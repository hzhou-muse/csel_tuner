/* MPI benchmark server: a long-lived job that serves measurement requests.
 *
 * Usage: mpiexec -n P mpisrv [max-m] [--plan FILE --out FILE [--deadline SEC]
 *                                     [--resume]] [--cold]
 *
 * Interactive mode (default): rank 0 reads one request per line from stdin
 * and writes one JSON reply per line to stdout. Batch mode (--plan): rank 0
 * reads the requests from FILE and writes the replies to --out; it first
 * writes an info record, stops cleanly when --deadline seconds have passed,
 * and with --resume skips the requests already answered in --out. Every
 * reply carries "seq", the request's index in the plan (blank lines and
 * lines starting with '#' are not requests).
 *
 * Requests are space-separated key=value tokens:
 *
 *   op=pingpong m=8113 pair=0,1 batches=3
 *   op=bcast m=65536 p=256 ppn=8 root=0 place=nic nodes=0:1 check=1
 *   op=allreduce m=65536 p=8 ppn=4 batches=3 target_us=1000 cache=hot
 *   op=info | op=quit
 *
 * Each request runs on a sub-communicator: the two ranks of `pair`, or p
 * ranks, ppn from each of p/ppn nodes. Nodes are MPI_COMM_TYPE_SHARED
 * domains numbered by their lowest world rank; `nodes=OFF:STRIDE` uses nodes
 * OFF, OFF+STRIDE, ... On each node `place=spread` takes ppn local ranks
 * evenly spaced over the node's n ranks (local ranks floor(j * n / ppn),
 * j < ppn; ppn need not divide n), `place=compact` the first ppn.
 * `place=nic` (default) is spread with each rank moved within its aligned
 * group of M local ranks so that the j-th rank has local rank = j (mod M),
 * M = MPISRV_NIC_MOD (default 1 = spread; M must divide n). MPICH picks a
 * rank's NIC by local rank mod the number of NICs close to it, so with M =
 * that number this gives the sub-communicator the NIC spread of a real
 * ppn-rank job; with spread, strides that are multiples of M put every rank
 * of a socket on one NIC (e.g. ppn 24 of 96 with 4 NICs per socket). Each
 * group of M local ranks must be bound to one socket. Other ranks wait for
 * the next request.
 *
 * Node topology comes from the site config, not from this code:
 * MPISRV_NIC_MOD (NICs per socket) and MPISRV_SOCKETS (sockets per node).
 * At startup they are checked against MPICH's MPI_INFO_ENV keys
 * num_close_nics / num_nics_available and against each rank's core binding
 * (socket from sysfs); the info record reports both and lists mismatches in
 * "topology_warnings".
 *
 * A measurement is: one untimed call, calibration of k calls per batch to
 * reach target_us/2, one warmup batch, one barrier, then `batches` batches of
 * k calls back to back. Batch j's latency is the synchronized latency
 * (max_i e_ij - max_i e_i,j-1) / k, where e_ij is rank i's time from its
 * barrier exit to the end of batch j. Ping-pong reports half the round trip.
 * sync=call instead puts a barrier before every call and times each call on
 * its own: batch j's value is the max over ranks of the rank's mean call time
 * in that batch (barriers excluded), i.e. latency without pipelining between
 * consecutive calls.
 * cvar.<NAME>=<value> (up to 4 per request) writes an MPI_T control variable
 * on every rank before the request and restores it after, e.g.
 * cvar.MPIR_CVAR_BCAST_INTRA_ALGORITHM=binomial; enum values go by name (MPI_T
 * enum items, or for older MPICH the description's value list) or number. It
 * needs an MPICH whose algorithm CVARs take effect when written at run time
 * (MPICH main since commit de0b8e6921), which the site config asserts with
 * MPISRV_CVAR_SWITCH=1; without it cvar.* requests are refused.
 * check=1 (bcast) first broadcasts a pattern and verifies it on every rank;
 * its reply also lists the local ranks used on the first node (node0_locals).
 * See docs/design.md.
 *
 * Environment: MPISRV_IDLE selects how ranks wait for the next request:
 *   spin  (default) MPI_Test every MPISRV_IDLE_NS ns (default 50000), with a
 *         pause-instruction loop in between: no MPI progress, no syscalls,
 *         and the core stays at speed. Rank 0 also spins while polling stdin.
 *   sleep same, but nanosleep in between. Cores drop to low clock speed and
 *         need tens of ms of work to recover (docs/design.md, section 3.4).
 *   busy  blocking MPI_Bcast (MPICH busy-polls its progress engine).
 *   split the members of the last request spin, all other ranks sleep (at
 *         start every rank sleeps), so idle cores do not take the measured
 *         cores' turbo budget. With MPISRV_SPINUP_US, the spin-up runs only
 *         when some member was sleeping.
 * MPISRV_SPINUP_US: before calibration, members run the operation for at
 * least this long; the reply's "spun_up" says whether it ran.
 * MPISRV_DEBUG: members print themselves to stderr.
 */
#define _GNU_SOURCE             /* sched_getaffinity */
#include <mpi.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>            /* strcasecmp */
#include <time.h>
#include <poll.h>
#include <unistd.h>

#define MAX_BATCHES 1000
#define MAX_COMMS 512
#define TAG_LEN 64
#define HOST_LEN 64
#define MAX_CVARS 4
#define CVAR_LEN 64

enum op { OP_NONE, OP_INFO, OP_QUIT, OP_PINGPONG, OP_ALLREDUCE, OP_BCAST };
static const char *op_names[] = { "none", "info", "quit", "pingpong", "allreduce", "bcast" };

struct req {
    int op;
    int seq;
    long m;
    int p, ppn;
    int place;                  /* PLACE_* */
    int node_off, node_stride;
    int pair[2];
    int root;
    int batches;
    double target_us;
    int cold;
    int check;
    int sync_call;              /* sync=call: barrier before every call */
    char tag[TAG_LEN];
    int ncvar;                  /* cvar.NAME=VALUE tokens, applied for this request only */
    char cvar_name[MAX_CVARS][CVAR_LEN];
    char cvar_val[MAX_CVARS][CVAR_LEN];
};

struct result {
    long m;
    int k;
    int check_err;              /* -1: not checked */
    int spun_up;
    double wall_us;
    double spread;
    double t_us[MAX_BATCHES];
};

static int world_rank, world_size;
static int my_node, my_local;   /* node index and rank within the node */
static int num_nodes, min_ppn;
static int nic_mod = 1;         /* MPISRV_NIC_MOD, for place=nic */
static int *socket_of;          /* per world rank; -1: not bound within one socket */
static int close_nics, nics_available;  /* from MPI_INFO_ENV; -1 unknown, -2 differ by rank */
static int nic_groups_ok;       /* every group of nic_mod local ranks on one socket */
static char topo_warn[1024];
enum place { PLACE_COMPACT, PLACE_SPREAD, PLACE_NIC };
static const char *place_names[] = { "compact", "spread", "nic" };
static int *node_of, *local_of; /* per world rank */
static long max_m;
static int cold_ok;
static long idle_ns = 50000;
enum idle { IDLE_SPIN, IDLE_SLEEP, IDLE_BUSY, IDLE_SPLIT };
static int idle_mode = IDLE_SPIN;
static int hot;                 /* split: member of the last request, so spinning */
static double spinup_us = 0;
static char *sbuf, *rbuf;       /* max_m each, 2 * max_m with --cold */
static FILE *in, *out;          /* rank 0: requests and replies */

static struct {
    struct req key;             /* layout fields only */
    MPI_Comm comm;
} comms[MAX_COMMS];
static int num_comms;

/* ---- topology ---------------------------------------------------------- */

static void init_topology(void)
{
    MPI_Comm node;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &node);
    MPI_Comm_rank(node, &my_local);
    /* number nodes by their lowest world rank */
    int leader = world_rank;
    MPI_Allreduce(MPI_IN_PLACE, &leader, 1, MPI_INT, MPI_MIN, node);
    MPI_Comm_free(&node);

    int *leaders = malloc(sizeof(int) * world_size);
    node_of = malloc(sizeof(int) * world_size);
    local_of = malloc(sizeof(int) * world_size);
    MPI_Allgather(&leader, 1, MPI_INT, leaders, 1, MPI_INT, MPI_COMM_WORLD);
    MPI_Allgather(&my_local, 1, MPI_INT, local_of, 1, MPI_INT, MPI_COMM_WORLD);
    int *index = malloc(sizeof(int) * world_size);
    num_nodes = 0;
    for (int r = 0; r < world_size; r++)
        if (leaders[r] == r)
            index[r] = num_nodes++;
    for (int r = 0; r < world_size; r++)
        node_of[r] = index[leaders[r]];
    my_node = node_of[world_rank];
    min_ppn = world_size;
    for (int n = 0; n < num_nodes; n++) {
        int cnt = 0;
        for (int r = 0; r < world_size; r++)
            if (node_of[r] == n)
                cnt++;
        if (cnt < min_ppn)
            min_ppn = cnt;
    }
    free(index);
    free(leaders);
}

/* Socket of the CPUs this rank may run on, -1 if they span sockets. */
static int my_socket(void)
{
    cpu_set_t set;
    if (sched_getaffinity(0, sizeof(set), &set))
        return -1;
    int sock = -2;
    for (int cpu = 0; cpu < CPU_SETSIZE; cpu++) {
        if (!CPU_ISSET(cpu, &set))
            continue;
        char path[128];
        snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/topology/physical_package_id", cpu);
        int s = -1;
        FILE *f = fopen(path, "r");
        if (f) {
            if (fscanf(f, "%d", &s) != 1)
                s = -1;
            fclose(f);
        }
        if (s < 0 || (sock != -2 && s != sock))
            return -1;
        sock = s;
    }
    return sock < 0 ? -1 : sock;
}

static int info_env_int(const char *key)
{
    char val[32];
    int len = sizeof(val), flag = 0;
    MPI_Info_get_string(MPI_INFO_ENV, key, &len, val, &flag);
    return flag ? atoi(val) : -1;
}

static void warn(const char *msg)
{
    size_t n = strlen(topo_warn);
    snprintf(topo_warn + n, sizeof(topo_warn) - n, "%s%s", n ? "; " : "", msg);
}

/* Site topology (MPISRV_NIC_MOD, MPISRV_SOCKETS), checked against the core
 * binding and the NIC counts MPICH reports. After init_topology. */
static void init_site(void)
{
    int mine[2] = { my_socket(), info_env_int("num_close_nics") };
    int *all = malloc(sizeof(int) * 2 * world_size);
    MPI_Allgather(mine, 2, MPI_INT, all, 2, MPI_INT, MPI_COMM_WORLD);
    socket_of = malloc(sizeof(int) * world_size);
    close_nics = all[1];
    int bound = 1;
    for (int r = 0; r < world_size; r++) {
        socket_of[r] = all[2 * r];
        bound &= socket_of[r] >= 0;
        if (all[2 * r + 1] != close_nics)
            close_nics = -2;
    }
    free(all);
    nics_available = info_env_int("num_nics_available");

    char msg[160];
    const char *env = getenv("MPISRV_NIC_MOD");
    if (env)
        nic_mod = atoi(env);
    if (nic_mod < 1 || min_ppn % nic_mod) {
        snprintf(msg, sizeof(msg), "MPISRV_NIC_MOD=%d does not divide %d ranks per node, using 1",
                 nic_mod, min_ppn);
        warn(msg);
        nic_mod = 1;
    }
    /* a single-node job may open only one NIC (MPICH then reports
     * num_close_nics = 1) and never uses the network: nothing to check */
    if (num_nodes == 1)
        ;
    else if (close_nics == -2)
        warn("ranks report different num_close_nics");
    else if (env && close_nics > 0 && close_nics != nic_mod) {
        snprintf(msg, sizeof(msg), "MPICH num_close_nics=%d but MPISRV_NIC_MOD=%d", close_nics, nic_mod);
        warn(msg);
    }
    if (!bound)
        warn("ranks not bound within one socket: sockets unknown");
    env = getenv("MPISRV_SOCKETS");
    if (env && bound && min_ppn >= atoi(env)) {
        for (int n = 0; n < num_nodes; n++) {
            int seen[64] = { 0 }, cnt = 0;
            for (int r = 0; r < world_size; r++)
                if (node_of[r] == n && socket_of[r] < 64 && !seen[socket_of[r]]++)
                    cnt++;
            if (cnt != atoi(env)) {
                snprintf(msg, sizeof(msg), "node %d: ranks on %d sockets, MPISRV_SOCKETS=%s", n, cnt, env);
                warn(msg);
                break;
            }
        }
    }
    /* group of local rank l: local ranks l - l % nic_mod ... on the same node */
    nic_groups_ok = 1;
    for (int r = 0; r < world_size && bound; r++)
        for (int q = 0; q < world_size; q++)
            if (node_of[q] == node_of[r] && local_of[q] == local_of[r] - local_of[r] % nic_mod &&
                socket_of[q] != socket_of[r])
                nic_groups_ok = 0;
    if (world_rank == 0 && topo_warn[0])
        fprintf(stderr, "mpisrv: topology: %s\n", topo_warn);
}

/* Local rank of the j-th member on each node. For place=nic, ranks j < j'
 * in one group of nic_mod have base ranks less than nic_mod apart, so
 * j' - j < nic_mod and their residues differ: the ranks are distinct. */
static int place_local(int j, const struct req *rq)
{
    if (rq->place == PLACE_COMPACT)
        return j;
    int base = (int) ((long) j * min_ppn / rq->ppn);
    if (rq->place == PLACE_SPREAD)
        return base;
    return base - base % nic_mod + j % nic_mod;
}

/* Whether world rank r belongs to the request's sub-communicator. */
static int is_member(int r, const struct req *rq)
{
    if (rq->op == OP_PINGPONG)
        return r == rq->pair[0] || r == rq->pair[1];
    int rel = node_of[r] - rq->node_off;
    if (rel < 0 || rel % rq->node_stride || rel / rq->node_stride >= rq->p / rq->ppn)
        return 0;
    for (int j = 0; j < rq->ppn; j++)
        if (place_local(j, rq) == local_of[r])
            return 1;
    return 0;
}

/* World rank of the sub-communicator's rank 0. */
static int comm_root(const struct req *rq)
{
    if (rq->op == OP_PINGPONG)
        return rq->pair[0];
    for (int r = 0; r < world_size; r++)
        if (is_member(r, rq))
            return r;
    return 0;
}

static int same_layout(const struct req *a, const struct req *b)
{
    if ((a->op == OP_PINGPONG) != (b->op == OP_PINGPONG))
        return 0;
    if (a->op == OP_PINGPONG)
        return a->pair[0] == b->pair[0] && a->pair[1] == b->pair[1];
    return a->p == b->p && a->ppn == b->ppn && a->place == b->place &&
        a->node_off == b->node_off && a->node_stride == b->node_stride;
}

/* Sub-communicator for a request; collective over MPI_COMM_WORLD the first
 * time a layout is seen. Returns MPI_COMM_NULL on non-members. */
static MPI_Comm get_comm(const struct req *rq)
{
    for (int i = 0; i < num_comms; i++)
        if (same_layout(&comms[i].key, rq))
            return comms[i].comm;

    int color = MPI_UNDEFINED, key = world_rank;
    if (is_member(world_rank, rq)) {
        color = 0;
        if (rq->op == OP_PINGPONG)
            key = world_rank == rq->pair[0] ? 0 : 1;
    }
    MPI_Comm c;
    MPI_Comm_split(MPI_COMM_WORLD, color, key, &c);
    if (num_comms < MAX_COMMS) {
        comms[num_comms].key = *rq;
        comms[num_comms].comm = c;
        num_comms++;
    }
    return c;
}

/* ---- requests ---------------------------------------------------------- */

static int tag_ok(const char *s)
{
    for (; *s; s++)
        if (!(('a' <= *s && *s <= 'z') || ('A' <= *s && *s <= 'Z') || ('0' <= *s && *s <= '9') ||
              strchr("_.:-/", *s)))
            return 0;
    return 1;
}

/* Parse a request line on rank 0. Returns NULL or an error message. */
/* ---- per-request control variables (MPI_T) ----------------------------- */

static int cvar_switch;         /* MPISRV_CVAR_SWITCH: the build applies CVAR writes at run time */

struct cvar {
    char name[CVAR_LEN];
    MPI_T_cvar_handle h;
    MPI_Datatype dtype;         /* MPI_INT (plain or enum) or MPI_CHAR */
    MPI_T_enum etype;
    int count;
    void *saved;                /* value before the request, restored after it */
    char *desc;                 /* MPI_T description: MPICH lists its enum values there */
};
static struct cvar cvars[64];
static int num_cvars;

/* The cached MPI_T handle of a control variable, or NULL with *err set. */
static struct cvar *find_cvar(const char *name, const char **err)
{
    for (int i = 0; i < num_cvars; i++)
        if (!strcmp(cvars[i].name, name))
            return &cvars[i];
    int idx, nlen = 0, dlen = 0, verb, bind, scope;
    struct cvar c = { { 0 } };
    if (num_cvars == (int) (sizeof(cvars) / sizeof(cvars[0]))) {
        *err = "too many distinct cvars";
        return NULL;
    }
    if (strlen(name) >= CVAR_LEN || MPI_T_cvar_get_index(name, &idx) != MPI_SUCCESS) {
        *err = "unknown cvar";
        return NULL;
    }
    MPI_T_cvar_get_info(idx, NULL, &nlen, &verb, &c.dtype, &c.etype, NULL, &dlen, &bind, &scope);
    c.desc = calloc((size_t) dlen + 1, 1);
    MPI_T_cvar_get_info(idx, NULL, &nlen, &verb, &c.dtype, &c.etype, c.desc, &dlen, &bind, &scope);
    if (bind != MPI_T_BIND_NO_OBJECT) {
        *err = "cvar is bound to an MPI object";
        return NULL;
    }
    if (c.dtype != MPI_INT && c.dtype != MPI_CHAR) {
        *err = "cvar type is not int, enum or string";
        return NULL;
    }
    if (MPI_T_cvar_handle_alloc(idx, NULL, &c.h, &c.count) != MPI_SUCCESS) {
        *err = "cannot allocate a handle for the cvar";
        return NULL;
    }
    strcpy(c.name, name);
    c.saved = calloc((size_t) c.count + CVAR_LEN, c.dtype == MPI_INT ? sizeof(int) : 1);
    cvars[num_cvars] = c;
    return &cvars[num_cvars++];
}

/* Convert a value string to the cvar's representation in buf (an int, or a
 * string of < CVAR_LEN chars); NULL on success, else the reason. */
static const char *cvar_value(const struct cvar *c, const char *val, void *buf)
{
    if (c->dtype == MPI_CHAR) {
        if (strlen(val) >= CVAR_LEN)
            return "cvar value too long";
        strcpy(buf, val);
        return NULL;
    }
    char *end;
    long v = strtol(val, &end, 0);
    if (*val && !*end) {                /* a number is taken as is */
        *(int *) buf = (int) v;
        return NULL;
    }
    if (c->etype != MPI_T_ENUM_NULL) {  /* MPICH main since f95b6ec93d */
        int n, len = 0;
        MPI_T_enum_get_info(c->etype, &n, NULL, &len);
        for (int i = 0; i < n; i++) {
            char item[256];
            int iv, ilen = sizeof(item);
            if (MPI_T_enum_get_item(c->etype, i, &iv, item, &ilen) == MPI_SUCCESS && !strcasecmp(item, val)) {
                *(int *) buf = iv;
                return NULL;
            }
        }
        return "value is not one of the cvar's enum items";
    }
    /* Older MPICH registers enum CVARs as plain MPI_INT; the value is the
     * position of the "name - text" line in the description (maint/extractcvars). */
    int pos = 0;
    for (const char *l = c->desc; l && *l; l = strchr(l, '\n') ? strchr(l, '\n') + 1 : NULL) {
        const char *w = l + strspn(l, " \t");
        size_t n = strcspn(w, " \t\n");
        const char *after = w + n + strspn(w + n, " \t");
        if (n && *after == '-') {
            if (n == strlen(val) && !strncasecmp(w, val, n)) {
                *(int *) buf = pos;
                return NULL;
            }
            pos++;
        }
    }
    if (pos)
        return "value is not one of the cvar's listed values";
    return "cvar value is not an integer";
}

/* All ranks: write the request's cvars (saving the old values) or restore them. */
static void apply_cvars(const struct req *rq, int restore)
{
    for (int k = restore ? rq->ncvar - 1 : 0; restore ? k >= 0 : k < rq->ncvar; k += restore ? -1 : 1) {
        const char *err = NULL;
        struct cvar *c = find_cvar(rq->cvar_name[k], &err);
        int buf[CVAR_LEN];
        if (c && !restore) {
            MPI_T_cvar_read(c->h, c->saved);
            err = cvar_value(c, rq->cvar_val[k], buf);
        }
        if (!err && MPI_T_cvar_write(c->h, restore ? c->saved : (void *) buf) != MPI_SUCCESS)
            err = "MPI_T_cvar_write failed";
        if (err) {              /* rank 0 validated it, so ranks disagree: give up */
            fprintf(stderr, "mpisrv: rank %d: %s: %s\n", world_rank, rq->cvar_name[k], err);
            MPI_Abort(MPI_COMM_WORLD, 3);
        }
    }
}

static const char *parse(char *line, struct req *rq)
{
    memset(rq, 0, sizeof(*rq));
    rq->op = OP_NONE;
    rq->p = world_size;
    rq->ppn = -1;
    rq->place = PLACE_NIC;
    rq->node_stride = 1;
    rq->pair[1] = 1;
    rq->batches = 3;
    rq->target_us = 1000.0;

    static char err[256];
    for (char *tok = strtok(line, " \t\r\n"); tok; tok = strtok(NULL, " \t\r\n")) {
        char *val = strchr(tok, '=');
        if (!val) {
            snprintf(err, sizeof(err), "token without '=': %.64s", tok);
            return err;
        }
        *val++ = '\0';
        if (!strcmp(tok, "op")) {
            for (int i = 1; i < (int) (sizeof(op_names) / sizeof(op_names[0])); i++)
                if (!strcmp(val, op_names[i]))
                    rq->op = i;
            if (rq->op == OP_NONE) {
                snprintf(err, sizeof(err), "unknown op: %.64s", val);
                return err;
            }
        } else if (!strcmp(tok, "m")) {
            rq->m = atol(val);
        } else if (!strcmp(tok, "p")) {
            rq->p = atoi(val);
        } else if (!strcmp(tok, "ppn")) {
            rq->ppn = atoi(val);
        } else if (!strcmp(tok, "place")) {
            rq->place = -1;
            for (int i = 0; i < 3; i++)
                if (!strcmp(val, place_names[i]))
                    rq->place = i;
            if (rq->place < 0)
                return "place must be nic, spread or compact";
        } else if (!strcmp(tok, "nodes")) {
            if (sscanf(val, "%d:%d", &rq->node_off, &rq->node_stride) < 1)
                return "nodes must be OFF or OFF:STRIDE";
        } else if (!strcmp(tok, "pair")) {
            if (sscanf(val, "%d,%d", &rq->pair[0], &rq->pair[1]) != 2)
                return "pair must be a,b";
        } else if (!strcmp(tok, "root")) {
            rq->root = atoi(val);
        } else if (!strcmp(tok, "batches")) {
            rq->batches = atoi(val);
        } else if (!strcmp(tok, "target_us")) {
            rq->target_us = atof(val);
        } else if (!strcmp(tok, "cache")) {
            if (strcmp(val, "hot") && strcmp(val, "cold"))
                return "cache must be hot or cold";
            rq->cold = !strcmp(val, "cold");
        } else if (!strcmp(tok, "check")) {
            rq->check = atoi(val);
        } else if (!strcmp(tok, "sync")) {
            if (strcmp(val, "batch") && strcmp(val, "call"))
                return "sync must be batch or call";
            rq->sync_call = !strcmp(val, "call");
        } else if (!strncmp(tok, "cvar.", 5)) {
            const char *cerr = NULL;
            int buf[CVAR_LEN];
            if (!cvar_switch)
                return "cvar.* needs an MPICH that applies CVAR writes at run time (MPISRV_CVAR_SWITCH=1)";
            if (rq->ncvar == MAX_CVARS)
                return "at most 4 cvar.* per request";
            for (int k = 0; k < rq->ncvar; k++)
                if (!strcmp(rq->cvar_name[k], tok + 5))
                    return "cvar given twice";
            struct cvar *c = find_cvar(tok + 5, &cerr);
            if (c)
                cerr = cvar_value(c, val, buf);
            if (cerr) {
                snprintf(err, sizeof(err), "%s: %.64s=%.64s", cerr, tok + 5, val);
                return err;
            }
            strcpy(rq->cvar_name[rq->ncvar], tok + 5);
            strcpy(rq->cvar_val[rq->ncvar++], val);
        } else if (!strcmp(tok, "tag")) {
            if (strlen(val) >= TAG_LEN || !tag_ok(val))
                return "tag must be < 64 chars of [A-Za-z0-9_.:-/]";
            strcpy(rq->tag, val);
        } else {
            snprintf(err, sizeof(err), "unknown key: %.64s", tok);
            return err;
        }
    }
    if (rq->op == OP_NONE)
        return "missing op";
    if (rq->ppn < 0)
        rq->ppn = rq->p < min_ppn ? rq->p : min_ppn;
    if (rq->m < 0 || rq->m > max_m) {
        snprintf(err, sizeof(err), "m must be in [0, %ld]", max_m);
        return err;
    }
    if (rq->batches < 1 || rq->batches > MAX_BATCHES)
        return "batches must be in [1, 1000]";
    if (rq->target_us <= 0)
        return "target_us must be > 0";
    if (rq->cold && !cold_ok)
        return "cache=cold needs the server started with --cold";
    if (rq->op == OP_PINGPONG) {
        int a = rq->pair[0], b = rq->pair[1];
        if (a == b || a < 0 || b < 0 || a >= world_size || b >= world_size)
            return "pair must be two different world ranks";
    }
    if (rq->op == OP_ALLREDUCE || rq->op == OP_BCAST) {
        if (rq->p < 1 || rq->p > world_size || rq->ppn < 1 || rq->p % rq->ppn)
            return "need 1 <= p <= world size and p a multiple of ppn";
        if (rq->node_off < 0 || rq->node_stride < 1 || rq->ppn > min_ppn ||
            rq->node_off + (rq->p / rq->ppn - 1) * rq->node_stride >= num_nodes)
            return "layout does not fit: check p/ppn, nodes=OFF:STRIDE and ppn <= ranks per node";
        if (rq->root < 0 || rq->root >= rq->p)
            return "root must be in [0, p)";
        if (rq->place == PLACE_NIC && !nic_groups_ok)
            return "place=nic: a group of MPISRV_NIC_MOD local ranks spans two sockets";
    }
    return NULL;
}

/* Pause between checks without MPI calls: spin keeps the core at speed,
 * sleep lets it idle. */
static void idle_pause(void)
{
    if (idle_mode == IDLE_SLEEP || (idle_mode == IDLE_SPLIT && !hot)) {
        struct timespec ts = { idle_ns / 1000000000, idle_ns % 1000000000 };
        nanosleep(&ts, NULL);
    } else {
        double t0 = MPI_Wtime();
        while ((MPI_Wtime() - t0) * 1e9 < idle_ns)
            for (int i = 0; i < 100; i++)
                __builtin_ia32_pause();
    }
}

/* Wait for the next request without driving MPI progress in between. */
static void bcast_request(struct req *rq)
{
    if (idle_mode == IDLE_BUSY) {
        MPI_Bcast(rq, sizeof(*rq), MPI_BYTE, 0, MPI_COMM_WORLD);
        return;
    }
    MPI_Request r;
    int done = 0;
    MPI_Ibcast(rq, sizeof(*rq), MPI_BYTE, 0, MPI_COMM_WORLD, &r);
    for (;;) {
        MPI_Test(&r, &done, MPI_STATUS_IGNORE);
        if (done)
            break;
        idle_pause();
    }
}

/* Rank 0: read the next line; from stdin, wait for input in the idle mode.
 * stdin is unbuffered (see main), so poll() sees every line not yet read even
 * when a driver sends several requests at once. */
static char *read_line(char *line, int len)
{
    if (in == stdin && idle_mode != IDLE_SLEEP) {
        struct pollfd pfd = { STDIN_FILENO, POLLIN, 0 };
        while (poll(&pfd, 1, 0) == 0)
            idle_pause();
    }
    return fgets(line, len, in);
}

/* ---- measurement ------------------------------------------------------- */

/* k calls of the operation; returns local elapsed seconds (with sync=call,
 * the sum of the calls' own times, barriers excluded) */
static double run(const struct req *rq, MPI_Comm c, int crank, long count, int k)
{
    long bytes = rq->op == OP_ALLREDUCE ? count * (long) sizeof(int) : count;
    double t0 = MPI_Wtime(), sum = 0;
    for (int i = 0; i < k; i++) {
        long off = rq->cold ? (i * bytes) % max_m : 0;
        if (rq->sync_call) {
            MPI_Barrier(c);
            t0 = MPI_Wtime();
        }
        /* ping-pong and bcast use one buffer, like bench/pingpong.c */
        char *s = sbuf + off, *r = rbuf + off;
        if (rq->op == OP_PINGPONG) {
            if (crank == 0) {
                MPI_Send(s, count, MPI_BYTE, 1, 0, c);
                MPI_Recv(s, count, MPI_BYTE, 1, 0, c, MPI_STATUS_IGNORE);
            } else {
                MPI_Recv(s, count, MPI_BYTE, 0, 0, c, MPI_STATUS_IGNORE);
                MPI_Send(s, count, MPI_BYTE, 0, 0, c);
            }
        } else if (rq->op == OP_BCAST) {
            MPI_Bcast(s, count, MPI_BYTE, rq->root, c);
        } else {
            MPI_Allreduce(s, r, count, MPI_INT, MPI_SUM, c);
        }
        if (rq->sync_call)
            sum += MPI_Wtime() - t0;
    }
    return rq->sync_call ? sum : MPI_Wtime() - t0;
}

/* Broadcast a pattern once and count wrong bytes over all ranks. */
static int check_bcast(const struct req *rq, MPI_Comm c, int crank)
{
    long n = rq->m;
    unsigned char *b = (unsigned char *) sbuf;
    for (long i = 0; i < n; i++)
        b[i] = crank == rq->root ? (unsigned char) (i * 131 + rq->seq + 7) : 0;
    MPI_Bcast(b, n, MPI_BYTE, rq->root, c);
    int bad = 0;
    for (long i = 0; i < n; i++)
        if (b[i] != (unsigned char) (i * 131 + rq->seq + 7))
            bad++;
    MPI_Allreduce(MPI_IN_PLACE, &bad, 1, MPI_INT, MPI_SUM, c);
    return bad;
}

static void measure(const struct req *rq, MPI_Comm c, struct result *res)
{
    int crank;
    MPI_Comm_rank(c, &crank);
    long count = rq->op == OP_ALLREDUCE ? rq->m / (long) sizeof(int) : rq->m;
    double w0 = MPI_Wtime();

    res->check_err = rq->check && rq->op == OP_BCAST ? check_bcast(rq, c, crank) : -1;
    run(rq, c, crank, count, 1);        /* first-call exclusion */
    res->spun_up = spinup_us > 0;
    if (res->spun_up && idle_mode == IDLE_SPLIT) {      /* only if some member was asleep */
        int cold = !hot;
        MPI_Allreduce(MPI_IN_PLACE, &cold, 1, MPI_INT, MPI_MAX, c);
        res->spun_up = cold;
    }
    if (res->spun_up) {                 /* time-based spin-up, decided by comm rank 0 */
        double s0 = MPI_Wtime();
        int more = 1;
        while (more) {
            run(rq, c, crank, count, 1);
            more = (MPI_Wtime() - s0) * 1e6 < spinup_us;
            MPI_Bcast(&more, 1, MPI_INT, 0, c);
        }
    }
    int k = 1;
    for (;;) {                          /* calibration */
        double t = run(rq, c, crank, count, k);
        MPI_Allreduce(MPI_IN_PLACE, &t, 1, MPI_DOUBLE, MPI_MAX, c);
        if (t * 1e6 >= rq->target_us / 2 || k >= (1 << 20))
            break;
        k *= 2;
    }
    run(rq, c, crank, count, k);        /* warmup */

    double e[MAX_BATCHES], emax[MAX_BATCHES], emin[MAX_BATCHES];
    MPI_Barrier(c);
    double t0 = MPI_Wtime();
    for (int j = 0; j < rq->batches; j++) {
        double b = run(rq, c, crank, count, k);
        e[j] = rq->sync_call ? (j ? e[j - 1] : 0.0) + b : MPI_Wtime() - t0;
    }
    MPI_Reduce(e, emax, rq->batches, MPI_DOUBLE, MPI_MAX, 0, c);
    MPI_Reduce(e, emin, rq->batches, MPI_DOUBLE, MPI_MIN, 0, c);
    if (crank == 0) {
        double div = rq->op == OP_PINGPONG ? 2.0 * k : k;
        for (int j = 0; j < rq->batches; j++)
            res->t_us[j] = (emax[j] - (j ? emax[j - 1] : 0.0)) / div * 1e6;
        int last = rq->batches - 1;
        res->spread = (emax[last] - emin[last]) / emax[last];
        res->k = k;
        res->m = rq->op == OP_ALLREDUCE ? count * (long) sizeof(int) : count;
        res->wall_us = (MPI_Wtime() - w0) * 1e6;
    }
}

/* ---- replies (rank 0) -------------------------------------------------- */

static void put_string(const char *s)
{
    for (; *s; s++)
        fputc(*s == '"' || *s == '\\' ? '\'' : (*s == '\n' ? ' ' : *s), out);
}

static void reply_error(int seq, const char *msg)
{
    fprintf(out, "{\"status\":\"error\",\"seq\":%d,\"msg\":\"", seq);
    put_string(msg);
    fprintf(out, "\"}\n");
    fflush(out);
}

/* Collective: hostnames of all nodes, the library, and the MPI/fabric env. */
static void reply_info(int seq)
{
    char host[HOST_LEN] = { 0 }, *hosts = NULL;
    gethostname(host, HOST_LEN - 1);
    if (world_rank == 0)
        hosts = malloc((size_t) HOST_LEN * world_size);
    MPI_Gather(host, HOST_LEN, MPI_CHAR, hosts, HOST_LEN, MPI_CHAR, 0, MPI_COMM_WORLD);
    if (world_rank != 0)
        return;

    char ver[MPI_MAX_LIBRARY_VERSION_STRING];
    int len;
    MPI_Get_library_version(ver, &len);
    char *nl = strchr(ver, '\n');
    if (nl)
        *nl = '\0';
    fprintf(out, "{\"status\":\"ok\",\"op\":\"info\",\"seq\":%d,\"world_size\":%d,\"nodes\":%d,"
            "\"min_ppn\":%d,\"nic_mod\":%d,\"num_close_nics\":%d,\"num_nics_available\":%d,"
            "\"max_m\":%ld,\"cvar_switch\":%d,\"library\":\"", seq, world_size, num_nodes, min_ppn, nic_mod,
            close_nics, nics_available, max_m, cvar_switch);
    put_string(ver);
    fprintf(out, "\",\"hosts\":[");
    for (int n = 0, first = 1; n < num_nodes; n++)
        for (int r = 0; r < world_size; r++)
            if (node_of[r] == n && local_of[r] == 0) {
                fprintf(out, "%s\"", first ? "" : ",");
                put_string(hosts + (size_t) r * HOST_LEN);
                fprintf(out, "\"");
                first = 0;
                break;
            }
    fprintf(out, "],\"node_size\":[");
    for (int n = 0; n < num_nodes; n++) {
        int cnt = 0;
        for (int r = 0; r < world_size; r++)
            cnt += node_of[r] == n;
        fprintf(out, "%s%d", n ? "," : "", cnt);
    }
    fprintf(out, "],\"node0_sockets\":[");
    for (int r = 0, n = 0; r < world_size; r++)
        if (node_of[r] == 0)
            fprintf(out, "%s%d", n++ ? "," : "", socket_of[r]);
    fprintf(out, "],\"topology_warnings\":\"");
    put_string(topo_warn);
    fprintf(out, "\",\"env\":{");
    extern char **environ;
    int first = 1;
    for (char **e = environ; *e; e++)
        if (!strncmp(*e, "MPIR_CVAR_", 10) || !strncmp(*e, "MPICH_", 6) || !strncmp(*e, "FI_", 3) ||
            !strncmp(*e, "MPISRV_", 7)) {
            char *eq = strchr(*e, '=');
            if (!eq)
                continue;
            fprintf(out, "%s\"%.*s\":\"", first ? "" : ",", (int) (eq - *e), *e);
            put_string(eq + 1);
            fprintf(out, "\"");
            first = 0;
        }
    fprintf(out, "}}\n");
    fflush(out);
    free(hosts);
}

static void reply_result(const struct req *rq, const struct result *res)
{
    fprintf(out, "{\"status\":\"ok\",\"seq\":%d,\"op\":\"%s\",\"m\":%ld,", rq->seq,
            op_names[rq->op], res->m);
    if (rq->op == OP_PINGPONG)
        fprintf(out, "\"pair\":[%d,%d],", rq->pair[0], rq->pair[1]);
    else
        fprintf(out, "\"p\":%d,\"ppn\":%d,\"place\":\"%s\",\"nodes\":\"%d:%d\",", rq->p, rq->ppn,
                place_names[rq->place], rq->node_off, rq->node_stride);
    if (rq->op == OP_BCAST)
        fprintf(out, "\"root\":%d,", rq->root);
    if (res->check_err >= 0) {
        fprintf(out, "\"check\":\"%s\",\"check_err\":%d,", res->check_err ? "fail" : "ok",
                res->check_err);
        /* placement record: local ranks used on the first node */
        fprintf(out, "\"node0_locals\":[");
        for (int r = 0, n = 0; r < world_size; r++)
            if (node_of[r] == rq->node_off && is_member(r, rq))
                fprintf(out, "%s%d", n++ ? "," : "", local_of[r]);
        fprintf(out, "],");
    }
    if (spinup_us > 0)
        fprintf(out, "\"spun_up\":%d,", res->spun_up);
    if (rq->ncvar) {
        fprintf(out, "\"cvars\":{");
        for (int k = 0; k < rq->ncvar; k++) {
            fprintf(out, "%s\"", k ? "," : "");
            put_string(rq->cvar_name[k]);
            fprintf(out, "\":\"");
            put_string(rq->cvar_val[k]);
            fprintf(out, "\"");
        }
        fprintf(out, "},");
    }
    fprintf(out, "\"cache\":\"%s\",\"sync\":\"%s\",\"k\":%d,\"t_us\":[", rq->cold ? "cold" : "hot",
            rq->sync_call ? "call" : "batch", res->k);
    for (int j = 0; j < rq->batches; j++)
        fprintf(out, "%s%.4f", j ? "," : "", res->t_us[j]);
    fprintf(out, "],\"t_rank_spread\":%.4f,\"wall_us\":%.1f,\"tag\":\"%s\"}\n",
            res->spread, res->wall_us, rq->tag);
    fflush(out);
}

/* ---- batch mode -------------------------------------------------------- */

/* --resume: mark the requests already answered in the output file. */
static char *load_done(const char *path, int *n)
{
    char *done = calloc(1, 1 << 20);
    *n = 0;
    FILE *f = fopen(path, "r");
    if (!f)
        return done;
    char line[65536];
    while (fgets(line, sizeof(line), f)) {
        char *s = strstr(line, "\"seq\":");
        int seq;
        if (s && sscanf(s + 6, "%d", &seq) == 1 && seq >= 0 && seq < (1 << 20) &&
            !strstr(line, "\"status\":\"deadline\"")) {
            done[seq] = 1;
            (*n)++;
        }
    }
    fclose(f);
    return done;
}

/* ------------------------------------------------------------------------ */

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int provided;
    MPI_T_init_thread(MPI_THREAD_SINGLE, &provided);
    cvar_switch = getenv("MPISRV_CVAR_SWITCH") && atoi(getenv("MPISRV_CVAR_SWITCH"));
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    double t_start = MPI_Wtime();

    const char *plan = NULL, *outpath = NULL;
    double deadline = 0;
    int resume = 0;
    max_m = 1L << 24;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--plan") && i + 1 < argc)
            plan = argv[++i];
        else if (!strcmp(argv[i], "--out") && i + 1 < argc)
            outpath = argv[++i];
        else if (!strcmp(argv[i], "--deadline") && i + 1 < argc)
            deadline = atof(argv[++i]);
        else if (!strcmp(argv[i], "--resume"))
            resume = 1;
        else if (!strcmp(argv[i], "--cold"))
            cold_ok = 1;
        else if (argv[i][0] >= '0' && argv[i][0] <= '9')
            max_m = atol(argv[i]);
        else {
            if (world_rank == 0)
                fprintf(stderr, "mpisrv: unknown argument %s\n", argv[i]);
            MPI_Abort(MPI_COMM_WORLD, 2);
        }
    }
    if (getenv("MPISRV_IDLE_NS"))
        idle_ns = atol(getenv("MPISRV_IDLE_NS"));
    const char *im = getenv("MPISRV_IDLE");
    if (im)
        idle_mode = !strcmp(im, "sleep") ? IDLE_SLEEP : !strcmp(im, "busy") ? IDLE_BUSY :
            !strcmp(im, "split") ? IDLE_SPLIT : IDLE_SPIN;
    if (getenv("MPISRV_SPINUP_US"))
        spinup_us = atof(getenv("MPISRV_SPINUP_US"));
    init_topology();
    init_site();

    long buflen = (cold_ok ? 2 : 1) * (max_m > 0 ? max_m : 1);
    if (posix_memalign((void **) &sbuf, 4096, buflen) || posix_memalign((void **) &rbuf, 4096, buflen))
        MPI_Abort(MPI_COMM_WORLD, 1);
    memset(sbuf, 1, buflen);
    memset(rbuf, 0, buflen);

    char *done = NULL;
    int ndone = 0;
    in = stdin;
    if (world_rank == 0 && !plan)
        setvbuf(stdin, NULL, _IONBF, 0);
    out = stdout;
    if (world_rank == 0 && plan) {
        if (!outpath) {
            fprintf(stderr, "mpisrv: --plan needs --out\n");
            MPI_Abort(MPI_COMM_WORLD, 2);
        }
        if (resume)
            done = load_done(outpath, &ndone);
        in = fopen(plan, "r");
        out = fopen(outpath, resume ? "a" : "w");
        if (!in || !out) {
            perror(!in ? plan : outpath);
            MPI_Abort(MPI_COMM_WORLD, 2);
        }
    }
    if (plan)                           /* batch mode starts with an info record */
        reply_info(-1);

    char line[4096];
    struct req rq;
    struct result res;
    int seq = -1;
    for (;;) {
        if (world_rank == 0) {
            /* answer bad requests here; only valid ones are broadcast */
            for (;;) {
                if (plan && deadline > 0 && MPI_Wtime() - t_start > deadline) {
                    fprintf(out, "{\"status\":\"deadline\",\"seq\":%d,\"elapsed_s\":%.1f}\n", seq + 1,
                            MPI_Wtime() - t_start);
                    fflush(out);
                    rq.op = OP_QUIT;
                    break;
                }
                if (!read_line(line, sizeof(line))) {
                    rq.op = OP_QUIT;
                    break;
                }
                char *s = line + strspn(line, " \t\r\n");
                if (!*s || *s == '#')
                    continue;
                seq++;
                if (done && seq < (1 << 20) && done[seq])
                    continue;
                const char *err = parse(line, &rq);
                rq.seq = seq;
                if (!err)
                    break;
                reply_error(seq, err);
            }
        }
        bcast_request(&rq);
        if (rq.op == OP_QUIT)
            break;
        if (rq.op == OP_INFO) {
            reply_info(rq.seq);
            continue;
        }

        apply_cvars(&rq, 0);
        MPI_Comm c = get_comm(&rq);
        int croot = comm_root(&rq);     /* world rank of comm rank 0 */
        if (getenv("MPISRV_DEBUG") && c != MPI_COMM_NULL)
            fprintf(stderr, "member seq=%d world=%d node=%d local=%d\n", rq.seq, world_rank, my_node,
                    my_local);
        if (c != MPI_COMM_NULL)
            measure(&rq, c, &res);
        hot = c != MPI_COMM_NULL;
        apply_cvars(&rq, 1);
        if (croot != 0) {
            if (world_rank == croot)
                MPI_Send(&res, sizeof(res), MPI_BYTE, 0, 1, MPI_COMM_WORLD);
            else if (world_rank == 0)
                MPI_Recv(&res, sizeof(res), MPI_BYTE, croot, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        if (world_rank == 0)
            reply_result(&rq, &res);
    }

    for (int i = 0; i < num_comms; i++)
        if (comms[i].comm != MPI_COMM_NULL)
            MPI_Comm_free(&comms[i].comm);
    if (world_rank == 0 && plan) {
        fclose(in);
        fclose(out);
    }
    free(done);
    free(sbuf);
    free(rbuf);
    free(node_of);
    free(local_of);
    free(socket_of);
    MPI_T_finalize();
    MPI_Finalize();
    return 0;
}
