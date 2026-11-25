# csel_tuner: the model file

The model file holds every measurement of a run with its provenance, and the
fits derived from them; collmodel writes it and adds to it, colltree reads
it. This document is its format reference; why the file is shaped this way
(data as the source of truth, units, incremental models) is in
[design.md](design.md), section 12. The reader and writer are
`csel_tuner/modelfile.py` (its docstring has the full grammar) and
`csel_tuner/model.py`.

## collmodel/6 (the format written)

A model file is a list of units, each one model of one collective measured
one way (design.md §12.1), named by its **tags**:

| Tag | Meaning |
|---|---|
| `coll` | the collective (`--op`) |
| `variant` | the input-condition assignment the requests had (`kinds.run_variant`): `inplace` where MPI allows `MPI_IN_PLACE`, `builtin_op`/`commutative` (always 1: mpisrv sums with `MPI_SUM`) for the reductions |
| `datatype` | mpisrv's element type: `int` for the reductions, `byte` otherwise |
| `counts` | v-operations only: the blocks' pattern (`--counts`) |
| `launch` | the launch kind (`--launch`, `--mem`): `host` (host buffers), `gpu` (device buffers with GPU_COLL) or `device` (device buffers without it) |
| `kind` | `param_map` in a param-map file's units (below); absent in a model's |

It is written as **collmodel/6**, line-oriented text (`*.model`), so grep,
diff, less and awk read it and a user can annotate it:

```
collmodel/6 created=2026-11-22T10:00:00Z
@unit coll=allreduce variant={"builtin_op":1,"commutative":1,"inplace":0} datatype=int launch=host
sizes 1 2 3 4 ...
domain {"N":[1,8],"ppn":[1,12],"m":[1,67108864]}
layouts 1x2 1x3 ... 8x12
conditions inplace not_inplace pof2 ...
mask 1x2 6144                      # nonzero layout masks
left_out nb
fit_version 2
job 0 {"start":...,"mode":"live",...}
run {"mode":"create",...}
---- 3fa2c1d09b7e4a55 ----         # an inner selection file, verbatim, named by its hash
{...}
---- end
@algo binomial
info {"leaf":null,"restrictions":[],"mask":0,"inner":[]}
penalty intra 0.0 0.1 ...
penalty inter -
curve 2x4 2.181421 2.190931 ...
meas 2 4 1 4406.8 0 - - 269.7395 269.521 272.277
pre 1 2 1 0
@algo circ_graph
info {"leaf":"MPIR_Allreduce_intra_circ_graph",...}
params 0 {"chunk_size":65536,"q_len":8}    # parameters set per request, by index
meas 2 4 262144 812.5 0 p0 - 101.2 100.9 101.4
```

### General syntax

- One record per line; fields are separated by runs of blanks. The first
  word of a line is its kind.
- A blank line, or a line whose first non-blank character is `#`, is
  ignored. A `#` anywhere else is ordinary text: data lines have no
  trailing comments (the `# ...` in the example above are annotations).
- Names (algorithms, conditions, hashes) never contain blanks; the writer
  refuses one that does.
- Small structured values are one-line JSON. A tag value is a bare word
  where it reads back as the same string, else JSON.
- Numbers are written as JSON writes them: an integer, or a decimal with a
  point or an exponent (`nan` and `inf` are refused).
- A layout is `NxPPN`: nodes x ranks per node, e.g. `2x4`.
- A file is a header, then units; a unit is an `@unit` line, its unit
  lines, then its `@algo` sections. A gzipped file reads as it is.

### Header

| Line | Meaning |
|---|---|
| `collmodel/6 created=<time>` | the first non-comment line: the format and the UTC time the file was written (ISO 8601) |

The writer adds two comment lines after it that name the columns of `meas`
and `pre`.

### Unit lines

| Line | Syntax | Meaning |
|---|---|---|
| `@unit` | `@unit KEY=VALUE ...` | starts a unit; its tags (table above), e.g. `coll=bcast variant={} datatype=byte launch=host` |
| `sizes` | `sizes m ...` | the size grid in bytes, ascending (4 per octave up to the largest size; one rank's block for the collectives with a block per rank). `curve` and `penalty` values are per grid size, in this order |
| `domain` | `domain {"N":[lo,hi],"ppn":[lo,hi],"m":[lo,hi]}` | the range the model covers: nodes, ranks per node, message size |
| `layouts` | `layouts NxPPN ...` | every allowed layout of the allocation, measured or not (the points colltree fits over) |
| `conditions` | `conditions NAME ...` | the restrictive conditions of the catalog (e.g. `inplace`, `not_inplace`, `pof2`): bit k of every mask is the k-th name |
| `mask` | `mask NxPPN BITS` | the conditions true at that layout, from the precheck, as a decimal bit mask over `conditions`; written only where nonzero (a layout without a `mask` line has 0) |
| `left_out` | `left_out ALG ...` | algorithms left out of the model: unrestricted ones that failed the precheck at some layout (may be empty) |
| `fit_version` | `fit_version K` | 2: the fit stores each algorithm's penalty; 1 (older): the penalty is refitted on load |
| `note` | `note TEXT` | free text to the end of the line, one per note (e.g. a precheck finding) |
| `job` | `job K {json}` | the unit's job K, numbered from 0 in order: one launch of collmodel that measured into the unit (below) |
| `run` | `run {json}` | one collmodel run on the unit, in order: what it did and how it stopped (below) |
| verbatim block | `---- HASH ----` ... `---- end` | an inner selection file: the tuning JSON a launch loaded for the inner calls of compositional algorithms, byte for byte. The label is the first 16 hex digits of its sha256; the reader checks it, so a hand edit cannot detach the measurements that name it |

### Algorithm lines

An `@algo` line starts an algorithm's section; the lines up to the next
`@algo` or `@unit` belong to it (the writer writes them in this order; a
`params` line must come before the `meas` lines that name it):

| Line | Syntax | Meaning |
|---|---|---|
| `@algo` | `@algo NAME` | the algorithm, by its catalog name (e.g. `binomial`, `nb`, `buffer_swap`) |
| `info` | `info {json}` | its catalog entry: `leaf` (the CSEL leaf function, or null for `MPIR_<Coll>_intra_<name>`), `restrictions` (condition names), `mask` (their bits), `inner` (the inner calls of a compositional algorithm, `section@comm[k=v]`), `params` (the leaf parameters at the CVAR values it ran with, below), `param_map` (an algorithm measured at a param map, design.md §14: `rule`, its `version`, `consts`, `hash`, and the calibration's `fit` report, not part of the hash; a map of another rule version than the code's is refused). Absent for an algorithm that was only prechecked |
| `fit_inner` | `fit_inner HASH` | a compositional algorithm: the inner selection file its curves ran under |
| `penalty` | `penalty intra\|inter A ...` or `penalty intra\|inter -` | the non-power-of-two penalty a(m) of its layout model, one value per grid size, for single-node (`intra`) or multi-node (`inter`) layouts; `-`: nothing fitted for that class (below) |
| `curve` | `curve NxPPN V ...` | log2 of the time in µs at every grid size, at one layout where it was sampled: measured sizes exactly, the others interpolated or, where pruned, the cross-layout prediction corrected by the measured residuals |
| `params` | `params K {json}` | parameter set K of its measurements, `{name: value}`, numbered from 0 in the order of first use; a `meas` line names one with `p<K>` |
| `meas` | `meas N PPN M WALL JOB FLAGS INNER T ...` | one measurement (columns below) |
| `pre` | `pre N PPN OK JOB [MSG]` | one precheck: whether the forced algorithm ran at the layout (`OK` 1 or 0), the job, and MPICH's error message (the rest of the line) when it did not |

The columns of `meas`:

| Column | Meaning |
|---|---|
| `N`, `PPN` | the layout: nodes, ranks per node |
| `M` | the message size in bytes (one rank's block for the collectives with a block per rank) |
| `WALL` | the request's full server-side cost in µs: untimed first call, calibration, warm-up and every timed batch (a replay bills it; a live run bills elapsed time) |
| `JOB` | the index of the `job` line that measured it |
| `FLAGS` | `-`, or in this order: `a` a drift anchor (an earlier point measured again), `g` a growth anchor (a fixed point for growth fits), `p<K>` ran with parameter set K (`params` line K). The fit skips `a` and `g` points |
| `INNER` | a compositional algorithm: the hash of the inner selection it ran under; else `-` |
| `T ...` | one value per timed batch: the batch's synchronized latency per call in µs (with `sync=call`, the slowest rank's mean call time; design.md §3.2). The fit takes their median |

A measurement that ran with algorithm parameters set per request (e.g. a
param map's) records them in a `params` line; a file without `params`
lines is one whose measurements all ran at the CVAR defaults, and a
`p<K>` without its `params` line is refused. The `"params"` key in `info`
is something else: the leaf parameters at the CVAR defaults (below).

### The JSON of `job` and `run`

A `job` holds the settings that shaped the job's requests and its
provenance; collmodel writes, among others (older files lack some):
`start`, `end` (Unix seconds), `mode` (`live` or `replay`), `tool`, `op`,
`tags`, `algs`, `exclude_algs`, `mem`, `inplace`, `counts`, `nodes`,
`ppn_max`, `max_m`, `grid_nodes`, `batches`, `sync`, `seed`, `prune`,
`probe_m`, `compositional`, `max_stage`, `layouts`, `anchors`, `growth`,
`base` (hash of the `--model` file it added to), `inner` and
`inner_selection` (hash of the inner selection file), `hosts`, `library`
(MPI_Get_library_version), `launch` (`cmd`, `env`), `topology_warnings`.
`collmodel --resume` refuses a job whose request-shaping settings differ
from the journal's first job (except fewer algorithms).

A `run` has `mode`: `create` (`algs`, `complete`, `stop`, `node_h`,
`n_measurements`, `sets`: per restriction set its algorithms, stop,
layouts, measurements, the builder's settings, estimate and checks;
`growth_anchors`), `add` (`algs`, `layouts`, `anchors`: the drift check,
`stop`, ...) or `import` (`from`, `journal`, `meta`), and `jobs`, the
indices of its jobs.

### Notes

- The unit keeps what its algorithms share (the layouts and their masks,
  the jobs, the runs, the inner selection files); an `@algo` section keeps
  what is one algorithm's own (its catalog entry, its fit, its records).
  These sections are the shape that per-algorithm model files will take.
- Records keep their order within an algorithm, so a unit converts between
  the text and the JSON formats unchanged.
- Measurements are columns instead of JSON objects, so the file is smaller
  than the JSON.
- `collmodel --convert old.json --out new.model` converts a JSON model;
  any load and save does the same.

## Leaf parameters, inner selections, the fit

An algorithm with `extra_params` in the catalog also has `"params"`, the
leaf parameters at the values the server ran with: mpisrv's info record
lists every global int and enum CVAR (`cvar_values`, enums by item name),
and each non-constant parameter takes `MPIR_CVAR_<COLL>_<cvar_param>`, as
MPICH's `maint/gen_coll.py` maps them. colltree writes them into the
leaves, so a tuning file runs an algorithm the way it was measured, also
under `--env` overrides. A boolean comes out as 0 or 1 (MPI_T reports it as
an int), which MPICH's JSON parser accepts since csel-json-bool-int. A
model without `params` (recorded earlier, or replay) gets colltree's
frozen table of the 2026-10 defaults.

In the JSON formats a measurement is an object: `alg`, `N`, `ppn`, `m`,
`t_us`, `wall_us`, `job`, and where they apply `"inner": "<hash>"`,
`"anchor": true`, `"growth": true` and `"params": {name: value}` (the
columns and flags of `meas`).

**The penalty.** An algorithm's model across layouts is a LayoutModel
(design.md §5) over all its curves:

```
T(N, ppn, m) = S(log2 N, log2 ppn, m) x (1 + a(m) s(p)),   p = N ppn
s(p) = 2 - p / 2^floor(log2 p) if p is not a power of two, else 0
```

S interpolates the curves in (log2 N, log2 ppn), single-node layouts only
among themselves. s(p) is 0 at a power of two, close to 1 just above one
and falls back to 0 at the next (s(5) = 0.75, s(6) = 0.5), so a(m) is how
much the algorithm loses just above a power of two at size m (0.5: up to
1.5x; it may be negative). It is fitted per class, `intra` (N = 1) and
`inter` (N > 1), by leave-one-out: for each a in a fixed grid (-0.3 to 3),
each measured layout inside the others' hull is predicted from the others,
the a with the least error is taken per size, then smoothed by a running
median over 3 sizes and clipped to [-0.5, 10]. A class has no penalty (`-`,
JSON `null`) when no such inner layout has p off a power of two. This
search is the only costly part of the fit, so the file stores a(m) and a
load builds the LayoutModel from it; a file of fit version 1 (before the
penalty was stored) refits it on load.

collmodel/2 and /3, written by earlier versions, convert with
`bin/collmodel --import`.

## Param-map files

A param-map file (`bin/algoparam`, design.md §14) is a model file too: one
unit per collective with the extra tag `kind=param_map`, holding the
algorithm's entry with its `param_map` and the calibration's measurements
with their parameters. colltree skips such units; collmodel `--param-map`
reads them.

## collmodel/5 and /4 (JSON, read only)

**collmodel/5** (JSON, read only) holds the same units:

```json
{"format": "collmodel/5", "created": "...",
 "units": [{"tags": {"coll": "allreduce", "variant": {"builtin_op": 1, "commutative": 1, "inplace": 0},
                     "datatype": "int", "launch": "host"},
            "sizes": [...], ...the rest as in a collmodel/4 file, below, without format, op and mem...}]}
```

A unit is self-contained (its jobs, inner files and runs are its own), so
units copy between files unchanged: a collmodel run puts its unit into
`--out` and keeps the file's other units (adding those of `--model` it
lacks, so adding to a model keeps its other units), `collmodel --merge` combines
files, `collmodel --model` and `colltree --unit` select units by their tags.
A collmodel/4 file reads as one unit; new tags get defaults (a unit written
before a tag existed has its default value).

**collmodel/4** (JSON, read only), one unit per file. Its fields are the unit's
fields in every format:

```json
{"format": "collmodel/4", "op": "bcast", "sizes": [...], "domain": {...}, "layouts": [[1, 2], ...],
 "conditions": ["inplace", ...], "algorithms": {"binomial": {"leaf": "...", "restrictions": [], "mask": 0,
                                                              "inner": []}, ...},
 "layout_masks": [[1, 2, 2048], ...], "left_out": ["nb"],
 "data": {"jobs": [{...}], "measurements": [{"alg": "binomial", "N": 2, "ppn": 4, "m": 1024,
                                             "t_us": [...], "wall_us": ..., "job": 0}, ...],
          "prechecks": [{"alg": ..., "N": ..., "ppn": ..., "ok": true, "msg": "", "job": 0}, ...],
          "inner_files": {"<hash>": "<tuning file text>"}},
 "fit": {"version": 2, "inner": {"smp": "<hash>"},
         "curves": {"binomial": [{"N": 2, "ppn": 4, "log2_t": [...]}, ...], ...},
         "penalty": {"binomial": {"intra": [...], "inter": null}, ...}},
 "runs": [{"mode": "create" | "add" | "import", "jobs": [...], "stop": ..., "node_h": ..., ...}]}
```

`"mem"` is the buffers' memory kind, `host` (also when the field is
absent) or `device`.

## collmodel/1 (one algorithm)

`collmodel --alg` (builder v1) writes one algorithm's model as JSON:

**collmodel/1** (one algorithm):

```json
{"format": "collmodel/1", "inputs": ["N", "ppn", "m"], "output": "t_us", "what": "bcast-latency",
 "alg": "binomial",
 "model": {"type": "layout-grid-pof2", "sizes": [1, 2, 3, ...],
           "layouts": [{"N": 2, "ppn": 1, "log2_t": [...]}, ...],
           "penalty": {"form": "1 + a*s", "a_intra": [...], "a_inter": [...]}},
 "domain": {"N": [1, 32], "ppn": [1, 96], "m": [1, 67108864]},
 "jumps": [{"N": 2, "ppn": 1, "lo": ..., "hi": ..., "ratio": ...}, ...],
 "meta": {"created": "...", "tool": "collmodel 0.1", "launch": {...}, "library": "...",
          "n_measurements": ..., "n_layouts": ..., "node_h": ..., "stop": "plateau",
          "estimate": {"med": ..., "p90": ...}, "checks": [...]}}
```

`log2_t` is the measured layout's curve on the size grid; prediction at
other layouts follows design.md §5 (`LayoutModel.from_json`).
