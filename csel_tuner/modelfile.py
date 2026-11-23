"""The text model file (collmodel/6, extension .model): the reader and writer.

A collmodel/6 file holds the same units as a collmodel/5 JSON file (model.py), as line-oriented
text that grep, diff, less and awk can read and a user can annotate:

    collmodel/6 created=2026-11-22T10:00:00Z
    @unit coll=bcast datatype=byte launch=host variant={}
    sizes 1 2 4 8
    domain {"N":[1,8],"ppn":[1,12],"m":[1,8]}
    layouts 1x2 1x4 2x4
    conditions inplace pof2
    mask 2x4 2                          # nonzero layout masks
    left_out nb
    fit_version 2
    note <text>                         # one per note
    job 0 {"start":...}                 # the unit's jobs, by index
    run {"mode":"create",...}           # one per collmodel run
    ---- 3fa2c1d09b7e4a55 ----          # an inner selection file, verbatim, keyed by its hash
    {...}
    ---- end
    @algo binomial                      # one section per algorithm
    info {"leaf":null,"restrictions":[],"mask":0,"inner":[]}
    fit_inner 3fa2c1d09b7e4a55
    penalty intra 0.0 0.1 ...           # or "-": nothing fitted
    penalty inter -
    curve 2x4 2.181421 2.190931 ...     # log2 T on the size grid
    meas 2 4 1 4406.8 0 - - 269.7395 269.521 ...
    pre 1 2 1 0 <msg>

Lines: blank lines and lines whose first non-blank character is # are ignored; fields are
separated by runs of blanks; a # elsewhere is ordinary text (no trailing comments in data).
Names (algorithms, conditions) contain no blanks; the writer refuses one that does. Small
structured values are one-line JSON. A verbatim block runs from "---- <label> ----" to a line
"---- end" and keeps everything between, byte for byte (its text always ends with a newline); the
reader checks an inner file's hash against its label.

meas: N ppn m wall_us job flags inner t_us..., flags "a" (drift anchor), "g" (growth anchor) or
"-", inner the hash of the inner selection or "-". pre: N ppn ok(1/0) job msg, msg the rest of
the line.

Jobs and inner files belong to the unit: a job measures many algorithms and an inner selection can
serve several. An @algo section holds what belongs to one algorithm: its catalog entry (info, absent
for an algorithm only prechecked), its fit and its records. Records keep their order within an
algorithm; model.Model.to_dict lists them grouped by algorithm in the same way, so a unit converts
between the two formats without change.
"""
import gzip
import json
import re

from .model import FIT_VERSION, file_hash

MAGIC = "collmodel/6"
_BARE = re.compile(r"^[A-Za-z0-9_.+:-]+$")


def is_text_model(head):
    """Whether the first bytes of a file are a collmodel/6 file (gzip is unwrapped by read)."""
    return head.lstrip()[:len(MAGIC)] == MAGIC.encode()


def read(path):
    """The units (as collmodel/5 unit dicts, for model.Model.from_dict) and the created stamp of a
    .model file, gzipped or not."""
    raw = open(path, "rb").read()
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    return loads(raw.decode(), path)


def write(path_or_file, units, created):
    """units: collmodel/5 unit dicts (Model.to_dict)."""
    text = dumps(units, created)
    if hasattr(path_or_file, "write"):
        path_or_file.write(text)
    else:
        with open(path_or_file, "w") as f:
            f.write(text)


# ---- writer --------------------------------------------------------------


def dumps(units, created):
    out = ["%s created=%s" % (MAGIC, created),
           "# meas: N ppn m wall_us job flags(a: anchor, g: growth) inner t_us...",
           "# pre: N ppn ok job msg"]
    for u in units:
        out += _unit_lines(u)
    return "\n".join(out) + "\n"


def _js(v):
    return json.dumps(v, separators=(",", ":"))


def _num(x):
    """A number as the JSON writer would give it, so text and JSON round trips agree."""
    return json.dumps(x)


def _name(s, what):
    s = str(s)
    if not s or any(c.isspace() for c in s):
        raise ValueError("%s %r: a name in a .model file must be non-empty and have no blanks" % (what, s))
    return s


def _lay(n, ppn):
    return "%dx%d" % (n, ppn)


def _bare_json(s):
    """A bare word that reads as JSON (a number, true, false, null), else None."""
    try:
        return json.loads(s)
    except ValueError:
        return None


def _tag_value(v):
    """A string as a bare word where it reads back as that string, else JSON."""
    if isinstance(v, str) and _BARE.match(v) and _bare_json(v) is None and v != "null":
        return v
    return _js(v)


def _unit_lines(u):
    tags = u["tags"]
    out = ["@unit " + " ".join("%s=%s" % (_name(k, "tag"), _tag_value(tags[k])) for k in tags)]
    out.append(" ".join(["sizes"] + [_num(m) for m in u["sizes"]]))
    out.append("domain " + _js(u["domain"]))
    out.append(" ".join(["layouts"] + [_lay(*l) for l in u["layouts"]]))
    out.append(" ".join(["conditions"] + [_name(c, "condition") for c in u["conditions"]]))
    out += ["mask %s %d" % (_lay(n, q), m) for n, q, m in u["layout_masks"] if m]
    out.append(" ".join(["left_out"] + [_name(a, "algorithm") for a in u["left_out"]]))
    fit = u["fit"]
    out.append("fit_version %d" % fit["version"])
    for n in u["notes"]:
        if "\n" in n:
            raise ValueError("note %r: a note in a .model file is one line" % n)
        out.append(("note " + n).rstrip())
    data = u["data"]
    out += ["job %d %s" % (i, _js(j)) for i, j in enumerate(data["jobs"])]
    out += ["run " + _js(r) for r in u["runs"]]
    for h, text in data["inner_files"].items():
        if file_hash(text) != h or not text.endswith("\n"):
            raise ValueError("inner file %s: its hash does not match, or it lacks a final newline" % h)
        out += ["---- %s ----" % h, text[:-1], "---- end"]

    algs = list(u["algorithms"])
    seen = set(algs)
    for r in data["measurements"] + data["prechecks"]:
        if r["alg"] not in seen:
            seen.add(r["alg"])
            algs.append(r["alg"])
    for a in list(fit["curves"]) + list(fit.get("penalty", {})) + list(fit["inner"]):
        if a not in seen:
            seen.add(a)
            algs.append(a)
    meas, pre = {}, {}
    for r in data["measurements"]:
        meas.setdefault(r["alg"], []).append(r)
    for r in data["prechecks"]:
        pre.setdefault(r["alg"], []).append(r)
    for a in algs:
        out.append("@algo " + _name(a, "algorithm"))
        if a in u["algorithms"]:
            out.append("info " + _js(u["algorithms"][a]))
        if a in fit["inner"]:
            out.append("fit_inner " + _name(fit["inner"][a], "hash"))
        p = fit.get("penalty", {}).get(a)
        if p is not None:
            for cls in ("intra", "inter"):
                v = p[cls]
                out.append(" ".join(["penalty", cls] + (["-"] if v is None else [_num(x) for x in v])))
        for c in fit["curves"].get(a, []):
            out.append(" ".join(["curve", _lay(c["N"], c["ppn"])] + [_num(x) for x in c["log2_t"]]))
        for r in meas.get(a, []):
            out.append(_meas_line(r))
        for r in pre.get(a, []):
            if "\n" in r["msg"]:
                raise ValueError("precheck message %r: one line in a .model file" % r["msg"])
            out.append(("pre %d %d %d %d %s" % (r["N"], r["ppn"], 1 if r["ok"] else 0, r["job"],
                                                r["msg"])).rstrip())
    return out


_MEAS_KEYS = {"alg", "N", "ppn", "m", "t_us", "wall_us", "job", "inner", "anchor", "growth"}


def _meas_line(r):
    extra = set(r) - _MEAS_KEYS
    if extra:
        raise ValueError("measurement %r: keys %s have no column in a .model file" % (r, sorted(extra)))
    flags = ("a" if r.get("anchor") else "") + ("g" if r.get("growth") else "")
    return " ".join(["meas", "%d" % r["N"], "%d" % r["ppn"], _num(r["m"]), _num(r["wall_us"]), "%d" % r["job"],
                     flags or "-", _name(r["inner"], "hash") if r.get("inner") else "-"]
                    + [_num(t) for t in r["t_us"]])


# ---- reader --------------------------------------------------------------


class ParseError(ValueError):
    pass


def loads(text, path="<model>"):
    """[unit dict], created."""
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    units, created = [], None
    u = alg = None
    i = 0

    def fail(msg):
        raise ParseError("%s:%d: %s" % (path, i + 1, msg))

    while i < len(lines):
        raw = lines[i]
        line = raw.strip()
        if not line or line.startswith("#"):
            i += 1
            continue
        if created is None:
            if not line.startswith(MAGIC):
                fail("not a %s file" % MAGIC)
            created = _keyvals(line[len(MAGIC):], fail).get("created", "")
            i += 1
            continue
        if line.startswith("---- ") and line.endswith(" ----") and len(line) > 10:
            label, start = line[5:-5].strip(), i
            body = []
            i += 1
            while i < len(lines) and lines[i].rstrip() != "---- end":
                body.append(lines[i])
                i += 1
            if i == len(lines):
                i = start
                fail("block %s has no '---- end'" % label)
            if u is None:
                i = start
                fail("block %s outside a unit" % label)
            content = "".join(b + "\n" for b in body)
            if file_hash(content) != label:
                i = start
                fail("inner file %s: its content hashes to %s (edited?); measurements name it by the label"
                     % (label, file_hash(content)))
            u["data"]["inner_files"][label] = content
            i += 1
            continue
        key, rest = (line.split(None, 1) + [""])[:2]
        f = rest.split()
        try:
            if key == "@unit":
                u = _new_unit(_keyvals(rest, fail))
                units.append(u)
                alg = None
            elif u is None:
                fail("%s before any @unit" % key)
            elif key == "@algo":
                if len(f) != 1:
                    fail("@algo takes one name")
                alg = f[0]
            elif alg is None:
                _unit_line(u, key, rest, f, fail)
            else:
                _algo_line(u, alg, key, rest, f, fail)
        except (ValueError, IndexError, KeyError) as e:
            if isinstance(e, ParseError):
                raise
            fail("%s: %s" % (key, e))
        i += 1
    if created is None:
        raise ParseError("%s: empty, not a %s file" % (path, MAGIC))
    return [_finish(u, path) for u in units], created


def _keyvals(text, fail):
    """'a=b c={"x":1}' -> {"a": "b", "c": {"x": 1}}: a value starting with {, [ or " is JSON,
    else a bare word."""
    out, s, dec = {}, text.strip(), json.JSONDecoder()
    while s:
        k, eq, s = s.partition("=")
        k = k.strip()
        if not eq or not k or any(c.isspace() for c in k):
            fail("expected KEY=VALUE at %r" % (k + eq + s)[:40])
        if s[:1] in ("{", "[", '"'):
            v, end = dec.raw_decode(s)
            s = s[end:]
        else:
            m = re.match(r"(\S*)(.*)", s, re.S)
            v, s = m.group(1), m.group(2)
            j = _bare_json(v)
            v = v if j is None and v != "null" else j
        out[k] = v
        s = s.lstrip()
    return out


def _new_unit(tags):
    return {"tags": tags, "sizes": [], "domain": {}, "layouts": [], "conditions": [], "algorithms": {},
            "layout_masks": [], "left_out": [], "notes": [],
            "data": {"jobs": [], "measurements": [], "prechecks": [], "inner_files": {}},
            "fit": {"version": FIT_VERSION, "inner": {}, "curves": {}, "penalty": {}}, "runs": [],
            "_masks": {}}


def _parse_lay(s):
    n, x, q = s.partition("x")
    if not x:
        raise ValueError("layout %r is not NxPPN" % s)
    return [int(n), int(q)]


def _number(s):
    """A number as the writer gives it (json.dumps): an int unless it has a point or an exponent
    (so "nan" and "inf" are refused)."""
    return float(s) if "." in s or "e" in s or "E" in s else int(s)


def _unit_line(u, key, rest, f, fail):
    if key == "sizes":
        u["sizes"] = [_number(x) for x in f]
    elif key == "domain":
        u["domain"] = json.loads(rest)
    elif key == "layouts":
        u["layouts"] = [_parse_lay(x) for x in f]
    elif key == "conditions":
        u["conditions"] = f
    elif key == "mask":
        u["_masks"][tuple(_parse_lay(f[0]))] = int(f[1])
    elif key == "left_out":
        u["left_out"] = f
    elif key == "fit_version":
        u["fit"]["version"] = int(f[0])
    elif key == "note":
        u["notes"].append(rest)
    elif key == "job":
        if int(f[0]) != len(u["data"]["jobs"]):
            fail("job %s out of order (expected %d)" % (f[0], len(u["data"]["jobs"])))
        u["data"]["jobs"].append(json.loads(rest[len(f[0]):]))
    elif key == "run":
        u["runs"].append(json.loads(rest))
    else:
        fail("unknown unit line %r" % key)


def _algo_line(u, alg, key, rest, f, fail):
    fit = u["fit"]
    if key == "info":
        u["algorithms"][alg] = json.loads(rest)
    elif key == "fit_inner":
        fit["inner"][alg] = f[0]
    elif key == "penalty":
        if f[0] not in ("intra", "inter"):
            fail("penalty class %r (intra or inter)" % f[0])
        fit["penalty"].setdefault(alg, {"intra": None, "inter": None})[f[0]] = \
            None if f[1:] == ["-"] else [_number(x) for x in f[1:]]
    elif key == "curve":
        n, q = _parse_lay(f[0])
        fit["curves"].setdefault(alg, []).append({"N": n, "ppn": q, "log2_t": [_number(x) for x in f[1:]]})
    elif key == "meas":
        if len(f) < 8:
            fail("meas needs N ppn m wall_us job flags inner t_us...")
        r = {"alg": alg, "N": int(f[0]), "ppn": int(f[1]), "m": _number(f[2]),
             "t_us": [_number(x) for x in f[7:]], "wall_us": _number(f[3]), "job": int(f[4])}
        if f[6] != "-":
            r["inner"] = f[6]
        if set(f[5]) - set("ag-"):
            fail("meas flags %r (a, g or -)" % f[5])
        if "a" in f[5]:
            r["anchor"] = True
        if "g" in f[5]:
            r["growth"] = True
        u["data"]["measurements"].append(r)
    elif key == "pre":
        parts = rest.split(None, 4)
        if len(parts) < 4 or parts[2] not in ("0", "1"):
            fail("pre needs N ppn ok(0/1) job [msg]")
        u["data"]["prechecks"].append({"alg": alg, "N": int(parts[0]), "ppn": int(parts[1]), "ok": parts[2] == "1",
                                       "msg": parts[4] if len(parts) > 4 else "", "job": int(parts[3])})
    else:
        fail("unknown @algo line %r" % key)


def _finish(u, path):
    """The unit dict as model.Model.from_dict takes it (masks over the layouts, 0 by default)."""
    masks = u.pop("_masks")
    for lay in masks:
        if list(lay) not in u["layouts"]:
            raise ParseError("%s: unit %s: mask %s is not one of its layouts" % (path, u["tags"], _lay(*lay)))
    u["layout_masks"] = [[n, q, masks.get((n, q), 0)] for n, q in u["layouts"]]
    return u
