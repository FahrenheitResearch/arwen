"""The assembled pipeline, graded at EVERY stage boundary.

Phase 1's end condition is that the assembled pipeline reproduces
``nt-levels.csv`` bitwise -- **graded at every capture boundary, not only
end to end**. End-to-end agreement would say "the answer is right"; a
boundary grade says which stage is wrong, and the difference is the whole
reason the capture files exist.

IT IS ALSO WHAT ESTABLISHES THE LEVEL BASE. Four static derivations gave
four answers (docs/ntiedtke/PORT-RECORD.md §32, §35), so ``NT_LEVEL_BASE`` is a PRIOR
and these grades are its test: a wrong base fails at its own stage, by
name, because :meth:`NtWorkspace.levels` reads the same rows whatever base
a stage was bound with (``test_ntiedtke_workspace.py``).

The fixture is 108 columns -- 18 cases across the six-member dx sweep --
which is one chunk. The multi-chunk path is a separate gate; see the note
at the foot of this file.
"""
from __future__ import annotations

import numpy as np
import pytest

from gpuwm.verify.ntiedtke_oracle import NT_NZ, load_csv, word

#: CSV column -> workspace array. Identical for all but one: the driver
#: spells the Exner function ``pi3d`` and the only kernel that reads it
#: (post_run) calls the parameter ``exner``, so ``exner`` IS the array.
_DRIVER_LEV = {"t3d": "t3d", "qv3d": "qv3d", "qc3d": "qc3d", "qi3d": "qi3d",
               "u3d": "u3d", "v3d": "v3d", "pcps": "pcps", "dz8w": "dz8w",
               "rho3d": "rho3d", "pi3d": "exner", "qvften": "qvften",
               "thften": "thften"}
_DRIVER_IFACE = ("p8w", "w")
_PREP_OUT = ("prsl", "ghtl", "omg", "tf", "qvf", "qcf", "qif", "uf", "vf",
             "qvftenz", "thftenz")
_PREP_IFACE_OUT = ("prsi", "ghti")
_CONV_OUT = ("ztp1", "zqp1", "zqsat", "pgeo", "pverv", "ptte", "pqte")

#: stage -> (levels capture, level fields, surface capture, surface fields)
#:
#: THE WALK, one boundary at a time. Each stage is launched from what the
#: previous stages left in the workspace -- nothing is re-seeded from the
#: fixture after the driver inputs -- so a boundary that passes proves the
#: whole chain up to it, and one that fails names the stage.
#:
#: The CSV column name and the workspace array name differ where the
#: reference gives one array two names. cutypen's outputs are the clearest
#: case: it writes ptu/pqu/plu/klab, and the fixture records them as
#: cutu/cuqu/culu/culab precisely so cuinin's answer and cutypen's answer
#: could be told apart in one file.
_WALK = (
    ("ntiedtke_cuinin", "nt-cuinin-levels.csv",
     {"ptenh": "ptenh", "pqenh": "pqenh", "pqsenh": "pqsenh",
      "ptu": "ptu", "pqu": "pqu", "ptd": "ptd", "pqd": "pqd",
      "puu": "puu", "pvu": "pvu", "pud": "pud", "pvd": "pvd",
      "plu": "plu", "klab": "klab"},
     "nt-cuinin-surface.csv", {"klwmin": "klwmin"}),
    ("ntiedtke_cutypen", "nt-cutypen-levels.csv",
     # cutypen writes its OWN arrays -- cutu/cuqu/culu/culab -- and uses
     # scr's slices for the plume it builds internally. cuinin's ptu/pqu/
     # plu/klab are untouched by it, which is why one fixture file can
     # hold both answers.
     {"cutu": "cutu", "cuqu": "cuqu", "culu": "culu", "culab": "culab"},
     "nt-cutypen-surface.csv",
     {"ldcum": "ldcum_o", "ktype": "ktype_o", "cubot": "cubot_o",
      "cutop": "cutop_o", "kdpl": "kdpl_o", "wbase": "wbase_o"}),
    ("ntiedtke_mfub", None, {}, "nt-mfub-surface.csv",
     {"ldcum": "ldcum", "zdhpbl": "zdhpbl", "upbl": "upbl",
      "zmfub": "zmfub"}),
    ("ntiedtke_cuascn", "nt-cuascn-out-levels.csv",
     {"ptu": "ptu", "pqu": "pqu", "plu": "plu", "pmfu": "pmfu",
      "pmfus": "pmfus", "pmfuq": "pmfuq", "pmful": "pmful",
      "plude": "plude", "pdmfup": "pdmfup", "plglac": "plglac",
      "pmfude_rate": "pmfude_rate", "ptenh_out": "ptenh",
      "pqenh_out": "pqenh", "klab": "klab"},
     "nt-cuascn-surface.csv",
     {"ldcum": "ldcum", "ktype": "ktype", "kcbot": "kcbot",
      "kctop": "kctop", "kctop0": "kctop0", "kdpl": "kdpl",
      "wup": "wup", "wbase": "wbase"}),
    # NO OUT-CAPTURE. cloud_depth's outputs (the ktype flip, kctop, kcbot,
    # ldcum) are graded by the NEXT stage's entry capture, which is the
    # same boundary seen from the other side. Listed so it RUNS and so its
    # absence from the grade is visible rather than an omission.
    ("ntiedtke_cloud_depth", None, {}, None, {}),
    ("ntiedtke_cudlfsn", "nt-cudlfsn-out-levels.csv",
     {"ptd": "ptd", "pqd": "pqd", "pud": "pud", "pvd": "pvd",
      "pmfd": "pmfd", "pmfds": "pmfds", "pmfdq": "pmfdq",
      "pdmfdp": "pdmfdp"},
     "nt-cudlfsn-out-surface.csv",
     {"kdtop": "kdtop", "lddraf": "lddraf", "prfl_out": "prfl"}),
    ("ntiedtke_cuddrafn", "nt-downdraft-levels.csv",
     {"zmfd": "pmfd", "zmfds": "pmfds", "zmfdq": "pmfdq",
      "zdmfdp": "pdmfdp", "ztd": "ptd", "zqd": "pqd",
      "pmfdde_rate": "pmfdde_rate"},
     "nt-cuddrafn-out-surface.csv", {"prfl_out": "prfl"}),
    ("ntiedtke_closure", None, {}, "nt-closure-surface.csv",
     {"zheat": "zheat", "zcape": "zcape", "zcape1": "zcape1",
      "zcape2": "zcape2", "ztauc": "ztauc", "ztaubl": "ztaubl",
      "ztau": "ztau_o", "zmfub": "zmfub", "zmfub1": "zmfub1",
      "upbl": "upbl"}),
    # NO OUT-CAPTURE: graded by cuflxn's entry.
    ("ntiedtke_updraft_scale", None, {}, None, {}),
    ("ntiedtke_cuflxn", "nt-cuflxn-out-levels.csv",
     {n: n for n in ("pmfu", "pmfd", "pmfus", "pmfds", "pmfuq", "pmfdq",
                     "pmful", "plglac", "pdmfup", "pdmfdp", "pdpmel",
                     "pmflxr", "pmflxs", "pqsen", "plude",
                     "pmfdde_rate", "pmfude_rate")},
     "nt-cuflxn-out-surface.csv",
     {"kdtop_out": "kdtop", "prain": "prain"}),
    ("ntiedtke_adjust", "nt-adjust-out-levels.csv",
     {n: n for n in ("pmfd", "pmfds", "pmfdq", "pdmfdp", "pdmfup",
                     "pmfdde_rate", "pmfude_rate")},
     "nt-adjust-out-surface.csv", {"prsfc": "prsfc", "pssfc": "pssfc"}),
    ("ntiedtke_cudtdqn", "nt-cudtdqn-out-levels.csv",
     {"ptent": "ptent", "ptenq": "ptenq", "pcte": "pcte"}, None, {}),
    # NO OUT-CAPTURE for either: graded by cududvn's entry.
    ("ntiedtke_momentum_profile", None, {}, None, {}),
    ("ntiedtke_momentum_rescale", None, {}, None, {}),
    ("ntiedtke_cududvn", "nt-cududvn-out-levels.csv",
     {"ptenu": "ptenu", "ptenv": "ptenv"}, None, {}),
    ("ntiedtke_ke_dissipation", "nt-kedis-out-levels.csv",
     {"ptte": "ptte"}, None, {}),
    ("ntiedtke_post_conversion", "nt-postconv-out-levels.csv",
     {n: n for n in ("pqc", "pqi", "pt", "pqv", "pu", "pv")},
     "nt-postconv-out-surface.csv", {"zprecc": "zprecc"}),
    ("ntiedtke_post_run", "nt-post-out-levels.csv",
     {n: n for n in ("rthcuten", "rqvcuten", "rqccuten", "rqicuten",
                     "rucuten", "rvcuten")},
     "nt-post-out-surface.csv", {"raincv": "raincv", "pratec": "pratec"}),
)


def _fixture():
    """Driver inputs and every captured boundary, keyed by column."""
    keys, inp = [], {}
    for r in load_csv("nt-prep-input.csv"):
        key = (int(r["case"]), float(r["dx"]))
        s = inp.setdefault(key, {})
        k = int(r["k"]) - 1
        for csv_name, ws_name in _DRIVER_LEV.items():
            s.setdefault(ws_name, np.zeros(NT_NZ, dtype=np.float32))
            if k < NT_NZ:
                s[ws_name][k] = word(r[csv_name])
        for f in _DRIVER_IFACE:
            s.setdefault(f, np.zeros(NT_NZ + 1, dtype=np.float32))[k] = word(r[f])
    sur = {}
    for r in load_csv("nt-surface.csv"):
        key = (int(r["case"]), float(r["dx"]))
        sur[key] = {"xland": word(r["xland"]), "hfx": word(r["hfx"]),
                    "qfx": word(r["qfx"]), "dx": np.float32(float(r["dx"]))}
    prep = {}
    for r in load_csv("nt-prep-levels.csv"):
        key = (int(r["case"]), float(r["dx"]))
        s = prep.setdefault(key, {})
        k = int(r["k"]) - 1
        for f in _PREP_OUT:
            s.setdefault(f, np.zeros(NT_NZ, dtype=np.float32))[k] = word(r[f])
        for f in _PREP_IFACE_OUT:
            s.setdefault(f, np.zeros(NT_NZ + 1, dtype=np.float32))[k] = word(r[f])
    conv = {}
    for r in load_csv("nt-conv-levels.csv"):
        key = (int(r["case"]), float(r["dx"]))
        s = conv.setdefault(key, {})
        k = int(r["k"]) - 1
        for f in _CONV_OUT:
            s.setdefault(f, np.zeros(NT_NZ, dtype=np.float32))[k] = word(r[f])
    keys = sorted(inp)
    return keys, inp, sur, prep, conv


_KEYS, _INP, _SUR, _PREP, _CONV = _fixture()


@pytest.fixture(scope="module")
def pipeline():
    """One chunk holding the whole fixture, seeded and walked to convert."""
    cp = pytest.importorskip("cupy")

    from gpuwm.core.ntiedtke import NtPipeline
    from gpuwm.verify.ntiedtke_oracle import NT_DT, NT_ITIMESTEP, NT_STEPCU

    p = NtPipeline(ncol=len(_KEYS), nz=NT_NZ, dt=float(NT_DT),
                   stepcu=NT_STEPCU, itimestep=NT_ITIMESTEP)

    def seed(name, per_key, rows):
        a = np.zeros((rows, len(_KEYS)), dtype=np.float32)
        for c, key in enumerate(_KEYS):
            a[:, c] = per_key[key][name][:rows]
        # bind(name, 0) is the allocation from row 1 -- the same anchor
        # levels() uses, one row longer so an (nz+1) interface array fits.
        p.w.bind(name, 0)[:rows, :] = cp.asarray(a)

    for f in _DRIVER_LEV.values():
        seed(f, _INP, NT_NZ)
    for f in _DRIVER_IFACE:
        seed(f, _INP, NT_NZ + 1)
    for f in ("xland", "hfx", "qfx", "dx"):
        p.w.bind(f, 1)[...] = cp.asarray(
            np.array([float(_SUR[k][f]) for k in _KEYS], dtype=np.float32))

    # SNAPSHOT AT EACH BOUNDARY, not at the end. Reading the final state
    # is the pipeline-level form of "a neighbour's capture will do" -- and
    # it hid a real bug for one commit: cutypen updates cuinin's ptu/pqu/
    # plu/klab IN PLACE (cumastrn:490 passes those very arrays), so grading
    # cuinin against the end state grades it against cutypen's answer.
    # It passed only while the two were wrongly allocated apart.
    snaps = {}

    def take(stage, fields, rows):
        snaps[stage] = {ws: p.w.bind(ws, 0)[:rows].get()
                        for ws in fields}

    p.zero_run_head()
    p.run_stage("ntiedtke_prep")
    p.run_stage("ntiedtke_convert")
    p.snapshot_forcing()
    take("ntiedtke_prep", _PREP_OUT + _PREP_IFACE_OUT, NT_NZ + 1)
    take("ntiedtke_convert", _CONV_OUT, NT_NZ)
    entries = {}
    for stage, _, lf, _, sf in _WALK:
        if stage in _ENTRY:
            entries[stage] = {ws: p.w.bind(ws, 0)[:NT_NZ].get()
                              for ws in _ENTRY[stage][1].values()}
        if stage == "ntiedtke_cududvn":
            # cumastrn:1019-1024, before cududvn overwrites pvom/pvol.
            p.snapshot_momentum()
        if stage == "ntiedtke_cuascn":
            # THE CHUNK-WIDE REDUCTION, and its soundness check. klab is
            # cutypen's by now, so this is the population cuascn will
            # actually descend -- not the fixture-wide one section 12
            # measured.
            p.reduce_llo3()
        p.run_stage(stage)
        cp.cuda.Stream.null.synchronize()
        snaps[stage] = {ws: p.w.bind(ws, 0)[:NT_NZ].get()
                        for ws in lf.values()}
        snaps[stage].update({ws: p.w.bind(ws, 1).get() for ws in sf.values()})
    cp.cuda.Stream.null.synchronize()
    p.stages.check_geometry()
    snaps["_entry"] = entries
    return p, snaps


def _load_capture(name, fields, rows):
    """One capture file, keyed by column, in workspace names."""
    out = {}
    for r in load_csv(name):
        key = (int(r["case"]), float(r["dx"]))
        s = out.setdefault(key, {})
        if rows:
            k = int(r["k"]) - 1
            for csv_name, ws_name in fields.items():
                s.setdefault(ws_name, np.zeros(rows, dtype=np.float32))
                s[ws_name][k] = word(r[csv_name]) if not _is_int(r[csv_name]) \
                    else np.float32(int(r[csv_name]))
        else:
            for csv_name, ws_name in fields.items():
                v = r[csv_name]
                s[ws_name] = (np.float32(int(v)) if _is_int(v) else word(v))
    return out


def _is_int(text):
    """The fixture writes integers as decimals and floats as hex words."""
    t = text.strip()
    return not (len(t) == 8 and all(c in "0123456789abcdefABCDEF" for c in t))


#: stage -> (entry levels capture, fields). BOTH SIDES OF A BOUNDARY.
#:
#: An out-capture says a stage computed the right answer FROM WHAT IT WAS
#: GIVEN. An entry capture says it was given the right thing. Grading only
#: the outputs leaves a stage with no out-capture able to corrupt the next
#: one's inputs and be blamed for it -- which is what happened: cuflxn's
#: pdmfdp and pmflxr failed while updraft_scale, which has no out-capture
#: and rescales the very arrays cuflxn reads, passed by not being graded.
_ENTRY = {
    "ntiedtke_uscale": ("nt-uscale-in-levels.csv",
                        {n: n for n in ("pmfu", "pmfus", "pmfuq", "pmful",
                                        "pdmfup", "plude", "pmfude_rate",
                                        "pmfd", "pmfds", "pmfdq", "pdmfdp",
                                        "pmfdde_rate")}),
    "ntiedtke_cuflxn": ("nt-cuflxn-in-levels.csv",
                        {"pten": "pten", "pqen": "pqen", "pqsen": "pqsen",
                         "ptenh": "ptenh", "pqenh": "pqenh",
                         "pmfu": "pmfu", "pmfd": "pmfd", "pmfus": "pmfus",
                         "pmfds": "pmfds", "pmfuq": "pmfuq",
                         "pmfdq": "pmfdq", "pmful": "pmful",
                         "pdmfup": "pdmfup", "pdmfdp": "pdmfdp",
                         "plglac_in": "plglac", "plude_in": "plude",
                         "pmfdde_rate_in": "pmfdde_rate"}),
}
_ENTRY["ntiedtke_updraft_scale"] = _ENTRY.pop("ntiedtke_uscale")

#: Fields defined only on the DEEP arm, and therefore gradeable only
#: there. In the reference these are scalars local to the ``do jl`` loop
#: inside ``if (ldcum .and. ktype == 1)``; on a column that misses the
#: branch a Fortran local retains the PREVIOUS column's value, and the
#: harness captures that. The port's kernel writes nothing, so the two
#: disagree on exactly the columns where neither number means anything.
#:
#: The closure's own parity test has always restricted these
#: (test_ntiedtke_prep_parity.py:1327). This walk graded them everywhere
#: and case 8 failed -- a grading error of mine, not a port defect, and
#: worth recording as one: an ungraded branch and a wrongly-graded one
#: look identical from the failure line.
_DEEP_ONLY = {"zheat", "zcape", "zcape1", "zcape2", "ztauc", "ztaubl",
              "ztau_o", "upbl"}


def _deep_columns():
    """ldcum at cuascn's exit AND ktype == 1 at the closure's entry."""
    ld, kt = {}, {}
    for r in load_csv("nt-cuascn-surface.csv"):
        ld[(int(r["case"]), float(r["dx"]))] = int(r["ldcum"]) != 0
    for r in load_csv("nt-downdraft-surface.csv"):
        kt[(int(r["case"]), float(r["dx"]))] = int(r["ktype_closure"])
    return {k for k in ld if ld[k] and kt.get(k) == 1}


_DEEP = _deep_columns()

_ENTRY_CAPTURES = {stage: _load_capture(f, fields, NT_NZ)
                   for stage, (f, fields) in _ENTRY.items()}

_CAPTURES = {
    stage: (_load_capture(lev, lf, NT_NZ) if lev else {},
            _load_capture(sur, sf, 0) if sur else {})
    for stage, lev, lf, sur, sf in _WALK
}


def _bits(a):
    """Raw words, whatever the dtype -- klab and friends are int32."""
    a = np.asarray(a)
    return a.view(np.uint32) if a.dtype.itemsize == 4 else a


def _grade(got, name, want_per_key, rows):
    bad = []
    for c, key in enumerate(_KEYS):
        g = got[:rows, c]
        e = want_per_key[key][name][:rows]
        if g.dtype != e.dtype:
            e = e.astype(g.dtype)
        d = np.nonzero(_bits(g) != _bits(e))[0]
        if d.size:
            bad.append((key, d.tolist()[:4], float(g[d[0]]), float(e[d[0]])))
    return bad


@pytest.mark.parametrize("field", _PREP_OUT)
def test_prep_boundary_is_bitwise(pipeline, field):
    """Every field ``ntiedtke_prep`` produces, at its own boundary."""
    _, snaps = pipeline
    bad = _grade(snaps["ntiedtke_prep"][field], field, _PREP, NT_NZ)
    assert not bad, f"{field}: {bad[:3]}"


@pytest.mark.parametrize("field", _PREP_IFACE_OUT)
def test_prep_interface_boundary_is_bitwise(pipeline, field):
    """``prsi`` and ``ghti`` are (nz+1) -- one row longer than the rest.

    The fixture captured only nz of them, so nz is what is graded; the
    surface interface row is graded through cuinin's use of it.
    """
    _, snaps = pipeline
    bad = _grade(snaps["ntiedtke_prep"][field], field, _PREP, NT_NZ)
    assert not bad, f"{field}: {bad[:3]}"


@pytest.mark.parametrize("field", _CONV_OUT)
def test_convert_boundary_is_bitwise(pipeline, field):
    """The second boundary, and the one that settles convert's base.

    The shape scan said ``convert`` was 1-based because the parity test
    allocates 51 rows; its body walks k = 0..nz-1. This grade is what
    decides, and it decides by failing if the prior is wrong.
    """
    _, snaps = pipeline
    bad = _grade(snaps["ntiedtke_convert"][field], field, _CONV, NT_NZ)
    assert not bad, f"{field}: {bad[:3]}"


@pytest.mark.parametrize(
    "stage,field",
    [(stage, ws) for stage, _, lf, _, _ in _WALK for ws in lf.values()])
def test_walked_level_boundary_is_bitwise(pipeline, stage, field):
    """Launched from what the previous stages left, not from a re-seed."""
    _, snaps = pipeline
    want, _ = _CAPTURES[stage]
    bad = _grade(snaps[stage][field], field, want, NT_NZ)
    assert not bad, f"{stage} / {field}: {bad[:3]}"


@pytest.mark.parametrize(
    "stage,field",
    [(stage, ws) for stage, _, _, _, sf in _WALK for ws in sf.values()])
def test_walked_surface_boundary_is_bitwise(pipeline, stage, field):
    _, snaps = pipeline
    _, want = _CAPTURES[stage]
    got = snaps[stage][field]
    if field in _DEEP_ONLY:
        assert len(_DEEP) >= 10, (
            f"only {len(_DEEP)} deep columns; {field} would be graded on "
            f"almost nothing")
    bad = []
    for c, key in enumerate(_KEYS):
        if field in _DEEP_ONLY and key not in _DEEP:
            continue
        g = np.asarray(got[c])
        e = np.asarray(want[key][field]).astype(g.dtype)
        if _bits(g.reshape(1)) != _bits(e.reshape(1)):
            bad.append((key, float(g), float(e)))
    assert not bad, f"{stage} / {field}: {bad[:3]}"


@pytest.mark.parametrize(
    "stage,field",
    [(st, ws) for st, (_, f) in _ENTRY.items() for ws in f.values()])
def test_stage_ENTRY_is_bitwise(pipeline, stage, field):
    """The other side of the boundary: was the stage given the right thing?

    This is where a stage with no out-capture stops being able to corrupt
    its successor invisibly.
    """
    _, snaps = pipeline
    bad = _grade(snaps["_entry"][stage][field], field,
                 _ENTRY_CAPTURES[stage], NT_NZ)
    assert not bad, f"{stage} entry / {field}: {bad[:3]}"


def test_the_walk_covers_the_stages_it_claims_to(pipeline):
    """A vacuity guard on the walk itself.

    Every stage in _WALK must have actually launched, in order. The order
    report is the device's own record, so this cannot be satisfied by a
    table that merely lists them.
    """
    from gpuwm.core.ntiedtke import NT_STAGE_NAMES

    p, _ = pipeline
    ids = {name: sid for sid, name in
           ((sid, "ntiedtke_" + n) for sid, n in NT_STAGE_NAMES.items())}
    expected = ["ntiedtke_prep", "ntiedtke_convert"] + [s for s, *_ in _WALK]
    p.stages.check_order([ids[name] for name in expected])


def test_the_fixture_is_one_chunk_and_the_multi_chunk_path_is_untested():
    """Stated, because the gate that ends Phase 1 cannot reach it.

    The cap is 17,920 columns and the fixture is 108, so every run of this
    file executes exactly ONE chunk. Workspace reuse between chunks, llo3
    being recomputed per chunk rather than leaking, and no column's state
    surviving into the next chunk's lane are all invisible here -- and WRF
    has no analogue to disagree with, because its decomposition is its own
    (review).

    The gate for that needs no oracle: the same 108 columns at caps of 32,
    64 and 108 must give byte-identical output. It is valid on this fixture
    only because §12's precondition makes llo3 invariant under re-chunking,
    and the day that precondition stops holding the chunking test stops
    being valid too.
    """
    from gpuwm.core.ntiedtke import NtWorkspace  # noqa: F401

    assert len(_KEYS) == 108
    assert len(_KEYS) < 17920, "the fixture would now span chunks"
