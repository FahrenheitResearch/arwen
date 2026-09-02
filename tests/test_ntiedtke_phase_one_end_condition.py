"""Phase 1's end condition, stated over the artifact it names.

**The assembled pipeline reproduces ``nt-levels.csv`` bitwise.** Not a
kernel count -- that number has three times measured a smaller thing than
it appeared to (docs/ntiedtke/PORT-RECORD.md §29). This file runs the twenty stages
from the driver inputs and compares the eight tendency columns of
``nt-levels.csv`` and the ``scale_fac``/``scale_fac2``/``raincv``/
``pratec`` columns of ``nt-surface.csv`` against the real WRF driver's own
answer.

AND THE CHUNKING GATE, which no oracle can provide. The cap is 17,920
columns and the fixture is 108, so the end-to-end run above executes
exactly ONE chunk: workspace reuse between chunks, ``llo3`` recomputed
rather than leaked, and no column's state surviving into the next chunk's
lane are all invisible to it. WRF has nothing to disagree with, because
its decomposition is its own (review).

The gate needs no oracle at all: the same 108 columns at caps of 32, 64
and 108 must give **byte-identical** output. Three caps rather than one
makes the claim *"chunking does not change the answer"* rather than *"this
chunking is safe"*.

**IT IS VALID ON THIS FIXTURE FOR A STATED REASON.** ``llo3`` is
chunk-wide, so re-chunking changes its population and could legitimately
change the answer. It cannot here: §12's precondition gate asserts that
every column entering cuascn with ``ldcum`` true carries
``klab(klev) > 0`` -- so ``llo3`` is true for any chunk containing any
such column, including a chunk of one, and is invariant under
re-chunking. (Not "108 of 108": the fixture is 108 columns but the
precondition is over the ones that trigger, and reading it over the wrong
population is how the first draft of the test below reported 60
violations of a property that holds.) **The day
that precondition stops holding, this test stops being valid**, and
:func:`test_the_chunking_gate_rests_on_the_llo3_precondition` is what says
so rather than leaving it to be rediscovered.
"""
from __future__ import annotations

import numpy as np
import pytest

from gpuwm.verify.ntiedtke_oracle import (NT_DT, NT_ITIMESTEP, NT_NZ,
                                          NT_STEPCU, load_csv, word)

_DRIVER = {"t3d": "t3d", "qv3d": "qv3d", "qc3d": "qc3d", "qi3d": "qi3d",
           "u3d": "u3d", "v3d": "v3d", "pcps": "pcps", "dz8w": "dz8w",
           "rho3d": "rho3d", "pi3d": "exner", "qvften": "qvften",
           "thften": "thften"}
_IFACE = ("p8w", "w")
_TEND = ("rthcuten", "rqvcuten", "rqccuten", "rqicuten", "rucuten",
         "rvcuten")
_SFC = ("scale_fac", "scale_fac2", "raincv", "pratec")


def _fixture():
    inp, want_lev, want_sfc, sur = {}, {}, {}, {}
    for r in load_csv("nt-levels.csv"):
        key = (int(r["case"]), float(r["dx"]))
        k = int(r["k"]) - 1
        d = inp.setdefault(key, {})
        for csv_name, ws in _DRIVER.items():
            d.setdefault(ws, np.zeros(NT_NZ, dtype=np.float32))
            if k < NT_NZ:
                d[ws][k] = word(r[csv_name])
        for f in _IFACE:
            d.setdefault(f, np.zeros(NT_NZ + 1, dtype=np.float32))[k] = \
                word(r[f])
        w = want_lev.setdefault(key, {})
        for f in _TEND:
            w.setdefault(f, np.zeros(NT_NZ, dtype=np.float32))
            if k < NT_NZ:
                w[f][k] = word(r[f])
    for r in load_csv("nt-surface.csv"):
        key = (int(r["case"]), float(r["dx"]))
        sur[key] = {"xland": word(r["xland"]), "hfx": word(r["hfx"]),
                    "qfx": word(r["qfx"]), "dx": np.float32(float(r["dx"]))}
        want_sfc[key] = {f: word(r[f]) for f in _SFC}
    return sorted(inp), inp, sur, want_lev, want_sfc


_KEYS, _INP, _SUR, _WANT_LEV, _WANT_SFC = _fixture()


def _run(keys):
    """The twenty stages, once, over exactly these columns."""
    import cupy as cp

    from gpuwm.core.ntiedtke import NtPipeline
    from tests.test_ntiedtke_pipeline_boundaries import _WALK

    p = NtPipeline(ncol=len(keys), nz=NT_NZ, dt=float(NT_DT),
                   stepcu=NT_STEPCU, itimestep=NT_ITIMESTEP)

    def seed(name, rows):
        a = np.zeros((rows, len(keys)), dtype=np.float32)
        for c, key in enumerate(keys):
            a[:, c] = _INP[key][name][:rows]
        p.w.bind(name, 0)[:rows, :] = cp.asarray(a)

    for f in _DRIVER.values():
        seed(f, NT_NZ)
    for f in _IFACE:
        seed(f, NT_NZ + 1)
    for f in ("xland", "hfx", "qfx", "dx"):
        p.w.bind(f, 1)[...] = cp.asarray(
            np.array([float(_SUR[k][f]) for k in keys], dtype=np.float32))

    p.zero_run_head()
    p.run_stage("ntiedtke_prep")
    p.run_stage("ntiedtke_convert")
    p.snapshot_forcing()
    for stage, *_ in _WALK:
        if stage == "ntiedtke_cuascn":
            p.reduce_llo3()
        if stage == "ntiedtke_cududvn":
            p.snapshot_momentum()
        p.run_stage(stage)
    cp.cuda.Stream.null.synchronize()
    p.stages.check_geometry()
    out = {f: cp.asnumpy(p.w.bind(f, 0)[:NT_NZ]) for f in _TEND}
    out.update({f: cp.asnumpy(p.w.bind(f, 1)) for f in _SFC})
    return out


@pytest.fixture(scope="module")
def whole():
    pytest.importorskip("cupy")
    return _run(_KEYS)


@pytest.mark.parametrize("field", _TEND)
def test_the_pipeline_reproduces_nt_levels_bitwise(whole, field):
    """PHASE 1'S END CONDITION, on the file it names."""
    bad = []
    for c, key in enumerate(_KEYS):
        g = whole[field][:, c]
        e = _WANT_LEV[key][field]
        d = np.nonzero(g.view(np.uint32) != e.view(np.uint32))[0]
        if d.size:
            bad.append((key, d.tolist()[:4], float(g[d[0]]),
                        float(e[d[0]])))
    assert not bad, f"{field}: {bad[:3]}"


@pytest.mark.parametrize("field", _SFC)
def test_the_pipeline_reproduces_nt_surface_bitwise(whole, field):
    bad = []
    for c, key in enumerate(_KEYS):
        g = np.float32(whole[field][c])
        e = np.float32(_WANT_SFC[key][field])
        if g.view(np.uint32) != e.view(np.uint32):
            bad.append((key, float(g), float(e)))
    assert not bad, f"{field}: {bad[:3]}"


def test_the_tendencies_are_not_all_zero(whole):
    """Green must not come from a pipeline that produced nothing.

    The most embarrassing way for an end-to-end gate to pass is for both
    sides to be zero, and 66 of 108 fixture columns do not trigger.
    """
    heated = sum(1 for c in range(len(_KEYS))
                 if np.any(whole["rthcuten"][:, c] != 0))
    assert heated >= 20, f"only {heated} columns have any heating"
    assert float(np.abs(whole["rthcuten"]).max()) > 1e-6


# ===========================================================================
# The chunking gate
# ===========================================================================

@pytest.mark.parametrize("cap", [32, 64, 108])
def test_chunking_does_not_change_the_answer(whole, cap):
    """Byte-identical output at every cap. No oracle involved.

    This is the one property WRF cannot arbitrate: its decomposition is
    its own, and there is no Fortran analogue of "did the workspace
    survive being reused across chunk boundaries". Three caps rather than
    one turns "this chunking is safe" into "chunking does not change the
    answer".
    """
    pytest.importorskip("cupy")
    pieces = [_KEYS[i:i + cap] for i in range(0, len(_KEYS), cap)]
    assert len(pieces) == (1 if cap >= len(_KEYS) else -(-len(_KEYS) // cap))
    got = {f: [] for f in _TEND + _SFC}
    for piece in pieces:
        out = _run(piece)
        for f in _TEND:
            got[f].append(out[f])
        for f in _SFC:
            got[f].append(out[f])
    bad = []
    for f in _TEND:
        joined = np.concatenate(got[f], axis=1)
        d = np.nonzero(joined.view(np.uint32)
                       != whole[f].view(np.uint32))
        if d[0].size:
            bad.append((f, int(d[0][0]), int(d[1][0])))
    for f in _SFC:
        joined = np.concatenate(got[f])
        d = np.nonzero(joined.view(np.uint32) != whole[f].view(np.uint32))[0]
        if d.size:
            bad.append((f, "surface", int(d[0])))
    assert not bad, (
        f"cap={cap} changes the answer at {bad[:4]}. Chunking must be "
        f"invisible: a column's result cannot depend on who it shares a "
        f"chunk with.")


def test_the_chunking_gate_rests_on_the_llo3_precondition():
    """WHY the test above is valid here, and when it stops being.

    ``llo3`` is a chunk-wide monotone OR-reduction, so re-chunking changes
    its population and could legitimately change the answer. It cannot on
    this fixture because every ``ldcum`` column carries ``klab(klev) > 0``
    -- so ``llo3`` is true for any chunk containing any triggering column,
    including a chunk of one.

    THE DAY THAT STOPS HOLDING, the chunking gate stops being valid, and
    this is the paragraph that says so rather than leaving it to be
    rediscovered (review). It reads the precondition from the fixture
    rather than trusting §12's recorded count.
    """
    # BOTH READ AT cuascn's ENTRY. The first version of this took ldcum
    # from nt-cuascn-surface.csv, which is cuascn's EXIT -- a different
    # population, since cuascn itself sets ldcum -- and klab from cuinin's
    # exit, which cutypen then rewrites. It reported 60 violations of a
    # precondition that holds. Reading a boundary's state from the wrong
    # side of the boundary is the error this whole file exists to prevent,
    # committed in the test that checks it.
    ld, klab_top = {}, {}
    for r in load_csv("nt-cuascn-in2-surface.csv"):
        ld[(int(r["case"]), float(r["dx"]))] = int(r["ldcum"]) != 0
    for r in load_csv("nt-cuascn-in-levels.csv"):
        if int(r["k"]) == NT_NZ:
            klab_top[(int(r["case"]), float(r["dx"]))] = int(r["klab"])
    triggering = [k for k, v in ld.items() if v]
    assert len(triggering) >= 20, f"only {len(triggering)} trigger"
    violations = [k for k in triggering if klab_top.get(k, 0) <= 0]
    assert not violations, (
        f"{len(violations)} triggering columns have klab(klev) == 0, so "
        f"llo3 would latch DURING cuascn's descent and is no longer "
        f"invariant under re-chunking. The chunking gate above is invalid "
        f"until this is resolved: {violations[:4]}")
