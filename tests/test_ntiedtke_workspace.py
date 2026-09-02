"""The assembler's workspace, and the proof that its boundary grade can fail.

WHY THE LAST PART MATTERS MOST (review). The assembler establishes each
stage's level-indexing base by GRADING rather than by declaring it, because
three static derivations gave three different answers (docs/ntiedtke/PORT-RECORD.md
§32). That design rests entirely on the grade being able to fail.

The mechanism to rule out is **bind-and-read symmetry**: if the read-back
used the same base as the bind, a wrong base would shift the write and
unshift the read, the comparison would pass, and the array would be left
mis-indexed relative to whatever the next stage expects. The grade would be
measuring the round trip instead of the binding -- the harness-agrees-with-
itself shape that has cost this port five instances.

It cannot happen here, and not by care: :meth:`NtWorkspace.levels` always
returns the same rows of the allocation regardless of base, so the read-back
is base-independent **by construction**. ``test_a_wrong_base_fails_a_real
_boundary_grade`` demonstrates it on a graded kernel rather than arguing it.
"""
from __future__ import annotations

import numpy as np
import pytest

from gpuwm.core.ntiedtke import (NT_LEVEL_F, NT_LEVEL_I, NT_SEEDS,
                                 NT_SURFACE_F, NT_SURFACE_I, NtWorkspace)
from gpuwm.verify.ntiedtke_oracle import NT_NZ, load_csv, word


def _ws(ncol=8, nz=NT_NZ):
    pytest.importorskip("cupy")
    return NtWorkspace(ncol=ncol, nz=nz)


def test_every_array_the_kernels_touch_is_allocated():
    """The name set is the kernels', not a list someone maintains.

    A kernel parameter with no workspace array is a stage the assembler
    cannot launch, and it would surface as a KeyError at run time rather
    than here.
    """
    names = set(NT_LEVEL_F) | set(NT_LEVEL_I) | set(NT_SURFACE_F) \
        | set(NT_SURFACE_I)
    assert len(names) == 167, len(names)
    for expected in ("ptu", "pmfu", "zprecc", "rthcuten", "klab", "ldcum"):
        assert expected in names


def test_the_two_bases_agree_on_where_level_one_lives():
    """The whole reason one allocation can serve both conventions.

    A 1-based stage writes level ``jk`` at allocation row ``jk``; a 0-based
    stage writes level ``k`` at row ``k`` of ``arr[1:]``, which is row
    ``k+1`` of the allocation. Level 1 is row 1 either way, which is what
    makes the halo row worth its 108 floats per array.
    """
    w = _ws()
    import cupy as cp
    one, zero = w.bind("ptu", 1), w.bind("ptu", 0)
    one[3, :] = 7.0
    assert float(zero[2, 0]) == 7.0, "the two views disagree by one row"
    assert one.shape[0] == zero.shape[0] + 1
    cp.cuda.Stream.null.synchronize()


def test_levels_is_base_INDEPENDENT():
    """The property the whole grading design rests on.

    ``levels`` takes the same rows whatever base a stage was bound with,
    so a wrong base cannot unshift itself on read-back.
    """
    w = _ws()
    before = w.levels("ptu").data.ptr
    w.bind("ptu", 0)
    w.bind("ptu", 1)
    assert w.levels("ptu").data.ptr == before
    assert w.levels("ptu").shape == (w.nz, w.ncol)
    # And it starts one row into the allocation, never at row 0.
    assert (w.levels("ptu").data.ptr
            == w.bind("ptu", 1)[1:].data.ptr)


def test_aliases_resolve_to_one_array_not_two():
    """``pten`` and ``ztp1`` are the same memory, because they are the same
    array under two of the reference's dummy names."""
    w = _ws()
    assert w.resolve("pten") == "ztp1"
    assert w.resolve("rn") == "zprecc"
    assert w.bind("pten", 1).data.ptr == w.bind("ztp1", 1).data.ptr
    # A chain is followed to its end.
    assert w.resolve("pvom") == "ptenu"
    # And a name that is nobody's alias resolves to itself.
    assert w.resolve("ptu") == "ptu"


def test_an_unknown_name_raises_rather_than_allocating():
    w = _ws()
    with pytest.raises(KeyError, match="not an array in the workspace"):
        w.bind("no_such_array", 1)


def test_a_bad_base_raises():
    w = _ws()
    with pytest.raises(ValueError, match="level base"):
        w.bind("ptu", 2)


def test_the_seed_table_is_the_module_s_and_not_a_copy():
    """One object, and the test suite reads the module's (review).

    A second copy of the seed table would be the failure this port has
    paid for four times, on the table with the widest blast radius in it.
    """
    from tests import test_ntiedtke_call_order_vs_source as gate
    assert gate.SEEDS is NT_SEEDS


# ===========================================================================
# The proof that a wrong base fails a REAL grade
# ===========================================================================

def _post_run_column():
    """One fixture column of cu_ntiedtke_post_run's inputs and answer."""
    key = (1, 1500.0)
    lev = {f: np.zeros(NT_NZ, dtype=np.float32) for f in
           ("exner", "qv", "qc", "qi", "t", "u", "v",
            "tf", "qvf", "qcf", "qif", "uf", "vf")}
    for r in load_csv("nt-post-in-levels.csv"):
        if (int(r["case"]), float(r["dx"])) != key:
            continue
        for f in lev:
            lev[f][int(r["k"]) - 1] = word(r[f])
    sur = {}
    for r in load_csv("nt-post-in-surface.csv"):
        if (int(r["case"]), float(r["dx"])) == key:
            sur = {"rn": word(r["rn"]), "dt": word(r["dt"]),
                   "stepcu": int(r["stepcu"])}
    want = np.zeros(NT_NZ, dtype=np.float32)
    for r in load_csv("nt-post-out-levels.csv"):
        if (int(r["case"]), float(r["dx"])) == key:
            want[int(r["k"]) - 1] = word(r["rthcuten"])
    return lev, sur, want


def _run_post_run(base):
    """Launch post_run through the workspace with a chosen level base."""
    import cupy as cp

    from gpuwm.core.ntiedtke import NtLaunchGeometry, NtStages

    lev, sur, want = _post_run_column()
    w = NtWorkspace(ncol=1, nz=NT_NZ)
    for name, col in lev.items():
        w.levels(name)[:, 0] = cp.asarray(col)
    w.bind("rn", 1)[0] = float(sur["rn"])

    order = ("exner", "qv", "qc", "qi", "t", "u", "v",
             "tf", "qvf", "qcf", "qif", "uf", "vf")
    args = [w.bind(n, base) for n in order]
    args.append(w.bind("rn", base))
    args += [w.bind(n, base) for n in
             ("rthcuten", "rqvcuten", "rqccuten", "rqicuten",
              "rucuten", "rvcuten")]
    args += [w.bind("raincv", base), w.bind("pratec", base)]
    args += [np.int32(1), np.int32(NT_NZ), np.int32(sur["stepcu"]),
             np.float32(float(sur["dt"]))]

    stages = NtStages(NtLaunchGeometry(ncol=1, nz=NT_NZ))
    stages.launch("ntiedtke_post_run", tuple(args))
    cp.cuda.Stream.null.synchronize()
    got = cp.asnumpy(w.levels("rthcuten"))[:, 0]
    return got, want


def test_the_right_base_reproduces_the_oracle():
    """post_run is 1-based, and bound that way it is bitwise."""
    pytest.importorskip("cupy")
    got, want = _run_post_run(1)
    d = np.nonzero(got.view(np.uint32) != want.view(np.uint32))[0]
    assert not d.size, f"levels {d.tolist()[:5]} differ at the correct base"


def test_a_wrong_base_fails_a_real_boundary_grade():
    """THE LOAD-BEARING TEST OF THE WHOLE DESIGN.

    Bind a 1-based stage with base 0 and the grade must fail. If it passed,
    every "the base is established by grading" claim in §32 would be empty,
    and a mis-indexed array would be handed to the next stage with every
    number finite and plausible.

    The failure is not subtle -- a whole column shifted by one level -- but
    subtlety is not the point. The point is that the read-back does not
    unshift it, and that is a property of ``levels`` rather than of care.
    """
    pytest.importorskip("cupy")
    got, want = _run_post_run(0)
    d = np.nonzero(got.view(np.uint32) != want.view(np.uint32))[0]
    assert d.size, (
        "binding a 1-based stage with base 0 still reproduced the oracle. "
        "The boundary grade is measuring the round trip rather than the "
        "binding, so it cannot establish the base and section 32's design "
        "does not hold.")


# ===========================================================================
# The census, and the numbers §26 priced the tile decision on
# ===========================================================================

def test_the_scratch_slot_count_is_DERIVED_from_the_kernel():
    """Not from the signature comment, which was stale by one.

    ``ntiedtke_cutypen`` takes one float pointer and slices it internally.
    Its signature said ``(11, nz+2, ncol)``; the body uses slots 0-9 and
    the graded parity test allocates ten. One stale comment was one whole
    level array -- 3.5 MiB at the capped tile, in a campaign where 50 MiB
    has to earn itself.

    So the count comes off the body. A slot the kernel starts using lands
    here rather than as a silent out-of-bounds read.
    """
    import re
    from pathlib import Path

    from gpuwm.core.ntiedtke import (NT_CUTYPEN_SCR_I_SLOTS,
                                     NT_CUTYPEN_SCR_SLOTS)

    cu = (Path(__file__).resolve().parents[1] / "gpuwm" / "core" / "kernels"
          / "ntiedtke.cu").read_text(encoding="utf-8")
    body = cu[cu.index("void ntiedtke_cutypen("):]
    body = body[:body.index("\n}\n")]
    slots = [int(m) for m in re.findall(r"scr\s*\+\s*(\d+)\s*\*\s*stride", body)]
    assert slots, "the scratch-slot scan found nothing in cutypen"
    assert max(slots) + 1 == NT_CUTYPEN_SCR_SLOTS, (
        f"cutypen uses slots up to {max(slots)}, so it needs "
        f"{max(slots) + 1}; NT_CUTYPEN_SCR_SLOTS says {NT_CUTYPEN_SCR_SLOTS}")
    assert NT_CUTYPEN_SCR_I_SLOTS == 1, "klab is the only int slab"


def test_aliases_are_not_allocated_and_copies_are():
    """Guarantee 2, landing where §7 said it would: with the assembler.

    ``pten`` and ``ztp1`` are ONE array under two of the reference's dummy
    names, so allocating both is dead memory. ``zqq`` and ``ztt`` are
    SNAPSHOTS and must have their own storage -- the post-conversion
    computes ``ptte - ztt``, which is identically zero if they share.
    """
    w = _ws()
    c = w.storage_census()
    assert c["aliased_away"] == 41, c
    assert w.bind("pten", 1).data.ptr == w.bind("ztp1", 1).data.ptr
    assert w.bind("zqq", 1).data.ptr != w.bind("pqte", 1).data.ptr
    assert w.bind("ztt", 1).data.ptr != w.bind("ptte", 1).data.ptr


def test_the_census_is_what_section_26_should_have_priced():
    """The count, measured, against the estimate the tile decision used.

    §26 priced capped-versus-uncapped on "the 75 distinct level arrays the
    assembled scheme holds" -- an estimate made before an assembler existed
    to count them. Measured: 89 level buffers, 37 surface, plus 11
    level-sized scratch slabs cutypen slices internally and which no name
    census could have seen.

    IT HAS FALLEN FOUR TIMES, 107 -> 101 -> 97 -> 89, and every step was a
    bug the pinned count caught. The last was the largest: eleven more
    names turned out to be second names for arrays the reference already
    had, found by walking all twenty stages -- ptent/ptenq are ptte/pqte
    (cudtdqn ACCUMULATES into the forcing), pt/pqv/pqc/pqi/pu/pv are
    tf/qvf/qcf/qif/uf/vf (cu_ntiedtke_run is called with them), idtop is
    kdtop, pmfub is zmfub, kctop0 is ictop0.

    The first two steps:

    * t/qv/qc/qi/u/v are post_run's names for t3d/qv3d/..., both classed
      "driver", both allocated. post_run would have differenced its
      tendencies against six buffers of zeros.
    * cutu/cuqu/culu/culab and ldcum/ictop0 are cutypen's names for arrays
      cuinin filled and mfub updates IN PLACE (cumastrn:490 passes ptu,
      pqu, ilab, plu). Allocated apart, every stage after cutypen would
      have read cuinin's plume instead of cutypen's.

    Neither is visible to the class-1 seed analysis -- the second set is
    never read before written, so NT_SEEDS has nothing to say about it.
    Storage identity is a different question and NT_ALIASES is where it
    is answered.

    The DECISION is unchanged and strengthened -- uncapped measures 2.3 GiB
    on the profile domain against §26's computed 1.5. The capped-versus-GF
    margin is deliberately NOT asserted here: §34 showed GF's figure prices
    its workspace and not its call, and §35 hands the comparison to two
    scheduled runs' peak VRAM rather than to any census.
    """
    c = _ws().storage_census()
    assert c["level"] == 89, c
    assert c["surface"] == 37, c
    assert c["scratch_level_equivalents"] == 11, c
