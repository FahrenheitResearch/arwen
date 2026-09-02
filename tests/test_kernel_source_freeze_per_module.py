"""Every CUDA kernel source pinned by SHA-256, ONE TEST NODE PER MODULE.

THE BREAKAGE THIS PREVENTS
--------------------------
The 2026-08-28 fault-injection audit disabled a physics process in a CUDA
kernel -- WSM6 rain accretion of cloud water, ``pracw = 0.0f * fminf(...)`` in
``gpuwm/core/kernels/wsm6.cu`` -- and ran the ENTIRE 172-file stage-1 leg
twice, pristine and injected, one pytest process per file:

    pristine tip : 37.5 min, 12 files rc!=0, 13 failing node ids
    injected     : 36.8 min, 12 files rc!=0, 13 failing node ids

Zero files differed.  Zero test node ids differed.  Time to detect: never.

Two separate causes, and this file addresses both.

1.  ``tests/test_mp8_frozen.py`` holds the digests, and it is NOT on
    ``tools/battery/stage1_files.txt``.  The list's own header (line 1026)
    says why: nine kernel modules had drifted from ``FROZEN_MODULE_DIGESTS``,
    so the file is red, and "a known-red file on a per-cut list buys a red
    battery, not a fixed test".  The same header writes "a frozen digest
    nothing checks is not a freeze."

2.  Even run directly it does not DISCRIMINATE.
    ``test_every_frozen_kernel_module_is_unchanged`` collects every drifted
    module into one dict and asserts once, so under the injected fault it
    fails with the IDENTICAL two node ids as the clean baseline.  Only the
    assertion message grows a key.  No exit code, no node id, no count moves.
    An already-red aggregate gate has stopped being a gate.

This file is the same claim at ONE NODE PER MODULE.  A tenth module going
red is a new node id, visible in the count and in the summary, whether or not
the other nine are red.

Measured on this tree at b47a400a5 (and identically at 659962929, the
published 2.5.8 tip): 56 of the 65 pinned modules still match
their historical freeze; nine have moved, each in a named commit, and are
re-pinned below to the bytes they now carry.  Re-pinning is not loosening --
every one of the 65 is pinned to a specific digest, and the drift is written
down with the commit that caused it, which is the amendment discipline the
repository already uses for a stale gate table.  Nothing here is exempt and
nothing here is skipped.

It hashes 91 text files.  It compiles nothing, imports no kernel and needs no
CUDA device.
"""

from __future__ import annotations

import hashlib
import importlib.util
import pathlib
import sys

import pytest

REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[1]
KERNELS = REPOSITORY_ROOT / "gpuwm" / "core" / "kernels"


def _frozen_module():
    """``tests/test_mp8_frozen.py`` loaded BY PATH.

    ``from tests import ...`` resolves to an unrelated ``tests`` package in
    site-packages on this box -- the same shadowing ARWEN-ORIENTATION
    section 9 records for ``gpuwm``.  The path is explicit and asserted so a
    tripwire cannot end up reading the wrong tree's pins.
    """

    path = REPOSITORY_ROOT / "tests" / "test_mp8_frozen.py"
    assert path.is_file(), f"{path} is missing; its digests are the source"
    spec = importlib.util.spec_from_file_location("gpuwm_frozen_pins", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["gpuwm_frozen_pins"] = module
    spec.loader.exec_module(module)
    assert pathlib.Path(module.__file__).resolve() == path
    return module


#: ``{module: (file_sha256_now, the commit that moved it)}`` for any module
#: that no longer matches ``FROZEN_MODULE_DIGESTS``.  This table makes a
#: drift explicit and CHECKED -- a further edit to a listed module fails
#: its own node exactly as an edit to a frozen one does -- and the gate
#: beneath makes it TEMPORARY: once a drift is ratified into the freeze,
#: its row here must go.
RE_PINNED_DRIFT: dict[str, tuple[str, str]] = {
    # EMPTY on purpose, and the gate below keeps it true: the nine drifted
    # modules this table shadow-pinned (diagnostics/nest for the two-way
    # feedback, kf and ysu for the column-workspace moves, rrtmg_lw's
    # buffer-march fix, rrtmg_sw's subnormal armor, and the contributed
    # RRTMGP optimisation's three units) were RATIFIED into
    # FROZEN_MODULE_DIGESTS with each owning lane's evidence cited beside
    # its hash, so the drift this table recorded no longer exists.  A new
    # drift enters here with its (sha, commit) pair until its own
    # ratification retires it the same way.
}

#: The kernel translation units that ``FROZEN_MODULE_DIGESTS`` never pinned.
#: The freeze deliberately ignores modules added after it was taken, which is
#: right for a FREEZE and wrong for a guard: measured on this tree at
#: b47a400a5 there were twenty-six, and they are not a tidy set of new mp=28
#: additions.  They include ``gf`` (Grell-Freitas cumulus), ``wdm6``,
#: ``milbrandt2``, ``myjpbl``, ``myjsfc``, ``mynn_dmp_sibling``,
#: ``mynn_scalar_mix``, ``noahmp_glacier``, ``shinhong``, ``sase`` and
#: ``tke_budget`` -- shipped physics whose CUDA source no digest anywhere in
#: the repository checked.  These are their bytes at b47a400a5, which is 2.5.8
#: plus the P3 CUDA port.
#:
#: This is a BASELINE pin, and it says so: it does not claim these bytes are
#: right, it claims they are the bytes 2.5.8 shipped, so the next change to
#: any of them is a decision somebody makes on purpose instead of an edit
#: nothing observes.  The comment above each entry is the commit that last
#: moved that file.
BASELINE_PINNED: dict[str, str] = {
    # 6106e4e31 feat(ftz): measure FP32 subnormal handling on all five compile routes
    "ftz_probe":
        "a8c76a2f19dce3eb9ce545f2caacd1891ac35c3de56c7d57e39e787f37e1717f",
    # d0c23dad0 feat(cumulus): New Tiedtke joins as cu_physics = 16 -- the glibc
    # float32 transcendentals leave gf.cu for the shared glibc_flt32.cuh the
    # loader prepends; a move, not an edit (517 non-trivial lines out, 517 in)
    "gf":
        "2e6a0b59b669e3c35fc0138be14d7ce2a8d7781b2c02f46ea41a23efe0764142",
    # 1ee7f0be0 tiles: name the streamed-run config table, and part it from cycle streaming
    "health_tile":
        "2943d5e226a61487aefbe7f191dc120420a4cfe3f96deef19c90c2bb8c15bead",
    # d76e25a82 feat(da): the LETKF factors its own matrices; cuSOLVER becomes optional
    "jacobi_eigh":
        "7e24eff5cf84ff6895e251aab6165d5e866c1eadfd3e4f33a740d5932631c23c",
    # 0c8f2305d Batch native KF output validation
    "kf_validation":
        "697a1cab3ab07d2e1464c03cad72c08bda809d461a33b5d67273e31ca2a71f56",
    # 5b912c2b9 Batch canonical microphysics validation
    "microphysics_validation":
        "a9e21aff3e9f011bf16a49ae3df3bf7d7688fc86bb8ba27ead08340839034e78",
    # d0c23dad0 feat(cumulus): New Tiedtke joins as cu_physics = 16 -- the scheme's
    # translation unit, bitwise against WRF v4.6.1 at every stage (tests/test_ntiedtke_*)
    "ntiedtke":
        "06daa934a71a02797fbaaaef243b83d4c43382e2dcd1cbd83505393d9373af7a",
    # c1563f187 fix(release-scan): the gate reads by content, and sees an escaped path
    "milbrandt2":
        "381aa37f9509ac8663e0333131ae44aa963820cd7712c88e89d555ef5b92b574",
    # f1e9adbf1 fix(myj): seed TKE_MYJ at WRF's EPSQ2, make the mutation controls real, decl
    "myjpbl":
        "d0e0b3dde6ba1729460694a3bf730ac82420e85d3979327953a7e45bf85719f1",
    # f1e9adbf1 fix(myj): seed TKE_MYJ at WRF's EPSQ2, make the mutation controls real, decl
    "myjsfc":
        "cccc66d730e535c7d35749c5d6b96e58af9205ec4e682d68e2dcc5b6d2949009",
    # 4a0bb3f69 mynn(mixscalars): MYNN-EDMF mixes the qn family, and the DMP unit exports it
    "mynn_dmp_sibling":
        "3684fde5c7647211ea0118d26232996a2e005995c5bfcb53b8318e439a828e68",
    # 4a0bb3f69 mynn(mixscalars): MYNN-EDMF mixes the qn family, and the DMP unit exports it
    "mynn_scalar_mix":
        "de9860af23c6bf612a0ef5df4c9720e6e59e3f736f904d749ee5fefbdfc386e8",
    # 342f8780d feat(glacier): NOAHMP_GLACIER ported, the sea-ice threshold configurable, na
    "noahmp_glacier":
        "6a200773433a257f562f38d3e32cff13555acea1a4ce8267054b60914a6b5219",
    # 2258c85e3 feat(p3): mp=50 runs on the card, because a host round trip per
    # step was the scheme.  Added by the P3 CUDA port AFTER 2.5.8 published; it
    # is the module this gate named unprompted the day it was written, which is
    # the case for the reverse test at the bottom of this file.
    # RE-PINNED by the 2.6.1 cold-start qvs-floor fix: the floor lands
    # default-on in all three arms (2026-08-31) -- step-1 sup/supi pin
    # at -1 instead of the stock 0/0 NaN, inert from step 2 on; the
    # measurement is tests/test_p3_port.py::
    # test_the_first_step_qvs_floor_pins_sup_at_minus_one plus the
    # from-step-2 off-path identity.
    # RE-PINNED AGAIN by the immersion-freezing overflow rescue folded off
    # p3/front-door-20260829: the droplet and rain branches gained WRF's own
    # commented-out double excursion, reached ONLY when the single-precision
    # chain overflows to +Inf.  A real 6 h GFS forecast on the shipped
    # p3-mp50 suite went non-finite at step 284 because it does (see the
    # block comment at the branch and gpuwm/core/p3.py
    # _rescue_overflowed_product).  The float path is byte-unchanged, so
    # this is a new branch, not a moved answer.
    "p3":
        "58b164ab9a650a25c330a5961e95c93fbaac753471ea44ecb71395aa3a8c3039",
    # 9c57c4ee9 feat(sase): CUDA mirror of the S3-12 additive e^{3/2} dissipation channel, p
    "sase":
        "9c49c1d06f5dfc30de04e2bed68b1d5ff4a4bf3426dadb943da03124e41d940d",
    # a084e0aeb fix(shinhong): the ULP table moved because the kernel compiler did, so it is
    "shinhong":
        "342e8c86f16262bf54137569b31e6637d4f8589281e304a76ec0a68a487286c6",
    # 58dfd599c feat(shinhong): CUDA mirror on the RTX 5090 -- dtheta bitwise, br DAZ counte
    "shinhong_validation":
        "e3714616c403a3f272600499164cf9b0215879d567dadcda31287dd33c600b87",
    # c1563f187 fix(release-scan): the gate reads by content, and sees an escaped path
    "thompson_aerosol_cold":
        "ff9efd45815288a6a79ff4cad0c5d06feba18c16728597d45cb2bb0ee9bc5188",
    # 0ebda6608 snapshot(mp28): the recovered aerosol-aware Thompson port, re-parented to it
    "thompson_aerosol_probe":
        "a83d3c9f8157b5702b504350ee93572c34378390917f8c037bf2762b27b0a91e",
    # c1563f187 fix(release-scan): the gate reads by content, and sees an escaped path
    "thompson_aerosol_sat":
        "b54711ac9e07dfe843378f53402b5c2bddb324e7bb90b1fff6629eed7540aea6",
    # c1563f187 fix(release-scan): the gate reads by content, and sees an escaped path
    "thompson_aerosol_sed":
        "d234db58a7d3cbb1c8442f00f3c1b587b1cde9486afa690ddf948f0c851531d7",
    # c1563f187 fix(release-scan): the gate reads by content, and sees an escaped path
    "thompson_aerosol_state":
        "856c00e10f3fb4cf802308ea86196a688d8bb75f5dcdb760f77e4e83b170ca50",
    # c1563f187 fix(release-scan): the gate reads by content, and sees an escaped path
    "thompson_aerosol_warm":
        "031fa75543cb940821354a5a5eeb619a33edf192598bc416e8e1ac0a99f3daa0",
    # 02cfd5301 feat(les): km_opt=2 restart carrier, lateral-boundary arm, TKE budget
    "tke_budget":
        "c7f6dc37f15b25fccbea50deef0c6d595c08b2ee4762f14eef169b654d54fccb",
    # 5165b9485 chore(wdm6): the divergence gets a citation, the constants get one home
    "wdm6":
        "b9a370f5cc9c480c1fa19b41c5c2ce4a395710ce3dae787141f627de15b5a062",
    # 5165b9485 chore(wdm6): the divergence gets a citation, the constants get one home
    "wdm6_refl":
        "5dff160d671d68c2236c964bfac94e0b8f275a6897b8840f860c0e1ddbf9fdcf",
    # c5afbc870 Batch YSU output validation
    "ysu_validation":
        "ed125e770df19cb3161c4a8bed53e55cc0d740f521f318cf9166e2a1063ddd25",
}

_FROZEN = _frozen_module()
PINNED: dict[str, str] = dict(
    (name, digests[0]) for name, digests in _FROZEN.FROZEN_MODULE_DIGESTS.items())
PINNED.update((name, sha) for name, (sha, _c) in RE_PINNED_DRIFT.items())
PINNED.update(BASELINE_PINNED)


def test_the_pin_table_was_read_at_all() -> None:
    """A gate over an empty table passes and protects nothing."""

    assert len(PINNED) >= 91, (
        f"only {len(PINNED)} kernel module(s) are pinned; ninety-one were at "
        "b47a400a5 (65 frozen + 26 baseline), so a pin table was read wrongly "
        "or emptied and this gate is checking almost nothing")


def test_every_re_pinned_module_is_still_a_module_that_drifted() -> None:
    """The re-pin table cannot outlive its reason.

    If a module in ``RE_PINNED_DRIFT`` comes back into agreement with
    the historical freeze -- someone reverts the change, or re-pins
    ``FROZEN_MODULE_DIGESTS`` properly -- this entry is stale and must be
    dropped, or the file reads as if nine modules are still adrift.
    """

    reconciled = sorted(
        name for name, (sha, _c) in RE_PINNED_DRIFT.items()
        if _FROZEN.FROZEN_MODULE_DIGESTS.get(name, (None,))[0] == sha)
    assert not reconciled, (
        f"{reconciled} now agree with FROZEN_MODULE_DIGESTS; remove them "
        "from RE_PINNED_DRIFT so the drift list stays true")


@pytest.mark.parametrize("module", sorted(PINNED))
def test_the_kernel_source_is_byte_identical_to_its_pin(module: str) -> None:
    """One node per translation unit.  A tenth red is a NEW node id."""

    source = KERNELS / f"{module}.cu"
    assert source.is_file(), (
        f"{source} is pinned and does not exist; a frozen kernel module "
        "disappeared")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    where = ("RE_PINNED_DRIFT" if module in RE_PINNED_DRIFT
             else "tests/test_mp8_frozen.py FROZEN_MODULE_DIGESTS")
    assert digest == PINNED[module], (
        f"gpuwm/core/kernels/{module}.cu changed.\n"
        f"  pinned in {where}\n"
        f"    expected {PINNED[module]}\n"
        f"    actual   {digest}\n"
        "  A CUDA kernel source moved.  If the change is intended, re-pin it "
        "in the same commit and say in that commit which physics changed and "
        "what measurement shows the new behaviour is right.  A silently "
        "edited kernel runs on every column of every step and no CPU test in "
        "this repository executes it.")


def test_every_kernel_source_on_disk_is_pinned_by_something() -> None:
    """The other direction: a kernel nobody pins is a kernel nobody guards.

    ``FROZEN_MODULE_DIGESTS`` deliberately ignores kernel modules added after
    the freeze.  That is the right call for a FREEZE and the wrong one for a
    guard, because it means a new ``.cu`` -- or a renamed one -- carries no
    digest at all.  The unpinned set is listed here with its size so that
    growth is a decision.
    """

    on_disk = {path.stem for path in sorted(KERNELS.glob("*.cu"))}
    unpinned = sorted(on_disk - set(PINNED))
    assert not unpinned, (
        f"{len(unpinned)} kernel source(s) under gpuwm/core/kernels are "
        f"pinned by no digest at all:\n  " + "\n  ".join(unpinned) +
        "\n  Every .cu in that directory was pinned at b47a400a5, so this is "
        "a CUDA translation unit that joined the product with nothing "
        "checking its bytes.  Add it to BASELINE_PINNED (or to "
        "the freeze) in the commit that adds the kernel.")
