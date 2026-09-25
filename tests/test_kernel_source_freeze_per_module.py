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
    # 3eb49b2fd: evaluation-time moist boundary conversion; measured
    # 73fc2406f evaluator/shared-dycore/public 1800-second forecast.
    # Raw + assembled compiler identities and resources are retained in
    # docs/measurements/lbc-time-2026-09-05/{windows,linux}.json.
    "lbc_time":
        "e98471547d79502806324b840eae701af51d3e6f7e21682d0d7602049513a8c2",
    # 6106e4e31 feat(ftz): measure FP32 subnormal handling on all five compile routes
    "ftz_probe":
        "a8c76a2f19dce3eb9ce545f2caacd1891ac35c3de56c7d57e39e787f37e1717f",
    # d0c23dad0 feat(cumulus): New Tiedtke joins as cu_physics = 16 -- the glibc
    # float32 transcendentals leave gf.cu for the shared glibc_flt32.cuh the
    # loader prepends; a move, not an edit (517 non-trivial lines out, 517 in)
    # RE-PINNED at 2.7.0 by the gamma replacement.  What moved, and the
    # measurement that says the new behaviour is right:
    #   * ArWen's earlier gamma is gone from the header this unit
    #     prepends, replaced by ArWen's own correctly rounded gamma, which is
    #     CORRECTLY ROUNDED on all 59,768,833 float32 of [0.25, 36] where
    #     glibc 2.39 is not on 23,575,230 of them (39.4440 per cent, worst 6
    #     ULP).  Measured against a 113-bit tgammaq oracle;
    #     tests/test_gf_gamma_correctly_rounded.py is the gate.
    #   * gf_libm_unary_probe drops from 7 slots to 4: gfk_lgamma_pos,
    #     gfk_expm1 and gfk_exp2 were reached only by that gamma block and
    #     are deleted with it.
    #   * gf_gfdrv_stage gains three scin slots (DINS_fzu_up/dn/sh), the
    #     same fzu_override gf_deep_stage and gf_shallow_stage already
    #     expose.  The shipped forecast passes 0 in all three; the parity
    #     suites pin them from the WRF capture, which is what keeps the
    #     216-column GFDRV boundary bitwise (host crosscheck: 83/83 level
    #     fields, 69/69 scalars, 39/39 integer fields, 126/126 shallow,
    #     0 driver gate failures).
    #   * the physics change is a DELIBERATE DIVERGENCE and is written up in
    #     docs/gf_gamma_known_delta.md: RAINCV/PRATEC move by at most 7.273
    #     per cent, median 1.613, on the committed capture, and no integer
    #     index field moves on any of the 216 columns.
    # RE-PINNED by the WRF-parity cup_up_aa0 repair (PAR-CU-GF-02).
    # module_cu_gf_deep.F:3024 is `IF(K.LT.KBCON(I))GO TO 100`, so the level
    # k == KBCON contributes to the undilute CAPE; the kernel wrote
    # `k <= kbcon` and dropped it.  The index convention is settled by the
    # sibling cup_up_aa1bl (:4048 `IF(k.gt.KBCON(i))`, gf.cu:1168
    # `if (k > kbcon) continue;`), and gpuwm.verify.gf_deep_ref:971 already
    # read `k < kbcon`.  Answers move only where the dropped level is the
    # whole of aa0: the committed 216-column oracle
    # (tests/test_gf_wrf461_parity.py) is byte-identical either way and
    # stays 2/2 green.  Measured by tests/test_gf_workspace.py::
    # test_cup_up_aa0_keeps_the_k_equals_kbcon_layer, which compiles gf.cu
    # and drives the device function through ctypes: 0.0 before, 0.0348961316
    # after, identical to the Python reference.  gf is not an mp=8
    # translation unit and thompson.cu is byte-unchanged.
    # Composed licence + parity source independently checked before this pin:
    # gamma CPU/device gates and the kbcon-layer probe: 17 passed
    # on 2026-09-04. Neither standalone branch digest names these bytes.
    # RE-PINNED at 2.7.6 from 334c55ab764732be by the notice correction: the
    # comments at lines 62-65 and 2820 stop describing the earlier gamma as
    # derived from glibc.  Comments only, measured: with // and /* */
    # comments removed the file is byte-identical to the 334c55ab bytes, and
    # it keeps its 4,061 lines, so every line number cited above still holds.
    "gf":
        "2ca7ac7bbeb01627ddeff8819b103c07081b6962125fb68d8f688627370184d6",
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
    # 6933d5762 fix(physics): Milbrandt-Yau joins the radar operator -- the pure
    # Z block of mp_milbrandt2mom_main's final diagnostics
    # (module_mp_milbrandt2mom.F:3400-3466), lifted out of the byte-frozen
    # milbrandt2.cu so gpuwm.da.obsop can launch H_Z(x) without the scheme's
    # state update.  It shipped with nothing checking its bytes at all, which
    # this pin closes; the duplication is held equal to the original by
    # tests/test_da_obsop_milbrandt_gpu.py::
    # test_the_operator_is_the_schemes_own_z_block_bitwise.
    "milbrandt2_zet":
        "dfab83f02bc23e2f4d019015203fb942cd4fbfe02bae4dc342b6816b062f532c",
    # f1e9adbf1 fix(myj): seed TKE_MYJ at WRF's EPSQ2, make the mutation controls real, decl
    "myjpbl":
        "d0e0b3dde6ba1729460694a3bf730ac82420e85d3979327953a7e45bf85719f1",
    # f1e9adbf1 fix(myj): seed TKE_MYJ at WRF's EPSQ2, make the mutation controls real, decl
    # RE-PINNED by the WRF-parity PBLH seed repair (MYJ-01): the height
    # accumulator seeded from ``dz_a[0]`` -- layer 0 of COLUMN 0 -- for every
    # thread, where module_sf_myjsfc.F:177-184 accumulates ZINT strictly
    # inside column I.  PBLH was wrong by ``dz[0][0] - dz[0][col]`` on every
    # column but the first, and it re-enters SFCDIF as BTGH (:431-435).
    # This MOVES ANSWERS on any domain with terrain.  Measured by
    # tests/test_myj_port.py::
    # test_the_surface_kernel_pblh_uses_each_columns_own_dz (two columns,
    # different dz[0] -- the old gate ran at shape (1,1), where col is
    # always 0 and the two indices coincide) and, device-free, by ::
    # test_the_surface_kernels_pblh_accumulator_is_column_local.
    "myjsfc":
        "334ae702f03f2572a2bb8e3590056b86127932431185aa4ec567b759817e938f",
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
    # RE-PINNED 2026-09-24 by the WRF v4.6.1 real-column repairs.  What
    # moved is WRF's own rule in each case, cited to module_mp_thompson.F
    # in the kernel, and the measurement that says the new behaviour is
    # right is tools/thompson_real_column_parity: WRF's Fortran run beside
    # the port's kernels compiled for the host on 137,200 columns of seven
    # saved real-data states, where every mp=28 process rate now agrees
    # to float32 rounding (17 rates had differed beyond 1 percent at up
    # to 2,457 levels a frame) and the echo within 0.024 dB (up to 8.7 dB
    # off in 505 to 939 cells of every forecast frame before); the
    # committed fixture tests/data/thompson_real_columns_wrf461.npz holds
    # it in tests/test_thompson_real_column_host_parity.py.
    # ce303c3e5: 5 micron crystals for number-less entry ice
    # (:1855-1858), the snow-cloud table's zero 6 micron bin (:4936),
    # the D0i minimum crystal mass (:2649, :2713), the graupel
    # sublimation number gate (:2703), the source-stage rain and
    # graupel balances (:3067-3091, :3118-3160); 217e84e18: the running
    # vapour carried unfloored (:3974); 08f1f9373: the ice mass/number
    # balance above 0 C (:3033-3055).
    # Previously ff9efd45815288a6.
    "thompson_aerosol_cold":
        "42fcf4c2d28a8e2be98e05dc0b169a08e0fe611dab69c27b0b62b4b6e9840abb",
    # 0ebda6608 snapshot(mp28): the recovered aerosol-aware Thompson port, re-parented to it
    "thompson_aerosol_probe":
        "a83d3c9f8157b5702b504350ee93572c34378390917f8c037bf2762b27b0a91e",
    # c1563f187 fix(release-scan): the gate reads by content, and sees an escaped path
    # RE-PINNED 2026-09-24 by the WRF v4.6.1 real-column repairs.  What
    # moved is WRF's own rule in each case, cited to module_mp_thompson.F
    # in the kernel, and the measurement that says the new behaviour is
    # right is tools/thompson_real_column_parity: WRF's Fortran run beside
    # the port's kernels compiled for the host on 137,200 columns of seven
    # saved real-data states, where every mp=28 process rate now agrees
    # to float32 rounding (17 rates had differed beyond 1 percent at up
    # to 2,457 levels a frame) and the echo within 0.024 dB (up to 8.7 dB
    # off in 505 to 939 cells of every forecast frame before); the
    # committed fixture tests/data/thompson_real_columns_wrf461.npz holds
    # it in tests/test_thompson_real_column_host_parity.py.
    # ce303c3e5: the adjustment exports L_qc (:3485) and the rain
    # evaporation writes L_qr and the :3568 rewrite into the rain
    # reference density (:3236); 217e84e18: the running vapour carried
    # unfloored (:3974).
    # Previously b54711ac9e07dfe8.
    "thompson_aerosol_sat":
        "44d3fb82ecfc49aea7711d09a73262802299fa4889406c8ec547210053ee5c04",
    # c1563f187 fix(release-scan): the gate reads by content, and sees an escaped path
    # RE-PINNED 2026-09-24 by the WRF v4.6.1 real-column repairs.  What
    # moved is WRF's own rule in each case, cited to module_mp_thompson.F
    # in the kernel, and the measurement that says the new behaviour is
    # right is tools/thompson_real_column_parity: WRF's Fortran run beside
    # the port's kernels compiled for the host on 137,200 columns of seven
    # saved real-data states, where every mp=28 process rate now agrees
    # to float32 rounding (17 rates had differed beyond 1 percent at up
    # to 2,457 levels a frame) and the echo within 0.024 dB (up to 8.7 dB
    # off in 505 to 939 cells of every forecast frame before); the
    # committed fixture tests/data/thompson_real_columns_wrf461.npz holds
    # it in tests/test_thompson_real_column_host_parity.py.
    # ce303c3e5: the terminal ice bound in its per-kilogram form
    # (:4024-4040) and the terminal bounds on the density the rain
    # evaporation left (:3572); c4f3fcc70: cloud at or below 1e-12
    # kg/kg carried to the phase cleanup (:3943-3966).
    # Previously d234db58a7d3cbb1.
    "thompson_aerosol_sed":
        "8380f654e902a4ecabb7d9b44c5be30f210d12ae3e0fdf6c93e1d93c8748abf1",
    # c1563f187 fix(release-scan): the gate reads by content, and sees an escaped path
    # RE-PINNED 2026-09-24 by the WRF v4.6.1 real-column repairs.  What
    # moved is WRF's own rule in each case, cited to module_mp_thompson.F
    # in the kernel, and the measurement that says the new behaviour is
    # right is tools/thompson_real_column_parity: WRF's Fortran run beside
    # the port's kernels compiled for the host on 137,200 columns of seven
    # saved real-data states, where every mp=28 process rate now agrees
    # to float32 rounding (17 rates had differed beyond 1 percent at up
    # to 2,457 levels a frame) and the echo within 0.024 dB (up to 8.7 dB
    # off in 505 to 939 cells of every forecast frame before); the
    # committed fixture tests/data/thompson_real_columns_wrf461.npz holds
    # it in tests/test_thompson_real_column_host_parity.py.
    # 217e84e18: WRF's no-microphysics column exit (:1646, :2020), the
    # new thompson_aa_micro_columns flag kernel, and the terminal apply
    # flooring vapour at 1e-10 in every other column (:3974); the final
    # vapour, nwfa and nifa have no unexplained cell on any frame.
    # Previously 856c00e10f3fb4cf.
    "thompson_aerosol_state":
        "a64cc5d86f2cfa0908cbff1ec0c0109b8e1fa2eba4947792aab362b9a260a7a8",
    # c1563f187 fix(release-scan): the gate reads by content, and sees an escaped path
    # RE-PINNED 2026-09-24 by the WRF v4.6.1 real-column repairs.  What
    # moved is WRF's own rule in each case, cited to module_mp_thompson.F
    # in the kernel, and the measurement that says the new behaviour is
    # right is tools/thompson_real_column_parity: WRF's Fortran run beside
    # the port's kernels compiled for the host on 137,200 columns of seven
    # saved real-data states, where every mp=28 process rate now agrees
    # to float32 rounding (17 rates had differed beyond 1 percent at up
    # to 2,457 levels a frame) and the echo within 0.024 dB (up to 8.7 dB
    # off in 505 to 939 cells of every forecast frame before); the
    # committed fixture tests/data/thompson_real_columns_wrf461.npz holds
    # it in tests/test_thompson_real_column_host_parity.py.
    # ce303c3e5: the source-stage rain and graupel balances
    # (:3067-3091, :3118-3160); 217e84e18: the running vapour carried
    # unfloored (:3974).
    # Previously 031fa75543cb9408.
    "thompson_aerosol_warm":
        "446212647a18e1660da8a65c417f975facaa789ff207691049853946ac0b5f51",
    # 02cfd5301 feat(les): km_opt=2 restart carrier, lateral-boundary arm, TKE budget
    "tke_budget":
        "c7f6dc37f15b25fccbea50deef0c6d595c08b2ee4762f14eef169b654d54fccb",
    # RE-PINNED 2026-09-20 after three commits moved the file and none
    # re-pinned it (red on every box since 2026-09-13; proof/node-reds-276):
    #   * 07d1ef7e3 (2026-09-12) fix(physics): publish current WDM6
    #     checkpoint identity -- the published registry row and the
    #     kernel's identity comment;
    #   * 0a7fc061f (2026-09-12) fix(physics): bound evolving WDM6 rain
    #     transport time -- every receiving layer covered by the existing
    #     fall-speed envelope, conservative interface transfers kept,
    #     temporal refinement verified independently
    #     (tests/test_wdm6_sedimentation.py, docs/wdm6_oracle_known_deltas.md);
    #   * a3f158bef (2026-09-13) the substep count is checked before it is
    #     converted to a signed counter; an unrepresentable count or an
    #     invalid density/thickness stops the affected column and reports
    #     a failed call through the health path, normal schedules
    #     unchanged and no iteration cap applied
    #     (tests/test_wdm6_count_safety.py, tests/test_wdm6_count_cuda.py).
    # The bytes below are a3f158bef's.
    "wdm6":
        "6f528e1b1df03047f48df9c2d495560896ba17d2ec2bcd3c1206748b2156b705",
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

# These are the complete headers the real loader prepends. common.cuh
# retains its existing authority; the other rows close the former *.cu-only
# gap without changing any source bytes. The GF gamma numerical evidence is
# docs/gf_gamma_known_delta.md and tests/test_gf_gamma_correctly_rounded.py;
# header assembly is independently checked by test_kernel_loader_inert.py.
PINNED_HEADERS = {
    "common.cuh": _FROZEN.COMMON_CUH_SHA256,
    # RE-PINNED at 2.7.6 from 794c7d4123bb0642 by the notice correction: the
    # comments at lines 235-237 and 344-353 stop describing the earlier gamma
    # as derived from glibc.  Comments only, measured: with comments removed
    # the header is byte-identical, and it keeps its 552 lines.
    "glibc_flt32.cuh": "b647335d355aefb5d9ece1726ff23667a01f09c8b263c136bd8a94764e904d4f",
    "rrtmgp_planck_common.cuh": "4e1a8214ea8e2a3dbd88cc2cda260a21ff678d98acf4f22c971ba0b51b4eba36",
    "thompson_aerosol_common.cuh": "07f5c144180b95dbc218480784c9cdaaeaf5ce6614180074a92299400906f97d",
}


@pytest.mark.parametrize("header", sorted(PINNED_HEADERS))
def test_prepended_header_is_byte_identical_to_its_pin(header):
    source = KERNELS / header
    assert source.is_file(), f"pinned CUDA header is missing: {source}"
    assert hashlib.sha256(source.read_bytes()).hexdigest() == PINNED_HEADERS[header], (
        f"prepended CUDA header {header} changed; retain its numerical authority "
        "and record the measured change before updating its pin")


def test_header_pins_cover_the_actual_loader_closure():
    from gpuwm.core.kernels import EXTRA_HEADERS
    used = {"common.cuh", *(header for headers in EXTRA_HEADERS.values() for header in headers)}
    assert used == set(PINNED_HEADERS)
    assert {path.name for path in KERNELS.glob("*.cuh")} == used


def test_header_fault_is_detected_even_when_module_pins_are_unchanged(tmp_path, monkeypatch):
    header = "glibc_flt32.cuh"
    (tmp_path / header).write_bytes((KERNELS / header).read_bytes() + b"\n// fault control\n")
    monkeypatch.setattr(sys.modules[__name__], "KERNELS", tmp_path)
    with pytest.raises(AssertionError, match="prepended CUDA header"):
        test_prepended_header_is_byte_identical_to_its_pin(header)


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
             else "BASELINE_PINNED in this file" if module in BASELINE_PINNED
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
