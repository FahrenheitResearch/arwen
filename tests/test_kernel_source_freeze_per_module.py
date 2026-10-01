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
    # RE-PINNED 2026-09-30 by the default-pieces speed lane (b11fc66ab): a
    # production column the first trigger rejects writes its zero outputs
    # at once, the environment above the trigger's reach is built only for
    # columns that pass it, and the driver launch is bounded to eight
    # resident blocks per SM.  No answer moved: the lane's A/B dumps after
    # 23 steps (sm_120) and a 1 h real HRRR GF forecast (sm_89) are
    # byte-identical to d6929cb8d.  Previously 2ca7ac7bbeb01627.
    "gf":
        "ad4148a861c20622a4e7278de970171b649bf400f57c49814e98a9a145547017",
    # 1ee7f0be0 tiles: name the streamed-run config table, and part it from cycle streaming
    "health_tile":
        "2943d5e226a61487aefbe7f191dc120420a4cfe3f96deef19c90c2bb8c15bead",
    # d76e25a82 feat(da): the LETKF factors its own matrices; cuSOLVER becomes optional
    "jacobi_eigh":
        "7e24eff5cf84ff6895e251aab6165d5e866c1eadfd3e4f33a740d5932631c23c",
    # 0c8f2305d Batch native KF output validation
    "kf_validation":
        "697a1cab3ab07d2e1464c03cad72c08bda809d461a33b5d67273e31ca2a71f56",
    # 5b912c2b9 Batch canonical microphysics validation; re-pinned by the
    # speed lane's one-launch ring guard (mp_ring_copy joins the unit: word
    # copies only, frame 0 B measured on sm_120 at NVRTC 13.4.92; every
    # two-moment and Thompson forecast byte-identical to d6929cb8d)
    "microphysics_validation":
        "9a40f5bda9065d98d7145dbda3473d22ad00a890aad223e5f2e913b16b547b28",
    # d0c23dad0 feat(cumulus): New Tiedtke joins as cu_physics = 16 -- the scheme's
    # translation unit, bitwise against WRF v4.6.1 at every stage (tests/test_ntiedtke_*)
    # RE-PINNED 2026-09-30 by the default-pieces speed lane (5310e64b2):
    # cutypen clears parcel scratch only for trials that can run and the
    # hoist checks are read once per call from integer device masks.  No
    # answer moved: the lane's A/B dumps (sm_120, the fused 350 x 200 path
    # included) and a 1 h real HRRR New Tiedtke forecast (sm_89) are
    # byte-identical to d6929cb8d.  Previously 06daa934a71a0279.
    "ntiedtke":
        "17fe4aac03ed6b5191bda49db9eebe6c51c29d0e542bafc8ea33989f363caa42",
    # c1563f187 fix(release-scan): the gate reads by content, and sees an escaped path
    # RE-PINNED by the speed lane's category-parallel sedimentation (no
    # reading moves): five thread rows per column block run rain, ice, snow,
    # graupel and hail at once (disjoint fields), then one thread per column
    # takes the snow constraints and precipitation totals in the original
    # order after a block barrier.  The mp9 suite's 1 h forecasts of two
    # convective cases (220 x 176 x 49) are byte-identical to d6929cb8d on
    # an RTX 5090.
    # Re-pinned for A146 on top of that: constant-divisor float divisions
    # spelled __fdiv_rn, because NVRTC compiles x / C as a multiply by the
    # rounded reciprocal on Blackwell targets. Previously 8b4219be.
    "milbrandt2":
        "654a4fd0c496e9f78028b1450fa8be1eb59239ee350029b06b02a7d53db43d23",
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
    # Re-pinned for A146 (a98f2482e): constant-divisor float divisions
    # spelled __fdiv_rn.  Previously 334ae702.
    "myjsfc":
        "7ce67cdd5459cc8154311b75378a937b066bab24fd7b67892cb92b3ec4fce637",
    # 4a0bb3f69 mynn(mixscalars): MYNN-EDMF mixes the qn family, and the DMP unit exports it
    # RE-PINNED by the 2026-09-30 MYNN speed lane (2f5dd16f9): level-major
    # column storage.  Addressing only; every output replayed bitwise
    # identical (tests/test_mynn_dmp_sibling.py records it).
    # Re-pinned for A146 (a98f2482e): constant-divisor float divisions
    # spelled __fdiv_rn.  Previously 90c71fa1.
    "mynn_dmp_sibling":
        "57344ddd0b30006febdd7108f5772e5c0cf76671d213812cd83ce614c28b2674",
    # 4a0bb3f69 mynn(mixscalars): MYNN-EDMF mixes the qn family, and the DMP unit exports it
    # RE-PINNED by the 2026-09-30 MYNN speed lane (2f5dd16f9): the flux kernel reads
    # level-major plume and column storage.  Addressing only; the
    # mixscalars replay is bitwise identical.
    "mynn_scalar_mix":
        "04c6c17fd0ee157515c74b90642e319f9f293d06192fdad568c438dfb28516cf",
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
    # RE-PINNED by the speed lane's level arms (no reading moves): four
    # threads per column share the level-local work of preparation,
    # process rates, sedimentation substeps and final diagnostics, each
    # level's statements unchanged; ``cuda`` now selects ``sedlevels``.
    # 1 h P3 forecasts of two convective cases (220 x 176 x 49) are
    # byte-identical to d6929cb8d on an RTX 5090 and an RTX 4090.
    # RE-PINNED again by the level-group shapes (no reading moves):
    # preparation runs eight groups of 16 columns and finishes both of a
    # level's stages before the next, process rates sixteen groups of 8,
    # sedimentation sixteen groups of 4 (fmaxf Courant maximum, exact).
    # The same two P3 forecasts stay byte-identical on an RTX 5090.
    # Re-pinned for A146 on top of that: constant-divisor float divisions
    # spelled __fdiv_rn, because NVRTC compiles x / C as a multiply by the
    # rounded reciprocal on Blackwell targets. Previously ebbd1f28.
    "p3":
        "449b70350553e31785c4a8a53c64d65ee846a3e77392e7a96ef8e58231b34080",
    # 9c57c4ee9 feat(sase): CUDA mirror of the S3-12 additive e^{3/2} dissipation channel, p
    # Re-pinned for A146 (a98f2482e): constant-divisor float divisions
    # spelled __fdiv_rn.  Previously 9c49c1d0.
    "sase":
        "af8c732fbc4945c517a965a7dc4c2e931667400064d01072f27728f4a67d7ae8",
    # a084e0aeb fix(shinhong): the ULP table moved because the kernel compiler did, so it is
    # RE-PINNED 2026-09-30 by the default-pieces speed lane (878435c39):
    # the column arrays move from a 17,160 B per-thread local frame into a
    # global workspace sized to the columns in flight (the YSU route).
    # Placement only; no arithmetic line moved.  The lane's A/B dumps
    # (sm_120) and a 1 h real HRRR Shin-Hong forecast (sm_89) are
    # byte-identical to d6929cb8d, and the ULP table gained its NVRTC
    # 13.4.92 sm_89 and sm_120 rows.  Previously 342e8c86f16262bf.
    # Re-pinned for A146 on top of that: constant-divisor float divisions
    # spelled __fdiv_rn, because NVRTC compiles x / C as a multiply by the
    # rounded reciprocal on Blackwell targets. Previously be8bf6ea.
    "shinhong":
        "6fd615d06165a26b806fa768e1088f9af1976ead1eedad447aa49474787109b7",
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
    # Re-pinned for A146 (a98f2482e): constant-divisor float divisions
    # spelled __fdiv_rn.  Previously 42fcf4c2.
    "thompson_aerosol_cold":
        "d8b8faa7ac626c96d0eb851afc1a6b7779bfa2bbae8ff138c35155ad445472d4",
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
    # RE-PINNED 2026-09-30 by the speed change that gives an empty column
    # an exact short path through the aerosol cloud fallout and appends
    # the level-parallel cloud fallout; no answer moved (bit tests and a
    # byte-identical 1 h mp=28 forecast on an RTX 4090).
    # Previously 8380f654e902a4ec.
    "thompson_aerosol_sed":
        "cff91a9693c01ab21b298c621f2786edd57901b585c3e27980f3a6701a136925",
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
    # Re-pinned for A146 (a98f2482e): constant-divisor float divisions
    # spelled __fdiv_rn.  Previously 44621264.  Re-pinned again by the
    # A146 review repair: the two lamc clamps divide by a header constant
    # (THOMPSON_AA_D0C, THOMPSON_AA_D0R * 2.0f), which compute_120 turns
    # into a reciprocal multiply; they now go through thompson_aa_div.
    "thompson_aerosol_warm":
        "633791f61719e99f8c9d1ed3a1e5989fd6839243ab84be74ed6c985e95493281",
    # 02cfd5301 feat(les): km_opt=2 restart carrier, lateral-boundary arm, TKE budget
    "tke_budget":
        "c7f6dc37f15b25fccbea50deef0c6d595c08b2ee4762f14eef169b654d54fccb",
    # The UW moist-turbulence PBL (bl_pbl_physics = 9, lane/europe-uw-pbl):
    # every output word of the product launcher equals WRF v4.7.1's on the
    # packaged oracle fixtures (tests/test_uwpbl_launcher_wrf471_parity.py).
    "uwpbl":
        "79ba54c179dbe7d3c7966de5e08d5d223eaf2c169fca02ba864f3baba0ec2423",
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
    # RE-PINNED 2026-09-30 (A144): the rain condensation cap at
    # module_mp_wdm6.F:1255 keeps the rate's sign, so rain with no number no
    # longer evaporates half the saturation deficit and feeds ice deposition
    # that took vapour below zero (both WDM6 convective cases stopped at step
    # 4).  WRF v4.6.1's Fortran goes negative on the same captured columns and
    # stays nonnegative with this one rule; both cases now run 1 h with the
    # vapour check on (tests/test_wdm6_numberless_rain.py,
    # docs/wdm6_oracle_known_deltas.md section 6).
    # Re-pinned for A146 (a98f2482e) on top: constant-divisor float
    # divisions spelled __fdiv_rn.  A144 alone was 53ac977a, A146 alone
    # 0df1f08e; previously 6f528e1b1df03047 (a3f158bef).
    "wdm6":
        "e1c1a666004e43f2471b69469b6938ec188d43e542330c741ed546c3a8d771aa",
    # 5165b9485 chore(wdm6): the divergence gets a citation, the constants get one home
    "wdm6_refl":
        "5dff160d671d68c2236c964bfac94e0b8f275a6897b8840f860c0e1ddbf9fdcf",
    # c5afbc870 Batch YSU output validation
    "ysu_validation":
        "ed125e770df19cb3161c4a8bed53e55cc0d740f521f318cf9166e2a1063ddd25",
    # lane/urban-ucm: the single-layer urban canopy model (sf_urban_physics=1),
    # bitwise against WRF v4.7.1's urban, lsm and noahmp_urban
    # (tests/test_urban_ucm_*wrf471_parity.py).  Moved on lane/urban-infra
    # 2026-09-30: the green-roof constants are read from __constant__
    # memory so NVRTC 12.9.86 cannot mis-fold their sums (bitwise under
    # 12.9.86 and 13.4.92 since).  Previously 6589c991.  Moved on
    # lane/urban-physics 2026-09-30: ucm_overrides blends the UCM's 2 m value
    # into Noah-MP's T2 as the absolute temperature it is, the one named
    # divergence from WRF's surface_driver.F:3393 (measured: bitwise against
    # WRF built with that line fixed, tests/test_urban_ucm_noahmp_wrf471_
    # parity.py; the 750 m Los Angeles run's city T2 had fallen 11 K below
    # its own skin at 1,900 m).  Previously a692015a.
    "urban_ucm":
        "30436e47dcb5c3b5ae84975e26cfa045ea08fa8a6a8b33d3604e3f496743d1f2",
    # a28017016 feat(urban-bep): MYJURB (MYJ under BEP/BEP+BEM), word-identical
    # to WRF v4.7.1 (tests/test_myjurb_wrf471_parity.py)
    # Re-pinned for A146 on lane/281-nvrtc-literal-div: the flag_bep lower
    # TKE correction's division by 11.788 (module_bl_myjurb.F:426-429) is
    # spelled __fdiv_rn, because NVRTC compiles x / C as a multiply by the
    # rounded reciprocal for sm_120 under -ftz=true.  sm_89 compiles the
    # same division either way.  Previously a25a77f1.
    "myjurb":
        "ff8c000f5fdbe9f43f903a00ef9a20c760635c5321e2912257ebfe2d172a26e8",
    # ebcf4edd2 lane/urban-bep: the BEP column (sf_urban_physics=2)
    # (tests/test_urban_bep_wrf471_parity.py)
    "urban_bep":
        "11dc0497fe8442684622e1fa08194f54d1d5daf0101ca210161dfcfe51212818",
    # 04118fa24 fix(urban-bem): BEP+BEM (sf_urban_physics=3) bit-identical on
    # sm_89 and sm_120; compiled through gpuwm/core/urban_bem.py's own unit
    # (tests/test_urban_bem_wrf471_parity.py)
    "urban_bep_bem":
        "1b84b9d2a7309de203f311a28e6d8db79ba4708800747123fe664ec8014107c5",
    # 466fce443 lane/urban-bep: the Noah/Noah-MP BEP surface couple
    # (tests/test_urban_bep_couple_wrf471_parity.py)
    "urban_bep_couple":
        "31c547530a46c866859b2de0b259ec7e77e42f69dde894860c2efb90684b7ef2",
    # The translation units the 2026-09-30 speed lanes added (eleven; the
    # megakernel lane's mynn_seaice_glue, phy_column and phy_glue left the
    # tree with its revert, A147, and their pins with them).  Each
    # lane shipped its .cu with no pin, which is what this file's reverse
    # gate reported at the six-lane merge 74a4374af; they are pinned at the
    # bytes that merge carries, in the sweep that found them rather than in
    # the commits that added them.  Every lane recorded its change as
    # byte-identical output against d6929cb8d.
    #
    # lane/speed-dycore-host, 66af318bc perf(dycore): the RK time-t copies
    # and tendency clears run as one word-copy and one word-clear launch,
    # and surface w, face mass and held heating each run as one point-local
    # launch (tests/test_rk_bookkeeping.py, test_surface_w_fused.py,
    # test_face_mass_fused.py, test_held_heating_fused.py).
    "face_mass":
        "e68a95a475c3b2b958cf0e40805537bac5c0c075600b2b2ef81dae4261010ec0",
    "held_heating":
        "9228ead79c478bf4c8c7dd15dec9b917dcbd6a46cff0478dee51c97c25881c89",
    "rk_bookkeeping":
        "f1e80e7d8b15b6266713c2694104ed7a08537ccdfa0721fbf8a4f0b7945f91c2",
    "surface_w":
        "f4b488281c8fb9a59afb676a88a9803e8951c3ad7d8a409205c60a896280ba38",
    # lane/speed-rrtmg-legacy.  rrtmg_legacy_adapter: added at 2c64326aa
    # (the adapter keeps the radiation call on the device), moved by
    # cfd6503b7 and last by 378191e61 (results, ozone and radius conversion
    # on the device).  rrtmg_legacy_prep and rrtmg_lw_chain_coalesced:
    # 668fa4c56 (the wrapper prep on the device, the coalesced g-point
    # slabs).  rrtmg_lw_zbatched: added at bfc177208 (one-launch taumol),
    # moved by a44612163 and last by 77ae92552 (a summing thread per
    # (column, level) row).  Exercised by
    # tests/test_rrtmg_legacy_device_glue.py, test_rrtmg_legacy_prep_device.py
    # and test_rrtmg_lw_batched_layout.py.
    "rrtmg_legacy_adapter":
        "b19b4e13cb424b50789d75c7c8fa6066cf9a84808f9f9f5e14250e85d63b4524",
    "rrtmg_legacy_prep":
        "857a68265a35ae893d118107519cbedc97e6bb4586d590bc6f896918fa62c815",
    "rrtmg_lw_chain_coalesced":
        "2682d172388d4a31ae11be0168bdd13b33be9c1b8e77e7ca92e86a8f9a822401",
    "rrtmg_lw_zbatched":
        "5611794c5c9e44815ad00e28af7367428beda949afe205654ecb959665de7a8e",
    # lane/281-zadvect-implicit (A158): WRF 4.7.1's implicit-explicit
    # vertical advection, graded word for word against WRF's compiled
    # routines by tests/test_zadvect_implicit.py.  Moved once on the lane:
    # ieva_solve_s takes WRF's t0 as a shift, so the theta solve reads
    # theta - t0 as WRF's does (a uniform column stays uniform).  Moved
    # again by lane/281-ieva-units (A179): ieva_solve_w's two boundary
    # terms take consistent units (the lower one uncouples the u/v
    # tendencies, the upper one divides by g), a declared divergence from
    # WRF 4.7.1 graded against WRF's routine with the same corrections.
    "ieva":
        "45740ad8cbbe738a5b69b32a03a63cd878e6cd3f0378e2a9d6216deea7448df3",
    # Lane 281-namelist-gaps: WRF v4.7.1 slope_rad / topo_shading
    # (module_radiation_driver.F toposhad/topo_rad_adj), held bit for bit
    # to WRF's Fortran by tests/test_topo_radiation.py on a card.
    "topo_radiation":
        "4edf8b687f486cf56b6adccd6f00ea233e15eb438c820b34cb6a0b0c9ae90f15",
    # Noah mosaic (sf_surface_mosaic = 1): WRF v4.7.1 lsm_mosaic and its
    # ordinary and glacial SFLX subtrees, bitwise against the byte-unmodified
    # WRF column oracle at every non-FTZ word
    # (tests/test_noah_mosaic_wrf471_parity.py).
    "noah_mosaic":
        "8dab69ae3a7cb90ad41436b97ade1f4728e4030d3fc0b86fced12089ff21f791",
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
    # A146 (a98f2482e): __fdiv_rn spellings; previously 07f5c144.
    "thompson_aerosol_common.cuh": "94876bfbc38db9c75540d24944a1744d3c40d29f9be9e88dff5dafe32b772760",
    # 399b1c017: glibc 2.43 float32 trig (Arm sinf/cosf, CORE-MATH tanf/
    # asinf/acosf/atanf) for the urban BEP column, generated and proven by
    # tools/glibc_trig_flt32_proof/.
    "glibc_trig_flt32.cuh": "bbdb54c85d361d208ea9b1a7cb49a33132026d17694ad0f42c8b0bf5460baaf0",
    # The UW PBL's prepended headers, in the loader's order
    # (gpuwm/core/kernels/__init__.py _EXTRA_HEADERS["uwpbl"]): the
    # binary64 libm (tools/uwpbl_wrf471_oracle/libm64 proves it), the
    # rounding-pinned vocabulary and the CAM routines, graded with uwpbl.cu.
    "glibc_flt64.cuh": "a14aff39d4acc74d9c782726d17d0bf7ef726c40235ddeb86d6a1a0bd7c4fcb8",
    "uwpbl_common.cuh": "93c768b4b0cf9ddb7dfe6f0259b8bd60c7ccb0ad768aca30c2535514fcb1cdc8",
    "uwpbl_wvsat.cuh": "b78b74eeebdb3db76d5518f1f44b3b034f489cc8ff5bc3e35ae937863d8ecfaa",
    "uwpbl_vdiff.cuh": "cb580058c0213365aebe99f75306ef43ab4b05ab53922775bd66bd5ba99535c7",
    "uwpbl_zisocl.cuh": "d2ff60bed748bb8211412e36fb7ab623cea6b220b7ab5999e5565505c5074275",
    "uwpbl_caleddy.cuh": "5581569d8b48d8d17a86a45fed7930e2505f75b7526f68055e43a84ce2a6de63",
    "uwpbl_eddy.cuh": "5377dc3a6d607441652542c2af6e00e2f69b52699178b395584813b74a5f02c4",
    "uwpbl_driver.cuh": "824f07e71d4de98808cdb8d67a6613e24c101529148eb2dcaaa592340d86e61c",
    # The fused RUC translation unit's own sources (gpuwm.core.ruc_tier,
    # RUC_FUSED_SOURCES), appended after ruc.cu rather than prepended.
    # PINNED at 2.8.1 by lane speed-ruc, which added them: the fused RUC call
    # whose every output word equals the array orchestration's.  sfctmp's is
    # generated by tools/ruc_fused/gen_sfctmp.py, the driver's aliases by
    # tools/ruc_fused/build_driver.py; regenerate, prove identity, re-pin.
    "ruc_fused_sfctmp.cuh": "97f28c5dbf215d13bb08ee75c79a25e6bbf62255536ec1f62b7977ea05f3ef5f",
    "ruc_fused_driver.cuh": "a38eb8389187b746accc140b3b90acf10721191b7a714c01276df1359ecf7455",
}

#: Headers a module composes ITSELF rather than through the loader's
#: EXTRA_HEADERS, pinned all the same.  urban_bem.cuh is generated by
#: tools/transcribe_urban_bem.py and compiled by gpuwm/core/urban_bem.py
#: (04118fa24) into the BEP+BEM unit.
COMPOSED_HEADERS = {
    "urban_bem.cuh": "b4cb2ac2ed49d7e8aeb71193a191eefbe4a6247624772428dd9ac322a89cdb99",
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
    from gpuwm.core.ruc_tier import RUC_FUSED_SOURCES
    used = {"common.cuh", *(header for headers in EXTRA_HEADERS.values() for header in headers),
            *RUC_FUSED_SOURCES}
    assert used == set(PINNED_HEADERS)
    assert {path.name for path in KERNELS.glob("*.cuh")} == used | set(COMPOSED_HEADERS)
    for header, sha in COMPOSED_HEADERS.items():
        assert hashlib.sha256((KERNELS / header).read_bytes()).hexdigest() == sha, (
            f"{header} changed; record the reading that moved it before its pin")


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


def test_mosaic_ucm_composed_source_is_pinned():
    # The composed Noah mosaic + UCM unit (NOAH_MOSAIC_UCM, urban_ucm.cu +
    # noah_mosaic.cu), bitwise against WRF v4.7.1 on the ucm and ucm_lcz
    # families (tests/test_noah_mosaic_ucm_wrf471_parity.py).  Moved when
    # urban_ucm.cu's Noah-MP-only ucm_overrides T2 arm changed; the mosaic
    # unit never launches ucm_overrides.  Previously d70a75c1.
    from gpuwm.core.noah_mosaic import mosaic_ucm_source
    assert hashlib.sha256(mosaic_ucm_source().encode()).hexdigest() == (
        '47dbe3c6a16088ac14b298fc9e30e99110c7e161dfaf6f09c16776298eddee59')
