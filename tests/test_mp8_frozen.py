"""The mp_physics=8 freeze gate -- WP-00 of the mp_physics=28 port.

This is the merge criterion for every other package on the
``feature/mp28-thompson-aerosol-aware`` branch.  Run it at the tip of each
package's work before that work is considered landable.

WHAT IT PROVES, AND WHY THAT IS ENOUGH
--------------------------------------
The mp=8 non-regression argument is a MECHANISM, not a measurement.
``gpuwm/core/kernels/__init__.py::load_module`` builds exactly one
``cupy.RawModule`` per ``.cu`` file from a single source string.  If that
string is byte-identical then the PTX is identical, therefore the register
allocation and FP contraction are identical, therefore every mp=8 result is
bit-identical.  No PTX-diff tooling is needed, and none should be relied on:
a PTX gate is a test that fails late and can be "fixed" by relaxing it.

So the primary assertion here is a digest of the *assembled* source string
-- captured by driving the REAL loader with a recording ``RawModule``, not
by re-implementing it -- for every ``.cu`` translation unit that existed at
the frozen commit.  That catches an edit to ``thompson.cu``, an edit to
``common.cuh``, a change to ``CUDA_DEFINES``, a change to ``_preamble()``,
and any loader change that is not perfectly inert.  WP-02's
``_EXTRA_HEADERS`` allow-list is exactly such a change, and this gate is
what holds it to its inertness claim.

Around the mechanism sit six receipts (R1..R6), each a real failure mode
that would otherwise produce a model that runs, stays stable and is
silently wrong -- the specific hazard of this port, where a half-converted
prognostic ``nc`` never raises anything.

R1  source identity of thompson.cu / thompson.py / every frozen module.
R2  the classic table contract (CCN_ACTIVATE.BIN must never appear in it).
R3  ``extra_moist_species`` -- Morrison's deliberate ``nc`` exclusion.
R4  the preflight allocation and scratch-arena surface.
R5  the nest-transition edge field codes (28 APPENDED, never inserted).
R6  ``acoustic`` n_mass selection plus the recorded ``_apply_thompson``
    launcher call graph, argument by argument.

Plus two fixture receipts: F1 freezes the 92 committed mp=8 oracle CSVs,
and F2 pins the four-file clean-rebuild exception together with a hermetic
witness for its cause (see :mod:`tools.mp8_freeze_receipt`).

New mp=28 files are expected and permitted everywhere: the gate pins what
existed and ignores additions.  It never passes because something was
deleted -- every pinned name must still be present.

This module does not import cupy in its own source and opens no device: the
loader capture replaces ``cupy.RawModule`` with a recorder before any
compile can happen.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools import mp8_freeze_receipt as freeze          # noqa: E402

# ==========================================================================
# PINS.  RE-ANCHORED at the 1.4.1 merge, and the rule is unchanged: nothing
# on this branch may move a value below, and a diff that has to edit this
# file has by definition changed mp=8.
#
# The pins were captured from an orphan snapshot of ArWen 1.3.1
# (789f61181fb0b198ace10775f3ea184eb5e786a3), the last tree before any
# mp_physics=28 work.  That tree is 371 commits behind the release line the
# port now sits on, and ten of the pinned kernels moved on that line while
# the port was off it: eight noahmp_* translation units, rrtmg_sw, and
# thompson itself, the last by 5e4af4e3 ("the rain MVD bound belongs to
# TAU+1") and cb765336 ("the rain-presence gate is a mass concentration").
# SCRATCH_SLOT_REGISTRY_MP8 gained physics_validation_status the same way.
#
# Re-pinning a freeze is exactly how a freeze gets defeated, so the new
# values were not taken on trust.  Every one of the ten was verified to be
# BYTE-IDENTICAL to integration/release-1.4.1 -- `git show
# integration/release-1.4.1:gpuwm/core/kernels/<name>.cu | sha256sum`
# equals `git show HEAD:...` for all ten -- before its digest was written
# here.  The compiled-string digests are safe for the same reason at one
# remove: PREAMBLE_SHA256 and COMMON_CUH_SHA256 below did NOT move, and
# kernels/__init__._EXTRA_HEADERS lists only the six mp=28 translation
# units, so _extra_header_text() returns "" for every frozen module and
# their assembled sources are still exactly _preamble() + source.
#
# The freeze therefore still asserts what it always asserted: mp=28 has not
# touched mp=8.  What it no longer asserts is that mp=8 never changes --
# it does, on its own line, in its own commits, and this branch inherits
# those by merging rather than by editing.
# ==========================================================================

#: The release-line commit these pins were re-captured against.  The
#: original capture point, 789f61181fb0b198ace10775f3ea184eb5e786a3, is an
#: orphan snapshot with no parent and no descendant; it cannot be diffed
#: against the current line, which is why the anchor moved.
FROZEN_COMMIT = "b15a2558a569c661e148c0db3cfc99896f3af91a"
FROZEN_COMMIT_ORIGINAL = "789f61181fb0b198ace10775f3ea184eb5e786a3"

# -- R1 --------------------------------------------------------------------

THOMPSON_CU_SHA256 = (
    "3ca6b7e902d2d77ca9881a66eb484141df552c8ea30a0076bd07023e7255e760")
#: sha256 of ``_preamble() + thompson.cu`` -- the exact string nvrtc sees.
#: THIS is the mp=8 numerics guarantee.
THOMPSON_COMPILED_SOURCE_SHA256 = (
    "8cb23f0a78b1e48a402266fd4f841b2facdc7a3ee4b7d53e3c5482aae973775d")
THOMPSON_COMPILED_SOURCE_LEN = 340875

COMMON_CUH_SHA256 = (
    "c78b17cb02ef67a2ad24d19e06e1129d7d5bcda74b972b38470fd33a6e58ff43")
PREAMBLE_SHA256 = (
    "1888bcf077e4251398b3baf7cca53b7c15ba993ab791992b7e278c3e32bb75e8")
PREAMBLE_LEN = 768

CUDA_DEFINES_PIN = {
    "G": 9.81, "RD": 287.0, "RV": 461.6, "CP": 1004.5, "CV": 717.5,
    "P0": 100000.0, "T0": 300.0, "GAMMA": 1.4,
    "RCP": 0.2857142857142857, "RCV": 0.4,
    "RVOVRD": 1.6083624362945557, "RERADIUS": 1.5698587127158556e-07,
    "XLV": 2500000.0, "SVP1": 0.6112, "SVP2": 17.67, "SVP3": 29.65,
    "SVPT0": 273.15, "RHOWATER": 1000.0, "EP2": 0.6217504332755632,
}

THOMPSON_PY_SHA256 = (
    "ede422fe6c2acba76771e40aabd231ad0edf8b391db5923535f67ac0b85f70e8")

#: ``gpuwm/core/thompson.py::__all__`` verbatim, in declaration order.
#: mp=28 launchers live in the new ``thompson_aerosol_*.py`` modules; not
#: one name may be added here.
THOMPSON_PY_ALL = (
    "launch_cloud_freezing",
    "launch_cloud_saturation_adjust",
    "launch_cloud_sedimentation",
    "launch_classic_graupel_number_init",
    "launch_classic_graupel_number_finalize",
    "launch_cold_cloud_source_network",
    "launch_cold_rain_source_network",
    "launch_cold_rain_snow_graupel_network",
    "launch_effective_radius",
    "launch_final_phase_cleanup",
    "launch_frozen_vapor_network",
    "launch_frozen_vapor_network_from_owner",
    "launch_graupel_cloud_riming",
    "launch_graupel_fallout_column_mask",
    "launch_hydrometeor_column_mask",
    "launch_graupel_sedimentation",
    "launch_graupel_melting",
    "launch_graupel_sublimation",
    "launch_ice_autoconversion",
    "launch_ice_deposition",
    "launch_ice_nucleation",
    "launch_ice_sedimentation",
    "launch_rain_evaporation",
    "launch_rain_freezing",
    "launch_rain_graupel_collection",
    "launch_rain_ice_collection",
    "launch_rain_snow_collection",
    "launch_rain_sedimentation",
    "launch_rain_self_collection",
    "launch_snow_sublimation",
    "launch_snow_cloud_riming",
    "launch_snow_ice_collection",
    "launch_snow_melting",
    "launch_snow_rime_conversion",
    "launch_snow_sedimentation",
    "launch_snow_vapor_exchange",
    "launch_warm_autoconversion",
    "launch_warm_process_network",
    "launch_warm_frozen_source_network",
    "launch_warm_frozen_source_network_from_owner",
    "launch_warm_rain_collection",
    "launch_warm_saturation_adjust",
)

#: The constant-Nt_c inventory, MEASURED on the frozen tree.  Every line
#: here hardcodes what mp=28 must make prognostic.  Recorded so that a
#: reviewer can see the port never "fixed" one of them in place, and so
#: that the claim in the port spec is checkable rather than asserted.
#:
#: NOTE, and this corrects the spec's summary paragraph: the measured
#: counts are 13 / 6 / 2 / 3, not 12 / 7 / 3.  ``thompson.cu:343`` is the
#: one CORRECT 2730.0f -- ``calc_effectRad`` genuinely uses WRF's integer
#: ``g_ratio`` PARAMETER there -- and the 272.0f site is line 1007, not
#: 1006.
THOMPSON_CU_LITERAL_SITES = {
    "100.0e6f": [343, 895, 1012, 2094, 2903, 2970, 3187, 3793, 4018, 4141,
                 4273, 4693, 6945],
    "2730.0f": [343, 895, 1012, 4018, 4141, 4693],
    "272.0f": [901, 1020],
    "cloud_number_bin = 65": [3946, 4359, 7018],
}

#: Every ``.cu`` translation unit present at the frozen commit, as
#: ``name -> (file sha256, assembled-compile-string sha256)``.  Modules
#: added later (the mp=28 ones) are ignored by the gate; a pinned name
#: that disappears is a failure.
FROZEN_MODULE_DIGESTS = {
    'acoustic': (
        # Re-pinned for the WPHI_MAX_LEV tier ladder (LES program P2): the
        # unconditional `#define WPHI_MAX_LEV 129` became an `#ifndef`
        # guard around the same literal, so the launcher can compile the
        # implicit w''-phi'' solve deeper than 128 levels.  acoustic is not
        # an mp=8 translation unit and the mp=8 numerics guarantee is
        # untouched.  The guard is a preprocessor no-op at the shipped
        # tier -- proven against a real host preprocessor, with a negative
        # control, in tests/test_acoustic_nz_tiers.py -- so this pin moves
        # while the compiled binary does not.
        '766285639b5379db461db1ca9a81ec12b45d36d097547fb59ab2db541b89768c',
        '9c8836bfb75aa56ef600dd83e31b6e2ffd0948547f27c13659c345062ac3417f'),
    'advection': (
        '8a88c2fc0ed833e9fc5bd55bd3f3f78752fbb9e68714122c2fb68adc368d2d7e',
        '3449e3bc306ef5ba9a374e0e04b6e0f7601cbe6e3d7b0aa6ba6b8edd91c8d16e'),
    'coriolis_map': (
        '53fc37398d086dcf54c317917872e5d8d473b83d6d217697335d5ae6bc96d143',
        '7981384362444c8e782abb80efb29cbf61d0a3183cc0c316799b3bf9f3c43671'),
    'diagnostics': (
        # Re-pinned for the two-way feedback landing (88fdf60b9,
        # "bitwise-gated" by its own suite).  Not an mp=8 unit.
        # Re-pinned again for the EOS spelling change: calc_p_alpha stopped
        # forming the layer geopotential thickness by differencing two
        # ~2.4e4 J/kg totals and stopped writing opt 2's log ratio as
        # log(pfd/pfu).  No physics moved -- the same two quantities are
        # computed by a spelling that does not cancel.  Measured against
        # the float64 mirror, relative error in p on a random 1 K state:
        # opt 1, 2.06e-6 -> 3.64e-7 at nz=16 and 2.89e-5 -> 4.13e-7 at
        # nz=160; opt 2, 4.25e-6 -> 4.54e-7 and 1.15e-4 -> 5.49e-7.  The
        # error stopped tracking 1/dz, which is what
        # tests/test_diagnostics.py::
        # test_eos_error_does_not_scale_with_vertical_resolution pins.
        '384bf67ce6fff289e611b363720c46cac54275ec91b35019866d8cd1970bb347',
        'eb691c936bea58f860a6e05d01874481ef9feba9c55bed4a59240b3e4317101a'),
    'diff6': (
        '7dbcfb2d4e259ad36a3d29705e936a276b56e9ea52511c5f82054749e38302e9',
        '563febbc809cd53695782a77095b3ab01f60c64866657091b790f792f39a5394'),
    'diff6_seam': (
        '776ed7053a2dd697b0401e87600073e8dcfcdb3f7136fa4fcf73bb0b4972b464',
        '7af0e3ddeb9dc13d94992bf66ff65af583dd36e229de73236d0ed1da39f05d92'),
    'diffusion': (
        '00fb2e5d5550680fef154b4f67c7e282ad7ca1b170df59abdea89f888dad91ef',
        'f4958de3298bfcd764a5fd848aedfbb13938043ab91132db419408951c0a061e'),
    'dycore': (
        # Re-pinned for the small-step lateral-boundary fix:
        # small_step_init_uv / small_step_finish_uv took WRF calc_mu_uv's
        # periodic_x branch unconditionally, so a specified / nested / open
        # domain drew its west boundary face mass from the EAST-most mass
        # column.  They now take boundary_x/boundary_y and clamp when the
        # axis is not periodic.  A PERIODIC run is bit-identical -- the two
        # branches differ only at i = 0 / i = nx and j = 0 / j = ny -- and
        # that is measured, not argued: tilestream/test_gate.py's whole
        # output (51 physics cases, dry through full physics + MYNN +
        # Noah-MP, plus every negative control) is byte-identical across the
        # change, as is the whole-inventory SHA-256 after 1 and 8 steps on
        # three periodic rungs.  mp=8 is periodic-agnostic and dycore is not
        # an mp=8 translation unit, so the mp=8 numerics guarantee is
        # untouched; see tests/test_small_step_lateral_wrap.py.
        'af1c30640e10bce8c567a4c3dde466375b3ff1b44a39aa2a56513b80be2f4533',
        'af656dc03896f103d79b0b80fe59950f3e12932caec6c52c0c034fb087aef117'),
    'health': (
        # RECOMPUTED at the tilestream port, over the MERGED health.cu that
        # carries both re-pins below.  The pin that arrived with the port
        # was the integration side's alone and described a source that does
        # not exist in this tree, so it was red by construction and a pass
        # would have been coincidence; both rationales are kept because both
        # edits are present in the merged source.  The two values are the
        # file's own sha256 and the sha256 of the assembled compile string,
        # both computed from text: no card and no nvrtc were involved, which
        # is why this recompute closed on the CPU shard.
        #
        # --- from v1.8.7 ---
        # Re-pinned for the validation-gate launch geometry: blockIdx.y now
        # selects the descriptor and blockIdx.x a chunk within it, so the
        # full-state scan is no longer one block per field.  health is not
        # an mp=8 translation unit -- it computes no model state at all, it
        # only reads state and writes the compact health record -- so the
        # mp=8 numerics guarantee is untouched.  The record itself is
        # unchanged by construction: the epilogue is the same atomicOr over
        # status bits and the same atomicMin over (field << 48) | index,
        # both order-independent, so which block visits an element cannot
        # change what is reported (tests/test_health.py).
        #
        # --- from integration ---
        # Re-pinned for the STREAMED SAFETY FOLD (defect2-observer-fold).
        # health.cu gained one kernel, health_partial_tile: the per-tile half
        # of stability_report, emitted after a tile's step and before its
        # interior is scattered, so that under [tiles] with a host store
        # the run loop's nan / w_max / CFL gate reduces over the STORE rather
        # than over a DomainState the sweep never writes.  ADDITIVE: nothing
        # that was in this translation unit changed, health_partial and
        # health_final are byte-identical, and mp=8 does not compile health
        # at all, so the mp=8 numerics guarantee is untouched.  The fold's
        # own proof is tilestream/test_obsfold.py; the whole-gate evidence is
        # that tilestream/test_gate.py passes 233/233 across the change,
        # negative controls included.
        'c904e487cbf3cf0b6f6bbf6acad56158d2c3392d3c2a8a2e3ec6200d89f0cb03',
        '5f01ec47943352f0239f690945ec20924a23a70bbba6b5ca2437724871314b0c'),
    'kessler': (
        'fecf2e8028fda0ed4cb47fccce4c602d4632048d2dcbdd163613685ded952fdc',
        '530faef7f3bc5e5600d7a5f1086c9e4d0914a3aeda735214072bed30907c05d7'),
    'kf': (
        # Re-pinned for the column-workspace move (48ff6b813), measured
        # on node-1 with its two declared bit-moving placements named in
        # the commit.  Not an mp=8 unit.
        #
        # RE-PINNED by the WRF-parity shallow-TIMEC repair (PAR-CU-KF-05).
        # module_cu_kfeta.F:1598-1600 sets TIMEC=2400. for the shallow arm
        # and then ROUNDS it to FLOAT(NINT(TIMEC/DT))*DT; :2571-2573 sets
        # TIMEC = 2400. again immediately before the feedback loop, and the
        # six tendencies at :2603-2640 divide by that un-rounded value.  The
        # kernel divided all six by the rounded one.  Everything WRF keeps
        # rounded stays rounded here: the closure, AINCMX/AINC, DTIME/DTT/
        # NSTEP, the TADVEC comparison (:2569) and TIMEC_KF (:2387).  This
        # MOVES ANSWERS on shallow KF columns whenever DT does not divide
        # 2400 -- the pinned case is DT=90, where the rounded TIMEC is 2430
        # and every shallow tendency was 1.23% low (1 - 2400/2430).  kf is
        # not an mp=8 translation unit and thompson.cu is byte-unchanged.
        # Measured against the float64 mirror by tests/test_kf.py::
        # test_shallow_feedback_tendencies_divide_by_an_unrounded_2400.
        '6a2207a03f0a8d93413eafe0dd47d214fcd0f98f2e18a967f7922b05dc959bb3',
        '30c369d5b1dadcf2555e7372b71047f42b893180bfdb6b14e1da924f83fc3bf6'),
    'lbc_flow': (
        # 4febd041f supplies resolved WDM6/NSSL inflow concentrations.
        # docs/dev/qnn-specified-inflow.md records 4 CPU + 10 GPU
        # edge/corner, velocity and scalar transport controls.
        '68a743950e30e308676fada38f96ea3139d283447028edfeb85d2d64c36441fa',
        '233391c535605271d70b32dc5ce85321cb0ed633340dae3fe2cd3ebcf08ab6ba'),
    'lbc_state': (
        'fadf66fea201e4eac56e8a58d72b11940325b142a4e08fcc0b8db80fd78b53ec',
        '4cd1c59322d6a800eaebee8182a5a2a25413c37ffe4681f56a164a71ebd3a47b'),
    'morrison': (
        # MOVER: the deposition-freezing cold-trap bound, carried to the
        # engine line from lane/level5-owner 0c54221d2.  UNLIKE the two
        # sedimentation rearrangements this pin previously recorded, this
        # one DOES change FP results, and it is a DECLARED DIVERGENCE from
        # WRF, not a transcription repair: WRF F:2902-2905 sets MNUCCD
        # with no vapor-availability test, and the F:3009-3015 FUDGEF
        # rescale tests only the two matching sign pairs, so a positive
        # MNUCCD over a subsaturated state is never rescaled.  Below the
        # 159.4887 K POLYSVP crossover measured on this transcription the
        # extrapolated liquid curve falls under the ice curve, F:1315
        # clamps QVI==QVS, and the 0.999*QVS trigger fires at or below ice
        # saturation: the unfixed kernel drove qv to -1.87e-4 kg/kg out of
        # 5.55e-8 kg/kg available at 156 K / 250 Pa, dt-independent at
        # dt = 10/50/300/900 s.  Neither WRF nor a p_top-limited regional
        # domain reaches that state.  The bound ALSO engages where WRF can
        # reach -- 44 of 227 MNUCCD executions over the 28-column WRF
        # oracle fixture, 189.88-199.96 K, number moment scaled 0.197-0.898
        # -- moving case 13's ni on 16 levels (67% relative), qi on 23
        # levels (1.4e-3) and qv at 1.8e-16.  On the sm_86 device NO pinned
        # per-field max_ulp moves and the overall 1,709,094,255 / sr is
        # unchanged; the fixture mismatch count rises 3,512 -> 3,554 of
        # 10,948.  Recorded in the morrison-mp10 registry warnings.
        #
        # RE-PINNED by the WRF-parity PGAM density repair (PAR-MP-MORR-10).
        # module_mp_morr_two_moment.F:3918-3920 builds the PGAM reference
        # density as DUM = PRES(K)/(287.15*T3D(K)) from the CURRENT T3D, and
        # by that line T3D has moved: the tendency apply (:3710), the
        # sedimentation evaporation (:3735-3758) and the ice melt and both
        # homogeneous freezings (:3805-3843) all precede it.  RHO(K) is
        # built once at :1325 and is NOT that density.  morr_bound scaled
        # the stale RHO by RD/287.15 instead, so PGAM -- and through it the
        # cloud-droplet spectral shape and EFFC -- came from the entry-time
        # temperature.  The sibling site :1558 is downstream of a T3D change
        # too (:1504, :1511), and morr_process_level applies the same melt
        # to *temp before its call, so the one repair corrects both call
        # sites.  This MOVES ANSWERS wherever a level's temperature changed
        # within the step: EFFC feeds RRTMG, so radiation moves with it.
        # morrison is not an mp=8 translation unit and thompson.cu is
        # byte-unchanged.  Measured by tests/test_morr_rimed_ice.py::
        # test_pgam_reference_density_is_rebuilt_from_the_current_temperature,
        # which compiles module_source("morrison") and drives morr_bound
        # through ctypes.
        '270af004b00f0962d2f39cb120d14c246fdfcb99770c0c45b855b4cd1adff3d1',
        '1c4148b5e49a63feb7deecffcde9dd523fccccc89d79f26918102366c34b6ca5'),
    'mynn_pbl': (
        'b53ab90e634e61367afadfaa77667c8f2eb2430fc061ce9976509fe0e2f4490e',
        '87f80d06cc7724fd1277eefbf91738fe8eb0e774768ed64292cb1157f19a2d84'),
    'mynn_surface': (
        'a94de3ff2da95c37e12b437123b4a3807ac1318c524b339609f6024f4d21f85b',
        '891ec5d565c720afabab57169f1a3b1aa95efc3d1ac84e87ffbd4ebb239c57fe'),
    'nest': (
        # e1bdd7741 + d17f1d08b preserve global terrain coordinates and
        # sequential donor arithmetic through bounded child operands.
        # Measured 8d317e5ec: 34 focused + 29 resident CUDA controls;
        # 295d6ec0a: public moving/restart, 137 exact arrays per domain.
        # Re-pinned with diagnostics for 88fdf60b9's parent smoothers.
        '7aa2d1102fbbedaa9655850d1dc3403f3d995f9d1167da8fdd5a8ab7a2ac569f',
        '7b627569381652445f132287d81bda319eb5c7a156f854c46e034cdb98a226ac'),
    'nest_microphysics': (
        # Re-pinned for the mp=9 (Milbrandt-Yau) mixed-edge ratification
        # (audit R-003): the generic microphysics_edge_field kernel gained
        # the parent's base and perturbation potential temperature, its
        # pressure and the scheme's ck constant vector, plus the reference
        # pressure / R-over-cp pair and the flag saying whether the base
        # theta is a column or a field -- all read by the mp=9 arm alone,
        # placeholders for every other target -- plus my2_edge_field and
        # the two MY2 helper functions.  The absolute temperature that arm
        # reads is formed IN the kernel: the host has no array to form it
        # into on the tile-streamed nest route, where the launcher is
        # handed transition_parent_window's bounded namespace rather than a
        # DomainState, and a windowed mp=9 edge died there with an
        # AttributeError.  The frozen mp8_to_mp18_mass_diagnosed_field
        # entry point in the same file is byte-unchanged (its own bitwise
        # test still binds it), the P3 and NSSL arms are untouched, and no
        # mp=8 trajectory reaches the changed kernel: the edge matrix
        # launches only on MIXED nest edges, which an all-mp8 run never
        # resolves.  MEASURED on the 5070 Ti by
        # tests/test_milbrandt_nest_edge_gpu.py, whose windowed arm is
        # equal cell for cell to the resident parent's.  Previously pinned
        # at 541288c6/1598bc17 for the host-built temperature plane, and at
        # add31c69/fdf0a5b4 for the mp=50 ratification.
        #
        # RE-PINNED AGAIN for the mp=16 / mp=28 mixed-edge ratification:
        # the same generic kernel gained one wdm6_ccn scalar and the two
        # entry arms, and its field-code comment grew codes 24, 25 and 26.
        # Both reasons above hold unchanged -- the frozen
        # mp8_to_mp18_mass_diagnosed_field entry point is byte-identical,
        # and an all-mp8 run never launches the edge matrix at all.
        # Previously pinned at 9031874d/ae17b2a2.
        '6584d2be8f237eb3991ab4ccef25d5bf8e9426de1dc5f4c2884afd624ffefc0a',
        'ab4befec0c24f28d9f096ca962543a8db5f2f9e9ca5c195ad3ce83916a7b99c2'),
    'noah': (
        # RE-PINNED by the WRF-parity FRZX repair (NOAH-01).  WRF renames
        # this quantity twice on its way down and gpuwm followed the NAME
        # instead of the ARGUMENT: REDPRM (module_sf_noahlsm.F:2477-2478)
        # builds FRZX = FRZK*FRZFACT; SFLX passes FRZX at :769 and :784;
        # the receiving dummy is spelled FRZFACT in NOPAC (:1909), SNOPAC
        # (:3015) and SMFLX (:2670); SMFLX passes it on at :2785/:2794/:2803
        # and SRT (:3655) names it back to FRZX, spending it at :3795 as
        # ACRT = CVFRZ*FRZX/DICE.  noah_column was handing noah_smflx the
        # bare FRZFACT, so the frozen-ground infiltration limit ran on a
        # dimensionless ratio instead of FRZK*FRZFACT and SRT's ACRT
        # exponent collapsed.  This MOVES ANSWERS, hard, on frozen ground:
        # measured against the WRF oracle the sfcrunoff distance falls from
        # 60,641,303 ULP to 2,812, and sh2o, smcrel and smois go from 6,508
        # / 4,729 / 1,627 ULP to exactly bitwise.  noah is not an mp=8
        # translation unit and thompson.cu is byte-unchanged.  Measured by
        # tests/test_noah_wrf461_parity.py (the re-pinned BASELINE_MAX_ULP
        # table), ::test_the_mirror_reproduces_wrfs_frozen_ground_infiltration
        # and ::test_the_kernel_hands_smflx_the_same_word_the_mirror_does.
        'd7ae4d2ccac5ca6c32c575031337a3dfa7dfe8c37a14bf64167e57be3ac373fc',
        'b8adda8aa53d0749c1a08f9a2e760930d7046654bfb04e6e72ffa170725461dc'),
    'noahmp_bareflux': (
        '54fb5065e95b24d4cf676e2deda29bae44b3e9305d3d98cbc1abf5ed55f444ce',
        'fbb19fc8b5668ea2edbcc1270f8ffe367475124ffa0ce99d3bc34639f3f31e9f'),
    'noahmp_driver': (
        'bd555be10ccade5a5bdddcaf4c56b7f4353dae1208fc48a5586fb7ce7d32d643',
        'f1913fe0054adb74188effa6499b799e989cfb1e96ecd676edcd43df0083030e'),
    'noahmp_energy': (
        # RE-PINNED at 2.7.0 by the licence cut.  `nmpe_tanhf` loses glibc's
        # redundant `if (ix == 0) return x;` -- the one edit in the FDLIBM
        # group attributable to reading glibc's text and to nothing else, so
        # it is expression this Apache-2.0 distribution should not carry.
        # FDLIBM never had it and the tree already shipped tanh without it
        # (mynn_pbl.cu, mynn_dmp_sibling.cu).  It is redundant because the
        # |x| < 2**-55 branch returns x*(1+x), which is x for both signed
        # zeros.  NOTHING MOVES: the two forms were compiled side by side
        # with the shipped bodies and a host shim and compared on all
        # 4,294,967,296 float32 bit patterns -- 0 differ, tanh(+0) = +0 and
        # tanh(-0) = -0 in both.  Four lines of comment came in with it.
        '4fd4b5a86ec97371e01aac13d2c5e40e40546b64a77c39e6fb674c96be69d209',
        'bd91acf9bf987c5116e894b67e0ecbe47ae5bad5f1237095fe822da7fcd6b0a2'),
    'noahmp_fluxprep': (
        'eef473608e9d1c0176574c6bd72183659249b42a022f8d63d77c611de506f4ed',
        '4ae139ef5d31234b5e9fdbc04c4cfd3c5156cd91d0326e5544b5b78f7c17fb31'),
    'noahmp_leaves': (
        # RE-PINNED at 2.7.0 by the licence cut, and for the same reason as
        # noahmp_energy above.  `r_log10`'s zero path was glibc's
        # `-two25 / fabsf(x)`, a division whose only work is to raise
        # divide-by-zero on the way to -inf; FDLIBM divides by a `zero`
        # variable instead, so the spelling is glibc's own.  ArWen returns the
        # -inf directly.  NOTHING MOVES: compared on all 4,294,967,296 float32
        # bit patterns -- 0 differ, log10(+-0) = 0xFF800000 in both.  Five
        # lines of comment came in with it.
        '4a3b6ce94993ab1c7055de39880e1209b54527751dcda07f26aa4b4a612f3497',
        'e27a5b92bd5f9ee64e0e9a9e496228e4b4b7b4a43ca3f7708a272a8234030ebd'),
    'noahmp_libm_slab': (
        '0144fa7d142a8d24f5f0f52bd0dade987312cd97c2a2efad9a6f33edb1a35fda',
        'c7cdc57aa7d3d507d2b57935df13e16dba87dd783384e28a7240b95a536f39a7'),
    'noahmp_radiation': (
        'c26d7a68ed86bd7182ddbea5e2005fb4761805dd5eaf13c3115e640eb2234c22',
        'a454d07fefa080f986423d2f74bf0fe1ecf85959638e02a7a447b44d68aea24a'),
    'noahmp_sflx': (
        '47f51fed351b3f720203c07ea2d5e5a8902f4162afe17b634f5196447728365f',
        'f17caed98ec58c4d9836778f3a9edea9e88cf13808dbe64d5e099fd81ffebc77'),
    'noahmp_snow': (
        'f46f850fb54acf3dadcaf7a7df9bb4c09d107be8de0c577b704536a20271b42a',
        'd375fe9595c514690fb7fe6db756b76f4c2341862dd5c3b70bae2131ccfcde22'),
    'noahmp_soilwater': (
        'a53adc6aaa1a46974b13497676e682c56587176442bf1a26003b4d377d87688e',
        'd9e8788e7bb479104faabbb3ca5d6ae698768a394a60f54b0e2840862bfbfa5a'),
    'noahmp_thermal': (
        '2598ca76b7f9c0d6de35631d5113901d1e25f24a014c8ea4e0df490158d16c86',
        'f7eef131507a29def54e2387686f34a4807851354f446684deaf5322dc226753'),
    'noahmp_vegeflux': (
        '2178b13989853a7433869bb35e7974df004ddb9f441a96946672c03730472b3c',
        '688723dc9eef069b5e82338a8023438636d61db2db1dfd583cd1ced03b923a19'),
    'noahmp_vegprecip': (
        '9ce5667599d111be0efc7cb0871e5719d70259bb95b30fbe923e01964b75c26c',
        '599e85f0fe59a28ae5fd5437e47db7f41c31abd21783979f66b987b212f1f41a'),
    'noahmp_water': (
        '4154bace0d97235503d4ca9ed6cb4877c8543762f2f384dfc8883fe3b2ed429e',
        '6cfeaef3fd00b8054d1761fd8bcfd6a920a6fb476f53986672ef300cbadd53ba'),
    'nssl2': (
        # RE-PINNED by the WRF-parity qxmin(lh) repair (NSSLA-01).  The Bigg
        # rain-freezing gate used the graupel minimum mixing ratio 1.e-7 set
        # at module_mp_nssl_2mom.F:2095, missing the overwrite eight lines
        # later at :2103: `IF ( lh .gt. 1 .and. lnh .gt. 1 ) qxmin(lh) =
        # 1.0e-12`.  Under the option-18 default (ipconc=5 by
        # module_physics_init.F:4633-4641) the index block at :1650-1667
        # gives lnh = 14 and lh = 7 (:658), so the override is
        # unconditionally live at both use sites -- the minimum-transfer
        # gate (:17653) and the volume/SETVT gate (:14205/:14213).  The
        # sibling nssl2.cu:4274 already used 1.0e-12f.  This MOVES ANSWERS
        # in the five-decade window between the two constants, where the
        # transfer was being suppressed to zero.  nssl2 is not an mp=8
        # translation unit and thompson.cu is byte-unchanged.  Measured on
        # the device by tests/test_nssl2_gpu.py::
        # test_bigg_rain_freezing_uses_the_two_moment_graupel_qxmin.
        # 9a8b0da23 rounds Bigg's rain-mass multiply before subtracting,
        # preserving the WRF FP32 state used by the later number bound.
        # This changes mp=18, not Thompson's mp=8 translation unit. The
        # exact parent source reproduces both old pins through the real
        # loader; retained NSSL GPU oracle comparisons grade arithmetic
        # without widening the existing tolerances.
        '0541eb4f5353d8379af80100fb231893698fcde0521706357a5afc7c88569679',
        '24a7e4af3fab46b6c9dffbc58438287eebc7c5151dde8c11ec815491476ff044'),
    'nssl2_diagnostics': (
        'a95ae9e0bc3dd20a13865cfa6d1148d2a78ee5d7c17c9c1bca9a0c8dbdf19868',
        '331b4a9734959260ab515216e24ee7100eac18d8e6d646b1e0e8bf21c0c23374'),
    # The three NSSL translation units below are re-pinned for the WRF
    # v4.6.1 variant family (nssl_hail_on / nssl_ccn_on).  Each gained one
    # trailing `int` kernel parameter and a branch on it: the CCN load and
    # store in nssl2_driver_support, the graupel-to-hail conversion call in
    # nssl2_fused_gs, and the nucleation pool in nssl2_nucond.  None of them
    # is an mp=8 translation unit, so the mp=8 numerics guarantee this file
    # exists to protect is untouched.
    #
    # The pins move; the mp18 default lane's OUTPUT does not.  That is the
    # justification, and it is measured rather than argued: with these exact
    # bytes, tools/nssl2_mp18_digest_probe.py reproduces all 30 committed
    # SHA-256 field digests in evidence/nssl2-variants/mp18-digest-baseline
    # .json byte for byte (re-run on this tree: 30 reproduced, 0
    # mismatched).  That is what rules out the FMA-contraction risk of
    # adding a parameter and a branch to a fused kernel.  Each digest below
    # was re-derived from the tree by this test's own fixture, not edited
    # to match.
    'nssl2_driver_support': (
        'd1c729369bdf59859f178622402f138e2c9b67f4f12fa5661162924c7cf542ec',
        '0ae24c4f406b91a4d849f5ce171838264be81254544dc04693550f3355a14511'),
    'nssl2_fused_gs': (
        # RE-PINNED by the WRF-parity vertical-velocity centering repair
        # (G-01).  The kernel averaged interface w to mass level TWICE,
        # delivering 0.25*w[k] + 0.5*w[k+1] + 0.25*w[k+2] where
        # module_mp_nssl_2mom.F:14174-14176 delivers 0.5*(w[k] + w[k+1]) --
        # a half-level upward shift plus 1-2-1 smoothing.  The premise of
        # the comment that put it there ("WRF's microphysics driver supplies
        # a mass-level W field") is false: solve_em.F hands the driver the
        # staggered grid%w_2 and :2827 copies it into the GS slab with no
        # de-staggering, so wvel = 0.5*(w(kp1)+w(kgs)) IS the single
        # interface-to-mass average and its Min(nz, kgs+1) clamp is on the
        # upper INTERFACE.  This MOVES ANSWERS for every mp_physics=18 run
        # with vertical shear in w: wvel is the linear factor in WRF's
        # icenucopt=1 primary-ice source (:20733-20738) and also its > 0
        # gate.  nssl2_fused_gs is not an mp=8 translation unit and
        # thompson.cu is byte-unchanged, so the mp=8 numerics guarantee is
        # untouched.  Measured against WRF's own instrumented oracle by
        # tests/test_nssl2_fused_gs.py::
        # test_official_wrf_oracle_pins_the_single_average_and_its_top_clamp
        # (all 240 rows satisfy w_center == 0.5*(w_lower+w_upper)) and ::
        # test_official_wrf_oracle_rows_reach_qiint_with_a_shifted_w_center
        # (12 rows reach the qiint computation, where the old rule overstated
        # wvel by 1.18x to 2.06x); the kernel's spelling is held by ::
        # test_cuda_centres_interface_w_onto_mass_levels_exactly_once, which
        # replaces a source pin that asserted the two-stage average.
        '8b4ad70fcde7a2fb5889c07045505c25d77440d1b96913adac1b54eaf1187e2a',
        '4606e9061c788322ffa6a92a3f9a1a58cd5bcd4b4e7753eaccd3b2c96cd72b30'),
    'nssl2_nucond': (
        # RE-PINNED by the WRF-parity raw-w repair (N-02).  The low-T cnuc
        # hack at module_mp_nssl_2mom.F:10122 tests the RAW staggered
        # element `w(igs(mgs),jgs,kgs(mgs))` -- the cell's own bottom face --
        # and the kernel averaged the two interfaces onto the mass level
        # first.  That is not WRF being sloppy: NUCOND spans :9611-12215 and
        # the only wvel assignment inside it is at :10381, 259 lines
        # downstream, so the routine has no averaged w to read at :10122.
        # The mass-level average survives everywhere WRF does compute one.
        # This MOVES ANSWERS on sheared columns below 265 K where the raw
        # face and the average straddle the 2.0 m/s gate.  nssl2_nucond is
        # not an mp=8 translation unit and thompson.cu is byte-unchanged.
        # Measured by tests/test_nssl2_contract.py::
        # test_the_low_temperature_cnuc_hack_reads_the_raw_staggered_w,
        # a two-sided gate: it also requires the mass-level average to
        # survive at its two legitimate sites, so it cannot go green by
        # deleting the averaging everywhere.
        '224e22e7f0ab40965444d9dbdca6796c6d43a3404bdbb434425954d44d2b2555',
        '311068aa8b1f2dd2d0a5cb38cf5d936c2b2e5ac9e6c5e4f764695e3aa01418a9'),
    'nssl2_qvexcess': (
        '6906dcd9f8822d73d87ff3cb6e545a1b1ddef567c16c658435d4f669f1f369dc',
        '89d27036499b7f57d780c594308766b83ef711fbc9d9c5d0ece2e79c270f6626'),
    'openbc': (
        # 6d9c9b999 folds CFL over owned columns of one domain sweep.
        # Measured 0d2a79ed4: 41 CUDA controls; public resident/streamed
        # 448 fields and resumed/continuous 150 fields exactly agree.
        # 399d95a86 appends the adaptive timestep's w_cfl_stat probe;
        # d7c5a9eca corrects its strict-threshold comment. The entire old
        # file remains an exact prefix: existing boundary kernels do not
        # change. Restoring the parent source through the real loader
        # reproduces both old pins; test_wrf_cfl_histogram grades the new
        # measurement against its independent CPU reference.
        'c7217de931f41cc30ccb8f31281ab8da2c1bf770c0621a16522f0b656eb3b872',
        'c15740b46301a3814fbe5c8ecab6daeff20e6997b012798873543c5aa601e01c'),
    'pd_advection': (
        '606e396872b2c42bafcff8d46d6a4c16d0f1c4f0fc796bb1728f4ae3678c309c',
        'd9e8649915c1a8bd0b65354131baa8fc29d00f921d61849bb4fa0d47b065c9ea'),
    'refl': (
        'ff4e3c6dd532be49fc866692e829e9a7efa7ac1c65a95b4cd1beb910f39d07b6',
        '8e3843a3884edee0ed8ade032401df5750ccde3df2d396a4d17a59267d0f27a1'),
    'rrtmg_lw': (
        # Re-pinned for the buffer-march positivity fix (7ce2f5de7,
        # dp == DELTAP bitwise for in-contract tops per its message).
        '391e459a9c174e07fc0fddba6de6e59ea836f04edccd00b7e0ae6c0d41ae2555',
        'c39e41b508935b52b22914c1385dc636451d8aa34e10b8eb316e4d7e98b944b9'),
    'rrtmg_lw_chain': (
        'a71a779ea3a733e422414ee5d5d885cb904972fe32ac2bfe6438d6c5214c1252',
        'ee8f4c95e52248d0b641a48a8532d44d6aff0b75344dda59a416a3905ae972c1'),
    'rrtmg_lw_taugb02_10_11_12': (
        'fe5d57d1eb2649005d2748ae40f849c340b03a5f3d3a61c7fb506d0c69792485',
        'f7d4c98b0ed523e2c5912d4810925c60c3bcbb52d3e1a40d06a2708686cb632c'),
    'rrtmg_lw_taugb03_05': (
        '5de3f0c7c700cff993d5d5a3caa868bbd68d810ea30a902e6e92d652e5612cf7',
        '8a86d6cfcd46ea76bbc3027cb1f68f8842d4e37de5b011d681a5c502c1794b2e'),
    'rrtmg_lw_taugb06_09': (
        'ab677eb0de0faeb2b7a3f4a5923c6777e11128ab7950d9b63abc9cdb61125157',
        '4899dd33838614a7cf94bf3e74291039ed7bb3b35806495a22d3e1c559d01c01'),
    'rrtmg_lw_taugb13_16': (
        '104c3157451dc878e3f9d751db1cf01a4368b1a2892825678370c5159a787ff6',
        '56f315a0258d902c69f99abacbe5e53005f50ce478725c146647d3df26c71522'),
    'rrtmg_mcica_wrf': (
        'edb6bcb71a9d0763d3576b602db0afa05ec5eddc9bc63548f4991f34bd6d718c',
        '07af9ba6f5a0ed7e3c735f0b574041aace48f306e2641e7563252ae992a2cfd9'),
    'rrtmg_sw': (
        # f7c2aadea removes an unused macro; host launch coverage now
        # includes every layer. Measured d8ef9d086: 156 CUDA controls
        # including independent WRF 80/129-layer reference fixtures.
        # Re-pinned for the one-instruction subnormal armor (65944605a,
        # exact in binary64 per its message; witness at 25ad40769).
        '0301818b55046062ef0b89edc1c2ddc85adb361a4bfd03bc278f56c5fb139aef',
        '288e905955ad26e2f216928eb81cba05800d75c0e29e2172f34b9f89002d36cd'),
    'rrtmgp_cloud': (
        '015aec6065be8a23bcec1ce5421ae28cfbc74de1d6a7713a75bc1a78d1f7bc08',
        '5976824ca813f3e40a8b6d73ccb88d39b333f110054bb502d482817a3a7c6ad7'),
    'rrtmgp_gas': (
        # Re-pinned for the contributed RRTMGP optimisation (51819ed0f,
        # plus af2fe5e7f's follow-up): cell-per-block gas optics with
        # shared memory, __f*_rn-pinned expressions.  Equivalence is the
        # commit's own bitwise gates
        # (test_gas_vmr_fused_kernel_matches_the_expression_reference and
        # siblings), run green on sm_86.  Not an mp=8 unit.
        '06249ff6c626dc914912c27465dc5cd3aa0ac3969f40b7cc515c639871594d04',
        '01ad92e76fedbe87270f701260a324fb182446657a59030b7023458ae6bc0b3e'),
    'rrtmgp_mcica': (
        # Re-pinned with rrtmgp_gas for 51819ed0f (mcica jump operators);
        # the same commit's bitwise gates cover it.  Not an mp=8 unit.
        'e59fe155d595c74a8c17619758091633652773a509a56030d3dc1a7d90b11039',
        '49a078da7c25f9ecc321b26a827a1972c641808613d0a8aac9143291805dfd29'),
    'rrtmgp_rte': (
        # Re-pinned for 51819ed0f: in-solver Planck derivation, fused
        # finalize, warp fold -- each with an in-commit bitwise
        # equivalence test against the kernel it replaces.  This unit
        # also carries the rrtmgp_planck_common.cuh header grant; the
        # loader-inertness gate below defers to
        # test_kernel_loader_inert's closed mapping for exactly the
        # modules the loader names.  Not an mp=8 unit.
        '22e0afb97b08a24e6daf30a3a03197558d6f452a1bfe225c729e83fdc12c6087',
        '3204abbd31c1600c43a2f1bbf7e7c2d416b5a4be77fac81041a8a9dcbd68d63e'),
    'rrtmgp_validation': (
        'c3a2554827e7c39db269d0fa1d5ad594d1623d6974be859a1ae3a044d438951f',
        '36f7a4dccc8c66be225e16fdd6088f0fafa7a357ac1aba9b4a99f98af297c370'),
    'ruc': (
        # Re-pinned for the RUC_NZS tier ladder (lane/ruc-column-nzs, step 3
        # of 3, final): the ladder, the 161-token macro substitution that
        # uses it, and the `#elif RUC_NZS == 6` arm of the depth table.  The
        # forecast column now compiles at both geometries WRF defines.
        #
        # The nine-level translation unit is unchanged through all three
        # steps.  Every macro expands to a bare decimal literal, so
        # `real zshalf[RUC_NZS]` is the token stream `real zshalf[9]` was,
        # and the `#elif` arm is deleted by the preprocessor at nine.
        #
        # The pin moves; the compiled binary does not.  Proved three ways in
        # tests/test_ruc_nzs_tier.py, all device-free: the shipped source is
        # run BACKWARDS to the pre-lift file and hashes to
        # d446b7462e4952416d3e21482b051823766a6f675163236686c7d9fab7fbbdb7
        # (the digest this row carried before), the token streams out of a
        # real host preprocessor are equal, and `nvcc -ptx` emits identical
        # PTX.  Each comparison has a negative control that must differ.
        #
        # MOVED A SECOND TIME, and this time the binary DOES change: the
        # `RUC_NZS DZSTOP` block.  ruc_soil_finalize computed
        # `dzstop = 1 / (0.01f - 0.0f)` -- WRF's NINE-level
        # zsmain(2) - zsmain(1), written as a literal rather than read from
        # ruc_soil_layer_depth, and therefore invisible to the macro sweep
        # above, which was looking for extents.  Harmless while nine was the
        # only geometry; at six it divides by 0.01 where the grid says 0.05,
        # and the kernel returned a ground heat flux five times too large --
        # MEASURED as grdflx -337.1 W m-2 against the host lane's -67.4 on
        # the same column, before the fix.
        #
        # WHY THIS ROW MAY MOVE FOR IT.  At nine levels
        # ruc_soil_layer_depth[1] IS 0.01f and [0] IS 0.00f, so the
        # subtraction, the divide and every number downstream are unchanged,
        # and that is measured on the hardware rather than argued: the full
        # 43-field host/device driver comparison in
        # tests/test_ruc_nzs_device.py is max_ulp 0 at nine, over snow-free,
        # water+ice and snow columns, and the eleven-file RUC suite is
        # unchanged against its lane baseline.  The generated code is NOT
        # unchanged -- a __constant__ load where an immediate was -- and
        # test_the_named_fix_is_a_real_change_to_the_generated_code asserts
        # that difference, so the fix cannot be waved through as inert.
        # The reconstruction above still reaches
        # d446b7462e4952416d3e21482b051823766a6f675163236686c7d9fab7fbbdb7,
        # through one extra named inversion step for this block, which is
        # what keeps "the lift changed nothing else" checkable.
        '2b176b92364530762032a815de270373143725c553437d19207152c84213ac1f',
        '55894935fdfde9f3ac683e2bbf151e0c20f677744b37a9a438aa61bb91635935'),
    'saxpy': (
        '8637cb5cb0a6878d59a32454a6ae662a8b18c0be4d94c067fbde1e4bf5bad079',
        '7b7083065716a2b3b58d47c3ac456ea8d0c1a38ec771219897917bb0b1b79cb2'),
    'sfclay': (
        # Re-pinned on the 1.6 release line for the sm_120 DAZ hardening.
        # A roughness small enough to overflow the FP32 quotient sent Inf
        # into every downstream similarity quantity, so the logarithm is
        # rescued in float64 and a NaN roughness now propagates instead of
        # being floored into a plausible finite contrast.  sfclay is not an
        # mp=8 translation unit and thompson.cu is byte-unchanged across
        # this line, so the mp=8 numerics guarantee is untouched; the
        # healthy path is proven bit-exact under pinned fixtures and an
        # adversarial sweep.
        #
        # RE-PINNED by the WRF-parity USTM repair (SFC-01).  The kernel
        # produced no USTM at all: module_sf_sfclay.F:800-804 (and
        # physics_mmm/sf_sfclayrev.F90:759-763) relaxes a SECOND friction
        # velocity on WSPDI = sqrt(ux*ux+vx*vx) -- the speed WITHOUT the
        # Beljaars/Mahrt-Sun vconv/vsgd correction, without WSPD's 0.1 floor
        # and without UST's land floor -- and USTM is unconditional Registry
        # state (Registry.EM_COMMON:1954) that WRF hands to tke_rhs and
        # vertical_diffusion_2 (module_first_rk_step_part2.F:914,:1066).
        # ArWen recomputed it in a CuPy post-pass gated on
        # km_opt in (2,3,4) and bl_pbl_physics == 0; that post-pass is now
        # deleted and the line lives where WRF has it.  This MOVES ANSWERS:
        # <=1 ULP on the LES path (the kernel contracts uu*uu+vv*vv into an
        # FMA where the three CuPy kernels did not) and first-order for
        # km_opt=2 with a PBL scheme on, where the TKE surface shear source
        # was identically zero.  sfclay is not an mp=8 translation unit and
        # thompson.cu is byte-unchanged, so the mp=8 numerics guarantee is
        # untouched.  Measured on the CPU authority by tests/test_sfclay.py::
        # test_ustm_relaxes_on_the_uncorrected_wind_speed and ::
        # test_ustm_takes_neither_the_wind_floor_nor_the_land_floor, with the
        # kernel held to the float64 mirror by ::
        # test_sfclay_kernel_writes_ustm_and_not_a_copy_of_ust and by every
        # standing SFCLAY_OUTPUTS sweep, which now grades ustm.
        '1e0687817889897b950b5ae47e0f0cff58f0a97b32e5b65bcf3bf1ce15215f22',
        'c26f5f0590d0e42ca033795ba801acd269e07fb7ed8f54213dfb385d5cff7f2b'),
    'smag2d': (
        # Re-pinned on the 1.5 integration line: feature/les-integration's
        # verified km_opt=2/3 work edits smag2d.cu after this table was
        # frozen on the mp28 lane.  smag2d is not an mp=8 translation unit;
        # the mp=8 numerics guarantee is untouched.
        #
        # Re-pinned again on the P1 moist lane, which adds the spec 3.3
        # MUTATION CONTROL to wrf_calc_n2: an `#ifdef
        # GPUWM_MUTATE_MOIST_N2_FORCE_DRY` block that assigns
        # `saturated = false` for instrument qualification.  Both digests
        # move because both are digests of SOURCE TEXT.  The generated code
        # of the production build does not: the guard's macro is never
        # defined in the assembly `load_module` builds -- asserted by
        # tests/test_les_moist_n2_mutation.py
        # ::test_the_mutation_define_is_absent_from_the_production_assembly
        # -- so the preprocessor deletes the block before nvrtc sees it, and
        # the production predicate line above it is byte-identical to what
        # it was.  The mutant is a SEPARATE translation unit compiled
        # through load_module_int_defines under its own cache key and its
        # own kernel-manifest entry; it is never this one.
        'c57ebd81fc6478ab58f4376043a0931feddf89c4146e7ed88b089133f559332e',
        '2e5a33dcfd34a46c47d8408206c7e13bb14ee80320858f9c7adf796c118223e8'),
    'spec_bdy': (
        'bcc7090fbbb8ea307bd6dd6c65ab9b8a3f56948c4752ae3d744127b450d20161',
        'bc03ed595bacc546d8e041fbb1d11b5bb3b3b90760ef06ea1dd1f0f18b4de931'),
    'thompson': (
        '3ca6b7e902d2d77ca9881a66eb484141df552c8ea30a0076bd07023e7255e760',
        '8cb23f0a78b1e48a402266fd4f841b2facdc7a3ee4b7d53e3c5482aae973775d'),
    'uh_diag': (
        'cbfc98e8d025a4511fd7f8a41ca4bd163c261da4a48dec22bb979ec5a496b14e',
        '9dc88c6e14b2aaaa4249a9f844dc231f105431623375c988a2894e322de2f3ea'),
    'vert_interp': (
        'ab608d651d99f882a42881f7b977b29d8879f474ea26404a79d06c77a86125d9',
        '65b0fe8a9aadda83084e0286d3f70570e251bb6d2fb1a0636a0530103b8d9d89'),
    'wsm6': (
        '0526192b79d90d3be7c733a475987216d37cc81b17f8de4f1fe3e4220a6b81d7',
        '1a6d20da0d450f235227fe609bdb12b368d96aec5ac231752074ff4dd9cc50e6'),
    'ysu': (
        # Re-pinned for the column-workspace move (17cf943ef): per-thread
        # local arrays became a caller-sized global workspace, stride-32
        # lane indexing, tile offset col0.  Placement-only -- proven by
        # --fmad=false digest equality (0 differing words of 903,168) and
        # the WRF v4.6.1 ULP-equality parity pins, with the instrument
        # validated in both directions
        # (docs/kernel_local_memory_bounds.md).  ysu is not an mp=8
        # translation unit; the mp=8 numerics guarantee is untouched.
        #
        # RE-PINNED AGAIN by the WRF-parity pblflg repair (par-pbl-ysu-01
        # and -04).  Two branch structures, both re-read at
        # phys/physics_mmm/bl_ysu.F90 before the edit: :703-728 guards the
        # thermal-enhanced Richardson sweep with if(pblflg(i)) and the
        # kernel did not, so a column WRF holds in the local-K regime was
        # being switched to full non-local YSU at convective onset; and
        # :765/:766 are two INDEPENDENT statements, so nesting the second
        # inside the first let a theta-li revival reach :832 with kpbl == 1
        # and index one level below the column.  This MOVES ANSWERS on both
        # paths -- see the release note -- and both are measured on the CPU
        # authority by tests/test_ysu.py::
        # test_the_thermal_enhanced_sweep_cannot_raise_pblflg and ::
        # test_a_theta_li_revival_that_leaves_kpbl_at_one_is_extinguished,
        # with the kernel's own spelling held to the mirror's by ::
        # test_the_kernel_spells_wrfs_two_pblflg_rules_like_the_mirror.
        # The same bytes also carry par-pbl-ysu-03 (the enhanced sweep's
        # result reaches the theta-li scan unclamped, because :718-728 has
        # no counterpart to the clamp at :646 and :823), landed in the same
        # window by the lane that owns it; the kernel and
        # gpuwm.verify.npref.np_ysu_column carry it identically.
        '251dc8469b13a18a86f8a4a6546ceb9bb7680407a3f4afa03290d1140a48964b',
        '4725831d8c4ad0660eff7d792f34c3eb8eb71e6e672293612100634c68200a7f'),
}

# -- R2 --------------------------------------------------------------------

CLASSIC_TABLE_ASSETS_PIN = (
    ("qr_acr_qg_V4.dat", 74_966_480,
     "89b779855847b2acdca1b40e24c5f1bd89b0c6ed105ca91a5a076d80c2437c3f"),
    ("qr_acr_qsV2.dat", 43_764_288,
     "47350be20bd59c9f31378dd5805ce7d35fd14bebcfafb4ade56626f6eed818d7"),
    ("freezeH2O.dat", 254_944_848,
     "c235d1ce6f8750a671b2273d0e216ed3acf9a869bfd52a14676826f87aab5c02"),
    ("thompson_aux_tables.dat", 6_164_536,
     "a1bda803cdb53aedce8a2970c04c355fad19e3744398e1c9b13a876f09730547"),
)
TABLE_SET_ID_PIN = "wrf-v4.6.1-classic-thompson-mp8-gfortran13-v1"
WRF_REFERENCE_VERSION_PIN = "v4.6.1"
WRF_REFERENCE_COMMIT_PIN = "d66e442fccc04111067e29274c9f9eaccc3cef28"

# -- R3 --------------------------------------------------------------------

EXTRA_MOIST_SPECIES_MP8 = ("qi", "qs", "qg", "nr", "ni")
EXTRA_MOIST_SPECIES_MP10 = ("qi", "qs", "qg", "nr", "ni", "ns", "ng")
TRANSPORTED_NUMBER_SPECIES_PIN = ("nr", "ni", "ns", "ng")

# -- R4.  Captured for the fixed probe config nx=8 ny=6 nz=4,
# moist=True, mp_physics=8 (see freeze.PREFLIGHT_PROBE_CONFIG).
# ------------------------------------------------------------------
STATE_ARRAY_SHAPES_MP8 = {
    'al': (4, 6, 8),
    'al_pp': (4, 6, 8),
    'alb': (4,),
    'alt': (4, 6, 8),
    'c1f': (5,),
    'c1h': (4,),
    'c2f': (5,),
    'c2h': (4,),
    'c3f': (5,),
    'c3h': (4,),
    'c4f': (5,),
    'c4h': (4,),
    'cosa': (6, 8),
    # The EOS's float64-derived base-thickness correction and the
    # float64-differenced full-level coefficient drops that replaced the
    # two cancelling subtractions in calc_p_alpha.  Derived setup, priced
    # like every other allocation; see gpuwm/core/kernels/diagnostics.cu.
    'dc3f': (4,),
    'dc4f': (4,),
    'dn': (4,),
    'dnw': (4,),
    'dphb_resid': (4,),
    'e': (6, 8),
    'effc': (4, 6, 8),
    'effi': (4, 6, 8),
    'effs': (4, 6, 8),
    'f': (6, 8),
    'fnm': (4,),
    'fnp': (4,),
    'h_diabatic': (4, 6, 8),
    'ht': (6, 8),
    'msft': (6, 8),
    'msfu': (6, 9),
    'msfv': (7, 8),
    'mu_pp': (6, 8),
    'mub2d': (6, 8),
    'mup': (6, 8),
    'mup0': (6, 8),
    'ni': (4, 6, 8),
    'ni0': (4, 6, 8),
    'nr': (4, 6, 8),
    'nr0': (4, 6, 8),
    'p': (4, 6, 8),
    'p_pp': (4, 6, 8),
    'p_pp_old': (4, 6, 8),
    'pb': (4,),
    'ph_pp': (5, 6, 8),
    'phb': (5,),
    'php': (5, 6, 8),
    'php0': (5, 6, 8),
    'qc': (4, 6, 8),
    'qc0': (4, 6, 8),
    'qg': (4, 6, 8),
    'qg0': (4, 6, 8),
    'qi': (4, 6, 8),
    'qi0': (4, 6, 8),
    'qr': (4, 6, 8),
    'qr0': (4, 6, 8),
    'qs': (4, 6, 8),
    'qs0': (4, 6, 8),
    'qv': (4, 6, 8),
    'qv0': (4, 6, 8),
    'rdn': (4,),
    'rdnw': (4,),
    'rmu_t': (6, 8),
    'rph_t': (5, 6, 8),
    'rth_t': (4, 6, 8),
    'ru_t': (4, 6, 9),
    'rv_t': (4, 7, 8),
    'rw_t': (5, 6, 8),
    'sina': (6, 8),
    'th_pp': (4, 6, 8),
    'thb': (4,),
    'thp': (4, 6, 8),
    'thp0': (4, 6, 8),
    'u': (4, 6, 9),
    'u0': (4, 6, 9),
    'u_pp': (4, 6, 9),
    'v': (4, 7, 8),
    'v0': (4, 7, 8),
    'v_pp': (4, 7, 8),
    'w': (5, 6, 8),
    'w0': (5, 6, 8),
    'w_pp': (5, 6, 8),
    'ww_pp': (5, 6, 8),
    'znu': (4,),
    'znw': (5,),
}
SCRATCH_SLOT_REGISTRY_MP8 = {
    'acoustic_a': (5, 6, 8),
    'acoustic_alpha': (5, 6, 8),
    'acoustic_c2a': (4, 6, 8),
    'acoustic_gamma': (5, 6, 8),
    'acoustic_mu_pp_old': (6, 8),
    'acoustic_th_pp_old': (4, 6, 8),
    'adv_ru': (4, 6, 9),
    'adv_rv': (4, 7, 8),
    'adv_rw': (5, 6, 8),
    'integration_health_aux_ptr': (2048,),
    'integration_health_bounds': (1024, 2),
    'integration_health_field_ptr': (2048,),
    'integration_health_field_size': (2048,),
    'integration_health_flags': (1024,),
    'integration_health_partial': (1, 9),
    'integration_health_planes': (1024,),
    'integration_health_result': (8,),
    'integration_health_status_bits': (2048,),
    'integration_health_validation': (4,),
    'moist_pd_q0': (4, 6, 8),
    'moist_rq_t': (4, 6, 8),
    'mp_dz8w': (4, 6, 8),
    'mp_graupelnc': (6, 8),
    'mp_graupelncv': (6, 8),
    'mp_pii': (4, 6, 8),
    'mp_rainnc': (6, 8),
    'mp_rainncv': (6, 8),
    'mp_snownc': (6, 8),
    'mp_snowncv': (6, 8),
    'mp_sr': (6, 8),
    'mp_th': (4, 6, 8),
    'mp_thompson_frozen_reference_density': (4, 6, 8),
    'mp_thompson_frozen_reference_temperature': (4, 6, 8),
    'mp_thompson_graupel_melt_marker': (4, 6, 8),
    'mp_thompson_graupel_number_shadow': (4, 6, 8),
    'mp_thompson_rain_reference_density': (4, 6, 8),
    'mp_thompson_snow_melt_marker': (4, 6, 8),
    'mp_thompson_snow_velocity_boost': (4, 6, 8),
    'mp_thompson_temperature': (4, 6, 8),
    'mp_z8w': (5, 6, 8),
    'pd_fxc': (4, 6, 9),
    'pd_fxl': (4, 6, 9),
    'pd_fyc': (4, 7, 8),
    'pd_fyl': (4, 7, 8),
    'pd_fzc': (5, 6, 8),
    'pd_fzl': (5, 6, 8),
    'physics_validation_status': (1,),
    'refl_10cm': (4, 6, 8),
    'refl_t': (4, 6, 8),
    'rk_ru': (4, 6, 9),
    'rk_ru_m': (4, 6, 9),
    'rk_rv': (4, 7, 8),
    'rk_rv_m': (4, 7, 8),
    'rk_ww': (5, 6, 8),
    'rk_ww_m': (5, 6, 8),
}
NEST_FIELD_KINDS_MP8 = (
    'u', 'v', 'w', 't', 'ph', 'mu',
    'qv', 'qc', 'qr', 'qi', 'qs', 'qg', 'nr', 'ni',
)
STATE_ARRAY_SHAPES_DIGEST = (
    '9bf527776f97f6e401d5c8084b31a58015f36c388eabba5dc3cc4eaefbaa124c')
SCRATCH_SLOT_REGISTRY_DIGEST = (
    'cfa4fe7ed787889825d504ebb122e0a7042cc8de177ae33367b6cfc8f3ec6d2c')
ORACLE_FIXTURE_COUNT = 92
#: RE-PINNED with the corrected oracle, not with an edit.  The Thompson
#: oracle lane found that five committed fixtures were the output of a
#: libmvec-linked build and regenerated the whole set devectorised
#: (fix/thompson-oracle-devectorized, 576b755b/8290189d), which this branch
#: merges rather than reproduces.  See the inverted witness below.
ORACLE_FIXTURE_AGGREGATE_SHA256 = (
    '6c7f555df2a44206b1cb12013b68c50bd57ac94f4dd0967f1bf0595b7a9eac53')

# -- R5 --------------------------------------------------------------------

PORTED_MP_PHYSICS_PIN = (1, 6, 8, 10, 18)
#: The 20 pre-existing nest-edge field codes.  ``28`` must be APPENDED to
#: ``PORTED_MP_PHYSICS``; inserting it renumbers this table and silently
#: re-points the ratified mp8 -> mp18 nest edge at different fields.
EDGE_FIELD_CODES_PIN = {
    "qv": 0, "qc": 1, "qr": 2, "qi": 3, "qs": 4, "qg": 5,
    "nr": 6, "ni": 7, "ns": 8, "ng": 9,
    "qh": 10, "qndrop": 11, "qnr": 12, "qni": 13, "qns": 14,
    "qng": 15, "qnh": 16, "qnn": 17, "qvolg": 18, "qvolh": 19,
}

# -- R6.  The recorded _apply_thompson call graph.  Arguments are
# identity labels, not values: 'state.qc' or 'scratch[mp_dz8w]'.
# ------------------------------------------------------------------
ADAPTER_CALLS_NO_REFL = (
    ('save_pre_mp_theta', (
        '<_HostAdapterState>',
     ), {}),
    ('launch_classic_graupel_number_init', (
        'state.qg',
        'scratch[mp_thompson_temperature]',
        'state.p',
        'state.qv',
        'scratch[mp_thompson_graupel_number_shadow]',
     ), {}),
    ('launch_frozen_vapor_network_from_owner', (
        'state.qi',
        'state.ni',
        'state.qs',
        'state.qg',
        'state.qr',
        'state.nr',
        'scratch[mp_thompson_temperature]',
        'state.p',
        'state.qv',
        '<classic-table-owner>',
        '10.0',
     ), {
        'graupel_number_shadow':
            'scratch[mp_thompson_graupel_number_shadow]',
        'qc':
            'state.qc',
        'snow_velocity_boost':
            'scratch[mp_thompson_snow_velocity_boost]',
    }),
    ('launch_warm_frozen_source_network_from_owner', (
        'state.qc',
        'state.qr',
        'state.nr',
        'state.qs',
        'state.qg',
        'scratch[mp_thompson_graupel_number_shadow]',
        'scratch[mp_thompson_graupel_melt_marker]',
        'scratch[mp_thompson_snow_melt_marker]',
        'scratch[mp_thompson_temperature]',
        'state.p',
        'state.qv',
        '<classic-table-owner>',
        '10.0',
     ), {}),
    ('launch_hydrometeor_column_mask', (
        'state.qr',
        'scratch[mp_rainncv]',
     ), {}),
    ('launch_hydrometeor_column_mask', (
        'state.qc',
        'scratch[mp_snowncv]',
     ), {}),
    ('launch_graupel_fallout_column_mask', (
        'scratch[mp_thompson_frozen_reference_temperature]',
        'state.qg',
        'scratch[mp_sr]',
     ), {}),
    ('launch_cloud_saturation_adjust', (
        'scratch[mp_thompson_temperature]',
        'state.p',
        'state.qv',
        'state.qc',
     ), {
        'reference_density':
            'scratch[mp_thompson_frozen_reference_density]',
        'reference_temperature':
            'scratch[mp_thompson_frozen_reference_temperature]',
    }),
    ('launch_rain_evaporation', (
        'state.qr',
        'state.nr',
        'scratch[mp_thompson_temperature]',
        'state.p',
        'state.qv',
        '10.0',
     ), {
        'graupel_melt_marker':
            'scratch[mp_thompson_graupel_melt_marker]',
        'reference_density':
            'scratch[mp_thompson_rain_reference_density]',
    }),
    ('launch_cloud_sedimentation', (
        'state.qc',
        'scratch[mp_thompson_temperature]',
        'state.p',
        'state.qv',
        'state.w[view:(3, 1, 1)]',
        'scratch[mp_dz8w]',
        '10.0',
     ), {
        'cloud_active_columns':
            'scratch[mp_snowncv]',
        'rain_active_columns':
            'scratch[mp_rainncv]',
        'reference_density':
            'scratch[mp_thompson_frozen_reference_density]',
    }),
    ('launch_ice_sedimentation', (
        'state.qi',
        'state.ni',
        'scratch[mp_thompson_temperature]',
        'state.p',
        'state.qv',
        'scratch[mp_dz8w]',
        'scratch[mp_rainnc]',
        'scratch[mp_rainncv]',
        'scratch[mp_snownc]',
        'scratch[mp_snowncv]',
        '10.0',
     ), {
        'reference_density':
            'scratch[mp_thompson_frozen_reference_density]',
    }),
    ('launch_snow_sedimentation', (
        'state.qs',
        'scratch[mp_thompson_temperature]',
        'state.p',
        'state.qv',
        'scratch[mp_dz8w]',
        'scratch[mp_rainnc]',
        'scratch[mp_rainncv]',
        'scratch[mp_snownc]',
        'scratch[mp_snowncv]',
        '10.0',
     ), {
        'accumulate_surface':
            'True',
        'melt_rain_nr':
            'state.nr',
        'melt_rain_qr':
            'state.qr',
        'reference_density':
            'scratch[mp_thompson_frozen_reference_density]',
        'reference_temperature':
            'scratch[mp_thompson_frozen_reference_temperature]',
        'snow_melt_marker':
            'scratch[mp_thompson_snow_melt_marker]',
        'velocity_boost':
            'scratch[mp_thompson_snow_velocity_boost]',
    }),
    ('launch_graupel_sedimentation', (
        'state.qg',
        'scratch[mp_thompson_temperature]',
        'state.p',
        'state.qv',
        'scratch[mp_dz8w]',
        'scratch[mp_rainnc]',
        'scratch[mp_rainncv]',
        'scratch[mp_graupelnc]',
        'scratch[mp_graupelncv]',
        '10.0',
     ), {
        'accumulate_surface':
            'True',
        'active_columns':
            'scratch[mp_sr]',
        'graupel_number_shadow':
            'scratch[mp_thompson_graupel_number_shadow]',
        'reference_density':
            'scratch[mp_thompson_frozen_reference_density]',
    }),
    ('launch_rain_sedimentation', (
        'state.qr',
        'state.nr',
        'scratch[mp_thompson_temperature]',
        'state.p',
        'state.qv',
        'scratch[mp_dz8w]',
        'scratch[mp_rainnc]',
        'scratch[mp_rainncv]',
        '10.0',
     ), {
        'accumulate_surface':
            'True',
        'reference_density':
            'scratch[mp_thompson_rain_reference_density]',
    }),
    ('launch_final_phase_cleanup', (
        'state.qc',
        'state.qi',
        'state.ni',
        'scratch[mp_thompson_temperature]',
        'state.p',
        'state.qv',
     ), {}),
    ('launch_classic_graupel_number_finalize', (
        'state.qg',
        'scratch[mp_thompson_temperature]',
        'state.p',
        'state.qv',
        'scratch[mp_thompson_graupel_number_shadow]',
     ), {}),
    ('launch_effective_radius', (
        'scratch[mp_thompson_temperature]',
        'state.p',
        'state.qv',
        'state.qc',
        'state.qi',
        'state.ni',
        'state.qs',
        'state.effc',
        'state.effi',
        'state.effs',
     ), {}),
    ('moist_physics_finish', (
        '<_HostAdapterState>',
        '<SimpleNamespace>',
        'scratch[mp_th]',
        '10.0',
     ), {}),
)

ADAPTER_CALLS_WITH_REFL = (
    ('save_pre_mp_theta', (
        '<_HostAdapterState>',
     ), {}),
    ('launch_classic_graupel_number_init', (
        'state.qg',
        'scratch[mp_thompson_temperature]',
        'state.p',
        'state.qv',
        'scratch[mp_thompson_graupel_number_shadow]',
     ), {}),
    ('launch_frozen_vapor_network_from_owner', (
        'state.qi',
        'state.ni',
        'state.qs',
        'state.qg',
        'state.qr',
        'state.nr',
        'scratch[mp_thompson_temperature]',
        'state.p',
        'state.qv',
        '<classic-table-owner>',
        '10.0',
     ), {
        'graupel_number_shadow':
            'scratch[mp_thompson_graupel_number_shadow]',
        'qc':
            'state.qc',
        'snow_velocity_boost':
            'scratch[mp_thompson_snow_velocity_boost]',
    }),
    ('launch_warm_frozen_source_network_from_owner', (
        'state.qc',
        'state.qr',
        'state.nr',
        'state.qs',
        'state.qg',
        'scratch[mp_thompson_graupel_number_shadow]',
        'scratch[mp_thompson_graupel_melt_marker]',
        'scratch[mp_thompson_snow_melt_marker]',
        'scratch[mp_thompson_temperature]',
        'state.p',
        'state.qv',
        '<classic-table-owner>',
        '10.0',
     ), {}),
    ('launch_hydrometeor_column_mask', (
        'state.qr',
        'scratch[mp_rainncv]',
     ), {}),
    ('launch_hydrometeor_column_mask', (
        'state.qc',
        'scratch[mp_snowncv]',
     ), {}),
    ('launch_graupel_fallout_column_mask', (
        'scratch[mp_thompson_frozen_reference_temperature]',
        'state.qg',
        'scratch[mp_sr]',
     ), {}),
    ('launch_cloud_saturation_adjust', (
        'scratch[mp_thompson_temperature]',
        'state.p',
        'state.qv',
        'state.qc',
     ), {
        'reference_density':
            'scratch[mp_thompson_frozen_reference_density]',
        'reference_temperature':
            'scratch[mp_thompson_frozen_reference_temperature]',
    }),
    ('launch_rain_evaporation', (
        'state.qr',
        'state.nr',
        'scratch[mp_thompson_temperature]',
        'state.p',
        'state.qv',
        '10.0',
     ), {
        'graupel_melt_marker':
            'scratch[mp_thompson_graupel_melt_marker]',
        'reference_density':
            'scratch[mp_thompson_rain_reference_density]',
    }),
    ('launch_cloud_sedimentation', (
        'state.qc',
        'scratch[mp_thompson_temperature]',
        'state.p',
        'state.qv',
        'state.w[view:(3, 1, 1)]',
        'scratch[mp_dz8w]',
        '10.0',
     ), {
        'cloud_active_columns':
            'scratch[mp_snowncv]',
        'rain_active_columns':
            'scratch[mp_rainncv]',
        'reference_density':
            'scratch[mp_thompson_frozen_reference_density]',
    }),
    ('launch_ice_sedimentation', (
        'state.qi',
        'state.ni',
        'scratch[mp_thompson_temperature]',
        'state.p',
        'state.qv',
        'scratch[mp_dz8w]',
        'scratch[mp_rainnc]',
        'scratch[mp_rainncv]',
        'scratch[mp_snownc]',
        'scratch[mp_snowncv]',
        '10.0',
     ), {
        'reference_density':
            'scratch[mp_thompson_frozen_reference_density]',
    }),
    ('launch_snow_sedimentation', (
        'state.qs',
        'scratch[mp_thompson_temperature]',
        'state.p',
        'state.qv',
        'scratch[mp_dz8w]',
        'scratch[mp_rainnc]',
        'scratch[mp_rainncv]',
        'scratch[mp_snownc]',
        'scratch[mp_snowncv]',
        '10.0',
     ), {
        'accumulate_surface':
            'True',
        'melt_rain_nr':
            'state.nr',
        'melt_rain_qr':
            'state.qr',
        'reference_density':
            'scratch[mp_thompson_frozen_reference_density]',
        'reference_temperature':
            'scratch[mp_thompson_frozen_reference_temperature]',
        'snow_melt_marker':
            'scratch[mp_thompson_snow_melt_marker]',
        'velocity_boost':
            'scratch[mp_thompson_snow_velocity_boost]',
    }),
    ('launch_graupel_sedimentation', (
        'state.qg',
        'scratch[mp_thompson_temperature]',
        'state.p',
        'state.qv',
        'scratch[mp_dz8w]',
        'scratch[mp_rainnc]',
        'scratch[mp_rainncv]',
        'scratch[mp_graupelnc]',
        'scratch[mp_graupelncv]',
        '10.0',
     ), {
        'accumulate_surface':
            'True',
        'active_columns':
            'scratch[mp_sr]',
        'graupel_number_shadow':
            'scratch[mp_thompson_graupel_number_shadow]',
        'reference_density':
            'scratch[mp_thompson_frozen_reference_density]',
    }),
    ('launch_rain_sedimentation', (
        'state.qr',
        'state.nr',
        'scratch[mp_thompson_temperature]',
        'state.p',
        'state.qv',
        'scratch[mp_dz8w]',
        'scratch[mp_rainnc]',
        'scratch[mp_rainncv]',
        '10.0',
     ), {
        'accumulate_surface':
            'True',
        'reference_density':
            'scratch[mp_thompson_rain_reference_density]',
    }),
    ('launch_final_phase_cleanup', (
        'state.qc',
        'state.qi',
        'state.ni',
        'scratch[mp_thompson_temperature]',
        'state.p',
        'state.qv',
     ), {}),
    ('launch_classic_graupel_number_finalize', (
        'state.qg',
        'scratch[mp_thompson_temperature]',
        'state.p',
        'state.qv',
        'scratch[mp_thompson_graupel_number_shadow]',
     ), {}),
    ('reflectivity', (
        '<_HostAdapterState>',
        '<SimpleNamespace>',
        'scratch[mp_thompson_temperature]',
        'state.p',
     ), {
        'thompson_graupel_number':
            'scratch[mp_thompson_graupel_number_shadow]',
    }),
    ('launch_effective_radius', (
        'scratch[mp_thompson_temperature]',
        'state.p',
        'state.qv',
        'state.qc',
        'state.qi',
        'state.ni',
        'state.qs',
        'state.effc',
        'state.effi',
        'state.effs',
     ), {}),
    ('moist_physics_finish', (
        '<_HostAdapterState>',
        '<SimpleNamespace>',
        'scratch[mp_th]',
        '10.0',
     ), {}),
)

ACOUSTIC_N_MASS = {
    "mp1": 3, "mp6": 6, "mp8": 6, "mp10": 6, "mp18": 7,
}

# -- F2 --------------------------------------------------------------------

ORACLE_REBUILD_EXCEPTION_FILES = frozenset({
    "warm-column.csv", "ice-column.csv",
    "mixed-column.csv", "mixed-surface.csv",
})
#: Levels (1-based) at which the three original fixtures' ``p_pa`` differs
#: by one float32 ulp from the other 43 committed fixtures.
ORACLE_PPA_DIVERGENT_LEVELS = (2, 3, 5, 6, 7, 8, 10, 14, 16, 17, 18, 19, 22)


# ==========================================================================
# R1 -- source identity
# ==========================================================================

@pytest.fixture(scope="module")
def r1():
    return freeze.receipt_r1_sources()


def test_thompson_cu_is_byte_frozen(r1):
    """The single most important assertion in the port."""
    module = r1["modules"]["thompson"]
    assert module["file_sha256"] == THOMPSON_CU_SHA256, (
        "gpuwm/core/kernels/thompson.cu was edited.  It is byte-frozen: "
        "the entire mp=8 numerics guarantee is that its compiled source "
        "string never moves.  mp=28 kernels belong in new .cu files.")


def test_thompson_compiled_source_string_is_frozen(r1):
    """Identical source string => identical PTX => identical FP results.

    Stronger than the file hash: this is what cupy actually compiles, so a
    change to ``_preamble()``, ``CUDA_DEFINES``, ``common.cuh`` or the
    loader itself fails here too.
    """
    module = r1["modules"]["thompson"]
    assert module["capture_method"] == "loader-capture", (
        "the compile string was reconstructed instead of captured from the "
        "real loader; the inertness claim is then unproven")
    assert module["compiled_source_len"] == THOMPSON_COMPILED_SOURCE_LEN
    assert (module["compiled_source_sha256"]
            == THOMPSON_COMPILED_SOURCE_SHA256)


def test_preamble_and_common_header_are_frozen(r1):
    assert r1["preamble_sha256"] == PREAMBLE_SHA256
    assert r1["preamble_len"] == PREAMBLE_LEN
    assert r1["common_cuh_sha256"] == COMMON_CUH_SHA256
    assert r1["cuda_defines"] == CUDA_DEFINES_PIN


def test_every_frozen_kernel_module_is_unchanged(r1):
    """All 65 pre-existing translation units, file AND compile string.

    New mp=28 ``.cu`` files are allowed and ignored; a MISSING pinned name
    is a failure.
    """
    modules = r1["modules"]
    missing = sorted(set(FROZEN_MODULE_DIGESTS) - set(modules))
    assert not missing, f"frozen kernel modules disappeared: {missing}"
    drift = {}
    for name, (file_sha, compiled_sha) in sorted(
            FROZEN_MODULE_DIGESTS.items()):
        got = modules[name]
        if (got["file_sha256"], got["compiled_source_sha256"]) != (
                file_sha, compiled_sha):
            drift[name] = {
                "expected": (file_sha, compiled_sha),
                "actual": (got["file_sha256"],
                           got["compiled_source_sha256"]),
            }
    assert not drift, f"kernel source drift: {drift}"


def test_loader_hook_is_inert_for_every_frozen_module(r1):
    """``load_module`` still assembles ``_preamble() + <name>.cu`` exactly.

    WP-02 adds an ``_EXTRA_HEADERS`` allow-list to the shared loader.  This
    is the assertion that holds it to "every module not named in the dict
    assembles a byte-identical source string".
    """
    import importlib.util as _ilu

    from gpuwm.core import kernels as _kernels

    _spec = _ilu.spec_from_file_location(
        "kernel_loader_inert_probe",
        Path(__file__).with_name("test_kernel_loader_inert.py"))
    _mod = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    _EXPECTED_HEADERS = _mod._EXPECTED_HEADERS

    # The header grant stays a CLOSED literal: the loader's own allow-list
    # must equal the mapping test_kernel_loader_inert pins, so a module
    # cannot gain a header here without that gate moving too.
    granted = {name: tuple(headers) for name, headers
               in _kernels._EXTRA_HEADERS.items()}
    assert granted == {name: tuple(headers) for name, headers
                       in _EXPECTED_HEADERS.items()}, (
        "the kernel loader's _EXTRA_HEADERS drifted from the closed "
        "mapping tests/test_kernel_loader_inert.py pins")
    not_inert = sorted(
        name for name in FROZEN_MODULE_DIGESTS
        if name not in granted
        and not r1["modules"][name]["loader_matches_preamble_plus_file"])
    assert not not_inert, (
        "the kernel loader no longer assembles _preamble() + <name>.cu for "
        f"these pre-existing modules: {not_inert}")


def test_thompson_py_has_no_new_launcher(r1):
    assert r1["thompson_py_sha256"] == THOMPSON_PY_SHA256
    assert r1["thompson_py_all"] == THOMPSON_PY_ALL
    assert (r1["thompson_py_launch_symbols"]
            == tuple(sorted(THOMPSON_PY_ALL))), (
        "gpuwm/core/thompson.py grew or lost a launch_* symbol; mp=28 "
        "launchers belong in gpuwm/core/thompson_aerosol_*.py")


def test_constant_droplet_number_inventory_is_unchanged(r1):
    """The 13 + 6 + 2 + 3 literal sites mp=28 must replace, not edit.

    If this moves, someone has been editing the frozen kernel in place --
    which the source digests would also catch, but this failure names the
    physics.
    """
    assert r1["thompson_cu_literal_sites"] == THOMPSON_CU_LITERAL_SITES


# ==========================================================================
# R2 -- classic table contract
# ==========================================================================

@pytest.fixture(scope="module")
def r2():
    return freeze.receipt_r2_tables()


def test_classic_table_assets_are_unchanged(r2):
    got = tuple((a["filename"], a["bytes"], a["sha256"])
                for a in r2["classic_table_assets"])
    assert got == CLASSIC_TABLE_ASSETS_PIN
    assert r2["table_set_id"] == TABLE_SET_ID_PIN
    assert r2["mp_physics"] == 8
    assert tuple(r2["number_species"]) == ("ni", "nr")
    assert tuple(r2["mass_species"]) == ("qv", "qc", "qr", "qi", "qs", "qg")
    assert r2["wrf_reference_version"] == WRF_REFERENCE_VERSION_PIN
    assert r2["wrf_reference_commit"] == WRF_REFERENCE_COMMIT_PIN


def test_aerosol_blob_never_enters_the_classic_contract(r2):
    """CCN_ACTIVATE.BIN is third-party parcel-model output only mp=28 reads.

    It ships with gpuwm as of 2026-08-01 and is listed in
    ``tables/MANIFEST.sha256``, which is exactly why this assertion still
    matters: that manifest is a ``sha256sum -c`` file nothing reads at run
    time, while ``CLASSIC_TABLE_ASSETS`` IS the four-asset inventory every
    mp=8 launch validates and every mp=8 restart identity binds.  Adding the
    blob there would change what a validated mp=8 table set means and make
    an mp=8 launch fail closed on a file it never reads.  WP-01 gives it its
    OWN contract and its own set id instead.
    """
    assert r2["aerosol_blob_in_classic_assets"] is False


# ==========================================================================
# R3 -- transported species
# ==========================================================================

def test_morrison_droplet_number_exclusion_survives():
    """``nc`` must not start being advected as a side effect of mp=28.

    mp_physics=10 already allocates ``state.nc`` and deliberately does not
    transport it.  A presence-based ``nc`` in
    ``TRANSPORTED_NUMBER_SPECIES`` would silently start advecting Morrison's
    diagnostic droplet number through all 8 generic dycore call sites.  The
    mp=10 probe below CARRIES ``nc``, so its absence from the result is
    proved rather than merely unexercised.
    """
    r3 = freeze.receipt_r3_species()
    assert tuple(r3["extra_moist_species_mp8"]) == EXTRA_MOIST_SPECIES_MP8
    assert tuple(r3["extra_moist_species_mp10"]) == EXTRA_MOIST_SPECIES_MP10
    assert "nc" not in r3["extra_moist_species_mp10"]
    assert "nwfa" not in r3["extra_moist_species_mp10"]
    assert "nifa" not in r3["extra_moist_species_mp10"]
    assert (tuple(r3["transported_number_species"])
            == TRANSPORTED_NUMBER_SPECIES_PIN)


# ==========================================================================
# R4 -- preflight allocation surface
# ==========================================================================

@pytest.fixture(scope="module")
def r4():
    return freeze.receipt_r4_preflight()


def test_mp8_state_allocation_list_is_unchanged(r4):
    """Every array DomainState allocates for mp=8, with its exact shape."""
    assert r4["probe_config"]["mp_physics"] == 8
    got = r4["state_array_shapes"]
    assert set(got) == set(STATE_ARRAY_SHAPES_MP8), {
        "added": sorted(set(got) - set(STATE_ARRAY_SHAPES_MP8)),
        "removed": sorted(set(STATE_ARRAY_SHAPES_MP8) - set(got)),
    }
    assert got == STATE_ARRAY_SHAPES_MP8
    assert r4["state_array_shapes_digest"] == STATE_ARRAY_SHAPES_DIGEST


def test_mp8_scratch_arena_layout_is_unchanged(r4):
    """The scratch registry IS the aliasing contract.

    ``_apply_thompson`` deliberately lifetime-aliases buffers (the graupel
    entry marker with the held-temperature buffer, RAINNCV/SNOWNCV/SR with
    column masks).  A new or resized slot moves the arena and can turn one
    of those aliases into a live conflict without any test failing on
    values.
    """
    got = r4["scratch_slot_registry"]
    assert set(got) == set(SCRATCH_SLOT_REGISTRY_MP8), {
        "added": sorted(set(got) - set(SCRATCH_SLOT_REGISTRY_MP8)),
        "removed": sorted(set(SCRATCH_SLOT_REGISTRY_MP8) - set(got)),
    }
    assert got == SCRATCH_SLOT_REGISTRY_MP8
    assert r4["scratch_slot_registry_digest"] == SCRATCH_SLOT_REGISTRY_DIGEST


def test_mp8_nest_field_kinds_are_unchanged(r4):
    assert tuple(r4["nest_field_kinds"]) == NEST_FIELD_KINDS_MP8
    assert "nc" not in r4["nest_field_kinds"]


# ==========================================================================
# R5 -- nest transition edge codes
# ==========================================================================

def test_edge_field_codes_for_the_twenty_pre_existing_names():
    """28 must be APPENDED to PORTED_MP_PHYSICS, never inserted.

    ``_EDGE_FIELD_CODES`` is ``enumerate`` over the de-duplicated union of
    every ported scheme's fields in PORTED_MP_PHYSICS order.  Inserting 28
    anywhere but the end renumbers the table, and the ratified mp8 -> mp18
    nest edge then selects different fields with no error anywhere.
    """
    r5 = freeze.receipt_r5_edge_codes()
    codes = r5["edge_field_codes"]
    for name, code in EDGE_FIELD_CODES_PIN.items():
        assert codes.get(name) == code, (
            f"nest edge field {name!r} moved from code {code} to "
            f"{codes.get(name)}")
    assert tuple(r5["ported_mp_physics"])[:5] == PORTED_MP_PHYSICS_PIN
    assert tuple(r5["all_edge_fields"])[:20] == tuple(EDGE_FIELD_CODES_PIN)


# ==========================================================================
# R6 -- acoustic selection and the adapter call graph
# ==========================================================================

def test_acoustic_moist_cq_selects_six_masses_for_mp8():
    """``calc_cq`` mass loading must not learn about aerosol numbers.

    WRF's ``calc_cq`` sums the ``moist`` Registry package; number moments
    live in the separate ``scalar`` package and are not mass loading.  mp=28
    adds nc/nwfa/nifa as scalars, so n_mass for mp=8 (and for mp=28) stays
    6.
    """
    n_mass = freeze.receipt_r6_call_graph()["acoustic_n_mass"]
    assert n_mass["mp8"] == 6
    assert n_mass == ACOUSTIC_N_MASS


def _as_tuple(calls):
    return tuple(
        (c["launcher"], tuple(c["args"]), dict(c["kwargs"])) for c in calls)


def test_apply_thompson_issues_the_identical_launcher_sequence():
    """Call-recording double over the real adapter.

    Every launcher, ``save_pre_mp_theta``, ``moist_physics_finish`` and the
    reflectivity entry point are replaced by spies, and the classic table
    owner by a sentinel, so no CUDA runs and no 380 MB table is read.  What
    is pinned is the ORDER of the calls and the IDENTITY of every argument
    -- which state field or which named scratch slot -- because that
    ordering and that aliasing are the mp=8 trajectory.

    The mp=28 adapter is a SEPARATE module
    (``gpuwm/core/thompson_aerosol.py``); nothing in this sequence may
    change to accommodate it.
    """
    recorded = _as_tuple(freeze.record_adapter_calls(refl_10cm_due=False))
    assert recorded == ADAPTER_CALLS_NO_REFL


def test_apply_thompson_reflectivity_call_graph_is_unchanged():
    recorded = _as_tuple(freeze.record_adapter_calls(refl_10cm_due=True))
    assert recorded == ADAPTER_CALLS_WITH_REFL


def test_adapter_still_feeds_cloud_sedimentation_the_lower_w_slice():
    """WRF copies ``w(i,k,j)`` into ``w1d(k)`` with no averaging.

    mp=28's ``activ_ncloud`` needs the same field and is far more sensitive
    to it, so this is the argument the aerosol adapter must match.  Pinned
    here as its own named claim because a silent switch to a mass-level
    average would still pass the whole-sequence comparison's shape checks.
    """
    calls = {c["launcher"]: c
             for c in freeze.record_adapter_calls(refl_10cm_due=False)}
    assert calls["launch_cloud_sedimentation"]["args"][4] == (
        "state.w[view:(3, 1, 1)]")


# ==========================================================================
# F1 -- the committed mp=8 oracle fixtures
# ==========================================================================

def test_committed_mp8_oracle_fixtures_are_frozen():
    """92 CSVs, byte-for-byte.

    WP-03 adds ``gpuwm/data/thompson/oracle-aero/``; it may not touch this
    directory.  Regenerating any file here moves a model-validated
    baseline.
    """
    f1 = freeze.receipt_f1_oracle_fixtures()
    assert f1["count"] == ORACLE_FIXTURE_COUNT
    assert f1["aggregate_sha256"] == ORACLE_FIXTURE_AGGREGATE_SHA256


# ==========================================================================
# F2 -- the documented four-file clean-rebuild exception
# ==========================================================================

def test_rebuild_exception_list_is_exactly_four_named_files():
    """No file may be added to the exception list to make a gate pass.

    The exception is a recorded historical fact about four fixtures, not a
    tolerance.  Everything else must rebuild byte-for-byte.
    """
    assert set(freeze.ORACLE_REBUILD_EXCEPTIONS) == (
        ORACLE_REBUILD_EXCEPTION_FILES)


def test_the_committed_fixtures_now_share_one_pressure_profile():
    """THE WITNESS, INVERTED, BECAUSE THE DEFECT IT WITNESSED WAS FIXED.

    This test used to be called
    ``test_the_three_original_fixtures_carry_a_foreign_pressure_profile``
    and it asserted the DEFECT: that the 46 committed column fixtures split
    into two ``p_pa`` profiles, 43 against {warm, mixed, ice}.  That was a
    true and carefully measured statement about a stale oracle, and the
    receipt beside it explained the split as a build-provenance difference
    -- "a different libm expf, i.e. a different machine or glibc" -- and
    recorded it rather than repairing it.

    The Thompson oracle lane found the actual mechanism and repaired it.
    It is not a different machine: from GCC 12 on, ``-O2`` implies
    ``-ftree-vectorize``, a vectorised ``exp``/``pow`` loop links glibc's
    libmvec SIMD entry points instead of the scalar routines, libmvec is
    not bit-identical to scalar libm, and whether any given loop vectorises
    is a cost-model decision that depends on how much UNRELATED source
    surrounds it.  ``run_column.F90`` was 227 lines when warm/mixed/ice
    were generated and roughly a thousand when the other 43 were, so the
    same ``-O2`` produced two different oracles.  ``build.sh`` now pins
    ``-fno-tree-vectorize``, which removes libmvec from the link entirely
    and makes the set invariant across gfortran 12/13/14/15 at -O1/-O2/-O3,
    and all five affected fixtures were regenerated
    (fix/thompson-oracle-devectorized, 576b755b and 8290189d).

    So the assertion INVERTS rather than being deleted: ONE profile, 46
    files, no minority group.  That is strictly stronger than what it
    asserted before, and a regression that reintroduces a second profile
    fails here exactly as the old form would have.

    The old docstring's remaining content is preserved because it is still
    the correction of record for the port spec's blocking unknown #2:

    HERMETIC witness for the CAUSE of the (former) exception.

    ``p_pa = p0 * exp(-z(k)/8000.0)`` (run_column.F90:219) is a pure
    function of the harness's own z grid and does not depend on the
    scenario, so all 46 committed column fixtures must print the same 24
    values.  They do not: there are exactly two profiles, and the minority
    one belongs to exactly {warm, mixed, ice} -- which
    ``gpuwm/data/thompson/PROVENANCE.md`` records as the first three columns
    ever generated.  Those three came from a different build of the harness
    (a different libm ``expf``: they differ by one float32 ulp at 13 of 24
    levels), and every other difference in those files -- including the
    ``after`` rows and mixed-surface's rainnc -- follows from it through the
    nonlinear scheme.

    This CORRECTS the port spec's blocking unknown #2, which described the
    drift as a float32-vs-float64 vapour seed with "p_pa, pii and theta
    byte-identical".  p_pa and pii are NOT byte-identical, and the k=1
    vapour coincidence the spec generalised from does not hold at k=2..24.

    Reading only committed repository bytes, this test needs no gfortran,
    no WRF tree and no rebuild -- so the exception stays verified on every
    run instead of resting on a note.
    """
    witness = freeze.fixture_provenance_witness()
    assert witness["distinct_p_pa_profiles"] == 1, (
        "the committed column fixtures split into "
        f"{witness['distinct_p_pa_profiles']} pressure profiles again; "
        "p_pa = p0*exp(-z/8000) does not depend on the scenario, so more "
        "than one profile means the set is a stratigraphy of builds rather "
        "than one oracle run.  The minority group is "
        f"{sorted(witness['minority_group'])}")
    assert witness["group_sizes"] == [46]
    assert witness["minority_group"] == []
    assert witness["p_pa_detail"] == {}, (
        "the witness is reporting a p_pa divergence again; see "
        "tests/test_thompson_oracle_provenance.py for the same property "
        "asserted from the fixture bytes directly")
    # The port spec's blocking unknown #2 claimed the drift was a
    # float32-vs-float64 vapour seed, generalising from a k=1 coincidence.
    # With the oracle corrected, even that coincidence is gone: the
    # committed k=1 qv is the float32-throughout evaluation, not the
    # float64-then-REAL(4) one.  Both halves of the spec's claim are now
    # measured false, and both are asserted so rather than described.
    assert witness["seed_k1_note"]["committed_matches_float64_path"] is False
    assert witness["seed_k1_note"]["generalises_to_other_levels"] is False


@pytest.mark.skipif(
    not os.environ.get("GPUWM_MP8_ORACLE_REBUILD_DIR"),
    reason="set GPUWM_MP8_ORACLE_REBUILD_DIR to a completed "
           "tools/thompson_wrf461_oracle/build.sh output directory")
def test_clean_oracle_rebuild_matches_except_the_four_documented_files():
    """Opt-in empirical gate: 4/4 .dat SHAs and 88/92 CSVs, exactly.

    Opt-in because it needs gfortran, the pristine WRF tree and ~380 MB of
    regenerated coefficient tables.  Measured on gfortran 13.3.0 / glibc
    2.39 / Ubuntu 24.04 during WP-00.
    """
    build_dir = Path(os.environ["GPUWM_MP8_ORACLE_REBUILD_DIR"])
    result = freeze.compare_rebuilt_oracle(build_dir)
    assert result["dat_all_match"] is True, result["dat_assets"]
    assert result["csv_missing"] == []
    assert result["csv_total"] == ORACLE_FIXTURE_COUNT
    assert set(result["csv_differing"]) == ORACLE_REBUILD_EXCEPTION_FILES
    assert result["csv_identical"] == (
        ORACLE_FIXTURE_COUNT - len(ORACLE_REBUILD_EXCEPTION_FILES))
    for name, diff in result["csv_differences"].items():
        pinned = freeze.ORACLE_REBUILD_EXCEPTIONS[name]
        assert diff["max_relative_overall"] <= (
            pinned["max_relative_overall"] * 1.001), (
            f"{name} drifted beyond the recorded deviation")
