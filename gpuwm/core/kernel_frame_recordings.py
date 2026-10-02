"""Per-thread local-frame tables, one per compile platform that read one.

``gpuwm.core.preflight`` prices the launch-time local-memory backing
store from the widest per-thread frame the selected kernel set compiles
to, and that reservation --
``(frame - default stack) x SMs x threads per SM`` -- is the largest
single non-pool term in the device budget.  Until 2026-08-20 the frames
lived in ``preflight`` as one flat dict whose provenance was prose, and
the gate that regenerates it asserted EXACT equality on whatever machine
happened to run it.  Both were wrong in the same way: a frame is what
NVRTC emitted for ONE target architecture at ONE compiler build, not a
property of the ``.cu`` source.

Measured the same NVRTC-plus-driver way on three compile platforms
(``tools/vram_reserve_probe.py frames``), the rows that disagree (the
sm_120 columns as re-read 2026-09-28, the sm_86 column as read
2026-08-20/21 and re-read at the cut):

  ======================  ==========  ==========  ==========
  module                  sm_120      sm_120      sm_86
                          13.0.48     13.3.33     13.0.48
  ======================  ==========  ==========  ==========
  gf                           72          72          88
  kf                          512         512         512
  noah                        176         176         224
  thompson_aerosol_warm         0           0         112
  ysu                           0           0           0
  nssl2_fused_gs              112         216         112
  rrtmgp_cloud                  0          40           0
  shinhong (to 2026-09-30) 13,000      17,160      14,040
  shinhong (workspace)          0           0           0
  noahmp_leaves               272         208         208
  ======================  ==========  ==========  ==========

``gf``, ``ysu`` and ``kf`` are the three rows that went to ZERO-ish on
every platform, and they are the reason this file exists in its present
shape.  All three kernels kept a whole column in the per-thread local
frame; on 2026-08-21 all three moved those arrays into a global
workspace (``gpuwm/core/kernels/gf.cu``, ``ysu.cu``, ``kf.cu``).  ``gf``
went 22,416 -> 88 B, ``ysu`` 9,232 -> 0 B, ``kf`` 24,064 -> 512 B.

``shinhong`` took the same route on 2026-09-30: its column arrays moved
into a global workspace sized to the columns in flight, 17,160 -> 0 B.
Every recording below was RE-READ at the new source that day rather
than carried over: sm_120 on an RTX 5090 through the production loader
under NVRTC 13.0.48, 13.0.88, 13.3.33 and 13.4.92, sm_89 on an RTX 4090
under 13.4.92, and sm_86 compile-only (NVRTC 13.0.48 PTX through ptxas
13.0.48, ``-v``), because the RTX 3080 stays out of GPU work.  All read
0 B, so the recordings keep their completeness claims.

``kf``'s 512 B is not a failed zero: ``tv_env`` and ``positive_energy``
stay on its stack on purpose, because they are the only two of its 54
column arrays whose placement moves an output bit.  512 B is half the
default stack, so the row still reserves nothing.

``ysu`` is the one that mattered to users: ``bl_pbl_physics = 1`` is the
wizard's default, so YSU's frame was the widest a BARE DEFAULT run
LAUNCHED, and MEASURED on weather-node-1 (RTX 5070 Ti, sm_120) it was
holding an 842.0 MiB launch-time reservation on every default run.

The sm_86 rows were briefly DROPPED on the theory that the RTX 3080 was
off limits and could not be re-read.  That turned out to be wrong -- the
card is in the machine and the GPU test suite already runs on it -- so on
2026-08-21 all three were RE-READ there at the post-workspace source
rather than left as holes or back-filled from another platform.  They
agree with sm_120: 88 B, 0 B and 512 B.  This recording is COMPLETE
again.

COMPLETE is a claim about the ``.cu`` files that compile ALONE, and those
are the whole key domain of a ``frames`` mapping: both readers --
``tools/vram_reserve_probe.py`` (``mode_frames``) and
``tests/test_preflight.py::test_the_recorded_local_frames_match_the_driver``
-- enumerate ``gpuwm/core/kernels/*.cu`` and drop whatever NVRTC refuses
standalone.  Three translation units the model actually LAUNCHES are
therefore outside every table here, because they exist only as
compositions: ``rrtmg_lw_legacy_chain``, ``rrtmg_sw_legacy`` and
``p3_composed`` (P3 one-category, ``mp_physics = 50``).  They are priced
from ``preflight.CHAINED_TRANSLATION_UNIT_FRAMES`` instead, and they may
not be moved here: a chained-unit key in a ``frames`` mapping raises at
import, because :func:`frame_ceiling` is checked for EXACT equality against
``preflight.KERNEL_MAX_LOCAL_SIZE_BYTES``
(``gpuwm/core/preflight.py:1663``) and a chained unit is barred from that
table by construction (``:2002``).  So the absence is structural rather
than an oversight -- and it has a price, which is why
:data:`CHAINED_UNITS_WITHOUT_A_PER_PLATFORM_ROW` records it per unit: each
is priced on EVERY platform from ONE reading, and
``under_priced_kernel_frames`` cannot report a drifting one, because its
``observed`` argument comes from that same ``.cu`` enumeration.

The platform-independent claims still live in tests, because a box with no
row here at all is the case those protect:
``tests/test_gf_workspace.py``, ``tests/test_ysu_workspace.py`` and
``tests/test_kf_workspace.py`` each assert the frame stays under the
1,024 B default stack on ANY card.

A NOTE ON WHAT THESE ROWS ARE, because it is easy to over-read them: the
driver takes the reservation at LAUNCH, not at module load.  MEASURED on
node-1 2026-08-21 with a two-kernel module -- compiling a module holding
a 16,384 B kernel reserved 0.0 MiB, and the 1,574.0 MiB appeared only
when that kernel was actually launched.  So a row here is a CEILING over
what a module could cost, and a module whose widest kernel a given
configuration never launches costs less than its row.  ``thompson`` is
the standing example: its 11,264 B is the ``KMAX=256`` template, and a
run with nz <= 64 launches only the ``_64`` variants (2,816 B measured).
Pricing the row is the safe direction and is what preflight does; it is
not what the driver charges.

Rows move with the ARCHITECTURE at a fixed compiler, with the COMPILER
BUILD at a fixed architecture, or with both (``noahmp_leaves``,
``shinhong``).  A row also moves with the SOURCE, and that is what the
2026-09-28 re-read found: every sm_120 recording was read again on an RTX
5090 (weather-node-2) and an RTX 5070 Ti (weather-node-4), the two cards
agreeing to the byte at every build, and three rows had gone stale.
``rrtmgp_rte`` still carried 5,152 B on 13.0.48 and 13.3.33, a reading
of the source before the RRTMGP optimisation (every build now compiles it
to 3,600), and ``gf`` (88 -> 72) and ``shinhong`` (14,040 -> 13,000) had
narrowed on 13.0.48 and 13.0.88 since those rows were read.  Stale-wide
rows are the safe direction, but not free: ``rrtmgp_rte`` was the widest
frame a Morrison, Kessler, P3 or microphysics-free configuration with
RRTMGP radiation launched, so every such run reserved backing store for
frame bytes no compiler emits (32 B per resident thread over Morrison's
5,120, the whole 1,552 B where nothing else is wider than 3,600).  The
sm_86 column predates the same source changes and is re-read at the cut,
so ``gf`` keeps its 88 B ceiling from it until then.  So the identity of
a recording is the pair
``(device_compute_capability, nvrtc_build)`` -- exactly two of the keys
:func:`gpuwm.certify.compile_platform.compile_platform_fingerprint`
already measures -- and neither the card's SM count nor its model name
belongs in it (the SM count is priced separately, off the live device).

``preflight.KERNEL_MAX_LOCAL_SIZE_BYTES`` is the element-wise MAXIMUM
over these tables, checked against them at import.  A rail gate that does
not know which platform it is on must never price BELOW a measurement:
one byte of under-priced frame is one byte times the whole
resident-thread capacity of device memory nobody charged for.  Adding a
fourth platform is adding a row here; nothing else changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping


@dataclass(frozen=True)
class KernelFrameRecording:
    """One compile platform's driver-read local-frame table.

    ``complete`` is False for a recording taken before every ``.cu`` in
    the tree existed, which is a statement about the reading and not
    about the box: the 2026-07/08 desktop readings predate
    ``health_tile.cu``, and a recording may not be back-filled with a
    value nothing measured.
    """

    box: str
    device: str
    compute_capability: str
    nvrtc_build: str
    platform_family: str
    measured: str
    complete: bool
    frames: Mapping[str, int]

    @property
    def platform_key(self) -> tuple[str, str]:
        """What decides the frames: the target arch and the compiler."""
        return (self.compute_capability, self.nvrtc_build)


#: The reading every shipped row came from: the development Windows desktop
#: while the RTX 5090 was in it (the card moved to weather-node-2 on
#: 2026-08-16, so this exact box no longer exists and the table can no
#: longer be regenerated on it).  Rows were taken 2026-07-26 and
#: extended 2026-07-31 / 2026-08-03 / 2026-08-10 as schemes landed; the
#: per-row prose that explains each number is where it has always been,
#: beside the ceiling in gpuwm/core/preflight.py.  ``health_tile.cu``
#: arrived with the out-of-core merge after the last of those readings,
#: which is why this recording is INCOMPLETE rather than carrying a
#: zero nobody measured.
SM120_NVRTC_13_0_48 = KernelFrameRecording(
    box='development-desktop (RTX 5090 era)',
    device='NVIDIA GeForce RTX 5090',
    compute_capability='120',
    nvrtc_build='13.0.48',
    platform_family='windows',
    measured='2026-07-26',
    complete=False,
    frames=MappingProxyType({
        'acoustic': 544,
        'advection': 0,
        'coriolis_map': 0,
        'diagnostics': 0,
        'diff6': 0,
        'diff6_seam': 0,
        'diffusion': 0,
        # Read 2026-10-02 on RTX 5090 and on node-4's RTX 5070 Ti, through
        # the production loader at this NVRTC: both coordinate-diffusion
        # exports 0 B.
        'diff_opt1': 0,
        # New merged leaf modules, read on RTX 5090 at the same
        # compiler. All exported kernels are 0 B, on node-4's RTX 5070 Ti too.
        'bandwidth_glue': 0,
        'noah_init': 0,
        'wrf_cold_start_w': 0,
        'dycore': 0,
        'ftz_probe': 0,
        # RE-READ 2026-09-28 on an RTX 5090 (node-2) and an RTX 5070 Ti
        # (node-4), NVRTC 13.0.48 build id CL-36260728 on both: gf 72,
        # rrtmgp_rte 3,600 and shinhong 13,000 (the rows below), every
        # other row here reproduced to the byte.  The old values (88,
        # 5,152, 14,040) were readings of older source; 5,152 predates
        # the RRTMGP optimisation that took rrtmgp_sw_2stream's three
        # per-thread column arrays off the stack.
        'gf': 72,
        'health': 0,
        'jacobi_eigh': 0,
        'kessler': 5120,
        'kf': 512,
        'kf_validation': 0,
        'lbc_flow': 0,
        'lbc_state': 0,
        'microphysics_validation': 0,
        'milbrandt2': 2048,
        'morrison': 5120,
        'myjpbl': 9232,
        'myjsfc': 0,
        'mynn_pbl': 0,
        'mynn_surface': 0,
        'nest': 0,
        'nest_microphysics': 0,
        'noah': 176,
        'noahmp_bareflux': 0,
        'noahmp_fluxprep': 0,
        'noahmp_leaves': 272,
        'noahmp_radiation': 0,
        'noahmp_sflx': 0,
        'noahmp_snow': 200,
        'noahmp_soilwater': 0,
        'noahmp_vegeflux': 0,
        'noahmp_vegprecip': 0,
        'noahmp_water': 224,
        'nssl2': 15504,
        'nssl2_diagnostics': 0,
        'nssl2_driver_support': 15504,
        'nssl2_fused_gs': 112,
        'nssl2_nucond': 0,
        'nssl2_qvexcess': 0,
        'openbc': 0,
        'pd_advection': 0,
        'refl': 18432,
        'rrtmg_lw': 0,
        'rrtmg_mcica_wrf': 0,
        'rrtmgp_cloud': 0,
        'rrtmgp_gas': 512,
        'rrtmgp_mcica': 0,
        'rrtmgp_rte': 3600,
        'rrtmgp_validation': 0,
        'ruc': 144,
        'sase': 6272,
        'saxpy': 0,
        'sfclay': 0,
        'shinhong': 0,
        'shinhong_validation': 0,
        'smag2d': 0,
        'spec_bdy': 0,
        'thompson': 11264,
        'thompson_aerosol_cold': 0,
        'thompson_aerosol_probe': 0,
        'thompson_aerosol_sat': 0,
        'thompson_aerosol_sed': 9216,
        'thompson_aerosol_state': 40,
        'thompson_aerosol_warm': 0,
        'tke_budget': 0,
        'uh_diag': 0,
        'vert_interp': 768,
        # RE-READ 2026-09-28, RTX 5090 (node-2), NVRTC 13.0.48,
        # after 308c2d39e (WDM6 rain mass and number conservation)
        # took falk and falkn, two per-level arrays, off the stack:
        # 9,776 -> 9,264 B at WDM6_KMAX 64, 11,568 B at 80.
        'wdm6': 9264,
        'wdm6_refl': 16128,
        'wsm6': 7216,
        'ysu': 0,
        'ysu_validation': 0,
    }),
)

#: weather-node-1, RTX 5070 Ti / Linux, 2026-08-20.  Same target
#: architecture as the recording above and a LATER NVRTC, which is what
#: isolates the compiler half: ``nssl2_fused_gs`` 112 -> 216,
#: ``rrtmgp_cloud`` 0 -> 40, ``shinhong`` 13,000 -> 17,160 (14,040 ->
#: 17,160 at the source both were first read at) and ``noahmp_leaves``
#: 272 -> 208 move with the build alone.  RE-READ 2026-09-28 on an RTX
#: 5090 (node-2) and an RTX 5070 Ti (node-4), NVRTC 13.3.33 build id
#: CL-37862127 on both: every row reproduced to the byte except
#: ``rrtmgp_rte``, 5,152 -> 3,600 (see its row).
SM120_NVRTC_13_3_33 = KernelFrameRecording(
    box='weather-node-1',
    device='NVIDIA GeForce RTX 5070 Ti',
    compute_capability='120',
    nvrtc_build='13.3.33',
    platform_family='linux',
    measured='2026-08-20',
    # INCOMPLETE as of 2026-08-31: mynn_scalar_mix.cu and
    # mynn_dmp_sibling.cu (4a0bb3f69, the MYNN-EDMF qn-family mixing
    # wave) postdate this reading and were never read on this compiler.
    # The compile platform is a property of the ENVIRONMENT the process
    # runs in -- the NVRTC library the resolved cuda-toolkit wheel put
    # on the loader path -- not of the box: the same node-1 read NVRTC
    # 13.0.88 from one venv (2026-08-31), 13.3.33 from another (the
    # Noah-MP composed reading of 2026-09-10) and 13.4.59 from a third
    # the same day (see :data:`RESOLVED_TOOLCHAIN_PINS`).  Extending this
    # table therefore means running the census in an environment whose
    # NVRTC is 13.3.33, not waiting for a machine; until someone does,
    # the two rows stay absent and are priced from the ceiling.  A
    # reading may not be back-filled with a value nothing measured.
    complete=False,
    frames=MappingProxyType({
        'acoustic': 544,
        'advection': 0,
        'coriolis_map': 0,
        'diagnostics': 0,
        'diff6': 0,
        'diff6_seam': 0,
        'diffusion': 0,
        # Read 2026-10-02 on RTX 5090 and on node-4's RTX 5070 Ti, through
        # the production loader at this NVRTC: both coordinate-diffusion
        # exports 0 B.
        'diff_opt1': 0,
        # New merged leaf modules, read on RTX 5090 at the same
        # compiler. All exported kernels are 0 B, on node-4's RTX 5070 Ti too.
        'bandwidth_glue': 0,
        'noah_init': 0,
        'wrf_cold_start_w': 0,
        'dycore': 0,
        'ftz_probe': 0,
        'gf': 72,
        'health': 0,
        'health_tile': 0,
        'jacobi_eigh': 0,
        'kessler': 5120,
        'kf': 512,
        'kf_validation': 0,
        'lbc_flow': 0,
        'lbc_state': 0,
        'microphysics_validation': 0,
        'milbrandt2': 2048,
        'morrison': 5120,
        'myjpbl': 9232,
        'myjsfc': 0,
        'mynn_pbl': 0,
        'mynn_surface': 0,
        'nest': 0,
        'nest_microphysics': 0,
        'noah': 176,
        'noahmp_bareflux': 0,
        'noahmp_fluxprep': 0,
        'noahmp_leaves': 208,
        'noahmp_radiation': 0,
        'noahmp_sflx': 0,
        'noahmp_snow': 200,
        'noahmp_soilwater': 0,
        'noahmp_vegeflux': 0,
        'noahmp_vegprecip': 0,
        'noahmp_water': 224,
        'nssl2': 15504,
        'nssl2_diagnostics': 0,
        'nssl2_driver_support': 15504,
        'nssl2_fused_gs': 216,
        'nssl2_nucond': 0,
        'nssl2_qvexcess': 0,
        'openbc': 0,
        'pd_advection': 0,
        'refl': 18432,
        'rrtmg_lw': 0,
        'rrtmg_mcica_wrf': 0,
        'rrtmgp_cloud': 40,
        'rrtmgp_gas': 512,
        'rrtmgp_mcica': 0,
        # RE-READ 2026-09-28 (node-2 and node-4): 5,152 was a reading of
        # the source before the RRTMGP optimisation took three per-thread
        # column arrays off rrtmgp_sw_2stream's stack.
        'rrtmgp_rte': 3600,
        'rrtmgp_validation': 0,
        'ruc': 144,
        'sase': 6272,
        'saxpy': 0,
        'sfclay': 0,
        'shinhong': 0,
        'shinhong_validation': 0,
        'smag2d': 0,
        'spec_bdy': 0,
        'thompson': 11264,
        'thompson_aerosol_cold': 0,
        'thompson_aerosol_probe': 0,
        'thompson_aerosol_sat': 0,
        'thompson_aerosol_sed': 9216,
        'thompson_aerosol_state': 40,
        'thompson_aerosol_warm': 0,
        'tke_budget': 0,
        'uh_diag': 0,
        'vert_interp': 768,
        # RE-READ 2026-09-28, RTX 5070 Ti (node-4), NVRTC 13.3.33,
        # after 308c2d39e (WDM6 rain mass and number conservation)
        # took falk and falkn, two per-level arrays, off the stack:
        # 9,776 -> 9,264 B at WDM6_KMAX 64, 11,568 B at 80.
        'wdm6': 9264,
        'wdm6_refl': 16128,
        'wsm6': 7216,
        'ysu': 0,
        'ysu_validation': 0,
    }),
)

#: The development Windows desktop as it stands now: RTX 3080, sm_86, on the
#: same NVRTC 13.0.48 the first recording used.  Holding the compiler
#: fixed is what isolates the architecture half: ``gf`` 22,416 ->
#: 23,984, ``noah`` 176 -> 224 and ``thompson_aerosol_warm`` 0 -> 112
#: compile WIDER here than the shipped rows, so those three rows were
#: under-pricing this card by 163,774,464 + 5,013,504 + 11,698,176 B
#: (0.17 GiB together, at 68 SMs x 1,536 threads) until the ceiling
#: took them up.
SM86_NVRTC_13_0_48 = KernelFrameRecording(
    box='development-desktop',
    device='NVIDIA GeForce RTX 3080',
    compute_capability='86',
    nvrtc_build='13.0.48',
    platform_family='windows',
    # The bulk of this table was read 2026-08-20; ``gf`` and ``ysu``
    # alone were re-read on the same card 2026-08-21, after the
    # column workspaces.  The field stays the bulk reading's date
    # because that is what the other 70 rows are.
    measured='2026-08-20',
    # COMPLETE again as of 2026-08-21.  Every hole is filled with a real
    # reading rather than a back-filled value: the RTX 3080 turned out to
    # be IN the machine and the GPU test suite already runs on it, so the
    # earlier "off limits, nobody can replace it" note was describing a
    # constraint that no longer holds.  ``gf``, ``ysu`` and ``kf`` were all
    # re-read on that card at the post-workspace source, same NVRTC.
    # Extended 2026-08-31 with the two MYNN-EDMF modules (see their rows),
    # and 2026-09-05 with lbc_time and ntiedtke at this compiler/architecture.
    # Extended 2026-09-11 with milbrandt2_zet, READ ON THIS CARD at this
    # NVRTC (see its row).  The flag went False for one day when that
    # module joined the tree unread here, which is what the docstring
    # says the flag is for; the hole is filled with a reading rather than
    # with the sm_120 value, so the claim is COMPLETE again on the same
    # terms as 2026-08-21 -- every standalone .cu in the tree has a number
    # this box produced.
    #
    # NOT COMPLETE since 2026-09-30: urban_ucm, urban_bep, urban_bep_couple
    # and myjurb (sf_urban_physics 1-3) joined the tree unread on this card,
    # which agents do not use (the desktop RTX 3080 is reserved for the
    # release install smoke).  They are priced from the sm_89 and sm_120
    # readings until someone reads them here; nothing is back-filled.
    # INCOMPLETE again from 2026-09-30: rrtmg_legacy_adapter.cu and
    # rrtmg_legacy_prep.cu (the legacy RRTMG adapter's device glue and
    # wrapper prep) joined the tree unread on this card; both are 0 B on
    # sm_120 at NVRTC 13.4.92.  The hole stays a hole until this
    # card reads it, as the docstring says the flag is for.  The same day
    # rrtmg_mcica_wrf.cu gained rmcw_fill_outputs_column (a copy-only
    # layout twin, 0 B on sm_120 and sm_89 at NVRTC 13.4.92): its 0 B row
    # below predates that kernel.
    # The same day uwpbl.cu (the UW moist-turbulence PBL, bl_pbl_physics = 9)
    # joined the tree unread on this card, which lanes do not use (it is kept
    # for the release install smoke).  It is priced from the sm_89 and sm_120
    # readings until it is read here; nothing is back-filled.
    complete=False,
    frames=MappingProxyType({
        'acoustic': 544,
        'advection': 0,
        'coriolis_map': 0,
        'diagnostics': 0,
        'diff6': 0,
        'diff6_seam': 0,
        'diffusion': 0,
        'dycore': 0,
        'ftz_probe': 0,
        # RE-READ 2026-08-21 on this card at the post-workspace
        # source: 88 B, the same value sm_120 compiles it to.  The
        # 23,984 B this row used to carry described gf.cu BEFORE the
        # column workspace and is gone with that source.
        'gf': 88,
        'health': 0,
        'health_tile': 0,
        'jacobi_eigh': 0,
        'kessler': 5120,
        # RE-MEASURED 2026-08-21 on this very box, not back-filled and
        # not dropped: 24,064 -> 512 B, the same 512 the two sm_120
        # rows read, and the launch-time reservation went 816.0 MiB (the
        # law exactly, at 68 SMs x 1,536) -> 0.0 MiB.  The `gf` row
        # above is absent because gf.cu's workspace landed without an
        # sm_86 reading; kf's did not have to, because the RTX 3080 is
        # in the machine this lane ran from and a frame is a compile
        # attribute -- no device memory, no exclusivity.  The bitwise
        # A/B was re-run here too: 6,569,984 graded words, 0 differ,
        # both controls firing.  That matters more than the frame,
        # because sm_86 is a different architecture and ptxas makes its
        # own FP-contraction choices.
        'kf': 512,
        'kf_validation': 0,
        'lbc_flow': 0,
        'lbc_state': 0,
        # Added from both driver-read entry points on this same platform,
        # 2026-09-05: linear16/rational36 registers; both local_size_bytes0.
        # Exact source/compiler receipts: docs/lbc_time_local_memory.md.
        'lbc_time': 0,
        'microphysics_validation': 0,
        'milbrandt2': 2048,
        # MEASURED 2026-09-11 on the RTX 3080 in this box at this NVRTC
        # (13.0.48, build id CL-36260728), by tools/vram_reserve_probe.py's
        # own mode_frames body bounded to one translation unit: the same
        # load_module, the same extern "C" __global__ symbol scan, the same
        # local_size_bytes attribute.  Bounded rather than a full sweep
        # because a sweep compiles every .cu, and eight rows are enough to
        # show the instrument agrees with the table it is extending --
        # milbrandt2 2048, gf 88, kf 512, ysu 0, noah 224,
        # thompson_aerosol_warm 112, nssl2_fused_gs 112 and
        # nest_microphysics 0 all came back equal to the rows already here,
        # in the same reading.  milbrandt2_zet.cu is the pure Z block lifted
        # out of milbrandt2.cu so the radar observation operator can launch
        # it without the scheme's state update; it holds no column, so
        # unlike its parent it reserves nothing.  The sm_120 reading of the
        # same source is NOT what this row carries and could not be: a
        # frame is what one compiler emitted for one architecture.
        'milbrandt2_zet': 0,
        'morrison': 5120,
        'myjpbl': 9232,
        'myjsfc': 0,
        # MEASURED 2026-08-31 on this card at this NVRTC (13.0.48), the
        # sm_86 campaign that closed the "no measurement platform in this
        # lane" declaration the two MYNN-EDMF modules (4a0bb3f69) landed
        # under: both compile to a 0 B frame, so neither reserves
        # anything on any card.  node-1 (sm_120, NVRTC 13.0.88 -- a
        # platform with no recording) read 0 B for both as well.
        'mynn_dmp_sibling': 0,
        'mynn_pbl': 0,
        'mynn_scalar_mix': 0,
        'mynn_surface': 0,
        'nest': 0,
        'nest_microphysics': 0,
        'noah': 224,
        'noahmp_bareflux': 0,
        'noahmp_fluxprep': 0,
        'noahmp_leaves': 208,
        'noahmp_radiation': 0,
        'noahmp_sflx': 0,
        'noahmp_snow': 200,
        'noahmp_soilwater': 0,
        'noahmp_vegeflux': 0,
        'noahmp_vegprecip': 0,
        'noahmp_water': 224,
        'nssl2': 15504,
        'nssl2_diagnostics': 0,
        'nssl2_driver_support': 15504,
        'nssl2_fused_gs': 112,
        'nssl2_nucond': 0,
        'nssl2_qvexcess': 0,
        # Added 2026-09-05 from all21 driver-read entry points on this
        # platform, closing the pre-existing hole in this complete table.
        # Every entry reports0B local; see docs/lbc_time_local_memory.md.
        'ntiedtke': 0,
        'openbc': 0,
        'pd_advection': 0,
        'refl': 18432,
        'rrtmg_lw': 0,
        'rrtmg_mcica_wrf': 0,
        'rrtmgp_cloud': 0,
        'rrtmgp_gas': 512,
        'rrtmgp_mcica': 0,
        # RE-MEASURED 2026-08-20 off this box's driver after the RRTMGP
        # optimisation landed: 5,152 -> 3,600.  ``rrtmgp_sw_2stream`` is the
        # widest kernel in the module and it lost three per-thread column
        # arrays -- denom[128], dif_dn[129], dif_up[129], 1,544 B together
        # and 1,552 after padding -- which the rewritten two-stream sweeps
        # now compute inline.  Nothing here is priced: the module sits far
        # below this platform's ceiling frames (kf 24,064, gf 23,984), so
        # the reservation is unchanged.  The two sm_120 recordings carried
        # 5,152 for that older source until they were re-read on
        # 2026-09-28 (node-2 and node-4), where every sm_120 build
        # compiles it to this same 3,600.
        'rrtmgp_rte': 3600,
        'rrtmgp_validation': 0,
        'ruc': 144,
        'sase': 6272,
        'saxpy': 0,
        'sfclay': 0,
        'shinhong': 0,
        'shinhong_validation': 0,
        'smag2d': 0,
        'spec_bdy': 0,
        'thompson': 11264,
        'thompson_aerosol_cold': 0,
        'thompson_aerosol_probe': 0,
        'thompson_aerosol_sat': 0,
        'thompson_aerosol_sed': 9216,
        'thompson_aerosol_state': 40,
        'thompson_aerosol_warm': 112,
        'tke_budget': 0,
        'uh_diag': 0,
        'vert_interp': 768,
        # The source this read no longer exists: 308c2d39e took the
        # frame to 9,264 B on every sm_120 build read 2026-09-28
        # (NVRTC 12.9.86, 13.0.48, 13.0.88, 13.3.33, 13.4.92).  This
        # card was read again at the 2.8.0 cut's Windows step
        # (2026-09-29): 9,264 B at WDM6_KMAX 64 on sm_86 as well.
        'wdm6': 9264,
        'wdm6_refl': 16128,
        'wsm6': 7216,
        # RE-READ 2026-08-21 on this card at the post-workspace
        # source: 0 B, as on both sm_120 builds.  The 7,184 B this row
        # used to carry described ysu.cu BEFORE the column workspace.
        # tests/test_ysu_workspace.py holds the platform-independent
        # claim on any box that has no row here at all.
        'ysu': 0,
        'ysu_validation': 0,
    }),
)

#: The same RTX 3080, read 2026-09-11 through the SHIPPED desktop
#: runtime instead of a CUDA-13 checkout environment: the interpreter
#: ArWen 2.7.2 installs carries cupy-cuda12x 14.2.0 and
#: nvidia-cuda-nvrtc-cu12 12.9.86, which is what
#: ``cupy-cuda12x[ctk]>=14.0`` resolves to (see
#: :data:`RESOLVED_TOOLCHAIN_PINS`), so sm_86 / NVRTC 12.9.86 is the
#: compile platform of every desktop install of this release.  CUDA
#: driver version read beside it: 13030.
#:
#: PARTIAL ON PURPOSE, and the reason is the reading's bound rather than
#: a hole: it is the CALIBRATION of the Noah-MP composed row for this
#: platform below, so it read the eight standalone stems whose values
#: this architecture's 13.0.48 recording already holds and which are the
#: ones known to move -- the three post-workspace frames (``gf``, ``kf``,
#: ``ysu``), the two rows that move with the ARCHITECTURE at a fixed
#: compiler (``noah`` 224, ``thompson_aerosol_warm`` 112), the one that
#: moves with the COMPILER BUILD at a fixed architecture
#: (``nssl2_fused_gs``), and ``milbrandt2`` / ``nest_microphysics`` as a
#: wide and a zero control.  Same instrument as the rest of the census
#: (``gpuwm.core.kernels.load_module``, then ``local_size_bytes`` over
#: every exported ``__global__``, widest per module), fresh process,
#: empty CuPy cache, zero launches.
#:
#: WHAT IT MEASURED: all eight reproduce their NVRTC 13.0.48 value on
#: this card to the byte, so the 13.0.48 -> 12.9.86 step moved none of
#: them on sm_86.  That is a statement about these eight stems and these
#: two builds; it licenses nothing about the stems that were not read,
#: which is why ``complete`` is False and no other key is written here.
#: The ceiling does not move: every value is at or below the element-wise
#: maximum the other recordings already carry.
SM86_NVRTC_12_9_86 = KernelFrameRecording(
    box='development-desktop',
    device='NVIDIA GeForce RTX 3080',
    compute_capability='86',
    nvrtc_build='12.9.86',
    platform_family='windows',
    measured='2026-09-11',
    complete=False,
    frames=MappingProxyType({
        'gf': 88,
        'kf': 512,
        'milbrandt2': 2048,
        'nest_microphysics': 0,
        'noah': 224,
        'nssl2_fused_gs': 112,
        'thompson_aerosol_warm': 112,
        'ysu': 0,
    }),
)


#: The compiler every fresh ``pip install gpuwm[gpu-cu13]`` has installed
#: since 2026-09-16, when cuda-toolkit 13.4.2 became the ``[ctk]``
#: resolution and pinned nvidia-cuda-nvrtc 13.4.92 (see
#: :data:`RESOLVED_TOOLCHAIN_PINS`).  Read 2026-09-28 with
#: ``tools/vram_reserve_probe.py frames`` in a venv installed from
#: ``cupy-cuda13x[ctk]>=14.0`` (CuPy 14.2.0, NVRTC build id CL-38855100),
#: fresh CuPy cache, on an RTX 5090 (weather-node-2, 170 SMs) and an RTX
#: 5070 Ti (weather-node-4, 70 SMs): the two cards agree to the byte on
#: every module.  Every row equals the 13.3.33 recording's, so the
#: 13.3 -> 13.4 step moved no standalone frame on this architecture, and
#: the five modules that recording never read (``lbc_time``,
#: ``milbrandt2_zet``, ``mynn_dmp_sibling``, ``mynn_scalar_mix``,
#: ``ntiedtke``) read 0 B.  No row is above the ceiling the other
#: recordings already carry, so adding it moves no shipped frame; what it
#: adds is the exact-equality leg of the driver gate on the compiler a
#: fresh CUDA-13 install actually runs.  Re-read whole on 2026-10-01 at the
#: GPU forcing-preparation lane's merge (843d2043e), fresh CuPy cache: every
#: row below equals that reading, which covers every standalone unit
#: (horizontal, real_init, real_init_math, thompson_cold_start and
#: portable_libm64_grade included).  Re-read 2026-10-02 on RTX 5090
#: through the production loader for the four 2.8.2 keys below
#: (bandwidth_glue, diff_opt1, noah_init, wrf_cold_start_w), and on
#: node-4's RTX 5070 Ti the same day at NVRTC 13.0.48, 13.0.88, 13.3.33,
#: 13.4.92 and 12.9.86: 0 B each on both cards.
SM120_NVRTC_13_4_92 = KernelFrameRecording(
    box='weather-node-2',
    device='NVIDIA GeForce RTX 5090',
    compute_capability='120',
    nvrtc_build='13.4.92',
    platform_family='linux',
    measured='2026-10-02',
    complete=True,
    frames=MappingProxyType({
        # horizontal.cu: fused RH adds 0 B; unit maximum stays 16 B, NVRTC 13.4.92.
        'horizontal': 16,
        # thompson_cold_start.cu (the card closure) and the test-only
        # portable_libm64_grade unit, read 2026-10-01 on this card at this
        # build with the rest of this recording: 0 B each.
        'thompson_cold_start': 0,
        'portable_libm64_grade': 0,
        # real_init.cu and real_init_math.cu (the card route of
        # initialize_real), every entry point read 2026-09-30 on this card
        # at NVRTC 13.4.92: 0 B.
        # REAL units re-read after IEEE neighbor repair, 2026-10-01.
        'real_init': 0,
        'real_init_math': 0,
        'acoustic': 544,
        'advection': 0,
        'coriolis_map': 0,
        'diagnostics': 0,
        'diff6': 0,
        'diff6_seam': 0,
        'diffusion': 0,
        # Read 2026-10-02 on RTX 5090 and on node-4's RTX 5070 Ti, through
        # the production loader at this NVRTC: both coordinate-diffusion
        # exports 0 B.
        'diff_opt1': 0,
        # New merged leaf modules, read on RTX 5090 at the same
        # compiler. All exported kernels are 0 B, on node-4's RTX 5070 Ti too.
        'bandwidth_glue': 0,
        'noah_init': 0,
        'wrf_cold_start_w': 0,
        'dycore': 0,
        # The dycore-host speed lane's four point-local units (66af318bc):
        # read 2026-09-30 on this box at this NVRTC (RTX 5090, fresh CuPy
        # cache, tools/vram_reserve_probe.py frames), every kernel 0 B, and
        # 0 B again on an RTX 4090 (sm_89) at NVRTC 13.4.92.  The same
        # reading reproduced every other row of this recording.
        'face_mass': 0,
        'held_heating': 0,
        'rk_bookkeeping': 0,
        'surface_w': 0,
        # zadvect_implicit's unit (A158): read 2026-10-01 on this box at this
        # NVRTC through the production loader at its shipped IEVA_KMAX = 65,
        # the five column solves 1,040 B each (two double columns of 65),
        # the split and column-mass kernels 0 B; 2,064 B at 129 and 4,112 B
        # at 257 (preflight.IEVA_TIER_FRAME).
        'ieva': 1040,
        'ftz_probe': 0,
        # Re-read 2026-09-30 on this box at this NVRTC after the
        # default-pieces speed lane's GF change (b11fc66ab: early exit for
        # trigger-rejected columns, eight resident blocks per SM): every
        # gf.cu kernel reads 0 B, where 72 B was read before.  Its GF A/B
        # dumps and 1 h real HRRR GF forecast are byte-identical to
        # d6929cb8d.
        'gf': 0,
        'health': 0,
        'health_tile': 0,
        'jacobi_eigh': 0,
        'kessler': 5120,
        # Re-read 2026-09-30 on this box at this NVRTC after the
        # default-pieces speed lane's KF change (08a1eed93: eight-warp
        # blocks under __launch_bounds__(256), trigger-predicted column
        # order): every kf.cu kernel reads 0 B, where 512 B was read before
        # (and 0 B again on an RTX 4090, sm_89, at NVRTC 13.4.92).  The
        # lane's KF-every-step A/B and its 1 h real HRRR forecasts are
        # byte-identical to d6929cb8d on both cards, so the placement change
        # moved no output bit.
        'kf': 0,
        'kf_validation': 0,
        'lbc_flow': 0,
        'lbc_state': 0,
        'lbc_time': 0,
        'microphysics_validation': 0,
        'milbrandt2': 2048,
        'milbrandt2_zet': 0,
        'morrison': 5120,
        'myjpbl': 9232,
        'myjsfc': 0,
        'mynn_dmp_sibling': 0,
        'mynn_pbl': 0,
        'mynn_scalar_mix': 0,
        'mynn_surface': 0,
        'nest': 0,
        'nest_microphysics': 0,
        'noah': 176,
        'noahmp_bareflux': 0,
        'noahmp_fluxprep': 0,
        'noahmp_leaves': 208,
        'noahmp_radiation': 0,
        'noahmp_sflx': 0,
        'noahmp_snow': 200,
        'noahmp_soilwater': 0,
        'noahmp_vegeflux': 0,
        'noahmp_vegprecip': 0,
        'noahmp_water': 224,
        'nssl2': 15504,
        'nssl2_diagnostics': 0,
        'nssl2_driver_support': 15504,
        'nssl2_fused_gs': 216,
        'nssl2_nucond': 0,
        'nssl2_qvexcess': 0,
        'ntiedtke': 0,
        'openbc': 0,
        'pd_advection': 0,
        'refl': 18432,
        # Read 2026-09-30 on this box at this NVRTC (the device-resident
        # legacy RRTMG adapter glue, tools/vram_reserve_probe.py frames reader):
        # every kernel 0 B.  The same for its device wrapper prep.  Re-read
        # the same day on an RTX PRO 4500 (sm_120, NVRTC 13.4) after the
        # result-grid, SWDOWN and radius kernels joined the unit: all twelve
        # 0 B, the prep unit's four 0 B.
        'rrtmg_legacy_adapter': 0,
        # Only this unit re-read 2026-10-02 at e035bb754 on this RTX 5090:
        # the production device-prep loader (--ftz=false), NVRTC 13.4.92,
        # reads all four kernels as 0 B after the LW spacing-guard fix.
        'rrtmg_legacy_prep': 0,
        'rrtmg_lw': 0,
        # Re-read 2026-09-30 with rmcw_fill_outputs_column added (the
        # longwave McICA slabs in the batched engine's layout): 0 B.
        'rrtmg_mcica_wrf': 0,
        'rrtmgp_cloud': 40,
        'rrtmgp_gas': 512,
        'rrtmgp_mcica': 0,
        'rrtmgp_rte': 3600,
        'rrtmgp_validation': 0,
        'ruc': 144,
        'sase': 6272,
        'saxpy': 0,
        'sfclay': 0,
        'shinhong': 0,
        'shinhong_validation': 0,
        'smag2d': 0,
        'spec_bdy': 0,
        'thompson': 11264,
        'thompson_aerosol_cold': 0,
        'thompson_aerosol_probe': 0,
        'thompson_aerosol_sat': 0,
        'thompson_aerosol_sed': 9216,
        'thompson_aerosol_state': 40,
        'thompson_aerosol_warm': 0,
        'tke_budget': 0,
        # Read 2026-09-30 on this box at this NVRTC, fresh CuPy cache (WRF
        # v4.7.1 slope_rad / topo_shading, lane 281-namelist-gaps): the
        # shadow scan and the surface adjustment 40 B each, the other four
        # entry points 0 B.
        'topo_radiation': 40,
        'uh_diag': 0,
        # uwpbl.cu joined after this reading.  Read on this card and
        # this NVRTC build 2026-09-30 (the loader's get_function().
        # attributes route of tools/vram_reserve_probe.py): uwpbl_columns
        # 864 B, 255 registers.  The same source is 3,184 B on sm_89 at
        # the same build (SM89_NVRTC_13_4_92).
        'uwpbl': 864,
        # Re-read 2026-10-01 at the lane's merge with integrate/2.8 (843d2043e)
        # on this card at this build, fresh CuPy cache: the WRF-real vertical
        # step that gives the Rust bridge's bits (9cd3a755f) is 512 B, where
        # the 2026-09-28 reading of the older source was 768 B.
        'vert_interp': 512,
        'wdm6': 9264,
        'wdm6_refl': 16128,
        'wsm6': 7216,
        'ysu': 0,
        'ysu_validation': 0,
        # The urban canopy models joined after this reading.  Read on this
        # card and this NVRTC build 2026-09-30 with tools/
        # vram_reserve_probe.py frames, at the source that reads the
        # green-roof constants from __constant__ memory (urban_ucm.cu):
        # urban_ucm 0 B (the same source is 72 B on sm_89, see
        # SM89_NVRTC_13_4_92), urban_bep and urban_bep_couple 0 B, and
        # myjurb 10,256 B -- MYJ's 9,232 B myjpbl frame plus MYJURB's
        # urban arrays, a (10,256 - 1,024) B x 170 SMs x 1,536 threads =
        # 2.2 GiB launch-time reservation.  BEP+BEM's composed unit is
        # priced in preflight.CHAINED_TRANSLATION_UNIT_FRAMES.
        'myjurb': 10256,
        'urban_bep': 0,
        'urban_bep_couple': 0,
        'urban_ucm': 0,
    }),
)

#: A167: sm_120 at NVRTC 12.9.86, the compiler of the default gpuwm[gpu]
#: extra (cupy-cuda12x[ctk], RESOLVED_TOOLCHAIN_PINS) and of the shipped
#: desktop runtime.  Until this reading the standalone tables had no
#: sm_120 row for it (only Noah-MP's composed units below), and the urban
#: canopy units, read 2026-09-30 under 13.4.92 only, compile WIDER here:
#: myjurb 12,304 B against 10,256 B under 13.4.92, a (2,048 B) x 70 SMs x
#: 1,536 threads = 0.205 GiB launch-time reservation the fit gate did not
#: charge on an RTX 5070 Ti (0.50 GiB on an RTX 5090's 170 SMs);
#: urban_ucm 552 B against 72 B and urban_bep 16 B against 0 B, both under
#: the 1,024 B default stack.  Those three rows raise the ceiling; every
#: other row is at or below it (gf 88, uwpbl 2,032, noahmp_leaves 272 read
#: as other recordings already carry them).  Read 2026-10-01 with
#: tools/vram_reserve_probe.py frames through the production loader, fresh
#: CuPy and driver caches, two processes on each of node-4's RTX 5070 Ti
#: (driver 13.2) and node-2's RTX 5090 (driver 13.3), CuPy 14.2.0 with
#: nvidia-cuda-nvrtc-cu12 12.9.86: the four readings agree to the byte.
#: COMPLETE: every ``.cu`` that compiles alone, the 15 that do not being
#: exactly preflight.UNMEASURED_KERNEL_MODULES.
#: Re-read 2026-10-02 on RTX 5090 through the production loader:
#: all 99 standalone rows, including the four new keys below.
#: Earlier cross-card readings above describe the preceding
#: 95-row source set. The new rows read 0 B on node-4's RTX 5070 Ti too
#: (2026-10-02, this build).
SM120_NVRTC_12_9_86 = KernelFrameRecording(
    box='weather-node-2',
    device='NVIDIA GeForce RTX 5090',
    compute_capability='120',
    nvrtc_build='12.9.86',
    platform_family='linux',
    measured='2026-10-02',
    complete=True,
    frames=MappingProxyType({
        'acoustic': 544,
        'advection': 0,
        'coriolis_map': 0,
        'diagnostics': 0,
        'diff6': 0,
        'diff6_seam': 0,
        'diffusion': 0,
        # Read 2026-10-02 on RTX 5090 and on node-4's RTX 5070 Ti, through
        # the production loader at this NVRTC: both coordinate-diffusion
        # exports 0 B.
        'diff_opt1': 0,
        # New merged leaf modules, read on RTX 5090 at the same
        # compiler. All exported kernels are 0 B, on node-4's RTX 5070 Ti too.
        'bandwidth_glue': 0,
        'noah_init': 0,
        'wrf_cold_start_w': 0,
        'dycore': 0,
        'face_mass': 0,
        'ftz_probe': 0,
        'gf': 88,
        'health': 0,
        'health_tile': 0,
        'held_heating': 0,
        # The card preparation's units (lane 281-gpu-forcing-prep), read
        # 2026-10-01 on node-2's RTX 5090 at this build with the rest of the
        # tree, fresh CuPy cache: wider here than under NVRTC 13.4 for the
        # float64 libm twins (real_init_math, thompson_cold_start and the
        # test-only grading unit 48 B, against 0 B at 13.4.92).
        'horizontal': 16,
        'portable_libm64_grade': 48,
        # REAL units re-read after IEEE neighbor repair, 2026-10-01.
        'real_init': 0,
        'real_init_math': 48,
        'thompson_cold_start': 48,
        'ieva': 1040,
        'jacobi_eigh': 0,
        'kessler': 5120,
        'kf': 0,
        'kf_validation': 0,
        'lbc_flow': 0,
        'lbc_state': 0,
        'lbc_time': 0,
        'microphysics_validation': 0,
        'milbrandt2': 2048,
        'milbrandt2_zet': 0,
        'morrison': 5120,
        'myjpbl': 9232,
        'myjsfc': 0,
        'myjurb': 12304,
        'mynn_dmp_sibling': 0,
        'mynn_pbl': 0,
        'mynn_scalar_mix': 0,
        'mynn_surface': 0,
        'nest': 0,
        'nest_microphysics': 0,
        'noah': 176,
        'noahmp_bareflux': 0,
        'noahmp_fluxprep': 0,
        'noahmp_leaves': 272,
        'noahmp_radiation': 0,
        'noahmp_sflx': 0,
        'noahmp_snow': 200,
        'noahmp_soilwater': 0,
        'noahmp_vegeflux': 0,
        'noahmp_vegprecip': 0,
        'noahmp_water': 224,
        'nssl2': 15504,
        'nssl2_diagnostics': 0,
        'nssl2_driver_support': 15504,
        'nssl2_fused_gs': 112,
        'nssl2_nucond': 0,
        'nssl2_qvexcess': 0,
        'ntiedtke': 0,
        'openbc': 0,
        'pd_advection': 0,
        'refl': 18432,
        'rk_bookkeeping': 0,
        'rrtmg_legacy_adapter': 0,
        'rrtmg_legacy_prep': 0,
        'rrtmg_lw': 0,
        'rrtmg_mcica_wrf': 0,
        'rrtmgp_cloud': 0,
        'rrtmgp_gas': 512,
        'rrtmgp_mcica': 0,
        'rrtmgp_rte': 3600,
        'rrtmgp_validation': 0,
        'ruc': 144,
        'sase': 6272,
        'saxpy': 0,
        'sfclay': 0,
        'shinhong': 0,
        'shinhong_validation': 0,
        'smag2d': 0,
        'spec_bdy': 0,
        'surface_w': 0,
        'thompson': 11264,
        'thompson_aerosol_cold': 0,
        'thompson_aerosol_probe': 0,
        'thompson_aerosol_sat': 0,
        'thompson_aerosol_sed': 9216,
        'thompson_aerosol_state': 40,
        'thompson_aerosol_warm': 0,
        'tke_budget': 0,
        'topo_radiation': 40,
        'uh_diag': 0,
        'urban_bep': 16,
        'urban_bep_couple': 0,
        'urban_ucm': 552,
        'uwpbl': 2032,
        # Re-read 2026-10-01 on node-2's RTX 5090 at this build (fresh CuPy
        # cache, the GPU forcing-preparation lane's merge 843d2043e): the
        # WRF-real vertical step that gives the Rust bridge's bits is 512 B.
        'vert_interp': 512,
        'wdm6': 9264,
        'wdm6_refl': 16128,
        'wsm6': 7216,
        'ysu': 0,
        'ysu_validation': 0,
    }),
)


#: node-1's RTX 4090 (sm_89) at NVRTC 13.4.92, the cupy-cuda13x[ctk] build
#: of a fresh gpu-cu13 install, read 2026-09-30 with tools/
#: vram_reserve_probe.py frames: every standalone ``.cu`` in the tree (the
#: 86 it compiles; the 13 it cannot are the composed fragments of
#: preflight.UNMEASURED_KERNEL_MODULES).  Taken for the urban canopy
#: models, whose urban_ucm frame differs by architecture (72 B here, 0 B on
#: sm_120) -- so the ceiling needed this architecture -- and recorded whole
#: rather than as four rows: every other module agrees with the shipped
#: ceiling, so it raises no other row.  NVRTC 12.9.86 (cupy-cuda12x) on the
#: same card read the same 86 values.  Re-read whole at the forward merge
#: onto integrate/2.8: 94 standalone modules, 16 composed fragments.
#: NOT COMPLETE since the namelist-gaps merge (2026-10-01): topo_radiation.cu
#: (WRF's slope_rad / topo_shading) joined the tree unread on this card.  It
#: is priced from the other recordings' readings (40 B on sm_120) until this
#: card reads it; nothing is back-filled.  ieva.cu (A158, zadvect_implicit)
#: is unread here too and is priced from its sm_120 reading, 1,040 B.
#: Both read on this card on 2026-10-01 (see the first rows below), and the
#: recording is complete again.
SM89_NVRTC_13_4_92 = KernelFrameRecording(
    box='weather-node-1',
    device='NVIDIA GeForce RTX 4090',
    compute_capability='89',
    nvrtc_build='13.4.92',
    platform_family='linux',
    measured='2026-09-30',
    complete=True,
    frames=MappingProxyType({
        # Read whole 2026-10-01 on this card at this build, fresh CuPy cache,
        # at the GPU forcing-preparation lane's merge with integrate/2.8
        # 57066783c (843d2043e): every row below equals that reading, and the
        # four units unread here until then read ieva 1,040 B (as its sm_120
        # pricing assumed), topo_radiation 40 B, thompson_cold_start 48 B and
        # the test-only portable_libm64_grade 48 B, so the recording is
        # complete again.
        'ieva': 1040,
        'topo_radiation': 40,
        'thompson_cold_start': 48,
        'portable_libm64_grade': 48,
        # The four 2.8.2 standalone units, read 2026-10-02 on this card at
        # this build through the production loader
        # (tools/vram_reserve_probe.py frames, fresh CuPy cache, at
        # integrate/2.8 ff39ff708): 0 B each, so the recording stays complete.
        'bandwidth_glue': 0,
        'diff_opt1': 0,
        'noah_init': 0,
        'wrf_cold_start_w': 0,
        # Re-read 2026-09-30 on this card at the urban tip forward-merged onto
        # integrate/2.8 (tools/vram_reserve_probe.py frames): every row
        # above and below read the same value again, and the eight
        # modules the 2026-09-30 speed lanes added (face_mass,
        # held_heating, mynn_seaice_glue, phy_glue, rk_bookkeeping,
        # rrtmg_legacy_adapter, rrtmg_legacy_prep, surface_w) read 0 B,
        # so the recording stays complete.  Re-read again at the merge of
        # integrate/2.8 10977552d: kf 512 -> 0 B (the default-pieces lane's
        # KF tiling) and shinhong 17,160 -> 0 B (its column arrays moved to a
        # global workspace, 878435c39); every other row read the same.
        # mynn_seaice_glue and phy_glue left the tree with the megakernel
        # revert (A147), and their rows with them.
        'acoustic': 544,
        'advection': 0,
        'coriolis_map': 0,
        'diagnostics': 0,
        'diff6': 0,
        'diff6_seam': 0,
        'diffusion': 0,
        'dycore': 0,
        'face_mass': 0,
        'ftz_probe': 0,
        'gf': 88,
        'health': 0,
        'health_tile': 0,
        'held_heating': 0,
        # horizontal.cu: fused RH adds 0 B; unit maximum stays 16 B, NVRTC 13.4.92.
        'horizontal': 16,
        'jacobi_eigh': 0,
        'kessler': 5120,
        'kf': 0,
        'kf_validation': 0,
        'lbc_flow': 0,
        'lbc_state': 0,
        'lbc_time': 0,
        'microphysics_validation': 0,
        'milbrandt2': 2048,
        'milbrandt2_zet': 0,
        'morrison': 5120,
        'myjpbl': 9232,
        'myjsfc': 0,
        'myjurb': 10256,
        'mynn_dmp_sibling': 0,
        'mynn_pbl': 0,
        'mynn_scalar_mix': 0,
        'mynn_surface': 0,
        'nest': 0,
        'nest_microphysics': 0,
        'noah': 224,
        'noahmp_bareflux': 0,
        'noahmp_fluxprep': 0,
        'noahmp_leaves': 208,
        'noahmp_radiation': 0,
        'noahmp_sflx': 0,
        'noahmp_snow': 200,
        'noahmp_soilwater': 0,
        'noahmp_vegeflux': 0,
        'noahmp_vegprecip': 0,
        'noahmp_water': 224,
        'nssl2': 15504,
        'nssl2_diagnostics': 0,
        'nssl2_driver_support': 15504,
        'nssl2_fused_gs': 112,
        'nssl2_nucond': 0,
        'nssl2_qvexcess': 0,
        'ntiedtke': 0,
        'openbc': 0,
        'pd_advection': 0,
        # real_init.cu and real_init_math.cu, read 2026-09-30 on this card at
        # this NVRTC: real_init 0 B; real_init_math 48 B (real_thermo).
        # REAL units re-read after IEEE neighbor repair, 2026-10-01.
        'real_init': 0,
        'real_init_math': 48,
        'refl': 18432,
        'rk_bookkeeping': 0,
        'rrtmg_legacy_adapter': 0,
        # Only this unit re-read 2026-10-02 after the LW spacing-guard fix:
        # RTX 4090, NVRTC 13.4.92, production device-prep loader,
        # all four kernels 0 B; the existing ceiling is unchanged.
        'rrtmg_legacy_prep': 0,
        'rrtmg_lw': 0,
        'rrtmg_mcica_wrf': 0,
        'rrtmgp_cloud': 40,
        'rrtmgp_gas': 512,
        'rrtmgp_mcica': 0,
        'rrtmgp_rte': 3600,
        'rrtmgp_validation': 0,
        'ruc': 144,
        'sase': 6272,
        'saxpy': 0,
        'sfclay': 0,
        'shinhong': 0,
        'shinhong_validation': 0,
        'smag2d': 0,
        'spec_bdy': 0,
        'surface_w': 0,
        'thompson': 11264,
        'thompson_aerosol_cold': 0,
        'thompson_aerosol_probe': 0,
        'thompson_aerosol_sat': 0,
        'thompson_aerosol_sed': 9216,
        'thompson_aerosol_state': 40,
        'thompson_aerosol_warm': 112,
        'tke_budget': 0,
        'uh_diag': 0,
        'urban_bep': 0,
        'urban_bep_couple': 0,
        'urban_ucm': 72,
        # The UW moist-turbulence PBL (integrate/2.8's own sm_89 reading of
        # it, merged into this recording): uwpbl_columns compiles to
        # 3,184 B here against 864 B on sm_120 at the same build.
        'uwpbl': 3184,
        # Re-read 2026-10-01 at the GPU forcing-preparation lane's merge
        # (843d2043e): the WRF-real vertical step that gives the Rust
        # bridge's bits is 512 B, where the older source read 768 B.
        'vert_interp': 512,
        'wdm6': 9264,
        'wdm6_refl': 16128,
        'wsm6': 7216,
        'ysu': 0,
        'ysu_validation': 0,
    }),
)


# gp-vert measured only this module on each card at the loaded compiler.
# The wider tiers read 1280 B and 2048 B; the table prices the default unit.
SM120_VERT_NVRTC_13_4_59 = KernelFrameRecording(
    box='node-2', device='NVIDIA GeForce RTX 5090',
    compute_capability='120', nvrtc_build='13.4.59',
    platform_family='linux', measured='2026-09-30', complete=False,
    frames=MappingProxyType({'vert_interp': 512}),
)
SM89_VERT_NVRTC_13_4_59 = KernelFrameRecording(
    box='node-1', device='NVIDIA GeForce RTX 4090',
    compute_capability='89', nvrtc_build='13.4.59',
    platform_family='linux', measured='2026-09-30', complete=False,
    frames=MappingProxyType({'vert_interp': 512}),
)


#: Every recording, oldest reading first.  Order is not significant to
#: the ceiling; it is the order a reader should walk them in.
SM120_NVRTC_13_4_59_COLD_START = KernelFrameRecording(
    box='weather-node-2', device='NVIDIA GeForce RTX 5090',
    compute_capability='120', nvrtc_build='13.4.59', platform_family='linux',
    measured='2026-09-30', complete=False,
    frames=MappingProxyType({'thompson_cold_start': 0}),
)
SM89_NVRTC_13_4_59_COLD_START = KernelFrameRecording(
    box='weather-node-1', device='NVIDIA GeForce RTX 4090',
    compute_capability='89', nvrtc_build='13.4.59', platform_family='linux',
    measured='2026-10-01', complete=False,
    # Re-read 2026-10-01 on this card at the lane's merge (843d2043e): 48 B,
    # where the closure lane's first reading of an earlier source was 0 B.
    frames=MappingProxyType({'thompson_cold_start': 48}),
)

KERNEL_LOCAL_FRAME_RECORDINGS: tuple[KernelFrameRecording, ...] = (
    # gp-libm64 records only its new grading unit on the measured compiler.
    KernelFrameRecording(
        box='node-2', device='NVIDIA GeForce RTX 5090',
        compute_capability='120', nvrtc_build='13.4.59',
        platform_family='linux', measured='2026-09-30', complete=False,
        # YSU's production module, including ysu_column_topo, re-read
        # 2026-10-02 on this card at this build: every export is 0 B.
        # This first partial recording is the one recording_for selects.
        frames=MappingProxyType({'portable_libm64_grade': 0, 'ysu': 0,
                                'bandwidth_glue': 0, 'diff_opt1': 0,
                                'wrf_cold_start_w': 0}),
    ),
    KernelFrameRecording(
        box='node-1', device='NVIDIA GeForce RTX 4090',
        compute_capability='89', nvrtc_build='13.4.59',
        platform_family='linux', measured='2026-09-30', complete=False,
        frames=MappingProxyType({'portable_libm64_grade': 48}),
    ),
    SM120_NVRTC_13_4_59_COLD_START,
    SM89_NVRTC_13_4_59_COLD_START,
    SM120_NVRTC_13_0_48,
    SM120_NVRTC_13_3_33,
    SM86_NVRTC_13_0_48,
    # ------------------------------------------------------------------
    # ADDED 2026-08-29, because cu_physics = 16 could not run without
    # it: kernel_local_frame_bytes is FAIL-CLOSED and raised on the
    # missing 'ntiedtke' row rather than charging zero. That is the
    # ninth site of the Phase 2 group and the third to announce itself
    # through a raise rather than a silent default.
    #
    # A FOURTH COMPILE PLATFORM, not a row bolted onto an existing one.
    # A frame is what NVRTC emitted for one architecture at one
    # compiler build -- this file's own opening paragraph -- so
    # writing 'ntiedtke': 0 into recordings measured on 13.0.48 and
    # 13.3.33 would have been asserting a measurement never taken.
    # Measured here with tools/vram_reserve_probe.py frames.
    #
    # IT RAISES NOTHING. Checked before adding: no module's ceiling
    # moves, so no existing budget changes. It adds three rows the
    # ceiling did not carry at all -- ntiedtke, mynn_dmp_sibling and
    # mynn_scalar_mix, all 0 B -- and the latter two were a latent
    # instance of the same fail-closed raise waiting for whoever
    # priced a run that selected them.
    KernelFrameRecording(
        box='node-1 (this workstation)',
        device='NVIDIA GeForce RTX 5070 Ti',
        compute_capability='120',
        nvrtc_build='13.0.88',
        platform_family='windows',
        measured='2026-08-29',
        # Every module the probe could compile on that date. This reading
        # predates lbc_time.cu (2026-09-04), so it cannot claim complete
        # coverage of today's source or borrow a row from another compiler.
        # Every value it did record still has an exact-equality driver gate.
        complete=False,
        frames=MappingProxyType({
            'acoustic': 544,
            'advection': 0,
            'coriolis_map': 0,
            'diagnostics': 0,
            'diff6': 0,
            'diff6_seam': 0,
            'diffusion': 0,
            # Read 2026-10-02 on RTX 5090 and RTX 5070 Ti, through the production
            # loader at this NVRTC: both coordinate-diffusion exports 0 B.
            'diff_opt1': 0,
            # New merged leaf modules, read on RTX 5090 at the same
            # compiler. All exported kernels are 0 B, on node-4's RTX 5070 Ti too.
            'bandwidth_glue': 0,
            'noah_init': 0,
            'wrf_cold_start_w': 0,
            'dycore': 0,
            'ftz_probe': 0,
            # RE-READ 2026-09-28 on an RTX 5090 (node-2) and an RTX 5070
            # Ti (node-4), NVRTC 13.0.88 build id CL-36424714 on both:
            # gf 88 -> 72 and shinhong 14,040 -> 13,000 (its row below)
            # since this recording was read; every other row reproduced
            # to the byte.
            'gf': 72,
            'health': 0,
            'health_tile': 0,
            'jacobi_eigh': 0,
            'kessler': 5120,
            'kf': 512,
            'kf_validation': 0,
            'lbc_flow': 0,
            'lbc_state': 0,
            'microphysics_validation': 0,
            'milbrandt2': 2048,
            'morrison': 5120,
            'myjpbl': 9232,
            'myjsfc': 0,
            'mynn_dmp_sibling': 0,
            'mynn_pbl': 0,
            'mynn_scalar_mix': 0,
            'mynn_surface': 0,
            'nest': 0,
            'nest_microphysics': 0,
            'noah': 176,
            'noahmp_bareflux': 0,
            'noahmp_fluxprep': 0,
            'noahmp_leaves': 272,
            'noahmp_radiation': 0,
            'noahmp_sflx': 0,
            'noahmp_snow': 200,
            'noahmp_soilwater': 0,
            'noahmp_vegeflux': 0,
            'noahmp_vegprecip': 0,
            'noahmp_water': 224,
            'nssl2': 15504,
            'nssl2_diagnostics': 0,
            'nssl2_driver_support': 15504,
            'nssl2_fused_gs': 112,
            'nssl2_nucond': 0,
            'nssl2_qvexcess': 0,
            'ntiedtke': 0,
            'openbc': 0,
            'pd_advection': 0,
            'refl': 18432,
            'rrtmg_lw': 0,
            'rrtmg_mcica_wrf': 0,
            'rrtmgp_cloud': 0,
            'rrtmgp_gas': 512,
            'rrtmgp_mcica': 0,
            'rrtmgp_rte': 3600,
            'rrtmgp_validation': 0,
            'ruc': 144,
            'sase': 6272,
            'saxpy': 0,
            'sfclay': 0,
            'shinhong': 0,
            'shinhong_validation': 0,
            'smag2d': 0,
            'spec_bdy': 0,
            'thompson': 11264,
            'thompson_aerosol_cold': 0,
            'thompson_aerosol_probe': 0,
            'thompson_aerosol_sat': 0,
            'thompson_aerosol_sed': 9216,
            'thompson_aerosol_state': 40,
            'thompson_aerosol_warm': 0,
            'tke_budget': 0,
            'uh_diag': 0,
            'vert_interp': 768,
            # RE-READ 2026-09-28, RTX 5070 Ti (node-4), NVRTC 13.0.88,
            # after 308c2d39e (WDM6 rain mass and number conservation)
            # took falk and falkn, two per-level arrays, off the stack:
            # 9,776 -> 9,264 B at WDM6_KMAX 64, 11,568 B at 80.
            'wdm6': 9264,
            'wdm6_refl': 16128,
            'wsm6': 7216,
            'ysu': 0,
            'ysu_validation': 0,
        }),
    ),

    # A partial recording of the new boundary module, independently measured
    # through the production loader on local WSL. No other module was read.
    KernelFrameRecording(
        box='development-desktop (WSL2)',
        device='NVIDIA GeForce RTX 3080',
        compute_capability='86',
        nvrtc_build='12.8.93',
        platform_family='linux',
        measured='2026-09-05',
        complete=False,
        frames=MappingProxyType({'lbc_time': 0}),
    ),

    # The compile platform of every desktop install of this release, read
    # 2026-09-11 through the runtime ArWen 2.7.2 installs.  Defined above
    # beside this architecture's other recording, where the eight stems it
    # read and why those eight are written out.
    SM86_NVRTC_12_9_86,
    # What a fresh gpu-cu13 install compiles on since 2026-09-16, read
    # 2026-09-28 on an RTX 5090 and an RTX 5070 Ti.  Defined above.
    SM120_NVRTC_13_4_92,
    SM120_NVRTC_12_9_86,
    # The same compiler on sm_89 (RTX 4090), read 2026-09-30 for the
    # urban canopy models and the UW moist-turbulence PBL.  Defined above.
    SM89_NVRTC_13_4_92,
    SM120_VERT_NVRTC_13_4_59,
    SM89_VERT_NVRTC_13_4_59,
)


#: Translation units that LAUNCH and deliberately have no row in any
#: recording above, each with the reason -- so the next reader finds a
#: decision instead of an absence.  Keys must be exactly
#: ``preflight.CHAINED_TRANSLATION_UNIT_FRAMES``; the gate is
#: ``tests/test_kernel_frame_recordings.py::
#: test_every_chained_translation_unit_says_why_it_has_no_recording``,
#: which fails on a fourth composed unit that arrives without a sentence
#: here.  What every entry has in common, and what the reasons are for:
#: a composed unit is priced on every platform from ONE reading, and the
#: cross-platform drift check (``preflight.under_priced_kernel_frames``,
#: driven from the ``.cu`` enumeration) is structurally unable to see it.
CHAINED_UNITS_WITHOUT_A_PER_PLATFORM_ROW = MappingProxyType({
    "noah_mosaic_ucm_unit":
        "glibc_flt32.cuh, unchanged urban_ucm.cu and noah_mosaic.cu compose "
        "one NOAH_MOSAIC_UCM translation unit through _mosaic_ucm_module, "
        "with C++17 and fmad disabled, so no *.cu enumeration reaches it.  "
        "Priced 1,040 B, noah_mosaic_ucm_column's local_size_bytes on sm_120 "
        "(RTX 5090) at NVRTC 12.9.86, the default install's compiler, read "
        "2026-10-01: the widest of four readings (sm_89 400 B at 12.9.86 "
        "and 13.4.92, sm_120 288 B at 13.4.92).  Re-read on any device by "
        "tests/test_noah_mosaic_driver.py::"
        "test_the_mosaic_units_compile_to_the_frames_they_are_priced_at.",
    # P3 one-category (mp_physics = 50), priced at 0 B.
    #
    # WHY IT CANNOT HAVE A ROW: the unit is ``noahmp_leaves.cu`` +
    # ``p3.cu`` assembled by ``gpuwm/core/p3_device.p3_source()``.
    # ``p3.cu`` borrows the tree's single audited glibc r_pow/r_exp/r_log
    # from ``noahmp_leaves.cu`` rather than carrying a second copy that
    # could drift, so it fails NVRTC standalone, sits in
    # ``preflight.UNMEASURED_KERNEL_MODULES``, and no ``*.cu`` glob can
    # reach it.  ``p3`` therefore has no measurable frame at all and a row
    # for it would be a value nothing measured, which is exactly what the
    # ``complete`` flag above exists to refuse.
    #
    # WHY THE 0 B IS STRUCTURAL, not lucky: ``p3.cu`` declares no
    # per-thread column array anywhere.  All eighteen ``(nk, ncol)``
    # companions -- twelve carriers and six of shared sedimentation
    # workspace, 72 B per grid cell -- live in the global workspace
    # ``gpuwm/core/p3_device.make_workspace()`` allocates, which is the
    # move ``gf.cu``, ``ysu.cu`` and ``kf.cu`` each made to get their own
    # frames to nothing.  P3's state inventory is what makes that cheap:
    # ONE ice category with a rime pair (qir mass, qib volume) and no qs
    # and no qg at all, so there is no snow or graupel column to carry.
    #
    # WHAT IS NOT STRUCTURAL: spilling.  The widest kernel of the unit,
    # ``p3k_kloopmain``, sits at 244 registers and the fused
    # ``p3k_fused_process`` at 250, against a 255-register ceiling, so a
    # different architecture or NVRTC build can spill where sm_120 /
    # 13.x did not.  A spill would make the priced 0 B under-charge by the
    # whole frame times the resident-thread capacity, and nothing in the
    # ``.cu`` enumeration would notice.  The leg that does notice, on any
    # box with a device, is ``tests/test_p3_cuda_gpu.py::
    # test_the_composed_unit_compiles_to_the_frame_it_is_priced_at``.
    "p3_composed":
        "noahmp_leaves.cu + p3.cu, assembled by "
        "gpuwm/core/p3_device.p3_source(); p3.cu cannot compile standalone "
        "(it borrows the tree's one glibc r_pow/r_exp/r_log), so no *.cu "
        "enumeration reaches it.  Priced 0 B from a single sm_120 / cupy "
        "14.2.0 reading, 2026-08-29; re-audited on any device by "
        "tests/test_p3_cuda_gpu.py::"
        "test_the_composed_unit_compiles_to_the_frame_it_is_priced_at.",
    # The legacy-RRTMG pair.  Not this lane's site: their provenance,
    # their 2026-07-27 reading and their drift bound are recorded where
    # they are priced (gpuwm/core/preflight.py
    # CHAINED_TRANSLATION_UNIT_FRAMES) and in
    # docs/rrtmg_legacy_integration.md section 6.  They are named here
    # because the gate is "every composed unit says why", and a gate that
    # covered only the newest one would let the fourth hide.
    "rrtmg_lw_legacy_chain":
        "rrtmg_lw.cu + rrtmg_lw_chain.cu + its address-only twin "
        "rrtmg_lw_chain_coalesced.cu + the four rrtmg_lw_taugb* band "
        "fragments + the batched entries rrtmg_lw_zbatched.cu as one unit (gpuwm/core/rrtmg_lw.py section 10); the "
        "fragments fail NVRTC standalone.  Priced 2,048 B from a single "
        "sm_120 / cupy 14.0.1 reading, 2026-07-27; bounded on a device by "
        "tests/test_rrtmg_lw_cuda.py (LOCAL_FRAME_BOUNDS).",
    "rrtmg_sw_legacy":
        "rrtmg_sw.cu through its own unit (gpuwm/core/rrtmg_sw.py); the "
        "fragment fails NVRTC standalone.  Priced 0 B from the same "
        "sm_120 / cupy 14.0.1 reading, 2026-07-27.",
    # BEP+BEM (sf_urban_physics = 3).
    "urban_bem_composed":
        "glibc_flt32.cuh + glibc_trig_flt32.cuh + urban_bem.cuh + "
        "urban_bep_bem.cu compiled as one unit with -fmad=false --ftz=false "
        "by gpuwm/core/urban_bem.py (_bem_module); urban_bep_bem.cu fails "
        "NVRTC standalone.  Priced 5,128 B, bep_bem_columns' sm_120 / "
        "NVRTC 12.9.86 reading of 2026-10-01, the widest of four readings "
        "(sm_89 4,368 B at 12.9.86 and 4,144 B at 13.4.92, sm_120 "
        "2,312 B at 13.4.92; bep_bem_class_init 0 B on all four).  "
        "Re-read on any device by tests/test_urban_bem_wrf471_parity.py::"
        "test_the_composed_bem_unit_compiles_within_the_priced_frame.",
    # Noah mosaic (sf_surface_mosaic = 1).
    "noah_mosaic_unit":
        "glibc_flt32.cuh + noah_mosaic.cu compiled as one unit with "
        "-std=c++17 --fmad=false by gpuwm/core/noah_mosaic.py "
        "(_mosaic_module); the *.cu glob compiles noah_mosaic.cu with the "
        "loader's options, which is not what launches.  Priced 688 B, "
        "noah_mosaic_column's local_size_bytes on sm_120 (RTX 5090) at "
        "NVRTC 12.9.86, the default install's compiler, read 2026-10-01: "
        "the widest of four readings (sm_89 240 B at 12.9.86 and 13.4.92, "
        "sm_120 176 B at 13.4.92).  Re-read on any device by "
        "tests/test_noah_mosaic_driver.py::"
        "test_the_mosaic_units_compile_to_the_frames_they_are_priced_at.",
    "terrain_drag_composed":
        "glibc_flt32.cuh + glibc_trig_flt32.cuh + terrain_drag.cu, assembled "
        "by gpuwm/core/terrain_drag.module_source() and compiled directly "
        "with -fmad=false --ftz=false; terrain_drag.cu fails standalone. "
        "Priced 1,184 B from sm_89 and sm_120 at NVRTC 13.4.59, measured "
        "2026-10-01 over every export; re-read 2026-10-02 on sm_120 at "
        "13.4.59, 13.4.92 and 12.9.86 with the same maximum. Re-audited by "
        "tests/test_terrain_drag_wrf471_parity.py::"
        "test_production_drag_frames_match_the_recorded_unit.",
})


# The direct production unit's compile/load-only readings, over every
# export.  These are separate from standalone .cu recordings: the unit
# requires the two glibc headers.  The source and options bind the readings
# to the composition that was measured, rather than only to its filename.
TERRAIN_DRAG_COMPOSED_SOURCE_SHA256 = (
    "ee9a70dfaff59d8e9d36f6b4a0214fa9124c72742b0bb60726639460f7b60fcf")
TERRAIN_DRAG_COMPOSED_OPTIONS = ("-std=c++17", "-fmad=false", "--ftz=false")
TERRAIN_DRAG_COMPOSED_FRAME_READINGS = MappingProxyType({
    ("89", "13.4.59"): MappingProxyType({
        "topo_wind_static": 0, "terrain_pbl_top": 0,
        "gwdo_column": 640, "gwdo_gsl_column": 1184}),
    ("120", "13.4.59"): MappingProxyType({
        "topo_wind_static": 0, "terrain_pbl_top": 0,
        "gwdo_column": 640, "gwdo_gsl_column": 1184}),
    ("120", "13.4.92"): MappingProxyType({
        "topo_wind_static": 0, "terrain_pbl_top": 0,
        "gwdo_column": 640, "gwdo_gsl_column": 1184}),
    ("120", "12.9.86"): MappingProxyType({
        "topo_wind_static": 0, "terrain_pbl_top": 0,
        "gwdo_column": 640, "gwdo_gsl_column": 1184}),
})


# ---------------------------------------------------------------------------
# Noah-MP runtime translation units: per-platform readings, and their own
# ceiling for a platform nobody has read.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ComposedUnitFrameRecording:
    """One compile platform's compile/load-only reading of the Noah-MP
    runtime translation units.

    WHAT IT MEASURES.  Each of the fifteen units in
    :data:`gpuwm.core.noahmp_kernel_sources.NOAHMP_TRANSLATION_UNITS` was
    compiled through the one production factory
    (:func:`gpuwm.core.noahmp_kernel_sources.compile_runtime_unit`, the
    same source string and option tuple a forecast hands NVRTC) in a
    fresh process with an empty CuPy cache, and every exported
    ``__global__`` of the loaded module was asked for
    ``CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES`` (``get_function(...).attributes
    ["local_size_bytes"]``).  ``frames`` is the maximum of that attribute
    over the unit's exports, keyed by the unit's pricing key.  No kernel
    was launched and no constant table was uploaded; a frame is a compile
    attribute, and the driver takes the reservation it prices at launch.

    WHY THESE ROWS ARE NOT IN A :class:`KernelFrameRecording`.  Five of
    the units are compositions (``noahmp_leaves.cu`` prepended to a
    fragment that borrows its r_pow/r_exp/r_log), so no ``*.cu`` glob
    reaches them and they may not carry a key in ``frames`` mappings the
    standalone ceiling is checked against.  They are priced from their
    own table: a card whose compile platform has a row here is priced
    from that row, and a card whose platform has none is priced from the
    element-wise ceiling over these rows -- the same rule the standalone
    census applies to an unrecorded platform -- with the basis printed
    beside the number so a user can see which of the two they got
    (:func:`gpuwm.core.noahmp_frame_provenance.frame_basis_for_profile`).
    The reading ``noahmp_leaves`` alone gives across the standalone
    platforms above (272 / 208 / 272 B) is the demonstration that the
    frame moves with the compiler build, which is why the row is keyed
    on the pair and why the ceiling, never an average or the nearest
    row, is what an unread pair is charged.  Adding a platform is adding
    a row, taken with ``tools/measure_noahmp_frames.py measure`` on that
    card; from then on that card is priced exactly.

    ``unit_identity`` binds each row to the exact unit it read:
    :meth:`gpuwm.core.noahmp_kernel_sources.RuntimeUnit.identity`'s
    ``identity_sha256`` at measurement time (ordered component hashes,
    the common preamble, the full composed source, the option tuple and
    every exported kernel).  A source or option edit changes the digest,
    the row stops matching the tree and is withdrawn from both the exact
    match and the ceiling until the platform is re-read; with no usable
    row left at all the estimator refuses --
    ``tests/test_noahmp_frame_provenance.py`` makes a stale row a red CPU
    test rather than a silent stale price.
    """

    box: str
    device: str
    compute_capability: str
    nvrtc_build: str
    platform_family: str
    measured: str
    #: Pricing key -> widest ``local_size_bytes`` over the unit's exports.
    frames: Mapping[str, int]
    #: Pricing key -> ``RuntimeUnit.identity()["identity_sha256"]`` read.
    unit_identity: Mapping[str, str]

    @property
    def platform_key(self) -> tuple[str, str]:
        """What decides the frames: the target arch and the compiler."""
        return (self.compute_capability, self.nvrtc_build)


#: Every Noah-MP composed-unit recording, oldest reading first.  A card
#: on a recorded platform is priced from its own row; a card on any other
#: platform from the element-wise maximum over the rows that describe
#: this tree (:func:`gpuwm.core.noahmp_frame_provenance.composed_frame_ceiling`).
NOAHMP_COMPOSED_FRAME_RECORDINGS: tuple[ComposedUnitFrameRecording, ...] = (
    # weather-node-1, RTX 5070 Ti (70 SMs x 1,536), Linux, sm_120 at NVRTC
    # 13.3.33 -- the same compile platform as SM120_NVRTC_13_3_33 above,
    # re-read 2026-09-10 for the fifteen Noah-MP runtime units with
    # `python tools/measure_noahmp_frames.py measure` (fresh process, empty
    # CuPy cache, compile_runtime_unit for each unit, function attributes
    # on all 108 exports, zero launches).  The three standalone stems that
    # ALSO carry a row in that recording agree with it to the byte
    # (noahmp_leaves 208, noahmp_snow 200, noahmp_water 224), which is the
    # cross-check that this reading and the standalone census are looking
    # at the same compiler.
    #
    # WHAT IT PRICES: the widest unit is noahmp_glacier_composed at 456 B,
    # under the 1,024 B fresh default stack, so on this platform Noah-MP
    # adds 0 B to the launch-time reservation -- the refusal that stood on
    # scheme 4 since the 1.8.8 sweep was guarding a term that costs nothing
    # here.  The reservation a Noah-MP configuration pays is then whatever
    # the rest of its kernel set pays (rrtmgp_rte at 3,600 B on this
    # platform, for instance), exactly as for scheme 2.
    ComposedUnitFrameRecording(
        box='weather-node-1',
        device='NVIDIA GeForce RTX 5070 Ti',
        compute_capability='120',
        nvrtc_build='13.3.33',
        platform_family='linux',
        measured='2026-09-10',
        frames=MappingProxyType({
            'noahmp_bareflux': 0,
            'noahmp_driver_composed': 352,
            'noahmp_energy_composed': 208,
            'noahmp_fluxprep': 0,
            'noahmp_glacier_composed': 456,
            'noahmp_leaves': 208,
            'noahmp_libm_slab_composed': 208,
            'noahmp_radiation': 0,
            'noahmp_sflx': 0,
            'noahmp_snow': 200,
            'noahmp_soilwater': 0,
            'noahmp_thermal_composed': 368,
            'noahmp_vegeflux_runtime': 0,
            'noahmp_vegprecip': 0,
            'noahmp_water': 224,
        }),
        unit_identity=MappingProxyType({
            'noahmp_bareflux':
                '0b93a806f083cf33808f147ea0ed104cbae32f26305e4476834dc0ddf747dfce',
            'noahmp_driver_composed':
                '287907ce202a69af463063c1308a7b2726582560999915cd9c7411908e16b18b',
            'noahmp_energy_composed':
                '531ed4d33a90371bd4c6ca0f2cac7a73704655960147421df7158add99933071',
            'noahmp_fluxprep':
                '42b862510256d660d7f2a4e8d60fcb1d2fbca923509686a061034d6b8896f0ec',
            'noahmp_glacier_composed':
                '569ceb5fb679388581bc1b6fc718aa0aa5304b31d1c9cbfe35d0974427d3e2d1',
            'noahmp_leaves':
                '39bdbf3a366ac8b8876600a6becaa46754adc1142c7722a7965cc98945c60878',
            'noahmp_libm_slab_composed':
                '88b3b10a4c35d6d86b199b77173cddf8e6d89be75da243adc773ff6b2dce959c',
            'noahmp_radiation':
                '77216cfeff3226232467eb91d8bf625ca661e9424f712ae8da748904eaee9acd',
            'noahmp_sflx':
                '76ca4db5df80bce0ba053e65d615d64058ffcf6efd1a8a1e482568f8fed48f80',
            'noahmp_snow':
                '39205d245c1e334367ad7bef5c521f527fbfd9895adc2613b2e246c4963e7262',
            'noahmp_soilwater':
                '2eb119965a7759ce3b8f29c84526634db02d3830c9378a3426add974eeb5910f',
            'noahmp_thermal_composed':
                '6d9d80383b8a9a818238ae04e66c0779239a532b2e601b9a1d9e3e8c6dfa22a0',
            'noahmp_vegeflux_runtime':
                '517bc16aa14c818c1f0185dab3ab0142b62fbe6004d96a60d2fffea250c716ae',
            'noahmp_vegprecip':
                '80e07bc9d36be385bf265617529f2c7b7111e0225b8fc738e357946bda1fd874',
            'noahmp_water':
                '69844cbf94941db0aeaeeb23cdf7f41aa4fdf9f0b7e51e140425af3348c125fb',
        }),
    ),
    # weather-node-1, the same RTX 5070 Ti, Linux, sm_120 at NVRTC 13.4.59
    # -- the compiler a FRESH `pip install gpuwm[gpu-cu13]` has installed
    # since 2026-09-09, when cuda-toolkit 13.4.1 (the `[ctk]` extra's
    # resolution) began pinning nvidia-cuda-nvrtc 13.4.59.  Read
    # 2026-09-10 with the same instrument and the same fresh-process
    # discipline as the row above (108 exports, zero launches), with the
    # 13.4.59 library first on the loader path.  Every frame and every
    # unit identity is byte-identical to the 13.3.33 row: on this
    # architecture the 13.3 -> 13.4 compiler step moved none of the
    # fifteen Noah-MP units, which is a measured fact about these two
    # builds and licenses NOTHING about a third -- a build with no row
    # is priced from the ceiling over the rows, and the basis says so.
    #
    # Why the row exists at all: the row above was taken from a venv
    # whose cuda-toolkit resolved in the 13.3.x window, and a release
    # that carried only it admitted Noah-MP on no fresh install anywhere,
    # because the package's own dependency spec no longer resolves to
    # that compiler.  tests/test_kernel_frame_recordings.py holds the
    # gate that keeps the recorded builds in step with what the spec
    # resolves to (:data:`RESOLVED_TOOLCHAIN_PINS`).
    ComposedUnitFrameRecording(
        box='weather-node-1',
        device='NVIDIA GeForce RTX 5070 Ti',
        compute_capability='120',
        nvrtc_build='13.4.59',
        platform_family='linux',
        measured='2026-09-10',
        frames=MappingProxyType({
            'noahmp_bareflux': 0,
            'noahmp_driver_composed': 352,
            'noahmp_energy_composed': 208,
            'noahmp_fluxprep': 0,
            'noahmp_glacier_composed': 456,
            'noahmp_leaves': 208,
            'noahmp_libm_slab_composed': 208,
            'noahmp_radiation': 0,
            'noahmp_sflx': 0,
            'noahmp_snow': 200,
            'noahmp_soilwater': 0,
            'noahmp_thermal_composed': 368,
            'noahmp_vegeflux_runtime': 0,
            'noahmp_vegprecip': 0,
            'noahmp_water': 224,
        }),
        unit_identity=MappingProxyType({
            'noahmp_bareflux':
                '0b93a806f083cf33808f147ea0ed104cbae32f26305e4476834dc0ddf747dfce',
            'noahmp_driver_composed':
                '287907ce202a69af463063c1308a7b2726582560999915cd9c7411908e16b18b',
            'noahmp_energy_composed':
                '531ed4d33a90371bd4c6ca0f2cac7a73704655960147421df7158add99933071',
            'noahmp_fluxprep':
                '42b862510256d660d7f2a4e8d60fcb1d2fbca923509686a061034d6b8896f0ec',
            'noahmp_glacier_composed':
                '569ceb5fb679388581bc1b6fc718aa0aa5304b31d1c9cbfe35d0974427d3e2d1',
            'noahmp_leaves':
                '39bdbf3a366ac8b8876600a6becaa46754adc1142c7722a7965cc98945c60878',
            'noahmp_libm_slab_composed':
                '88b3b10a4c35d6d86b199b77173cddf8e6d89be75da243adc773ff6b2dce959c',
            'noahmp_radiation':
                '77216cfeff3226232467eb91d8bf625ca661e9424f712ae8da748904eaee9acd',
            'noahmp_sflx':
                '76ca4db5df80bce0ba053e65d615d64058ffcf6efd1a8a1e482568f8fed48f80',
            'noahmp_snow':
                '39205d245c1e334367ad7bef5c521f527fbfd9895adc2613b2e246c4963e7262',
            'noahmp_soilwater':
                '2eb119965a7759ce3b8f29c84526634db02d3830c9378a3426add974eeb5910f',
            'noahmp_thermal_composed':
                '6d9d80383b8a9a818238ae04e66c0779239a532b2e601b9a1d9e3e8c6dfa22a0',
            'noahmp_vegeflux_runtime':
                '517bc16aa14c818c1f0185dab3ab0142b62fbe6004d96a60d2fffea250c716ae',
            'noahmp_vegprecip':
                '80e07bc9d36be385bf265617529f2c7b7111e0225b8fc738e357946bda1fd874',
            'noahmp_water':
                '69844cbf94941db0aeaeeb23cdf7f41aa4fdf9f0b7e51e140425af3348c125fb',
        }),
    ),
    # weather-node-1, the same RTX 5070 Ti (70 SMs x 1,536), Linux, sm_120
    # at NVRTC 12.9.86 -- the compiler every fresh CUDA-12 install
    # compiles on.  `cupy-cuda12x[ctk]>=14.0` resolves cuda-toolkit
    # 12.9.2.0, which pins nvidia-cuda-nvrtc-cu12 12.9.86; re-resolved
    # against the live index on 2026-09-11 (`python
    # tools/measure_noahmp_frames.py resolve --extra gpu-cu12`, exit 0)
    # and read the same day inside a venv installed from that
    # requirement, with the same instrument and the same fresh-process
    # discipline as the rows above (fifteen units, 108 exports, empty
    # CuPy cache, zero launches).
    #
    # Why the row exists: gpu-cu12 is not a minority spelling.  It is
    # what `gpuwm[gpu]` and `gpuwm[all]` alias to, and it is what the
    # packaged desktop runtime installs, so before this reading every
    # CUDA-12 install refused sf_surface_physics = 4 by name on every
    # card while carrying no way to reach the scheme at all.
    #
    # WHAT MOVED against the 13.x rows on this same card, and why it is
    # the compiler: noahmp_leaves reads 272 B here against 208 B at
    # 13.3.33 / 13.4.59, and the two units whose maximum IS the leaves
    # frame move with it (noahmp_energy_composed and
    # noahmp_libm_slab_composed, 208 -> 272); noahmp_driver_composed
    # reads 288 B against 352 B.  Every other unit is unmoved
    # (noahmp_glacier_composed 456, noahmp_thermal_composed 368,
    # noahmp_snow 200, noahmp_water 224, the rest 0).  That leaves value
    # is the one the standalone census already records for the older
    # compiler family on this architecture -- SM120_NVRTC_13_0_48 and
    # SM120_NVRTC_13_0_88 both read noahmp_leaves 272 -- and this
    # reading's three standalone stems agree with the 13.0.88 row to the
    # byte (leaves 272, snow 200, water 224), which is the cross-check
    # that the instrument and the standalone census are looking at the
    # same compiler.
    #
    # WHAT IT PRICES: the widest unit is noahmp_glacier_composed at
    # 456 B, under the 1,024 B fresh default stack, so Noah-MP adds 0 B
    # to the launch-time reservation on this platform too.
    ComposedUnitFrameRecording(
        box='weather-node-1',
        device='NVIDIA GeForce RTX 5070 Ti',
        compute_capability='120',
        nvrtc_build='12.9.86',
        platform_family='linux',
        measured='2026-09-11',
        frames=MappingProxyType({
            'noahmp_bareflux': 0,
            'noahmp_driver_composed': 288,
            'noahmp_energy_composed': 272,
            'noahmp_fluxprep': 0,
            'noahmp_glacier_composed': 456,
            'noahmp_leaves': 272,
            'noahmp_libm_slab_composed': 272,
            'noahmp_radiation': 0,
            'noahmp_sflx': 0,
            'noahmp_snow': 200,
            'noahmp_soilwater': 0,
            'noahmp_thermal_composed': 368,
            'noahmp_vegeflux_runtime': 0,
            'noahmp_vegprecip': 0,
            'noahmp_water': 224,
        }),
        unit_identity=MappingProxyType({
            'noahmp_bareflux':
                '0b93a806f083cf33808f147ea0ed104cbae32f26305e4476834dc0ddf747dfce',
            'noahmp_driver_composed':
                '287907ce202a69af463063c1308a7b2726582560999915cd9c7411908e16b18b',
            'noahmp_energy_composed':
                '531ed4d33a90371bd4c6ca0f2cac7a73704655960147421df7158add99933071',
            'noahmp_fluxprep':
                '42b862510256d660d7f2a4e8d60fcb1d2fbca923509686a061034d6b8896f0ec',
            'noahmp_glacier_composed':
                '569ceb5fb679388581bc1b6fc718aa0aa5304b31d1c9cbfe35d0974427d3e2d1',
            'noahmp_leaves':
                '39bdbf3a366ac8b8876600a6becaa46754adc1142c7722a7965cc98945c60878',
            'noahmp_libm_slab_composed':
                '88b3b10a4c35d6d86b199b77173cddf8e6d89be75da243adc773ff6b2dce959c',
            'noahmp_radiation':
                '77216cfeff3226232467eb91d8bf625ca661e9424f712ae8da748904eaee9acd',
            'noahmp_sflx':
                '76ca4db5df80bce0ba053e65d615d64058ffcf6efd1a8a1e482568f8fed48f80',
            'noahmp_snow':
                '39205d245c1e334367ad7bef5c521f527fbfd9895adc2613b2e246c4963e7262',
            'noahmp_soilwater':
                '2eb119965a7759ce3b8f29c84526634db02d3830c9378a3426add974eeb5910f',
            'noahmp_thermal_composed':
                '6d9d80383b8a9a818238ae04e66c0779239a532b2e601b9a1d9e3e8c6dfa22a0',
            'noahmp_vegeflux_runtime':
                '517bc16aa14c818c1f0185dab3ab0142b62fbe6004d96a60d2fffea250c716ae',
            'noahmp_vegprecip':
                '80e07bc9d36be385bf265617529f2c7b7111e0225b8fc738e357946bda1fd874',
            'noahmp_water':
                '69844cbf94941db0aeaeeb23cdf7f41aa4fdf9f0b7e51e140425af3348c125fb',
        }),
    ),
    # development-desktop, NVIDIA GeForce RTX 3080 (68 SMs x 1,536),
    # Windows, sm_86 at NVRTC 12.9.86 -- the compile platform of every
    # DESKTOP install of this release.  Read 2026-09-11 with `python
    # tools/measure_noahmp_frames.py measure` through the interpreter the
    # shipped ArWen 2.7.2 desktop runtime installs, which carries
    # cupy-cuda12x 14.2.0 and nvidia-cuda-nvrtc-cu12 12.9.86: exactly what
    # `cupy-cuda12x[ctk]>=14.0` resolves to (:data:`RESOLVED_TOOLCHAIN_PINS`),
    # so the reading is of the compiler the user's own installation
    # compiles on rather than of a checkout environment beside it.  Same
    # instrument and same discipline as the two rows above: fresh process,
    # empty CuPy cache, compile_runtime_unit for each of the fifteen units,
    # function attributes on all 108 exports, zero launches.
    #
    # Why the row exists: the desktop runtime is CUDA-12, and with only the
    # two sm_120 / NVRTC 13.x rows every desktop install refused
    # sf_surface_physics = 4 by name -- on the one card class whose
    # standalone frames this tree has read since 2026-08-20.
    #
    # WHAT IT PRICES: the widest unit is noahmp_glacier_composed at 456 B,
    # under the 1,024 B fresh default stack this card reports, so Noah-MP
    # adds 0 B to the launch-time reservation here as well; a scheme-4
    # configuration pays whatever the rest of its kernel set pays.
    #
    # WHAT MOVED: noahmp_driver_composed reads 304 B against the 352 B of
    # both sm_120 / NVRTC 13.x rows and the 288 B of the sm_120 row at
    # this same compiler, and noahmp_leaves (with the two units whose
    # maximum it is) reads 208 B against sm_120's 272 B at this compiler.
    # Architecture and build each move frames, which is why the row is
    # keyed on the pair; a platform with no row is priced from the
    # ceiling over the rows, and the basis says so.  Every unit identity
    # matches the other three rows.
    #
    # CALIBRATION, in the same sitting and on the same card: the three
    # stems that also carry a standalone row on this architecture agree
    # with SM86_NVRTC_13_0_48 to the byte (noahmp_leaves 208, noahmp_snow
    # 200, noahmp_water 224), and the bounded eight-stem pass recorded as
    # SM86_NVRTC_12_9_86 above reproduces every one of that recording's
    # values at this compiler.  That is what says this reading and the
    # standalone census are looking at the same card and the same
    # compiler, with the 13.0.48 -> 12.9.86 step visible in nothing.
    ComposedUnitFrameRecording(
        box='development-desktop',
        device='NVIDIA GeForce RTX 3080',
        compute_capability='86',
        nvrtc_build='12.9.86',
        platform_family='windows',
        measured='2026-09-11',
        frames=MappingProxyType({
            'noahmp_bareflux': 0,
            'noahmp_driver_composed': 304,
            'noahmp_energy_composed': 208,
            'noahmp_fluxprep': 0,
            'noahmp_glacier_composed': 456,
            'noahmp_leaves': 208,
            'noahmp_libm_slab_composed': 208,
            'noahmp_radiation': 0,
            'noahmp_sflx': 0,
            'noahmp_snow': 200,
            'noahmp_soilwater': 0,
            'noahmp_thermal_composed': 368,
            'noahmp_vegeflux_runtime': 0,
            'noahmp_vegprecip': 0,
            'noahmp_water': 224,
        }),
        unit_identity=MappingProxyType({
            'noahmp_bareflux':
                '0b93a806f083cf33808f147ea0ed104cbae32f26305e4476834dc0ddf747dfce',
            'noahmp_driver_composed':
                '287907ce202a69af463063c1308a7b2726582560999915cd9c7411908e16b18b',
            'noahmp_energy_composed':
                '531ed4d33a90371bd4c6ca0f2cac7a73704655960147421df7158add99933071',
            'noahmp_fluxprep':
                '42b862510256d660d7f2a4e8d60fcb1d2fbca923509686a061034d6b8896f0ec',
            'noahmp_glacier_composed':
                '569ceb5fb679388581bc1b6fc718aa0aa5304b31d1c9cbfe35d0974427d3e2d1',
            'noahmp_leaves':
                '39bdbf3a366ac8b8876600a6becaa46754adc1142c7722a7965cc98945c60878',
            'noahmp_libm_slab_composed':
                '88b3b10a4c35d6d86b199b77173cddf8e6d89be75da243adc773ff6b2dce959c',
            'noahmp_radiation':
                '77216cfeff3226232467eb91d8bf625ca661e9424f712ae8da748904eaee9acd',
            'noahmp_sflx':
                '76ca4db5df80bce0ba053e65d615d64058ffcf6efd1a8a1e482568f8fed48f80',
            'noahmp_snow':
                '39205d245c1e334367ad7bef5c521f527fbfd9895adc2613b2e246c4963e7262',
            'noahmp_soilwater':
                '2eb119965a7759ce3b8f29c84526634db02d3830c9378a3426add974eeb5910f',
            'noahmp_thermal_composed':
                '6d9d80383b8a9a818238ae04e66c0779239a532b2e601b9a1d9e3e8c6dfa22a0',
            'noahmp_vegeflux_runtime':
                '517bc16aa14c818c1f0185dab3ab0142b62fbe6004d96a60d2fffea250c716ae',
            'noahmp_vegprecip':
                '80e07bc9d36be385bf265617529f2c7b7111e0225b8fc738e357946bda1fd874',
            'noahmp_water':
                '69844cbf94941db0aeaeeb23cdf7f41aa4fdf9f0b7e51e140425af3348c125fb',
        }),
    ),
    # weather-node-2, RTX 5090 (170 SMs x 1,536), Linux, sm_120 at NVRTC
    # 13.4.92 -- the compiler a FRESH `pip install gpuwm[gpu-cu13]` has
    # installed since 2026-09-16, when cuda-toolkit 13.4.2 (the `[ctk]`
    # extra's resolution) began pinning nvidia-cuda-nvrtc 13.4.92.  Read
    # 2026-09-28 with `python tools/measure_noahmp_frames.py measure` in a
    # venv installed from `cupy-cuda13x[ctk]>=14.0` (CuPy 14.2.0, NVRTC
    # build id CL-38855100), same instrument and same fresh-process
    # discipline as the rows above (fifteen units, 108 exports, empty CuPy
    # cache, zero launches).  weather-node-4's RTX 5070 Ti (70 SMs) read
    # the same day in its own venv of the same resolution gave the same
    # frames and the same unit identities, so the row is a reading of the
    # platform and not of one card.
    #
    # WHAT MOVED: nothing.  Every frame and every unit identity equals the
    # 13.3.33 and 13.4.59 rows, and the three standalone stems agree with
    # SM120_NVRTC_13_4_92 to the byte (noahmp_leaves 208, noahmp_snow 200,
    # noahmp_water 224).  What the row changes is the basis: without it a
    # fresh gpu-cu13 install was priced from the ceiling over the other
    # rows (noahmp_leaves and the two units whose maximum it is at 272 B,
    # the CUDA-12 reading) and plan review said "not measured on this
    # card" about the card class the docs name as measured.
    ComposedUnitFrameRecording(
        box='weather-node-2',
        device='NVIDIA GeForce RTX 5090',
        compute_capability='120',
        nvrtc_build='13.4.92',
        platform_family='linux',
        measured='2026-09-28',
        frames=MappingProxyType({
            'noahmp_bareflux': 0,
            'noahmp_driver_composed': 352,
            'noahmp_energy_composed': 208,
            'noahmp_fluxprep': 0,
            'noahmp_glacier_composed': 456,
            'noahmp_leaves': 208,
            'noahmp_libm_slab_composed': 208,
            'noahmp_radiation': 0,
            'noahmp_sflx': 0,
            'noahmp_snow': 200,
            'noahmp_soilwater': 0,
            'noahmp_thermal_composed': 368,
            'noahmp_vegeflux_runtime': 0,
            'noahmp_vegprecip': 0,
            'noahmp_water': 224,
        }),
        unit_identity=MappingProxyType({
            'noahmp_bareflux':
                '0b93a806f083cf33808f147ea0ed104cbae32f26305e4476834dc0ddf747dfce',
            'noahmp_driver_composed':
                '287907ce202a69af463063c1308a7b2726582560999915cd9c7411908e16b18b',
            'noahmp_energy_composed':
                '531ed4d33a90371bd4c6ca0f2cac7a73704655960147421df7158add99933071',
            'noahmp_fluxprep':
                '42b862510256d660d7f2a4e8d60fcb1d2fbca923509686a061034d6b8896f0ec',
            'noahmp_glacier_composed':
                '569ceb5fb679388581bc1b6fc718aa0aa5304b31d1c9cbfe35d0974427d3e2d1',
            'noahmp_leaves':
                '39bdbf3a366ac8b8876600a6becaa46754adc1142c7722a7965cc98945c60878',
            'noahmp_libm_slab_composed':
                '88b3b10a4c35d6d86b199b77173cddf8e6d89be75da243adc773ff6b2dce959c',
            'noahmp_radiation':
                '77216cfeff3226232467eb91d8bf625ca661e9424f712ae8da748904eaee9acd',
            'noahmp_sflx':
                '76ca4db5df80bce0ba053e65d615d64058ffcf6efd1a8a1e482568f8fed48f80',
            'noahmp_snow':
                '39205d245c1e334367ad7bef5c521f527fbfd9895adc2613b2e246c4963e7262',
            'noahmp_soilwater':
                '2eb119965a7759ce3b8f29c84526634db02d3830c9378a3426add974eeb5910f',
            'noahmp_thermal_composed':
                '6d9d80383b8a9a818238ae04e66c0779239a532b2e601b9a1d9e3e8c6dfa22a0',
            'noahmp_vegeflux_runtime':
                '517bc16aa14c818c1f0185dab3ab0142b62fbe6004d96a60d2fffea250c716ae',
            'noahmp_vegprecip':
                '80e07bc9d36be385bf265617529f2c7b7111e0225b8fc738e357946bda1fd874',
            'noahmp_water':
                '69844cbf94941db0aeaeeb23cdf7f41aa4fdf9f0b7e51e140425af3348c125fb',
        }),
    ),
)


@dataclass(frozen=True)
class ResolvedToolchainPin:
    """The NVRTC build one GPU extra of this package installs.

    WHAT SETS THE COMPILE PLATFORM.  A CuPy wheel carries no compiler
    headers, so the ``gpu-cu13`` / ``gpu-cu12`` extras name
    ``cupy-cuda1Nx[ctk]``, whose ``[ctk]`` extra requires
    ``cuda-toolkit[...]==1N.*``, and each cuda-toolkit release pins
    ``nvidia-cuda-nvrtc`` to ONE exact build.  That build -- not the box,
    not the driver, not the card -- is the NVRTC half of the compile
    platform every fresh install compiles on, and it changes the day a
    new cuda-toolkit release lands on the index.  The frames in the
    tables above are readings of (architecture, NVRTC build) pairs, so a
    release whose Noah-MP rows were all read on a build the spec no longer
    resolves to admits Noah-MP on no fresh install at all, while its docs
    say it does.  The first such release was this one, before the 13.4.59
    row: the 13.3.33 reading came from a venv resolved in the 13.3.x
    window (cuda-toolkit 13.3.1), and cuda-toolkit 13.4.1 replaced it on
    the index on 2026-09-09.

    So the resolution is DECLARED here, dated, and gated twice:

    * ``tests/test_kernel_frame_recordings.py`` (no network) asserts that
      ``requirement`` is byte-for-byte the pyproject extra's entry, and
      that every ``current`` pin with ``noahmp_architectures`` has a
      :class:`ComposedUnitFrameRecording` at ``(arch, nvrtc_build)`` for
      each architecture -- so an edit to the extra, or a re-pin here
      without a reading, is a red CPU test rather than a refusal on every
      user's machine;
    * ``tools/measure_noahmp_frames.py resolve`` (network) resolves the
      requirement against the live index with ``pip install --dry-run
      --report`` and compares the ``nvidia-cuda-nvrtc`` version it
      returns with ``nvrtc_build``, so the declaration is re-checked at
      the cut and the day a new cuda-toolkit release moves the compiler
      is the day the table is known to need a row.

    ``current`` is False for a build the spec USED to resolve to: its
    rows still price the installs that resolved in that window, and the
    declaration says which window that was.
    """

    extra: str
    requirement: str
    cuda_toolkit: str
    nvrtc_distribution: str
    nvrtc_build: str
    resolved: str
    current: bool
    #: Compute capabilities on which this release prices Noah-MP from a
    #: reading of the card's own platform for this build: each must have
    #: a composed row at ``(arch, nvrtc_build)``.  Empty means "no reading
    #: on this build; a card on it is priced from the ceiling over the
    #: recorded platforms, and the basis says so".
    noahmp_architectures: tuple[str, ...]


#: The resolutions this release was checked against.  Re-resolve with
#: ``python tools/measure_noahmp_frames.py resolve`` before a cut; a
#: changed ``nvidia-cuda-nvrtc`` version is a new compile platform and
#: needs a new row from ``measure`` on each architecture listed.
RESOLVED_TOOLCHAIN_PINS: tuple[ResolvedToolchainPin, ...] = (
    # 2026-09-28, pip 26.2.1 against PyPI (`python
    # tools/measure_noahmp_frames.py resolve`): cupy-cuda13x 14.2.0 ->
    # cuda-toolkit 13.4.2 (uploaded 2026-09-16) -> nvidia-cuda-nvrtc
    # 13.4.92.  The compiler every fresh gpu-cu13 install has today; both
    # tables carry an sm_120 reading of it (SM120_NVRTC_13_4_92 and the
    # composed row), read the same day on an RTX 5090 and an RTX 5070 Ti.
    ResolvedToolchainPin(
        extra='gpu-cu13',
        requirement='cupy-cuda13x[ctk]>=14.0',
        cuda_toolkit='13.4.2',
        nvrtc_distribution='nvidia-cuda-nvrtc',
        nvrtc_build='13.4.92',
        resolved='2026-09-28',
        current=True,
        noahmp_architectures=('120',),
    ),
    # The window before it: cuda-toolkit 13.4.1.0 (uploaded 2026-09-09)
    # pinned nvidia-cuda-nvrtc 13.4.59, resolved 2026-09-10 with pip 25.2.
    # An install resolved between 2026-09-09 and the 13.4.2 upload on
    # 2026-09-16 runs this compiler and is priced from the 13.4.59 rows.
    ResolvedToolchainPin(
        extra='gpu-cu13',
        requirement='cupy-cuda13x[ctk]>=14.0',
        cuda_toolkit='13.4.1.0',
        nvrtc_distribution='nvidia-cuda-nvrtc',
        nvrtc_build='13.4.59',
        resolved='2026-09-10',
        current=False,
        noahmp_architectures=('120',),
    ),
    # The window before it: cuda-toolkit 13.3.x (13.3.1 on the venv the
    # first Noah-MP reading came from) pinned nvidia-cuda-nvrtc 13.3.33.
    # An install resolved between the 13.3.x upload and 2026-09-08 runs
    # this compiler and is priced from the 13.3.33 rows.
    ResolvedToolchainPin(
        extra='gpu-cu13',
        requirement='cupy-cuda13x[ctk]>=14.0',
        cuda_toolkit='13.3.1',
        nvrtc_distribution='nvidia-cuda-nvrtc',
        nvrtc_build='13.3.33',
        resolved='2026-09-08',
        current=False,
        noahmp_architectures=('120',),
    ),
    # 2026-09-10, same resolution method: cupy-cuda12x 14.2.0 ->
    # cuda-toolkit 12.9.2.0 -> nvidia-cuda-nvrtc-cu12 12.9.86.  Re-resolved
    # against the live index on 2026-09-11 with the same command, which
    # returned the same cuda-toolkit release and the same NVRTC build.
    # This is the compiler the shipped desktop runtime carries and the
    # extra `gpuwm[gpu]` and `gpuwm[all]` alias to, and two architectures
    # were read on it 2026-09-11 inside environments installed from this
    # very requirement: sm_120 on weather-node-1 and sm_86 on the platform
    # the packaged desktop runtime compiles on.  Both are declared here:
    # the declaration is the statement that a fresh install of this extra
    # lands on a recorded platform, and the gate in
    # tests/test_kernel_frame_recordings.py checks rows and pins against
    # each other in both directions.  A CUDA-12 card of any other
    # architecture is priced from the ceiling over the recorded rows with
    # the basis stated; a `measure` run on that card inside a gpu-cu12
    # environment, then a row and this tuple, makes it exact.
    ResolvedToolchainPin(
        extra='gpu-cu12',
        requirement='cupy-cuda12x[ctk]>=14.0',
        cuda_toolkit='12.9.2.0',
        nvrtc_distribution='nvidia-cuda-nvrtc-cu12',
        nvrtc_build='12.9.86',
        resolved='2026-09-10',
        current=True,
        noahmp_architectures=('120', '86'),
    ),
)


def noahmp_composed_recording_for(
        fingerprint: Mapping[str, object] | None
) -> ComposedUnitFrameRecording | None:
    """The Noah-MP composed-unit recording taken on THIS compile platform.

    Same two keys and the same refusal to read "unavailable" as a match
    as :func:`recording_for`.  ``None`` means the platform has no row of
    its own; the estimator then prices it from the Noah-MP ceiling and
    says so (:func:`gpuwm.core.noahmp_frame_provenance.frame_basis_for_profile`).
    """
    if not fingerprint:
        return None
    capability = fingerprint.get("device_compute_capability")
    build = fingerprint.get("nvrtc_build")
    for value in (capability, build):
        if not isinstance(value, str) or not value or value == "unavailable":
            return None
    for recording in NOAHMP_COMPOSED_FRAME_RECORDINGS:
        if recording.platform_key == (capability, build):
            return recording
    return None


def frame_ceiling() -> dict[str, int]:
    """The element-wise maximum over every recording.

    What a gate is allowed to charge when it does not know which compile
    platform it is on.  Never below a measurement, by construction.
    """
    ceiling: dict[str, int] = {}
    for recording in KERNEL_LOCAL_FRAME_RECORDINGS:
        for module, frame in recording.frames.items():
            if frame > ceiling.get(module, -1):
                ceiling[module] = int(frame)
    return ceiling


#: How a frame priced by :func:`assumed_frame_bound` is described wherever
#: it is printed, so a reader can tell an assumption from a reading.
ASSUMED_BOUND_PHRASE = "assumed bound, not measured"


def assumed_frame_bound() -> int:
    """The widest per-thread frame any module has ever been recorded at.

    What a module with no reading on ANY platform is charged.  It is an
    assumed BOUND, not a measurement: nothing this tree has measured is
    wider, so a reservation built on it is never short, and every place
    that prices from it says :data:`ASSUMED_BOUND_PHRASE` beside the
    number.  Pricing from it is what a missing reading costs; refusing
    the run instead was the 2.7.3 defect this replaces.
    """
    widest = max(frame_ceiling().values(), default=0)
    for recording in NOAHMP_COMPOSED_FRAME_RECORDINGS:
        for frame in recording.frames.values():
            widest = max(widest, int(frame))
    return int(widest)


def recording_for(fingerprint: Mapping[str, object] | None
                  ) -> KernelFrameRecording | None:
    """The recording taken on THIS compile platform, or ``None``.

    Matched on the two fingerprint keys that decide code generation.  An
    unresolved or absent key never matches: "unavailable" must not be
    read as "the reference box", which is how a fingerprint that measured
    nothing would otherwise license an exact-equality assertion.
    """
    if not fingerprint:
        return None
    capability = fingerprint.get("device_compute_capability")
    build = fingerprint.get("nvrtc_build")
    for value in (capability, build):
        if not isinstance(value, str) or not value or value == "unavailable":
            return None
    for recording in KERNEL_LOCAL_FRAME_RECORDINGS:
        if recording.platform_key == (capability, build):
            return recording
    return None
