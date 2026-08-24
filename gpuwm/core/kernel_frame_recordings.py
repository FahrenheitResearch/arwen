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
(``tools/vram_reserve_probe.py frames``), the rows that disagree:

  ======================  ==========  ==========  ==========
  module                  sm_120      sm_120      sm_86
                          13.0.48     13.3.33     13.0.48
  ======================  ==========  ==========  ==========
  gf                           88          72          88
  kf                          512         512         512
  noah                        176         176         224
  thompson_aerosol_warm         0           0         112
  ysu                           0           0           0
  nssl2_fused_gs              112         216         112
  rrtmgp_cloud                  0          40           0
  shinhong                 14,040      17,160      14,040
  noahmp_leaves               272         208         208
  rrtmgp_rte                5,152       5,152       3,600
  ======================  ==========  ==========  ==========

``gf``, ``ysu`` and ``kf`` are the three rows that went to ZERO-ish on
every platform, and they are the reason this file exists in its present
shape.  All three kernels kept a whole column in the per-thread local
frame; on 2026-08-21 all three moved those arrays into a global
workspace (``gpuwm/core/kernels/gf.cu``, ``ysu.cu``, ``kf.cu``).  ``gf``
went 22,416 -> 88 B, ``ysu`` 9,232 -> 0 B, ``kf`` 24,064 -> 512 B.

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

Four rows move with the ARCHITECTURE at a fixed compiler, four move with
the COMPILER BUILD at a fixed architecture, and ``noahmp_leaves`` moves
with both.  ``rrtmgp_rte`` is the one row that moves with neither: the
sm_86 reading is post-RRTMGP-optimisation (2026-08-20) and the two sm_120
readings are of the source as it stood before it, on boxes that are no
longer reachable to re-read.  The ceiling therefore over-prices sm_120
here until one of them is measured again, which is the safe direction and
costs nothing -- the module is nowhere near the widest frame on any
platform.  So the identity of a recording is the pair
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


#: The reading every shipped row came from: Drew's Windows desktop
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
    box='drew-desktop (RTX 5090 era)',
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
        'dycore': 0,
        'ftz_probe': 0,
        'gf': 88,
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
        'rrtmgp_rte': 5152,
        'rrtmgp_validation': 0,
        'ruc': 144,
        'sase': 6272,
        'saxpy': 0,
        'sfclay': 0,
        'shinhong': 14040,
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
        'wdm6': 9776,
        'wdm6_refl': 16128,
        'wsm6': 7216,
        'ysu': 0,
        'ysu_validation': 0,
    }),
)

#: weather-node-1, RTX 5070 Ti / Linux, 2026-08-20.  Same target
#: architecture as the recording above and a LATER NVRTC, which is what
#: isolates the compiler half: ``nssl2_fused_gs`` 112 -> 216,
#: ``rrtmgp_cloud`` 0 -> 40, ``shinhong`` 14,040 -> 17,160 and
#: ``noahmp_leaves`` 272 -> 208 move with the build alone.
SM120_NVRTC_13_3_33 = KernelFrameRecording(
    box='weather-node-1',
    device='NVIDIA GeForce RTX 5070 Ti',
    compute_capability='120',
    nvrtc_build='13.3.33',
    platform_family='linux',
    measured='2026-08-20',
    complete=True,
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
        'rrtmgp_rte': 5152,
        'rrtmgp_validation': 0,
        'ruc': 144,
        'sase': 6272,
        'saxpy': 0,
        'sfclay': 0,
        'shinhong': 17160,
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
        'wdm6': 9776,
        'wdm6_refl': 16128,
        'wsm6': 7216,
        'ysu': 0,
        'ysu_validation': 0,
    }),
)

#: Drew's Windows desktop as it stands now: RTX 3080, sm_86, on the
#: same NVRTC 13.0.48 the first recording used.  Holding the compiler
#: fixed is what isolates the architecture half: ``gf`` 22,416 ->
#: 23,984, ``noah`` 176 -> 224 and ``thompson_aerosol_warm`` 0 -> 112
#: compile WIDER here than the shipped rows, so those three rows were
#: under-pricing this card by 163,774,464 + 5,013,504 + 11,698,176 B
#: (0.17 GiB together, at 68 SMs x 1,536 threads) until the ceiling
#: took them up.
SM86_NVRTC_13_0_48 = KernelFrameRecording(
    box='drew-desktop',
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
    complete=True,
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
        'microphysics_validation': 0,
        'milbrandt2': 2048,
        'morrison': 5120,
        'myjpbl': 9232,
        'myjsfc': 0,
        'mynn_pbl': 0,
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
        # the reservation is unchanged.  The two sm_120 recordings still
        # carry 5,152 for a source that no longer exists; neither box is
        # reachable to re-read, and an over-priced frame is the safe
        # direction for a rail gate.
        'rrtmgp_rte': 3600,
        'rrtmgp_validation': 0,
        'ruc': 144,
        'sase': 6272,
        'saxpy': 0,
        'sfclay': 0,
        'shinhong': 14040,
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
        'wdm6': 9776,
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


#: Every recording, oldest reading first.  Order is not significant to
#: the ceiling; it is the order a reader should walk them in.
KERNEL_LOCAL_FRAME_RECORDINGS: tuple[KernelFrameRecording, ...] = (
    SM120_NVRTC_13_0_48,
    SM120_NVRTC_13_3_33,
    SM86_NVRTC_13_0_48,
)


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
