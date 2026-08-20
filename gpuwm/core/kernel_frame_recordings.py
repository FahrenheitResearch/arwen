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
  gf                       22,416      22,416      23,984
  noah                        176         176         224
  thompson_aerosol_warm         0           0         112
  ysu                       9,232       9,232       7,184
  nssl2_fused_gs              112         216         112
  rrtmgp_cloud                  0          40           0
  shinhong                 14,040      17,160      14,040
  noahmp_leaves               272         208         208
  ======================  ==========  ==========  ==========

Four rows move with the ARCHITECTURE at a fixed compiler, four move with
the COMPILER BUILD at a fixed architecture, and ``noahmp_leaves`` moves
with both.  So the identity of a recording is the pair
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
        'gf': 22416,
        'health': 0,
        'jacobi_eigh': 0,
        'kessler': 5120,
        'kf': 24064,
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
        'ysu': 9232,
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
        'gf': 22416,
        'health': 0,
        'health_tile': 0,
        'jacobi_eigh': 0,
        'kessler': 5120,
        'kf': 24064,
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
        'ysu': 9232,
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
        'gf': 23984,
        'health': 0,
        'health_tile': 0,
        'jacobi_eigh': 0,
        'kessler': 5120,
        'kf': 24064,
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
        'thompson_aerosol_warm': 112,
        'tke_budget': 0,
        'uh_diag': 0,
        'vert_interp': 768,
        'wdm6': 9776,
        'wdm6_refl': 16128,
        'wsm6': 7216,
        'ysu': 7184,
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
