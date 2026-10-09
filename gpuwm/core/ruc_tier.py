"""The RUC soil column's compile-time geometry tier.

``gpuwm/core/kernels/ruc.cu`` sizes every per-thread soil scratch array from
``RUC_NZS`` and selects its level table with the same macro.  This module is
the one place that decides what ``RUC_NZS`` a given soil geometry compiles
with, and it is the RUC analogue of the ``WPHI_MAX_LEV`` ladder that
:mod:`gpuwm.core.acoustic` owns for the implicit w''-phi'' solve.

WHY THIS IS ITS OWN MODULE, and not three functions in
:mod:`gpuwm.core.ruc_gpu` where the launchers that use them live:
``ruc_gpu`` imports CuPy at module scope, so importing it requires a CuPy
install.  The whole value of :func:`ruc_kernel_source` is that the string
NVRTC will receive can be digested, preprocessed and compiled to PTX on a
box with no CuPy and no card -- that is what makes the nine-level
bit-identity claim in ``tests/test_ruc_nzs_tier.py`` a measurement rather
than an assertion.  A tier helper that can only be imported next to a GPU
cannot carry that proof, so the tier lives here, beside the contract, and
``ruc_gpu`` imports it.  :mod:`gpuwm.core.kernels` is CuPy-free to import
for the same reason (its ``import cupy`` calls are inside the loaders).
"""

from __future__ import annotations

from gpuwm.core.kernels import (get_kernel, get_kernel_int_defines,
                                module_source, module_source_int_defines)
from gpuwm.core.ruc_contract import (NUM_SOIL_LAYERS,
                                     WRF_SUPPORTED_NUM_SOIL_LAYERS)

#: The RUC translation unit's name in :mod:`gpuwm.core.kernels`.
RUC_MODULE = "ruc"

#: SOILPROP's two WRF lineages, by name (``RunConfig.ruc_soilprop``).
#:
#: ``wrf_45``: WRF v4.0 to v4.5 ``phys/module_sf_ruclsm.F`` SOILPROP
#: (v4.5.2 :6154, :6213-6216, :6245), byte-identical in the operational
#: RAP/HRRR branch (:6343, :6402-6407, :6434).  Soil-water diffusivity and
#: hydraulic conductivity are normalised by the moisture above the residual,
#: ``(theta - qmin) / (theta_sat - qmin)``, and Johansen's mineral
#: conductivity is 2.0 at every quartz fraction.
#:
#: ``wrf_461``: WRF v4.6.1 (:6198-6202, :6261-6267, :6289): total moisture
#: over porosity, and 3.0 below 20 percent quartz.  In dry soil this
#: diffusivity is 2.5 to 8 times the v4.5 value.  MEASURED on a 3 km
#: afternoon cut of a native operational-HRRR start (2026-10-02 21Z, 106,671
#: land cells): the top soil level rose from 0.161 to 0.187 m3/m3 in the first
#: hour, fed from the levels below, where the operational model's own top
#: level fell to 0.157; latent heat flux 207 W/m2 against its 142.  Kept
#: selectable by name for WRF v4.6.1 column parity; it is not the default.
#:
#: Two layers, two defaults, on purpose.  The forecast runtime
#: (``ruc_lsm_step``, the fused step, every loader in this module) defaults
#: to ``wrf_45`` and is always handed ``RunConfig.ruc_soilprop``.  The
#: transcription leaves under it (``gpuwm.core.ruc`` and ``ruc_gpu``:
#: ``soilprop``, ``soil``, ``snowsoil``, ``sfctmp``, ``LSMRUC``) default to
#: ``wrf_461``, the lineage their WRF v4.6.1 oracles were recorded with, so a
#: bare leaf call is still the oracle's arithmetic; the runtime passes the
#: name at every call it makes into them.
RUC_SOILPROP_FORMS = ("wrf_45", "wrf_461")
RUC_SOILPROP_DEFAULT = "wrf_45"

#: LSMRUC's cold start of the ground vapour and condensate (QVG, QCG) when
#: the run starts without them (``RunConfig.ruc_qvg_cold_start``).
#:
#: ``air``: the operational RAP/HRRR branch, ``module_sf_ruclsm.F:479-483``
#: there: an invalid QVG takes the lowest-level vapour and QCG becomes 0; the
#: branch has no separate QCG check.  Its comment gives the reason: QSG times
#: MAVAIL is a bad approximation where MAVAIL is very low.  No public WRF
#: carries it.
#:
#: ``wrf``: public WRF v4.6.1 ``:505-514`` (v4.5.2 ``:481`` alike): an
#: invalid QCG takes the lowest-level condensate, an invalid QVG takes
#: saturation at the skin times moisture availability.
#:
#: Generic forecasts default to ``wrf``. The operational namelist importer
#: and configuration recipe explicitly select ``air``. SOILTEMP's energy
#: balance carries the vapour storage of the constant-flux layer from OLD QVG
#: (``xlv*r210*qvg`` with ``r211 = 0.5*conflx/delt``, ten to thirty times
#: the exchange coefficient at a 12 to 20 s step), so the first steps tie
#: the skin to whatever QVG the start supplied.  The branch's fallback is
#: rarely run there (it cycles QVG); every gpuwm forecast is a cold start.
#: MEASURED, the RUC runtime fixture (moist loam, 12 s step): ``air`` puts
#: the first-step skin at 296.3 K against 302.2 K, sensible heat at -105
#: against +58 W/m2, and after 20 steps the skin is still 2.7 K colder.
#: MEASURED, a 3 km cut of a native operational-HRRR start (106,671 land
#: cells): ``air`` moves the 2 m dewpoint by -0.013 / -0.016 K at f01 / f02
#: of a 21Z start and by +0.018 K of an 06Z start against the operational
#: model's own files.
RUC_QVG_COLD_START_FORMS = ("air", "wrf")
RUC_QVG_COLD_START_DEFAULT = "wrf"

#: How SFCDIAGS_RUCLSM writes T2, TH2 and Q2 (``RunConfig.ruc_2m_diagnostic``).
#:
#: ``flux``: public WRF ``module_sf_sfcdiags_ruclsm.F:53-146``, the flux form,
#: and the default.
#:
#: ``log_profile``: adds the block the operational RAP/HRRR branch carries at
#: ``:150-179`` there and no public WRF 3.9 to 4.7.1 has.  With dT = T1 - TSK,
#: dQ = qlev1 - QSFCmr and dz1 = half the lowest layer: where dT > 0,
#: fh = min(max(1 - dT/10, 0.01), 1) and fac = log(2.05/(0.05 + fh)) /
#: log((dz1 + 0.05)/(0.05 + fh)), T2 = TSK + fac*(T1 - TSK) and TH2 from it;
#: where dQ > 0, the same with fh from dQ/0.003, Q2 = QSFCmr + fac*(qlev1 -
#: QSFCmr), with no saturation cap.  Elsewhere the flux values stand.
#:
#: MEASURED on a 3 km cut of a native operational-HRRR start (106,671 land
#: cells), against the operational model's own files: on an 06Z start (97
#: percent of land stable) it lowers T2 by 0.33 to 0.36 K, which moves the
#: mean from -0.03 / +0.03 / +0.04 K to -0.39 / -0.32 / -0.29 K at f01 to
#: f03, and moves the 2 m dewpoint by -0.008 to -0.011 K; on a 21Z start it
#: lowers T2 by 0.06 / 0.14 K and leaves Q2 unchanged.  Either the
#: operational files' 2 m values do not come from this block as ported or
#: the port differs from the branch elsewhere. Generic forecasts keep the
#: flux default; the operational namelist importer and configuration recipe
#: explicitly select log_profile to reproduce the fork diagnostic block.
RUC_2M_DIAGNOSTIC_FORMS = ("flux", "log_profile")
RUC_2M_DIAGNOSTIC_DEFAULT = "flux"

#: The RUC snow scheme's two WRF lineages (``RunConfig.ruc_snow``).
#:
#: ``wrf_45``: WRF v4.0 to v4.5 SFCTMP, SNOWSOIL and SNOWTEMP, which the
#: operational RAP/HRRR branch carries line for line apart from dead code.
#: Constant snow conductivity 0.265 W/m/K (4.6.1's ``isncond_opt = 1``);
#: snow cover ``min(1, snhei/(2*snhei_crit))`` with the critical depth taken
#: after compaction and new-snow density (4.6.1's ``isncovr_opt = 1``
#: formula, at the branch's position), and not rebuilt after the snow
#: column; fresh-snow albedo kept from the depth on the ground; melt cap
#: independent of the step; melt bookkeeping scaled by cover under the
#: snow mosaic; SNOWFALLAC grown by new snow less its melt; and the
#: ``ktau = 1`` snow cover default from snow water.
#:
#: ``wrf_461``: WRF v4.6.1 (``isncond_opt = 2``, ``isncovr_opt = 2``), the
#: form every earlier gpuwm build ran and the WRF v4.6.1 oracle pins.
#:
#: The generic forecast and transcription leaves default to ``wrf_461``.
#: The operational namelist importer and configuration recipe select
#: ``wrf_45`` explicitly. The two differ only where there is snow.
RUC_SNOW_FORMS = ("wrf_45", "wrf_461")
RUC_SNOW_DEFAULT = "wrf_461"


def ruc_snow_form(value) -> str:
    """The snow lineage name, or refuse an unknown one by name."""
    if type(value) is not str or value not in RUC_SNOW_FORMS:
        raise ValueError(
            f"RUC ruc_snow={value!r} must be one of {RUC_SNOW_FORMS}: "
            "'wrf_45' is the WRF v4.0-4.5 snow scheme the operational "
            "RAP/HRRR branch carries and 'wrf_461' the WRF v4.6.1 rewrite; "
            "the two differ in snow conductivity, cover, melt and albedo, so "
            "an unknown name cannot select either")
    return value


def ruc_soilprop_form(value) -> str:
    """The SOILPROP lineage name, or refuse an unknown one by name."""
    if type(value) is not str or value not in RUC_SOILPROP_FORMS:
        raise ValueError(
            f"RUC ruc_soilprop={value!r} must be one of {RUC_SOILPROP_FORMS}: "
            "'wrf_45' normalises soil-water diffusivity by the moisture above "
            "the residual (WRF v4.0-4.5, the operational RAP/HRRR form) and "
            "'wrf_461' by total moisture over porosity (WRF v4.6.1); the two "
            "move different water between soil levels, so an unknown name "
            "cannot select either")
    return value


def ruc_qvg_cold_start_form(value) -> int:
    """The kernel selector (1 for ``air``, 0 for ``wrf``), or refuse."""
    if type(value) is not str or value not in RUC_QVG_COLD_START_FORMS:
        raise ValueError(
            f"RUC ruc_qvg_cold_start={value!r} must be one of "
            f"{RUC_QVG_COLD_START_FORMS}: 'air' starts the ground vapour from "
            "the lowest-level air (the operational RAP/HRRR form) and 'wrf' "
            "from saturation at the skin times moisture availability (public "
            "WRF); the two start different surface humidity, so an unknown "
            "name cannot select either")
    return int(value == "air")


def ruc_2m_diagnostic_form(value) -> int:
    """The kernel selector (1 for ``log_profile``, 0 for ``flux``), or refuse."""
    if type(value) is not str or value not in RUC_2M_DIAGNOSTIC_FORMS:
        raise ValueError(
            f"RUC ruc_2m_diagnostic={value!r} must be one of "
            f"{RUC_2M_DIAGNOSTIC_FORMS}: 'flux' writes T2 and Q2 by public "
            "WRF's flux form and 'log_profile' adds the operational RAP/HRRR "
            "branch's logarithmic profile where the air is warmer or moister "
            "than the surface; the two write different 2 m values, so an "
            "unknown name cannot select either")
    return int(value == "log_profile")


def ruc_module_defines(nzs: int, soilprop: str = RUC_SOILPROP_DEFAULT,
                       snow: str = RUC_SNOW_DEFAULT,
                       ) -> tuple[tuple[str, int], ...]:
    """Integer defines the RUC module compiles with at this soil geometry.

    At nine levels geometry adds no define. Named physics forms still bind
    their defines: wrf_461 snow always compiles GPUWM_SNOW_WRF461, while the
    explicit wrf_45 form leaves it absent. This mapping is independent of the
    generic selector default. Six levels also bind RUC_NZS.
    """
    nzs = int(nzs)
    if nzs not in WRF_SUPPORTED_NUM_SOIL_LAYERS:
        raise ValueError(
            f"RUC soil geometry {nzs} is not one of "
            f"{WRF_SUPPORTED_NUM_SOIL_LAYERS}")
    lineage = (() if ruc_soilprop_form(soilprop) == RUC_SOILPROP_DEFAULT
               else (("GPUWM_SOILPROP_WRF461", 1),))
    lineage += (() if ruc_snow_form(snow) == "wrf_45"
                else (("GPUWM_SNOW_WRF461", 1),))
    if nzs == NUM_SOIL_LAYERS:
        return lineage
    return (("RUC_NZS", nzs),) + lineage


def ruc_kernel(func: str, nzs: int, soilprop: str = RUC_SOILPROP_DEFAULT,
               snow: str = RUC_SNOW_DEFAULT):
    """The RUC kernel ``func`` compiled at this geometry's tier.

    At nine levels the geometry itself adds no define. The generic snow
    form binds GPUWM_SNOW_WRF461; the explicit fork form leaves it absent.
    Each geometry and named lineage has its own compiled module key.
    """
    defines = ruc_module_defines(nzs, soilprop, snow)
    if not defines:
        return get_kernel(RUC_MODULE, func)
    return get_kernel_int_defines(RUC_MODULE, func, defines)


def ruc_kernel_source(nzs: int, soilprop: str = RUC_SOILPROP_DEFAULT,
                      snow: str = RUC_SNOW_DEFAULT) -> str:
    """The exact string NVRTC receives for RUC at ``nzs``.  CPU-only.

    Imports no CuPy and touches no device, so the identity of the nine-level
    translation unit is testable on any box.
    """
    defines = ruc_module_defines(nzs, soilprop, snow)
    if not defines:
        return module_source(RUC_MODULE)
    return module_source_int_defines(RUC_MODULE, defines)


# ---------------------------------------------------------------------------
# The fused RUC translation unit.
# ---------------------------------------------------------------------------

#: The fused column kernels' manifest name.
RUC_FUSED_MODULE = "ruc_fused"

#: The fused kernels' own sources, appended in this order after ``ruc.cu``.
#: They are ``.cuh`` fragments, not ``.cu`` modules: neither compiles alone,
#: because both call the leaf bodies ``ruc.cu`` defines.
RUC_FUSED_SOURCES = ("ruc_fused_sfctmp.cuh", "ruc_fused_driver.cuh")

#: ``ruc.cu``'s leaves are ``extern "C" __global__`` functions whose body
#: finds its column as ``blockIdx.x * blockDim.x + threadIdx.x`` and returns
#: past ``n``.  Compiled once more with ``__global__`` spelled ``__device__``,
#: the SAME text becomes a set of column functions a fused kernel calls from
#: the thread that owns that column, so the leaf arithmetic has one source
#: and ``ruc.cu`` does not change by a byte.  Every FP operation in it is an
#: explicit round-to-nearest intrinsic, so the calling context cannot
#: contract or reorder it.
#:
#: NVRTC does not honour ``#pragma push_macro``/``pop_macro`` (measured: a
#: kernel declared after the pop is still a device function), so the close
#: restores the spelling CUDA's host_defines.h gives ``__global__``.
_RUC_AS_DEVICE_OPEN = (
    "#undef __global__\n"
    "#define __global__ __device__\n")
_RUC_AS_DEVICE_CLOSE = (
    "\n#undef __global__\n"
    "#define __global__ __location__(global)\n")


def ruc_fused_source(nzs: int, *, kernel_dir=None,
                     soilprop: str = RUC_SOILPROP_DEFAULT,
                     snow: str = RUC_SNOW_DEFAULT) -> str:
    """The exact string NVRTC receives for the fused RUC unit.  CPU-only.

    The preamble, then the tier define exactly where
    :func:`gpuwm.core.kernels.module_source_int_defines` places it for
    ``ruc.cu`` (before its text, so the ladder sees it), then ``ruc.cu``
    with ``__global__`` read as ``__device__``, then the fused sources.
    ``kernel_dir`` composes it from another tree's kernel files (the A146
    census gates the tree it scans, A193).
    """
    from pathlib import Path

    from gpuwm.core.kernels import _ENCODING, _KDIR, _preamble

    kdir = _KDIR if kernel_dir is None else Path(kernel_dir)
    defines = ruc_module_defines(nzs, soilprop, snow)
    prefix = "".join(f"#define {key} {value}\n" for key, value in defines)
    parts = [_preamble(kdir), prefix,
             (kdir / "glibc_flt32.cuh").read_text(encoding=_ENCODING),
             (kdir / "glibc_trig_flt32.cuh").read_text(encoding=_ENCODING),
             _RUC_AS_DEVICE_OPEN,
             (kdir / f"{RUC_MODULE}.cu").read_text(encoding=_ENCODING),
             _RUC_AS_DEVICE_CLOSE]
    parts += [(kdir / name).read_text(encoding=_ENCODING)
              for name in RUC_FUSED_SOURCES]
    return "".join(parts)


def _ruc_fused_module_key(nzs: int, soilprop: str = RUC_SOILPROP_DEFAULT,
                          snow: str = RUC_SNOW_DEFAULT) -> str:
    from gpuwm.core.kernels import MODULE_KEY_ROOT

    defines = ruc_module_defines(nzs, soilprop, snow)
    key = f"{MODULE_KEY_ROOT}:{RUC_FUSED_MODULE}"
    if defines:
        key += "[" + ",".join(f"{k}={v}" for k, v in defines) + "]"
    return key


_RUC_FUSED_MODULES: dict[tuple[int, int, str, str], object] = {}


def ruc_fused_kernel(func: str, nzs: int, soilprop: str = RUC_SOILPROP_DEFAULT,
                     snow: str = RUC_SNOW_DEFAULT):
    """A kernel of the fused RUC unit at this soil geometry.

    One compile per geometry per process, recorded in the kernel manifest
    under its own key like every other translation unit.
    """
    import cupy as cp

    nzs = int(nzs)
    soilprop = ruc_soilprop_form(soilprop)
    snow = ruc_snow_form(snow)
    owner = (int(cp.cuda.Device().id), nzs, soilprop, snow)
    module = _RUC_FUSED_MODULES.get(owner)
    if module is None:
        import cupy as cp

        from gpuwm.certify.kernel_manifest import record_module
        from gpuwm.core.kernels import _compile_observed

        source = ruc_fused_source(nzs, soilprop=soilprop, snow=snow)
        key = _ruc_fused_module_key(nzs, soilprop, snow)
        module = cp.RawModule(code=source, options=("-std=c++17",),
                              name_expressions=None)
        _compile_observed(module, key)
        record_module(key, source=source, options=("-std=c++17",),
                      module=module)
        _RUC_FUSED_MODULES[owner] = module
    return module.get_function(func)
