"""Full-state model health validation for synchronized run boundaries.

The production path uses exactly one descriptor-driven CUDA kernel launch.
Every descriptor supplies a field pointer, bounds, optional auxiliary array,
and a stable status-class bit.  The kernel ORs every failing class and records
the lexicographically first ``(field, flat-index)`` pair.  Only that compact
record is copied to the host.  No value is clipped or repaired: a failed gate
is terminal for the worker.

The NumPy mirror below deliberately shares the descriptor/rule construction
with the CUDA path.  It is used by CPU corruption tests and by host-side LBC
or nest-table construction checks before those tables are uploaded.
"""

from __future__ import annotations

import dataclasses
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


# A four-domain NSSL-2 step currently reaches 527 descriptors after its lazy
# persistent microphysics and nest scratch is materialized.  Keep a power-of-
# two ceiling with substantial headroom so scheduled physics can add state
# without invalidating a run after it has already advanced.
MAX_HEALTH_FIELDS = 1024
_INDEX_MASK = (1 << 48) - 1

# Stable status classes.  The individual failing field remains available in
# ValidationReport.first_bad_field; these bits make the device record compact
# while still reporting every affected physical/storage class in one pass.
STATUS_BITS = {
    "wind": 1 << 0,
    "geopotential": 1 << 1,
    "coupled_mass": 1 << 2,
    "pressure": 1 << 3,
    "specific_volume": 1 << 4,
    "theta": 1 << 5,
    "moisture": 1 << 6,
    "moment": 1 << 7,
    "effective_radius": 1 << 8,
    "held_tendency": 1 << 9,
    "lateral_boundary": 1 << 10,
    "nest_table": 1 << 11,
    "soil_temperature": 1 << 12,
    "soil_moisture": 1 << 13,
    "surface": 1 << 14,
    "generic": 1 << 15,
    "particle_volume": 1 << 16,
}

_LOWER = 1 << 0
_UPPER = 1 << 1
_STRICT_LOWER = 1 << 2
_ADD_AUX = 1 << 3
_AUX_LEVEL = 1 << 4
_INT32_STORAGE = 1 << 5

# Chunk sizing for the fused gate.  One block per descriptor left the whole
# check waiting on the single largest field while the rest of the device sat
# idle, so the launch also spreads each descriptor over ``ceil(size / chunk)``
# blocks.  The chunk is driven by the TOTAL element count, targeting roughly
# _CHUNK_TARGET_BLOCKS blocks over the whole inventory, so one huge descriptor
# cannot explode the count of blocks that exit immediately on the short ones.
# The measured time is flat over chunks of 8192-65536 elements, so the floor
# only has to keep tiny inventories from launching a block per 256 elements.
_CHUNK_TARGET_BLOCKS = 8192
_MIN_CHUNK_ELEMENTS = 4096

# These are immutable setup-time category maps, not floating model state.
# They cannot contain NaN/Inf, and their legal ranges depend on the selected
# Noah vegetation/soil tables (which validate them at initialization).  Keep
# them in collect_state_fields so the real inventory and dtype remain pinned,
# but deliberately exclude only these exact names from the per-step GPU gate.
# Every other non-float32 field fails descriptor construction unless it has an
# explicit range policy below.
GPU_INTEGER_EXCLUSIONS = frozenset({
    "surface.ivgtyp",
    "surface.isltyp",
    # SINT geometry index maps (T10/F16 device tables): built FP64-on-host,
    # range-validated against the REAL parent extents by nest_interp's
    # _check_table at registration, stored int32, and immutable thereafter
    # (the kernels consume them read-only; no step writes them).  Their
    # legal ranges depend on per-domain parent extents, which a fixed
    # FieldRule cannot express -- same treatment as the Noah category
    # fields above.  One entry per stagger class (mass/x-stag/y-stag).
    "nest.scratch.nest_sint_ci_m", "nest.scratch.nest_sint_ip_m",
    "nest.scratch.nest_sint_cj_m", "nest.scratch.nest_sint_jp_m",
    "nest.scratch.nest_sint_ci_x", "nest.scratch.nest_sint_ip_x",
    "nest.scratch.nest_sint_cj_x", "nest.scratch.nest_sint_jp_x",
    "nest.scratch.nest_sint_ci_y", "nest.scratch.nest_sint_ip_y",
    "nest.scratch.nest_sint_cj_y", "nest.scratch.nest_sint_jp_y",
})
_GPU_RANGE_CHECKED_INT32 = frozenset({
    # Noah writes one of four energy-balance solution cases.
    "surface.ebal",
    # YSU writes a one-based model level; zero is the pre-first-call value.
    "surface.kpbl",
    # MYNN's EDMF plume top, also a one-based model level, but capped one
    # level lower than kpbl: module_bl_mynn.F:6392 ends DMP_mf with the
    # unconditional ``ktop=MIN(ktop,KTE-1)`` on every exit path, and the
    # only place the value grows is ``ktop=MAX(ktop,k)`` (:6362) inside
    # ``do k=kts+1,kte-1``.  Zero means "no plume" -- DMP_mf sets it at
    # :6033 before the ascent, mynn_bl_driver sets the whole row at :646
    # ahead of the initflag block, so it is also the pre-first-call value
    # and the value for bl_mynn_edmf=0.  collect_state_fields supplies the
    # state-dependent nz-1 cap.
    "surface.ktop_plume",
    # Noah-MP's top snow layer index: minus the number of ACTIVE snow layers,
    # so it is negative-or-zero and a naive ">= 0" bound would fire on the
    # first snowpack column.  The range is [-NSNOW, 0], NSNOW fixed at 3 by
    # module_sf_noahmpdrv.F:628.  Every writer in the pinned tree stays inside
    # it: SNOW_INIT (:2382-2402) assigns 0, -1, -2 or -3 and nothing else;
    # SNOWFALL sets -1 (module_sf_noahmplsm.F:6589); COMBINE only ever does
    # ISNOW = ISNOW + 1 (:6697, :6776) or ISNOW = 0 (:6727), both toward zero;
    # and DIVIDE closes with ISNOW = -MSNO (:6902) where MSNO reaches 3 at
    # most.  gpuwm's port carries the identical statements
    # (noahmp_driver.py:260-279, noahmp_snow.py:234, :364, :387, :428, :519).
    # Not nz-dependent, so no state-dependent cap is needed.
    "surface.isnowxy",
    # Noah-MP's plant growth stage.  GROWING_GDD
    # (module_sf_noahmplsm.F:10783-10797) is the only writer and assigns 1..8
    # and nothing else; 0 is the Registry cold value and, under the admitted
    # opt_crop=0, the value forever, because CARBON_CROP never runs and
    # module_sf_noahmpdrv.F:1397 writes the unchanged INOUT back.  So [0, 8]
    # bounds both the crop and the no-crop case.
    "surface.pgsxy",
})

#: ``NSNOW``, WRF's fixed snow-layer count (``module_sf_noahmpdrv.F:628``), and
#: therefore the magnitude of the most negative legal ``ISNOWXY``.  Spelled
#: here rather than imported so this module keeps no dependency on a physics
#: runner; ``tests/test_health_integer_policy.py`` asserts it equals
#: ``gpuwm.core.noahmp_runtime.NSNOW`` so the two cannot drift apart.
NOAHMP_SNOW_LAYERS = 3

#: The largest plant growth stage ``GROWING_GDD`` assigns.
NOAHMP_MAX_GROWTH_STAGE = 8


def gpu_integer_policy(name: str, dtype: Any) -> int | None:
    """Device-gate storage policy for one collected field.

    Returns the descriptor storage flags for a field the gate checks, or
    ``None`` for a field carrying a documented exclusion.  Raises
    :class:`TypeError` for a dtype with no declared policy, which is the
    point of the function: an integer field added to any physics driver
    must be classified deliberately before a forecast can run, rather than
    silently admitted to a float32 gate.

    Module level rather than inline in :meth:`StateHealthValidator._refresh`
    so that the host-side census, the CPU tests and the device inventory all
    classify through this one implementation instead of three spellings of
    it.  A CPU test can therefore falsify the real gate without a device.
    """
    dtype = np.dtype(dtype)
    if dtype == np.dtype(np.float32):
        return 0
    if dtype == np.dtype(np.int32):
        if name in GPU_INTEGER_EXCLUSIONS:
            return None
        if name in _GPU_RANGE_CHECKED_INT32:
            return _INT32_STORAGE
    raise TypeError(
        f"GPU health field {name!r} has unsupported dtype "
        f"{dtype}; add an explicit integer range policy or a "
        "documented exclusion before this inventory can run")


@dataclass(frozen=True)
class FieldRule:
    """Bounds and status class for one health field."""

    status_class: str
    lower: float | None = None
    upper: float | None = None
    strict_lower: bool = False

    @property
    def status_bit(self) -> int:
        return STATUS_BITS[self.status_class]


@dataclass(frozen=True)
class HealthField:
    """One array plus optional auxiliary values used by a derived check.

    ``aux_mode='direct'`` checks ``values + auxiliary`` elementwise.
    ``aux_mode='level'`` broadcasts a one-dimensional auxiliary profile over
    each horizontal plane (the theta base profile case).
    """

    name: str
    values: Any
    rule: FieldRule
    auxiliary: Any | None = None
    aux_mode: str | None = None
    plane_size: int = 0


@dataclass(frozen=True)
class ValidationReport:
    """Compact result of one full-state validation pass."""

    ok: bool
    status_bits: int
    first_bad_field: str | None = None
    first_bad_index: tuple[int, ...] | None = None
    first_bad_flat_index: int | None = None
    first_bad_value: float | None = None
    reason: str | None = None
    phase: str | None = None

    @property
    def failing_classes(self) -> tuple[str, ...]:
        return tuple(name for name, bit in STATUS_BITS.items()
                     if self.status_bits & bit)


class HealthCheckError(FloatingPointError):
    """A synchronized model-state health gate failed."""

    def __init__(self, report: ValidationReport):
        self.report = report
        location = ("unknown" if report.first_bad_index is None
                    else str(report.first_bad_index))
        phase = "" if report.phase is None else f" during {report.phase}"
        super().__init__(
            f"full-state health gate failed{phase}: "
            f"{report.first_bad_field}{location}: {report.reason}; "
            f"classes={report.failing_classes}")


_FINITE = FieldRule("generic")


def _leaf(name: str) -> str:
    return name.lower().replace("[", ".").replace("]", "").split(".")[-1]


def rule_for_field(name: str) -> FieldRule:
    """Return the shared CPU/CUDA rule for a named model field."""
    lower_name = name.lower()
    leaf = _leaf(name)
    if lower_name.startswith("lbc."):
        return FieldRule("lateral_boundary")
    if lower_name.startswith("nest."):
        return FieldRule("nest_table")
    if lower_name.startswith("held.") or leaf == "h_diabatic":
        return FieldRule("held_tendency")
    if leaf in ("u", "v"):
        return FieldRule("wind", -500.0, 500.0)
    if leaf == "w":
        return FieldRule("wind", -200.0, 200.0)
    if leaf == "thp":
        # State collection supplies total theta through the base-state
        # auxiliary descriptor; the perturbation itself is never bounded.
        # 600 K intentionally admits the legal ~495 K WK82 20-km sounding
        # plus a useful perturbation margin while still catching runaway
        # thermodynamic state.  The previous 500 K cap could reject a legal
        # idealized state even though production currently installs this gate
        # only on real-case runs.
        return FieldRule("theta", 100.0, 600.0)
    if leaf == "php":
        return FieldRule("geopotential", -1.0e8, 1.0e8)
    if leaf == "mup":
        # State collection adds MUB2D, checking the coupled denominator.
        return FieldRule("coupled_mass", 0.0, 2.0e6, strict_lower=True)
    if leaf == "p":
        return FieldRule("pressure", 0.0, 2.0e6, strict_lower=True)
    if leaf == "al":
        # AL is the perturbation specific volume and may be negative.
        return FieldRule("specific_volume", -1.0e4, 1.0e4)
    if leaf == "alt":
        return FieldRule("specific_volume", 0.0, 1.0e4,
                         strict_lower=True)
    if leaf in ("qv", "qc", "qr", "qi", "qs", "qg", "qh"):
        return FieldRule("moisture", 0.0, 1.0)
    if leaf in ("nc", "nr", "ni", "ns", "ng", "qndrop", "qnr", "qni",
                "qns", "qng", "qnh", "qnn",
                # mp=28 aerosol number tracers, per kilogram.  WRF's own
                # terminal clamp holds them inside [11.1E6, 9999.E6] and
                # [5.0E3, 9999.E6] (module_mp_thompson.F:3977-3982), and
                # the deliberately unclamped surface emission at :1310-1327
                # can push the lowest level above the ceiling between the
                # clamp and the next call's entry pack.  The existing
                # 1.0e15 "moment" ceiling is four decades of headroom over
                # that, so this rule catches a blow-up without pretending
                # to enforce the scheme's own bounds.
                "nwfa", "nifa"):
        return FieldRule("moment", 0.0, 1.0e15)
    if leaf in ("qvolg", "qvolh"):
        return FieldRule("particle_volume", 0.0, 1.0)
    if leaf in ("effc", "effr", "effi", "effs"):
        # Morrison stores these diagnostics in microns.
        return FieldRule("effective_radius", 0.0, 1.0e6)
    if leaf in ("tsk", "tslb"):
        return FieldRule("soil_temperature", 150.0, 400.0)
    if leaf in ("smois", "sh2o"):
        return FieldRule("soil_moisture", 0.0, 1.0)
    if lower_name.startswith("surface."):
        if leaf == "ebal":
            return FieldRule("surface", 0.0, 3.0)
        if leaf == "kpbl":
            # collect_state_fields supplies the state-dependent nz cap.
            return FieldRule("surface", 0.0)
        if leaf == "ktop_plume":
            # collect_state_fields supplies the state-dependent nz-1 cap.
            return FieldRule("surface", 0.0)
        if leaf == "isnowxy":
            # Noah-MP counts ACTIVE snow layers downward from the surface, so
            # the index is negative: 0 is snow-free and -NSNOW is a full pack.
            # Fixed by module_sf_noahmpdrv.F:628, not by the domain, so this
            # needs no state-dependent cap.  See _GPU_RANGE_CHECKED_INT32.
            return FieldRule("surface", float(-NOAHMP_SNOW_LAYERS), 0.0)
        if leaf == "pgsxy":
            return FieldRule("surface", 0.0,
                             float(NOAHMP_MAX_GROWTH_STAGE))
        return FieldRule("surface")
    return _FINITE


def field_from_array(name: str, values: Any, *, auxiliary: Any | None = None,
                     aux_mode: str | None = None,
                     plane_size: int = 0) -> HealthField:
    """Build one descriptor using :func:`rule_for_field`."""
    return HealthField(name, values, rule_for_field(name), auxiliary,
                       aux_mode, int(plane_size))


def _is_array(value: Any) -> bool:
    return (hasattr(value, "shape") and hasattr(value, "dtype")
            and hasattr(value, "size"))


def _walk_arrays(value: Any, prefix: str, *, seen: set[int] | None = None,
                 depth: int = 0) -> list[HealthField]:
    """Recursively collect arrays from one explicitly selected container."""
    if seen is None:
        seen = set()
    if value is None or depth > 8:
        return []
    if _is_array(value):
        return [field_from_array(prefix, value)]
    identity = id(value)
    if identity in seen:
        return []
    seen.add(identity)
    result: list[HealthField] = []
    if isinstance(value, Mapping):
        items = sorted(value.items(), key=lambda item: str(item[0]))
    elif dataclasses.is_dataclass(value):
        items = [(field.name, getattr(value, field.name))
                 for field in dataclasses.fields(value)]
    elif isinstance(value, Sequence) and not isinstance(
            value, (str, bytes, bytearray)):
        items = list(enumerate(value))
    elif hasattr(value, "__dict__"):
        items = sorted(vars(value).items())
    else:
        return result
    for key, child in items:
        if isinstance(child, (str, bytes, int, float, bool, type(None))):
            continue
        result.extend(_walk_arrays(
            child, f"{prefix}.{key}", seen=seen, depth=depth + 1))
    return result


def collect_state_fields(state: Any, *, backend: str = "cpu",
                         extra_tables: Mapping[str, Any] | None = None
                         ) -> tuple[HealthField, ...]:
    """Collect the mutable synchronized state covered by the health gate.

    The explicit top-level list follows the restart manifest's cross-step
    state.  Physics-held arrays, surface/soil fields, resident LBC tables,
    and any nest-owned containers are added separately.  ``extra_tables``
    is the integration surface for multi-domain donor/receiver tables.
    """
    if backend not in ("cpu", "gpu"):
        raise ValueError("backend must be 'cpu' or 'gpu'")
    result: list[HealthField] = []
    for name in (
            "u", "v", "w", "thp", "php", "mup", "p", "al", "alt",
            "qv", "qc", "qr", "qi", "qs", "qg", "nc", "nr", "ni",
            "ns", "ng", "nwfa", "nifa",
            "qh", "qndrop", "qnr", "qni", "qns", "qng",
            "qnh", "qnn", "qvolg", "qvolh", "effc", "effr", "effi",
            "effs", "h_diabatic"):
        value = getattr(state, name, None)
        if value is None:
            continue
        if name == "mup" and getattr(state, "mub2d", None) is not None:
            result.append(field_from_array(
                name, value, auxiliary=state.mub2d, aux_mode="direct"))
        elif name == "thp" and getattr(state, "thb", None) is not None:
            thb = state.thb
            if getattr(thb, "ndim", 0) == 1:
                plane = int(np.prod(value.shape[1:], dtype=np.int64))
                result.append(field_from_array(
                    name, value, auxiliary=thb, aux_mode="level",
                    plane_size=plane))
            else:
                result.append(field_from_array(
                    name, value, auxiliary=thb, aux_mode="direct"))
        else:
            result.append(field_from_array(name, value))

    driver = getattr(state, "physics", None)
    if driver is not None:
        for tendency_name in (
                "pbl_tendencies", "radiation_tendencies",
                "cumulus_tendencies"):
            tendency = getattr(driver, tendency_name, None)
            result.extend(_walk_arrays(
                tendency, f"held.{tendency_name.removesuffix('_tendencies')}"))
        for name in (
                "rthratenlw", "rthratensw", "cu_nca", "cu_pratec",
                "cu_raincv", "cu_rates", "_pending_rainbl"):
            value = getattr(driver, name, None)
            if value is not None:
                result.extend(_walk_arrays(value, f"held.{name}"))
        surface_fields = _walk_arrays(getattr(driver, "fields", None),
                                      "surface")
        nz = int(getattr(getattr(state, "p", None), "shape", (0,))[0])
        for field in surface_fields:
            if field.name == "surface.kpbl":
                field = dataclasses.replace(
                    field, rule=FieldRule("surface", 0.0, float(nz)))
            elif field.name == "surface.ktop_plume":
                # MYNN's plume top sits one level BELOW kpbl's ceiling:
                # module_bl_mynn.F:6392 closes DMP_mf with
                # ``ktop=MIN(ktop,KTE-1)`` unconditionally, and gpuwm's port
                # carries the same clamp (kernels/mynn_pbl.cu:3073,
                # mynn_pbl.py:2979).  The ``max(..., 0)`` matters only for a
                # state that carries no pressure array, where nz is 0 and the
                # sole legal value is the never-written 0.
                field = dataclasses.replace(
                    field, rule=FieldRule("surface", 0.0,
                                          float(max(nz - 1, 0))))
            result.append(field)
        result.extend(_walk_arrays(getattr(driver, "microphysics", None),
                                   "surface.microphysics"))

    # Restart-persistent scratch is mutable cross-step state too.  Keep this
    # explicit (matching restart.py's fail-loud philosophy) so transient RK
    # work buffers are not mistaken for synchronized state.
    persistent_scratch = {
        "mp_rainnc", "mp_rainncv", "mp_snownc", "mp_snowncv",
        "mp_graupelnc", "mp_graupelncv", "mp_hailnc", "mp_hailncv",
        "mp_sr", "mp_kessler_sr",
        "cu_rainc", "cu_nca", "cu_pratec", "cu_raincv",
        "cu_rthcuten", "cu_rqvcuten", "cu_rqccuten", "cu_rqicuten",
        "cu_rqrcuten", "cu_rqscuten",
    }
    for slot, value in sorted(getattr(state, "_scratch", {}).items()):
        if slot in persistent_scratch:
            prefix = ("held.scratch" if slot.startswith("cu_")
                      else "surface.microphysics.scratch")
            result.append(field_from_array(f"{prefix}.{slot}", value))

    if backend == "gpu":
        # Current LBC upload deliberately packs every interval/field/side
        # value and tendency into one resident scratch allocation.  Validate
        # that allocation once instead of emitting ~100 overlapping views.
        scratch = getattr(state, "_scratch", {})
        packed_lbc = scratch.get("lbc_forcing_tables")
        if packed_lbc is not None:
            result.append(field_from_array("lbc.forcing_tables", packed_lbc))
            for slot, value in sorted(scratch.items()):
                if slot.startswith("lbc_weights_"):
                    result.append(field_from_array(f"lbc.{slot}", value))
        else:
            result.extend(_walk_arrays(
                getattr(state, "_lateral_boundary_device", None), "lbc"))
    else:
        result.extend(_walk_arrays(
            getattr(state, "lateral_boundaries", None), "lbc"))

    # Task-13/14 may attach nest tables under different owner-selected names.
    # Selecting only names containing ``nest`` avoids scanning transient RK
    # scratch while still making the health interface future-proof.
    for name, value in sorted(vars(state).items()):
        if "nest" in name.lower() and value is not None:
            result.extend(_walk_arrays(value, f"nest.{name}"))
    for slot, value in sorted(getattr(state, "_scratch", {}).items()):
        if slot.startswith("nest_"):
            result.append(field_from_array(f"nest.scratch.{slot}", value))
    if extra_tables:
        for name, value in sorted(extra_tables.items()):
            result.extend(_walk_arrays(value, f"nest.{name}"))

    if len(result) > MAX_HEALTH_FIELDS:
        raise ValueError(
            f"health field inventory has {len(result)} descriptors; "
            f"MAX_HEALTH_FIELDS={MAX_HEALTH_FIELDS} must be raised explicitly")
    return tuple(result)


def _cpu_values(field: HealthField) -> np.ndarray:
    values = np.asarray(field.values)
    if field.auxiliary is None:
        return values
    auxiliary = np.asarray(field.auxiliary)
    if field.aux_mode == "direct":
        if auxiliary.shape != values.shape:
            raise ValueError(
                f"{field.name} auxiliary shape {auxiliary.shape} != "
                f"field shape {values.shape}")
        return values + auxiliary
    if field.aux_mode == "level":
        if auxiliary.ndim != 1 or values.shape[0] != auxiliary.size:
            raise ValueError(
                f"{field.name} level auxiliary shape {auxiliary.shape} is "
                f"incompatible with {values.shape}")
        return values + auxiliary.reshape((-1,) + (1,) * (values.ndim - 1))
    raise ValueError(f"unknown auxiliary mode {field.aux_mode!r}")


def _bad_mask(values: np.ndarray, rule: FieldRule) -> np.ndarray:
    bad = ~np.isfinite(values)
    if rule.lower is not None:
        bad |= values <= rule.lower if rule.strict_lower else values < rule.lower
    if rule.upper is not None:
        bad |= values > rule.upper
    return bad


def _reason(value: float, rule: FieldRule) -> str:
    if not math.isfinite(value):
        return f"non-finite value {value!r}"
    if rule.lower is not None and (
            value <= rule.lower if rule.strict_lower else value < rule.lower):
        op = ">" if rule.strict_lower else ">="
        return f"value {value!r} violates lower bound {op} {rule.lower!r}"
    if rule.upper is not None and value > rule.upper:
        return f"value {value!r} exceeds upper bound {rule.upper!r}"
    return "unknown invariant failure"


def validate_fields_cpu(fields: Mapping[str, Any] | Sequence[HealthField],
                        *, phase: str | None = None) -> ValidationReport:
    """Validate host arrays with the exact production rule descriptors."""
    if isinstance(fields, Mapping):
        descriptors = tuple(field_from_array(name, value)
                            for name, value in fields.items())
    else:
        descriptors = tuple(fields)
    status = 0
    first: tuple[int, int, np.ndarray, HealthField] | None = None
    for field_number, field in enumerate(descriptors):
        values = _cpu_values(field)
        bad = _bad_mask(values, field.rule)
        if bool(np.any(bad)):
            status |= field.rule.status_bit
            flat = int(np.flatnonzero(np.ravel(bad, order="C"))[0])
            if first is None:
                first = field_number, flat, values, field
    if first is None:
        return ValidationReport(True, status, phase=phase)
    _, flat, values, field = first
    value = float(np.ravel(values, order="C")[flat])
    return ValidationReport(
        False, status, field.name,
        tuple(int(i) for i in np.unravel_index(flat, values.shape)),
        flat, value, _reason(value, field.rule), phase)


def validate_state_cpu(state: Any, *, phase: str | None = None,
                       extra_tables: Mapping[str, Any] | None = None,
                       raise_on_error: bool = False) -> ValidationReport:
    """CPU mirror for a state-like object (used by corruption fixtures)."""
    report = validate_fields_cpu(
        collect_state_fields(state, backend="cpu", extra_tables=extra_tables),
        phase=phase)
    if raise_on_error and not report.ok:
        raise HealthCheckError(report)
    return report


def cuda_source() -> str:
    """Complete NVRTC source for offline compilation (no device needed)."""
    from gpuwm.core.kernels import _preamble
    path = Path(__file__).with_name("kernels") / "health.cu"
    return _preamble() + path.read_text(encoding="utf-8")


def _u64_words(values: Sequence[int]) -> np.ndarray:
    return np.asarray(values, dtype=np.uint64).view(np.uint32)


def _word_floats(words: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(words, dtype=np.uint32).view(np.float32)


def _cuda_pointer(value: Any, name: str) -> int:
    interface = getattr(value, "__cuda_array_interface__", None)
    if interface is None:
        raise TypeError(f"GPU health field {name!r} is not a CUDA array")
    if not bool(value.flags.c_contiguous):
        raise TypeError(f"GPU health field {name!r} must be C-contiguous")
    return int(value.data.ptr)


class StateHealthValidator:
    """Reusable one-launch CUDA validator for one prepared DomainState."""

    def __init__(self, state: Any, *, extra_tables: Mapping[str, Any] | None = None):
        self.state = state
        self.extra_tables = extra_tables
        # The restart classifier already treats integration_health_* scratch
        # as rebuilt.  Metadata is bit-packed into FP32 storage because the
        # state scratch API intentionally has one model dtype.
        self._ptr = state.scratch((MAX_HEALTH_FIELDS * 2,),
                                  "integration_health_field_ptr")
        self._aux = state.scratch((MAX_HEALTH_FIELDS * 2,),
                                  "integration_health_aux_ptr")
        self._size = state.scratch((MAX_HEALTH_FIELDS * 2,),
                                   "integration_health_field_size")
        self._bounds = state.scratch((MAX_HEALTH_FIELDS, 2),
                                     "integration_health_bounds")
        self._flags = state.scratch((MAX_HEALTH_FIELDS,),
                                    "integration_health_flags")
        self._planes = state.scratch((MAX_HEALTH_FIELDS,),
                                     "integration_health_planes")
        self._status = state.scratch((MAX_HEALTH_FIELDS * 2,),
                                     "integration_health_status_bits")
        self._result = state.scratch((4,), "integration_health_validation")
        self._chunk = _MIN_CHUNK_ELEMENTS
        self._chunk_count = 1
        self.fields: tuple[HealthField, ...] = ()
        self.excluded_integer_fields: tuple[HealthField, ...] = ()

    def _refresh(self) -> None:
        collected = collect_state_fields(
            self.state, backend="gpu", extra_tables=self.extra_tables)
        fields: list[HealthField] = []
        excluded: list[HealthField] = []
        storage_flags: list[int] = []
        pointers_by_field: list[int] = []
        seen: set[tuple] = set()
        for field in collected:
            flag = gpu_integer_policy(field.name, field.values.dtype)
            if flag is None:
                excluded.append(field)
                continue
            pointer = _cuda_pointer(field.values, field.name)
            # collect_state_fields reaches the cumulus-tendency and
            # microphysics-accumulator buffers twice, once as a driver
            # attribute and once as restart-persistent scratch.  A second
            # descriptor over the identical allocation, extent and rule reads
            # the same bytes to the same verdict, and atomicMin already keeps
            # the first registration, so scanning it again buys nothing.  The
            # auxiliary enters the key by object identity rather than by base
            # pointer: anything short of provably the same scan is kept.
            key = (pointer, int(field.values.size), flag, field.rule,
                   id(field.auxiliary), field.aux_mode, field.plane_size)
            if key in seen:
                continue
            seen.add(key)
            fields.append(field)
            storage_flags.append(flag)
            pointers_by_field.append(pointer)
        fields_tuple = tuple(fields)
        pointers = np.zeros(MAX_HEALTH_FIELDS, dtype=np.uint64)
        auxiliaries = np.zeros(MAX_HEALTH_FIELDS, dtype=np.uint64)
        sizes = np.zeros(MAX_HEALTH_FIELDS, dtype=np.uint64)
        bounds = np.zeros((MAX_HEALTH_FIELDS, 2), dtype=np.float32)
        flags = np.zeros(MAX_HEALTH_FIELDS, dtype=np.uint32)
        planes = np.zeros(MAX_HEALTH_FIELDS, dtype=np.uint32)
        bits = np.zeros(MAX_HEALTH_FIELDS, dtype=np.uint64)
        for index, field in enumerate(fields):
            pointers[index] = pointers_by_field[index]
            sizes[index] = int(field.values.size)
            rule = field.rule
            flags[index] |= storage_flags[index]
            if rule.lower is not None:
                flags[index] |= _LOWER
                bounds[index, 0] = np.float32(rule.lower)
            if rule.upper is not None:
                flags[index] |= _UPPER
                bounds[index, 1] = np.float32(rule.upper)
            if rule.strict_lower:
                flags[index] |= _STRICT_LOWER
            if field.auxiliary is not None:
                if np.dtype(field.auxiliary.dtype) != np.dtype(np.float32):
                    raise TypeError(
                        f"GPU health auxiliary for {field.name!r} must be "
                        "float32")
                auxiliaries[index] = _cuda_pointer(
                    field.auxiliary, field.name + " auxiliary")
                flags[index] |= _ADD_AUX
                if field.aux_mode == "level":
                    flags[index] |= _AUX_LEVEL
                    planes[index] = field.plane_size
                elif field.aux_mode != "direct":
                    raise ValueError(
                        f"unknown auxiliary mode {field.aux_mode!r}")
            bits[index] = rule.status_bit
        self._ptr.set(_word_floats(_u64_words(pointers)))
        self._aux.set(_word_floats(_u64_words(auxiliaries)))
        self._size.set(_word_floats(_u64_words(sizes)))
        self._bounds.set(bounds)
        self._flags.set(_word_floats(flags))
        self._planes.set(_word_floats(planes))
        self._status.set(_word_floats(_u64_words(bits)))
        initial = np.asarray([0, (1 << 64) - 1], dtype=np.uint64)
        self._result.set(_word_floats(_u64_words(initial)))
        scanned = sizes[:len(fields)]
        self._chunk = max(_MIN_CHUNK_ELEMENTS,
                          int(scanned.sum()) // _CHUNK_TARGET_BLOCKS)
        largest = int(scanned.max(initial=0))
        self._chunk_count = max(1, (largest + self._chunk - 1) // self._chunk)
        self.fields = fields_tuple
        self.excluded_integer_fields = tuple(excluded)

    def validate(self, *, phase: str | None = None) -> ValidationReport:
        """Launch the fused gate once and return its compact host result."""
        import cupy as cp
        from gpuwm.core.kernels import get_kernel

        self._refresh()
        nfield = len(self.fields)
        if not nfield:
            return ValidationReport(True, 0, phase=phase)
        kernel = get_kernel("health", "validate_full_state")
        kernel((self._chunk_count, nfield), (256,), (
            self._ptr, self._aux, self._size, self._bounds, self._flags,
            self._planes, self._status, self._result, np.int32(nfield),
            np.uint64(self._chunk)))
        words = cp.asnumpy(self._result).view(np.uint64)
        status, packed = (int(words[0]), int(words[1]))
        if packed == (1 << 64) - 1:
            return ValidationReport(True, status, phase=phase)
        field_number = packed >> 48
        flat = packed & _INDEX_MASK
        if field_number >= nfield:
            raise RuntimeError(
                f"health kernel returned invalid field id {field_number}")
        field = self.fields[field_number]
        value = float(cp.asnumpy(field.values.reshape(-1)[flat]))
        if field.auxiliary is not None:
            if field.aux_mode == "direct":
                value += float(cp.asnumpy(
                    field.auxiliary.reshape(-1)[flat]))
            else:
                value += float(cp.asnumpy(
                    field.auxiliary.reshape(-1)[flat // field.plane_size]))
        return ValidationReport(
            False, status, field.name,
            tuple(int(i) for i in np.unravel_index(
                flat, tuple(field.values.shape))),
            flat, value, _reason(value, field.rule), phase)

    def require_healthy(self, *, phase: str | None = None
                        ) -> ValidationReport:
        report = self.validate(phase=phase)
        if not report.ok:
            raise HealthCheckError(report)
        return report

    def phase_observer(self, phase: str) -> ValidationReport:
        """Dycore-compatible debug observer: validate at a named phase."""
        return self.require_healthy(phase=phase)


def benchmark_validator(state: Any, *, repeats: int = 20,
                        step_wall_seconds: float | None = None,
                        extra_tables: Mapping[str, Any] | None = None
                        ) -> dict[str, float | int | None]:
    """Controller-only GPU pre-flight hook for the <=2% overhead gate.

    This function executes CUDA and is intentionally never called by CPU
    tests or lane workers.  The controller supplies the measured d01 step
    wall time to obtain ``overhead_percent``.
    """
    if repeats < 1:
        raise ValueError("repeats must be positive")
    import cupy as cp

    validator = StateHealthValidator(state, extra_tables=extra_tables)
    validator.require_healthy(phase="validator-benchmark-warmup")
    cp.cuda.runtime.deviceSynchronize()
    started = time.perf_counter()
    for _ in range(repeats):
        validator.require_healthy(phase="validator-benchmark")
    cp.cuda.runtime.deviceSynchronize()
    elapsed = time.perf_counter() - started
    per_call = elapsed / repeats
    overhead = (None if step_wall_seconds is None else
                100.0 * per_call / float(step_wall_seconds))
    return {"repeats": repeats, "elapsed_seconds": elapsed,
            "seconds_per_validation": per_call,
            "step_wall_seconds": step_wall_seconds,
            "overhead_percent": overhead}


__all__ = [
    "FieldRule", "HealthCheckError", "HealthField", "MAX_HEALTH_FIELDS",
    "STATUS_BITS", "StateHealthValidator", "ValidationReport",
    "benchmark_validator", "collect_state_fields", "cuda_source",
    "field_from_array", "rule_for_field", "validate_fields_cpu",
    "validate_state_cpu",
]
