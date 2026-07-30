"""Produce a gpuwm N1.5 nest-force candidate from an instrumented WRF dump.

Operand adjudication (WRF v4.6.1 reference bundle): ``module_integrate.F``
advances the parent solve and clock before calling ``med_nest_force``
(:393-396, :409-423).  ``mediation_force_domain.F`` then couples the parent
and child in place (:111-135), calls ``force_domain_em_part2`` (:167-178),
uncouples child then parent (:179-202), and only afterwards clears
``first_force`` and resets child ``dtbc`` (:203-206).  The instrumentation's
table hook is immediately after the force callee returns and before either
uncouple.  Inside ``bdy_interp1``, the tendency is
``(SINT(cfld)-nfld)/cdt`` and the new value table is ``nfld``
(``interp_fcn.F:2578-2584``; analogous side writes at :2587-2616).

Therefore the interpolation donor is the *before d01* parent state, already
one parent step ahead when FORCE begins.  The previous child boundary target
in the tendency recurrence is the *before d02* child state.  The two ``after``
samples diagnose WRF's non-bit-inert reciprocal uncoupling pass and must not
feed either operand.  This producer restores the complete ``before`` input
records, forms coupled FP32 fields in fresh CPU scratch through gpuwm's
REAL-emulation mirror, and never couples or uncouples a restored state in
place.  With ``use_theta_m=0``, WRF ``t_2`` is dry-theta-minus-300; it is
restored as gpuwm ``thp = (t_2 + 300) - t_init`` before mirror coupling.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from types import SimpleNamespace
import sys
from typing import Mapping

import numpy as np


_SCRIPT_PATH = Path(__file__).resolve()


def _validate_repository_layout(repo: Path, script_path: Path) -> Path:
    """Require one self-consistent producer, dump reader, and gpuwm tree."""
    resolved_repo = repo.resolve()
    resolved_script = script_path.resolve()
    gpuwm_package = resolved_repo / "gpuwm" / "__init__.py"
    dump_reader_path = resolved_script.with_name("dump_reader.py")
    if not gpuwm_package.is_file() or not dump_reader_path.is_file():
        raise RuntimeError(
            "N1.5 producer repository layout mismatch: "
            f"gpuwm package={gpuwm_package}; "
            f"dump_reader beside producer={dump_reader_path}")
    return resolved_repo


def _repo_root() -> Path:
    """Resolve from this module, never from the caller's working directory."""
    return _validate_repository_layout(_SCRIPT_PATH.parents[2], _SCRIPT_PATH)


sys.path.insert(0, str(_SCRIPT_PATH.parent))
sys.path.insert(0, str(_repo_root()))

from dump_reader import (
    BoundaryClockSample,
    DumpFormatError,
    ForceInputKey,
    NestForceDump,
    TableKey,
    read_dump,
)

from gpuwm.core.clock import DomainClock, DomainTicks  # noqa: E402
from gpuwm.core.nest_interp import register_nest  # noqa: E402
from gpuwm.namelist_import import parse_namelist  # noqa: E402
from gpuwm.verify.npref import (np_couple_nest_field, np_nest_force,  # noqa: E402
                                np_sint)


PINNED_NAMELIST = Path(__file__).with_name("namelist.input.n1p5")
_SIDES = {"west": "xs", "east": "xe", "south": "ys", "north": "ye"}
_STATE_FIELDS = ("u", "v", "w", "t", "ph", "mu")
_AUX_FIELDS = ("mub", "t_init", "c1h", "c2h", "c1f", "c2f",
               "msft", "msfu", "msfv")


class ProducerError(DumpFormatError):
    """The dump cannot independently drive the pinned N1.5 producer."""


@dataclass(frozen=True)
class DomainGeometry:
    grid_id: int
    parent_id: int
    nx: int
    ny: int
    nz: int
    parent_grid_ratio: int
    parent_time_step_ratio: int
    i_parent_start: int
    j_parent_start: int


@dataclass(frozen=True)
class PinnedCase:
    parent: DomainGeometry
    child: DomainGeometry
    parent_dt: np.float32
    child_dt: np.float32
    mp_physics: int
    spec_bdy_width: int
    spec_zone: int
    relax_zone: int
    use_theta_m: int


def _column(section: Mapping[str, list], key: str, count: int) -> list:
    try:
        values = list(section[key])
    except KeyError as exc:
        raise ProducerError(
            f"pinned namelist is missing required setting {key!r}") from exc
    if len(values) == 1:
        values *= count
    if len(values) != count:
        raise ProducerError(
            f"pinned namelist {key} has {len(values)} values; expected {count}")
    return values


def read_pinned_case(path: str | Path = PINNED_NAMELIST) -> PinnedCase:
    """Read only the geometry/force settings that govern the N1.5 mirror."""
    sections = parse_namelist(path)
    try:
        domains = sections["domains"]
        physics = sections["physics"]
        dynamics = sections["dynamics"]
        boundary = sections["bdy_control"]
    except KeyError as exc:
        raise ProducerError(
            f"pinned namelist is missing &{exc.args[0]}") from exc

    max_dom_values = domains.get("max_dom", [])
    if max_dom_values != [2]:
        raise ProducerError(
            f"pinned namelist max_dom is {max_dom_values!r}; expected [2]")
    count = 2
    e_we = [int(v) for v in _column(domains, "e_we", count)]
    e_sn = [int(v) for v in _column(domains, "e_sn", count)]
    e_vert = [int(v) for v in _column(domains, "e_vert", count)]
    grid_id = [int(v) for v in _column(domains, "grid_id", count)]
    parent_id = [int(v) for v in _column(domains, "parent_id", count)]
    grid_ratio = [int(v) for v in _column(
        domains, "parent_grid_ratio", count)]
    time_ratio = [int(v) for v in _column(
        domains, "parent_time_step_ratio", count)]
    ipos = [int(v) for v in _column(domains, "i_parent_start", count)]
    jpos = [int(v) for v in _column(domains, "j_parent_start", count)]
    if grid_id != [1, 2] or parent_id != [0, 1]:
        raise ProducerError(
            "pinned namelist must describe the d01 parent and d02 child")
    if grid_ratio[0] != 1 or time_ratio[0] != 1:
        raise ProducerError("pinned d01 ratios must both be one")

    geometry = tuple(
        DomainGeometry(
            grid_id[n], parent_id[n], e_we[n] - 1, e_sn[n] - 1,
            e_vert[n] - 1, grid_ratio[n], time_ratio[n], ipos[n], jpos[n],
        )
        for n in range(count)
    )
    parent_dt = np.float32(_column(domains, "time_step", count)[0])
    child_dt = np.float32(
        np.float32(parent_dt) / np.float32(geometry[1].parent_time_step_ratio))
    return PinnedCase(
        parent=geometry[0],
        child=geometry[1],
        parent_dt=parent_dt,
        child_dt=child_dt,
        mp_physics=int(_column(physics, "mp_physics", count)[1]),
        spec_bdy_width=int(_column(boundary, "spec_bdy_width", count)[1]),
        spec_zone=int(_column(boundary, "spec_zone", count)[1]),
        relax_zone=int(_column(boundary, "relax_zone", count)[1]),
        use_theta_m=int(_column(dynamics, "use_theta_m", count)[1]),
    )


def _same_fp32(left: float | np.float32, right: float | np.float32) -> bool:
    return np.float32(left).tobytes() == np.float32(right).tobytes()


def validate_pinned_metadata(dump: NestForceDump, case: PinnedCase) -> None:
    """Fail before arithmetic if the dump is not the pinned d01+d02 case."""
    md = dump.metadata
    mismatches: list[str] = []

    def check(label: str, actual, expected) -> None:
        if actual != expected:
            mismatches.append(f"{label}={actual!r}, expected {expected!r}")

    check("domain_id", md.domain_id, case.child.grid_id)
    check("parent_id", md.parent_id, case.parent.grid_id)
    check("mp_physics", md.mp_physics, case.mp_physics)
    check("spec_bdy_width", md.spec_bdy_width, case.spec_bdy_width)
    check("nids", md.nids, 1)
    check("nide", md.nide, case.child.nx + 1)
    check("njds", md.njds, 1)
    check("njde", md.njde, case.child.ny + 1)
    check("nkds", md.nkds, 1)
    check("nkde", md.nkde, case.child.nz + 1)
    if not _same_fp32(md.parent_dt, case.parent_dt):
        mismatches.append(
            f"parent_dt={md.parent_dt!r}, expected FP32 {float(case.parent_dt)!r}")
    if not _same_fp32(md.child_dt, case.child_dt):
        mismatches.append(
            f"child_dt={md.child_dt!r}, expected FP32 {float(case.child_dt)!r}")
    if case.use_theta_m != 0:
        mismatches.append(
            f"use_theta_m={case.use_theta_m!r}, expected 0 for dry-theta tables")
    if mismatches:
        raise ProducerError("pinned geometry mismatch: " + "; ".join(mismatches))


def _field_inventory(dump: NestForceDump) -> tuple[tuple[str, str], ...]:
    inventory = sorted({(key.family, key.field) for key in dump.tables})
    state = sorted(field for family, field in inventory if family == "state")
    if state != sorted(_STATE_FIELDS):
        raise ProducerError(
            f"state table inventory is {state}; expected {list(_STATE_FIELDS)}")
    names = [field for _, field in inventory]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ProducerError(
            f"field names collide across table families: {duplicates}")
    return tuple(inventory)


def _shape(geometry: DomainGeometry, family: str, field: str) -> tuple[int, ...]:
    nz, ny, nx = geometry.nz, geometry.ny, geometry.nx
    if family in {"moist", "scalar"}:
        return nz, ny, nx
    if family == "state":
        return {
            "mu": (ny, nx),
            "u": (nz, ny, nx + 1),
            "v": (nz, ny + 1, nx),
            "w": (nz + 1, ny, nx),
            "t": (nz, ny, nx),
            "ph": (nz + 1, ny, nx),
        }[field]
    return {
        "mub": (ny, nx),
        "t_init": (nz, ny, nx),
        "c1h": (nz,),
        "c2h": (nz,),
        "c1f": (nz + 1,),
        "c2f": (nz + 1,),
        "msft": (ny, nx),
        "msfu": (ny, nx + 1),
        "msfv": (ny + 1, nx),
    }[field]


def _required_inputs(
    dump: NestForceDump,
    case: PinnedCase,
) -> dict[ForceInputKey, tuple[int, ...]]:
    inventory = _field_inventory(dump)
    required: dict[ForceInputKey, tuple[int, ...]] = {}
    for geometry in (case.parent, case.child):
        for family, field in inventory:
            key = ForceInputKey("before", geometry.grid_id, family, field)
            required[key] = _shape(geometry, family, field)
        for field in _AUX_FIELDS:
            key = ForceInputKey("before", geometry.grid_id, "aux", field)
            required[key] = _shape(geometry, "aux", field)
    return required


def validate_force_inputs(dump: NestForceDump, case: PinnedCase) -> None:
    required = _required_inputs(dump, case)
    actual = set(dump.force_inputs)
    missing = sorted(str(key) for key in set(required) - actual)
    extra = sorted(str(key) for key in actual - set(required))
    if missing:
        raise ProducerError("missing force input field(s): " + ", ".join(missing))
    if extra:
        raise ProducerError("unexpected force input field(s): " + ", ".join(extra))
    for key, expected in required.items():
        values = dump.force_inputs[key]
        if values.shape != expected:
            raise ProducerError(
                f"force input geometry mismatch for {key}: "
                f"shape={values.shape}, expected={expected}")
        if values.dtype != np.float32 or not values.flags.c_contiguous:
            raise ProducerError(f"force input {key} is not canonical float32")


def _input(dump: NestForceDump, domain_id: int, family: str, field: str):
    return dump.force_inputs[ForceInputKey("before", domain_id, family, field)]


def _restore_state(
    dump: NestForceDump,
    geometry: DomainGeometry,
    inventory: tuple[tuple[str, str], ...],
) -> SimpleNamespace:
    domain_id = geometry.grid_id
    t_wrf = _input(dump, domain_id, "state", "t")
    thb = _input(dump, domain_id, "aux", "t_init")
    # Mirror the stored FP32 representation changes explicitly.  np_couple's
    # t branch converts thb+thp back to WRF's theta-minus-300 operand.
    thp = np.asarray(
        np.asarray(t_wrf + np.float32(300.0), dtype=np.float32) - thb,
        dtype=np.float32,
    )
    thp.setflags(write=False)
    state = SimpleNamespace(
        mup=_input(dump, domain_id, "state", "mu"),
        u=_input(dump, domain_id, "state", "u"),
        v=_input(dump, domain_id, "state", "v"),
        w=_input(dump, domain_id, "state", "w"),
        thp=thp,
        php=_input(dump, domain_id, "state", "ph"),
        mub2d=_input(dump, domain_id, "aux", "mub"),
        thb=thb,
        c1h=_input(dump, domain_id, "aux", "c1h"),
        c2h=_input(dump, domain_id, "aux", "c2h"),
        c1f=_input(dump, domain_id, "aux", "c1f"),
        c2f=_input(dump, domain_id, "aux", "c2f"),
        msft=_input(dump, domain_id, "aux", "msft"),
        msfu=_input(dump, domain_id, "aux", "msfu"),
        msfv=_input(dump, domain_id, "aux", "msfv"),
        has_msf=True,
    )
    for family, field in inventory:
        if family in {"moist", "scalar"}:
            setattr(state, field, _input(dump, domain_id, family, field))
    return state


def _sample_from_input(
    dump: NestForceDump,
    domain_id: int,
    geometry: DomainGeometry,
) -> np.ndarray:
    mu = _input(dump, domain_id, "state", "mu")
    t = _input(dump, domain_id, "state", "t")
    u = _input(dump, domain_id, "state", "u")
    v = _input(dump, domain_id, "state", "v")
    w = _input(dump, domain_id, "state", "w")
    ph = _input(dump, domain_id, "state", "ph")
    nz, ny, nx = geometry.nz, geometry.ny, geometry.nx
    return np.asarray([
        mu[0, 0], t[0, 0, 0], u[0, 0, 0], v[0, 0, 0],
        w[0, 0, 0], ph[0, 0, 0],
        mu[ny - 1, nx - 1], t[nz - 1, ny - 1, nx - 1],
        u[nz - 1, ny - 1, nx - 1],
        v[nz - 1, ny - 1, nx - 1],
        w[nz - 1, ny - 1, nx - 1],
        ph[nz - 1, ny - 1, nx - 1],
    ], dtype=np.float32)


def validate_sample_adjudication(dump: NestForceDump, case: PinnedCase) -> None:
    if len(dump.prognostic_samples) != 4:
        raise ProducerError(
            "missing force-bracketing prognostic samples; expected "
            "before d01, before d02, after d01, after d02")
    before = {
        sample.domain_id: sample for sample in dump.prognostic_samples
        if sample.phase == "before"
    }
    for geometry in (case.parent, case.child):
        sample = before.get(geometry.grid_id)
        if sample is None:
            raise ProducerError(
                f"missing before d{geometry.grid_id:02d} prognostic sample")
        restored = _sample_from_input(dump, geometry.grid_id, geometry)
        if not np.array_equal(restored.view(np.uint32),
                              sample.values.view(np.uint32)):
            mismatch = int(np.flatnonzero(
                restored.view(np.uint32) != sample.values.view(np.uint32))[0])
            raise ProducerError(
                f"before d{geometry.grid_id:02d} prognostic sample mismatch "
                f"at value {mismatch}: input={float(restored[mismatch])!r}, "
                f"PROG={float(sample.values[mismatch])!r}")


def _registrations(case: PinnedCase) -> dict[str, object]:
    ratio = case.child.parent_grid_ratio
    return {
        stagger: register_nest(
            nri=ratio,
            nrj=ratio,
            i_parent_start=case.child.i_parent_start,
            j_parent_start=case.child.j_parent_start,
            child_nx=case.child.nx,
            child_ny=case.child.ny,
            parent_nx=case.parent.nx,
            parent_ny=case.parent.ny,
            stagger="" if stagger == "m" else stagger,
            wrapper="bdy",
        )
        for stagger in ("m", "x", "y")
    }


def build_candidate(
    dump: NestForceDump,
    case: PinnedCase,
) -> dict[TableKey, np.ndarray]:
    """Run the complete non-mutating CPU REAL-emulation mirror chain."""
    validate_pinned_metadata(dump, case)
    validate_force_inputs(dump, case)
    validate_sample_adjudication(dump, case)
    inventory = _field_inventory(dump)
    parent = _restore_state(dump, case.parent, inventory)
    child = _restore_state(dump, case.child, inventory)
    registrations = _registrations(case)
    mirror = np_nest_force(
        parent,
        child,
        registrations,
        field_names=tuple(field for _, field in inventory),
        parent_dt_fp32=np.float32(dump.metadata.parent_dt),
        parent_interval_ticks=case.child.parent_time_step_ratio,
        spec_zone=case.spec_zone,
        relax_zone=case.relax_zone,
        spec_bdy_width=case.spec_bdy_width,
        dtype=np.float32,
    )
    by_field = {field: family for family, field in inventory}
    result: dict[TableKey, np.ndarray] = {}
    for field, sides in mirror.items():
        family = by_field[field]
        for side, suffix in _SIDES.items():
            for kind_index, kind in enumerate(("value", "tendency")):
                key = TableKey(family, field, suffix, kind)
                values = sides[side][kind_index]
                if side in {"south", "north"}:
                    # gpuwm's resident application layout is (z,width,x);
                    # the dump/NPZ comparator contract is uniformly
                    # (z,along_boundary,width).
                    values = np.swapaxes(values, 1, 2)
                values = np.ascontiguousarray(values, dtype=np.float32)
                result[key] = values
    expected = set(dump.tables)
    if set(result) != expected:
        missing = sorted(str(key) for key in expected - set(result))
        extra = sorted(str(key) for key in set(result) - expected)
        raise ProducerError(
            f"candidate table inventory mismatch: missing={missing}, extra={extra}")
    for key, values in result.items():
        if values.shape != dump.tables[key].shape:
            raise ProducerError(
                f"candidate geometry mismatch for {key}: "
                f"shape={values.shape}, expected={dump.tables[key].shape}")
    validate_boundary_clock(dump, case)
    return result


def _fp32_boundary_expression(
    tables: Mapping[TableKey, np.ndarray],
    field: str,
    side: str,
    dtbc: np.float32,
) -> np.float32:
    value = np.float32(tables[TableKey("state", field, side, "value")][0, 0, 0])
    tendency = np.float32(
        tables[TableKey("state", field, side, "tendency")][0, 0, 0])
    return np.float32(value + np.float32(dtbc * tendency))


def _clock_values(
    tables: Mapping[TableKey, np.ndarray],
    dtbc: np.float32,
) -> np.ndarray:
    return np.asarray([
        _fp32_boundary_expression(tables, "u", "xs", dtbc),
        _fp32_boundary_expression(tables, "v", "ys", dtbc),
        _fp32_boundary_expression(tables, "t", "xe", dtbc),
        _fp32_boundary_expression(tables, "w", "ye", dtbc),
        _fp32_boundary_expression(tables, "ph", "xs", dtbc),
        _fp32_boundary_expression(tables, "mu", "ys", dtbc),
    ], dtype=np.float32)


def _child_clock(case: PinnedCase) -> DomainClock:
    ratio = case.child.parent_time_step_ratio
    spec = DomainTicks(
        grid_id=case.child.grid_id,
        parent_id=case.parent.grid_id,
        parent_time_step_ratio=ratio,
        step_ticks=1,
        dt_fp32=case.child_dt,
        history_ticks=ratio,
        restart_ticks=None,
        radt_ticks=None,
        stepra=None,
        cudt_ticks=None,
        stepcu=None,
        bldt_ticks=None,
        stepbl=None,
    )
    clock = DomainClock(spec, tick_den=1, run_ticks=ratio)
    clock.mark_force()
    return clock


def validate_boundary_clock(
    dump: NestForceDump,
    case: PinnedCase,
) -> None:
    """Validate WRF-recurrent FP32 dtbc and the six solve-side oracles."""
    ratio = case.child.parent_time_step_ratio
    samples: tuple[BoundaryClockSample, ...] = dump.boundary_clock_samples
    if len(samples) != ratio:
        raise ProducerError(
            f"missing boundary-clock records: got {len(samples)}, expected {ratio}")
    clock = _child_clock(case)
    for ordinal, sample in enumerate(samples, start=1):
        clock.prepare_step()
        dtbc = clock.dtbc_launch_fp32
        if sample.step_ordinal != ordinal or not _same_fp32(sample.dtbc, dtbc):
            raise ProducerError(
                f"boundary-clock mismatch at ordinal {ordinal}: "
                f"dump ordinal/dtbc={sample.step_ordinal}/{sample.dtbc!r}, "
                f"clock.py recurrence={float(dtbc)!r}")
        reference_values = _clock_values(dump.tables, dtbc)
        if not np.array_equal(reference_values.view(np.uint32),
                              sample.values.view(np.uint32)):
            mismatch = int(np.flatnonzero(
                reference_values.view(np.uint32)
                != sample.values.view(np.uint32))[0])
            raise ProducerError(
                f"boundary-clock oracle mismatch at ordinal {ordinal}, "
                f"value {mismatch}: tables={float(reference_values[mismatch])!r}, "
                f"DBDY={float(sample.values[mismatch])!r}")


def _canonical_strip(
    values: np.ndarray,
    side: str,
    width: int,
) -> np.ndarray:
    """Extract canonical ``(vertical, along, width)`` boundary storage."""
    source = np.asarray(values, dtype=np.float32)
    if source.ndim == 2:
        source = source[None]
    if side == "xs":
        result = source[:, :, :width]
    elif side == "xe":
        result = source[:, :, -1:-width - 1:-1]
    elif side == "ys":
        result = np.swapaxes(source[:, :width, :], 1, 2)
    elif side == "ye":
        result = np.swapaxes(source[:, -1:-width - 1:-1, :], 1, 2)
    else:
        raise ProducerError(f"unknown boundary side {side!r}")
    return np.ascontiguousarray(result, dtype=np.float32)


def _fp32_successor(
    value: np.ndarray,
    tendency: np.ndarray,
    cdt: np.float32,
) -> np.ndarray:
    increment = np.asarray(
        np.float32(cdt) * np.asarray(tendency, dtype=np.float32),
        dtype=np.float32,
    )
    return np.asarray(
        np.asarray(value, dtype=np.float32) + increment,
        dtype=np.float32,
    )


def _coupled_metric(
    reference: np.ndarray,
    candidate: np.ndarray,
    band: float,
) -> dict[str, object]:
    """Apply the registered relative form at coupled-value scale."""
    ref64 = np.asarray(reference, dtype=np.float32).astype(np.float64)
    candidate32 = np.asarray(candidate, dtype=np.float32)
    candidate64 = candidate32.astype(np.float64)
    absolute = np.abs(candidate64 - ref64)
    table_scale = float(np.max(np.abs(ref64), initial=0.0))
    denominator = np.abs(ref64) + table_scale
    failures = absolute > float(band) * denominator
    normalized = np.zeros_like(absolute)
    nonzero = denominator != 0.0
    np.divide(absolute, denominator, out=normalized, where=nonzero)
    normalized[~nonzero & (absolute != 0.0)] = np.inf
    worst = np.unravel_index(int(np.argmax(normalized)), normalized.shape)
    return {
        "pass": not bool(np.any(failures)),
        "band_fail_count": int(np.count_nonzero(failures)),
        "bit_mismatch_count": int(np.count_nonzero(
            np.asarray(reference, dtype=np.float32).view(np.uint32)
            != candidate32.view(np.uint32))),
        "max_metric": float(np.max(normalized, initial=0.0)),
        "max_abs_error": float(np.max(absolute, initial=0.0)),
        "table_scale": table_scale,
        "worst_index": [int(index) for index in worst],
    }


def build_successor_diagnostics(
    dump: NestForceDump,
    candidate: Mapping[TableKey, np.ndarray],
    case: PinnedCase,
) -> dict[str, object]:
    """Report theta successors and assert gpuwm's stored recurrence."""
    # Import the landed comparator rather than duplicating its table logic or
    # changing its semantics.  Its expected raw theta misses remain visible.
    from compare_nest_force import BAND, compare

    comparison = compare(dump, dict(candidate))
    comparison_summary = {
        name: comparison[name]
        for name in (
            "pass", "band", "metric", "reference_tables",
            "candidate_tables", "missing", "extra", "value_max_metric",
            "tendency_max_metric",
        )
    }
    comparison_summary["failing_tables"] = [
        record["key"] for record in comparison["tables"]
        if not record.get("pass", False)
    ]

    inventory = _field_inventory(dump)
    parent = _restore_state(dump, case.parent, inventory)
    parent_coupled = np_couple_nest_field(parent, "t", dtype=np.float32)
    raw_gpuwm_successor = np.asarray(
        np_sint(parent_coupled, _registrations(case)["m"], dtype=np.float32),
        dtype=np.float32,
    )
    width = min(
        max(int(case.spec_zone), int(case.relax_zone) + 1),
        int(case.spec_bdy_width),
    )
    cdt = np.float32(dump.metadata.parent_dt)
    tables: dict[str, object] = {}
    successor_pass = True
    recurrence_pass = True
    successor_max = 0.0
    recurrence_max = 0.0
    for side in ("xs", "xe", "ys", "ye"):
        tendency_key = TableKey("state", "t", side, "tendency")
        value_key = TableKey("state", "t", side, "value")
        wrf_successor = _fp32_successor(
            dump.tables[value_key], dump.tables[tendency_key], cdt)
        gpuwm_successor = _fp32_successor(
            candidate[value_key], candidate[tendency_key], cdt)
        successor = _coupled_metric(wrf_successor, gpuwm_successor, BAND)
        recurrence = _coupled_metric(
            _canonical_strip(raw_gpuwm_successor, side, width),
            gpuwm_successor,
            BAND,
        )
        successor_pass = successor_pass and bool(successor["pass"])
        recurrence_pass = recurrence_pass and bool(recurrence["pass"])
        successor_max = max(successor_max, float(successor["max_metric"]))
        recurrence_max = max(recurrence_max, float(recurrence["max_metric"]))
        tables[str(tendency_key)] = {
            "gpuwm_vs_wrf_successor": successor,
            "gpuwm_stored_table_recurrence": recurrence,
        }

    if not recurrence_pass:
        failed = [
            key for key, record in tables.items()
            if not record["gpuwm_stored_table_recurrence"]["pass"]
        ]
        raise ProducerError(
            "gpuwm stored-table recurrence exceeds the registered 1e-6 "
            f"coupled-value band: {failed}")

    return {
        "registered_comparator": comparison_summary,
        "successor_metrics": {
            "band": BAND,
            "metric": (
                "abs(candidate-reference) <= band*"
                "(abs(reference)+max_abs(reference_table))"
            ),
            "fp32_expression": "fl32(VALUE + fl32(parent_dt*TENDENCY))",
            "successor_pass": successor_pass,
            "stored_table_recurrence_pass": recurrence_pass,
            "successor_max_metric": successor_max,
            "stored_table_recurrence_max_metric": recurrence_max,
            "tables": tables,
        },
    }


def write_candidate(
    dump: NestForceDump,
    path: str | Path,
    *,
    namelist_path: str | Path = PINNED_NAMELIST,
    successor_metrics: bool = False,
) -> dict[str, object]:
    """Build, validate, and write the exact comparator-facing NPZ mapping."""
    out_path = Path(path)
    case = read_pinned_case(namelist_path)
    candidate = build_candidate(dump, case)
    diagnostics = (
        build_successor_diagnostics(dump, candidate, case)
        if successor_metrics else {}
    )
    with out_path.open("wb") as stream:
        np.savez(stream, **{
            str(key): values for key, values in sorted(candidate.items())
        })
    report = {
        "output": str(out_path),
        "table_count": len(candidate),
        "dtype": "float32",
        "parent_operand": "before d01",
        "child_previous_target": "before d02",
        "ignored_recurrence_operands": ["after d01", "after d02"],
        "dtbc": [float(sample.dtbc)
                 for sample in dump.boundary_clock_samples],
    }
    report.update(diagnostics)
    return report


def produce(
    dump_path: str | Path,
    out_path: str | Path,
    *,
    namelist_path: str | Path = PINNED_NAMELIST,
    successor_metrics: bool = False,
) -> dict[str, object]:
    source = Path(dump_path).resolve()
    destination = Path(out_path).resolve()
    if source == destination:
        raise ProducerError("output NPZ must not alias the input dump")
    dump = read_dump(source)
    return write_candidate(
        dump,
        destination,
        namelist_path=namelist_path,
        successor_metrics=successor_metrics,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dump", type=Path)
    parser.add_argument("out_npz", type=Path)
    parser.add_argument(
        "--summary", action="store_true",
        help="print the producer summary as JSON after writing the NPZ")
    parser.add_argument(
        "--successor-metrics", action="store_true",
        help=("include the registered comparator summary plus per-theta-table "
              "FP32 successor and gpuwm recurrence metrics"))
    args = parser.parse_args()
    report = produce(
        args.dump,
        args.out_npz,
        successor_metrics=args.successor_metrics,
    )
    if args.summary:
        print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
