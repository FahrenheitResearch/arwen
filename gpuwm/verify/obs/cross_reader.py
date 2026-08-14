"""Does the mandated science core read a history tape the way we do?

The battery scores two models' output through one reader.  Every number it
will ever publish therefore rests on a premise nobody has measured: that
``wrf-rust`` opens a tape written by *either* writer and hands back the
values that are actually in the file.  A battery built on that premise
instead of on a receipt would report reader disagreement as forecast skill,
and nobody downstream could tell the two apart.

So this module measures the premise.  For one frame from each writer it
asks the science core for a registered quantity, derives the same quantity
from a plain NetCDF read, and records the difference.  Three properties
make the answer worth having:

* **The control is independent of the thing it controls.**  The control
  path opens the file with ``netCDF4`` and does its own arithmetic.  It
  never calls the science core -- an instrument that agreed because it
  asked the same code twice would measure nothing.  This is the single
  place in :mod:`gpuwm.verify.obs` that computes a diagnostic quantity
  itself, and it exists only to be checked against.

* **The tolerances are mechanism-derived, not calibrated.**  A passthrough
  read and a maximum-over-a-column are exact operations: an FP32 value
  widened to FP64 is exact, and a maximum is one of its own inputs, so the
  registered tolerance is zero and no observation can move it.  The one
  derived quantity gets a tolerance bounded by its own arithmetic, quoted
  in the registration beside the reason.

* **The reader is pinned by distribution, not by hope.**  The receipt
  records which module answered, where it came from, and whether it is the
  mandated distribution version -- a receipt that cannot name its reader is
  not evidence, the same standard the evaluator commit is held to.

Two things the receipt is explicit about:

* the earth-rotation sign, which is **settled**.  Stored ``SINALPHA`` and
  ``COSALPHA`` carry ``alpha = cone * (STAND_LON - XLONG)``, positive west
  of the standard meridian, and the registered control is WRF's own
  ``earth_u``/``earth_v`` applied to exactly that angle.  A second lineage
  negates both the angle and the formula, which cancels to the identical
  rotation; there is no convention split to adjudicate.  The opposite-sign
  result is still computed and carried beside the scored one, but it is a
  **magnitude control on the sign, not a rival convention**: it prices what
  a sign error would cost on the domain being read.  The adjudication, its
  source citations and its geometric ground truth are the rotation addendum
  under ``docs/public/receipts/obsbattery/``.

* reflectivity provenance.  The registered quantity is the column maximum
  of the *stored* ``REFL_10CM``, which is each model's own
  scheme-consistent operator.  The science core also offers a recomputed
  ``maxdbz``; it is a different operator, not a different read of this one,
  and it is not what this receipt scores.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from gpuwm import science_core as _pin

SCHEMA_ID = "gpuwm.obs-cross-reader/v1"

#: The mandated science core, named by distribution.  ``importlib.metadata``
#: answers for the distribution; the module's own ``__version__`` attribute
#: is recorded beside it but is not the authority, because an attribute the
#: upstream forgets to bump would silently become the pin.
SCIENCE_CORE_DISTRIBUTION = _pin.SCIENCE_CORE_DISTRIBUTION
#: The floor of the certified window.  Kept under the old name because
#: receipts quote it; the verdict below tests the WINDOW, so a newer
#: wrf-rust inside 0.2.x passes instead of failing a receipt for being new.
SCIENCE_CORE_VERSION = _pin.SCIENCE_CORE_FLOOR
SCIENCE_CORE_REQUIREMENT = _pin.SCIENCE_CORE_REQUIREMENT
SCIENCE_CORE_IMPORT = "wrf"

#: Frames carry one record each in every writer the battery consumes; when a
#: file carries more, the leading record is the one scored.  Registered so a
#: later multi-record tape cannot silently change which values were compared.
TIME_INDEX = 0

#: The receipt that settles the earth-rotation sign, cited from the
#: registration so every future receipt carries the pointer rather than the
#: open question the first one carried.
ROTATION_ADJUDICATION = (
    "docs/public/receipts/obsbattery/"
    "B0-CROSS-READER-ROTATION-ADDENDUM-20260803.md")

#: The angle stored in ``SINALPHA``/``COSALPHA``, as WPS geogrid writes it.
#: Quoted in the registration so a receipt states which angle its rotation
#: was applied to instead of leaving a reader to infer it.
STORED_ROTATION_ANGLE = "alpha = cone * (STAND_LON - XLONG)"

_EVALUATOR_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")

_STATUS_SCORED = "scored"
_STATUS_UNAVAILABLE = "unavailable"

#: How two writers' spellings of one global attribute are compared.
#:
#: Exact equality is the wrong test for a projection constant.  The ARW
#: history-tape convention stores geographic constants as FP32, and a writer
#: that keeps them in FP64 records the same projection as a different
#: decimal -- a centre latitude of 39.6848 against 39.684818267822266 is one
#: domain written at two widths, not two domains.  Numeric attributes are
#: therefore compared after rounding both sides to FP32, the width the
#: convention stores them in; at 40 degrees that is a tolerance of about
#: 0.4 m, which no two distinct cases could hide inside.  Everything else --
#: grid dimensions, projection codes, the simulation start date -- is
#: compared exactly, because those are exact quantities in every writer.
PAIRING_NUMERIC_COMPARISON = "equal after rounding both sides to float32"

#: Global attributes that decide whether two frames are the same case.  A
#: cross-reader receipt claims both writers were exercised on one case, and
#: that claim is checked against the files rather than against the caller's
#: directory names.
#:
#: ``CEN_LAT`` and ``CEN_LON`` are deliberately absent.  They are not
#: projection constants: one writer echoes the requested centre and another
#: records the centre its grid actually landed on, so on one domain they
#: disagree by a couple of metres while describing the same grid.  Comparing
#: them would test which convention a writer follows.  The georeference
#: those attributes summarise is compared directly instead, below, which is
#: the stronger test.
PAIRING_ATTRIBUTES: tuple[str, ...] = (
    "SIMULATION_START_DATE",
    "MAP_PROJ",
    "DX",
    "DY",
    "TRUELAT1",
    "TRUELAT2",
    "STAND_LON",
    "WEST-EAST_GRID_DIMENSION",
    "SOUTH-NORTH_GRID_DIMENSION",
    "BOTTOM-TOP_GRID_DIMENSION",
)

#: The stored georeference, compared cell by cell across the two sides.
PAIRING_FIELDS: tuple[str, ...] = ("XLAT", "XLONG")

#: Degrees.  Two writers evaluate the same map projection with independent
#: FP32 implementations, so their latitude/longitude arrays agree to a few
#: FP32 ULP -- of order 1e-5 degrees at these magnitudes.  1e-4 degrees sits
#: an order above that and two orders below the battery's finest grid
#: spacing (3 km is 0.027 degrees of latitude), so independent arithmetic
#: fits inside it and a genuinely different domain placement cannot.
PAIRING_FIELD_ABS_TOL_DEGREES = 1e-4


@dataclass(frozen=True)
class ReaderQuantity:
    """One registered quantity, and how each side of the check produces it.

    ``kind`` says what a disagreement would mean.  ``passthrough`` and
    ``reduction`` are exact operations on the stored bytes, so any nonzero
    difference is a reader disagreement about the file.  ``derived`` runs
    arithmetic on several stored fields, so its tolerance admits floating
    point association and nothing more.
    """

    name: str
    units: str
    kind: str
    science_core_call: str
    control_recipe: str
    control_inputs: tuple[str, ...]
    abs_tol: float
    tolerance_reason: str
    alternate_control: str | None = None
    alternate_control_reason: str | None = None


REGISTERED_QUANTITIES: tuple[ReaderQuantity, ...] = (
    ReaderQuantity(
        name="t2",
        units="K",
        kind="passthrough",
        science_core_call="getvar(frame, 't2')",
        control_recipe="T2[time_index]",
        control_inputs=("T2",),
        abs_tol=0.0,
        tolerance_reason=(
            "a stored FP32 value widened to FP64 is exact, so the two "
            "reads of one array can only differ if they disagree about "
            "which bytes the array is"),
    ),
    ReaderQuantity(
        name="uvmet10_u",
        units="m s-1",
        kind="derived",
        science_core_call="getvar(frame, 'uvmet10')[0]",
        control_recipe="U10*COSALPHA - V10*SINALPHA",
        control_inputs=("U10", "V10", "SINALPHA", "COSALPHA"),
        abs_tol=1e-9,
        tolerance_reason=(
            "two FP64 products and one FP64 sum over inputs bounded by "
            "1e2 round to order 1e-14; 1e-9 m s-1 sits five decades above "
            "that bound and twelve below any wind difference a forecast "
            "reader could act on, so it admits association order and "
            "nothing else"),
        alternate_control="U10*COSALPHA + V10*SINALPHA",
        alternate_control_reason=(
            "the stored angle inside the opposite formula shape -- a "
            "rotation no shipping WRF post-processor computes, and not a "
            "rival convention.  Kept as a magnitude control on the "
            "rotation sign: it prices what a sign error would cost on the "
            "domain being read.  The convention is settled in favour of "
            "the registered recipe by " + ROTATION_ADJUDICATION),
    ),
    ReaderQuantity(
        name="uvmet10_v",
        units="m s-1",
        kind="derived",
        science_core_call="getvar(frame, 'uvmet10')[1]",
        control_recipe="U10*SINALPHA + V10*COSALPHA",
        control_inputs=("U10", "V10", "SINALPHA", "COSALPHA"),
        abs_tol=1e-9,
        tolerance_reason=(
            "two FP64 products and one FP64 sum over inputs bounded by "
            "1e2 round to order 1e-14; 1e-9 m s-1 sits five decades above "
            "that bound and twelve below any wind difference a forecast "
            "reader could act on, so it admits association order and "
            "nothing else"),
        alternate_control="V10*COSALPHA - U10*SINALPHA",
        alternate_control_reason=(
            "the stored angle inside the opposite formula shape -- a "
            "rotation no shipping WRF post-processor computes, and not a "
            "rival convention.  Kept as a magnitude control on the "
            "rotation sign: it prices what a sign error would cost on the "
            "domain being read.  The convention is settled in favour of "
            "the registered recipe by " + ROTATION_ADJUDICATION),
    ),
    ReaderQuantity(
        name="refl_10cm_column_max",
        units="dBZ",
        kind="reduction",
        science_core_call="getvar(frame, 'REFL_10CM').max(axis=vertical)",
        control_recipe="REFL_10CM[time_index].max(axis=vertical)",
        control_inputs=("REFL_10CM",),
        abs_tol=0.0,
        tolerance_reason=(
            "a maximum returns one of its own inputs, so a column maximum "
            "over an exactly-widened array introduces no arithmetic; any "
            "nonzero difference is a disagreement about the array or about "
            "which axis is vertical"),
    ),
)

#: Every stored variable some registered control needs.
CONTROL_VARIABLES: tuple[str, ...] = tuple(sorted(
    {name for quantity in REGISTERED_QUANTITIES
     for name in quantity.control_inputs}))


def _control_t2(fields: Mapping[str, np.ndarray]) -> np.ndarray:
    return fields["T2"]


def _control_uvmet10_u(fields: Mapping[str, np.ndarray]) -> np.ndarray:
    return fields["U10"] * fields["COSALPHA"] - fields["V10"] * fields["SINALPHA"]


def _control_uvmet10_v(fields: Mapping[str, np.ndarray]) -> np.ndarray:
    return fields["U10"] * fields["SINALPHA"] + fields["V10"] * fields["COSALPHA"]


def _control_refl_column_max(fields: Mapping[str, np.ndarray]) -> np.ndarray:
    return np.max(fields["REFL_10CM"], axis=0)


def _alternate_uvmet10_u(fields: Mapping[str, np.ndarray]) -> np.ndarray:
    return fields["U10"] * fields["COSALPHA"] + fields["V10"] * fields["SINALPHA"]


def _alternate_uvmet10_v(fields: Mapping[str, np.ndarray]) -> np.ndarray:
    return fields["V10"] * fields["COSALPHA"] - fields["U10"] * fields["SINALPHA"]


_CONTROLS: dict[str, Callable[[Mapping[str, np.ndarray]], np.ndarray]] = {
    "t2": _control_t2,
    "uvmet10_u": _control_uvmet10_u,
    "uvmet10_v": _control_uvmet10_v,
    "refl_10cm_column_max": _control_refl_column_max,
}

_ALTERNATE_CONTROLS: dict[
    str, Callable[[Mapping[str, np.ndarray]], np.ndarray]] = {
    "uvmet10_u": _alternate_uvmet10_u,
    "uvmet10_v": _alternate_uvmet10_v,
}


# ---------------------------------------------------------------------------
# the science core
# ---------------------------------------------------------------------------


class ScienceCoreReader(Protocol):
    """The seam the receipt measures through.

    The battery's reader is :class:`WrfRustReader`.  The protocol exists so
    the harness's own pins can drive a stub that answers wrong on purpose --
    a control that cannot be made to fail proves nothing -- and never so a
    missing science core can be papered over: there is no fallback
    implementation, and importing the core is what
    :meth:`WrfRustReader.provenance` does first.
    """

    def provenance(self) -> dict[str, Any]:
        ...

    def open(self, path: Path) -> Any:
        ...

    def quantity(self, handle: Any, quantity: ReaderQuantity) -> np.ndarray:
        ...


def _home_relative(path: Path) -> str:
    """The path with the user-profile prefix spelled ``~``.

    Receipts under docs/public ship, and the release snapshot gate
    refuses a developer-absolute path in anything that ships.  The
    username is machine identity, not provenance; everything after it --
    which checkout, which module file -- is the provenance and is kept.
    """
    try:
        return "~/" + path.relative_to(Path.home()).as_posix()
    except ValueError:
        return str(path)


def _home_relative_url(url: str) -> str:
    """``file:`` URLs take the same ``~`` spelling; other schemes pass."""
    if not url.startswith("file:"):
        return url
    prefix = Path.home().as_posix()
    for spelling in (f"file:///{prefix}/", f"file://{prefix}/"):
        if url.startswith(spelling):
            return "file:///~/" + url[len(spelling):]
    return url


class WrfRustReader:
    """The mandated science core, asked through its shipped Python API."""

    def __init__(self) -> None:
        import wrf  # noqa: PLC0415 -- the registration must stay importable

        self._core = wrf

    def provenance(self) -> dict[str, Any]:
        import importlib.metadata as metadata  # noqa: PLC0415

        try:
            distribution_version: str | None = metadata.version(
                SCIENCE_CORE_DISTRIBUTION)
        except metadata.PackageNotFoundError:
            distribution_version = None
        direct_url: str | None = None
        editable: bool | None = None
        if distribution_version is not None:
            raw = metadata.distribution(
                SCIENCE_CORE_DISTRIBUTION).read_text("direct_url.json")
            if raw:
                try:
                    parsed = json.loads(raw)
                except ValueError:
                    parsed = {}
                direct_url = parsed.get("url")
                editable = bool(parsed.get("dir_info", {}).get("editable"))
        module_file = getattr(self._core, "__file__", None)
        return {
            "import_name": SCIENCE_CORE_IMPORT,
            "distribution": SCIENCE_CORE_DISTRIBUTION,
            "distribution_version": distribution_version,
            "module_file": (_home_relative(Path(module_file).resolve())
                            if module_file else None),
            "module_version_attribute": getattr(
                self._core, "__version__", None),
            "installed_from": (_home_relative_url(direct_url)
                               if direct_url else direct_url),
            "installed_editable": editable,
        }

    def open(self, path: Path) -> Any:
        return self._core.WrfFile(str(path))

    def quantity(self, handle: Any, quantity: ReaderQuantity) -> np.ndarray:
        if quantity.name == "t2":
            return _as_f64(self._core.getvar(handle, "t2",
                                             timeidx=TIME_INDEX))
        if quantity.name in ("uvmet10_u", "uvmet10_v"):
            pair = _as_f64(self._core.getvar(handle, "uvmet10",
                                             timeidx=TIME_INDEX))
            if pair.ndim != 3 or pair.shape[0] != 2:
                raise ValueError(
                    "the science core returned a uvmet10 of shape "
                    f"{pair.shape}; the registered call expects a leading "
                    "component axis of length 2")
            return pair[0] if quantity.name == "uvmet10_u" else pair[1]
        if quantity.name == "refl_10cm_column_max":
            column = _as_f64(self._core.getvar(handle, "REFL_10CM",
                                               timeidx=TIME_INDEX))
            if column.ndim != 3:
                raise ValueError(
                    "the science core returned a REFL_10CM of shape "
                    f"{column.shape}; the registered reduction expects a "
                    "leading vertical axis")
            return np.max(column, axis=0)
        raise KeyError(
            f"{quantity.name!r} is registered but this reader does not know "
            "how to ask the science core for it")


def _as_f64(value: Any) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(value), dtype=np.float64)


# ---------------------------------------------------------------------------
# registration
# ---------------------------------------------------------------------------


def make_registration() -> dict[str, object]:
    """The scoring contract, pinned before any frame is opened."""
    return {
        "schema": SCHEMA_ID,
        "science_core": {
            "distribution": SCIENCE_CORE_DISTRIBUTION,
            "version": SCIENCE_CORE_VERSION,
            "import_name": SCIENCE_CORE_IMPORT,
            "role": (
                "mandated core for reading and diagnosing history tapes; "
                "this receipt measures it, and never replaces it"),
        },
        "control": {
            "reader": "netCDF4",
            "auto_mask": False,
            "dtype": "float64",
            "independence": (
                "the control never calls the science core; its arithmetic "
                "is written out in each quantity's control_recipe"),
        },
        "time_index": TIME_INDEX,
        "quantities": {
            quantity.name: {
                "units": quantity.units,
                "kind": quantity.kind,
                "science_core_call": quantity.science_core_call,
                "control_recipe": quantity.control_recipe,
                "control_inputs": list(quantity.control_inputs),
                "abs_tol": quantity.abs_tol,
                "tolerance_reason": quantity.tolerance_reason,
                "alternate_control": quantity.alternate_control,
                "alternate_control_reason": quantity.alternate_control_reason,
                "alternate_control_is_rival_convention": False,
            }
            for quantity in REGISTERED_QUANTITIES
        },
        "earth_rotation": {
            "stored_angle": STORED_ROTATION_ANGLE,
            "status": "settled",
            "settled_by": ROTATION_ADJUDICATION,
            "scope": ("northern-hemisphere Lambert conformal, MAP_PROJ = 1; "
                      "the southern-hemisphere sign flip and the other "
                      "projections are unaudited"),
        },
        "pairing": {
            "attributes": list(PAIRING_ATTRIBUTES),
            "numeric_comparison": PAIRING_NUMERIC_COMPARISON,
            "other_comparison": "exact",
            "also_compared": ["valid_time"],
            "georeference_fields": list(PAIRING_FIELDS),
            "georeference_abs_tol_degrees": PAIRING_FIELD_ABS_TOL_DEGREES,
        },
        "verdict_rule": (
            "PASS iff every registered quantity scored on every side "
            "within its registered tolerance, the science core answered "
            "at the pinned distribution version, and the sides agree on "
            "every pairing attribute"),
    }


def canonical_json(document: object) -> str:
    """Stable JSON text: the receipt is a function of its inputs alone."""
    return json.dumps(document, sort_keys=True, indent=2,
                      separators=(",", ": "), ensure_ascii=True)


def _stable_hash(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=True).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def registration_sha256(registration: Mapping[str, object] | None = None
                        ) -> str:
    """SHA-256 over the canonical encoding of the registration."""
    if registration is None:
        registration = make_registration()
    return _stable_hash(registration)


def sha256_file(path: Path, *, block_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(block_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_evaluator_commit(root: Path | None = None) -> str | None:
    """The 40-hex commit of the tree that produced a receipt, or None.

    None is a refusal, not a default: a receipt whose evaluator cannot be
    named is not evidence.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(root) if root is not None else None,
            capture_output=True, text=True, check=False, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    candidate = result.stdout.strip()
    return candidate if _EVALUATOR_COMMIT_RE.match(candidate) else None


# ---------------------------------------------------------------------------
# the control read
# ---------------------------------------------------------------------------


def _open_dataset(path: Path):
    import netCDF4  # noqa: PLC0415 -- the registration must stay importable

    return netCDF4.Dataset(str(path))


def _record(variable) -> np.ndarray:
    """The registered record of a stored variable, as FP64.

    Auto-masking is off deliberately: a fill value replaced by a mask would
    compare equal to whatever the science core returned there, which is the
    one thing a reader control must not do.
    """
    variable.set_auto_mask(False)
    if variable.dimensions and variable.dimensions[0] == "Time":
        array = variable[TIME_INDEX]
    else:
        array = variable[:]
    return _as_f64(array)


def read_control_fields(path: Path) -> dict[str, np.ndarray]:
    """Every stored variable a registered control needs, as FP64."""
    return _read_named(path, CONTROL_VARIABLES)


def read_pairing_fields(path: Path) -> dict[str, np.ndarray]:
    """The stored georeference, as FP64."""
    return _read_named(path, PAIRING_FIELDS)


def _read_named(path: Path, names: tuple[str, ...]) -> dict[str, np.ndarray]:
    fields: dict[str, np.ndarray] = {}
    with _open_dataset(path) as dataset:
        for name in names:
            variable = dataset.variables.get(name)
            if variable is None:
                continue
            fields[name] = _record(variable)
    return fields


def frame_identity(path: Path) -> dict[str, object]:
    """The attributes that say which case and which frame this file is."""
    with _open_dataset(path) as dataset:
        attributes = {
            name: _plain(dataset.getncattr(name))
            for name in PAIRING_ATTRIBUTES if name in dataset.ncattrs()
        }
        title = (_plain(dataset.getncattr("TITLE")).strip()
                 if "TITLE" in dataset.ncattrs() else None)
        times = dataset.variables.get("Times")
        if times is not None:
            times.set_auto_mask(False)
            valid_time = b"".join(
                np.asarray(times[TIME_INDEX]).tolist()).decode(
                    "ascii", "replace")
        else:
            valid_time = None
    return {
        "title": title,
        "valid_time": valid_time,
        "attributes": attributes,
    }


def _plain(value: object) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, bytes):
        return value.decode("ascii", "replace")
    return value


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------


def compare_arrays(core: np.ndarray, control: np.ndarray) -> dict[str, object]:
    """Difference metrics between the two reads of one quantity.

    Non-finite elements are not averaged away.  Where one side is finite and
    the other is not the arrays disagree about the value, so those elements
    are counted separately and the difference metrics are taken over the
    elements both sides call finite.
    """
    core = _as_f64(core)
    control = _as_f64(control)
    if core.shape != control.shape:
        return {
            "shape_core": list(core.shape),
            "shape_control": list(control.shape),
            "shape_agrees": False,
            "max_abs_diff": None,
            "rms_diff": None,
            "differing_elements": None,
            "elements": None,
            "nonfinite_disagreements": None,
        }
    core_finite = np.isfinite(core)
    control_finite = np.isfinite(control)
    both = core_finite & control_finite
    delta = np.abs(core[both] - control[both])
    return {
        "shape_core": list(core.shape),
        "shape_control": list(control.shape),
        "shape_agrees": True,
        "max_abs_diff": float(np.max(delta)) if delta.size else 0.0,
        "rms_diff": (float(np.sqrt(np.mean(np.square(delta))))
                     if delta.size else 0.0),
        "differing_elements": int(np.count_nonzero(delta)),
        "elements": int(core.size),
        "nonfinite_disagreements": int(
            np.count_nonzero(core_finite != control_finite)),
    }


def score_side(path: Path, *, reader: ScienceCoreReader) -> dict[str, object]:
    """Score every registered quantity on one frame."""
    path = Path(path)
    fields = read_control_fields(path)
    handle = reader.open(path)
    quantities: dict[str, object] = {}
    for quantity in REGISTERED_QUANTITIES:
        missing = [name for name in quantity.control_inputs
                   if name not in fields]
        if missing:
            quantities[quantity.name] = {
                "status": _STATUS_UNAVAILABLE,
                "reason": ("the frame does not store "
                           f"{', '.join(missing)}, which the registered "
                           "control needs"),
                "verdict": None,
            }
            continue
        try:
            core_values = reader.quantity(handle, quantity)
        except Exception as error:  # noqa: BLE001 -- recorded, not swallowed
            quantities[quantity.name] = {
                "status": _STATUS_UNAVAILABLE,
                "reason": (f"the science core refused {quantity.name!r}: "
                           f"{type(error).__name__}: {error}"),
                "verdict": None,
            }
            continue
        control_values = _CONTROLS[quantity.name](fields)
        metrics = compare_arrays(core_values, control_values)
        entry: dict[str, object] = {
            "status": _STATUS_SCORED,
            "units": quantity.units,
            "kind": quantity.kind,
            "abs_tol": quantity.abs_tol,
            "metrics": metrics,
        }
        alternate = _ALTERNATE_CONTROLS.get(quantity.name)
        if alternate is not None:
            entry["diagnostic_alternate_control"] = {
                "recipe": quantity.alternate_control,
                "reason": quantity.alternate_control_reason,
                "metrics": compare_arrays(core_values, alternate(fields)),
                "gated": False,
                "is_rival_convention": False,
                "means": "cost of a rotation-sign error on this domain",
                "settled_by": ROTATION_ADJUDICATION,
            }
        maximum = metrics["max_abs_diff"]
        within = (metrics["shape_agrees"]
                  and metrics["nonfinite_disagreements"] == 0
                  and maximum is not None
                  and float(maximum) <= quantity.abs_tol)
        entry["verdict"] = "PASS" if within else "FAIL"
        quantities[quantity.name] = entry
    return {
        "identity": frame_identity(path),
        "quantities": quantities,
    }


def attribute_values_agree(values: list[object]) -> bool:
    """Do two writers' spellings of one attribute describe one thing?

    Booleans are integers in Python and are deliberately not treated as
    numbers here: a flag is an exact quantity, and rounding one to FP32
    would make ``True`` and ``1.0`` agree, which is not what any attribute
    on this list means.
    """
    if len(values) < 2:
        return True
    if any(value is None for value in values):
        return all(value is None for value in values)
    numeric = [value for value in values
               if isinstance(value, (int, float))
               and not isinstance(value, bool)]
    if len(numeric) == len(values):
        rounded = {float(np.float32(value)) for value in numeric}
        return len(rounded) == 1
    return len({json.dumps(value, sort_keys=True)
                for value in values}) == 1


def check_pairing(sides: Mapping[str, Mapping[str, object]],
                  georeference: Mapping[str, Mapping[str, np.ndarray]]
                  | None = None) -> dict[str, object]:
    """Do the scored frames come from one case, at one valid time?

    Checked against the files' own attributes and their own stored
    georeference.  A cross-reader receipt whose two sides are different
    cases still proves each reader path, and proves nothing about the pair,
    so the distinction is recorded rather than assumed away.
    """
    labels = sorted(sides)
    if len(labels) < 2:
        return {
            "status": "single_side",
            "same_case": False,
            "reason": ("a cross-reader receipt pairs two writers; "
                       f"{len(labels)} side(s) were scored"),
            "differing_attributes": {},
            "georeference": {},
            "distinct_writers": False,
            "writer_titles": {label: sides[label]["identity"]["title"]
                              for label in labels},
        }
    identities = {label: sides[label]["identity"] for label in labels}
    differing: dict[str, object] = {}
    for name in PAIRING_ATTRIBUTES:
        values = {label: identities[label]["attributes"].get(name)
                  for label in labels}
        if not attribute_values_agree(list(values.values())):
            differing[name] = values
    valid_times = {label: identities[label]["valid_time"] for label in labels}
    if not attribute_values_agree(list(valid_times.values())):
        differing["valid_time"] = valid_times

    geo_report: dict[str, object] = {}
    if georeference is not None:
        reference = labels[0]
        for name in PAIRING_FIELDS:
            present = [label for label in labels
                       if name in georeference.get(label, {})]
            if len(present) != len(labels):
                geo_report[name] = {
                    "status": _STATUS_UNAVAILABLE,
                    "reason": (f"{name} is not stored on "
                               + ", ".join(sorted(
                                   set(labels) - set(present))))}
                differing[name] = "not stored on every side"
                continue
            worst = 0.0
            shapes_agree = True
            for label in labels[1:]:
                left = georeference[reference][name]
                right = georeference[label][name]
                if left.shape != right.shape:
                    shapes_agree = False
                    break
                worst = max(worst, float(np.max(np.abs(left - right))))
            if not shapes_agree:
                geo_report[name] = {
                    "status": _STATUS_SCORED,
                    "shapes_agree": False,
                    "max_abs_diff_degrees": None,
                    "abs_tol_degrees": PAIRING_FIELD_ABS_TOL_DEGREES,
                    "verdict": "FAIL",
                }
                differing[name] = "the sides store different grid shapes"
                continue
            within = worst <= PAIRING_FIELD_ABS_TOL_DEGREES
            geo_report[name] = {
                "status": _STATUS_SCORED,
                "shapes_agree": True,
                "max_abs_diff_degrees": worst,
                "abs_tol_degrees": PAIRING_FIELD_ABS_TOL_DEGREES,
                "verdict": "PASS" if within else "FAIL",
            }
            if not within:
                differing[name] = (
                    f"max |delta| {worst:.6g} degrees exceeds "
                    f"{PAIRING_FIELD_ABS_TOL_DEGREES:g}")

    titles = {identities[label]["title"] for label in labels}
    return {
        "status": "paired" if not differing else "mismatched",
        "same_case": not differing,
        "reason": (None if not differing else
                   "the sides disagree on "
                   f"{', '.join(sorted(differing))}"),
        "differing_attributes": differing,
        "georeference": geo_report,
        "distinct_writers": len(titles) == len(labels),
        "writer_titles": {label: identities[label]["title"]
                          for label in labels},
    }


def check_science_core_pin(provenance: Mapping[str, object]
                           ) -> dict[str, object]:
    """Is the module that answered inside the certified version window?"""
    version = provenance.get("distribution_version")
    if version is None:
        return {
            "status": "unresolved",
            "ok": False,
            "expected": SCIENCE_CORE_REQUIREMENT,
            "observed": None,
            "reason": (f"{SCIENCE_CORE_DISTRIBUTION} is not an installed "
                       "distribution here, so the reader that answered "
                       "cannot be named"),
        }
    ok = _pin.version_supported(version)
    attribute = provenance.get("module_version_attribute")
    note = None
    if attribute is not None and str(attribute) != str(version):
        note = (f"the module's __version__ attribute is {attribute!r} while "
                f"the installed distribution is {version!r}; the "
                "distribution is the authority and the attribute is "
                "recorded, not gated")
    return {
        "status": "pinned" if ok else "mismatch",
        "ok": ok,
        "expected": SCIENCE_CORE_REQUIREMENT,
        "observed": str(version),
        "reason": (None if ok else
                   f"the battery requires {SCIENCE_CORE_REQUIREMENT}; "
                   f"{version} answered"),
        "version_attribute_note": note,
    }


#: What a receipt is evidence *of*.  A receipt scored on a registered
#: battery case speaks for that case; one scored on any other matched pair
#: proves the reader paths and nothing about the battery's cases.  Which one
#: a receipt is depends on whether its caller could name a case, so it is
#: recorded from that fact rather than inferred from the frames.
SCOPE_BATTERY_CASE = "battery-case"
SCOPE_READER_QUALIFICATION = "reader-qualification"


def build_cross_reader_receipt(sides: Mapping[str, Path], *,
                               evaluator_commit: str,
                               reader: ScienceCoreReader | None = None,
                               case_id: str | None = None,
                               ) -> dict[str, object]:
    """Score one frame per writer and return the receipt document.

    ``case_id`` names the registered battery case the frames belong to, when
    they belong to one.  Left unnamed, the receipt says so: it qualifies the
    reader on the writers it was given, and claims nothing about a case.
    """
    if not _EVALUATOR_COMMIT_RE.match(str(evaluator_commit)):
        raise ValueError(
            "evaluator_commit must be a 40-hex commit id; a receipt that "
            "cannot name its evaluator is not evidence (got "
            f"{evaluator_commit!r})")
    if not sides:
        raise ValueError("a cross-reader receipt needs at least one frame")

    if reader is None:
        reader = WrfRustReader()
    provenance = reader.provenance()
    registration = make_registration()

    scored: dict[str, object] = {}
    georeference: dict[str, dict[str, np.ndarray]] = {}
    inputs: list[dict[str, object]] = []
    for label in sorted(sides):
        path = Path(sides[label])
        scored[label] = score_side(path, reader=reader)
        georeference[label] = read_pairing_fields(path)
        inputs.append({
            "side": label,
            "name": path.name,
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        })

    pairing = check_pairing(scored, georeference)
    pin = check_science_core_pin(provenance)

    verdicts = [entry["verdict"]
                for side in scored.values()
                for entry in side["quantities"].values()
                if entry["verdict"] is not None]
    unavailable = sorted({
        f"{label}:{name}"
        for label, side in scored.items()
        for name, entry in side["quantities"].items()
        if entry["status"] == _STATUS_UNAVAILABLE
    })
    readers_agree = bool(verdicts) and "FAIL" not in verdicts
    verdict = "PASS" if (readers_agree and pin["ok"] and pairing["same_case"]
                         and not unavailable) else "FAIL"

    return {
        "schema": SCHEMA_ID,
        "evaluator_commit": evaluator_commit,
        "battery_case_id": case_id,
        "scope": (SCOPE_BATTERY_CASE if case_id
                  else SCOPE_READER_QUALIFICATION),
        "registration": registration,
        "registration_sha256": registration_sha256(registration),
        "science_core_provenance": provenance,
        "science_core_pin": pin,
        "inputs": inputs,
        "sides": scored,
        "pairing": pairing,
        "readers_agree": readers_agree,
        "unavailable_quantities": unavailable,
        "verdict": verdict,
    }


def render_markdown(receipt: Mapping[str, object]) -> str:
    """A reviewer-facing table of the same numbers the receipt carries."""
    pin = receipt["science_core_pin"]
    provenance = receipt["science_core_provenance"]
    pairing = receipt["pairing"]
    lines = [
        "# cross-reader receipt: the science core against an independent read",
        "",
        f"- schema: `{receipt['schema']}`",
        f"- evaluator commit: `{receipt['evaluator_commit']}`",
        f"- scope: `{receipt['scope']}`"
        + (f", battery case `{receipt['battery_case_id']}`"
           if receipt["battery_case_id"] else
           " (no registered battery case named; this receipt qualifies the "
           "reader on the writers it was given and claims nothing about a "
           "battery case)"),
        f"- registration sha256: `{receipt['registration_sha256']}`",
        f"- science core: `{provenance['distribution']}"
        f"=={provenance['distribution_version']}` "
        f"(pin {pin['expected']}, {pin['status']})",
        f"- science core module: `{provenance['module_file']}`",
        f"- pairing: {pairing['status']}"
        + (f" ({pairing['reason']})" if pairing["reason"] else ""),
        f"- verdict: **{receipt['verdict']}**",
        "",
        "| side | quantity | units | kind | tol | max abs diff | rms diff | "
        "differing elements | verdict |",
        "|---|---|---|---|---:|---:|---:|---:|---|",
    ]
    for label in sorted(receipt["sides"]):
        side = receipt["sides"][label]
        for name in sorted(side["quantities"]):
            entry = side["quantities"][name]
            if entry["status"] != _STATUS_SCORED:
                lines.append(
                    f"| {label} | {name} | - | - | - | - | - | - | "
                    f"{entry['status']} |")
                continue
            metrics = entry["metrics"]
            lines.append(
                f"| {label} | {name} | {entry['units']} | {entry['kind']} | "
                f"{entry['abs_tol']:g} | "
                f"{_cell(metrics['max_abs_diff'])} | "
                f"{_cell(metrics['rms_diff'])} | "
                f"{_cell(metrics['differing_elements'])} | "
                f"{entry['verdict']} |")
    if pairing["georeference"]:
        lines.extend([
            "",
            "## pairing: the stored georeference across the sides",
            "",
            "| field | max abs diff (deg) | tol (deg) | verdict |",
            "|---|---:|---:|---|",
        ])
        for name in sorted(pairing["georeference"]):
            entry = pairing["georeference"][name]
            lines.append(
                f"| {name} | "
                f"{_cell(entry.get('max_abs_diff_degrees'))} | "
                f"{_cell(entry.get('abs_tol_degrees'))} | "
                f"{entry.get('verdict', entry['status'])} |")
    diagnostics = [
        (label, name, entry["diagnostic_alternate_control"])
        for label in sorted(receipt["sides"])
        for name, entry in sorted(receipt["sides"][label]["quantities"].items())
        if entry.get("diagnostic_alternate_control")
    ]
    if diagnostics:
        lines.extend([
            "",
            "## rotation-sign magnitude control (recorded, never gated)",
            "",
            "Not a rival convention: the stored angle inside the opposite "
            "formula shape, which no shipping WRF post-processor computes. "
            "The column below is what a rotation-sign error would cost on "
            f"this domain. Settled by `{ROTATION_ADJUDICATION}`.",
            "",
            "| side | quantity | sign-error recipe | cost of a sign error |",
            "|---|---|---|---:|",
        ])
        for label, name, diagnostic in diagnostics:
            lines.append(
                f"| {label} | {name} | `{diagnostic['recipe']}` | "
                f"{_cell(diagnostic['metrics']['max_abs_diff'])} |")
    return "\n".join(lines) + "\n"


def _cell(value: object) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


__all__ = [
    "CONTROL_VARIABLES",
    "PAIRING_ATTRIBUTES",
    "PAIRING_FIELDS",
    "PAIRING_FIELD_ABS_TOL_DEGREES",
    "PAIRING_NUMERIC_COMPARISON",
    "REGISTERED_QUANTITIES",
    "ROTATION_ADJUDICATION",
    "STORED_ROTATION_ANGLE",
    "SCHEMA_ID",
    "SCIENCE_CORE_DISTRIBUTION",
    "SCIENCE_CORE_IMPORT",
    "SCIENCE_CORE_REQUIREMENT",
    "SCIENCE_CORE_VERSION",
    "SCOPE_BATTERY_CASE",
    "SCOPE_READER_QUALIFICATION",
    "TIME_INDEX",
    "ReaderQuantity",
    "ScienceCoreReader",
    "WrfRustReader",
    "attribute_values_agree",
    "build_cross_reader_receipt",
    "canonical_json",
    "check_pairing",
    "check_science_core_pin",
    "compare_arrays",
    "frame_identity",
    "make_registration",
    "read_control_fields",
    "read_pairing_fields",
    "registration_sha256",
    "render_markdown",
    "resolve_evaluator_commit",
    "score_side",
    "sha256_file",
]
