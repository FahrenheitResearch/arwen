"""Cases, arms, paths and window classes for the nowcast radar-heating evaluation.

Everything the box scripts need to know about a case or an arm comes from
``cases.toml`` and ``arms.toml`` through this module, so the shell scripts
carry no case or arm knowledge of their own.  Orchestration only: nothing
here reads a weather array.

Box layout (``--root``, default ``/work/nh``)::

    <root>/<case>/raw/                 HRRR and RAP GRIB2 (anonymous S3)
    <root>/<case>/windows/<name>/      heating windows (one dir per window end)
    <root>/<case>/nowcast/<name>/      frames roots (gpuwm-obs.nowcast-frames.v1)
    <root>/<case>/arms/<ARM>/          experiment.toml, arm.json, out/, markers
    <root>/<case>/preps/<HHMM>z-<N>h/  one preparation per start and length:
                                       experiment.toml, namelist.wps, prepared/
    <root>/<case>/sheets/, scores/     rw_compare sheets, score tables, decision inputs

One preparation per start time AND run length.  The door binds the SHA-256
of the config its preparation read and refuses any other; it cuts an arm's
``[radar_heating]`` table out before that check (work package 2's
``detach_table_from_config``), so A, C and E share one preparation.  It
does not cut the length: the door runs the config's ``run_seconds`` (a
shorter ``--run-seconds`` only warns), so a 6 h arm (C60, E3D) bound to an
18 h preparation would be refused, and given the 18 h config it would run
18 h.  The preparation key is therefore ``<start>-<length>``.

CLI (used by the shell scripts)::

    python -m tools.heat_eval.plan case-ids [--roles decision,anchor]
    python -m tools.heat_eval.plan arm-ids --case ID [--plan core|full]
    python -m tools.heat_eval.plan fetch-list --case ID [--plan core|full]
    python -m tools.heat_eval.plan field --case ID [--arm ARM] NAME
    python -m tools.heat_eval.plan prep-inputs --root R --case ID --key 1800z-18h
    python -m tools.heat_eval.plan check-windows --root R --case ID --arm ARM
"""

from __future__ import annotations

import argparse
import json
import sys
import tomllib
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Mapping, Sequence

HERE = Path(__file__).resolve().parent
CASES_FILE = HERE / "cases.toml"
ARMS_FILE = HERE / "arms.toml"
DEFAULT_ROOT = Path("/work/nh")

CASES_SCHEMA = "gpuwm.heat-eval-cases.v1"
ARMS_SCHEMA = "gpuwm.heat-eval-arms.v1"
WINDOW_SCHEMA = "gpuwm-obs.radar-tten-ref.v1"

#: The keys of the forecast door's ``[radar_heating]`` table (WP2).
RADAR_HEATING_KEYS = (
    "windows", "window_minutes", "active_minutes", "latent_heat_period_min",
    "strict_suppression", "pbl_extension", "mp_tend_lim")

#: What an arm row may say.  Anything else is refused: an arm that differs
#: from A in a second knob makes its comparison meaningless.
ARM_ROW_KEYS = frozenset({
    "id", "label", "column", "plan", "roles", "start_offset_minutes",
    "run_hours", "core_run_hours", "lead_class", "cards", "min_cards",
    "heating"})
CASE_ROW_KEYS = frozenset({
    "id", "date", "role", "order", "goes_east", "region", "start_hour"})
LEAD_CLASSES = ("none", "observed", "forecast", "oracle")
CASE_ROLES = ("decision", "anchor", "reserve")
PLANS = ("core", "full")

#: The case step (``case_steps.sh``) that writes each window set, so the box
#: chain knows which step an arm waits for and which failure ends it.
WINDOW_STEPS = {
    "levelii-b": "windows-b",
    "levelii-t0": "windows-t0",
    "levelii-fwd": "windows-fwd",
    "stormscope": "windows-nowcast",
    "mrms2d": "windows-oracle",
}

HRRR_BUCKET = "https://noaa-hrrr-bdp-pds.s3.amazonaws.com"
RAP_BUCKET = "https://noaa-rap-pds.s3.amazonaws.com"


class PlanError(ValueError):
    """A case or arm row, or a window set, that the evaluation refuses."""


def utc(text: str) -> datetime:
    """``2026-10-01T18:00:00Z`` / ``...+00:00`` / ``20261001T1800Z`` as UTC."""
    value = str(text).strip()
    for fmt in ("%Y%m%dT%H%MZ", "%Y%m%dT%H%M%SZ"):
        try:
            return datetime.strptime(value, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def stamp(time: datetime) -> str:
    """The window directory name: ``YYYYmmddTHHMMZ``."""
    return time.astimezone(timezone.utc).strftime("%Y%m%dT%H%MZ")


def iso(time: datetime) -> str:
    return time.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass(frozen=True)
class Case:
    id: str
    date: str
    role: str
    order: int
    goes_east: str
    region: str
    start_hour: int
    boundary_cycle_offset_hours: int
    boundary_cadence_hours: int
    forecast_hours: int
    hrrr_reference_hours: int
    g3_growth: float
    g3_threshold_dbz: float

    @property
    def t0(self) -> datetime:
        day = datetime.strptime(self.date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        return day + timedelta(hours=self.start_hour)

    @property
    def boundary_cycle(self) -> datetime:
        return self.t0 + timedelta(hours=self.boundary_cycle_offset_hours)

    def lead_hours(self, valid: datetime) -> float:
        """Hours after t0: the key every score is filed under.

        Arm B starts an hour before t0, so its own lead of a valid time is
        one more than every other arm's; keying by t0 lines the arms up by
        valid time, which is what a sheet column and a score row compare.
        """
        return (valid - self.t0).total_seconds() / 3600.0


@dataclass(frozen=True)
class Arm:
    id: str
    label: str
    column: str
    plan: str
    roles: tuple[str, ...]
    start_offset_minutes: int
    run_hours: int
    core_run_hours: int
    lead_class: str
    cards: int
    min_cards: int
    heating: Mapping[str, object] = field(default_factory=dict)

    @property
    def heated(self) -> bool:
        return bool(self.heating)

    def start(self, case: Case) -> datetime:
        return case.t0 + timedelta(minutes=self.start_offset_minutes)

    def run_seconds(self, plan: str) -> int:
        hours = self.run_hours if plan == "full" else self.core_run_hours
        return int(hours) * 3600

    def end(self, case: Case, plan: str) -> datetime:
        return self.start(case) + timedelta(seconds=self.run_seconds(plan))

    def in_plan(self, plan: str, case: Case) -> bool:
        if plan not in PLANS:
            raise PlanError(f"unknown plan {plan!r}; one of {PLANS}")
        if self.plan == "full" and plan != "full":
            return False
        return not self.roles or case.role in self.roles

    def prep_key(self, case: "Case", plan: str) -> str:
        """The preparation this arm runs on: its start and its length."""
        return prep_key(self.start(case), self.run_seconds(plan))

    def needs(self, case: "Case", plan: str) -> list[str]:
        """The case steps this arm waits for: its preparation, then its windows."""
        steps = [f"prep-{self.prep_key(case, plan)}"]
        if self.heated:
            name = str(self.heating["windows"])
            if name not in WINDOW_STEPS:
                raise PlanError(f"arm {self.id}: no case step writes the window set {name!r}")
            steps.append(WINDOW_STEPS[name])
        return steps

    def window_ends(self, case: Case) -> list[datetime]:
        """The window ends the door reads: start + n x window_minutes."""
        if not self.heated:
            return []
        step = int(self.heating["window_minutes"])
        active = int(self.heating["active_minutes"])
        return [self.start(case) + timedelta(minutes=step * n)
                for n in range(1, active // step + 1)]


def _read_toml(path: Path) -> dict:
    with open(path, "rb") as handle:
        return tomllib.load(handle)


def load_cases(path: Path = CASES_FILE) -> list[Case]:
    doc = _read_toml(Path(path))
    if doc.get("schema") != CASES_SCHEMA:
        raise PlanError(f"{path}: schema {doc.get('schema')!r}, need {CASES_SCHEMA}")
    defaults = dict(doc.get("defaults", {}))
    cases = []
    for row in doc.get("case", []):
        unknown = set(row) - CASE_ROW_KEYS
        if unknown:
            raise PlanError(f"case row {row.get('id')!r} carries unknown keys {sorted(unknown)}")
        if row.get("role") not in CASE_ROLES:
            raise PlanError(f"case {row.get('id')!r}: role {row.get('role')!r} not in {CASE_ROLES}")
        cases.append(Case(
            id=str(row["id"]), date=str(row["date"]), role=str(row["role"]),
            order=int(row["order"]), goes_east=str(row["goes_east"]),
            region=str(row.get("region", "")),
            start_hour=int(row.get("start_hour", defaults.get("start_hour", 18))),
            boundary_cycle_offset_hours=int(defaults.get("boundary_cycle_offset_hours", -3)),
            boundary_cadence_hours=int(defaults.get("boundary_cadence_hours", 3)),
            forecast_hours=int(defaults.get("forecast_hours", 18)),
            hrrr_reference_hours=int(defaults.get("hrrr_reference_hours", 18)),
            g3_growth=float(defaults.get("g3_growth", 1.5)),
            g3_threshold_dbz=float(defaults.get("g3_threshold_dbz", 35.0))))
    ids = [case.id for case in cases]
    if len(set(ids)) != len(ids):
        raise PlanError("two case rows share an id")
    return sorted(cases, key=lambda case: case.order)


def arm_from_row(row: Mapping[str, object], *, oracle_columns: Sequence[str] = ("E", "E3D")) -> Arm:
    """One arm row, refused when it says more than an arm may say."""
    label = row.get("id", "?")
    unknown = set(row) - ARM_ROW_KEYS
    if unknown:
        raise PlanError(
            f"arm {label!r} sets {sorted(unknown)}; an arm may change only its start, its "
            f"length and the [radar_heating] table, because an arm that differs from A in a "
            f"second knob makes the comparison meaningless")
    heating = dict(row.get("heating", {}) or {})
    extra = set(heating) - set(RADAR_HEATING_KEYS)
    if extra:
        raise PlanError(f"arm {label!r}: {sorted(extra)} are not [radar_heating] keys")
    lead_class = str(row.get("lead_class", "none"))
    if lead_class not in LEAD_CLASSES:
        raise PlanError(f"arm {label!r}: lead_class {lead_class!r} not in {LEAD_CLASSES}")
    if heating:
        for key in ("windows", "window_minutes", "active_minutes"):
            if key not in heating:
                raise PlanError(f"arm {label!r}: a heated arm names [radar_heating] {key}")
        step, active = int(heating["window_minutes"]), int(heating["active_minutes"])
        if step <= 0 or active <= 0 or active % step:
            raise PlanError(
                f"arm {label!r}: window_minutes {step} does not divide active_minutes "
                f"{active}; part of the forced period would run unforced while the record "
                f"says forced")
        if lead_class == "none":
            raise PlanError(f"arm {label!r} heats but says lead_class none")
    elif lead_class != "none":
        raise PlanError(f"arm {label!r} has lead_class {lead_class!r} but no heating")
    column = str(row["column"])
    if lead_class == "oracle" and column not in oracle_columns:
        raise PlanError(
            f"arm {label!r} reads oracle windows but sits in column {column!r}; future "
            f"observations would be scored as forecast skill")
    plan = str(row.get("plan", "core"))
    if plan not in PLANS:
        raise PlanError(f"arm {label!r}: plan {plan!r} not in {PLANS}")
    roles = tuple(str(role) for role in row.get("roles", ()))
    if any(role not in CASE_ROLES for role in roles):
        raise PlanError(f"arm {label!r}: roles {roles} not all in {CASE_ROLES}")
    return Arm(
        id=str(row["id"]), label=str(row.get("label", row["id"])), column=column,
        plan=plan, roles=roles,
        start_offset_minutes=int(row.get("start_offset_minutes", 0)),
        run_hours=int(row["run_hours"]),
        core_run_hours=int(row.get("core_run_hours", row["run_hours"])),
        lead_class=lead_class, cards=int(row.get("cards", 2)),
        min_cards=int(row.get("min_cards", row.get("cards", 2))), heating=heating)


@dataclass(frozen=True)
class ArmTable:
    arms: tuple[Arm, ...]
    oracle_columns: tuple[str, ...]
    forecast_columns: tuple[str, ...]

    def arm(self, arm_id: str) -> Arm:
        for arm in self.arms:
            if arm.id == arm_id:
                return arm
        raise PlanError(f"no arm {arm_id!r}; arms are {[arm.id for arm in self.arms]}")

    def for_case(self, case: Case, plan: str) -> list[Arm]:
        return [arm for arm in self.arms if arm.in_plan(plan, case)]


def load_arms(path: Path = ARMS_FILE) -> ArmTable:
    doc = _read_toml(Path(path))
    if doc.get("schema") != ARMS_SCHEMA:
        raise PlanError(f"{path}: schema {doc.get('schema')!r}, need {ARMS_SCHEMA}")
    oracle = tuple(doc.get("oracle_columns", ("E", "E3D")))
    forecast = tuple(doc.get("forecast_columns", ()))
    if set(oracle) & set(forecast):
        raise PlanError("a column cannot be both an oracle and a forecast column")
    arms = tuple(arm_from_row(row, oracle_columns=oracle) for row in doc.get("arm", []))
    ids = [arm.id for arm in arms]
    if len(set(ids)) != len(ids) or "A" not in ids:
        raise PlanError("arm ids must be unique and include the control A")
    if arms[ids.index("A")].heated:
        raise PlanError("arm A is the unheated control")
    return ArmTable(arms=arms, oracle_columns=oracle, forecast_columns=forecast)


def case_by_id(case_id: str, cases: Sequence[Case] | None = None) -> Case:
    for case in cases if cases is not None else load_cases():
        if case.id == case_id:
            return case
    raise PlanError(f"no case {case_id!r}")


# -- paths ------------------------------------------------------------------

def case_dir(root: Path, case: Case) -> Path:
    return Path(root) / case.id


def arm_dir(root: Path, case: Case, arm: Arm) -> Path:
    return case_dir(root, case) / "arms" / arm.id


def prep_key(start: datetime, run_seconds: int) -> str:
    """``1800z-18h``: a preparation is one start time and one run length."""
    if int(run_seconds) % 3600:
        raise PlanError(f"a run of {run_seconds} s is not whole hours; the boundaries are hourly")
    return f"{start.astimezone(timezone.utc):%H%M}z-{int(run_seconds) // 3600}h"


def prep_dir(root: Path, case: Case, key: str) -> Path:
    """The preparation ``key`` of a case (the module docstring says why per length)."""
    return case_dir(root, case) / "preps" / key


#: The door case of gate G0 (work package 2's ``GPUWM_HEAT_BOX_DOOR``): one
#: hour from t0, long enough for a forced period and a pass-through and one
#: history frame after both, short enough to run twice in minutes.
G0_DOOR_RUN_SECONDS = 3600


def g0_door_key(case: Case) -> str:
    return prep_key(case.t0, G0_DOOR_RUN_SECONDS)


def windows_root(root: Path, case: Case, arm: Arm) -> Path:
    return case_dir(root, case) / "windows" / str(arm.heating["windows"])


# -- inputs -----------------------------------------------------------------

def boundary_leads(case: Case, start: datetime, end: datetime) -> list[int]:
    """Boundary file leads from the case's cycle covering ``[start, end]``."""
    first = int((start - case.boundary_cycle).total_seconds() // 3600)
    last = (end - case.boundary_cycle).total_seconds() / 3600.0
    if first < 0:
        raise PlanError(f"{case.id}: a run starting {iso(start)} precedes its boundary cycle")
    leads, lead = [], first
    while True:
        leads.append(lead)
        if lead >= last:
            return leads
        lead += case.boundary_cadence_hours


def starts_for(case: Case, arms: Sequence[Arm]) -> list[datetime]:
    """The distinct start times (one preparation each), earliest first."""
    return sorted({arm.start(case) for arm in arms})


def fetch_list(case: Case, arms: Sequence[Arm], plan: str) -> list[tuple[str, str]]:
    """``(url, relative destination)`` for every GRIB2 object the case needs."""
    rows: list[tuple[str, str]] = []
    cycle = case.boundary_cycle
    for start in starts_for(case, arms):
        end = max(arm.end(case, plan) for arm in arms if arm.start(case) == start)
        day = start.strftime("%Y%m%d")
        for product in ("wrfnatf00", "wrfprsf00", "wrfsfcf00"):
            name = f"hrrr.t{start.hour:02d}z.{product}.grib2"
            rows.append((f"{HRRR_BUCKET}/hrrr.{day}/conus/{name}", f"raw/{name}"))
        for lead in boundary_leads(case, start, end):
            name = f"rap.t{cycle.hour:02d}z.awp130bgrbf{lead:02d}.grib2"
            rows.append((f"{RAP_BUCKET}/rap.{cycle.strftime('%Y%m%d')}/{name}", f"raw/{name}"))
    day = case.t0.strftime("%Y%m%d")
    for lead in range(1, case.hrrr_reference_hours + 1):
        name = f"hrrr.t{case.t0.hour:02d}z.wrfsfcf{lead:02d}.grib2"
        rows.append((f"{HRRR_BUCKET}/hrrr.{day}/conus/{name}", f"reference/{name}"))
    seen, unique = set(), []
    for url, dest in rows:
        if dest not in seen:
            seen.add(dest)
            unique.append((url, dest))
    return unique


# -- heating windows --------------------------------------------------------

def classify_window(receipt: Mapping[str, object], issue_time: datetime) -> str:
    """``observed``, ``forecast`` or ``oracle`` for one window receipt.

    A window the adapter labelled keeps its label.  An unlabelled window
    (``rw_nexrad grid-ref`` writes none) is classed by time: one ending at
    or before the case's issue time (t0) is a causal observation; one ending
    after it holds observations the forecast could not have had, so it is
    an oracle whatever made it.

    A ``forecast`` label is held to its own record too: a window whose
    source says it is not causal, or was issued after the case's issue time,
    carries information the forecast could not have had, so it is an oracle
    whatever its label says.
    """
    window = receipt.get("window") or {}
    if not isinstance(window, Mapping) or not window.get("end"):
        raise PlanError("a heating window with no window.end cannot be classed")
    end = utc(str(window["end"]))
    source = receipt.get("source") or {}
    if not isinstance(source, Mapping):
        source = {}
    label = source.get("lead_class")
    if label is None:
        return "observed" if end <= issue_time else "oracle"
    if label not in ("forecast", "oracle", "observed"):
        raise PlanError(f"unknown window lead_class {label!r}")
    if label == "observed" and end > issue_time:
        return "oracle"
    if label == "forecast":
        if source.get("causal") is False:
            return "oracle"
        issued = source.get("issue_time")
        if issued and utc(str(issued)) > issue_time:
            return "oracle"
    return str(label)


def arm_window_classes(root: Path, case: Case, arm: Arm) -> list[dict[str, object]]:
    """Every window the arm will read, with its class; refused when missing."""
    rows = []
    base = windows_root(root, case, arm)
    for end in arm.window_ends(case):
        receipt_path = base / stamp(end) / "ref.json"
        if not receipt_path.is_file():
            raise PlanError(
                f"{arm.id}: window {receipt_path} is missing; part of the forced period would "
                f"run on the model while the record says forced")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt.get("schema") != WINDOW_SCHEMA:
            raise PlanError(f"{receipt_path}: schema {receipt.get('schema')!r}")
        status = receipt.get("status", "READY")
        if status != "READY":
            raise PlanError(f"{receipt_path}: status {status!r}")
        rows.append({"end": iso(end), "class": classify_window(receipt, case.t0),
                     "receipt": str(receipt_path),
                     "data_sha256": (receipt.get("data") or {}).get("sha256")})
    return rows


def check_arm_windows(root: Path, case: Case, arm: Arm) -> dict[str, object]:
    """The arm's windows, refused when their classes break the arm's column."""
    rows = arm_window_classes(root, case, arm)
    classes = sorted({str(row["class"]) for row in rows})
    if arm.lead_class == "forecast" and classes != ["forecast"]:
        raise PlanError(
            f"arm {arm.id} is a forecast arm but reads {classes} windows; an oracle window "
            f"in a forecast column would score future observations as skill")
    if arm.lead_class == "observed" and classes != ["observed"]:
        raise PlanError(
            f"arm {arm.id} is the observed-radar arm but reads {classes} windows ending after "
            f"t0 {iso(case.t0)}")
    return {"arm": arm.id, "lead_class": arm.lead_class, "classes": classes, "windows": rows}


def prep_inputs(root: Path, case: Case, key: str) -> list[Path]:
    """Write a preparation's initial-inputs.json and namelist.wps; return its RAP files.

    The start is the HRRR analysis at the preparation's start time (HRRR
    native levels, with the pressure-level and surface files as the soil and
    vegetation donors); the boundaries are the case's RAP cycle at every
    cadence step that covers the run the preparation's config names.
    """
    from gpuwm.experiment import load_experiment
    from gpuwm.hrrr_prepared_bundle import render_wps_namelist

    folder = prep_dir(root, case, key)
    config = _read_toml(folder / "experiment.toml")
    start = config["experiment"]["start_time"].replace(tzinfo=timezone.utc)
    end = start + timedelta(seconds=float(config["experiment"]["run_seconds"]))
    raw = case_dir(root, case) / "raw"
    hour = f"{start.hour:02d}"
    initial = {
        "schema": "gpuwm-initial-source-v1", "source": "hrrr-native",
        "input_files": [str(raw / f"hrrr.t{hour}z.wrfnatf00.grib2")],
        "supplements": {
            "soil_surface_data": [str(raw / f"hrrr.t{hour}z.wrfprsf00.grib2")],
            "vegetation_surface_data": [str(raw / f"hrrr.t{hour}z.wrfsfcf00.grib2")]},
    }
    (folder / "initial-inputs.json").write_text(json.dumps(initial, indent=1), encoding="utf-8")
    (folder / "namelist.wps").write_text(
        render_wps_namelist(load_experiment(folder / "experiment.toml"),
                            interval_seconds=3600.0 * case.boundary_cadence_hours),
        encoding="utf-8")
    cycle = case.boundary_cycle
    files = [raw / f"rap.t{cycle.hour:02d}z.awp130bgrbf{lead:02d}.grib2"
             for lead in boundary_leads(case, start, end)]
    missing = [str(path) for path in files + [Path(p) for p in initial["input_files"]] if not path.is_file()]
    if missing:
        raise PlanError(f"{case.id} {key}: inputs not fetched: {missing}")
    return files


# -- CLI --------------------------------------------------------------------

def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m tools.heat_eval.plan")
    sub = parser.add_subparsers(dest="command", required=True)
    ids = sub.add_parser("case-ids")
    ids.add_argument("--roles", default="anchor,decision")
    arms = sub.add_parser("arm-ids")
    arms.add_argument("--case", required=True)
    arms.add_argument("--plan", default="core", choices=PLANS)
    fetch = sub.add_parser("fetch-list")
    fetch.add_argument("--case", required=True)
    fetch.add_argument("--plan", default="core", choices=PLANS)
    value = sub.add_parser("field")
    value.add_argument("--case", required=True)
    value.add_argument("--arm")
    value.add_argument("--plan", default="core", choices=PLANS)
    value.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    value.add_argument("name")
    prep = sub.add_parser("prep-inputs")
    prep.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    prep.add_argument("--case", required=True)
    prep.add_argument("--key", required=True, help="the preparation key, e.g. 1800z-18h")
    check = sub.add_parser("check-windows")
    check.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    check.add_argument("--case", required=True)
    check.add_argument("--arm", required=True)
    check.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    table = load_arms()
    try:
        if args.command == "case-ids":
            roles = set(args.roles.split(","))
            print("\n".join(case.id for case in load_cases() if case.role in roles))
        elif args.command == "arm-ids":
            case = case_by_id(args.case)
            print("\n".join(arm.id for arm in table.for_case(case, args.plan)))
        elif args.command == "fetch-list":
            case = case_by_id(args.case)
            for url, dest in fetch_list(case, table.for_case(case, args.plan), args.plan):
                print(f"{url}\t{dest}")
        elif args.command == "field":
            case = case_by_id(args.case)
            arm = table.arm(args.arm) if args.arm else None
            values = {
                "t0": iso(case.t0), "date": case.date, "role": case.role,
                "goes_east": case.goes_east, "boundary_cycle": iso(case.boundary_cycle),
                "t0_stamp": stamp(case.t0), "t0_hour": f"{case.t0.hour:02d}",
                "arms": " ".join(a.id for a in table.for_case(case, args.plan)),
                "g0_prep": str(prep_dir(args.root, case, g0_door_key(case))),
                "g0_prep_key": g0_door_key(case),
            }
            if arm is not None:
                values.update({
                    "start": iso(arm.start(case)), "start_hour": f"{arm.start(case).hour:02d}",
                    "run_seconds": str(arm.run_seconds(args.plan)), "cards": str(arm.cards),
                    "min_cards": str(arm.min_cards), "lead_class": arm.lead_class,
                    "column": arm.column, "heated": "1" if arm.heated else "0",
                    "windows": str(arm.heating.get("windows", "")),
                    "needs": " ".join(arm.needs(case, args.plan)),
                    "prep_key": arm.prep_key(case, args.plan),
                    "prep": str(prep_dir(args.root, case, arm.prep_key(case, args.plan))),
                })
            if args.name not in values:
                raise PlanError(f"unknown field {args.name!r}; known: {sorted(values)}")
            print(values[args.name])
        elif args.command == "prep-inputs":
            case = case_by_id(args.case)
            for path in prep_inputs(args.root, case, args.key):
                print(path)
        elif args.command == "check-windows":
            case = case_by_id(args.case)
            record = check_arm_windows(args.root, case, table.arm(args.arm))
            text = json.dumps(record, indent=2) + "\n"
            if args.out:
                args.out.write_text(text, encoding="utf-8")
            sys.stdout.write(text)
    except PlanError as error:
        print(f"refused: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
