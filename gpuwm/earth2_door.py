"""One canonical Earth-2 job, shared by the CLI and job-form worker.

Only orchestration belongs here. Model identities and cadence come from the
Earth-2 authority; acquisition, cyclone placement, ensembles and products
remain owned by their existing doors.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

from gpuwm.earth2_run import TABLE, model_table, resolve_model

SCHEMA = "woof-earth2-job-v1"
SCHEMA_FILE = TABLE.with_name("woof-earth2-job.v1.schema.json")


def canonical_json(spec: dict) -> str:
    return json.dumps(spec, sort_keys=True, indent=2, allow_nan=False) + "\n"


def _named_row(name, rows, *, kind):
    wanted = str(name or "").strip().lower()
    for key, row in rows.items():
        if wanted in (key.lower(), *(str(a).lower() for a in row.get("aliases", ()))):
            return key, row
    raise ValueError(f"Unknown {kind} {name!r}; valid names: " + ", ".join(sorted(rows)))


def diagnostic_rows(table=None):
    table = model_table() if table is None else table
    return {key: row for key, row in table["models"].items()
            if row.get("kind") == "diagnostic"}


def _forecast(name):
    table = model_table()
    rows = {key: row for key, row in table["models"].items()
            if row.get("kind", "prognostic") != "diagnostic"}
    key, row = _named_row(name, rows, kind="model")
    # Reuse the family's profile, account and weight refusals.
    resolve_model(key)
    return key, row


def _source(name):
    table = model_table()
    wanted = str(name).strip().lower()
    rows = {k: r for k, r in table["models"].items() if r.get("kind") != "diagnostic"}
    for key, row in rows.items():
        names = (key, *row.get("aliases", ()))
        if wanted in names or wanted in tuple("earth2-" + n for n in names):
            return _forecast(key)[0]
    from gpuwm.source_adapters import get_source_adapter, source_adapters
    try:
        adapter = get_source_adapter(wanted)
    except ValueError:
        adapter = None
    if adapter is not None:
        return adapter.source_id
    valid = sorted(set(rows) | {a.source_id for a in source_adapters()})
    raise ValueError(f"Unknown source {name!r}; valid names: " + ", ".join(valid))


def _positive_int(value, label):
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a positive integer")
    try:
        n = int(value)
    except (ValueError, TypeError, OverflowError):
        raise ValueError(f"{label} must be a positive integer") from None
    if n <= 0 or str(value).strip() not in (str(n), f"{n}.0"):
        raise ValueError(f"{label} must be a positive integer")
    return n


def _cycle(value):
    if not value or str(value).lower() == "latest":
        raise ValueError("--cycle must be an explicit UTC initialization, YYYY-MM-DDTHH")
    try:
        moment = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        raise ValueError("--cycle must be an explicit UTC initialization, YYYY-MM-DDTHH") from None
    moment = moment.replace(tzinfo=timezone.utc) if moment.tzinfo is None else moment.astimezone(timezone.utc)
    if moment.minute or moment.second or moment.microsecond:
        raise ValueError("--cycle must name a whole UTC hour")
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


def _list(value):
    return [value] if isinstance(value, str) else list(value or ())


def build_spec(*, model, cycle, hours=24, ic_source="era5", storm=None,
               nest=None, nest_km=None, area=None, ensemble=None, mix=None,
               diagnose=None, export=None, card=None) -> dict:
    """Build a validated job; serialise with :func:`canonical_json`.

    ``nest`` accepts the CLI's ``2.5km`` spelling. ``nest_km`` accepts the
    job form's numeric spacing. ``mix`` accepts ``a:n,b:m`` or row dicts.
    No directories, downloads or GPU contexts are created here.
    """
    model, row = _forecast(model)
    hours = _positive_int(hours, "--hours")
    table = model_table()
    ic_source, _ = _named_row(ic_source, table["runtime"]["initial_condition_sources"], kind="initial-condition source")
    if nest is not None and nest_km is not None:
        raise ValueError("Supply nest or nest_km once")
    spacing = nest if nest is not None else nest_km
    if spacing is not None:
        try:
            spacing = float(str(spacing).removesuffix("km"))
        except (ValueError, TypeError):
            raise ValueError("--nest must be a positive spacing in km, e.g. 2.5km") from None
        if not math.isfinite(spacing) or spacing <= 0:
            raise ValueError("--nest must be a positive spacing in km")
    if area is not None:
        area = area.split(",") if isinstance(area, str) else area
        try:
            area = [float(v) for v in area]
        except (ValueError, TypeError):
            raise ValueError("--area must be south,west,north,east") from None
        if (len(area) != 4 or not all(math.isfinite(v) for v in area)
                or not -90 <= area[0] < area[2] <= 90
                or not all(-180 <= area[i] <= 180 for i in (1, 3)) or area[1] >= area[3]):
            raise ValueError("--area needs ordered south,west,north,east bounds within +/-90 and +/-180")
    members = None if ensemble is None else _positive_int(ensemble, "--ensemble")
    if mix is not None:
        if isinstance(mix, str):
            try:
                mix = [dict(zip(("source", "members"), s.rsplit(":", 1), strict=True))
                       for s in mix.split(",")]
            except ValueError:
                raise ValueError("--mix needs source:members entries separated by commas") from None
        counts = {}
        for item in mix:
            source = _source(item["source"])
            counts[source] = counts.get(source, 0) + _positive_int(item["members"], "mix members")
        if not counts:
            raise ValueError("--mix needs at least one source:members entry")
        mix = [{"source": k, "members": n} for k, n in sorted(counts.items())]
        total = sum(counts.values())
        if members is not None and members != total:
            raise ValueError(f"--ensemble {members} disagrees with --mix total {total}")
        members = total
    elif members is not None:
        mix = [{"source": model, "members": members}]
    diagnostics = sorted({_named_row(v, diagnostic_rows(table), kind="diagnostic")[0]
                          for v in _list(diagnose)})
    exports = sorted(set(_list(export)))
    if set(exports) - {"earth2"}:
        raise ValueError("Unknown export; valid names: earth2")
    if storm is not None:
        storm = str(storm).strip()
        if not storm:
            raise ValueError("--storm must be auto, a storm name or an ATCF id")
        if storm.lower() == "auto":
            storm = "auto"
    spec = dict(schema=SCHEMA, model=model, ic_source=ic_source, cycle=_cycle(cycle),
                hours=hours, storm=storm, nest_km=spacing, area=area,
                ensemble=None if members is None else dict(members=members, mix=mix),
                diagnose=diagnostics, export=exports,
                table=dict(file=TABLE.name, sha256=hashlib.sha256(TABLE.read_bytes()).hexdigest()))
    if card is not None:
        from gpuwm.domain_wizard import declared_card_gib
        card = str(card).strip()
        declared_card_gib(card)
        spec["card"] = card
    return spec


def load_spec(path):
    """Validate saved jobs as strictly as CLI input, including table identity."""
    import jsonschema
    spec = json.loads(Path(path).read_text(encoding="utf-8"))
    try:
        jsonschema.Draft202012Validator(json.loads(SCHEMA_FILE.read_text())).validate(spec)
    except jsonschema.ValidationError as error:
        raise ValueError(f"Invalid JobSpec at {'.'.join(map(str, error.path))}: {error.message}") from None
    options = {k: spec[k] for k in ("model", "cycle", "hours", "ic_source", "storm", "nest_km", "area", "diagnose", "export")}
    if "card" in spec:
        options["card"] = spec["card"]
    if spec["ensemble"] is not None:
        options.update(ensemble=spec["ensemble"]["members"], mix=spec["ensemble"]["mix"])
    rebuilt = build_spec(**options)
    if spec["table"] != rebuilt["table"]:
        raise ValueError("JobSpec table SHA256 differs from the installed Earth-2 table; rebuild the spec against this table before running")
    if spec != rebuilt:
        raise ValueError("JobSpec is not canonical; resolve aliases and sort diagnostics/mix through build_spec")
    return rebuilt


def register_cli(subparsers):
    table = model_table()
    parser = subparsers.add_parser("earth2", help="run an Earth-2 forecast and optional moving nest",
        description="Earth-2 model rows: " + ", ".join(sorted(table["models"])))
    parser.add_argument("--model", help="table id or alias")
    parser.add_argument("--ic-source", default=None, help="initial conditions: " + ", ".join(table["runtime"]["initial_condition_sources"]))
    parser.add_argument("--cycle", help="explicit initialization in UTC, YYYY-MM-DDTHH")
    parser.add_argument("--hours", type=int, default=None)
    parser.add_argument("--storm", help="auto, ATCF id or storm name; auto ranks cycle strength within --area or the configured basin")
    parser.add_argument("--nest", help="inner grid spacing, e.g. 2.5km")
    parser.add_argument("--area", help="south,west,north,east; without --nest, author a 3 km regional domain")
    parser.add_argument("--card", help="planning card name or capacity, passed to domain/cyclone-setup")
    parser.add_argument("--ensemble", type=int)
    parser.add_argument("--mix", help="parent sources and counts, e.g. source:18,source:18")
    parser.add_argument("--diagnose", action="append", help="diagnostic table rows: " + ", ".join(sorted(diagnostic_rows(table))))
    parser.add_argument("--export", action="append", choices=("earth2",))
    parser.add_argument("--dry-run", action="store_true", help="check the job against every refusal the real run applies and write its canonical JobSpec, without fetching or running")
    parser.add_argument("--spec-out", type=Path)
    parser.add_argument("--spec", type=Path)
    parser.add_argument("--earth2-input", type=Path,
                        help="reuse a completed fetch directory after checking model, cycle, horizon, coverage and artifact hashes; auto placement searches a regional input's area")
    parser.add_argument("--out", type=Path, default=Path("earth2-output"))
    from gpuwm.geog_assets import add_geog_root_argument
    add_geog_root_argument(parser, help="geography tree the regional or nest preparation reads "
                           "(default: the staged WPS_GEOG tree); a dry run checks the named tree")
    parser.set_defaults(func=main)


def main(args):
    try:
        keys = ("model", "cycle", "hours", "ic_source", "storm", "nest", "area", "ensemble", "mix", "diagnose", "export", "card")
        options = {k: getattr(args, k) for k in keys if getattr(args, k) is not None}
        if args.spec:
            if options:
                raise ValueError("--spec cannot be combined with job flags; edit or rebuild the saved job")
            spec = load_spec(args.spec)
        else:
            spec = build_spec(**options)
        if args.dry_run:
            # A dry run answers "would this job run?": it applies the same refusals as the real run
            # (licence per --diagnose row, runtime refusals, account-gated weights, nest needs storm,
            # required modules and binaries, the ensemble mix, the regional --area requirement), so a
            # clean dry run is not followed by a refused real run. These checks do no I/O beyond lookups.
            from gpuwm.earth2_job import _preflight
            _preflight(spec)
            geog_root = getattr(args, "geog_root", None)
            if geog_root is not None:
                # The tree the real run's prep reads, refused the way `gpuwm go` refuses it.
                from gpuwm.go_cli import geography_refusal
                refusal = geography_refusal(Path(geog_root).expanduser().resolve())
                if refusal:
                    raise ValueError(refusal)
                print(f"earth2: geography {Path(geog_root).expanduser().resolve()}", file=sys.stderr)
        document = canonical_json(spec)
        if args.spec_out:
            args.spec_out.parent.mkdir(parents=True, exist_ok=True)
            args.spec_out.write_text(document, encoding="utf-8")
        if args.dry_run:
            print(document, end="")
            return 0
        from gpuwm.earth2_job import run_job
        return run_job(spec, args.out, earth2_input=getattr(args, "earth2_input", None),
                       geog_root=getattr(args, "geog_root", None))
    except (ValueError, TypeError, KeyError, OSError, RuntimeError) as error:
        print(f"earth2: {error}", file=sys.stderr)
        return 2
