"""Bounded regional CWP acquisition through the existing native GOES door.

Transport lists/fetches native records, then calls the existing pack, join,
QC and grid owners. No granule decoder, brightness-temperature operator or
assumed East/West assignment lives here. Each product keeps its actual
navigation. Overlap is resolved once per model column, never averaged as
independent satellite observations.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from datetime import timedelta, timezone, datetime
import hashlib
import json
from pathlib import Path
import tempfile
import time
from typing import Callable

import numpy as np

from gpuwm.obs.goes_cwp import (CwpErrorModel, GriddedCwp, SuperobPolicy,
    grid_cwp, join_cloud_top, read_cloudtop_pack, read_cwp_pack, no_join_receipt)
from gpuwm.obs.goes_cwp_policy import (
    ERROR_BASIS, default_errors, stamp, utc,
)

ACQUISITION_SCHEMA = "arwen.regional-cwp-window.v1"
LIST_SCHEMA = "gpuwm-obs.goes-list.v1"
FETCH_SCHEMA = "gpuwm-obs.goes-fetch.v1"
REQUIRED_PRODUCTS = ("COD", "CPS", "ACTP")
OPTIONAL_PRODUCTS = ("ACHA",)


@dataclass(frozen=True)
class Source:
    satellite: str
    sector: str
    mode: int

    @property
    def id(self):
        return f"{self.satellite}-{self.sector}-M{self.mode}"


# Supported identities, not operational assignments or geographical gates.
# Real listings determine availability; decoded native pixels determine
# coverage. Retired satellites remain usable for historical cases.
SOURCES = tuple(Source(satellite, sector, mode)
                for satellite in ("G16", "G17", "G18", "G19")
                for sector in ("C", "F") for mode in (6, 3, 4))


@dataclass(frozen=True)
class AcquisitionPolicy:
    version: str = ACQUISITION_SCHEMA
    max_age_seconds: int = 1800
    # A directory-discovery margin for supported scan modes, not permission
    # to use old observations. Actual end times are screened separately.
    scan_lookback_seconds: int = 1800
    error_model: dict = field(default_factory=default_errors)
    superob: dict = field(default_factory=lambda: asdict(SuperobPolicy()))
    sources: tuple[Source, ...] = SOURCES

    def validate(self):
        if self.version != ACQUISITION_SCHEMA:
            raise ValueError("CWP policy version changed; regenerate the reviewed plan")
        for name in ("max_age_seconds", "scan_lookback_seconds"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"CWP {name} must be a positive integer")
        CwpErrorModel(**self.error_model).validate()
        SuperobPolicy(**self.superob).validate()
        if not self.sources or len({s.id for s in self.sources}) != len(self.sources):
            raise ValueError("CWP sources must be nonempty and distinct")
        for source in self.sources:
            if source not in SOURCES:
                raise ValueError(f"unsupported CWP source {source}; register its native contract first")

    def to_payload(self):
        self.validate()
        return asdict(self)

    @classmethod
    def from_payload(cls, value):
        value = dict(value)
        value["sources"] = tuple(Source(**s) for s in value.get("sources", ()))
        policy = cls(**value)
        policy.validate()
        return policy


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def asset(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}


def _json(path, value):
    """The ensemble manifest owner performs atomic publication."""
    from gpuwm.ensemble.manifest import write_json_atomically
    write_json_atomically(Path(path), value)


def _now():
    return datetime.now(timezone.utc)


def scan_id(source, scan_start):
    # Revision and mode are deliberately excluded. A revised copy of the
    # same physical scan cannot become another independent observation.
    return f"{source.satellite}/{source.sector}/{stamp(scan_start)}"


@dataclass(frozen=True)
class Scan:
    source: Source
    start: datetime
    end: datetime
    granules: dict
    publication: datetime | None

    @property
    def id(self):
        return scan_id(self.source, self.start)


class AcquisitionError(RuntimeError):
    """One automatic source failed; other sources may still serve the window."""


def select_scans(record, source, *, when, publication_cutoff, policy, used_scan_ids=()):
    """Interpret the native listing, without nominal scan-minute assumptions.

    A complete optical trio is required. ACHA is optional and must be from
    the exact same scan and available by the publication cutoff. The native
    ``complete`` flag covers its requested product set, not just the trio.
    Late but unconsumed scans may enter this window within max_age_seconds.
    """
    when, cutoff = utc(when), utc(publication_cutoff)
    if not isinstance(record, dict):
        raise AcquisitionError("native listing is not an object")
    expected = dict(schema=LIST_SCHEMA, satellite=source.satellite, sector=source.sector, mode=source.mode)
    for key, value in expected.items():
        if record.get(key) != value:
            raise AcquisitionError(f"native listing {key} is {record.get(key)!r}, expected {value!r}")
    if set(record.get("products", ())) != set(REQUIRED_PRODUCTS + OPTIONAL_PRODUCTS):
        raise AcquisitionError("native listing returned a different product set")
    used = set(used_scan_ids)
    choices, decisions = [], []
    seen = set()
    for scan in record.get("scans", ()):
        start = utc(scan["scan_start"])
        identity = scan_id(source, start)
        if identity in seen:
            raise AcquisitionError("native listing repeats a scan identity")
        seen.add(identity)
        decision = {"scan_id": identity, "scan_start": stamp(start)}
        decisions.append(decision)
        if identity in used:
            decision["status"] = "already_consumed"
            continue
        by_product = {}
        for row in scan.get("granules", ()):
            product = row["product"]
            if product not in REQUIRED_PRODUCTS + OPTIONAL_PRODUCTS or product in by_product:
                raise AcquisitionError("native scan repeats or invents a product")
            if utc(row["scan_start"]) != start or utc(row["scan_end"]) < start:
                raise AcquisitionError("native granules do not share a valid scan interval")
            prefix = f"OR_ABI-L2-{product}{source.sector}-M{source.mode}_{source.satellite}_"
            if (Path(row["key"]).name != row["filename"] or not row["filename"].startswith(prefix)
                    or int(row["size_bytes"]) <= 0):
                raise AcquisitionError("native granule identity or size contradicts its source")
            by_product[product] = dict(row)
        missing = [p for p in REQUIRED_PRODUCTS if p not in by_product]
        if missing:
            decision.update(status="incomplete", missing_products=missing)
            continue
        rows = {p: by_product[p] for p in REQUIRED_PRODUCTS}
        end = max(utc(r["scan_end"]) for r in rows.values())
        publications = [utc(r["last_modified"]) for r in rows.values() if r.get("last_modified")]
        publication = max(publications) if len(publications) == len(rows) else None
        if start > when or end > when:
            decision["status"] = "future_scan"
            continue
        if (when - end).total_seconds() > policy.max_age_seconds:
            decision["status"] = "stale"
            continue
        if any(value > cutoff for value in publications):
            decision["status"] = "published_after_cutoff"
            continue
        top = by_product.get("ACHA")
        if (top is not None and (not top.get("last_modified")
                or utc(top["last_modified"]) <= cutoff) and utc(top["scan_end"]) <= when):
            rows["ACHA"] = top
            end = max(end, utc(top["scan_end"]))
            publication = max(publication, utc(top["last_modified"])) if publication is not None and top.get("last_modified") else None
            decision["cloud_top"] = "paired_candidate"
        else:
            decision["cloud_top"] = "missing_or_not_yet_available; named fallback placement"
        decision.update(status="eligible", scan_end=stamp(end), publication_utc=stamp(publication) if publication else None)
        if publication is None:
            decision['warning'] = 'Source publication time is unknown; actual local receipt is recorded, without claiming availability at discovery start.'
        choices.append(Scan(source, start, end, rows, publication))
    choices.sort(key=lambda s: (s.end, s.start, s.id), reverse=True)
    return choices, decisions


class NativeGoes:
    """Use one reviewed executable throughout a window and record each call."""
    def __init__(self, binary, *, expected_sha256=None):
        from gpuwm.obs.frontdoor import GOES
        self.door = GOES
        self.binary = Path(binary).resolve()
        if expected_sha256 is not None and sha256(self.binary) != expected_sha256:
            raise AcquisitionError("the reviewed GOES executable changed; review this case again")
        ok, reason = GOES.probe(self.binary)
        if not ok:
            raise AcquisitionError(reason)
        self.identity = asset(self.binary)

    def call(self, subcommand, arguments, *, schema):
        if sha256(self.binary) != self.identity["sha256"]:
            raise AcquisitionError("the reviewed GOES executable changed during the window")
        return self.door.run(subcommand, arguments, schema=schema, binary=self.binary)


def _args(source, start, end, products):
    return ["--satellite", source.satellite, "--sector", source.sector,
            "--mode", str(source.mode), "--products", ",".join(products),
            "--start", stamp(start), "--end", stamp(end)]


def _fetch(scan, native, directory, cache, now):
    directory.mkdir(parents=True, exist_ok=True)
    products = tuple(scan.granules)
    # A singleton interval preserves the fractional scan start. --limit-scans
    # alone would keep the first, not the latest, scan in the native listing.
    record = native.call("fetch", _args(scan.source, scan.start, scan.start, products)
        + ["--cache", str(cache), "--out", str(directory / "granules"), "--complete-only"], schema=FETCH_SCHEMA)
    received = stamp(now())
    if not isinstance(record, dict):
        raise AcquisitionError("native fetch record is not an object")
    for key, value in (("satellite", scan.source.satellite), ("sector", scan.source.sector),
                       ("mode", scan.source.mode), ("schema", FETCH_SCHEMA)):
        if record.get(key) != value:
            raise AcquisitionError(f"GOES fetch changed {key}")
    scans = record.get("scans", ())
    if len(scans) != 1 or not scans[0].get("complete") or utc(scans[0]["scan_start"]) != scan.start:
        raise AcquisitionError("GOES fetch did not return exactly the selected complete scan")
    files = scans[0].get("files", ())
    if len(files) != len(products) or {f["product"] for f in files} != set(products):
        raise AcquisitionError("GOES fetch returned a different product set")
    paths, records = {}, []
    for row in files:
        expected = scan.granules[row["product"]]
        path = Path(row["path"]).resolve()
        if (row["key"] != expected["key"] or path.name != expected["filename"]
                or row["bytes"] != expected["size_bytes"]
                or path.stat().st_size != expected["size_bytes"] or sha256(path) != row["sha256"]):
            raise AcquisitionError("GOES fetch changed the selected revision or its bytes; scan not used")
        paths[row["product"]] = path
        records.append({**asset(path), "product": row["product"], "key": row["key"],
                        "url": expected["url"], "publication_utc": expected.get("last_modified") or None,
                        # For old native size-only cache entries the original
                        # arrival is unknown. Never relabel the cache mtime.
                        "first_receipt_utc": None if row.get("cache_hit") else received,
                        "available_on_disk_utc": received,
                        "arrival_basis": "unverified cache arrival" if row.get("cache_hit") else "after native fetch returned"})
    _json(directory / "fetch.json", record)
    return paths, records


def _prove_pack(pack, scan, paths, products):
    if pack.meta["satellite"] != scan.source.satellite or pack.meta["sector"] != scan.source.sector:
        raise AcquisitionError("decoded pack navigation belongs to a different satellite or sector")
    if utc(pack.meta["scan_start"]) != scan.start:
        raise AcquisitionError("decoded pack belongs to a different scan")
    # The pack top-level end is the first source's end. The selection's
    # complete interval is the maximum of all product end times.
    if not scan.start <= utc(pack.meta["scan_end"]) <= scan.end:
        raise AcquisitionError("decoded pack interval contradicts its listing")
    sources = pack.meta.get("sources", ())
    if len(sources) != len(products) or {r["product"] for r in sources} != set(products):
        raise AcquisitionError("decoded pack has different source products")
    for row in sources:
        path = paths[row["product"]]
        if row["filename"] != path.name or row["sha256"] != sha256(path):
            raise AcquisitionError("decoded pack source digest does not match the fetched bytes")


def _build(scan, native, directory, cache, grid, policy, now):
    directory.mkdir(parents=True, exist_ok=True)
    # Fetch the required trio independently. Optional placement transport
    # must not make a complete optical retrieval unavailable.
    core = {p: scan.granules[p] for p in REQUIRED_PRODUCTS}
    core_scan = replace(scan, granules=core,
        end=max(utc(r["scan_end"]) for r in core.values()),
        publication=max(utc(r["last_modified"]) for r in core.values()) if all(r.get("last_modified") for r in core.values()) else None)
    paths, records = _fetch(core_scan, native, directory / "optical", cache, now)
    cwp_path = directory / "cwp.goespack"
    built = native.call("cwp", ["--cod", str(paths["COD"]), "--cps", str(paths["CPS"]),
        "--actp", str(paths["ACTP"]), "--out", str(cwp_path)], schema="gpuwm-obs.goes-cwp-build.v1")
    _json(directory / "cwp-build.json", built)
    pack = read_cwp_pack(cwp_path)
    _prove_pack(pack, core_scan, paths, REQUIRED_PRODUCTS)
    heights, join = None, no_join_receipt("no causally eligible ACHA product for this scan")
    top_path = None
    top_failure = None
    top = None
    used_end, used_publication = core_scan.end, core_scan.publication
    if "ACHA" in scan.granules:
        top_path = directory / "cloudtop.goespack"
        try:
            top_scan = replace(scan, granules={"ACHA": scan.granules["ACHA"]})
            top_paths, top_records = _fetch(top_scan, native, directory / "placement", cache, now)
            paths.update(top_paths)
            top_record = native.call("cloud-top", ["--acha", str(paths["ACHA"]),
                "--pairs-with", str(cwp_path), "--out", str(top_path)], schema="gpuwm-obs.goes-cloudtop-build.v1")
            _json(directory / "cloudtop-build.json", top_record)
            top = read_cloudtop_pack(top_path)
            _prove_pack(top, scan, paths, ("ACHA",))
            heights, join = join_cloud_top(pack, top, method="nearest")
            records.extend(top_records)
            used_end, used_publication = scan.end, scan.publication
        except (OSError, RuntimeError, ValueError) as exc:
            # Optional placement can fail without turning missing water-path
            # retrievals into observations. The CWP trio is still required.
            top_failure = str(exc)
            heights, join = None, no_join_receipt("ACHA unusable: " + top_failure)
    out = grid_cwp(pack, grid, error_model=CwpErrorModel(**policy.error_model),
        policy=SuperobPolicy(**policy.superob), cloud_top_m=heights, join_receipt=join)
    out.provenance["error_model"]["initial_settings_basis"] = ERROR_BASIS
    mask = out.cwp_mask.astype(bool)
    provenance = {"scan_id": scan.id, "satellite": scan.source.satellite, "sector": scan.source.sector,
        "mode": scan.source.mode, "scan_start": stamp(scan.start), "scan_end": stamp(used_end),
        "optical_scan_end": stamp(core_scan.end), "publication_utc": stamp(used_publication) if used_publication else None, "source_files": records,
        "publication_basis": "provider times" if used_publication else "provider time incomplete; actual local receipt retained",
        "pack": pack.provenance(), "cloud_top_failure": top_failure,
        "gridding": out.provenance, "eligible_columns": int(np.count_nonzero(mask)),
        "cloud_top_pack": top.provenance() if heights is not None and top is not None else None}
    return out, provenance


FIELDS = ("cwp_obs", "cwp_mask", "cwp_err", "cwp_class", "cwp_count", "cwp_pixels", "cloud_top_height_m", "obs_level")


class Mosaic:
    """Select one existing observation at each exact column through Rust.

    Candidates are ordered by optical end, then stable source identity.
    The native nearest-plan tie rule picks the first co-located source.
    A column identity check prevents a missing column borrowing a neighbour.
    Values and errors are selected, never averaged or interpolated.
    """
    def __init__(self, grid):
        self.grid = grid
        self.offers = []

    def offer(self, product, record):
        self.offers.append((product, record))

    def finish(self, acquisition):
        from gpuwm import obs_regrid_bridge as native
        grid = self.grid
        shape = (grid.ny, grid.nx)
        fields = dict(cwp_obs=np.zeros(shape), cwp_mask=np.zeros(shape, np.int8), cwp_err=np.zeros(shape),
            cwp_class=np.full(shape, -1, np.int8), cwp_count=np.zeros(shape, np.int32), cwp_pixels=np.zeros(shape, np.int32),
            cloud_top_height_m=np.full(shape, np.nan), obs_level=np.full(shape, -1, np.int32))
        owner = np.full(shape, -1, np.int64)
        offers = sorted(self.offers, key=lambda item: (-utc(item[1]["optical_scan_end"]).timestamp(), item[1]["scan_id"]))
        positions = [np.flatnonzero(product.cwp_mask) for product, _ in offers]
        if positions and sum(len(values) for values in positions):
            cells = np.concatenate(positions)
            source_owner = np.concatenate([np.full(len(values), i) for i, values in enumerate(positions)])
            source_shape = (1, len(cells))
            lat = np.asarray(grid.lat).reshape(-1)[cells].reshape(source_shape)
            lon = np.asarray(grid.lon).reshape(-1)[cells].reshape(source_shape)
            index, reachable, _ = native.build_plan(method="nearest", source_latitude=lat,
                source_longitude=lon, destination_latitude=grid.lat,
                destination_longitude=grid.lon, max_distance_m=1.)
            # The positive search radius is an API requirement, not a spatial
            # tolerance: only the identical target-column identity can survive.
            reachable &= cells[index] == np.arange(grid.ny * grid.nx).reshape(shape)
            def selected(values, validity=None):
                values = np.asarray(values).reshape(source_shape)
                valid = np.ones(source_shape, bool) if validity is None else np.asarray(validity).reshape(source_shape)
                return native.apply_plan(method="nearest", source_index=index, reachable=reachable,
                    source_shape=source_shape, destination_shape=shape, values=values, valid=valid)
            owner_values, owner_mask = selected(source_owner)
            owner[owner_mask] = owner_values[owner_mask].astype(np.int64)
            for name in FIELDS:
                values = np.concatenate([np.asarray(getattr(product, name)).reshape(-1)[indices]
                    for (product, _), indices in zip(offers, positions)])
                # Missing optional cloud-top placement remains missing.
                mapped, valid = selected(values, np.isfinite(values))
                fields[name][valid] = mapped[valid]
        contributing = []
        for i in sorted(int(i) for i in np.unique(owner) if i >= 0):
            record = dict(offers[i][1], owner_index=i, selected_columns=int(np.count_nonzero(owner == i)))
            contributing.append(record)
        acquisition["scans"] = contributing
        acquisition["consumed_scan_ids"] = sorted({r["scan_id"] for r in contributing})
        flat = owner.reshape(-1)
        edges = np.r_[0, np.flatnonzero(np.diff(flat)) + 1, flat.size]
        acquisition["column_owners_rle"] = [[int(a), int(b-a), int(flat[a])]
            for a, b in zip(edges[:-1], edges[1:]) if flat[a] >= 0]
        mask = fields["cwp_mask"].astype(bool)
        counts = {"observations": int(mask.sum()),
            **{"observations_" + name: int(np.count_nonzero(mask & (fields["cwp_class"] == klass)))
               for klass, name in ((0, "clear"), (1, "liquid"), (2, "ice"))}}
        provenance = {"acquisition": acquisition, "counts": counts,
            "join": {"method": "per_source_nearest", "sources": [r["gridding"]["join"] for r in contributing]},
            "error_model": {**CwpErrorModel(**acquisition["policy"]["error_model"]).to_payload(), "initial_settings_basis": ERROR_BASIS},
            "superob": SuperobPolicy(**acquisition["policy"]["superob"]).to_payload(),
            "dqf_policy": [{"scan_id": r["scan_id"], "products": r["gridding"]["dqf_policy"]} for r in contributing]}
        return GriddedCwp(**fields, counts=counts, provenance=provenance)


@dataclass
class WindowResult:
    path: Path | None
    manifest_path: Path
    receipt: dict
    assets: list[dict]


def acquire_window(*, grid, when, directory, cache, native, policy=None,
                   used_scan_ids=(), now: Callable = _now, write_grid=None):
    """Create an owned output generation; the caller freezes its manifest.

    Source publication is cut at discovery start. Actual local receipt and
    completion are recorded afterwards, so a retrospective download is not
    labelled as information that was locally available at the analysis time.
    No sleeps, cadence changes, forecast execution or model resizing occur.
    """
    policy = AcquisitionPolicy() if policy is None else policy
    policy.validate()
    when, started = utc(when), time.monotonic()
    cutoff = utc(now())
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    generation = Path(tempfile.mkdtemp(prefix="generation-", dir=directory)).resolve()
    cache = Path(cache).resolve()
    receipt = {"schema": ACQUISITION_SCHEMA, "assigned_analysis_time": stamp(when),
        "grid_identity_sha256": grid.identity_sha256(), "policy": policy.to_payload(),
        "publication_cutoff_utc": stamp(cutoff), "provider": getattr(native, "identity", None),
        "selection": "newest valid scan per source, older fallback only after empty/failing scans; one winner per column",
        "overlap": "newer optical scan end, then stable source id; exact-column Rust selection, no independent duplicates",
        "max_age_seconds": policy.max_age_seconds,
        "time_operator": "3D analysis at analysis time; scan interval and age retained, no trajectory interpolation",
        "coverage": "native decoded geolocation and target-grid superob coverage, not named operational assignments",
        "sources": []}
    mosaic = Mosaic(grid)
    consumed_here = set(used_scan_ids)
    for source in sorted(policy.sources, key=lambda s: s.id):
        item = {"source": asdict(source), "status": "listing", "attempts": []}
        receipt["sources"].append(item)
        source_dir = generation / source.id
        source_dir.mkdir()
        try:
            listed = native.call("list", _args(source,
                when - timedelta(seconds=policy.max_age_seconds + policy.scan_lookback_seconds), when,
                REQUIRED_PRODUCTS + OPTIONAL_PRODUCTS), schema=LIST_SCHEMA)
            _json(source_dir / "listing.json", listed)
            choices, decisions = select_scans(listed, source, when=when, publication_cutoff=cutoff,
                policy=policy, used_scan_ids=consumed_here)
            item["decisions"] = decisions
            item["status"] = "empty"
            for index, scan in enumerate(choices):
                attempt = {"scan_id": scan.id, "status": "building"}
                item["attempts"].append(attempt)
                try:
                    out, evidence = _build(scan, native, source_dir / f"scan-{index:03d}", cache, grid, policy, now)
                    attempt.update(status="empty" if not np.any(out.cwp_mask) else "candidate", counts=out.counts)
                    if np.any(out.cwp_mask):
                        mosaic.offer(out, evidence)
                        consumed_here.add(scan.id)
                        item["status"] = "candidate"
                        break
                except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
                    attempt.update(status="unavailable", reason=f"{type(exc).__name__}: {exc}")
                    item["status"] = "unavailable"
        except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
            item.update(status="unavailable", reason=f"{type(exc).__name__}: {exc}")
    receipt["information_cutoff_utc"] = stamp(now())
    product = mosaic.finish(receipt)
    receipt["observed_columns"] = int(np.count_nonzero(product.cwp_mask))
    receipt["status"] = ("ready" if receipt["observed_columns"] else
        "unavailable" if all(s["status"] == "unavailable" for s in receipt["sources"]) else "empty")
    receipt["max_observation_age_seconds"] = max(((when - utc(r["optical_scan_end"])).total_seconds() for r in receipt["scans"]), default=None)
    lag = (utc(receipt["information_cutoff_utc"]) - when).total_seconds()
    receipt["latency_behind_real_time_s"] = lag
    # Same vocabulary and thresholds as Global FetchRecord. Do not import
    # the global spectral/radiance dependency graph to classify one number.
    receipt["latency_class"] = "fast" if lag <= 3600 else "replay" if lag <= 86400 else "retrospective"
    output = None
    if receipt["observed_columns"]:
        if write_grid is None:
            from gpuwm.obs.goes_grid import write_goes_grid
            write_grid = write_goes_grid
        output = generation / "cwp.nc"
        receipt["product"] = write_grid(output, product, grid, valid_time=stamp(when))
        if not output.is_file():
            raise AcquisitionError("the GOES grid owner returned without publishing a product")
    # Include native raw inputs, listings and packs, not just the final grid.
    # Uncommitted failed generations are never substituted for frozen inputs.
    assets = [asset(p) for p in sorted(generation.rglob("*")) if p.is_file()]
    receipt["assets"] = assets
    receipt["completed_utc"] = stamp(now())
    receipt["wall_seconds"] = time.monotonic() - started
    receipt["timing_scope"] = "through product publication and asset hashing; final JSON seal excluded"
    manifest = generation / "acquisition.json"
    _json(manifest, receipt)
    return WindowResult(output, manifest, receipt, [*assets, asset(manifest)])


def cwp_operator_refusal(state, setup, run_cfg):
    """Prove the existing column operator before auto acquisition is enabled.

    Unsupported species are not replaced with zeros or a different operator.
    Automatic CWP is reported unavailable; other observation routes continue.
    Explicit incompatible satellite input is refused by the preparation caller.
    """
    from gpuwm.da.obsop_cwp import checkpoint_cwp_provider, CwpOperatorError
    try:
        provider = checkpoint_cwp_provider(run_cfg,
            **{key: setup[key] for key in ("c1h", "c2h", "dnw", "mub2d")})
        provider(0, state)
    except CwpOperatorError as exc:
        return ("CWP column operator is unavailable: " + str(exc)
                + " Use a configuration carrying the required condensate species, "
                "or a separately reviewed phase composition; other streams remain usable.")
    return None
