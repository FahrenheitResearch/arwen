"""CPU-only real-case input, static-data, and table preflight.

The public surface is deliberately setup-only: :func:`build_input_catalog`
hashes and decodes resolved inputs into immutable metadata, while
:func:`preflight_report` runs every independent check and returns *all*
failures in one actionable report.  This module never imports CuPy or creates
a model state, so ``gpuwm check`` fails before GPU setup.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np

from gpuwm.ingest.grib import (Era5DecodeResult, Era5Snapshot,
                               cached_era5_forcing, canonical_units,
                               inspect_grib1_envelopes, parse_vtable)
from gpuwm import data_assets
from gpuwm.ingest.horiz import _MASKED_SEARCH_RADIUS
from gpuwm.static.geog import GeogDataset


_REQUIRED_PRESSURE = frozenset({"Z", "T", "U", "V", "RH"})
_REQUIRED_SURFACE = frozenset({
    "U10", "V10", "T2", "D2", "LANDSEA", "PSFC", "SKINTEMP", "SST",
    "SNOW_EC", "ST000007", "ST007028", "ST028100", "ST100289",
    "SM000007", "SM007028", "SM028100", "SM100289",
})
_WATER_OPTIONAL_FIELDS = frozenset(("SST", "SEAICE"))
_REQUIRED_GEOG = (
    "topo_gmted2010_30s",
    "modis_landuse_20class_30s_with_lakes",
    "soiltype_top_30s",
    "soiltype_bot_30s",
    "greenfrac_fpar_modis",
    "lai_modis_10m",
    "albedo_modis",
    "maxsnowalb_modis",
    "soiltemp_1deg",
)

# Broad physical plausibility bounds.  ERA5 pressure-level RH can extend
# beyond 0..100 % in supersaturated/extrapolated columns, so the limits retain
# those legitimate values while still catching corrupt/unit-wrong fields.
_FIELD_BOUNDS: Mapping[str, tuple[float, float]] = MappingProxyType({
    "Z": (-2.0e4, 6.0e5),
    "HGT": (-2.0e3, 6.0e4),
    "T": (150.0, 350.0),
    "U": (-200.0, 200.0),
    "V": (-200.0, 200.0),
    "RH": (-10.0, 150.0),
    "U10": (-150.0, 150.0),
    "V10": (-150.0, 150.0),
    "T2": (150.0, 350.0),
    "D2": (150.0, 350.0),
    "PSFC": (1.0e4, 1.2e5),
    "PMSL": (8.0e4, 1.2e5),
    "SKINTEMP": (150.0, 350.0),
    "SST": (260.0, 330.0),
    "SEAICE": (0.0, 1.0),
    "LANDSEA": (0.0, 1.0),
    "SNOW_EC": (0.0, 100.0),
    "SOILGEO": (-2.0e4, 6.0e4),
    "ST000007": (150.0, 350.0),
    "ST007028": (150.0, 350.0),
    "ST028100": (150.0, 350.0),
    "ST100289": (150.0, 350.0),
    # GRIB simple-packing roundoff produces tiny negative values (observed
    # native ERA5 minimum -9.52e-4); values outside [-0.01, 1.1] are corrupt.
    "SM000007": (-0.01, 1.1),
    "SM007028": (-0.01, 1.1),
    "SM028100": (-0.01, 1.1),
    "SM100289": (-0.01, 1.1),
})


@dataclass(frozen=True)
class CatalogFile:
    role: str
    path: Path
    sha256: str
    size: int
    product_id: str = ""
    provenance: str = ""

    def as_dict(self) -> dict[str, object]:
        return {
            "role": self.role,
            "path": str(self.path),
            "sha256": self.sha256,
            "size": self.size,
            "product_id": self.product_id,
            "provenance": self.provenance,
        }


@dataclass(frozen=True)
class SpatialCoverage:
    shape: tuple[int, int]
    latitude_min: float
    latitude_max: float
    longitude_min: float
    longitude_max: float
    latitude_order: str
    longitude_order: str


@dataclass(frozen=True)
class CatalogMask:
    variable: str
    valid_time: datetime
    reason: str
    mask: np.ndarray

    def __post_init__(self) -> None:
        mask = np.asarray(self.mask, dtype=bool).copy()
        mask.setflags(write=False)
        object.__setattr__(self, "mask", mask)

    @property
    def count(self) -> int:
        return int(np.count_nonzero(self.mask))


@dataclass(frozen=True)
class LBCRecord:
    index: int
    start_time: datetime
    end_time: datetime
    start_seconds: float
    end_seconds: float
    delta_seconds: float


def build_lbc_records(valid_times: Sequence[datetime]) -> tuple[LBCRecord, ...]:
    """Build interval metadata from the *actual* forcing time deltas."""

    times = tuple(valid_times)
    if len(times) < 2 or any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError("LBC valid times must be strictly increasing (at least two)")
    origin = times[0]
    records = []
    for index, (start, end) in enumerate(zip(times, times[1:])):
        start_s = (start - origin).total_seconds()
        end_s = (end - origin).total_seconds()
        records.append(LBCRecord(
            index, start, end, start_s, end_s, end_s - start_s
        ))
    return tuple(records)


def colon_free_output_filename(domain_id: int, valid_time: datetime) -> str:
    """Full H_M_S WRF-style output name, portable to Windows."""

    if domain_id < 1:
        raise ValueError("domain_id must be positive")
    return valid_time.strftime(f"wrfout_d{domain_id:02d}_%Y-%m-%d_%H_%M_%S")


@dataclass(frozen=True)
class InputCatalog:
    files: tuple[CatalogFile, ...]
    product_id: str
    provenance: Mapping[str, object]
    raw_valid_times: tuple[datetime, ...]
    valid_times: tuple[datetime, ...]
    excluded_valid_times: tuple[datetime, ...]
    levels_hpa: tuple[float, ...]
    inventory: tuple[str, ...]
    units: Mapping[str, str]
    spatial_coverage: SpatialCoverage | None
    masks: Mapping[tuple[datetime, str], CatalogMask]
    snapshots: tuple[Era5Snapshot, ...]
    field_sources: Mapping[tuple[datetime, str], tuple[Path, ...]]
    #: Optional validated ``[static.highres]`` config carried from
    #: :class:`gpuwm.case_data.CaseDataConfig` so per-domain static builds
    #: reached only through the catalog (child initialization) can honour
    #: it.  ``None`` -- the default for every adapter that does not set
    #: it -- is the identity 30-arc-second path.
    static_highres: object | None = None
    #: The resolved water-temperature policy, carried for the same reason
    #: as ``static_highres``: nested-child initialization sees the catalog
    #: and not the case data, and a child that assembled its lakes under a
    #: different policy than its parent would seam at the nest boundary.
    water_temperature_policy: str | None = None
    #: The resolved ``[ingest] soil_texture_downscale`` policy, carried for
    #: exactly the reason above one line at a time: a nested child sees the
    #: catalog and not the case data, it re-ingests the SAME forcing onto a
    #: FINER grid, and a child that kept the forcing mesh in its soil while
    #: its parent did not would seam at the nest boundary in every field
    #: soil moisture drives.
    soil_texture_downscale: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "provenance",
                           MappingProxyType(dict(self.provenance)))
        object.__setattr__(self, "units", MappingProxyType(dict(self.units)))
        object.__setattr__(self, "masks", MappingProxyType(dict(self.masks)))
        object.__setattr__(self, "field_sources", MappingProxyType({
            key: tuple(paths) for key, paths in self.field_sources.items()
        }))

    @property
    def lbc_records(self) -> tuple[LBCRecord, ...]:
        return build_lbc_records(self.valid_times) if len(self.valid_times) >= 2 else ()

    @property
    def coverage_seconds(self) -> float:
        if len(self.valid_times) < 2:
            return 0.0
        return (self.valid_times[-1] - self.valid_times[0]).total_seconds()

    @property
    def run_ceiling_seconds(self) -> float:
        return self.coverage_seconds

    @property
    def fingerprint(self) -> str:
        payload = [item.as_dict() for item in sorted(
            self.files, key=lambda item: (item.role, str(item.path))
        )]
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    @property
    def file_hashes(self) -> Mapping[str, str]:
        return MappingProxyType({str(item.path): item.sha256 for item in self.files})

    @property
    def run_provenance(self) -> Mapping[str, object]:
        """JSON-ready record embedded in the future run setup fingerprint."""

        return MappingProxyType({
            "input_catalog_sha256": self.fingerprint,
            "product_id": self.product_id,
            "valid_times": tuple(t.isoformat() for t in self.valid_times),
            "files": tuple(item.as_dict() for item in self.files),
            "catalog_provenance": dict(self.provenance),
        })


@dataclass(frozen=True)
class PreflightIssue:
    code: str
    message: str
    path: Path | None = None
    variable: str | None = None
    index: tuple[int, ...] | None = None
    #: "error" fails the preflight; "warning" is REPORTED and the run
    #: proceeds.  Warnings are the advisory findings -- conservative
    #: meteorological envelopes, cadence pedantry -- where the input is
    #: readable and self-consistent but outside a blessed window.
    severity: str = "error"

    def format(self) -> str:
        address = []
        if self.path is not None:
            address.append(f"file={self.path}")
        if self.variable is not None:
            address.append(f"variable={self.variable}")
        if self.index is not None:
            address.append(f"index={self.index}")
        suffix = f" ({', '.join(address)})" if address else ""
        return f"[{self.code}] {self.message}{suffix}"


class CatalogBuildError(ValueError):
    def __init__(self, issues: Sequence[PreflightIssue]):
        self.issues = tuple(issues)
        super().__init__("input catalog failed:\n" + "\n".join(
            f"- {issue.format()}" for issue in self.issues
        ))


@dataclass(frozen=True)
class PreflightReport:
    catalog: InputCatalog
    failures: tuple[PreflightIssue, ...]
    checks: tuple[str, ...]

    @property
    def errors(self) -> tuple[PreflightIssue, ...]:
        return tuple(issue for issue in self.failures
                     if issue.severity != "warning")

    @property
    def warnings(self) -> tuple[PreflightIssue, ...]:
        return tuple(issue for issue in self.failures
                     if issue.severity == "warning")

    @property
    def ok(self) -> bool:
        # Warnings report; only errors refuse (warn-not-block).
        return not self.errors

    @property
    def passed(self) -> bool:
        return self.ok

    @property
    def run_ceiling_seconds(self) -> float:
        return self.catalog.run_ceiling_seconds

    def format(self) -> str:
        lines = [
            "gpuwm input preflight: " + ("PASS" if self.ok else "FAIL"),
            f"product={self.catalog.product_id or 'unknown'}",
            f"catalog_sha256={self.catalog.fingerprint}",
            f"valid_times={','.join(t.isoformat() for t in self.catalog.valid_times)}",
            f"run_ceiling_seconds={self.catalog.run_ceiling_seconds:g}",
        ]
        errors, warnings = self.errors, self.warnings
        if errors:
            lines.append(f"failures={len(errors)} (all independent checks ran)")
            lines.extend(f"- {issue.format()}" for issue in errors)
        else:
            lines.append(f"checks={len(self.checks)}")
        if warnings:
            lines.append(f"warnings={len(warnings)} (reported, not blocking)")
            lines.extend(f"- WARN {issue.format()}" for issue in warnings)
        return "\n".join(lines)

    def raise_for_failures(self) -> None:
        if self.errors:
            raise ValueError(self.format())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _catalog_file(role: str, path: Path, *, product_id: str = "",
                  provenance: str = "",
                  identity_path: Path | None = None) -> CatalogFile:
    resolved = path.resolve()
    identity = (resolved if identity_path is None
                else Path(identity_path).resolve())
    return CatalogFile(role, identity, _sha256(resolved),
                       resolved.stat().st_size, product_id, provenance)


def _product_id(case_data) -> str:
    declared = getattr(case_data, "product_id", None)
    if declared:
        return str(declared)
    identity = (case_data.vtable_identity()
                if hasattr(case_data, "vtable_identity")
                else case_data.vtable)
    name = Path(identity).name.upper()
    return "ERA5" if "ERA5" in name else name


def _declared_table_root(case_data) -> Path | None:
    """A caller-supplied flat table tree, or None for the packaged set.

    The native WRF distribution builds one of these
    (``tools/build_native_wrf_distribution.py``): a single directory
    holding copies of the ``gpuwm/data`` subtrees in their original
    layout.  When one is declared it answers for every asset, exactly as
    it always did.
    """

    declared = (getattr(case_data, "table_root", None)
                or getattr(case_data, "bundled_table_root", None))
    return None if declared is None else Path(declared)


def _table_root(case_data) -> Path:
    """The root a declared tree names, else this wheel's ``gpuwm/data``.

    Do not join packaged asset paths onto this: use :func:`_table_asset`.
    Since 2.5.0 two of those directories ship in the ``gpuwm-data``
    companion distribution and only the resolver knows which.
    """

    declared = _declared_table_root(case_data)
    return declared if declared is not None \
        else data_assets.package_data_root()


def _table_asset(case_data, relative: str) -> Path:
    """One ``gpuwm/data``-relative asset, at the root that holds it.

    Written as a function rather than a join so the two call sites below
    -- the existence sweep and the per-file NetCDF dimension checks --
    cannot drift apart.  They did not drift before because both joined
    the same ``root``; once ``rrtmgp/`` moved to the companion, a sweep
    that found the companion path and a check that rebuilt the in-wheel
    one would have made every dimension check silently skip, since the
    check keys on the path the sweep recorded.
    """

    declared = _declared_table_root(case_data)
    if declared is not None:
        return declared.joinpath(*relative.split("/"))
    return data_assets.data_path(relative)


def _table_asset_map(case_data, exp=None) -> dict[str, Path]:
    """``gpuwm/data``-relative key -> resolved path, in check order.

    The KEY is what ``_TABLE_SHA256`` is written against, so it travels
    with the path instead of being recovered from it.  Recovering it by
    ``path.relative_to(root)`` is what broke when ``rrtmgp/`` moved to
    the companion: the relative_to raised, the fallback used the bare
    filename, no pin matched that spelling, and four pinned SHA-256
    checks passed by finding nothing to compare.
    """

    relatives = [
        "kf_lutab/kf_lutab.npz",
        "rrtmgp/rrtmgp-gas-lw-g256.nc",
        "rrtmgp/rrtmgp-gas-sw-g224.nc",
        "rrtmgp/rrtmgp-clouds-lw-bnd.nc",
        "rrtmgp/rrtmgp-clouds-sw-bnd.nc",
        "noah_tables/LANDUSE.TBL",
        "noah_tables/VEGPARM.TBL",
        "noah_tables/SOILPARM.TBL",
        "noah_tables/GENPARM.TBL",
    ]
    if exp is not None:
        from gpuwm.config import radiation_scheme_ids
        if any(radiation_scheme_ids(dc.run)[0] == 1 for dc in exp.domains):
            relatives.append("wrf_radiation/RRTM_DATA")
    return {relative: _table_asset(case_data, relative)
            for relative in relatives}


def _table_files(case_data, exp=None) -> tuple[Path, ...]:
    return tuple(_table_asset_map(case_data, exp).values())


def _select_contiguous_times(times: Sequence[datetime], interval_s: float | None
                             ) -> tuple[tuple[datetime, ...],
                                        tuple[datetime, ...], str]:
    raw = tuple(sorted(set(times)))
    if interval_s is None or len(raw) < 2:
        return raw, (), "all decoded valid times"
    runs: list[list[datetime]] = [[raw[0]]]
    for value in raw[1:]:
        delta = (value - runs[-1][-1]).total_seconds()
        if math.isclose(delta, interval_s, rel_tol=0.0, abs_tol=1.0e-9):
            runs[-1].append(value)
        else:
            runs.append([value])
    if len(runs) == 1:
        return raw, (), f"all times match declared {interval_s:g} s interval"
    longest = max(len(run) for run in runs)
    winners = [run for run in runs if len(run) == longest]
    if len(winners) != 1:
        # Ambiguity is preserved for preflight to diagnose; do not silently
        # choose one equal-length forcing window.
        return raw, (), (f"ambiguous contiguous runs at declared "
                         f"{interval_s:g} s interval")
    selected = tuple(winners[0])
    excluded = tuple(value for value in raw if value not in selected)
    return selected, excluded, (
        f"unique longest contiguous run at declared {interval_s:g} s interval; "
        f"excluded date/time cross-product values"
    )


def _empty_catalog(product_id: str = "") -> InputCatalog:
    return InputCatalog((), product_id, {}, (), (), (), (), (), {}, None,
                        {}, (), {})


def _build_input_catalog(case_data) -> tuple[InputCatalog,
                                              tuple[PreflightIssue, ...]]:
    issues: list[PreflightIssue] = []
    product_id = _product_id(case_data)
    files: list[CatalogFile] = []
    forcing_hashes: dict[Path, str] = {}

    def add_file(role: str, raw_path, *, provenance: str = "",
                 identity_path=None) -> None:
        path = Path(raw_path)
        identity = Path(path if identity_path is None else identity_path)
        if not path.is_file():
            issues.append(PreflightIssue(
                "missing-file", f"required {role} file does not exist",
                path=identity,
            ))
            return
        try:
            record = _catalog_file(
                role, path, product_id=product_id if role == "forcing" else "",
                provenance=provenance, identity_path=identity,
            )
            files.append(record)
            if role == "forcing":
                forcing_hashes[path.resolve()] = record.sha256
        except Exception as exc:
            issues.append(PreflightIssue(
                "hash", f"could not SHA-256 hash {role}: {exc}", path=identity,
            ))

    forcing_paths = tuple(Path(path) for path in case_data.forcing)
    forcing_identities = tuple(Path(path) for path in (
        case_data.forcing_identity()
        if hasattr(case_data, "forcing_identity") else case_data.forcing))
    for path, identity in zip(forcing_paths, forcing_identities):
        add_file(
            "forcing", path, provenance="resolved CaseData forcing",
            identity_path=identity)
    vtable_identity = (
        case_data.vtable_identity()
        if hasattr(case_data, "vtable_identity") else case_data.vtable)
    wps_identity = (
        case_data.wps_namelist_identity()
        if hasattr(case_data, "wps_namelist_identity")
        else case_data.wps_namelist)
    add_file(
        "vtable", case_data.vtable, provenance="declared Vtable/schema",
        identity_path=vtable_identity)
    add_file("wps_namelist", case_data.wps_namelist,
             provenance="declared WPS grid source",
             identity_path=wps_identity)
    source_orography = getattr(case_data, "source_orography", None)
    source_orography_identity = (
        case_data.source_orography_identity()
        if hasattr(case_data, "source_orography_identity")
        else source_orography)
    if source_orography is not None and hasattr(source_orography, "path"):
        add_file("source_orography", source_orography.path,
                 provenance=f"variable={getattr(source_orography, 'variable', '')}",
                 identity_path=source_orography_identity.path)
    water_overlay = getattr(case_data, "water_temperature_overlay", None)
    if water_overlay is not None:
        water_overlay_identity = (
            case_data.water_temperature_overlay_identity()
            if hasattr(case_data, "water_temperature_overlay_identity")
            else water_overlay)
        add_file("water_temperature_overlay", water_overlay,
                 provenance="declared hi-res water-temperature analysis",
                 identity_path=water_overlay_identity)

    # Retrieval request/log/checksum sidecars are first-class provenance when
    # all forcing products share a staging directory (the May-1999 layout).
    parents = {path.resolve().parent for path in forcing_identities
               if path.exists()}
    if len(parents) == 1:
        parent = next(iter(parents))
        for name in ("retrieve.py", "retrieve.log", "SHA256SUMS.txt"):
            sidecar = parent / name
            if sidecar.is_file():
                add_file("forcing_provenance", sidecar,
                         provenance="retrieval request/log/recorded checksum")

    # Catalog the packaged runtime tables and GEOG schemas.  Preflight adds
    # the exact footprint-intersecting tile hashes once the experiment grid is
    # available; hashing the entire 14+ GiB WPS tree is neither selection-aware
    # nor useful.
    for path in _table_files(case_data):
        add_file("bundled_table", path)
    geog_root = Path(case_data.geog_root)
    for name in getattr(case_data, "geog_datasets", _REQUIRED_GEOG):
        dataset_name = str(name)
        if isinstance(getattr(case_data, "geog_datasets", None), Mapping):
            dataset_name = str(name)
        index = geog_root / dataset_name / "index"
        if index.is_file():
            add_file("geog_index", index, provenance=f"dataset={dataset_name}")

    entries = ()
    units: Mapping[str, str] = {}
    try:
        entries = parse_vtable(case_data.vtable)
        units = canonical_units(entries)
    except Exception as exc:
        issues.append(PreflightIssue(
            "vtable", f"could not parse the declared Vtable: {exc}",
            path=Path(case_data.vtable),
        ))

    decodable: list[Path] = []
    content_hashes: list[str] = []
    for path in forcing_paths:
        resolved = path.resolve()
        if resolved not in forcing_hashes:
            continue
        try:
            inspect_grib1_envelopes(resolved)
        except Exception as exc:
            issues.append(PreflightIssue(
                "grib-encoding", str(exc), path=resolved,
            ))
            continue
        decodable.append(resolved)
        content_hashes.append(forcing_hashes[resolved])

    decoded: Era5DecodeResult | None = None
    if decodable and entries:
        try:
            decoded = cached_era5_forcing(
                decodable, case_data.vtable, content_sha256=content_hashes
            )
        except Exception as exc:
            issues.append(PreflightIssue(
                "grib-decode",
                f"could not decode/merge forcing inputs: {exc}",
                path=decodable[0] if len(decodable) == 1 else None,
            ))
    elif not decodable:
        issues.append(PreflightIssue(
            "grib-decode", "no structurally valid GRIB1 forcing file remains"
        ))

    if decoded is None:
        catalog = replace(
            _empty_catalog(product_id), files=tuple(files), units=units,
            provenance={"encoding": "native GRIB1", "decode": "failed"},
        )
        return catalog, tuple(issues)

    raw_times = tuple(snapshot.valid_time for snapshot in decoded.snapshots)
    selected_times, excluded_times, selection = _select_contiguous_times(
        raw_times, getattr(case_data, "forcing_interval_s", None)
    )
    selected_set = set(selected_times)
    snapshots = tuple(
        snapshot for snapshot in decoded.snapshots
        if snapshot.valid_time in selected_set
    )
    field_sources = {
        key: paths for key, paths in decoded.field_sources.items()
        if key[0] in selected_set
    }

    levels = (tuple(float(value) for value in snapshots[0].levels_hpa)
              if snapshots else ())
    inventory = (tuple(sorted(snapshots[0].fields)) if snapshots else ())
    coverage = None
    masks: dict[tuple[datetime, str], CatalogMask] = {}
    if snapshots:
        first = snapshots[0]
        coverage = SpatialCoverage(
            shape=(first.latitude.size, first.longitude.size),
            latitude_min=float(np.min(first.latitude)),
            latitude_max=float(np.max(first.latitude)),
            longitude_min=float(np.min(first.longitude)),
            longitude_max=float(np.max(first.longitude)),
            latitude_order=("ascending" if first.latitude[-1] > first.latitude[0]
                            else "descending"),
            longitude_order=("ascending" if first.longitude[-1] > first.longitude[0]
                             else "descending"),
        )
        for snapshot in snapshots:
            land = snapshot.fields.get("LANDSEA")
            if land is not None:
                land_mask = np.isfinite(land) & (land >= 0.5)
                masks[(snapshot.valid_time, "LANDSEA")] = CatalogMask(
                    "LANDSEA", snapshot.valid_time, "land cells (value >= 0.5)",
                    land_mask,
                )
            seaice = snapshot.fields.get("SEAICE")
            if seaice is not None:
                masks[(snapshot.valid_time, "SEAICE")] = CatalogMask(
                    "SEAICE", snapshot.valid_time,
                    "sea-ice cells (finite value >= 0.5)",
                    np.isfinite(seaice) & (seaice >= 0.5) & (seaice <= 1.0),
                )
            for name in _WATER_OPTIONAL_FIELDS:
                bitmap = decoded.bitmap_missing.get((snapshot.valid_time, name))
                if bitmap is not None and np.any(bitmap):
                    masks[(snapshot.valid_time, name + "_MISSING")] = CatalogMask(
                        name, snapshot.valid_time,
                        "native GRIB bitmap missing values",
                        bitmap,
                    )

    provenance = {
        "product_id": product_id,
        "encoding": "native GRIB1 (vendored grib-core 0.1.0)",
        "vtable_schema": str(Path(vtable_identity).resolve()),
        "time_selection": selection,
        "raw_valid_times": tuple(value.isoformat() for value in raw_times),
        "excluded_valid_times": tuple(value.isoformat()
                                      for value in excluded_times),
    }
    catalog = InputCatalog(
        files=tuple(files), product_id=product_id, provenance=provenance,
        raw_valid_times=raw_times, valid_times=selected_times,
        excluded_valid_times=excluded_times, levels_hpa=levels,
        inventory=inventory, units=units, spatial_coverage=coverage,
        masks=masks, snapshots=snapshots, field_sources=field_sources,
    )
    return catalog, tuple(issues)


def _missing_required_inventory(catalog: InputCatalog) -> list[str]:
    """Required forcing variables absent from the decoded inventory.

    One definition, two callers: the run-start gate in
    :func:`build_input_catalog` and the aggregated report's forcing scan.
    """

    return sorted((_REQUIRED_PRESSURE | _REQUIRED_SURFACE)
                  - set(catalog.inventory))


def build_input_catalog(case_data) -> InputCatalog:
    """Build a hashed, decoded catalog from resolved :class:`CaseDataConfig`.

    Structural/hash/decode defects are aggregated in :class:`CatalogBuildError`.
    Full experiment-dependent checks (coverage, p_top, GEOG, orography, tables)
    are performed by :func:`preflight_report`.

    This IS the run path's ingest gate -- ``gpuwm run`` calls it and never
    calls :func:`preflight_report`, which is reachable only through
    ``gpuwm check``.  So the one class of defect that can never produce a
    forecast, a forcing product MISSING REQUIRED VARIABLES, is refused
    here by name rather than surfacing hundreds of lines downstream as a
    ``KeyError`` from whichever consumer happened to index the absent
    field first (soil preprocessing on the single-domain path, the
    D2/RH2 surface-humidity choice on the multi-domain one).

    Deliberately NARROW.  Only absence is refused.  Every judgement call
    about the VALUES -- finiteness, meteorological bounds, spatial and
    temporal coverage -- stays advisory in ``gpuwm check``, because a run
    that would almost certainly have worked must not be blocked by a
    threshold nobody agreed to.  Absence is not a threshold: the variables
    are the ones the initializer unconditionally indexes.
    """

    catalog, issues = _build_input_catalog(case_data)
    missing = _missing_required_inventory(catalog) if not issues else []
    if missing:
        issues = (*issues, PreflightIssue(
            "inventory",
            f"forcing inventory is missing {missing}; the declared forcing "
            "product supplies no such variable at any valid time. ERA5 "
            "single-level (surface) and pressure-level products are "
            "separate CDS downloads -- declare both, or one file "
            "containing both. `gpuwm check` reports the full input estate."))
    if issues:
        raise CatalogBuildError(issues)
    # Optional hi-res water-temperature overlay (task #71): the catalog's
    # SERVED snapshots carry it, because nested-child initialization reads
    # its source snapshot from here.  The raw-decode integrity scans above
    # ran on the raw snapshots; the overlay file itself is already a
    # hashed catalog record, so the run identity moves with the data.
    # Absent key: this branch never runs and the catalog is untouched.
    water_overlay_path = getattr(
        case_data, "water_temperature_overlay", None)
    if water_overlay_path is not None:
        from gpuwm.ingest.water_overlay import (
            cached_water_temperature_overlay,
            overlay_snapshots,
        )

        overlay = cached_water_temperature_overlay(water_overlay_path)
        replaced, _receipt = overlay_snapshots(catalog.snapshots, overlay)
        catalog = replace(catalog, snapshots=replaced)
    # Optional [static.highres] config rides the catalog so child-domain
    # static builds (which see only the catalog) can honour it.  Absent
    # key: the catalog is untouched and the 30s path is byte-identical.
    highres = getattr(case_data, "static_highres", None)
    if highres is not None:
        catalog = replace(catalog, static_highres=highres)
    from gpuwm.ingest.water_temperature import (
        resolve_water_temperature_policy)
    from gpuwm.ingest.soil_downscale import declared_soil_texture_downscale
    catalog = replace(
        catalog,
        water_temperature_policy=resolve_water_temperature_policy(case_data),
        soil_texture_downscale=declared_soil_texture_downscale(case_data))
    return catalog


def _first_index(mask: np.ndarray) -> tuple[int, ...]:
    return tuple(int(value) for value in np.argwhere(mask)[0])


def _source_for(catalog: InputCatalog, valid_time: datetime,
                variable: str) -> Path | None:
    paths = catalog.field_sources.get((valid_time, variable), ())
    return paths[0] if paths else None


def _land_or_coastal_support(land_mask: np.ndarray) -> np.ndarray:
    """Land plus the frozen interpolator's coastal search neighborhood."""

    land = np.asarray(land_mask, dtype=bool)
    if land.ndim != 2:
        raise ValueError("LANDSEA mask must be two-dimensional")
    ny, nx = land.shape
    radius = _MASKED_SEARCH_RADIUS
    padded = np.pad(land, radius, mode="constant", constant_values=False)
    return np.logical_or.reduce(tuple(
        padded[dj:dj + ny, di:di + nx]
        for dj in range(2 * radius + 1) for di in range(2 * radius + 1)
    ))


def _stable_native_bitmap_fields(catalog: InputCatalog) -> set[str]:
    """Water-optional fields whose native missing mask is time invariant."""

    stable: set[str] = set()
    for name in _WATER_OPTIONAL_FIELDS:
        by_time: list[np.ndarray] = []
        for snapshot in catalog.snapshots:
            value = snapshot.fields.get(name)
            if value is None:
                by_time = []
                break
            record = catalog.masks.get((snapshot.valid_time, name + "_MISSING"))
            by_time.append(
                record.mask if record is not None else np.zeros(value.shape, dtype=bool)
            )
        if by_time and any(np.any(mask) for mask in by_time):
            reference = by_time[0]
            if all(np.array_equal(mask, reference) for mask in by_time[1:]):
                stable.add(name)
    return stable


def _scan_forcing(catalog: InputCatalog) -> tuple[list[PreflightIssue], list[str]]:
    issues: list[PreflightIssue] = []
    checks = ["forcing inventory", "forcing finiteness",
              "meteorological bounds", "source masks"]
    missing = _missing_required_inventory(catalog)
    if missing:
        issues.append(PreflightIssue(
            "inventory", f"forcing inventory is missing {missing}"
        ))

    stable_native_bitmaps = _stable_native_bitmap_fields(catalog)
    for snapshot in catalog.snapshots:
        if tuple(sorted(snapshot.fields)) != catalog.inventory:
            issues.append(PreflightIssue(
                "inventory",
                f"field inventory differs at {snapshot.valid_time}: "
                f"{sorted(snapshot.fields)}",
            ))
        land = snapshot.fields.get("LANDSEA")
        land_mask = (np.isfinite(land) & (land >= 0.5)
                     if land is not None else None)
        tolerated_water_missing = (
            _land_or_coastal_support(land_mask)
            if land_mask is not None else None
        )
        for name, value in snapshot.fields.items():
            finite = np.isfinite(value)
            if not np.all(finite):
                bad = ~finite
                if (name in stable_native_bitmaps
                        and tolerated_water_missing is not None):
                    # The frozen ingest path treats SST/SEAICE as undefined on
                    # land and repairs fractional-coast disagreement from a
                    # nearby finite water value (with SKINTEMP as the SST
                    # fallback).  Exempt only source-proven native bitmap
                    # cells and only when the bitmap is identical at every
                    # valid time.  SST holes are limited to land/coastal
                    # support because SKINTEMP is its fallback.  SEAICE holes
                    # may occur on open water and are explicitly zero-filled
                    # by horizontal ingest (no analyzed ice support).
                    native = catalog.masks.get(
                        (snapshot.valid_time, name + "_MISSING")
                    )
                    if native is not None:
                        tolerated = (native.mask if name == "SEAICE" else
                                     native.mask & tolerated_water_missing)
                        bad = bad & ~tolerated
                if np.any(bad):
                    index = _first_index(bad)
                    issues.append(PreflightIssue(
                        "nonfinite",
                        f"{name} at {snapshot.valid_time} contains "
                        f"{value[index]!r}",
                        path=_source_for(catalog, snapshot.valid_time, name),
                        variable=name, index=index,
                    ))
            bounds = _FIELD_BOUNDS.get(name)
            if bounds is None:
                continue
            lo, hi = bounds
            if name == "SEAICE":
                # WRF module_soil_pre.F treats values >200 as GRIB flags and
                # repairs them to zero.  Values in (1,200] remain invalid.
                active = finite & ((value < lo)
                                   | ((value > hi) & (value <= 200.0)))
            else:
                active = finite & ((value < lo) | (value > hi))
            if np.any(active):
                index = _first_index(active)
                # Advisory envelope, not a physical impossibility: the
                # field decoded, is finite, and sits outside a
                # conservative window.  Reported, never blocking.
                issues.append(PreflightIssue(
                    "meteorological-bounds",
                    f"{name} at {snapshot.valid_time} = {value[index]:g} "
                    f"outside [{lo:g}, {hi:g}]",
                    path=_source_for(catalog, snapshot.valid_time, name),
                    variable=name, index=index, severity="warning",
                ))
    return issues, checks


def _check_levels(exp, catalog: InputCatalog
                  ) -> tuple[list[PreflightIssue], list[str]]:
    issues: list[PreflightIssue] = []
    checks = ["pressure levels vs p_top", "below-surface level support"]
    levels = np.asarray(catalog.levels_hpa, dtype=np.float64)
    if levels.size == 0:
        return [PreflightIssue("levels", "forcing has no pressure levels")], checks
    if np.any(~np.isfinite(levels)) or np.any(np.diff(levels) <= 0.0):
        issues.append(PreflightIssue(
            "levels", f"pressure levels must be finite/strictly increasing: "
            f"{levels.tolist()}", variable="pressure_level",
        ))
        return issues, checks
    p_top = float(exp.vertical.p_top)
    if levels[0] * 100.0 > p_top:
        issues.append(PreflightIssue(
            "level-coverage",
            f"top pressure level {levels[0]:g} hPa does not reach p_top "
            f"{p_top:g} Pa; need a level <= {p_top / 100.0:g} hPa",
            variable="pressure_level", index=(0,),
        ))
    if levels[-1] < 1000.0:
        issues.append(PreflightIssue(
            "below-surface-support",
            f"deepest pressure level is {levels[-1]:g} hPa; ERA5 real-data "
            "initialization requires the 1000 hPa level for terrain columns",
            variable="pressure_level", index=(int(levels.size - 1),),
        ))
    return issues, checks


def _check_temporal(exp, case_data, catalog: InputCatalog
                    ) -> tuple[list[PreflightIssue], list[str]]:
    issues: list[PreflightIssue] = []
    checks = ["forecast/LBC coverage", "actual-delta LBC records",
              "whole-step output cadence", "colon-free output names"]
    times = catalog.valid_times
    start = exp.start_time
    end = start + timedelta(seconds=float(exp.run_seconds))
    if start not in times:
        issues.append(PreflightIssue(
            "missing-time",
            f"forcing is missing experiment start valid time {start}; "
            f"catalog times are {list(times)}",
        ))
    if len(times) < 2:
        issues.append(PreflightIssue(
            "lbc-coverage", "at least two forcing valid times are required"
        ))
    else:
        try:
            records = build_lbc_records(times)
            if any(record.delta_seconds <= 0.0 for record in records):
                raise ValueError("non-positive LBC interval")
        except Exception as exc:
            issues.append(PreflightIssue("lbc-records", str(exc)))
        if times[0] > start or times[-1] < end:
            issues.append(PreflightIssue(
                "run-ceiling",
                f"run_seconds={exp.run_seconds:g} needs {start} .. {end}, but "
                f"validated forcing covers {times[0]} .. {times[-1]} "
                f"({catalog.run_ceiling_seconds:g} s)",
            ))

    interval = getattr(case_data, "forcing_interval_s", None)
    if interval is not None and start in catalog.raw_valid_times:
        expected = start
        raw = set(catalog.raw_valid_times)
        while expected <= end:
            if expected not in raw:
                issues.append(PreflightIssue(
                    "missing-time",
                    f"missing forcing valid time {expected} at declared "
                    f"{float(interval):g} s interval",
                ))
            expected += timedelta(seconds=float(interval))

    for dc in exp.domains:
        # Divisibility is judged on the EXACT rational dt (T1 schema
        # authority), never the chained-FP32 kernel value: 900 / (5/3 s)
        # is exactly 540 while 900 / float32(5/3) aliases to 540.0000129.
        dt_exact = exp.dt_exact(dc.grid_id)
        cadence_exact = Fraction(dc.history_interval_s).limit_denominator(
            10**9)
        ratio_exact = cadence_exact / dt_exact
        cadence = float(dc.history_interval_s)
        if ratio_exact.denominator != 1 or ratio_exact < 1:
            issues.append(PreflightIssue(
                "output-cadence",
                f"domain d{dc.grid_id:02d} history_interval_s={cadence:g} "
                f"is not a whole number of exact dt="
                f"{dt_exact} s steps (exact ratio={ratio_exact})",
            ))
        domain_start = exp.domain_start_time(dc.grid_id)
        sample = colon_free_output_filename(dc.grid_id, domain_start)
        if ":" in sample or not sample.endswith(
                domain_start.strftime("%Y-%m-%d_%H_%M_%S")):
            issues.append(PreflightIssue(
                "output-filename",
                f"domain d{dc.grid_id:02d} output name is not full H_M_S "
                f"colon-free: {sample}",
            ))
    return issues, checks


def output_records(exp, domain_id: int) -> tuple[tuple[int, datetime, str], ...]:
    """Resolved output step/time/name records for cadence fixture gates."""

    dc = exp.domain(domain_id)
    start = exp.domain_start_time(domain_id)
    offset = float(exp.domain_start_offset_exact(domain_id))
    cadence_steps = round(float(dc.history_interval_s) / float(dc.run.dt))
    total = int(math.floor(
        (float(exp.run_seconds) - offset) / dc.history_interval_s))
    return tuple(
        (
            index * cadence_steps,
            start + timedelta(seconds=index * dc.history_interval_s),
            colon_free_output_filename(
                domain_id,
                start + timedelta(seconds=index * dc.history_interval_s),
            ),
        )
        for index in range(total + 1)
    )


def _grid_for(exp, case_data):
    # Runtime grid construction is CPU-only and already cross-checks the WPS
    # namelist against the experiment projection/dimensions.
    from gpuwm.runtime import experiment_grid
    if len(exp.domains) == 1:
        return experiment_grid(exp, case_data)
    # Multi-domain: the T2 runtime is single-domain until Task 14, but the
    # preflight only needs the ROOT grid -- external forcing feeds d01
    # exclusively, and every child's geog footprint lies inside the root's
    # at tile granularity (per-domain static windows are Task 12's).
    from gpuwm.static.lambert import grids_from_projection_config
    return grids_from_projection_config(exp)[0]


def _check_spatial(exp, case_data, catalog: InputCatalog
                   ) -> tuple[list[PreflightIssue], list[str], object | None]:
    issues: list[PreflightIssue] = []
    checks = ["forcing spatial coverage"]
    try:
        grid = _grid_for(exp, case_data)
    except Exception as exc:
        issues.append(PreflightIssue(
            "grid", f"could not resolve experiment grid for coverage: {exc}",
            path=Path(case_data.wps_namelist),
        ))
        return issues, checks, None
    if catalog.spatial_coverage is None:
        issues.append(PreflightIssue(
            "spatial-coverage", "forcing has no decoded spatial coverage"
        ))
        return issues, checks, grid

    lat, lon = grid.latlon_mass()
    coverage = catalog.spatial_coverage
    lat_bad = ((lat < coverage.latitude_min - 1.0e-9)
               | (lat > coverage.latitude_max + 1.0e-9))
    center = 0.5 * (coverage.longitude_min + coverage.longitude_max)
    mapped_lon = center + np.mod(lon - center + 180.0, 360.0) - 180.0
    lon_bad = ((mapped_lon < coverage.longitude_min - 1.0e-9)
               | (mapped_lon > coverage.longitude_max + 1.0e-9))
    bad = lat_bad | lon_bad
    if np.any(bad):
        index = _first_index(bad)
        issues.append(PreflightIssue(
            "spatial-coverage",
            f"domain point lat/lon=({lat[index]:g}, {lon[index]:g}) lies "
            f"outside forcing lat [{coverage.latitude_min:g}, "
            f"{coverage.latitude_max:g}] lon [{coverage.longitude_min:g}, "
            f"{coverage.longitude_max:g}]",
            variable="forcing_grid", index=index,
        ))
    return issues, checks, grid


def _check_orography(exp, case_data, catalog
                     ) -> tuple[list[PreflightIssue], list[str]]:
    issues: list[PreflightIssue] = []
    checks = ["source orography shape/dtype/finiteness"]
    source = getattr(case_data, "source_orography", None)
    if source is not None and "SOILGEO" in catalog.inventory:
        issues.append(PreflightIssue(
            "orography-conflict",
            "declared source_orography "
            f"{source.path} variable={source.variable} conflicts with "
            "forcing catalog SOILGEO via era5_z_invariant; declare exactly "
            "one source",
            path=Path(source.path), variable=str(source.variable),
        ))
    if source is None:
        # Task 4's forcing-catalog invariant provider is validated as SOILGEO.
        return issues, checks
    # Runtime resolves the artifact PER DOMAIN; checking only one domain
    # leaves a check-time false green on every other nest.
    bound: list[tuple[object, object]] = []
    if hasattr(source, "by_domain"):
        for dc in exp.domains:
            try:
                artifact = case_data.source_orography_for_domain(
                    int(dc.grid_id))
            except Exception as exc:
                issues.append(PreflightIssue(
                    "orography-declaration",
                    f"d{dc.grid_id:02d} source orography did not resolve: "
                    f"{exc}",
                ))
                continue
            bound.append((dc, artifact))
    elif hasattr(source, "path"):
        bound.append(
            (exp.domain(int(getattr(case_data, "output_domain", 1))),
             source))
    else:
        return issues, checks
    for dc, artifact in bound:
        path = Path(artifact.path)
        variable = str(artifact.variable)
        checks.append(
            f"source orography shape/dtype/finiteness d{dc.grid_id:02d}")
        try:
            # The VALUES come out of Drew's Rust decoder, in the split
            # `gpuwm.downscale._parent_geometry` established: a source
            # orography field is meteorological data whoever wrote the file,
            # and reading one is decode work.  `np.isfinite` below stays in
            # Python because it is a validation predicate over data the
            # bridge has already delivered, not a second decode of it.
            from gpuwm import netcdf_bridge          # noqa: PLC0415

            with netcdf_bridge.open_dataset(path) as dataset:
                if variable not in dataset.variables:
                    issues.append(PreflightIssue(
                        "orography-variable",
                        f"d{dc.grid_id:02d} source orography variable is "
                        f"absent; available variables: "
                        f"{sorted(dataset.variables)}",
                        path=path, variable=variable,
                    ))
                    continue
                raw = dataset.variables[variable]
                shape = tuple(raw.shape)
                expected = (int(dc.run.ny), int(dc.run.nx))
                if shape == expected:
                    selector: object = slice(None)
                elif len(shape) == 3 and shape[0] == 1 \
                        and shape[1:] == expected:
                    selector = 0
                else:
                    issues.append(PreflightIssue(
                        "orography-shape",
                        f"orography has shape {shape}; expected {expected} "
                        f"or {(1, *expected)} for d{dc.grid_id:02d}",
                        path=path, variable=variable,
                    ))
                    continue
                # Asked of the STORED type from the inventory, ahead of any
                # decode.  The decoder promotes every numeric type to f64, so
                # asking the decoded array would always answer "numeric" --
                # and a character orography cannot be decoded at all, so the
                # question has to be settled before the values are pulled.
                if not np.issubdtype(raw.dtype, np.number):
                    issues.append(PreflightIssue(
                        "orography-dtype",
                        f"d{dc.grid_id:02d} orography dtype {raw.dtype} "
                        "is not numeric",
                        path=path, variable=variable,
                    ))
                    continue
                value = np.asarray(raw[selector])
                bad = ~np.isfinite(value)
                if np.any(bad):
                    index = _first_index(bad)
                    issues.append(PreflightIssue(
                        "orography-nonfinite",
                        f"d{dc.grid_id:02d} orography contains "
                        f"{value[index]!r}", path=path,
                        variable=variable, index=index,
                    ))
        except Exception as exc:
            issues.append(PreflightIssue(
                "orography-read",
                f"could not validate d{dc.grid_id:02d} source orography: "
                f"{exc}",
                path=path, variable=variable,
            ))
    return issues, checks


def _geog_declarations(case_data) -> tuple[tuple[str, bool], ...]:
    declared = getattr(case_data, "geog_datasets", _REQUIRED_GEOG)
    sparse_names = set(str(value) for value in getattr(
        case_data, "sparse_geog_datasets", ()))
    if isinstance(declared, Mapping):
        result = []
        for name, policy in declared.items():
            if isinstance(policy, Mapping):
                sparse = bool(policy.get("sparse", False))
            else:
                sparse = bool(policy)
            result.append((str(name), sparse or str(name) in sparse_names))
        return tuple(result)
    return tuple((str(name), str(name) in sparse_names) for name in declared)


def _geog_window(ds: GeogDataset, grid, case_data, name: str
                 ) -> tuple[int, int, int, int]:
    overrides = getattr(case_data, "geog_coverage_windows", {})
    if name in overrides:
        values = tuple(int(value) for value in overrides[name])
        if len(values) != 4:
            raise ValueError(
                f"geog_coverage_windows[{name!r}] must be (x0,x1,y0,y1)"
            )
        return values
    if grid is None:
        raise ValueError("experiment grid is unavailable")
    # Static construction samples a three-model-cell extended grid and then
    # adds three source cells for interpolation support.  Use its exact
    # extended-grid coordinates here so preflight proves every tile the build
    # can read, rather than only the unextended mass-point footprint.
    from gpuwm.static.build import _DomainSampler
    from gpuwm.static.projection import ProjectedGrid

    if isinstance(grid, ProjectedGrid):
        sampler = _DomainSampler(grid)
        lat, lon = sampler.lat_c, sampler.lon_c
    else:
        # Lightweight grid adapters used by callers/tests predate the static
        # sampler contract.  Production experiment grids are ProjectedGrid
        # subclasses (lambert/mercator/polar).
        lat, lon = grid.latlon_mass()
    x, y = ds.latlon_to_xy(lat, lon)
    if ds.wraps_x and float(x.max() - x.min()) > ds.nx_global / 2.0:
        x = np.where(x < ds.nx_global / 2.0, x + ds.nx_global, x)
    margin = int(getattr(case_data, "geog_coverage_margin", 3))
    x0 = int(np.floor(np.min(x))) - margin
    x1 = int(np.ceil(np.max(x))) + margin
    y0 = max(1, int(np.floor(np.min(y))) - margin)
    y1 = min(ds.ny_global, int(np.ceil(np.max(y))) + margin)
    return x0, x1, y0, y1


def _needed_tile_origins(ds: GeogDataset, window: tuple[int, int, int, int]
                         ) -> tuple[tuple[int, int], ...]:
    x0, x1, y0, y1 = window
    x = np.arange(x0, x1 + 1, dtype=np.int64)
    if ds.wraps_x:
        x = (x - 1) % ds.nx_global + 1
    x = x[(x >= 1) & (x <= ds.nx_global)]
    y = np.arange(max(y0, 1), min(y1, ds.ny_global) + 1, dtype=np.int64)
    xs = np.unique((x - 1) // ds.index.tile_x * ds.index.tile_x + 1)
    ys = np.unique((y - 1) // ds.index.tile_y * ds.index.tile_y + 1)
    return tuple((int(x_origin), int(y_origin))
                 for y_origin in ys for x_origin in xs)


def _check_geog(case_data, grid) -> tuple[list[PreflightIssue], list[str],
                                          tuple[CatalogFile, ...]]:
    issues: list[PreflightIssue] = []
    checks = ["WPS GEOG tile completeness", "GEOG selected-tile hashes"]
    records: list[CatalogFile] = []
    root = Path(case_data.geog_root)
    for name, sparse in _geog_declarations(case_data):
        path = root / name
        if not path.is_dir():
            issues.append(PreflightIssue(
                "geog-dataset", f"required WPS GEOG dataset {name!r} is absent",
                path=path,
            ))
            continue
        try:
            ds = GeogDataset(path, sparse=sparse)
            window = _geog_window(ds, grid, case_data, name)
            coverage = ds.tile_coverage_mask(*window)
            if not sparse and not np.all(coverage):
                j, i = _first_index(~coverage)
                source_x = window[0] + i
                source_y = window[2] + j
                if ds.wraps_x:
                    source_x = (source_x - 1) % ds.nx_global + 1
                tile_x = ((source_x - 1) // ds.index.tile_x
                          * ds.index.tile_x + 1)
                tile_y = ((source_y - 1) // ds.index.tile_y
                          * ds.index.tile_y + 1)
                issues.append(PreflightIssue(
                    "missing-geog-tile",
                    f"dataset {name} has unexplained fill at source index "
                    f"(x={source_x}, y={source_y}); missing tile origin "
                    f"({tile_x}, {tile_y}). Declare sparse only for a "
                    "documented sparse dataset",
                    path=path, variable=name, index=(source_y, source_x),
                ))

            bdr = ds.index.tile_bdr
            expected_bytes = (ds.index.nz
                              * (ds.index.tile_y + 2 * bdr)
                              * (ds.index.tile_x + 2 * bdr)
                              * ds.index.wordsize)
            for origin in _needed_tile_origins(ds, window):
                tile = ds.tiles.get(origin)
                if tile is None:
                    continue
                actual = tile.stat().st_size
                if actual != expected_bytes:
                    issues.append(PreflightIssue(
                        "geog-tile-size",
                        f"dataset {name} tile {tile.name} has {actual} bytes; "
                        f"expected {expected_bytes}", path=tile,
                        variable=name,
                    ))
                    continue
                try:
                    records.append(_catalog_file(
                        "geog_tile", tile,
                        provenance=f"dataset={name};window={window}",
                    ))
                except Exception as exc:
                    issues.append(PreflightIssue(
                        "hash", f"could not hash selected GEOG tile: {exc}",
                        path=tile, variable=name,
                    ))
        except Exception as exc:
            issues.append(PreflightIssue(
                "geog-dataset", f"could not inventory dataset {name}: {exc}",
                path=path, variable=name,
            ))
    return issues, checks, tuple(records)


_TABLE_SHA256 = MappingProxyType({
    "kf_lutab/kf_lutab.npz":
        "3248ad89f73084d615a55e319ee3a0151a3b6a07402143d745fe1afb11751f73",
    "rrtmgp/rrtmgp-gas-lw-g256.nc":
        "4048360199d1917ed8f2ccaae2ec097d0f990da3bbad9830337b739b4fa01be7",
    "rrtmgp/rrtmgp-gas-sw-g224.nc":
        "584f1dd41ea9fc07d4ee3754eb1dafbd46ad3161cd6fd20fa06b6922b6f0702e",
    "rrtmgp/rrtmgp-clouds-lw-bnd.nc":
        "09d6704c5b863b4c3ceb417d20bb3076ec492e6bf2dfbcc9f3c5996a3706f0b0",
    "rrtmgp/rrtmgp-clouds-sw-bnd.nc":
        "7671835992a45afe66244b591a02c0b3df73d7d59ecb746bbffd9763497651cd",
    "noah_tables/LANDUSE.TBL":
        "cafdb5f4982b88c93f2cb321f18d9559e7c03b6d213388d5ae3d07b7280caa08",
    "noah_tables/VEGPARM.TBL":
        "4612c5c2d7e5398ae925ff363634ff98d6587a4aaa7440eaf1bbcc248dad6090",
    "noah_tables/SOILPARM.TBL":
        "efa3e1ad5dc6bd64665fcf8a2336ea4e4a2ca7fc8190a9305d363a4b13b2045b",
    "noah_tables/GENPARM.TBL":
        "b8d1829a745cb266a9e253e2ea13f2ae2493f05c1c40c62d785c794d4ee30a9a",
    "wrf_radiation/RRTM_DATA":
        "45dc91514cf018e133301a1b667deeb17f5b65c49ccfaa9140761a8028bc8ae0",
})

_RRTMGP_DIMS = MappingProxyType({
    "rrtmgp-gas-lw-g256.nc": {
        "bnd": 16, "temperature": 14, "pressure_interp": 60,
        "mixing_fraction": 9, "gpt": 256, "pressure": 59,
        "absorber": 19, "absorber_ext": 20,
    },
    "rrtmgp-gas-sw-g224.nc": {
        "bnd": 14, "temperature": 14, "pressure_interp": 60,
        "mixing_fraction": 9, "gpt": 224, "pressure": 59,
        "absorber": 19, "absorber_ext": 20,
    },
    "rrtmgp-clouds-lw-bnd.nc": {
        "nrghice": 3, "nband": 16, "nsize_ice": 18, "nsize_liq": 20,
    },
    "rrtmgp-clouds-sw-bnd.nc": {
        "nrghice": 3, "nband": 14, "nsize_ice": 18, "nsize_liq": 20,
    },
})


def _check_table_hash(path: Path, relative: str,
                      issues: list[PreflightIssue]) -> bool:
    """Pinned-checksum check for one table, keyed by its declared name.

    ``relative`` is the ``gpuwm/data``-relative spelling ``_TABLE_SHA256``
    is written against.  It is PASSED rather than derived from ``path``,
    because the two no longer share a root: since 2.5.0 ``rrtmgp/`` and
    ``thompson/tables/`` resolve inside the ``gpuwm-data`` companion.
    """

    expected = _TABLE_SHA256.get(relative)
    if not path.is_file():
        issues.append(PreflightIssue(
            "table-missing", "bundled table file is absent", path=path,
        ))
        return False
    actual = _sha256(path)
    if expected is not None and actual != expected:
        issues.append(PreflightIssue(
            "table-checksum",
            f"SHA-256 {actual} does not match pinned {expected}", path=path,
        ))
    return True


def _check_tables(case_data, exp=None) -> tuple[list[PreflightIssue], list[str]]:
    issues: list[PreflightIssue] = []
    checks = ["KF LUT checksum/shape/dtype/finiteness",
              "RRTMGP NetCDF checksum/shape/dtype/finiteness",
              "Noah tables checksum/shape/dtype/finiteness"]
    assets = _table_asset_map(case_data, exp)
    present = {path: _check_table_hash(path, relative, issues)
               for relative, path in assets.items()}

    kf_path = assets["kf_lutab/kf_lutab.npz"]
    if present.get(kf_path):
        expected = {
            "temperature": ((250, 220), np.dtype("float32")),
            "qsat": ((250, 220), np.dtype("float32")),
            "thetae_base": ((220,), np.dtype("float32")),
            "log_ratio": ((200,), np.dtype("float32")),
            "pressure_top": ((), np.dtype("float32")),
            "pressure_reciprocal": ((), np.dtype("float32")),
            "thetae_reciprocal": ((), np.dtype("float32")),
        }
        try:
            with np.load(kf_path, allow_pickle=False) as archive:
                if set(archive.files) != set(expected):
                    issues.append(PreflightIssue(
                        "table-inventory",
                        f"KF LUT members {sorted(archive.files)}; expected "
                        f"{sorted(expected)}", path=kf_path,
                    ))
                for name, (shape, dtype) in expected.items():
                    if name not in archive:
                        continue
                    value = np.asarray(archive[name])
                    if value.shape != shape:
                        issues.append(PreflightIssue(
                            "table-shape",
                            f"KF LUT {name} shape {value.shape}; expected {shape}",
                            path=kf_path, variable=name,
                        ))
                    if value.dtype != dtype:
                        issues.append(PreflightIssue(
                            "table-dtype",
                            f"KF LUT {name} dtype {value.dtype}; expected {dtype}",
                            path=kf_path, variable=name,
                        ))
                    bad = ~np.isfinite(value)
                    if np.any(bad):
                        index = _first_index(np.atleast_1d(bad))
                        issues.append(PreflightIssue(
                            "table-nonfinite",
                            f"KF LUT {name} contains a non-finite value",
                            path=kf_path, variable=name,
                            index=(() if value.ndim == 0 else index),
                        ))
        except Exception as exc:
            issues.append(PreflightIssue(
                "table-read", f"could not read KF LUT: {exc}", path=kf_path,
            ))

    try:
        import netCDF4
    except Exception as exc:
        issues.append(PreflightIssue(
            "table-read", f"netCDF4 is unavailable for RRTMGP checks: {exc}"
        ))
        netCDF4 = None
    if netCDF4 is not None:
        for name, dimensions in _RRTMGP_DIMS.items():
            path = assets.get(f"rrtmgp/{name}")
            if path is None or not present.get(path):
                continue
            try:
                with netCDF4.Dataset(path) as dataset:
                    for dim, size in dimensions.items():
                        actual = (len(dataset.dimensions[dim])
                                  if dim in dataset.dimensions else None)
                        if actual != size:
                            issues.append(PreflightIssue(
                                "table-shape",
                                f"RRTMGP dimension {dim}={actual}; expected {size}",
                                path=path, variable=dim,
                            ))
                    dtype_expect = (
                        {"kmajor": np.dtype("float64"),
                         "key_species": np.dtype("int32")}
                        if "gas" in name else
                        {"extice": np.dtype("float32"),
                         "extliq": np.dtype("float32")}
                    )
                    for variable, dtype in dtype_expect.items():
                        if variable not in dataset.variables:
                            issues.append(PreflightIssue(
                                "table-inventory",
                                f"RRTMGP variable {variable} is absent",
                                path=path, variable=variable,
                            ))
                        elif np.dtype(dataset[variable].dtype) != dtype:
                            issues.append(PreflightIssue(
                                "table-dtype",
                                f"RRTMGP {variable} dtype "
                                f"{dataset[variable].dtype}; expected {dtype}",
                                path=path, variable=variable,
                            ))
                    dataset.set_auto_mask(False)
                    for variable, handle in dataset.variables.items():
                        if not np.issubdtype(handle.dtype, np.number):
                            continue
                        value = np.asarray(handle[:])
                        bad = ~np.isfinite(value)
                        if np.any(bad):
                            index = _first_index(np.atleast_1d(bad))
                            issues.append(PreflightIssue(
                                "table-nonfinite",
                                f"RRTMGP {variable} contains a non-finite value",
                                path=path, variable=variable,
                                index=(() if value.ndim == 0 else index),
                            ))
            except Exception as exc:
                issues.append(PreflightIssue(
                    "table-read", f"could not read RRTMGP table: {exc}",
                    path=path,
                ))

    rrtm_path = assets.get("wrf_radiation/RRTM_DATA")
    if rrtm_path in present:
        checks.append("WRF RRTM_DATA checksum/record-layout/finiteness")
        if present[rrtm_path]:
            try:
                from gpuwm.core.rrtm import load_rrtm_raw_tables
                tables = load_rrtm_raw_tables(rrtm_path)
                if len(tables.arrays) != 42 or tables.nbytes != 749120:
                    issues.append(PreflightIssue(
                        "table-shape",
                        "RRTM_DATA decoded inventory is not the pinned "
                        f"42 arrays / 749120 bytes: got "
                        f"{len(tables.arrays)} / {tables.nbytes}",
                        path=rrtm_path,
                    ))
            except Exception as exc:
                issues.append(PreflightIssue(
                    "table-read", f"could not decode RRTM_DATA: {exc}",
                    path=rrtm_path,
                ))

    noah_dir = assets["noah_tables/LANDUSE.TBL"].parent
    if all(present.get(noah_dir / name) for name in
           ("LANDUSE.TBL", "VEGPARM.TBL", "SOILPARM.TBL", "GENPARM.TBL")):
        try:
            from gpuwm.core.landuse import load_landuse_table
            from gpuwm.core.noah import load_tables
            landuse = load_landuse_table(tbl_dir=noah_dir)
            tables = load_tables(tbl_dir=noah_dir)
            if landuse.values.shape != (2, 61, 7):
                issues.append(PreflightIssue(
                    "table-shape",
                    f"LANDUSE values shape {landuse.values.shape}; "
                    "expected (2, 61, 7)",
                    path=noah_dir / "LANDUSE.TBL", variable="values",
                ))
            elif not np.isfinite(landuse.values).all():
                index = _first_index(~np.isfinite(landuse.values))
                issues.append(PreflightIssue(
                    "table-nonfinite", "LANDUSE values are non-finite",
                    path=noah_dir / "LANDUSE.TBL", variable="values",
                    index=index,
                ))
            for name, shape in {
                    "shdtbl": (20,), "laimintbl": (20,),
                    "z0maxtbl": (20,), "bb": (19,), "maxsmc": (19,),
                    "satdk": (19,), "slope_data": (9,)}.items():
                value = np.asarray(getattr(tables, name))
                if value.shape != shape:
                    issues.append(PreflightIssue(
                        "table-shape",
                        f"Noah {name} shape {value.shape}; expected {shape}",
                        path=noah_dir, variable=name,
                    ))
                if not np.issubdtype(value.dtype, np.number):
                    issues.append(PreflightIssue(
                        "table-dtype", f"Noah {name} dtype {value.dtype} "
                        "is not numeric", path=noah_dir, variable=name,
                    ))
                elif not np.isfinite(value).all():
                    index = _first_index(~np.isfinite(value))
                    issues.append(PreflightIssue(
                        "table-nonfinite", f"Noah {name} is non-finite",
                        path=noah_dir, variable=name, index=index,
                    ))
        except Exception as exc:
            issues.append(PreflightIssue(
                "table-read", f"could not parse Noah tables: {exc}",
                path=noah_dir,
            ))
    return issues, checks


def preflight_report(exp, case_data) -> PreflightReport:
    """Run every CPU input/static/table check and aggregate all failures."""

    catalog, build_issues = _build_input_catalog(case_data)
    failures = list(build_issues)
    checks: list[str] = ["resolved input SHA-256 catalog",
                         "native GRIB1 envelope/decode contract"]

    for checker in (
            lambda: _scan_forcing(catalog),
            lambda: _check_levels(exp, catalog),
            lambda: _check_temporal(exp, case_data, catalog),
            lambda: _check_orography(exp, case_data, catalog),
            lambda: _check_tables(case_data, exp)):
        try:
            found, completed = checker()
            failures.extend(found)
            checks.extend(completed)
        except Exception as exc:  # preserve the aggregate contract
            failures.append(PreflightIssue(
                "preflight-internal",
                f"an independent CPU checker could not complete: {exc}",
            ))

    grid = None
    try:
        found, completed, grid = _check_spatial(exp, case_data, catalog)
        failures.extend(found)
        checks.extend(completed)
    except Exception as exc:
        failures.append(PreflightIssue(
            "preflight-internal", f"spatial checker could not complete: {exc}"
        ))

    geog_records: tuple[CatalogFile, ...] = ()
    try:
        found, completed, geog_records = _check_geog(case_data, grid)
        failures.extend(found)
        checks.extend(completed)
    except Exception as exc:
        failures.append(PreflightIssue(
            "preflight-internal", f"GEOG checker could not complete: {exc}"
        ))

    if geog_records:
        known = {(item.role, item.path) for item in catalog.files}
        additions = tuple(item for item in geog_records
                          if (item.role, item.path) not in known)
        catalog = replace(catalog, files=catalog.files + additions)
    return PreflightReport(catalog, tuple(failures), tuple(checks))


def _check_command(args) -> int:
    import io
    import os
    import sys
    import tomllib

    from gpuwm.case_data import load_experiment_case
    from gpuwm.config_authority import read_config_authority
    from gpuwm.experiment import is_experiment_toml_bytes

    # In --json mode stdout belongs to the memory estimator's machine
    # report; the input-preflight text goes to stderr so the composed
    # `gpuwm check CONFIG --json` emits parseable JSON on stdout.
    stream = sys.stderr if getattr(args, "json", False) else sys.stdout
    # A legacy RunConfig-shaped TOML has no [case_data] declared-input
    # table, so there is nothing for the input preflight to check.
    # Returning success lets the composed ``gpuwm check`` advance to the
    # memory estimator, which wraps the RunConfig as a one-domain
    # experiment (gpuwm.core.preflight._load_experiment_any) -- the same
    # acceptance the estimator registrar has always had.
    #
    # No `.exists()` pre-check any more: it existed so a missing file
    # could "fall through to the experiment loader's own error", which
    # is precisely the traceback-at-exit-1 behaviour this release
    # replaces.  `read_config_authority` now decides the path's kind and
    # refuses in one sentence.
    authority = read_config_authority(args.config)
    if (authority is not None
            and not is_experiment_toml_bytes(authority.payload)):
        print("input preflight: legacy RunConfig-shaped config declares no "
              "[case_data] inputs; skipping to the memory estimator",
              file=stream)
        return 0
    # An experiment TOML without [case_data] is not consumable by the
    # config-driven run front door -- `gpuwm domain --source gfs|hrrr`
    # emits this shape deliberately, because those tables feed the
    # rw-wps/gpuwm-wrf-init native initialization front door, which
    # performs its own hash-bound input validation.  Say so plainly,
    # then advance to the memory estimator: geometry/physics/VRAM
    # validation is exactly what `gpuwm check` can honestly certify for
    # these configs.
    if authority is not None:
        if "case_data" not in tomllib.load(io.BytesIO(authority.payload)):
            print(
                "input preflight: not applicable -- no [case_data] "
                "table (the config-driven run front door consumes "
                "the ERA5 native-GRIB1 route only; GFS/HRRR feed the "
                "rw-wps/gpuwm-wrf-init native front door, which "
                "validates its own inputs).  Continuing to the "
                "memory preflight.",
                file=stream)
            return 0
    exp, case_data = load_experiment_case(args.config)
    report = preflight_report(exp, case_data)
    print(report.format(), file=stream)
    return 0 if report.ok else 1


def register_cli(subparsers):
    """Extend the controller-owned ``gpuwm check`` with input preflight.

    The memory registrar owns the public subcommand and its flags.  Installing
    the namespaced handler on the parent parser makes registration order
    irrelevant without creating a competing ``check`` parser.  When a check
    parser already exists, preserve its dispatcher; a controller may replace
    that dispatcher explicitly with the combined input-then-memory policy.
    """

    subparsers.container.set_defaults(
        ingest_preflight_handler=_check_command
    )
    parser = subparsers.choices.get("check")
    if parser is None:
        return None
    if not any(getattr(action, "dest", None) == "config"
               for action in parser._actions):
        parser.add_argument("config", type=Path, metavar="CONFIG",
                            help="experiment TOML with [case_data]")
    parser.set_defaults(ingest_preflight_handler=_check_command)
    if parser.get_default("func") is None:
        parser.set_defaults(func=_check_command)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="gpuwm check")
    parser.add_argument("config", type=Path, metavar="CONFIG")
    return _check_command(parser.parse_args(argv))


__all__ = [
    "CatalogBuildError", "CatalogFile", "CatalogMask", "InputCatalog",
    "LBCRecord", "PreflightIssue", "PreflightReport", "SpatialCoverage",
    "build_input_catalog", "build_lbc_records", "colon_free_output_filename",
    "main", "output_records", "preflight_report", "register_cli",
]
