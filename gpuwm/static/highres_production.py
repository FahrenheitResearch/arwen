"""Config-driven high-resolution static geography for real-data cases.

This is the production wrapper around the proven pilot machinery in
:mod:`gpuwm.static.highres`: the ``[static.highres]`` TOML surface, the
US-interior refusal gates, the per-footprint tiled fetch/cache
(:mod:`gpuwm.static.highres_fetch`), and the per-domain receipts.  The
science -- area averaging to WPS spherical grids, the NLCD->MODIS-21
crosswalk, the USDA soil-texture triangle, the water-mask merge with donor
climatology fill and TMN recompute -- is reused wholesale, not rewritten.

Scope (deliberate, refuse-loudly outside it):

- US-interior footprints only.  A footprint outside the joint 3DEP/Annual
  NLCD coverage envelope refuses; a footprint whose 30-arc-second baseline
  land use reports WRF ocean category anywhere refuses, because the
  pilot's inland-water rule (NLCD open water -> WRF lake 21) is not
  coast-safe.  ``on_refuse`` selects between a hard error (default) and an
  explicit, receipted fallback to the unchanged 30s baseline.
- Replaced fields are exactly the pilot's split: terrain, land-use
  fractions/index/mask, and soil fractions/categories.  Monthly
  climatologies stay 30s with counted donor fill; TMN is recomputed.
- An enabled block that ends up replacing zero cells is itself a refusal:
  an enabled feature that did nothing must never look like it ran.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np

from .build import HALO
from .highres import (BoundRaster, NLCD_TO_MODIS21_INLAND,
                      build_highres_overrides, merge_highres_overrides)
from .highres_fetch import (ANNUAL_NLCD_SOURCE_URL, ANNUAL_NLCD_LICENSE,
                            SOILGRIDS_COMPONENTS, SOILGRIDS_CRS,
                            SOILGRIDS_DEPTHS, SOILGRIDS_LICENSE,
                            SOILGRIDS_NODATA, SOILGRIDS_SCALE,
                            SOILGRIDS_SOURCE_URL, THREE_DEP_LICENSE,
                            THREE_DEP_SOURCE_URL, CoverageError,
                            derive_landcover_window, derive_terrain_window,
                            domain_footprint, fetch_annual_nlcd,
                            fetch_soilgrids, fetch_three_dep_tiles,
                            nlcd_year_for, FootprintBBox)
from .lambert import LambertGrid

RECEIPT_SCHEMA = "gpuwm-static-highres-receipt-v1"

#: Envelope where the declared sources are JOINTLY published: 3DEP staged
#: 1/3 arc-second tiles and the Annual NLCD conterminous-US collection.
#: This is a source-coverage constant, not a policy taste bound.
US_COVERAGE_ENVELOPE = FootprintBBox(
    lat_min=24.0, lat_max=49.5, lon_min=-125.0, lon_max=-66.5)

#: The crosswalk targets WRF's MODIS 21-category inventory with these two
#: water categories; a baseline using any other inventory cannot be merged.
_MODIS21_ISWATER = 17
_MODIS21_ISLAKE = 21

_REPLACED_FIELDS = ("HGT_M", "LANDUSEF", "LANDMASK", "LU_INDEX",
                    "SOILCTOP", "SCT_DOM", "SOILCBOT", "SCB_DOM")

_ON_REFUSE_CHOICES = ("error", "fallback-30s")


class HighresRefusal(RuntimeError):
    """A named, deliberate refusal of the high-resolution path."""

    def __init__(self, reason: str, detail: str):
        super().__init__(f"[static.highres] refused ({reason}): {detail}")
        self.reason = reason
        self.detail = detail


@dataclass(frozen=True)
class HighresStaticConfig:
    """Validated ``[static.highres]`` block."""

    enabled: bool
    cache_root: Path
    on_refuse: str = "error"

    def echo(self) -> dict[str, str]:
        return {
            "enabled": "true" if self.enabled else "false",
            "cache_root": str(self.cache_root),
            "on_refuse": self.on_refuse,
        }


def parse_static_table(raw, *, source: str, base_dir
                       ) -> HighresStaticConfig | None:
    """Validate one ``[static]`` table (fail-loud); ``None`` when absent.

    Follows the governance idiom of :mod:`gpuwm.case_data`: no key is
    ignored, because a dropped key runs a default under the name of the
    user's value.
    """
    if raw is None:
        return None
    from gpuwm.experiment import did_you_mean

    if not isinstance(raw, dict):
        raise ValueError(
            f"[static] of {source} must be a table, got {raw!r}.")
    unknown = sorted(set(raw) - {"highres"})
    if unknown:
        named = ", ".join(
            f"{key!r}{did_you_mean(key, ('highres',))}" for key in unknown)
        raise ValueError(
            f"[static] of {source} does not have a table {named}; the one "
            "known sub-table is [static.highres].")
    table = raw.get("highres")
    if table is None:
        raise ValueError(
            f"[static] of {source} declares nothing; declare "
            "[static.highres] or remove the table.")
    if not isinstance(table, dict):
        raise ValueError(
            f"[static.highres] of {source} must be a table, got {table!r}.")

    known = ("enabled", "cache_root", "on_refuse")
    unknown = sorted(set(table) - set(known))
    if unknown:
        named = ", ".join(
            f"{key!r}{did_you_mean(key, known)}" for key in unknown)
        raise ValueError(
            f"[static.highres] of {source} does not have a key {named}; no "
            "key is ignored, because a dropped key runs a default under "
            f"the name of your value.  Known keys: {sorted(known)}.")
    missing = [key for key in ("enabled", "cache_root") if key not in table]
    if missing:
        raise ValueError(
            f"[static.highres] of {source} is missing required key(s) "
            f"{missing}: the fetch cache location is declared, never "
            "implicit.")

    enabled = table["enabled"]
    if not isinstance(enabled, bool):
        raise ValueError(
            f"enabled in [static.highres] of {source} must be a boolean, "
            f"got {enabled!r}.")

    raw_root = table["cache_root"]
    if not isinstance(raw_root, str) or not raw_root:
        raise ValueError(
            f"cache_root in [static.highres] of {source} must be a "
            f"non-empty path string, got {raw_root!r}.")
    from gpuwm.case_data import expand_path_variables
    cache_root = Path(expand_path_variables(raw_root, "cache_root", source))
    if not cache_root.is_absolute():
        cache_root = Path(base_dir) / cache_root

    on_refuse = table.get("on_refuse", "error")
    if on_refuse not in _ON_REFUSE_CHOICES:
        raise ValueError(
            f"on_refuse in [static.highres] of {source} must be one of "
            f"{list(_ON_REFUSE_CHOICES)}, got {on_refuse!r}.  'error' "
            "stops the case at a refusal; 'fallback-30s' proceeds on the "
            "unchanged 30-arc-second baseline and says so in the receipt.")

    return HighresStaticConfig(enabled=enabled, cache_root=cache_root,
                               on_refuse=str(on_refuse))


# ---------------------------------------------------------------------------
# Refusal gates
# ---------------------------------------------------------------------------

def _require_lambert(grid) -> None:
    if not isinstance(grid, LambertGrid):
        raise HighresRefusal(
            "unsupported-projection",
            f"the production high-resolution path is proven on WRF "
            f"spherical Lambert grids only; this domain uses "
            f"{getattr(grid, 'map_proj', type(grid).__name__)!r}")


def _require_modis21(landuse_attrs) -> None:
    iswater = int(landuse_attrs["ISWATER"])
    islake = landuse_attrs["ISLAKE"]
    islake = None if islake in (None, "") else int(islake)
    if iswater != _MODIS21_ISWATER or islake != _MODIS21_ISLAKE:
        raise HighresRefusal(
            "landuse-inventory-mismatch",
            "the NLCD crosswalk targets WRF's MODIS 21-category land use "
            f"(ISWATER={_MODIS21_ISWATER}, ISLAKE={_MODIS21_ISLAKE}); the "
            f"selected baseline dataset declares ISWATER={iswater}, "
            f"ISLAKE={islake}")


def _require_us_interior(bbox: FootprintBBox, baseline, *,
                         iswater: int) -> None:
    if not US_COVERAGE_ENVELOPE.contains(bbox):
        raise HighresRefusal(
            "outside-us-coverage",
            f"domain+halo footprint {bbox.as_dict()} leaves the joint "
            "publication envelope of USGS 3DEP 1/3 arc-second staged tiles "
            "and the Annual NLCD conterminous-US collection "
            f"({US_COVERAGE_ENVELOPE.as_dict()}); the declared sources do "
            "not exist there")
    lu_index = np.asarray(baseline["LU_INDEX"])
    ocean_cells = int(np.count_nonzero(lu_index == float(iswater)))
    if ocean_cells:
        raise HighresRefusal(
            "coastal-footprint",
            f"{ocean_cells} cell(s) of the domain's own 30-arc-second "
            f"baseline LU_INDEX carry WRF ocean category {iswater} "
            "(coast-detection method: the baseline's water field, which "
            "separates ocean from inland lakes).  The high-resolution "
            "inland-water rule maps NLCD open water to WRF lake category "
            f"{_MODIS21_ISLAKE} and is not coast-safe: at a coast the "
            "ocean/lake split needs a coastline-aware rule this path does "
            "not implement")


# ---------------------------------------------------------------------------
# Source binding
# ---------------------------------------------------------------------------

def _bound(fetched, *, source_id: str, role: str, source_url: str,
           license_id: str, license_url: str, nominal_resolution: str,
           reference_year=None, crs_override=None, nodata_override=None,
           scale_factor: float = 1.0) -> BoundRaster:
    return BoundRaster(
        path=Path(fetched.path), sha256=fetched.sha256, source_id=source_id,
        role=role, source_url=source_url, license_id=license_id,
        license_url=license_url, nominal_resolution=nominal_resolution,
        expected_bytes=int(fetched.bytes), reference_year=reference_year,
        crs_override=crs_override, nodata_override=nodata_override,
        scale_factor=scale_factor)


def _fetch_and_bind(bbox: FootprintBBox, cache_root: Path, case_date: date,
                    *, urlopen=None):
    """Fetch/cache everything one footprint needs; return bound sources."""
    tiles = fetch_three_dep_tiles(bbox, cache_root, urlopen=urlopen)
    terrain_window = derive_terrain_window(tiles, bbox, cache_root)
    terrain = _bound(
        terrain_window, source_id="usgs-3dep-13as", role="terrain",
        source_url=THREE_DEP_SOURCE_URL, license_id=THREE_DEP_LICENSE[0],
        license_url=THREE_DEP_LICENSE[1],
        nominal_resolution="1/3 arc-second (~10 m)")

    year, anachronism_years = nlcd_year_for(case_date)
    bundle, raster = fetch_annual_nlcd(year, cache_root, urlopen=urlopen)
    landcover_window = derive_landcover_window(raster, bbox, cache_root)
    landcover = _bound(
        landcover_window, source_id=f"annual-nlcd-{year}", role="landcover",
        source_url=ANNUAL_NLCD_SOURCE_URL,
        license_id=ANNUAL_NLCD_LICENSE[0],
        license_url=ANNUAL_NLCD_LICENSE[1], nominal_resolution="30 m",
        reference_year=year)

    soil_fetched = fetch_soilgrids(bbox, cache_root, urlopen=urlopen)
    soil_sources = {
        key: _bound(
            item, source_id="soilgrids-v2",
            role=f"soil_{key[0]}_{key[1]}", source_url=SOILGRIDS_SOURCE_URL,
            license_id=SOILGRIDS_LICENSE[0],
            license_url=SOILGRIDS_LICENSE[1], nominal_resolution="250 m",
            crs_override=SOILGRIDS_CRS, nodata_override=SOILGRIDS_NODATA,
            scale_factor=SOILGRIDS_SCALE)
        for key, item in soil_fetched.items()
    }
    fetch_manifest = {
        "three_dep_tiles": [item.receipt() for item in tiles],
        "terrain_window": terrain_window.receipt(),
        "annual_nlcd_bundle": bundle.receipt(),
        "annual_nlcd_raster": raster.receipt(),
        "landcover_window": landcover_window.receipt(),
        "soilgrids_windows": {
            f"{component}_{depth}": soil_fetched[(component, depth)].receipt()
            for component in SOILGRIDS_COMPONENTS
            for depth in SOILGRIDS_DEPTHS
        },
        "bytes_fetched": sum(
            item.bytes for item in (*tiles, bundle)
            if not item.cache_hit) + sum(
            item.bytes for item in soil_fetched.values()
            if not item.cache_hit),
        "nlcd_year": year,
        "nlcd_anachronism_years": anachronism_years,
    }
    return terrain, landcover, soil_sources, fetch_manifest


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

def _replacement_counts(baseline, merged) -> dict[str, int]:
    counts: dict[str, int] = {}
    for name in _REPLACED_FIELDS:
        before = np.asarray(baseline[name])
        after = np.asarray(merged[name])
        changed = before != after
        while changed.ndim > 2:          # category/fraction leading axes
            changed = changed.any(axis=0)
        counts[name] = int(np.count_nonzero(changed))
    return counts


def _grid_identity(grid, domain_id: int) -> dict[str, object]:
    return {
        "domain_id": int(domain_id),
        "map_proj": getattr(grid, "map_proj", "lambert"),
        "ref_lat": float(grid.ref_lat), "ref_lon": float(grid.ref_lon),
        "truelat1": float(grid.truelat1), "truelat2": float(grid.truelat2),
        "stand_lon": float(grid.stand_lon),
        "dx": float(grid.dx), "dy": float(grid.dy),
        "e_we": int(grid.e_we), "e_sn": int(grid.e_sn),
    }


def _write_receipt(config: HighresStaticConfig, receipt: dict) -> Path:
    identity = hashlib.sha256(json.dumps(
        receipt["grid"], sort_keys=True).encode("utf-8")).hexdigest()[:16]
    domain_id = int(receipt["grid"]["domain_id"])
    path = (Path(config.cache_root) / "receipts"
            / f"static_highres_{identity}_d{domain_id:02d}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(receipt, indent=2, sort_keys=True,
                         allow_nan=False) + "\n"
    temporary = path.with_name(f".{path.name}.partial-{os.getpid()}")
    temporary.write_text(encoded, encoding="utf-8")
    os.replace(temporary, path)
    receipt["receipt_path"] = str(path.resolve())
    return path


def apply_highres_statics(baseline, grid, *, config, domain_id: int,
                          case_date: date, landuse_attrs, urlopen=None):
    """Replace one domain's static fields from high-resolution sources.

    Returns ``(fields, receipt)``.  ``config`` absent or disabled is the
    identity: the exact ``baseline`` object comes back with ``None`` and
    nothing is fetched, printed, or written.  Refusals follow
    ``config.on_refuse``: ``"error"`` raises :class:`HighresRefusal`,
    ``"fallback-30s"`` returns the unchanged baseline beneath a receipt
    that names the refusal.
    """
    if config is None or not getattr(config, "enabled", False):
        return baseline, None

    receipt: dict[str, object] = {
        "schema": RECEIPT_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": config.echo(),
        "grid": _grid_identity(grid, domain_id),
        "case_date": case_date.isoformat(),
    }
    try:
        fields, detail = _apply(baseline, grid, config=config,
                                case_date=case_date,
                                landuse_attrs=landuse_attrs,
                                urlopen=urlopen)
        receipt.update(detail)
        receipt["status"] = "APPLIED"
        path = _write_receipt(config, receipt)
        print(f"[static.highres] d{int(domain_id):02d}: APPLIED "
              f"(cells replaced: {detail['cells_replaced']['total']} of "
              f"{detail['cells_replaced']['cell_count']}; "
              f"receipt {path})")
        return fields, receipt
    except HighresRefusal as refusal:
        receipt["status"] = "REFUSED"
        receipt["refusal"] = {"reason": refusal.reason,
                              "detail": refusal.detail}
        if config.on_refuse != "fallback-30s":
            raise HighresRefusal(
                refusal.reason,
                refusal.detail + "  (on_refuse = \"error\"; set on_refuse "
                "= \"fallback-30s\" in [static.highres] to proceed on the "
                "unchanged 30-arc-second baseline)") from refusal
        receipt["fallback"] = "unchanged 30-arc-second baseline statics"
        path = _write_receipt(config, receipt)
        print(f"[static.highres] d{int(domain_id):02d}: REFUSED "
              f"({refusal.reason}) -- proceeding on the 30s baseline per "
              f"on_refuse = \"fallback-30s\" (receipt {path})")
        return baseline, receipt


def _apply(baseline, grid, *, config: HighresStaticConfig,
           case_date: date, landuse_attrs, urlopen=None):
    _require_lambert(grid)
    _require_modis21(landuse_attrs)
    bbox = domain_footprint(grid, HALO)
    _require_us_interior(bbox, baseline, iswater=_MODIS21_ISWATER)

    try:
        terrain, landcover, soil_sources, fetch_manifest = _fetch_and_bind(
            bbox, config.cache_root, case_date, urlopen=urlopen)
    except CoverageError as error:
        raise HighresRefusal("missing-source-coverage", str(error)) \
            from error

    soil_fallback = {"SOILCTOP": np.asarray(baseline["SOILCTOP"]),
                     "SOILCBOT": np.asarray(baseline["SOILCBOT"])}
    overrides, source_audit = build_highres_overrides(
        grid, terrain=terrain, landcover=landcover,
        soil_sources=soil_sources, soil_fallback=soil_fallback,
        landcover_mapping=NLCD_TO_MODIS21_INLAND, halo=HALO)
    merged, merge_audit = merge_highres_overrides(baseline, overrides)

    counts = _replacement_counts(baseline, merged)
    cell_count = int(np.asarray(baseline["HGT_M"]).size)
    changed_any = np.zeros(np.asarray(baseline["HGT_M"]).shape, dtype=bool)
    for name in _REPLACED_FIELDS:
        before = np.asarray(baseline[name])
        after = np.asarray(merged[name])
        changed = before != after
        while changed.ndim > 2:
            changed = changed.any(axis=0)
        changed_any |= changed
    total_changed = int(np.count_nonzero(changed_any))
    if total_changed == 0:
        raise HighresRefusal(
            "zero-cells-replaced",
            "the enabled high-resolution overlay produced fields identical "
            "to the 30-arc-second baseline on every cell; an enabled block "
            "that changes nothing must not present itself as applied")

    detail = {
        "footprint": bbox.as_dict(),
        "halo_cells": HALO,
        "fetch": fetch_manifest,
        "sources": source_audit.pop("sources"),
        "override_audit": source_audit,
        "merge_audit": merge_audit,
        "cells_replaced": {**counts, "total": total_changed,
                           "cell_count": cell_count},
        "fields_replaced": list(_REPLACED_FIELDS),
        "fields_retained_30s": ["GREENFRAC", "LAI12M", "ALBEDO12M",
                                "SNOALB", "SOILTEMP"],
        "tmn": "recomputed from merged SOILTEMP/HGT_M over the new mask",
    }
    year = fetch_manifest["nlcd_year"]
    anachronism = fetch_manifest["nlcd_anachronism_years"]
    if anachronism:
        detail["anachronism"] = (
            f"Annual NLCD begins {year if year == 1985 else 1985}; the "
            f"case date {case_date.isoformat()} uses the {year} map, a "
            f"{anachronism}-year land-cover anachronism.  A modern map is "
            "never a silent historical replacement; it is named here.")
    return merged, detail


def refuse_inert_highres(config_path, *, lane: str) -> None:
    """Refuse an enabled ``[static.highres]`` block on a lane that ignores it.

    Only the experiment route (``gpuwm.runtime``) builds high-resolution
    statics.  The direct adapters and the native-HRRR static path build
    their GEOG fields through their own seams and load their config with
    :func:`gpuwm.experiment.load_experiment`, which never reads the
    ``[static]`` table at all -- so before this refusal existed, a user
    who enabled the block and ran one of those routes got the 30 arc
    second baseline with no indication that the thing they asked for had
    not happened.  A silently inert setting is the failure mode this
    project refuses on principle: it reads, afterwards, as a setting that
    took effect.

    Absent or ``enabled = false`` blocks pass, because they ask for
    nothing.  ``lane`` names the route in the refusal, so the message
    says which path could not honor it rather than that something,
    somewhere, was wrong.
    """
    import io
    import tomllib

    path = Path(config_path)
    try:
        raw = tomllib.load(io.BytesIO(path.read_bytes()))
    except (OSError, tomllib.TOMLDecodeError):
        # Not this function's error to report: the lane's own loader
        # reads the same file and says it better.
        return
    highres = parse_static_table(
        raw.get("static"), source=str(path), base_dir=path.parent)
    if highres is None or not getattr(highres, "enabled", False):
        return
    raise ValueError(
        f"{path} enables [static.highres], and the {lane} cannot honor "
        "it: that route builds its static geography through its own seam "
        "and never consults the block, so the run would silently use the "
        "30 arc second baseline under the name of your setting. Either "
        "run this case on the experiment route (`gpuwm run`), which "
        "applies high-resolution statics, or set enabled = false to ask "
        "for the baseline deliberately.")


__all__ = [
    "HighresRefusal", "HighresStaticConfig", "RECEIPT_SCHEMA",
    "US_COVERAGE_ENVELOPE", "apply_highres_statics", "parse_static_table",
    "refuse_inert_highres",
]
