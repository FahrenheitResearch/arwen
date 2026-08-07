"""Build ``GPWMGOES`` packs in memory, for tests that need one.

An independent transcription of the framing in
``tools/rustwx/crates/rw-goes/src/pack.rs`` -- deliberately written from
the Rust writer's own field list rather than from
:mod:`gpuwm.obs.goes_pack`, so that a reader test is a test of two
independent readings of one contract and not of a module agreeing with
itself.

``tests/test_goes_pack_crosslane.py`` closes the loop the only way that
really settles it: it hands a pack this module wrote to the **Rust**
``rw_goes verify``, so the bytes are judged by the other lane's real
decoder.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

MAGIC = b"GPWMGOES"
VERSION = 1
HEADER_BYTES = 64

CWP_SCHEMA = "gpuwm-obs.goes-cwp.v1"
CWP_SCHEMA_V2 = "gpuwm-obs.goes-cwp.v2"
CLOUDTOP_SCHEMA = "gpuwm-obs.goes-cloudtop.v1"
CLOUDTOP_SCHEMA_V2 = "gpuwm-obs.goes-cloudtop.v2"

#: GOES-19 CONUS, as the real granules carry it.
PROJECTION = {
    "perspective_point_height_m": 35786023.0,
    "semi_major_axis_m": 6378137.0,
    "semi_minor_axis_m": 6356752.31414,
    "longitude_of_projection_origin_deg": -75.0,
    "sweep_angle_axis": "x",
}

COEFFICIENTS = {
    "formula": "CWP [g m-2] = (2/3) * rho(phase) * tau * r_e",
    "liquid_density_g_cm3": 1.0,
    "ice_density_g_cm3": 0.917,
    "ice_coefficient_provisional": True,
    "mixed_phase_takes_ice_branch_provisional": True,
    "clear_sky_emits_zero": True,
}

#: The bridge corrected this string on 2026-08-06: it used to name
#: ``gpuwm-obs.goes-cwp.v1`` literally, which shipped a wrong schema name
#: inside v2 cloud-top packs.  It now names ``pairs_with_schema`` instead,
#: carrying no version.  Nothing in this reader parses it -- it is prose
#: for humans -- but the fixture tracks the real string so a test never
#: asserts against a sentence the bridge has stopped writing.
NO_REGRID = (
    "none: planes are on the granules' own fixed grid, bit-identical; any "
    "join to the pairs_with_schema sibling's grid is the consumer's "
    "explicit choice")


def source(product: str, *, condemn_mask: int | None = 88,
           total: int = 100, finite: int = 90,
           dqf_plane: str | None = None) -> dict:
    """One ``SourceEntry`` row.

    ``condemn_mask`` defaults to 88 (snow/sea-ice 8 | twilight 16 |
    glint 64), which is what the live packs actually carry -- verified
    against g19_conus_20260804_1801.  Bits 256/512 are outside it by
    design, which is why the operator inflates them rather than gating.

    ``dqf_plane`` names this product's per-pixel DQF plane (v2 only).
    """

    return {
        "product": product,
        "filename": f"OR_ABI-L2-{product}C-M6_G19_s20262161801170.nc",
        "bytes": 1234567,
        "sha256": hashlib.sha256(product.encode()).hexdigest(),
        "dqf_rule": "bitfield" if condemn_mask is not None else "enumerated",
        **({"condemn_mask": condemn_mask} if condemn_mask is not None else {}),
        **({"dqf_plane": dqf_plane} if dqf_plane is not None else {}),
        "dqf": {"total": total, "primary_missing": 3, "dqf_missing": 2,
                "dqf_bad": 4, "masked": total - finite, "finite": finite},
    }


def derive_cwp(cod, cps, phase, *, coefficients=None) -> np.ndarray:
    """The bridge's own CWP derivation, in the bridge's operand order.

    ``rw_sat::cwp::cloud_water_path_g_m2``: clear sky is a genuine 0.0,
    unknown phase and missing or negative inputs are NaN, and the product
    is formed as ``((2/3 * rho) * tau) * r_e`` in float32.
    """

    coefficients = COEFFICIENTS if coefficients is None else coefficients
    cod = np.asarray(cod, dtype=np.float32)
    cps = np.asarray(cps, dtype=np.float32)
    phase = np.asarray(phase, dtype=np.float32)
    out = np.full(cod.shape, np.nan, dtype=np.float32)
    two_thirds = np.float32(2.0) / np.float32(3.0)
    usable = (np.isfinite(phase) & (phase >= 0.0) & (phase <= 5.0)
              & (np.floor(phase) == phase))
    codes = np.where(usable, phase, -1.0).astype(np.int64)
    inputs_ok = (np.isfinite(cod) & np.isfinite(cps)
                 & (cod >= np.float32(0.0)) & (cps >= np.float32(0.0)))
    out[codes == 0] = np.float32(0.0)
    for code, key in ((1, "liquid_density_g_cm3"),
                      (2, "liquid_density_g_cm3"),
                      (3, "ice_density_g_cm3"),
                      (4, "ice_density_g_cm3")):
        rho = np.float32(coefficients[key])
        where = (codes == code) & inputs_ok
        if np.any(where):
            out[where] = ((two_thirds * rho) * cod[where]) * cps[where]
    return out


def _encode(meta: dict, payload: bytes) -> bytes:
    meta_json = json.dumps(meta).encode("utf-8")
    header = bytearray(HEADER_BYTES)
    header[0:8] = MAGIC
    header[8:12] = np.uint32(VERSION).tobytes()
    header[12:16] = np.uint32(len(meta_json)).tobytes()
    header[16:24] = np.uint64(len(payload)).tobytes()
    return bytes(header) + meta_json + payload


def _pack_planes(named):
    """``(payload, planes, plane_order, arrays)`` from ordered planes."""

    payload = bytearray()
    planes: dict[str, str] = {}
    order: list[str] = []
    arrays: dict[str, dict] = {}
    for index, (name, values) in enumerate(named):
        values = np.ascontiguousarray(values, dtype="<f4")
        key = f"a{index:05d}"
        offset = len(payload)
        payload.extend(values.tobytes(order="C"))
        planes[name] = key
        order.append(name)
        arrays[key] = {"dtype": "<f4", "shape": list(values.shape),
                       "offset": offset, "bytes": values.nbytes}
    return bytes(payload), planes, order, arrays


def write_cwp_pack(path, *, cod, cps, phase, lat, lon, cwp=None,
                   satellite="G19", sector="C",
                   scan_start="2026-08-04T18:01:17.0Z",
                   scan_end="2026-08-04T18:06:17.0Z",
                   x_scan_rad=None, y_scan_rad=None,
                   projection=None, coefficients=None, status="READY",
                   schema=CWP_SCHEMA, window=None, sources=None,
                   cloud_top_height_m=None, dqf_planes=None) -> Path:
    """Write one CWP pack, v1 or v2.

    ``dqf_planes`` is ``{product: (ny, nx) array}``; supplying it appends
    the per-pixel DQF planes AFTER lat/lon (v2 layout, indices 0-5
    unmoved) and names each in its source row's ``dqf_plane``.
    """

    cod = np.asarray(cod, dtype=np.float32)
    ny, nx = cod.shape
    coefficients = COEFFICIENTS if coefficients is None else coefficients
    if cwp is None:
        cwp = derive_cwp(cod, cps, phase, coefficients=coefficients)
    named = [("cwp", cwp), ("phase", phase), ("cod", cod), ("cps", cps)]
    if cloud_top_height_m is not None:
        named.append(("cloud_top_height_m", cloud_top_height_m))
    named += [("lat", lat), ("lon", lon)]
    if dqf_planes:
        for product in ("COD", "CPS", "ACTP"):
            if product in dqf_planes:
                named.append((f"{product.lower()}_dqf",
                              dqf_planes[product]))
    payload, planes, order, arrays = _pack_planes(named)
    if sources is None:
        def _plane_name(product):
            return (f"{product.lower()}_dqf"
                    if dqf_planes and product in dqf_planes else None)
        sources = [source("COD", dqf_plane=_plane_name("COD")),
                   source("CPS", dqf_plane=_plane_name("CPS")),
                   source("ACTP", condemn_mask=None,
                          dqf_plane=_plane_name("ACTP"))]
    meta = {
        "schema": schema,
        "status": status,
        "satellite": satellite,
        "sector": sector,
        "scan_start": scan_start,
        "scan_end": scan_end,
        "sources": sources,
        "projection": dict(PROJECTION if projection is None else projection),
        "nx": int(nx),
        "ny": int(ny),
        "x_scan_rad": (list(np.linspace(-0.05, 0.05, nx)) if x_scan_rad is None
                       else list(map(float, x_scan_rad))),
        "y_scan_rad": (list(np.linspace(0.10, 0.06, ny)) if y_scan_rad is None
                       else list(map(float, y_scan_rad))),
        "planes": planes,
        "plane_order": order,
        "arrays": arrays,
        "payload_bytes": len(payload),
        "content_sha256": hashlib.sha256(payload).hexdigest(),
        "cwp_counts": {"clear_zero": 1, "liquid": 1, "supercooled": 0,
                       "mixed": 0, "ice": 1, "unknown": 0,
                       "phase_missing": 0, "input_missing": 0, "finite": 3},
        "coefficients": dict(coefficients),
    }
    if window is not None:
        meta["window"] = list(window)
    path = Path(path)
    path.write_bytes(_encode(meta, payload))
    return path


def write_cloudtop_pack(path, *, cloud_top_height_m, lat, lon,
                        cloud_top_pressure_hpa=None,
                        satellite="G19", sector="C",
                        scan_start="2026-08-04T18:01:17.0Z",
                        scan_end="2026-08-04T18:06:17.0Z",
                        x_scan_rad=None, y_scan_rad=None,
                        projection=None, status="READY",
                        schema=CLOUDTOP_SCHEMA, window=None,
                        sibling=None, dqf_planes=None) -> Path:
    """Write one cloud-top pack, v1 or v2.

    ``dqf_planes`` is ``{product: array}`` for ACHA/CTP; supplying it
    appends the per-pixel DQF planes after lat/lon and names each in its
    source row, which a v2 pack must carry.
    """

    heights = np.asarray(cloud_top_height_m, dtype=np.float32)
    ny, nx = heights.shape
    products = ["ACHA"] + (["CTP"] if cloud_top_pressure_hpa is not None
                           else [])
    if schema.endswith(".v2") and dqf_planes is None:
        dqf_planes = {name: np.zeros((ny, nx), np.float32)
                      for name in products}
    named = [("cloud_top_height_m", heights)]
    if cloud_top_pressure_hpa is not None:
        named.append(("cloud_top_pressure_hpa", cloud_top_pressure_hpa))
    named += [("lat", lat), ("lon", lon)]
    if dqf_planes:
        for product in products:
            if product in dqf_planes:
                named.append((f"{product.lower()}_dqf",
                              dqf_planes[product]))
    payload, planes, order, arrays = _pack_planes(named)
    meta = {
        "schema": schema,
        "status": status,
        "satellite": satellite,
        "sector": sector,
        "scan_start": scan_start,
        "scan_end": scan_end,
        "sources": [
            source(name, condemn_mask=None,
                   dqf_plane=(f"{name.lower()}_dqf"
                              if dqf_planes and name in dqf_planes else None))
            for name in products],
        "projection": dict(PROJECTION if projection is None else projection),
        "nx": int(nx),
        "ny": int(ny),
        "x_scan_rad": (list(np.linspace(-0.05, 0.05, nx)) if x_scan_rad is None
                       else list(map(float, x_scan_rad))),
        "y_scan_rad": (list(np.linspace(0.10, 0.06, ny)) if y_scan_rad is None
                       else list(map(float, y_scan_rad))),
        "planes": planes,
        "plane_order": order,
        "arrays": arrays,
        "payload_bytes": len(payload),
        "content_sha256": hashlib.sha256(payload).hexdigest(),
        "pairs_with_schema": (CWP_SCHEMA_V2 if schema.endswith(".v2")
                              else CWP_SCHEMA),
        "regrid": NO_REGRID,
    }
    if window is not None:
        meta["window"] = list(window)
    if sibling is not None:
        meta["sibling"] = dict(sibling)
    path = Path(path)
    path.write_bytes(_encode(meta, payload))
    return path


def sibling_block(cwp_path) -> dict:
    """The ``sibling`` block ``rw_goes cloud-top --pairs-with`` writes."""

    from gpuwm.obs.goes_pack import read_goes_pack

    pack = read_goes_pack(cwp_path)
    return {
        "schema": pack.schema,
        "filename": Path(cwp_path).name,
        "content_sha256": pack.meta["content_sha256"],
        "nx": int(pack.meta["nx"]),
        "ny": int(pack.meta["ny"]),
    }
