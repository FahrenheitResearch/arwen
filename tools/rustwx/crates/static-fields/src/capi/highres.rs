//! LANE 3 seam: high-resolution overlay compute.
//!
//! One request-JSON call per operation, field sets in and out by
//! handle.  The Python production shell keeps its TOML/receipt/refusal
//! policy and calls these for every byte-transforming step; the fetch
//! driver never opens a raster.
//!
//! Request schemas (all UTF-8 JSON):
//!
//! * `gpuwm_static_highres_terrain`: `{grid_spec?, halo?, smooth_passes?,
//!   terrain: BoundRasterSpec}` — a nonzero grid handle wins over
//!   `grid_spec`; handle 0 uses `grid_spec` (the Python bridge always
//!   sends the spec so both routes work before and after lane 1 lands).
//! * `gpuwm_static_highres_overrides`: adds `landcover`,
//!   `landcover_mapping` (pairs), `soil_sources` (name → spec, name =
//!   `component_depth`), `soil_fallback_handle?`, `iswater?`,
//!   `islake?`, `depth_weights?`.
//! * `gpuwm_static_highres_merge`: `{mode: "terrain"|"all"}` on two
//!   fieldset handles; the merged handle's audit JSON is queried with
//!   `gpuwm_static_highres_audit_json` and released with
//!   `gpuwm_static_highres_audit_drop`.
//! * `gpuwm_static_highres_derive_window`: `{kind:
//!   "terrain-window"|"global-terrain-window"|"landcover-window", ...}`
//!   writing the derived GeoTIFF and answering its audit JSON.
//!
//! Additive helpers for the parity harness and the routed Python
//! bodies: `gpuwm_static_highres_fieldset_new` (arrays → fieldset
//! handle) and `gpuwm_static_highres_usda` (triangle on raw arrays).

use std::collections::BTreeMap;
use std::sync::Mutex;

use super::{
    bytes, clear_error, register_fieldset, set_error, utf8, with_fieldset,
    with_grid, ERR, OK,
};
use crate::error::StaticError;
use crate::highres::{self, BoundRasterSpec};
use crate::projection::GridSpec;
use crate::raster::warp;
use crate::raster::{geotiff, Crs, Raster};
use crate::types::{Field, FieldSet, Grid2, Stack3};
use crate::HALO;

static AUDITS: Mutex<BTreeMap<u64, String>> = Mutex::new(BTreeMap::new());

fn remember_audit(handle: u64, audit: String) {
    AUDITS
        .lock()
        .expect("audit registry poisoned")
        .insert(handle, audit);
}

fn resolve_spec(
    grid: u64,
    from_request: Option<GridSpec>,
) -> Result<GridSpec, String> {
    if grid != 0 {
        if let Some(spec) = with_grid(grid, |grid| grid.spec.clone()) {
            return Ok(spec);
        }
        return Err(format!("unknown grid handle {grid}"));
    }
    from_request.ok_or_else(|| {
        "highres request carries no grid_spec and no grid handle".into()
    })
}

fn copy_out(buf: *mut u8, cap: usize, payload: &str) -> i64 {
    let raw = payload.as_bytes();
    if !buf.is_null() && cap > 0 {
        let n = raw.len().min(cap);
        unsafe { std::ptr::copy_nonoverlapping(raw.as_ptr(), buf, n) };
    }
    raw.len() as i64
}

// ---------------------------------------------------------------------------
// Requests
// ---------------------------------------------------------------------------

#[derive(serde::Deserialize)]
struct TerrainRequest {
    #[serde(default)]
    grid_spec: Option<GridSpec>,
    #[serde(default)]
    halo: Option<usize>,
    #[serde(default = "one")]
    smooth_passes: usize,
    terrain: BoundRasterSpec,
}

fn one() -> usize {
    1
}

#[derive(serde::Deserialize)]
struct OverridesRequest {
    #[serde(default)]
    grid_spec: Option<GridSpec>,
    #[serde(default)]
    halo: Option<usize>,
    #[serde(default = "one")]
    smooth_passes: usize,
    terrain: BoundRasterSpec,
    landcover: BoundRasterSpec,
    landcover_mapping: Vec<(i64, i64)>,
    /// `component_depth` (e.g. `sand_0-5cm`) → source.
    soil_sources: BTreeMap<String, BoundRasterSpec>,
    #[serde(default)]
    soil_fallback_handle: Option<u64>,
    #[serde(default = "default_iswater")]
    iswater: i64,
    #[serde(default = "default_islake")]
    islake: i64,
    /// layer name → ordered (depth, weight) pairs.
    #[serde(default = "default_depth_weights")]
    depth_weights: Vec<(String, Vec<(String, f64)>)>,
}

fn default_iswater() -> i64 {
    17
}

fn default_islake() -> i64 {
    21
}

fn default_depth_weights() -> Vec<(String, Vec<(String, f64)>)> {
    vec![
        (
            "top_0_30cm".into(),
            vec![
                ("0-5cm".into(), 5.0),
                ("5-15cm".into(), 10.0),
                ("15-30cm".into(), 15.0),
            ],
        ),
        (
            "bottom_30_100cm".into(),
            vec![("30-60cm".into(), 30.0), ("60-100cm".into(), 40.0)],
        ),
    ]
}

#[derive(serde::Deserialize)]
struct MergeRequest {
    mode: String,
}

#[derive(serde::Deserialize)]
struct DeriveRequest {
    kind: String,
    #[serde(default)]
    tiles: Vec<std::path::PathBuf>,
    #[serde(default)]
    source: Option<std::path::PathBuf>,
    /// [west, south, east, north] in the tiles' own CRS units.
    #[serde(default)]
    bounds: Option<[f64; 4]>,
    /// [lat_min, lat_max, lon_min, lon_max] footprint for the
    /// landcover clip (densified into the source CRS here).
    #[serde(default)]
    bounds_lonlat: Option<[f64; 4]>,
    #[serde(default)]
    margin_m: Option<f64>,
    #[serde(default)]
    resolution_deg: Option<f64>,
    #[serde(default)]
    sea_level_fill: Option<f64>,
    #[serde(default)]
    source_nodata: Option<f64>,
    out_path: std::path::PathBuf,
}

// ---------------------------------------------------------------------------
// Terrain-only build
// ---------------------------------------------------------------------------

/// Build the terrain-only override for a grid from a derived terrain
/// window.  Writes a field-set handle carrying `HGT_M`.  LANE 3.
///
/// # Safety
/// `request_json`/`request_len` readable UTF-8; `out_handle` writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gpuwm_static_highres_terrain(
    grid: u64,
    request_json: *const u8,
    request_len: usize,
    out_handle: *mut u64,
) -> i32 {
    clear_error();
    let Some(text) = (unsafe { utf8(request_json, request_len) }) else {
        return set_error("highres request pointer/UTF-8 invalid");
    };
    let request: TerrainRequest = match serde_json::from_str(text) {
        Ok(request) => request,
        Err(err) => return set_error(format!("highres terrain JSON: {err}")),
    };
    let spec = match resolve_spec(grid, request.grid_spec) {
        Ok(spec) => spec,
        Err(message) => return set_error(message),
    };
    let halo = request.halo.unwrap_or(HALO);
    let built = request.terrain.open().and_then(|raster| {
        highres::build_terrain_grid(
            &spec,
            &raster,
            halo,
            request.smooth_passes,
        )
    });
    match built {
        Err(err) => set_error(err.to_string()),
        Ok(hgt) => {
            if out_handle.is_null() {
                return set_error("out_handle is null");
            }
            let mut fields = FieldSet::default();
            fields.fields.insert("HGT_M".into(), Field::Plane(hgt));
            unsafe { *out_handle = register_fieldset(fields) };
            OK
        }
    }
}

// ---------------------------------------------------------------------------
// Full overlay build
// ---------------------------------------------------------------------------

fn build_overrides(
    spec: &GridSpec,
    request: &OverridesRequest,
) -> Result<(FieldSet, String), StaticError> {
    let halo = request.halo.unwrap_or(HALO);
    let extended = highres::extended_spec(spec, halo);
    let ny = (spec.e_sn - 1) as usize;
    let nx = (spec.e_we - 1) as usize;
    let (eny, enx) =
        ((extended.e_sn - 1) as usize, (extended.e_we - 1) as usize);
    let crop_stack = |stack: &Stack3| -> Stack3 {
        let mut out = Stack3 {
            planes: stack.planes,
            ny,
            nx,
            data: vec![0.0; stack.planes * ny * nx],
        };
        for plane in 0..stack.planes {
            for row in 0..ny {
                let src =
                    plane * stack.ny * stack.nx + (row + halo) * stack.nx + halo;
                let dst = plane * ny * nx + row * nx;
                out.data[dst..dst + nx]
                    .copy_from_slice(&stack.data[src..src + nx]);
            }
        }
        out
    };

    // Terrain leg.
    let hgt = {
        let raster = request.terrain.open()?;
        highres::build_terrain_grid(
            spec,
            &raster,
            halo,
            request.smooth_passes,
        )?
    };

    // Land-cover leg.
    let (raw, nodata) = {
        request.landcover.verify()?;
        geotiff::read_band1_raw(
            &request.landcover.path,
            request
                .landcover
                .crs_override
                .as_deref()
                .map(Crs::parse_override)
                .transpose()?,
            request.landcover.nodata_override,
        )?
    };
    let mapping: BTreeMap<i64, i64> =
        request.landcover_mapping.iter().copied().collect();
    let luf_extended = highres::resample_mapped_categories(
        &raw,
        &request.landcover.path.display().to_string(),
        &extended,
        &mapping,
        21,
        nodata,
    )?;
    highres::require_coverage("land cover", &luf_extended.data)?;
    let luf = crop_stack(&luf_extended);
    let islake = if request.islake > 0 { Some(request.islake) } else { None };
    let landmask =
        crate::fields::landmask_from_landusef(&luf, request.iswater, islake)?;
    let lu_index = crate::fields::lu_index_from_landusef(
        &luf,
        &landmask,
        request.iswater,
        islake,
    )?;

    // Soil leg.
    let mut audit_layers: BTreeMap<String, serde_json::Value> =
        BTreeMap::new();
    let mut soil_fields: BTreeMap<String, Field> = BTreeMap::new();
    for (layer_name, weights) in &request.depth_weights {
        // Read + co-register the component/depth planes.
        let mut planes: BTreeMap<String, Vec<Vec<f64>>> = BTreeMap::new();
        let mut geometry: Option<(usize, usize, [f64; 6], Crs)> = None;
        for component in ["sand", "silt", "clay"] {
            let mut stack: Vec<Vec<f64>> = Vec::new();
            for (depth, _) in weights {
                let key = format!("{component}_{depth}");
                let source =
                    request.soil_sources.get(&key).ok_or_else(|| {
                        StaticError::Missing(format!(
                            "missing SoilGrids sources: [('{component}', \
                             '{depth}')]"
                        ))
                    })?;
                let raster = source.open()?;
                match &geometry {
                    None => {
                        geometry = Some((
                            raster.ny,
                            raster.nx,
                            raster.transform,
                            raster.crs.clone(),
                        ))
                    }
                    Some((gny, gnx, gtransform, gcrs)) => {
                        if raster.ny != *gny
                            || raster.nx != *gnx
                            || raster.transform != *gtransform
                            || raster.crs != *gcrs
                        {
                            return Err(StaticError::Invalid(
                                "SoilGrids source rasters are not \
                                 co-registered"
                                    .into(),
                            ));
                        }
                    }
                }
                stack.push(raster.values);
            }
            planes.insert(component.into(), stack);
        }
        let (gny, gnx, gtransform, gcrs) = geometry.unwrap();
        let weight_values: Vec<f64> =
            weights.iter().map(|(_, weight)| *weight).collect();
        let (category, valid, raw_total) = highres::soilgrids_categories(
            &planes,
            &weight_values,
            gny * gnx,
        )?;
        let carrier = Raster {
            ny: gny,
            nx: gnx,
            values: vec![0.0; 0],
            transform: gtransform,
            crs: gcrs,
        };
        let (dst_crs, dst_transform, (dny, dnx)) =
            highres::raster_geometry(&extended)?;
        let fractions_extended = warp::reproject_category_fractions(
            &category,
            &valid,
            &carrier,
            &dst_crs,
            dst_transform,
            dny,
            dnx,
            16,
        )?;
        let mut fractions = crop_stack(&fractions_extended);
        debug_assert_eq!((dny, dnx), (eny, enx));

        // Water pillars, missing-land fallback, land normalization —
        // the `build_highres_overrides` soil-layer tail, verbatim.
        let water: Vec<bool> =
            landmask.data.iter().map(|value| *value == 0.0).collect();
        for cell in 0..ny * nx {
            if water[cell] {
                for plane in 0..16 {
                    fractions.data[plane * ny * nx + cell] =
                        if plane == 13 { 1.0 } else { 0.0 };
                }
            }
        }
        let mut missing_land: Vec<bool> = vec![false; ny * nx];
        for cell in 0..ny * nx {
            if water[cell] {
                continue;
            }
            let mut all_finite = true;
            let mut total = 0.0f64;
            for plane in 0..16 {
                let value = fractions.data[plane * ny * nx + cell];
                if !value.is_finite() {
                    all_finite = false;
                } else {
                    total += value;
                }
            }
            missing_land[cell] = !all_finite || total <= 0.0;
        }
        let fallback_name = if layer_name == "top_0_30cm" {
            "SOILCTOP"
        } else {
            "SOILCBOT"
        };
        let missing_count =
            missing_land.iter().filter(|flag| **flag).count();
        if missing_count > 0 {
            let fallback = request
                .soil_fallback_handle
                .and_then(|handle| {
                    with_fieldset(handle, |set| {
                        set.fields.get(fallback_name).cloned()
                    })
                    .flatten()
                })
                .ok_or_else(|| {
                    StaticError::Invalid(format!(
                        "SoilGrids {layer_name} lacks coverage over \
                         {missing_count} target land cells and no \
                         {fallback_name} fallback was supplied"
                    ))
                })?;
            let Field::Stack(fallback) = fallback else {
                return Err(StaticError::Invalid(format!(
                    "{fallback_name} fallback is not a category stack"
                )));
            };
            if (fallback.planes, fallback.ny, fallback.nx) != (16, ny, nx) {
                return Err(StaticError::Invalid(format!(
                    "{fallback_name} fallback shape ({}, {}, {}) differs \
                     from (16, {ny}, {nx})",
                    fallback.planes, fallback.ny, fallback.nx
                )));
            }
            for cell in 0..ny * nx {
                if missing_land[cell] {
                    for plane in 0..16 {
                        fractions.data[plane * ny * nx + cell] =
                            fallback.data[plane * ny * nx + cell];
                    }
                }
            }
        }
        for cell in 0..ny * nx {
            if water[cell] {
                continue;
            }
            let mut total = 0.0f64;
            for plane in 0..16 {
                total += fractions.data[plane * ny * nx + cell];
            }
            for plane in 0..16 {
                fractions.data[plane * ny * nx + cell] /= total;
            }
        }
        let dom = crate::fields::dominant_category(&fractions)?;
        let (frac_name, dom_name) = if layer_name == "top_0_30cm" {
            ("SOILCTOP", "SCT_DOM")
        } else {
            ("SOILCBOT", "SCB_DOM")
        };
        soil_fields.insert(frac_name.into(), Field::Stack(fractions));
        soil_fields.insert(dom_name.into(), Field::Plane(dom));

        let totals: Vec<f64> = raw_total
            .iter()
            .zip(&valid)
            .filter(|(_, ok)| **ok)
            .map(|(value, _)| *value)
            .collect();
        let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
        for value in &totals {
            lo = lo.min(*value);
            hi = hi.max(*value);
        }
        audit_layers.insert(
            layer_name.clone(),
            serde_json::json!({
                "raw_component_total_percent_min": lo,
                "raw_component_total_percent_max": hi,
                "valid_source_pixels":
                    valid.iter().filter(|ok| **ok).count(),
                "fallback_land_cells": missing_count,
            }),
        );
    }

    let mut fields = FieldSet::default();
    fields.fields.insert("HGT_M".into(), Field::Plane(hgt));
    fields.fields.insert("LANDUSEF".into(), Field::Stack(luf));
    fields.fields.insert("LANDMASK".into(), Field::Plane(landmask));
    fields.fields.insert("LU_INDEX".into(), Field::Plane(lu_index));
    for (name, field) in soil_fields {
        fields.fields.insert(name, field);
    }
    let audit = serde_json::json!({ "halo_cells": halo, "soil": audit_layers });
    Ok((fields, audit.to_string()))
}

/// Build the full overlay (terrain + land cover + soil) override set.
/// LANE 3.
///
/// # Safety
/// `request_json`/`request_len` readable UTF-8; `out_handle` writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gpuwm_static_highres_overrides(
    grid: u64,
    request_json: *const u8,
    request_len: usize,
    out_handle: *mut u64,
) -> i32 {
    clear_error();
    let Some(text) = (unsafe { utf8(request_json, request_len) }) else {
        return set_error("highres request pointer/UTF-8 invalid");
    };
    let request: OverridesRequest = match serde_json::from_str(text) {
        Ok(request) => request,
        Err(err) => return set_error(format!("highres overrides JSON: {err}")),
    };
    let spec = match resolve_spec(grid, request.grid_spec.clone()) {
        Ok(spec) => spec,
        Err(message) => return set_error(message),
    };
    match build_overrides(&spec, &request) {
        Err(err) => set_error(err.to_string()),
        Ok((fields, audit)) => {
            if out_handle.is_null() {
                return set_error("out_handle is null");
            }
            let handle = register_fieldset(fields);
            remember_audit(handle, audit);
            unsafe { *out_handle = handle };
            OK
        }
    }
}

// ---------------------------------------------------------------------------
// Merge
// ---------------------------------------------------------------------------

/// Merge overrides into a baseline field set (terrain-only or full,
/// selected by the request), returning a merged field-set handle; the
/// audit JSON is queried with `gpuwm_static_highres_audit_json`.
/// LANE 3.
///
/// # Safety
/// `request_json`/`request_len` readable UTF-8; `out_handle` writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gpuwm_static_highres_merge(
    baseline: u64,
    overrides: u64,
    request_json: *const u8,
    request_len: usize,
    out_handle: *mut u64,
) -> i32 {
    clear_error();
    let Some(text) = (unsafe { utf8(request_json, request_len) }) else {
        return set_error("highres request pointer/UTF-8 invalid");
    };
    let request: MergeRequest = match serde_json::from_str(text) {
        Ok(request) => request,
        Err(err) => return set_error(format!("highres merge JSON: {err}")),
    };
    let Some(baseline_set) = with_fieldset(baseline, FieldSet::clone) else {
        return set_error(format!("unknown fieldset handle {baseline}"));
    };
    let Some(override_set) = with_fieldset(overrides, FieldSet::clone) else {
        return set_error(format!("unknown fieldset handle {overrides}"));
    };
    let merged = match request.mode.as_str() {
        "terrain" => {
            let hgt = match override_set.fields.get("HGT_M") {
                Some(Field::Plane(grid)) => grid.clone(),
                _ => {
                    return set_error(
                        "terrain-only overrides missing ['HGT_M']",
                    )
                }
            };
            highres::merge_terrain_override(&baseline_set, &hgt)
        }
        "all" => {
            highres::merge_highres_overrides(&baseline_set, &override_set)
        }
        other => {
            return set_error(format!(
                "unknown merge mode {other:?} (terrain|all)"
            ))
        }
    };
    match merged {
        Err(err) => set_error(err.to_string()),
        Ok((fields, audit)) => {
            if out_handle.is_null() {
                return set_error("out_handle is null");
            }
            let handle = register_fieldset(fields);
            let audit_json = serde_json::to_string(&audit)
                .unwrap_or_else(|_| "{}".into());
            remember_audit(handle, audit_json);
            unsafe { *out_handle = handle };
            OK
        }
    }
}

/// Copy the audit JSON remembered for a fieldset handle; returns the
/// full length, or -1 with the error set.
///
/// # Safety
/// `buf` writable for `cap` bytes, or null with `cap` 0.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gpuwm_static_highres_audit_json(
    handle: u64,
    buf: *mut u8,
    cap: usize,
) -> i64 {
    clear_error();
    let registry = AUDITS.lock().expect("audit registry poisoned");
    match registry.get(&handle) {
        None => {
            drop(registry);
            set_error(format!("no audit recorded for fieldset {handle}"));
            ERR as i64
        }
        Some(audit) => copy_out(buf, cap, audit),
    }
}

/// Release the audit JSON remembered for a fieldset handle.
#[unsafe(no_mangle)]
pub extern "C" fn gpuwm_static_highres_audit_drop(handle: u64) {
    AUDITS.lock().expect("audit registry poisoned").remove(&handle);
}

// ---------------------------------------------------------------------------
// Fieldset construction from caller arrays
// ---------------------------------------------------------------------------

#[derive(serde::Deserialize)]
struct FieldsetSpecEntry {
    name: String,
    planes: usize,
    ny: usize,
    nx: usize,
}

#[derive(serde::Deserialize)]
struct FieldsetSpec {
    fields: Vec<FieldsetSpecEntry>,
}

/// Build a fieldset handle from caller-supplied f64 arrays: `spec`
/// names every field and its dims, `data` carries the concatenated
/// C-order float64 planes in spec order.  LANE 3 (harness + routed
/// merge bodies).
///
/// # Safety
/// `spec_json`/`spec_len` readable UTF-8; `data` readable for
/// `data_len` f64; `out_handle` writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gpuwm_static_highres_fieldset_new(
    spec_json: *const u8,
    spec_len: usize,
    data: *const f64,
    data_len: usize,
    out_handle: *mut u64,
) -> i32 {
    clear_error();
    let Some(text) = (unsafe { utf8(spec_json, spec_len) }) else {
        return set_error("fieldset spec pointer/UTF-8 invalid");
    };
    let spec: FieldsetSpec = match serde_json::from_str(text) {
        Ok(spec) => spec,
        Err(err) => return set_error(format!("fieldset spec JSON: {err}")),
    };
    let total: usize = spec
        .fields
        .iter()
        .map(|entry| entry.planes * entry.ny * entry.nx)
        .sum();
    if total != data_len {
        return set_error(format!(
            "fieldset spec declares {total} values, caller offered \
             {data_len}"
        ));
    }
    if data.is_null() && data_len > 0 {
        return set_error("fieldset data pointer is null");
    }
    let values =
        unsafe { std::slice::from_raw_parts(data, data_len) };
    let mut fields = FieldSet::default();
    let mut cursor = 0usize;
    for entry in &spec.fields {
        let n = entry.planes * entry.ny * entry.nx;
        let slice = values[cursor..cursor + n].to_vec();
        cursor += n;
        let field = if entry.planes == 1 {
            Field::Plane(Grid2 { ny: entry.ny, nx: entry.nx, data: slice })
        } else {
            Field::Stack(Stack3 {
                planes: entry.planes,
                ny: entry.ny,
                nx: entry.nx,
                data: slice,
            })
        };
        fields.fields.insert(entry.name.clone(), field);
    }
    if out_handle.is_null() {
        return set_error("out_handle is null");
    }
    unsafe { *out_handle = register_fieldset(fields) };
    OK
}

// ---------------------------------------------------------------------------
// USDA triangle on raw arrays
// ---------------------------------------------------------------------------

/// The USDA texture triangle on raw f64 arrays; writes i16 categories.
/// LANE 3 (routed `usda_texture_category` body).
///
/// # Safety
/// `sand`/`silt`/`clay` readable for `n` f64; `out` writable for `n`
/// i16.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gpuwm_static_highres_usda(
    sand: *const f64,
    silt: *const f64,
    clay: *const f64,
    n: usize,
    out: *mut i16,
) -> i32 {
    clear_error();
    if (sand.is_null() || silt.is_null() || clay.is_null() || out.is_null())
        && n > 0
    {
        return set_error("usda buffers are null");
    }
    let sand = unsafe { std::slice::from_raw_parts(sand, n) };
    let silt = unsafe { std::slice::from_raw_parts(silt, n) };
    let clay = unsafe { std::slice::from_raw_parts(clay, n) };
    match highres::usda_texture_category(sand, silt, clay) {
        Err(err) => set_error(err.to_string()),
        Ok(categories) => {
            unsafe {
                std::ptr::copy_nonoverlapping(categories.as_ptr(), out, n)
            };
            OK
        }
    }
}

// ---------------------------------------------------------------------------
// The warp substrate: one call per `resample_*` Python entry point
// ---------------------------------------------------------------------------

#[derive(serde::Deserialize)]
struct ResampleRequest {
    /// `continuous` | `mapped-categories` | `soil-categories`.
    kind: String,
    grid_spec: GridSpec,
    #[serde(default)]
    source: Option<BoundRasterSpec>,
    #[serde(default)]
    method: Option<String>,
    #[serde(default)]
    mapping: Vec<(i64, i64)>,
    #[serde(default = "sixteen")]
    category_count: usize,
    /// `component_depth` (e.g. `sand_0-5cm`) -> source.
    #[serde(default)]
    soil_sources: BTreeMap<String, BoundRasterSpec>,
    /// Ordered `(depth, weight)` pairs; the accumulation order defines
    /// the depth-weighted mean's bits.
    #[serde(default)]
    depth_weights: Vec<(String, f64)>,
}

fn sixteen() -> usize {
    16
}

/// Read + co-register the SoilGrids component planes, take the
/// depth-weighted mean, classify it and warp the categories onto the
/// grid: `_soilgrids_categories` followed by `_resample_category_array`,
/// which is how `build_highres_overrides` uses them.
fn soil_category_fractions(
    request: &ResampleRequest,
) -> Result<(Stack3, serde_json::Value), StaticError> {
    if request.depth_weights.is_empty() {
        return Err(StaticError::Invalid(
            "soil-categories requires depth_weights".into(),
        ));
    }
    let mut planes: BTreeMap<String, Vec<Vec<f64>>> = BTreeMap::new();
    let mut geometry: Option<(usize, usize, [f64; 6], Crs)> = None;
    for component in ["sand", "silt", "clay"] {
        let mut stack: Vec<Vec<f64>> = Vec::new();
        for (depth, _) in &request.depth_weights {
            let key = format!("{component}_{depth}");
            let source = request.soil_sources.get(&key).ok_or_else(|| {
                StaticError::Missing(format!(
                    "missing SoilGrids sources: [('{component}', \
                     '{depth}')]"
                ))
            })?;
            let raster = source.open()?;
            match &geometry {
                None => {
                    geometry = Some((
                        raster.ny,
                        raster.nx,
                        raster.transform,
                        raster.crs.clone(),
                    ))
                }
                Some((gny, gnx, gtransform, gcrs)) => {
                    if raster.ny != *gny
                        || raster.nx != *gnx
                        || raster.transform != *gtransform
                        || raster.crs != *gcrs
                    {
                        return Err(StaticError::Invalid(
                            "SoilGrids source rasters are not \
                             co-registered"
                                .into(),
                        ));
                    }
                }
            }
            stack.push(raster.values);
        }
        planes.insert(component.into(), stack);
    }
    let (gny, gnx, gtransform, gcrs) = geometry.expect("three components");
    let weights: Vec<f64> =
        request.depth_weights.iter().map(|(_, w)| *w).collect();
    let (category, valid, raw_total) =
        highres::soilgrids_categories(&planes, &weights, gny * gnx)?;
    let carrier = Raster {
        ny: gny,
        nx: gnx,
        values: vec![0.0; 0],
        transform: gtransform,
        crs: gcrs,
    };
    let (dst_crs, dst_transform, (dny, dnx)) =
        highres::raster_geometry(&request.grid_spec)?;
    let fractions = warp::reproject_category_fractions(
        &category,
        &valid,
        &carrier,
        &dst_crs,
        dst_transform,
        dny,
        dnx,
        request.category_count,
    )?;
    let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
    for (value, ok) in raw_total.iter().zip(&valid) {
        if *ok {
            lo = lo.min(*value);
            hi = hi.max(*value);
        }
    }
    let audit = serde_json::json!({
        "raw_component_total_percent_min": lo,
        "raw_component_total_percent_max": hi,
        "valid_source_pixels": valid.iter().filter(|ok| **ok).count(),
    });
    Ok((fractions, audit))
}

fn resample(
    request: &ResampleRequest,
) -> Result<(FieldSet, String), StaticError> {
    let mut fields = FieldSet::default();
    match request.kind.as_str() {
        "continuous" => {
            let source = request.source.as_ref().ok_or_else(|| {
                StaticError::Invalid(
                    "continuous resampling requires a source raster".into(),
                )
            })?;
            let method = warp::Resampling::parse(
                request.method.as_deref().unwrap_or("average"),
            )?;
            let raster = source.open()?;
            let plane = highres::resample_continuous(
                &raster,
                &request.grid_spec,
                method,
            )?;
            fields.fields.insert("VALUES".into(), Field::Plane(plane));
            Ok((fields, "{}".into()))
        }
        "mapped-categories" => {
            let source = request.source.as_ref().ok_or_else(|| {
                StaticError::Invalid(
                    "mapped-category resampling requires a source raster"
                        .into(),
                )
            })?;
            source.verify()?;
            let (raw, nodata) = geotiff::read_band1_raw(
                &source.path,
                source
                    .crs_override
                    .as_deref()
                    .map(Crs::parse_override)
                    .transpose()?,
                source.nodata_override,
            )?;
            let mapping: BTreeMap<i64, i64> =
                request.mapping.iter().copied().collect();
            let fractions = highres::resample_mapped_categories(
                &raw,
                &source.path.display().to_string(),
                &request.grid_spec,
                &mapping,
                request.category_count,
                nodata,
            )?;
            fields
                .fields
                .insert("FRACTIONS".into(), Field::Stack(fractions));
            Ok((fields, "{}".into()))
        }
        "soil-categories" => {
            let (fractions, audit) = soil_category_fractions(request)?;
            fields
                .fields
                .insert("FRACTIONS".into(), Field::Stack(fractions));
            Ok((fields, audit.to_string()))
        }
        other => Err(StaticError::Invalid(format!(
            "unknown resample kind {other:?} (continuous, \
             mapped-categories, soil-categories)"
        ))),
    }
}

/// The warp substrate behind `gpuwm.static.highres`'s `resample_*`
/// entry points: decode one bound raster (or the six co-registered
/// SoilGrids planes), then reproject onto the model grid.  Writes a
/// field-set handle carrying `VALUES` (continuous) or `FRACTIONS`
/// (categorical); the soil kind also remembers an audit document,
/// queried with `gpuwm_static_highres_audit_json`.  LANE 3.
///
/// # Safety
/// `request_json`/`request_len` readable UTF-8; `out_handle` writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gpuwm_static_highres_resample(
    request_json: *const u8,
    request_len: usize,
    out_handle: *mut u64,
) -> i32 {
    clear_error();
    let Some(text) = (unsafe { utf8(request_json, request_len) }) else {
        return set_error("highres request pointer/UTF-8 invalid");
    };
    let request: ResampleRequest = match serde_json::from_str(text) {
        Ok(request) => request,
        Err(err) => return set_error(format!("highres resample JSON: {err}")),
    };
    match resample(&request) {
        Err(err) => set_error(err.to_string()),
        Ok((fields, audit)) => {
            if out_handle.is_null() {
                return set_error("out_handle is null");
            }
            let handle = register_fieldset(fields);
            remember_audit(handle, audit);
            unsafe { *out_handle = handle };
            OK
        }
    }
}

// ---------------------------------------------------------------------------
// Point transforms
// ---------------------------------------------------------------------------

#[derive(serde::Deserialize)]
struct TransformRequest {
    /// A recorded `crs_override` string, or `epsg:4326`.
    to: String,
}

/// Transform `n` lon/lat degree pairs (EPSG:4326, x=lon) in place into
/// the CRS named by the request.  The fetch driver's window snap needs
/// exactly this and nothing else, so it crosses as points rather than
/// dragging a projection object into Python.  LANE 3.
///
/// # Safety
/// `request_json`/`request_len` readable UTF-8; `x`/`y` readable and
/// writable for `n` f64 each.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gpuwm_static_highres_transform_points(
    request_json: *const u8,
    request_len: usize,
    x: *mut f64,
    y: *mut f64,
    n: usize,
) -> i32 {
    clear_error();
    let Some(text) = (unsafe { utf8(request_json, request_len) }) else {
        return set_error("highres request pointer/UTF-8 invalid");
    };
    let request: TransformRequest = match serde_json::from_str(text) {
        Ok(request) => request,
        Err(err) => {
            return set_error(format!("highres transform JSON: {err}"))
        }
    };
    if (x.is_null() || y.is_null()) && n > 0 {
        return set_error("transform_points buffers are null");
    }
    let to = match Crs::parse_override(&request.to) {
        Ok(crs) => crs,
        Err(err) => return set_error(err.to_string()),
    };
    let xs = unsafe { std::slice::from_raw_parts_mut(x, n) };
    let ys = unsafe { std::slice::from_raw_parts_mut(y, n) };
    match crate::raster::transform_points(&Crs::Geographic, &to, xs, ys) {
        Err(err) => set_error(err.to_string()),
        Ok(()) => OK,
    }
}

// ---------------------------------------------------------------------------
// Derived windows
// ---------------------------------------------------------------------------

/// A derive-window refusal, plus whether it is a SOURCE-COVERAGE
/// refusal rather than a defect.
///
/// The distinction is not cosmetic and it is not inferable from the
/// message: `gpuwm.static.highres_production` routes a coverage
/// refusal through the user's `on_refuse` policy (`fallback-30s`
/// proceeds on the unchanged 30-arc-second baseline), while anything
/// else is a fault that must stop the case.  Collapsing the two would
/// let a corrupt raster be silently answered with baseline terrain, so
/// the seam reports it as a distinct return code (-2) rather than
/// asking Python to pattern-match a sentence.
struct DeriveFailure {
    coverage: bool,
    error: StaticError,
}

impl DeriveFailure {
    fn coverage(message: String) -> Self {
        DeriveFailure { coverage: true, error: StaticError::Invalid(message) }
    }
}

impl From<StaticError> for DeriveFailure {
    fn from(error: StaticError) -> Self {
        DeriveFailure { coverage: false, error }
    }
}

impl From<std::io::Error> for DeriveFailure {
    fn from(error: std::io::Error) -> Self {
        DeriveFailure { coverage: false, error: StaticError::Io(error) }
    }
}

fn derive_window(request: &DeriveRequest) -> Result<String, DeriveFailure> {
    match request.kind.as_str() {
        "terrain-window" | "global-terrain-window" => {
            let bounds = request.bounds.ok_or_else(|| {
                StaticError::Invalid(format!(
                    "derive kind {:?} requires bounds [w, s, e, n]",
                    request.kind
                ))
            })?;
            if request.tiles.is_empty() {
                return Err(StaticError::Invalid(
                    "terrain window derivation requires >= 1 tile".into(),
                )
                .into());
            }
            let mut tiles: Vec<Raster> = Vec::with_capacity(
                request.tiles.len(),
            );
            let mut first_nodata: Option<f64> = None;
            for (index, path) in request.tiles.iter().enumerate() {
                let (raster, nodata) =
                    geotiff::read_band1_raw(path, Some(Crs::Geographic), None)?;
                if index == 0 {
                    first_nodata = nodata;
                }
                // Mask the tile's own nodata so painting skips it.
                let mut raster = raster;
                if let Some(sentinel) = nodata {
                    for value in raster.values.iter_mut() {
                        if *value == sentinel {
                            *value = f64::NAN;
                        }
                    }
                }
                tiles.push(raster);
            }
            let global = request.kind == "global-terrain-window";
            let resolution =
                if global { Some(request.resolution_deg.unwrap_or(1.0 / 3600.0)) } else { None };
            let (mut mosaic, holes) = warp::mosaic(
                &tiles,
                bounds,
                resolution,
                request.source_nodata,
            )?;
            let audit;
            if global {
                let fill = request.sea_level_fill.unwrap_or(0.0);
                // The Python fills on the float32 plane; replicate the
                // f32 fill value bit for bit.
                let fill = fill as f32 as f64;
                for value in mosaic.values.iter_mut() {
                    if value.is_nan() {
                        *value = fill;
                    } else {
                        // Round-trip through f32 like the Python's
                        // `dtype="float32"` mosaic plane.
                        *value = *value as f32 as f64;
                    }
                }
                audit = serde_json::json!({
                    "output_resolution_deg":
                        resolution.unwrap_or(1.0 / 3600.0),
                    "output_shape": [mosaic.ny, mosaic.nx],
                    "sea_level_filled_pixels": holes,
                    "total_pixels": mosaic.ny * mosaic.nx,
                    "sea_level_fill_m": request.sea_level_fill.unwrap_or(0.0),
                    "source_nodata": request.source_nodata,
                    "resampling":
                        "nearest (latitude-banded source resolutions)",
                });
                geotiff::write_band1(
                    &request.out_path,
                    &mosaic,
                    geotiff::SampleType::F32,
                    None,
                )?;
            } else {
                audit = serde_json::json!({
                    "output_shape": [mosaic.ny, mosaic.nx],
                    "hole_pixels": holes,
                    "total_pixels": mosaic.ny * mosaic.nx,
                    "nodata": first_nodata,
                });
                geotiff::write_band1(
                    &request.out_path,
                    &mosaic,
                    geotiff::SampleType::F32,
                    first_nodata,
                )?;
            }
            Ok(audit.to_string())
        }
        "landcover-window" => {
            let source_path = request.source.as_ref().ok_or_else(|| {
                StaticError::Invalid(
                    "landcover-window requires a source raster".into(),
                )
            })?;
            let bbox = request.bounds_lonlat.ok_or_else(|| {
                StaticError::Invalid(
                    "landcover-window requires bounds_lonlat".into(),
                )
            })?;
            let margin = request.margin_m.unwrap_or(2000.0);
            let mut reader = geotiff::TiffReader::open(source_path)?;
            let crs = reader.crs.clone().ok_or_else(|| {
                StaticError::Invalid(format!(
                    "raster {source_path:?} has no CRS and declares no \
                     crs_override"
                ))
            })?;
            // `_densified_bounds`: 41x41 lon/lat mesh -> source CRS.
            let [lat_min, lat_max, lon_min, lon_max] = bbox;
            let projection = crs.point_projection()?;
            let mut west = f64::INFINITY;
            let mut south = f64::INFINITY;
            let mut east = f64::NEG_INFINITY;
            let mut north = f64::NEG_INFINITY;
            for j in 0..41 {
                let lat =
                    lat_min + (lat_max - lat_min) * j as f64 / 40.0;
                for i in 0..41 {
                    let lon =
                        lon_min + (lon_max - lon_min) * i as f64 / 40.0;
                    let (x, y) = projection.forward(lon, lat);
                    if x.is_finite() && y.is_finite() {
                        west = west.min(x);
                        east = east.max(x);
                        south = south.min(y);
                        north = north.max(y);
                    }
                }
            }
            west -= margin;
            south -= margin;
            east += margin;
            north += margin;
            // Fractional window + explicit outward rounding, the
            // Python's arithmetic.
            let t = reader.transform;
            let col_off_f = (west - t[2]) / t[0];
            let row_off_f = (north - t[5]) / t[4];
            let width_f = (east - west) / t[0];
            let height_f = (south - north) / t[4];
            let col_off = col_off_f.floor();
            let row_off = row_off_f.floor();
            let width =
                (width_f + (col_off_f - col_off)).ceil() as i64;
            let height =
                (height_f + (row_off_f - row_off)).ceil() as i64;
            let (col_off, row_off) = (col_off as i64, row_off as i64);
            let full_w = reader.width as i64;
            let full_h = reader.height as i64;
            let clip_col = col_off.max(0);
            let clip_row = row_off.max(0);
            let clip_w = (col_off + width).min(full_w) - clip_col;
            let clip_h = (row_off + height).min(full_h) - clip_row;
            if clip_w <= 0 || clip_h <= 0 {
                return Err(DeriveFailure::coverage(format!(
                    "footprint [{lat_min}, {lat_max}] x [{lon_min}, \
                     {lon_max}] lies outside the land-cover raster \
                     extent of {}",
                    source_path.display()
                )));
            }
            if clip_w != width || clip_h != height {
                return Err(DeriveFailure::coverage(format!(
                    "footprint [{lat_min}, {lat_max}] x [{lon_min}, \
                     {lon_max}] is only partially covered by the \
                     land-cover raster {}; the source window would be \
                     truncated from {width}x{height} to \
                     {clip_w}x{clip_h} pixels",
                    source_path.display()
                )));
            }
            let nodata = reader.nodata;
            let values = reader.read_window_raw(
                clip_col as usize,
                clip_row as usize,
                clip_w as usize,
                clip_h as usize,
            )?;
            let window = Raster {
                ny: clip_h as usize,
                nx: clip_w as usize,
                values,
                transform: [
                    t[0],
                    t[1],
                    t[2] + t[0] * clip_col as f64,
                    t[3],
                    t[4],
                    t[5] + t[4] * clip_row as f64,
                ],
                crs,
            };
            geotiff::write_band1(
                &request.out_path,
                &window,
                geotiff::SampleType::U8,
                nodata,
            )?;
            Ok(serde_json::json!({
                "output_shape": [window.ny, window.nx],
                "window": [clip_col, clip_row, clip_w, clip_h],
                "nodata": nodata,
            })
            .to_string())
        }
        other => Err(StaticError::Invalid(format!(
            "unknown derive-window kind {other:?} (terrain-window, \
             global-terrain-window, landcover-window)"
        ))
        .into()),
    }
}

/// Derive one cached raster window (mosaic/clip/fill; replaces the
/// Python `derive_*_window` byte transforms).  Writes the derived
/// GeoTIFF and copies its audit JSON into `buf`, returning the full
/// length; on refusal the error is set and the return is negative:
/// **-2 for a SOURCE-COVERAGE refusal** (the footprint reaches past
/// the published raster, which the caller's `on_refuse` policy may
/// legitimately answer with the 30-arc-second baseline) and -1 for
/// everything else (a defect that must stop the case).  LANE 3.
///
/// # Safety
/// `request_json`/`request_len` readable UTF-8; `buf` writable for
/// `cap` bytes or null with `cap` 0.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gpuwm_static_highres_derive_window(
    request_json: *const u8,
    request_len: usize,
    buf: *mut u8,
    cap: usize,
) -> i64 {
    clear_error();
    let Some(text) = (unsafe { utf8(request_json, request_len) }) else {
        set_error("highres request pointer/UTF-8 invalid");
        return -1;
    };
    let request: DeriveRequest = match serde_json::from_str(text) {
        Ok(request) => request,
        Err(err) => {
            set_error(format!("highres derive JSON: {err}"));
            return -1;
        }
    };
    match derive_window(&request) {
        Err(failure) => {
            set_error(failure.error.to_string());
            if failure.coverage {
                -2
            } else {
                -1
            }
        }
        Ok(audit) => copy_out(buf, cap, &audit),
    }
}

/// Keep the shared byte helper referenced (mirror of grid.rs).
#[allow(dead_code)]
fn _reserved(ptr: *const u8, len: usize) -> usize {
    unsafe { bytes(ptr, len) }.map_or(0, <[u8]>::len)
}
