//! Build isobaric sounding volumes from a WRF file.
//!
//! Updated from BowEcho v0.30.5's memory-hardened volume builder so Rusty
//! Weather owns the optimized implementation consumed by desktop hosts.
//!
//! WRF is on native (eta) levels, but the skew-T builder
//! ([`rw_ui::skewt::build_sounding_column`]) needs the same `*_iso` isobaric
//! 3D variables the model ingest writes for HRRR/GFS: `temperature_iso`,
//! `dewpoint_iso`, `u_iso`, `v_iso`, `height_iso`. This module reads WRF's 3D
//! fields through `wrf-core`'s `getvar` (which already handles destaggering,
//! theta -> T, geopotential -> height, and QVAPOR -> Td) and log-pressure
//! interpolates each column onto the canonical isobaric levels, so imported
//! WRF runs produce soundings exactly like the downloaded models do.
#![allow(dead_code)]
// `try_interpolate_iso_volumes` takes the five column fields + shape as separate
// slices by design (the shared raw/post-processed reader contract); factoring
// them into a struct would only obscure the call sites.
#![allow(clippy::too_many_arguments)]

use rustwx_core::checked_volume_elements;
use rw_store::PressureVolumeInput;
use wrf_core::{ComputeOpts, VarOutput, WrfFile, getvar};

const STANDARD_LEVEL_COUNT: usize = 37;
const MAX_NATIVE_F64_COMPONENT_COUNT: u128 = 8;
const ISO_VOLUME_COUNT: u128 = 5;
const SURFACE_F32_PLANE_COUNT: u128 = 5;
/// Ceiling for buffers owned directly by the volume path. This deliberately
/// excludes wrf-core's memoization cache, whose lifetime is managed separately.
/// The known 800x800x79 workflow needs 3,722,240,000 bytes (~3.47 GiB) by this
/// accounting and therefore remains supported with useful allocation headroom.
const MAX_WRF_VOLUME_OWNED_BYTES: u128 = 4 * 1024 * 1024 * 1024;

/// Canonical isobaric levels (hPa), matching the model-ingest convention
/// (`100..=1000` step 25 -> 37 levels). Levels outside a column's model range
/// are left NaN and pruned by the sounding column builder.
fn standard_levels() -> Vec<u16> {
    (100..=1000u16).step_by(25).collect()
}

/// Preflight the complete known owned working set before any 3-D reads or
/// output allocations. The five `*_iso` products all share the canonical
/// 37-level shape. Raw wrfout owns at most six native-sized f64 components;
/// postprocessed severe processing can retain pressure/QVAPOR alongside its
/// derived hPa/dewpoint arrays and therefore reaches eight. We conservatively
/// budget eight for every caller. Five f32 lowest-level surface planes are also
/// included.
///
/// Callers that receive an error must not begin the 3-D volume read. A caller
/// that already owns independent 2-D products may retain them; a volume-only
/// caller may instead return the error. The returned value is the total known
/// owned byte count. The per-volume store ceiling remains an independent check
/// in addition to the aggregate 4 GiB desktop working-set ceiling.
pub(crate) fn preflight_iso_volume_shape(nz: usize, cells: usize) -> Result<u64, String> {
    if nz < 2 {
        return Err(format!(
            "WRF native pressure volume requires at least two levels, got {nz}"
        ));
    }
    if cells == 0 {
        return Err("WRF grid has zero cells".to_string());
    }
    let iso_elements = checked_volume_elements(STANDARD_LEVEL_COUNT, cells).map_err(|err| {
        format!("canonical {STANDARD_LEVEL_COUNT}-level WRF pressure volume is unsupported: {err}")
    })?;

    let native_bytes = checked_byte_product(
        "native WRF volume buffers",
        &[
            MAX_NATIVE_F64_COMPONENT_COUNT,
            nz as u128,
            cells as u128,
            std::mem::size_of::<f64>() as u128,
        ],
    )?;
    let iso_bytes = checked_byte_product(
        "isobaric WRF output buffers",
        &[
            ISO_VOLUME_COUNT,
            iso_elements as u128,
            std::mem::size_of::<f32>() as u128,
        ],
    )?;
    let surface_bytes = checked_byte_product(
        "WRF surface fallback buffers",
        &[
            SURFACE_F32_PLANE_COUNT,
            cells as u128,
            std::mem::size_of::<f32>() as u128,
        ],
    )?;
    let owned_bytes = native_bytes
        .checked_add(iso_bytes)
        .and_then(|bytes| bytes.checked_add(surface_bytes))
        .ok_or_else(|| "WRF volume owned-byte total overflows u128".to_string())?;
    if owned_bytes > MAX_WRF_VOLUME_OWNED_BYTES {
        return Err(format!(
            "WRF volume path requires {owned_bytes} known owned bytes for {nz} levels x {cells} cells, exceeding the {MAX_WRF_VOLUME_OWNED_BYTES}-byte (4 GiB) desktop ceiling"
        ));
    }
    u64::try_from(owned_bytes)
        .map_err(|_| format!("WRF volume owned-byte total {owned_bytes} does not fit u64"))
}

fn checked_byte_product(name: &str, factors: &[u128]) -> Result<u128, String> {
    factors.iter().try_fold(1u128, |product, &factor| {
        product
            .checked_mul(factor)
            .ok_or_else(|| format!("{name} factors {factors:?} overflow u128"))
    })
}

/// One isobaric volume ready for the store writer: owned row-major planes.
pub struct IsoVolume {
    pub name: String,
    pub units: String,
    /// `(level_hpa, plane)` where each plane holds `ny * nx` row-major values.
    pub levels: Vec<(u16, Vec<f32>)>,
}

impl IsoVolume {
    /// Borrowed view for the store writer's [`PressureVolumeInput`].
    pub fn as_input(&self) -> PressureVolumeInput<'_> {
        PressureVolumeInput {
            name: &self.name,
            units: &self.units,
            selector_template: serde_json::json!({
                "source": "wrf",
                "field": self.name,
                "vertical": "isobaric",
            }),
            levels: self
                .levels
                .iter()
                .map(|(hpa, plane)| (*hpa, plane.as_slice()))
                .collect(),
        }
    }
}

/// Lowest-model-level surface fallbacks, in the units the skew-T expects
/// (Pa, K, K, m/s, m/s). Used to synthesize the 2D surface fields a split
/// `wrf3d` file (CONUS404 / GDEX CONUS-II) omits — chiefly `PSFC` — so the
/// sounding can still start near the surface. Callers expose these substitutes
/// only under explicit `approx_*` names, never as true 2 m/10 m products. Each
/// plane is row-major `ny * nx`.
pub struct SurfaceFallback {
    pub surface_pressure_pa: Vec<f32>,
    pub temperature_2m_k: Vec<f32>,
    pub dewpoint_2m_k: Vec<f32>,
    pub u_10m: Vec<f32>,
    pub v_10m: Vec<f32>,
}

/// Read WRF 3D fields for `timeidx` and interpolate them to the canonical
/// isobaric levels, returning the five `*_iso` volumes the skew-T needs plus
/// the lowest-model-level [`SurfaceFallback`] (so callers can fill in any 2D
/// surface field the file omits).
///
/// `cells` is the horizontal grid size (`ny * nx`) of the hour being written;
/// every returned plane matches it. Fails (leaving the caller to skip volumes
/// and still write the 2D fields) if the required 3D fields are unreadable.
///
/// `progress` receives per-stage messages (which 3D field is being read /
/// getvar'd, then interpolation percentage) — on a 250 m grid each stage is
/// tens of seconds, and both import paths surface these lines in the dock.
/// The isobaric levels the production direct chart recipes plot.  RH and
/// absolute vorticity interpolate at these levels only: unlike the five
/// sounding fields they feed no skew-T column, so full 37-level volumes
/// would be pure store weight.
pub(crate) const CHART_RECIPE_LEVELS_HPA: [u16; 6] = [200, 250, 300, 500, 700, 850];

pub fn build_iso_volumes(
    file: &WrfFile,
    timeidx: usize,
    cells: usize,
    progress: &mut dyn FnMut(String),
) -> Result<(Vec<IsoVolume>, Vec<IsoVolume>, SurfaceFallback), String> {
    // This must precede the first getvar: an otherwise valid 2-D grid can be
    // too large for either a 37-level dense store volume or the aggregate
    // native/output working set. In that case callers omit these products
    // rather than reading several enormous native fields.
    let (nx, ny, nz) = (file.nx, file.ny, file.nz);
    let file_cells = checked_dimension_product("WRF horizontal grid", &[ny, nx])?;
    if cells != file_cells {
        return Err(format!(
            "WRF caller supplied {cells} horizontal cells, but file dimensions [{ny}, {nx}] describe {file_cells}"
        ));
    }
    preflight_iso_volume_shape(nz, cells)?;
    let read = |name: &str, stage: &str| -> Result<VarOutput, String> {
        getvar(file, name, Some(timeidx), &ComputeOpts::default())
            .map_err(|err| format!("read WRF {name} ({stage}): {err}"))
    };

    progress("reading WRF pressure (sounding field 1/5)".to_string());
    let pressure = read("pressure", "sounding field 1/5")?; // hPa, [nz, ny, nx]
    let expected_3d = check_native_3d_output(&pressure, "pressure", nz, ny, nx)?;

    progress("reading WRF temperature (sounding field 2/5)".to_string());
    let temp = read("temp", "sounding field 2/5")?; // K
    check_native_3d_output(&temp, "temp", nz, ny, nx)?;
    progress("reading WRF dewpoint (sounding field 3/5)".to_string());
    let td = read("td", "sounding field 3/5")?; // degC
    check_native_3d_output(&td, "td", nz, ny, nx)?;
    progress("reading WRF height (sounding field 4/5)".to_string());
    let height = read("height", "sounding field 4/5")?; // m MSL
    check_native_3d_output(&height, "height", nz, ny, nx)?;

    // Earth-relative winds. `uvmet` returns [u_earth.., v_earth..]
    // (2 * nz * cells). There is intentionally NO ua/va fallback: those are
    // grid-relative components, while the store's canonical u_iso/v_iso fields
    // drive geographic sounding wind barbs. Publishing ua/va under those names
    // silently rotates every profile away from true north on projected grids.
    // Split without copying: on a 50 M-cell grid the two halves are ~400 MB
    // each, and `to_vec`-ing them while the 800 MB source was still alive
    // measurably spiked the peak working set of the whole import.
    progress("reading WRF winds (sounding field 5/5)".to_string());
    let uvmet = read("uvmet", "sounding field 5/5")?;
    let wind_data = validate_earth_relative_uvmet(uvmet, nz, ny, nx, expected_3d)?;

    // Chart-recipe planes: RH and absolute vorticity at the six production
    // chart levels.  Read while the memoized cache is still warm (their
    // dependency chains share the intermediates above), interpolate each at
    // once, and DROP the native 3-D array before the next read -- the peak
    // working set grows by exactly one native field, keeping the raw-wrfout
    // path inside the preflight's eight-component budget.  Either field's
    // failure degrades to a progress note: the sounding volumes and every
    // 2-D product still publish, and the chart recipes that needed the
    // missing planes stay honestly "not stored".
    let mut recipe_volumes: Vec<IsoVolume> = Vec::new();
    for (var, stage, volume_name, units) in [
        ("rh", "chart-level relative humidity", "rh_chart_levels", "%"),
        ("avo", "chart-level absolute vorticity", "avo_chart_levels", "s-1"),
    ] {
        progress(format!("reading WRF {var} ({stage})"));
        match read(var, stage) {
            Ok(output) => match check_native_3d_output(&output, var, nz, ny, nx) {
                Ok(_) => {
                    let planes = interpolate_field_at_levels(
                        &pressure.data,
                        &output.data,
                        nz,
                        cells,
                        &CHART_RECIPE_LEVELS_HPA,
                    )?;
                    recipe_volumes.push(IsoVolume {
                        name: volume_name.to_string(),
                        units: units.to_string(),
                        levels: planes,
                    });
                }
                Err(err) => progress(format!("{volume_name} skipped: {err}")),
            },
            Err(err) => progress(format!("{volume_name} skipped: {err}")),
        }
    }

    // The hour's LAST `getvar` is behind us, and every input the interpolator
    // needs is owned above — release wrf-core's memoized 3-D f64 intermediates
    // NOW, before the interpolation loop and the store write. `getvar`
    // memoizes every intermediate (full pressure, theta, temperature,
    // geopotential, heights, QVAPOR, destaggered winds, …) inside `WrfFile`
    // and only evicts on a timestep CHANGE; on the 800×800×79 Enderlin grid
    // that cache is ~5 GB of dead weight from here on. Clearing any EARLIER
    // was measured to more than double the peak (every read recomputes its
    // whole dependency chain — see docs/wrf-import-large-grids.md); clearing
    // here costs zero recompute. catch_unwind: a poisoned cache mutex (from a
    // caught diagnostic panic upstream) must not fail the volumes.
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| file.clear_cache()));

    // Dewpoint arrives in degC from wrf-core's `td`; the shared interpolator
    // works in Kelvin like every other field. Convert in place — a separate
    // Kelvin copy is another ~400 MB on large grids.
    let mut dewpoint_k = td.data;
    for value in &mut dewpoint_k {
        *value += 273.15;
    }
    // Borrow the two contiguous components directly. Vec::split_off would
    // allocate a seventh native-sized f64 buffer while retaining the original
    // allocation's two-component capacity. Borrowing keeps raw wrfout at its
    // actual six components and preserves the conservative cap's headroom.
    let (u_wind, v_wind) = wind_data.split_at(expected_3d);
    let (volumes, surface) = try_interpolate_iso_volumes(
        &pressure.data,
        &temp.data,
        &dewpoint_k,
        &height.data,
        u_wind,
        v_wind,
        nz,
        cells,
        progress,
    )?;
    Ok((volumes, recipe_volumes, surface))
}

/// Interpolate one native `[nz, ny, nx]` field onto the given isobaric
/// levels -- the same per-column `bracket` + `lerp` walk the five-field
/// interpolator runs, applied to a single field.  Cells outside a column's
/// model pressure range stay NaN, exactly like the sounding volumes.
fn interpolate_field_at_levels(
    pressure_hpa: &[f64],
    data: &[f64],
    nz: usize,
    cells: usize,
    levels: &[u16],
) -> Result<Vec<(u16, Vec<f32>)>, String> {
    let expected = checked_dimension_product("chart-level field", &[nz, cells])?;
    if pressure_hpa.len() != expected || data.len() != expected {
        return Err(format!(
            "chart-level interpolation inputs must be [nz={nz}, cells={cells}]; got pressure {} and field {}",
            pressure_hpa.len(),
            data.len()
        ));
    }
    let mut planes = try_init_planes("chart_levels", levels.len(), cells)?;
    let mut col_p = vec![0.0f64; nz];
    for c in 0..cells {
        for k in 0..nz {
            col_p[k] = pressure_hpa[k * cells + c];
        }
        for (li, &lev) in levels.iter().enumerate() {
            let Some((k, t)) = bracket(&col_p, f64::from(lev)) else {
                continue;
            };
            let (i0, i1) = (k * cells + c, (k + 1) * cells + c);
            if let Some(value) = lerp(data[i0], data[i1], t) {
                planes[li][c] = value as f32;
            }
        }
    }
    Ok(pack(levels, planes))
}

/// Interpolate pre-read WRF column fields onto the canonical isobaric levels
/// and derive the lowest-level surface fallback. All inputs are row-major
/// `[nz, ny, nx]` (index `k * cells + c`) in skew-T units: pressure hPa,
/// temperature K, dewpoint K, height m, winds m/s. Shared by the raw-wrfout
/// ([`build_iso_volumes`]) and post-processed (`TK`/`Z`/`P`) reader paths.
///
/// File readers must call [`preflight_iso_volume_shape`] with trustworthy
/// metadata before reading their 3-D inputs. This function repeats that guard,
/// validates all input lengths, and uses fallible reservations for the large
/// output, surface, and scratch buffers.
pub(crate) fn try_interpolate_iso_volumes(
    pressure_hpa: &[f64],
    temp_k: &[f64],
    dewpoint_k: &[f64],
    height_m: &[f64],
    u_ms: &[f64],
    v_ms: &[f64],
    nz: usize,
    cells: usize,
    progress: &mut dyn FnMut(String),
) -> Result<(Vec<IsoVolume>, SurfaceFallback), String> {
    preflight_iso_volume_shape(nz, cells)?;
    validate_interpolation_inputs(
        pressure_hpa,
        temp_k,
        dewpoint_k,
        height_m,
        u_ms,
        v_ms,
        nz,
        cells,
    )?;

    let levels = standard_levels();
    debug_assert_eq!(levels.len(), STANDARD_LEVEL_COUNT);
    let planes = IsoPlanes::try_new(levels.len(), cells)?;
    let surface = try_surface_fallback(pressure_hpa, temp_k, dewpoint_k, u_ms, v_ms, cells)?;
    let mut column_pressure = Vec::new();
    column_pressure
        .try_reserve_exact(nz)
        .map_err(|err| format!("reserve {nz}-level WRF pressure column: {err}"))?;
    column_pressure.resize(nz, 0.0);
    Ok(interpolate_iso_volumes_with_allocations(
        pressure_hpa,
        temp_k,
        dewpoint_k,
        height_m,
        u_ms,
        v_ms,
        nz,
        cells,
        &levels,
        planes,
        surface,
        column_pressure,
        progress,
    ))
}

struct IsoPlanes {
    temperature: Vec<Vec<f32>>,
    dewpoint: Vec<Vec<f32>>,
    u_wind: Vec<Vec<f32>>,
    v_wind: Vec<Vec<f32>>,
    height: Vec<Vec<f32>>,
}

impl IsoPlanes {
    fn try_new(levels: usize, cells: usize) -> Result<Self, String> {
        Ok(Self {
            temperature: try_init_planes("temperature_iso", levels, cells)?,
            dewpoint: try_init_planes("dewpoint_iso", levels, cells)?,
            u_wind: try_init_planes("u_iso", levels, cells)?,
            v_wind: try_init_planes("v_iso", levels, cells)?,
            height: try_init_planes("height_iso", levels, cells)?,
        })
    }
}

fn interpolate_iso_volumes_with_allocations(
    pressure_hpa: &[f64],
    temp_k: &[f64],
    dewpoint_k: &[f64],
    height_m: &[f64],
    u_ms: &[f64],
    v_ms: &[f64],
    nz: usize,
    cells: usize,
    levels: &[u16],
    mut planes: IsoPlanes,
    surface: SurfaceFallback,
    mut col_p: Vec<f64>,
    progress: &mut dyn FnMut(String),
) -> (Vec<IsoVolume>, SurfaceFallback) {
    let progress_step = (cells / 10).max(1);
    for c in 0..cells {
        if c % progress_step == 0 {
            progress(format!(
                "interpolating 5 sounding fields to {} isobaric levels — {}%",
                levels.len(),
                c * 100 / cells
            ));
        }
        for k in 0..nz {
            col_p[k] = pressure_hpa[k * cells + c];
        }
        for (li, &lev) in levels.iter().enumerate() {
            let Some((k, t)) = bracket(&col_p, f64::from(lev)) else {
                continue;
            };
            let (i0, i1) = (k * cells + c, (k + 1) * cells + c);
            if let Some(value) = lerp(temp_k[i0], temp_k[i1], t) {
                planes.temperature[li][c] = value as f32;
            }
            if let Some(value) = lerp(dewpoint_k[i0], dewpoint_k[i1], t) {
                planes.dewpoint[li][c] = value as f32;
            }
            if let Some(value) = lerp(u_ms[i0], u_ms[i1], t) {
                planes.u_wind[li][c] = value as f32;
            }
            if let Some(value) = lerp(v_ms[i0], v_ms[i1], t) {
                planes.v_wind[li][c] = value as f32;
            }
            if let Some(value) = lerp(height_m[i0], height_m[i1], t) {
                planes.height[li][c] = value as f32;
            }
        }
    }

    let volumes = vec![
        IsoVolume {
            name: "temperature_iso".to_string(),
            units: "K".to_string(),
            levels: pack(levels, planes.temperature),
        },
        IsoVolume {
            name: "dewpoint_iso".to_string(),
            units: "K".to_string(),
            levels: pack(levels, planes.dewpoint),
        },
        IsoVolume {
            name: "u_iso".to_string(),
            units: "m/s".to_string(),
            levels: pack(levels, planes.u_wind),
        },
        IsoVolume {
            name: "v_iso".to_string(),
            units: "m/s".to_string(),
            levels: pack(levels, planes.v_wind),
        },
        IsoVolume {
            name: "height_iso".to_string(),
            units: "gpm".to_string(),
            levels: pack(levels, planes.height),
        },
    ];
    (volumes, surface)
}

fn check_native_3d_output(
    out: &VarOutput,
    name: &str,
    nz: usize,
    ny: usize,
    nx: usize,
) -> Result<usize, String> {
    let expected_shape = [nz, ny, nx];
    if out.shape.as_slice() != expected_shape.as_slice() {
        return Err(format!(
            "WRF {name} has shape {:?}, expected exact native shape {expected_shape:?}",
            out.shape
        ));
    }
    let expected = checked_dimension_product("WRF native 3-D field", &expected_shape)?;
    if out.data.len() != expected {
        return Err(format!(
            "WRF {name} shape {expected_shape:?} describes {expected} values, but the output contains {}",
            out.data.len()
        ));
    }
    Ok(expected)
}

fn validate_earth_relative_uvmet(
    uvmet: VarOutput,
    nz: usize,
    ny: usize,
    nx: usize,
    expected_component_values: usize,
) -> Result<Vec<f64>, String> {
    let expected_shape = [2, nz, ny, nx];
    if uvmet.shape.as_slice() != expected_shape.as_slice() {
        return Err(format!(
            "WRF uvmet has shape {:?}, expected exact two-component native shape {expected_shape:?}",
            uvmet.shape,
        ));
    }
    let expected_total = expected_component_values.checked_mul(2).ok_or_else(|| {
        "WRF uvmet two-component length overflows the platform address space".to_string()
    })?;
    if uvmet.data.len() != expected_total {
        return Err(format!(
            "WRF uvmet has {} values, expected {expected_total} for two earth-relative components",
            uvmet.data.len()
        ));
    }
    Ok(uvmet.data)
}

/// Multiply dimensions supplied by an untrusted file without relying on
/// release-mode wrapping (or a debug-mode panic). The caller can then report a
/// malformed shape as an ordinary import error.
fn checked_dimension_product(name: &str, dimensions: &[usize]) -> Result<usize, String> {
    dimensions.iter().try_fold(1usize, |product, &dimension| {
        product.checked_mul(dimension).ok_or_else(|| {
            format!("{name} dimensions {dimensions:?} overflow the platform address space")
        })
    })
}

fn validate_interpolation_inputs(
    pressure_hpa: &[f64],
    temp_k: &[f64],
    dewpoint_k: &[f64],
    height_m: &[f64],
    u_ms: &[f64],
    v_ms: &[f64],
    nz: usize,
    cells: usize,
) -> Result<(), String> {
    let expected = checked_dimension_product("WRF native 3-D field", &[nz, cells])?;
    if nz < 2 {
        return Err(format!(
            "WRF native 3-D fields require at least two levels, got {nz}"
        ));
    }
    for (name, values) in [
        ("pressure", pressure_hpa),
        ("temperature", temp_k),
        ("dewpoint", dewpoint_k),
        ("height", height_m),
        ("u wind", u_ms),
        ("v wind", v_ms),
    ] {
        if values.len() != expected {
            return Err(format!(
                "WRF {name} has {} values, expected {expected} for {nz} levels x {cells} cells",
                values.len()
            ));
        }
    }
    Ok(())
}

fn try_surface_fallback(
    pressure_hpa: &[f64],
    temp_k: &[f64],
    dewpoint_k: &[f64],
    u_ms: &[f64],
    v_ms: &[f64],
    cells: usize,
) -> Result<SurfaceFallback, String> {
    Ok(SurfaceFallback {
        surface_pressure_pa: try_surface_plane(
            "approx_surface_pressure",
            pressure_hpa,
            cells,
            |value| (value * 100.0) as f32,
        )?,
        temperature_2m_k: try_surface_plane("approx_temperature_2m", temp_k, cells, |value| {
            value as f32
        })?,
        dewpoint_2m_k: try_surface_plane("approx_dewpoint_2m", dewpoint_k, cells, |value| {
            value as f32
        })?,
        u_10m: try_surface_plane("approx_u_10m", u_ms, cells, |value| value as f32)?,
        v_10m: try_surface_plane("approx_v_10m", v_ms, cells, |value| value as f32)?,
    })
}

fn try_surface_plane(
    name: &str,
    source: &[f64],
    cells: usize,
    convert: impl Fn(f64) -> f32,
) -> Result<Vec<f32>, String> {
    let mut plane = Vec::new();
    plane
        .try_reserve_exact(cells)
        .map_err(|err| format!("reserve {cells}-cell WRF surface plane '{name}': {err}"))?;
    plane.extend(source.iter().take(cells).map(|value| convert(*value)));
    Ok(plane)
}

fn try_init_planes(name: &str, levels: usize, cells: usize) -> Result<Vec<Vec<f32>>, String> {
    let mut planes = Vec::new();
    planes
        .try_reserve_exact(levels)
        .map_err(|err| format!("reserve {levels} WRF pressure planes for '{name}': {err}"))?;
    for level_index in 0..levels {
        let mut plane = Vec::new();
        plane.try_reserve_exact(cells).map_err(|err| {
            format!(
                "reserve {cells}-cell WRF pressure plane {level_index}/{levels} for '{name}': {err}"
            )
        })?;
        plane.resize(cells, f32::NAN);
        planes.push(plane);
    }
    Ok(planes)
}

fn pack(levels: &[u16], planes: Vec<Vec<f32>>) -> Vec<(u16, Vec<f32>)> {
    levels.iter().copied().zip(planes).collect()
}

/// Locate the native levels bracketing `target` hPa in a WRF column (pressure
/// decreasing with index, level 0 nearest the surface) and return the lower
/// level index plus the log-pressure interpolation weight. `None` when the
/// target sits below the lowest level or above the model top.
fn bracket(col_p: &[f64], target: f64) -> Option<(usize, f64)> {
    for k in 0..col_p.len().saturating_sub(1) {
        let (pk, pk1) = (col_p[k], col_p[k + 1]);
        if !pk.is_finite() || !pk1.is_finite() || pk == pk1 {
            continue;
        }
        let (hi, lo) = if pk >= pk1 { (pk, pk1) } else { (pk1, pk) };
        if target <= hi && target >= lo {
            let t = (target.ln() - pk.ln()) / (pk1.ln() - pk.ln());
            return Some((k, t));
        }
    }
    None
}

fn lerp(a: f64, b: f64, t: f64) -> Option<f64> {
    (a.is_finite() && b.is_finite()).then_some(a + t * (b - a))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn standard_levels_span_the_isobaric_ladder() {
        let levels = standard_levels();
        assert_eq!(levels.len(), 37);
        assert_eq!(*levels.first().unwrap(), 100);
        assert_eq!(*levels.last().unwrap(), 1000);
    }

    #[test]
    fn volume_preflight_checks_store_and_owned_working_set_ceilings_without_allocating() {
        let largest_supported_grid = rustwx_core::MAX_VOLUME_ELEMENTS / STANDARD_LEVEL_COUNT;
        assert!(
            u128::from(preflight_iso_volume_shape(2, largest_supported_grid).unwrap())
                < MAX_WRF_VOLUME_OWNED_BYTES
        );

        let error = preflight_iso_volume_shape(2, largest_supported_grid + 1)
            .expect_err("one cell past the 37-level ceiling must be omitted");
        assert!(error.contains("37-level"), "unexpected error: {error}");
        assert!(
            error.contains(&rustwx_core::MAX_VOLUME_ELEMENTS.to_string()),
            "shared ceiling must be visible in the error: {error}"
        );

        assert_eq!(
            preflight_iso_volume_shape(79, 800 * 800).unwrap(),
            3_722_240_000,
            "known 800x800x79 workflow remains below the 4 GiB owned-buffer cap"
        );
        assert_eq!(
            preflight_iso_volume_shape(92, 800 * 800).unwrap(),
            4_254_720_000,
            "largest accepted level count for this grid remains just under 4 GiB"
        );
        let aggregate_error = preflight_iso_volume_shape(93, 800 * 800)
            .expect_err("one level beyond the aggregate boundary must fail before getvar");
        assert!(
            aggregate_error.contains("4 GiB"),
            "unexpected aggregate error: {aggregate_error}"
        );
        assert!(preflight_iso_volume_shape(usize::MAX, 1).is_err());
        assert!(preflight_iso_volume_shape(1, 1).is_err());
        assert!(preflight_iso_volume_shape(2, 0).is_err());
        assert!(checked_byte_product("test", &[u128::MAX, 2]).is_err());
    }

    #[test]
    fn bracket_interpolates_in_log_pressure_and_clamps_to_range() {
        // Decreasing pressure with index (level 0 nearest the surface).
        let col = [1000.0, 850.0, 700.0, 500.0];
        // Midway between 1000 and 850 in ln-p.
        let (k, t) = bracket(&col, 925.0).expect("in range");
        assert_eq!(k, 0);
        let expected = (925f64.ln() - 1000f64.ln()) / (850f64.ln() - 1000f64.ln());
        assert!((t - expected).abs() < 1e-9);
        // Below the lowest level and above the top are both out of range.
        assert!(bracket(&col, 1013.0).is_none());
        assert!(bracket(&col, 300.0).is_none());
    }

    #[test]
    fn lerp_skips_non_finite_endpoints() {
        assert_eq!(lerp(0.0, 10.0, 0.5), Some(5.0));
        assert_eq!(lerp(f64::NAN, 10.0, 0.5), None);
        assert_eq!(lerp(0.0, f64::NAN, 0.5), None);
    }

    #[test]
    fn malformed_dimension_products_return_errors_instead_of_overflowing() {
        let error = checked_dimension_product("test field", &[2, usize::MAX, 2])
            .expect_err("oversized file dimensions must fail closed");
        assert!(error.contains("overflow"));
        assert_eq!(checked_dimension_product("test field", &[2, 3, 4]), Ok(24));
    }

    #[test]
    fn checked_interpolator_rejects_bad_native_shape_before_output_allocation() {
        let one_value = [1.0];
        let error = try_interpolate_iso_volumes(
            &one_value,
            &one_value,
            &one_value,
            &one_value,
            &one_value,
            &one_value,
            2,
            1,
            &mut |_| {},
        )
        .err()
        .expect("two levels x one cell requires two values per input");
        assert!(error.contains("expected 2"), "unexpected error: {error}");
    }

    #[test]
    fn native_outputs_require_exact_declared_wrf_shapes() {
        let valid = VarOutput {
            data: vec![1.0, 2.0],
            shape: vec![2, 1, 1],
            units: "K".to_string(),
            description: "valid native field".to_string(),
        };
        assert_eq!(check_native_3d_output(&valid, "temp", 2, 1, 1), Ok(2));

        let transposed = VarOutput {
            data: vec![1.0, 2.0],
            shape: vec![1, 1, 2],
            units: "K".to_string(),
            description: "wrongly shaped field".to_string(),
        };
        let error = check_native_3d_output(&transposed, "temp", 2, 1, 1)
            .expect_err("same-length transposed shape must fail closed");
        assert!(
            error.contains("exact native shape"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn uvmet_validation_rejects_grid_relative_or_malformed_fallback_shapes() {
        let uvmet = VarOutput {
            data: vec![1.0, 2.0, 3.0, 4.0],
            shape: vec![2, 2, 1, 1],
            units: "m/s".to_string(),
            description: "earth-relative wind".to_string(),
        };
        let winds =
            validate_earth_relative_uvmet(uvmet, 2, 1, 1, 2).expect("valid uvmet components");
        let (u, v) = winds.split_at(2);
        assert_eq!(u, &[1.0, 2.0]);
        assert_eq!(v, &[3.0, 4.0]);

        let one_component = VarOutput {
            data: vec![1.0, 2.0],
            shape: vec![1, 2, 1, 1],
            units: "m/s".to_string(),
            description: "grid-relative wind".to_string(),
        };
        assert!(
            validate_earth_relative_uvmet(one_component, 2, 1, 1, 2)
                .expect_err("one grid-relative component must not be accepted")
                .contains("two-component native shape")
        );
    }

    /// The shared interpolator must stream progress (both import paths surface
    /// it) and still produce correct planes — guard for the progress plumbing.
    #[test]
    fn interpolate_streams_progress_and_interpolates() {
        // 2 columns × 3 levels, pressure decreasing with index.
        let pressure = vec![1000.0, 1000.0, 850.0, 850.0, 700.0, 700.0];
        let temp = vec![300.0, 301.0, 290.0, 291.0, 280.0, 281.0];
        let dewp = vec![295.0, 296.0, 285.0, 286.0, 275.0, 276.0];
        let height = vec![100.0, 110.0, 1500.0, 1510.0, 3000.0, 3010.0];
        let u = vec![1.0; 6];
        let v = vec![2.0; 6];

        let mut messages = Vec::new();
        let (volumes, surface) = try_interpolate_iso_volumes(
            &pressure,
            &temp,
            &dewp,
            &height,
            &u,
            &v,
            3,
            2,
            &mut |message| messages.push(message),
        )
        .expect("small valid volume");

        assert!(
            messages
                .iter()
                .all(|message| message.contains("isobaric levels")),
            "unexpected progress lines: {messages:?}"
        );
        assert!(!messages.is_empty(), "interpolation must report progress");

        // 850 hPa is an exact native level: temperature lands unchanged.
        let temps = &volumes[0];
        assert_eq!(temps.name, "temperature_iso");
        let (_, plane_850) = temps
            .levels
            .iter()
            .find(|(hpa, _)| *hpa == 850)
            .expect("850 hPa plane");
        assert!((plane_850[0] - 290.0).abs() < 1e-3);
        assert!((plane_850[1] - 291.0).abs() < 1e-3);
        // Surface fallback comes from level 0 in Pa/K.
        assert!((surface.surface_pressure_pa[0] - 100_000.0).abs() < 1e-3);
        assert!((surface.temperature_2m_k[1] - 301.0).abs() < 1e-3);
    }
}
