//! CF regular lat-lon pairs through the production panel and sheet seams.
//! Source variable names and conversions are data rows. No model dispatch.

use std::path::{Path, PathBuf};
use std::time::Instant;

use chrono::{DateTime, Utc};
use rayon::prelude::*;
use rustwx_core::{CanonicalField, FieldSelector, GridProjection, ModelId};
use rustwx_render::{ColorScale, ExtendMode, LegendControls, RenderDensity, WeatherPalette};
use serde::{Deserialize, Serialize};

use crate::compare::{SheetHeader, compose_sheet, difference, difference_scale, ladder_scale};
use crate::panel::{PanelRequest, layout_path, render_panel};

pub const USAGE: &str = "rw_compare --mode latlon-pair --file A.nc --file B.nc \\\n  --label A=LABEL --label B=LABEL --field z500 --field t850 --field t2m \\\n  --field msl --field u10m_v10m_speed --field tcwv \\\n  [--region global|lat0,lat1,lon0,lon1] --valid YYYY-MM-DDTHH:MMZ --out-dir DIR \\\n  [--threads 12] [--width 1200 --height 760] [--variable-table TABLE.json]\n\
--print-variable-table prints the built-in JSON table for editing.\n\
Variables must be 2-D (lat,lon) or (lon,lat), on regular 1-D axes.\n\
Only coincident coordinates are compared, with cosine-latitude weights.\n\
A region may cross the date line (e.g. -30,30,150,-150).";

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PlaneSpec {
    /// One scalar or two vector components. Components are combined before conversion.
    pub variables: Vec<String>,
    pub factor: f64,
    pub offset: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum StyleSpec {
    Production {
        selector: FieldSelector,
        stored_units: String,
        /// Transform the production legend into the row's display units.
        factor: f64,
        offset: f64,
    },
    HeightLadder {
        first: f64,
        step: f64,
        bins: usize,
    },
    Neutral {
        min: f32,
        max: f32,
    },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VariableRow {
    pub field: String,
    pub a: PlaneSpec,
    pub b: PlaneSpec,
    pub product_slug: String,
    pub title: String,
    pub display_units: String,
    pub style: StyleSpec,
    /// Smallest useful difference step, in display units. Larger spreads adapt upward.
    pub difference_step: f64,
}

pub fn variable_table() -> Vec<VariableRow> {
    let scalar = |prefix: &str, field: &str, factor, offset| PlaneSpec {
        variables: vec![format!("{prefix}_{field}")],
        factor,
        offset,
    };
    let production = |selector, stored_units: &str, factor, offset| StyleSpec::Production {
        selector,
        stored_units: stored_units.into(),
        factor,
        offset,
    };
    let rows = [
        (
            "z500",
            "500mb_height",
            "500 hPa geopotential height",
            "dam",
            1.0 / 98.0665,
            0.0,
            StyleSpec::HeightLadder {
                first: 480.0,
                step: 6.0,
                bins: 21,
            },
            0.1,
        ),
        (
            "t850",
            "850mb_temperature",
            "850 hPa temperature",
            "degC",
            1.0,
            -273.15,
            production(
                FieldSelector::isobaric(CanonicalField::Temperature, 850),
                "K",
                1.0,
                0.0,
            ),
            0.1,
        ),
        (
            "t2m",
            "2m_temperature",
            "2 m temperature",
            "degC",
            1.0,
            -273.15,
            production(
                FieldSelector::height_agl(CanonicalField::Temperature, 2),
                "K",
                5.0 / 9.0,
                -160.0 / 9.0,
            ),
            0.1,
        ),
        (
            "msl",
            "mean_sea_level_pressure",
            "Mean sea level pressure",
            "hPa",
            0.01,
            0.0,
            StyleSpec::Neutral {
                min: 940.0,
                max: 1040.0,
            },
            0.1,
        ),
        (
            "u10m_v10m_speed",
            "10m_wind_speed",
            "10 m wind speed",
            "m/s",
            1.0,
            0.0,
            production(
                FieldSelector::height_agl(CanonicalField::WindSpeed, 10),
                "m/s",
                1.0 / 1.943_844_492_440_6,
                0.0,
            ),
            0.1,
        ),
        (
            "tcwv",
            "total_column_water_vapour",
            "Total column water vapour",
            "mm",
            1.0,
            0.0,
            production(
                FieldSelector::entire_atmosphere(CanonicalField::PrecipitableWater),
                "mm",
                25.4,
                0.0,
            ),
            0.1,
        ),
    ];
    rows.into_iter()
        .map(|(field, slug, title, units, factor, offset, style, step)| {
            // Default rows read the Earth-2 vocabulary name itself in both files
            // (z500, t2m, msl, u10m ...); a file that prefixes or renames its
            // variables supplies --variable-table, never a code path.
            let mut a = scalar("a", field, factor, offset);
            let mut b = scalar("b", field, factor, offset);
            if field == "u10m_v10m_speed" {
                a.variables = vec!["u10m".into(), "v10m".into()];
                b.variables = vec!["u10m".into(), "v10m".into()];
            } else {
                a.variables = vec![field.into()];
                b.variables = vec![field.into()];
            }
            VariableRow {
                field: field.into(),
                a,
                b,
                product_slug: slug.into(),
                title: title.into(),
                display_units: units.into(),
                style,
                difference_step: step,
            }
        })
        .collect()
}

fn find_row<'a>(rows: &'a [VariableRow], field: &str) -> Result<&'a VariableRow, String> {
    rows.iter().find(|r| r.field == field).ok_or_else(|| {
        format!(
            "unknown --field {field:?}; variable table knows: {}",
            rows.iter()
                .map(|r| r.field.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        )
    })
}

fn validate_table(rows: &[VariableRow]) -> Result<(), String> {
    let mut fields = std::collections::HashSet::new();
    let mut slugs = std::collections::HashSet::new();
    for row in rows {
        if row.field.is_empty()
            || !fields.insert(&row.field)
            || !slugs.insert(&row.product_slug)
            || crate::panel::safe_component(&row.product_slug, "product") != row.product_slug
        {
            return Err(format!(
                "invalid or duplicate field/product in table: {:?}",
                row.field
            ));
        }
        for plane in [&row.a, &row.b] {
            if !(1..=2).contains(&plane.variables.len())
                || plane.variables.iter().any(|v| v.is_empty())
                || !plane.factor.is_finite()
                || plane.factor == 0.0
                || !plane.offset.is_finite()
            {
                return Err(format!(
                    "{}: table requires one scalar or two wind components and a finite conversion",
                    row.field
                ));
            }
        }
        if !row.difference_step.is_finite() || row.difference_step <= 0.0 {
            return Err(format!("{}: difference_step must be positive", row.field));
        }
    }
    Ok(())
}

#[derive(Clone, Debug)]
struct Axis {
    values: Vec<f64>,
    indices: Vec<usize>,
    original_len: usize,
}

fn wrap_lon(lon: f64) -> f64 {
    (lon + 180.0).rem_euclid(360.0) - 180.0
}

/// Canonical south-to-north / west-to-east axes, retaining the source permutation.
fn axis(values: &[f64], longitude: bool) -> Result<Axis, String> {
    let name = if longitude { "lon" } else { "lat" };
    if values.len() < 2
        || values.iter().any(|x| {
            !x.is_finite()
                || (!longitude && x.abs() > 90.0)
                || (longitude && (*x < -180.0 || *x > 360.0))
        })
    {
        return Err(format!(
            "{name}: expected at least two finite geographic coordinates"
        ));
    }
    // CF axes are monotonic before longitude normalization. Reject scrambled inputs.
    if !values.windows(2).all(|v| v[1] > v[0]) && !values.windows(2).all(|v| v[1] < v[0]) {
        return Err(format!("{name}: coordinate axis is not strictly monotonic"));
    }
    let spacing = (values[1] - values[0]).abs();
    if values
        .windows(2)
        .any(|v| ((v[1] - v[0]).abs() - spacing).abs() > 1e-4 * spacing.max(1.0))
    {
        return Err(format!("{name}: coordinate axis is not regular"));
    }
    let mut pairs: Vec<_> = values
        .iter()
        .enumerate()
        .map(|(i, &x)| (if longitude { wrap_lon(x) } else { x }, i))
        .collect();
    pairs.sort_by(|a, b| a.0.total_cmp(&b.0));
    // A duplicated cyclic endpoint is the same physical meridian. Keep it once.
    pairs.dedup_by(|a, b| (a.0 - b.0).abs() < 1e-7);
    Ok(Axis {
        values: pairs.iter().map(|p| p.0).collect(),
        indices: pairs.iter().map(|p| p.1).collect(),
        original_len: values.len(),
    })
}

struct Grid {
    lat: Axis,
    lon: Axis,
    lat_dim: String,
    lon_dim: String,
}

fn read_grid(nc: &netcrust::File) -> Result<Grid, String> {
    let read_axis = |name, longitude| -> Result<(Axis, String), String> {
        let var = nc
            .variable(name)
            .ok_or_else(|| format!("missing coordinate variable {name:?}"))?;
        if var.ndim() != 1 {
            return Err(format!(
                "{name}: coordinate must be 1-D, got {:?}",
                var.shape()
            ));
        }
        let data = var.array_f64().map_err(|e| format!("{name}: {e}"))?;
        Ok((
            axis(data.values(), longitude)?,
            var.dimensions()[0].name().to_string(),
        ))
    };
    let (lat, lat_dim) = read_axis("lat", false)?;
    let (lon, lon_dim) = read_axis("lon", true)?;
    if lat_dim == lon_dim {
        return Err("lat and lon must use distinct dimensions".into());
    }
    Ok(Grid {
        lat,
        lon,
        lat_dim,
        lon_dim,
    })
}

/// Decode CF fill and packing in the stored domain, before table conversion.
fn read_variable(nc: &netcrust::File, name: &str, grid: &Grid) -> Result<Vec<f64>, String> {
    let var = nc
        .variable(name)
        .ok_or_else(|| format!("missing variable {name:?} named by variable table"))?;
    let dims: Vec<_> = var.dimensions().iter().map(|d| d.name()).collect();
    let transposed = dims == [grid.lon_dim.as_str(), grid.lat_dim.as_str()];
    if dims != [grid.lat_dim.as_str(), grid.lon_dim.as_str()] && !transposed {
        return Err(format!(
            "{name}: expected (lat,lon) or (lon,lat) dimensions, got {dims:?}"
        ));
    }
    let data = var.array_f64().map_err(|e| format!("{name}: {e}"))?;
    let fill: Vec<f64> = ["_FillValue", "missing_value"]
        .iter()
        .filter_map(|n| var.attribute(n))
        .filter_map(|a| a.value().as_f64_vec())
        .flatten()
        .collect();
    let factor = var
        .attribute("scale_factor")
        .and_then(|a| a.as_f64())
        .unwrap_or(1.0);
    let offset = var
        .attribute("add_offset")
        .and_then(|a| a.as_f64())
        .unwrap_or(0.0);
    Ok(grid
        .lat
        .indices
        .par_iter()
        .flat_map_iter(|&j| {
            grid.lon.indices.iter().map({
                let data = data.values();
                let fill = &fill;
                move |&i| {
                    let x = data[if transposed {
                        i * grid.lat.original_len + j
                    } else {
                        j * grid.lon.original_len + i
                    }];
                    if !x.is_finite() || fill.contains(&x) {
                        f64::NAN
                    } else {
                        x * factor + offset
                    }
                }
            })
        })
        .collect())
}

fn convert_plane(spec: &PlaneSpec, components: &[Vec<f64>]) -> Vec<f32> {
    (0..components[0].len())
        .into_par_iter()
        .map(|i| {
            let a = components[0][i];
            let value = if components.len() == 2 {
                let b = components[1][i];
                if a.is_finite() && b.is_finite() {
                    a.hypot(b)
                } else {
                    f64::NAN
                }
            } else {
                a
            };
            (value * spec.factor + spec.offset) as f32
        })
        .collect()
}

fn read_plane(nc: &netcrust::File, grid: &Grid, spec: &PlaneSpec) -> Result<Vec<f32>, String> {
    let components = spec
        .variables
        .iter()
        .map(|v| read_variable(nc, v, grid))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(convert_plane(spec, &components))
}

#[derive(Clone, Copy, Debug)]
struct Region {
    south: f64,
    north: f64,
    west: f64,
    span: f64,
}

fn parse_region(text: &str) -> Result<Option<Region>, String> {
    if text == "global" {
        return Ok(None);
    }
    let v = text
        .split(',')
        .map(str::parse::<f64>)
        .collect::<Result<Vec<_>, _>>()
        .map_err(|_| format!("invalid --region {text:?}"))?;
    if v.len() != 4
        || v.iter().any(|x| !x.is_finite())
        || v[0] < -90.0
        || v[1] > 90.0
        || v[0] >= v[1]
        || v[2] < -180.0
        || v[2] > 360.0
        || v[3] < -180.0
        || v[3] > 360.0
    {
        return Err(format!(
            "invalid --region {text:?}: use south,north,west,east"
        ));
    }
    let raw = v[3] - v[2];
    let span = if raw.abs() >= 360.0 {
        360.0
    } else {
        raw.rem_euclid(360.0)
    };
    if span == 0.0 {
        return Err("region has zero longitude width".into());
    }
    Ok(Some(Region {
        south: v[0],
        north: v[1],
        west: wrap_lon(v[2]),
        span,
    }))
}

fn common_axis(a: &[f64], b: &[f64]) -> Vec<(usize, usize)> {
    let (mut i, mut j) = (0, 0);
    let mut out = Vec::new();
    while i < a.len() && j < b.len() {
        if (a[i] - b[j]).abs() <= 1e-5 {
            out.push((i, j));
            i += 1;
            j += 1;
        } else if a[i] < b[j] {
            i += 1;
        } else {
            j += 1;
        }
    }
    out
}

struct CommonGrid {
    ny: usize,
    nx: usize,
    lat: Vec<f32>,
    lon: Vec<f32>,
    a_indices: Vec<usize>,
    b_indices: Vec<usize>,
}

fn common_grid(a: &Grid, b: &Grid, region: Option<Region>) -> Result<CommonGrid, String> {
    let lat: Vec<_> = common_axis(&a.lat.values, &b.lat.values)
        .into_iter()
        .filter(|&(j, _)| {
            region.is_none_or(|r| a.lat.values[j] >= r.south && a.lat.values[j] <= r.north)
        })
        .collect();
    let mut lon: Vec<_> = common_axis(&a.lon.values, &b.lon.values)
        .into_iter()
        .filter(|&(i, _)| {
            region.is_none_or(|r| (a.lon.values[i] - r.west).rem_euclid(360.0) <= r.span + 1e-5)
        })
        .collect();
    if let Some(r) = region {
        lon.sort_by(|&(i, _), &(k, _)| {
            (a.lon.values[i] - r.west)
                .rem_euclid(360.0)
                .total_cmp(&(a.lon.values[k] - r.west).rem_euclid(360.0))
        });
    }
    if lat.len() < 2 || lon.len() < 2 {
        return Err("fewer than two coincident lat/lon coordinates in region; grids are matched by coordinate, never by array position".into());
    }
    let mut out = CommonGrid {
        ny: lat.len(),
        nx: lon.len(),
        lat: Vec::new(),
        lon: Vec::new(),
        a_indices: Vec::new(),
        b_indices: Vec::new(),
    };
    for (aj, bj) in lat {
        for &(ai, bi) in &lon {
            out.lat.push(a.lat.values[aj] as f32);
            let longitude = if let Some(r) = region {
                r.west + (a.lon.values[ai] - r.west).rem_euclid(360.0)
            } else {
                a.lon.values[ai]
            };
            out.lon.push(longitude as f32);
            out.a_indices.push(aj * a.lon.values.len() + ai);
            out.b_indices.push(bj * b.lon.values.len() + bi);
        }
    }
    Ok(out)
}

#[derive(Clone, Debug, Serialize)]
pub struct PairStats {
    pub common_finite_cells: usize,
    pub weight_sum: f64,
    pub bias: f64,
    pub rmse: f64,
    /// None for a constant field, where spatial correlation is undefined.
    pub pattern_correlation: Option<f64>,
    pub a_min: f32,
    pub a_max: f32,
    pub b_min: f32,
    pub b_max: f32,
    pub difference_min: f64,
    pub difference_max: f64,
}

pub fn pair_stats(a: &[f32], b: &[f32], lat: &[f32]) -> Result<PairStats, String> {
    if a.len() != b.len() || a.len() != lat.len() {
        return Err("statistics shape mismatch".into());
    }
    // Fixed chunks preserve reduction order across thread counts, with f64 accumulation.
    let chunks: Vec<_> = a
        .par_chunks(4096)
        .zip(b.par_chunks(4096))
        .zip(lat.par_chunks(4096))
        .map(|((a, b), lat)| {
            let mut sums = [0.0f64; 5];
            let mut count = 0;
            for ((&x, &y), &phi) in a.iter().zip(b).zip(lat) {
                if !x.is_finite() || !y.is_finite() || !phi.is_finite() {
                    continue;
                }
                let (x, y) = (f64::from(x), f64::from(y));
                let w = f64::from(phi).to_radians().cos().max(0.0);
                let d = x - y;
                for (sum, value) in sums.iter_mut().zip([w, w * x, w * y, w * d, w * d * d]) {
                    *sum += value;
                }
                count += 1;
            }
            (sums, count)
        })
        .collect();
    let mut sums = [0.0; 5];
    let mut count = 0;
    for (s, n) in chunks {
        for (sum, value) in sums.iter_mut().zip(s) {
            *sum += value;
        }
        count += n;
    }
    if sums[0] <= 1e-12 {
        return Err("no common finite cells with positive area weight".into());
    }
    let (ma, mb) = (sums[1] / sums[0], sums[2] / sums[0]);
    let chunks: Vec<_> = a
        .par_chunks(4096)
        .zip(b.par_chunks(4096))
        .zip(lat.par_chunks(4096))
        .map(|((a, b), lat)| {
            let mut cov = [0.0; 3];
            for ((&x, &y), &phi) in a.iter().zip(b).zip(lat) {
                if !x.is_finite() || !y.is_finite() || !phi.is_finite() {
                    continue;
                }
                let (x, y) = (f64::from(x) - ma, f64::from(y) - mb);
                let w = f64::from(phi).to_radians().cos().max(0.0);
                for (sum, value) in cov.iter_mut().zip([w * x * x, w * y * y, w * x * y]) {
                    *sum += value;
                }
            }
            cov
        })
        .collect();
    let mut cov = [0.0; 3];
    for s in chunks {
        for (sum, v) in cov.iter_mut().zip(s) {
            *sum += v;
        }
    }
    let mut stats = PairStats {
        common_finite_cells: count,
        weight_sum: sums[0],
        bias: sums[3] / sums[0],
        rmse: (sums[4] / sums[0]).sqrt(),
        pattern_correlation: (cov[0] > 0.0 && cov[1] > 0.0)
            .then(|| (cov[2] / (cov[0] * cov[1]).sqrt()).clamp(-1.0, 1.0)),
        a_min: f32::INFINITY,
        a_max: f32::NEG_INFINITY,
        b_min: f32::INFINITY,
        b_max: f32::NEG_INFINITY,
        difference_min: f64::INFINITY,
        difference_max: f64::NEG_INFINITY,
    };
    for (&x, &y) in a.iter().zip(b) {
        if x.is_finite() && y.is_finite() {
            stats.a_min = stats.a_min.min(x);
            stats.a_max = stats.a_max.max(x);
            stats.b_min = stats.b_min.min(y);
            stats.b_max = stats.b_max.max(y);
            stats.difference_min = stats.difference_min.min(f64::from(x) - f64::from(y));
            stats.difference_max = stats.difference_max.max(f64::from(x) - f64::from(y));
        }
    }
    Ok(stats)
}

struct Style {
    scale: ColorScale,
    tick: Option<f64>,
    legend: LegendControls,
    density: RenderDensity,
}
fn style(row: &VariableRow) -> Result<Style, String> {
    match &row.style {
        StyleSpec::Production {
            selector,
            stored_units,
            factor,
            offset,
        } => {
            if !factor.is_finite() || *factor <= 0.0 || !offset.is_finite() {
                return Err(format!("{}: invalid legend conversion", row.field));
            }
            let s = rustwx_products::viewer::operational_style_for_store_variable(
                &row.product_slug,
                &serde_json::to_value(selector).map_err(|e| e.to_string())?,
                stored_units,
                ModelId::WrfGdex,
            )
            .ok_or_else(|| format!("{}: no production style for selector", row.field))?;
            let mut scale = s.scale.resolved_discrete();
            for level in &mut scale.levels {
                *level = *level * factor + offset;
            }
            // Paired continuous fields show calm/dry cells too, with endpoint extension.
            scale.mask_below = None;
            scale.extend = ExtendMode::Both;
            Ok(Style {
                scale: ColorScale::Discrete(scale),
                tick: s.cbar_tick_step.map(|v| v * factor),
                legend: s.colormap_options.legend,
                density: s.colormap_options.render_density,
            })
        }
        StyleSpec::HeightLadder { first, step, bins } => {
            if !first.is_finite() || !step.is_finite() || *step <= 0.0 || *bins < 2 || *bins > 4096
            {
                return Err("invalid height ladder".into());
            }
            Ok(Style {
                scale: ladder_scale(
                    *first,
                    *step,
                    *bins,
                    &rustwx_render::weather::weather_palette(WeatherPalette::IsothermHeight),
                ),
                tick: Some(step * 2.0),
                legend: LegendControls::default(),
                density: RenderDensity::default(),
            })
        }
        StyleSpec::Neutral { min, max } => {
            if !min.is_finite() || !max.is_finite() || min >= max {
                return Err("invalid neutral ladder range".into());
            }
            let s = rustwx_products::viewer::generic_style_for_prescaled_store_variable(
                &row.product_slug,
                &row.display_units,
                Some((*min, *max)),
            );
            Ok(Style {
                scale: s.scale,
                tick: s.cbar_tick_step,
                legend: s.colormap_options.legend,
                density: s.colormap_options.render_density,
            })
        }
    }
}

/// Cover typical errors while retaining a neutral bin and explicit open ends.
fn adaptive_difference_step(stats: &PairStats, minimum: f64) -> f64 {
    let target = (2.5 * stats.rmse / 5.0).max(minimum);
    let decade = 10f64.powf(target.log10().floor());
    [1.0, 2.0, 2.5, 5.0, 10.0]
        .into_iter()
        .map(|n| n * decade)
        .find(|&n| n >= target)
        .unwrap_or(target)
}

struct Options {
    files: Vec<PathBuf>,
    labels: [String; 2],
    fields: Vec<String>,
    out: PathBuf,
    valid: DateTime<Utc>,
    region: Option<Region>,
    region_label: String,
    threads: usize,
    width: u32,
    height: u32,
    rows: Vec<VariableRow>,
}

fn options(args: &[String]) -> Result<Options, String> {
    let (mut files, mut fields) = (Vec::new(), Vec::new());
    let mut labels = ["A".to_string(), "B".to_string()];
    let (mut out, mut valid, mut table) = (None, None, None);
    let (mut threads, mut width, mut height) = (12usize, 1200u32, 760u32);
    let mut region_label = "global".to_string();
    let mut i = 0;
    while i < args.len() {
        let key = &args[i];
        i += 1;
        let value = args.get(i).ok_or_else(|| format!("{key}: missing value"))?;
        i += 1;
        match key.as_str() {
            "--mode" if value == "latlon-pair" => {}
            "--file" => files.push(PathBuf::from(value)),
            "--field" => fields.push(value.clone()),
            "--label" => {
                let (side, label) = value
                    .split_once('=')
                    .ok_or("--label requires A=LABEL or B=LABEL")?;
                let index = match side {
                    "A" => 0,
                    "B" => 1,
                    _ => return Err(format!("unknown label side {side:?}")),
                };
                if label.trim().is_empty() {
                    return Err("empty label".into());
                }
                labels[index] = label.to_string();
            }
            "--out-dir" => out = Some(PathBuf::from(value)),
            "--valid" => valid = Some(crate::nowcast_frames::parse_utc(value)?),
            "--region" => region_label = value.clone(),
            "--threads" => threads = value.parse().map_err(|_| "invalid --threads")?,
            "--width" => width = value.parse().map_err(|_| "invalid --width")?,
            "--height" => height = value.parse().map_err(|_| "invalid --height")?,
            "--variable-table" => table = Some(PathBuf::from(value)),
            _ => return Err(format!("unknown latlon-pair option {key:?}")),
        }
    }
    if files.len() != 2 {
        return Err("latlon-pair requires exactly two --file arguments, A then B".into());
    }
    if fields.is_empty() {
        return Err("latlon-pair requires at least one --field".into());
    }
    if threads == 0 || threads > 256 || width < 640 || height < 400 {
        return Err("use 1..256 threads and panels at least 640x400".into());
    }
    let rows = if let Some(path) = table {
        serde_json::from_slice(
            &std::fs::read(&path).map_err(|e| format!("{}: {e}", path.display()))?,
        )
        .map_err(|e| format!("{}: {e}", path.display()))?
    } else {
        variable_table()
    };
    validate_table(&rows)?;
    let mut seen = std::collections::HashSet::new();
    for field in &fields {
        find_row(&rows, field)?;
        if !seen.insert(field) {
            return Err(format!("duplicate --field {field:?}"));
        }
    }
    Ok(Options {
        files,
        labels,
        fields,
        out: out.ok_or("missing --out-dir")?,
        valid: valid.ok_or("missing --valid")?,
        region: parse_region(&region_label)?,
        region_label,
        threads,
        width,
        height,
        rows,
    })
}

#[derive(Serialize)]
struct DeletedFile {
    path: PathBuf,
    bytes: u64,
}
#[derive(Serialize)]
struct FieldReceipt {
    field: String,
    units: String,
    stats: PairStats,
    difference_step: f64,
    ny: usize,
    nx: usize,
    path: PathBuf,
    stages: serde_json::Value,
    deleted_files: Vec<DeletedFile>,
}

fn render_field(opt: &Options, row: &VariableRow) -> Result<FieldReceipt, String> {
    let start = Instant::now();
    let read = |path: &Path, spec: &PlaneSpec| -> Result<(Grid, Vec<f32>), String> {
        let file = netcrust::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let grid = read_grid(&file).map_err(|e| format!("{}: {e}", path.display()))?;
        let plane =
            read_plane(&file, &grid, spec).map_err(|e| format!("{}: {e}", path.display()))?;
        Ok((grid, plane))
    };
    let (left, right) = rayon::join(
        || read(&opt.files[0], &row.a),
        || read(&opt.files[1], &row.b),
    );
    let ((ag, av), (bg, bv)) = (left?, right?);
    let read_seconds = start.elapsed().as_secs_f64();
    let stage = Instant::now();
    let grid = common_grid(&ag, &bg, opt.region)?;
    let a: Vec<_> = grid.a_indices.par_iter().map(|&i| av[i]).collect();
    let b: Vec<_> = grid.b_indices.par_iter().map(|&i| bv[i]).collect();
    let stats = pair_stats(&a, &b, &grid.lat)?;
    let stats_seconds = stage.elapsed().as_secs_f64();
    let step = adaptive_difference_step(&stats, row.difference_step);
    let style = style(row)?;
    let delta = difference(&a, &b);
    let domain = if opt.region.is_some() {
        "latlon-region"
    } else {
        "latlon-global"
    };
    let stem = format!(
        "latlon-pair_{}_{}",
        row.field,
        opt.valid.format("%Y%m%dT%H%MZ")
    );
    let path = layout_path(
        &opt.out,
        domain,
        &row.product_slug,
        &opt.valid.format("%Y-%m-%d").to_string(),
        &stem,
    );
    let stage = Instant::now();
    let mut panels = Vec::new();
    let mut parts = Vec::new();
    let mut render_seconds = Vec::new();
    for (index, values) in [a, b, delta].into_iter().enumerate() {
        let panel_start = Instant::now();
        let part = path.with_file_name(format!(".{stem}-{}-{index}.png", std::process::id()));
        let is_delta = index == 2;
        render_panel(PanelRequest {
            lat_deg: &grid.lat,
            lon_deg: &grid.lon,
            projection: Some(&GridProjection::Geographic),
            ny: grid.ny,
            nx: grid.nx,
            values,
            product_slug: row.product_slug.clone(),
            title: if is_delta {
                "A minus B".into()
            } else {
                format!("{}: {}", ["A", "B"][index], opt.labels[index])
            },
            display_units: row.display_units.clone(),
            scale: if is_delta {
                difference_scale(step)
            } else {
                style.scale.clone()
            },
            cbar_tick_step: if is_delta {
                Some(step * 2.0)
            } else {
                style.tick
            },
            legend: if is_delta {
                LegendControls::default()
            } else {
                style.legend
            },
            render_density: style.density,
            subtitle_left: opt.valid.format("Valid %Y-%m-%d %H:%MZ").to_string(),
            subtitle_center: None,
            subtitle_right: if is_delta {
                "A - B; open scale ends".into()
            } else {
                "CF lat-lon".into()
            },
            width: opt.width,
            height: opt.height,
            contours: Vec::new(),
            colorbar: true,
            overlays: None,
            annotations: None,
            out_path: part.clone(),
        })?;
        panels.push(
            image::open(&part)
                .map_err(|e| format!("{}: {e}", part.display()))?
                .to_rgba8(),
        );
        parts.push(part);
        render_seconds.push(panel_start.elapsed().as_secs_f64());
    }
    let render_wall = stage.elapsed().as_secs_f64();
    let stage = Instant::now();
    let corr = stats
        .pattern_correlation
        .map(|x| format!("{x:.5}"))
        .unwrap_or_else(|| "undefined".into());
    let sheet = compose_sheet(
        &panels,
        &SheetHeader {
            title: format!("{} ({})", row.title, row.display_units),
            subtitle: format!(
                "{} | {} | area-weighted A-B bias {:+.3}, RMSE {:.3}, pattern r {} | n={} | {}",
                opt.valid.format("%Y-%m-%d %H:%MZ"),
                opt.region_label,
                stats.bias,
                stats.rmse,
                corr,
                stats.common_finite_cells,
                row.display_units
            ),
        },
    )?;
    sheet
        .save(&path)
        .map_err(|e| format!("{}: {e}", path.display()))?;
    let compose_seconds = stage.elapsed().as_secs_f64();
    let mut deleted_files = Vec::new();
    for part in parts {
        let bytes = std::fs::metadata(&part).map_err(|e| e.to_string())?.len();
        std::fs::remove_file(&part).map_err(|e| e.to_string())?;
        deleted_files.push(DeletedFile { path: part, bytes });
    }
    let receipt = FieldReceipt {
        field: row.field.clone(),
        units: row.display_units.clone(),
        stats,
        difference_step: step,
        ny: grid.ny,
        nx: grid.nx,
        path,
        stages: serde_json::json!({"read_and_convert_seconds":read_seconds,"common_grid_and_stats_seconds":stats_seconds,
            "render_panels_seconds":render_wall,"individual_panel_seconds":render_seconds,"compose_write_seconds":compose_seconds,"total_seconds":start.elapsed().as_secs_f64(),
            "parallelism":format!("shared {}-thread Rayon pool; fields, paired reads, rows and production raster work",opt.threads)}),
        deleted_files,
    };
    println!(
        "LATLON_STATS {}",
        serde_json::to_string(&receipt).map_err(|e| e.to_string())?
    );
    println!("RENDERED {} {}", row.product_slug, receipt.path.display());
    Ok(receipt)
}

pub fn cli(args: &[String]) -> Result<(), String> {
    if args.iter().any(|a| a == "--help" || a == "-h") {
        println!("{USAGE}");
        return Ok(());
    }
    if args.iter().any(|a| a == "--print-variable-table") {
        println!(
            "{}",
            serde_json::to_string_pretty(&variable_table()).map_err(|e| e.to_string())?
        );
        return Ok(());
    }
    let start = Instant::now();
    let opt = options(args)?;
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(opt.threads)
        .build()
        .map_err(|e| e.to_string())?;
    let results: Vec<_> = pool.install(|| {
        opt.fields
            .par_iter()
            .map(|field| render_field(&opt, find_row(&opt.rows, field).unwrap()))
            .collect()
    });
    let results = results.into_iter().collect::<Result<Vec<_>, _>>()?;
    let receipt = serde_json::json!({"schema":"rustwx.latlon-pair.v1","valid":opt.valid,"files":opt.files,"labels":opt.labels,
        "region":opt.region_label,"threads":opt.threads,"wall_seconds":start.elapsed().as_secs_f64(),"variable_table":opt.rows,"fields":results});
    let path = opt.out.join(format!(
        "latlon-pair-{}.json",
        opt.valid.format("%Y%m%dT%H%MZ")
    ));
    std::fs::write(
        &path,
        serde_json::to_vec_pretty(&receipt).map_err(|e| e.to_string())?,
    )
    .map_err(|e| format!("{}: {e}", path.display()))?;
    println!("LATLON_RECEIPT {}", path.display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn table_names_and_conversions() {
        let rows = variable_table();
        validate_table(&rows).unwrap();
        assert!(
            find_row(&rows, "unknown_temperature")
                .unwrap_err()
                .contains("unknown_temperature")
        );
        for (field, raw, expected) in [
            ("z500", 58839.9, 600.0),
            ("msl", 101325.0, 1013.25),
            ("t850", 273.15, 0.0),
            ("t2m", 303.15, 30.0),
            ("tcwv", 42.0, 42.0),
        ] {
            let row = find_row(&rows, field).unwrap();
            let value = convert_plane(&row.a, &[vec![raw]])[0];
            assert!((value - expected).abs() < 1e-3, "{field}: {value}");
            assert_eq!(row.a.variables[0], field);
            assert_eq!(row.b.variables[0], field);
            let _ = style(row).unwrap();
        }
        let row = find_row(&rows, "u10m_v10m_speed").unwrap();
        let speed = convert_plane(&row.a, &[vec![3.0, f64::NAN], vec![4.0, 4.0]]);
        assert_eq!(speed[0], 5.0);
        assert!(speed[1].is_nan());
    }
    #[test]
    fn latitude_order_and_longitude_wrap_keep_values_attached() {
        let lat = axis(&[90.0, 0.0, -90.0], false).unwrap();
        let lon = axis(&[0.0, 90.0, 180.0, 270.0, 360.0], true).unwrap();
        assert_eq!(lat.values, [-90.0, 0.0, 90.0]);
        assert_eq!(lat.indices, [2, 1, 0]);
        assert_eq!(lon.values, [-180.0, -90.0, 0.0, 90.0]);
        assert_eq!(lon.indices[0..2], [2, 3]);
        assert!(axis(&[0.0, 2.0, 1.0], false).is_err());
        assert!(axis(&[0.0, 1.0, 3.0], false).is_err());
        let a = Grid {
            lat,
            lon,
            lat_dim: "lat".into(),
            lon_dim: "lon".into(),
        };
        let b = Grid {
            lat: axis(&[-90.0, 0.0, 90.0], false).unwrap(),
            lon: axis(&[-180.0, -90.0, 0.0, 90.0], true).unwrap(),
            lat_dim: "lat".into(),
            lon_dim: "lon".into(),
        };
        let common = common_grid(&a, &b, None).unwrap();
        assert_eq!(common.a_indices, common.b_indices);
        let region = parse_region("-20,20,170,-80").unwrap();
        assert!(common_grid(&a, &b, region).is_err()); // only one latitude in crop
    }
    #[test]
    fn cf_reader_reorders_transposes_unpacks_and_refuses_named_variables() {
        use netcdf_writer::{AttrValue, NcFormat, NcType, NcWriter, Schema, VarData};
        let path = std::env::temp_dir().join(format!("latlon-pair-cf-{}.nc", std::process::id()));
        let mut schema = Schema::new(NcFormat::Offset64);
        let lat_dim = schema.def_dim("latitude", 3, false).unwrap();
        let lon_dim = schema.def_dim("longitude", 4, false).unwrap();
        let lat = schema.def_var("lat", NcType::Double, &[lat_dim]).unwrap();
        let lon = schema.def_var("lon", NcType::Double, &[lon_dim]).unwrap();
        let plane = schema
            .def_var("packed", NcType::Short, &[lon_dim, lat_dim])
            .unwrap();
        schema
            .put_var_attr(plane, "scale_factor", AttrValue::Doubles(vec![0.5]))
            .unwrap();
        schema
            .put_var_attr(plane, "add_offset", AttrValue::Doubles(vec![273.15]))
            .unwrap();
        schema
            .put_var_attr(plane, "_FillValue", AttrValue::Shorts(vec![-999]))
            .unwrap();
        let mut writer = NcWriter::create(&path, schema).unwrap();
        writer
            .write_var(lat, VarData::F64(&[60.0, 0.0, -60.0]))
            .unwrap();
        writer
            .write_var(lon, VarData::F64(&[0.0, 90.0, 180.0, 270.0]))
            .unwrap();
        writer
            .write_var(
                plane,
                VarData::I16(&[0, 2, 4, 10, 12, 14, 20, 22, 24, 30, 32, -999]),
            )
            .unwrap();
        writer.finish().unwrap();
        let file = netcrust::open(&path).unwrap();
        let grid = read_grid(&file).unwrap();
        let values = read_plane(
            &file,
            &grid,
            &PlaneSpec {
                variables: vec!["packed".into()],
                factor: 1.0,
                offset: -273.15,
            },
        )
        .unwrap();
        // First canonical row is 60 S; normalized longitudes start at 180 E.
        assert_eq!(values[0], 12.0);
        assert!(values[1].is_nan());
        assert_eq!(&values[2..4], &[2.0, 7.0]);
        assert_eq!(values[4], 11.0);
        assert_eq!(values[8], 10.0);
        let error = read_variable(&file, "missing_model_variable", &grid).unwrap_err();
        assert!(error.contains("missing_model_variable"), "{error}");
        let reversed = Grid {
            lat: axis(&[-60.0, 0.0, 60.0], false).unwrap(),
            lon: axis(&[-180.0, -90.0, 0.0, 90.0], true).unwrap(),
            lat_dim: "lat".into(),
            lon_dim: "lon".into(),
        };
        let crop = common_grid(&grid, &reversed, parse_region("-60,60,90,-90").unwrap()).unwrap();
        assert_eq!(crop.nx, 3);
        assert_eq!(&crop.lon[..3], &[90.0, 180.0, 270.0]);
        assert_eq!(&crop.a_indices[..3], &[3, 0, 1]);
        let encoded = serde_json::to_vec(&variable_table()).unwrap();
        let decoded: Vec<VariableRow> = serde_json::from_slice(&encoded).unwrap();
        assert_eq!(
            find_row(&decoded, "z500").unwrap().product_slug,
            "500mb_height"
        );
        // The lane removes this spent fixture and records its size after validation.
    }

    #[test]
    fn finite_area_weights_and_pattern_correlation() {
        let s = pair_stats(
            &[2.0, 4.0, 100.0, f32::NAN],
            &[1.0, 2.0, f32::NAN, 1.0],
            &[0.0, 60.0, 0.0, 0.0],
        )
        .unwrap();
        assert_eq!(s.common_finite_cells, 2);
        assert!((s.bias - 4.0 / 3.0).abs() < 1e-12);
        assert!((s.rmse - 2.0f64.sqrt()).abs() < 1e-12);
        assert!((s.pattern_correlation.unwrap() - 1.0).abs() < 1e-12);
        assert_eq!(
            pair_stats(&[1.0, 1.0], &[2.0, 3.0], &[0.0, 0.0])
                .unwrap()
                .pattern_correlation,
            None
        );
        assert!(pair_stats(&[f32::NAN], &[0.0], &[0.0]).is_err());
    }
}
