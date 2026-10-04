//! LANE 2.  The static-field build (`build_static` / `_build_static`
//! in `gpuwm/static/build.py`) plus the landmask/dominant-category
//! rules and `monthly_interp_to_date`.
//!
//! Output contract (the native static inventory, float64, cropped to
//! the mass grid `(e_sn-1, e_we-1)`):
//!
//! | field       | shape          | notes                                    |
//! |-------------|----------------|------------------------------------------|
//! | HGT_M       | (ny, nx)       | gcell+four_pt+average_4pt, one           |
//! |             |                | smth-desmth_special pass                 |
//! | LANDUSEF    | (21, ny, nx)   | fractional gcell + four_pt, f32-recip    |
//! |             |                | normalized                               |
//! | LANDMASK    | (ny, nx)       | water frac >= 0.5 -> 0                   |
//! | LU_INDEX    | (ny, nx)       | dominant water/land type (lake beats     |
//! |             |                | ocean only strictly)                     |
//! | SOILCTOP/BOT| (16, ny, nx)   | four_pt only                             |
//! | SCT/SCB_DOM | (ny, nx)       | plain argmax + 1, lowest wins ties       |
//! | GREENFRAC   | (12, ny, nx)   | monthly, water-masked 0                  |
//! | LAI12M      | (12, ny, nx)   | monthly, water-masked 0                  |
//! | ALBEDO12M   | (12, ny, nx)   | monthly, fill 8, water-masked 8          |
//! | SNOALB      | (ny, nx)       | month 1, water-masked 0                  |
//! | SOILTEMP    | (ny, nx)       | sixteen_pt-led sequence, water-masked 0  |
//! | TMN         | (ny, nx)       | SOILTEMP - 0.0065*HGT_M on land          |
//!
//! The GEOG directory selection (WPS `geog_data_res` token resolution,
//! `GeogSelection`) STAYS PYTHON -- it is namelist/config orchestration
//! -- and arrives here as nine resolved dataset paths.

use std::path::PathBuf;

use crate::error::{Result, StaticError};
use crate::geog::GeogDataset;
use crate::interp::InterpOp;
use crate::projection::ProjectedGrid;
use crate::sampler::DomainSampler;
use crate::smooth::{apply_terrain_smoothing, TerrainSmoothing, WPS_DEFAULT};
use crate::types::{Field, FieldSet, Grid2, Stack3};

/// The nine resolved GEOG dataset directories for one build
/// (`GeogSelection` after token resolution; order and names mirror
/// `_DEFAULT_GEOG_DIRS` keys).
#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct GeogPaths {
    pub terrain: PathBuf,
    pub landuse: PathBuf,
    pub soil_top: PathBuf,
    pub soil_bottom: PathBuf,
    pub greenfrac: PathBuf,
    pub lai: PathBuf,
    pub albedo: PathBuf,
    pub snow_albedo: PathBuf,
    pub soil_temperature: PathBuf,
}

/// Build every native static field for `grid` (`build_static`).
/// Coverage receipts are returned inside the `FieldSet`.  LANE 2.
pub fn build_static(
    grid: &ProjectedGrid,
    paths: &GeogPaths,
    halo: usize,
) -> Result<FieldSet> {
    build_static_smoothed(grid, paths, halo, WPS_DEFAULT)
}

/// [`build_static`] with one domain's terrain smoothing.
pub fn build_static_smoothed(
    grid: &ProjectedGrid,
    paths: &GeogPaths,
    halo: usize,
    smoothing: TerrainSmoothing,
) -> Result<FieldSet> {
    let sampler = DomainSampler::new(grid, halo)?;
    build_static_with_sampler_smoothed(&sampler, paths, smoothing)
}

/// Build just terrain with the identical sampler and smoother as the full set.
pub fn build_terrain(grid: &ProjectedGrid, terrain: &std::path::Path, halo: usize) -> Result<FieldSet> {
    build_terrain_smoothed(grid, terrain, halo, WPS_DEFAULT)
}

/// [`build_terrain`] with one domain's terrain smoothing.
pub fn build_terrain_smoothed(
    grid: &ProjectedGrid,
    terrain: &std::path::Path,
    halo: usize,
    smoothing: TerrainSmoothing,
) -> Result<FieldSet> {
    let dom = DomainSampler::new(grid, halo)?;
    let (hgt, coverage) = sampled_terrain(&dom, terrain, smoothing)?;
    let mut set = FieldSet::default();
    set.fields.insert("HGT_M".to_string(), Field::Plane(hgt));
    set.coverage_reports.insert("terrain".to_string(), coverage);
    Ok(set)
}

fn sampled_terrain(
    dom: &DomainSampler<'_>,
    terrain: &std::path::Path,
    smoothing: TerrainSmoothing,
) -> Result<(Grid2, String)> {
    let topo = GeogDataset::open(terrain, None)?;
    let win = dom.window(&topo, 3)?;
    let coverage = dom.require_source_coverage(&topo, &win, "terrain")?;
    let hgt = dom.continuous(&topo, &win, 0,
        &[InterpOp::FourPt, InterpOp::Average4Pt], 0.0, true, None)?;
    let hgt = apply_terrain_smoothing(&hgt, smoothing)?;
    Ok((crop_grid(dom, &hgt), coverage))
}

/// Crop an extended plane to the mass grid.
fn crop_grid(dom: &DomainSampler<'_>, g: &Grid2) -> Grid2 {
    let mut data = Vec::with_capacity(dom.ny * dom.nx);
    for j in 0..dom.ny {
        let row = (j + dom.halo) * dom.nxe + dom.halo;
        data.extend_from_slice(&g.data[row..row + dom.nx]);
    }
    Grid2 {
        ny: dom.ny,
        nx: dom.nx,
        data,
    }
}

fn crop_stack(dom: &DomainSampler<'_>, s: &Stack3) -> Stack3 {
    let n = dom.nye * dom.nxe;
    let mut data = Vec::with_capacity(s.planes * dom.ny * dom.nx);
    for z in 0..s.planes {
        for j in 0..dom.ny {
            let row = z * n + (j + dom.halo) * dom.nxe + dom.halo;
            data.extend_from_slice(&s.data[row..row + dom.nx]);
        }
    }
    Stack3 {
        planes: s.planes,
        ny: dom.ny,
        nx: dom.nx,
        data,
    }
}

/// The build against an already-constructed sampler (`_build_static`).
/// Split out so mesh-fixture tests and the future assembler can drive
/// it; `build_static` is the production entry.
pub fn build_static_with_sampler(
    dom: &DomainSampler<'_>,
    paths: &GeogPaths,
) -> Result<FieldSet> {
    build_static_with_sampler_smoothed(dom, paths, WPS_DEFAULT)
}

/// [`build_static_with_sampler`] with one domain's terrain smoothing.
pub fn build_static_with_sampler_smoothed(
    dom: &DomainSampler<'_>,
    paths: &GeogPaths,
    smoothing: TerrainSmoothing,
) -> Result<FieldSet> {
    let mut set = FieldSet::default();

    let require = |set: &mut FieldSet,
                       field: &str,
                       ds: &GeogDataset,
                       win: &crate::geog::GeogWindow|
     -> Result<()> {
        let receipt = dom.require_source_coverage(ds, win, field)?;
        if set.coverage_reports.contains_key(field) {
            return Err(StaticError::Invalid(format!(
                "duplicate static source-coverage field '{field}'"
            )));
        }
        set.coverage_reports.insert(field.to_string(), receipt);
        Ok(())
    };

    // --- terrain: average_gcell(4.0)+four_pt+average_4pt, fill 0, one
    //     smoother-desmoother pass (choice arbitrated by geo_em HGT_M).
    let (hgt, coverage) = sampled_terrain(dom, &paths.terrain, smoothing)?;
    set.coverage_reports.insert("terrain".to_string(), coverage);

    // --- landuse -> LANDUSEF / LANDMASK / LU_INDEX ---------------------
    let lu_ds = GeogDataset::open(&paths.landuse, None)?;
    // Python `index.iswater or 17`: absent AND zero both fall back.
    let iswater = match lu_ds.index.iswater {
        None | Some(0) => 17,
        Some(w) => w,
    };
    let islake = match lu_ds.index.islake {
        Some(l) if l < 0 => {
            return Err(StaticError::Invalid(format!(
                "landuse index declares negative islake={l}"
            )))
        }
        other => other,
    };
    let win = dom.window(&lu_ds, 3)?;
    require(&mut set, "landuse", &lu_ds, &win)?;
    let luf_e = dom.categorical(&lu_ds, &win, true)?;
    let luf = crop_stack(dom, &luf_e);
    let landmask_e = landmask_from_landusef(&luf_e, iswater, islake)?;
    let landmask = crop_grid(dom, &landmask_e);
    let lu_index =
        lu_index_from_landusef(&luf, &landmask, iswater, islake)?;
    drop(win);
    drop(lu_ds);
    drop(luf_e);

    // --- soil categories ----------------------------------------------
    let mut soils: Vec<(&str, &str, &PathBuf)> = Vec::new();
    soils.push(("SOILCTOP", "soil_top", &paths.soil_top));
    soils.push(("SOILCBOT", "soil_bottom", &paths.soil_bottom));
    let mut soil_out: Vec<(String, Stack3, Grid2)> = Vec::new();
    for (name, field, path) in soils {
        let ds = GeogDataset::open(path, None)?;
        let win = dom.window(&ds, 3)?;
        require(&mut set, field, &ds, &win)?;
        let frac = crop_stack(dom, &dom.categorical(&ds, &win, false)?);
        let dominant = dominant_category(&frac)?;
        soil_out.push((name.to_string(), frac, dominant));
    }

    let water: Vec<bool> =
        landmask.data.iter().map(|&v| v == 0.0).collect();
    let water_e: Vec<bool> =
        landmask_e.data.iter().map(|&v| v == 0.0).collect();
    let active_e: Vec<bool> = water_e.iter().map(|&w| !w).collect();

    // --- monthly climatologies + masked scalars -----------------------
    let monthly = |set: &mut FieldSet,
                       field: &str,
                       path: &PathBuf,
                       seq: &[InterpOp],
                       fill: f64,
                       mask_fill: f64|
     -> Result<Stack3> {
        let ds = GeogDataset::open(path, None)?;
        let nz = ds.index.nz() as usize;
        let mut months = Stack3 {
            planes: nz,
            ny: dom.ny,
            nx: dom.nx,
            data: Vec::with_capacity(nz * dom.ny * dom.nx),
        };
        for z in 0..nz {
            let win = dom.window_plane(&ds, 3, z)?;
            if z == 0 {
                // Coverage is geometric and identical for every source month.
                require(set, field, &ds, &win)?;
            }
            let plane = dom.continuous(
                &ds,
                &win,
                z,
                seq,
                fill,
                true,
                Some(&active_e),
            )?;
            months.data.extend_from_slice(&crop_grid(dom, &plane).data);
        }
        let plane_len = dom.ny * dom.nx;
        for z in 0..nz {
            for (k, &w) in water.iter().enumerate() {
                if w {
                    months.data[z * plane_len + k] = mask_fill;
                }
            }
        }
        Ok(months)
    };

    let seq = [
        InterpOp::FourPt,
        InterpOp::Average4Pt,
        InterpOp::Average16Pt,
        InterpOp::Search,
    ];
    let greenfrac =
        monthly(&mut set, "greenfrac", &paths.greenfrac, &seq, 0.0, 0.0)?;
    // GEOGRID.TBL.ARW's `default` dataset is the pre-aggregated
    // 10-minute product; `modis_lai` selects the 30-second tree
    // upstream (path resolution stays Python).
    let lai = monthly(&mut set, "lai", &paths.lai, &seq, 0.0, 0.0)?;
    let albedo =
        monthly(&mut set, "albedo", &paths.albedo, &seq, 8.0, 8.0)?;
    let snoalb_stack = monthly(
        &mut set,
        "snow_albedo",
        &paths.snow_albedo,
        &seq,
        0.0,
        0.0,
    )?;
    let soiltemp_stack = monthly(
        &mut set,
        "soil_temperature",
        &paths.soil_temperature,
        &[
            InterpOp::SixteenPt,
            InterpOp::FourPt,
            InterpOp::Average4Pt,
            InterpOp::Average16Pt,
            InterpOp::Search,
        ],
        0.0,
        0.0,
    )?;
    let plane_len = dom.ny * dom.nx;
    let snoalb = Grid2 {
        ny: dom.ny,
        nx: dom.nx,
        data: snoalb_stack.data[..plane_len].to_vec(),
    };
    let soiltemp = Grid2 {
        ny: dom.ny,
        nx: dom.nx,
        data: soiltemp_stack.data[..plane_len].to_vec(),
    };

    // --- deep soil temperature, elevation-corrected (real.exe input;
    //     WRF share/module_soil_pre.F:973, land only) ------------------
    let tmn = Grid2 {
        ny: dom.ny,
        nx: dom.nx,
        data: landmask
            .data
            .iter()
            .zip(soiltemp.data.iter().zip(hgt.data.iter()))
            .map(|(&mask, (&st, &h))| {
                if mask > 0.5 { st - 0.0065 * h } else { st }
            })
            .collect(),
    };

    set.fields.insert("HGT_M".to_string(), Field::Plane(hgt));
    set.fields.insert("LANDUSEF".to_string(), Field::Stack(luf));
    set.fields
        .insert("LANDMASK".to_string(), Field::Plane(landmask));
    set.fields
        .insert("LU_INDEX".to_string(), Field::Plane(lu_index));
    for (name, frac, dominant) in soil_out {
        let dom_name = if name == "SOILCTOP" {
            "SCT_DOM"
        } else {
            "SCB_DOM"
        };
        set.fields.insert(name, Field::Stack(frac));
        set.fields
            .insert(dom_name.to_string(), Field::Plane(dominant));
    }
    set.fields
        .insert("GREENFRAC".to_string(), Field::Stack(greenfrac));
    set.fields.insert("LAI12M".to_string(), Field::Stack(lai));
    set.fields
        .insert("ALBEDO12M".to_string(), Field::Stack(albedo));
    set.fields.insert("SNOALB".to_string(), Field::Plane(snoalb));
    set.fields
        .insert("SOILTEMP".to_string(), Field::Plane(soiltemp));
    set.fields.insert("TMN".to_string(), Field::Plane(tmn));
    Ok(set)
}

/// One sub-grid orographic statistic as GEOGRID.TBL.ARW builds it for WRF's
/// terrain drag (`topo_wind`, `gwd_opt`).  Every row the table carries for
/// these fields is `continuous` with `fill_missing = 0`; they differ in two
/// switches only, which is all this request carries:
///
/// * `gcell`: `average_gcell(4.0)+four_pt+average_4pt` (VAR_SSO, the
///   terrain's own sequence) when true, else `average_4pt` alone (CON, VAR,
///   OA1-4, OL1-4 and the GSL large- and small-scale statistics);
/// * `masked_water`: `masked = water`, so a water cell of the GEOG land use
///   takes `fill_missing` and is never interpolated (the GREENFRAC
///   treatment above, on the same land mask).
#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
#[serde(deny_unknown_fields)]
pub struct OrographicField {
    pub name: String,
    pub path: PathBuf,
    pub gcell: bool,
    pub masked_water: bool,
    #[serde(default)]
    pub masked_land: bool,
    #[serde(default = "default_gcell_ratio")]
    pub gcell_ratio: f32,
    #[serde(default)]
    pub search_depth: Option<u32>,
    #[serde(default)]
    pub fill_missing: f64,
}

fn default_gcell_ratio() -> f32 { 4.0 }

/// The orographic fields to build, and the land-use dataset whose mask the
/// `masked = water` rows read (the same dataset `build_static` reads).
#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
#[serde(deny_unknown_fields)]
pub struct OrographicRequest {
    #[serde(default)]
    pub landuse: Option<PathBuf>,
    pub fields: Vec<OrographicField>,
}

/// Build the requested orographic statistics for `grid` (no other field).
pub fn build_orographic(
    grid: &ProjectedGrid,
    request: &OrographicRequest,
    halo: usize,
) -> Result<FieldSet> {
    let dom = DomainSampler::new(grid, halo)?;
    build_orographic_with_sampler(&dom, request)
}

/// [`build_orographic`] against an already-constructed sampler.
pub fn build_orographic_with_sampler(
    dom: &DomainSampler<'_>,
    request: &OrographicRequest,
) -> Result<FieldSet> {
    let mut set = FieldSet::default();
    let masked = request.fields.iter().any(|f| f.masked_water || f.masked_land);
    let (water, active_e) = if masked {
        let Some(path) = request.landuse.as_ref() else {
            return Err(StaticError::Invalid(
                "a masked = water orographic field needs the land-use                  dataset its mask comes from"
                    .to_string(),
            ));
        };
        let lu_ds = GeogDataset::open(path, None)?;
        let iswater = match lu_ds.index.iswater {
            None | Some(0) => 17,
            Some(w) => w,
        };
        let islake = match lu_ds.index.islake {
            Some(l) if l < 0 => {
                return Err(StaticError::Invalid(format!(
                    "landuse index declares negative islake={l}"
                )))
            }
            other => other,
        };
        let win = dom.window(&lu_ds, 3)?;
        let luf_e = dom.categorical(&lu_ds, &win, true)?;
        let landmask_e = landmask_from_landusef(&luf_e, iswater, islake)?;
        let landmask = crop_grid(dom, &landmask_e);
        let water: Vec<bool> =
            landmask.data.iter().map(|&v| v == 0.0).collect();
        let active_e: Vec<bool> =
            landmask_e.data.iter().map(|&v| v != 0.0).collect();
        (water, Some(active_e))
    } else {
        (Vec::new(), None)
    };
    for field in &request.fields {
        if set.fields.contains_key(&field.name) {
            return Err(StaticError::Invalid(format!(
                "orographic field '{}' requested twice",
                field.name
            )));
        }
        let ds = GeogDataset::open(&field.path, None)?;
        let win = dom.window(&ds, 3)?;
        let receipt = dom.require_source_coverage(&ds, &win, &field.name)?;
        set.coverage_reports.insert(field.name.clone(), receipt);
        if field.masked_water && field.masked_land {
            return Err(StaticError::Invalid(format!(
                "field '{}' masks both land and water, leaving no active cells", field.name)));
        }
        if !field.gcell_ratio.is_finite() || field.gcell_ratio < 0.0
            || !field.fill_missing.is_finite() {
            return Err(StaticError::Invalid(format!(
                "field '{}' has invalid interpolation operands", field.name)));
        }
        let seq: Vec<InterpOp> = if let Some(depth) = field.search_depth {
            vec![InterpOp::SearchDepth(depth)]
        } else if field.gcell {
            vec![InterpOp::FourPt, InterpOp::Average4Pt]
        } else {
            vec![InterpOp::Average4Pt]
        };
        let water_active = active_e.as_ref().map(|a|
            a.iter().map(|&v| !v).collect::<Vec<bool>>());
        let active = if field.masked_water {
            active_e.as_deref()
        } else if field.masked_land {
            water_active.as_deref()
        } else {
            None
        };
        let plane =
            dom.continuous_wps(&ds, &win, 0, &seq, field.fill_missing,
                               field.gcell, active, field.gcell_ratio)?;
        let mut cropped = crop_grid(dom, &plane);
        if field.masked_water || field.masked_land {
            for (value, &w) in cropped.data.iter_mut().zip(&water) {
                if (field.masked_water && w) || (field.masked_land && !w) {
                    *value = field.fill_missing;
                }
            }
        }
        set.fields.insert(field.name.clone(), Field::Plane(cropped));
    }
    Ok(set)
}

/// LANDMASK from LANDUSEF (`landmask_from_landusef`): 0 where the
/// water fraction (iswater + islake) is >= 0.5, else 1.  LANE 2.
pub fn landmask_from_landusef(
    luf: &Stack3,
    iswater: i64,
    islake: Option<i64>,
) -> Result<Grid2> {
    let n = luf.ny * luf.nx;
    let water_plane = plane_index(luf, iswater)?;
    let lake_plane = match islake {
        Some(l) => Some(plane_index(luf, l)?),
        None => None,
    };
    let mut data = Vec::with_capacity(n);
    for k in 0..n {
        let mut w = luf.plane(water_plane)[k];
        if let Some(lp) = lake_plane {
            w += luf.plane(lp)[k];
        }
        data.push(if w >= 0.5 { 0.0 } else { 1.0 });
    }
    Ok(Grid2 {
        ny: luf.ny,
        nx: luf.nx,
        data,
    })
}

fn plane_index(luf: &Stack3, category: i64) -> Result<usize> {
    let plane = category - 1;
    if plane < 0 || plane as usize >= luf.planes {
        return Err(StaticError::Invalid(format!(
            "category {category} is outside the {}-plane land-use stack",
            luf.planes
        )));
    }
    Ok(plane as usize)
}

/// Dominant landuse (`lu_index_from_landusef`): water cells get the
/// dominant water type (lake only when strictly larger than ocean);
/// land cells the dominant land category, water types excluded; ties
/// to the lowest category.  LANE 2.
pub fn lu_index_from_landusef(
    luf: &Stack3,
    landmask: &Grid2,
    iswater: i64,
    islake: Option<i64>,
) -> Result<Grid2> {
    let n = luf.ny * luf.nx;
    if landmask.ny != luf.ny || landmask.nx != luf.nx {
        return Err(StaticError::Invalid(format!(
            "landmask is {}x{}, land-use stack is {}x{}",
            landmask.ny, landmask.nx, luf.ny, luf.nx
        )));
    }
    let water_plane = plane_index(luf, iswater)?;
    let lake_plane = match islake {
        Some(l) => Some(plane_index(luf, l)?),
        None => None,
    };
    let mut data = Vec::with_capacity(n);
    for k in 0..n {
        // argmax over the water-masked stack: first maximum wins
        // (lowest category), water planes forced to -1.
        let mut best = f64::NEG_INFINITY;
        let mut best_z = 0usize;
        for z in 0..luf.planes {
            let v = if z == water_plane || Some(z) == lake_plane {
                -1.0
            } else {
                luf.plane(z)[k]
            };
            if v > best {
                best = v;
                best_z = z;
            }
        }
        let dom_land = best_z as f64 + 1.0;
        let dom_water = match lake_plane {
            Some(lp) => {
                if luf.plane(lp)[k] > luf.plane(water_plane)[k] {
                    islake.expect("lake plane implies islake") as f64
                } else {
                    iswater as f64
                }
            }
            None => iswater as f64,
        };
        data.push(if landmask.data[k] < 0.5 {
            dom_water
        } else {
            dom_land
        });
    }
    Ok(Grid2 {
        ny: luf.ny,
        nx: luf.nx,
        data,
    })
}

/// Plain dominant category (`dominant_category`): argmax + 1, first
/// (lowest) category wins ties.  LANE 2.
pub fn dominant_category(frac: &Stack3) -> Result<Grid2> {
    if frac.planes == 0 {
        return Err(StaticError::Invalid(
            "dominant_category of an empty stack".to_string(),
        ));
    }
    let n = frac.ny * frac.nx;
    let mut data = Vec::with_capacity(n);
    for k in 0..n {
        let mut best = f64::NEG_INFINITY;
        let mut best_z = 0usize;
        for z in 0..frac.planes {
            let v = frac.plane(z)[k];
            if v > best {
                best = v;
                best_z = z;
            }
        }
        data.push(best_z as f64 + 1.0);
    }
    Ok(Grid2 {
        ny: frac.ny,
        nx: frac.nx,
        data,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testsupport::{
        assert_bits_f64, golden_dir, json, load_package, read_f64,
    };

    fn stack(dims: &[usize], data: Vec<f64>) -> Stack3 {
        Stack3 {
            planes: dims[0],
            ny: dims[1],
            nx: dims[2],
            data,
        }
    }

    #[test]
    fn landmask_and_dominant_rules_match_the_python() {
        let dir = golden_dir().join("fields");
        let spec = json(&dir.join("goldens.json"));
        let (dims, data) =
            read_f64(&dir.join(spec["luf"].as_str().unwrap()));
        let luf = stack(&dims, data);
        let iswater = spec["iswater"].as_i64().unwrap();
        let islake = spec["islake"].as_i64().unwrap();
        for (key, got) in [
            (
                "landmask",
                landmask_from_landusef(&luf, iswater, Some(islake))
                    .unwrap(),
            ),
            (
                "landmask_nolake",
                landmask_from_landusef(&luf, iswater, None).unwrap(),
            ),
            ("dominant", dominant_category(&luf).unwrap()),
        ] {
            let (_, want) =
                read_f64(&dir.join(spec[key].as_str().unwrap()));
            assert_bits_f64(&got.data, &want, key);
        }
        let (_, lm) =
            read_f64(&dir.join(spec["landmask"].as_str().unwrap()));
        let landmask = Grid2 {
            ny: dims[1],
            nx: dims[2],
            data: lm,
        };
        let got =
            lu_index_from_landusef(&luf, &landmask, iswater, Some(islake))
                .unwrap();
        let (_, want) =
            read_f64(&dir.join(spec["lu_index"].as_str().unwrap()));
        assert_bits_f64(&got.data, &want, "lu_index");
        let (_, lm) = read_f64(
            &dir.join(spec["landmask_nolake"].as_str().unwrap()),
        );
        let landmask = Grid2 {
            ny: dims[1],
            nx: dims[2],
            data: lm,
        };
        let got = lu_index_from_landusef(&luf, &landmask, iswater, None)
            .unwrap();
        let (_, want) =
            read_f64(&dir.join(spec["lu_index_nolake"].as_str().unwrap()));
        assert_bits_f64(&got.data, &want, "lu_index_nolake");
    }

    #[test]
    fn monthly_interp_matches_wrf_real_transcription() {
        let dir = golden_dir().join("fields");
        let spec = json(&dir.join("goldens.json"));
        let (dims, data) =
            read_f64(&dir.join(spec["monthly"].as_str().unwrap()));
        let monthly = stack(&dims, data);
        for case in spec["monthly_cases"].as_array().unwrap() {
            let year = case["year"].as_i64().unwrap() as i32;
            let julian = case["julian"].as_u64().unwrap() as u32;
            let mids: Vec<u32> = case["mid_month_julian"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as u32)
                .collect();
            let mids: [u32; 12] = mids.try_into().unwrap();
            let got =
                monthly_interp_to_date(&monthly, year, julian, &mids)
                    .unwrap();
            let (_, want) =
                read_f64(&dir.join(case["out"].as_str().unwrap()));
            assert_bits_f64(
                &got.data,
                &want,
                &format!("monthly {year}-{julian}"),
            );
        }
    }

    fn check_build(tag: &str) {
        let Some(pkg) = load_package(tag) else {
            eprintln!("SKIP: WPS_GEOG reference tree not present");
            return;
        };
        let dom = pkg.sampler();
        let set =
            build_static_with_sampler(&dom, &pkg.geog_paths()).unwrap();
        let build = pkg.meta["build"].as_object().unwrap();
        let mut want_names: Vec<&String> = build.keys().collect();
        want_names.sort();
        let got_names: Vec<&String> = set.fields.keys().collect();
        assert_eq!(got_names, want_names, "{tag}: field inventory");
        for (name, entry) in build {
            let field = set.fields.get(name).unwrap();
            let (planes, ny, nx) = field.dims();
            let shape: Vec<usize> = entry["shape"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as usize)
                .collect();
            let got_shape = if shape.len() == 2 {
                vec![ny, nx]
            } else {
                vec![planes, ny, nx]
            };
            assert_eq!(got_shape, shape, "{tag}/{name}: shape");
            let (_, want) = read_f64(
                &pkg.dir.join(entry["file"].as_str().unwrap()),
            );
            assert_bits_f64(
                field.data(),
                &want,
                &format!("{tag}/{name}"),
            );
        }
        // every mandatory dataset carries a coverage receipt
        for field in [
            "terrain",
            "landuse",
            "soil_top",
            "soil_bottom",
            "greenfrac",
            "lai",
            "albedo",
            "snow_albedo",
            "soil_temperature",
        ] {
            assert!(
                set.coverage_reports.contains_key(field),
                "{tag}: no coverage receipt for {field}"
            );
        }
    }

    #[test]
    fn coarse_build_is_byte_identical_to_the_python() {
        check_build("coarse");
    }

    #[test]
    fn subkm_build_is_byte_identical_to_the_python() {
        check_build("subkm");
    }
}

/// WRF real's `monthly_interp_to_date` (module_initialize_real.F
/// :8023-8089): 15th-of-month anchors, 31-day fictitious year-wrap
/// anchors, WHOLE-day target, integer-day linear weights.  The date
/// arrives resolved to `(year, julian_day)` plus the twelve
/// mid-month Julian days by the Python caller so no calendar library
/// enters the crate.  LANE 2.
pub fn monthly_interp_to_date(
    monthly: &Stack3,
    year: i32,
    julian_day: u32,
    mid_month_julian: &[u32; 12],
) -> Result<Grid2> {
    if monthly.planes != 12 {
        return Err(StaticError::Invalid(
            "monthly climatology must have a leading 12-axis".to_string(),
        ));
    }
    let mut middle = [0i64; 14];
    for month in 1..=12usize {
        middle[month] =
            year as i64 * 1000 + mid_month_julian[month - 1] as i64;
    }
    middle[0] = middle[1] - 31;
    middle[13] = middle[12] + 31;
    let target = year as i64 * 1000 + julian_day as i64;
    for anchor in 0..13usize {
        if middle[anchor] < target && target <= middle[anchor + 1] {
            let (month1, month2) = if anchor == 0 || anchor == 12 {
                (12usize, 1usize)
            } else {
                (anchor, anchor + 1)
            };
            let w2 = (target - middle[anchor]) as f64;
            let w1 = (middle[anchor + 1] - target) as f64;
            let span = (middle[anchor + 1] - middle[anchor]) as f64;
            let n = monthly.ny * monthly.nx;
            let p2 = monthly.plane(month2 - 1);
            let p1 = monthly.plane(month1 - 1);
            let mut data = Vec::with_capacity(n);
            for k in 0..n {
                data.push((p2[k] * w2 + p1[k] * w1) / span);
            }
            return Ok(Grid2 {
                ny: monthly.ny,
                nx: monthly.nx,
                data,
            });
        }
    }
    Err(StaticError::Invalid(format!(
        "no mid-month interval brackets year={year} julian={julian_day}"
    )))
}
