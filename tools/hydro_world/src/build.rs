//! The global fabric builder (contract stages 1-9) for any lon/lat box.
//! Table-driven: the source row names the inputs and the method, the recipe
//! table holds every channel and level-pool number. Two methods share one
//! assembly step:
//! - `dem-d8` (open, the worldwide default): river network and catchments
//!   derived from the DEM itself (flow.rs), lakes from HydroLAKES;
//! - `vector-basins` (opt-in cross-check): river polylines and basin
//!   polygons with downstream ids (HydroRIVERS + HydroBASINS), see vector.rs.
use crate::contract::*;
use crate::grid::{BBox, Grid};
use crate::source::{BuildSpec, SourceRow};
use crate::{derive, flow, geo, grid, polygonize, shp, source, tiff, Result};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::time::Instant;

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Recipes {
    pub schema: String,
    pub owner: String,
    pub channel: Channel,
    pub level_pool: LevelPool,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Channel {
    pub provenance: String,
    pub top_width_a: f64,
    pub top_width_b: f64,
    pub bottom_width_fraction: f64,
    pub compound_top_width_factor: f64,
    pub side_slope_a: f64,
    pub side_slope_b: f64,
    pub manning_by_min_order: Vec<(f64, f64)>,
    pub compound_manning_factor: f64,
    pub min_slope: f64,
    pub musk_s: f64,
    pub musx: f64,
    pub qi_m3s: f64,
    pub kchan_mm_h: f64,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct LevelPool {
    pub provenance: String,
    pub weir_c: f64,
    pub weir_l_m: f64,
    pub orifice_c: f64,
    pub orifice_a_m2: f64,
    pub dam_length: f64,
    pub ifd: f64,
    pub pool_depth_per_mean_depth: f64,
    pub max_above_weir_fraction: f64,
    pub default_mean_depth_m: f64,
}

pub fn load_recipes(path: &Path) -> Result<Recipes> {
    let r: Recipes = serde_json::from_slice(&std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?)?;
    if r.schema != "arwen.hydro-world.recipes.v1" {
        return Err(format!("{}: schema {} is not arwen.hydro-world.recipes.v1", path.display(), r.schema).into());
    }
    Ok(r)
}

#[derive(Debug, Default, Serialize)]
pub struct Receipt {
    pub stages: Vec<StageTime>,
    pub counts: BTreeMap<String, Value>,
}

#[derive(Debug, Serialize)]
pub struct StageTime {
    pub stage: String,
    pub ms: f64,
    /// "N threads (rayon)" or "1 thread" with the reason.
    pub parallelism: String,
}

impl Receipt {
    pub fn time<T>(&mut self, stage: &str, par: &str, f: impl FnOnce() -> T) -> T {
        let t = Instant::now();
        let v = f();
        self.stages.push(StageTime { stage: stage.into(), ms: t.elapsed().as_secs_f64() * 1e3, parallelism: par.into() });
        v
    }
    pub fn count(&mut self, k: &str, v: impl Serialize) {
        self.counts.insert(k.into(), serde_json::to_value(v).unwrap_or(Value::Null));
    }
}

pub fn threads() -> String {
    format!("{} threads (rayon)", rayon::current_num_threads())
}

/// Method-independent hydrography the assembly step turns into a fabric.
pub struct Hydro {
    pub reaches: Vec<ReachIn>,
    pub lakes: Vec<LakeIn>,
    /// Dam records for the `FABRIC.gpkg.dams.json` sidecar (dem-d8 rows with a `dams` input).
    pub dams: Vec<Value>,
}

pub struct ReachIn {
    pub down: Option<usize>,
    /// lon/lat, upstream to downstream, ending at the downstream nexus point.
    pub line: Vec<(f64, f64)>,
    /// Exterior and holes, lon/lat.
    pub divide: Option<(Vec<(f64, f64)>, Vec<Vec<(f64, f64)>>)>,
    pub local_area_km2: f64,
    pub slope: f64,
    pub hf_id: f64,
    pub terrain: Option<Terrain>,
}

#[derive(Debug, Clone, Copy, Serialize)]
pub struct Terrain {
    pub elevation_mean_m: f64,
    pub slope_mean_deg: f64,
    pub aspect_mean_deg: f64,
    pub centroid_lat_deg: f64,
    pub centroid_lon_deg: f64,
}

pub struct LakeIn {
    pub hylak: i64,
    pub area_km2: f64,
    pub mean_depth_m: f64,
    pub weir_e_m: f64,
    pub outlet: (f64, f64),
    /// Reach whose downstream end is the lake outlet.
    pub reach: usize,
}

pub struct Options {
    pub sources: PathBuf,
    pub row: String,
    pub bbox: BBox,
    pub data_root: PathBuf,
    pub accept: Vec<String>,
    pub params: Option<PathBuf>,
    pub revision: String,
    pub device: Device,
    pub parent_grid: Option<crate::parent_grid::ParentGrid>,
}

/// Where the DEM per-cell passes run. Both give bit-identical results
/// (`rw_hydro_world hydrography-compare`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum Device {
    /// The GPU when a CUDA driver, a device and NVRTC load; otherwise all CPU
    /// cores, with the reason printed and written to the receipt.
    Auto,
    /// The GPU or an error.
    Cuda,
    /// All CPU cores (Rayon).
    Cpu,
}

impl Device {
    pub fn parse(s: &str) -> Result<Device> {
        match s {
            "auto" => Ok(Device::Auto),
            "cuda" | "gpu" => Ok(Device::Cuda),
            "cpu" => Ok(Device::Cpu),
            _ => Err(format!("--device {s}: use auto, cuda or cpu").into()),
        }
    }
}

/// `--row auto`: the first rule of the table's `auto` list whose limits the
/// box meets (e.g. GLO-30 for small boxes, GLO-90 otherwise).
pub fn resolve_row(table: &source::SourceTable, row: &str, b: &BBox) -> Result<String> {
    resolve(table, row, b).map(|(r, _)| r)
}

/// The row and the integer factor its grid is coarsened by over this box
/// (1: the row's own grid). A rule with `coarsen` keeps its row past
/// `max_cells` on the coarsened grid; a named row coarsens only as far as the
/// cell index needs. Either way no box is refused for its size.
pub fn resolve(table: &source::SourceTable, row: &str, b: &BBox) -> Result<(String, u32)> {
    let base = |id: &str| table.rows.iter().find(|r| r.id == id).and_then(|r| r.build.as_ref()).and_then(|s| s.grid_arcsec);
    if row != "auto" {
        return Ok((row.to_string(), base(row).map_or(1, |res| grid::coarsening(b, res, None))));
    }
    for rule in &table.auto {
        table.rows.iter().find(|r| r.id == rule.row).ok_or_else(|| format!("auto rule names unknown row {}", rule.row))?;
        let Some(res) = base(&rule.row) else {
            return Ok((rule.row.clone(), 1));
        };
        let (ny, nx) = Grid::dims(b, res);
        let cells = ny.checked_mul(nx).unwrap_or(usize::MAX);
        if rule.max_cells.is_none_or(|m| cells <= m) {
            return Ok((rule.row.clone(), grid::coarsening(b, res, None)));
        }
        if rule.coarsen {
            return Ok((rule.row.clone(), grid::coarsening(b, res, rule.max_cells)));
        }
    }
    Err("--row auto: no rule of the source table's auto list fits this box".into())
}

pub struct Built {
    pub fabric: Fabric,
    pub receipt: Receipt,
    pub terrain: BTreeMap<String, Value>,
    pub dams: Vec<Value>,
}

pub fn build(o: &Options) -> Result<Built> {
    let mut rc = Receipt::default();
    let table_bytes = std::fs::read(&o.sources)?;
    let table = source::load(o.sources.to_str().ok_or("sources path")?)?;
    let (row_id, coarsen) = resolve(&table, &o.row, &o.bbox)?;
    let row = table.rows.iter().find(|r| r.id == row_id).ok_or_else(|| format!("no source row {row_id}"))?;
    source::admit_with(row, &o.accept)?;
    let spec = row.build.clone().ok_or_else(|| format!("source row {} has no build spec (its recipe lives in its owner's tool)", row.id))?;
    rc.count("grid_coarsening", coarsen);
    let data_dir = o.sources.parent().unwrap_or(Path::new("."));
    let recipes = load_recipes(&data_dir.join(&spec.recipes))?;
    rc.count("source_row", &row.id);
    if o.row == "auto" {
        rc.count("source_row_selected_by", "auto rules of the source table");
    }
    rc.count("bbox", o.bbox);
    let hydro = match spec.method.as_str() {
        "dem-d8" => dem_hydro(row, &spec, &o.bbox, &o.data_root, coarsen, o.device, o.parent_grid.as_ref(), &mut rc)?,
        "vector-basins" => crate::vector::vector_hydro(row, &spec, &o.bbox, &o.data_root, coarsen, &mut rc)?,
        m => return Err(format!("build method {m} is not dem-d8 or vector-basins").into()),
    };
    let params = match &o.params {
        Some(p) => Some(serde_json::from_slice::<BTreeMap<String, Value>>(&std::fs::read(p)?)?),
        None => None,
    };
    let sha = {
        use sha2::Digest;
        format!("{:x}", sha2::Sha256::digest(&table_bytes))
    };
    let (fabric, terrain) = rc.time("assemble contract tables", &threads(), || assemble(row, &spec, &recipes, &o.bbox, &hydro, &o.revision, &sha, params.as_ref()))?;
    Ok(Built { fabric, receipt: rc, terrain, dams: hydro.dams })
}

pub fn load_tiles(row: &SourceRow, b: &BBox, root: &Path) -> Result<(HashMap<(i32, i32), tiff::Raster>, Vec<String>)> {
    load_tile_list(row, root, &b.tiles())
}

/// Decode the named 1-degree DEM tiles on all threads; tiles listed in the
/// tile directory's ABSENT.txt (open sea) come back in the second list.
pub fn load_tile_list(row: &SourceRow, root: &Path, tiles: &[(i32, i32)]) -> Result<(HashMap<(i32, i32), tiff::Raster>, Vec<String>)> {
    let dem = row.inputs.get("dem").ok_or("row has no dem input")?;
    let pat = dem.local.as_deref().ok_or("dem input has no local tile pattern")?;
    let absent_file = root.join(Path::new(pat).parent().unwrap_or(Path::new(""))).join("ABSENT.txt");
    let absent: HashSet<String> = std::fs::read_to_string(&absent_file).map(|s| s.lines().map(|l| l.trim().to_string()).collect()).unwrap_or_default();
    let decoded: Vec<Result<Option<((i32, i32), tiff::Raster)>>> = tiles
        .par_iter()
        .map(|&(la, lo)| {
            let rel = source::tile_path(pat, la, lo);
            let p = root.join(&rel);
            if p.exists() {
                Ok(Some(((la, lo), tiff::read(&p)?)))
            } else if absent.contains(Path::new(&rel).file_name().and_then(|f| f.to_str()).unwrap_or("")) {
                Ok(None)
            } else {
                Err(format!("DEM tile {} is not present and not listed in {} (fetch it with tools/hydro_world/hydro_world_build.py)", p.display(), absent_file.display()).into())
            }
        })
        .collect();
    let mut m = HashMap::new();
    let mut missing = Vec::new();
    for (t, d) in tiles.iter().zip(decoded) {
        match d? {
            Some((k, r)) => {
                m.insert(k, r);
            }
            None => missing.push(source::tile_path(pat, t.0, t.1)),
        }
    }
    Ok((m, missing))
}

pub struct LakeRecord {
    pub hylak: i64,
    pub area_km2: f64,
    pub depth_m: f64,
    pub wshd_km2: f64,
    pub lon: f64,
    pub lat: f64,
}

pub struct DamRecord {
    pub id: i64,
    pub name: String,
    pub hylak: i64,
    pub height_m: f64,
    pub length_m: f64,
    pub capacity_mcm: f64,
    pub year: i64,
    pub main_use: String,
    pub lake_control: String,
    pub catchment_km2: f64,
    pub lon: f64,
    pub lat: f64,
}

/// Dam barriers inside the box from the row's `dams` input (GDW v1.0
/// barriers: the column map names id, name, lake link, height, length,
/// capacity, year, main use, lake control, catchment area and the dam point).
pub fn load_dams(row: &SourceRow, b: &BBox, root: &Path) -> Result<Vec<DamRecord>> {
    let Some(inp) = row.inputs.get("dams") else { return Ok(Vec::new()) };
    let p = root.join(inp.local.as_deref().ok_or("dams input has no local path")?);
    let col = |meaning: &str| -> Result<String> {
        inp.columns.iter().find(|(_, v)| v.split(' ').next() == Some(meaning)).map(|(k, _)| k.clone()).ok_or_else(|| format!("dams input maps no column to '{meaning}'").into())
    };
    let dbf = shp::Dbf::open(&p.with_extension("dbf"))?;
    let num = |m: &str| -> Result<Vec<f64>> { dbf.f64s(&col(m)?) };
    let (id, hylak, year) = (dbf.i64s(&col("id")?)?, dbf.i64s(&col("lake_id")?)?, dbf.i64s(&col("year")?)?);
    let (hgt, len, cap, catch, lon, lat) = (num("height_m")?, num("length_m")?, num("capacity_mcm")?, num("catchment_km2")?, num("lon")?, num("lat")?);
    let (name, usev, ctrl) = (dbf.strings(&col("name")?)?, dbf.strings(&col("main_use")?)?, dbf.strings(&col("lake_control")?)?);
    Ok((0..dbf.n)
        .filter(|&i| b.contains(lon[i], lat[i]))
        .map(|i| DamRecord { id: id[i], name: name[i].trim().to_string(), hylak: hylak[i], height_m: hgt[i], length_m: len[i], capacity_mcm: cap[i], year: year[i], main_use: usev[i].trim().to_string(), lake_control: ctrl[i].trim().to_string(), catchment_km2: catch[i], lon: lon[i], lat: lat[i] })
        .collect())
}

pub fn load_lakes(row: &SourceRow, spec: &BuildSpec, b: &BBox, root: &Path) -> Result<Vec<LakeRecord>> {
    let inp = row.inputs.get("lakes").ok_or("row has no lakes input")?;
    let p = root.join(inp.local.as_deref().ok_or("lakes input has no local path")?);
    let col = |meaning: &str| -> Result<String> {
        inp.columns.iter().find(|(_, v)| v.split(' ').next() == Some(meaning)).map(|(k, _)| k.clone()).ok_or_else(|| format!("lakes input maps no column to '{meaning}'").into())
    };
    let dbf = shp::Dbf::open(&p.with_extension("dbf"))?;
    let id = dbf.i64s(&col("id")?)?;
    let area = dbf.f64s(&col("area_km2")?)?;
    let depth = dbf.f64s(&col("mean_depth_m")?)?;
    let wshd = dbf.f64s(&col("catchment_km2")?)?;
    let lon = dbf.f64s(&col("outlet_lon")?)?;
    let lat = dbf.f64s(&col("outlet_lat")?)?;
    Ok((0..dbf.n)
        .into_par_iter()
        .filter(|&i| area[i] >= spec.lake_min_area_km2 && b.contains(lon[i], lat[i]))
        .map(|i| LakeRecord { hylak: id[i], area_km2: area[i], depth_m: depth[i], wshd_km2: wshd[i], lon: lon[i], lat: lat[i] })
        .collect())
}

/// Stage 1 and the sea mask: the working grid, the DEM on it, and the sea.
/// The DEM on the working grid: the row's own grid (`coarsen` 1, tiles
/// mosaicked exactly) or that grid coarsened by `coarsen`, each cell the
/// block mean of the row's own grid (grid::aggregate). The coarsened path
/// streams 1-degree tile rows north to south, so host memory holds the
/// tiles of at most the tile rows one band of coarse rows touches.
pub fn dem_on_grid(row: &SourceRow, b: &BBox, root: &Path, res: f64, coarsen: u32, sea_value: Option<f32>, rc: &mut Receipt) -> Result<(Grid, Vec<f32>, Vec<bool>)> {
    let g = Grid::for_box(b, res * coarsen as f64)?;
    rc.count("grid", g);
    if coarsen <= 1 {
        let (tiles, absent) = rc.time("1 read DEM tiles", &threads(), || load_tiles(row, b, root))?;
        rc.count("dem_tiles_read", tiles.len());
        rc.count("dem_tiles_absent_sea", absent);
        let (z, missing) = rc.time("1 mosaic DEM onto grid", &threads(), || grid::mosaic(&g, &tiles))?;
        return Ok((g, z, missing));
    }
    rc.count("grid_coarsening_note", format!("working grid {}\" = {coarsen} x the row's {res}\" grid; each cell is the block mean of the {coarsen} x {coarsen} row-grid cells it holds", res * coarsen as f64));
    let box_tiles = b.tiles();
    let (la_min, la_max) = box_tiles.iter().fold((i32::MAX, i32::MIN), |(a, z), t| (a.min(t.0), z.max(t.0)));
    let lat_tile = |lat: f64| ((lat - 1e-9).floor() as i32).clamp(la_min, la_max);
    let span = |r: usize| {
        let v: Vec<i32> = grid::block_lats(&g, coarsen, res, r).map(lat_tile).collect();
        (*v.iter().min().unwrap(), *v.iter().max().unwrap())
    };
    let mut z = vec![0f32; g.len()];
    let mut missing = vec![false; g.len()];
    let (mut read, mut absent_all, mut peak_tiles) = (0usize, Vec::new(), 0usize);
    let mut tiles: HashMap<(i32, i32), tiff::Raster> = HashMap::new();
    let mut loaded_rows: HashSet<i32> = HashSet::new();
    rc.time("1 read DEM tiles + block-mean onto the coarsened grid (tile rows streamed north to south)", &threads(), || -> Result<()> {
        let mut r0 = 0;
        while r0 < g.ny {
            // One band: the coarse rows whose northernmost tile row is the same.
            let (_, top) = span(r0);
            let mut r1 = r0 + 1;
            let mut bottom = span(r0).0;
            while r1 < g.ny && span(r1).1 == top {
                bottom = bottom.min(span(r1).0);
                r1 += 1;
            }
            tiles.retain(|k, _| k.0 <= top);
            loaded_rows.retain(|&la| la <= top);
            let want: Vec<(i32, i32)> = box_tiles.iter().copied().filter(|t| t.0 >= bottom && t.0 <= top && !loaded_rows.contains(&t.0)).collect();
            if !want.is_empty() {
                let (m, absent) = load_tile_list(row, root, &want)?;
                read += m.len();
                absent_all.extend(absent);
                tiles.extend(m);
                loaded_rows.extend(want.iter().map(|t| t.0));
            }
            peak_tiles = peak_tiles.max(tiles.len());
            grid::aggregate(&g, coarsen, res, &tiles, b, sea_value, r0..r1, &mut z[r0 * g.nx..r1 * g.nx], &mut missing[r0 * g.nx..r1 * g.nx])?;
            r0 = r1;
        }
        Ok(())
    })?;
    rc.count("dem_tiles_read", read);
    rc.count("dem_tiles_absent_sea", absent_all);
    rc.count("dem_tiles_resident_peak", peak_tiles);
    Ok((g, z, missing))
}

pub fn prepare_dem(row: &SourceRow, spec: &BuildSpec, b: &BBox, root: &Path, coarsen: u32, rc: &mut Receipt) -> Result<(Grid, Vec<f32>, Vec<bool>)> {
    let res = spec.grid_arcsec.ok_or("dem-d8 needs grid_arcsec")?;
    let (g, z, missing) = dem_on_grid(row, b, root, res, coarsen, spec.sea_value, rc)?;
    let sea = rc.time("2 sea mask", "1 thread (breadth-first flood from sea seeds; seeds found on all threads)", || flow::sea_mask(&g, &z, &missing, spec.sea_value));
    drop(missing);
    rc.count("sea_cells", sea.par_iter().filter(|s| **s).count());
    Ok((g, z, sea))
}

/// The per-cell passes on all CPU cores (flow.rs).
pub fn cpu_passes(g: &Grid, z: &[f32], sea: &[bool], spec: &BuildSpec, rc: &mut Receipt) -> Result<(Vec<f32>, Vec<u8>, Vec<u64>)> {
    let (filled, fstats) = rc.time("2 depression fill (tiled Priority-Flood)", &threads(), || flow::fill(g, z, sea, spec.fill_tile.unwrap_or(512)))?;
    rc.count("fill", &fstats);
    let (dir, flats) = rc.time("2 D8 directions + flat resolution", &format!("{} (flat labelling 1 thread, linear)", threads()), || flow::directions(g, &filled, sea))?;
    rc.count("flats", &flats);
    let (acc, astats) = rc.time("2 flow accumulation", &threads(), || flow::accumulate(g, &dir))?;
    rc.count("accumulation", &astats);
    Ok((filled, dir, acc))
}

/// The per-cell passes on the GPU; Err when CUDA cannot start or a pass fails.
#[cfg(feature = "cuda")]
pub fn gpu_passes(g: &Grid, z: &[f32], sea: &[bool], keep_filled: bool, rc: &mut Receipt) -> Result<crate::gpu::Passes> {
    let t = Instant::now();
    let gpu = crate::gpu::Gpu::new()?;
    let par = format!("GPU ({}, CUDA)", gpu.device);
    rc.stages.push(StageTime { stage: "2 CUDA start (context, NVRTC compile, module load)".into(), ms: t.elapsed().as_secs_f64() * 1e3, parallelism: format!("{par}; NVRTC compile 1 thread, {:.0} ms", gpu.compile_ms) });
    let p = gpu.passes(g, z, sea, keep_filled)?;
    for (name, ms) in &p.stages {
        rc.stages.push(StageTime { stage: format!("2 {name}"), ms: *ms, parallelism: par.clone() });
    }
    rc.count("device", json!({"used": "cuda", "name": gpu.device}));
    rc.count("fill", &p.fill);
    rc.count("flats", &p.flats);
    rc.count("flat_relaxation_sweeps", &p.flat_sweeps);
    rc.count("accumulation", &p.accum);
    Ok(p)
}

/// Directions and accumulated areas on the chosen device.
pub fn per_cell_passes(g: &Grid, z: &[f32], sea: &[bool], spec: &BuildSpec, device: Device, rc: &mut Receipt) -> Result<(Vec<u8>, Vec<u64>)> {
    #[cfg(feature = "cuda")]
    if device != Device::Cpu {
        match gpu_passes(g, z, sea, false, rc) {
            Ok(p) => return Ok((p.dir, p.acc)),
            Err(e) if device == Device::Auto => {
                eprintln!("rw_hydro_world: CUDA path not used ({e}); running fill, D8 and accumulation on all {} CPU threads (identical results)", rayon::current_num_threads());
                rc.count("device", json!({"used": "cpu", "cuda_not_used": e.to_string()}));
            }
            Err(e) => return Err(e),
        }
    }
    #[cfg(not(feature = "cuda"))]
    match device {
        Device::Cuda => return Err("--device cuda: this build has no CUDA support (cargo feature cuda)".into()),
        _ => rc.count("device", json!({"used": "cpu", "cuda_not_used": "built without the cuda feature"})),
    }
    if device == Device::Cpu {
        rc.count("device", json!({"used": "cpu", "requested": "cpu"}));
    }
    let (_filled, dir, acc) = cpu_passes(g, z, sea, spec, rc)?;
    Ok((dir, acc))
}

/// `hydrography-compare`: the per-cell passes on all CPU cores and on the GPU
/// for the same box; every filled-surface float, direction and accumulated
/// area must be equal.
#[cfg(feature = "cuda")]
pub fn compare_passes(o: &Options) -> Result<Value> {
    let table = source::load(o.sources.to_str().ok_or("sources path")?)?;
    let (row_id, coarsen) = resolve(&table, &o.row, &o.bbox)?;
    let row = table.rows.iter().find(|r| r.id == row_id).ok_or_else(|| format!("no source row {row_id}"))?;
    source::admit_with(row, &o.accept)?;
    let spec = row.build.clone().ok_or("row has no build spec")?;
    let mut prep = Receipt::default();
    let (g, z, sea) = prepare_dem(row, &spec, &o.bbox, &o.data_root, coarsen, &mut prep)?;
    let mut cpu = Receipt::default();
    let tc = Instant::now();
    let (cf, cd, ca) = cpu_passes(&g, &z, &sea, &spec, &mut cpu)?;
    let cpu_ms = tc.elapsed().as_secs_f64() * 1e3;
    let mut gpu = Receipt::default();
    let tg = Instant::now();
    let p = gpu_passes(&g, &z, &sea, true, &mut gpu)?;
    let gpu_ms = tg.elapsed().as_secs_f64() * 1e3;
    let gf = p.filled.as_ref().ok_or("GPU filled surface not returned")?;
    let diff_f = cf.par_iter().zip(gf.par_iter()).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
    let diff_d = cd.par_iter().zip(p.dir.par_iter()).filter(|(a, b)| a != b).count();
    let diff_a = ca.par_iter().zip(p.acc.par_iter()).filter(|(a, b)| a != b).count();
    let first = |v: Vec<usize>| v.first().map(|&i| json!({"cell": i, "row": i / g.nx, "col": i % g.nx}));
    let first_d = first((0..g.len()).into_par_iter().filter(|&i| cd[i] != p.dir[i]).min().into_iter().collect());
    let first_f = first((0..g.len()).into_par_iter().filter(|&i| cf[i].to_bits() != gf[i].to_bits()).min().into_iter().collect());
    let sha = |b: &[u8]| {
        use sha2::Digest;
        format!("{:x}", sha2::Sha256::digest(b))
    };
    let acc_bytes: Vec<u8> = ca.iter().flat_map(|x| x.to_le_bytes()).collect();
    Ok(json!({
        "schema": "arwen.hydro-world.hydrography-compare.v1",
        "source_row": row.id,
        "bbox": o.bbox,
        "grid": g,
        "cells": g.len(),
        "identical": diff_f == 0 && diff_d == 0 && diff_a == 0,
        "differing": {"filled_f32_bits": diff_f, "d8_directions": diff_d, "accumulated_area_units": diff_a},
        "first_difference": {"filled": first_f, "directions": first_d},
        "sha256": {"directions": sha(&cd), "accumulation_le_u64": sha(&acc_bytes)},
        "prepare": {"stages": prep.stages, "counts": prep.counts},
        "cpu": {"total_ms": cpu_ms, "threads": rayon::current_num_threads(), "stages": cpu.stages, "counts": cpu.counts},
        "gpu": {"total_ms": gpu_ms, "stages": gpu.stages, "counts": gpu.counts},
    }))
}

#[allow(clippy::too_many_arguments)]
fn dem_hydro(row: &SourceRow, spec: &BuildSpec, b: &BBox, root: &Path, coarsen: u32, device: Device, parent: Option<&crate::parent_grid::ParentGrid>, rc: &mut Receipt) -> Result<Hydro> {
    let thr_km2 = spec.stream_threshold_km2.ok_or("dem-d8 needs stream_threshold_km2")?;
    let (g, z, mut sea) = prepare_dem(row, spec, b, root, coarsen, rc)?;
    if let Some(parent) = parent {
        let excluded = rc.time("1 actual parent-grid domain mask", &threads(), || parent.mask(&g, &mut sea))?;
        rc.count("parent_grid", parent);
        rc.count("parent_grid_boundary_cells_excluded", excluded);
        rc.count("parent_grid_boundary_policy", "complete DEM cells inside projected atmosphere; open drainage boundary");
    }
    let (dir, acc) = per_cell_passes(&g, &z, &sea, spec, device, rc)?;
    let thr = (thr_km2 * 1e6 * flow::AREA_UNITS_PER_M2).round() as u64;
    let to_km2 = |u: u64| u as f64 / flow::AREA_UNITS_PER_M2 / 1e6;
    // Lakes: snap each HydroLAKES pour point to the stream cell nearby whose
    // upstream area best matches the lake's catchment area.
    let lakes = rc.time("7 read lakes", &threads(), || load_lakes(row, spec, b, root))?;
    rc.count("lakes_in_box", lakes.len());
    let radius = spec.lake_snap_radius.round() as isize;
    let snapped: Vec<Option<(usize, usize)>> = rc.time("7 snap lakes to streams", &threads(), || {
        lakes
            .par_iter()
            .enumerate()
            .map(|(li, l)| {
                let (r0, c0) = g.rowcol(l.lon, l.lat)?;
                let mut best: Option<(f64, isize, usize)> = None;
                for dr in -radius..=radius {
                    for dc in -radius..=radius {
                        let (r, c) = (r0 as isize + dr, c0 as isize + dc);
                        if r < 0 || c < 0 || r as usize >= g.ny || c as usize >= g.nx {
                            continue;
                        }
                        let i = r as usize * g.nx + c as usize;
                        if sea[i] || acc[i] < thr {
                            continue;
                        }
                        let a = to_km2(acc[i]);
                        let score = if l.wshd_km2.is_finite() && l.wshd_km2 > 0.0 {
                            let ratio = a / l.wshd_km2;
                            if ratio > spec.lake_snap_area_ratio || ratio < 1.0 / spec.lake_snap_area_ratio {
                                continue;
                            }
                            ratio.ln().abs()
                        } else {
                            -a
                        };
                        let d2 = dr * dr + dc * dc;
                        if best.is_none_or(|(s, bd, _)| score < s || (score == s && d2 < bd)) {
                            best = Some((score, d2, i));
                        }
                    }
                }
                best.map(|(_, _, i)| (li, i))
            })
            .collect()
    });
    let mut at_cell: HashMap<usize, usize> = HashMap::new();
    for (li, cell) in snapped.iter().flatten() {
        let keep = match at_cell.get(cell) {
            Some(&o) => lakes[*li].area_km2 > lakes[o].area_km2 || (lakes[*li].area_km2 == lakes[o].area_km2 && lakes[*li].hylak < lakes[o].hylak),
            None => true,
        };
        if keep {
            at_cell.insert(*cell, *li);
        }
    }
    rc.count("lakes_unsnapped", snapped.iter().filter(|s| s.is_none()).count());
    rc.count("lakes_sharing_outlet_dropped", snapped.iter().flatten().count() - at_cell.len());
    let mut brk = vec![false; g.len()];
    for &c in at_cell.keys() {
        brk[c] = true;
    }
    let reaches = rc.time("3 stream cells and reaches", &threads(), || flow::reaches(&g, &dir, &acc, thr, &brk));
    let reaches = match spec.min_reach_length_km {
        Some(km) if km > 0.0 => {
            let (r, st) = rc.time("4 refactor: merge short reaches downstream", "1 thread (upstream-to-downstream merge order); lengths and relabelling on all threads", || flow::collapse_short(&g, &dir, reaches, km * 1000.0, &brk));
            rc.count("refactor_short_reaches", &st);
            r
        }
        _ => reaches,
    };
    let (labels, rounds) = rc.time("3 catchments (pointer jumping)", &threads(), || flow::catchments(&g, &dir, &reaches.cell_reach));
    rc.count("pointer_jumping_rounds", rounds);
    let nr = reaches.cells.len();
    rc.count("reaches", nr);
    let polys = rc.time("3 polygonize catchments", &threads(), || polygonize::polygonize(&g, &labels, nr));
    // Zonal statistics per catchment (area, terrain), deterministic chunk order.
    let zonal = rc.time("6 zonal terrain statistics", &threads(), || zonal(&g, &z, &labels, nr));
    let unlabeled_land: u64 = (0..g.ny)
        .into_par_iter()
        .map(|r| {
            let a = (g.cell_area_m2(r) * flow::AREA_UNITS_PER_M2).round() as u64;
            (0..g.nx).filter(|&c| labels[r * g.nx + c] == 0 && !sea[r * g.nx + c]).count() as u64 * a
        })
        .sum();
    let land: u64 = (0..g.ny).into_par_iter().map(|r| (0..g.nx).filter(|&c| !sea[r * g.nx + c]).count() as u64 * (g.cell_area_m2(r) * flow::AREA_UNITS_PER_M2).round() as u64).sum();
    rc.count("land_km2", to_km2(land));
    rc.count("land_not_in_any_catchment_km2", to_km2(unlabeled_land));
    let sliver_cells: u64 = polys.iter().flatten().map(|p| p.dropped_cells).sum();
    rc.count("divide_polygons_multipart", polys.iter().flatten().filter(|p| p.parts > 1).count());
    rc.count("divide_sliver_cells_outside_kept_part", sliver_cells);
    let half = g.res() / 2.0;
    let center = |i: usize| (g.lon(i % g.nx), g.lat(i / g.nx));
    let global_index = |i: usize| -> f64 {
        let row_g = ((90.0 * 3600.0 - (g.top_as - (i / g.nx) as f64 * g.res_as)) / g.res_as).round();
        let col_g = ((g.left_as + (i % g.nx) as f64 * g.res_as + 180.0 * 3600.0) / g.res_as).round();
        row_g * (360.0 * 3600.0 / g.res_as).round() + col_g
    };
    let out: Vec<ReachIn> = rc.time("4 reach geometry, length, slope", &threads(), || {
        (0..nr)
            .into_par_iter()
            .map(|k| {
                let cells = &reaches.cells[k];
                let last = *cells.last().unwrap() as usize;
                let mut line: Vec<(f64, f64)> = cells.iter().map(|&c| center(c as usize)).collect();
                let mut zs: Vec<f64> = cells.iter().map(|&c| z[c as usize] as f64).collect();
                match flow::receiver(&g, &dir, last).filter(|_| reaches.down[k].is_some()) {
                    Some(d) => {
                        line.push(center(d));
                        zs.push(z[d] as f64);
                    }
                    None => {
                        // Terminal: extend half a cell towards the sea neighbour
                        // or the nearest box edge, so the reach has its own length.
                        let (r, c) = ((last / g.nx) as isize, (last % g.nx) as isize);
                        let mut step = None;
                        for kk in 0..8 {
                            if let Some(n) = flow::neighbour(&g, last, kk) {
                                if sea[n] {
                                    step = Some((flow::DC[kk] as f64, -(flow::DR[kk] as f64)));
                                    break;
                                }
                            }
                        }
                        let step = step.unwrap_or_else(|| {
                            let cand = [(r, (0.0, 1.0)), (g.ny as isize - 1 - r, (0.0, -1.0)), (c, (-1.0, 0.0)), (g.nx as isize - 1 - c, (1.0, 0.0))];
                            cand.iter().min_by_key(|x| x.0).unwrap().1
                        });
                        let (lo, la) = center(last);
                        line.push((lo + step.0 * half, la + step.1 * half));
                        zs.push(*zs.last().unwrap());
                    }
                }
                let mut dist = vec![0.0f64; line.len()];
                for j in 1..line.len() {
                    dist[j] = dist[j - 1] + geo::geodesic_m(line[j - 1].0, line[j - 1].1, line[j].0, line[j].1);
                }
                // Bed slope: least-squares fit of DEM elevation along the reach.
                let n = dist.len() as f64;
                let (mx, my) = (dist.iter().sum::<f64>() / n, zs.iter().sum::<f64>() / n);
                let (mut sxy, mut sxx) = (0.0, 0.0);
                for j in 0..dist.len() {
                    sxy += (dist[j] - mx) * (zs[j] - my);
                    sxx += (dist[j] - mx).powi(2);
                }
                let slope = if sxx > 0.0 { -sxy / sxx } else { 0.0 };
                let divide = polys[k].as_ref().map(|p| {
                    let ring = |r: &Vec<(u32, u32)>| r.iter().map(|&(ci, cj)| g.corner(ci as usize, cj as usize)).collect::<Vec<_>>();
                    (ring(&p.exterior), p.holes.iter().map(ring).collect())
                });
                let zt = &zonal[k];
                ReachIn {
                    down: reaches.down[k].map(|d| d as usize),
                    line,
                    divide,
                    local_area_km2: to_km2(zt.area_units),
                    slope,
                    hf_id: global_index(last),
                    terrain: zt.terrain,
                }
            })
            .collect()
    });
    let mut lake_out = Vec::new();
    let mut terminal_lakes = 0;
    let mut keys: Vec<_> = at_cell.iter().collect();
    keys.sort();
    for (&cell, &li) in keys {
        let k = reaches.cell_reach[cell] as usize - 1;
        if reaches.down[k].is_none() {
            terminal_lakes += 1;
            continue;
        }
        let l = &lakes[li];
        lake_out.push(LakeIn { hylak: l.hylak, area_km2: l.area_km2, mean_depth_m: l.depth_m, weir_e_m: z[cell] as f64, outlet: center(cell), reach: k });
    }
    rc.count("lakes_at_domain_outlet_dropped", terminal_lakes);
    rc.count("lakes_snapped_to_inner_reaches", lake_out.len());
    // Dams (GDW barriers): snap each dam point to the stream cell nearby whose
    // upstream area best matches the dam's catchment area (as lakes), name the
    // reach it sits on and, through the HydroLAKES id, the fabric lake it controls.
    let dams = rc.time("7 read dams", "1 thread (attribute-table scan, tens of ms)", || load_dams(row, b, root))?;
    let ns = row.id_namespace;
    let nid = |k: usize| ns * 10_000_000_000 + k as i64 + 1;
    let lake_nexus: HashMap<i64, String> = lake_out.iter().filter_map(|l| reaches.down[l.reach].map(|d| (l.hylak, format!("nex-{}", nid(d as usize))))).collect();
    let dam_rows: Vec<Value> = rc.time("7 snap dams to streams", &threads(), || {
        dams.par_iter()
            .map(|dm| {
                let snap = g.rowcol(dm.lon, dm.lat).and_then(|(r0, c0)| {
                    let mut best: Option<(f64, isize, usize)> = None;
                    for dr in -radius..=radius {
                        for dc in -radius..=radius {
                            let (r, c) = (r0 as isize + dr, c0 as isize + dc);
                            if r < 0 || c < 0 || r as usize >= g.ny || c as usize >= g.nx {
                                continue;
                            }
                            let i = r as usize * g.nx + c as usize;
                            if sea[i] || reaches.cell_reach[i] == 0 {
                                continue;
                            }
                            let a = to_km2(acc[i]);
                            let score = if dm.catchment_km2.is_finite() && dm.catchment_km2 > 0.0 {
                                let ratio = a / dm.catchment_km2;
                                if ratio > spec.lake_snap_area_ratio || ratio < 1.0 / spec.lake_snap_area_ratio {
                                    continue;
                                }
                                ratio.ln().abs()
                            } else {
                                0.0
                            };
                            let d2 = dr * dr + dc * dc;
                            if best.is_none_or(|(s, bd, _)| score < s || (score == s && d2 < bd)) {
                                best = Some((score, d2, i));
                            }
                        }
                    }
                    best.map(|(_, _, i)| i)
                });
                let reach = snap.map(|i| reaches.cell_reach[i] as usize - 1);
                json!({
                    "gdw_id": dm.id,
                    "name": dm.name,
                    "lon": dm.lon,
                    "lat": dm.lat,
                    "flowpath_id": reach.map(|k| format!("wb-{}", nid(k))),
                    "snapped_upstream_area_km2": snap.map(|i| to_km2(acc[i])),
                    "gdw_catchment_km2": dm.catchment_km2,
                    "hylak_id": (dm.hylak > 0).then_some(dm.hylak),
                    "fabric_lake_nexus": lake_nexus.get(&dm.hylak),
                    "dam_height_m": dm.height_m,
                    "dam_length_m": dm.length_m,
                    "capacity_mcm": dm.capacity_mcm,
                    "year": dm.year,
                    "main_use": dm.main_use,
                    "lake_control": dm.lake_control,
                })
            })
            .collect()
    });
    rc.count("dams_in_box", dam_rows.len());
    rc.count("dams_snapped_to_network", dam_rows.iter().filter(|d| !d["flowpath_id"].is_null()).count());
    rc.count("dams_on_fabric_lakes", dam_rows.iter().filter(|d| !d["fabric_lake_nexus"].is_null()).count());
    rc.count("fabric_lakes_with_a_gdw_dam", dam_rows.iter().filter_map(|d| d["fabric_lake_nexus"].as_str()).collect::<HashSet<_>>().len());
    Ok(Hydro { reaches: out, lakes: lake_out, dams: dam_rows })
}

pub struct Zonal {
    pub area_units: u64,
    pub terrain: Option<Terrain>,
}

/// Per-label area and terrain means (Horn 1981 slope and aspect on the raw
/// DEM), accumulated per row chunk and merged in chunk order.
fn zonal(g: &Grid, z: &[f32], labels: &[u32], nlabels: usize) -> Vec<Zonal> {
    #[derive(Default, Clone, Copy)]
    struct Acc {
        a: u64,
        w: f64,
        z: f64,
        s: f64,
        sa: f64,
        ca: f64,
        lat: f64,
        lon: f64,
    }
    let rows_per = 32;
    let chunks: Vec<HashMap<u32, Acc>> = (0..g.ny.div_ceil(rows_per))
        .into_par_iter()
        .map(|ch| {
            let mut m: HashMap<u32, Acc> = HashMap::new();
            for r in ch * rows_per..((ch + 1) * rows_per).min(g.ny) {
                let area = g.cell_area_m2(r);
                let au = (area * flow::AREA_UNITS_PER_M2).round() as u64;
                let (dx, dy) = g.spacing_m(r);
                let lat = g.lat(r);
                for c in 0..g.nx {
                    let i = r * g.nx + c;
                    let l = labels[i];
                    if l == 0 {
                        continue;
                    }
                    let zz = |rr: isize, cc: isize| -> f64 {
                        let rr = rr.clamp(0, g.ny as isize - 1) as usize;
                        let cc = cc.clamp(0, g.nx as isize - 1) as usize;
                        z[rr * g.nx + cc] as f64
                    };
                    let (ri, ci) = (r as isize, c as isize);
                    let (a1, a2, a3) = (zz(ri - 1, ci - 1), zz(ri - 1, ci), zz(ri - 1, ci + 1));
                    let (b1, b3) = (zz(ri, ci - 1), zz(ri, ci + 1));
                    let (c1, c2, c3) = (zz(ri + 1, ci - 1), zz(ri + 1, ci), zz(ri + 1, ci + 1));
                    let dzdx = ((a3 + 2.0 * b3 + c3) - (a1 + 2.0 * b1 + c1)) / (8.0 * dx);
                    let dzdn = ((a1 + 2.0 * a2 + a3) - (c1 + 2.0 * c2 + c3)) / (8.0 * dy);
                    let slope = (dzdx * dzdx + dzdn * dzdn).sqrt().atan().to_degrees();
                    let e = m.entry(l).or_default();
                    e.a += au;
                    e.w += area;
                    e.z += area * z[i] as f64;
                    e.s += area * slope;
                    if dzdx != 0.0 || dzdn != 0.0 {
                        // Aspect: direction the slope faces, clockwise from north.
                        let asp = (-dzdx).atan2(-dzdn);
                        e.sa += area * asp.sin();
                        e.ca += area * asp.cos();
                    }
                    e.lat += area * lat;
                    e.lon += area * g.lon(c);
                }
            }
            m
        })
        .collect();
    let mut tot = vec![Acc::default(); nlabels];
    for m in chunks {
        let mut ks: Vec<_> = m.into_iter().collect();
        ks.sort_by_key(|x| x.0);
        for (l, a) in ks {
            let t = &mut tot[l as usize - 1];
            t.a += a.a;
            t.w += a.w;
            t.z += a.z;
            t.s += a.s;
            t.sa += a.sa;
            t.ca += a.ca;
            t.lat += a.lat;
            t.lon += a.lon;
        }
    }
    tot.into_iter()
        .map(|t| Zonal {
            area_units: t.a,
            terrain: (t.w > 0.0).then(|| Terrain {
                elevation_mean_m: t.z / t.w,
                slope_mean_deg: t.s / t.w,
                aspect_mean_deg: if t.sa == 0.0 && t.ca == 0.0 { 0.0 } else { (t.sa.atan2(t.ca).to_degrees() + 360.0) % 360.0 },
                centroid_lat_deg: t.lat / t.w,
                centroid_lon_deg: t.lon / t.w,
            }),
        })
        .collect()
}

fn manning(ch: &Channel, order: f64) -> f64 {
    let mut n = ch.manning_by_min_order.first().map(|x| x.1).unwrap_or(0.05);
    for &(o, v) in &ch.manning_by_min_order {
        if order >= o {
            n = v;
        }
    }
    n
}

/// Contract tables from method-independent hydrography (stages 4, 5, 7, 8, 9).
pub fn assemble(row: &SourceRow, spec: &BuildSpec, rec: &Recipes, b: &BBox, h: &Hydro, revision: &str, table_sha: &str, params: Option<&BTreeMap<String, Value>>) -> Result<(Fabric, BTreeMap<String, Value>)> {
    let nr = h.reaches.len();
    let ns = row.id_namespace;
    let nid = |k: usize| ns * 10_000_000_000 + k as i64 + 1;
    let (lat0, lon0) = ((b.south + b.north) / 2.0, (b.west + b.east) / 2.0);
    let proj = geo::Laea::new(lat0, lon0);
    let srs_id = 100_000;
    let srs = Srs {
        srs_name: format!("ARWEN LAEA {} {}", geo::fmt_num(lat0), geo::fmt_num(lon0)),
        srs_id,
        organization: "ARWEN".into(),
        organization_coordsys_id: srs_id as i64,
        definition: proj.wkt2(&format!("ARWEN Lambert Azimuthal Equal Area lat0={} lon0={}", geo::fmt_num(lat0), geo::fmt_num(lon0))),
        description: Some("WGS84 Lambert Azimuthal Equal Area centred on the fabric domain (HYDRO-WORLD contract section 4)".into()),
    };
    let srs_i = srs_id as i32;
    // Topology: upstream lists, Kahn order (hydroseq), Strahler, totals, levels.
    let mut ups: Vec<Vec<usize>> = vec![Vec::new(); nr];
    for (k, r) in h.reaches.iter().enumerate() {
        if let Some(d) = r.down {
            ups[d].push(k);
        }
    }
    let mut indeg: Vec<usize> = ups.iter().map(|u| u.len()).collect();
    let mut queue: std::collections::VecDeque<usize> = (0..nr).filter(|&k| indeg[k] == 0).collect();
    let mut order = Vec::with_capacity(nr);
    while let Some(k) = queue.pop_front() {
        order.push(k);
        if let Some(d) = h.reaches[k].down {
            indeg[d] -= 1;
            if indeg[d] == 0 {
                queue.push_back(d);
            }
        }
    }
    if order.len() != nr {
        return Err(format!("reach network has a cycle ({} reaches never ordered)", nr - order.len()).into());
    }
    let mut hydroseq = vec![0i64; nr];
    let mut strahler = vec![1u32; nr];
    let mut tot = vec![0f64; nr];
    for (rank, &k) in order.iter().enumerate() {
        hydroseq[k] = rank as i64 + 1;
        let u = &ups[k];
        if !u.is_empty() {
            let m = u.iter().map(|&x| strahler[x]).max().unwrap();
            strahler[k] = if u.iter().filter(|&&x| strahler[x] == m).count() >= 2 { m + 1 } else { m };
        }
        tot[k] = h.reaches[k].local_area_km2 + u.iter().map(|&x| tot[x]).sum::<f64>();
    }
    // Main stems: from each outlet up the largest-drainage branch.
    let mut mainstem = vec![0i64; nr];
    for &k in order.iter().rev() {
        if h.reaches[k].down.is_none() {
            mainstem[k] = nid(k);
        }
        let u = &ups[k];
        if let Some(&big) = u.iter().max_by(|&&a, &&b| tot[a].total_cmp(&tot[b]).then(b.cmp(&a))) {
            for &x in u {
                mainstem[x] = if x == big { mainstem[k] } else { nid(x) };
            }
        }
    }
    let lake_at: HashMap<usize, &LakeIn> = h.lakes.iter().map(|l| (l.reach, l)).collect();
    let toid = |k: usize| match h.reaches[k].down {
        Some(d) => format!("nex-{}", nid(d)),
        None => format!("tnx-{}", nid(k)),
    };
    let proj_line = |pts: &[(f64, f64)]| pts.iter().map(|&(x, y)| proj.forward(x, y)).collect::<Vec<_>>();
    let per: Vec<(Flowpath, Divide, FlowpathAttributes, NetworkRow)> = (0..nr)
        .into_par_iter()
        .map(|k| {
            let r = &h.reaches[k];
            let n = nid(k);
            let len_m = geo::polyline_geodesic_m(&r.line);
            let lengthkm = len_m / 1000.0;
            let terminal = r.down.is_none();
            let fp = Flowpath {
                fid: 0,
                geom: Some(Geometry(geo::gpkg_linestring(srs_i, &proj_line(&r.line)))),
                id: Some(format!("wb-{n}")),
                toid: Some(toid(k)),
                mainstem: Some(mainstem[k]),
                order: Some(strahler[k] as f64),
                hydroseq: Some(hydroseq[k]),
                lengthkm: Some(lengthkm),
                areasqkm: Some(r.local_area_km2),
                tot_drainage_areasqkm: Some(tot[k]),
                has_divide: Some(r.divide.is_some()),
                divide_id: r.divide.as_ref().map(|_| format!("cat-{n}")),
            };
            let dv = Divide {
                fid: 0,
                geom: r.divide.as_ref().map(|(e, holes)| Geometry(geo::gpkg_polygon(srs_i, &proj_line(e), &holes.iter().map(|h| proj_line(h)).collect::<Vec<_>>()))),
                divide_id: Some(format!("cat-{n}")),
                toid: Some(toid(k)),
                kind: Some(if terminal { "terminal" } else { "network" }.into()),
                ds_id: None,
                areasqkm: Some(r.local_area_km2),
                id: Some(format!("wb-{n}")),
                lengthkm: Some(lengthkm),
                tot_drainage_areasqkm: Some(tot[k]),
                has_flowline: Some(true),
            };
            let ch = &rec.channel;
            let tw = ch.top_width_a * tot[k].max(1e-6).powf(ch.top_width_b);
            let nman = manning(ch, strahler[k] as f64);
            let fa = FlowpathAttributes {
                fid: 0,
                id: Some(format!("wb-{n}")),
                rl_gages: None,
                rl_nhd_waterbody_comid: lake_at.get(&k).map(|l| l.hylak.to_string()),
                qi: Some(ch.qi_m3s),
                mus_k: Some(ch.musk_s),
                mus_x: Some(ch.musx),
                n: Some(nman),
                so: Some(r.slope.max(ch.min_slope)),
                ch_slp: Some(ch.side_slope_a * tot[k].max(1e-6).powf(ch.side_slope_b)),
                btm_wdth: Some(ch.bottom_width_fraction * tw),
                time: Some(0.0),
                kchan: Some(ch.kchan_mm_h),
                n_cc: Some(ch.compound_manning_factor * nman),
                top_wdth_cc: Some(ch.compound_top_width_factor * tw),
                top_wdth: Some(tw),
                length_m: Some(len_m),
            };
            let nw = NetworkRow {
                fid: 0,
                id: Some(format!("wb-{n}")),
                toid: Some(toid(k)),
                divide_id: r.divide.as_ref().map(|_| format!("cat-{n}")),
                ds_id: None,
                mainstem: Some(mainstem[k]),
                hl_id: None,
                hydroseq: Some(hydroseq[k]),
                hl_uri: None,
                hf_source: Some(row.hf_source.clone()),
                hf_id: Some(r.hf_id),
                lengthkm: Some(lengthkm),
                areasqkm: Some(r.local_area_km2),
                tot_drainage_areasqkm: Some(tot[k]),
                kind: Some(if terminal { "terminal" } else { "nexus" }.into()),
                vpu: None,
            };
            (fp, dv, fa, nw)
        })
        .collect();
    let mut flowpaths = Vec::with_capacity(nr);
    let mut divides = Vec::with_capacity(nr);
    let mut attrs = Vec::with_capacity(nr);
    let mut network = Vec::with_capacity(2 * nr);
    for (fp, dv, fa, nw) in per {
        flowpaths.push(fp);
        if dv.geom.is_some() {
            divides.push(dv);
        }
        attrs.push(fa);
        network.push(nw);
    }
    // Lakes and their WBOut hydrolocations (stage 7), sorted by outlet nexus.
    let mut lakes_sorted: Vec<&LakeIn> = h.lakes.iter().collect();
    lakes_sorted.sort_by_key(|l| (nid(h.reaches[l.reach].down.unwrap()), l.hylak));
    let lp = &rec.level_pool;
    let mut lakes = Vec::new();
    let mut hls = Vec::new();
    let mut poi: HashMap<String, (String, String)> = HashMap::new();
    for (j, l) in lakes_sorted.iter().enumerate() {
        let d = h.reaches[l.reach].down.unwrap();
        let nex = format!("nex-{}", nid(d));
        if poi.contains_key(&nex) {
            continue;
        }
        let hl_id = (j + 1).to_string();
        let uri = format!("WBOut-{}", l.hylak);
        let depth = if l.mean_depth_m.is_finite() && l.mean_depth_m > 0.0 { l.mean_depth_m } else { lp.default_mean_depth_m } * lp.pool_depth_per_mean_depth;
        let lkmxe = l.weir_e_m + lp.max_above_weir_fraction * depth;
        let (x, y) = proj.forward(l.outlet.0, l.outlet.1);
        lakes.push(Lake {
            fid: 0,
            geom: Some(Geometry(geo::gpkg_point(srs_i, x, y))),
            id: Some(nex.clone()),
            toid: Some(format!("wb-{}", nid(d))),
            hl_id: Some(hl_id.clone()),
            hl_reference: Some("WBOut".into()),
            hl_link: Some(l.hylak as f64),
            hl_uri: Some(uri.clone()),
            dam_length: Some(lp.dam_length),
            ifd: Some(lp.ifd),
            lk_area: Some(l.area_km2),
            lk_mx_e: Some(lkmxe),
            orifice_a: Some(lp.orifice_a_m2),
            orifice_c: Some(lp.orifice_c),
            orifice_e: Some(lkmxe - depth),
            time: Some(0.0),
            weir_c: Some(lp.weir_c),
            weir_e: Some(l.weir_e_m),
            weir_l: Some(lp.weir_l_m),
        });
        hls.push(Hydrolocation {
            fid: 0,
            geom: Some(Geometry(geo::gpkg_point(srs_i, x, y))),
            hl_id: Some(hl_id.clone()),
            hl_reference: Some("WBOut".into()),
            hl_link: Some(l.hylak.to_string()),
            hl_uri: Some(uri.clone()),
            hl_position: Some("outflow".into()),
            id: Some(nex.clone()),
        });
        poi.insert(nex, (hl_id, uri));
    }
    // Nexus points: one per receiving reach (its upstream end), one terminal per outlet reach.
    let mut nexus = Vec::new();
    for k in 0..nr {
        let n = nid(k);
        if !ups[k].is_empty() {
            let id = format!("nex-{n}");
            let (x, y) = proj.forward(h.reaches[k].line[0].0, h.reaches[k].line[0].1);
            let p = poi.get(&id);
            nexus.push(Nexus {
                fid: 0,
                geom: Some(Geometry(geo::gpkg_point(srs_i, x, y))),
                id: Some(id.clone()),
                toid: Some(format!("wb-{n}")),
                hl_id: p.map(|p| p.0.clone()),
                hl_uri: Some(p.map(|p| p.1.clone()).unwrap_or_else(|| "NA".into())),
                kind: Some(if p.is_some() { "poi" } else { "nexus" }.into()),
            });
            network.push(NetworkRow { fid: 0, id: Some(id), toid: Some(format!("wb-{n}")), divide_id: None, ds_id: None, mainstem: None, hl_id: p.map(|p| p.0.clone()), hydroseq: None, hl_uri: p.map(|p| p.1.clone()), hf_source: None, hf_id: None, lengthkm: None, areasqkm: None, tot_drainage_areasqkm: None, kind: Some("nexus".into()), vpu: None });
        }
        if h.reaches[k].down.is_none() {
            let id = format!("tnx-{n}");
            let end = *h.reaches[k].line.last().unwrap();
            let (x, y) = proj.forward(end.0, end.1);
            nexus.push(Nexus { fid: 0, geom: Some(Geometry(geo::gpkg_point(srs_i, x, y))), id: Some(id.clone()), toid: Some("wb-0".into()), hl_id: None, hl_uri: Some("NA".into()), kind: Some("nexus".into()) });
            network.push(NetworkRow { fid: 0, id: Some(id), toid: Some("wb-0".into()), divide_id: None, ds_id: None, mainstem: None, hl_id: None, hydroseq: None, hl_uri: None, hf_source: None, hf_id: None, lengthkm: None, areasqkm: None, tot_drainage_areasqkm: None, kind: Some("terminal".into()), vpu: None });
        }
    }
    // Terrain sidecar and model_attributes (only when a parameter packet covers every divide).
    let mut terrain = BTreeMap::new();
    for (k, r) in h.reaches.iter().enumerate() {
        if let (Some(t), true) = (r.terrain, r.divide.is_some()) {
            terrain.insert(format!("cat-{}", nid(k)), json!({"area_km2": r.local_area_km2, "terrain": t}));
        }
    }
    let model_attributes = match params {
        None => None,
        Some(p) => {
            let mut rows = Vec::new();
            for d in &divides {
                let id = d.divide_id.clone().unwrap();
                let t = terrain.get(&id).and_then(|v| v.get("terrain")).ok_or_else(|| format!("{id}: no terrain statistics"))?;
                let mut obj = p.get(&id).and_then(|v| v.as_object()).cloned().ok_or_else(|| format!("parameter packet has no row for {id}"))?;
                for (kk, v) in t.as_object().unwrap() {
                    obj.insert(kk.clone(), v.clone());
                }
                obj.insert("fid".into(), json!(0));
                obj.insert("divide_id".into(), json!(id));
                rows.push(serde_json::from_value::<ModelAttributes>(Value::Object(obj)).map_err(|e| format!("parameter packet row {id}: {e}"))?);
            }
            Some(rows)
        }
    };
    let terms = row
        .inputs
        .iter()
        .filter_map(|(k, i)| i.licence.name.as_ref().map(|n| format!("{k}: {} {} ({n})", i.dataset, i.version)))
        .collect::<Vec<_>>()
        .join("; ");
    let utc = utc_now();
    let md = [
        ("contract", CONTRACT.to_string()),
        ("builder", format!("rw_hydro_world build ({})", spec.method)),
        ("builder_revision", revision.to_string()),
        ("source_row", row.id.clone()),
        ("source_table_sha256", table_sha.to_string()),
        ("data_terms", terms),
        ("data_usage", if row.inputs.values().any(|i| i.licence.status == "research-only") {
            "research-only; never bundled, fetched by default or shipped".to_string()
        } else { "source-specific terms apply".to_string() }),
        ("id_namespace", ns.to_string()),
        ("horizontal_crs", srs.definition.clone()),
        ("vertical_datum", row.vertical_datum.clone()),
        ("soil_class_table", spec.soil_class_table.clone()),
        ("veg_class_table", spec.veg_class_table.clone()),
        ("domain", b.wkt()),
        ("build_utc", utc.clone()),
    ];
    let fabric_metadata = Some(md.iter().map(|(k, v)| MetadataRow { fid: 0, key: Some(k.to_string()), value: Some(v.clone()) }).collect());
    let mut f = Fabric {
        contract: CONTRACT.into(),
        srs,
        contents: BTreeMap::new(),
        divides,
        flowpaths,
        nexus,
        network,
        flowpath_attributes: attrs,
        hydrolocations: hls,
        lakes,
        model_attributes,
        fabric_metadata,
    };
    derive::canonicalize(&mut f);
    // gpkg_contents: identifier = table name, extent of feature tables in the fabric CRS.
    let env = |geoms: Vec<&Option<Geometry>>| -> Option<[f64; 4]> {
        let mut e = [f64::MAX, f64::MAX, f64::MIN, f64::MIN];
        let mut any = false;
        for g in geoms.into_iter().flatten() {
            for &(x, y) in geo::gpkg_rings(&g.0).into_iter().flatten().flatten().collect::<Vec<_>>().iter() {
                e = [e[0].min(x), e[1].min(y), e[2].max(x), e[3].max(y)];
                any = true;
            }
        }
        any.then_some(e)
    };
    let mut contents = BTreeMap::new();
    let mut put = |name: &str, bbox: Option<[f64; 4]>| {
        contents.insert(name.to_string(), Contents { identifier: Some(name.to_string()), description: Some(String::new()), last_change: md[12].1.clone(), bbox });
    };
    put("divides", env(f.divides.iter().map(|x| &x.geom).collect()));
    put("flowpaths", env(f.flowpaths.iter().map(|x| &x.geom).collect()));
    put("nexus", env(f.nexus.iter().map(|x| &x.geom).collect()));
    put("hydrolocations", env(f.hydrolocations.iter().map(|x| &x.geom).collect()));
    put("lakes", env(f.lakes.iter().map(|x| &x.geom).collect()));
    for t in ["network", "flowpath_attributes", "fabric_metadata"] {
        put(t, None);
    }
    if f.model_attributes.is_some() {
        put("model_attributes", None);
    }
    f.contents = contents;
    if let Some(m) = f.fabric_metadata.as_mut() {
        for (i, r) in m.iter_mut().enumerate() {
            r.fid = i as i64 + 1;
        }
    }
    Ok((f, terrain))
}

/// Topological levels of a reach network (engine GPU ordering depth).
pub fn level_count(f: &Fabric) -> usize {
    let idx: HashMap<&str, usize> = f.flowpaths.iter().enumerate().map(|(i, x)| (x.id.as_deref().unwrap_or(""), i)).collect();
    let nex: HashMap<&str, &str> = f.nexus.iter().map(|x| (x.id.as_deref().unwrap_or(""), x.toid.as_deref().unwrap_or(""))).collect();
    let down: Vec<Option<usize>> = f.flowpaths.iter().map(|x| nex.get(x.toid.as_deref().unwrap_or("")).and_then(|t| idx.get(t).copied())).collect();
    let mut indeg = vec![0usize; down.len()];
    for d in down.iter().flatten() {
        indeg[*d] += 1;
    }
    let mut lvl = vec![0usize; down.len()];
    let mut q: Vec<usize> = (0..down.len()).filter(|&i| indeg[i] == 0).collect();
    let mut maxl = 0;
    while let Some(i) = q.pop() {
        maxl = maxl.max(lvl[i]);
        if let Some(d) = down[i] {
            lvl[d] = lvl[d].max(lvl[i] + 1);
            indeg[d] -= 1;
            if indeg[d] == 0 {
                q.push(d);
            }
        }
    }
    maxl + 1
}

pub fn utc_now() -> String {
    let s = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0) as i64;
    let (days, rem) = (s.div_euclid(86_400), s.rem_euclid(86_400));
    // Civil from days (Hinnant).
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    format!("{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}Z", rem / 3600, (rem % 3600) / 60, rem % 60)
}
