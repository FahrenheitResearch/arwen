//! Gridded quantities share a target grid, complete family rosters and native
//! observation readers. Reflectivity never passes through a dBZ averaging remap.
use super::*;
use flate2::read::GzDecoder;
use grib_core::grib2::{grid_latlon, unpack_message, Grib2File};
use netcdf_writer::{AttrValue, NcFormat, NcType, NcWriter, Schema, VarData};
use rayon::prelude::*;
use rw_nexrad::grid_composite::{haversine_m, parse_when, Frames, Lattice};
use sha2::{Digest, Sha256};
use std::{collections::HashMap, io::Read, time::Instant};

type NativeResult<T> = std::result::Result<T, String>;
const MAX_DISTANCE_M: f64 = 5000.0;
const THRESHOLDS: [f64; 3] = [20.0, 35.0, 45.0];
#[derive(Deserialize)]
pub struct GridMatch {
    quantity: String,
    valid_time: String,
    target_grid: PathBuf,
    observation: PathBuf,
    /// Explicit identities independent of the files that happen to exist.
    required_families: BTreeMap<String, Vec<String>>,
    members: Vec<GridMember>,
    boundary_rows: usize,
    interior_rim_m: f64,
    max_distance_m: f64,
    /// Omit for family scores only. Empty object selects fixed 0.5 / 0.5.
    #[serde(default)]
    blend: Option<BTreeMap<String, f64>>,
    /// Optional cut of the target grid, [y0, y1, x0, x1) in target cells, so a
    /// large lattice (the 1024 x 1792 nowcast crop of HRRR) is scored over one region.
    #[serde(default)]
    target_window: Option<[usize; 4]>,
}
#[derive(Deserialize)]
struct GridMember {
    id: String,
    family: String,
    format: String,
    file: PathBuf,
    #[serde(default)]
    sample: Option<usize>,
}
struct Plane {
    values: Vec<f64>,
    latitude: Vec<f64>,
    longitude: Vec<f64>,
    receipt: Value,
}
fn pin(path: &Path) -> NativeResult<Value> {
    let mut file = std::fs::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut hash = Sha256::new();
    let mut buffer = vec![0; 1024 * 1024];
    loop {
        let n = file.read(&mut buffer).map_err(|e| e.to_string())?;
        if n == 0 {
            break;
        }
        hash.update(&buffer[..n]);
    }
    Ok(json!({"path":path,"sha256":format!("{:x}",hash.finalize()),
        "bytes":file.metadata().map_err(|e| e.to_string())?.len()}))
}
fn covered(v: f64) -> bool {
    v.is_finite() && v > -900.0
}
fn column_max(volume: &[f64], cells: usize) -> NativeResult<Vec<f64>> {
    if cells == 0 || volume.len() % cells != 0 || volume.is_empty() {
        return Err("REFL_10CM must have full vertical mass-grid planes".into());
    }
    // A missing level is not evidence of a clear column. Preserve the hole.
    Ok((0..cells)
        .into_par_iter()
        .map(|i| {
            let mut maximum = f64::NEG_INFINITY;
            for level in volume.chunks_exact(cells) {
                if !covered(level[i]) || level[i].abs() > 1e10 {
                    return f64::NAN;
                }
                maximum = maximum.max(level[i]);
            }
            maximum
        })
        .collect())
}
fn wrf(path: &Path, expected: &str) -> NativeResult<Plane> {
    let file = WrfFile::open(path).map_err(|e| e.to_string())?;
    valid_time(&file, expected).map_err(|e| e.to_string())?;
    let values = file.read_var("REFL_10CM", 0).map_err(|e| e.to_string())?;
    if values.len() != file.nxy() * file.nz {
        return Err("REFL_10CM vertical shape differs".into());
    }
    Ok(Plane {
        values: column_max(&values, file.nxy())?,
        latitude: field(&file, "XLAT").map_err(|e| e.to_string())?,
        longitude: field(&file, "XLONG").map_err(|e| e.to_string())?,
        receipt: pin(path)?,
    })
}
/// The scoring grid.  A 3 km WRF history (its own valid time must match), or a
/// nowcast-frames root (its 3 km lattice, with a frame at the valid time), so an
/// AI family can be scored against MRMS when no physics history covers the case.
/// Either may be cut by `target_window`.
fn target_geometry(request: &GridMatch) -> NativeResult<(usize, usize, Vec<f64>, Vec<f64>, Value)> {
    let (ny, nx, lat, lon, receipt) = if request.target_grid.join("nowcast.json").is_file() {
        let frames = Frames::load(&request.target_grid)?;
        frames.frame_at(parse_when(&request.valid_time)?)?;
        let Lattice::Lambert(l) = &frames.lattice else {
            return Err("nowcast target grid must be a projected (Lambert) lattice".into());
        };
        if (l.dx.abs() - 3000.0).abs() > 1.0 || (l.dy.abs() - 3000.0).abs() > 1.0 {
            return Err("33 km FSS contract requires a 3 km target grid".into());
        }
        let (ny, nx) = frames.lattice.shape();
        let coordinates: Vec<_> = (0..ny * nx)
            .into_par_iter()
            .map(|i| frames.lattice.cell_latlon(i / nx, i % nx))
            .collect();
        (ny, nx, coordinates.iter().map(|v| v.0).collect(), coordinates.iter().map(|v| v.1).collect(),
         json!({"kind":"nowcast-frames lattice","metadata":pin(&request.target_grid.join("nowcast.json"))?}))
    } else {
        let target = WrfFile::open(&request.target_grid).map_err(|e| e.to_string())?;
        valid_time(&target, &request.valid_time).map_err(|e| e.to_string())?;
        if (target.dx - 3000.0).abs() > 1.0 || (target.dy - 3000.0).abs() > 1.0 {
            return Err("33 km FSS contract requires a 3 km target grid".into());
        }
        (target.ny, target.nx, field(&target, "XLAT").map_err(|e| e.to_string())?,
         field(&target, "XLONG").map_err(|e| e.to_string())?, pin(&request.target_grid)?)
    };
    let Some([y0, y1, x0, x1]) = request.target_window else {
        return Ok((ny, nx, lat, lon, receipt));
    };
    if !(y0 < y1 && y1 <= ny && x0 < x1 && x1 <= nx) {
        return Err(format!("target_window {:?} is outside the {ny} x {nx} target grid", [y0, y1, x0, x1]));
    }
    let cut = |v: &[f64]| -> Vec<f64> { (y0..y1).flat_map(|j| v[j * nx + x0..j * nx + x1].to_vec()).collect() };
    Ok((y1 - y0, x1 - x0, cut(&lat), cut(&lon), json!({"grid":receipt,"window_y0_y1_x0_x1":[y0, y1, x0, x1]})))
}
fn nowcast(member: &GridMember, expected: &str) -> NativeResult<Plane> {
    let frames = Frames::load(&member.file)?;
    let document: Value = serde_json::from_slice(
        &std::fs::read(member.file.join("nowcast.json")).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    let end = parse_when(expected)?;
    if frames.kind != "nowcast"
        || !frames.causal
        || frames.issue_time > end
        || document["units"] != "dBZ"
        || parse_when(
            document["latest_input_time"]
                .as_str()
                .ok_or("nowcast has no latest input time")?,
        )? > frames.issue_time
    {
        return Err("nowcast must be causal refc in dBZ with inputs ending by issue".into());
    }
    let sample = member
        .sample
        .ok_or("nowcast member requires zero-based sample")?;
    if sample >= frames.members {
        return Err("required nowcast sample is missing".into());
    }
    let (_, entry) = frames.frame_at(end)?;
    let all = frames.read(entry)?;
    let (ny, nx) = frames.lattice.shape();
    let coordinates: Vec<_> = (0..ny * nx)
        .into_par_iter()
        .map(|i| frames.lattice.cell_latlon(i / nx, i % nx))
        .collect();
    let inputs = member.file.join(
        document["inputs"]["file"]
            .as_str()
            .ok_or("nowcast has no input ledger")?,
    );
    let input_pin = pin(&inputs)?;
    if input_pin["sha256"] != document["inputs"]["sha256"] {
        return Err("nowcast input ledger digest differs".into());
    }
    Ok(Plane {
        values: all[sample * ny * nx..(sample + 1) * ny * nx]
            .iter()
            .map(|&v| v as f64)
            .collect(),
        latitude: coordinates.iter().map(|v| v.0).collect(),
        longitude: coordinates.iter().map(|v| v.1).collect(),
        receipt: json!({"metadata":pin(&member.file.join("nowcast.json"))?,"frame":{"valid_time":entry.valid,"file":entry.file,"sha256":entry.sha256,"shape":entry.shape,"dtype":entry.dtype},
            "inputs":input_pin,"source":document["source"],"sample":sample,"seed":document["seed"],"sampler":document["sampler"]}),
    })
}
fn mrms(path: &Path, expected: &str) -> NativeResult<Plane> {
    let raw = std::fs::read(path).map_err(|e| e.to_string())?;
    let mut bytes = Vec::new();
    if raw.starts_with(&[0x1f, 0x8b]) {
        GzDecoder::new(&raw[..])
            .read_to_end(&mut bytes)
            .map_err(|e| e.to_string())?;
    } else {
        bytes = raw;
    }
    let grib = Grib2File::from_bytes(&bytes).map_err(|e| e.to_string())?;
    if grib.messages.len() != 1 {
        return Err("MRMS requires exactly one composite message".into());
    }
    let m = &grib.messages[0];
    // MRMS composite local parameter, 500 m MSL product identifier. No QPE or base substitution.
    if (
        m.discipline,
        m.product.parameter_category,
        m.product.parameter_number,
        m.product.level_type,
    ) != (209, 10, 0, 102)
        || m.product.level_value != 500.0
        || m.grid.template != 0
        || m.grid.scan_mode & 0x30 != 0
    {
        return Err("observation is not MergedReflectivityQCComposite_00.50 regular GRIB2".into());
    }
    let offset = (parse_when(expected)? - m.reference_time.and_utc()).num_seconds();
    if !(0..=120).contains(&offset) {
        return Err("MRMS time must be at or before valid time, within 120 s".into());
    }
    let values = unpack_message(m).map_err(|e| e.to_string())?;
    let (latitude, longitude) = grid_latlon(&m.grid).map_err(|e| e.to_string())?;
    let receipt = json!({"input":pin(path)?,"product":"MergedReflectivityQCComposite_00.50",
        "valid_time":m.reference_time.to_string(),"offset_seconds":offset,"source_shape":[m.grid.ny,m.grid.nx],
        "no_coverage_cells":values.iter().filter(|v|!covered(**v)).count(),
        "observed_clear_cells":values.iter().filter(|v|**v == -99.0).count(),
        "mask":"nonfinite or <= -900 missing; -99 observed clear air, unchanged"});
    Ok(Plane {
        values,
        latitude,
        longitude: longitude
            .iter()
            .map(|&v| rw_obs::seam::wrap_longitude(v))
            .collect(),
        receipt,
    })
}
fn xyz(lat: f64, lon: f64) -> [f64; 3] {
    let (p, l) = (lat.to_radians(), lon.to_radians());
    [
        6371000.0 * p.cos() * l.cos(),
        6371000.0 * p.cos() * l.sin(),
        6371000.0 * p.sin(),
    ]
}
fn bucket(p: [f64; 3]) -> (i32, i32, i32) {
    (
        (p[0] / MAX_DISTANCE_M).floor() as i32,
        (p[1] / MAX_DISTANCE_M).floor() as i32,
        (p[2] / MAX_DISTANCE_M).floor() as i32,
    )
}
/// Exact nearest covered centre in a 5 km spherical cap. Cartesian buckets
/// prune candidates only; the final choice uses great-circle metres and a
/// stable source-index tie rule. No arithmetic on dBZ values is performed.
fn nearest(plane: &Plane, lat: &[f64], lon: &[f64]) -> NativeResult<(Vec<f64>, Value)> {
    if plane.values.len() != plane.latitude.len() || plane.values.len() != plane.longitude.len() {
        return Err("source values and coordinate lengths differ".into());
    }
    let targets: Vec<_> = lat.iter().zip(lon).map(|(&a, &b)| xyz(a, b)).collect();
    let mut low = [f64::INFINITY; 3];
    let mut high = [f64::NEG_INFINITY; 3];
    for p in &targets {
        for k in 0..3 {
            low[k] = low[k].min(p[k] - MAX_DISTANCE_M);
            high[k] = high[k].max(p[k] + MAX_DISTANCE_M);
        }
    }
    let mut bins: HashMap<(i32, i32, i32), Vec<usize>> = HashMap::new();
    for (i, &value) in plane.values.iter().enumerate() {
        if !covered(value) {
            continue;
        }
        let p = xyz(plane.latitude[i], plane.longitude[i]);
        if (0..3).all(|k| p[k] >= low[k] && p[k] <= high[k]) {
            bins.entry(bucket(p)).or_default().push(i);
        }
    }
    let mapped: Vec<_> = targets
        .par_iter()
        .enumerate()
        .map(|(i, &p)| {
            let (a, b, c) = bucket(p);
            let mut best = (MAX_DISTANCE_M + 1.0, usize::MAX);
            for x in a - 1..=a + 1 {
                for y in b - 1..=b + 1 {
                    for z in c - 1..=c + 1 {
                        if let Some(indices) = bins.get(&(x, y, z)) {
                            for &s in indices {
                                let d = haversine_m(
                                    lat[i],
                                    lon[i],
                                    plane.latitude[s],
                                    plane.longitude[s],
                                    6371000.0,
                                );
                                if d <= MAX_DISTANCE_M && (d, s) < best {
                                    best = (d, s);
                                }
                            }
                        }
                    }
                }
            }
            if best.1 == usize::MAX {
                (f64::NAN, 0.0, -1i64)
            } else {
                (plane.values[best.1], best.0, best.1 as i64)
            }
        })
        .collect();
    let indices: Vec<u8> = mapped.iter().flat_map(|v| v.2.to_le_bytes()).collect();
    Ok((
        mapped.iter().map(|v| v.0).collect(),
        json!({"method":"nearest-valid-centre","max_distance_m":MAX_DISTANCE_M,
        "max_used_distance_m":mapped.iter().map(|v|v.1).fold(0.0,f64::max),
        "source_index_sha256":hex_sha256(&indices),"unreachable_cells":mapped.iter().filter(|v|v.2 < 0).count()}),
    ))
}
fn ratio(a: f64, b: f64) -> Option<f64> {
    (b > 0.0).then_some(a / b)
}
fn probability(columns: &[&Vec<f64>], threshold: f64, mask: &[u8]) -> Vec<f64> {
    (0..mask.len())
        .into_par_iter()
        .map(|i| {
            if mask[i] == 0 {
                f64::NAN
            } else {
                columns.iter().filter(|c| c[i] >= threshold).count() as f64 / columns.len() as f64
            }
        })
        .collect()
}
fn threshold_score(
    p: &[f64],
    obs: &[f64],
    mask: &[u8],
    ny: usize,
    nx: usize,
    threshold: f64,
) -> NativeResult<Value> {
    let mut h = 0u64;
    let mut m = 0u64;
    let mut f = 0u64;
    let mut c = 0u64;
    let mut bs = 0.0;
    let mut oevent = vec![0.0; obs.len()];
    for i in 0..obs.len() {
        if mask[i] == 0 {
            continue;
        }
        let o = obs[i] >= threshold;
        let e = p[i] >= 0.5;
        oevent[i] = u8::from(o) as f64;
        match (o, e) {
            (true, true) => h += 1,
            (true, false) => m += 1,
            (false, true) => f += 1,
            (false, false) => c += 1,
        }
        bs += (p[i] - oevent[i]).powi(2);
    }
    let n = h + m + f + c;
    if n == 0 {
        return Err("no common covered interior cells".into());
    }
    let fss = obs_score::masked_probability_fss(p, &oevent, mask, ny, nx, 5)?;
    Ok(
        json!({"threshold":threshold,"fss":fss,"fss_definition":"masked neighbourhood ensemble probability versus observed binary event; 11x11 zero boundary",
        "brier_score":bs/n as f64,"samples":n,"observed_events":h+m,"forecast_events":h+f,
        "expected_forecast_events":p.iter().zip(mask).filter(|(_,m)|**m!=0).map(|(v,_)|*v).sum::<f64>(),
        "hits":h,"misses":m,"false_alarms":f,"correct_negatives":c,
        "csi":ratio(h as f64,(h+m+f) as f64),"pod":ratio(h as f64,(h+m) as f64),
        "far":ratio(f as f64,(h+f) as f64),"frequency_bias":ratio((h+f) as f64,(h+m) as f64),
        "contingency_probability_cutoff":0.5}),
    )
}
fn validate_roster(request: &GridMatch) -> NativeResult<()> {
    if request.required_families.is_empty() || request.members.is_empty() {
        return Err("required family roster is empty".into());
    }
    let mut actual: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    let mut ids = BTreeSet::new();
    let mut sources = BTreeSet::new();
    for m in &request.members {
        if m.id.is_empty() || !ids.insert(&m.id) || m.id.contains(['\t', '\n', '\r']) {
            return Err("member ids must be distinct labels".into());
        }
        if !["physics", "ai"].contains(&m.family.as_str()) {
            return Err("family must be physics or ai".into());
        }
        if !matches!(
            (m.family.as_str(), m.format.as_str()),
            ("physics", "wrf-history") | ("ai", "nowcast-frames")
        ) {
            return Err("physics requires WRF history; ai requires nowcast frames".into());
        }
        if !sources.insert((&m.file, m.sample)) {
            return Err("duplicate member file/sample would change probability mass".into());
        }
        actual
            .entry(m.family.clone())
            .or_default()
            .insert(m.id.clone());
    }
    let mut expected = BTreeMap::new();
    for (family, roster) in &request.required_families {
        let set: BTreeSet<_> = roster.iter().cloned().collect();
        if roster.is_empty() || set.len() != roster.len() {
            return Err("required roster is empty or duplicated".into());
        }
        expected.insert(family.clone(), set);
    }
    if actual != expected {
        return Err(
            "missing required member or unexpected family member; scored frame fails".into(),
        );
    }
    Ok(())
}
fn blend_weights(request: &GridMatch) -> NativeResult<Option<BTreeMap<String, f64>>> {
    let Some(declared) = &request.blend else {
        return Ok(None);
    };
    let weights = if declared.is_empty() {
        BTreeMap::from([("physics".into(), 0.5), ("ai".into(), 0.5)])
    } else {
        declared.clone()
    };
    if weights.keys().collect::<Vec<_>>() != request.required_families.keys().collect::<Vec<_>>()
        || weights.values().any(|v| !v.is_finite() || *v < 0.0)
        || (weights.values().sum::<f64>() - 1.0).abs() > 1e-12
    {
        return Err("experimental blend requires all declared families and nonnegative fixed weights summing to one".into());
    }
    Ok(Some(weights))
}
fn product(
    path: &Path,
    lat: &[f64],
    lon: &[f64],
    ny: usize,
    nx: usize,
    valid: &str,
    columns: &[&Vec<f64>],
    probabilities: &[Vec<f64>],
    mask: &[u8],
) -> NativeResult<()> {
    let mut s = Schema::new(NcFormat::Cdf5);
    let y = s
        .def_dim("south_north", ny, false)
        .map_err(|e| e.to_string())?;
    let x = s
        .def_dim("west_east", nx, false)
        .map_err(|e| e.to_string())?;
    let t = s
        .def_dim("refl_threshold", 3, false)
        .map_err(|e| e.to_string())?;
    let m = s
        .def_dim("refl_member", columns.len(), false)
        .map_err(|e| e.to_string())?;
    for (k, v) in [
        ("ensemble_contract", "gpuwm-resident-ensemble-products.v1"),
        ("probability_scale", "fraction"),
        ("valid_time", valid),
    ] {
        s.put_global_attr(k, AttrValue::Text(v.into()))
            .map_err(|e| e.to_string())?;
    }
    s.put_global_attr(
        "ensemble_members",
        AttrValue::Ints(vec![columns.len() as i32]),
    )
    .map_err(|e| e.to_string())?;
    let mut arrays: Vec<(usize, Vec<f32>)> = Vec::new();
    let mut add = |name: &str, dims: &[usize], units: &str, values: Vec<f32>| -> NativeResult<()> {
        let v = s
            .def_var(name, NcType::Float, dims)
            .map_err(|e| e.to_string())?;
        s.put_var_attr(v, "units", AttrValue::Text(units.into()))
            .map_err(|e| e.to_string())?;
        if name == "refl_probability" {
            s.put_var_attr(v, "comparison", AttrValue::Text("ge".into()))
                .map_err(|e| e.to_string())?;
        }
        arrays.push((v, values));
        Ok(())
    };
    add(
        "XLAT",
        &[y, x],
        "degrees_north",
        lat.iter().map(|v| *v as f32).collect(),
    )?;
    add(
        "XLONG",
        &[y, x],
        "degrees_east",
        lon.iter().map(|v| *v as f32).collect(),
    )?;
    // Arithmetic ensemble dBZ mean is a renderer metadata requirement. Only
    // the observed singleton mean is drawn; scores never threshold a mean.
    add(
        "refl_mean",
        &[y, x],
        "dBZ",
        (0..mask.len())
            .map(|i| {
                if mask[i] == 0 {
                    f32::NAN
                } else {
                    (columns.iter().map(|c| c[i]).sum::<f64>() / columns.len() as f64) as f32
                }
            })
            .collect(),
    )?;
    add(
        "refl_members",
        &[m, y, x],
        "dBZ",
        columns
            .iter()
            .flat_map(|c| {
                c.iter()
                    .zip(mask)
                    .map(|(v, m)| if *m == 0 { f32::NAN } else { *v as f32 })
            })
            .collect(),
    )?;
    add(
        "refl_thresholds",
        &[t],
        "dBZ",
        THRESHOLDS.iter().map(|v| *v as f32).collect(),
    )?;
    add(
        "refl_probability",
        &[t, y, x],
        "1",
        probabilities.iter().flatten().map(|v| *v as f32).collect(),
    )?;
    add(
        "refl_finite_count",
        &[y, x],
        "1",
        mask.iter()
            .map(|v| if *v == 0 { 0.0 } else { columns.len() as f32 })
            .collect(),
    )?;
    let mut w = NcWriter::create(path, s).map_err(|e| e.to_string())?;
    for (id, data) in arrays {
        w.write_var(id, VarData::F32(&data))
            .map_err(|e| e.to_string())?;
    }
    w.finish().map_err(|e| e.to_string())?;
    Ok(())
}
pub fn matching(request: &GridMatch, out: &Path) -> Result<Value> {
    matching_native(request, out).map_err(err)
}
fn matching_native(request: &GridMatch, out: &Path) -> NativeResult<Value> {
    let began = Instant::now();
    if request.quantity != "composite_reflectivity"
        || request.max_distance_m != MAX_DISTANCE_M
        || !request.interior_rim_m.is_finite()
        || request.interior_rim_m < 0.0
    {
        return Err(
            "gridded quantity requires composite_reflectivity and explicit max_distance_m=5000"
                .into(),
        );
    }
    validate_roster(request)?;
    let weights = blend_weights(request)?;
    let (ny, nx, lat, lon, target_receipt) = target_geometry(request)?;
    let geometry = geometry_hash(&lat, &lon);
    let rim = request.boundary_rows + (request.interior_rim_m / 3000.0).ceil() as usize;
    if 2 * rim >= ny || 2 * rim >= nx {
        return Err("interior rim removes the target grid".into());
    }
    let obs_start = Instant::now();
    let obs = mrms(&request.observation, &request.valid_time)?;
    let (observed, obs_remap) = nearest(&obs, &lat, &lon)?;
    let obs_seconds = obs_start.elapsed().as_secs_f64();
    let observation_receipt = obs.receipt;
    drop(obs.values);
    drop(obs.latitude);
    drop(obs.longitude);
    let member_start = Instant::now();
    let loaded: NativeResult<Vec<_>>=request.members.par_iter().map(|m| {
        let stage=Instant::now();
        let plane=match m.format.as_str() {"wrf-history"=>wrf(&m.file,&request.valid_time)?,"nowcast-frames"=>nowcast(m,&request.valid_time)?,_=>return Err("unknown gridded member format".into())};
        let (values,remap)=if geometry_hash(&plane.latitude,&plane.longitude)==geometry {
            (plane.values,json!({"method":"identical coordinate centres","max_distance_m":MAX_DISTANCE_M,"max_used_distance_m":0.0}))
        } else { nearest(&plane,&lat,&lon)? };
        if values.iter().any(|v|covered(*v) && (*v < -100.0 || *v>100.0)) { return Err("reflectivity outside [-100,100] dBZ seam".into()); }
        Ok((values,json!({"id":m.id,"family":m.family,"input":plane.receipt,"remap":remap,"wall_seconds":stage.elapsed().as_secs_f64()})))
    }).collect();
    let loaded = loaded?;
    let member_seconds = member_start.elapsed().as_secs_f64();
    // Shared support is fixed across every required member in every family.
    // Holes mask cells, never shrink the member denominator.
    let mask: Vec<u8> = (0..ny * nx)
        .into_par_iter()
        .map(|i| {
            u8::from(
                i / nx >= rim
                    && i / nx < ny - rim
                    && i % nx >= rim
                    && i % nx < nx - rim
                    && covered(observed[i])
                    && loaded.iter().all(|c| covered(c.0[i])),
            )
        })
        .collect();
    if !mask.contains(&1) {
        return Err("no common covered interior cells".into());
    }
    let stage = Instant::now();
    std::fs::create_dir_all(out).map_err(|e| e.to_string())?;
    let mut families = BTreeMap::new();
    let mut family_probs = BTreeMap::new();
    let mut products = Vec::new();
    for (family, roster) in &request.required_families {
        let indices: Vec<_> = request
            .members
            .iter()
            .enumerate()
            .filter_map(|(i, m)| (m.family == *family).then_some(i))
            .collect();
        let columns: Vec<_> = indices.iter().map(|i| &loaded[*i].0).collect();
        let probabilities: Vec<_> = THRESHOLDS
            .par_iter()
            .map(|&t| probability(&columns, t, &mask))
            .collect();
        let scores: NativeResult<Vec<_>> = THRESHOLDS
            .par_iter()
            .enumerate()
            .map(|(k, &t)| threshold_score(&probabilities[k], &observed, &mask, ny, nx, t))
            .collect();
        let member_scores: NativeResult<Vec<_>> = indices
            .par_iter()
            .map(|&i| {
                let cols = vec![&loaded[i].0];
                let scores: NativeResult<Vec<_>> = THRESHOLDS
                    .iter()
                    .map(|&t| {
                        threshold_score(&probability(&cols, t, &mask), &observed, &mask, ny, nx, t)
                    })
                    .collect();
                Ok(json!({"id":request.members[i].id,"threshold_scores":scores?}))
            })
            .collect();
        let path = out.join(format!("{family}.nc"));
        product(
            &path,
            &lat,
            &lon,
            ny,
            nx,
            &request.valid_time,
            &columns,
            &probabilities,
            &mask,
        )?;
        products.push(json!({"family":family,"path":path,"sha256":pin(&path)?["sha256"],"products":["prob","postage"]}));
        families.insert(family.clone(),json!({"member_ids":roster,"members":columns.len(),"threshold_scores":scores?,"member_scores":member_scores?}));
        family_probs.insert(family.clone(), probabilities);
    }
    let blend = if let Some(weights) = weights {
        let probabilities: Vec<Vec<f64>> = (0..3)
            .map(|k| {
                (0..mask.len())
                    .into_par_iter()
                    .map(|i| {
                        if mask[i] == 0 {
                            f64::NAN
                        } else {
                            weights.iter().map(|(f, w)| family_probs[f][k][i] * w).sum()
                        }
                    })
                    .collect()
            })
            .collect();
        let scores: NativeResult<Vec<_>> = THRESHOLDS
            .par_iter()
            .enumerate()
            .map(|(k, &t)| threshold_score(&probabilities[k], &observed, &mask, ny, nx, t))
            .collect();
        Some(
            json!({"label":"experimental fixed-weight family probability blend","weights":weights,"threshold_scores":scores?}),
        )
    } else {
        None
    };
    let observed_columns = vec![&observed];
    let observed_prob: Vec<_> = THRESHOLDS
        .iter()
        .map(|&t| probability(&observed_columns, t, &mask))
        .collect();
    let path = out.join("observed.nc");
    product(
        &path,
        &lat,
        &lon,
        ny,
        nx,
        &request.valid_time,
        &observed_columns,
        &observed_prob,
        &mask,
    )?;
    products.push(
        json!({"family":"observed","path":path,"sha256":pin(&path)?["sha256"],"products":["mean"]}),
    );
    Ok(
        json!({"schema":"gpuwm-ensemble-grid-match.v1","status":"complete","quantity":request.quantity,"units":"dBZ","valid_time":request.valid_time,
        "grid":{"shape":[ny,nx],"geometry_sha256":geometry,"input":target_receipt,"boundary_rows":request.boundary_rows,"interior_rim_m":request.interior_rim_m},
        "observation":observation_receipt,"observation_remap":obs_remap,"mask_sha256":hex_sha256(&mask),"scored_cells":mask.iter().filter(|v|**v!=0).count(),
        "families":families,"blend":blend,"source_receipts":loaded.iter().map(|v|&v.1).collect::<Vec<_>>(),"products":products,
        "neighbourhood":{"cells":[11,11],"dx_m":3000,"length_m":33000,"boundary":"zero","validity":"common intersection of observation, required members, and declared interior"},
        "timing":{"observation_decode_remap":{"seconds":obs_seconds,"parallelism":format!("{} threads for remap; native GRIB decode",rayon::current_num_threads())},
            "member_read_hash_composite_remap":{"seconds":member_seconds,"parallelism":format!("{} threads",rayon::current_num_threads())},
            "scores_products":{"seconds":stage.elapsed().as_secs_f64(),"parallelism":format!("{} threads",rayon::current_num_threads())},
            "total_seconds":began.elapsed().as_secs_f64()}}),
    )
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn vertical_max_preserves_missing_and_clear_air() {
        let out = column_max(&[-99., 20., -999., 10., 45., 50.], 3).unwrap();
        assert_eq!(&out[..2], &[10., 45.]);
        assert!(out[2].is_nan());
    }
    #[test]
    fn nearest_valid_does_not_average_dbz_or_score_no_coverage() {
        let p = Plane {
            values: vec![-999., -99., 45.],
            latitude: vec![40.; 3],
            longitude: vec![-85., -84.99, -84.98],
            receipt: json!({}),
        };
        let (v, _) = nearest(&p, &[40., 40.], &[-85., -86.]).unwrap();
        assert_eq!(v[0], -99.);
        assert!(v[1].is_nan());
    }
    #[test]
    fn unequal_families_are_not_pooled_and_missing_roster_fails() {
        let raw = json!({"quantity":"composite_reflectivity","valid_time":"2000-01-01T00:00Z","target_grid":"x","observation":"o",
            "required_families":{"physics":["p1","p2"],"ai":["a1"]},"members":[
                {"id":"p1","family":"physics","format":"wrf-history","file":"p1"},
                {"id":"p2","family":"physics","format":"wrf-history","file":"p2"},
                {"id":"a1","family":"ai","format":"nowcast-frames","file":"a1","sample":0}],
            "boundary_rows":0,"interior_rim_m":0,"max_distance_m":5000,"blend":{}});
        let mut r: GridMatch = serde_json::from_value(raw).unwrap();
        validate_roster(&r).unwrap();
        let weights = blend_weights(&r).unwrap().unwrap();
        let physics = probability(&[&vec![0.], &vec![0.]], 35., &[1]);
        let ai = probability(&[&vec![50.]], 35., &[1]);
        assert_eq!(weights["physics"] * physics[0] + weights["ai"] * ai[0], 0.5);
        r.members.pop();
        assert!(validate_roster(&r).is_err());
    }
    #[test]
    fn analytical_events_brier_fss_and_zero_denominators() {
        let score =
            threshold_score(&[1., 0., 1., 0.], &[50., 50., 0., 0.], &[1; 4], 2, 2, 35.).unwrap();
        assert_eq!(score["hits"], 1);
        assert_eq!(score["misses"], 1);
        assert_eq!(score["false_alarms"], 1);
        assert_eq!(score["brier_score"], 0.5);
        assert_eq!(score["fss"], 1.0);
        let clear = threshold_score(&[0.; 4], &[-99.; 4], &[1; 4], 2, 2, 35.).unwrap();
        assert_eq!(clear["fss"], 1.);
        assert_eq!(clear["csi"], Value::Null);
    }
    #[test]
    fn nowcast_reads_pinned_refc_sample_and_refuses_missing_or_future_input() {
        let root = std::env::temp_dir().join(format!("grid-match-nowcast-{}", std::process::id()));
        std::fs::create_dir_all(&root).unwrap();
        let values: Vec<u8> = [-99f32, 35., 45., f32::NAN, 0., 20., 40., 50.]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        std::fs::write(root.join("refc.f32"), &values).unwrap();
        std::fs::write(root.join("inputs.json"), b"{\"objects\":[]}").unwrap();
        let mut doc = json!({"schema":"gpuwm-obs.nowcast-frames.v1","status":"READY","variable":"refc","units":"dBZ",
            "source":{"kind":"nowcast","id":"any-table-row","revision":"pinned"},"members":2,
            "issue_time":"2000-01-01T00:00Z","latest_input_time":"2000-01-01T00:00Z","causal":true,
            "lattice":{"kind":"latlon","lat0":40.,"lon0":-85.,"dlat":0.03,"dlon":0.03,"nx":2,"ny":2,"row_order":"south_to_north",
                "corners_latlon":{"j0_i0":[40.,-85.],"j0_in":[40.,-84.97],"jn_i0":[40.03,-85.],"jn_in":[40.03,-84.97]}},
            "frames":[{"valid":"2000-01-01T00:10Z","file":"refc.f32","shape":[2,2,2],"dtype":"float32-le","sha256":hex_sha256(&values)}],
            "inputs":{"file":"inputs.json","sha256":pin(&root.join("inputs.json")).unwrap()["sha256"]}});
        let save = |d: &Value| {
            std::fs::write(root.join("nowcast.json"), serde_json::to_vec(d).unwrap()).unwrap()
        };
        save(&doc);
        let mut member = GridMember {
            id: "a1".into(),
            family: "ai".into(),
            format: "nowcast-frames".into(),
            file: root.clone(),
            sample: Some(1),
        };
        assert_eq!(
            nowcast(&member, "2000-01-01T00:10Z").unwrap().values,
            vec![0., 20., 40., 50.]
        );
        member.sample = Some(0);
        assert!(nowcast(&member, "2000-01-01T00:10Z").unwrap().values[3].is_nan());
        member.sample = Some(2);
        assert!(nowcast(&member, "2000-01-01T00:10Z").is_err());
        member.sample = Some(0);
        doc["latest_input_time"] = json!("2000-01-01T00:01Z");
        save(&doc);
        assert!(nowcast(&member, "2000-01-01T00:10Z").is_err());
        // A nowcast root is accepted as the scoring grid only on a projected 3 km lattice.
        let request: GridMatch = serde_json::from_value(json!({"quantity":"composite_reflectivity",
            "valid_time":"2000-01-01T00:10Z","target_grid":root,"observation":"o",
            "required_families":{"ai":["a1"]},"members":[{"id":"a1","family":"ai","format":"nowcast-frames","file":root,"sample":0}],
            "boundary_rows":0,"interior_rim_m":0.0,"max_distance_m":5000.0,"target_window":[0,1,0,1]})).unwrap();
        let refused = target_geometry(&request).err().unwrap();
        assert!(refused.contains("Lambert"), "{refused}");
    }
}
