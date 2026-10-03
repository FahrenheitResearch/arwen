//! Native forward-operator qualification and a bounded CPU benchmark.
//! The generated atmosphere is analytic test data, not an observed case.
use std::collections::BTreeMap;
use std::error::Error;
use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

use bowecho_simradar::geo::aeqd_inverse_km;
use bowecho_simradar::radar_core::{MomentType, beam_height_above_radar_m};
use bowecho_simradar::wrf_scene_inventory::WrfSourceIdentity;
use bowecho_simradar::{
    BeamIntegration, ModelRadarFields, SyntheticRadarComputePreference, SyntheticRadarConfig,
    WrfFile, WrfRadarFields, read_wrf_radar_fields_for_config, try_build_synthetic_volume,
};
use chrono::{DateTime, Utc};
use netcdf_writer::{AttrValue, NcFormat, NcType, NcWriter, Schema, VarData};
use serde_json::json;

type Result<T> = std::result::Result<T, Box<dyn Error>>;
const LAT: f64 = 35.0;
const LON: f64 = -98.0;
const DX_M: f64 = 3000.0;

fn mass_height(k: usize) -> f32 {
    if k == 0 {
        50.0
    } else {
        350.0 + 500.0 * (k - 1) as f32
    }
}
fn face_height(k: usize) -> f32 {
    if k == 0 {
        0.0
    } else {
        100.0 + 500.0 * (k - 1) as f32
    }
}
fn horizontal_dbz(x_km: f64, y_km: f64, shift_km: f64) -> f32 {
    let a = ((x_km - 24.0 - shift_km).powi(2) / 1800.0 + (y_km - 30.0).powi(2) / 3200.0)
        .exp()
        .recip();
    let b = ((x_km + 55.0 - shift_km).powi(2) / 3200.0 + (y_km + 20.0).powi(2) / 1800.0)
        .exp()
        .recip();
    (5.0 + 40.0 * a.max(0.8 * b)) as f32
}
fn xyz_dbz(base: f32, height: f32) -> f32 {
    base - 0.0008 * (height - 2000.0).max(0.0)
}
fn model(nx: usize, ny: usize, nz: usize, shift_km: f64) -> ModelRadarFields {
    let horizontal = nx * ny;
    let cells = horizontal * nz;
    let mut lat = Vec::with_capacity(horizontal);
    let mut lon = Vec::with_capacity(horizontal);
    let mut base = Vec::with_capacity(horizontal);
    for j in 0..ny {
        let y = (j as f64 - (ny - 1) as f64 * 0.5) * DX_M * 0.001;
        for i in 0..nx {
            let x = (i as f64 - (nx - 1) as f64 * 0.5) * DX_M * 0.001;
            let (la, lo) = aeqd_inverse_km(LAT, LON, x, y);
            lat.push(la as f32);
            lon.push(lo as f32);
            base.push(horizontal_dbz(x, y, shift_km));
        }
    }
    let mut height = Vec::with_capacity(cells);
    let mut dbz = Vec::with_capacity(cells);
    for k in 0..nz {
        let z = mass_height(k);
        height.extend(std::iter::repeat_n(z, horizontal));
        dbz.extend(base.iter().map(|&v| xyz_dbz(v, z)));
    }
    ModelRadarFields {
        nx,
        ny,
        nz,
        latitude_deg: lat,
        longitude_deg: lon,
        height_msl_m: height,
        reflectivity_dbz: dbz,
        eastward_wind_mps: vec![12.0; cells],
        northward_wind_mps: vec![-4.0; cells],
        upward_wind_mps: vec![0.0; cells],
        terrain_msl_m: vec![0.0; horizontal],
        grid_spacing_m: Some(DX_M),
        model_label: "WOOF analytic qualification".to_owned(),
    }
}
fn config(id: &str, x_km: f64, y_km: f64) -> SyntheticRadarConfig {
    let (lat, lon) = aeqd_inverse_km(LAT, LON, x_km, y_km);
    let mut c = SyntheticRadarConfig::default();
    c.site_id = id.to_owned();
    c.site_name = Some("WOOF analytic qualification".to_owned());
    c.site_lat_deg = Some(lat);
    c.site_lon_deg = Some(lon);
    c.antenna_msl_m = Some(100.0);
    c.compute_preference = SyntheticRadarComputePreference::Cpu;
    c.beam_integration = BeamIntegration::Balanced;
    c.ref_gate_texture = false;
    c.vel_gate_texture = false;
    c.dual_pol = false;
    c.instrument_noise = false;
    c.terminal_fall_speed = false;
    c.spectrum_width = false;
    c.propagation = false;
    c
}
fn write_history(path: &Path, input: &ModelRadarFields, minute: usize) -> Result<()> {
    let (nx, ny, nz) = (input.nx, input.ny, input.nz);
    let cells = nx * ny;
    let time_label = format!("2026-10-02_00:{minute:02}:00");
    let mut s = Schema::new(NcFormat::Offset64);
    let t = s.def_dim("Time", 0, true)?;
    let sl = s.def_dim("DateStrLen", 19, false)?;
    let x = s.def_dim("west_east", nx, false)?;
    let y = s.def_dim("south_north", ny, false)?;
    let z = s.def_dim("bottom_top", nz, false)?;
    let xs = s.def_dim("west_east_stag", nx + 1, false)?;
    let ys = s.def_dim("south_north_stag", ny + 1, false)?;
    let zs = s.def_dim("bottom_top_stag", nz + 1, false)?;
    for (name, value) in [
        ("TITLE", "WOOF analytic radar test atmosphere"),
        ("START_DATE", "2026-10-02_00:00:00"),
        ("SIMULATION_START_DATE", "2026-10-02_00:00:00"),
        ("GRIDTYPE", "C"),
    ] {
        s.put_global_attr(name, AttrValue::Text(value.to_owned()))?;
    }
    for (name, value) in [
        ("MAP_PROJ", 6),
        ("GRID_ID", 1),
        ("PARENT_ID", 0),
        ("MP_PHYSICS", 0),
    ] {
        s.put_global_attr(name, AttrValue::Ints(vec![value]))?;
    }
    for (name, value) in [
        ("DX", DX_M as f32),
        ("DY", DX_M as f32),
        ("CEN_LAT", LAT as f32),
        ("CEN_LON", LON as f32),
        ("TRUELAT1", LAT as f32),
        ("TRUELAT2", LAT as f32),
        ("STAND_LON", LON as f32),
        ("POLE_LAT", 90.0),
        ("POLE_LON", 0.0),
    ] {
        s.put_global_attr(name, AttrValue::Floats(vec![value]))?;
    }
    let mut ids = BTreeMap::new();
    ids.insert("Times", s.def_var("Times", NcType::Char, &[t, sl])?);
    for name in [
        "XLAT", "XLONG", "HGT", "REFC", "SINALPHA", "COSALPHA", "T2", "PSFC", "U10", "V10",
    ] {
        ids.insert(name, s.def_var(name, NcType::Float, &[t, y, x])?);
    }
    for name in ["T", "P", "PB", "REFL_10CM", "QVAPOR"] {
        ids.insert(name, s.def_var(name, NcType::Float, &[t, z, y, x])?);
    }
    ids.insert("U", s.def_var("U", NcType::Float, &[t, z, y, xs])?);
    ids.insert("V", s.def_var("V", NcType::Float, &[t, z, ys, x])?);
    for name in ["W", "PH", "PHB"] {
        ids.insert(name, s.def_var(name, NcType::Float, &[t, zs, y, x])?);
    }
    for name in ["REFC", "REFL_10CM"] {
        s.put_var_attr(ids[name], "units", AttrValue::Text("dBZ".to_owned()))?;
    }
    let mut w = NcWriter::create(path, s)?;
    w.write_record(0, ids["Times"], VarData::Char(time_label.as_bytes()))?;
    w.write_record(0, ids["XLAT"], VarData::F32(&input.latitude_deg))?;
    w.write_record(0, ids["XLONG"], VarData::F32(&input.longitude_deg))?;
    w.write_record(0, ids["HGT"], VarData::F32(&input.terrain_msl_m))?;
    w.write_record(
        0,
        ids["REFC"],
        VarData::F32(&input.reflectivity_dbz[..cells]),
    )?;
    for (name, value) in [
        ("SINALPHA", 0.0),
        ("COSALPHA", 1.0),
        ("T2", 295.0),
        ("PSFC", 100_000.0),
        ("U10", 12.0),
        ("V10", -4.0),
    ] {
        w.write_record(0, ids[name], VarData::F32(&vec![value; cells]))?;
    }
    for (name, value) in [("T", 0.0), ("P", 0.0), ("PB", 80_000.0), ("QVAPOR", 0.01)] {
        w.write_record(0, ids[name], VarData::F32(&vec![value; cells * nz]))?;
    }
    w.write_record(0, ids["REFL_10CM"], VarData::F32(&input.reflectivity_dbz))?;
    w.write_record(0, ids["U"], VarData::F32(&vec![12.0; (nx + 1) * ny * nz]))?;
    w.write_record(0, ids["V"], VarData::F32(&vec![-4.0; nx * (ny + 1) * nz]))?;
    w.write_record(
        0,
        ids["W"],
        VarData::F32(&vec![input.upward_wind_mps[0]; cells * (nz + 1)]),
    )?;
    w.write_record(0, ids["PH"], VarData::F32(&vec![0.0; cells * (nz + 1)]))?;
    let phb: Vec<_> = (0..=nz)
        .flat_map(|k| std::iter::repeat_n(face_height(k) * 9.80665, cells))
        .collect();
    w.write_record(0, ids["PHB"], VarData::F32(&phb))?;
    w.finish()?;
    Ok(())
}
// Independent quadrature reference on the analytic rectilinear grid. The
// production sampler locates cells in latitude/longitude; this reference
// locates them directly in the known x/y coordinates and reads stored REFC.
fn pulse_reference(refc: &[f64], nx: usize, ny: usize, azimuth: f64, range: f64) -> (f64, f64) {
    let sigma = 0.95 / (2.0 * (2.0 * std::f64::consts::LN_2).sqrt());
    let mut points = vec![(0.0, 0.0, 0.0, 4.0)];
    for az in [-1.0, 1.0] {
        for el in [-1.0, 1.0] {
            for r in [-0.35, 0.35] {
                points.push((az, el, r, 1.0));
            }
        }
    }
    let (mut power, mut column_power, mut weight) = (0.0, 0.0, 0.0);
    for (da, de, dr, w) in points {
        let az = (azimuth + da * sigma).to_radians();
        let el = (0.5 + de * sigma).to_radians();
        let r = range + dr * 250.0;
        let radius = 6_371_000.0 * (4.0 / 3.0);
        let h = (r * r + radius * radius + 2.0 * r * radius * el.sin()).sqrt() - radius;
        let ground = radius * (r * el.cos() / (radius + h)).asin();
        let x = ground * az.sin() / DX_M + (nx - 1) as f64 * 0.5;
        let y = ground * az.cos() / DX_M + (ny - 1) as f64 * 0.5;
        let i = x.floor() as usize;
        let j = y.floor() as usize;
        let (fx, fy) = (x - i as f64, y - j as f64);
        let z = 100.0 + h;
        if i + 1 >= nx || j + 1 >= ny || z < 50.0 {
            continue;
        }
        let mut horizontal = 0.0;
        for (col, weight) in [
            (j * nx + i, (1.0 - fx) * (1.0 - fy)),
            (j * nx + i + 1, fx * (1.0 - fy)),
            ((j + 1) * nx + i, (1.0 - fx) * fy),
            ((j + 1) * nx + i + 1, fx * fy),
        ] {
            horizontal += weight * 10.0f64.powf(refc[col] * 0.1);
        }
        let lo = (0..29)
            .find(|&k| z >= f64::from(mass_height(k)) && z <= f64::from(mass_height(k + 1)))
            .unwrap();
        let f = (z - f64::from(mass_height(lo))) / f64::from(mass_height(lo + 1) - mass_height(lo));
        let low = 10.0f64.powf(-0.00008 * (f64::from(mass_height(lo)) - 2000.0).max(0.0));
        let high = 10.0f64.powf(-0.00008 * (f64::from(mass_height(lo + 1)) - 2000.0).max(0.0));
        power += w * horizontal * (low * (1.0 - f) + high * f);
        column_power += w * horizontal;
        weight += w;
    }
    (
        10.0 * (power / weight).log10(),
        10.0 * (column_power / weight).log10(),
    )
}

fn rss_kib() -> Option<u64> {
    let status = fs::read_to_string("/proc/self/status").ok()?;
    status
        .lines()
        .find(|line| line.starts_with("VmHWM:"))?
        .split_whitespace()
        .nth(1)?
        .parse()
        .ok()
}
fn qualify(out: &Path) -> Result<serde_json::Value> {
    let mut beams = File::create(out.join("beam-heights.csv"))?;
    writeln!(
        beams,
        "range_m,tilt_deg,reference_height_m,bowecho_height_m,error_m"
    )?;
    let mut max_height_error = 0.0f64;
    for tilt in [0.5f64, 1.5, 5.0, 19.5] {
        for range in [0.0f64, 25_000.0, 50_000.0, 100_000.0, 150_000.0, 230_000.0] {
            let effective = 6_371_000.0 * (4.0 / 3.0);
            let numerator = range * range + 2.0 * range * effective * tilt.to_radians().sin();
            let reference = numerator / ((effective * effective + numerator).sqrt() + effective);
            let actual = beam_height_above_radar_m(range, tilt);
            let error = actual - reference;
            max_height_error = max_height_error.max(error.abs());
            writeln!(beams, "{range},{tilt},{reference},{actual},{error}")?;
        }
    }
    assert!(max_height_error < 1.0e-7);
    let input = model(201, 161, 30, 0.0);
    let history = out.join("wrfout_d01_2026-10-02_00_00_00.nc");
    write_history(&history, &input, 0)?;
    let file = WrfFile::open(&history)?;
    let refc = file.read_var("REFC", 0)?;
    assert_eq!(refc.len(), input.nx * input.ny);
    let second = model(201, 161, 30, 3.0);
    write_history(&out.join("wrfout_d01_2026-10-02_00_05_00.nc"), &second, 5)?;
    let mut scan = config("TEST", 0.0, 0.0);
    scan.elevations_deg = vec![0.5, 0.9, 1.3];
    let source = WrfSourceIdentity("analytic-native-history".to_owned());
    let read = read_wrf_radar_fields_for_config(&file, &source, 0, &scan)?;
    let direct = WrfRadarFields::from_model_fields(input)?;
    let mut max_read_error = 0.0f32;
    for (a, b) in read.height_msl.iter().zip(&direct.height_msl) {
        max_read_error = max_read_error.max((a - b).abs());
    }
    assert!(max_read_error < 0.002);
    assert_eq!(read.dbz, direct.dbz);
    assert_eq!(read.u, direct.u);
    assert_eq!(read.v, direct.v);
    let valid = DateTime::parse_from_rfc3339("2026-10-02T00:00:00Z")?.with_timezone(&Utc);
    let volume = try_build_synthetic_volume(&read, valid, &scan)?;
    let cut = &volume.cuts[0];
    let vel = &cut.moments[&MomentType::Velocity];
    let refl = &cut.moments[&MomentType::Reflectivity];
    let mut velocities = File::create(out.join("radial-velocity.csv"))?;
    let mut refs = File::create(out.join("reflectivity-refc.csv"))?;
    writeln!(velocities, "azimuth_deg,reference_mps,simulated_mps")?;
    writeln!(
        refs,
        "range_m,model_refc_dbz,matched_model_dbz,simulated_dbz,error_dbz"
    )?;
    let mut max_velocity_error = 0.0f64;
    let (mut sum_ref_sq, mut max_ref_error, mut count) = (0.0f64, 0.0f64, 0usize);
    let mut sum_column_sq = 0.0f64;
    for row in 0..vel.radial_count() {
        let az = cut.radials[vel.radial_indices[row]].azimuth_deg as f64;
        let expected = (12.0 * az.to_radians().sin() - 4.0 * az.to_radians().cos())
            * 0.5f64.to_radians().cos();
        let actual = vel
            .scaled_value(row, 40)
            .ok_or("missing analytic velocity gate")? as f64;
        max_velocity_error = max_velocity_error.max((actual - expected).abs());
        writeln!(velocities, "{az},{expected},{actual}")?;
        for gate in (20..400).step_by(4) {
            let range = (refl.gate_range.first_gate_m
                + gate as i32 * refl.gate_range.gate_spacing_m) as f64;
            let (expected, column_refc) = pulse_reference(&refc, 201, 161, az, range);
            if let Some(actual) = refl.scaled_value(row, gate) {
                let error = actual as f64 - expected;
                if !actual.is_finite() {
                    continue;
                }
                sum_ref_sq += error * error;
                sum_column_sq += (actual as f64 - column_refc).powi(2);
                max_ref_error = max_ref_error.max(error.abs());
                count += 1;
                writeln!(refs, "{range},{column_refc},{expected},{actual},{error}")?;
            }
        }
    }
    assert!(
        max_velocity_error < 0.02,
        "wind projection error {max_velocity_error}"
    );
    let ref_rmse = (sum_ref_sq / count as f64).sqrt();
    assert!(ref_rmse < 0.02, "matched footprint RMSE {ref_rmse}");
    assert!(
        max_ref_error < 0.1,
        "matched footprint max error {max_ref_error}"
    );
    Ok(
        json!({"kind":"analytic_operator_qualification", "observation_skill_test":false,
        "beam_height_max_abs_error_m":max_height_error,
        "history_height_max_abs_error_m":max_read_error,
        "history_reflectivity_and_wind_arrays_exact":true,
        "wind_projection_max_abs_error_mps":max_velocity_error,
        "lowest_tilt_refc_paired_gates":count,
        "matched_footprint_rmse_dbz":ref_rmse,
        "lowest_tilt_refc_rmse_dbz":(sum_column_sq/count as f64).sqrt(),
        "matched_footprint_max_abs_error_dbz":max_ref_error,
        "reflectivity_scope":"Stored analytic REFC plane sampled in linear Z; matched nine-point footprint including vertical variation; lowest 0.5 degree tilt at 5 to 100 km",
        "fixture_shape":[201,161,30], "fixture_resolution_m":DX_M,
        "fixture_times":["2026-10-02T00:00:00Z","2026-10-02T00:05:00Z"]}),
    )
}
fn benchmark(out: &Path, write_full_history: bool) -> Result<serde_json::Value> {
    let setup = Instant::now();
    let mut input = model(1799, 1059, 50, 0.0);
    // Commit the W plane too; a zero-filled allocation can share OS zero pages.
    input.upward_wind_mps.fill(0.1);
    let array_seconds = setup.elapsed().as_secs_f64();
    let mut history = serde_json::Value::Null;
    if write_full_history {
        let dir = out.join("conus");
        fs::create_dir_all(&dir)?;
        let path = dir.join("wrfout_d01_2026-10-02_00_00_00.nc");
        let started = Instant::now();
        write_history(&path, &input, 0)?;
        history = json!({"path":path,"seconds":started.elapsed().as_secs_f64(),"bytes":fs::metadata(&path)?.len()});
    }
    let lut_start = Instant::now();
    let fields = WrfRadarFields::from_model_fields(input)?;
    let lut_seconds = lut_start.elapsed().as_secs_f64();
    let valid = DateTime::parse_from_rfc3339("2026-10-02T00:00:00Z")?.with_timezone(&Utc);
    let mut timings = Vec::new();
    let start = Instant::now();
    for (id, x, y) in [
        ("TEST1", -100.0, -50.0),
        ("TEST2", 0.0, 0.0),
        ("TEST3", 100.0, 50.0),
    ] {
        let scan = config(id, x, y);
        let clock = Instant::now();
        let volume = try_build_synthetic_volume(&fields, valid, &scan)?;
        let seconds = clock.elapsed().as_secs_f64();
        let gates: usize = volume
            .cuts
            .iter()
            .map(|cut| cut.radials.len() * cut.radials[0].gate_range.gate_count as usize)
            .sum();
        let finite: usize = volume
            .cuts
            .iter()
            .map(|cut| {
                let grid = &cut.moments[&MomentType::Reflectivity];
                (0..grid.radial_count())
                    .map(|row| {
                        (0..grid.gate_range.gate_count)
                            .filter(|&gate| {
                                grid.scaled_value(row, gate).is_some_and(f32::is_finite)
                            })
                            .count()
                    })
                    .sum::<usize>()
            })
            .sum();
        timings.push(json!({"site":id,"seconds":seconds,"tilts":volume.cuts.len(),"geometric_gates":gates,"finite_reflectivity_gates":finite}));
    }
    Ok(
        json!({"kind":"cpu_reference_synthetic_benchmark","observation_skill_test":false,
        "shape":[1799,1059,50],"cells":1799u64*1059*50,"resolution_m":DX_M,
        "rayon_threads":std::env::var("RAYON_NUM_THREADS").unwrap_or_default(),
        "array_generation_seconds":array_seconds,"geolocation_seconds":lut_seconds,
        "history":history, "volume_planes_committed":true,
        "three_site_seconds_including_output_inspection":start.elapsed().as_secs_f64(),
        "peak_rss_kib":rss_kib(),"beam_integration":"Balanced","gate_spacing_m":250,
        "azimuth_count":720,"max_range_m":230000,"timings":timings,
        "scope":"Full resident CONUS-size arrays; scalar reflectivity and velocity; excludes history read and radar file encoding"}),
    )
}
fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let out = PathBuf::from(
        args.next()
            .ok_or("usage: qualify OUTPUT_DIR [--conus] [--write-conus-history]")?,
    );
    let options: Vec<_> = args.collect();
    let full = options.iter().any(|v| v == "--write-conus-history");
    let conus = full || options.iter().any(|v| v == "--conus");
    fs::create_dir_all(&out)?;
    let science = qualify(&out)?;
    fs::write(
        out.join("science.json"),
        serde_json::to_vec_pretty(&science)?,
    )?;
    println!("{}", serde_json::to_string(&science)?);
    if conus {
        let bench = benchmark(&out, full)?;
        fs::write(
            out.join("benchmark.json"),
            serde_json::to_vec_pretty(&bench)?,
        )?;
        println!("{}", serde_json::to_string(&bench)?);
    }
    Ok(())
}
