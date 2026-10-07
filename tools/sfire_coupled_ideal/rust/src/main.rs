//! Native NetCDF decoding and fire/atmosphere verification metrics.
use netcrust::{DataArray, File};
use rustwx_contour::{
    ContourEngine, ContourLevels, LineSegment, Point2, RectilinearGrid, ScalarField2D,
};
use serde::Serialize;
use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
const GRAVITY: f64 = 9.81;
const CP: f64 = 1004.5;
const RD: f64 = 287.0;

#[derive(Serialize)]
struct FieldSummary {
    words: usize,
    min: f64,
    max: f64,
    mean: f64,
    rms: f64,
}

#[derive(Serialize)]
struct Frame {
    file: String,
    time_seconds: f64,
    fire_shape: [usize; 2],
    fire_area_m2: f64,
    levelset_negative_area_m2: f64,
    perimeter_m: f64,
    sensible_power_w: f64,
    latent_power_w: f64,
    remaining_fuel_cell_fraction_sum: f64,
    dry_air_mass_kg: Option<f64>,
    water_vapor_mass_kg: Option<f64>,
    dry_enthalpy_proxy_j: Option<f64>,
    mass_weighted_theta_k: Option<f64>,
    theta_profile_k: Option<Vec<f64>>,
    qv_profile_kg_kg: Option<Vec<f64>>,
    mass_metric_missing_fields: Vec<String>,
    upward_velocity_profile_max_m_s: Vec<f64>,
    fields: BTreeMap<String, FieldSummary>,
}

#[derive(Serialize)]
struct FieldDifference {
    words: usize,
    different_f32_words: usize,
    max_ulp: u64,
    max_abs: f64,
    bias: f64,
    rmse: f64,
}

#[derive(Serialize)]
struct PairedFrame {
    time_seconds: f64,
    reference: Frame,
    candidate: Frame,
    area_difference_m2: f64,
    perimeter_difference_m: f64,
    sensible_power_difference_w: f64,
    latent_power_difference_w: f64,
    burned_cell_iou: f64,
    perimeter_midpoint_mean_distance_m: Option<f64>,
    perimeter_sampled_hausdorff_m: Option<f64>,
    differences: BTreeMap<String, FieldDifference>,
}

#[derive(Serialize)]
struct HistoryIntegral {
    start_seconds: f64,
    end_seconds: f64,
    samples: usize,
    sensible_heat_j_trapezoid: f64,
    latent_heat_j_trapezoid: f64,
    dry_air_mass_change_kg: Option<f64>,
    vapor_mass_change_kg: Option<f64>,
    theta_change_k: Option<f64>,
}

fn history_integral<'a>(frames: impl Iterator<Item = &'a Frame>) -> Result<HistoryIntegral> {
    let frames: Vec<_> = frames.collect();
    if frames.is_empty() {
        return Err("cannot integrate an empty history".into());
    }
    let first = frames[0];
    let last = frames[frames.len() - 1];
    let mut sensible = 0.;
    let mut latent = 0.;
    for pair in frames.windows(2) {
        let dt = pair[1].time_seconds - pair[0].time_seconds;
        if dt <= 0. {
            return Err("history times must increase".into());
        }
        sensible += 0.5 * dt * (pair[1].sensible_power_w + pair[0].sensible_power_w);
        latent += 0.5 * dt * (pair[1].latent_power_w + pair[0].latent_power_w);
    }
    Ok(HistoryIntegral {
        start_seconds: first.time_seconds,
        end_seconds: last.time_seconds,
        samples: frames.len(),
        sensible_heat_j_trapezoid: sensible,
        latent_heat_j_trapezoid: latent,
        dry_air_mass_change_kg: last.dry_air_mass_kg.zip(first.dry_air_mass_kg).map(|(a,b)|a-b),
        vapor_mass_change_kg: last.water_vapor_mass_kg.zip(first.water_vapor_mass_kg).map(|(a,b)|a-b),
        theta_change_k: last.mass_weighted_theta_k.zip(first.mass_weighted_theta_k).map(|(a,b)|a-b),
    })
}

fn read(file: &File, name: &str) -> Result<DataArray> {
    let data = file.read_array_f64_first_record_or_all(name)?;
    if data
        .values()
        .iter()
        .any(|v| !v.is_finite() || v.abs() > 1.0e30)
    {
        return Err(format!("{name} contains a nonfinite or uninitialized value").into());
    }
    Ok(data)
}

fn attribute(file: &File, name: &str) -> Result<f64> {
    file.attribute(name)
        .and_then(|v| v.as_f64())
        .filter(|v| v.is_finite())
        .ok_or_else(|| format!("missing finite {name} attribute").into())
}

fn summary(values: &[f64]) -> Result<FieldSummary> {
    if values.is_empty() {
        return Err("empty field".into());
    }
    let n = values.len() as f64;
    Ok(FieldSummary {
        words: values.len(),
        min: values.iter().copied().fold(f64::INFINITY, f64::min),
        max: values.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        mean: values.iter().sum::<f64>() / n,
        rms: (values.iter().map(|v| v * v).sum::<f64>() / n).sqrt(),
    })
}

fn trim_fire(data: &DataArray, ny: usize, nx: usize) -> Result<Vec<f64>> {
    let shape = data.shape();
    if shape.len() != 2 || shape[0] < ny || shape[1] < nx {
        return Err(format!("fire grid {:?} cannot contain {ny}x{nx}", shape).into());
    }
    Ok((0..ny)
        .flat_map(|j| {
            data.values()[j * shape[1]..j * shape[1] + nx]
                .iter()
                .copied()
        })
        .collect())
}

fn perimeter_segments(
    values: Vec<f64>,
    ny: usize,
    nx: usize,
    dx: f64,
    dy: f64,
) -> Result<Vec<LineSegment>> {
    let grid = RectilinearGrid::new(
        (0..nx).map(|i| (i as f64 + 0.5) * dx).collect(),
        (0..ny).map(|j| (j as f64 + 0.5) * dy).collect(),
    )?;
    let field = ScalarField2D::new(grid, values)?;
    let topology = ContourEngine::new().extract_isolines(&field, &ContourLevels::new(vec![0.0])?);
    Ok(topology
        .layers
        .into_iter()
        .flat_map(|l| l.segments)
        .map(|s| s.geometry)
        .collect())
}

fn perimeter(values: Vec<f64>, ny: usize, nx: usize, dx: f64, dy: f64) -> Result<f64> {
    Ok(perimeter_segments(values, ny, nx, dx, dy)?
        .iter()
        .map(|s| s.length_squared().sqrt())
        .sum())
}

fn point_distance(point: Point2, line: &LineSegment) -> f64 {
    let dx = line.end.x - line.start.x;
    let dy = line.end.y - line.start.y;
    let norm = dx * dx + dy * dy;
    let fraction = if norm > 0. {
        ((point.x - line.start.x) * dx + (point.y - line.start.y) * dy) / norm
    } else {
        0.
    };
    let nearest = line.start.lerp(line.end, fraction.clamp(0., 1.));
    ((point.x - nearest.x).powi(2) + (point.y - nearest.y).powi(2)).sqrt()
}

fn perimeter_distances(a: &[LineSegment], b: &[LineSegment]) -> (Option<f64>, Option<f64>) {
    if a.is_empty() && b.is_empty() {
        return (Some(0.), Some(0.));
    }
    if a.is_empty() || b.is_empty() {
        return (None, None);
    }
    let mut max_distance: f64 = 0.;
    let mut total = 0.;
    let mut length = 0.;
    for (from, to) in [(a, b), (b, a)] {
        for segment in from {
            let weight = segment.length_squared().sqrt();
            let middle = segment.start.lerp(segment.end, 0.5);
            for point in [segment.start, middle, segment.end] {
                let distance = to
                    .iter()
                    .map(|line| point_distance(point, line))
                    .fold(f64::INFINITY, f64::min);
                max_distance = max_distance.max(distance);
                if point == middle {
                    total += weight * distance;
                    length += weight;
                }
            }
        }
    }
    (
        Some(if length > 0. { total / length } else { 0. }),
        Some(max_distance),
    )
}

fn frame(path: &Path, sr_x: usize, sr_y: usize) -> Result<Frame> {
    let file = netcrust::open(path)?;
    let t = read(&file, "T")?;
    if t.shape().len() != 3 {
        return Err("T must have level,y,x dimensions".into());
    }
    let [nz, ny, nx] = [t.shape()[0], t.shape()[1], t.shape()[2]];
    let dx = attribute(&file, "DX")?;
    let dy = attribute(&file, "DY")?;
    let fdx = dx / sr_x as f64;
    let fdy = dy / sr_y as f64;
    let fnx = nx * sr_x;
    let fny = ny * sr_y;
    let lfn = trim_fire(&read(&file, "LFN")?, fny, fnx)?;
    let fire_area = trim_fire(&read(&file, "FIRE_AREA")?, fny, fnx)?;
    let fuel = trim_fire(&read(&file, "FUEL_FRAC")?, fny, fnx)?;
    let sensible = trim_fire(&read(&file, "FGRNHFX")?, fny, fnx)?;
    let latent = trim_fire(&read(&file, "FGRNQFX")?, fny, fnx)?;
    if fire_area
        .iter()
        .chain(&fuel)
        .any(|v| *v < -1e-5 || *v > 1.0 + 1e-5)
    {
        return Err("fire area or fuel fraction is outside [0,1]".into());
    }
    let mu = read(&file, "MU")?;
    let mub = read(&file, "MUB")?;
    let qv = read(&file, "QVAPOR")?;
    let w = read(&file, "W")?;
    let p = read(&file, "P")?;
    let pb = read(&file, "PB")?;
    let mass_metric_missing_fields: Vec<_> = ["DNW", "C1H", "C2H"].iter()
        .filter(|name| file.variable(name).is_none()).map(|name| name.to_string()).collect();
    let mass_coordinates = if mass_metric_missing_fields.is_empty() {
        Some((read(&file, "DNW")?, read(&file, "C1H")?, read(&file, "C2H")?))
    } else { None };
    let mapx = read(
        &file,
        if file.variable("MAPFAC_MX").is_some() {
            "MAPFAC_MX"
        } else {
            "MAPFAC_M"
        },
    )?;
    let mapy = read(
        &file,
        if file.variable("MAPFAC_MY").is_some() {
            "MAPFAC_MY"
        } else {
            "MAPFAC_M"
        },
    )?;
    let horizontal = nx * ny;
    if mu.len() != horizontal
        || mub.len() != horizontal
        || mapx.len() != horizontal
        || mapy.len() != horizontal
        || qv.shape() != t.shape()
        || p.shape() != t.shape()
        || pb.shape() != t.shape()
        || w.shape() != [nz + 1, ny, nx]
    {
        return Err("atmospheric mass, eta or stagger dimensions disagree".into());
    }
    if let Some((dnw,c1h,c2h)) = &mass_coordinates {
        if dnw.len()!=nz || c1h.len()!=nz || c2h.len()!=nz {
            return Err("native atmospheric eta profile dimensions disagree".into());
        }
    }
    if mapx.values().iter().chain(mapy.values()).any(|v| *v <= 0.) {
        return Err(
            "mass integration requires the positive directional map factors used by WRF dynamics"
                .into(),
        );
    }
    let mut dry = 0.;
    let mut vapor = 0.;
    let mut heat = 0.;
    let mut theta_total = 0.;
    let mut theta_profile = Vec::new();
    let mut vapor_profile = Vec::new();
    let mut w_profile = Vec::new();
    if let Some((dnw,c1h,c2h)) = &mass_coordinates {
     for k in 0..nz {
        let mut mass_k = 0.;
        let mut theta_k = 0.;
        let mut vapor_k = 0.;
        for j in 0..horizontal {
            let mass = (c1h.values()[k] * (mu.values()[j] + mub.values()[j]) + c2h.values()[k])
                * (-dnw.values()[k])
                * dx
                * dy
                / (GRAVITY * mapx.values()[j] * mapy.values()[j]);
            if !mass.is_finite() || mass <= 0. {
                return Err("nonfinite or nonpositive dry atmospheric mass".into());
            }
            let theta = t.values()[k * horizontal + j] + 300.;
            let moisture = qv.values()[k * horizontal + j];
            let pressure = p.values()[k * horizontal + j] + pb.values()[k * horizontal + j];
            if pressure <= 0. {
                return Err("nonpositive atmospheric pressure".into());
            }
            mass_k += mass;
            theta_k += mass * theta;
            vapor_k += mass * moisture;
            heat += mass * CP * theta * (pressure / 100000.).powf(RD / CP);
        }
        theta_profile.push(theta_k / mass_k);
        vapor_profile.push(vapor_k / mass_k);
        dry += mass_k;
        vapor += vapor_k;
        theta_total += theta_k;
     }
    }
    for k in 0..nz {
        w_profile.push(w.values()[k * horizontal..(k + 1) * horizontal]
            .iter().copied().fold(f64::NEG_INFINITY, f64::max));
    }
    if p.values().iter().zip(pb.values()).any(|(a,b)| a+b<=0.) {
        return Err("nonpositive atmospheric pressure".into());
    }
    if [dry, vapor, heat, theta_total]
        .iter()
        .any(|v| !v.is_finite())
    {
        return Err("atmospheric integral overflowed; refusing a null numeric receipt".into());
    }
    let mut fields = BTreeMap::new();
    for name in [
        "U",
        "V",
        "W",
        "T",
        "QVAPOR",
        "P",
        "MU",
        "PH",
        "FIRE_AREA",
        "FUEL_FRAC",
        "LFN",
        "FGRNHFX",
        "FGRNQFX",
        "ROS_FRONT",
    ] {
        let data = read(&file, name)?;
        let vals = if data.shape().len() == 2 && name.starts_with('F')
            || ["LFN", "ROS_FRONT"].contains(&name)
        {
            trim_fire(&data, fny, fnx)?
        } else {
            data.into_values()
        };
        fields.insert(name.into(), summary(&vals)?);
    }
    // Optional coupled smoke and moisture diagnostics remain native-decoded.
    // Absence is legitimate for the original ideal reference and chem-off runs.
    for name in [
        "fire_smoke", "FIRE_SMOKE", "SFIRE_SMOKE_SFC", "SFIRE_SMOKE_COLUMN",
        "SFIRE_SMOKE_FUEL_SOURCE", "SFIRE_SMOKE_INJECTED", "SFIRE_SMOKE_EMITTED",
        "FMC_G", "FMC_GC", "ROS",
    ] {
        if file.variable(name).is_some() {
            let data = read(&file, name)?;
            fields.insert(name.into(), summary(data.values())?);
        }
    }
    let seconds = read(&file, "XTIME")?.values()[0] * 60.;
    Ok(Frame {
        file: path.display().to_string(),
        time_seconds: seconds,
        fire_shape: [fny, fnx],
        fire_area_m2: fire_area.iter().sum::<f64>() * fdx * fdy,
        levelset_negative_area_m2: lfn.iter().filter(|v| **v < 0.).count() as f64 * fdx * fdy,
        perimeter_m: perimeter(lfn, fny, fnx, fdx, fdy)?,
        sensible_power_w: sensible.iter().sum::<f64>() * fdx * fdy,
        latent_power_w: latent.iter().sum::<f64>() * fdx * fdy,
        remaining_fuel_cell_fraction_sum: fuel.iter().sum(),
        dry_air_mass_kg: mass_coordinates.as_ref().map(|_|dry),
        water_vapor_mass_kg: mass_coordinates.as_ref().map(|_|vapor),
        dry_enthalpy_proxy_j: mass_coordinates.as_ref().map(|_|heat),
        mass_weighted_theta_k: mass_coordinates.as_ref().map(|_|theta_total/dry),
        theta_profile_k: mass_coordinates.as_ref().map(|_|theta_profile),
        qv_profile_kg_kg: mass_coordinates.as_ref().map(|_|vapor_profile),
        mass_metric_missing_fields,
        upward_velocity_profile_max_m_s: w_profile,
        fields,
    })
}

fn key(v: f32) -> i64 {
    let bits = v.to_bits();
    if bits & 0x80000000 != 0 {
        -(bits as i64 & 0x7fffffff)
    } else {
        bits as i64
    }
}
fn difference(a: &[f64], b: &[f64]) -> Result<FieldDifference> {
    if a.len() != b.len() || a.is_empty() {
        return Err("paired field lengths disagree".into());
    }
    let mut result = FieldDifference {
        words: a.len(),
        different_f32_words: 0,
        max_ulp: 0,
        max_abs: 0.,
        bias: 0.,
        rmse: 0.,
    };
    for (&a, &b) in a.iter().zip(b) {
        let af = a as f32;
        let bf = b as f32;
        result.different_f32_words += usize::from(af.to_bits() != bf.to_bits());
        result.max_ulp = result.max_ulp.max((key(af) - key(bf)).unsigned_abs());
        let d = b - a;
        result.max_abs = result.max_abs.max(d.abs());
        result.bias += d;
        result.rmse += d * d;
    }
    result.bias /= a.len() as f64;
    result.rmse = (result.rmse / a.len() as f64).sqrt();
    Ok(result)
}

fn files(root: &Path) -> Result<BTreeMap<i64, PathBuf>> {
    let mut result = BTreeMap::new();
    for entry in fs::read_dir(root)? {
        let path = entry?.path();
        if !path.is_file()
            || !path
                .file_name()
                .unwrap()
                .to_string_lossy()
                .starts_with("wrfout")
        {
            continue;
        }
        let file = netcrust::open(&path)?;
        let time = read(&file, "XTIME")?.values()[0] * 60.;
        let key = (time * 1000.).round() as i64;
        if result.insert(key, path).is_some() {
            return Err("duplicate history time".into());
        }
    }
    if result.is_empty() {
        return Err("no wrfout frames".into());
    }
    Ok(result)
}

fn run() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() != 5 {
        return Err("usage: sfire-ideal-compare REFERENCE_DIR CANDIDATE_DIR SR_X SR_Y OUTPUT_JSON; use - as CANDIDATE_DIR to summarize a reference".into());
    }
    let sx = args[2].parse::<usize>()?;
    let sy = args[3].parse::<usize>()?;
    if sx == 0 || sy == 0 {
        return Err("fire refinement must be positive".into());
    }
    let left = files(Path::new(&args[0]))?;
    if args[1] == "-" {
        let frames = left
            .values()
            .map(|p| frame(p, sx, sy))
            .collect::<Result<Vec<_>>>()?;
        let integral = history_integral(frames.iter())?;
        fs::write(
            &args[4],
            serde_json::to_string_pretty(
                &serde_json::json!({"schema":"sfire-ideal-summary-v1","frames":frames,"integral":integral,
            "perimeter":"rustwx-contour marching squares of LFN=0 at fire cell centers", "enthalpy":"dry-air Cp*T proxy; excludes latent and potential energy, open-boundary fluxes and pressure work"}),
            )? + "\n",
        )?;
        return Ok(());
    }
    let right = files(Path::new(&args[1]))?;
    if left.keys().ne(right.keys()) {
        return Err("reference/candidate history timelines disagree".into());
    }
    let mut pairs = Vec::new();
    for (time, path) in left {
        let candidate = &right[&time];
        let lf = netcrust::open(&path)?;
        let rf = netcrust::open(candidate)?;
        let reference = frame(&path, sx, sy)?;
        let candidate_frame = frame(candidate, sx, sy)?;
        let ny = reference.fire_shape[0];
        let nx = reference.fire_shape[1];
        let la = trim_fire(&read(&lf, "LFN")?, ny, nx)?;
        let lb = trim_fire(&read(&rf, "LFN")?, ny, nx)?;
        let intersection = la
            .iter()
            .zip(&lb)
            .filter(|(a, b)| **a < 0. && **b < 0.)
            .count();
        let union = la
            .iter()
            .zip(&lb)
            .filter(|(a, b)| **a < 0. || **b < 0.)
            .count();
        let burned_cell_iou = if union == 0 {
            1.
        } else {
            intersection as f64 / union as f64
        };
        let lines_a = perimeter_segments(
            la,
            ny,
            nx,
            attribute(&lf, "DX")? / sx as f64,
            attribute(&lf, "DY")? / sy as f64,
        )?;
        let lines_b = perimeter_segments(
            lb,
            ny,
            nx,
            attribute(&rf, "DX")? / sx as f64,
            attribute(&rf, "DY")? / sy as f64,
        )?;
        let (mean_distance, max_distance) = perimeter_distances(&lines_a, &lines_b);
        let mut diffs = BTreeMap::new();
        for name in reference.fields.keys() {
            if !candidate_frame.fields.contains_key(name) {
                continue; // Optional diagnostics are compared only when both cases publish them.
            }
            let a = read(&lf, name)?;
            let b = read(&rf, name)?;
            let is_fire = [
                "FIRE_AREA",
                "FUEL_FRAC",
                "LFN",
                "FGRNHFX",
                "FGRNQFX",
                "ROS_FRONT",
                "FMC_G", "ROS",
            ]
            .contains(&name.as_str());
            let (a, b) = if is_fire {
                (
                    trim_fire(&a, reference.fire_shape[0], reference.fire_shape[1])?,
                    trim_fire(&b, reference.fire_shape[0], reference.fire_shape[1])?,
                )
            } else {
                if a.shape() != b.shape() {
                    return Err(format!("{name} paired shapes disagree").into());
                }
                (a.into_values(), b.into_values())
            };
            diffs.insert(name.clone(), difference(&a, &b)?);
        }
        pairs.push(PairedFrame {
            time_seconds: reference.time_seconds,
            area_difference_m2: candidate_frame.fire_area_m2 - reference.fire_area_m2,
            perimeter_difference_m: candidate_frame.perimeter_m - reference.perimeter_m,
            sensible_power_difference_w: candidate_frame.sensible_power_w
                - reference.sensible_power_w,
            latent_power_difference_w: candidate_frame.latent_power_w - reference.latent_power_w,
            burned_cell_iou,
            perimeter_midpoint_mean_distance_m: mean_distance,
            perimeter_sampled_hausdorff_m: max_distance,
            reference,
            candidate: candidate_frame,
            differences: diffs,
        });
    }
    let reference_integral = history_integral(pairs.iter().map(|p| &p.reference))?;
    let candidate_integral = history_integral(pairs.iter().map(|p| &p.candidate))?;
    fs::write(
        &args[4],
        serde_json::to_string_pretty(
            &serde_json::json!({"schema":"sfire-ideal-comparison-v1","frames":pairs,
        "reference_integral":reference_integral,"candidate_integral":candidate_integral,
        "perimeter":"rustwx-contour marching squares of LFN=0 at fire cell centers",
        "heat_integral":"trapezoidal estimate at stored history times; not every dynamics step",
        "perimeter_distance":"symmetric nearest-segment distance; length-weighted midpoint mean and maximum at vertices plus midpoints, a sampled Hausdorff estimate",
        "word_grading":"stored fields promoted from native NetCDF then cast to f32; signed ULP distance with zero identified numerically"}),
        )? + "\n",
    )?;
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn zero_contour_has_metric_length() {
        let vals = vec![-1., 0., 1., -1., 0., 1., -1., 0., 1.];
        assert!((perimeter(vals, 3, 3, 2., 3.).unwrap() - 6.).abs() < 1.0e-12);
    }
    #[test]
    fn word_and_absolute_grades_are_independent() {
        let a = vec![0., 1., -1.];
        let b = vec![-0., f32::from_bits(1f32.to_bits() + 1) as f64, -1.];
        let result = difference(&a, &b).unwrap();
        assert_eq!(result.different_f32_words, 2);
        assert_eq!(result.max_ulp, 1);
        assert!(result.rmse > 0.);
    }
    #[test]
    fn parallel_perimeters_retain_their_metric_offset() {
        let a = vec![LineSegment::new(Point2::new(0., 0.), Point2::new(0., 10.))];
        let b = vec![LineSegment::new(Point2::new(3., 0.), Point2::new(3., 10.))];
        assert_eq!(perimeter_distances(&a, &b), (Some(3.), Some(3.)));
        assert_eq!(perimeter_distances(&a, &[]), (None, None));
        assert_eq!(perimeter_distances(&[], &[]), (Some(0.), Some(0.)));
    }
}
