//! Read exported rows through rustwx-io and compare three geographic points
//! against independent arithmetic on the original history carriers.
//! Usage: check_rows WRFOUT PRESSURE.grib2 SURFACE.grib2
use rustwx_core::{CanonicalField as F, FieldSelector, SelectedField2D};
use wrf_core::WrfFile;

fn nearest(lat: &[f64], lon: &[f64], a: f64, b: f64) -> usize {
    (0..lat.len())
        .min_by(|&i, &j| {
            let d = |c: usize| (lat[c] - a).powi(2) + ((lon[c] - b) * a.to_radians().cos()).powi(2);
            d(i).total_cmp(&d(j))
        })
        .unwrap()
}
fn at(field: &SelectedField2D, lat: f64, lon: f64) -> f64 {
    let ys: Vec<f64> = field.grid.lat_deg.iter().map(|&v| f64::from(v)).collect();
    let xs: Vec<f64> = field.grid.lon_deg.iter().map(|&v| f64::from(v)).collect();
    f64::from(field.values[nearest(&ys, &xs, lat, lon)])
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 4 {
        return Err("usage: check_rows WRFOUT PRESSURE.grib2 SURFACE.grib2".into());
    }
    let file = WrfFile::open(&args[1])?;
    let prs = std::fs::read(&args[2])?;
    let sfc = std::fs::read(&args[3])?;
    let mut decoded = rustwx_io::extract_fields_from_bytes(
        &prs,
        &[
            FieldSelector::isobaric(F::SpecificHumidity, 850),
            FieldSelector::isobaric(F::Omega, 500),
            FieldSelector::isobaric(F::GeometricVerticalVelocity, 500),
        ],
    )?;
    decoded.extend(rustwx_io::extract_fields_from_bytes(
        &sfc,
        &[
            FieldSelector::height_agl(F::UWind, 100),
            FieldSelector::height_agl(F::VWind, 100),
        ],
    )?);
    let read = |name: &str| file.read_var(name, 0);
    let p = read("P")?;
    let pb = read("PB")?;
    let theta = read("T")?;
    let q = read("QVAPOR")?;
    let w = read("W")?;
    let ph = read("PH")?;
    let phb = read("PHB")?;
    let hgt = read("HGT")?;
    let u = read("U")?;
    let v = read("V")?;
    let sin = read("SINALPHA")?;
    let cos = read("COSALPHA")?;
    let lat = read("XLAT")?;
    let lon = read("XLONG")?;
    let condensates: Vec<_> = ["QCLOUD", "QRAIN", "QICE", "QSNOW", "QGRAUP", "QHAIL"]
        .iter()
        .filter(|name| file.has_var(name))
        .map(|name| read(name))
        .collect::<Result<_, _>>()?;
    let n = file.nxy();
    let nx = file.nx;
    let ny = file.ny;
    println!("source={} reader=rustwx_io points=3", args[1]);
    let stats = |name: &str, mut values: Vec<f32>| {
        values.retain(|v| v.is_finite());
        values.sort_by(f32::total_cmp);
        if !values.is_empty() {
            let q = |fraction: f64| values[((values.len() - 1) as f64 * fraction) as usize];
            println!(
                "stats {name} n={} min={} p01={} median={} p99={} max={}",
                values.len(),
                values[0],
                q(0.01),
                q(0.5),
                q(0.99),
                values[values.len() - 1]
            );
        }
    };
    for field in &decoded[..3] {
        stats(&field.selector.to_string(), field.values.clone());
    }
    stats(
        "wind100_m_s",
        decoded[3]
            .values
            .iter()
            .zip(&decoded[4].values)
            .map(|(u, v)| u.hypot(*v))
            .collect(),
    );
    let land = read("LANDMASK")?;
    let u10 = read("U10")?;
    let v10 = read("V10")?;
    // The GRIB reader normalizes scan order. Align source cells explicitly
    // before combining decoded winds with raw history land/10 m fields.
    let ys: Vec<f64> = decoded[3]
        .grid
        .lat_deg
        .iter()
        .map(|&v| f64::from(v))
        .collect();
    let xs: Vec<f64> = decoded[3]
        .grid
        .lon_deg
        .iter()
        .map(|&v| f64::from(v))
        .collect();
    let origin = nearest(&ys, &xs, lat[0], lon[0]) as isize;
    let dx = (nearest(&ys, &xs, lat[nx - 1], lon[nx - 1]) as isize - origin) / (nx - 1) as isize;
    let dy = (nearest(&ys, &xs, lat[n - nx], lon[n - nx]) as isize - origin) / (ny - 1) as isize;
    let source_to_grid: Vec<usize> = (0..n)
        .map(|c| {
            let d =
                usize::try_from(origin + (c % nx) as isize * dx + (c / nx) as isize * dy).unwrap();
            assert!(
                (ys[d] - lat[c]).abs() < 0.05 && (xs[d] - lon[c]).abs() < 0.05,
                "source/GRIB grid mismatch at {c}"
            );
            d
        })
        .collect();
    stats(
        "land_wind100_over_wind10_speed10_ge_1",
        (0..n)
            .filter_map(|c| {
                let ten = u10[c].hypot(v10[c]);
                (land[c] > 0.5 && ten >= 1.).then(|| {
                    decoded[3].values[source_to_grid[c]].hypot(decoded[4].values[source_to_grid[c]])
                        / ten as f32
                })
            })
            .collect(),
    );
    let mut points = vec![
        ("Gulf", 29.76, -95.37),
        ("Plains", 37.69, -97.34),
        ("Midwest", 41.60, -93.60),
    ];
    // Diagnose a real proof-run finding: omega's domain extrema exceeded
    // the expected synoptic range. Check those cells against the same hand
    // formula to distinguish source W from a packing/interpolation error.
    for (label, minimum) in [("OmegaMinimum", true), ("OmegaMaximum", false)] {
        let c = (0..n)
            .filter(|&c| decoded[1].values[c].is_finite())
            .min_by(|&a, &b| {
                if minimum {
                    decoded[1].values[a].total_cmp(&decoded[1].values[b])
                } else {
                    decoded[1].values[b].total_cmp(&decoded[1].values[a])
                }
            })
            .unwrap();
        points.push((
            label,
            f64::from(decoded[1].grid.lat_deg[c]),
            f64::from(decoded[1].grid.lon_deg[c]),
        ));
    }
    for (name, a, b) in points {
        let c = nearest(&lat, &lon, a, b);
        let i = c % nx;
        let j = c / nx;
        let pressure = |k: usize| p[k * n + c] + pb[k * n + c];
        let mass_w = |k: usize| 0.5 * (w[k * n + c] + w[(k + 1) * n + c]);
        let omega = |k: usize| {
            let r = pressure(k);
            let t = (theta[k * n + c] + 300.) * (r / 100000.).powf(0.2857142857);
            -r * 9.80665 * mass_w(k) / (287.058 * t * (1. + 0.61 * q[k * n + c].max(0.)))
        };
        let specific = |k: usize| {
            let vap = q[k * n + c].max(0.);
            vap / (1.
                + vap
                + condensates
                    .iter()
                    .map(|f| f[k * n + c].max(0.))
                    .sum::<f64>())
        };
        let interp = |target: f64, f: &dyn Fn(usize) -> f64| {
            if target >= pressure(0) {
                return f(0);
            }
            for k in 0..file.nz - 1 {
                if pressure(k) >= target && target >= pressure(k + 1) {
                    let weight = (target / pressure(k)).ln() / (pressure(k + 1) / pressure(k)).ln();
                    return f(k) + weight * (f(k + 1) - f(k));
                }
            }
            f64::NAN
        };
        let z = |k: usize| {
            0.5 * (ph[k * n + c] + phb[k * n + c] + ph[(k + 1) * n + c] + phb[(k + 1) * n + c])
                / 9.80665
                - hgt[c]
        };
        let mut wind = (f64::NAN, f64::NAN);
        for k in 0..file.nz - 1 {
            if z(k) <= 100. && 100. <= z(k + 1) {
                let weight = (100. - z(k)) / (z(k + 1) - z(k));
                let ug = |l: usize| {
                    0.5 * (u[(l * ny + j) * (nx + 1) + i] + u[(l * ny + j) * (nx + 1) + i + 1])
                };
                let vg = |l: usize| {
                    0.5 * (v[(l * (ny + 1) + j) * nx + i] + v[(l * (ny + 1) + j + 1) * nx + i])
                };
                let x = ug(k) + weight * (ug(k + 1) - ug(k));
                let y = vg(k) + weight * (vg(k + 1) - vg(k));
                wind = (x * cos[c] - y * sin[c], x * sin[c] + y * cos[c]);
                break;
            }
        }
        let expected = [
            interp(85000., &specific),
            interp(50000., &omega),
            interp(50000., &mass_w),
            wind.0,
            wind.1,
        ];
        for ((field, want), tol) in decoded
            .iter()
            .zip(expected)
            .zip([2e-7, 2e-4, 2e-5, 2e-4, 2e-4])
        {
            if name.starts_with("Omega") && field.selector.field != F::Omega {
                continue;
            }
            let got = at(field, lat[c], lon[c]);
            let error = (got - want).abs();
            println!("{name} ij={i},{j} lat={:.5} lon={:.5} {} decoded={got:.9} hand={want:.9} abs_error={error:.3e}",lat[c],lon[c],field.selector);
            if !got.is_finite() || error > tol {
                return Err(
                    format!("{} mismatch at {name}: {error} > {tol}", field.selector).into(),
                );
            }
        }
    }
    Ok(())
}
