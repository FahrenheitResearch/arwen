//! Criterion benchmarks for metrust meteorological calculations.
//!
//! Run: `cargo bench --package metrust`
//! Reports land in `target/criterion/`.
//!
//! Deterministic data generators (no RNG) use trigonometric formulas
//! to produce realistic-looking synthetic weather data.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

use metrust::calc::thermo;
use metrust::calc::wind;
use metrust::calc::severe;
use metrust::calc::smooth;

// Re-exports from wx-math that metrust uses
use wx_math::dynamics;
use wx_math::thermo as wx_thermo;

// ---------------------------------------------------------------------------
// Deterministic data generators
// ---------------------------------------------------------------------------

/// Sinusoidal 1D array — no RNG needed.
fn synthetic_1d(n: usize, base: f64, scale: f64) -> Vec<f64> {
    (0..n)
        .map(|i| base + scale * (i as f64 * 0.1).sin())
        .collect()
}

/// Smooth 2D field — sinusoidal in x and y.
fn synthetic_grid(nx: usize, ny: usize, base: f64, scale: f64) -> Vec<f64> {
    (0..ny * nx)
        .map(|idx| {
            let j = idx / nx;
            let i = idx % nx;
            base + scale * (i as f64 * 0.05).sin() * (j as f64 * 0.05).cos()
        })
        .collect()
}

/// Realistic sounding profile: pressure, temperature, dewpoint, height AGL.
fn synthetic_sounding(n: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let p: Vec<f64> = (0..n)
        .map(|i| 1000.0 - (i as f64 * 900.0 / n as f64))
        .collect();
    let t: Vec<f64> = (0..n)
        .map(|i| 30.0 - 70.0 * (i as f64 / n as f64) + 3.0 * (i as f64 * 0.2).sin())
        .collect();
    let td: Vec<f64> = (0..n)
        .map(|i| 20.0 - 60.0 * (i as f64 / n as f64) + 2.0 * (i as f64 * 0.15).sin())
        .collect();
    let h: Vec<f64> = (0..n).map(|i| i as f64 * 100.0).collect();
    (p, t, td, h)
}

/// Synthetic wind profile for storm motion calculations.
fn synthetic_wind_profile(n: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let u: Vec<f64> = (0..n)
        .map(|i| 5.0 + 25.0 * (i as f64 / n as f64) + 3.0 * (i as f64 * 0.3).sin())
        .collect();
    let v: Vec<f64> = (0..n)
        .map(|i| -5.0 + 15.0 * (i as f64 / n as f64) + 2.0 * (i as f64 * 0.25).cos())
        .collect();
    let z: Vec<f64> = (0..n).map(|i| i as f64 * 120.0).collect();
    (u, v, z)
}

// ---------------------------------------------------------------------------
// 1. Scalar thermodynamics
// ---------------------------------------------------------------------------

fn bench_scalar_thermo(c: &mut Criterion) {
    let mut group = c.benchmark_group("scalar_thermo");

    group.bench_function("potential_temperature", |b| {
        b.iter(|| wx_thermo::potential_temperature(black_box(850.0), black_box(25.0)))
    });

    group.bench_function("saturation_vapor_pressure", |b| {
        b.iter(|| wx_thermo::vappres(black_box(25.0)))
    });

    group.bench_function("equivalent_potential_temperature", |b| {
        b.iter(|| {
            wx_thermo::equivalent_potential_temperature(
                black_box(850.0), black_box(25.0), black_box(18.0),
            )
        })
    });

    group.bench_function("wet_bulb_temperature", |b| {
        b.iter(|| {
            wx_thermo::wet_bulb_temperature(black_box(850.0), black_box(25.0), black_box(18.0))
        })
    });

    group.bench_function("lcl", |b| {
        b.iter(|| thermo::lcl(black_box(1000.0), black_box(25.0), black_box(18.0)))
    });

    group.bench_function("mixing_ratio", |b| {
        b.iter(|| thermo::mixing_ratio(black_box(850.0), black_box(20.0)))
    });

    group.bench_function("dewpoint_from_rh", |b| {
        b.iter(|| thermo::dewpoint_from_relative_humidity(black_box(25.0), black_box(60.0)))
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// 2. Array thermodynamics
// ---------------------------------------------------------------------------

fn bench_array_thermo(c: &mut Criterion) {
    let mut group = c.benchmark_group("array_thermo");
    let n = 1000;

    let pressures = synthetic_1d(n, 850.0, 50.0);
    let temperatures = synthetic_1d(n, 20.0, 10.0);

    group.bench_function("potential_temperature_1000", |b| {
        b.iter(|| {
            pressures.iter().zip(temperatures.iter())
                .map(|(&p, &t)| wx_thermo::potential_temperature(p, t))
                .collect::<Vec<f64>>()
        })
    });

    group.bench_function("saturation_vapor_pressure_1000", |b| {
        b.iter(|| {
            temperatures.iter()
                .map(|&t| wx_thermo::vappres(t))
                .collect::<Vec<f64>>()
        })
    });

    // Parcel profile on 100 levels
    let (p100, _, _, _) = synthetic_sounding(100);
    group.bench_function("parcel_profile_100", |b| {
        b.iter(|| wx_thermo::parcel_profile(black_box(&p100), black_box(25.0), black_box(18.0)))
    });

    group.bench_function("dry_lapse_100", |b| {
        b.iter(|| wx_thermo::dry_lapse(black_box(&p100), black_box(25.0)))
    });

    group.bench_function("moist_lapse_100", |b| {
        b.iter(|| wx_thermo::moist_lapse(black_box(&p100), black_box(25.0)))
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// 3. CAPE/CIN
// ---------------------------------------------------------------------------

fn bench_cape_cin(c: &mut Criterion) {
    let mut group = c.benchmark_group("cape_cin");
    group.sample_size(20);

    for &n in &[100, 500] {
        let (p, t, td, h) = synthetic_sounding(n);

        group.bench_with_input(BenchmarkId::new("surface", n), &n, |b, _| {
            b.iter(|| {
                thermo::cape_cin(
                    black_box(&p), black_box(&t), black_box(&td), black_box(&h),
                    black_box(1000.0), black_box(30.0), black_box(20.0),
                    black_box("sb"), black_box(100.0), black_box(300.0), black_box(None),
                )
            })
        });

        group.bench_with_input(BenchmarkId::new("mixed_layer", n), &n, |b, _| {
            b.iter(|| {
                thermo::cape_cin(
                    black_box(&p), black_box(&t), black_box(&td), black_box(&h),
                    black_box(1000.0), black_box(30.0), black_box(20.0),
                    black_box("ml"), black_box(100.0), black_box(300.0), black_box(None),
                )
            })
        });

        group.bench_with_input(BenchmarkId::new("most_unstable", n), &n, |b, _| {
            b.iter(|| {
                thermo::cape_cin(
                    black_box(&p), black_box(&t), black_box(&td), black_box(&h),
                    black_box(1000.0), black_box(30.0), black_box(20.0),
                    black_box("mu"), black_box(100.0), black_box(300.0), black_box(None),
                )
            })
        });
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// 4. Grid kinematics
// ---------------------------------------------------------------------------

fn bench_grid_kinematics(c: &mut Criterion) {
    let mut group = c.benchmark_group("grid_kinematics");

    for &(nx, ny) in &[(100, 100), (500, 500)] {
        let u = synthetic_grid(nx, ny, 10.0, 5.0);
        let v = synthetic_grid(nx, ny, -3.0, 4.0);
        let theta = synthetic_grid(nx, ny, 300.0, 10.0);
        let dx = 3000.0;
        let dy = 3000.0;
        let label = format!("{}x{}", nx, ny);

        group.bench_with_input(BenchmarkId::new("divergence", &label), &(), |b, _| {
            b.iter(|| dynamics::divergence(&u, &v, nx, ny, dx, dy))
        });

        group.bench_with_input(BenchmarkId::new("vorticity", &label), &(), |b, _| {
            b.iter(|| dynamics::vorticity(&u, &v, nx, ny, dx, dy))
        });

        group.bench_with_input(BenchmarkId::new("advection", &label), &(), |b, _| {
            b.iter(|| dynamics::advection(&theta, &u, &v, nx, ny, dx, dy))
        });

        group.bench_with_input(BenchmarkId::new("frontogenesis", &label), &(), |b, _| {
            b.iter(|| dynamics::frontogenesis_2d(&theta, &u, &v, nx, ny, dx, dy))
        });

        group.bench_with_input(BenchmarkId::new("total_deformation", &label), &(), |b, _| {
            b.iter(|| dynamics::total_deformation(&u, &v, nx, ny, dx, dy))
        });
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// 5. Smoothing
// ---------------------------------------------------------------------------

fn bench_smoothing(c: &mut Criterion) {
    let mut group = c.benchmark_group("smoothing");

    for &(nx, ny) in &[(200, 200), (500, 500)] {
        let data = synthetic_grid(nx, ny, 300.0, 15.0);
        let label = format!("{}x{}", nx, ny);

        group.bench_with_input(BenchmarkId::new("smooth_gaussian/sigma2", &label), &(), |b, _| {
            b.iter(|| smooth::smooth_gaussian(&data, nx, ny, 2.0))
        });

        group.bench_with_input(BenchmarkId::new("smooth_gaussian/sigma5", &label), &(), |b, _| {
            b.iter(|| smooth::smooth_gaussian(&data, nx, ny, 5.0))
        });
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// 6. Wind
// ---------------------------------------------------------------------------

fn bench_wind(c: &mut Criterion) {
    let mut group = c.benchmark_group("wind");

    for &n in &[1000, 10000] {
        let u = synthetic_1d(n, 10.0, 5.0);
        let v = synthetic_1d(n, -3.0, 4.0);
        let speed = synthetic_1d(n, 15.0, 8.0);
        let direction = synthetic_1d(n, 225.0, 45.0);

        group.bench_with_input(BenchmarkId::new("wind_speed", n), &n, |b, _| {
            b.iter(|| dynamics::wind_speed(&u, &v))
        });

        group.bench_with_input(BenchmarkId::new("wind_direction", n), &n, |b, _| {
            b.iter(|| dynamics::wind_direction(&u, &v))
        });

        group.bench_with_input(BenchmarkId::new("wind_components", n), &n, |b, _| {
            b.iter(|| dynamics::wind_components(&speed, &direction))
        });
    }

    // Profile-based wind calcs
    let (u, v, z) = synthetic_wind_profile(100);

    group.bench_function("bulk_shear_100", |b| {
        b.iter(|| wind::bulk_shear(&u, &v, &z, 0.0, 6000.0))
    });

    group.bench_function("storm_relative_helicity_100", |b| {
        b.iter(|| wind::storm_relative_helicity(&u, &v, &z, 1000.0, 10.0, 5.0))
    });

    group.bench_function("bunkers_storm_motion_100", |b| {
        let p: Vec<f64> = z.iter().map(|h| 1013.25 * (1.0 - 0.0065 * h / 288.15_f64).powf(5.2561)).collect();
        b.iter(|| wind::bunkers_storm_motion(&p, &u, &v, &z))
    });

    group.bench_function("corfidi_storm_motion_100", |b| {
        b.iter(|| wind::corfidi_storm_motion(&u, &v, &z, black_box(10.0), black_box(-3.0)))
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// 7. Interpolation (1D profile interpolation via wx-math)
// ---------------------------------------------------------------------------

fn bench_interpolation(c: &mut Criterion) {
    let mut group = c.benchmark_group("interpolation");

    // Profile interpolation: interp_linear across sounding levels
    for &n in &[100, 1000, 10000] {
        let x = synthetic_1d(n, 0.0, 1.0);
        let y = synthetic_1d(n, 300.0, 15.0);

        group.bench_with_input(BenchmarkId::new("interp_linear_sweep", n), &n, |b, _| {
            b.iter(|| {
                // Interpolate at 50 target points across the range
                let targets: Vec<f64> = (0..50)
                    .map(|i| x[0] + (x[n - 1] - x[0]) * i as f64 / 49.0)
                    .collect();
                targets.iter().map(|&tgt| {
                    wx_thermo::interp_linear(tgt, x[0], x[n - 1], y[0], y[n - 1])
                }).collect::<Vec<f64>>()
            })
        });
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// 8. Severe weather parameters
// ---------------------------------------------------------------------------

fn bench_severe(c: &mut Criterion) {
    let mut group = c.benchmark_group("severe");

    // Scalar severe params
    group.bench_function("significant_tornado_parameter", |b| {
        b.iter(|| {
            severe::significant_tornado_parameter(
                black_box(2500.0), black_box(800.0),
                black_box(250.0), black_box(25.0),
            )
        })
    });

    group.bench_function("supercell_composite_parameter", |b| {
        b.iter(|| {
            severe::supercell_composite_parameter(
                black_box(3000.0), black_box(300.0), black_box(25.0),
            )
        })
    });

    group.bench_function("critical_angle", |b| {
        b.iter(|| {
            severe::critical_angle(
                black_box(10.0), black_box(5.0),
                black_box(3.0), black_box(-2.0),
                black_box(15.0), black_box(8.0),
            )
        })
    });

    // Batch: 1000 scalar STP calls
    let n = 1000;
    let cape = synthetic_1d(n, 2000.0, 1500.0);
    let lcl = synthetic_1d(n, 1000.0, 500.0);
    let srh = synthetic_1d(n, 200.0, 150.0);
    let shear = synthetic_1d(n, 25.0, 15.0);

    group.bench_function("stp_batch_1000", |b| {
        b.iter(|| {
            cape.iter().zip(lcl.iter().zip(srh.iter().zip(shear.iter())))
                .map(|(&c, (&l, (&s, &sh)))| severe::significant_tornado_parameter(c, l, s, sh))
                .collect::<Vec<f64>>()
        })
    });

    group.bench_function("scp_batch_1000", |b| {
        b.iter(|| {
            cape.iter().zip(srh.iter().zip(shear.iter()))
                .map(|(&c, (&s, &sh))| severe::supercell_composite_parameter(c, s, sh))
                .collect::<Vec<f64>>()
        })
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// Criterion harness
// ---------------------------------------------------------------------------

criterion_group!(
    benches,
    bench_scalar_thermo,
    bench_array_thermo,
    bench_cape_cin,
    bench_grid_kinematics,
    bench_smoothing,
    bench_wind,
    bench_interpolation,
    bench_severe,
);
criterion_main!(benches);
