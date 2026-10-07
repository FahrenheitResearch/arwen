//
// Group E tests: analytic columns against the binary64 reference, the CUDA
// kernels bit for bit against the CPU path, the kernel manifest, and (when
// WOOF_POST_FIXTURES names the staged oracle directory) the distance to the
// black-box fixtures of the clean-room specification, section 9.2.

use rw_post::group_e::{
    self, Carriers, GustMethod, N_OUT, OUT_NAMES, Options,
    column::{self, Column, Surface},
};

fn col(z: &[f64], tk: &[f64], p: &[f64], thv: &[f64], u: &[f64], v: &[f64]) -> Column {
    let f = |a: &[f64]| a.iter().map(|&x| x as f32).collect::<Vec<f32>>();
    Column { theta_v: f(thv), tk: f(tk), p: f(p), z: f(z), u: f(u), v: f(v), tke: None }
}

fn surface(hgt: f64, psfc: f64) -> Surface {
    Surface { hgt: hgt as f32, psfc: psfc as f32, ..Surface::default() }
}

fn idx(name: &str) -> usize {
    OUT_NAMES.iter().position(|n| *n == name).unwrap()
}

/// A dry, linear-lapse column: T = t0 - lapse (z - zsfc) up to ztrop, then
/// isothermal; pressure exponential in height (ln p linear in z).
fn lapse_column(zsfc: f64, t0: f64, lapse: f64, ztrop: f64, dz: f64, nz: usize) -> (Column, Vec<f64>) {
    let mut z = Vec::new();
    let mut t = Vec::new();
    let mut p = Vec::new();
    for k in 0..nz {
        let zz = zsfc + 20.0 + dz * k as f64;
        let tt = if zz - zsfc < ztrop { t0 - lapse * (zz - zsfc) } else { t0 - lapse * ztrop };
        z.push(zz);
        t.push(tt);
        p.push(100_000.0 * (-(zz - zsfc) / 8000.0).exp());
    }
    let thv: Vec<f64> = t.iter().zip(&p).map(|(t, p)| t * (100_000.0 / p).powf(2.0 / 7.0)).collect();
    let u = vec![5.0; nz];
    let v = vec![0.0; nz];
    (col(&z, &t, &p, &thv, &u, &v), z)
}

#[test]
fn wind80_interpolates_in_height_and_from_the_10m_wind_below_the_lowest_level() {
    let z = [30.0, 100.0, 200.0, 400.0];
    let u: Vec<f64> = z.iter().map(|z| 2.0 + 0.05 * z).collect();
    let v: Vec<f64> = z.iter().map(|z| -1.0 - 0.01 * z).collect();
    let c = col(&z, &[280.0; 4], &[1e5; 4], &[290.0; 4], &u, &v);
    let (a, b) = column::wind80(&c, 0.0, 1.0, 1.0);
    assert!((a - 6.0).abs() < 1e-6 && (b + 1.8).abs() < 1e-6, "{a} {b}");
    // lowest level above 80 m AGL: blend from the 10 m wind (spec D25)
    let z = [150.0, 300.0, 500.0, 800.0];
    let c = col(&z, &[280.0; 4], &[1e5; 4], &[290.0; 4], &[9.5, 10.0, 11.0, 12.0], &[0.0; 4]);
    let (a, _) = column::wind80(&c, 0.0, 3.0, 0.0);
    assert!((a - 6.25).abs() < 1e-9, "{a}");
    // terrain offset
    let z = [1530.0, 1600.0, 1700.0];
    let c = col(&z, &[280.0; 3], &[1e5; 3], &[290.0; 3], &[1.0, 2.0, 3.0], &[0.0; 3]);
    let (a, _) = column::wind80(&c, 1500.0, 0.0, 0.0);
    assert!((a - (1.0 + 50.0 / 70.0)).abs() < 1e-6, "{a}");
    // no level reaches 80 m: missing (no extrapolation)
    let c = col(&[2.0, 6.0], &[280.0; 2], &[1e5; 2], &[290.0; 2], &[1.0, 2.0], &[0.0; 2]);
    assert!(column::wind80(&c, 0.0, 0.0, 0.0).0.is_nan());
}

#[test]
fn freezing_levels_on_a_linear_lapse_column_are_exact() {
    let (c, _) = lapse_column(100.0, 290.0, 0.0065, 11_000.0, 250.0, 80);
    let mut s = surface(100.0, 100_000.0);
    s.t2 = Some(290.0 - 0.0065 * 2.0);
    s.q2 = Some(0.0);
    let o = column::fields(&c, &s, Options::default());
    let zc = 100.0 + (290.0 - 273.15) / 0.0065;
    let pc = 100_000.0 * (-(zc - 100.0) / 8000.0_f64).exp();
    assert!((f64::from(o[idx("freezing_height")]) - zc).abs() < 0.01, "{}", o[idx("freezing_height")]);
    assert!((f64::from(o[idx("freezing_pressure")]) - pc).abs() < 0.05, "{} vs {pc}", o[idx("freezing_pressure")]);
    assert_eq!(o[idx("highest_freezing_height")], o[idx("freezing_height")]);
    let z10 = 100.0 + (290.0 - 263.15) / 0.0065;
    let z20 = 100.0 + (290.0 - 253.15) / 0.0065;
    assert!((f64::from(o[idx("minus10_height")]) - z10).abs() < 0.01);
    assert!((f64::from(o[idx("minus20_height")]) - z20).abs() < 0.01);
}

#[test]
fn cold_ground_and_a_warm_nose() {
    // ground below freezing, warm layer 1000-1500 m AGL, cold above
    let nz = 60;
    let mut z = Vec::new();
    let mut t = Vec::new();
    let mut p = Vec::new();
    for k in 0..nz {
        let zz = 20.0 + 100.0 * k as f64;
        let tt = if zz < 1000.0 { 271.0 + 0.005 * zz } else if zz < 1500.0 { 276.0 } else { 276.0 - 0.0065 * (zz - 1500.0) };
        z.push(zz);
        t.push(tt);
        p.push(100_000.0 * (-zz / 8000.0_f64).exp());
    }
    let c = col(&z, &t, &p, &t, &vec![5.0; nz], &vec![0.0; nz]);
    let mut s = surface(0.0, 100_000.0);
    s.t2 = Some(270.5);
    let o = column::fields(&c, &s, Options::default());
    assert_eq!(o[idx("freezing_height")], 0.0);
    assert_eq!(o[idx("freezing_pressure")], 100_000.0);
    let top = 1500.0 + (276.0 - 273.15) / 0.0065;
    assert!((f64::from(o[idx("highest_freezing_height")]) - top).abs() < 0.01, "{}", o[idx("highest_freezing_height")]);
    // warm ground: lowest crossing below the nose
    let mut s2 = s;
    s2.t2 = Some(275.0);
    let c2 = col(&z, &t.iter().map(|x| x + 4.0).collect::<Vec<_>>(), &p, &t, &vec![5.0; nz], &vec![0.0; nz]);
    let o2 = column::fields(&c2, &s2, Options::default());
    assert!(o2[idx("freezing_height")] > 1500.0);
}

#[test]
fn tropopause_caps_the_highest_crossings_and_ignores_low_inversions() {
    let (mut c, _) = lapse_column(0.0, 288.15, 0.0065, 11_000.0, 250.0, 81);
    let k = column::tropopause_level(&c);
    assert!((f64::from(c.z[k]) - 11_020.0).abs() < 1.0, "tropopause at {}", c.z[k]);
    // an isothermal layer near the ground (p > 550 hPa) is not a tropopause
    for kk in 0..6 {
        c.tk[kk] = 288.15;
    }
    let k2 = column::tropopause_level(&c);
    assert_eq!(k2, k);
    // a warm column whose -20 C crossing lies only above the tropopause (a
    // stratosphere cooling at 1.5 K/km): not reported
    let (mut c3, _) = lapse_column(0.0, 300.0, 0.0040, 11_000.0, 250.0, 81);
    for kk in 0..c3.z.len() {
        let h = f64::from(c3.z[kk]);
        if h > 11_000.0 {
            c3.tk[kk] = (256.0 - 0.0015 * (h - 11_000.0)) as f32;
        }
    }
    let kt = column::tropopause_level(&c3);
    assert!((f64::from(c3.z[kt]) - 11_020.0).abs() < 1.0, "tropopause at {}", c3.z[kt]);
    let mut s = surface(0.0, 100_000.0);
    s.t2 = Some(300.0);
    let o = column::fields(&c3, &s, Options::default());
    assert!(o[idx("minus20_height")].is_nan(), "{}", o[idx("minus20_height")]);
    assert!(o[idx("minus10_height")].is_finite());
}

/// Independent naive bulk-Richardson height (written separately from
/// column.rs, std math) for the analytic check.
fn naive_pbl(z: &[f64], thv: &[f64], u: &[f64], ust: f64, zsfc: f64) -> f64 {
    let ri = |k: usize| -> f64 {
        let d = (u[k] - u[0]).powi(2) + 100.0 * ust * ust;
        9.81 / thv[0] * (thv[k] - thv[0]) * (z[k] - z[0]) / d.max(1e-10)
    };
    for k in 1..z.len() {
        if ri(k) >= 0.25 {
            let r0 = if k == 1 { 0.0 } else { ri(k - 1) };
            return z[k - 1] + (0.25 - r0) / (ri(k) - r0) * (z[k] - z[k - 1]) - zsfc;
        }
    }
    f64::NAN
}

#[test]
fn bulk_richardson_height_matches_an_independent_reference() {
    let nz = 40;
    let z: Vec<f64> = (0..nz).map(|k| 210.0 + 60.0 * k as f64).collect();
    let thv: Vec<f64> = z.iter().map(|&z| if z < 1400.0 { 301.0 } else { 301.0 + 0.008 * (z - 1400.0) }).collect();
    let u: Vec<f64> = z.iter().map(|&z| 3.0 + 0.002 * z).collect();
    let c = col(&z, &thv, &vec![9e4; nz], &thv, &u, &vec![0.0; nz]);
    let h = column::pbl_height(&c, 200.0, Some(0.35));
    let z32: Vec<f64> = c.z.iter().map(|&x| f64::from(x)).collect();
    let t32: Vec<f64> = c.theta_v.iter().map(|&x| f64::from(x)).collect();
    let u32_: Vec<f64> = c.u.iter().map(|&x| f64::from(x)).collect();
    let want = naive_pbl(&z32, &t32, &u32_, f64::from(0.35_f32), 200.0);
    assert!(h > 1200.0 && h < 1600.0, "{h}");
    assert!((h - want).abs() < 1e-6, "{h} vs {want}");
    // no UST: missing, never treated as zero
    assert!(column::pbl_height(&c, 200.0, None).is_nan());
    // neutral, no crossing: missing
    let c2 = col(&z, &vec![300.0; nz], &vec![9e4; nz], &vec![300.0; nz], &u, &vec![0.0; nz]);
    assert!(column::pbl_height(&c2, 200.0, Some(0.3)).is_nan());
}

#[test]
fn similarity_gust_neutral_and_convective() {
    let (c, _) = lapse_column(0.0, 300.0, 0.0065, 11_000.0, 100.0, 60);
    let mut s = surface(0.0, 100_000.0);
    s.u10 = 6.0;
    s.v10 = 8.0;
    s.ust = Some(0.5);
    s.t2 = Some(300.0);
    s.th2 = Some(300.0);
    s.q2 = Some(0.01);
    s.hfx = Some(0.0);
    s.qfx = Some(0.0);
    let o = column::fields(&c, &s, Options::default());
    assert!((f64::from(o[idx("gust")]) - (10.0 + 7.71 * 0.5)).abs() < 1e-5, "{}", o[idx("gust")]);
    // convective: (u*^3 + zi kappa g w'thv' / (24 thv))^(1/3) = u* (1 - zi / (24 L))^(1/3)
    s.hfx = Some(300.0);
    let o = column::fields(&c, &s, Options::default());
    let zi = f64::from(o[idx("pbl_height")]);
    let r = 0.01_f64;
    let vf = (1.0 + r / (287.0 / 461.6)) / (1.0 + r);
    let rho = 100_000.0 / (287.0 * 300.0 * vf);
    let wthv = 300.0 / (rho * 1004.5) * (1.0 + (461.6 / 287.0 - 1.0) * r / (1.0 + r));
    let thv = 300.0 * vf;
    let l = -(0.5_f64.powi(3)) * thv / (0.4 * 9.81 * wthv);
    let want = 10.0 + 7.71 * 0.5 * (1.0 - zi / (24.0 * l)).powf(1.0 / 3.0);
    if zi.is_finite() {
        assert!((f64::from(o[idx("gust")]) - want).abs() < 1e-4, "{} vs {want}", o[idx("gust")]);
    }
    // missing flux carrier: missing gust
    s.hfx = None;
    assert!(column::fields(&c, &s, Options::default())[idx("gust")].is_nan());
}

// ---------------------------------------------------------------------------
// Device identity
// ---------------------------------------------------------------------------

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
    }
}

struct Frame {
    nx: usize,
    ny: usize,
    nz: usize,
    v: std::collections::BTreeMap<&'static str, Vec<f32>>,
}

/// A deterministic synthetic WRF-like frame with varied regimes: warm and
/// cold columns, terrain, stable and convective layers, some missing cells.
fn synthetic_frame(nx: usize, ny: usize, nz: usize) -> Frame {
    let mut r = Lcg(12345);
    let n = nx * ny;
    let mut m = std::collections::BTreeMap::new();
    let mut hgt = vec![0f32; n];
    let mut ph = vec![0f32; n * (nz + 1)];
    let phb = vec![0f32; n * (nz + 1)];
    let mut t = vec![0f32; n * nz];
    let mut p = vec![0f32; n * nz];
    let pb = vec![0f32; n * nz];
    let mut qv = vec![0f32; n * nz];
    let mut tke = vec![0f32; n * nz];
    let mut u = vec![0f32; nz * ny * (nx + 1)];
    let mut v = vec![0f32; nz * (ny + 1) * nx];
    let mut t2 = vec![0f32; n];
    let mut q2 = vec![0f32; n];
    let mut psfc = vec![0f32; n];
    for j in 0..ny {
        for i in 0..nx {
            let c = j * nx + i;
            let zs = 1500.0 * r.next() * (i as f64 / nx as f64);
            hgt[c] = zs as f32;
            let tsfc = 255.0 + 45.0 * (j as f64 / ny as f64) + 3.0 * r.next();
            let lapse = 0.004 + 0.005 * r.next();
            let ztrop = 9000.0 + 6000.0 * r.next();
            let mut zi = zs;
            for k in 0..=nz {
                ph[k * n + c] = (zi * 9.81) as f32;
                if k < nz {
                    let dz = 30.0 + 40.0 * k as f64 + 5.0 * r.next();
                    let zm = zi + 0.5 * dz;
                    let h = zm - zs;
                    let mut tk = if h < ztrop { tsfc - lapse * h } else { tsfc - lapse * ztrop };
                    if (i + j) % 7 == 0 && h > 800.0 && h < 1300.0 {
                        tk += 8.0; // elevated warm nose
                    }
                    let pp = 101_325.0 * (-(h + zs) / 8200.0).exp();
                    let theta = tk * (100_000.0 / pp).powf(2.0 / 7.0);
                    t[k * n + c] = (theta - 300.0) as f32;
                    p[k * n + c] = pp as f32;
                    qv[k * n + c] = (0.012 * (-h / 2500.0).exp() * r.next()) as f32;
                    tke[k * n + c] = if h < 1500.0 { (2.0 * r.next()) as f32 } else { 0.01 };
                    zi += dz;
                }
            }
            t2[c] = (tsfc + 1.0 * (r.next() - 0.5)) as f32;
            q2[c] = (0.01 * r.next()) as f32;
            psfc[c] = (101_325.0 * (-zs / 8200.0).exp()) as f32;
        }
    }
    for x in u.iter_mut() {
        *x = (20.0 * r.next() - 5.0) as f32;
    }
    for x in v.iter_mut() {
        *x = (20.0 * r.next() - 10.0) as f32;
    }
    let mk2 = |r: &mut Lcg, a: f64, b: f64| (0..n).map(|_| (a + b * r.next()) as f32).collect::<Vec<f32>>();
    let ust = mk2(&mut r, 0.0, 0.8);
    let hfx = mk2(&mut r, -50.0, 400.0);
    let qfx = mk2(&mut r, 0.0, 2e-4);
    let u10 = mk2(&mut r, -8.0, 16.0);
    let v10 = mk2(&mut r, -8.0, 16.0);
    let sa = mk2(&mut r, -0.3, 0.6);
    let ca: Vec<f32> = sa.iter().map(|s| (1.0 - s * s).sqrt()).collect();
    // missing cells
    let mut t2m = t2.clone();
    t2m[3] = f32::NAN;
    let mut ustm = ust.clone();
    ustm[5] = f32::NAN;
    t[2 * n + 7] = f32::NAN;
    m.insert("T", t);
    m.insert("P", p);
    m.insert("PB", pb);
    m.insert("QVAPOR", qv);
    m.insert("PH", ph);
    m.insert("PHB", phb);
    m.insert("U", u);
    m.insert("V", v);
    m.insert("TKE", tke);
    m.insert("HGT", hgt);
    m.insert("PSFC", psfc);
    m.insert("U10", u10);
    m.insert("V10", v10);
    m.insert("UST", ustm);
    m.insert("T2", t2m);
    m.insert("Q2", q2);
    m.insert("HFX", hfx);
    m.insert("QFX", qfx);
    m.insert("SINALPHA", sa);
    m.insert("COSALPHA", ca);
    Frame { nx, ny, nz, v: m }
}

impl Frame {
    /// Group A's shared state of the synthetic frame.  The synthetic pressure
    /// is stored as P_HYD, so the state's pressure coordinate is exactly it.
    fn state(&self) -> rw_post::state::ColumnState {
        use rw_post::state::{MassInput, Shape, StateInputs, build_cpu};
        let shape = Shape { nx: self.nx, ny: self.ny, nz: self.nz };
        let v = shape.mass_volume();
        let mut mass = vec![0.0f32; MassInput::COUNT * v];
        let mut present = 0u32;
        for (slot, key) in [(MassInput::T, "T"), (MassInput::P, "P"), (MassInput::PB, "PB"), (MassInput::QVapor, "QVAPOR"), (MassInput::PHyd, "P")] {
            mass[slot as usize * v..(slot as usize + 1) * v].copy_from_slice(&self.v[key]);
            present |= 1 << slot as u32;
        }
        let n = shape.ncell();
        let inp = StateInputs {
            shape,
            mass,
            present,
            ph: self.v["PH"].clone(),
            phb: self.v["PHB"].clone(),
            u_stag: self.v["U"].clone(),
            v_stag: self.v["V"].clone(),
            mu: vec![0.0; n],
            mub: vec![0.0; n],
            c3f: vec![0.0; self.nz + 1],
            c4f: vec![0.0; self.nz + 1],
            p_top: 0.0,
        };
        build_cpu(&inp).unwrap()
    }

    fn carriers<'a>(&'a self, state: &'a rw_post::state::ColumnState) -> Carriers<'a> {
        let g = |k: &str| self.v[k].as_slice();
        Carriers {
            nx: self.nx,
            ny: self.ny,
            nz: self.nz,
            state,
            tke: Some(g("TKE")),
            hgt: g("HGT"),
            psfc: g("PSFC"),
            u10: g("U10"),
            v10: g("V10"),
            ust: Some(g("UST")),
            t2: Some(g("T2")),
            th2: None,
            q2: Some(g("Q2")),
            hfx: Some(g("HFX")),
            qfx: Some(g("QFX")),
            lh: None,
            sinalpha: Some(g("SINALPHA")),
            cosalpha: Some(g("COSALPHA")),
        }
    }
}

#[test]
fn synthetic_frame_produces_every_field_somewhere() {
    let f = synthetic_frame(31, 19, 35);
    let st = f.state();
    let c = f.carriers(&st);
    let o = group_e::compute_cpu(&c, Options::default()).unwrap();
    assert!(o.omitted.is_empty(), "{:?}", o.omitted);
    for name in OUT_NAMES {
        let p = o.plane(name).unwrap();
        let valid = p.iter().filter(|x| x.is_finite()).count();
        assert!(valid > p.len() / 2, "{name}: only {valid} of {} valid", p.len());
    }
}

#[cfg(any(windows, target_os = "linux"))]
#[test]
fn gpu_is_bitwise_identical_to_the_cpu_reference() {
    // Skip only when no card is visible; with a card granted, any failure
    // to load or run the kernels fails the test (audit item 9).
    let granted = std::env::var("CUDA_VISIBLE_DEVICES").map(|v| !v.trim().is_empty()).unwrap_or(false);
    let g = match rw_post::gpu::GpuPost::open(0) {
        Ok(g) => g,
        Err(e) if !granted => {
            eprintln!("skipping GPU test, no CUDA device visible: {e}");
            return;
        }
        Err(e) => panic!("a CUDA device is visible but the post kernels did not load: {e}"),
    };
    let f = synthetic_frame(67, 41, 50);
    let st = f.state();
    let c = f.carriers(&st);
    let exec = group_e::cuda::GroupEExecutor::new(&g);
    for opt in [
        Options::default(),
        Options { earth_winds: true, gust: GustMethod::MixedLayerTke },
    ] {
        let cpu = group_e::compute_cpu(&c, opt).unwrap();
        let n = c.ncell();
        let (gpu, _) = exec.run(&c, opt).unwrap();
        for (i, name) in OUT_NAMES.iter().enumerate().take(N_OUT) {
            let a = &cpu.planes[i * n..(i + 1) * n];
            let b = &gpu[i * n..(i + 1) * n];
            let d = a.iter().zip(b).filter(|(x, y)| x.to_bits() != y.to_bits()).count();
            assert_eq!(d, 0, "{opt:?} {name}: {d} of {n} cells differ");
        }
        // repeatability
        let (again, _) = exec.run(&c, opt).unwrap();
        assert!(gpu.iter().zip(&again).all(|(x, y)| x.to_bits() == y.to_bits()));
    }
}

// ---------------------------------------------------------------------------
// Black-box fixture distance (spec 9.2).  Needs WOOF_POST_FIXTURES.
// ---------------------------------------------------------------------------

fn fixture(name: &str) -> Option<Vec<f32>> {
    let dir = std::env::var("WOOF_POST_FIXTURES").ok()?;
    let b = std::fs::read(std::path::Path::new(&dir).join(name)).ok()?;
    Some(b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
}

/// The fixtures mark an unavailable value with NaN, 1e20 or 9.99e20.
fn om(x: f32) -> f64 {
    if x.is_nan() || x.abs() >= 1.0e19 { f64::NAN } else { f64::from(x) }
}

#[test]
#[ignore = "needs WOOF_POST_FIXTURES (staged black-box fixtures, SPEC 9.2); run by the gate with --include-ignored"]
fn fixture_wind80() {
    let (Some(inp), Some(outp)) = (fixture("wind80-inputs.f32"), fixture("wind80-outputs.f32")) else {
        panic!("WOOF_POST_FIXTURES is not set or lacks this fixture (audit item 9: no silent pass)");
    };
    let mut worst: f64 = 0.0;
    for case in 0..12 {
        let w = &inp[case * 31..(case + 1) * 31];
        let o = &outp[case * 14..(case + 1) * 14];
        let zs = f64::from(w[0]);
        let z: Vec<f64> = w[1..7].iter().map(|&x| f64::from(x)).collect();
        let u: Vec<f64> = (0..6).map(|k| 0.5 * (f64::from(w[7 + 2 * k]) + f64::from(w[8 + 2 * k]))).collect();
        let v: Vec<f64> = (0..6).map(|k| 0.5 * (f64::from(w[19 + 2 * k]) + f64::from(w[20 + 2 * k]))).collect();
        let c = col(&z, &[280.0; 6], &[1e5; 6], &[290.0; 6], &u, &v);
        // the fixture carries no 10 m wind
        let (a, b) = column::wind80(&c, zs, f64::NAN, f64::NAN);
        let (ra, rb) = (om(o[12]), om(o[13]));
        let below = z[0] - zs > 80.0;
        let above = z[5] - zs < 80.0;
        let d = (a - ra).abs().max((b - rb).abs());
        println!(
            "wind80 case {case:2}: ours ({a:9.4}, {b:9.4}) oracle ({ra:9.4}, {rb:9.4}) {}",
            if below {
                "below lowest level: oracle extrapolates, ours needs U10 (D25)".to_owned()
            } else if above {
                "80 m above the top level: oracle extrapolates, ours missing".to_owned()
            } else {
                format!("|d| = {d:.2e}")
            }
        );
        if !below && !above {
            worst = worst.max(d);
        }
    }
    assert!(worst < 1e-4, "wind80 graded cases differ by {worst}");
}

#[test]
#[ignore = "needs WOOF_POST_FIXTURES (staged black-box fixtures, SPEC 9.2); run by the gate with --include-ignored"]
fn fixture_pbl_height() {
    let (Some(inp), Some(outp)) = (fixture("pbl-inputs.f32"), fixture("pbl-outputs.f32")) else {
        panic!("WOOF_POST_FIXTURES is not set or lacks this fixture (audit item 9: no silent pass)");
    };
    let h: Vec<u32> = inp[..4].iter().map(|x| x.to_bits()).collect();
    let (nx, ny, nz, nc) = (h[0] as usize, h[1] as usize, h[2] as usize, h[3] as usize);
    let n = nx * ny;
    let per = 6 * n * nz + n;
    let mut all = Vec::new();
    for case in 0..nc {
        let d = &inp[4 + case * per..4 + (case + 1) * per];
        let vol = |q: usize, k: usize, cell: usize| f64::from(d[q * n * nz + k * n + cell]);
        let mut diffs = Vec::new();
        for j in 1..ny - 1 {
            for i in 1..nx - 1 {
                let cell = j * nx + i;
                // top-down -> bottom-up
                let lv = |q: usize| (0..nz).rev().map(|k| vol(q, k, cell)).collect::<Vec<f64>>();
                let (p, t, q, u, v, z) = (lv(0), lv(1), lv(2), lv(3), lv(4), lv(5));
                let thv: Vec<f64> = (0..nz)
                    .map(|k| {
                        let r = q[k] / (1.0 - q[k]);
                        t[k] * (100_000.0 / p[k]).powf(2.0 / 7.0) * (1.0 + r / (287.0 / 461.6)) / (1.0 + r)
                    })
                    .collect();
                let c = col(&z, &t, &p, &thv, &u, &v);
                let zs = f64::from(d[6 * n * nz + cell]) / 9.81;
                // the fixture carries no u*: run with u* = 0
                let ours = column::pbl_height(&c, zs, Some(0.0));
                let ref_ = om(outp[case * n + cell]);
                diffs.push((ours, ref_));
            }
        }
        let both: Vec<f64> = diffs.iter().filter(|(a, b)| a.is_finite() && b.is_finite()).map(|(a, b)| a - b).collect();
        let only_ours = diffs.iter().filter(|(a, b)| a.is_finite() && !b.is_finite()).count();
        let only_ref = diffs.iter().filter(|(a, b)| !a.is_finite() && b.is_finite()).count();
        let maxd = both.iter().fold(0.0_f64, |m, x| m.max(x.abs()));
        let rms = (both.iter().map(|x| x * x).sum::<f64>() / both.len().max(1) as f64).sqrt();
        println!(
            "pbl case {case}: centre ours {:8.1} oracle {:8.1}; interior both={} max|d|={maxd:.1} rms={rms:.1} only-ours={only_ours} only-oracle={only_ref}",
            diffs[diffs.len() / 2].0,
            diffs[diffs.len() / 2].1,
            both.len()
        );
        all.extend(both);
    }
    let rms = (all.iter().map(|x| x * x).sum::<f64>() / all.len().max(1) as f64).sqrt();
    println!("pbl all interior cells: n={} rms={rms:.1} m", all.len());
}

#[test]
#[ignore = "needs WOOF_POST_FIXTURES (staged black-box fixtures, SPEC 9.2); run by the gate with --include-ignored"]
fn fixture_isotherm_control() {
    let (Some(inp), Some(outp)) = (fixture("isotherm-control-inputs.f32"), fixture("isotherm-control-outputs.f32")) else {
        panic!("WOOF_POST_FIXTURES is not set or lacks this fixture (audit item 9: no silent pass)");
    };
    let names = [
        "freezing_height",
        "freezing_pressure",
        "highest_freezing_height",
        "highest_freezing_pressure",
        "minus10_height",
        "minus10_pressure",
        "minus20_height",
        "minus20_pressure",
    ];
    let mut diffs: Vec<Vec<(f64, f64)>> = vec![Vec::new(); 8];
    for rec in 0..24 {
        let w = &inp[rec * 285..(rec + 1) * 285];
        let th2 = w[0];
        let r2 = w[1];
        let hgt = f64::from(w[2]);
        let p: Vec<f64> = w[3..59].iter().map(|&x| f64::from(x)).collect();
        let t: Vec<f64> = w[59..115].iter().map(|&x| f64::from(x)).collect();
        let pint0 = f64::from(w[115]);
        let z: Vec<f64> = w[229..285].iter().map(|&x| f64::from(x)).collect();
        let c = col(&z, &t, &p, &t, &[0.0; 56], &[0.0; 56]);
        let mut s = surface(hgt, pint0);
        if om(th2).is_finite() {
            s.th2 = Some(th2);
        }
        if om(r2).is_finite() {
            s.q2 = Some(r2);
        }
        let o = column::fields(&c, &s, Options::default());
        // our -10/-20 pressures are not catalog fields; recompute them for the table
        let ours = [
            f64::from(o[idx("freezing_height")]),
            f64::from(o[idx("freezing_pressure")]),
            f64::from(o[idx("highest_freezing_height")]),
            f64::from(o[idx("highest_freezing_pressure")]),
            f64::from(o[idx("minus10_height")]),
            f64::NAN,
            f64::from(o[idx("minus20_height")]),
            f64::NAN,
        ];
        let base = rec * 72;
        for pl in 0..8 {
            let r = om(outp[base + pl * 9 + 4]);
            diffs[pl].push((ours[pl], r));
        }
    }
    for (pl, name) in names.iter().enumerate() {
        let both: Vec<f64> = diffs[pl].iter().filter(|(a, b)| a.is_finite() && b.is_finite()).map(|(a, b)| a - b).collect();
        if both.is_empty() {
            println!("isotherm {name}: not produced by ours or by oracle");
            continue;
        }
        let maxd = both.iter().fold(0.0_f64, |m, x| m.max(x.abs()));
        let rms = (both.iter().map(|x| x * x).sum::<f64>() / both.len() as f64).sqrt();
        let only_ours = diffs[pl].iter().filter(|(a, b)| a.is_finite() && !b.is_finite()).count();
        let only_ref = diffs[pl].iter().filter(|(a, b)| !a.is_finite() && b.is_finite()).count();
        println!("isotherm {name}: n={} max|d|={maxd:.2} rms={rms:.2} only-ours={only_ours} only-oracle={only_ref}", both.len());
        for (rec, (a, b)) in diffs[pl].iter().enumerate() {
            if (a - b).abs() > 1.0 || a.is_finite() != b.is_finite() {
                println!("    record {rec:2}: ours {a:10.2} oracle {b:10.2}");
            }
        }
    }
}
