//
// Group C tests on a synthetic frame: the analytic convective sounding of
// Mon. Wea. Rev. 110 (1982) 504-520 (theta and relative humidity profiles
// with a moisture cap) with per-column surface perturbations and a curved
// hodograph.  CPU invariants, missing-value isolation, and (when a CUDA
// device is present) GPU against the f64 CPU reference plus run-to-run
// bitwise identity.

use rw_post::severe::{self, ColumnState, FIELDS, NPLANES, thermo};

struct Synth {
    nz: usize,
    n: usize,
    p: Vec<f32>,
    tk: Vec<f32>,
    r: Vec<f32>,
    z_mass: Vec<f32>,
    u: Vec<f32>,
    v: Vec<f32>,
    p_int: Vec<f32>,
    z_int: Vec<f32>,
    zsfc: Vec<f32>,
    psfc: Vec<f32>,
    t2: Vec<f32>,
    q2: Vec<f32>,
    p2: Vec<f32>,
    u10: Vec<f32>,
    v10: Vec<f32>,
}

impl Synth {
    fn state(&self) -> ColumnState<'_> {
        ColumnState {
            nz: self.nz,
            ncell: self.n,
            p: &self.p,
            tk: &self.tk,
            r: &self.r,
            z_mass: &self.z_mass,
            u: &self.u,
            v: &self.v,
            p_int: &self.p_int,
            z_int: &self.z_int,
            zsfc: &self.zsfc,
            psfc: &self.psfc,
            t2: &self.t2,
            q2: &self.q2,
            p2: &self.p2,
            u10: &self.u10,
            v10: &self.v10,
        }
    }
}

fn synth(n: usize) -> Synth {
    let nz = 50;
    let dz = 400.0;
    let mut s = Synth {
        nz,
        n,
        p: vec![0.0; nz * n],
        tk: vec![0.0; nz * n],
        r: vec![0.0; nz * n],
        z_mass: vec![0.0; nz * n],
        u: vec![0.0; nz * n],
        v: vec![0.0; nz * n],
        p_int: vec![0.0; (nz + 1) * n],
        z_int: vec![0.0; (nz + 1) * n],
        zsfc: vec![0.0; n],
        psfc: vec![0.0; n],
        t2: vec![0.0; n],
        q2: vec![0.0; n],
        p2: vec![0.0; n],
        u10: vec![0.0; n],
        v10: vec![0.0; n],
    };
    for c in 0..n {
        let f = c as f64 / n.max(2) as f64;
        let zsfc = 300.0 * f;
        let theta0 = 297.0 + 6.0 * f;
        let qcap = 0.010 + 0.006 * f;
        let shear = 10.0 + 25.0 * f;
        let prof = |z: f64| -> (f64, f64) {
            let ztr = 12_000.0;
            let th = if z <= ztr { theta0 + (343.0 - theta0) * (z / ztr).powf(1.25) } else { 343.0 * (thermo::G / (thermo::CP * 213.0) * (z - ztr)).exp() };
            let rh = if z <= ztr { 1.0 - 0.75 * (z / ztr).powf(1.25) } else { 0.25 };
            (th, rh)
        };
        // Hydrostatic Exner integration from the ground (1000 hPa at sea level adjusted for terrain).
        let mut pi = (1.0 - zsfc / 8000.0 * 0.12f64).max(0.5);
        let mut z = 0.0;
        let step: f64 = 10.0;
        let at = |zt: f64, pi: &mut f64, z: &mut f64| {
            while *z < zt - 1e-9 {
                let h = step.min(zt - *z);
                let (th, _) = prof(*z + 0.5 * h);
                *pi -= thermo::G / (thermo::CP * th) * h;
                *z += h;
            }
            thermo::P0 * pi.powf(1.0 / thermo::KAPPA)
        };
        let mut pint = vec![0.0; nz + 1];
        for k in 0..=nz {
            pint[k] = at(k as f64 * dz, &mut pi, &mut z);
        }
        // Re-integrate for mass levels (fresh integration so order is simple).
        let mut pi2 = (1.0 - zsfc / 8000.0 * 0.12f64).max(0.5);
        let mut z2 = 0.0;
        for k in 0..nz {
            let zm = (k as f64 + 0.5) * dz;
            let pk = at(zm, &mut pi2, &mut z2);
            let (th, rh) = prof(zm);
            let t = thermo::temperature_from_theta(th, pk);
            let r = (rh * thermo::rsat(t, pk)).min(qcap);
            let ang = zm / 6000.0 * 1.8;
            let i = k * n + c;
            s.p[i] = pk as f32;
            s.tk[i] = t as f32;
            s.r[i] = r as f32;
            s.z_mass[i] = (zsfc + zm) as f32;
            s.u[i] = (shear * (1.0 - ang.cos())) as f32;
            s.v[i] = (shear * ang.sin()) as f32;
        }
        for k in 0..=nz {
            s.p_int[k * n + c] = pint[k] as f32;
            s.z_int[k * n + c] = (zsfc + k as f64 * dz) as f32;
        }
        let (th0, _) = prof(2.0);
        let ps = pint[0];
        s.zsfc[c] = zsfc as f32;
        s.psfc[c] = ps as f32;
        let p2 = ps * (-thermo::G * 2.0 / (thermo::RD * 300.0)).exp();
        s.p2[c] = p2 as f32;
        s.t2[c] = (thermo::temperature_from_theta(th0 + 1.5 * f, p2)) as f32;
        s.q2[c] = (qcap * (0.95 + 0.1 * f)) as f32;
        let a10 = 10.0 / 6000.0 * 1.8;
        s.u10[c] = (shear * (1.0 - f64::cos(a10)) * 0.8) as f32;
        s.v10[c] = (shear * a10.sin() * 0.8) as f32;
    }
    s
}

#[test]
fn cpu_invariants_on_the_analytic_sounding() {
    let s = synth(64);
    let f = severe::compute_cpu(&s.state()).unwrap();
    let pl = |name| f.plane(name).unwrap();
    for c in 0..s.n {
        for (cape, cin) in [("sbcape", "sbcin"), ("mlcape", "mlcin"), ("mucape", "mucin"), ("cape_best180", "cin_best180")] {
            let (a, b) = (pl(cape)[c], pl(cin)[c]);
            assert!(a.is_finite() && a >= 0.0, "{cape}[{c}] = {a}");
            assert!(b.is_finite() && b <= 0.0, "{cin}[{c}] = {b}");
            if a == 0.0 {
                assert_eq!(b, 0.0);
            }
        }
        // Every column of this sounding is conditionally unstable.
        assert!(pl("mucape")[c] > 300.0, "mucape {c} {}", pl("mucape")[c]);
        assert!(pl("lcl_height")[c] >= 0.0 && pl("lcl_height")[c] < 4000.0);
        assert!(pl("bulk_shear_0_6km")[c] > 5.0);
        // Clockwise-turning hodograph: positive helicity for the right mover.
        assert!(pl("srh_0_3km")[c] > 0.0, "srh03 {c} {}", pl("srh_0_3km")[c]);
        let bulk = (pl("shear_u_0_6km")[c].powi(2) + pl("shear_v_0_6km")[c].powi(2)).sqrt();
        assert!((bulk - pl("bulk_shear_0_6km")[c]).abs() < 1e-3);
        let ehi = pl("sbcape")[c] as f64 * pl("srh_0_1km")[c] as f64 / 160_000.0;
        assert!((ehi - pl("ehi_0_1km")[c] as f64).abs() <= 1e-4 * ehi.abs().max(1.0));
    }
}

#[test]
fn moist_end_of_the_analytic_sounding_is_strongly_unstable() {
    // The moistest column (theta0 about 303 K, 16 g/kg cap) is close to the
    // published convective case of this sounding, a few thousand J/kg.
    let s = synth(64);
    let f = severe::compute_cpu(&s.state()).unwrap();
    let mu = f.plane("mucape").unwrap()[63];
    assert!(mu > 1500.0 && mu < 6000.0, "mucape {mu}");
    let sb = f.plane("sbcape").unwrap()[63];
    assert!(sb > 1000.0, "sbcape {sb}");
}

#[test]
fn missing_input_stays_in_its_cell() {
    let mut s = synth(16);
    s.tk[3 * s.n + 5] = f32::NAN;
    s.u[2 * s.n + 7] = f32::NAN;
    let f = severe::compute_cpu(&s.state()).unwrap();
    assert!(f.plane("sbcape").unwrap()[5].is_nan());
    assert!(f.plane("storm_motion_u").unwrap()[5].is_finite());
    assert!(f.plane("storm_motion_u").unwrap()[7].is_nan());
    assert!(f.plane("sbcape").unwrap()[7].is_finite());
    for c in [0usize, 4, 6, 8, 15] {
        for name in FIELDS {
            assert!(f.plane(name).unwrap()[c].is_finite(), "{name}[{c}]");
        }
    }
}

#[test]
fn gpu_matches_cpu_reference_and_is_deterministic() {
    // Skip only when no card is visible; with a card granted, any failure
    // to load or run the kernels fails the test.
    let granted = std::env::var("CUDA_VISIBLE_DEVICES").map(|v| !v.trim().is_empty()).unwrap_or(false);
    let gp = match rw_post::gpu::GpuPost::open(0) {
        Ok(g) => g,
        Err(e) if !granted => {
            eprintln!("skipping GPU test, no CUDA device visible: {e}");
            return;
        }
        Err(e) => panic!("a CUDA device is visible but the Group C kernels did not load: {e}"),
    };
    let g = severe::gpu::SevereGpu::new(&gp);
    let mut s = synth(4096);
    s.tk[3 * s.n + 11] = f32::NAN;
    let st = s.state();
    let cpu = severe::compute_cpu(&st).unwrap();
    let (a, _) = g.compute(&st).unwrap();
    let (b, _) = g.compute(&st).unwrap();
    assert!(a.planes.iter().zip(&b.planes).all(|(x, y)| x.to_bits() == y.to_bits()), "GPU run-to-run identity");
    for i in 0..NPLANES {
        let n = s.n;
        let (x, y) = (&cpu.planes[i * n..(i + 1) * n], &a.planes[i * n..(i + 1) * n]);
        for c in 0..n {
            let (p, q) = (x[c], y[c]);
            // One maths library on both devices: the same bits (D30).
            assert!(p.to_bits() == q.to_bits() || (p.is_nan() && q.is_nan()), "{}[{c}] cpu {p} gpu {q}", FIELDS[i]);
        }
    }
}
