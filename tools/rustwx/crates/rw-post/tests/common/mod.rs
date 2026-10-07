//! Shared test helpers: a synthetic but realistic frame.

#![allow(dead_code)]

use rw_post::hybrid::InterfaceCoefficients;
use rw_post::state::{MassInput, Shape, StateInputs};
use rw_post::surface::{SurfaceInput, SurfaceInputs};

/// Deterministic pseudo-random numbers in [0, 1) (64-bit LCG).
pub struct Lcg(pub u64);
impl Lcg {
    pub fn next(&mut self) -> f64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// A hybrid-coordinate frame with a standard-atmosphere-like column,
/// terrain from 0 to 3 km, moisture and condensate, and sheared winds.
pub fn synthetic_state(nx: usize, ny: usize, nz: usize, seed: u64) -> StateInputs {
    let shape = Shape { nx, ny, nz };
    let n = shape.ncell();
    let mut rng = Lcg(seed);
    let p_top = 5000.0f32;
    let znw: Vec<f32> = (0..=nz).map(|k| {
        let x = k as f64 / nz as f64;
        (1.0 - x.powf(1.4)) as f32
    }).collect();
    let coeff = InterfaceCoefficients::resolve(nz, None, &znw, 2, Some(0.2), p_top).unwrap();
    let mut mass = vec![0.0f32; MassInput::COUNT * shape.mass_volume()];
    let vol = shape.mass_volume();
    let mut ph = vec![0.0f32; shape.interface_volume()];
    let mut phb = vec![0.0f32; shape.interface_volume()];
    let mut mu = vec![0.0f32; n];
    let mut mub = vec![0.0f32; n];
    for c in 0..n {
        let terrain = 3000.0 * rng.next();
        let psfc = 101325.0 * (1.0 - 2.25577e-5 * terrain).powf(5.25588);
        let mud = psfc - p_top as f64 - 300.0 * rng.next();
        mub[c] = (mud * 0.97) as f32;
        mu[c] = (mud - mub[c] as f64) as f32;
        let mut z = terrain;
        let mut p_prev = psfc;
        for k in 0..=nz {
            let pd = coeff.c3f[k] as f64 * mud + coeff.c4f[k] as f64 + p_top as f64;
            if k > 0 {
                let tmean = (288.15 - 0.0065 * z).max(216.65);
                z += 287.0 * tmean / 9.81 * (p_prev / pd).ln();
            }
            p_prev = pd;
            let gz = 9.81 * z;
            phb[k * n + c] = (gz * 0.9) as f32;
            ph[k * n + c] = (gz - phb[k * n + c] as f64) as f32;
        }
        for k in 0..nz {
            let pd = 0.5 * ((coeff.c3f[k] + coeff.c3f[k + 1]) as f64 * mud + (coeff.c4f[k] + coeff.c4f[k + 1]) as f64) + p_top as f64;
            let zk = (ph[k * n + c] + phb[k * n + c]) as f64 / 9.81;
            let tk = (288.15 - 0.0065 * zk).max(216.65) + 4.0 * (rng.next() - 0.5);
            let theta = tk * (1e5 / pd).powf(2.0 / 7.0);
            let at = k * n + c;
            mass[MassInput::T as usize * vol + at] = (theta - 300.0) as f32;
            mass[MassInput::PB as usize * vol + at] = (pd * 0.98) as f32;
            mass[MassInput::P as usize * vol + at] = (pd - (pd * 0.98) as f32 as f64 + 20.0 * (rng.next() - 0.5)) as f32;
            let rv = 0.016 * (-(zk - terrain) / 2500.0).exp() * rng.next();
            mass[MassInput::QVapor as usize * vol + at] = rv as f32;
            mass[MassInput::QCloud as usize * vol + at] = if rng.next() > 0.7 { (1e-3 * rng.next()) as f32 } else { 0.0 };
            mass[MassInput::QRain as usize * vol + at] = if rng.next() > 0.8 { (2e-3 * rng.next()) as f32 } else { -1e-9 };
            mass[MassInput::QIce as usize * vol + at] = (1e-5 * rng.next()) as f32;
            mass[MassInput::QSnow as usize * vol + at] = (1e-4 * rng.next()) as f32;
            mass[MassInput::QGraupel as usize * vol + at] = 0.0;
        }
    }
    let present = (1 << MassInput::T as u32)
        | (1 << MassInput::P as u32)
        | (1 << MassInput::PB as u32)
        | (1 << MassInput::QVapor as u32)
        | (1 << MassInput::QCloud as u32)
        | (1 << MassInput::QRain as u32)
        | (1 << MassInput::QIce as u32)
        | (1 << MassInput::QSnow as u32)
        | (1 << MassInput::QGraupel as u32);
    let u_stag: Vec<f32> = (0..nz * ny * (nx + 1)).map(|_| (40.0 * rng.next() - 10.0) as f32).collect();
    let v_stag: Vec<f32> = (0..nz * (ny + 1) * nx).map(|_| (30.0 * rng.next() - 15.0) as f32).collect();
    StateInputs {
        shape,
        mass,
        present,
        ph,
        phb,
        u_stag,
        v_stag,
        mu,
        mub,
        c3f: coeff.c3f,
        c4f: coeff.c4f,
        p_top,
    }
}

/// Surface carriers matching a synthetic state.
pub fn synthetic_surface(st: &StateInputs, seed: u64) -> SurfaceInputs {
    let n = st.shape.ncell();
    let mut rng = Lcg(seed);
    let mut s = SurfaceInputs::new(n);
    let mut plane = |f: &mut dyn FnMut(usize, &mut Lcg) -> f32| -> Vec<f32> { (0..n).map(|c| f(c, &mut rng)).collect() };
    let hgt: Vec<f32> = (0..n).map(|c| (st.ph[c] + st.phb[c]) / 9.81).collect();
    let psfc: Vec<f32> = (0..n).map(|c| st.mu[c] + st.mub[c] + st.p_top + 150.0).collect();
    s.set(SurfaceInput::Hgt, &hgt);
    s.set(SurfaceInput::Psfc, &psfc);
    s.set(SurfaceInput::T2, &plane(&mut |_, r| (250.0 + 60.0 * r.next()) as f32));
    s.set(SurfaceInput::Th2, &plane(&mut |_, r| (255.0 + 60.0 * r.next()) as f32));
    s.set(SurfaceInput::Q2, &plane(&mut |c, r| if c % 53 == 0 { -1e-6 } else { (0.02 * r.next()) as f32 }));
    s.set(SurfaceInput::Tsk, &plane(&mut |_, r| (250.0 + 70.0 * r.next()) as f32));
    s.set(SurfaceInput::Snow, &plane(&mut |c, r| if c % 3 == 0 { (200.0 * r.next()) as f32 } else { 0.0 }));
    s.set(SurfaceInput::SnowH, &plane(&mut |_, r| (r.next() - 0.1) as f32));
    s.set(SurfaceInput::SnowC, &plane(&mut |c, _| if c % 3 == 0 { 1.0 } else { 0.0 }));
    s.set(SurfaceInput::Hfx, &plane(&mut |_, r| (600.0 * r.next() - 100.0) as f32));
    s.set(SurfaceInput::Lh, &plane(&mut |_, r| (700.0 * r.next() - 50.0) as f32));
    s.set(SurfaceInput::Qfx, &plane(&mut |_, r| (3e-4 * r.next() - 2e-5) as f32));
    s.set(SurfaceInput::GrdFlx, &plane(&mut |_, r| (200.0 * r.next() - 100.0) as f32));
    s.set(SurfaceInput::Ust, &plane(&mut |_, r| (1.5 * r.next()) as f32));
    s.set(SurfaceInput::Glw, &plane(&mut |_, r| (150.0 + 350.0 * r.next()) as f32));
    s.set(SurfaceInput::SwDown, &plane(&mut |_, r| (1000.0 * r.next()) as f32));
    s.set(SurfaceInput::VegFra, &plane(&mut |_, r| (100.0 * r.next()) as f32));
    s.set(SurfaceInput::IvgTyp, &plane(&mut |c, _| (1 + c % 21) as f32));
    s.set(SurfaceInput::SeaIce, &plane(&mut |_, r| r.next() as f32));
    s.set(SurfaceInput::Znt, &plane(&mut |_, r| (2.0 * r.next()) as f32));
    s.set(SurfaceInput::Lai, &plane(&mut |_, r| (7.0 * r.next()) as f32));
    s.set(SurfaceInput::XLat, &plane(&mut |_, r| (180.0 * r.next() - 90.0) as f32));
    s.set(SurfaceInput::XLong, &plane(&mut |_, r| (360.0 * r.next() - 180.0) as f32));
    s
}

/// Read a little-endian f32 file from tests/data.
pub fn data(name: &str) -> Vec<f32> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data").join(name);
    let bytes = std::fs::read(path).expect("fixture");
    bytes.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect()
}
