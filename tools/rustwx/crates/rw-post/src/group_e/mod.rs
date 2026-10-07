//
// WOOF post-processor, group E: boundary layer, near-surface winds and
// isotherm levels (clean-room post specification, section 8).
//
// Clean-room implementation written only from that specification and the
// public sources it cites; see the crate documentation for the list.

pub mod column;
#[cfg(any(windows, target_os = "linux"))]
pub mod cuda;
pub mod history;

use rayon::prelude::*;

pub use column::{GustMethod, N_OUT, OUT_NAMES, Options};

/// Where the group E computation runs (spec 3.5, `--post-device`).
pub use crate::PostDevice;

/// Group E's inputs of one frame: Group A's shared column state (spec 3.4)
/// and the 2D carriers (f32, `[ny][nx]`), plus TKE when the history has it.
/// `None` = the carrier is absent from the history; absence is never zero.
#[derive(Clone, Copy, Debug)]
pub struct Carriers<'a> {
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    /// The shared column state (theta_v, T, pressure coordinate, mass
    /// height and destaggered winds are read from it).
    pub state: &'a crate::state::ColumnState,
    /// Turbulent kinetic energy per unit mass, [nz][ny][nx] (gust method b).
    pub tke: Option<&'a [f32]>,
    /// 2D [ny][nx].
    pub hgt: &'a [f32],
    pub psfc: &'a [f32],
    pub u10: &'a [f32],
    pub v10: &'a [f32],
    pub ust: Option<&'a [f32]>,
    pub t2: Option<&'a [f32]>,
    pub th2: Option<&'a [f32]>,
    pub q2: Option<&'a [f32]>,
    pub hfx: Option<&'a [f32]>,
    pub qfx: Option<&'a [f32]>,
    pub lh: Option<&'a [f32]>,
    pub sinalpha: Option<&'a [f32]>,
    pub cosalpha: Option<&'a [f32]>,
}

impl Carriers<'_> {
    #[must_use]
    pub fn ncell(&self) -> usize {
        self.nx * self.ny
    }

    /// Checks every array length against the grid.
    pub fn validate(&self) -> Result<(), String> {
        let n = self.ncell();
        let n3 = n * self.nz;
        if self.nz < 2 || n == 0 {
            return Err(format!("grid {}x{}x{} is too small", self.nx, self.ny, self.nz));
        }
        let s = self.state.shape;
        if (s.nx, s.ny, s.nz) != (self.nx, self.ny, self.nz) {
            return Err("the shared state and the carriers have different grids".to_owned());
        }
        let check = |name: &str, len: usize, want: usize| {
            if len == want { Ok(()) } else { Err(format!("{name} has {len} values, the grid needs {want}")) }
        };
        if let Some(a) = self.tke {
            check("TKE", a.len(), n3)?;
        }
        for (name, a) in [("HGT", self.hgt), ("PSFC", self.psfc), ("U10", self.u10), ("V10", self.v10)] {
            check(name, a.len(), n)?;
        }
        for (name, a) in [
            ("UST", self.ust),
            ("T2", self.t2),
            ("TH2", self.th2),
            ("Q2", self.q2),
            ("HFX", self.hfx),
            ("QFX", self.qfx),
            ("LH", self.lh),
            ("SINALPHA", self.sinalpha),
            ("COSALPHA", self.cosalpha),
        ] {
            if let Some(a) = a {
                check(name, a.len(), n)?;
            }
        }
        Ok(())
    }
}

/// The ten group E planes, f32 with NaN for missing, in `OUT_NAMES` order.
#[derive(Clone, Debug)]
pub struct Output {
    pub nx: usize,
    pub ny: usize,
    /// [N_OUT][ny][nx].
    pub planes: Vec<f32>,
    /// Fields the carriers cannot support, with the reason (manifest text).
    /// Their planes are all NaN.
    pub omitted: Vec<(&'static str, String)>,
    /// "cpu" or the GPU name and kernel image used.
    pub device: String,
}

impl Output {
    #[must_use]
    pub fn plane(&self, name: &str) -> Option<&[f32]> {
        let i = OUT_NAMES.iter().position(|n| *n == name)?;
        let n = self.nx * self.ny;
        Some(&self.planes[i * n..(i + 1) * n])
    }
}

/// Which outputs the carriers cannot support, and why.
#[must_use]
pub fn omissions(c: &Carriers<'_>, opt: Options) -> Vec<(&'static str, String)> {
    let mut v = Vec::new();
    if c.ust.is_none() {
        v.push(("pbl_height", "UST absent: the bulk Richardson height uses the b u*^2 term (spec D23)".to_owned()));
    }
    match opt.gust {
        GustMethod::Similarity => {
            let mut miss = Vec::new();
            if c.ust.is_none() {
                miss.push("UST");
            }
            if c.hfx.is_none() {
                miss.push("HFX");
            }
            if c.qfx.is_none() && c.lh.is_none() {
                miss.push("QFX or LH");
            }
            if c.q2.is_none() {
                miss.push("Q2");
            }
            if c.t2.is_none() && c.th2.is_none() {
                miss.push("T2 or TH2");
            }
            if !miss.is_empty() {
                v.push(("gust", format!("similarity gust needs {} (absent)", miss.join(", "))));
            }
        }
        GustMethod::MixedLayerTke => {
            if c.tke.is_none() || c.ust.is_none() || (c.t2.is_none() && c.th2.is_none()) {
                v.push(("gust", "mixed-layer gust needs TKE, UST (for the boundary-layer depth) and T2 or TH2".to_owned()));
            }
        }
    }
    if opt.earth_winds && (c.sinalpha.is_none() || c.cosalpha.is_none()) {
        for name in ["u80", "v80"] {
            v.push((name, "earth-relative winds requested but SINALPHA/COSALPHA absent".to_owned()));
        }
    }
    v
}

/// The slots of the shared column state (spec 3.4) group E reads, as f32
/// planes, level-major [k][cell].
#[derive(Clone, Debug)]
pub struct State {
    pub theta_v: Vec<f32>,
    pub tk: Vec<f32>,
    pub p: Vec<f32>,
    pub z: Vec<f32>,
    pub u: Vec<f32>,
    pub v: Vec<f32>,
}

/// Group E's view of Group A's state: virtual potential temperature,
/// temperature, the pressure coordinate (D1), mass height and winds.
#[must_use]
pub fn state_cpu(c: &Carriers<'_>) -> State {
    use crate::state::MassState as M;
    let st = c.state;
    State {
        theta_v: st.mass_slot(M::ThetaV).to_vec(),
        tk: st.mass_slot(M::Tk).to_vec(),
        p: st.mass_slot(M::P).to_vec(),
        z: st.mass_slot(M::ZMass).to_vec(),
        u: st.mass_slot(M::U).to_vec(),
        v: st.mass_slot(M::V).to_vec(),
    }
}

fn opt2d(a: Option<&[f32]>, cell: usize) -> Option<f32> {
    a.map(|x| x[cell])
}

/// CPU field computation from a built state, bitwise twin of
/// `woof_post_e_fields_v1`.
#[must_use]
pub fn fields_cpu(c: &Carriers<'_>, st: &State, opt: Options) -> Vec<f32> {
    let n = c.ncell();
    let nz = c.nz;
    let earth_ok = opt.earth_winds && c.sinalpha.is_some() && c.cosalpha.is_some();
    let per_cell: Vec<[f32; N_OUT]> = (0..n)
        .into_par_iter()
        .map(|cell| {
            let gather = |a: &[f32]| (0..nz).map(|k| a[k * n + cell]).collect::<Vec<f32>>();
            let col = column::Column {
                theta_v: gather(&st.theta_v),
                tk: gather(&st.tk),
                p: gather(&st.p),
                z: gather(&st.z),
                u: gather(&st.u),
                v: gather(&st.v),
                tke: c.tke.map(gather),
            };
            let s = column::Surface {
                hgt: c.hgt[cell],
                psfc: c.psfc[cell],
                u10: c.u10[cell],
                v10: c.v10[cell],
                ust: opt2d(c.ust, cell),
                t2: opt2d(c.t2, cell),
                th2: opt2d(c.th2, cell),
                q2: opt2d(c.q2, cell),
                hfx: opt2d(c.hfx, cell),
                qfx: opt2d(c.qfx, cell),
                lh: opt2d(c.lh, cell),
                sinalpha: if earth_ok { opt2d(c.sinalpha, cell) } else { None },
                cosalpha: if earth_ok { opt2d(c.cosalpha, cell) } else { None },
            };
            column::fields(&col, &s, Options { earth_winds: earth_ok, gust: opt.gust })
        })
        .collect();
    let mut planes = vec![f32::NAN; N_OUT * n];
    for (cell, v) in per_cell.into_iter().enumerate() {
        for (f, x) in v.into_iter().enumerate() {
            planes[f * n + cell] = x;
        }
    }
    planes
}

fn blank_omitted(planes: &mut [f32], n: usize, omitted: &[(&'static str, String)]) {
    for (name, _) in omitted {
        if let Some(i) = OUT_NAMES.iter().position(|x| x == name) {
            planes[i * n..(i + 1) * n].fill(f32::NAN);
        }
    }
}

/// Group E on the CPU (binary64 reference path).
pub fn compute_cpu(c: &Carriers<'_>, opt: Options) -> Result<Output, String> {
    c.validate()?;
    let st = state_cpu(c);
    let mut planes = fields_cpu(c, &st, opt);
    let omitted = omissions(c, opt);
    blank_omitted(&mut planes, c.ncell(), &omitted);
    Ok(Output { nx: c.nx, ny: c.ny, planes, omitted, device: "cpu".to_owned() })
}

/// Group E on the requested device.  `Auto` falls back to the CPU when no
/// opened GPU is given or the device path fails; `Gpu` fails instead.
#[cfg(any(windows, target_os = "linux"))]
pub fn compute(c: &Carriers<'_>, opt: Options, gpu: Option<&crate::gpu::GpuPost>, device: PostDevice) -> Result<Output, String> {
    match (device, gpu) {
        (PostDevice::Cpu, _) => compute_cpu(c, opt),
        (PostDevice::Gpu, Some(g)) => compute_gpu(c, opt, g),
        (PostDevice::Gpu, None) => Err("PostDevice::Gpu needs an opened GPU".to_owned()),
        (PostDevice::Auto, Some(g)) => match compute_gpu(c, opt, g) {
            Ok(o) => Ok(o),
            Err(_) => compute_cpu(c, opt),
        },
        (PostDevice::Auto, None) => compute_cpu(c, opt),
    }
}

/// Group E on an opened kernel set.
#[cfg(any(windows, target_os = "linux"))]
pub fn compute_gpu(c: &Carriers<'_>, opt: Options, g: &crate::gpu::GpuPost) -> Result<Output, String> {
    c.validate()?;
    let exec = cuda::GroupEExecutor::new(g);
    let (mut planes, _) = exec.run(c, opt)?;
    let omitted = omissions(c, opt);
    blank_omitted(&mut planes, c.ncell(), &omitted);
    Ok(Output { nx: c.nx, ny: c.ny, planes, omitted, device: exec.describe() })
}
