//
// WOOF post-processor, group E: CUDA executor.
//
// Launches the group E field kernel of the one kernel set
// (kernels/group_e.cu inside kernels/woof_post.cu) on a
// [`crate::gpu::GpuPost`].  The column state is Group A's shared state.

use cudarc::driver::{CudaSlice, LaunchConfig, PushKernelArg};

use super::{Carriers, GustMethod, N_OUT, Options};
use crate::gpu::GpuPost;

pub const FIELDS_KERNEL: &str = "woof_post_e_fields_v1";
const THREADS: u32 = 128;

// Flag bits shared with kernels/group_e.cu.
const WF_UST: i32 = 1;
const WF_T2: i32 = 2;
const WF_TH2: i32 = 4;
const WF_Q2: i32 = 8;
const WF_HFX: i32 = 16;
const WF_QFX: i32 = 32;
const WF_LH: i32 = 64;
const WF_TKE: i32 = 128;
const WF_EARTH: i32 = 256;
const WF_ROT: i32 = 512;
const WF_GUST_B: i32 = 1024;

fn derr(what: &str, e: cudarc::driver::DriverError) -> String {
    format!("CUDA {what}: {e}")
}

pub struct GroupEExecutor<'g> {
    g: &'g GpuPost,
}

/// Timings of one run, seconds.
#[derive(Clone, Copy, Debug, Default)]
pub struct Timing {
    pub upload: f64,
    pub compute: f64,
    pub download: f64,
}

impl<'g> GroupEExecutor<'g> {
    pub fn new(g: &'g GpuPost) -> Self {
        Self { g }
    }

    #[must_use]
    pub fn describe(&self) -> String {
        self.g.device.describe()
    }

    fn up(&self, a: &[f32]) -> Result<CudaSlice<f32>, String> {
        self.g.stream().clone_htod(a).map_err(|e| derr("upload", e))
    }

    fn up_opt(&self, a: Option<&[f32]>) -> Result<Option<CudaSlice<f32>>, String> {
        a.map(|x| self.up(x)).transpose()
    }

    /// Runs the state and field kernels; returns [N_OUT][cell] planes.
    pub fn run(&self, c: &Carriers<'_>, opt: Options) -> Result<(Vec<f32>, Timing), String> {
        let n = c.ncell();
        let n3 = n * c.nz;
        let nz = i32::try_from(c.nz).map_err(|_| "nz too large")?;
        let ncell = i32::try_from(n).map_err(|_| "grid too large")?;
        let null: u64 = 0;
        // Device bytes this frame needs: six state volumes, 2D carriers, outputs.
        let floats = 6 * n3 + c.tke.map_or(0, <[f32]>::len) + 13 * n + N_OUT * n;
        let need = floats * std::mem::size_of::<f32>();
        let (free, _) = self.g.memory().map_err(|e| format!("CUDA memory query: {e}"))?;
        if need + need / 8 > free {
            return Err(format!("CUDA device has {free} bytes free, the frame needs about {need}"));
        }
        let t0 = std::time::Instant::now();

        // Group A's shared state: the six slots group E reads.
        let st = super::state_cpu(c);
        let thv = self.up(&st.theta_v)?;
        let tk = self.up(&st.tk)?;
        let pc = self.up(&st.p)?;
        let zm = self.up(&st.z)?;
        let um = self.up(&st.u)?;
        let vm = self.up(&st.v)?;
        let tke = self.up_opt(c.tke)?;
        let hgt = self.up(c.hgt)?;
        let psfc = self.up(c.psfc)?;
        let u10 = self.up(c.u10)?;
        let v10 = self.up(c.v10)?;
        let ust = self.up_opt(c.ust)?;
        let t2 = self.up_opt(c.t2)?;
        let th2 = self.up_opt(c.th2)?;
        let q2 = self.up_opt(c.q2)?;
        let hfx = self.up_opt(c.hfx)?;
        let qfx = self.up_opt(c.qfx)?;
        let lh = self.up_opt(c.lh)?;
        let earth = opt.earth_winds && c.sinalpha.is_some() && c.cosalpha.is_some();
        let sa = if earth { self.up_opt(c.sinalpha)? } else { None };
        let ca = if earth { self.up_opt(c.cosalpha)? } else { None };
        let stream = self.g.stream();
        let mut out = stream.alloc_zeros::<f32>(N_OUT * n).map_err(|e| derr("alloc", e))?;
        stream.synchronize().map_err(|e| derr("sync", e))?;
        let t1 = std::time::Instant::now();

        let cfg = LaunchConfig {
            grid_dim: ((n as u32).div_ceil(THREADS), 1, 1),
            block_dim: (THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut flags = 0_i32;
        let mut set = |present: bool, bit: i32| {
            if present {
                flags |= bit;
            }
        };
        set(c.ust.is_some(), WF_UST);
        set(c.t2.is_some(), WF_T2);
        set(c.th2.is_some(), WF_TH2);
        set(c.q2.is_some(), WF_Q2);
        set(c.hfx.is_some(), WF_HFX);
        set(c.qfx.is_some(), WF_QFX);
        set(c.lh.is_some(), WF_LH);
        set(c.tke.is_some(), WF_TKE);
        set(earth, WF_EARTH | WF_ROT);
        set(opt.gust == GustMethod::MixedLayerTke, WF_GUST_B);
        {
            let f = self.g.function(FIELDS_KERNEL).map_err(|e| format!("CUDA resolve {FIELDS_KERNEL}: {e}"))?;
            let mut l = stream.launch_builder(f);
            l.arg(&thv).arg(&tk).arg(&pc).arg(&zm).arg(&um).arg(&vm);
            for a in [&tke] {
                match a {
                    Some(a) => l.arg(a),
                    None => l.arg(&null),
                };
            }
            l.arg(&hgt).arg(&psfc).arg(&u10).arg(&v10);
            for a in [&ust, &t2, &th2, &q2, &hfx, &qfx, &lh, &sa, &ca] {
                match a {
                    Some(a) => l.arg(a),
                    None => l.arg(&null),
                };
            }
            l.arg(&ncell).arg(&nz).arg(&flags);
            l.arg(&mut out);
            unsafe { l.launch(cfg) }.map_err(|e| derr("launch field kernel", e))?;
        }
        stream.synchronize().map_err(|e| derr("sync", e))?;
        let t2_ = std::time::Instant::now();
        let planes = stream.clone_dtoh(&out).map_err(|e| derr("download", e))?;
        let t3 = std::time::Instant::now();
        Ok((
            planes,
            Timing {
                upload: (t1 - t0).as_secs_f64(),
                compute: (t2_ - t1).as_secs_f64(),
                download: (t3 - t2_).as_secs_f64(),
            },
        ))
    }
}
