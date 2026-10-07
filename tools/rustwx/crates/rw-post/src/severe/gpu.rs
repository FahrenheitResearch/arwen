//
// CUDA path for Group C: the parcel, wind and index kernels of the one
// kernel set (kernels/severe_c.cu inside kernels/woof_post.cu), launched on
// the stream of a [`crate::gpu::GpuPost`].  Uploads the column-state slices
// Group C reads, runs three kernels (parcels with one thread per column per
// parcel type, wind profile, STP/EHI) and downloads the output planes.

use std::time::Instant;

use cudarc::driver::{LaunchConfig, PushKernelArg};

use crate::gpu::GpuPost;

use super::{ColumnState, NPLANES, SevereFields};

const PARCELS: &str = "woof_severe_parcels_v1";
const WINDS: &str = "woof_severe_winds_v1";
const INDICES: &str = "woof_severe_indices_v1";
const THREADS: u32 = 128;
const PARCEL_TYPES: u32 = 4;
const KEEP_PLANES: usize = 5;
#[derive(Debug)]
pub struct GpuError(pub String);

impl std::fmt::Display for GpuError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for GpuError {}

fn err<E: std::fmt::Debug>(what: &str) -> impl FnOnce(E) -> GpuError + '_ {
    move |e| GpuError(format!("{what}: {e:?}"))
}

/// Wall-clock split of one frame on the device (seconds).
#[derive(Clone, Copy, Debug, Default)]
pub struct GpuTiming {
    pub upload: f64,
    pub kernels: f64,
    pub download: f64,
}

pub struct SevereGpu<'g> {
    g: &'g GpuPost,
    pub device_name: String,
    pub compute_capability: (i32, i32),
    pub artifact: String,
}

impl<'g> SevereGpu<'g> {
    /// Group C on an opened kernel set.
    pub fn new(g: &'g GpuPost) -> Self {
        Self {
            g,
            device_name: g.device.name.clone(),
            compute_capability: g.device.compute_capability,
            artifact: g.device.kernel_artifact.clone(),
        }
    }

    /// Free and total device memory (bytes).
    pub fn memory(&self) -> Result<(usize, usize), GpuError> {
        self.g.memory().map_err(err("memory info"))
    }

    /// Device bytes one frame needs (inputs, outputs and the f64 keep planes).
    pub fn bytes_needed(state: &ColumnState<'_>) -> usize {
        let n = state.ncell;
        let nz = state.nz;
        4 * (6 * nz * n + 2 * (nz + 1) * n + 7 * n + NPLANES * n) + 8 * KEEP_PLANES * n
    }

    pub fn compute(&self, state: &ColumnState<'_>) -> Result<(SevereFields, GpuTiming), GpuError> {
        state.validate().map_err(|e| GpuError(e.0))?;
        let s = self.g.stream();
        let n = state.ncell;
        let t0 = Instant::now();
        let up = |v: &[f32], what: &'static str| s.clone_htod(v).map_err(err(what));
        let p = up(state.p, "upload p")?;
        let tk = up(state.tk, "upload tk")?;
        let r = up(state.r, "upload r")?;
        let zm = up(state.z_mass, "upload z_mass")?;
        let u = up(state.u, "upload u")?;
        let v = up(state.v, "upload v")?;
        let pi = up(state.p_int, "upload p_int")?;
        let zi = up(state.z_int, "upload z_int")?;
        let zsfc = up(state.zsfc, "upload zsfc")?;
        let psfc = up(state.psfc, "upload psfc")?;
        let t2 = up(state.t2, "upload t2")?;
        let q2 = up(state.q2, "upload q2")?;
        let p2 = up(state.p2, "upload p2")?;
        let u10 = up(state.u10, "upload u10")?;
        let v10 = up(state.v10, "upload v10")?;
        let mut out = s.alloc_zeros::<f32>(NPLANES * n).map_err(err("allocate outputs"))?;
        let mut keep = s.alloc_zeros::<f64>(KEEP_PLANES * n).map_err(err("allocate keep planes"))?;
        s.synchronize().map_err(err("synchronize after upload"))?;
        let t1 = Instant::now();

        let nz_i = state.nz as i32;
        let n_i = n as i32;
        let blocks = (n as u32).div_ceil(THREADS);
        let mut b = s.launch_builder(self.g.function(PARCELS).map_err(err("resolve parcel kernel"))?);
        b.arg(&nz_i)
            .arg(&n_i)
            .arg(&p)
            .arg(&tk)
            .arg(&r)
            .arg(&pi)
            .arg(&zi)
            .arg(&zsfc)
            .arg(&psfc)
            .arg(&t2)
            .arg(&q2)
            .arg(&p2)
            .arg(&mut out)
            .arg(&mut keep);
        unsafe {
            b.launch(LaunchConfig { grid_dim: (blocks, PARCEL_TYPES, 1), block_dim: (THREADS, 1, 1), shared_mem_bytes: 0 })
        }
        .map_err(err("launch parcel kernel"))?;
        let mut b = s.launch_builder(self.g.function(WINDS).map_err(err("resolve wind kernel"))?);
        b.arg(&nz_i).arg(&n_i).arg(&zm).arg(&u).arg(&v).arg(&zsfc).arg(&u10).arg(&v10).arg(&mut out).arg(&mut keep);
        unsafe { b.launch(LaunchConfig { grid_dim: (blocks, 1, 1), block_dim: (THREADS, 1, 1), shared_mem_bytes: 0 }) }
            .map_err(err("launch wind kernel"))?;
        let mut b = s.launch_builder(self.g.function(INDICES).map_err(err("resolve index kernel"))?);
        b.arg(&n_i).arg(&keep).arg(&mut out);
        unsafe { b.launch(LaunchConfig { grid_dim: (blocks, 1, 1), block_dim: (THREADS, 1, 1), shared_mem_bytes: 0 }) }
            .map_err(err("launch index kernel"))?;
        s.synchronize().map_err(err("synchronize after kernels"))?;
        let t2_ = Instant::now();
        let planes = s.clone_dtoh(&out).map_err(err("download outputs"))?;
        s.synchronize().map_err(err("synchronize after download"))?;
        let t3 = Instant::now();
        Ok((
            SevereFields { ncell: n, planes },
            GpuTiming {
                upload: (t1 - t0).as_secs_f64(),
                kernels: (t2_ - t1).as_secs_f64(),
                download: (t3 - t2_).as_secs_f64(),
            },
        ))
    }
}
