//! The one CUDA loader of the post-processor (specification 3.5).
//!
//! Every group's kernels are one translation unit, `kernels/woof_post.cu`
//! (it includes `post_group_a.cu`, `group_b.cu`, `severe_c.cu` and
//! `group_e.cu` with the shared maths library `woof_math.cuh`), built into
//! ONE kernel set: CUBINs for the two certified Blackwell architectures
//! (sm_100 and sm_120) plus a compute_75 PTX every other card JIT-compiles,
//! all described by ONE manifest, `kernels/woof_post.manifest.json`.
//!
//! Why only two CUBINs: every image here is embedded in each binary that
//! links this crate, and the full twelve-architecture set (19.2 MB of CUBIN)
//! put the 2.8.7 manylinux wheel at about 105 MB, over PyPI's
//! 100,000,000-byte per-file limit, so the release could not be uploaded.
//! Byte identity is certified on Blackwell (sm_100, sm_120); Ada, Ampere,
//! Hopper and the other Blackwell variants run the same kernels from the
//! PTX, which encodes the same `--fmad=false` exact-division arithmetic.
//! [`EMBEDDED_ARTIFACTS`] is the set, and `tests/gpu.rs` pins it.
//! [`GpuPost::open`] loads that image once and resolves every kernel; the
//! groups launch through [`GpuPost::function`] on [`GpuPost::stream`].
//!
//! Driver-only, dynamically loaded CUDA (vendored cudarc 0.19.8): no
//! toolkit and no NVRTC at run time.

use std::sync::Arc;

use std::collections::BTreeMap;

use cudarc::{
    driver::{CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg},
    nvrtc::Ptx,
};
use thiserror::Error;

use crate::state::{ColumnState, MassState, Shape, StateInputs, check_inputs};
use crate::surface::{SurfaceField, SurfaceInput, SurfaceInputs, SurfaceOptions};

const PTX: &str = include_str!("../kernels/woof_post.ptx");
pub const MANIFEST: &str = include_str!("../kernels/woof_post.manifest.json");
const THREADS: u32 = 128;

/// Every kernel of the set, resolved when a device opens.
pub const KERNELS: [&str; 21] = [
    // Group A: column state, surface, rotation, probes.
    "woof_post_state_v1",
    "woof_post_surface_v1",
    "woof_post_rotate_v1",
    "woof_post_libm_probe_v1",
    "woof_post_math64_probe_v1",
    "woof_post_thermo_probe_v1",
    // Group B: pressure levels, sea-level pressure, column water.
    "woof_post_b_column",
    "woof_post_b_mem_inputs",
    "woof_post_b_restrict",
    "woof_post_b_jacobi",
    "woof_post_b_prolong",
    "woof_post_b_mem_column",
    "woof_post_b_smooth",
    "woof_post_b_copy64",
    "woof_post_b_store_f32",
    // Group C: parcels, shear, helicity, severe indices.
    "woof_severe_parcels_v1",
    "woof_severe_winds_v1",
    "woof_severe_indices_v1",
    // Group E: boundary layer, near-surface winds, isotherms.
    "woof_post_e_fields_v1",
    // Group D: reflectivity, cloud, visibility, simulated infrared.
    "woof_post_d_fields_v1",
    "woof_post_d_ir_v1",
];

#[derive(Debug, Error)]
pub enum GpuError {
    #[error("{operation}: {message}")]
    Driver { operation: &'static str, message: String },
    #[error("invalid input: {0}")]
    Input(String),
}

fn drv<E: std::fmt::Display>(operation: &'static str) -> impl FnOnce(E) -> GpuError {
    move |e| GpuError::Driver { operation, message: e.to_string() }
}

/// The device a frame ran on, for the manifest.
#[derive(Clone, Debug)]
pub struct DeviceInfo {
    pub ordinal: usize,
    pub name: String,
    pub compute_capability: (i32, i32),
    pub total_memory_bytes: usize,
    pub kernel_artifact: String,
}

impl DeviceInfo {
    /// One line for manifests and logs.
    pub fn describe(&self) -> String {
        format!(
            "gpu {} (sm_{}{}, {})",
            self.name, self.compute_capability.0, self.compute_capability.1, self.kernel_artifact
        )
    }
}

/// Every image compiled into this crate, by file name: the sm_100 and
/// sm_120 CUBINs and the compute_75 PTX, nothing else (see the module
/// documentation for why the other architectures are not embedded).
pub const EMBEDDED_ARTIFACTS: [&str; 3] = ["woof_post_sm100.cubin", "woof_post_sm120.cubin", "woof_post.ptx"];

/// The checked-in image for an architecture: its CUBIN, else the PTX.
fn artifact_for(major: i32, minor: i32) -> (String, Ptx) {
    macro_rules! cubin {
        ($arch:literal) => {
            (
                concat!("sm_", stringify!($arch), " CUBIN").to_owned(),
                Ptx::from_binary(
                    include_bytes!(concat!("../kernels/woof_post_sm", stringify!($arch), ".cubin")).to_vec(),
                ),
            )
        };
    }
    match major * 10 + minor {
        100 => cubin!(100),
        120 => cubin!(120),
        _ => ("compute_75 PTX".to_owned(), Ptx::from_src(PTX)),
    }
}

/// Every checked-in artifact by file name (manifest authentication).
pub fn checked_in_artifact(file: &str) -> Option<&'static [u8]> {
    Some(match file {
        "woof_post_sm100.cubin" => include_bytes!("../kernels/woof_post_sm100.cubin"),
        "woof_post_sm120.cubin" => include_bytes!("../kernels/woof_post_sm120.cubin"),
        "woof_post.ptx" => PTX.as_bytes(),
        _ => return None,
    })
}

/// The post-processor's kernel set bound to one device.
pub struct GpuPost {
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    functions: BTreeMap<&'static str, CudaFunction>,
    pub device: DeviceInfo,
}

/// The 3D carriers resident on the device (packed as in [`StateInputs`]).
pub struct DeviceStateInputs {
    pub shape: Shape,
    pub present: u32,
    pub p_top: f32,
    pub mass: CudaSlice<f32>,
    pub ph: CudaSlice<f32>,
    pub phb: CudaSlice<f32>,
    pub u_stag: CudaSlice<f32>,
    pub v_stag: CudaSlice<f32>,
    pub mu: CudaSlice<f32>,
    pub mub: CudaSlice<f32>,
    pub c3f: CudaSlice<f32>,
    pub c4f: CudaSlice<f32>,
}

/// The column state resident on the device (level-major, as on the CPU).
pub struct DeviceState {
    pub shape: Shape,
    pub mass: CudaSlice<f32>,
    pub interface: CudaSlice<f32>,
}

fn grid(n: usize) -> LaunchConfig {
    LaunchConfig { grid_dim: ((n as u32).div_ceil(THREADS), 1, 1), block_dim: (THREADS, 1, 1), shared_mem_bytes: 0 }
}

impl GpuPost {
    /// Open device `ordinal` and load the kernels.
    ///
    /// A machine with no NVIDIA driver library is an `Err`, not a panic:
    /// cudarc's dynamic loader panics on its first driver call when neither
    /// `libcuda`/`nvcuda` loads, which took `--post-device auto` (the GRIB2
    /// export default) down instead of to the CPU on such a machine, and
    /// failed the GPU tests' own no-card skip on public CI.  The probe is
    /// cudarc's, over the same library names its loader tries; with the
    /// library present nothing changes.
    pub fn open(ordinal: usize) -> Result<Self, GpuError> {
        // SAFETY: loads (and drops) the driver library by name; no symbol
        // is called.
        if !unsafe { cudarc::driver::sys::is_culib_present() } {
            return Err(GpuError::Driver {
                operation: "open CUDA device",
                message: "no NVIDIA driver library (libcuda/nvcuda) could be loaded".into(),
            });
        }
        let ctx = CudaContext::new(ordinal).map_err(drv("open CUDA device"))?;
        let name = ctx.name().map_err(drv("device name"))?;
        let (major, minor) = ctx.compute_capability().map_err(drv("compute capability"))?;
        let total = ctx.total_mem().map_err(drv("device memory"))?;
        let (label, image) = artifact_for(major, minor);
        let module = ctx.load_module(image).map_err(drv("load post kernels"))?;
        let mut functions = BTreeMap::new();
        for name in KERNELS {
            functions.insert(name, module.load_function(name).map_err(drv("resolve post kernel"))?);
        }
        Ok(Self {
            stream: ctx.default_stream(),
            functions,
            device: DeviceInfo {
                ordinal,
                name,
                compute_capability: (major, minor),
                total_memory_bytes: total,
                kernel_artifact: label,
            },
            ctx,
        })
    }

    /// A kernel of the set by name.
    pub fn function(&self, name: &str) -> Result<&CudaFunction, GpuError> {
        self.functions
            .get(name)
            .ok_or_else(|| GpuError::Input(format!("kernel {name} is not in the post kernel set")))
    }

    /// The stream every group launches on (one stream: launches are ordered).
    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    /// The device context.
    pub fn context(&self) -> &Arc<CudaContext> {
        &self.ctx
    }

    /// Free and total device memory in bytes.
    pub fn memory(&self) -> Result<(usize, usize), GpuError> {
        self.ctx.mem_get_info().map_err(drv("memory info"))
    }

    /// Bytes the state build needs on the device for a shape.
    pub fn state_bytes(shape: &Shape) -> usize {
        let words = crate::state::MassInput::COUNT * shape.mass_volume()
            + 2 * shape.interface_volume()
            + shape.nz * shape.ny * (shape.nx + 1)
            + shape.nz * (shape.ny + 1) * shape.nx
            + MassState::COUNT * shape.mass_volume()
            + 2 * shape.interface_volume()
            + 2 * shape.ncell();
        words * 4
    }

    pub fn synchronize(&self) -> Result<(), GpuError> {
        self.stream.synchronize().map_err(drv("synchronize"))
    }

    /// Build the column state on the device (upload, then the state kernel).
    pub fn build_state(&self, inp: &StateInputs) -> Result<DeviceState, GpuError> {
        let dev = self.upload_state_inputs(inp)?;
        self.run_state(&dev)
    }

    /// Upload the 3D carriers once.  The packed hydrometeor slots stay on
    /// the device for the groups that consume them after the state.
    pub fn upload_state_inputs(&self, inp: &StateInputs) -> Result<DeviceStateInputs, GpuError> {
        check_inputs(inp).map_err(GpuError::Input)?;
        let s = &self.stream;
        Ok(DeviceStateInputs {
            shape: inp.shape,
            present: inp.present,
            p_top: inp.p_top,
            mass: s.clone_htod(&inp.mass).map_err(drv("upload mass inputs"))?,
            ph: s.clone_htod(&inp.ph).map_err(drv("upload PH"))?,
            phb: s.clone_htod(&inp.phb).map_err(drv("upload PHB"))?,
            u_stag: s.clone_htod(&inp.u_stag).map_err(drv("upload U"))?,
            v_stag: s.clone_htod(&inp.v_stag).map_err(drv("upload V"))?,
            mu: s.clone_htod(&inp.mu).map_err(drv("upload MU"))?,
            mub: s.clone_htod(&inp.mub).map_err(drv("upload MUB"))?,
            c3f: s.clone_htod(&inp.c3f).map_err(drv("upload C3F"))?,
            c4f: s.clone_htod(&inp.c4f).map_err(drv("upload C4F"))?,
        })
    }

    /// The state kernel on device-resident carriers.
    pub fn run_state(&self, d: &DeviceStateInputs) -> Result<DeviceState, GpuError> {
        let s = &self.stream;
        let shape = d.shape;
        let mut mass = s.alloc_zeros::<f32>(MassState::COUNT * shape.mass_volume()).map_err(drv("allocate state"))?;
        let mut interface = s.alloc_zeros::<f32>(2 * shape.interface_volume()).map_err(drv("allocate interfaces"))?;
        let (nx, ny, nz) = (shape.nx as u32, shape.ny as u32, shape.nz as u32);
        let mut launch = s.launch_builder(self.function("woof_post_state_v1")?);
        launch
            .arg(&d.mass)
            .arg(&d.present)
            .arg(&d.ph)
            .arg(&d.phb)
            .arg(&d.u_stag)
            .arg(&d.v_stag)
            .arg(&d.mu)
            .arg(&d.mub)
            .arg(&d.c3f)
            .arg(&d.c4f)
            .arg(&d.p_top)
            .arg(&nx)
            .arg(&ny)
            .arg(&nz)
            .arg(&mut mass)
            .arg(&mut interface);
        unsafe { launch.launch(grid(shape.ncell())) }.map_err(drv("launch state kernel"))?;
        Ok(DeviceState { shape, mass, interface })
    }

    /// Copy a device state to the host.
    pub fn download_state(&self, st: &DeviceState) -> Result<ColumnState, GpuError> {
        Ok(ColumnState {
            shape: st.shape,
            mass: self.stream.clone_dtoh(&st.mass).map_err(drv("download state"))?,
            interface: self.stream.clone_dtoh(&st.interface).map_err(drv("download interfaces"))?,
        })
    }

    /// Surface fields, taking the lowest-level virtual temperature from a
    /// device-resident state when one is given (else from `inp`).
    pub fn surface(
        &self,
        inp: &SurfaceInputs,
        opt: &SurfaceOptions,
        state: Option<&DeviceState>,
    ) -> Result<Vec<f32>, GpuError> {
        let s = &self.stream;
        let n = inp.ncell;
        let mut planes = s.clone_htod(&inp.planes).map_err(drv("upload surface inputs"))?;
        let mut present = inp.present;
        if let Some(st) = state {
            if st.shape.ncell() != n {
                return Err(GpuError::Input("state and surface grids differ".into()));
            }
            let tv0 = MassState::Tv as usize * st.shape.mass_volume();
            let slot = SurfaceInput::TvLowest as usize * n;
            s.memcpy_dtod(&st.mass.slice(tv0..tv0 + n), &mut planes.slice_mut(slot..slot + n))
                .map_err(drv("copy lowest virtual temperature"))?;
            present |= 1 << SurfaceInput::TvLowest as u32;
        }
        let mut out = s.alloc_zeros::<f32>(SurfaceField::COUNT * n).map_err(drv("allocate surface output"))?;
        let ncell = n as u32;
        let mut launch = s.launch_builder(self.function("woof_post_surface_v1")?);
        launch.arg(&planes).arg(&present).arg(opt).arg(&ncell).arg(&mut out);
        unsafe { launch.launch(grid(n)) }.map_err(drv("launch surface kernel"))?;
        s.clone_dtoh(&out).map_err(drv("download surface output"))
    }

    /// Rotate `u.len() / sina.len()` levels of grid-relative winds.
    pub fn rotate(&self, u: &[f32], v: &[f32], sina: &[f32], cosa: &[f32]) -> Result<(Vec<f32>, Vec<f32>), GpuError> {
        let ncell = sina.len();
        if cosa.len() != ncell || u.len() != v.len() || ncell == 0 || u.len() % ncell != 0 {
            return Err(GpuError::Input("rotation sizes".into()));
        }
        let s = &self.stream;
        let du = s.clone_htod(u).map_err(drv("upload u"))?;
        let dv = s.clone_htod(v).map_err(drv("upload v"))?;
        let dsa = s.clone_htod(sina).map_err(drv("upload SINALPHA"))?;
        let dca = s.clone_htod(cosa).map_err(drv("upload COSALPHA"))?;
        let mut ue = s.alloc_zeros::<f32>(u.len()).map_err(drv("allocate ue"))?;
        let mut ve = s.alloc_zeros::<f32>(u.len()).map_err(drv("allocate ve"))?;
        let (nc, planes) = (ncell as u32, (u.len() / ncell) as u32);
        let mut launch = s.launch_builder(self.function("woof_post_rotate_v1")?);
        launch.arg(&du).arg(&dv).arg(&dsa).arg(&dca).arg(&nc).arg(&planes).arg(&mut ue).arg(&mut ve);
        unsafe { launch.launch(grid(u.len())) }.map_err(drv("launch rotation"))?;
        Ok((s.clone_dtoh(&ue).map_err(drv("download ue"))?, s.clone_dtoh(&ve).map_err(drv("download ve"))?))
    }

    /// [expf, logf, sinf, cosf] of each x, for the identity tests.
    pub fn libm_probe(&self, x: &[f32]) -> Result<Vec<f32>, GpuError> {
        let s = &self.stream;
        let dx = s.clone_htod(x).map_err(drv("upload probe"))?;
        let mut out = s.alloc_zeros::<f32>(4 * x.len()).map_err(drv("allocate probe"))?;
        let n = x.len() as u32;
        let mut launch = s.launch_builder(self.function("woof_post_libm_probe_v1")?);
        launch.arg(&dx).arg(&n).arg(&mut out);
        unsafe { launch.launch(grid(x.len())) }.map_err(drv("launch libm probe"))?;
        s.clone_dtoh(&out).map_err(drv("download probe"))
    }

    /// [ln, exp, pow(|x|, y), cbrt, sin, cos] of each (x, y) in binary64.
    pub fn math64_probe(&self, x: &[f64], y: &[f64]) -> Result<Vec<f64>, GpuError> {
        let s = &self.stream;
        let dx = s.clone_htod(x).map_err(drv("upload probe x"))?;
        let dy = s.clone_htod(y).map_err(drv("upload probe y"))?;
        let mut out = s.alloc_zeros::<f64>(6 * x.len()).map_err(drv("allocate probe"))?;
        let n = x.len() as u32;
        let f = self.function("woof_post_math64_probe_v1")?;
        let mut launch = s.launch_builder(f);
        launch.arg(&dx).arg(&dy).arg(&n).arg(&mut out);
        unsafe { launch.launch(grid(x.len())) }.map_err(drv("launch math64 probe"))?;
        s.clone_dtoh(&out).map_err(drv("download probe"))
    }

    /// [es_water, es_ice, e, dewpoint, rh, tv, theta_e] per (T, p, r).
    pub fn thermo_probe(&self, tk: &[f32], p: &[f32], r: &[f32]) -> Result<Vec<f32>, GpuError> {
        let s = &self.stream;
        let dt = s.clone_htod(tk).map_err(drv("upload T"))?;
        let dp = s.clone_htod(p).map_err(drv("upload p"))?;
        let dr = s.clone_htod(r).map_err(drv("upload r"))?;
        let mut out = s.alloc_zeros::<f32>(7 * tk.len()).map_err(drv("allocate probe"))?;
        let n = tk.len() as u32;
        let mut launch = s.launch_builder(self.function("woof_post_thermo_probe_v1")?);
        launch.arg(&dt).arg(&dp).arg(&dr).arg(&n).arg(&mut out);
        unsafe { launch.launch(grid(tk.len())) }.map_err(drv("launch thermo probe"))?;
        s.clone_dtoh(&out).map_err(drv("download probe"))
    }
}
