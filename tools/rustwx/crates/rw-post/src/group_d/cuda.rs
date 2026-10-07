//! Group D on the GPU: launches `woof_post_d_fields_v1` and
//! `woof_post_d_ir_v1` of the one kernel
//! set on Group A's device-resident state and the packed carriers the state
//! build uploaded (one upload per frame), plus the group D carriers.

use cudarc::driver::{CudaSlice, LaunchConfig, PushKernelArg};

use super::{Carriers, FIELDS, N_COLUMN, N_OUT, O_CEIL, O_IR, O_LOW, Options, Output, blank_omitted, flags, ir_source, omissions, validate};
use crate::gpu::{DeviceState, DeviceStateInputs, GpuPost};
use crate::state::{ColumnState, StateInputs};

pub const FIELDS_KERNEL: &str = "woof_post_d_fields_v1";
pub const IR_KERNEL: &str = "woof_post_d_ir_v1";
const THREADS: u32 = 128;

fn derr<E: std::fmt::Display>(what: &'static str) -> impl FnOnce(E) -> String {
    move |e| format!("CUDA {what}: {e}")
}

/// Timings of one run, seconds.
#[derive(Clone, Copy, Debug, Default)]
pub struct Timing {
    pub upload: f64,
    pub kernel: f64,
    pub download: f64,
}

/// Group D on device-resident state and carriers.
pub fn compute_device(
    g: &GpuPost,
    dstate: &DeviceState,
    dinp: &DeviceStateInputs,
    inp: &StateInputs,
    c: &Carriers<'_>,
    opt: &Options,
) -> Result<(Output, Timing), String> {
    let shape = dstate.shape;
    if dinp.shape != shape || inp.shape != shape {
        return Err("device state and carriers have different shapes".into());
    }
    let n = shape.ncell();
    let (fl, source) = flags(inp, c, opt);
    let s = g.stream();
    let t0 = std::time::Instant::now();
    let up = |a: Option<&[f32]>| -> Result<Option<CudaSlice<f32>>, String> {
        a.map(|x| s.clone_htod(x).map_err(derr("upload group D carrier"))).transpose()
    };
    let refl = up(c.refl)?;
    let cf = up(c.cldfra)?;
    let aer = up(c.aerosol_km)?;
    let hgt = up(c.hgt)?;
    let olr = up(c.olr)?;
    let mut out = s.alloc_zeros::<f32>(N_COLUMN * n).map_err(derr("allocate group D output"))?;
    let mut out_ir = s.alloc_zeros::<f32>(n).map_err(derr("allocate group D IR output"))?;
    s.synchronize().map_err(derr("synchronize"))?;
    let t1 = std::time::Instant::now();
    let null: u64 = 0;
    let ncell = i32::try_from(n).map_err(|_| "grid too large".to_owned())?;
    let nz = i32::try_from(shape.nz).map_err(|_| "too many levels".to_owned())?;
    let f = g.function(FIELDS_KERNEL).map_err(|e| e.to_string())?;
    let mut l = s.launch_builder(f);
    l.arg(&dstate.mass).arg(&dinp.mass);
    for a in [&refl, &cf, &aer, &hgt] {
        match a {
            Some(a) => l.arg(a),
            None => l.arg(&null),
        };
    }
    l.arg(&ncell).arg(&nz).arg(&fl).arg(&opt.contrast).arg(&mut out);
    let cfg = LaunchConfig { grid_dim: ((n as u32).div_ceil(THREADS), 1, 1), block_dim: (THREADS, 1, 1), shared_mem_bytes: 0 };
    unsafe { l.launch(cfg) }.map_err(derr("launch group D kernel"))?;
    let fi = g.function(IR_KERNEL).map_err(|e| e.to_string())?;
    let mut li = s.launch_builder(fi);
    li.arg(&dstate.mass).arg(&dstate.interface).arg(&dinp.mass);
    match &olr {
        Some(a) => li.arg(a),
        None => li.arg(&null),
    };
    li.arg(&ncell).arg(&nz).arg(&fl).arg(&mut out_ir);
    unsafe { li.launch(cfg) }.map_err(derr("launch group D IR kernel"))?;
    s.synchronize().map_err(derr("synchronize"))?;
    let t2 = std::time::Instant::now();
    let mut planes = s.clone_dtoh(&out).map_err(derr("download group D output"))?;
    let ir = s.clone_dtoh(&out_ir).map_err(derr("download group D IR output"))?;
    debug_assert_eq!(O_IR, N_COLUMN);
    planes.extend_from_slice(&ir);
    debug_assert_eq!(planes.len(), N_OUT * n);
    let t3 = std::time::Instant::now();
    let omitted = omissions(inp, c, opt);
    blank_omitted(&mut planes, n, &omitted);
    let cf_used = FIELDS[O_LOW..=O_CEIL].iter().any(|f| !omitted.iter().any(|(id, _)| *id == f.id));
    Ok((
        Output {
            nx: shape.nx,
            ny: shape.ny,
            planes,
            omitted,
            cloud_fraction_source: if cf_used { source } else { None },
            ir_source: ir_source(fl),
            device: g.device.describe(),
        },
        Timing { upload: (t1 - t0).as_secs_f64(), kernel: (t2 - t1).as_secs_f64(), download: (t3 - t2).as_secs_f64() },
    ))
}

/// Group D from host-side state: uploads the packed carriers and the
/// state, then runs [`compute_device`].
pub fn compute_host(g: &GpuPost, state: &ColumnState, inp: &StateInputs, c: &Carriers<'_>, opt: &Options) -> Result<Output, String> {
    validate(state, inp, c)?;
    let dinp = g.upload_state_inputs(inp).map_err(|e| e.to_string())?;
    let s = g.stream();
    let dstate = DeviceState {
        shape: state.shape,
        mass: s.clone_htod(&state.mass).map_err(derr("upload state"))?,
        interface: s.clone_htod(&state.interface).map_err(derr("upload interfaces"))?,
    };
    compute_device(g, &dstate, &dinp, inp, c, opt).map(|(o, _)| o)
}
