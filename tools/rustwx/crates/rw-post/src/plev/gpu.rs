//! The CUDA path of Group B (kernels/group_b.cu, part of the one kernel set
//! kernels/woof_post.cu loaded by [`crate::gpu::GpuPost`]).  It reads Group
//! A's shared column state; no toolkit, no NVRTC at run time.

use std::sync::Arc;
use std::time::Instant;

use cudarc::driver::{CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};

use crate::plev::consts::*;
use crate::gpu::GpuPost;
use crate::plev::frame::Frame;
use crate::plev::*;
use crate::plev::membrane;

const THREADS: u32 = 128;

fn e<E: std::fmt::Display>(what: &'static str) -> impl FnOnce(E) -> String {
    move |err| format!("CUDA {what}: {err}")
}

struct Kernels {
    column: CudaFunction,
    mem_inputs: CudaFunction,
    restrict: CudaFunction,
    jacobi: CudaFunction,
    prolong: CudaFunction,
    mem_column: CudaFunction,
    smooth: CudaFunction,
    copy64: CudaFunction,
    store_f32: CudaFunction,
}

fn cfg1(n: usize) -> LaunchConfig {
    LaunchConfig { grid_dim: ((n as u32).div_ceil(THREADS), 1, 1), block_dim: (THREADS, 1, 1), shared_mem_bytes: 0 }
}

fn cfg2(n: usize, planes: usize) -> LaunchConfig {
    LaunchConfig {
        grid_dim: ((n as u32).div_ceil(THREADS), planes as u32, 1),
        block_dim: (THREADS, 1, 1),
        shared_mem_bytes: 0,
    }
}

fn kernels(g: &GpuPost) -> Result<Kernels, String> {
    let f = |name: &str| g.function(name).cloned().map_err(|err| format!("CUDA resolve {name}: {err}"));
    Ok(Kernels {
        column: f("woof_post_b_column")?,
        mem_inputs: f("woof_post_b_mem_inputs")?,
        restrict: f("woof_post_b_restrict")?,
        jacobi: f("woof_post_b_jacobi")?,
        prolong: f("woof_post_b_prolong")?,
        mem_column: f("woof_post_b_mem_column")?,
        smooth: f("woof_post_b_smooth")?,
        copy64: f("woof_post_b_copy64")?,
        store_f32: f("woof_post_b_store_f32")?,
    })
}

fn nmi_of(nm: usize) -> i32 {
    nm as i32
}

fn up(stream: &Arc<CudaStream>, v: &[f32]) -> Result<CudaSlice<f32>, String> {
    stream.clone_htod(v).map_err(e("upload"))
}

/// A double-buffered level of the membrane cascade.
struct Level {
    nx: usize,
    ny: usize,
    a: CudaSlice<f64>,
    b: CudaSlice<f64>,
    mask: CudaSlice<u8>,
    /// true when `b` holds the current values.
    in_b: bool,
}

impl Level {
    fn cur(&self) -> &CudaSlice<f64> {
        if self.in_b { &self.b } else { &self.a }
    }
}

fn jacobi(stream: &Arc<CudaStream>, k: &Kernels, lv: &mut Level, nact: usize, count: usize) -> Result<(), String> {
    let n = lv.nx * lv.ny;
    let (nx, ny, na) = (lv.nx as i32, lv.ny as i32, nact as i32);
    for _ in 0..count {
        let mut b = stream.launch_builder(&k.jacobi);
        b.arg(&nx).arg(&ny).arg(&na);
        if lv.in_b {
            b.arg(&lv.b).arg(&lv.mask).arg(&mut lv.a);
        } else {
            b.arg(&lv.a).arg(&lv.mask).arg(&mut lv.b);
        }
        unsafe { b.launch(cfg2(n, nact)) }.map_err(e("launch jacobi"))?;
        lv.in_b = !lv.in_b;
    }
    Ok(())
}

pub fn compute(f: &Frame, levels: &[f64], g: &GpuPost) -> Result<Output, String> {
    if f.nz > GPU_NZ_MAX {
        return Err(format!("{} levels exceed the CUDA column limit {GPU_NZ_MAX}", f.nz));
    }
    let mut timings: Vec<(&'static str, f64)> = Vec::new();
    let mut clock = Instant::now();
    let mut lap = |what: &'static str, stream: &Arc<CudaStream>| -> Result<(), String> {
        stream.synchronize().map_err(e("synchronize"))?;
        timings.push((what, clock.elapsed().as_secs_f64()));
        clock = Instant::now();
        Ok(())
    };
    let kset = kernels(g)?;
    let k = &kset;
    let stream = Arc::clone(g.stream());
    lap("setup", &stream)?;
    let n2 = f.ncell();
    let nl = levels.len();
    let mem_levels = membrane_levels();
    let nm = mem_levels.len();
    let base2 = 6 * nl;

    // Group A's shared state (one upload of the state slots this group reads).
    let st = up(&stream, &f.state.mass)?;
    let si = up(&stream, &f.state.interface)?;
    let psfc = up(&stream, f.psfc)?;
    let hgt = up(&stream, f.hgt)?;
    let lev = stream.clone_htod(levels).map_err(e("upload"))?;
    let mlev = stream.clone_htod(&mem_levels).map_err(e("upload"))?;
    let mut out32 = stream.alloc_zeros::<f32>((base2 + N2D) * n2).map_err(e("alloc out32"))?;
    let mut out64 = stream.alloc_zeros::<f64>((nm + N64_EXTRA) * n2).map_err(e("alloc out64"))?;
    lap("upload", &stream)?;

    let (nx, ny, nz) = (f.nx as i32, f.ny as i32, f.nz as i32);
    let (nli, nmi) = (nl as i32, nmi_of(nm));
    {
        let mut b = stream.launch_builder(&k.column);
        b.arg(&nx).arg(&ny).arg(&nz).arg(&st).arg(&si).arg(&psfc).arg(&hgt);
        b.arg(&lev).arg(&nli).arg(&mlev).arg(&nmi);
        b.arg(&mut out32).arg(&mut out64);
        unsafe { b.launch(cfg1(n2)) }.map_err(e("launch column"))?;
    }
    lap("column kernel", &stream)?;
    // Inputs are no longer needed.
    drop((st, si, psfc, lev));

    // Membrane.
    let pg = stream
        .clone_dtoh(&out64.slice((nm + Q_PGROUND) * n2..(nm + Q_PGROUND + 1) * n2))
        .map_err(e("download ground pressure"))?;
    let active = membrane::active_levels(&mem_levels, &pg);
    let nact = active.len();
    let mut act_of_m = vec![-1i32; nm];
    for (a, &m) in active.iter().enumerate() {
        act_of_m[m] = a as i32;
    }
    let act_of_m_d = stream.clone_htod(&act_of_m).map_err(e("upload"))?;
    let i1000 = levels.iter().position(|&p| p == 100000.0);
    let n2i = n2 as i64;
    let slp_plane = (base2 + P2_MEMBRANE_SLP) as i64;
    let z_plane = i1000.map_or(-1i64, |l| (l * 6) as i64);
    if nact > 0 {
        let dims = membrane::pyramid(f.nx, f.ny);
        let mut lv: Vec<Level> = Vec::with_capacity(dims.len());
        for &(x, y) in &dims {
            let n = x * y * nact;
            lv.push(Level {
                nx: x,
                ny: y,
                a: stream.alloc_zeros::<f64>(n).map_err(e("alloc membrane"))?,
                b: stream.alloc_zeros::<f64>(n).map_err(e("alloc membrane"))?,
                mask: stream.alloc_zeros::<u8>(n).map_err(e("alloc membrane"))?,
                in_b: false,
            });
        }
        let active_i: Vec<i32> = active.iter().map(|&m| m as i32).collect();
        let act_d = stream.clone_htod(&active_i).map_err(e("upload"))?;
        let nacti = nact as i32;
        {
            let l0 = &mut lv[0];
            let mut b = stream.launch_builder(&k.mem_inputs);
            b.arg(&n2i).arg(&nacti).arg(&act_d).arg(&mlev).arg(&nmi).arg(&out64);
            b.arg(&mut l0.a).arg(&mut l0.mask);
            unsafe { b.launch(cfg2(n2, nact)) }.map_err(e("launch membrane inputs"))?;
        }
        for l in 1..dims.len() {
            let (fine, coarse) = lv.split_at_mut(l);
            let fl = &fine[l - 1];
            let cl = &mut coarse[0];
            let (nxf, nyf, nxc, nyc) = (fl.nx as i32, fl.ny as i32, cl.nx as i32, cl.ny as i32);
            let mut b = stream.launch_builder(&k.restrict);
            b.arg(&nxf).arg(&nyf).arg(&nxc).arg(&nyc).arg(&nacti);
            b.arg(fl.cur()).arg(&fl.mask).arg(&mut cl.a).arg(&mut cl.mask);
            unsafe { b.launch(cfg2(cl.nx * cl.ny, nact)) }.map_err(e("launch restrict"))?;
            cl.in_b = false;
        }
        let last = dims.len() - 1;
        jacobi(&stream, k, &mut lv[last], nact, MEMBRANE_COARSE_SWEEPS)?;
        for l in (0..last).rev() {
            {
                let (fine, coarse) = lv.split_at_mut(l + 1);
                let fl = &mut fine[l];
                let cl = &coarse[0];
                let (nxf, nyf, nxc, nyc) = (fl.nx as i32, fl.ny as i32, cl.nx as i32, cl.ny as i32);
                let mut b = stream.launch_builder(&k.prolong);
                b.arg(&nxf).arg(&nyf).arg(&nxc).arg(&nyc).arg(&nacti).arg(cl.cur()).arg(&fl.mask);
                if fl.in_b {
                    b.arg(&mut fl.b);
                } else {
                    b.arg(&mut fl.a);
                }
                unsafe { b.launch(cfg2(fl.nx * fl.ny, nact)) }.map_err(e("launch prolong"))?;
            }
            jacobi(&stream, k, &mut lv[l], nact, MEMBRANE_LEVEL_SWEEPS)?;
        }
        let mut b = stream.launch_builder(&k.mem_column);
        b.arg(&n2i).arg(&nmi).arg(&mlev).arg(&act_of_m_d).arg(lv[0].cur()).arg(&out64);
        b.arg(&hgt).arg(&slp_plane).arg(&z_plane).arg(&mut out32);
        unsafe { b.launch(cfg1(n2)) }.map_err(e("launch membrane column"))?;
    } else {
        let none = stream.alloc_zeros::<f64>(1).map_err(e("alloc"))?;
        let mut b = stream.launch_builder(&k.mem_column);
        b.arg(&n2i).arg(&nmi).arg(&mlev).arg(&act_of_m_d).arg(&none).arg(&out64);
        b.arg(&hgt).arg(&slp_plane).arg(&z_plane).arg(&mut out32);
        unsafe { b.launch(cfg1(n2)) }.map_err(e("launch membrane column"))?;
    }

    lap("membrane", &stream)?;
    // MAPS smoothing.
    let passes = maps_smoothing_passes(f.dx);
    let mut sa = stream.alloc_zeros::<f64>(n2).map_err(e("alloc"))?;
    let mut sb = stream.alloc_zeros::<f64>(n2).map_err(e("alloc"))?;
    {
        let off = ((nm + Q_MAPS_RAW) * n2) as i64;
        let mut b = stream.launch_builder(&k.copy64);
        b.arg(&n2i).arg(&out64).arg(&off).arg(&mut sa);
        unsafe { b.launch(cfg1(n2)) }.map_err(e("launch copy"))?;
    }
    for _ in 0..passes {
        for along_y in [0i32, 1i32] {
            {
                let mut b = stream.launch_builder(&k.smooth);
                b.arg(&nx).arg(&ny).arg(&along_y).arg(&sa).arg(&mut sb);
                unsafe { b.launch(cfg1(n2)) }.map_err(e("launch smooth"))?;
            }
            std::mem::swap(&mut sa, &mut sb);
        }
    }
    {
        let off = ((base2 + P2_MAPS) * n2) as i64;
        let mut b = stream.launch_builder(&k.store_f32);
        b.arg(&n2i).arg(&sa).arg(&mut out32).arg(&off);
        unsafe { b.launch(cfg1(n2)) }.map_err(e("launch store"))?;
    }
    lap("maps smoothing", &stream)?;
    let host = stream.clone_dtoh(&out32).map_err(e("download"))?;
    lap("download", &stream)?;
    let mut o = split_output(f, levels, host, passes, nact, g.device.describe());
    o.timings = timings;
    Ok(o)
}
