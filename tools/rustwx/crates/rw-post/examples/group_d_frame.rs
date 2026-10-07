//! Run group D (reflectivity, cloud, visibility) on one WRF-format WOOF
//! history frame on the CPU and on GPU 0, from Group A's shared column
//! state, and write every plane plus a summary with the CPU/GPU bit
//! comparison, GPU run-to-run check and timings.
//!
//!   group_d_frame HISTORY OUTDIR [--no-gpu] [--cf model|model-only|diagnosed]
//!       [--diag xu|presence] [--interp z|dbz] [--contrast 5|2]
//!       [--allow-missing-aerosol] [--no-refl] [--tile N]
//!
//! Layout (little-endian f32, x fastest, south row first):
//!   OUTDIR/{cpu,gpu}/{field}.f32, OUTDIR/summary.json
//! `--tile N` also times the kernel on the frame tiled N x N.

use std::path::{Path, PathBuf};
use std::time::Instant;

use rw_post::group_d::{self, CloudFractionSource, DiagnosedCloud, FIELDS, Options, ReflInterp, history::OwnedCarriers};
use rw_post::history;
use rw_post::state::{ColumnState, MassState, Shape, StateInputs};
use serde_json::json;

fn write_f32(path: &Path, v: &[f32]) -> std::io::Result<()> {
    if let Some(d) = path.parent() {
        std::fs::create_dir_all(d)?;
    }
    let b: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
    std::fs::write(path, b)
}

fn diff(a: &[f32], b: &[f32]) -> usize {
    a.iter().zip(b).filter(|(x, y)| x.to_bits() != y.to_bits()).count()
}

/// Tile a level-major volume of `slots` x `nz` planes N x N.
fn tile(v: &[f32], planes: usize, nx: usize, ny: usize, t: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(v.len() * t * t);
    for p in 0..planes {
        let src = &v[p * nx * ny..(p + 1) * nx * ny];
        for _ty in 0..t {
            for j in 0..ny {
                for _tx in 0..t {
                    out.extend_from_slice(&src[j * nx..(j + 1) * nx]);
                }
            }
        }
    }
    out
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: group_d_frame HISTORY OUTDIR [options]");
        std::process::exit(2);
    }
    let hist = PathBuf::from(&args[1]);
    let out = PathBuf::from(&args[2]);
    let mut use_gpu = true;
    let mut opt = Options::default();
    let mut tiles = 0usize;
    let mut no_refl = false;
    let mut i = 3;
    while i < args.len() {
        let a = args[i].as_str();
        let mut val = || {
            i += 1;
            args.get(i).cloned().ok_or(format!("{a} needs a value"))
        };
        match a {
            "--no-gpu" => use_gpu = false,
            "--no-refl" => no_refl = true,
            "--allow-missing-aerosol" => opt.allow_missing_aerosol = true,
            "--cf" => {
                opt.cloud_fraction = match val()?.as_str() {
                    "model" => CloudFractionSource::ModelThenDiagnosed,
                    "model-only" => CloudFractionSource::ModelOnly,
                    "diagnosed" => CloudFractionSource::Diagnosed,
                    o => return Err(format!("--cf {o}").into()),
                }
            }
            "--diag" => {
                opt.diagnosed = match val()?.as_str() {
                    "xu" => DiagnosedCloud::XuRandall1996,
                    "presence" => DiagnosedCloud::CondensatePresence,
                    o => return Err(format!("--diag {o}").into()),
                }
            }
            "--interp" => opt.refl_interp = if val()? == "dbz" { ReflInterp::LinearDbz } else { ReflInterp::LinearZ },
            "--contrast" => {
                opt.contrast = if val()? == "2" { group_d::CONTRAST_2PCT } else { group_d::CONTRAST_MOR_5PCT }
            }
            "--tile" => tiles = val()?.parse()?,
            other => return Err(format!("unknown argument {other}").into()),
        }
        i += 1;
    }
    std::fs::create_dir_all(&out)?;

    let t = Instant::now();
    let mut frame = history::Frame::read(&hist, 0)?;
    let owned = OwnedCarriers::read(&hist)?;
    let read_s = t.elapsed().as_secs_f64();
    let shape = frame.shape;
    let n = shape.ncell();
    let mut c = owned.carriers();
    if no_refl {
        // Measurement mode: the RIP-note volume from the species instead.
        c.refl = None;
    }

    let a_cpu = rw_post::group_a_cpu(&mut frame)?;
    let t = Instant::now();
    let d_cpu = group_d::compute_cpu(&a_cpu.state, &frame.state, &c, &opt)?;
    let cpu_s = t.elapsed().as_secs_f64();
    for (k, f) in FIELDS.iter().enumerate() {
        write_f32(&out.join("cpu").join(format!("{}.f32", f.id)), &d_cpu.planes[k * n..(k + 1) * n])?;
    }
    let mut summary = json!({
        "history": hist.display().to_string(),
        "shape": {"nx": shape.nx, "ny": shape.ny, "nz": shape.nz},
        "options": format!("{opt:?}"),
        "aerosol_source": owned.aerosol_source,
        "carriers": {"REFL_10CM": c.refl.is_some(), "CLDFRA": c.cldfra.is_some(), "HGT": c.hgt.is_some()},
        "cloud_fraction_source": d_cpu.cloud_fraction_source,
        "omitted": d_cpu.omitted.iter().map(|(a, b)| json!([a, b])).collect::<Vec<_>>(),
        "read_seconds": read_s,
        "cpu_seconds": cpu_s,
        "rayon_threads": rayon::current_num_threads(),
    });

    if use_gpu {
        let g = rw_post::gpu::GpuPost::open(0)?;
        summary["gpu_device"] = json!(g.device.describe());
        // Group A's state built on the device, then group D on it.
        let dinp = g.upload_state_inputs(&frame.state)?;
        let dstate = g.run_state(&dinp)?;
        let st_gpu = g.download_state(&dstate)?;
        summary["state_words_differing_cpu_gpu"] = json!(diff(&a_cpu.state.mass, &st_gpu.mass));
        let (d1, _) = group_d::cuda::compute_device(&g, &dstate, &dinp, &frame.state, &c, &opt)?;
        let (d2, tm) = group_d::cuda::compute_device(&g, &dstate, &dinp, &frame.state, &c, &opt)?;
        for (k, f) in FIELDS.iter().enumerate() {
            write_f32(&out.join("gpu").join(format!("{}.f32", f.id)), &d2.planes[k * n..(k + 1) * n])?;
        }
        let mut per = serde_json::Map::new();
        for (k, f) in FIELDS.iter().enumerate() {
            per.insert(f.id.to_owned(), json!(diff(&d_cpu.planes[k * n..(k + 1) * n], &d2.planes[k * n..(k + 1) * n])));
        }
        summary["gpu_cpu_bit_differences"] = serde_json::Value::Object(per);
        summary["gpu_run_to_run"] = json!(diff(&d1.planes, &d2.planes));
        summary["gpu_timing"] = json!({"upload": tm.upload, "kernel": tm.kernel, "download": tm.download});
        eprintln!("group D cpu/gpu: {} run-to-run {}", summary["gpu_cpu_bit_differences"], summary["gpu_run_to_run"]);

        if tiles > 1 {
            let (nx, ny, nz) = (shape.nx, shape.ny, shape.nz);
            let big = Shape { nx: nx * tiles, ny: ny * tiles, nz };
            let st_big = ColumnState {
                shape: big,
                mass: tile(&a_cpu.state.mass, MassState::COUNT * nz, nx, ny, tiles),
                interface: vec![0.0; 1],
            };
            let in_big = StateInputs {
                shape: big,
                mass: tile(&frame.state.mass, rw_post::state::MassInput::COUNT * nz, nx, ny, tiles),
                present: frame.state.present,
                ph: vec![],
                phb: vec![],
                u_stag: vec![],
                v_stag: vec![],
                mu: vec![],
                mub: vec![],
                c3f: vec![],
                c4f: vec![],
                p_top: frame.state.p_top,
            };
            let refl_b = c.refl.map(|v| tile(v, nz, nx, ny, tiles));
            let cf_b = c.cldfra.map(|v| tile(v, nz, nx, ny, tiles));
            let aer_b = c.aerosol_km.map(|v| tile(v, 1, nx, ny, tiles));
            let hgt_b = c.hgt.map(|v| tile(v, 1, nx, ny, tiles));
            let cb = group_d::Carriers { refl: refl_b.as_deref(), cldfra: cf_b.as_deref(), aerosol_km: aer_b.as_deref(), hgt: hgt_b.as_deref(), olr: None };
            let s = g.stream();
            let t_up = Instant::now();
            let d_state = rw_post::gpu::DeviceState { shape: big, mass: s.clone_htod(&st_big.mass)?, interface: s.alloc_zeros::<f32>(1)? };
            let d_in = rw_post::gpu::DeviceStateInputs {
                shape: big,
                present: in_big.present,
                p_top: in_big.p_top,
                mass: s.clone_htod(&in_big.mass)?,
                ph: s.alloc_zeros::<f32>(1)?,
                phb: s.alloc_zeros::<f32>(1)?,
                u_stag: s.alloc_zeros::<f32>(1)?,
                v_stag: s.alloc_zeros::<f32>(1)?,
                mu: s.alloc_zeros::<f32>(1)?,
                mub: s.alloc_zeros::<f32>(1)?,
                c3f: s.alloc_zeros::<f32>(1)?,
                c4f: s.alloc_zeros::<f32>(1)?,
            };
            s.synchronize()?;
            let state_upload_s = t_up.elapsed().as_secs_f64();
            let _ = group_d::cuda::compute_device(&g, &d_state, &d_in, &in_big, &cb, &opt)?;
            let mut kern = Vec::new();
            let mut last = None;
            for _ in 0..5 {
                let (o, tm) = group_d::cuda::compute_device(&g, &d_state, &d_in, &in_big, &cb, &opt)?;
                kern.push(tm.kernel);
                last = Some((o, tm));
            }
            kern.sort_by(f64::total_cmp);
            let (o_big, tm) = last.expect("timed");
            let t = Instant::now();
            let cpu_big = group_d::compute_cpu(&st_big, &in_big, &cb, &opt)?;
            let cpu_big_s = t.elapsed().as_secs_f64();
            summary["tiled_timing"] = json!({
                "grid": [nz, big.ny, big.nx],
                "kernel_s_median": kern[2],
                "carrier_upload_s": tm.upload,
                "download_s": tm.download,
                "state_and_hydrometeor_upload_s": state_upload_s,
                "cpu_s": cpu_big_s,
                "gpu_cpu_bit_differences": diff(&cpu_big.planes, &o_big.planes),
            });
            eprintln!("tiled: {}", summary["tiled_timing"]);
        }
    }
    std::fs::write(out.join("summary.json"), serde_json::to_vec_pretty(&summary)?)?;
    Ok(())
}
