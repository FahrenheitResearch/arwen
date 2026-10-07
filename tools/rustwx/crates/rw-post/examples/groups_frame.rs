//! Run every Rust group (A state and surface, B, C, E) on one WRF-format
//! WOOF history frame, on the CPU and on GPU 0, from ONE shared column state
//! (Group A's), and write each group's planes plus a summary with timings
//! and the CPU/GPU bit comparison.
//!
//!   groups_frame HISTORY OUTDIR [--no-gpu] [--gust similarity|tke]
//!
//! Layout (little-endian f32, x fastest, south row first):
//!   OUTDIR/summary.json
//!   OUTDIR/b/new/{cpu,gpu}/{plev,pwat,mslp,maps_mslp,membrane_slp,t700}.f32
//!   OUTDIR/b/new/summary.json      (nx, ny, levels_pa, plev_fields)
//!   OUTDIR/c/{field}.{cpu,gpu}.f32
//!   OUTDIR/e/{cpu,gpu}/{field}.f32

use std::path::{Path, PathBuf};
use std::time::Instant;

use rw_post::group_e::{self, GustMethod, OUT_NAMES, Options, history::OwnedCarriers};
use rw_post::plev::{self, PLEV_FIELDS, frame::Frame as BFrame};
use rw_post::severe::{self, ColumnState as CState, FIELDS};
use rw_post::state::ColumnState;
use rw_post::surface::SurfaceField;
use rw_post::{history, state};
use serde_json::{Value, json};

fn write_f32(path: &Path, v: &[f32]) -> std::io::Result<()> {
    if let Some(d) = path.parent() {
        std::fs::create_dir_all(d)?;
    }
    let b: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
    std::fs::write(path, b)
}

/// Words that differ in bits (two NaNs count as equal).
fn diff(a: &[f32], b: &[f32]) -> usize {
    a.iter().zip(b).filter(|(x, y)| x.to_bits() != y.to_bits() && !(x.is_nan() && y.is_nan())).count()
}

fn save_b(dir: &Path, o: &plev::Output) -> std::io::Result<()> {
    write_f32(&dir.join("plev.f32"), &o.plev)?;
    write_f32(&dir.join("pwat.f32"), &o.pwat)?;
    write_f32(&dir.join("mslp.f32"), &o.mslp)?;
    write_f32(&dir.join("maps_mslp.f32"), &o.maps_mslp)?;
    write_f32(&dir.join("membrane_slp.f32"), &o.membrane_slp)?;
    write_f32(&dir.join("t700.f32"), &o.t700)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: groups_frame HISTORY OUTDIR [--no-gpu] [--gust similarity|tke]");
        std::process::exit(2);
    }
    let hist = PathBuf::from(&args[1]);
    let out = PathBuf::from(&args[2]);
    let mut use_gpu = true;
    let mut opt = Options::default();
    let mut i = 3;
    while i < args.len() {
        match args[i].as_str() {
            "--no-gpu" => use_gpu = false,
            "--gust" => {
                i += 1;
                opt.gust = if args[i] == "tke" { GustMethod::MixedLayerTke } else { GustMethod::Similarity };
            }
            other => return Err(format!("unknown argument {other}").into()),
        }
        i += 1;
    }
    std::fs::create_dir_all(&out)?;

    // ---- read: Group A's frame, plus the 2D carriers groups C and E read.
    let t = Instant::now();
    let mut frame = history::Frame::read(&hist, 0)?;
    let owned = OwnedCarriers::read(&hist)?;
    let file = netcrust::File::open(&hist)?;
    let dx = file.attribute("DX").and_then(|a| a.as_f64()).ok_or("DX attribute")?;
    let read_s = t.elapsed().as_secs_f64();
    let shape = frame.shape;
    let n = shape.ncell();
    let has_phyd = frame.state.has(state::MassInput::PHyd);
    let pressure_source = if has_phyd {
        "history P_HYD on mass levels; interfaces by moist hydrostatic integration (Group A state)"
    } else {
        "moist hydrostatic integration of the dry column mass (Group A state, D1)"
    };
    let vertical_source = frame.coefficients.as_str();

    // ---- Group A on the CPU: the one shared state, and the surface fields.
    let t = Instant::now();
    let a_cpu = rw_post::group_a_cpu(&mut frame)?;
    let a_cpu_s = t.elapsed().as_secs_f64();
    let st: &ColumnState = &a_cpu.state;
    let surf = |f: SurfaceField| &a_cpu.surface[f as usize * n..(f as usize + 1) * n];

    let mut summary = json!({
        "history": hist.display().to_string(),
        "shape": {"nx": shape.nx, "ny": shape.ny, "nz": shape.nz},
        "dx": dx,
        "pressure_source": pressure_source,
        "vertical_source": vertical_source,
        "read_seconds": read_s,
        "group_a_cpu_seconds": a_cpu_s,
        "rayon_threads": rayon::current_num_threads(),
    });

    let gpu = if use_gpu { Some(rw_post::gpu::GpuPost::open(0)?) } else { None };
    if let Some(g) = &gpu {
        summary["gpu_device"] = json!(g.device.describe());
        let mut f2 = history::Frame::read(&hist, 0)?;
        let t = Instant::now();
        let a_gpu = rw_post::group_a_gpu(g, &mut f2)?;
        let a_gpu_s = t.elapsed().as_secs_f64();
        summary["group_a"] = json!({
            "gpu_seconds_incl_upload": a_gpu_s,
            "state_mass_words_differing": diff(&a_cpu.state.mass, &a_gpu.state.mass),
            "state_interface_words_differing": diff(&a_cpu.state.interface, &a_gpu.state.interface),
            "surface_words_differing": diff(&a_cpu.surface, &a_gpu.surface),
        });
        eprintln!("group A cpu/gpu: {}", summary["group_a"]);
    }

    // ---- Group B.
    let bf = BFrame::new(st, &owned.vars["PSFC"], &owned.vars["HGT"], dx, pressure_source, vertical_source);
    let levels: Vec<f64> = (0..37).map(|i| 10000.0 + 2500.0 * i as f64).collect();
    let bdir = out.join("b/new");
    let t = Instant::now();
    let b_cpu = plev::compute_cpu(&bf, &levels)?;
    let b_cpu_s = t.elapsed().as_secs_f64();
    save_b(&bdir.join("cpu"), &b_cpu)?;
    let mut bsum = json!({
        "history": hist.display().to_string(), "nx": shape.nx, "ny": shape.ny, "nz": shape.nz, "dx": dx,
        "levels_pa": levels, "plev_fields": PLEV_FIELDS, "cpu_seconds": b_cpu_s,
        "maps_passes": b_cpu.maps_passes, "membrane_active_levels": b_cpu.membrane_active_levels,
        "pressure_source": b_cpu.pressure_source, "vertical_source": b_cpu.vertical_source,
    });
    if let Some(g) = &gpu {
        let _warm = plev::gpu::compute(&bf, &levels, g)?;
        let t = Instant::now();
        let b_gpu = plev::gpu::compute(&bf, &levels, g)?;
        bsum["gpu_seconds"] = json!(t.elapsed().as_secs_f64());
        bsum["gpu_device"] = json!(b_gpu.device);
        save_b(&bdir.join("gpu"), &b_gpu)?;
        let d = json!({
            "plev": diff(&b_cpu.plev, &b_gpu.plev),
            "pwat": diff(&b_cpu.pwat, &b_gpu.pwat),
            "mslp": diff(&b_cpu.mslp, &b_gpu.mslp),
            "maps_mslp": diff(&b_cpu.maps_mslp, &b_gpu.maps_mslp),
            "membrane_slp": diff(&b_cpu.membrane_slp, &b_gpu.membrane_slp),
            "t700": diff(&b_cpu.t700, &b_gpu.t700),
            "gpu_run_to_run": diff(&_warm.plev, &b_gpu.plev) + diff(&_warm.mslp, &b_gpu.mslp) + diff(&_warm.maps_mslp, &b_gpu.maps_mslp),
        });
        bsum["gpu_cpu_bit_differences"] = d;
    }
    std::fs::write(bdir.join("summary.json"), serde_json::to_vec_pretty(&bsum)?)?;
    eprintln!("group B: {}", bsum.get("gpu_cpu_bit_differences").unwrap_or(&Value::Null));
    summary["group_b"] = bsum;

    // ---- Group C.
    let v = &owned.vars;
    let q2: Vec<f32> = v["Q2"].clone();
    let cs = CState::from_shared(
        st,
        &v["HGT"],
        &v["PSFC"],
        surf(SurfaceField::T2),
        &q2,
        surf(SurfaceField::ShelterPressure),
        &v["U10"],
        &v["V10"],
    );
    let cdir = out.join("c");
    let t = Instant::now();
    let c_cpu = severe::compute_cpu(&cs)?;
    let c_cpu_s = t.elapsed().as_secs_f64();
    for (i, name) in FIELDS.iter().enumerate() {
        write_f32(&cdir.join(format!("{name}.cpu.f32")), &c_cpu.planes[i * n..(i + 1) * n])?;
    }
    let mut csum = json!({"cpu_seconds": c_cpu_s, "inputs": "Group A shared state; t2 and p2 from Group A surface; Q2, U10, V10, HGT, PSFC from the history"});
    if let Some(g) = &gpu {
        let sg = severe::gpu::SevereGpu::new(g);
        let (a, _) = sg.compute(&cs)?;
        let (b, tb) = sg.compute(&cs)?;
        for (i, name) in FIELDS.iter().enumerate() {
            write_f32(&cdir.join(format!("{name}.gpu.f32")), &b.planes[i * n..(i + 1) * n])?;
        }
        let mut per = serde_json::Map::new();
        for (i, name) in FIELDS.iter().enumerate() {
            per.insert(name.to_string(), json!(diff(&c_cpu.planes[i * n..(i + 1) * n], &b.planes[i * n..(i + 1) * n])));
        }
        csum["gpu_timing"] = json!({"upload": tb.upload, "kernels": tb.kernels, "download": tb.download});
        csum["gpu_run_to_run"] = json!(diff(&a.planes, &b.planes));
        csum["gpu_cpu_bit_differences"] = Value::Object(per);
    }
    std::fs::write(cdir.join("summary.json"), serde_json::to_vec_pretty(&csum)?)?;
    eprintln!("group C: {}", csum.get("gpu_cpu_bit_differences").unwrap_or(&Value::Null));
    summary["group_c"] = csum;

    // ---- Group E.
    let ec = owned.carriers(st);
    let edir = out.join("e");
    let t = Instant::now();
    let e_cpu = group_e::compute_cpu(&ec, opt)?;
    let e_cpu_s = t.elapsed().as_secs_f64();
    for (i, name) in OUT_NAMES.iter().enumerate() {
        write_f32(&edir.join("cpu").join(format!("{name}.f32")), &e_cpu.planes[i * n..(i + 1) * n])?;
    }
    let mut esum = json!({
        "cpu_seconds": e_cpu_s,
        "gust_method": format!("{:?}", opt.gust),
        "tke_source": owned.tke_source,
        "omitted": e_cpu.omitted.iter().map(|(a, b)| json!([a, b])).collect::<Vec<_>>(),
    });
    if let Some(g) = &gpu {
        let e_gpu = group_e::compute_gpu(&ec, opt, g)?;
        let e_gpu2 = group_e::compute_gpu(&ec, opt, g)?;
        for (i, name) in OUT_NAMES.iter().enumerate() {
            write_f32(&edir.join("gpu").join(format!("{name}.f32")), &e_gpu.planes[i * n..(i + 1) * n])?;
        }
        let mut per = serde_json::Map::new();
        for (i, name) in OUT_NAMES.iter().enumerate() {
            per.insert(name.to_string(), json!(diff(&e_cpu.planes[i * n..(i + 1) * n], &e_gpu.planes[i * n..(i + 1) * n])));
        }
        esum["gpu_run_to_run"] = json!(diff(&e_gpu.planes, &e_gpu2.planes));
        esum["gpu_cpu_bit_differences"] = Value::Object(per);
    }
    std::fs::write(edir.join("summary.json"), serde_json::to_vec_pretty(&esum)?)?;
    eprintln!("group E: {}", esum.get("gpu_cpu_bit_differences").unwrap_or(&Value::Null));
    summary["group_e"] = esum;

    std::fs::write(out.join("summary.json"), serde_json::to_vec_pretty(&summary)?)?;
    Ok(())
}
