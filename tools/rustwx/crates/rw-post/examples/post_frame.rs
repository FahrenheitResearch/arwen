//! Run Group A on one history frame, on the CPU, the GPU or both.
//!
//! ```text
//! post_frame HISTORY --out DIR [--device cpu|gpu|both|auto] [--record N]
//!            [--reference-stride S]
//! ```
//!
//! Writes one little-endian f32 plane per available product to
//! `DIR/<id>.f32` (WRF order: x fastest, rows south to north), `cosz.f32`
//! and `p2.f32`, and `DIR/manifest.json` with the GRIB identity, method,
//! availability, statistics, timings, the CPU/GPU identity check and the
//! binary64 reference check.

use std::path::PathBuf;
use std::time::Instant;

use rw_post::history::Frame;
use rw_post::reference;
use rw_post::state::{ColumnState, InterfaceState, MassInput, MassState};
use rw_post::surface::{SurfaceField, availability};
use serde_json::{Value, json};

/// Repeat each level of a `[levels][ny][nx]` array `t` x `t` times.
fn tile_levels(v: &[f32], nx: usize, ny: usize, t: usize) -> Vec<f32> {
    let levels = v.len() / (nx * ny);
    let (bx, by) = (nx * t, ny * t);
    let mut out = Vec::with_capacity(levels * bx * by);
    for k in 0..levels {
        for j in 0..by {
            for i in 0..bx {
                out.push(v[(k * ny + j % ny) * nx + i % nx]);
            }
        }
    }
    out
}

/// Tile a staggered array whose fastest axis has `nx + sx` and next `ny + sy`
/// points: the tiled grid has `t nx + sx` by `t ny + sy`.
fn tile_staggered(v: &[f32], nx: usize, ny: usize, sx: usize, sy: usize, t: usize) -> Vec<f32> {
    let (wx, wy) = (nx + sx, ny + sy);
    let levels = v.len() / (wx * wy);
    let (bx, by) = (nx * t + sx, ny * t + sy);
    let mut out = Vec::with_capacity(levels * bx * by);
    for k in 0..levels {
        for j in 0..by {
            let sj = if j == by - 1 && sy == 1 { ny } else { j % ny };
            for i in 0..bx {
                let si = if i == bx - 1 && sx == 1 { nx } else { i % nx };
                out.push(v[(k * wy + sj) * wx + si]);
            }
        }
    }
    out
}

/// A `t` x `t` horizontal tiling of a frame, for timing at larger sizes.
fn tile_frame(frame: &mut Frame, t: usize) {
    use rw_post::surface::SurfaceInputs;
    let s = frame.state.shape;
    let (nx, ny) = (s.nx, s.ny);
    let st = &mut frame.state;
    st.mass = tile_levels(&st.mass, nx, ny, t);
    st.ph = tile_levels(&st.ph, nx, ny, t);
    st.phb = tile_levels(&st.phb, nx, ny, t);
    st.mu = tile_levels(&st.mu, nx, ny, t);
    st.mub = tile_levels(&st.mub, nx, ny, t);
    st.u_stag = tile_staggered(&st.u_stag, nx, ny, 1, 0, t);
    st.v_stag = tile_staggered(&st.v_stag, nx, ny, 0, 1, t);
    st.shape.nx = nx * t;
    st.shape.ny = ny * t;
    frame.shape = st.shape;
    let old = std::mem::replace(&mut frame.surface, SurfaceInputs::new(frame.shape.ncell()));
    frame.surface.planes = tile_levels(&old.planes, nx, ny, t);
    frame.surface.present = old.present;
}

fn bits_differ(a: &[f32], b: &[f32]) -> usize {
    a.iter().zip(b).filter(|(x, y)| x.to_bits() != y.to_bits()).count()
}

fn stats(v: &[f32]) -> Value {
    let finite: Vec<f64> = v.iter().filter(|x| x.is_finite()).map(|&x| x as f64).collect();
    if finite.is_empty() {
        return json!({"finite": 0});
    }
    let min = finite.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = finite.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mean = finite.iter().sum::<f64>() / finite.len() as f64;
    json!({"finite": finite.len(), "missing": v.len() - finite.len(), "min": min, "max": max, "mean": mean})
}

/// Binary64 reference check of the state on every `stride`-th column and of
/// the shelter fields on every cell.
fn reference_check(frame: &Frame, st: &ColumnState, surface: &[f32], stride: usize) -> Value {
    let s = frame.state.shape;
    let n = s.ncell();
    let (nx, ny, nz) = (s.nx, s.ny, s.nz);
    let inp = &frame.state;
    let col = |slot: MassInput, c: usize| -> Vec<f64> {
        (0..nz).map(|k| inp.slot(slot)[k * n + c] as f64).collect()
    };
    let mut worst = [0.0f64; 6]; // p_int, p, tk, tv, z_mass, theta
    let mut checked = 0usize;
    let mut c = 0usize;
    while c < n {
        let condensate: Vec<Vec<f64>> = [
            MassInput::QCloud,
            MassInput::QRain,
            MassInput::QIce,
            MassInput::QSnow,
            MassInput::QGraupel,
            MassInput::QHail,
        ]
        .into_iter()
        .filter(|&q| inp.has(q))
        .map(|q| col(q, c))
        .collect();
        let phyd = inp.has(MassInput::PHyd).then(|| col(MassInput::PHyd, c));
        let ph: Vec<f64> = (0..=nz).map(|k| inp.ph[k * n + c] as f64).collect();
        let phb: Vec<f64> = (0..=nz).map(|k| inp.phb[k * n + c] as f64).collect();
        let c3f: Vec<f64> = inp.c3f.iter().map(|&x| x as f64).collect();
        let c4f: Vec<f64> = inp.c4f.iter().map(|&x| x as f64).collect();
        let r = reference::column(&reference::ColumnIn {
            t_pert: &col(MassInput::T, c),
            p_pert: &col(MassInput::P, c),
            p_base: &col(MassInput::PB, c),
            qv: &col(MassInput::QVapor, c),
            condensate: &condensate,
            p_hyd: phyd.as_deref(),
            ph: &ph,
            phb: &phb,
            mu_d: inp.mu[c] as f64 + inp.mub[c] as f64,
            c3f: &c3f,
            c4f: &c4f,
            p_top: inp.p_top as f64,
        });
        for k in 0..=nz {
            let d = (st.interface_slot(InterfaceState::PInt)[k * n + c] as f64 - r.p_int[k]).abs();
            worst[0] = worst[0].max(d);
        }
        for k in 0..nz {
            let at = k * n + c;
            let pairs = [
                (MassState::P, r.p[k], 1),
                (MassState::Tk, r.tk[k], 2),
                (MassState::Tv, r.tv[k], 3),
                (MassState::ZMass, r.z_mass[k], 4),
                (MassState::Theta, r.theta[k], 5),
            ];
            for (slot, refv, w) in pairs {
                let d = (st.mass_slot(slot)[at] as f64 - refv).abs();
                worst[w] = worst[w].max(d);
            }
        }
        checked += 1;
        c += stride.max(1);
    }
    // Shelter fields on every cell.
    let plane = |f: SurfaceField| &surface[f as usize * n..(f as usize + 1) * n];
    let mut sw = [0.0f64; 4]; // p2, t2, td2, rh2
    let sfc = &frame.surface;
    use rw_post::surface::SurfaceInput as I;
    if sfc.has(I::Psfc) && sfc.has(I::Q2) && sfc.has(I::TvLowest) && (sfc.has(I::T2) || sfc.has(I::Th2)) {
        for c in 0..n {
            let t2 = sfc.has(I::T2).then(|| sfc.plane(I::T2)[c] as f64);
            let th2 = if sfc.has(I::Th2) { sfc.plane(I::Th2)[c] as f64 } else { f64::NAN };
            let (p2, t2r, _q2, td2, rh2) = reference::shelter(
                sfc.plane(I::Psfc)[c] as f64,
                sfc.plane(I::TvLowest)[c] as f64,
                t2,
                th2,
                sfc.plane(I::Q2)[c] as f64,
            );
            let ours = [
                plane(SurfaceField::ShelterPressure)[c],
                plane(SurfaceField::T2)[c],
                plane(SurfaceField::Td2)[c],
                plane(SurfaceField::Rh2)[c],
            ];
            for (w, (o, r)) in ours.iter().zip([p2, t2r, td2, rh2]).enumerate() {
                if r.is_finite() && o.is_finite() {
                    sw[w] = sw[w].max((*o as f64 - r).abs());
                }
            }
        }
    }
    // Solar zenith cosine against the binary64 algorithm on every cell.
    let mut cz = 0.0f64;
    if sfc.has(I::XLat) && sfc.has(I::XLong) {
        let jd = frame.valid.julian_date();
        let ut = frame.valid.ut_hours();
        for c in 0..n {
            let r = reference::cosz(jd, ut, sfc.plane(I::XLat)[c] as f64, sfc.plane(I::XLong)[c] as f64);
            cz = cz.max((plane(SurfaceField::Cosz)[c] as f64 - r).abs());
        }
    }
    let _ = (nx, ny);
    json!({
        "columns_checked": checked,
        "state_max_abs": {"p_int_pa": worst[0], "p_pa": worst[1], "tk_k": worst[2], "tv_k": worst[3], "z_mass_m": worst[4], "theta_k": worst[5]},
        "shelter_max_abs": {"p2_pa": sw[0], "t2_k": sw[1], "td2_k": sw[2], "rh2_pct": sw[3]},
        "cosz_max_abs": cz,
    })
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut history = None;
    let mut out = None;
    let mut device = "both".to_owned();
    let mut record = 0usize;
    let mut stride = 97usize;
    let mut tile = 1usize;
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--out" => out = it.next().map(PathBuf::from),
            "--device" => device = it.next().cloned().unwrap_or_default(),
            "--record" => record = it.next().and_then(|s| s.parse().ok()).unwrap_or(0),
            "--reference-stride" => stride = it.next().and_then(|s| s.parse().ok()).unwrap_or(97),
            "--tile" => tile = it.next().and_then(|s| s.parse().ok()).unwrap_or(1),
            other => history = Some(PathBuf::from(other)),
        }
    }
    let history = history.ok_or("usage: post_frame HISTORY --out DIR [--device cpu|gpu|both|auto]")?;
    let out = out.ok_or("--out is required")?;
    std::fs::create_dir_all(&out)?;

    let t0 = Instant::now();
    let mut frame = Frame::read(&history, record)?;
    let read_s = t0.elapsed().as_secs_f64();
    if tile > 1 {
        tile_frame(&mut frame, tile);
    }
    let shape = frame.state.shape;
    eprintln!("read {} ({}x{}x{}) in {read_s:.2} s", history.display(), shape.nx, shape.ny, shape.nz);

    let mut timings = serde_json::Map::new();
    timings.insert("read_s".into(), json!(read_s));
    let mut cpu = None;
    if device == "cpu" || device == "both" {
        let t = Instant::now();
        let a = rw_post::group_a_cpu(&mut frame)?;
        timings.insert("cpu_group_a_s".into(), json!(t.elapsed().as_secs_f64()));
        cpu = Some(a);
    }
    let mut gpu_out = None;
    let mut gpu_info = Value::Null;
    #[cfg(any(windows, target_os = "linux"))]
    if device == "gpu" || device == "both" || device == "auto" {
        let choice = if device == "auto" { rw_post::PostDevice::Auto } else { rw_post::PostDevice::Gpu };
        if let Some(g) = rw_post::open_device(choice, &shape)? {
            // Warm-up run (module load and first-touch), then timed phases.
            let _ = rw_post::group_a_gpu(&g, &mut frame)?;
            g.synchronize()?;
            let t = Instant::now();
            let dins = g.upload_state_inputs(&frame.state)?;
            g.synchronize()?;
            timings.insert("gpu_upload_3d_s".into(), json!(t.elapsed().as_secs_f64()));
            let t = Instant::now();
            let dst = g.run_state(&dins)?;
            g.synchronize()?;
            timings.insert("gpu_state_kernel_s".into(), json!(t.elapsed().as_secs_f64()));
            let t = Instant::now();
            let sfc = g.surface(&frame.surface, &frame.options, Some(&dst))?;
            g.synchronize()?;
            timings.insert("gpu_surface_upload_kernel_download_s".into(), json!(t.elapsed().as_secs_f64()));
            let t = Instant::now();
            let gst = g.download_state(&dst)?;
            timings.insert("gpu_full_state_download_s_for_checks_only".into(), json!(t.elapsed().as_secs_f64()));
            drop((dins, dst));
            frame.attach_tv_lowest(&rw_post::tv_lowest(&gst));
            let d = &g.device;
            let a = rw_post::GroupA {
                state: gst,
                surface: sfc,
                device: format!("gpu {} (sm_{}{}, {})", d.name, d.compute_capability.0, d.compute_capability.1, d.kernel_artifact),
            };
            gpu_info = json!({
                "name": g.device.name,
                "compute_capability": format!("{}.{}", g.device.compute_capability.0, g.device.compute_capability.1),
                "kernel_artifact": g.device.kernel_artifact,
            });
            gpu_out = Some(a);
        }
    }

    let mut identity = Value::Null;
    if let (Some(c), Some(g)) = (&cpu, &gpu_out) {
        identity = json!({
            "state_mass_words_differing": bits_differ(&c.state.mass, &g.state.mass),
            "state_interface_words_differing": bits_differ(&c.state.interface, &g.state.interface),
            "surface_words_differing": bits_differ(&c.surface, &g.surface),
            "words_compared": c.state.mass.len() + c.state.interface.len() + c.surface.len(),
        });
        eprintln!("CPU/GPU identity: {identity}");
    }
    let primary = gpu_out.as_ref().or(cpu.as_ref()).ok_or("no device ran")?;
    let reference = reference_check(&frame, &primary.state, &primary.surface, stride);
    eprintln!("binary64 reference: {reference}");

    let n = shape.ncell();
    let mut fields = Vec::new();
    for spec in SurfaceField::PRODUCTS {
        let avail = availability(&spec, &frame.surface, &frame.options);
        let plane = primary.plane(spec.field);
        let mut entry = json!({
            "id": spec.id,
            "grib2": {"discipline": spec.discipline, "category": spec.category, "number": spec.number,
                       "level_type": spec.level_type, "level_value": spec.level_value},
            "units": spec.units,
            "method": spec.method,
        });
        match avail {
            Ok(()) => {
                let bytes: Vec<u8> = plane.iter().flat_map(|v| v.to_le_bytes()).collect();
                std::fs::write(out.join(format!("{}.f32", spec.id)), bytes)?;
                entry["available"] = json!(true);
                entry["stats"] = stats(plane);
            }
            Err(reason) => {
                entry["available"] = json!(false);
                entry["omitted"] = json!(reason);
            }
        }
        fields.push(entry);
    }
    // Diagnostic: the hydrostatic ground pressure of the state (interface 0),
    // compared with other exporters' surface pressure as a cross-check of
    // the hydrostatic reconstruction.  Not a product (D3 picks PSFC).
    let ground: Vec<f32> = primary.state.interface_slot(InterfaceState::PInt)[..n].to_vec();
    std::fs::write(
        out.join("hydrostatic_ground_pressure.f32"),
        ground.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>(),
    )?;
    let diagnostics = json!([{
        "id": "hydrostatic_ground_pressure",
        "grib2": {"discipline": 0, "category": 3, "number": 0, "level_type": 1, "level_value": 0.0},
        "units": "Pa",
        "method": "moist hydrostatic integration of the dry column (state p_int at the ground)",
        "available": true,
        "stats": stats(&ground),
    }]);
    for (name, f) in [("cosz", SurfaceField::Cosz), ("p2", SurfaceField::ShelterPressure)] {
        let plane = primary.plane(f);
        let bytes: Vec<u8> = plane.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(out.join(format!("{name}.f32")), bytes)?;
    }
    let manifest = json!({
        "history": history.display().to_string(),
        "valid": frame.valid_text,
        "shape": {"nx": shape.nx, "ny": shape.ny, "nz": shape.nz, "ncell": n},
        "device": primary.device,
        "gpu": gpu_info,
        "coefficients": frame.coefficients.as_str(),
        "solar": {"declination_deg": frame.solar.declination_deg, "hour_angle0_deg": frame.solar.hour_angle0_deg},
        "land_use_table": frame.land_use_table,
        "sf_surface_physics": frame.sf_surface_physics,
        "options": {"vegfra_scale": frame.options.vegfra_scale, "latent_from_qfx": frame.options.latent_from_qfx,
                     "snow_cover_derived": frame.options.snow_cover_derived},
        "p_hyd_present": frame.state.has(MassInput::PHyd),
        "carriers": frame.carriers,
        "timings": timings,
        "cpu_gpu_identity": identity,
        "reference_check": reference,
        "fields": fields,
        "diagnostics": diagnostics,
    });
    std::fs::write(out.join("manifest.json"), serde_json::to_vec_pretty(&manifest)?)?;
    println!("{}", serde_json::to_string(&manifest["timings"])?);
    Ok(())
}
