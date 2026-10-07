//! Group D: reflectivity, cloud and visibility (spec section 7).
//!
//! CPU tests check the methods against hand-computed values; the device test
//! (`--ignored`, needs an NVIDIA GPU) checks that the GPU returns the CPU's
//! bits in every mode, and fails, not skips, when the kernels do not load.

mod common;

use rw_post::group_d::{self, Carriers, CloudFractionSource, DiagnosedCloud, Options, ReflInterp};
use rw_post::state::{ColumnState, MassInput, MassState, StateInputs, build_cpu};

fn frame(nx: usize, ny: usize, nz: usize, seed: u64) -> (StateInputs, ColumnState, Vec<f32>) {
    let inp = common::synthetic_state(nx, ny, nz, seed);
    let st = build_cpu(&inp).unwrap();
    let n = inp.shape.ncell();
    let hgt: Vec<f32> = (0..n).map(|c| (inp.ph[c] + inp.phb[c]) / 9.81).collect();
    (inp, st, hgt)
}

fn set_slot(inp: &mut StateInputs, slot: MassInput, f: impl Fn(usize, usize) -> f32) {
    let v = inp.shape.mass_volume();
    let n = inp.shape.ncell();
    for k in 0..inp.shape.nz {
        for c in 0..n {
            inp.mass[slot as usize * v + k * n + c] = f(k, c);
        }
    }
    inp.present |= 1 << slot as u32;
}

fn differing(a: &[f32], b: &[f32]) -> usize {
    a.iter().zip(b).filter(|(x, y)| x.to_bits() != y.to_bits()).count()
}

#[test]
fn composite_and_1km_reflectivity_from_refl_10cm() {
    let (inp, st, hgt) = frame(6, 5, 30, 11);
    let n = inp.shape.ncell();
    let nz = inp.shape.nz;
    let refl: Vec<f32> = (0..nz * n).map(|i| (i / n) as f32 * 1.5 - 10.0).collect();
    let c = Carriers { refl: Some(&refl), cldfra: None, aerosol_km: None, hgt: Some(&hgt), olr: None };
    let o = group_d::compute_cpu(&st, &inp, &c, &Options::default()).unwrap();
    let comp = o.plane("composite_reflectivity").unwrap();
    let r1 = o.plane("reflectivity_1km").unwrap();
    let zm = st.mass_slot(MassState::ZMass);
    for cell in 0..n {
        assert_eq!(comp[cell], (nz - 1) as f32 * 1.5 - 10.0);
        let k = (0..nz).find(|&k| f64::from(zm[k * n + cell]) - f64::from(hgt[cell]) >= 1000.0).unwrap();
        let zl = f64::from(zm[(k - 1) * n + cell]) - f64::from(hgt[cell]);
        let zh = f64::from(zm[k * n + cell]) - f64::from(hgt[cell]);
        let w = (1000.0 - zl) / (zh - zl);
        let (dl, dh) = (f64::from(refl[(k - 1) * n + cell]), f64::from(refl[k * n + cell]));
        let zz = 10f64.powf(dl / 10.0) + w * (10f64.powf(dh / 10.0) - 10f64.powf(dl / 10.0));
        let want = (10.0 * zz.log10()).max(-20.0);
        assert!((f64::from(r1[cell]) - want).abs() < 1e-4, "cell {cell}: {} vs {want}", r1[cell]);
        assert!(f64::from(r1[cell]) >= dl.max(-20.0) && f64::from(r1[cell]) <= dh);
    }
    // The dBZ alternative is one switch away and gives a different answer.
    let opt = Options { refl_interp: ReflInterp::LinearDbz, ..Options::default() };
    let o2 = group_d::compute_cpu(&st, &inp, &c, &opt).unwrap();
    assert!(differing(o2.plane("reflectivity_1km").unwrap(), r1) > 0);
}

#[test]
fn species_reflectivity_has_a_floor_and_grows_with_rain() {
    let (mut inp, _, hgt) = frame(4, 3, 12, 12);
    // Doubling the rain each cell outruns the column-to-column air-density spread.
    set_slot(&mut inp, MassInput::QRain, |k, c| if k == 2 { 3e-4 * (1u32 << c) as f32 } else { 0.0 });
    set_slot(&mut inp, MassInput::QSnow, |_, _| 0.0);
    set_slot(&mut inp, MassInput::QGraupel, |_, _| 0.0);
    let st = build_cpu(&inp).unwrap();
    let c = Carriers { refl: None, cldfra: None, aerosol_km: None, hgt: Some(&hgt), olr: None };
    let o = group_d::compute_cpu(&st, &inp, &c, &Options::default()).unwrap();
    let comp = o.plane("composite_reflectivity").unwrap();
    for cell in 1..comp.len() {
        assert!(comp[cell] > comp[cell - 1], "monotone in rain: {comp:?}");
    }
    // 0.3 g/kg of rain is a few tens of dBZ (Marshall-Palmer intercept).
    assert!(comp[0] > 15.0 && comp[0] < 40.0, "{}", comp[0]);
    set_slot(&mut inp, MassInput::QRain, |_, _| 0.0);
    let st = build_cpu(&inp).unwrap();
    let o = group_d::compute_cpu(&st, &inp, &c, &Options::default()).unwrap();
    assert!(o.plane("composite_reflectivity").unwrap().iter().all(|&d| d == -20.0));
}

#[test]
fn cloud_layers_base_top_and_ceiling_from_cldfra() {
    let (mut inp, _, hgt) = frame(3, 3, 30, 13);
    let n = inp.shape.ncell();
    let nz = inp.shape.nz;
    set_slot(&mut inp, MassInput::QCloud, |k, _| if (3..=5).contains(&k) { 2e-4 } else { 0.0 });
    set_slot(&mut inp, MassInput::QIce, |_, _| 0.0);
    set_slot(&mut inp, MassInput::QSnow, |_, _| 0.0);
    let st = build_cpu(&inp).unwrap();
    let cf: Vec<f32> = (0..nz * n).map(|i| if (3..=5).contains(&(i / n)) { 0.8 } else { 0.0 }).collect();
    let c = Carriers { refl: None, cldfra: Some(&cf), aerosol_km: None, hgt: Some(&hgt), olr: None };
    let o = group_d::compute_cpu(&st, &inp, &c, &Options::default()).unwrap();
    assert!(o.cloud_fraction_source.unwrap().starts_with("CLDFRA"));
    let zm = st.mass_slot(MassState::ZMass);
    let pf = st.mass_slot(MassState::PFull);
    for cell in 0..n {
        let base = o.plane("cloud_base").unwrap()[cell];
        let top = o.plane("cloud_top").unwrap()[cell];
        let ceil = o.plane("cloud_ceiling").unwrap()[cell];
        assert!(base > zm[2 * n + cell] && base <= zm[3 * n + cell], "base {base}");
        assert!(top >= zm[5 * n + cell] && top < zm[6 * n + cell], "top {top}");
        assert!(ceil >= base && ceil <= zm[3 * n + cell], "ceiling {ceil}");
        let in_low = (3..=5).any(|k| pf[k * n + cell] > 68000.0);
        let low = o.plane("low_cloud").unwrap()[cell];
        assert_eq!(low, if in_low { 80.0 } else { 0.0 });
    }
    // Thin cloud (fraction below 0.5): base and top, but no ceiling.
    let thin: Vec<f32> = cf.iter().map(|&v| v * 0.5).collect();
    let c = Carriers { cldfra: Some(&thin), ..c };
    let o = group_d::compute_cpu(&st, &inp, &c, &Options::default()).unwrap();
    assert!(o.plane("cloud_base").unwrap().iter().all(|v| v.is_finite()));
    assert!(o.plane("cloud_ceiling").unwrap().iter().all(|v| v.is_nan()));
}

#[test]
fn cloud_fraction_source_switch() {
    let (mut inp, _, hgt) = frame(5, 4, 20, 14);
    // Saturate level 4 and put condensate there; dry the rest.
    let st0 = build_cpu(&inp).unwrap();
    let n = inp.shape.ncell();
    let v = inp.shape.mass_volume();
    let tk = st0.mass_slot(MassState::Tk).to_vec();
    let pf = st0.mass_slot(MassState::PFull).to_vec();
    for k in 0..inp.shape.nz {
        for c in 0..n {
            let i = k * n + c;
            let t = f64::from(tk[i]) - 273.15;
            let es = if t >= 0.0 { 610.94 * (17.625 * t / (t + 243.04)).exp() } else { 611.21 * (22.587 * t / (t + 273.86)).exp() };
            let rs = 0.622 * es / (f64::from(pf[i]) - es);
            let (r, qc) = if k == 4 { (rs * 1.01, 3e-4) } else if k == 8 { (rs * 0.9, 1e-5) } else { (rs * 0.3, 0.0) };
            inp.mass[MassInput::QVapor as usize * v + i] = r as f32;
            inp.mass[MassInput::QCloud as usize * v + i] = qc as f32;
            inp.mass[MassInput::QIce as usize * v + i] = 0.0;
            inp.mass[MassInput::QSnow as usize * v + i] = 0.0;
        }
    }
    let st = build_cpu(&inp).unwrap();
    let c = Carriers { refl: None, cldfra: None, aerosol_km: None, hgt: Some(&hgt), olr: None };
    // RULINGS 1 default: no CLDFRA, so the diagnosed fraction is used.
    let o = group_d::compute_cpu(&st, &inp, &c, &Options::default()).unwrap();
    assert!(o.cloud_fraction_source.unwrap().contains("Xu and Randall"));
    assert!(!o.omitted.iter().any(|(id, _)| *id == "cloud_ceiling"));
    let ceil = o.plane("cloud_ceiling").unwrap();
    let zm = st.mass_slot(MassState::ZMass);
    for cell in 0..n {
        // Saturated level 4 is overcast (fraction 1): the ceiling is there or just below.
        assert!(ceil[cell] > zm[3 * n + cell] && ceil[cell] <= zm[4 * n + cell], "ceiling {}", ceil[cell]);
        assert!(o.plane("cloud_top").unwrap()[cell] >= zm[8 * n + cell], "subsaturated cloud at level 8 counts");
    }
    // ModelOnly reproduces the pre-ruling policy: six fields omitted.
    let opt = Options { cloud_fraction: CloudFractionSource::ModelOnly, ..Options::default() };
    let o = group_d::compute_cpu(&st, &inp, &c, &opt).unwrap();
    assert_eq!(o.omitted.iter().filter(|(id, _)| id.contains("cloud")).count(), 6);
    assert!(o.cloud_fraction_source.is_none());
    // Condensate presence: binary fraction.
    let opt = Options { diagnosed: DiagnosedCloud::CondensatePresence, ..Options::default() };
    let o = group_d::compute_cpu(&st, &inp, &c, &opt).unwrap();
    assert!(o.plane("mid_cloud").unwrap().iter().chain(o.plane("low_cloud").unwrap()).all(|&v| v == 0.0 || v == 100.0));
}

#[test]
fn visibility_haze_fog_and_aerosol_policy() {
    let (mut inp, _, hgt) = frame(4, 4, 6, 15);
    let n = inp.shape.ncell();
    set_slot(&mut inp, MassInput::QVapor, |_, _| 1e-4);
    set_slot(&mut inp, MassInput::QCloud, |k, c| if k == 0 && c % 2 == 1 { 1e-3 } else { 0.0 });
    for s in [MassInput::QRain, MassInput::QIce, MassInput::QSnow, MassInput::QGraupel] {
        set_slot(&mut inp, s, |_, _| 0.0);
    }
    let st = build_cpu(&inp).unwrap();
    let c = Carriers { refl: None, cldfra: None, aerosol_km: None, hgt: Some(&hgt), olr: None };
    // RULINGS 10: ON by default without an aerosol carrier (hydrometeors
    // plus the clear-air baseline); the switch still omits it with a reason.
    assert!(group_d::VIS_WITHOUT_AEROSOL && Options::default().allow_missing_aerosol);
    let off = Options { allow_missing_aerosol: false, ..Options::default() };
    let o = group_d::compute_cpu(&st, &inp, &c, &off).unwrap();
    assert!(o.omitted.iter().any(|(id, why)| *id == "visibility" && why.contains("absent smoke")));
    assert!(o.plane("visibility").unwrap().iter().all(|v| v.is_nan()));
    let o = group_d::compute_cpu(&st, &inp, &c, &Options::default()).unwrap();
    assert!(!o.omitted.iter().any(|(id, _)| *id == "visibility"), "{:?}", o.omitted);
    let vis = o.plane("visibility").unwrap();
    // The manifest note names the baseline when there is no carrier, and
    // the carriers when there are.
    let note = group_d::aerosol_note("absent", &Options::default());
    assert!(note.contains("clear-air baseline") && note.contains("RULINGS 10"), "{note}");
    assert!(group_d::aerosol_note("absent", &off).contains("omitted"));
    assert!(group_d::aerosol_note("AEXTC55", &Options::default()).starts_with("AEXTC55"));
    let aer = vec![1.0f32; n];
    let c2 = Carriers { aerosol_km: Some(&aer), ..c };
    let o2 = group_d::compute_cpu(&st, &inp, &c2, &Options::default()).unwrap();
    let vis2 = o2.plane("visibility").unwrap();
    for cell in 0..n {
        if cell % 2 == 0 {
            assert_eq!(vis[cell], 20000.0, "dry clear air is capped at 20 km");
            // 1 km-1 of smoke plus the haze term: MOR just under 3 km.
            assert!(vis2[cell] > 2700.0 && vis2[cell] < 2996.0, "{}", vis2[cell]);
        } else {
            // About 1.2 g m-3 of cloud water: tens of metres (Kunkel 1984).
            assert!(vis[cell] > 5.0 && vis[cell] < 40.0, "fog {}", vis[cell]);
        }
    }
    // The 2 % contrast alternative is one switch away and gives longer ranges.
    let opt2 = Options { contrast: group_d::CONTRAST_2PCT, ..Options::default() };
    let o3 = group_d::compute_cpu(&st, &inp, &c2, &opt2).unwrap();
    assert!(o3.plane("visibility").unwrap()[0] > vis2[0]);
}

#[test]
fn absent_carriers_omit_their_fields_with_a_reason() {
    let (mut inp, st, _) = frame(3, 3, 8, 16);
    inp.present &= !(1 << MassInput::QRain as u32);
    let c = Carriers { refl: None, cldfra: None, aerosol_km: None, hgt: None, olr: None };
    let o = group_d::compute_cpu(&st, &inp, &c, &Options::default()).unwrap();
    let omitted: Vec<&str> = o.omitted.iter().map(|(id, _)| *id).collect();
    assert!(omitted.contains(&"composite_reflectivity"));
    assert!(omitted.contains(&"reflectivity_1km"));
    assert!(omitted.contains(&"visibility"));
    assert!(o.plane("composite_reflectivity").unwrap().iter().all(|v| v.is_nan()));
}

#[cfg(any(windows, target_os = "linux"))]
#[test]
#[ignore = "needs an NVIDIA GPU"]
fn gpu_matches_cpu_bit_for_bit_in_every_mode() {
    use rw_post::gpu::GpuPost;
    let g = GpuPost::open(0).expect("an NVIDIA GPU with the post kernels");
    for (nx, ny, nz, seed) in [(37, 29, 50, 21u64), (64, 33, 20, 22), (5, 3, 2, 23)] {
        let (inp, st, hgt) = frame(nx, ny, nz, seed);
        let n = inp.shape.ncell();
        let dinp = g.upload_state_inputs(&inp).unwrap();
        let dst = g.run_state(&dinp).unwrap();
        assert_eq!(differing(&g.download_state(&dst).unwrap().mass, &st.mass), 0, "state");
        let mut rng = common::Lcg(seed);
        let refl: Vec<f32> = (0..nz * n).map(|_| (80.0 * rng.next() - 30.0) as f32).collect();
        let cf: Vec<f32> = (0..nz * n).map(|_| { let x = rng.next(); if x < 0.5 { 0.0 } else { x as f32 } }).collect();
        let aer: Vec<f32> = (0..n).map(|_| (0.5 * rng.next()) as f32).collect();
        let olr: Vec<f32> = (0..n).map(|_| (90.0 + 230.0 * rng.next()) as f32).collect();
        let olr_thin: Vec<f32> = (0..n).map(|i| if i % 7 == 0 { f32::NAN } else { (120.0 + 180.0 * rng.next()) as f32 }).collect();
        for (use_refl, use_cf, diag, interp, contrast, use_aer) in [
            (true, true, DiagnosedCloud::XuRandall1996, ReflInterp::LinearZ, group_d::VIS_CONTRAST, true),
            (false, false, DiagnosedCloud::XuRandall1996, ReflInterp::LinearZ, group_d::VIS_CONTRAST, true),
            (false, false, DiagnosedCloud::CondensatePresence, ReflInterp::LinearDbz, group_d::CONTRAST_2PCT, true),
            (true, false, DiagnosedCloud::XuRandall1996, ReflInterp::LinearDbz, group_d::CONTRAST_MOR_5PCT, true),
            // RULINGS 10 default: no aerosol carrier, clear-air baseline.
            (true, true, DiagnosedCloud::XuRandall1996, ReflInterp::LinearZ, group_d::VIS_CONTRAST, false),
        ] {
            let c = Carriers {
                refl: use_refl.then_some(refl.as_slice()),
                cldfra: use_cf.then_some(cf.as_slice()),
                aerosol_km: use_aer.then_some(aer.as_slice()),
                hgt: Some(&hgt),
                olr: use_cf.then_some(olr.as_slice()).or(Some(&olr_thin)),
            };
            let opt = Options { diagnosed: diag, refl_interp: interp, contrast, ..Options::default() };
            let cpu = group_d::compute_cpu(&st, &inp, &c, &opt).unwrap();
            let (gpu, _) = group_d::cuda::compute_device(&g, &dst, &dinp, &inp, &c, &opt).unwrap();
            let (gpu2, _) = group_d::cuda::compute_device(&g, &dst, &dinp, &inp, &c, &opt).unwrap();
            assert_eq!(differing(&cpu.planes, &gpu.planes), 0, "{nx}x{ny}x{nz} {use_refl} {use_cf} {diag:?} {interp:?}");
            assert_eq!(differing(&gpu.planes, &gpu2.planes), 0, "run to run");
            assert!(cpu.planes.iter().any(|v| v.is_finite()));
            assert!(cpu.plane("visibility").unwrap().iter().any(|v| v.is_finite()), "visibility on (RULINGS 10)");
        }
    }
}


#[test]
fn simulated_ir_olr_relation_hand_values() {
    use rw_post::group_d::column::{OGE_A, OGE_B, SIGMA_SB, olr_brightness_temperature};
    // The root satisfies T_f = T_b (a + b T_b) for the flux-equivalent T_f.
    for olr in [90.0f64, 150.0, 220.0, 280.0, 320.0] {
        let tb = olr_brightness_temperature(olr);
        let tf = (olr / SIGMA_SB).powf(0.25);
        assert!((tb * (OGE_A + OGE_B * tb) - tf).abs() < 1e-9, "{olr}: {tb}");
    }
    // Cold cloud: the window Tb sits on the flux temperature; a warm clear
    // scene is warmer than its flux temperature (the window sees lower).
    let t_cold = olr_brightness_temperature(100.0);
    assert!((t_cold - (100.0f64 / SIGMA_SB).powf(0.25)).abs() < 1.0, "{t_cold}");
    let t_warm = olr_brightness_temperature(280.0);
    assert!(t_warm > 290.0 && t_warm < 296.0, "{t_warm}");
    assert!(olr_brightness_temperature(0.0).is_nan());
    assert!(olr_brightness_temperature(-5.0).is_nan());
    assert!(olr_brightness_temperature(f64::NAN).is_nan());
    // Beyond the relation's range (T_f > a^2 / (-4 b)) there is no root.
    assert!(olr_brightness_temperature(2000.0).is_nan());
}

#[test]
fn simulated_ir_emission_level_or_olr() {
    use rw_post::group_d::column::{IR_K_ICE, olr_brightness_temperature};
    use rw_post::state::InterfaceState;
    let (mut inp, _, hgt) = frame(4, 3, 20, 31);
    let n = inp.shape.ncell();
    let nz = inp.shape.nz;
    // Columns 0..6 carry a thick ice layer at level 14, the rest are clear.
    set_slot(&mut inp, MassInput::QCloud, |_, _| 0.0);
    set_slot(&mut inp, MassInput::QSnow, |_, _| 0.0);
    set_slot(&mut inp, MassInput::QIce, |k, c| if k == 14 && c < 6 { 2e-3 } else { 0.0 });
    let st = build_cpu(&inp).unwrap();
    let olr: Vec<f32> = (0..n).map(|c| 150.0 + 10.0 * c as f32).collect();
    let c = Carriers { refl: None, cldfra: None, aerosol_km: None, hgt: Some(&hgt), olr: Some(&olr) };
    let o = group_d::compute_cpu(&st, &inp, &c, &Options::default()).unwrap();
    let ir = o.plane("simulated_ir").unwrap();
    let tk = st.mass_slot(MassState::Tk);
    let pint = st.interface_slot(InterfaceState::PInt);
    for cell in 0..n {
        let want = if cell < 6 {
            let dp = f64::from(pint[14 * n + cell]) - f64::from(pint[15 * n + cell]);
            let dtau = IR_K_ICE * 2e-3f32 as f64 * 1000.0 * dp / 9.81;
            assert!(dtau > 1.0, "the layer must be opaque: {dtau}");
            let (ta, tl) = (f64::from(tk[15 * n + cell]), f64::from(tk[14 * n + cell]));
            ta + (1.0 / dtau) * (tl - ta)
        } else {
            olr_brightness_temperature(f64::from(olr[cell]))
        };
        assert!((f64::from(ir[cell]) - want).abs() < 1e-3, "cell {cell}: {} vs {want}", ir[cell]);
    }
    assert_eq!(o.ir_source.map(|s| s.starts_with("emission level")), Some(true));
    let _ = nz;
    // Without QICE every column takes the OLR branch, and says so.
    inp.present &= !(1 << MassInput::QIce as u32);
    let st2 = build_cpu(&inp).unwrap();
    let o2 = group_d::compute_cpu(&st2, &inp, &c, &Options::default()).unwrap();
    let ir2 = o2.plane("simulated_ir").unwrap();
    for cell in 0..n {
        assert_eq!(ir2[cell], olr_brightness_temperature(f64::from(olr[cell])) as f32);
    }
    assert_eq!(o2.ir_source.map(|s| s.starts_with("OLR relation only")), Some(true));
    // Without OLR the field is omitted with a reason, never zero.
    let c3 = Carriers { olr: None, ..c };
    let o3 = group_d::compute_cpu(&st, &inp, &c3, &Options::default()).unwrap();
    assert!(o3.omitted.iter().any(|(id, why)| *id == "simulated_ir" && why.contains("OLR")));
    assert!(o3.plane("simulated_ir").unwrap().iter().all(|v| v.is_nan()));
    assert!(o3.ir_source.is_none());
}

#[cfg(any(windows, target_os = "linux"))]
#[test]
#[ignore = "needs an NVIDIA GPU"]
fn gpu_simulated_ir_matches_cpu_in_both_branches() {
    use rw_post::group_d::column::olr_brightness_temperature;
    use rw_post::gpu::GpuPost;
    let g = GpuPost::open(0).expect("an NVIDIA GPU with the post kernels");
    for (nx, ny, nz, seed) in [(41, 23, 50, 41u64), (9, 7, 3, 42)] {
        let (mut inp, _, hgt) = frame(nx, ny, nz, seed);
        let n = inp.shape.ncell();
        let mut rng = common::Lcg(seed);
        // Ice in every third column (random amounts, opaque or not), no
        // liquid or snow, so clear, thin and opaque columns all occur.
        let qi: Vec<f32> = (0..nz * n).map(|i| { let x = rng.next(); if (i % n) % 3 == 0 { (3e-3 * x) as f32 } else { 0.0 } }).collect();
        set_slot(&mut inp, MassInput::QIce, |k, c| qi[k * n + c]);
        set_slot(&mut inp, MassInput::QCloud, |_, _| 0.0);
        set_slot(&mut inp, MassInput::QSnow, |_, _| 0.0);
        let olr: Vec<f32> = (0..n).map(|i| if i % 11 == 0 { f32::NAN } else { (90.0 + 230.0 * rng.next()) as f32 }).collect();
        for drop_ice in [false, true] {
            if drop_ice {
                inp.present &= !(1 << MassInput::QIce as u32);
            }
            let st = build_cpu(&inp).unwrap();
            let dinp = g.upload_state_inputs(&inp).unwrap();
            let dst = g.run_state(&dinp).unwrap();
            let c = Carriers { refl: None, cldfra: None, aerosol_km: None, hgt: Some(&hgt), olr: Some(&olr) };
            let cpu = group_d::compute_cpu(&st, &inp, &c, &Options::default()).unwrap();
            let (gpu, _) = group_d::cuda::compute_device(&g, &dst, &dinp, &inp, &c, &Options::default()).unwrap();
            let (a, b) = (cpu.plane("simulated_ir").unwrap(), gpu.plane("simulated_ir").unwrap());
            assert_eq!(differing(a, b), 0, "{nx}x{ny}x{nz} drop_ice={drop_ice}");
            assert_eq!(differing(&cpu.planes, &gpu.planes), 0, "all planes {nx}x{ny}x{nz}");
            // Both branches ran: OLR-relation cells, and (with QICE) opaque
            // columns that are finite even where OLR is missing.
            let olr_cells = (0..n).filter(|&i| olr[i].is_finite() && a[i] == olr_brightness_temperature(f64::from(olr[i])) as f32).count();
            let opaque_nan_olr = (0..n).filter(|&i| olr[i].is_nan() && a[i].is_finite()).count();
            assert!(olr_cells > 0, "OLR branch never ran");
            if drop_ice {
                assert_eq!(olr_cells, (0..n).filter(|&i| olr[i].is_finite()).count());
                assert!((0..n).all(|i| olr[i].is_finite() || a[i].is_nan()), "a NaN OLR cell stays missing in the OLR branch");
            } else {
                assert!(opaque_nan_olr > 0, "emission branch never ran");
            }
        }
    }
}
