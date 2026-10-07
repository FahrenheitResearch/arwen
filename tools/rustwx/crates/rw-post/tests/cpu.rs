//! CPU path against the binary64 reference and the staged surface fixture.

mod common;

use rw_post::reference as r64;
use rw_post::solar::{SolarFrame, UtcTime};
use rw_post::state::{InterfaceState, MassInput, MassState, build_cpu};
use rw_post::surface::{SurfaceField, SurfaceInput, SurfaceInputs, SurfaceOptions, surface_cell, surface_cpu};
use rw_post::thermo;

#[test]
fn thermo_helpers_match_binary64() {
    let mut rng = common::Lcg(7);
    let (mut es, mut ei, mut td, mut rh, mut tv, mut te) = (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for _ in 0..200_000 {
        let t = 190.0 + 140.0 * rng.next();
        let p = 2000.0 + 104000.0 * rng.next();
        let rr = 0.035 * rng.next() * rng.next();
        let (t32, p32, r32) = (t as f32, p as f32, rr as f32);
        let (t, p, rr) = (t32 as f64, p32 as f64, r32 as f64);
        es = es.max(((thermo::es_water(t32) as f64) / r64::es_water(t) - 1.0).abs());
        ei = ei.max(((thermo::es_ice(t32) as f64) / r64::es_ice(t) - 1.0).abs());
        let q32 = r32 / (1.0 + r32);
        let e32 = thermo::vapour_pressure(q32, p32);
        let e = r64::vapour_pressure(rr / (1.0 + rr), p);
        if rr > 1e-6 {
            td = td.max((thermo::dewpoint(e32) as f64 - r64::dewpoint(e)).abs());
            te = te.max(((thermo::theta_e(t32, p32, r32) as f64) / r64::theta_e(t, p, rr) - 1.0).abs());
        }
        rh = rh.max((thermo::rh(e32, t32) as f64 - r64::rh(e, t)).abs());
        tv = tv.max((thermo::virtual_temperature(t32, r32) as f64 - r64::virtual_temperature(t, rr)).abs());
    }
    eprintln!("max rel es_w {es:e} es_i {ei:e} theta_e {te:e}; max abs Td {td:e} K, RH {rh:e} %, Tv {tv:e} K");
    assert!(es < 3e-6 && ei < 3e-6, "{es} {ei}");
    assert!(td < 2e-3, "{td}");
    assert!(rh < 2e-3, "{rh}");
    assert!(tv < 1e-4, "{tv}");
    assert!(te < 2e-5, "{te}");
}

#[test]
fn magnus_values_from_the_paper() {
    // AERK at 0 C is its own constant, and es(Td(e)) = e.
    assert_eq!(thermo::es_water(273.15), 610.94);
    for e in [10.0f32, 100.0, 611.0, 2000.0, 7000.0] {
        let back = thermo::es_water(thermo::dewpoint(e));
        assert!((back / e - 1.0).abs() < 3e-6, "{e} {back}");
    }
    assert!(thermo::dewpoint(0.0).is_nan());
    assert!(thermo::rh(f32::NAN, 280.0).is_nan());
    assert_eq!(thermo::rh(5000.0, 273.15), 100.0);
}

#[test]
fn state_matches_binary64_column() {
    let inp = common::synthetic_state(9, 7, 30, 11);
    let st = build_cpu(&inp).unwrap();
    let s = inp.shape;
    let n = s.ncell();
    let (mut dpi, mut dp, mut dtk, mut dtv, mut dz) = (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for c in 0..n {
        let col = |slot: MassInput| -> Vec<f64> { (0..s.nz).map(|k| inp.slot(slot)[k * n + c] as f64).collect() };
        let cond: Vec<Vec<f64>> = [MassInput::QCloud, MassInput::QRain, MassInput::QIce, MassInput::QSnow, MassInput::QGraupel]
            .into_iter()
            .map(col)
            .collect();
        let ph: Vec<f64> = (0..=s.nz).map(|k| inp.ph[k * n + c] as f64).collect();
        let phb: Vec<f64> = (0..=s.nz).map(|k| inp.phb[k * n + c] as f64).collect();
        let c3f: Vec<f64> = inp.c3f.iter().map(|&x| x as f64).collect();
        let c4f: Vec<f64> = inp.c4f.iter().map(|&x| x as f64).collect();
        let r = r64::column(&r64::ColumnIn {
            t_pert: &col(MassInput::T),
            p_pert: &col(MassInput::P),
            p_base: &col(MassInput::PB),
            qv: &col(MassInput::QVapor),
            condensate: &cond,
            p_hyd: None,
            ph: &ph,
            phb: &phb,
            mu_d: inp.mu[c] as f64 + inp.mub[c] as f64,
            c3f: &c3f,
            c4f: &c4f,
            p_top: inp.p_top as f64,
        });
        for k in 0..=s.nz {
            dpi = dpi.max((st.interface_slot(InterfaceState::PInt)[k * n + c] as f64 - r.p_int[k]).abs());
        }
        for k in 0..s.nz {
            dp = dp.max((st.at(MassState::P, k, c) as f64 - r.p[k]).abs());
            dtk = dtk.max((st.at(MassState::Tk, k, c) as f64 - r.tk[k]).abs());
            dtv = dtv.max((st.at(MassState::Tv, k, c) as f64 - r.tv[k]).abs());
            dz = dz.max((st.at(MassState::ZMass, k, c) as f64 - r.z_mass[k]).abs());
        }
        // ground interface: the moist column weighs more than the dry one
        let pdry = inp.c3f[0] as f64 * (inp.mu[c] as f64 + inp.mub[c] as f64) + inp.c4f[0] as f64 + inp.p_top as f64;
        assert!(r.p_int[0] > pdry);
    }
    eprintln!("max abs: p_int {dpi} Pa, p {dp} Pa, Tk {dtk} K, Tv {dtv} K, z {dz} m");
    assert!(dpi < 0.1 && dp < 0.1, "{dpi} {dp}");
    assert!(dtk < 1e-3 && dtv < 1e-3, "{dtk} {dtv}");
    // f32 heights near 20 km have a 2 mm unit in the last place.
    assert!(dz < 5e-3, "{dz}");
    // Destaggering is the exact two-face mean.
    let (nx, ny) = (s.nx, s.ny);
    for k in 0..s.nz {
        for j in 0..ny {
            for i in 0..nx {
                let c = j * nx + i;
                let u = 0.5 * (inp.u_stag[(k * ny + j) * (nx + 1) + i] + inp.u_stag[(k * ny + j) * (nx + 1) + i + 1]);
                let v = 0.5 * (inp.v_stag[(k * (ny + 1) + j) * nx + i] + inp.v_stag[(k * (ny + 1) + j + 1) * nx + i]);
                assert_eq!(st.at(MassState::U, k, c), u);
                assert_eq!(st.at(MassState::V, k, c), v);
            }
        }
    }
}

#[test]
fn p_hyd_is_used_when_present() {
    let mut inp = common::synthetic_state(4, 3, 12, 5);
    let v = inp.shape.mass_volume();
    let at = MassInput::PHyd as usize * v;
    for w in 0..v {
        inp.mass[at + w] = 12345.0 + w as f32;
    }
    inp.present |= 1 << MassInput::PHyd as u32;
    let st = build_cpu(&inp).unwrap();
    assert_eq!(st.mass_slot(MassState::P)[7], 12352.0);
}

#[test]
fn shelter_fields_match_binary64() {
    let inp = common::synthetic_state(13, 11, 20, 3);
    let st = build_cpu(&inp).unwrap();
    let mut s = common::synthetic_surface(&inp, 4);
    let n = inp.shape.ncell();
    s.set(SurfaceInput::TvLowest, &st.mass_slot(MassState::Tv)[..n]);
    let solar = SolarFrame::at(&UtcTime::parse_wrf("2026-10-03_22:00:00").unwrap());
    let opt = SurfaceOptions::new(&solar);
    // Two passes: with T2, and with only TH2 (fallback through p2).
    for with_t2 in [true, false] {
        let mut s2 = s.clone();
        if !with_t2 {
            s2.present &= !(1 << SurfaceInput::T2 as u32);
        }
        let out = surface_cpu(&s2, &opt);
        let plane = |f: SurfaceField| &out[f as usize * n..(f as usize + 1) * n];
        let mut worst = [0.0f64; 4];
        for c in 0..n {
            let (p2, t2, _q2, td2, rh2) = r64::shelter(
                s2.plane(SurfaceInput::Psfc)[c] as f64,
                s2.plane(SurfaceInput::TvLowest)[c] as f64,
                with_t2.then(|| s2.plane(SurfaceInput::T2)[c] as f64),
                s2.plane(SurfaceInput::Th2)[c] as f64,
                s2.plane(SurfaceInput::Q2)[c] as f64,
            );
            let ours = [plane(SurfaceField::ShelterPressure)[c], plane(SurfaceField::T2)[c], plane(SurfaceField::Td2)[c], plane(SurfaceField::Rh2)[c]];
            for (w, (o, r)) in ours.iter().zip([p2, t2, td2, rh2]).enumerate() {
                if r.is_nan() {
                    assert!(o.is_nan(), "cell {c} field {w}: {o} vs NaN");
                } else {
                    worst[w] = worst[w].max((*o as f64 - r).abs());
                }
            }
            assert!(plane(SurfaceField::Td2)[c].is_nan() || plane(SurfaceField::Td2)[c] <= plane(SurfaceField::T2)[c]);
        }
        eprintln!("with_t2={with_t2}: max abs p2 {} Pa, t2 {} K, td2 {} K, rh2 {} %", worst[0], worst[1], worst[2], worst[3]);
        assert!(worst[0] < 0.02 && worst[1] < 1e-4 && worst[2] < 2e-3 && worst[3] < 2e-3, "{worst:?}");
    }
}

#[test]
fn cosz_matches_binary64_algorithm() {
    let mut rng = common::Lcg(99);
    for _ in 0..200 {
        let t = UtcTime {
            year: 2000 + (rng.next() * 40.0) as i32,
            month: 1 + (rng.next() * 12.0) as u32,
            day: 1 + (rng.next() * 28.0) as u32,
            hour: (rng.next() * 24.0) as u32,
            minute: (rng.next() * 60.0) as u32,
            second: 0,
        };
        let s = SolarFrame::at(&t);
        for _ in 0..50 {
            let lat = (180.0 * rng.next() - 90.0) as f32;
            let lon = (360.0 * rng.next() - 180.0) as f32;
            let ours = rw_post::solar::cosz(lat, lon, s.sin_dec, s.cos_dec, s.ha0_deg) as f64;
            let r = r64::cosz(t.julian_date(), t.ut_hours(), lat as f64, lon as f64);
            assert!((ours - r).abs() < 2e-5, "{t:?} {lat} {lon}: {ours} vs {r}");
        }
    }
}

/// The staged surface fixture (an old-module output on given inputs): graded
/// within GRIB packing precision except where the specification rules a
/// different method; those cells are listed and checked against the rule.
#[test]
fn surface_fixture() {
    let inp = common::data("surface-inputs.f32");
    let out = common::data("surface-outputs.f32");
    let n = 10;
    let slots = [
        SurfaceInput::Th2,
        SurfaceInput::Q2,
        SurfaceInput::Tsk,
        SurfaceInput::Snow,
        SurfaceInput::SnowH,
        SurfaceInput::Hfx,
        SurfaceInput::Qfx,
        SurfaceInput::GrdFlx,
        SurfaceInput::Ust,
        SurfaceInput::Glw,
        SurfaceInput::SwDown,
        SurfaceInput::VegFra,
        SurfaceInput::IvgTyp,
        SurfaceInput::SeaIce,
        SurfaceInput::Znt,
        SurfaceInput::ShdMin,
        SurfaceInput::ShdMax,
        SurfaceInput::Lai,
        SurfaceInput::Psfc,
        SurfaceInput::Cosz,
    ];
    let mut s = SurfaceInputs::new(n);
    for (f, slot) in slots.iter().enumerate() {
        s.set(*slot, &inp[f * n..(f + 1) * n]);
    }
    let mut opt = SurfaceOptions::new(&SolarFrame::at(&UtcTime::parse_wrf("2026-01-01_00:00:00").unwrap()));
    opt.latent_from_qfx = 1; // the fixture's QFX path is the RUC case
    opt.snow_cover_derived = 1;
    // 19 graded outputs in the specification's 4.2 order (from
    // potential_temperature2 on).
    let fields = [
        SurfaceField::PotentialTemperature2,
        SurfaceField::SpecificHumidity2,
        SurfaceField::SkinTemperature,
        SurfaceField::SnowWater,
        SurfaceField::SnowDepth,
        SurfaceField::SnowCover,
        SurfaceField::SensibleHeatFlux,
        SurfaceField::LatentHeatFlux,
        SurfaceField::GroundHeatFlux,
        SurfaceField::FrictionVelocity,
        SurfaceField::DownwardLongwave,
        SurfaceField::DownwardShortwave,
        SurfaceField::VegetationFraction,
        SurfaceField::VegetationType,
        SurfaceField::SeaIceFraction,
        SurfaceField::RoughnessLength,
        SurfaceField::VegetationMin,
        SurfaceField::VegetationMax,
        SurfaceField::LeafAreaIndex,
    ];
    let mut divergences = Vec::new();
    for c in 0..n {
        let ours = surface_cell(&s, &opt, c);
        for (w, f) in fields.iter().enumerate() {
            let want = out[w * n + c];
            let got = ours[*f as usize];
            // Packing precision of the exporter for these products.
            let tol = 1e-4f32.max(want.abs() * 2e-7);
            let agree = (got - want).abs() <= tol || (got.is_nan() && want.is_nan());
            if !agree {
                divergences.push((*f, c, inp_value(&inp, n, c), got, want));
            }
        }
    }
    for d in &divergences {
        eprintln!("divergence {:?} cell {}: ours {} old {}", d.0, d.1, d.3, d.4);
    }
    // Every divergence must be one the specification rules on purpose.
    for (f, c, _, got, want) in &divergences {
        let ruled = match f {
            // D28: Lv = 2.501e6 (the old output used 2.5e6): 0.04 percent, ratio 1.0004.
            SurfaceField::LatentHeatFlux => ((got / want) - 1.0004).abs() < 1e-5 || (*want == 0.0 && *got == 0.0),
            // Lower bound 0 only (no upper cap): SNOW 6000 and SNOWH 60 stay.
            SurfaceField::SnowWater | SurfaceField::SnowDepth => *c == 7,
            // D27 night guard at cosz <= 0 (the old output zeroed at a
            // small positive threshold too).
            SurfaceField::DownwardShortwave => *c == 2,
            // Q2 = -1: Q2 floored at 0 gives 0 (the old output wrote 99999).
            SurfaceField::SpecificHumidity2 => *c == 5 && *got == 0.0,
            // D26: snow cover from the Noah depletion curve; reference only.
            SurfaceField::SnowCover => true,
            _ => false,
        };
        assert!(ruled, "unruled divergence {f:?} cell {c}: {got} vs {want}");
    }
}

fn inp_value(inp: &[f32], n: usize, c: usize) -> f32 {
    inp[c % n]
}

#[test]
fn wind_rotation_is_orthonormal() {
    let mut rng = common::Lcg(17);
    for _ in 0..1000 {
        let a = (2.0 * std::f64::consts::PI * rng.next()) as f32;
        let (sa, ca) = (a.sin(), a.cos());
        let (u, v) = ((40.0 * rng.next() - 20.0) as f32, (40.0 * rng.next() - 20.0) as f32);
        let (ue, ve) = rw_post::wind::rotate(u, v, sa, ca);
        let s0 = (u as f64).hypot(v as f64);
        let s1 = (ue as f64).hypot(ve as f64);
        assert!((s0 - s1).abs() < 1e-4 * s0.max(1.0));
        // back-rotation by -a
        let (ub, vb) = rw_post::wind::rotate(ue, ve, -sa, ca);
        assert!((ub - u).abs() < 2e-5 * s0.max(1.0) as f32 && (vb - v).abs() < 2e-5 * s0.max(1.0) as f32);
    }
}

/// `Times(Time, DateStrLen)` reads as one label per record on both
/// containers.  Breakage prevented: netCDF-4 WRF histories decode NC_CHAR
/// one character per element, and the exporter refused every such file
/// with "holds 19 times" (b1-height lane, 2026-10-05).
#[test]
fn times_read_one_label_per_record_on_classic_and_netcdf4() {
    let want = ["2024-05-25_19:00:00", "2024-05-25_20:00:00"];
    for name in ["times-classic.nc", "times-netcdf4.nc"] {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data").join(name);
        let file = netcrust::File::open(&path).expect("fixture opens");
        let got = rw_post::history::read_times(&file).expect("Times reads");
        assert_eq!(got, want, "{name}");
    }
}
