//! Shared decoded-hour fixture plumbing for the numeric wiring contracts.

use super::{LEVELS, NX, NY, NZ, NXY};
use rustwx_products::gridded::{PressureFields, SurfaceFields, mixing_ratio_from_dewpoint_k};

pub struct Synthetic {
    pub lat: Vec<f64>,
    pub lon: Vec<f64>,
    pub psfc_pa: Vec<f64>,
    pub t2_k: Vec<f64>,
    pub q2_kgkg: Vec<f64>,
    pub u10_ms: Vec<f64>,
    pub v10_ms: Vec<f64>,
    pub orog_m: Vec<f64>,
    pub temperature_k: Vec<(u16, Vec<f64>)>,
    pub dewpoint_k: Vec<(u16, Vec<f64>)>,
    pub u_ms: Vec<(u16, Vec<f64>)>,
    pub v_ms: Vec<(u16, Vec<f64>)>,
    pub height_m: Vec<(u16, Vec<f64>)>,
}

pub fn flatten_with(planes: &[(u16, Vec<f64>)], convert: impl Fn(f64) -> f64) -> Vec<f64> {
    let mut out = Vec::with_capacity(NZ * NXY);
    for &level in &LEVELS {
        let (_, plane) = planes
            .iter()
            .find(|(have, _)| *have == level)
            .expect("synthetic volumes carry every level");
        out.extend(plane.iter().map(|&value| convert(value)));
    }
    out
}

pub fn decoded_pair(synthetic: &Synthetic) -> (SurfaceFields, PressureFields) {
    let qvapor_kgkg_3d: Vec<f64> = LEVELS
        .iter()
        .flat_map(|&level| {
            let (_, plane) = synthetic
                .dewpoint_k
                .iter()
                .find(|(have, _)| *have == level)
                .unwrap();
            plane
                .iter()
                .map(move |&td_k| mixing_ratio_from_dewpoint_k(f64::from(level), td_k))
        })
        .collect();
    (
        SurfaceFields {
            lat: synthetic.lat.clone(),
            lon: synthetic.lon.clone(),
            nx: NX,
            ny: NY,
            projection: None,
            psfc_pa: synthetic.psfc_pa.clone(),
            orog_m: synthetic.orog_m.clone(),
            orog_is_proxy: false,
            t2_k: synthetic.t2_k.clone(),
            q2_kgkg: synthetic.q2_kgkg.clone(),
            u10_ms: synthetic.u10_ms.clone(),
            v10_ms: synthetic.v10_ms.clone(),
            native_sbcape_jkg: None,
            native_mlcape_jkg: None,
            native_mucape_jkg: None,
            native_pblh_m: None,
        },
        PressureFields {
            pressure_levels_hpa: LEVELS.iter().map(|&level| f64::from(level)).collect(),
            pressure_3d_pa: None,
            temperature_c_3d: flatten_with(&synthetic.temperature_k, |v| v - 273.15),
            qvapor_kgkg_3d,
            u_ms_3d: flatten_with(&synthetic.u_ms, |v| v),
            v_ms_3d: flatten_with(&synthetic.v_ms, |v| v),
            gh_m_3d: flatten_with(&synthetic.height_m, |v| v),
            omega_pa_s_3d: None,
            absolute_vorticity_s_3d: None,
            cloud_liquid_kgkg_3d: None,
            cloud_ice_kgkg_3d: None,
            rain_kgkg_3d: None,
            snow_kgkg_3d: None,
            graupel_kgkg_3d: None,
        },
    )
}

pub fn assert_bits_eq(actual: &[f32], expected: &[f32], context: &str) {
    assert_eq!(actual.len(), expected.len(), "{context}: length mismatch");
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert_eq!(
            a.to_bits(),
            e.to_bits(),
            "{context}: value mismatch at index {i} (actual {a}, expected {e})"
        );
    }
}
