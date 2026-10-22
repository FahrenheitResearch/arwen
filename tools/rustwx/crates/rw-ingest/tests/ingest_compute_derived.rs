//! Wiring proof for the ingest derived precompute path: `sbcape` and
//! `srh_0_3km` grids computed through
//! `ingest_compute::compute_derived_2d_from_inputs` must match calling the
//! underlying rustwx-calc kernels directly with identically prepared
//! inputs, bit-exactly — same code path, same height-AGL assembly, same
//! per-level pressure vector, same final f64 -> f32 cast. Input decode no
//! longer lives in the ingest module: the surface/pressure pair comes from
//! the render lanes' own products decoder
//! (`rustwx_products::gridded::decode_store_thermo_pair`), so this test
//! constructs the decoded pair directly and pins the compute fan-out.

use rustwx_calc::{
    EcapeVolumeInputs, GridShape as CalcGridShape, SurfaceInputs, VolumeShape, WindGridInputs,
    compute_sbcape_cin, compute_srh_03km_hemispheric,
};
use rustwx_products::gridded::mixing_ratio_from_dewpoint_k;

#[path = "common/compute_fixture.rs"]
mod fixture;
use fixture::{Synthetic, flatten_with, decoded_pair, assert_bits_eq};

use rw_ingest::ingest_compute::{self, ProductsComputeInputs};

const NX: usize = 3;
const NY: usize = 2;
const NXY: usize = NX * NY;
/// Descending pressure (ground up) — the order the decode lane aligns to.
const LEVELS: [u16; 5] = [1000, 925, 850, 700, 500];
const NZ: usize = LEVELS.len();

fn synthetic() -> Synthetic {
    let mut lat = Vec::with_capacity(NXY);
    let mut lon = Vec::with_capacity(NXY);
    for y in 0..NY {
        for x in 0..NX {
            lat.push(35.0 + 0.01 * y as f64);
            lon.push(-97.0 + 0.01 * x as f64);
        }
    }
    let psfc_pa: Vec<f64> = (0..NXY).map(|ij| 97_500.0 + 50.0 * ij as f64).collect();
    let t2_k: Vec<f64> = (0..NXY).map(|ij| 302.0 + 0.3 * ij as f64).collect();
    let td2_k: Vec<f64> = (0..NXY).map(|ij| 295.0 + 0.2 * ij as f64).collect();
    let q2_kgkg: Vec<f64> = psfc_pa
        .iter()
        .zip(td2_k.iter())
        .map(|(&psfc, &td_k)| mixing_ratio_from_dewpoint_k(psfc / 100.0, td_k))
        .collect();
    let u10_ms: Vec<f64> = (0..NXY).map(|ij| 2.0 + 0.5 * ij as f64).collect();
    let v10_ms: Vec<f64> = (0..NXY).map(|ij| 5.0 - 0.25 * ij as f64).collect();
    let orog_m: Vec<f64> = (0..NXY).map(|ij| 300.0 + 5.0 * ij as f64).collect();

    // Per-level planes keyed on (k, ij) so any reordering or misalignment
    // of levels/planes changes values and breaks the bit-compare.
    let plane = |base: f64, dk: f64, dij: f64, k: usize| -> Vec<f64> {
        (0..NXY)
            .map(|ij| base + dk * k as f64 + dij * ij as f64)
            .collect()
    };
    let by_level = |base: f64, dk: f64, dij: f64| -> Vec<(u16, Vec<f64>)> {
        LEVELS
            .iter()
            .enumerate()
            .map(|(k, &level)| (level, plane(base, dk, dij, k)))
            .collect()
    };
    Synthetic {
        lat,
        lon,
        psfc_pa,
        t2_k,
        q2_kgkg,
        u10_ms,
        v10_ms,
        orog_m,
        temperature_k: by_level(301.0, -7.5, 0.2),
        dewpoint_k: by_level(294.0, -8.5, 0.15),
        u_ms: by_level(3.0, 4.0, 0.3),
        v_ms: by_level(6.0, 2.5, -0.2),
        // Geopotential heights: well above the orography (300 m + a bit) and
        // strictly increasing by far more than the +1 m monotonic clamp, so
        // the AGL assembly is a pure (gh - orog) on this data.
        height_m: by_level(400.0, 1400.0, 2.0),
    }
}

#[test]
fn ingest_derived_matches_direct_calc_kernels_bit_exactly() {
    let synthetic = synthetic();

    // --- the ingest path: the decoded pair through the store compute lane
    //     (ProductsComputeInputs::new builds the shared height-AGL volume
    //     by the in-place gh transform this test re-derives serially) ---
    let (surface, pressure) = decoded_pair(&synthetic);
    let qvapor_kgkg_3d_input = pressure.qvapor_kgkg_3d.clone();
    let mut inputs = ProductsComputeInputs::new(surface, pressure);
    // keep_winds = false: the no-heavy shape, where the wind volumes leave
    // RAM mid-stage — outputs must still match the direct kernels exactly.
    let derived = ingest_compute::compute_derived_2d_from_inputs(&mut inputs, false)
        .expect("derived precompute must succeed on the synthetic hour");
    assert_eq!(
        derived.len(),
        29,
        "all 29 non-heavy recipes must realize; got: {:?}",
        derived.iter().map(|grid| grid.name).collect::<Vec<_>>()
    );
    let take = |name: &str| {
        derived
            .iter()
            .find(|grid| grid.name == name)
            .unwrap_or_else(|| panic!("derived output missing '{name}'"))
    };

    // --- the direct path: identical prep, then the calc kernels the
    //     derived lane dispatches to ---
    let temperature_c_3d = flatten_with(&synthetic.temperature_k, |v| v - 273.15);
    let qvapor_kgkg_3d = qvapor_kgkg_3d_input;
    let u_3d = flatten_with(&synthetic.u_ms, |v| v);
    let v_3d = flatten_with(&synthetic.v_ms, |v| v);
    let gh_3d = flatten_with(&synthetic.height_m, |v| v);

    // Height AGL exactly as the derived lane assembles it: (gh - orog)
    // clamped at 0, then forced monotonic by >= 1 m per level upward.
    let mut height_agl_3d: Vec<f64> = gh_3d
        .iter()
        .enumerate()
        .map(|(idx, &value)| (value - synthetic.orog_m[idx % NXY]).max(0.0))
        .collect();
    for k in 1..NZ {
        for ij in 0..NXY {
            let min_height = height_agl_3d[(k - 1) * NXY + ij] + 1.0;
            if height_agl_3d[k * NXY + ij] < min_height {
                height_agl_3d[k * NXY + ij] = min_height;
            }
        }
    }

    // The compute lane has no pressure_3d_pa here, so the volume pressure is
    // the per-level vector (hPa * 100).
    let pressure_pa: Vec<f64> = LEVELS
        .iter()
        .map(|&level| f64::from(level) * 100.0)
        .collect();

    let calc_grid = CalcGridShape::new(NX, NY).unwrap();
    let surface_inputs = SurfaceInputs {
        psfc_pa: &synthetic.psfc_pa,
        t2_k: &synthetic.t2_k,
        q2_kgkg: &synthetic.q2_kgkg,
        u10_ms: &synthetic.u10_ms,
        v10_ms: &synthetic.v10_ms,
    };
    let volume_inputs = EcapeVolumeInputs {
        pressure_pa: &pressure_pa,
        temperature_c: &temperature_c_3d,
        qvapor_kgkg: &qvapor_kgkg_3d,
        height_agl_m: &height_agl_3d,
        u_ms: &u_3d,
        v_ms: &v_3d,
        nz: NZ,
    };
    let sb = compute_sbcape_cin(calc_grid, volume_inputs, surface_inputs, None)
        .expect("direct sbcape kernel");
    let srh_03km = compute_srh_03km_hemispheric(
        WindGridInputs {
            shape: VolumeShape::new(calc_grid, NZ).unwrap(),
            u_3d_ms: &u_3d,
            v_3d_ms: &v_3d,
            height_agl_3d_m: &height_agl_3d,
        },
        &synthetic.lat,
    )
    .expect("direct srh kernel");

    let expected_sbcape: Vec<f32> = sb.cape_jkg.iter().map(|&value| value as f32).collect();
    let expected_srh: Vec<f32> = srh_03km.iter().map(|&value| value as f32).collect();

    let sbcape = take("sbcape");
    assert_eq!(sbcape.units, "J/kg", "sbcape units");
    assert_bits_eq(&sbcape.values, &expected_sbcape, "sbcape");
    assert!(
        sbcape.values.iter().any(|value| value.is_finite()),
        "synthetic hour must produce at least one finite sbcape value, \
         or the comparison is NaN-trivial"
    );

    let srh = take("srh_0_3km");
    assert_eq!(srh.units, "m^2/s^2", "srh_0_3km units");
    assert_bits_eq(&srh.values, &expected_srh, "srh_0_3km");
    assert!(
        srh.values.iter().any(|value| value.is_finite()),
        "synthetic hour must produce at least one finite srh_0_3km value"
    );
}
