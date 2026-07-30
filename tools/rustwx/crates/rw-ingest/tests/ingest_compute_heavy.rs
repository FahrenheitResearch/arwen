//! Wiring proof for the ingest heavy (ECAPE-class) precompute path:
//! `sbecape` computed through `ingest_compute::compute_heavy_2d_from_inputs`
//! must match calling the rustwx-calc ECAPE triplet kernel directly with
//! identically prepared inputs, bit-exactly — same code path, same
//! height-AGL assembly, same per-level pressure vector, same slug fan-out
//! and native-CAPE plumbing, same final f64 -> f32 cast. Input decode no
//! longer lives in the ingest module: the surface/pressure pair comes from
//! the render lanes' own products decoder
//! (`rustwx_products::gridded::decode_store_thermo_pair`, which also
//! carries the native CAPE planes), so this test constructs the decoded
//! pair directly and pins the compute fan-out.

use rustwx_calc::{
    EcapeTripletOptions, EcapeVolumeInputs, GridShape as CalcGridShape, SurfaceInputs,
    compute_ecape_triplet_with_failure_mask_from_parts,
};
use rustwx_products::derived::store_heavy_recipe_slugs;
use rustwx_products::gridded::{PressureFields, SurfaceFields, mixing_ratio_from_dewpoint_k};

// The bin-shared module carries both compute stages; this test exercises
// only the heavy lane, so the non-heavy entry is intentionally unused here.
use rw_ingest::ingest_compute::{self, ProductsComputeInputs};

const NX: usize = 3;
const NY: usize = 2;
const NXY: usize = NX * NY;
/// Descending pressure (ground up) — the order the decode lane aligns to.
/// Deeper than the derived test's profile: ECAPE's entraining ascent and
/// storm motion need a full troposphere to produce finite values.
const LEVELS: [u16; 9] = [1000, 925, 850, 700, 500, 400, 300, 250, 200];
const NZ: usize = LEVELS.len();

/// A warm, moist, strongly sheared synthetic hour with a conditionally
/// unstable troposphere: enough instability and shear that sbecape is
/// finite and nonzero somewhere, so the comparison is not all-NaN-trivial.
struct Synthetic {
    lat: Vec<f64>,
    lon: Vec<f64>,
    psfc_pa: Vec<f64>,
    t2_k: Vec<f64>,
    q2_kgkg: Vec<f64>,
    u10_ms: Vec<f64>,
    v10_ms: Vec<f64>,
    orog_m: Vec<f64>,
    temperature_k: Vec<(u16, Vec<f64>)>,
    dewpoint_k: Vec<(u16, Vec<f64>)>,
    u_ms: Vec<(u16, Vec<f64>)>,
    v_ms: Vec<(u16, Vec<f64>)>,
    height_m: Vec<(u16, Vec<f64>)>,
}

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
    let td2_k: Vec<f64> = (0..NXY).map(|ij| 296.0 + 0.2 * ij as f64).collect();
    let q2_kgkg: Vec<f64> = psfc_pa
        .iter()
        .zip(td2_k.iter())
        .map(|(&psfc, &td_k)| mixing_ratio_from_dewpoint_k(psfc / 100.0, td_k))
        .collect();
    let u10_ms: Vec<f64> = (0..NXY).map(|ij| 3.0 + 0.5 * ij as f64).collect();
    let v10_ms: Vec<f64> = (0..NXY).map(|ij| 6.0 - 0.25 * ij as f64).collect();
    let orog_m: Vec<f64> = (0..NXY).map(|ij| 100.0 + 5.0 * ij as f64).collect();

    // Per-level base values keyed on (k, ij) so any reordering or
    // misalignment of levels/planes changes values and breaks the
    // bit-compare. The profile is a loaded, sheared springtime sounding.
    let plane = |bases: [f64; NZ], dij: f64| -> Vec<(u16, Vec<f64>)> {
        LEVELS
            .iter()
            .zip(bases.iter())
            .map(|(&level, &base)| {
                (
                    level,
                    (0..NXY)
                        .map(|ij| base + dij * ij as f64)
                        .collect::<Vec<f64>>(),
                )
            })
            .collect()
    };
    let temperature_k = plane(
        [
            300.0, 295.0, 290.0, 279.0, 258.0, 246.0, 230.0, 221.0, 213.0,
        ],
        0.2,
    );
    let dewpoint_k = plane(
        [
            295.0, 292.0, 287.0, 271.0, 244.0, 230.0, 214.0, 205.0, 197.0,
        ],
        0.15,
    );
    let u_ms = plane([4.0, 9.0, 13.0, 18.0, 24.0, 28.0, 33.0, 36.0, 38.0], 0.3);
    let v_ms = plane([8.0, 11.0, 9.0, 6.0, 4.0, 3.0, 2.0, 1.0, 0.0], -0.2);
    // Geopotential heights: above the orography and strictly increasing by
    // far more than the +1 m monotonic clamp, so the AGL assembly is a pure
    // (gh - orog) on this data.
    let height_m = plane(
        [
            120.0, 780.0, 1500.0, 3100.0, 5750.0, 7300.0, 9400.0, 10700.0, 12200.0,
        ],
        2.0,
    );

    Synthetic {
        lat,
        lon,
        psfc_pa,
        t2_k,
        q2_kgkg,
        u10_ms,
        v10_ms,
        orog_m,
        temperature_k,
        dewpoint_k,
        u_ms,
        v_ms,
        height_m,
    }
}

/// Flatten `(level, plane)` pairs into `[k][ij]` in LEVELS order with a
/// per-value conversion — the layout the decode lane hands over.
fn flatten_with(planes: &[(u16, Vec<f64>)], convert: impl Fn(f64) -> f64) -> Vec<f64> {
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

fn native_planes() -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    (
        (0..NXY).map(|ij| 600.0 + 40.0 * ij as f64).collect(),
        (0..NXY).map(|ij| 450.0 + 30.0 * ij as f64).collect(),
        (0..NXY).map(|ij| 700.0 + 25.0 * ij as f64).collect(),
    )
}

/// The decoded pair exactly as `decode_store_thermo_pair` shapes it, with
/// optional native CAPE planes riding on the surface fields.
fn decoded_pair(synthetic: &Synthetic, with_native_cape: bool) -> (SurfaceFields, PressureFields) {
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
    let (native_sb, native_ml, native_mu) = if with_native_cape {
        let (sb, ml, mu) = native_planes();
        (Some(sb), Some(ml), Some(mu))
    } else {
        (None, None, None)
    };
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
            native_sbcape_jkg: native_sb,
            native_mlcape_jkg: native_ml,
            native_mucape_jkg: native_mu,
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

fn assert_bits_eq(actual: &[f32], expected: &[f32], context: &str) {
    assert_eq!(actual.len(), expected.len(), "{context}: length mismatch");
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert_eq!(
            a.to_bits(),
            e.to_bits(),
            "{context}: value mismatch at index {i} (actual {a}, expected {e})"
        );
    }
}

#[test]
fn ingest_heavy_sbecape_matches_direct_ecape_triplet_bit_exactly() {
    let synthetic = synthetic();

    // --- the ingest path: the decoded pair (with native CAPE) through the
    //     heavy stage exactly as rw_ingest runs it (by value; the shared
    //     height-AGL volume comes from the in-place gh transform this test
    //     re-derives serially below) ---
    let (surface, pressure) = decoded_pair(&synthetic, true);
    let qvapor_kgkg_3d_input = pressure.qvapor_kgkg_3d.clone();
    let inputs = ProductsComputeInputs::new(surface, pressure);
    let heavy = ingest_compute::compute_heavy_2d_from_inputs(inputs)
        .expect("heavy precompute must succeed on the synthetic hour");

    let expected_slugs = store_heavy_recipe_slugs();
    assert_eq!(
        heavy.grids.iter().map(|grid| grid.name).collect::<Vec<_>>(),
        expected_slugs,
        "all heavy recipes must realize, in inventory order, when native CAPE is present"
    );
    assert!(
        heavy.skipped.is_empty(),
        "nothing may skip with native CAPE present: {:?}",
        heavy.skipped
    );

    // --- the direct path: identical prep, then the rustwx-calc ECAPE
    //     triplet kernel the heavy lane dispatches to ---
    let temperature_c_3d = flatten_with(&synthetic.temperature_k, |v| v - 273.15);
    let qvapor_kgkg_3d = qvapor_kgkg_3d_input;
    let u_3d = flatten_with(&synthetic.u_ms, |v| v);
    let v_3d = flatten_with(&synthetic.v_ms, |v| v);
    let gh_3d = flatten_with(&synthetic.height_m, |v| v);

    // Height AGL exactly as the heavy lane assembles it: (gh - orog)
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

    // prepare_heavy_volume is called with include_pressure_3d = false, so
    // the volume pressure is the per-level vector (hPa * 100).
    let pressure_pa: Vec<f64> = LEVELS
        .iter()
        .map(|&level| f64::from(level) * 100.0)
        .collect();

    let calc_grid = CalcGridShape::new(NX, NY).unwrap();
    let triplet = compute_ecape_triplet_with_failure_mask_from_parts(
        calc_grid,
        EcapeVolumeInputs {
            pressure_pa: &pressure_pa,
            temperature_c: &temperature_c_3d,
            qvapor_kgkg: &qvapor_kgkg_3d,
            height_agl_m: &height_agl_3d,
            u_ms: &u_3d,
            v_ms: &v_3d,
            nz: NZ,
        },
        SurfaceInputs {
            psfc_pa: &synthetic.psfc_pa,
            t2_k: &synthetic.t2_k,
            q2_kgkg: &synthetic.q2_kgkg,
            u10_ms: &synthetic.u10_ms,
            v10_ms: &synthetic.v10_ms,
        },
        // The heavy lane pins right-moving storm motion (see
        // compute_ecape_map_fields_with_prepared_volume).
        EcapeTripletOptions::new("right_moving"),
    )
    .expect("direct ECAPE triplet kernel");

    let take = |name: &str| {
        heavy
            .grids
            .iter()
            .find(|grid| grid.name == name)
            .unwrap_or_else(|| panic!("heavy output missing '{name}'"))
    };

    let expected_sbecape: Vec<f32> = triplet
        .sb
        .fields
        .ecape_jkg
        .iter()
        .map(|&value| value as f32)
        .collect();
    let sbecape = take("sbecape");
    assert_eq!(sbecape.units, "J/kg", "sbecape units");
    assert_bits_eq(&sbecape.values, &expected_sbecape, "sbecape");
    assert!(
        sbecape.values.iter().any(|value| value.is_finite()),
        "synthetic hour must produce at least one finite sbecape value, \
         or the comparison is NaN-trivial"
    );

    let expected_sbecin: Vec<f32> = triplet
        .sb
        .fields
        .cin_jkg
        .iter()
        .map(|&value| value as f32)
        .collect();
    let sbecin = take("sbecin");
    assert_eq!(sbecin.units, "J/kg", "sbecin units");
    assert_bits_eq(&sbecin.values, &expected_sbecin, "sbecin");

    let expected_sbncape: Vec<f32> = triplet
        .sb
        .fields
        .ncape_jkg
        .iter()
        .map(|&value| value as f32)
        .collect();
    let sbncape = take("sbncape");
    assert_eq!(sbncape.units, "J/kg", "sbncape units");
    assert_bits_eq(&sbncape.values, &expected_sbncape, "sbncape");

    assert_eq!(
        heavy.ecape_failure_count,
        triplet.total_failure_count(),
        "failure count must ride through unchanged"
    );

    // Native ratio plumbing: stored ratio grid == stored ecape / native
    // plane wherever both are finite (the lane NaNs cells whose native
    // CAPE denominator is below its 100 J/kg floor).
    let (native_sb, _, _) = native_planes();
    let ratio = take("sb_ecape_native_cape_ratio");
    assert_eq!(ratio.units, "ratio", "sb native ratio units");
    let mut finite_ratio_checked = 0usize;
    #[allow(clippy::needless_range_loop)] // ij indexes four parallel arrays
    for ij in 0..NXY {
        let ecape = f64::from(sbecape.values[ij]);
        if ratio.values[ij].is_finite() && ecape.is_finite() {
            let expected = (triplet.sb.fields.ecape_jkg[ij] / native_sb[ij]) as f32;
            assert_eq!(
                ratio.values[ij].to_bits(),
                expected.to_bits(),
                "sb native ratio at index {ij}"
            );
            finite_ratio_checked += 1;
        }
    }
    assert!(
        finite_ratio_checked > 0,
        "at least one finite native-ratio cell must exist, or the ratio check is trivial"
    );
}

#[test]
fn ingest_heavy_without_native_cape_skips_only_the_native_ratios() {
    let synthetic = synthetic();
    let (surface, pressure) = decoded_pair(&synthetic, false);
    let inputs = ProductsComputeInputs::new(surface, pressure);
    let heavy = ingest_compute::compute_heavy_2d_from_inputs(inputs)
        .expect("heavy precompute must succeed without native CAPE");

    let expected_realized: Vec<&str> = store_heavy_recipe_slugs()
        .into_iter()
        .filter(|slug| !slug.ends_with("_ecape_native_cape_ratio"))
        .collect();
    assert_eq!(
        heavy.grids.iter().map(|grid| grid.name).collect::<Vec<_>>(),
        expected_realized,
        "every non-native-ratio heavy recipe must realize"
    );
    let skipped_slugs: Vec<&str> = heavy.skipped.iter().map(|(slug, _)| *slug).collect();
    assert_eq!(
        skipped_slugs,
        vec![
            "sb_ecape_native_cape_ratio",
            "ml_ecape_native_cape_ratio",
            "mu_ecape_native_cape_ratio",
        ],
        "exactly the three native-ratio recipes must skip"
    );
    for (slug, reason) in &heavy.skipped {
        assert!(
            reason.contains("native") && reason.contains("CAPE"),
            "skip reason for '{slug}' must document the missing native CAPE plane, got: {reason}"
        );
    }
}
