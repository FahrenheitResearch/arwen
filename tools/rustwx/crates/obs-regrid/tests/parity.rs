//! Bitwise parity against the REAL scipy/numpy remap.
//!
//! Every golden under `golden/cases/` is the output of
//! `gpuwm.verify.obs.regrid` -- the shipped `scipy.spatial.cKDTree` plan
//! build and the shipped `numpy.add.at` apply -- run on real staged
//! observation and model bytes by `golden/gen_regrid_goldens.py`.  The
//! remapped values, indices and masks are compared on IEEE bit patterns:
//! a remapped float that differs in the last bit fails.
//!
//! One field is bitwise only where it can be: `max_used_distance_m` is
//! built from the platform's `sin`, `cos` and `asin`, whose last bits can
//! differ between libraries.  The goldens were written on Windows
//! (UCRT); glibc 2.43 disagrees on 4,361 of the 56,100 unit-vector
//! components of the real cases' grids, by up to 2 ULP, and the chord of
//! two vectors 1.35e-3 apart magnifies that to 227 ULP of the metre
//! distance (8579.868677808428 against 8579.868677808841, 4.1e-10 m),
//! which failed the public CI's Linux job on 2.7.6, 2.7.7 and 2.8.0.  So
//! the manifest records a hash of the reference's unit vectors: where this
//! platform's vectors hash the same the distance is compared to the bit,
//! and where they differ it is held to `trig_disagreement_bound_m`.
//! Every other field stays bitwise on every platform.
//!
//! One case is exempt from index parity and says so out loud:
//! `synthetic_tie_degenerate`, where two source cells sit at the same
//! point and scipy's answer is traversal order rather than a rule.  Its
//! perturbed control, `synthetic_tie_control`, is one ULP of longitude
//! away, has a geometric answer, and is NOT exempt.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use obs_regrid::{Method, apply_plan, build_plan};

// --------------------------------------------------------------------------
// golden IO (the lane-2 "GWARR1" container, same spelling)
// --------------------------------------------------------------------------

enum Array {
    F64(Vec<f64>),
    I64(Vec<i64>),
    U8(Vec<u8>),
}

fn cases_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("golden")
        .join("cases")
}

fn read_arr(path: &Path) -> (Vec<usize>, Array) {
    let bytes = std::fs::read(path)
        .unwrap_or_else(|err| panic!("golden {} unreadable: {err}", path.display()));
    assert!(
        bytes.len() >= 10 && &bytes[..8] == b"GWARR1\x00\x00",
        "golden {} has a bad header",
        path.display()
    );
    let code = bytes[8];
    let ndim = bytes[9] as usize;
    let mut dims = Vec::with_capacity(ndim);
    let mut offset = 10;
    for _ in 0..ndim {
        dims.push(u64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap()) as usize);
        offset += 8;
    }
    let count: usize = dims.iter().product();
    let payload = &bytes[offset..];
    let data = match code {
        0 => Array::F64(
            payload
                .chunks_exact(8)
                .take(count)
                .map(|chunk| f64::from_le_bytes(chunk.try_into().unwrap()))
                .collect(),
        ),
        2 => Array::I64(
            payload
                .chunks_exact(8)
                .take(count)
                .map(|chunk| i64::from_le_bytes(chunk.try_into().unwrap()))
                .collect(),
        ),
        3 => Array::U8(payload[..count].to_vec()),
        other => panic!("golden {} has dtype code {other}", path.display()),
    };
    (dims, data)
}

fn f64s(directory: &Path, name: &str) -> Vec<f64> {
    match read_arr(&directory.join(format!("{name}.bin"))).1 {
        Array::F64(values) => values,
        _ => panic!("{name} is not float64"),
    }
}

fn i64s(directory: &Path, name: &str) -> Vec<i64> {
    match read_arr(&directory.join(format!("{name}.bin"))).1 {
        Array::I64(values) => values,
        _ => panic!("{name} is not int64"),
    }
}

fn bools(directory: &Path, name: &str) -> Vec<bool> {
    match read_arr(&directory.join(format!("{name}.bin"))).1 {
        Array::U8(values) => values.into_iter().map(|byte| byte != 0).collect(),
        _ => panic!("{name} is not a mask"),
    }
}

// --------------------------------------------------------------------------
// a hand-rolled reader for the manifest, so the crate keeps zero deps
// --------------------------------------------------------------------------

/// The manifest fields this test consumes, pulled out with a scanner
/// rather than a JSON crate: adding serde_json to a dependency-free
/// crate for six scalars per case would be a supply-chain entry bought
/// with test convenience.
struct CaseSpec {
    name: String,
    method: Method,
    max_distance_m: f64,
    max_used_distance_m: f64,
    unit_vectors_fnv1a64: Option<u64>,
    source_shape: (usize, usize),
    destination_shape: (usize, usize),
    unreachable_destination_cells: usize,
}

fn hex_float(text: &str) -> f64 {
    let digits = text.trim().trim_start_matches("0x");
    f64::from_bits(u64::from_str_radix(digits, 16).expect("hex float"))
}

fn hex_u64(text: &str) -> u64 {
    u64::from_str_radix(text.trim().trim_start_matches("0x"), 16).expect("hex u64")
}

fn field<'a>(block: &'a str, key: &str) -> &'a str {
    let needle = format!("\"{key}\":");
    let start = block
        .find(&needle)
        .unwrap_or_else(|| panic!("manifest block has no {key}"))
        + needle.len();
    let rest = &block[start..];
    let end = rest
        .find(|c| c == ',' || c == '\n')
        .unwrap_or(rest.len());
    rest[..end].trim().trim_matches('"')
}

fn shape(block: &str, key: &str) -> (usize, usize) {
    let needle = format!("\"{key}\": [");
    let start = block.find(&needle).expect("shape") + needle.len();
    let rest = &block[start..];
    let end = rest.find(']').expect("shape close");
    let parts: Vec<usize> = rest[..end]
        .split(',')
        .map(|piece| piece.trim().parse().expect("shape number"))
        .collect();
    (parts[0], parts[1])
}

fn read_manifest() -> Vec<CaseSpec> {
    let text = std::fs::read_to_string(cases_dir().join("MANIFEST.json"))
        .expect("golden/cases/MANIFEST.json is missing; regenerate with golden/gen_regrid_goldens.py");
    let mut specs = Vec::new();
    // Every case block starts at its own `"name":` key; the manifest is
    // machine-written with one case per object, so splitting on that key
    // is unambiguous.
    for block in text.split("\"name\": \"").skip(1) {
        let name = block[..block.find('"').expect("name close")].to_string();
        let method = match field(block, "method") {
            "nearest" => Method::Nearest,
            "cell_average" => Method::CellAverage,
            other => panic!("unknown method {other} in {name}"),
        };
        specs.push(CaseSpec {
            name,
            method,
            max_distance_m: hex_float(field(block, "max_distance_m")),
            max_used_distance_m: hex_float(field(block, "max_used_distance_m")),
            unit_vectors_fnv1a64: block
                .contains("\"unit_vectors_fnv1a64\":")
                .then(|| hex_u64(field(block, "unit_vectors_fnv1a64"))),
            source_shape: shape(block, "source_shape"),
            destination_shape: shape(block, "destination_shape"),
            unreachable_destination_cells: field(block, "unreachable_destination_cells")
                .parse()
                .expect("unreachable count"),
        });
    }
    assert!(!specs.is_empty(), "the manifest declares no cases");
    specs
}

// --------------------------------------------------------------------------
// the parity run
// --------------------------------------------------------------------------

/// FNV-1a 64 over this platform's unit vectors, source then destination,
/// each component's little-endian bytes: the same hash
/// `golden/gen_regrid_goldens.py` records from the reference's vectors.
fn unit_vectors_fnv1a64(source: &[[f64; 3]], destination: &[[f64; 3]]) -> u64 {
    let mut value: u64 = 0xcbf2_9ce4_8422_2325;
    for point in source.iter().chain(destination.iter()) {
        for component in point {
            for byte in component.to_le_bytes() {
                value = (value ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3);
            }
        }
    }
    value
}

/// The diagnostic's precision contract when platform unit vectors differ.
///
/// This bound permits 2 ULP of scalar sin/cos disagreement, the largest
/// measured disagreement between glibc 2.43 and the UCRT on these vectors.
/// glibc's x86_64 accuracy table reports measured scalar sin/cos errors
/// of 1 ULP; UCRT documents results usually within 1 ULP of the correctly
/// rounded value, with possible larger errors.  Neither is a universal
/// accuracy guarantee:
/// this gate deliberately rejects disagreement beyond the propagated bound.
/// See sourceware.org/glibc/manual/2.39/html_node/Errors-in-Math-Functions.html
/// and learn.microsoft.com/cpp/c-runtime-library/floating-point-support.
/// A unit-vector component is `cos * cos`, `cos * sin`
/// or `sin`, at most 1 in magnitude, so it can disagree by 4 ULP of the
/// trig values plus one rounding of the product: 5 * 2^-53.  Each axis of
/// the difference of two vectors takes two such components and the chord
/// is the length of three axes, so it moves by at most
/// sqrt(3) * 10 * 2^-53; `2 R asin(chord / 2)` scales that by
/// `R / sqrt(1 - chord^2 / 4)`.  `asin` itself (2 ULP of a value near
/// chord / 2) and the final multiplications add a few ULP of the distance.
fn trig_disagreement_bound_m(distance_m: f64) -> f64 {
    let unit = f64::EPSILON / 2.0; // 2^-53
    let half_chord = (distance_m / obs_regrid::EARTH_RADIUS_M / 2.0).sin();
    let slope = obs_regrid::EARTH_RADIUS_M / (1.0 - half_chord * half_chord).sqrt();
    let ulp = f64::from_bits(distance_m.abs().to_bits() + 1) - distance_m.abs();
    slope * 3f64.sqrt() * 10.0 * unit + 8.0 * ulp
}

/// The one case whose SOURCE INDEX is exempt, and the reason.
const TIE_EXEMPT: &str = "synthetic_tie_degenerate";

#[test]
fn every_golden_case_matches_the_real_python_bit_for_bit() {
    let specs = read_manifest();
    let mut seen: BTreeMap<String, bool> = BTreeMap::new();

    for spec in &specs {
        let directory = cases_dir().join(&spec.name);
        let source_latitude = f64s(&directory, "source_latitude");
        let source_longitude = f64s(&directory, "source_longitude");
        let destination_latitude = f64s(&directory, "destination_latitude");
        let destination_longitude = f64s(&directory, "destination_longitude");
        let values = f64s(&directory, "values");
        let valid = bools(&directory, "valid");
        let expected_index = i64s(&directory, "source_index");
        let expected_reachable = bools(&directory, "reachable");
        let expected_values = f64s(&directory, "out_values");
        let expected_valid = bools(&directory, "out_valid");

        let plan = build_plan(
            spec.method,
            &source_latitude,
            &source_longitude,
            spec.source_shape,
            &destination_latitude,
            &destination_longitude,
            spec.destination_shape,
            spec.max_distance_m,
        )
        .unwrap_or_else(|err| panic!("{}: build_plan refused: {err}", spec.name));

        // --- the plan ---
        assert_eq!(
            plan.reachable, expected_reachable,
            "{}: the reachability mask differs from scipy's",
            spec.name
        );
        assert_eq!(
            obs_regrid::unreachable_destination_cells(&plan.reachable),
            spec.unreachable_destination_cells,
            "{}: the receipt's unreachable count differs",
            spec.name
        );
        let same_trig = spec.unit_vectors_fnv1a64.is_some_and(|recorded| {
            let source = obs_regrid::unit_vectors(&source_latitude, &source_longitude).unwrap();
            let destination =
                obs_regrid::unit_vectors(&destination_latitude, &destination_longitude).unwrap();
            unit_vectors_fnv1a64(&source, &destination) == recorded
        });
        if same_trig {
            assert_eq!(
                plan.max_used_distance_m.to_bits(),
                spec.max_used_distance_m.to_bits(),
                "{}: max_used_distance_m differs in its bits ({} vs {}) although this \
                 platform's unit vectors are the reference's to the bit",
                spec.name,
                plan.max_used_distance_m,
                spec.max_used_distance_m
            );
        } else {
            let apart = (plan.max_used_distance_m - spec.max_used_distance_m).abs();
            let bound = trig_disagreement_bound_m(spec.max_used_distance_m);
            assert!(
                apart <= bound,
                "{}: max_used_distance_m {} is {apart:e} m from the reference's {}, more than \
                 the {bound:e} m two C libraries' sin and cos can explain",
                spec.name,
                plan.max_used_distance_m,
                spec.max_used_distance_m
            );
            println!(
                "{}: this platform's sin/cos differ from the reference's; \
                 max_used_distance_m {apart:e} m apart, bound {bound:e} m",
                spec.name
            );
        }

        if spec.name == TIE_EXEMPT {
            // The DOCUMENTED divergence, asserted rather than skipped:
            // obs-regrid answers by its own rule, and that rule is
            // lowest flat index.
            assert_eq!(
                plan.source_index[0], 0,
                "{}: the tie rule is lowest flat index wins",
                spec.name
            );
            seen.insert(spec.name.clone(), plan.source_index == expected_index);
            continue;
        }
        assert_eq!(
            plan.source_index, expected_index,
            "{}: the integer mapping differs from scipy's",
            spec.name
        );

        // --- the apply ---
        let destination_cells = spec.destination_shape.0 * spec.destination_shape.1;
        let mut out_values = vec![f64::NAN; destination_cells];
        let mut out_valid = vec![false; destination_cells];
        apply_plan(
            spec.method,
            &plan.source_index,
            &plan.reachable,
            spec.source_shape,
            spec.destination_shape,
            &values,
            &valid,
            &mut out_values,
            &mut out_valid,
        )
        .unwrap_or_else(|err| panic!("{}: apply_plan refused: {err}", spec.name));

        assert_eq!(
            out_valid, expected_valid,
            "{}: the remapped validity differs from numpy's",
            spec.name
        );
        for (slot, (got, want)) in out_values.iter().zip(expected_values.iter()).enumerate() {
            assert_eq!(
                got.to_bits(),
                want.to_bits(),
                "{}: destination cell {slot} differs in its bits ({got} vs {want})",
                spec.name
            );
        }
        seen.insert(spec.name.clone(), true);
    }

    // The set the manifest promises must be the set that ran.
    for required in [
        "real_nearest_obs_to_model",
        "real_cell_average_obs_to_model",
        "real_nearest_model_to_obs",
        "real_nearest_tight_bound",
        "synthetic_bound_edge_below",
        "synthetic_bound_edge_above",
        "synthetic_tie_degenerate",
        "synthetic_tie_control",
    ] {
        assert!(
            seen.contains_key(required),
            "the golden set is missing {required}; regenerate it"
        );
    }
}

#[test]
fn every_case_records_the_reference_unit_vectors_and_the_bound_stays_small() {
    // Without the hash the distance is never compared to the bit, so a
    // regeneration that dropped it would quietly move every platform to the
    // bounded comparison.
    for spec in read_manifest() {
        assert!(
            spec.unit_vectors_fnv1a64.is_some(),
            "{}: the manifest records no unit_vectors_fnv1a64; regenerate with \
             golden/gen_regrid_goldens.py",
            spec.name
        );
    }
    // The bound is nanometres: a remap that picked a different neighbour
    // moves the distance by a grid spacing, metres to kilometres.
    let bound = trig_disagreement_bound_m(8579.868677808428);
    assert!(bound > 4.2e-10 && bound < 2.0e-8, "bound {bound:e} m");
}

#[test]
fn the_tight_bound_case_actually_leaves_most_of_the_domain_unreachable() {
    // A bound case where everything is reachable proves nothing about
    // the bound.  This asserts the golden set still exercises the branch
    // it was built for, so a regeneration against different bytes cannot
    // quietly turn it into a second copy of the loose-bound case.
    let spec = read_manifest()
        .into_iter()
        .find(|spec| spec.name == "real_nearest_tight_bound")
        .expect("the tight-bound case");
    let cells = spec.destination_shape.0 * spec.destination_shape.1;
    assert!(
        spec.unreachable_destination_cells * 2 > cells,
        "the tight-bound golden leaves only {} of {cells} destination \
         cells unreachable, which no longer exercises the bound",
        spec.unreachable_destination_cells
    );
}

#[test]
fn the_bound_edge_pair_straddles_the_flip() {
    // Same reasoning one layer down: the two synthetic bound-edge cases
    // are one ULP of metres apart and MUST disagree, or the strict
    // squared predicate is untested.
    let specs = read_manifest();
    let below = specs
        .iter()
        .find(|spec| spec.name == "synthetic_bound_edge_below")
        .expect("below");
    let above = specs
        .iter()
        .find(|spec| spec.name == "synthetic_bound_edge_above")
        .expect("above");
    assert_eq!(below.unreachable_destination_cells, 1, "below must reject");
    assert_eq!(above.unreachable_destination_cells, 0, "above must accept");
    assert!(
        below.max_distance_m < above.max_distance_m,
        "the pair is not ordered"
    );
}

#[test]
fn the_documented_divergence_has_a_perturbed_control_that_is_not_exempt() {
    // "Never bit-exact to a bug": the tie case is exempt from index
    // parity because scipy has no rule there.  That exemption is only
    // accurate if its perturbed twin -- the same grid with one ULP of
    // longitude, where the answer IS a fact -- is held to full parity.
    let specs = read_manifest();
    assert!(
        specs.iter().any(|spec| spec.name == "synthetic_tie_control"),
        "the tie exemption has no perturbed control"
    );
    assert_eq!(
        specs
            .iter()
            .filter(|spec| spec.name == TIE_EXEMPT)
            .count(),
        1,
        "exactly one case may be exempt from index parity"
    );
}
