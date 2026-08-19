//! Bitwise parity against the REAL scipy/numpy remap.
//!
//! Every golden under `golden/cases/` is the output of
//! `gpuwm.verify.obs.regrid` -- the shipped `scipy.spatial.cKDTree` plan
//! build and the shipped `numpy.add.at` apply -- run on real staged
//! observation and model bytes by `golden/gen_regrid_goldens.py`.  The
//! comparison here is on IEEE bit patterns, not a tolerance: a float
//! that differs in the last bit fails.
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
    source_shape: (usize, usize),
    destination_shape: (usize, usize),
    unreachable_destination_cells: usize,
}

fn hex_float(text: &str) -> f64 {
    let digits = text.trim().trim_start_matches("0x");
    f64::from_bits(u64::from_str_radix(digits, 16).expect("hex float"))
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
        assert_eq!(
            plan.max_used_distance_m.to_bits(),
            spec.max_used_distance_m.to_bits(),
            "{}: max_used_distance_m differs in its bits ({} vs {})",
            spec.name,
            plan.max_used_distance_m,
            spec.max_used_distance_m
        );

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
    // honest if its perturbed twin -- the same grid with one ULP of
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
