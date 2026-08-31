//! What a paired-run score promises its callers before it reads a byte.
//!
//! A campaign scoreboard indexes its rows by metric key and refuses a run
//! whose frame ladder is not the registered one.  Both of those are contract,
//! not implementation: a scorer that spelled a key differently would produce
//! right numbers under names its own evidence does not carry, and one that
//! scored whatever frames it happened to find would silently score a
//! different experiment.  So both are pinned here, where they can be checked
//! without a history stream.

use std::fs;
use std::path::{Path, PathBuf};

use rw_fieldcmp::runscore::{score, DomainSpec, MetricKeys, RunScoreRequest};

/// A request over two directories under `root`, with the registered
/// half-hour, five-minute ladder.
fn request(root: &Path, domains: &[(&str, f64)]) -> RunScoreRequest {
    RunScoreRequest {
        left: root.join("left"),
        right: root.join("right"),
        frame_prefix: "wrfout".to_string(),
        start: "1974-04-03T12:00:00".to_string(),
        run_seconds: 1800,
        cadence_seconds: 300,
        domains: domains
            .iter()
            .map(|(label, dx_m)| DomainSpec {
                label: label.to_string(),
                dx_m: *dx_m,
                score_neighborhood: false,
            })
            .collect(),
        fields: vec!["T".to_string()],
        low_pass_width_m: 6000.0,
        interior_exclusion_cells: 5,
        boundary_width_cells: 5,
        composite_variable: "REFL_10CM".to_string(),
        threshold: 40.0,
        neighborhood_radius_m: 5000.0,
        object_connectivity: 8,
        object_min_area_km2: 25.0,
        keys: MetricKeys::default(),
    }
}

/// Lay out empty files with history-frame names.  The ladder is checked
/// before anything is opened, which is what lets a staging mistake be caught
/// without decoding thirty gigabytes first.
fn lay_out(root: &Path, domain: &str, minutes: &[i64]) -> PathBuf {
    for arm in ["left", "right"] {
        let directory = root.join(arm);
        fs::create_dir_all(&directory).expect("arm directory");
        for minute in minutes {
            let name = format!("wrfout_{domain}_1974-04-03_12_{minute:02}_00");
            fs::write(directory.join(name), b"").expect("frame");
        }
    }
    root.to_path_buf()
}

/// A scratch root private to THIS process.
///
/// The process id is not decoration.  Without it two concurrent runs of
/// this suite on one box share `%TEMP%\rw-runscore-<name>`, and the
/// `remove_dir_all` below deletes the fixtures the other run is part way
/// through laying out; the loser panics with
/// `Os { code: 3, kind: NotFound }` at the `fs::write` in `lay_out`,
/// which reads like a bug in the code under test and is not.  It is not
/// hypothetical: it took out every shard of a parallel mutation run
/// (tools/battery/run_mutation_gate.py runs the suite in N copies at
/// once) before a single mutant had been tested.  Every other suite in
/// this workspace that writes under `temp_dir()` already scopes its root
/// this way.
fn scratch(name: &str) -> PathBuf {
    let root = std::env::temp_dir().join(format!("rw-runscore-{}-{name}", std::process::id()));
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).expect("scratch root");
    root
}

#[test]
fn a_short_ladder_is_refused_by_name() {
    let root = scratch("short-ladder");
    lay_out(&root, "d01", &[0, 5, 10, 15, 20, 25]);
    let error = score(&request(&root, &[("d01", 12000.0)]))
        .expect_err("a six-frame ladder is not the registered seven");
    let message = error.to_string();
    assert!(message.contains("frame times"), "{message}");
    assert!(message.contains("1500"), "the found ladder is named: {message}");
    assert!(message.contains("1800"), "the wanted ladder is named: {message}");
}

#[test]
fn an_off_cadence_frame_is_refused_rather_than_snapped_to_the_ladder() {
    let root = scratch("off-cadence");
    lay_out(&root, "d01", &[0, 5, 10, 15, 20, 25, 31]);
    let error = score(&request(&root, &[("d01", 12000.0)])).expect_err("31 minutes is not 30");
    assert!(error.to_string().contains("frame times"), "{error}");
}

#[test]
fn a_duration_the_cadence_does_not_divide_is_refused() {
    let root = scratch("ragged-cadence");
    lay_out(&root, "d01", &[0, 5, 10, 15, 20, 25, 30]);
    let mut plan = request(&root, &[("d01", 12000.0)]);
    plan.cadence_seconds = 700;
    let error = score(&plan).expect_err("1800 is not a whole number of 700s intervals");
    assert!(error.to_string().contains("whole number"), "{error}");
}

#[test]
fn a_request_with_nothing_to_score_says_so() {
    let root = scratch("empty-request");
    lay_out(&root, "d01", &[0, 5, 10, 15, 20, 25, 30]);
    let mut plan = request(&root, &[]);
    assert!(score(&plan).is_err());
    plan = request(&root, &[("d01", 12000.0)]);
    plan.fields.clear();
    assert!(score(&plan).is_err());
}

/// The scorer is told which domain it is looking at, so a directory holding
/// several domains' frames does not fail the ladder check on the ones it was
/// not asked about.
#[test]
fn other_domains_in_the_same_directory_are_ignored() {
    let root = scratch("mixed-domains");
    lay_out(&root, "d01", &[0, 5, 10, 15, 20, 25, 30]);
    lay_out(&root, "d02", &[0, 5]);
    let error = score(&request(&root, &[("d01", 12000.0)]))
        .expect_err("the empty frames cannot be decoded");
    // Past the ladder check, so the complaint is about reading, not staging.
    let message = error.to_string();
    assert!(!message.contains("frame times"), "{message}");
    assert!(message.contains("cannot"), "{message}");
}

/// Two domains under one label would write the same keys twice and the
/// scoreboard would show the second one's numbers under both names.
#[test]
fn a_repeated_domain_label_is_refused() {
    let root = scratch("repeated-label");
    lay_out(&root, "d01", &[0, 5, 10, 15, 20, 25, 30]);
    let error = score(&request(&root, &[("d01", 12000.0), ("d01", 3000.0)]))
        .expect_err("one label cannot name two domains");
    assert!(error.to_string().contains("named twice"), "{error}");
}

/// The key spellings a scoreboard indexes on.  Three colon-separated fields,
/// category first, domain second; the category strings are the caller's.
#[test]
fn metric_keys_are_category_domain_subject() {
    let keys = MetricKeys::default();
    for category in [&keys.state, &keys.boundary, &keys.object, &keys.neighborhood] {
        assert!(!category.contains(':'), "{category} would split its own key");
        assert!(!category.is_empty());
    }
    assert_eq!(keys.state, "low_pass_state_rmse");
    assert_eq!(keys.boundary, "applied_boundary_increment_error");
    assert_eq!(keys.object, "storm_object_timing_difference");
    assert_eq!(keys.object_subject, "first_object");
    // The neighbourhood defaults name no domain, because which domain carries
    // that row is a caller's pin rather than the metric's property.
    assert!(!keys.neighborhood.contains("d0"), "{}", keys.neighborhood);
}
