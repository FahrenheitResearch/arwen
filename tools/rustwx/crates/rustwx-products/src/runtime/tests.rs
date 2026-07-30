use super::*;
use rustwx_core::CycleSpec;

fn latest() -> LatestRun {
    LatestRun {
        model: rustwx_core::ModelId::Gfs,
        cycle: CycleSpec::new("20260415", 18).unwrap(),
        source: SourceId::Nomads,
    }
}

#[test]
fn build_single_pair_plan_emits_one_fetch_key_for_global_models() {
    let plan = build_single_pair_plan(&latest(), 12, None, None);
    assert_eq!(plan.bundles.len(), 2);
    assert_eq!(plan.fetch_keys().len(), 1);
    assert_eq!(plan.fetch_keys()[0].native_product, "pgrb2.0p25");
}

#[test]
fn nomads_fetch_concurrency_is_bounded_but_not_serial() {
    assert_eq!(fetch_concurrency_for_source(SourceId::Nomads, 0), 0);
    assert_eq!(fetch_concurrency_for_source(SourceId::Nomads, 1), 1);
    assert_eq!(fetch_concurrency_for_source(SourceId::Nomads, 3), 3);
    assert_eq!(fetch_concurrency_for_source(SourceId::Nomads, 6), 3);
}

#[test]
fn archive_fetch_concurrency_uses_all_fetch_keys() {
    assert_eq!(fetch_concurrency_for_source(SourceId::Aws, 6), 6);
}

#[test]
fn loaded_bundle_set_exposes_failure_accessors() {
    use rustwx_core::{CanonicalBundleId, ModelId};
    let plan = build_single_pair_plan(
        &LatestRun {
            model: ModelId::Hrrr,
            cycle: CycleSpec::new("20260415", 18).unwrap(),
            source: SourceId::Aws,
        },
        6,
        None,
        None,
    );
    let sfc_key = plan
        .fetch_keys()
        .into_iter()
        .find(|key| key.native_product == "sfc")
        .expect("hrrr plan has a sfc fetch key");
    let prs_key = plan
        .fetch_keys()
        .into_iter()
        .find(|key| key.native_product == "prs")
        .expect("hrrr plan has a prs fetch key");
    let surface_bundle_id = plan
        .bundles
        .iter()
        .find(|b| b.id.bundle == CanonicalBundleDescriptor::SurfaceAnalysis)
        .map(|b| b.id.clone())
        .expect("plan contains surface bundle");

    let mut fetch_failures: BTreeMap<BundleFetchKey, String> = BTreeMap::new();
    fetch_failures.insert(sfc_key.clone(), "simulated 404".to_string());

    let mut bundle_failures: BTreeMap<CanonicalBundleId, String> = BTreeMap::new();
    bundle_failures.insert(surface_bundle_id.clone(), "simulated 404".to_string());

    let loaded = LoadedBundleSet {
        latest: plan.latest(),
        forecast_hour: plan.forecast_hour,
        plan,
        fetched: BTreeMap::new(),
        fetch_failures,
        surface_decodes: BTreeMap::new(),
        pressure_decodes: BTreeMap::new(),
        bundle_failures,
        timing: LoadedBundleTiming::default(),
    };

    assert!(!loaded.all_fetches_succeeded());
    assert_eq!(loaded.fetch_failure(&sfc_key), Some("simulated 404"));
    assert_eq!(loaded.fetch_failure(&prs_key), None);
    assert_eq!(
        loaded.bundle_failure(&surface_bundle_id),
        Some("simulated 404")
    );
}

#[test]
fn build_single_pair_plan_emits_two_fetch_keys_for_hrrr() {
    let plan = build_single_pair_plan(
        &LatestRun {
            model: rustwx_core::ModelId::Hrrr,
            cycle: CycleSpec::new("20260415", 18).unwrap(),
            source: SourceId::Aws,
        },
        6,
        None,
        None,
    );
    assert_eq!(plan.bundles.len(), 2);
    assert_eq!(plan.fetch_keys().len(), 2);
    let products: Vec<_> = plan
        .fetch_keys()
        .iter()
        .map(|k| k.native_product.clone())
        .collect();
    assert!(products.contains(&"sfc".to_string()));
    assert!(products.contains(&"prs".to_string()));
}

#[test]
fn build_single_pair_plan_emits_two_fetch_keys_for_rrfs() {
    let plan = build_single_pair_plan(
        &LatestRun {
            model: rustwx_core::ModelId::RrfsA,
            cycle: CycleSpec::new("20260415", 18).unwrap(),
            source: SourceId::Aws,
        },
        6,
        None,
        None,
    );
    assert_eq!(plan.bundles.len(), 2);
    assert_eq!(plan.fetch_keys().len(), 2);
    let products: Vec<_> = plan
        .fetch_keys()
        .iter()
        .map(|k| k.native_product.clone())
        .collect();
    assert!(products.contains(&"nat-na".to_string()));
    assert!(products.contains(&"prs-na".to_string()));
}

#[test]
fn rrfs_pair_fetch_requests_use_subset_patterns() {
    let plan = build_single_pair_plan(
        &LatestRun {
            model: rustwx_core::ModelId::RrfsA,
            cycle: CycleSpec::new("20260415", 18).unwrap(),
            source: SourceId::Aws,
        },
        6,
        None,
        None,
    );
    let nat_key = plan
        .fetch_keys()
        .into_iter()
        .find(|key| key.native_product == "nat-na")
        .expect("rrfs plan includes nat-na");
    let prs_key = plan
        .fetch_keys()
        .into_iter()
        .find(|key| key.native_product == "prs-na")
        .expect("rrfs plan includes prs-na");
    let nat_request = build_fetch_request(&plan, &nat_key).expect("nat-na request builds");
    let prs_request = build_fetch_request(&plan, &prs_key).expect("prs-na request builds");
    assert!(
        nat_request
            .variable_patterns
            .contains(&"TMP:2 m above ground".to_string())
    );
    assert!(
        nat_request
            .variable_patterns
            .contains(&"UGRD:10 m above ground".to_string())
    );
    assert_eq!(
        prs_request.variable_patterns,
        vec![
            "HGT".to_string(),
            "GP".to_string(),
            "TMP".to_string(),
            "SPFH".to_string(),
            "DPT".to_string(),
            "RH".to_string(),
            "UGRD".to_string(),
            "VGRD".to_string(),
        ]
    );
}

#[test]
fn rrfs_shared_fetch_with_unsubsetted_native_consumer_uses_whole_file() {
    use rustwx_core::BundleRequirement;

    let latest = LatestRun {
        model: rustwx_core::ModelId::RrfsA,
        cycle: CycleSpec::new("20260415", 18).unwrap(),
        source: SourceId::Aws,
    };
    let mut builder = crate::planner::ExecutionPlanBuilder::new(&latest, 6);
    builder.require(&BundleRequirement::new(
        CanonicalBundleDescriptor::PressureAnalysis,
        6,
    ));
    builder.require(
        &BundleRequirement::new(CanonicalBundleDescriptor::NativeAnalysis, 6)
            .with_native_override("prs-na".to_string()),
    );
    let plan = builder.build();
    let prs_key = plan
        .fetch_keys()
        .into_iter()
        .find(|key| key.native_product == "prs-na")
        .expect("rrfs plan includes prs-na");
    let request = build_fetch_request(&plan, &prs_key).expect("request builds");
    assert!(
        request.variable_patterns.is_empty(),
        "shared fetch should keep whole-file bytes when any consumer lacks an explicit subset contract"
    );
}

#[test]
fn direct_native_requirement_patterns_reach_fetch_request() {
    use rustwx_core::BundleRequirement;

    let latest = LatestRun {
        model: rustwx_core::ModelId::Gfs,
        cycle: CycleSpec::new("20260415", 18).unwrap(),
        source: SourceId::Aws,
    };
    let mut builder = crate::planner::ExecutionPlanBuilder::new(&latest, 6);
    let requirement = BundleRequirement::new(CanonicalBundleDescriptor::NativeAnalysis, 6)
        .with_native_override("pgrb2.0p25");
    builder.require_with_logical_family_and_patterns(
        &requirement,
        Some("pgrb2.0p25"),
        ["TMP:500 mb", "UGRD:500 mb", "VGRD:500 mb"],
    );
    let plan = builder.build();
    let key = plan
        .fetch_keys()
        .into_iter()
        .find(|key| key.native_product == "pgrb2.0p25")
        .expect("direct plan includes pgrb2.0p25");
    let request = build_fetch_request(&plan, &key).expect("request builds");
    assert_eq!(
        request.variable_patterns,
        vec![
            "TMP:500 mb".to_string(),
            "UGRD:500 mb".to_string(),
            "VGRD:500 mb".to_string(),
        ]
    );
}

#[test]
fn hrrr_direct_native_sfc_patterns_do_not_expand_to_surface_mini_bundle() {
    use rustwx_core::BundleRequirement;

    let latest = LatestRun {
        model: rustwx_core::ModelId::Hrrr,
        cycle: CycleSpec::new("20260415", 18).unwrap(),
        source: SourceId::Aws,
    };
    let mut builder = crate::planner::ExecutionPlanBuilder::new(&latest, 0);
    let requirement = BundleRequirement::new(CanonicalBundleDescriptor::NativeAnalysis, 0)
        .with_native_override("sfc");
    builder.require_with_logical_family_and_patterns(
        &requirement,
        Some("sfc"),
        [
            "TMP:2 m above ground",
            "MSLMA:mean sea level",
            "UGRD:10 m above ground",
            "VGRD:10 m above ground",
        ],
    );
    let plan = builder.build();
    let key = plan
        .fetch_keys()
        .into_iter()
        .find(|key| key.native_product == "sfc")
        .expect("direct plan includes HRRR sfc");

    let request = build_fetch_request(&plan, &key).expect("request builds");

    assert_eq!(
        request.variable_patterns,
        vec![
            "MSLMA:mean sea level".to_string(),
            "TMP:2 m above ground".to_string(),
            "UGRD:10 m above ground".to_string(),
            "VGRD:10 m above ground".to_string(),
        ]
    );
    assert!(
        !request
            .variable_patterns
            .contains(&"APCP:surface".to_string()),
        "direct recipe subsets should not inherit the generic HRRR surface mini-bundle"
    );
}

#[test]
fn gfs_shared_direct_and_unsubsetted_derived_alias_uses_whole_file() {
    use rustwx_core::BundleRequirement;

    let latest = LatestRun {
        model: rustwx_core::ModelId::Gfs,
        cycle: CycleSpec::new("20260415", 18).unwrap(),
        source: SourceId::Aws,
    };
    let mut builder = crate::planner::ExecutionPlanBuilder::new(&latest, 0);
    let requirement = BundleRequirement::new(CanonicalBundleDescriptor::NativeAnalysis, 0)
        .with_native_override("pgrb2.0p25");
    builder.require_with_logical_family_and_patterns(
        &requirement,
        Some("direct"),
        ["TMP:2 m above ground"],
    );
    builder.require_with_logical_family(&requirement, Some("thermo-native:pgrb2.0p25"));
    let plan = builder.build();
    let key = plan
        .fetch_keys()
        .into_iter()
        .find(|key| key.native_product == "pgrb2.0p25")
        .expect("shared GFS plan includes pgrb2.0p25");

    let request = build_fetch_request(&plan, &key).expect("request builds");

    assert!(
        request.variable_patterns.is_empty(),
        "shared GFS direct+derived fetch should use whole-file bytes when a derived alias has no indexed subset contract"
    );
}
