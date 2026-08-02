use super::*;

#[test]
fn direct_catalog_keeps_supported_and_blocked_matrix() {
    let catalog = build_supported_products_catalog();
    let entry = catalog
        .direct
        .iter()
        .find(|entry| entry.slug == "composite_reflectivity_uh")
        .expect("direct catalog should include native reflectivity + UH");
    assert_eq!(
        entry.id,
        ProductId::new(ProductKind::Direct, "composite_reflectivity_uh")
    );
    assert_eq!(entry.status, ProductCatalogStatus::Partial);
    assert!(
        entry.support.iter().any(|target| {
            target.model == Some(ModelId::Hrrr)
                && matches!(target.status, ProductTargetStatus::Supported)
        }),
        "HRRR should support composite_reflectivity_uh"
    );
    assert!(
        entry.support.iter().any(|target| {
            target.model == Some(ModelId::Gfs)
                && matches!(target.status, ProductTargetStatus::Blocked)
        }),
        "GFS should still report blockers for composite_reflectivity_uh"
    );
    assert_eq!(entry.maturity, ProductMaturity::Operational);
    assert!(entry.flags.is_empty());
    let provenance = entry
        .product_metadata
        .as_ref()
        .and_then(|metadata| metadata.provenance.as_ref())
        .expect("direct entry should expose typed product provenance");
    assert_eq!(provenance.lineage, rustwx_core::ProductLineage::Direct);
}

#[test]
fn direct_catalog_marks_hrrr_layout_composites_as_panel_products() {
    let catalog = build_supported_products_catalog();
    let cloud_levels = catalog
        .direct
        .iter()
        .find(|entry| entry.slug == "cloud_cover_levels")
        .expect("cloud_cover_levels should stay in the direct lane");
    assert_eq!(
        cloud_levels.render_style.as_deref(),
        Some("weather_panel_grid")
    );
    assert!(
        cloud_levels
            .notes
            .iter()
            .any(|note| note.contains("composite panel"))
    );
    assert!(cloud_levels.support.iter().any(|target| {
        target.model == Some(ModelId::Hrrr)
            && matches!(target.status, ProductTargetStatus::Supported)
    }));

    let precipitation_type = catalog
        .direct
        .iter()
        .find(|entry| entry.slug == "precipitation_type")
        .expect("precipitation_type should stay in the direct lane");
    assert_eq!(
        precipitation_type.render_style.as_deref(),
        Some("weather_panel_grid")
    );
    assert!(
        precipitation_type
            .notes
            .iter()
            .any(|note| note.contains("freezing-rain"))
    );
    assert_eq!(precipitation_type.maturity, ProductMaturity::Operational);
    assert!(
        precipitation_type
            .product_metadata
            .as_ref()
            .and_then(|metadata| metadata.provenance.as_ref())
            .expect("direct composite should carry provenance")
            .flags
            .contains(&rustwx_core::ProductSemanticFlag::Composite)
    );
    let hrrr_support = precipitation_type
        .support
        .iter()
        .find(|target| target.model == Some(ModelId::Hrrr))
        .unwrap();
    assert_eq!(
        hrrr_support.source_routes,
        vec![ProductSourceRoute::DirectNativeCompositeExact]
    );
}

#[test]
fn derived_catalog_includes_intentional_blockers() {
    let catalog = build_supported_products_catalog();
    let entry = catalog
        .derived
        .iter()
        .find(|entry| entry.slug == "stp_effective")
        .expect("catalog should include blocked stp_effective entry");
    assert_eq!(entry.status, ProductCatalogStatus::Blocked);
    assert_eq!(entry.maturity, ProductMaturity::Operational);
    assert!(entry.flags.is_empty());
    assert_eq!(
        entry
            .product_metadata
            .as_ref()
            .and_then(|metadata| metadata.provenance.as_ref())
            .expect("blocked derived entries should still expose typed provenance")
            .lineage,
        rustwx_core::ProductLineage::Derived
    );
    assert_eq!(entry.support.len(), built_in_models().len());
    assert!(
        entry
            .support
            .iter()
            .all(|target| matches!(target.status, ProductTargetStatus::Blocked))
    );
    assert!(
        entry
            .support
            .iter()
            .flat_map(|target| target.blockers.iter())
            .any(|reason| reason.contains("effective SRH") || reason.contains("EBWD")),
        "blocked derived entries should carry the current blocker text"
    );
}

#[test]
fn direct_catalog_offers_terrain_and_never_silently_drops_it() {
    // The terrain frame is what makes a wind or precipitation frame
    // legible.  It renders from a plane the WRF import has always
    // published; what was missing was a catalog row asking for it.  This
    // asserts the row exists AND is not blocked -- the two failure modes
    // that look identical to a viewer staring at a directory with no
    // terrain image in it.
    let catalog = build_supported_products_catalog();
    let entry = catalog
        .direct
        .iter()
        .find(|entry| entry.slug == "terrain_height")
        .expect("terrain_height must be in the direct catalog");
    assert_eq!(entry.title, "Terrain Height");
    assert_eq!(entry.render_style.as_deref(), Some("weather_terrain"));
    // The model identity the wrfout store lane carries.  HREF is the one
    // blocked target, for the pre-existing reason that it serves only its
    // explicit ensprod recipes -- a restriction terrain shares with every
    // other deterministic product, not a terrain-specific gap.
    let wrf = entry
        .support
        .iter()
        .find(|target| target.model == Some(ModelId::WrfGdex))
        .expect("the wrfout store lane's model must appear in the support matrix");
    assert_eq!(wrf.status, ProductTargetStatus::Supported, "{wrf:?}");
    assert!(wrf.blockers.is_empty(), "{wrf:?}");
    let blocked: Vec<&str> = entry
        .support
        .iter()
        .filter(|target| target.status == ProductTargetStatus::Blocked)
        .map(|target| target.target.as_str())
        .collect();
    assert_eq!(blocked, vec!["href"], "{:?}", entry.support);
}

#[test]
fn derived_catalog_includes_depth_specific_ehi_products() {
    let catalog = build_supported_products_catalog();
    let entry = catalog
        .derived
        .iter()
        .find(|entry| entry.slug == "ehi_0_1km")
        .expect("catalog should include supported ehi_0_1km entry");
    assert_eq!(entry.status, ProductCatalogStatus::Supported);
    assert!(!entry.experimental);
    assert_eq!(entry.maturity, ProductMaturity::Operational);
    assert!(entry.notes.iter().any(|note| note.contains("0-1 km SRH")));
    assert_eq!(
        entry
            .product_metadata
            .as_ref()
            .and_then(|metadata| metadata.provenance.as_ref())
            .expect("derived entry should carry provenance")
            .lineage,
        rustwx_core::ProductLineage::Derived
    );
    assert!(
        entry
            .support
            .iter()
            .all(|target| target.source_routes == vec![ProductSourceRoute::CanonicalDerived])
    );
}

#[test]
fn derived_catalog_exposes_fastest_native_routes_per_model() {
    let catalog = build_supported_products_catalog();
    let mlcape = catalog
        .derived
        .iter()
        .find(|entry| entry.slug == "mlcape")
        .expect("catalog should expose mlcape");
    let gfs = mlcape
        .support
        .iter()
        .find(|target| target.model == Some(ModelId::Gfs))
        .unwrap();
    assert_eq!(
        gfs.source_routes,
        vec![
            ProductSourceRoute::CanonicalDerived,
            ProductSourceRoute::NativeProxy
        ]
    );

    let sbcape = catalog
        .derived
        .iter()
        .find(|entry| entry.slug == "sbcape")
        .expect("catalog should expose sbcape");
    let hrrr = sbcape
        .support
        .iter()
        .find(|target| target.model == Some(ModelId::Hrrr))
        .unwrap();
    assert_eq!(
        hrrr.source_routes,
        vec![
            ProductSourceRoute::CanonicalDerived,
            ProductSourceRoute::NativeExact
        ]
    );
}

#[test]
fn catalog_marks_proxy_and_proof_products_explicitly() {
    let catalog = build_supported_products_catalog();

    let proxy = catalog
        .derived
        .iter()
        .find(|entry| entry.slug == "scp_mu_0_3km_0_6km_proxy")
        .expect("catalog should expose proxy SCP entry");
    assert_eq!(proxy.maturity, ProductMaturity::Experimental);
    assert!(proxy.experimental);
    assert!(proxy.flags.contains(&ProductSemanticFlag::Proxy));

    let proof = catalog
        .heavy
        .iter()
        .find(|entry| entry.slug == "severe_proof_panel")
        .expect("catalog should expose proof heavy panel");
    assert_eq!(
        proof.id,
        ProductId::new(ProductKind::Bundled, "severe_proof_panel")
    );
    assert_eq!(proof.maturity, ProductMaturity::Proof);
    assert!(proof.experimental);
    assert!(proof.flags.contains(&ProductSemanticFlag::ProofOriented));
    assert!(proof.flags.contains(&ProductSemanticFlag::Proxy));
    assert_eq!(
        proof
            .product_metadata
            .as_ref()
            .and_then(|metadata| metadata.provenance.as_ref())
            .expect("heavy proof entry should expose provenance")
            .lineage,
        rustwx_core::ProductLineage::Bundled
    );
}

#[test]
fn ecape_catalog_entries_are_regular_derived_products() {
    let catalog = build_supported_products_catalog();
    assert_eq!(catalog.heavy.len(), 1);
    assert_eq!(catalog.heavy[0].slug, "severe_proof_panel");
    for slug in [
        "sbecape",
        "mlecape",
        "muecape",
        "sb_ecape_derived_cape_ratio",
        "ml_ecape_derived_cape_ratio",
        "mu_ecape_derived_cape_ratio",
        "ecape_ehi_0_3km",
    ] {
        let entry = catalog
            .derived
            .iter()
            .find(|entry| entry.slug == slug)
            .unwrap_or_else(|| panic!("catalog should expose {slug} as derived"));
        assert!(
            entry
                .runners
                .iter()
                .any(|runner| runner == "hrrr_derived_batch")
        );
    }
}

#[test]
fn heavy_derived_ecape_entries_do_not_route_through_non_ecape_hour() {
    let catalog = build_supported_products_catalog();
    let sbecape = catalog
        .derived
        .iter()
        .find(|entry| entry.slug == "sbecape")
        .expect("catalog should expose sbecape derived entry");
    assert!(
        sbecape
            .runners
            .iter()
            .any(|runner| runner == "derived_batch")
    );
    assert!(
        sbecape
            .runners
            .iter()
            .any(|runner| runner == "hrrr_derived_batch")
    );
    assert!(
        !sbecape
            .runners
            .iter()
            .any(|runner| runner == "non_ecape_hour")
    );
    assert!(
        !sbecape
            .runners
            .iter()
            .any(|runner| runner == "hrrr_non_ecape_hour")
    );
}

#[test]
fn severe_catalog_entry_is_supported_for_all_built_in_models() {
    let catalog = build_supported_products_catalog();
    let severe = catalog
        .heavy
        .iter()
        .find(|entry| entry.slug == "severe_proof_panel")
        .expect("catalog should expose severe proof panel entry");
    assert_eq!(severe.title, "Severe Map Set");
    assert!(severe.runners.iter().any(|runner| runner == "severe_batch"));
    assert_eq!(severe.support.len(), built_in_models().len());
    assert!(
        severe
            .support
            .iter()
            .all(|target| matches!(target.status, ProductTargetStatus::Supported))
    );
}

#[test]
fn catalog_reroutes_legacy_surface_aliases_into_derived_lane() {
    let catalog = build_supported_products_catalog();
    for slug in ["2m_theta_e_10m_winds", "2m_heat_index", "2m_wind_chill"] {
        assert!(
            catalog.direct.iter().all(|entry| entry.slug != slug),
            "{slug} should not remain in the direct/native lane"
        );
    }

    let theta_e = catalog
        .derived
        .iter()
        .find(|entry| entry.slug == "theta_e_2m_10m_winds")
        .expect("catalog should expose canonical theta-e product");
    assert_eq!(
        theta_e.id,
        ProductId::new(ProductKind::Derived, "theta_e_2m_10m_winds")
    );
    assert!(
        theta_e
            .aliases
            .iter()
            .any(|alias| alias.slug == "2m_theta_e_10m_winds")
    );
    assert!(
        theta_e
            .aliases
            .iter()
            .any(|alias| alias.id == ProductId::new(ProductKind::Derived, "2m_theta_e_10m_winds"))
    );
    assert!(
        theta_e
            .notes
            .iter()
            .any(|note| note.contains("derived lane"))
    );
    let identity = theta_e
        .product_metadata
        .as_ref()
        .and_then(|metadata| metadata.identity.as_ref())
        .expect("catalog entry should expose canonical identity");
    assert_eq!(identity.canonical, theta_e.id);
    assert!(
        identity
            .alias_slugs
            .contains(&"2m_theta_e_10m_winds".to_string())
    );

    let heat_index = catalog
        .derived
        .iter()
        .find(|entry| entry.slug == "heat_index_2m")
        .expect("catalog should expose canonical heat index product");
    assert!(
        heat_index
            .aliases
            .iter()
            .any(|alias| alias.slug == "2m_heat_index")
    );

    let wind_chill = catalog
        .derived
        .iter()
        .find(|entry| entry.slug == "wind_chill_2m")
        .expect("catalog should expose canonical wind chill product");
    assert!(
        wind_chill
            .aliases
            .iter()
            .any(|alias| alias.slug == "2m_wind_chill")
    );
}

#[test]
fn catalog_tracks_legacy_one_hour_qpf_name_in_windowed_lane() {
    let catalog = build_supported_products_catalog();
    assert!(
        catalog.direct.iter().all(|entry| entry.slug != "1h_qpf"),
        "1h_qpf should not remain in the direct/native lane"
    );
    let qpf_1h = catalog
        .windowed
        .iter()
        .find(|entry| entry.slug == "qpf_1h")
        .expect("catalog should expose canonical 1-hour QPF windowed product");
    assert_eq!(qpf_1h.id, ProductId::new(ProductKind::Windowed, "qpf_1h"));
    assert!(qpf_1h.aliases.iter().any(|alias| alias.slug == "1h_qpf"));
    assert!(
        qpf_1h
            .aliases
            .iter()
            .any(|alias| alias.id == ProductId::new(ProductKind::Windowed, "1h_qpf"))
    );
    assert!(
        qpf_1h
            .notes
            .iter()
            .any(|note| note.contains("windowed lane")),
        "catalog notes should keep 1h_qpf routed into the windowed story"
    );
    assert_eq!(
        qpf_1h
            .product_metadata
            .as_ref()
            .and_then(|metadata| metadata.provenance.as_ref())
            .expect("windowed entry should expose typed provenance")
            .window,
        Some(rustwx_core::ProductWindowSpec {
            process: rustwx_core::StatisticalProcess::Accumulation,
            duration_hours: Some(1),
        })
    );
    assert_eq!(
        qpf_1h
            .product_metadata
            .as_ref()
            .and_then(|metadata| metadata.identity.as_ref())
            .expect("windowed entry should expose canonical identity")
            .canonical,
        qpf_1h.id
    );
}

#[test]
fn windowed_catalog_marks_hr_rr_windowed_products_supported() {
    let catalog = build_supported_products_catalog();
    assert_eq!(catalog.windowed.len(), 49);
    // Non-QPF windowed support is an input-requirement gate: a source
    // serving the input fields is Supported, one that does not is
    // Blocked with the missing field named -- so entries are Supported
    // or Partial, never wholly Blocked, and HRRR (which serves every
    // windowed input) is Supported on every entry.
    assert!(catalog.windowed.iter().all(|entry| matches!(
        entry.status,
        ProductCatalogStatus::Supported | ProductCatalogStatus::Partial
    )));
    assert!(catalog.windowed.iter().all(|entry| {
        entry.support.iter().any(|target| {
            target.model == Some(ModelId::Hrrr)
                && matches!(target.status, ProductTargetStatus::Supported)
        })
    }));
    // The wrfout import lane (WrfGdex identity) serves 2-5 km UH, so
    // the UH windows list it as supported -- the STEP17 unlock.
    let uh_run_max = catalog
        .windowed
        .iter()
        .find(|entry| entry.slug == "uh_2to5km_run_max")
        .expect("catalog should expose the UH run-max windowed product");
    assert!(uh_run_max.support.iter().any(|target| {
        target.model == Some(ModelId::WrfGdex)
            && matches!(target.status, ProductTargetStatus::Supported)
    }));
    // Blocked rows must justify themselves with the missing input
    // field -- never with a bare model identity.
    for entry in &catalog.windowed {
        for target in &entry.support {
            if matches!(target.status, ProductTargetStatus::Blocked) {
                assert!(
                    !target.blockers.is_empty()
                        && target
                            .blockers
                            .iter()
                            .all(|blocker| blocker.contains("input field")),
                    "windowed blocker must name the missing input field: {:?}",
                    target
                );
            }
        }
    }
    assert!(
        catalog
            .windowed
            .iter()
            .all(|entry| entry.maturity == ProductMaturity::Operational)
    );
    assert!(catalog.windowed.iter().any(|entry| {
        entry.slug == "qpf_6h"
            && entry
                .runners
                .iter()
                .any(|runner| runner == "hrrr_non_ecape_hour")
            && entry.support[0].blockers.is_empty()
    }));
    let qpf_total = catalog
        .windowed
        .iter()
        .find(|entry| entry.slug == "qpf_total")
        .expect("catalog should expose total-QPF windowed product");
    assert!(
        qpf_total
            .support
            .iter()
            .any(|target| target.model == Some(ModelId::Gfs))
    );
    assert!(
        qpf_total
            .support
            .iter()
            .any(|target| target.model == Some(ModelId::Nbm))
    );
    assert!(
        catalog
            .windowed
            .iter()
            .any(|entry| entry.slug == "10m_wind_0_48h_max")
    );
    assert!(
        catalog
            .windowed
            .iter()
            .any(|entry| entry.slug == "2m_temp_0_24h_max")
    );
    assert!(
        catalog
            .windowed
            .iter()
            .any(|entry| entry.slug == "2m_temp_0_24h_range")
    );
    assert!(
        catalog
            .windowed
            .iter()
            .any(|entry| entry.slug == "2m_rh_0_48h_range")
    );
    assert!(
        catalog
            .windowed
            .iter()
            .any(|entry| entry.slug == "2m_dewpoint_24_48h_min")
    );
    assert!(
        catalog
            .windowed
            .iter()
            .any(|entry| entry.slug == "2m_vpd_0_24h_max")
    );
}

#[test]
fn summary_counts_proxy_and_proof_entries() {
    let catalog = build_supported_products_catalog();
    assert!(catalog.summary.experimental_entries >= 1);
    assert!(catalog.summary.proof_entries >= 1);
    assert!(catalog.summary.proxy_entries >= 2);
}
