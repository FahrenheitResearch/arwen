use rustwx_products::catalog::{
    ProductCatalogEntry, ProductTargetStatus, SupportedProductsCatalog,
    build_supported_products_catalog,
};
use serde::Serialize;
use serde_json::{Map, Value, json};

const PRODUCT_CATALOG_INVENTORY_V1: &str =
    include_str!("fixtures/product_catalog_inventory_v1.json");

const SENTINEL_PRODUCTS: &[(&str, &str)] = &[
    ("direct", "2m_temperature"),
    ("direct", "composite_reflectivity_uh"),
    ("direct", "mslp_10m_winds"),
    ("direct", "total_qpf"),
    ("derived", "fire_weather_composite"),
    ("derived", "scp_mu_0_3km_0_6km_proxy"),
    ("derived", "theta_e_2m_10m_winds"),
    ("heavy", "severe_proof_panel"),
    ("windowed", "2m_temp_0_24h_max"),
    ("windowed", "qpf_1h"),
    ("windowed", "qpf_total"),
    ("windowed", "uh_2to5km_run_max"),
];

#[test]
fn supported_product_catalog_inventory_matches_fixture() {
    let actual = product_catalog_inventory();
    if std::env::var_os("RW_UPDATE_PRODUCT_CATALOG_FIXTURE").is_some() {
        // Deliberate product-surface changes regenerate the fixture in
        // place; the diff is then reviewed like any other source change.
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/product_catalog_inventory_v1.json"
        );
        // The committed fixture blob carries CRLF endings; match them
        // so a regeneration diff shows only real content changes.
        let pretty = serde_json::to_string_pretty(&actual).expect("inventory serializes");
        let crlf = (pretty + "\n").replace('\n', "\r\n");
        std::fs::write(path, crlf).expect("fixture should be writable");
    }
    let expected: Value =
        serde_json::from_str(PRODUCT_CATALOG_INVENTORY_V1).expect("fixture should be valid JSON");

    assert_eq!(
        actual, expected,
        "product catalog inventory changed; update the fixture only for intentional product-surface changes"
    );
}

fn product_catalog_inventory() -> Value {
    let catalog = build_supported_products_catalog();
    json!({
        "schema": "rustwx-products.catalog_inventory.v1",
        "summary": catalog.summary,
        "lanes": {
            "direct": lane_slugs(&catalog.direct),
            "derived": lane_slugs(&catalog.derived),
            "heavy": lane_slugs(&catalog.heavy),
            "windowed": lane_slugs(&catalog.windowed),
        },
        "sentinels": sentinel_entries(&catalog),
    })
}

fn lane_slugs(entries: &[ProductCatalogEntry]) -> Vec<String> {
    let mut slugs = entries
        .iter()
        .map(|entry| entry.slug.clone())
        .collect::<Vec<_>>();
    slugs.sort();
    slugs
}

fn sentinel_entries(catalog: &SupportedProductsCatalog) -> Value {
    let mut entries = Map::new();
    for (lane, slug) in SENTINEL_PRODUCTS {
        let entry = entries_for_lane(catalog, lane)
            .iter()
            .find(|entry| entry.slug == *slug)
            .unwrap_or_else(|| panic!("sentinel product missing: {lane}:{slug}"));
        entries.insert(format!("{lane}:{slug}"), sentinel_entry(lane, entry));
    }
    Value::Object(entries)
}

fn entries_for_lane<'a>(
    catalog: &'a SupportedProductsCatalog,
    lane: &str,
) -> &'a [ProductCatalogEntry] {
    match lane {
        "direct" => &catalog.direct,
        "derived" => &catalog.derived,
        "heavy" => &catalog.heavy,
        "windowed" => &catalog.windowed,
        _ => panic!("unsupported product catalog lane: {lane}"),
    }
}

fn sentinel_entry(lane: &str, entry: &ProductCatalogEntry) -> Value {
    let mut flags = entry
        .flags
        .iter()
        .map(serialized_string)
        .collect::<Vec<_>>();
    flags.sort();

    let mut aliases = entry
        .aliases
        .iter()
        .map(|alias| alias.slug.clone())
        .collect::<Vec<_>>();
    aliases.sort();

    let supported_targets = entry
        .support
        .iter()
        .filter(|target| matches!(target.status, ProductTargetStatus::Supported))
        .count();
    let blocked_targets = entry
        .support
        .iter()
        .filter(|target| matches!(target.status, ProductTargetStatus::Blocked))
        .count();

    json!({
        "lane": lane,
        "slug": entry.slug,
        "title": entry.title,
        "status": serialized_string(&entry.status),
        "maturity": serialized_string(&entry.maturity),
        "flags": flags,
        "runners": entry.runners,
        "aliases": aliases,
        "supported_targets": supported_targets,
        "blocked_targets": blocked_targets,
    })
}

fn serialized_string<T: Serialize + ?Sized>(value: &T) -> String {
    serde_json::to_value(value)
        .expect("value should serialize")
        .as_str()
        .expect("serialized value should be a string")
        .to_string()
}
