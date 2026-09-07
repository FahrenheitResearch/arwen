//! Bundled research recommendations. Native commands own configuration and fit.
use serde_json::Value;
use std::sync::OnceLock;

pub fn catalog() -> &'static Value {
    static DATA: OnceLock<Value> = OnceLock::new();
    DATA.get_or_init(|| serde_json::from_str(include_str!("../../../gpuwm/data/tui/research-workspaces.json"))
        .expect("bundled research catalog"))
}
pub fn rows(key: &str) -> &'static [Value] { catalog()[key].as_array().expect("catalog array") }
pub fn hardware() -> &'static Value {
    static DATA: OnceLock<Value> = OnceLock::new();
    DATA.get_or_init(|| serde_json::from_str(include_str!("../../../gpuwm/data/tui/research-hardware-profiles.json")).expect("bundled hardware profiles"))
}
pub fn text(row: &'static Value, key: &str) -> &'static str { row[key].as_str().unwrap_or("") }
pub fn strings<'a>(row: &'a Value, key: &str) -> Vec<&'a str> {
    row[key].as_array().map(|v| v.iter().filter_map(Value::as_str).collect()).unwrap_or_default()
}
pub fn config(index: usize) -> Option<&'static Value> { rows("configurations").get(index) }
pub fn config_by_id(id: &str) -> Option<&'static Value> { rows("configurations").iter().find(|r| r["id"] == id) }
pub fn method(row: &'static Value) -> &'static str {
    match text(row, "method") {
        "regional" => "Regional coverage",
        "nested" => "Focused nested forecast",
        "moving_nest" => "Feature-following nest",
        "archived_downscale" => "Archived-parent downscale",
        "controlled_scenario" => "Controlled initial-state experiment",
        _ => "Research configuration",
    }
}
pub fn family_for_config(row: &'static Value) -> Option<&'static str> {
    let leaf = rows("leaves").iter().find(|v| v["id"] == row["leaf_id"])?;
    let submode = rows("submodes").iter().find(|v| v["id"] == leaf["submode_id"])?;
    Some(text(submode, "family_id"))
}
pub fn setup_route(row: &'static Value) -> crate::workflows::Route {
    if text(row, "method") == "archived_downscale" { crate::workflows::Route::Downscale }
    else if row["requires_existing_state"] == true && text(row, "method") != "controlled_scenario" { crate::workflows::Route::Open }
    else { crate::workflows::Route::New }
}
fn ladder(geometry: &Value) -> String {
    let dx = geometry["root_dx_km"].as_f64().unwrap_or(12.0);
    let ratios = geometry["nest_ratios"].as_array().cloned().unwrap_or_default();
    let mut spacing = vec![format!("{dx}")];
    let mut fine = dx;
    for ratio in ratios.iter().filter_map(Value::as_f64) {
        fine /= ratio;
        spacing.push(format!("{fine:.2}").trim_end_matches('0').trim_end_matches('.').to_owned());
    }
    spacing.join(" → ")
}
pub fn config_summary(row: &'static Value) -> String {
    let hours = row["geometry"]["forecast_hours"].as_f64().unwrap_or(6.0);
    format!("{} · {} km · {hours} h intent", method(row), ladder(&row["geometry"]))
}
pub fn diagnostic_label(name: &str) -> String {
    match name {
        "var:wrf_lapse_rate_0_3km" => "0-3 km plain-temperature lapse rate".into(),
        "var:wrf_lapse_rate_700_500" => "700-500 hPa plain-temperature lapse rate".into(),
        _ => crate::plotsettings::product_label(name),
    }
}
pub fn hardware_preview(row: &'static Value) -> String {
    use crate::workflows::Route;
    match setup_route(row) {
        Route::Downscale => return "8 / 12 / 16 / 24 / 32 GiB: native point sizing fits one child to the actual parent and available memory. Parent spacing, chosen ratio, cadence and surface state remain authoritative.".into(),
        Route::Open => return "Open the supplied scenario first. Its actual state and grid remain authoritative; use Fit a starter to compare resource budgets.".into(),
        _ => {}
    }
    let original = &row["geometry"];
    let controlled = text(row, "method") == "controlled_scenario";
    let intent = if original["nest_ratios"].as_array().is_some_and(Vec::is_empty) { "regional" }
        else if original["preferred_finest_dx_km"].as_f64().unwrap_or(4.0) >= 3.0 { "mesoscale" } else { "storm" };
    let mut lines = vec!["These are profile choices, not checked fits. For an 8 GiB target start with Auto or profile 8. Higher profiles can fail the same question's minimum study area on a smaller budget.".into()];
    for class in ["8", "12", "16", "24", "32"] {
        let profile = &hardware()["profiles"][class];
        let broad = intent == "storm" && profile["storm_context"].is_object()
            && original["minimum_root_span_km"].as_f64().unwrap_or(0.0) >= profile["storm_context_minimum_km"].as_f64().unwrap_or(f64::MAX);
        let geometry = if controlled { original } else if broad { &profile["storm_context"] } else { &profile[intent] };
        let nz = if controlled { 49 } else { profile["nz"].as_u64().unwrap_or(49) };
        lines.push(format!("{class:>2} GiB profile  {} km · {nz} levels{} · fit not checked", ladder(geometry), if broad { " · wider context" } else { "" }));
    }
    lines.push("Actual available memory determines domain sizes; native study-area checks can refuse a fit. Auto selects a profile supported by sampled capacity and free memory.".into());
    lines.join("\n")
}
pub fn detail(row: &'static Value) -> String {
    let mut parts = vec![
        text(row, "research_question").to_owned(),
        format!("Validation status  {}. Review the actual inputs, run and diagnostics before drawing scientific conclusions.", text(row, "validation_status")),
        format!("Method  {}", method(row)),
        format!("Catalog geometry intent  {}. The profile below selects the actual requested ladder; native creation reports the fitted grid.", config_summary(row)),
        format!("Minimum root study span  {} km in each direction", row["geometry"]["minimum_root_span_km"]),
        format!("Profile grid requests\n{}", hardware_preview(row)),
        format!("Study area  {}", text(&row["geometry"], "extent_intent")),
        format!("Compare  {}", text(row, "comparison")),
        format!("Plots  {}", strings(row, "diagnostics").into_iter().map(diagnostic_label).collect::<Vec<_>>().join(", ")),
        format!("Inputs  {}", strings(row, "input_requirements").join(" ")),
    ];
    let further = strings(row, "further_analysis");
    if !further.is_empty() {
        parts.insert(1, format!("Additional analysis required\n{}", further.join("\n")));
    }
    if matches!(text(row, "method"), "nested" | "controlled_scenario") {
        parts.push("Fixed child coverage  The event can leave the child's finite footprint. Review every generated child span and the full expected path before running.".into());
    }
    if row["tracker"].is_object() {
        let tracker = &row["tracker"];
        parts.push(format!("Following  {} · threshold {} {} · every {} s. {}",
            text(tracker, "field"), tracker["threshold"], text(tracker, "units"), tracker["cadence_seconds"], text(tracker, "notes")));
        parts.push("Centroid radius  The current native starting value is 50 km and remains configurable in Domains/Tracking. Its suitability for a particular cell or vortex is unvalidated. Review it against the actual child and movement corridor.".into());
    }
    if row["scenario"].is_object() {
        let scenario = &row["scenario"];
        parts.push(format!("Initial state  +{} K warm bubble · {} m AGL · radius {} km · half-depth {} m · preserve RH {}. Placement: {}.",
            scenario["amplitude_k"], scenario["center_height_m"], scenario["radius_km"], scenario["depth_m"], scenario["rh_preserve"], text(scenario, "placement")));
    }
    parts.push(format!("Limits  {}", strings(row, "limitations").join(" ")));
    let citations = strings(row, "citation_ids").iter().filter_map(|id| rows("citations").iter().find(|c| c["id"] == *id))
        .map(|c| format!("{}: {}", text(c, "title"), text(c, "url"))).collect::<Vec<_>>();
    parts.push(format!("Research sources\n{}", citations.join("\n")));
    parts.join("\n\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn catalog_has_complete_research_paths_and_three_recommendations() {
        assert_eq!(catalog()["schema_version"], 1);
        for (collection, recommendations) in [("families", "recommended_config_ids"), ("submodes", "recommended_config_ids"), ("leaves", "configuration_ids")] {
            for row in rows(collection) {
                let ids = strings(row, recommendations);
                assert!(ids.len() >= 3, "{} needs three recommendations", row["id"]);
                for id in ids { assert!(config_by_id(id).is_some(), "Missing {id}"); }
            }
        }
        for row in rows("configurations") {
            assert!(family_for_config(row).and_then(crate::workflows::mode).is_some());
            assert!(crate::plotsettings::Selection::preset_id(text(row, "plot_preset")).is_ok());
            assert!(!detail(row).is_empty());
        }
    }
    #[test]
    fn profile_choices_do_not_claim_fits_and_keep_the_same_question_remedy() {
        let preview = hardware_preview(config_by_id("rotation.structure").unwrap());
        assert!(preview.contains("8 GiB target start with Auto or profile 8"));
        assert!(preview.contains("same question's minimum study area"));
        for class in [8, 12, 16, 24, 32] {
            let row = preview.lines().find(|line| line.trim_start().starts_with(&format!("{class} GiB profile"))).unwrap();
            assert!(row.ends_with("fit not checked"));
        }
    }
}
