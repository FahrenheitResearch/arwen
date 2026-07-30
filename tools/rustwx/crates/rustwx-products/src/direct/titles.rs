use rustwx_core::ModelId;

use crate::shared_context::static_title_with_suffix;

use super::types::DirectBatchRequest;

fn dataset_token_from_product(product: &str) -> Option<&str> {
    let token = product.split(['-', '_']).next().unwrap_or(product);
    if is_gdex_dataset_token(token) {
        Some(token)
    } else {
        None
    }
}

pub(super) fn is_gdex_dataset_token(token: &str) -> bool {
    token.len() > 1 && token.starts_with('d') && token[1..].chars().all(|ch| ch.is_ascii_digit())
}

fn native_stat_label_from_product(product: &str) -> Option<String> {
    let token = product
        .rsplit('/')
        .next()
        .unwrap_or(product)
        .trim()
        .to_ascii_lowercase();
    let mut token = token
        .strip_suffix("_3hrly")
        .or_else(|| token.strip_suffix("_hourly"))
        .or_else(|| token.strip_suffix("_1hrly"))
        .unwrap_or(token.as_str())
        .to_string();
    token = token.replace('-', "_");
    for suffix in ["_conus", "_ak", "_hi", "_pr"] {
        if let Some(stripped) = token.strip_suffix(suffix) {
            token = stripped.to_string();
            break;
        }
    }
    let label = match token.as_str() {
        "mean" => "Mean".to_string(),
        "avg" | "avrg" => "Average".to_string(),
        "spread" | "sprd" => "Spread".to_string(),
        "std" | "stddev" | "stdev" => "Std Dev".to_string(),
        "min" | "minimum" => "Min".to_string(),
        "max" | "maximum" => "Max".to_string(),
        "prob" | "probability" => "Probability".to_string(),
        "eas" => "EAS".to_string(),
        "lpmm" => "Localized PMM".to_string(),
        "pmmn" | "pmm" => "Probability-Matched Mean".to_string(),
        "ffri" => "FFRI".to_string(),
        value if value.len() >= 2 && value.starts_with('p') => {
            let digits = &value[1..];
            if digits.chars().all(|ch| ch.is_ascii_digit()) {
                value.to_ascii_uppercase()
            } else {
                return None;
            }
        }
        _ => return None,
    };
    Some(label)
}

pub(super) fn native_stat_label_for_request(
    request: &DirectBatchRequest,
    planned_product: Option<&str>,
) -> Option<String> {
    planned_product
        .and_then(|planned| {
            request
                .product_overrides
                .get(planned)
                .and_then(|product| native_stat_label_from_product(product))
                .or_else(|| native_stat_label_from_product(planned))
        })
        .or_else(|| {
            request
                .product_overrides
                .values()
                .find_map(|product| native_stat_label_from_product(product))
        })
}

fn model_title_prefix(model: ModelId) -> String {
    model.as_str().replace('-', " ").to_ascii_uppercase()
}

pub(super) fn apply_native_stat_title_prefix(
    model: ModelId,
    stat_label: &str,
    base_title: &str,
) -> String {
    let model_label = model_title_prefix(model);
    let stat_prefix = format!("{model_label} {stat_label} ");
    if base_title.starts_with(&stat_prefix) {
        return base_title.to_string();
    }
    let model_prefix = format!("{model_label} ");
    if let Some(without_model) = base_title.strip_prefix(&model_prefix) {
        return format!("{model_label} {stat_label} {without_model}");
    }
    format!("{model_label} {stat_label} {base_title}")
}

fn direct_title_for_request(
    request: &DirectBatchRequest,
    planned_product: Option<&str>,
    base_title: &str,
) -> String {
    let mut title = base_title.to_string();
    if let Some(stat_label) = native_stat_label_for_request(request, planned_product) {
        title = apply_native_stat_title_prefix(request.model, &stat_label, &title);
    }
    if request.model != ModelId::WrfGdex || is_local_wrf_netcdf_request(request) {
        return static_title_with_suffix(title);
    }

    let dataset = planned_product
        .and_then(|planned| {
            request
                .product_overrides
                .get(planned)
                .and_then(|product| dataset_token_from_product(product))
                .or_else(|| dataset_token_from_product(planned))
        })
        .or_else(|| {
            request
                .product_overrides
                .values()
                .find_map(|product| dataset_token_from_product(product))
        })
        .unwrap_or("d612005");
    static_title_with_suffix(format!("{title} ({dataset})"))
}

fn is_local_wrf_netcdf_request(request: &DirectBatchRequest) -> bool {
    request
        .subtitle_right_override
        .as_deref()
        .is_some_and(|subtitle| subtitle.to_ascii_lowercase().contains("local wrf netcdf"))
}

pub(super) fn direct_title_for_planned_product(
    request: &DirectBatchRequest,
    planned_product: &str,
    base_title: &str,
) -> String {
    direct_title_for_request(request, Some(planned_product), base_title)
}

pub(super) fn direct_panel_title_for_request(
    request: &DirectBatchRequest,
    base_title: &str,
) -> String {
    direct_title_for_request(request, None, base_title)
}
