use rustwx_core::ModelId;

use crate::shared_context::static_title_with_suffix;

use super::types::DerivedBatchRequest;

pub(super) fn dataset_token_from_product(product: &str) -> Option<&str> {
    let token = product.split(['-', '_']).next().unwrap_or(product);
    if is_gdex_dataset_token(token) {
        Some(token)
    } else {
        None
    }
}

pub(super) fn derived_title_for_model(model: ModelId, base_title: &str) -> String {
    if model == ModelId::WrfGdex {
        let dataset = dataset_token_from_product("d612005-hist2d").unwrap_or("d612005");
        static_title_with_suffix(format!("{base_title} ({dataset})"))
    } else {
        static_title_with_suffix(base_title)
    }
}

pub(super) fn derived_title_for_request(request: &DerivedBatchRequest, base_title: &str) -> String {
    if is_local_wrf_netcdf_request(request) {
        return static_title_with_suffix(base_title);
    }
    if request.model != ModelId::WrfGdex {
        return static_title_with_suffix(base_title);
    }

    let dataset = request
        .surface_product_override
        .as_deref()
        .and_then(dataset_token_from_product)
        .or_else(|| {
            request
                .pressure_product_override
                .as_deref()
                .and_then(dataset_token_from_product)
        })
        .unwrap_or("d612005");
    static_title_with_suffix(format!("{base_title} ({dataset})"))
}

fn is_local_wrf_netcdf_request(request: &DerivedBatchRequest) -> bool {
    request.model == ModelId::WrfGdex
        && [
            request.surface_product_override.as_deref(),
            request.pressure_product_override.as_deref(),
        ]
        .into_iter()
        .flatten()
        .any(|product| product.eq_ignore_ascii_case("local_wrf_netcdf"))
}

#[derive(Clone, Copy, Default)]
pub(super) struct DerivedRenderOverrides<'a> {
    pub(super) output_suffix: Option<&'a str>,
    pub(super) subtitle_left: Option<&'a str>,
    pub(super) subtitle_right: Option<&'a str>,
}

pub(super) fn sanitize_output_suffix(value: &str) -> String {
    let mut out = String::new();
    for ch in value.chars() {
        if ch.is_ascii_alphanumeric() {
            out.push(ch.to_ascii_lowercase());
        } else if matches!(ch, '_' | '-' | '.') {
            out.push(ch);
        } else if ch.is_whitespace() {
            out.push('_');
        }
    }
    out.trim_matches(['_', '-', '.']).to_string()
}

pub(super) fn derived_output_suffix(output_suffix: Option<&str>) -> String {
    let mut suffix = String::new();
    if let Some(output_suffix) = output_suffix {
        let sanitized = sanitize_output_suffix(output_suffix);
        if !sanitized.is_empty() {
            suffix.push('_');
            suffix.push_str(&sanitized);
        }
    }
    suffix
}

pub(super) fn is_gdex_dataset_token(token: &str) -> bool {
    token.len() > 1 && token.starts_with('d') && token[1..].chars().all(|ch| ch.is_ascii_digit())
}
