//! The domain folder token every renderer writes under: `d02-3km`,
//! `d03-250m`, or `d02` when the file declares no usable `DX`.
//!
//! One owner, so the weather products (`rw_wrfbatch`) and the coupled-fire
//! products ([`crate::sfire`]) land in the same `<out>/<domain>/<product>/`
//! folder of a run instead of two siblings (`d01-1km` and `d01`).

/// `3km`, `1.5km`, `333m` -- the resolution half of the output token.
/// Sub-kilometre nests read as integer metres because `0.333km` is a
/// worse label for a 333 m nest than `333m` is.
pub fn resolution_token(spacing_m: f64) -> Option<String> {
    let (value, unit) = spacing_parts(spacing_m)?;
    Some(format!("{value}{unit}"))
}

/// The number and unit of a grid spacing, as the tokens spell them.
pub fn spacing_parts(spacing_m: f64) -> Option<(String, &'static str)> {
    if !spacing_m.is_finite() || spacing_m <= 0.0 {
        return None;
    }
    let metres = spacing_m.round();
    if metres >= 1_000.0 {
        Some((trimmed_number(spacing_m / 1_000.0), "km"))
    } else {
        Some((format!("{metres:.0}"), "m"))
    }
}

/// Three decimals at most, with the trailing zeros dropped: `3`, `1.5`,
/// `1.333`.
pub fn trimmed_number(value: f64) -> String {
    let text = format!("{value:.3}");
    let trimmed = text.trim_end_matches('0').trim_end_matches('.');
    if trimmed.is_empty() {
        "0".to_string()
    } else {
        trimmed.to_string()
    }
}

/// `d02-3km` from a domain name and its spacing, `d02` without a usable one.
pub fn domain_folder(domain: &str, spacing_m: Option<f64>) -> String {
    match spacing_m.and_then(resolution_token) {
        Some(resolution) => format!("{domain}-{resolution}"),
        None => domain.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_folder_carries_the_resolution_when_the_file_declares_one() {
        assert_eq!(domain_folder("d01", Some(1_000.0)), "d01-1km");
        assert_eq!(domain_folder("d03", Some(250.0)), "d03-250m");
        assert_eq!(domain_folder("d02", Some(1_333.3333)), "d02-1.333km");
        assert_eq!(domain_folder("d02", None), "d02");
        assert_eq!(domain_folder("d02", Some(0.0)), "d02");
    }
}
