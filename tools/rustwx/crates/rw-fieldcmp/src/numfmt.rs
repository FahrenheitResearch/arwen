//! Number rendering that matches the reference judge's table character for
//! character.
//!
//! The reference prints its summary statistics through a general-format
//! specifier with four significant digits, which is not a format Rust ships:
//! it chooses between fixed and scientific notation by decimal exponent,
//! trims trailing zeros afterwards, and pads the exponent to two digits.  A
//! table that differs only in `1e-5` versus `1.000e-05` is a table nobody can
//! diff, so the rule is reproduced here rather than approximated.

/// Render with `significant` significant digits, choosing fixed or
/// scientific notation the way the reference general format does.
///
/// `signed` forces a leading `+` on non-negative values, which the reference
/// uses for every difference column.
pub fn general(value: f64, significant: usize, signed: bool) -> String {
    let precision = significant.max(1);
    if value.is_nan() {
        return if signed { "+nan".to_string() } else { "nan".to_string() };
    }
    if value.is_infinite() {
        let body = if value.is_sign_negative() { "-inf" } else { "inf" };
        return if signed && !value.is_sign_negative() {
            format!("+{body}")
        } else {
            body.to_string()
        };
    }

    // The decimal exponent is read off the scientific rendering rather than
    // from log10, so a value that rounds up into the next decade (9.9996 at
    // four digits) is classified by the digits actually printed.
    let scientific = format!("{:.*e}", precision - 1, value);
    let exponent: i32 = scientific
        .split('e')
        .nth(1)
        .and_then(|tail| tail.parse().ok())
        .unwrap_or(0);

    let rendered = if exponent >= -4 && exponent < precision as i32 {
        let places = (precision as i32 - 1 - exponent).max(0) as usize;
        trim_fraction(&format!("{value:.places$}"))
    } else {
        let (mantissa, _) = scientific.split_once('e').unwrap_or((scientific.as_str(), "0"));
        let sign = if exponent < 0 { '-' } else { '+' };
        format!("{}e{}{:02}", trim_fraction(mantissa), sign, exponent.abs())
    };

    if signed && !rendered.starts_with('-') {
        format!("+{rendered}")
    } else {
        rendered
    }
}

/// Render with a fixed number of decimal places.
pub fn fixed(value: f64, places: usize, signed: bool) -> String {
    if value.is_nan() {
        return if signed { "+nan".to_string() } else { "nan".to_string() };
    }
    let rendered = format!("{value:.places$}");
    if signed && !rendered.starts_with('-') {
        format!("+{rendered}")
    } else {
        rendered
    }
}

/// Drop the trailing zeros of a fractional part, then a bare trailing point.
fn trim_fraction(rendered: &str) -> String {
    if !rendered.contains('.') {
        return rendered.to_string();
    }
    let trimmed = rendered.trim_end_matches('0');
    trimmed.strip_suffix('.').unwrap_or(trimmed).to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every case here is a literal string taken from the reference judge's
    /// shipped table, so the test fails if the renderer drifts from the
    /// output contract rather than merely from a style opinion.
    #[test]
    fn general_matches_the_shipped_table() {
        for (value, expected) in [
            (0.0f64, "0"),
            (283.1234, "283.1"),
            (1526.0, "1526"),
            (4715.0, "4715"),
            (16.25, "16.25"),
            (0.004462, "0.004462"),
            (0.0075315, "0.007532"),
            (300.0, "300"),
            (2.226e-07, "2.226e-07"),
            (7.5e-11, "7.5e-11"),
            (-0.00293, "-0.00293"),
            (1.0e-4, "0.0001"),
            // Rounds up into the 1e-4 decade, so it prints fixed, not
            // scientific: the exponent that selects the notation is the one
            // the rounded digits carry, not the one the input carries.
            (9.9999e-5, "0.0001"),
            (9.9994e-5, "9.999e-05"),
            (1.2345e-5, "1.234e-05"),
        ] {
            assert_eq!(general(value, 4, false), expected, "value {value}");
        }
    }

    #[test]
    fn signed_general_matches_the_delta_columns() {
        for (value, expected) in [
            (0.0f64, "+0"),
            (1.307e-05, "+1.307e-05"),
            (-7.5e-11, "-7.5e-11"),
            (-30.71, "-30.71"),
            (0.05178, "+0.05178"),
            (-0.0009476, "-0.0009476"),
        ] {
            assert_eq!(general(value, 4, true), expected, "value {value}");
        }
    }

    #[test]
    fn a_value_that_rounds_into_the_next_decade_is_classified_by_its_digits() {
        // Four significant digits turn 9999.6 into 1.000e+04, whose exponent
        // is 4 and therefore outside the fixed-notation window.
        assert_eq!(general(9999.6, 4, false), "1e+04");
        assert_eq!(general(9999.4, 4, false), "9999");
    }

    #[test]
    fn fixed_matches_the_domain_sum_and_ratio_columns() {
        assert_eq!(fixed(20272.0, 1, false), "20272.0");
        assert_eq!(fixed(0.4854, 1, true), "+0.5");
        assert_eq!(fixed(-2.1, 1, true), "-2.1");
        assert_eq!(fixed(f64::NAN, 1, true), "+nan");
        assert_eq!(fixed(44.9, 1, false), "44.9");
    }
}
