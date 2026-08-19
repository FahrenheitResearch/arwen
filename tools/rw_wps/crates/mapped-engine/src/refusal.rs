//! The refusal grammar of the seam contract.
//!
//! Every failure this engine can reach leaves the process as one JSON
//! object on the LAST stderr line, schema `gpuwm-mapped-refusal-v1`, with
//! a `class` token drawn from the closed set below.  The set is closed
//! because the Python side maps class -> exception type: an unlisted
//! class is a defect there, and growing the list is a contract change
//! (design doc §3.3).
//!
//! `message` is the engine's own diagnosis; `remedy` is what the caller
//! can do about it WITHOUT knowing gpuwm's install state.  Python owns
//! install-state remedies (missing bridges, staged binaries) and re-words
//! them, which is why nothing here mentions ladders or wheels.

use std::fmt;

/// A list of names as Python's `repr` writes it: `['a', 'b']`.
///
/// Refusal messages are compared between the two engines byte for byte,
/// so a message that interpolates a list has to render it the way the
/// Python engine's f-string does.  Rust's `{:?}` writes `["a", "b"]`,
/// which is a different sentence — three real sources' inspection
/// reports differed on nothing else.  The quoting rule is Python's own:
/// single quotes unless the value contains one and no double quote.
pub fn python_list_repr<S: AsRef<str>>(values: &[S]) -> String {
    let mut text = String::from("[");
    for (position, value) in values.iter().enumerate() {
        if position > 0 {
            text.push_str(", ");
        }
        text.push_str(&python_repr(value.as_ref()));
    }
    text.push(']');
    text
}

/// A byte string as Python's `repr` writes it: `b'GRIB'`.
///
/// `gpuwm.ingest.grib.inspect_grib1_envelopes` interpolates the four
/// observed marker bytes with `{...!r}` when a concatenated GRIB1 file
/// loses its message alignment, so the engine's sentence for the same
/// condition has to spell those bytes Python's way: the `b` prefix,
/// single quotes unless the value contains one and no double quote,
/// `\n`/`\r`/`\t`/`\\` named, and every other non-printable octet as a
/// two-digit lowercase `\xNN`.
pub fn python_bytes_repr(value: &[u8]) -> String {
    let quote = if value.contains(&b'\'') && !value.contains(&b'"') {
        '"'
    } else {
        '\''
    };
    let mut text = String::with_capacity(value.len() + 3);
    text.push('b');
    text.push(quote);
    for byte in value {
        match byte {
            b'\\' => text.push_str("\\\\"),
            b'\n' => text.push_str("\\n"),
            b'\r' => text.push_str("\\r"),
            b'\t' => text.push_str("\\t"),
            other if char::from(*other) == quote => {
                text.push('\\');
                text.push(quote);
            }
            other if (0x20..0x7f).contains(other) => text.push(char::from(*other)),
            other => text.push_str(&format!("\\x{other:02x}")),
        }
    }
    text.push(quote);
    text
}

/// A list of numbers as Python's `repr` writes it: `[100.0, 500.0]`.
pub fn python_float_list_repr(values: &[f64]) -> String {
    let mut text = String::from("[");
    for (position, value) in values.iter().enumerate() {
        if position > 0 {
            text.push_str(", ");
        }
        text.push_str(&python_float_repr(*value));
    }
    text.push(']');
    text
}

/// One f64 as Python's `repr` writes it.
///
/// Python prints the shortest round-tripping decimal, switches to
/// exponent form when the exponent is below -4 or at or above 16, and
/// always keeps a fractional part on an integral value.  Rust's `{}`
/// agrees on the digits but never adds the trailing `.0` and never
/// switches to exponent form, and `{:e}` spells the exponent `1e30`
/// where Python spells it `1e+30`.
pub fn python_float_repr(value: f64) -> String {
    if !value.is_finite() {
        // Python's spellings; `{}` would give `NaN` and `inf`.
        return if value.is_nan() {
            "nan".to_owned()
        } else if value > 0.0 {
            "inf".to_owned()
        } else {
            "-inf".to_owned()
        };
    }
    let magnitude = value.abs();
    if value != 0.0 && (magnitude < 1e-4 || magnitude >= 1e16) {
        let text = format!("{value:e}");
        // Rust: `1.5e30` / `1.5e-7`.  Python: `1.5e+30` / `1.5e-07`.
        let (mantissa, exponent) = text.split_once('e').expect("exponent form");
        let (sign, digits) = match exponent.strip_prefix('-') {
            Some(digits) => ('-', digits),
            None => ('+', exponent),
        };
        // No `.0` padding on the mantissa here: Python writes `1e+16`,
        // not `1.0e+16`.  The trailing `.0` rule applies only to the
        // decimal form.
        return format!("{mantissa}e{sign}{digits:0>2}");
    }
    let text = format!("{value}");
    if text.contains('.') {
        text
    } else {
        format!("{text}.0")
    }
}

/// One string as Python's `repr` writes it.
pub fn python_repr(value: &str) -> String {
    let quote = if value.contains('\'') && !value.contains('"') {
        '"'
    } else {
        '\''
    };
    let mut text = String::with_capacity(value.len() + 2);
    text.push(quote);
    for character in value.chars() {
        match character {
            '\\' => text.push_str("\\\\"),
            '\n' => text.push_str("\\n"),
            '\r' => text.push_str("\\r"),
            '\t' => text.push_str("\\t"),
            other if other == quote => {
                text.push('\\');
                text.push(other);
            }
            other => text.push(other),
        }
    }
    text.push(quote);
    text
}

/// Refusal classes, 1:1 with `gpuwm.mapped_engine_bridge.REFUSAL_CLASSES`.
pub mod class {
    /// argv grammar the engine could not parse.
    pub const USAGE: &str = "usage";
    /// a declared subcommand or option that this build does not implement.
    pub const NOT_IMPLEMENTED: &str = "not_implemented";
    /// a declared input file is absent, unreadable, or empty.
    pub const MISSING_INPUT: &str = "missing_input";
    /// the mapping document is not a well-formed `rw-wps.mapping.v1`.
    pub const MAPPING_INVALID: &str = "mapping_invalid";
    /// the input manifest does not describe the supplied bytes.
    pub const MANIFEST_MISMATCH: &str = "manifest_mismatch";
    /// no record satisfied a selector that must match.
    pub const SELECTOR_UNMATCHED: &str = "selector_unmatched";
    /// observed grid octets contradict the mapping's declared grid.
    pub const GRID_MISMATCH: &str = "grid_mismatch";
    /// the byte decode itself failed.
    pub const DECODE_FAILED: &str = "decode_failed";
    /// a canonical frame invariant does not hold.
    pub const FRAME_INVALID: &str = "frame_invalid";
    /// the staged valid times cannot bound a forecast.  Split out of
    /// `frame_invalid` when the Python engine promoted the same
    /// condition to its own `ForcingSeriesRefusal` class (the doors
    /// work): the class table maps 1:1 onto the exception the same
    /// condition raises there, so this condition needs its own token.
    pub const FORCING_SERIES: &str = "forcing_series";
    /// an authority's bytes moved under the decode.
    pub const AUTHORITY_MOVED: &str = "authority_moved";
}

/// One refusal: the contract object, before it is serialized.
#[derive(Debug, Clone)]
pub struct Refusal {
    pub class: &'static str,
    pub message: String,
    pub remedy: String,
}

impl Refusal {
    pub fn new(class: &'static str, message: impl Into<String>, remedy: impl Into<String>) -> Self {
        Self {
            class,
            message: message.into(),
            remedy: remedy.into(),
        }
    }

    /// Serialize as the single stderr line the contract specifies.
    pub fn to_json(&self) -> String {
        let value = serde_json::json!({
            "schema": crate::REFUSAL_SCHEMA,
            "class": self.class,
            "message": self.message,
            "remedy": self.remedy,
        });
        value.to_string()
    }
}

impl fmt::Display for Refusal {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}: {}", self.class, self.message)
    }
}

pub type Result<T> = std::result::Result<T, Refusal>;

/// The remedy every "this mapping does not describe these bytes" refusal
/// shares: the two vocabularies are both printed by the message, and the
/// caller decides which side is wrong.
pub const REMEDY_MAPPING_OR_INPUT: &str =
    "compare the mapping's declaration against the supplied bytes; a wrong \
     declaration is contract work, a wrong file is input data";

pub fn usage(message: impl Into<String>) -> Refusal {
    Refusal::new(
        class::USAGE,
        message,
        "run one of: decode, compose, inspect",
    )
}

pub fn missing_input(message: impl Into<String>) -> Refusal {
    Refusal::new(
        class::MISSING_INPUT,
        message,
        "supply the file at the path named in the input list",
    )
}

pub fn mapping_invalid(message: impl Into<String>) -> Refusal {
    Refusal::new(
        class::MAPPING_INVALID,
        message,
        "fix the mapping document; it must satisfy rw-wps.mapping.v1",
    )
}

pub fn manifest_mismatch(message: impl Into<String>) -> Refusal {
    Refusal::new(
        class::MANIFEST_MISMATCH,
        message,
        "re-stage the inputs the manifest describes, or re-issue the manifest",
    )
}

pub fn selector_unmatched(message: impl Into<String>) -> Refusal {
    Refusal::new(
        class::SELECTOR_UNMATCHED,
        message,
        REMEDY_MAPPING_OR_INPUT,
    )
}

pub fn grid_mismatch(message: impl Into<String>) -> Refusal {
    Refusal::new(class::GRID_MISMATCH, message, REMEDY_MAPPING_OR_INPUT)
}

pub fn decode_failed(message: impl Into<String>) -> Refusal {
    Refusal::new(
        class::DECODE_FAILED,
        message,
        "the supplied bytes did not decode; re-fetch the object and \
         compare its checksum against the acquisition manifest",
    )
}

pub fn frame_invalid(message: impl Into<String>) -> Refusal {
    Refusal::new(
        class::FRAME_INVALID,
        message,
        "the decoded frame does not satisfy the canonical contract; the \
         message names the invariant that failed",
    )
}

pub fn forcing_series(message: impl Into<String>) -> Refusal {
    Refusal::new(
        class::FORCING_SERIES,
        message,
        "stage the whole forcing window this run needs: the first valid \
         time is the initial condition and every later one is a lateral \
         boundary, so a bounded run needs at least two times on one \
         uniform cadence",
    )
}

pub fn authority_moved(message: impl Into<String>) -> Refusal {
    Refusal::new(
        class::AUTHORITY_MOVED,
        message,
        "an authority's bytes changed during the decode; re-run against a \
         quiescent tree",
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lists_render_the_way_pythons_f_string_renders_them() {
        // Refusal messages are compared with the Python engine's byte for
        // byte, so these are not "a reasonable formatting" -- they are the
        // output of `repr()` on the same values, measured.
        assert_eq!(python_list_repr(&["terrain_height"]), "['terrain_height']");
        assert_eq!(
            python_list_repr(&["land_fraction", "skin_temperature"]),
            "['land_fraction', 'skin_temperature']"
        );
        assert_eq!(python_list_repr::<&str>(&[]), "[]");
        assert_eq!(python_repr("it's"), "\"it's\"");
        assert_eq!(python_repr("say \"hi\""), "'say \"hi\"'");
    }

    #[test]
    fn byte_strings_render_the_way_pythons_repr_renders_them() {
        // Measured against CPython: `repr(v)` for each value below.
        assert_eq!(python_bytes_repr(b"GRIB"), "b'GRIB'");
        assert_eq!(python_bytes_repr(b"\x00\x01ab"), "b'\\x00\\x01ab'");
        assert_eq!(python_bytes_repr(b"\x7f\xff"), "b'\\x7f\\xff'");
        assert_eq!(python_bytes_repr(b"a\nb\\c"), "b'a\\nb\\\\c'");
        assert_eq!(python_bytes_repr(b"it's"), "b\"it's\"");
        assert_eq!(python_bytes_repr(b"'\""), "b'\\'\"'");
        assert_eq!(python_bytes_repr(b""), "b''");
    }

    #[test]
    fn floats_render_the_way_pythons_repr_renders_them() {
        // Measured against CPython: `repr(v)` for each value below.
        let values = [
            100.0f64,
            500.0,
            101_500.0,
            0.0,
            -0.5,
            1e16,
            1.5e30,
            1e-4,
            9.9e-5,
            1.5e-7,
            0.1,
            1_234_567_890_123_456.0,
        ];
        assert_eq!(
            python_float_list_repr(&values),
            "[100.0, 500.0, 101500.0, 0.0, -0.5, 1e+16, 1.5e+30, 0.0001, \
             9.9e-05, 1.5e-07, 0.1, 1234567890123456.0]"
        );
        assert_eq!(python_float_repr(f64::INFINITY), "inf");
        assert_eq!(python_float_repr(f64::NEG_INFINITY), "-inf");
        assert_eq!(python_float_repr(f64::NAN), "nan");
    }

    #[test]
    fn refusal_json_carries_the_schema_and_escapes_its_text() {
        let refusal = Refusal::new(class::DECODE_FAILED, "a \"quoted\" path", "do \\ that");
        let parsed: serde_json::Value = serde_json::from_str(&refusal.to_json()).unwrap();
        assert_eq!(parsed["schema"], crate::REFUSAL_SCHEMA);
        assert_eq!(parsed["class"], class::DECODE_FAILED);
        assert_eq!(parsed["message"], "a \"quoted\" path");
        assert_eq!(parsed["remedy"], "do \\ that");
    }
}
