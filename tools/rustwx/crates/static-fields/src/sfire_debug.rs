//! WRF SFIRE `write_array_m3` formatted debug arrays.
use crate::error::{Result, StaticError};
use std::io::{BufWriter, Write};
use std::path::PathBuf;

#[derive(Debug, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FireDebugRequest {
    pub name: String,
    /// Existing output directory. A relative directory uses the process cwd.
    pub path: PathBuf,
    /// Contiguous C-order dimensions: vertical, south-north, west-east.
    pub shape: [usize; 3],
    /// WRF inclusive bounds: its, ite, jts, jte, kts, kte.
    pub bounds: [i32; 6],
    pub step: u64,
}

/// A default-REAL E20.12 record, including the newline.
fn real_record(value: f32) -> String {
    let body = if value.is_nan() {
        "NaN".to_owned()
    } else if value.is_infinite() {
        if value.is_sign_negative() { "-Infinity" } else { "Infinity" }.to_owned()
    } else {
        let sign = if value.is_sign_negative() { "-" } else { "" };
        if value == 0.0 {
            format!("{sign}0.000000000000E+00")
        } else {
            // f64 represents every default-REAL value exactly. Rust rounds
            // these twelve significant decimal digits before shifting the
            // decimal point to the Fortran E descriptor's leading zero.
            let scientific = format!("{:.11e}", f64::from(value).abs());
            let (coefficient, exponent) = scientific.split_once('e').unwrap();
            let digits = coefficient.replace('.', "");
            let exponent = exponent.parse::<i32>().unwrap() + 1;
            format!("{sign}0.{digits}E{exponent:+03}")
        }
    };
    format!("{body:>20}\n")
}

pub fn write_array(request: &FireDebugRequest, values: &[f32]) -> Result<PathBuf> {
    if request.name.is_empty()
        || request.name.len() > 115
        || !request.name.bytes().all(|byte| byte.is_ascii_alphanumeric() || byte == b'_')
    {
        return Err(StaticError::Invalid(
            "SFIRE debug name must be a field identifier without directory components".into(),
        ));
    }
    if request.step == 0 {
        return Err(StaticError::Invalid("SFIRE debug step must be positive".into()));
    }
    let expected = request.shape.iter().try_fold(1_usize, |length, size| {
        if *size == 0 { None } else { length.checked_mul(*size) }
    });
    if expected != Some(values.len()) {
        return Err(StaticError::Invalid(
            "SFIRE debug array length differs from its nonempty C-order shape".into(),
        ));
    }
    for (dimension, bounds) in request.bounds.chunks_exact(2).enumerate() {
        let extent = i64::from(bounds[1]) - i64::from(bounds[0]) + 1;
        if extent <= 0 || extent as u64 != request.shape[2 - dimension] as u64 {
            return Err(StaticError::Invalid(
                "SFIRE debug inclusive bounds differ from its C-order shape".into(),
            ));
        }
    }
    // WRF's I8.8 filename overflows to stars after step 99999999. Keep the
    // complete step number so distinct long runs cannot overwrite each other.
    let output = request.path.join(format!("{}_{:08}.txt", request.name, request.step));
    let mut writer = BufWriter::new(std::fs::File::create(&output)?);
    for bound in request.bounds {
        writer.write_all(real_record(bound as f32).as_bytes())?;
    }
    for value in values {
        writer.write_all(real_record(*value).as_bytes())?;
    }
    writer.flush()?;
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn compiled_wrf_default_real_fixture_is_byte_identical() {
        let metadata: Vec<serde_json::Value> = serde_json::from_str(include_str!(
            "../../../../sfire_wrf471_oracle/fixtures/debug/cases.json")).unwrap();
        let fixtures: [(&[u8], &str); 3] = [
            (include_bytes!("../../../../sfire_wrf471_oracle/fixtures/debug/special.f32"),
             include_str!("../../../../sfire_wrf471_oracle/fixtures/debug/special_00000001.txt")),
            (include_bytes!("../../../../sfire_wrf471_oracle/fixtures/debug/header_edges.f32"),
             include_str!("../../../../sfire_wrf471_oracle/fixtures/debug/header_edges_00000031.txt")),
            (include_bytes!("../../../../sfire_wrf471_oracle/fixtures/debug/broad.f32"),
             include_str!("../../../../sfire_wrf471_oracle/fixtures/debug/broad_99999999.txt")),
        ];
        for (mut case, (raw, expected)) in metadata.into_iter().zip(fixtures) {
            case["path"] = ".".into();
            let request: FireDebugRequest = serde_json::from_value(case).unwrap();
            let mut text = String::new();
            for bound in request.bounds { text.push_str(&real_record(bound as f32)); }
            for word in raw.chunks_exact(4) {
                let value = f32::from_bits(u32::from_le_bytes(word.try_into().unwrap()));
                text.push_str(&real_record(value));
            }
            assert_eq!(text, expected, "{}", request.name);
        }
    }

    #[test]
    fn default_real_records_keep_signed_zero_and_full_float_range() {
        for (value, expected) in [
            (0.0, "  0.000000000000E+00\n"),
            (-0.0, " -0.000000000000E+00\n"),
            (1.0, "  0.100000000000E+01\n"),
            (-1.0, " -0.100000000000E+01\n"),
            (f32::from_bits(1), "  0.140129846432E-44\n"),
            (f32::MIN_POSITIVE, "  0.117549435082E-37\n"),
            (f32::MAX, "  0.340282346639E+39\n"),
            (f32::INFINITY, "            Infinity\n"),
            (f32::NEG_INFINITY, "           -Infinity\n"),
            (f32::NAN, "                 NaN\n"),
        ] {
            assert_eq!(real_record(value), expected);
        }
    }

    #[test]
    fn malformed_bounds_and_path_components_are_refused_before_io() {
        let mut request = FireDebugRequest { name: "field".into(), path: Path::new(".").into(),
            shape: [1, 1, 1], bounds: [1, 1, 1, 1, 1, 1], step: 1 };
        request.name = "../field".into();
        assert!(write_array(&request, &[0.0]).unwrap_err().to_string().contains("identifier"));
        request.name = "field".into();
        request.bounds[1] = 2;
        assert!(write_array(&request, &[0.0]).unwrap_err().to_string().contains("bounds"));
        request.bounds[1] = 1;
        assert!(write_array(&request, &[]).unwrap_err().to_string().contains("length"));
    }
}
