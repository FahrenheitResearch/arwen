//! Generic GRIB2 selector stacks in stored scan order.
#[path = "stack_support/json.rs"]
mod json;
#[path = "stack_support/sha256.rs"]
mod sha256;
use chrono::{Datelike, Duration, NaiveDateTime, Timelike};
use grib_core::grib2::{unpack_message, Grib2File, Grib2Message, GridDefinition};
use json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fs::{self, File};
use std::io::{BufReader, Read};
use std::path::Path;
/// WMO Code Table 4.5 type 1 is the ground/water surface, without a
/// numerical vertical coordinate. ecCodes presents its level as zero even
/// when scale/value octets are missing. Keep the parser's stored descriptors
/// unchanged and use the surface convention at the inventory/stack boundary.
/// Evidence: tests/test_grib2_stack.py::test_eccodes_committed_oracle_identifiers_coefficients_and_stack_bits.
pub fn first_surface_level(product: &grib_core::grib2::ProductDefinition) -> f64 {
    if product.level_type == 1 {
        0.0
    } else {
        product.level_value
    }
}

/// Decode the encoded regular axis without a host-language transform.
/// The increment remains authoritative, including equal cyclic endpoints.
#[no_mangle]
pub unsafe extern "C" fn gpuwm_grib2_axis_f64(
    first: f64, last: f64, increment: f64, count: usize,
    negative: u8, output: *mut f64,
) -> i32 {
    if count == 0 || output.is_null() || negative > 1
        || !first.is_finite() || !last.is_finite()
        || !increment.is_finite() || increment < 0.0
        || count > (isize::MAX as usize) / std::mem::size_of::<f64>() {
        return 1;
    }
    let signed_increment = if negative == 0 { increment } else { -increment };
    let output = std::slice::from_raw_parts_mut(output, count);
    for (index, value) in output.iter_mut().enumerate() {
        *value = first + signed_increment * index as f64;
    }
    0
}

fn iso(t: NaiveDateTime, filename: bool) -> String {
    if filename {
        format!(
            "{:04}-{:02}-{:02}T{:02}{:02}{:02}Z",
            t.year(),
            t.month(),
            t.day(),
            t.hour(),
            t.minute(),
            t.second()
        )
    } else {
        format!(
            "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z",
            t.year(),
            t.month(),
            t.day(),
            t.hour(),
            t.minute(),
            t.second()
        )
    }
}
type Result<T> = std::result::Result<T, Box<dyn Error>>;
fn num(n: impl Into<f64>) -> Value {
    Value::Num(n.into())
}
fn text(s: impl Into<String>) -> Value {
    Value::Str(s.into())
}
fn object(items: Vec<(&str, Value)>) -> Value {
    Value::Object(items.into_iter().map(|(k, v)| (k.into(), v)).collect())
}
fn optional(n: Option<u16>) -> Value {
    n.map_or(Value::Null, |n| num(n))
}
fn nums(n: &[f64]) -> Value {
    Value::Array(n.iter().copied().map(num).collect())
}
#[derive(Debug)]
struct Selector {
    key: String,
    discipline: u8,
    category: u8,
    parameter: u8,
    constituent: Option<u16>,
    aerosol: Option<u16>,
    level_type: u8,
    levels: Option<Vec<f64>>,
}
fn integer(v: &Value, max: u32) -> Result<u32> {
    let n = v.number()?;
    if n < 0.0 || n > max as f64 || n.fract() != 0.0 {
        return Err(format!("selector integer outside 0..{max}: {n}").into());
    }
    Ok(n as u32)
}
fn opt(v: &Value) -> Result<Option<u16>> {
    if *v == Value::Null {
        Ok(None)
    } else {
        Ok(Some(integer(v, 65534)? as u16))
    }
}
fn spec(v: &Value) -> Result<Vec<Selector>> {
    if v.get("schema")?.string()? != "gpuwm-grib2-stack-v1" {
        return Err("unsupported stack SPEC schema".into());
    }
    let mut keys = BTreeSet::new();
    let mut out = Vec::new();
    for s in v.get("select")?.array()? {
        let key = s.get("key")?.string()?.to_owned();
        if key.is_empty()
            || !key
                .bytes()
                .all(|c| c.is_ascii_alphanumeric() || c == b'_' || c == b'-')
            || !keys.insert(key.clone())
        {
            return Err(format!("unsafe or duplicate selector key {key:?}").into());
        }
        let levels = if *s.get("levels")? == Value::Null {
            None
        } else {
            let values = s
                .get("levels")?
                .array()?
                .iter()
                .map(Value::number)
                .collect::<std::result::Result<Vec<_>, _>>()?;
            if values.is_empty()
                || values
                    .iter()
                    .enumerate()
                    .any(|(i, x)| values[..i].contains(x))
            {
                return Err(format!("key {key}: empty or duplicate requested levels").into());
            }
            Some(values)
        };
        out.push(Selector {
            key,
            discipline: integer(s.get("discipline")?, 255)? as u8,
            category: integer(s.get("category")?, 255)? as u8,
            parameter: integer(s.get("parameter")?, 255)? as u8,
            constituent: opt(s.get("constituent_type")?)?,
            aerosol: opt(s.get("aerosol_type")?)?,
            level_type: integer(s.get("level_type")?, 255)? as u8,
            levels,
        });
    }
    if out.is_empty() {
        return Err("the stack would be empty: no selectors".into());
    }
    Ok(out)
}
impl Selector {
    fn matches(&self, m: &Grib2Message) -> bool {
        let p = &m.product;
        self.discipline == m.discipline
            && self.category == p.parameter_category
            && self.parameter == p.parameter_number
            && self.level_type == p.level_type
            && self
                .constituent
                .map_or(true, |v| p.constituent_type == Some(v))
            && self.aerosol.map_or(true, |v| p.aerosol_type == Some(v))
            && self
                .levels
                .as_ref()
                .map_or(true, |l| l.contains(&first_surface_level(p)))
    }
}
struct Identity {
    grid: GridDefinition,
    coords: Vec<f64>,
    pdt: u16,
}
struct Stack {
    key: String,
    reference: NaiveDateTime,
    step: u32,
    unit: u8,
    valid: NaiveDateTime,
    levels: Vec<(f64, Vec<u8>)>,
    seen: Vec<Value>,
}
fn valid(m: &Grib2Message) -> Result<NaiveDateTime> {
    if let Some(end) = m.product.end_of_interval {
        return Ok(end);
    }
    let seconds = match m.product.time_range_unit {
        0 => 60,
        1 => 3600,
        2 => 86400,
        10 => 10800,
        11 => 21600,
        12 => 43200,
        13 => 1,
        u => {
            return Err(format!(
                "forecast time unit {u} is not a fixed duration supported by stack v1"
            )
            .into())
        }
    };
    m.reference_time
        .checked_add_signed(Duration::seconds(
            i64::from(m.product.forecast_time) * seconds,
        ))
        .ok_or_else(|| "valid time overflows".into())
}
fn seen(m: &Grib2Message) -> Value {
    object(vec![
        ("discipline", num(m.discipline)),
        ("category", num(m.product.parameter_category)),
        ("parameter", num(m.product.parameter_number)),
        ("constituent_type", optional(m.product.constituent_type)),
        ("aerosol_type", optional(m.product.aerosol_type)),
        ("level_type", num(m.product.level_type)),
    ])
}
fn collect(
    input: &Path,
    selectors: &[Selector],
) -> Result<(BTreeMap<String, Identity>, Vec<Stack>)> {
    let mut ids: BTreeMap<String, Identity> = BTreeMap::new();
    let mut stacks: Vec<Stack> = Vec::new();
    let mut reader = BufReader::new(File::open(input)?);
    loop {
        let mut header = [0u8; 16];
        if reader.read(&mut header[..1])? == 0 {
            break;
        }
        reader
            .read_exact(&mut header[1..])
            .map_err(|e| format!("truncated GRIB2 header: {e}"))?;
        if &header[..4] != b"GRIB" || header[7] != 2 {
            return Err("invalid GRIB2 envelope marker or edition".into());
        }
        let length = usize::try_from(u64::from_be_bytes(header[8..16].try_into()?))?;
        if length < 20 {
            return Err("GRIB2 envelope shorter than 20 bytes".into());
        }
        let mut bytes = Vec::new();
        bytes.try_reserve_exact(length)?;
        bytes.resize(length, 0);
        bytes[..16].copy_from_slice(&header);
        reader
            .read_exact(&mut bytes[16..])
            .map_err(|e| format!("truncated GRIB2 envelope: {e}"))?;
        for m in Grib2File::from_bytes(&bytes)?.messages {
            let mut decoded = None;
            for s in selectors.iter().filter(|s| s.matches(&m)) {
                let p = &m.product;
                if !first_surface_level(p).is_finite() {
                    return Err(format!("key {}: non-finite level", s.key).into());
                }
                // 3.0 regular latitude/longitude, or 3.30 Lambert conformal
                // (HRRR's native grid): the Lambert parameters go into the
                // manifest so the reader can project targets onto the
                // stored grid; nothing here resamples.
                if m.grid.template != 0 && m.grid.template != 30 {
                    return Err(format!(
                        "key {}: grid template {} unsupported; v1 reads 3.0 and 3.30",
                        s.key, m.grid.template
                    )
                    .into());
                }
                if m.grid.is_reduced || m.grid.nx == 0 || m.grid.ny == 0 {
                    return Err(format!("key {}: non-rectangular or empty grid", s.key).into());
                }
                if p.coordinate_values.iter().any(|v| !v.is_finite()) {
                    return Err(format!("key {}: non-finite coordinate values", s.key).into());
                }
                if let Some(id) = ids.get(&s.key) {
                    if id.pdt != p.template {
                        return Err(format!(
                            "key {}: mixed PDTs {} and {}",
                            s.key, id.pdt, p.template
                        )
                        .into());
                    }
                    if id.grid != m.grid {
                        return Err(format!("key {}: records disagree on grid", s.key).into());
                    }
                    if id.coords != p.coordinate_values {
                        return Err(format!(
                            "key {}: records disagree on Section 4 coordinate values",
                            s.key
                        )
                        .into());
                    }
                } else {
                    ids.insert(
                        s.key.clone(),
                        Identity {
                            grid: m.grid.clone(),
                            coords: p.coordinate_values.clone(),
                            pdt: p.template,
                        },
                    );
                }
                let time = valid(&m)?;
                let ix = stacks.iter().position(|t| {
                    t.key == s.key
                        && t.reference == m.reference_time
                        && t.step == p.forecast_time
                        && t.unit == p.time_range_unit
                });
                let ix = ix.unwrap_or_else(|| {
                    stacks.push(Stack {
                        key: s.key.clone(),
                        reference: m.reference_time,
                        step: p.forecast_time,
                        unit: p.time_range_unit,
                        valid: time,
                        levels: Vec::new(),
                        seen: Vec::new(),
                    });
                    stacks.len() - 1
                });
                let stack = &mut stacks[ix];
                if stack.valid != time {
                    return Err(format!("key {}: records disagree on valid time", s.key).into());
                }
                if stack
                    .levels
                    .iter()
                    .any(|(l, _)| *l == first_surface_level(p))
                {
                    return Err(format!(
                        "key {}: duplicate (key, time, level) record at {} level {}",
                        s.key,
                        time,
                        first_surface_level(p)
                    )
                    .into());
                }
                if decoded.is_none() {
                    let values = unpack_message(&m)?;
                    if values.len() != m.grid.nx as usize * m.grid.ny as usize {
                        return Err(format!(
                            "key {}: decoded point count disagrees with grid",
                            s.key
                        )
                        .into());
                    }
                    decoded = Some(
                        values
                            .into_iter()
                            .flat_map(|v| (v as f32).to_le_bytes())
                            .collect::<Vec<_>>(),
                    );
                }
                stack
                    .levels
                    .push((first_surface_level(p), decoded.as_ref().unwrap().clone()));
                let row = seen(&m);
                if !stack.seen.contains(&row) {
                    stack.seen.push(row)
                }
            }
        }
    }
    for s in selectors {
        if !ids.contains_key(&s.key) {
            return Err(format!(
                "key {}: selector matches zero records; the stack would be empty",
                s.key
            )
            .into());
        }
        for stack in stacks.iter_mut().filter(|t| t.key == s.key) {
            stack.levels.sort_by(|a, b| a.0.total_cmp(&b.0));
            if let Some(requested) = &s.levels {
                for l in requested {
                    if !stack.levels.iter().any(|(v, _)| v == l) {
                        return Err(format!(
                            "key {}: requested level {} missing at {}",
                            s.key, l, stack.valid
                        )
                        .into());
                    }
                }
            }
        }
    }
    stacks.sort_by_key(|t| (t.key.clone(), t.reference, t.step, t.unit));
    Ok((ids, stacks))
}
fn run(input: &Path, specification: &Value, output: &Path) -> Result<()> {
    let selectors = spec(specification)?;
    let (ids, stacks) = collect(input, &selectors)?;
    if output.exists() {
        return Err(format!("refusing to overwrite existing output {output:?}").into());
    }
    let mut filenames = BTreeSet::new();
    for t in &stacks {
        if !filenames.insert(format!("{}__{}.f32le", t.key, iso(t.valid, true))) {
            return Err(format!(
                "key {}: valid time filename collision at {}",
                t.key, t.valid
            )
            .into());
        }
    }
    fs::create_dir(output)?;
    let mut rows = Vec::new();
    for t in stacks {
        // Basic ISO time avoids ':' which is not a portable filename character.
        let filename = format!("{}__{}.f32le", t.key, iso(t.valid, true));
        let data: Vec<u8> = t
            .levels
            .iter()
            .flat_map(|(_, b)| b.iter().copied())
            .collect();
        fs::write(output.join(&filename), &data)?;
        let id = &ids[&t.key];
        let g = &id.grid;
        rows.push(object(vec![
            ("key", text(t.key)),
            ("file", text(filename)),
            ("sha256", text(sha256::hex_sha256(&data))),
            (
                "levels",
                nums(&t.levels.iter().map(|(l, _)| *l).collect::<Vec<_>>()),
            ),
            ("pdt", num(id.pdt)),
            ("selector_values", Value::Array(t.seen)),
            ("reference_time", text(iso(t.reference, false))),
            ("forecast_step", num(t.step)),
            ("forecast_unit", num(t.unit)),
            ("valid_time", text(iso(t.valid, false))),
            ("grid", {
                let mut fields = vec![
                    ("template", num(g.template)),
                    ("nx", num(g.nx)),
                    ("ny", num(g.ny)),
                    ("lat1", num(g.lat1)),
                    ("lat2", num(g.lat2)),
                    ("lon1", num(g.lon1)),
                    ("lon2", num(g.lon2)),
                    ("dlat", num(g.dy)),
                    ("dlon", num(g.dx)),
                    ("scan_mode", num(g.scan_mode)),
                ];
                // Template 3.0 manifests stay byte for byte what they were.
                if g.template == 30 {
                    fields.push(("latin1", num(g.latin1)));
                    fields.push(("latin2", num(g.latin2)));
                    fields.push(("lov", num(g.lov)));
                }
                object(fields)
            }),
        ]));
    }
    let coordinates = Value::Object(
        ids.iter()
            .map(|(k, id)| (k.clone(), nums(&id.coords)))
            .collect(),
    );
    let manifest = object(vec![
        ("schema", text("gpuwm-grib2-stack-v1")),
        ("outputs", Value::Array(rows)),
        ("coordinate_values", coordinates),
    ]);
    fs::write(output.join("stack.json"), manifest.render() + "\n")?;
    Ok(())
}
/// Stack selected GRIB2 fields through the already bundled CPU library.
/// Paths are UTF-8, NUL-terminated strings. Failures use the calling thread
/// bridge message slot in src/wif_ffi.rs without changing ABI 1.
///
/// # Safety
/// Each non-null pointer must address a valid NUL-terminated readable string.
#[no_mangle]
pub unsafe extern "C" fn gpuwm_grib2_stack(
    input_path: *const std::os::raw::c_char,
    spec_json_path: *const std::os::raw::c_char,
    output_dir: *const std::os::raw::c_char,
) -> i32 {
    use crate::wif_ffi::{set_last_error, ERR_INPUT_FORMAT, ERR_INPUT_OPEN};
    use crate::{ERR_NULL, ERR_PANIC, OK};
    use std::ffi::CStr;
    use std::panic::{catch_unwind, AssertUnwindSafe};
    if input_path.is_null() || spec_json_path.is_null() || output_dir.is_null() {
        set_last_error(
            "gpuwm_grib2_stack: input_path, spec_json_path and output_dir must not be null".into(),
        );
        return ERR_NULL;
    }
    let result = catch_unwind(AssertUnwindSafe(|| -> Result<()> {
        let path = |pointer, label| -> Result<std::path::PathBuf> {
            let value = CStr::from_ptr(pointer)
                .to_str()
                .map_err(|error| format!("{label} is not valid UTF-8: {error}"))?;
            if value.is_empty() {
                return Err(format!("{label} is empty").into());
            }
            Ok(Path::new(value).to_path_buf())
        };
        let input = path(input_path, "input_path")?;
        let specification = path(spec_json_path, "spec_json_path")?;
        let output = path(output_dir, "output_dir")?;
        let specification = json::parse(&fs::read_to_string(specification)?)?;
        run(&input, &specification, &output)
    }));
    match result {
        Ok(Ok(())) => {
            set_last_error(String::new());
            OK
        }
        Ok(Err(error)) => {
            let code = if error.downcast_ref::<std::io::Error>().is_some() {
                ERR_INPUT_OPEN
            } else {
                ERR_INPUT_FORMAT
            };
            set_last_error(format!("gpuwm_grib2_stack: {error}"));
            code
        }
        Err(_) => {
            set_last_error("gpuwm_grib2_stack: native stack operation panicked".into());
            ERR_PANIC
        }
    }
}
#[cfg(test)]
mod axis_tests {
    use super::gpuwm_grib2_axis_f64;

    #[test]
    fn encoded_axes_preserve_direction_and_cyclic_increment() {
        for (first, last, increment, negative, expected) in [
            (40.0, 39.0, 0.5, 1, [40.0, 39.5, 39.0]),
            (0.0, 0.0, 120.0, 0, [0.0, 120.0, 240.0]),
        ] {
            let mut values = [f64::NAN; 3];
            assert_eq!(unsafe { gpuwm_grib2_axis_f64(
                first, last, increment, values.len(), negative, values.as_mut_ptr()) }, 0);
            assert_eq!(values.map(f64::to_bits), expected.map(f64::to_bits));
        }
    }

    #[test]
    fn malformed_encoded_axes_refuse_before_writing() {
        let mut values = [123.0; 2];
        for (first, last, increment, count, negative) in [
            (f64::NAN, 0.0, 1.0, 2, 0), (0.0, f64::INFINITY, 1.0, 2, 0),
            (0.0, 0.0, -1.0, 2, 0), (0.0, 0.0, 1.0, 0, 0),
            (0.0, 0.0, 1.0, 2, 2), (0.0, 0.0, f64::NAN, 2, 0),
        ] {
            assert_eq!(unsafe { gpuwm_grib2_axis_f64(
                first, last, increment, count, negative, values.as_mut_ptr()) }, 1);
            assert_eq!(values, [123.0; 2]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    fn ffi_run(input: &Path, specification: &Value, output: &Path) -> Result<()> {
        use std::ffi::CString;
        let spec_path = input.with_extension("spec.json");
        fs::write(&spec_path, specification.render())?;
        let input = CString::new(input.to_str().unwrap())?;
        let spec = CString::new(spec_path.to_str().unwrap())?;
        let output = CString::new(output.to_str().unwrap())?;
        let code = unsafe { gpuwm_grib2_stack(input.as_ptr(), spec.as_ptr(), output.as_ptr()) };
        if code == 0 {
            return Ok(());
        }
        let mut message = vec![0u8; 8192];
        let count =
            unsafe { crate::wif_ffi::gpuwm_bridge_last_error(message.as_mut_ptr(), message.len()) };
        Err(String::from_utf8_lossy(&message[..count])
            .into_owned()
            .into())
    }

    // Envelope construction follows grib-core/src/grib2/unpack.rs:3131.
    fn section(number: u8, length: usize) -> Vec<u8> {
        let mut v = vec![0; length];
        v[..4].copy_from_slice(&(length as u32).to_be_bytes());
        v[4] = number;
        v
    }
    fn message(template: u16, level: u32, discipline: u8, value: f64) -> Vec<u8> {
        let mut s1 = section(1, 21);
        s1[12..14].copy_from_slice(&2026u16.to_be_bytes());
        s1[14] = 9;
        s1[15] = 30;
        let mut s3 = section(3, 72);
        s3[6..10].copy_from_slice(&4u32.to_be_bytes());
        s3[30..34].copy_from_slice(&2u32.to_be_bytes());
        s3[34..38].copy_from_slice(&2u32.to_be_bytes());
        s3[14] = 6;
        s3[46..50].copy_from_slice(&40_000_000u32.to_be_bytes());
        s3[55..59].copy_from_slice(&39_000_000u32.to_be_bytes());
        s3[59..63].copy_from_slice(&1_000_000u32.to_be_bytes());
        s3[63..67].copy_from_slice(&1_000_000u32.to_be_bytes());
        s3[67..71].copy_from_slice(&1_000_000u32.to_be_bytes());
        let shift = if template == 40 { 2 } else { 0 };
        let mut s4 = section(4, 34 + shift + 8);
        s4[5..7].copy_from_slice(&2u16.to_be_bytes());
        s4[7..9].copy_from_slice(&template.to_be_bytes());
        s4[9] = if discipline == 192 { 210 } else { 20 };
        s4[10] = if discipline == 192 { 1 } else { 2 };
        if shift == 2 {
            s4[11..13].copy_from_slice(&5u16.to_be_bytes())
        }
        s4[17 + shift] = 1;
        s4[18 + shift..22 + shift].copy_from_slice(&3u32.to_be_bytes());
        s4[22 + shift] = 105;
        s4[24 + shift..28 + shift].copy_from_slice(&level.to_be_bytes());
        s4[28 + shift] = 255;
        s4[34 + shift..38 + shift].copy_from_slice(&1000f32.to_be_bytes());
        s4[38 + shift..42 + shift].copy_from_slice(&0.5f32.to_be_bytes());
        let mut s5 = section(5, 12);
        s5[5..9].copy_from_slice(&4u32.to_be_bytes());
        s5[9..11].copy_from_slice(&4u16.to_be_bytes());
        s5[11] = 2;
        let mut s6 = section(6, 6);
        s6[5] = 255;
        let mut s7 = section(7, 37);
        for i in 0..4 {
            s7[5 + 8 * i..13 + 8 * i].copy_from_slice(&(value + i as f64).to_be_bytes());
        }
        let mut out = b"GRIB\0\0\0\x02".to_vec();
        out[6] = discipline;
        out.extend_from_slice(
            &((16 + s1.len() + s3.len() + s4.len() + s5.len() + s6.len() + s7.len() + 4) as u64)
                .to_be_bytes(),
        );
        for s in [s1, s3, s4, s5, s6, s7] {
            out.extend(s)
        }
        out.extend(b"7777");
        out
    }
    struct Scratch(std::path::PathBuf);
    impl Scratch {
        fn new() -> Self {
            static COUNT: AtomicUsize = AtomicUsize::new(0);
            let path = std::env::temp_dir()
                .join(format!(
                    "stack-test-{}-{}",
                    std::process::id(),
                    COUNT.fetch_add(1, Ordering::Relaxed)
                ));
            fs::create_dir_all(&path).unwrap();
            Self(path)
        }
    }
    impl Drop for Scratch {
        fn drop(&mut self) {
            // Fixtures belong to this test's unique directory in TMPDIR.
            let _ = fs::remove_dir_all(&self.0);
        }
    }
    fn selectors(levels: &str) -> Value {
        json::parse(&format!(r#"{{"schema":"gpuwm-grib2-stack-v1","select":[{{"key":"gas","discipline":0,"category":20,"parameter":2,"constituent_type":null,"aerosol_type":null,"level_type":105,"levels":{levels}}},{{"key":"local","discipline":192,"category":210,"parameter":1,"constituent_type":null,"aerosol_type":null,"level_type":105,"levels":null}}]}}"#)).unwrap()
    }
    fn input(s: &Scratch, messages: Vec<Vec<u8>>) -> std::path::PathBuf {
        let p = s.0.join("input.grib2");
        fs::write(&p, messages.concat()).unwrap();
        p
    }
    #[test]
    fn interleaved_local_and_chemical_records_stack_in_ascending_levels() {
        let s = Scratch::new();
        let p = input(
            &s,
            vec![
                message(40, 137, 0, 10.0),
                message(0, 2, 192, 20.0),
                message(40, 1, 0, 30.0),
                message(0, 1, 192, 40.0),
            ],
        );
        let out = s.0.join("stacks");
        ffi_run(&p, &selectors("[1,137]"), &out).unwrap();
        let m = json::parse(&fs::read_to_string(out.join("stack.json")).unwrap()).unwrap();
        let rows = m.get("outputs").unwrap().array().unwrap();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].get("levels").unwrap(), &nums(&[1.0, 137.0]));
        assert_eq!(rows[0].get("pdt").unwrap(), &num(40u16));
        assert_eq!(rows[1].get("pdt").unwrap(), &num(0u16));
        let data = fs::read(out.join(rows[0].get("file").unwrap().string().unwrap())).unwrap();
        let f: Vec<_> = data
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        assert_eq!(f, vec![30., 31., 32., 33., 10., 11., 12., 13.]);
        assert_eq!(
            rows[0].get("sha256").unwrap().string().unwrap(),
            sha256::hex_sha256(&data)
        );
        assert_eq!(
            rows[0].get("valid_time").unwrap().string().unwrap(),
            "2026-09-30T03:00:00Z"
        );
        assert_eq!(
            m.get("coordinate_values").unwrap().get("gas").unwrap(),
            &nums(&[1000., 0.5])
        );
    }
    fn refusal(messages: Vec<Vec<u8>>, levels: &str, expected: &str) {
        let s = Scratch::new();
        let p = input(&s, messages);
        let out = s.0.join("stacks");
        let e = ffi_run(&p, &selectors(levels), &out)
            .unwrap_err()
            .to_string();
        assert!(e.contains(expected), "{e}");
        assert!(!out.exists());
    }
    #[test]
    fn zero_match_refuses() {
        refusal(
            vec![message(40, 1, 0, 0.)],
            "null",
            "the stack would be empty",
        )
    }
    #[test]
    fn duplicate_record_refuses() {
        refusal(
            vec![message(40, 1, 0, 0.), message(40, 1, 0, 0.)],
            "null",
            "duplicate (key, time, level)",
        )
    }
    #[test]
    fn requested_level_missing_refuses() {
        refusal(
            vec![message(40, 1, 0, 0.), message(0, 1, 192, 0.)],
            "[1,2]",
            "requested level 2 missing",
        )
    }
    #[test]
    fn mixed_pdts_refuse() {
        refusal(
            vec![message(40, 1, 0, 0.), message(0, 2, 0, 0.)],
            "null",
            "mixed PDTs",
        )
    }
    #[test]
    fn coordinate_disagreement_refuses() {
        let mut m = message(40, 2, 0, 0.);
        let start = 16 + 21 + 72 + 36;
        m[start..start + 4].copy_from_slice(&2000f32.to_be_bytes());
        refusal(vec![message(40, 1, 0, 0.), m], "null", "coordinate values")
    }
    #[test]
    fn grid_disagreement_refuses() {
        let mut m = message(40, 2, 0, 0.);
        m[16 + 21 + 71] = 64;
        refusal(vec![message(40, 1, 0, 0.), m], "null", "disagree on grid")
    }
    fn gas_only() -> Value {
        json::parse(r#"{"schema":"gpuwm-grib2-stack-v1","select":[{"key":"gas","discipline":0,"category":20,"parameter":2,"constituent_type":null,"aerosol_type":null,"level_type":105,"levels":null}]}"#).unwrap()
    }
    /// The same message on a 2 x 2 Lambert conformal grid (template 3.30,
    /// HRRR's projection): an 81-octet Section 3 replaces the 72-octet one.
    fn lambert(level: u32, value: f64) -> Vec<u8> {
        let m = message(0, level, 0, value);
        let start = 16 + 21;
        let mut s3 = section(3, 81);
        s3[6..10].copy_from_slice(&4u32.to_be_bytes());
        s3[12..14].copy_from_slice(&30u16.to_be_bytes());
        s3[14] = 6;
        s3[30..34].copy_from_slice(&2u32.to_be_bytes());
        s3[34..38].copy_from_slice(&2u32.to_be_bytes());
        s3[38..42].copy_from_slice(&21_138_123u32.to_be_bytes());
        s3[42..46].copy_from_slice(&237_280_472u32.to_be_bytes());
        s3[47..51].copy_from_slice(&38_500_000u32.to_be_bytes());
        s3[51..55].copy_from_slice(&262_500_000u32.to_be_bytes());
        s3[55..59].copy_from_slice(&3_000_000u32.to_be_bytes());
        s3[59..63].copy_from_slice(&3_000_000u32.to_be_bytes());
        s3[64] = 64;
        s3[65..69].copy_from_slice(&38_500_000u32.to_be_bytes());
        s3[69..73].copy_from_slice(&38_500_000u32.to_be_bytes());
        let mut out = m[..start].to_vec();
        out.extend_from_slice(&s3);
        out.extend_from_slice(&m[start + 72..]);
        let length = out.len() as u64;
        out[8..16].copy_from_slice(&length.to_be_bytes());
        out
    }
    #[test]
    fn lambert_records_stack_and_carry_their_projection() {
        let s = Scratch::new();
        let p = input(&s, vec![lambert(2, 5.0), lambert(1, 1.0)]);
        let out = s.0.join("stacks");
        ffi_run(&p, &gas_only(), &out).unwrap();
        let m = json::parse(&fs::read_to_string(out.join("stack.json")).unwrap()).unwrap();
        let row = &m.get("outputs").unwrap().array().unwrap()[0];
        let grid = row.get("grid").unwrap();
        assert_eq!(grid.get("template").unwrap(), &num(30u16));
        assert_eq!(grid.get("latin1").unwrap(), &num(38.5));
        assert_eq!(grid.get("lov").unwrap(), &num(262.5));
        assert_eq!(grid.get("dlon").unwrap(), &num(3000.0));
        assert_eq!(grid.get("scan_mode").unwrap(), &num(64u8));
        let data = fs::read(out.join(row.get("file").unwrap().string().unwrap())).unwrap();
        let f: Vec<_> = data
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        assert_eq!(f, vec![1., 2., 3., 4., 5., 6., 7., 8.]);
    }
    #[test]
    fn regular_grid_manifests_carry_no_lambert_keys() {
        let s = Scratch::new();
        let p = input(&s, vec![message(0, 1, 0, 1.0)]);
        let out = s.0.join("stacks");
        ffi_run(&p, &gas_only(), &out).unwrap();
        let text = fs::read_to_string(out.join("stack.json")).unwrap();
        assert!(!text.contains("latin1") && !text.contains("\"lov\""), "{text}");
    }
    #[test]
    fn unsupported_grid_refuses_by_number() {
        let mut m = message(40, 1, 0, 0.);
        m[16 + 21 + 12..16 + 21 + 14].copy_from_slice(&99u16.to_be_bytes());
        refusal(vec![m], "null", "grid template 99")
    }
    #[test]
    fn truncated_envelope_refuses() {
        let mut m = message(40, 1, 0, 0.);
        m.pop();
        refusal(vec![m], "null", "truncated GRIB2 envelope")
    }
    #[test]
    fn json_and_sha_known_answers() {
        assert_eq!(
            sha256::hex_sha256(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        assert_eq!(
            json::parse(r#""\ud83d\ude00""#).unwrap().string().unwrap(),
            "\u{1f600}"
        );
        for s in [
            "01",
            "1.",
            "1e",
            "[1,]",
            r#"{"a":1,"a":2}"#,
            "null trailing",
        ] {
            assert!(json::parse(s).is_err(), "{s}");
        }
    }
}
