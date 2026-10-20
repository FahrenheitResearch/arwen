//! Private referee diagnostic: match observed native MC calls by all eight
//! static parameter words, never by call ordinal modulo a sorted reach list.
//! Ambiguous fingerprints are excluded explicitly. No numerical runtime path.
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, env, error::Error, fs, path::Path, time::Instant};

fn digest(bytes: &[u8]) -> String { format!("{:x}", Sha256::digest(bytes)) }

fn array(root: &Path, manifest: &Value, name: &str, dtype: &str, width: usize)
    -> Result<Vec<u8>, Box<dyn Error>> {
    let spec = &manifest["arrays"][name];
    if spec["dtype"] != dtype || spec["shape"].as_array().map(Vec::len) != Some(1)
        || spec["strides"] != json!([width]) {
        return Err(format!("unsupported array layout: {name}").into());
    }
    let relative = Path::new(spec["path"].as_str().ok_or("missing path")?);
    if relative.components().any(|p| !matches!(p, std::path::Component::Normal(_))) {
        return Err("nonlocal array path".into());
    }
    let bytes = fs::read(root.join(relative))?;
    if bytes.len() != spec["shape"][0].as_u64().ok_or("missing extent")? as usize * width
        || spec["sha256"] != digest(&bytes) {
        return Err(format!("array seal/extent mismatch: {name}").into());
    }
    Ok(bytes)
}

fn main() -> Result<(), Box<dyn Error>> {
    let start = Instant::now();
    let args: Vec<_> = env::args().collect();
    if args.len() != 4 { return Err("usage: native_receiver NETWORK_DIR CALLS_RAW RECEIPT_JSON".into()); }
    let root = Path::new(&args[1]);
    let manifest_bytes = fs::read(root.join("manifest.json"))?;
    let manifest: Value = serde_json::from_slice(&manifest_bytes)?;
    if manifest["dt"] != 300 || manifest["steps"] != 24
        || manifest["qts_subdivisions"] != 12 || manifest["assume_short_ts"] != true {
        return Err("expected actual 24x300s/12-subdivision short-ts observation".into());
    }
    let ids_raw = array(root, &manifest, "network-parameters-row-index", "<i8", 8)?;
    let ids: Vec<_> = ids_raw.chunks_exact(8).map(|b| i64::from_le_bytes(b.try_into().unwrap())).collect();
    let lateral_ids = array(root, &manifest, "lateral-row-index", "<i8", 8)?;
    if lateral_ids != ids_raw { return Err("native parameter/lateral rows differ; explicit join required".into()); }
    let mut names: Vec<_> = manifest["arrays"].as_object().ok_or("missing arrays")?.keys()
        .filter(|k| k.starts_with("lateral-column-")).cloned().collect();
    names.sort();
    if names.len() != 2 { return Err("expected two actual native hourly columns".into()); }
    let columns: Vec<Vec<u32>> = names.iter().map(|name| {
        Ok(array(root, &manifest, name, "<f8", 8)?.chunks_exact(8)
            .map(|b| (f64::from_le_bytes(b.try_into().unwrap()) as f32).to_bits()).collect())
    }).collect::<Result<_, Box<dyn Error>>>()?;
    let parameters: Vec<Vec<u32>> = ["dx", "bw", "tw", "twcc", "n", "ncc", "cs", "s0"]
        .iter().map(|name| {
            Ok(array(root, &manifest, &format!("network-parameters-column-{name}"), "<f8", 8)?
                .chunks_exact(8).map(|b| (f64::from_le_bytes(b.try_into().unwrap()) as f32).to_bits()).collect())
        }).collect::<Result<_, Box<dyn Error>>>()?;
    if columns.iter().chain(parameters.iter()).any(|p| p.len() != ids.len()) {
        return Err("parameter/lateral extent mismatch".into());
    }
    let mut fingerprints: BTreeMap<[u32; 8], Vec<usize>> = BTreeMap::new();
    for row in 0..ids.len() {
        fingerprints.entry(std::array::from_fn(|i| parameters[i][row])).or_default().push(row);
    }
    let calls = fs::read(&args[2])?;
    if calls.is_empty() || calls.len() % 108 != 0 { return Err("invalid native 108-byte call records".into()); }
    let mut observed: BTreeMap<usize, Vec<(usize, u32)>> = BTreeMap::new();
    let (mut ambiguous, mut unmatched) = (0usize, 0usize);
    for (ordinal, call) in calls.chunks_exact(108).enumerate() {
        let word = |i| u32::from_le_bytes(call[i*4..i*4+4].try_into().unwrap());
        if word(0) != 300f32.to_bits() { return Err("non-300s native call".into()); }
        let fingerprint = std::array::from_fn(|i| word(i+5));
        match fingerprints.get(&fingerprint).map(Vec::as_slice) {
            Some([row]) => observed.entry(*row).or_default().push((ordinal, word(4))),
            Some(_) => ambiguous += 1,
            None => unmatched += 1,
        }
    }
    let (mut compared, mut different, mut changed_reaches) = (0usize, 0usize, 0usize);
    let mut witnesses = Vec::new();
    let mut incomplete = Vec::new();
    for (row, values) in &observed {
        if values.len() != 24 { incomplete.push(json!({"id":ids[*row], "calls":values.len()})); continue; }
        let expected = [columns[0][*row], columns[1][*row]];
        for (step, (_, value)) in values.iter().enumerate() {
            compared += 1;
            different += usize::from(*value != expected[step/12]);
        }
        if expected[0] != expected[1] {
            changed_reaches += 1;
            if witnesses.len() < 4 {
                witnesses.push(json!({"reach_id": ids[*row], "hourly_column_bits": expected,
                    "hourly_column_values": expected.map(f32::from_bits),
                    "calls_one_based_step": values.iter().enumerate().map(|(step,(ordinal,bits))|
                        json!({"step":step+1,"call_ordinal_zero_based":ordinal,"qlat_bits":bits})).collect::<Vec<_>>() }));
            }
        }
    }
    let receipt = json!({"schema":"native-troute-receiver-hour-transition-v1",
        "manifest_sha256":digest(&manifest_bytes), "calls_sha256":digest(&calls),
        "calls_bytes":calls.len(), "calls":calls.len()/108,
        "native_dt_seconds":300,"native_steps":24,"qts_subdivisions":12,"assume_short_ts":true,
        "native_lateral_columns":names,"native_parameter_rows":ids.len(),
        "unique_fingerprint_observed_reaches":observed.len(),"ambiguous_calls_excluded":ambiguous,
        "unmatched_calls_excluded":unmatched,"incomplete_reaches_excluded":incomplete,
        "compared_lateral_words":compared,"different_lateral_words":different,
        "reaches_with_different_hourly_inputs":changed_reaches,"witnesses":witnesses,
        "mapping":"all eight actual static float32 parameter words unique among native parameter rows; per-reach observed order",
        "scope":"shipped prescribed hourly laterals in native routing; NOT connected CFE delivery, physical area conversion, or full hydro validation",
        "seconds":start.elapsed().as_secs_f64()});
    fs::write(&args[3], serde_json::to_vec_pretty(&receipt)?)?;
    if different != 0 || compared == 0 || changed_reaches == 0 || !incomplete.is_empty() {
        return Err("receiver witness has differences, incomplete reaches, or no transition evidence".into());
    }
    println!("{compared} lateral words compared, {different} different; {changed_reaches} hourly transitions; {ambiguous} ambiguous and {unmatched} unmatched calls excluded");
    Ok(())
}
