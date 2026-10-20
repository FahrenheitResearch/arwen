// SPDX-License-Identifier: Apache-2.0
// Replay consumer: extract observed CFE depth and original binary nexus words.
// No runoff arithmetic, interpolated routing phases, or fabricated basin mapping.
use std::{env, fs, io::Write, path::Path, time::Instant};
fn number(line: &str, key: &str) -> Result<f64, Box<dyn std::error::Error>> {
    Ok(line.split_once(&format!("\"{key}\":" )).ok_or("missing number")?.1
        .split([',', '}']).next().ok_or("missing value")?.parse()?)
}
fn write(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    fs::OpenOptions::new().write(true).create_new(true).open(path)?.write_all(bytes)
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = env::args().collect();
    if args.len() != 5 { return Err("usage: publication-fixture FRAMEWORK NEXUS ROSTER NEW_DIR".into()); }
    let start = Instant::now();
    let roster = fs::read_to_string(&args[3])?;
    let mut columns = Vec::new(); let mut areas = Vec::new(); let mut receivers = Vec::new();
    let mut ids = Vec::new(); let mut nodes = Vec::new(); let mut count = None;
    for (column, row) in roster.lines().enumerate() {
        let fields: Vec<_> = row.split('\t').collect();
        if fields.len() != 3 { return Err("roster must be catchment, nexus, native area_sqkm".into()); }
        if !fields[..2].iter().all(|id| !id.is_empty() && id.bytes().all(|c| c.is_ascii_alphanumeric() || b"-_".contains(&c))) {
            return Err("unsafe roster ID".into());
        }
        if ids.contains(&fields[0].to_owned()) || nodes.contains(&fields[1].to_owned()) { return Err("this publication witness needs distinct native nexuses".into()); }
        let area: f64 = fields[2].parse()?;
        if !area.is_finite() || area <= 0.0 { return Err("invalid native area".into()); }
        areas.extend_from_slice(&area.to_le_bytes());
        receivers.extend_from_slice(&i32::try_from(column)?.to_le_bytes());
        ids.push(fields[0].to_owned()); nodes.push(fields[1].to_owned());
        let dir = Path::new(&args[1]).join(fields[0]).join("bmi_c_cfe");
        let raw = fs::read(Path::new(&args[2]).join(format!("{}_output.raw", fields[1])))?;
        let mut values = Vec::new();
        for (step, line) in fs::read_to_string(dir.join("steps.jsonl"))?.lines().enumerate() {
            if number(line, "step")? != step as f64 || number(line, "requested_delta")? != 3600.0
                || number(line, "time_before")? != step as f64 * 3600.0
                || number(line, "time_after")? != (step + 1) as f64 * 3600.0 { return Err("native clock differs".into()); }
            let stem = dir.join(format!("step-{step}-outputs"));
            let meta = fs::read_to_string(stem.with_extension("jsonl"))?;
            let q: Vec<_> = meta.lines().filter(|s| s.contains("\"name\":\"Q_OUT\"")).collect();
            if q.len() != 1 || !q[0].contains("\"type\":\"double\"") || !q[0].contains("\"units\":\"m\"") || number(q[0], "nbytes")? != 8.0 { return Err("native depth type/units differ".into()); }
            let offset = number(q[0], "offset")?;
            if offset < 0.0 || offset.fract() != 0.0 { return Err("bad output offset".into()); }
            let data = fs::read(stem.with_extension("bin"))?;
            let depth: [u8; 8] = data.get(offset as usize..offset as usize+8).ok_or("depth outside payload")?.try_into()?;
            let record = raw.get(step*35..(step+1)*35).ok_or("nexus record missing")?;
            if i64::from_le_bytes(record[..8].try_into()?) != step as i64 { return Err("nexus clock differs".into()); }
            std::str::from_utf8(&record[8..27])?;
            let discharge: [u8; 8] = record[27..35].try_into()?;
            if !f64::from_le_bytes(depth).is_finite() || !f64::from_le_bytes(discharge).is_finite() { return Err("nonfinite native word".into()); }
            values.push((depth, discharge));
        }
        if values.is_empty() || raw.len() != values.len()*35 || count.is_some_and(|n| n != values.len()) { return Err("native extents differ".into()); }
        count = Some(values.len()); columns.push(values);
    }
    let steps = count.ok_or("empty roster")?;
    let mut depths = Vec::new(); let mut expected = Vec::new();
    for step in 0..steps { for column in &columns {
        depths.extend_from_slice(&column[step].0); expected.extend_from_slice(&column[step].1);
    }}
    let out = Path::new(&args[4]); fs::create_dir(out)?;
    for (name, bytes) in [("depth.raw", depths), ("expected.raw", expected), ("area.raw", areas), ("receiver.raw", receivers)] { write(&out.join(name), &bytes)?; }
    let strings = |values: &[String]| values.iter().map(|v| format!("\"{v}\"")).collect::<Vec<_>>().join(",");
    let report = format!("{{\"schema\":\"arwen.cfe.publication-fixture/v1\",\"steps\":{steps},\"catchment_ids\":[{}],\"node_ids\":[{}],\"period_seconds\":3600,\"area_units\":\"km2\",\"order\":\"c\",\"parallelism\":\"one Rust host process\",\"wall_ms\":{},\"scope\":\"native hourly publication only; no 300-second routing policy or physical event\"}}\n", strings(&ids), strings(&nodes), start.elapsed().as_secs_f64()*1000.0);
    write(&out.join("fixture.json"), report.as_bytes())?;
    Ok(())
}
