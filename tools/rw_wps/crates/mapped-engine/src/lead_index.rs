//! Strict wgrib2 sidecar selection; no GRIB values are decoded here.
use crate::refusal::{frame_invalid, Result};
use serde_json::{json, Value};
pub fn select(text: &str, lead: u64, object_bytes: u64) -> Result<Value> {
    let mut rows = Vec::new();
    for line in text.lines().filter(|s| !s.trim().is_empty()) {
        let fields: Vec<&str> = line.split(':').collect();
        if fields.len() < 7 { return Err(frame_invalid("lead index has an incomplete record")); }
        let ordinal = fields[0].parse::<usize>().map_err(|_| frame_invalid("lead index record ordinal is invalid"))?;
        let offset = fields[1].parse::<u64>().map_err(|_| frame_invalid("lead index byte offset is invalid"))?;
        if ordinal != rows.len()+1 || offset >= object_bytes || (rows.is_empty() && offset != 0)
            || rows.last().is_some_and(|r: &(u64, bool)| r.0 >= offset) {
            return Err(frame_invalid("lead index offsets or ordinals are not a complete increasing object inventory"));
        }
        let wanted = format!("{lead} hour fcst");
        rows.push((offset, fields[5] == wanted || (lead == 0 && fields[5] == "anl")));
    }
    let mut ranges = Vec::new();
    for (i, (start, keep)) in rows.iter().enumerate() {
        let end = rows.get(i+1).map_or(object_bytes, |r| r.0);
        if end-start < 20 { return Err(frame_invalid("lead index record is shorter than a GRIB message")); }
        if *keep { ranges.push(json!([start, end-start])); }
    }
    if ranges.is_empty() { return Err(frame_invalid(format!("lead index has no instantaneous f{lead:03} records"))); }
    Ok(json!({"schema":"gpuwm-lead-index-ranges-v1", "lead":lead,"object_bytes":object_bytes,"ranges":ranges}))
}
pub fn run(args: &[String]) -> Result<Value> {
    if args.len()!=3 { return Err(frame_invalid("index-ranges requires INDEX LEAD OBJECT_BYTES")); }
    let text = std::fs::read_to_string(&args[0]).map_err(|e| frame_invalid(e.to_string()))?;
    let lead = args[1].parse().map_err(|_| frame_invalid("invalid lead"))?;
    let bytes = args[2].parse().map_err(|_| frame_invalid("invalid object length"))?;
    select(&text, lead, bytes)
}
#[cfg(test)] mod tests {
 use super::*;
 #[test] fn chooses_exact_lead_and_last_record() {
 let text="1:0:d=x:TMP:1000 mb:3 hour fcst:ENS=x\n2:30:d=x:TMP:900 mb:6 hour fcst:ENS=x\n3:70:d=x:TMP:800 mb:3 hour fcst:ENS=x\n";
 assert_eq!(select(text,3,120).unwrap()["ranges"],json!([[0,30],[70,50]]));
 assert!(select(text,0,120).is_err());
 assert!(select(&text.replace("2:30", "2:0"),3,120).is_err());
 assert!(select(&text.replace("3 hour fcst", "0-3 hour ave fcst"),3,120).is_err());
 }
}
