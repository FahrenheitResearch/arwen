//! Apply a case-specific signed-off native defect set without changing raw parity.
//! Native defect: read_geogrid.c:69-76,99-110 leaves unread malloc bytes unset.
//! This proof instrument never changes static generation or ordinary comparisons.
use std::{collections::BTreeSet, path::{Path, PathBuf}};
use serde::{Deserialize, Serialize};
use rw_mpas::{staticfile::compare, sha256_file};

#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Pin { path: PathBuf, sha256: String }
#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    schema: String,
    case_id: String,
    proposal: Pin,
    ruling: Pin,
    native_sha256: String,
    hex_sha256: String,
    field: String,
    dimensions: Vec<(String, usize)>,
    indices: Vec<usize>,
}
fn pinned(pin: &Pin) -> Result<Vec<u8>, String> {
    if sha256_file(&pin.path).map_err(|e| e.to_string())? != pin.sha256 {
        return Err(format!("changed pinned input {}", pin.path.display()));
    }
    std::fs::read(&pin.path).map_err(|e| e.to_string())
}
fn assess(manifest: &Manifest, native: &Path, hex: &Path) -> Result<serde_json::Value, String> {
    if manifest.schema != "mpas-approved-declared-residue/v1" { return Err("unknown residue manifest".into()); }
    // Authorization is the case-specific ruling pinned by the delivered manifest.
    let _ruling = pinned(&manifest.ruling)?;
    let proposal: serde_json::Value = serde_json::from_slice(&pinned(&manifest.proposal)?).map_err(|e|e.to_string())?;
    let cases = proposal["cases"].as_array().ok_or("missing proposed cases")?;
    let cases: Vec<_> = cases.iter().filter(|p| p["case_id"] == manifest.case_id).collect();
    if cases.len()!=1 { return Err("residue requires exactly one proposed case".into()); }
    let proposed = cases[0];
    if proposed["field"] != manifest.field || proposed["field_dimensions"] != serde_json::json!(manifest.dimensions)
        || proposed["indices"] != serde_json::json!(manifest.indices) {
        return Err("declared residue differs from exact proposed field, dimensions or indices".into());
    }
    let count = manifest.dimensions.iter().try_fold(1usize, |n, (_, len)| n.checked_mul(*len)).ok_or("dimensions overflow")?;
    let allowed: BTreeSet<_> = manifest.indices.iter().copied().collect();
    if allowed.is_empty() || allowed.len()!=manifest.indices.len() || allowed.iter().any(|i|*i>=count) {
        return Err("empty, duplicated or out-of-bounds residue indices".into());
    }
    let raw = compare::compare(native, hex).map_err(|e|e.to_string())?;
    if raw.reference_sha256!=manifest.native_sha256 || raw.candidate_sha256!=manifest.hex_sha256 {
        return Err("residue manifest does not bind these exact native and hex containers".into());
    }
    let witness = compare::differing_word_witness(native, hex).map_err(|e|e.to_string())?;
    let file=netcrust::File::open(native).map_err(|e|e.to_string())?;
    let field=file.variable(&manifest.field).ok_or("missing approved field")?;
    let dims: Vec<_> = field.dimensions().iter().map(|d|(d.name().to_owned(),d.len())).collect();
    if dims!=manifest.dimensions { return Err("approved field dimensions changed".into()); }
    let mut residue_differences=0usize;
    let mut outside_differences=0usize;
    for f in &witness.fields {
        for word in &f.words {
            if f.name==manifest.field && allowed.contains(&word.index) { residue_differences+=1; }
            else { outside_differences+=1; }
        }
    }
    let total: usize=raw.fields.iter().map(|f|f.n).sum();
    Ok(serde_json::json!({
        "schema":"mpas-declared-residue-assessment/v1",
        "tier":"DECLARED RESIDUE", "reason":"native reads undefined memory",
        "case_id":manifest.case_id,"manifest":manifest,
        "raw_comparison":raw,"raw_word_witness":witness,
        "declared_residue_words":allowed.len(),"different_declared_residue_words":residue_differences,
        "compared_words_outside_residue":total-allowed.len(),
        "different_words_outside_residue":outside_differences,
        "passed":outside_differences==0,
        "whole_model_proof":false
    }))
}
fn main() -> Result<(), String> {
    let args:Vec<_>=std::env::args().skip(1).collect();
    if args.len()!=4 { return Err("usage: static_declared_residue MANIFEST.json NATIVE.nc HEX.nc REPORT.json".into()); }
    let manifest=serde_json::from_slice(&std::fs::read(&args[0]).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;
    let report=assess(&manifest,Path::new(&args[1]),Path::new(&args[2]))?;
    std::fs::write(&args[3],serde_json::to_vec_pretty(&report).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;
    if report["passed"]!=true { return Err("stored words differ outside approved residue".into()); }
    println!("DECLARED RESIDUE; {} words excluded; 0 differences outside; raw parity unchanged",report["declared_residue_words"]);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use rw_store::netcdf_classic::{NcClassicWriter,NcDim,NcFormat,NcVarDef,NcType,NcData};
    #[test]
    fn exact_residue_keeps_raw_nan_bits_and_rejects_expansion_tampering_and_foreign_words() {
        let root=std::env::temp_dir().join(format!("static-residue-{}",std::process::id()));
        std::fs::create_dir_all(&root).unwrap();
        let native=root.join("native.nc"); let hex=root.join("hex.nc");
        let write=|p:&Path,values:&[f32]| {
            let mut w=NcClassicWriter::create(p,NcFormat::Offset64,vec![NcDim::fixed("nCells",2)],vec![],vec![NcVarDef::new("value",NcType::Float,vec![0])],0).unwrap();
            w.put("value",NcData::Floats(values)).unwrap(); w.finish().unwrap();
        };
        write(&native,&[f32::from_bits(0x7fc00001),2.]); write(&hex,&[f32::from_bits(0x7fc00002),2.]);
        let proposal=root.join("proposal.json");
        std::fs::write(&proposal,serde_json::to_vec(&serde_json::json!({"cases":[{"case_id":"fixture","field":"value","field_dimensions":[["nCells",2]],"indices":[0]}]})).unwrap()).unwrap();
        let ruling=root.join("ruling.txt");std::fs::write(&ruling,"fixture approval").unwrap();
        let mut m=Manifest{schema:"mpas-approved-declared-residue/v1".into(),case_id:"fixture".into(),proposal:Pin{sha256:sha256_file(&proposal).unwrap(),path:proposal},ruling:Pin{sha256:sha256_file(&ruling).unwrap(),path:ruling},native_sha256:sha256_file(&native).unwrap(),hex_sha256:sha256_file(&hex).unwrap(),field:"value".into(),dimensions:vec![("nCells".into(),2)],indices:vec![0]};
        let a=assess(&m,&native,&hex).unwrap();
        assert_eq!(a["tier"],"DECLARED RESIDUE");assert_eq!(a["passed"],true);
        assert_eq!(a["raw_comparison"]["word_identical"],false);
        assert_eq!(a["raw_word_witness"]["fields"][0]["words"][0]["reference"],0x7fc00001u64);
        m.indices.push(1);assert!(assess(&m,&native,&hex).is_err());m.indices.pop();
        let saved=m.hex_sha256.clone();m.hex_sha256="foreign".into();assert!(assess(&m,&native,&hex).is_err());m.hex_sha256=saved;
        m.case_id="other".into();assert!(assess(&m,&native,&hex).is_err());m.case_id="fixture".into();
        write(&hex,&[f32::from_bits(0x7fc00002),3.]);m.hex_sha256=sha256_file(&hex).unwrap();
        let a=assess(&m,&native,&hex).unwrap();assert_eq!(a["passed"],false);assert_eq!(a["different_words_outside_residue"],1);
        std::fs::write(&m.ruling.path,"tampered").unwrap();assert!(assess(&m,&native,&hex).is_err());
        std::fs::remove_dir_all(root).unwrap();
    }
}
