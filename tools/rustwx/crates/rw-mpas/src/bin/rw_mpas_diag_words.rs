//! Test receipt extractor: preserve native netCDF binary32 words in Rust.
use std::io::Write;
use std::path::Path;
use sha2::{Digest,Sha256};
fn main() -> Result<(),Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len()<4 { return Err("usage: rw_mpas_diag_words <netcdf> <directory> <field>...".into()); }
    let file=netcrust::open(&args[1])?; let dir=Path::new(&args[2]); std::fs::create_dir_all(dir)?;
    let mut records=vec![];
    for name in &args[3..] {
        let var=file.variable(name).ok_or_else(||format!("missing field {name}"))?;
        let dims: Vec<String>=var.dimensions().iter().map(|d|d.name().into()).collect();
        let shape=var.shape();
        let mut raw=vec![];
        if name == "iLev_DT" {
            for v in file.read_array::<i32>(name)?.iter() { raw.extend_from_slice(&v.to_le_bytes()); }
        } else {
            for v in file.read_array::<f32>(name)?.iter() { raw.extend_from_slice(&v.to_bits().to_le_bytes()); }
        }
        let sha=format!("{:x}",Sha256::digest(&raw));
        let mut out=std::fs::File::create(dir.join(format!("{name}.raw")))?;out.write_all(&raw)?;
        records.push(serde_json::json!({"field":name,"dimensions":dims,"shape":shape,"words":raw.len()/4,"sha256":sha}));
    }
    std::fs::write(dir.join("fields.json"),serde_json::to_vec_pretty(&records)?)?;
    Ok(())
}
