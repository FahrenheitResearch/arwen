// SPDX-License-Identifier: Apache-2.0
// Independent diagnostic expectations from sealed engineering terrain outputs.
use nextgen_replay::{file_sha256, sealed_bytes, Seal};
use rayon::prelude::*;
use serde::Deserialize;
use serde_json::json;
use std::{fs, path::PathBuf, time::Instant};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Input {
    shape: [usize; 2],
    initial_depth: Seal,
    final_depth: Seal,
    active: Seal,
    terrain_receipt: Seal,
    dem_manifest: Seal,
    source_identity: serde_json::Value,
    #[serde(default)]
    history: Option<History>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct History { frames: usize, depth: Seal }

fn main() -> Result<(), String> {
    let start = Instant::now();
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 3 { return Err("terrain_depth_fixture INPUT.json NEW_DIRECTORY".into()); }
    let input: Input = serde_json::from_slice(&fs::read(&args[1]).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;
    let count = input.shape[0].checked_mul(input.shape[1]).filter(|n|*n>0).ok_or("invalid shape")?;
    let bytes = count.checked_mul(4).ok_or("shape overflow")?;
    let (a,(b,mask)) = rayon::join(||sealed_bytes(&input.initial_depth), ||rayon::join(||sealed_bytes(&input.final_depth),||sealed_bytes(&input.active)));
    let (a,b,mask)=(a?,b?,mask?);
    if a.len()!=bytes || b.len()!=bytes || mask.len()!=bytes { return Err("raw shape/size mismatch".into()); }
    let history = input.history.as_ref().map(|h| {
        let size = bytes.checked_mul(h.frames).filter(|_|h.frames>0).ok_or("invalid history shape")?;
        let raw = sealed_bytes(&h.depth)?;
        if raw.len()!=size || raw[size-bytes..]!=b { return Err("history size or final frame mismatch".into()); }
        Ok::<_,String>((raw,h.frames))
    }).transpose()?;
    let receipt: serde_json::Value=serde_json::from_slice(&sealed_bytes(&input.terrain_receipt)?).map_err(|e|e.to_string())?;
    sealed_bytes(&input.dem_manifest)?;
    if receipt["checks"]["health"].as_u64()!=Some(0) { return Err("source receipt must report healthy engineering output".into()); }
    let expected: Result<Vec<_>,String>=(0..count).into_par_iter().map(|p| {
        let at=p*4;
        let m=i32::from_le_bytes(mask[at..at+4].try_into().unwrap());
        if m==0 { return Ok((0x7fc00000u32,0x7fc00000u32)); }
        if m!=1 { return Err(format!("invalid active mask at {p}")); }
        let x=f32::from_le_bytes(a[at..at+4].try_into().unwrap());
        let y=f32::from_le_bytes(b[at..at+4].try_into().unwrap());
        if !x.is_finite() || !y.is_finite() || x<0. || y<0. { return Err(format!("invalid engineering depth at {p}")); }
        let mut maximum=x.max(y);
        if let Some((raw,frames))=&history {
            for frame in 0..*frames {
                let offset=frame*bytes+at;
                let value=f32::from_le_bytes(raw[offset..offset+4].try_into().unwrap());
                if !value.is_finite() || value<0. { return Err(format!("invalid history depth at {frame}/{p}")); }
                maximum=maximum.max(value);
            }
        }
        Ok((y.to_bits(),maximum.to_bits()))
    }).collect();
    let expected=expected?;
    let root=PathBuf::from(&args[2]);fs::create_dir(&root).map_err(|e|e.to_string())?;
    let mut arrays=serde_json::Map::new();
    for (name,seal,dtype,shape) in [
        ("initial_depth",&input.initial_depth,"f32",input.shape.to_vec()),
        ("final_depth",&input.final_depth,"f32",input.shape.to_vec()),
        ("active",&input.active,"i32",input.shape.to_vec()),
    ] { arrays.insert(name.into(),json!({"path":seal.path,"sha256":seal.sha256,"dtype":dtype,"shape":shape,"order":"c"})); }
    for (name,raw,dtype,shape) in [
        ("expected_depth",expected.iter().flat_map(|v|v.0.to_le_bytes()).collect::<Vec<_>>(),"f32",input.shape.to_vec()),
        ("expected_maximum",expected.iter().flat_map(|v|v.1.to_le_bytes()).collect::<Vec<_>>(),"f32",input.shape.to_vec()),
        ("healthy",0u32.to_le_bytes().to_vec(),"u8",vec![4]),
    ] {
        let path=root.join(format!("{name}.raw"));fs::write(&path,raw).map_err(|e|e.to_string())?;
        arrays.insert(name.into(),json!({"path":path,"sha256":file_sha256(&path)?,"dtype":dtype,"shape":shape,"order":"c"}));
    }
    let mut evidence=vec![input.terrain_receipt,input.dem_manifest];
    if let Some(h)=&input.history { evidence.push(h.depth.clone()); }
    let result=json!({"schema":"nextgen.inundation.engineering-fixture/v1","origin":"engineering-depth-replay",
        "scope":"Public DEM and imposed-water engineering experiment; no event forcing, gauge skill, basin/channel mapping or native routing parity",
        "arrays":arrays,"source_identity":input.source_identity,
        "evidence":evidence,
        "observations":1+input.history.as_ref().map_or(1,|h|h.frames),
        "source_health":"expected zero from sealed owner receipt; runner separately records whether actual health is live or reconstructed",
        "parallelism":format!("Rust Rayon {} workers; sealed I/O and per-cell expectations",rayon::current_num_threads()),
        "prepare_wall_ms":start.elapsed().as_secs_f64()*1000.});
    fs::write(root.join("fixture.json"),serde_json::to_vec_pretty(&result).unwrap()).map_err(|e|e.to_string())?;
    println!("{} cells; {}",count,result["parallelism"]);
    Ok(())
}
