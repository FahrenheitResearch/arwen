use std::collections::BTreeMap;
use std::error::Error;
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};

use grib_core::grib2::{grid_latlon, unpack_message, Grib2File, GridDefinition};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

mod checkpoint;

type Result<T> = std::result::Result<T, Box<dyn Error>>;

fn frames(path: &Path) -> Result<Vec<PathBuf>> {
    let dir = path.join("out/wrfout");
    let mut files: Vec<_> = fs::read_dir(&dir)?.filter_map(|e| e.ok()).map(|e| e.path())
        .filter(|p| p.file_name().unwrap().to_string_lossy().starts_with("wrfout_d01_")).collect();
    files.sort();
    if files.len() != 3 { return Err(format!("{} has {} frames, expected 3", dir.display(), files.len()).into()); }
    Ok(files)
}

fn plane(file: &netcrust::File, name: &str, count: usize) -> Result<Vec<f64>> {
    let data = file.read_array_f64_first_record_or_all(name)?.into_values();
    if data.len() != count { return Err(format!("{name} has {} values, expected {count}", data.len()).into()); }
    Ok(data)
}

fn checkpoints(path: &Path) -> Result<Vec<PathBuf>> {
    fn walk(path:&Path, files:&mut Vec<PathBuf>) -> Result<()> {
        for entry in fs::read_dir(path)? {
            let path=entry?.path();
            if path.is_dir() {walk(&path,files)?;}
            else if path.file_name().unwrap().to_string_lossy().starts_with("gpuwmrst_d01_") && path.extension().and_then(|x|x.to_str())==Some("npz") {files.push(path);}
        }
        Ok(())
    }
    let mut files=Vec::new();walk(&path.join("out"),&mut files)?;files.sort();
    if files.len()!=2{return Err(format!("{} has {} checkpoints, expected 2",path.display(),files.len()).into());}
    Ok(files)
}

fn checkpoint_name_matches(name:&str, expected_stem:&str) -> bool {
    let Some(stem)=name.strip_suffix(".npz") else{return false;};
    if stem==expected_stem{return true;}
    let Some(nonce)=stem.strip_prefix(expected_stem).and_then(|suffix|suffix.strip_prefix("__")) else{return false;};
    nonce.len()==32 && nonce.bytes().all(|byte|byte.is_ascii_hexdigit())
}

fn checkpoint_time(path:&Path, cycle:usize, hour:usize) -> Result<Value> {
    let expected=format!("gpuwmrst_d01_2026-10-02_{:02}_00_00",cycle+hour);
    if !path.file_name().and_then(|x|x.to_str()).is_some_and(|name|checkpoint_name_matches(name,&expected)){return Err(format!("checkpoint path does not carry expected valid time: {}",path.display()).into());}
    let mut archive=zip::ZipArchive::new(fs::File::open(path)?)?;
    let mut entry=archive.by_name("__gpuwm_restart_header__.npy")?;
    if entry.size()>4*1024*1024{return Err("checkpoint metadata exceeds bounded header size".into());}
    let mut bytes=Vec::new();entry.read_to_end(&mut bytes)?;
    if bytes.get(..6)!=Some(b"\x93NUMPY"){return Err("checkpoint metadata is not NPY".into());}
    let start=match bytes.get(6){Some(1)=>10,Some(2)|Some(3)=>12,_=>return Err("unsupported metadata NPY version".into())};
    let size=if start==10{u16::from_le_bytes(bytes.get(8..10).ok_or("truncated metadata NPY length")?.try_into()?) as usize}
        else{u32::from_le_bytes(bytes.get(8..12).ok_or("truncated metadata NPY length")?.try_into()?) as usize};
    let header=std::str::from_utf8(bytes.get(start..start+size).ok_or("truncated metadata NPY header")?)?;
    if !header.contains("'|u1'"){return Err("checkpoint metadata is not its declared byte array".into());}
    let metadata:Value=serde_json::from_slice(bytes.get(start+size..).ok_or("truncated checkpoint JSON")?)?;
    let elapsed=metadata["elapsed_seconds"].as_f64().ok_or("checkpoint elapsed time is missing")?;
    if elapsed!=hour as f64*3600.0{return Err(format!("checkpoint elapsed {elapsed} differs from hour {hour}").into());}
    Ok(json!({"path":path,"elapsed_seconds":elapsed,"expected_filename_stem":expected,"optional_tree_nonce":"32 hexadecimal characters"}))
}

#[cfg(test)]
mod tests {
    use super::checkpoint_name_matches;

    #[test]
    fn checkpoint_time_stem_and_optional_tree_nonce_are_strict() {
        let stem="gpuwmrst_d01_2026-10-02_22_00_00";
        assert!(checkpoint_name_matches(&format!("{stem}.npz"),stem));
        assert!(checkpoint_name_matches(&format!("{stem}__0123456789abcdef0123456789abcdef.npz"),stem));
        assert!(!checkpoint_name_matches("gpuwmrst_d01_2026-10-02_23_00_00__0123456789abcdef0123456789abcdef.npz",stem));
        assert!(!checkpoint_name_matches(&format!("{stem}__0123456789abcdef0123456789abcdeg.npz"),stem));
        assert!(!checkpoint_name_matches(&format!("{stem}__0123456789abcdef.npz"),stem));
        assert!(!checkpoint_name_matches(&format!("{stem}__0123456789abcdef0123456789abcdef0.npz"),stem));
    }
}

fn history_time(file:&netcrust::File, path:&Path, cycle:usize, hour:usize) -> Result<()> {
    let expected=format!("2026-10-02_{:02}:00:00",cycle+hour);
    if file.variable("Times").is_none(){return Err(format!("{} has no Times field",path.display()).into());}
    let times=file.read_strings("Times")?;
    if times.len()!=1 || times[0].trim_matches(['\0',' '])!=expected{return Err(format!("{} has Times {times:?}, expected {expected}",path.display()).into());}
    if file.variable("XTIME").is_some(){
        let xtime=file.read_f64_first_record_or_all("XTIME")?;
        if xtime!=vec![hour as f64*60.0]{return Err(format!("{} has XTIME {xtime:?}, expected {} minutes",path.display(),hour*60).into());}
    }
    Ok(())
}

fn read_grib(path: &Path, selector: (u8,u8,u8,u8,f64), hour: usize, cycle:usize) -> Result<(Vec<f64>,GridDefinition,Value)> {
    let file = Grib2File::open(&path.to_string_lossy())?;
    if file.messages.len() != 1 { return Err(format!("{} contains {} messages", path.display(), file.messages.len()).into()); }
    let message = &file.messages[0];
    let actual = (message.discipline, message.product.parameter_category, message.product.parameter_number,
                  message.product.level_type, message.product.level_value);
    if actual != selector || message.product.template != 0 || message.product.forecast_time != hour as u32 {
        return Err(format!("{} has selector {actual:?}, PDT {}, lead {}", path.display(), message.product.template, message.product.forecast_time).into());
    }
    let expected_reference=format!("2026-10-02 {cycle:02}:00:00");
    if message.reference_time.to_string()!=expected_reference || message.product.time_range_unit!=1{
        return Err(format!("{} has reference time {} / forecast unit {}, expected {expected_reference} / hours",path.display(),message.reference_time,message.product.time_range_unit).into());
    }
    let metadata = json!({"path":path, "selector":[actual.0,actual.1,actual.2,actual.3,actual.4],
        "pdt":message.product.template, "forecast_hour":message.product.forecast_time,
        "grid":[message.grid.nx,message.grid.ny], "scan_mode":message.grid.scan_mode,"reference_time":message.reference_time.to_string(),"forecast_unit":message.product.time_range_unit});
    Ok((unpack_message(message)?,message.grid.clone(),metadata))
}

fn stats(values: &[f64], mask: &[bool]) -> Result<Value> {
    let mut selected: Vec<f64> = values.iter().zip(mask).filter_map(|(&x,&m)|m.then_some(x)).collect();
    if selected.iter().any(|x| !x.is_finite()) { return Err("nonfinite scored field".into()); }
    if selected.is_empty() { return Ok(json!({"cells":0})); }
    let mean = selected.iter().sum::<f64>() / selected.len() as f64;
    selected.sort_by(f64::total_cmp);
    let quantile = |p:f64| selected[((selected.len()-1) as f64*p).round() as usize];
    Ok(json!({"cells":selected.len(),"mean":mean,"min":selected[0],"max":selected[selected.len()-1],
              "p10":quantile(0.1),"p50":quantile(0.5),"p90":quantile(0.9)}))
}

fn comparison(left: &[f64], right: &[f64], mask: &[bool]) -> Result<Value> {
    let delta: Vec<_> = left.iter().zip(right).map(|(a,b)|a-b).collect();
    let mut s=stats(&delta,mask)?;
    let n=mask.iter().filter(|&&m|m).count();
    if n>0 { s["rmse"]=json!((delta.iter().zip(mask).filter_map(|(&v,&m)|m.then_some(v*v)).sum::<f64>()/n as f64).sqrt()); }
    Ok(s)
}

fn dewpoint(q: &[f64], p: &[f64]) -> Vec<f64> {
    q.iter().zip(p).map(|(q,p)| {
        let e=q*p/(0.622+q);
        let x=(e.max(1e-3)/611.2).ln();
        273.15+243.5*x/(17.67-x)
    }).collect()
}

fn digest_frame(path: &Path) -> Result<Value> {
    let mut source=fs::File::open(path)?;
    let mut file_hash=Sha256::new();
    let mut block=vec![0_u8;1024*1024];
    loop { let count=source.read(&mut block)?;if count==0{break;}file_hash.update(&block[..count]); }
    let f=netcrust::File::open(path)?;
    let mut vars=f.variables()?;
    vars.sort_by(|a,b|a.name().cmp(b.name()));
    let mut total=Sha256::new();
    let mut fields=BTreeMap::new();
    for var in vars {
        if matches!(var.dtype(),netcrust::DataType::Char | netcrust::DataType::String) { continue; }
        let values=f.read_array_f64_first_record_or_all(var.name())?;
        let mut hash=Sha256::new();
        hash.update(var.name().as_bytes());
        hash.update(format!("{:?}{:?}",var.dtype(),values.shape()).as_bytes());
        for value in values.into_values() { hash.update(value.to_le_bytes()); }
        let hex=format!("{:x}",hash.finalize());
        total.update(var.name().as_bytes());total.update(hex.as_bytes());
        fields.insert(var.name().to_string(),hex);
    }
    Ok(json!({"path":path,"file_sha256":format!("{:x}",file_hash.finalize()),"numeric_variables":fields.len(),"digest":format!("{:x}",total.finalize()),"fields":fields,
              "digest_encoding":"name, dtype, shape, promoted exact f64 numeric words; character Times excluded"}))
}

fn main() -> Result<()> {
    let args:Vec<String>=std::env::args().collect();
    if args.len()<6 { return Err("usage: solar-albedo-score RAW_ROOT CYCLE_HOUR OUTPUT_JSON ARM=RUNDIR [ARM=RUNDIR...]".into()); }
    let raw=PathBuf::from(&args[1]);let cycle:usize=args[2].parse()?;
    let arms:BTreeMap<String,PathBuf>=args[4..].iter().map(|a| {
        let (key,path)=a.split_once('=').expect("ARM=RUNDIR");(key.to_string(),PathBuf::from(path))
    }).collect();
    let mut arm_frames=BTreeMap::new();
    for (arm,path) in &arms { arm_frames.insert(arm.clone(),frames(path)?); }
    let mut arm_checkpoints=BTreeMap::new();
    for (arm,path) in &arms {arm_checkpoints.insert(arm.clone(),checkpoints(path)?);}
    let first=netcrust::File::open(&arm_frames["off"][0])?;
    let lat=first.read_array_f64_first_record_or_all("XLAT")?;
    let shape=lat.shape();let ny=shape[shape.len()-2];let nx=shape[shape.len()-1];let count=nx*ny;
    let lat=lat.into_values();let lon=plane(&first,"XLONG",count)?;
    let land=plane(&first,"LANDMASK",count)?;
    let (hland,grid,_)=read_grib(&raw.join(format!("{cycle:02}z/hrrr.t{cycle:02}z.f00.land.grib2")),(2,0,0,1,0.0),0,cycle)?;
    let (hlat,hlon)=grid_latlon(&grid)?;let hnx=grid.nx as usize;let hny=grid.ny as usize;
    let normalize=|x:f64| if x>180.0{x-360.0}else{x};
    let (anchor,_)=(0..hlat.len()).map(|i|(i,(hlat[i]-lat[0]).abs()+(normalize(hlon[i])-lon[0]).abs()))
        .min_by(|a,b|a.1.total_cmp(&b.1)).ok_or("empty HRRR grid")?;
    let aj=anchor/hnx;let ai=anchor%hnx;let mut best=(f64::INFINITY,0,0);
    for j in aj.saturating_sub(1)..=(aj+1).min(hny-ny) {
        for i in ai.saturating_sub(1)..=(ai+1).min(hnx-nx) {
            let mut worst:f64=0.0;
            for k in 0..count { let h=(j+k/nx)*hnx+i+k%nx;worst=worst.max((hlat[h]-lat[k]).abs()).max((normalize(hlon[h])-lon[k]).abs()); }
            if worst<best.0 { best=(worst,j,i); }
        }
    }
    // The recorded cut's WPS sphere and HRRR sphere differ by about
    // 0.0024 degrees in its prior registration receipt. Keep the actual
    // discrepancy visible, while rejecting an offset approaching a cell.
    if best.0>0.005 { return Err(format!("HRRR aligned-grid coordinate discrepancy {} degrees",best.0).into()); }
    let crop=|v:Vec<f64>| -> Vec<f64> {(0..count).map(|k|v[(best.1+k/nx)*hnx+best.2+k%nx]).collect()};
    let hland=crop(hland);
    let mask:Vec<_>=(0..count).map(|k|k/nx>=10&&k/nx<ny-10&&k%nx>=10&&k%nx<nx-10&&land[k]>0.5&&hland[k]>0.5).collect();
    let mut result=json!({"schema":"solar-albedo-cut-score-v1","cycle":format!("2026-10-02T{cycle:02}:00:00Z"),
       "grid":{"nx":nx,"ny":ny,"hrrr_j_i":[best.1,best.2],"max_coordinate_difference_deg":best.0,"boundary_dropped":10,
               "common_interior_land_cells":mask.iter().filter(|&&x|x).count()},
       "net_surface_sw_reference":"instantaneous HRRR DSWRF minus USWRF","hours":{},"frame_digests":{}});
    let specifications=[("tsk",(0,0,0,1,0.0)),("t2",(0,0,0,103,2.0)),("td2",(0,0,6,103,2.0)),
        ("lh",(0,0,10,1,0.0)),("hfx",(0,0,11,1,0.0)),("swdown",(0,4,7,1,0.0)),("swup",(0,4,8,1,0.0))];
    for hour in 0..=2 {
        let mut hr=BTreeMap::new();let mut metadata=Vec::new();
        for (name,selector) in specifications {
            let (v,current_grid,m)=read_grib(&raw.join(format!("{cycle:02}z/hrrr.t{cycle:02}z.f{hour:02}.{name}.grib2")),selector,hour,cycle)?;
            if current_grid!=grid{return Err(format!("HRRR {name} f{hour:02} grid differs from reference LAND grid").into());}
            hr.insert(name.to_string(),crop(v));metadata.push(m);
        }
        hr.insert("gsw".to_string(),hr["swdown"].iter().zip(&hr["swup"]).map(|(d,u)|d-u).collect());
        let mut row=json!({"hrrr":{},"selectors":metadata});
        for (name,v) in &hr { row["hrrr"][name]=stats(v,&mask)?; }
        let mut arm_values=BTreeMap::new();
        for (arm,files) in &arm_frames {
            let f=netcrust::File::open(&files[hour])?;
            history_time(&f,&files[hour],cycle,hour)?;
            for (name,reference) in [("XLAT",&lat),("XLONG",&lon),("LANDMASK",&land)]{
                if plane(&f,name,count)?!=*reference{return Err(format!("{arm} f{hour:02} {name} differs from OFF initial geometry/mask").into());}
            }
            let checkpoint_time_receipt=if hour>0{checkpoint_time(&arm_checkpoints[arm][hour-1],cycle,hour)?}else{Value::Null};
            let mut fields=BTreeMap::new();
            let mut provenance=BTreeMap::new();
            for (label,variable,checkpoint_name) in [("tsk","TSK","tsk"),("t2","T2","t2"),("lh","LH","lh"),("hfx","HFX","hfx"),("gsw","GSW","gsw"),("swdown","SWDOWN","swdown"),("albedo","ALBEDO","albedo"),("coszen","COSZEN","coszen")] {
                if f.variable(variable).is_some() {
                    fields.insert(label.to_string(),plane(&f,variable,count)?);
                    provenance.insert(label.to_string(),json!({"source":"history","path":files[hour],"variable":variable}));
                } else if hour>0 {
                    let checkpoint=&arm_checkpoints[arm][hour-1];
                    let selected=checkpoint::read_checkpoint_plane(checkpoint,checkpoint_name,nx,ny)?;
                    fields.insert(label.to_string(),selected.values);
                    provenance.insert(label.to_string(),json!({"source":"hourly checkpoint","path":checkpoint,"array":selected.array}));
                } else {
                    provenance.insert(label.to_string(),json!({"source":null,"reason":"history field absent and no initial checkpoint; no value inferred"}));
                }
            }
            fields.insert("td2".to_string(),dewpoint(&plane(&f,"Q2",count)?,&plane(&f,"PSFC",count)?));
            let low:Vec<_>=if let Some(cos)=fields.get("coszen"){mask.iter().zip(cos).map(|(&m,&s)|m&&s>0.0&&s<=0.5).collect()}else{vec![false;count]};
            let day:Vec<_>=if let Some(cos)=fields.get("coszen"){mask.iter().zip(cos).map(|(&m,&s)|m&&s>0.0).collect()}else{vec![false;count]};
            let mut detail=json!({"fields":{},"minus_hrrr":{},"low_sun_coszen_0_to_0p5":{},"daylight":{},"field_provenance":provenance,"checkpoint_time":checkpoint_time_receipt});
            for (name,v) in &fields {
                detail["fields"][name]=stats(v,&mask)?;
                if let Some(h)=hr.get(name) { detail["minus_hrrr"][name]=comparison(v,h,&mask)?; }
                detail["low_sun_coszen_0_to_0p5"][name]=stats(v,&low)?;
                detail["daylight"][name]=stats(v,&day)?;
            }
            row[arm]=detail;
            arm_values.insert(arm.clone(),fields);
            result["frame_digests"][arm][format!("f{hour:02}")]=digest_frame(&files[hour])?;
        }
        row["on_minus_off"]=json!({});
        for (name,v) in &arm_values["on"] {
            if let Some(off)=arm_values["off"].get(name){row["on_minus_off"][name]=comparison(v,off,&mask)?;}
        }
        result["hours"][format!("f{hour:02}")]=row;
    }
    let mut neutrality=true;
    if arms.contains_key("base"){
        let mut proof=Vec::new();
        for hour in 0..=2{
            let key=format!("f{hour:02}");
            let base=&result["frame_digests"]["base"][&key];let off=&result["frame_digests"]["off"][&key];
            let same=base["fields"]==off["fields"];
            neutrality &= same;
            proof.push(json!({"hour":key,"all_numeric_variable_words_identical":same,"whole_file_bytes_identical":base["file_sha256"]==off["file_sha256"],"base_digest":base["digest"],"off_digest":off["digest"]}));
        }
        result["baseline_off_history_identity"]=json!(proof);
    }
    fs::write(&args[3],serde_json::to_string_pretty(&result)?+"\n")?;
    println!("{}",serde_json::to_string(&json!({"output":args[3],"grid":result["grid"],"hours":result["hours"].as_object().unwrap().keys().collect::<Vec<_>>() }))?);
    if !neutrality{return Err("baseline and OFF history field sets or numeric words differ; receipt retained".into());}
    Ok(())
}
