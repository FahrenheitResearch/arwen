//! Analytic winter presentation fixture. No forecast skill is implied.
use std::{fs, path::{Path, PathBuf}, process::Command};
use netcdf_writer::{AttrValue, NcFormat, NcType, NcWriter, Schema, VarData};

fn frame(dir: &Path, lead: i64) -> PathBuf {
    const NX: usize = 96;
    const NY: usize = 72;
    const NZ: usize = 4;
    let cells = NX * NY;
    let init = chrono::NaiveDateTime::parse_from_str("2026-10-08_18:00:00", "%Y-%m-%d_%H:%M:%S").unwrap();
    let stamp = (init + chrono::Duration::hours(lead)).format("%Y-%m-%d_%H:%M:%S").to_string();
    let mut schema = Schema::new(NcFormat::Offset64);
    let time = schema.def_dim("Time", 0, true).unwrap();
    let strlen = schema.def_dim("DateStrLen", 19, false).unwrap();
    let z = schema.def_dim("bottom_top", NZ, false).unwrap();
    let zs = schema.def_dim("bottom_top_stag", NZ + 1, false).unwrap();
    let y = schema.def_dim("south_north", NY, false).unwrap();
    let x = schema.def_dim("west_east", NX, false).unwrap();
    for name in ["START_DATE", "SIMULATION_START_DATE"] {
        schema.put_global_attr(name, AttrValue::Text("2026-10-08_18:00:00".into())).unwrap();
    }
    for (name, value) in [("MAP_PROJ",6), ("GRID_ID",2)] {
        schema.put_global_attr(name, AttrValue::Ints(vec![value])).unwrap();
    }
    for name in ["DX", "DY"] { schema.put_global_attr(name, AttrValue::Floats(vec![750.])).unwrap(); }
    let times = schema.def_var("Times", NcType::Char, &[time,strlen]).unwrap();
    let mut planes: Vec<(usize, Vec<f32>)> = Vec::new();
    let envelope: Vec<f32> = (0..cells).map(|i| {
        let a = (i % NX) as f32 / NX as f32;
        let b = (i / NX) as f32 / NY as f32;
        (-(a-0.45).powi(2)*8. - (b-0.55).powi(2)*5.).exp()
    }).collect();
    for (name, units, values) in [
        ("XLAT","degree_north",(0..cells).map(|i|47.05+(i/NX) as f32*0.00675).collect()),
        ("XLONG","degree_east",(0..cells).map(|i|-112.05+(i%NX) as f32*0.0098).collect()),
        ("HGT","m",vec![1000.;cells]),
        ("PSFC","Pa",vec![90000.;cells]),
        ("T2","K",(0..cells).map(|i|263.+14.*(i%NX) as f32/NX as f32).collect()),
        ("SNOWNC","mm",envelope.iter().map(|v|v*lead as f32*0.25).collect()),
        ("GRAUPELNC","mm",envelope.iter().map(|v|v*lead as f32*0.01).collect()),
        ("SNOWFALLAC","mm",envelope.iter().map(|v|v*lead as f32*3.).collect()),
    ] {
        let id = schema.def_var(name,NcType::Float,&[time,y,x]).unwrap();
        schema.put_var_attr(id,"units",AttrValue::Text(units.into())).unwrap();
        planes.push((id,values));
    }
    let pressures = [85000_f64,70000.,50000.,30000.];
    for name in ["T","P","PB","QVAPOR","QCLOUD","QRAIN","QSNOW","QGRAUP"] {
        let id = schema.def_var(name,NcType::Float,&[time,z,y,x]).unwrap();
        let values = (0..NZ*cells).map(|n| {
            let k=n/cells; let i=n%cells;
            match name {
                "T" => ((268.-k as f64*8.)/(pressures[k]/100000.).powf(0.2857142857)-300.) as f32,
                "PB" => pressures[k] as f32, "P" => 0.,
                "QVAPOR" => 0.002*envelope[i],
                "QCLOUD" => 0.0003*envelope[i],
                "QRAIN" => if i%NX>NX/2 {0.001*envelope[i]} else {0.0001*envelope[i]},
                "QSNOW" => if i%NX>NX/2 {0.0001*envelope[i]} else {0.001*envelope[i]},
                _ => 0.,
            }
        }).collect();
        planes.push((id,values));
    }
    for name in ["PH","PHB"] {
        let id=schema.def_var(name,NcType::Float,&[time,zs,y,x]).unwrap();
        let values=(0..(NZ+1)*cells).map(|n| {
            if name=="PH" {0.} else {(1000.+[0.,1000.,3000.,6000.,10000.][n/cells])*9.81}
        }).collect();
        planes.push((id,values));
    }
    let path=dir.join(format!("wrfout_d02_{}",stamp.replace(':',"_")));
    let mut writer=NcWriter::create(&path,schema).unwrap();
    writer.write_record(0,times,VarData::Char(stamp.as_bytes())).unwrap();
    for (id,values) in planes { writer.write_record(0,id,VarData::F32(&values)).unwrap(); }
    writer.finish().unwrap();
    path
}

#[test]
fn native_default_source_label_uses_public_name_and_declared_version() {
    let retained=std::env::var_os("RW_LABEL_FIXTURE_DIR").map(PathBuf::from);
    let root=retained.clone().unwrap_or_else(||std::env::temp_dir().join(format!("wrf-labels-{}",std::process::id())));
    fs::create_dir_all(&root).unwrap();
    for lead in [0,72] { frame(&root,lead); }
    let end=frame(&root,96);
    let expected_title=root.join("expected-title.json");
    fs::write(&expected_title, r#"{"title":"24 h Kuchera snowfall ending Mon 12 Oct 18Z (d02 750 m)"}"#).unwrap();
    let mut pixels=Vec::new();
    for (arm,label) in [("default",None),("public",Some(concat!("WOOF ",env!("GPUWM_ENGINE_VERSION")))),("expected-title",None)] {
        let out=root.join(arm);
        let mut cmd=Command::new(env!("CARGO_BIN_EXE_rw_wrfbatch"));
        cmd.args(["--products","snow_kuchera_window","--layout","fixed","--width","1000","--height","800"])
            .arg("--store-root").arg(root.join("store")).arg("--out-dir").arg(&out)
            .env("GPUWM_SNOW_SINCE","2026-10-11_18:00:00").env_remove("RUSTWX_THEME");
        if let Some(label)=label {cmd.args(["--source-label",label]);}
        if arm == "expected-title" {cmd.arg("--annotate").arg(&expected_title);}
        let output=cmd.arg(&end).output().unwrap();
        assert!(output.status.success(),"{}\n{}",String::from_utf8_lossy(&output.stdout),String::from_utf8_lossy(&output.stderr));
        let png=fs::read_dir(&out).unwrap().filter_map(Result::ok).find(|e|e.path().extension().is_some_and(|s|s=="png")).unwrap().path();
        pixels.push(fs::read(png).unwrap());
    }
    assert_eq!(pixels[0],pixels[1],"bare native render must stamp WOOF and the actual engine version");
    assert!(pixels[0] == pixels[2], "bare whole-hour render must title its resolved 24 h window before the domain label");
    if retained.is_none() {fs::remove_dir_all(root).unwrap();}
}
