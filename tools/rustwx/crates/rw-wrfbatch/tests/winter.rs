use netcdf_writer::{AttrValue, NcFormat, NcType, NcWriter, Schema, VarData};
use rw_wrfbatch::wrf_process::{spawn_process_paths, WrfProcessMessage, WrfProcessOptions};
use rusty_weather::render_all::StoreFieldSource;
use rustwx_core::{CanonicalField, FieldSelector};
use std::path::{Path, PathBuf};
use std::process::Command;

const NX: usize = 36;
const NY: usize = 28;
const NZ: usize = 4;

fn fixture(dir: &Path, hour: usize) -> PathBuf {
    let mut s = Schema::new(NcFormat::Offset64);
    let t = s.def_dim("Time", 0, true).unwrap();
    let strlen = s.def_dim("DateStrLen", 19, false).unwrap();
    let z = s.def_dim("bottom_top", NZ, false).unwrap();
    let y = s.def_dim("south_north", NY, false).unwrap();
    let x = s.def_dim("west_east", NX, false).unwrap();
    let zs = s.def_dim("bottom_top_stag", NZ + 1, false).unwrap();
    let ys = s.def_dim("south_north_stag", NY + 1, false).unwrap();
    let xs = s.def_dim("west_east_stag", NX + 1, false).unwrap();
    for name in ["START_DATE", "SIMULATION_START_DATE"] {
        s.put_global_attr(name, AttrValue::Text("2026-01-01_00:00:00".into())).unwrap();
    }
    s.put_global_attr("MAP_PROJ", AttrValue::Ints(vec![6])).unwrap();
    s.put_global_attr("GRID_ID", AttrValue::Ints(vec![1])).unwrap();
    s.put_global_attr("DX", AttrValue::Floats(vec![16000.])).unwrap();
    let times = s.def_var("Times", NcType::Char, &[t, strlen]).unwrap();
    let mut arrays = Vec::new();
    let mut add = |name: &str, dims: &[usize], values: Vec<f32>, units: &str| {
        let id = s.def_var(name, NcType::Float, dims).unwrap();
        s.put_var_attr(id, "units", AttrValue::Text(units.into())).unwrap();
        arrays.push((id, values));
    };
    for (name, value, units) in [
        ("HGT", 1000., "m"), ("PBLH", 500., "m"),
        ("T2", 270., "K"), ("PSFC", 90000., "Pa"),
        ("U10", 5., "m/s"), ("V10", 0., "m/s"),
        ("SNOWNC", hour as f32, "mm"), ("GRAUPELNC", 0., "mm"),
        ("SNOWFALLAC", 10. * hour as f32, "mm"),
    ] { add(name, &[t,y,x], vec![value; NX*NY], units); }
    add("XLAT", &[t,y,x], (0..NY).flat_map(|j| vec![42. + j as f32 * 0.2; NX]).collect(), "degree_north");
    add("XLONG", &[t,y,x], (0..NY).flat_map(|_| (0..NX).map(|i| -116. + i as f32 * 0.2)).collect(), "degree_east");
    let pressure = [90000.0f64,80000.,65000.,50000.];
    let temperature = [270.0f64,272.,268.,260.];
    add("T", &[t,z,y,x], (0..NZ).flat_map(|k| vec![(temperature[k]/(pressure[k]/100000.).powf(0.2857142857)-300.) as f32; NX*NY]).collect(), "K");
    add("P", &[t,z,y,x], vec![0.; NX*NY*NZ], "Pa");
    add("PB", &[t,z,y,x], pressure.into_iter().flat_map(|p| vec![p as f32; NX*NY]).collect(), "Pa");
    for (name,value) in [("QVAPOR",0.001),("QRAIN",0.000001),("QSNOW",0.0002),("QGRAUP",0.)] {
        add(name, &[t,z,y,x], vec![value; NX*NY*NZ], "kg/kg");
    }
    add("U", &[t,z,y,xs], vec![20.; (NX+1)*NY*NZ], "m/s");
    add("V", &[t,z,ys,x], vec![0.; NX*(NY+1)*NZ], "m/s");
    add("PH", &[t,zs,y,x], vec![0.; NX*NY*(NZ+1)], "m2/s2");
    add("PHB", &[t,zs,y,x], [1000.,1300.,1700.,2300.,4000.].into_iter().flat_map(|h| vec![h*9.81; NX*NY]).collect(), "m2/s2");
    let label = format!("2026-01-01_{hour:02}:00:00");
    let path = dir.join(format!("wrfout_d01_{}", label.replace(':', "_")));
    let mut w = NcWriter::create(&path, s).unwrap();
    w.write_record(0, times, VarData::Char(label.as_bytes())).unwrap();
    for (id,v) in arrays { w.write_record(0,id,VarData::F32(&v)).unwrap(); }
    w.finish().unwrap();
    path
}

// Tiny actual GRIB2 envelopes, decoded by the production Rust GRIB reader.
fn grib(parameter: u8, statistic: u8, value: f32) -> Vec<u8> {
    let section=|number: u8,len: usize| {
        let mut b=vec![0u8;len];b[..4].copy_from_slice(&(len as u32).to_be_bytes());b[4]=number;b
    };
    let mut s1=section(1,21);s1[5..7].copy_from_slice(&98u16.to_be_bytes());s1[10]=35;s1[11]=1;
    s1[12..14].copy_from_slice(&2026u16.to_be_bytes());s1[14]=1;s1[15]=1;
    let mut s3=section(3,72);s3[6..10].copy_from_slice(&4u32.to_be_bytes());s3[14]=6;
    s3[30..34].copy_from_slice(&2u32.to_be_bytes());s3[34..38].copy_from_slice(&2u32.to_be_bytes());
    for (offset,v) in [(46,48i32),(50,-117),(55,42),(59,-109)] {
        let bits=(v.unsigned_abs()*1000000) | if v<0 {0x80000000} else {0};
        s3[offset..offset+4].copy_from_slice(&bits.to_be_bytes());
    }
    s3[63..67].copy_from_slice(&8000000u32.to_be_bytes());s3[67..71].copy_from_slice(&6000000u32.to_be_bytes());
    let mut s4=section(4,58);s4[7..9].copy_from_slice(&8u16.to_be_bytes());s4[9]=1;s4[10]=parameter;
    s4[17]=1;s4[22]=1;s4[28]=255;s4[34..36].copy_from_slice(&2026u16.to_be_bytes());
    s4[36]=1;s4[37]=1;s4[38]=6;s4[41]=1;s4[46]=statistic;s4[47]=2;s4[48]=1;
    s4[49..53].copy_from_slice(&6u32.to_be_bytes());s4[53]=255;
    let mut s5=section(5,12);s5[5..9].copy_from_slice(&4u32.to_be_bytes());s5[9..11].copy_from_slice(&4u16.to_be_bytes());s5[11]=1;
    let mut s6=section(6,6);s6[5]=255;
    let mut s7=section(7,21);for i in 0..4 {s7[5+4*i..9+4*i].copy_from_slice(&value.to_be_bytes());}
    let body=[s1,s3,s4,s5,s6,s7].concat();
    let mut out=b"GRIB\0\0\0\x02".to_vec();out.extend(((body.len()+20) as u64).to_be_bytes());out.extend(body);out.extend(b"7777");out
}

#[test]
fn winter_planes_and_real_renderer_binaries_produce_output() {
    let root = std::env::temp_dir().join(format!("winter-{}",std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    let _baseline = fixture(&root,0);
    let end = fixture(&root,6);
    let store = root.join("store");
    let task = spawn_process_paths(vec![end.clone()],store.clone(),WrfProcessOptions::default());
    let import = loop {
        match task.rx.recv().unwrap() {
            WrfProcessMessage::Progress(_) => {},
            WrfProcessMessage::Done(r) => break r.unwrap(),
        }
    };
    let src = StoreFieldSource::open(&store,&import.model,&import.run,6).unwrap();
    let gust = src.fetch(&FieldSelector::height_agl(CanonicalField::WindGust,10)).expect("missing winter gust plane");
    assert_eq!(gust.values.len(),NX*NY);
    assert!(gust.values.iter().all(|&v| v == 16.25));
    for (field,want) in [(CanonicalField::CategoricalRain,0.),(CanonicalField::CategoricalFreezingRain,0.),(CanonicalField::CategoricalIcePellets,0.),(CanonicalField::CategoricalSnow,1.)] {
        let plane = src.fetch(&FieldSelector::surface(field)).expect("missing categorical plane");
        assert!(plane.values.iter().all(|&v| v == want));
    }
    let level = src.generic_grid("wrf_snow_level_ft").unwrap();
    assert!(level.values.iter().all(|&v| v.is_finite() && v >= 3280.));
    for field in [CanonicalField::ModelSnowfall,CanonicalField::Snowfall10to1] {
        assert!(src.fetch(&FieldSelector::surface(field)).unwrap().values.iter().all(|&v| v == 60.));
    }
    let run = |exe: &str,args: Vec<String>| {
        let out = Command::new(exe).args(&args).output().unwrap();
        println!("COMMAND {exe} {}\n{}\n{}",args.join(" "),String::from_utf8_lossy(&out.stdout),String::from_utf8_lossy(&out.stderr));
        assert!(out.status.success());
        out
    };
    let text = |p: &Path| p.display().to_string();
    run(env!("CARGO_BIN_EXE_rw_wrfbatch"),vec!["--store-root".into(),text(&root.join("batch-store")),"--out-dir".into(),text(&root.join("pictures")),"--products".into(),"10m_wind_gusts,precipitation_type,var:wrf_snow_level_ft,snowfall_window,snow_10to1_window".into(),"--width".into(),"1000".into(),"--height".into(),"700".into(),text(&end)]);
    run(env!("CARGO_BIN_EXE_rw_ensbatch"),vec!["--store-root".into(),text(&root.join("ens-store")),"--out-dir".into(),text(&root.join("ensemble")),"--member".into(),format!("0={}",end.display()),"--member".into(),format!("1={}",end.display()),"--field".into(),"snow10_win".into(),"--products".into(),"prob".into(),"--threshold".into(),"5".into(),"--thresholds".into(),"15,25".into()]);
    let gfs=root.join("gfs");let ecmwf=root.join("ecmwf");
    std::fs::create_dir_all(&gfs).unwrap();std::fs::create_dir_all(&ecmwf).unwrap();
    std::fs::write(gfs.join("bucket.grib2"),[grib(8,1,4.),grib(195,0,0.5)].concat()).unwrap();
    std::fs::write(ecmwf.join("snow.grib2"),grib(198,1,0.002)).unwrap();
    run(option_env!("CARGO_BIN_EXE_rw_refsnow").unwrap_or("rw_refsnow-missing"),vec!["--store-root".into(),text(&root.join("ref-store")),"--out".into(),text(&root.join("reference.png")),"--start-lead".into(),"0".into(),"--end-lead".into(),"6".into(),"--woof".into(),format!("Synthetic={}",end.display()),"--gfs".into(),format!("Synthetic GFS={}",gfs.display()),"--ecmwf".into(),format!("Synthetic IFS={}",ecmwf.display()),"--columns".into(),"3".into()]);
    assert!(root.join("reference.png").is_file());
    let binary=std::fs::read(option_env!("CARGO_BIN_EXE_rw_refsnow").unwrap_or("rw_refsnow-missing")).unwrap();
    let stamp=concat!("GPUWM_BRIDGE_SOURCE_REV=",env!("GPUWM_BRIDGE_SOURCE_REV")).as_bytes();
    assert!(binary.windows(stamp.len()).any(|part| part==stamp),"reference renderer lost its source revision stamp");
    println!("WINTER_FIXTURE={}",root.display());
    // The lane removes this manifest-listed fixture after copying small receipts.
}
