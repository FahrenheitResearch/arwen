//! Synthetic full static-driver inputs, encoded by Rust. Invoke with an owned
//! output geography directory and the regenerated native climatology directory.
//! MPAS: mpas_init_atm_static.F:369-1250,1559-2182 (input layouts).
use std::{fs,path::PathBuf};
fn main() {
    let args:Vec<_>=std::env::args_os().skip(1).collect();
    assert!((2..=4).contains(&args.len()),"static_oracle_fixture GEOG CLIMATOLOGY [MISSING_RKIND [INDEX_TRAILER]]");
    let missing = args.get(2).map(|value| value.to_str().unwrap()).unwrap_or("255");
    missing.parse::<f32>().expect("missing value must be RKIND real metadata");
    let root=PathBuf::from(&args[0]);let climate=PathBuf::from(&args[1]);
    assert!(!root.exists(),"fixture root must be new");
    fs::create_dir_all(&root).unwrap();
    for (name,nz,categorical,scale) in [
        ("topo_gmted2010_30s",1,false,"0.25"),
        ("topo_30s",1,false,"0.25"),
        ("modis_landuse_20class_30s",1,true,"1"),
        ("modis_landuse_20class_15s",1,true,"1"),
        ("landuse_30s",1,true,"1"),
        ("soiltype_top_30s",1,true,"1"),
        ("bnu_soiltype_top",1,true,"1"),
        ("greenfrac_fpar_modis",12,false,"0.01"),
        ("albedo_modis",12,false,"0.1"),
        ("maxsnowalb_modis",1,false,"0.25"),
        ("soilgrids/soilcomp",8,false,"0.125"),
        ("soilgrids/texture_layer1",1,true,"1"),
        ("soilgrids/texture_layer2",1,true,"1"),
        ("soilgrids/texture_layer3",1,true,"1"),
        ("soilgrids/texture_layer4",1,true,"1"),
    ] {
        let directory=root.join(name);fs::create_dir_all(&directory).unwrap();
        let (classes,water,ice,label)=if name=="landuse_30s" {(24,16,24,"USGS")} else {(21,17,15,"MODIFIED_IGBP_MODIS_NOAH")};
        let index=format!("type={}\nprojection=regular_ll\ndx=45\ndy=45\nknown_x=1\nknown_y=1\nknown_lat=-67.5\nknown_lon=-157.5\ntile_x=8\ntile_y=4\ntile_z={nz}\ntile_bdr=1\nwordsize=2\nsigned=no\nscale_factor={scale}\nmissing_value={missing}\ncategory_min=1\ncategory_max={classes}\niswater={water}\nisice={ice}\nisoilwater=14\nmminlu={label}\n",if categorical {"categorical"} else {"continuous"});
        fs::write(directory.join("index"),index).unwrap();
        if let Some(trailer) = args.get(3) {
            use std::io::Write;
            let mut index = fs::OpenOptions::new().append(true).open(directory.join("index")).unwrap();
            writeln!(index,"{}",trailer.to_str().unwrap()).unwrap();
        }
        let mut bytes=Vec::new();
        for z in 0..nz {for j in 0..6 {for i in 0..10 {
            let p=j*10+i;
            let value:u16=if categorical {
                if p%11==0 {255} else {((p*5)%classes+1) as u16}
            } else if (p+z)%7==0 {255} else {((p*13+z*17)%101) as u16};
            bytes.extend_from_slice(&value.to_be_bytes());
        }}}
        fs::write(directory.join("00001-00008.00001-00004"),bytes).unwrap();
    }
    for name in ["soiltemp_1deg","greenfrac","albedo_ncep","maxsnowalb"] {
        let directory=root.join(name);fs::create_dir_all(&directory).unwrap();
        for entry in fs::read_dir(climate.join(name)).unwrap() {
            let entry=entry.unwrap();if entry.file_type().unwrap().is_file() {fs::copy(entry.path(),directory.join(entry.file_name())).unwrap();}
        }
    }
}
