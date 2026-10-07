//! Renderer control from captured static/perimeter data. ROS and heat are zero
//! controls, and this file does not contain an atmospheric model integration.
use netcdf_writer::{AttrValue,NcFormat,NcType,NcWriter,Schema,VarData};
use static_fields::{sfire::{self,FireStaticLoadRequest},projection::ProjectedGrid,types::Stagger};

fn run() -> Result<(),Box<dyn std::error::Error>> {
    let args=std::env::args().collect::<Vec<_>>();
    if args.len()!=3 {return Err("usage: fire-render-control CAPTURED_STATIC_BUNDLE OUTPUT_NETCDF".into());}
    let (fields,metadata)=sfire::load(&FireStaticLoadRequest {path:args[1].clone().into(),expected_sha256:None,expected_grid_spec:None})?;
    let observed=metadata.observed_initialization.as_ref().ok_or("Control requires a captured observed perimeter")?;
    let time=chrono::DateTime::from_timestamp_millis(observed["observed_utc_epoch_ms"].as_i64().ok_or("Observed UTC is missing")?).ok_or("Observed UTC cannot be represented")?;
    let stamp=time.format("%Y-%m-%d_%H:%M:%S").to_string();
    let spec=&metadata.grid_spec;
    let ny=(spec.e_sn-1) as usize;let nx=(spec.e_we-1) as usize;
    let [fy,fx]=metadata.fine_shape;
    let mut schema=Schema::new(NcFormat::Offset64);
    let record=schema.def_dim("Time",0,true)?;let chars=schema.def_dim("DateStrLen",19,false)?;
    let z=schema.def_dim("bottom_top",1,false)?;
    let y=schema.def_dim("south_north",ny,false)?;let x=schema.def_dim("west_east",nx,false)?;
    let fine_y=schema.def_dim("south_north_subgrid",fy,false)?;
    let fine_x=schema.def_dim("west_east_subgrid",fx,false)?;
    for (name,value) in [("TITLE","SFIRE static-data renderer control"),("START_DATE",stamp.as_str()),
        ("SIMULATION_START_DATE",stamp.as_str()),("FIRE_COORDINATE_MODE","geographic")]
        {schema.put_global_attr(name,AttrValue::Text(value.into()))?;}
    for (name,value) in [("GRID_ID",3),("MAP_PROJ",1),("SR_X",metadata.sr_x as i32),("SR_Y",metadata.sr_y as i32)]
        {schema.put_global_attr(name,AttrValue::Ints(vec![value]))?;}
    for (name,value) in [("DX",spec.dx),("DY",spec.dy),("TRUELAT1",spec.truelat1),("TRUELAT2",spec.truelat2),
        ("STAND_LON",spec.stand_lon),("CEN_LAT",spec.ref_lat),("CEN_LON",spec.ref_lon)]
        {schema.put_global_attr(name,AttrValue::Floats(vec![value as f32]))?;}
    let times=schema.def_var("Times",NcType::Char,&[record,chars])?;
    let latitude=schema.def_var("XLAT",NcType::Float,&[record,y,x])?;
    let longitude=schema.def_var("XLONG",NcType::Float,&[record,y,x])?;
    let theta=schema.def_var("T",NcType::Float,&[record,z,y,x])?;
    let mut fine=Vec::new();
    for name in ["FXLAT","FXLONG","ZSF","LFN","FIRE_AREA","ROS_FRONT","FGRNHFX"] {
        fine.push((name,schema.def_var(name,NcType::Float,&[record,fine_y,fine_x])?));
    }
    let grid=ProjectedGrid::new(spec.clone())?;
    let (lat,lon)=grid.latlon(Stagger::Mass);
    let lat=lat.data.iter().map(|v|*v as f32).collect::<Vec<_>>();
    let lon=lon.data.iter().map(|v|*v as f32).collect::<Vec<_>>();
    let mut writer=NcWriter::create(&args[2],schema)?;
    writer.write_record(0,times,VarData::Char(stamp.as_bytes()))?;
    writer.write_record(0,latitude,VarData::F32(&lat))?;
    writer.write_record(0,longitude,VarData::F32(&lon))?;
    writer.write_record(0,theta,VarData::F32(&vec![0.;ny*nx]))?;
    for (name,variable) in fine {
        let values=match name {
            "FXLAT"|"FXLONG"|"ZSF"=>fields.fields[name].data().iter().map(|v|*v as f32).collect(),
            "LFN"=>fields.fields["LFN_HIST"].data().iter().map(|v|*v as f32).collect(),
            "FIRE_AREA"=>fields.fields["LFN_HIST"].data().iter().map(|v|if *v<=0. {1.} else {0.}).collect(),
            _=>vec![0.;fy*fx],
        };
        writer.write_record(0,variable,VarData::F32(&values))?;
    }
    writer.finish()?;
    println!("Captured perimeter renderer control: {fy}x{fx}, ROS_FRONT=0, FGRNHFX=0; no coupled atmospheric integration");
    Ok(())
}
fn main() {if let Err(error)=run() {eprintln!("{error}");std::process::exit(1);}}
