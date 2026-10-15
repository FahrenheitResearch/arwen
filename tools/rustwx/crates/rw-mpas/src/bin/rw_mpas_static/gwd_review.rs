//! Proof visualization only: native GWD fields gathered by the existing Rust
//! spherical nearest-cell renderer weights. No interpolation enters word proof.
//! MPAS-A v8.4.2 mpas_init_atm_gwd.F:102-333 defines this ten-field inventory.
use std::path::Path;
use rw_mpas::{weights::{read_mesh_coordinates,build_weights}, window::{Window,LatLonWindow,ProjAttr}};
use rw_store::netcdf_classic::{NcAttr,NcClassicWriter,NcData,NcDim,NcFormat,NcType,NcVarDef};
const FIELDS: [&str;10] = ["var2d","con","oa1","oa2","oa3","oa4","ol1","ol2","ol3","ol4"];
fn run(args: Vec<String>) -> Result<(),Box<dyn std::error::Error>> {
    if args.len()!=4 {return Err("usage: rw_mpas_static gwd-review GRID.nc STATIC.nc FRAME.nc MANIFEST.json".into());}
    let grid=Path::new(&args[0]); let source=Path::new(&args[1]); let out=Path::new(&args[2]);
    let mesh=read_mesh_coordinates(grid,None)?;
    let window=Window::LatLon(LatLonWindow{south:-89.75,north:89.75,west:-179.75,east:179.75,spacing_degrees:0.5,
        description:"GWD static global nearest-cell review; no smoothing".into()});
    let weights=build_weights(&mesh,"static-review",&window)?;
    let (ny,nx)=weights.shape();
    let mut attrs=vec![NcAttr::text("TITLE","MPAS GWD STATIC REVIEW"),
        NcAttr::text("GPUWM_MODEL_LABEL","MPAS GWD static"),
        NcAttr::text("SIMULATION_START_DATE","2000-01-01_00:00:00"),
        NcAttr::text("MPAS_NONCLAIM","Nearest-cell visualization only; native-cell word proof is independent"),
        NcAttr::text("MPAS_SOURCE_SHA256",rw_mpas::sha256_file(source)?)];
    for (name,value) in window.projection_attributes()? {attrs.push(match value {
        ProjAttr::Int(v)=>NcAttr::int(name,v),ProjAttr::Float(v)=>NcAttr::float(name,v as f32),ProjAttr::Text(v)=>NcAttr::text(name,v)});}
    let dims=vec![NcDim::record("Time"),NcDim::fixed("DateStrLen",19),NcDim::fixed("south_north",ny),
        NcDim::fixed("west_east",nx),NcDim::fixed("bottom_top",1)];
    let mut vars=vec![NcVarDef::new("Times",NcType::Char,vec![0,1]),
        NcVarDef::new("XLAT",NcType::Float,vec![0,2,3]),NcVarDef::new("XLONG",NcType::Float,vec![0,2,3]),
        NcVarDef::new("T",NcType::Float,vec![0,4,2,3]),NcVarDef::new("XTIME",NcType::Float,vec![0])];
    for name in FIELDS {vars.push(NcVarDef::new(name,NcType::Float,vec![0,2,3]).with_attrs(vec![
        NcAttr::int("FieldType",104),NcAttr::text("MemoryOrder","XY "),NcAttr::text("stagger",""),
        NcAttr::text("description",format!("native GWD {name}")),NcAttr::text("units",if name=="var2d" {"m"}else{"1"})]));}
    let mut writer=NcClassicWriter::create(out,NcFormat::Offset64,dims,attrs,vars,1)?;
    writer.put_record_text("Times",0,"2000-01-01_00:00:00")?;
    let lat:Vec<f32>=weights.target_latitude.iter().map(|v|*v as f32).collect();
    let lon:Vec<f32>=weights.target_longitude.iter().map(|v|*v as f32).collect();
    writer.put_record("XLAT",0,NcData::Floats(&lat))?;writer.put_record("XLONG",0,NcData::Floats(&lon))?;
    writer.put_record("T",0,NcData::Floats(&vec![0.0;ny*nx]))?;writer.put_record("XTIME",0,NcData::Floats(&[0.0]))?;
    let file=netcrust::File::open(source)?;let mut fields=vec![];
    for name in FIELDS {
        let variable=file.variable(name).ok_or("missing required GWD variable")?;
        if variable.dtype()!=&netcrust::DataType::F32 {return Err("GWD must be NC_FLOAT".into());}
        let values:Vec<f32>=file.read_array::<f32>(name)?.iter().copied().collect();
        if values.len()!=mesh.n_cells() {return Err("GWD cell count mismatch".into());}
        let gathered=weights.gather(&values,1)?;
        writer.put_record(name,0,NcData::Floats(&gathered))?;
        fields.push(serde_json::json!({"name":name,"native_cells":values.len(),"display_points":gathered.len()}));
    }
    writer.finish()?;
    std::fs::write(&args[3],serde_json::to_vec_pretty(&serde_json::json!({"schema":"i3-static-review-frame/v1",
        "grid_sha256":rw_mpas::sha256_file(grid)?,"source_sha256":rw_mpas::sha256_file(source)?,
        "frame_sha256":rw_mpas::sha256_file(out)?,"weights_sha256":weights.weights_sha256,
        "method":rw_mpas::weights::RESAMPLE_METHOD,"word_proof":false,"fields":fields}))?)?;
    Ok(())
}
/// `rw_mpas_static gwd-review ...`, or this executable invoked as
/// `rw_mpas_gwd_review`.
pub fn main_with(argv: Vec<std::ffi::OsString>) -> std::process::ExitCode {
    let Ok(args) = crate::utf8_arguments("rw_mpas_gwd_review", argv) else { return std::process::ExitCode::from(2) };
    if let Err(e)=run(args){eprintln!("{e}");return std::process::ExitCode::FAILURE;}
    std::process::ExitCode::SUCCESS
}
