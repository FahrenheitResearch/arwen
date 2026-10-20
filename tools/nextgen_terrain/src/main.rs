// Original ArWen implementation, Apache-2.0. Raster/projection substrate: MIT.
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use static_fields::raster::{Crs, geotiff, transform_points};
use std::{collections::BTreeMap, error::Error, fs, path::{Path, PathBuf}, process::Command, time::Instant};
type Result<T> = std::result::Result<T, Box<dyn Error>>;

#[derive(Deserialize, Serialize)]
struct Grid { nx: usize, ny: usize, x0: f64, y0: f64, dx: f64, epsg: u32 }
#[derive(Deserialize)]
struct Request {
    source_table: PathBuf, source: String, output: PathBuf, grid: Grid,
    /// Offline mode uses a previously sealed TIFF, with mandatory expected SHA.
    input: Option<PathBuf>, expected_sha256: Option<String>,
}
#[derive(Deserialize, Serialize)]
struct Source {
    id: String, format: String, url: String,
    #[serde(default)] parameters: BTreeMap<String,String>,
    licence: String, licence_url: String, attribution: String,
    units: String, vertical_reference: String,
}
#[derive(Deserialize)]
struct Table { schema: String, sources: Vec<Source> }
fn sha(bytes: &[u8]) -> String { format!("{:x}", Sha256::digest(bytes)) }
fn write_field(out: &Path, name: &str, dtype: &str, data: Vec<u8>, grid: &Grid, units: &str) -> Result<Value> {
    let file = format!("{name}.bin");
    fs::write(out.join(&file), &data)?;
    Ok(json!({"path":file,"dtype":dtype,"shape":[grid.ny,grid.nx],
        "order":"C","units":units,"bytes":data.len(),"sha256":sha(&data)}))
}
fn f32_bytes(values: &[f32]) -> Vec<u8> {
    let mut b=vec![0;values.len()*4];
    b.par_chunks_mut(4).zip(values.par_iter()).for_each(|(b,v)| b.copy_from_slice(&v.to_le_bytes())); b
}
fn i32_bytes(values: &[i32]) -> Vec<u8> {
    let mut b=vec![0;values.len()*4];
    b.par_chunks_mut(4).zip(values.par_iter()).for_each(|(b,v)| b.copy_from_slice(&v.to_le_bytes())); b
}
fn run() -> Result<()> {
    let start=Instant::now();
    let request_path=std::env::args_os().nth(1).ok_or("usage: rw_terrain REQUEST.json")?;
    let request_bytes=fs::read(request_path)?;
    let r:Request=serde_json::from_slice(&request_bytes)?;
    let g=&r.grid;
    let n=g.nx.checked_mul(g.ny).filter(|&n|n>0 && n<=i32::MAX as usize).ok_or("invalid grid size")?;
    if ![g.x0,g.y0,g.dx].iter().all(|v|v.is_finite()) || g.dx<=0. || g.epsg!=5070 {
        return Err("grid must be finite, square, positive spacing EPSG:5070".into());
    }
    let table_bytes=fs::read(&r.source_table)?;
    let table:Table=serde_json::from_slice(&table_bytes)?;
    if table.schema!="arwen.terrain-sources/v1" { return Err("unsupported source table".into()); }
    let source=table.sources.into_iter().find(|s|s.id==r.source).ok_or("source row missing")?;
    if source.format!="geotiff" || source.units!="m" { return Err("requires GeoTIFF elevations in metres".into()); }
    if r.output.exists() { return Err("output already exists; choose a fresh receipt directory".into()); }
    fs::create_dir_all(&r.output)?;
    let substitutions=[("west",g.x0.to_string()),("south",g.y0.to_string()),
        ("east",(g.x0+g.nx as f64*g.dx).to_string()),("north",(g.y0+g.ny as f64*g.dx).to_string()),
        ("nx",g.nx.to_string()),("ny",g.ny.to_string()),("epsg",g.epsg.to_string())];
    let expand=|s:&str| { let mut text=s.to_string(); for (k,v) in &substitutions { text=text.replace(&format!("{{{k}}}"),v); } text };
    let parameters:BTreeMap<_,_>=source.parameters.iter().map(|(k,v)|(k.clone(),expand(v))).collect();
    let downloaded=r.output.join("source.tif");
    let input=if let Some(path)=&r.input {
        if r.expected_sha256.is_none() { return Err("offline input requires expected_sha256".into()); } path
    } else {
        let mut cmd=Command::new("curl");
        cmd.args(["--fail","--location","--silent","--show-error","--max-time","90","--get"]);
        cmd.arg(expand(&source.url));
        for (k,v) in &parameters {cmd.arg("--data-urlencode").arg(format!("{k}={v}"));}
        let result=cmd.arg("--output").arg(&downloaded).status()?;
        if !result.success() { return Err(format!("source download failed: {result}").into()); }
        &downloaded
    };
    let source_bytes=fs::read(input)?;
    let source_sha=sha(&source_bytes);
    if let Some(expected)=&r.expected_sha256 {if expected!=&source_sha {return Err("source SHA256 mismatch".into());}}
    let fetch_ms=start.elapsed().as_secs_f64()*1000.;
    // Inspect tags before accepting any grid override: coordinates and CRS must
    // come from the TIFF itself, never fabricated from the request.
    let raster=geotiff::read_band1(input,None,None,1.)?;
    let crs=Crs::AlbersConusNad83{lat_1:29.5,lat_2:45.5,lat_0:23.,lon_0:-96.,false_easting:0.,false_northing:0.};
    let expected=[g.dx,0.,g.x0,0.,-g.dx,g.y0+g.ny as f64*g.dx];
    if raster.nx!=g.nx || raster.ny!=g.ny || raster.crs!=crs ||
       raster.transform.iter().zip(expected).any(|(a,b)|(a-b).abs()>0.001) {
        return Err(format!("source grid/CRS differs: {:?}, {:?}",raster.transform,raster.crs).into());
    }
    let decode_ms=start.elapsed().as_secs_f64()*1000.-fetch_ms;
    let z:Vec<f32>=raster.values.par_iter().map(|&v|v as f32).collect();
    let active:Vec<i32>=z.par_iter().map(|v|i32::from(v.is_finite())).collect();
    let valid=active.par_iter().map(|&v|v as usize).sum::<usize>();
    if valid==0 {return Err("DEM has no finite cells".into());}
    let mut lon:Vec<f64>=(0..n).into_par_iter().map(|p|g.x0+(p%g.nx) as f64*g.dx+g.dx/2.).collect();
    let mut lat:Vec<f64>=(0..n).into_par_iter().map(|p|g.y0+(g.ny-p/g.nx) as f64*g.dx-g.dx/2.).collect();
    transform_points(&crs,&Crs::Geographic,&mut lon,&mut lat)?;
    if lon.par_iter().chain(lat.par_iter()).any(|v|!v.is_finite()) {return Err("nonfinite geolocation".into());}
    let lon:Vec<f32>=lon.par_iter().map(|&v|v as f32).collect();
    let lat:Vec<f32>=lat.par_iter().map(|&v|v as f32).collect();
    let downstream:Vec<i32>=(0..n).into_par_iter().map(|p| {
        if active[p]==0 {return -1;} let (x,y)=(p%g.nx,p/g.nx);
        let mut best=0.;let mut target=-1;
        for (ox,oy) in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(1,-1),(-1,1),(1,1)] {
            let (xx,yy)=(x as isize+ox,y as isize+oy);
            if xx<0||yy<0||xx>=g.nx as isize||yy>=g.ny as isize {continue;}
            let q=yy as usize*g.nx+xx as usize;
            if active[q]==0 {continue;}
            let distance=if ox!=0&&oy!=0 {g.dx*2_f64.sqrt()} else {g.dx};
            let slope=(z[p] as f64-z[q] as f64)/distance;
            if slope>best {best=slope;target=q as i32;}
        } target
    }).collect();
    let prepare_ms=start.elapsed().as_secs_f64()*1000.-fetch_ms-decode_ms;
    let mut arrays=BTreeMap::new();
    for (name,values,units) in [("elevation_m",&z,"m"),("latitude",&lat,"degrees_north"),("longitude",&lon,"degrees_east")] {
        arrays.insert(name,write_field(&r.output,name,"<f4",f32_bytes(values),g,units)?);
    }
    for (name,values,units) in [("active",&active,"0/1"),("downstream_cell",&downstream,"flat zero-based cell index; -1 pit/inactive/boundary")] {
        arrays.insert(name,write_field(&r.output,name,"<i4",i32_bytes(values),g,units)?);
    }
    let receipt=json!({"schema":"arwen.nextgen.terrain-grid/v1","source":source,
        "source_table_sha256":sha(&table_bytes),"request_sha256":sha(&request_bytes),
        "download_parameters":parameters,"source_sha256":source_sha,"source_bytes":source_bytes.len(),
        "grid":g,"row_order":"north-first, x increasing east; y0 is southern grid edge", "arrays":arrays,
        "valid_dem_cells":valid,"total_cells":n,"active_semantics":"finite DEM rectangle; not hydrofabric basin mask",
        "topology":"unconditioned topographic D8; no fill/breach; pits retained; no channel assignment",
        "parallelism":{"rayon_threads":rayon::current_num_threads(),"loops":"grid fields, geolocation, D8, encode use Rayon; TIFF decode setup only","gpu":"none; startup tool"},
        "timings_ms":{"fetch_hash":fetch_ms,"decode":decode_ms,"prepare":prepare_ms,
            "write_hash":start.elapsed().as_secs_f64()*1000.-fetch_ms-decode_ms-prepare_ms,
            "total":start.elapsed().as_secs_f64()*1000.}});
    fs::write(r.output.join("manifest.json"),serde_json::to_vec_pretty(&receipt)?)?;
    println!("{}",serde_json::to_string(&receipt)?); Ok(())
}
fn main() {if let Err(e)=run() {eprintln!("rw_terrain: {e}");std::process::exit(1);}}
