// Original Apache-2.0. Bathymetry and forcing preparation, never Python data work.
use rayon::prelude::*;
use serde_json::{Value,json};
use sha2::{Sha256,Digest};
use std::{fs,path::{Path,PathBuf},error::Error,time::Instant};
type R<T> = Result<T,Box<dyn Error>>;
fn sha(b:&[u8])->String {format!("{:x}",Sha256::digest(b))}
fn text<'a>(v:&'a Value,k:&str)->R<&'a str>{v[k].as_str().ok_or_else(||format!("missing {k}").into())}
fn read_nc(p:&Path,v:&str)->R<Vec<f64>>{Ok(netcrust::open(p)?.read_array_f64(v)?.values().to_vec())}
fn write(out:&Path,name:&str,v:&[f32],shape:&[usize])->R<Value>{
 let mut b=vec![0u8;v.len()*4]; b.par_chunks_mut(4).zip(v.par_iter()).for_each(|(b,v)|b.copy_from_slice(&v.to_le_bytes()));
 fs::write(out.join(format!("{name}.bin")),&b)?;
 Ok(json!({"path":format!("{name}.bin"),"dtype":"<f4","shape":shape,"sha256":sha(&b),"bytes":b.len()}))
}
fn f32s(v:&[f64])->Vec<f32>{v.par_iter().map(|&v|v as f32).collect()}
fn bracket(a:&[f64],v:f64)->Option<(usize,f64)>{
 if a.len()<2{return None} let asc=a[1]>a[0];
 if (asc&&(v<a[0]||v>a[a.len()-1]))||(!asc&&(v>a[0]||v<a[a.len()-1])){return None}
 let mut lo=0;let mut hi=a.len()-1;
 while hi-lo>1 {let m=(lo+hi)/2;if (asc&&a[m]<=v)||(!asc&&a[m]>=v){lo=m}else{hi=m}}
 Some((lo,(v-a[lo])/(a[lo+1]-a[lo])))
}
fn sample(z:&[f64],nx:usize,x:usize,y:usize,fx:f64,fy:f64)->f32{
 let v=[z[y*nx+x],z[y*nx+x+1],z[(y+1)*nx+x],z[(y+1)*nx+x+1]];
 if v.iter().any(|v|!v.is_finite()||v.abs()>20000.){return f32::NAN}
 ((1.-fy)*((1.-fx)*v[0]+fx*v[1])+fy*((1.-fx)*v[2]+fx*v[3])) as f32
}
fn neighbours(i:usize,nx:usize,ny:usize)->Vec<usize>{let(x,y)=(i%nx,i/nx);let mut v=Vec::with_capacity(4);if x>0{v.push(i-1)}if x+1<nx{v.push(i+1)}if y>0{v.push(i-nx)}if y+1<ny{v.push(i+nx)}v}
fn validate_rotation(cos:&[f32],sin:&[f32])->R<()> {
 if cos.len()!=sin.len()||cos.par_iter().zip(sin).any(|(&c,&s)|!c.is_finite()||!s.is_finite()||(c*c+s*s-1.).abs()>2.0e-4){return Err("wind rotation must be finite unit cosine/sine pairs".into())} Ok(())
}
fn prepare(r:&Value,out:&Path)->R<Value>{
 let gp=Path::new(text(r,"grid_file")?);let gf=netcrust::open(gp)?;
 let latname=r["latitude"].as_str().unwrap_or("latitude");let lonname=r["longitude"].as_str().unwrap_or("longitude");
 let mut lat=read_nc(gp,latname)?;let mut lon=read_nc(gp,lonname)?;
 let shape=gf.variable(latname).ok_or("missing latitude")?.shape();
 let rect=shape.len()==1;
 let lonshape=gf.variable(lonname).ok_or("missing longitude")?.shape();
 if (!rect&&lonshape!=shape)||(rect&&lonshape.len()!=1){return Err("target latitude/longitude coordinate dimensions differ".into())}
 if r["cosalpha"].as_str().is_some()!=r["sinalpha"].as_str().is_some()||(!rect&&r["cosalpha"].as_str().is_none()){return Err("projected target needs both cosalpha and sinalpha: identity rotation would misdirect wind stress".into())}
 let flipy=rect&&lat.len()>1&&lat[1]<lat[0];let flipx=rect&&lon.len()>1&&lon[1]<lon[0];
 if rect&&(lat.len()<2||lon.len()<2){return Err("target axes need at least two cells".into())}
 let (ny,nx,yy,xx)=if rect {
  for a in [&mut lat,&mut lon]{let sign=(a[1]-a[0]).signum();if a.iter().any(|v|!v.is_finite())||a.windows(2).any(|w|(w[1]-w[0])*sign<=0.){return Err("grid axes must be finite strictly monotone".into())}if sign<0.{a.reverse()}}
  let(ny,nx)=(lat.len(),lon.len());
  (ny,nx,(0..nx*ny).into_par_iter().map(|i|lat[i/nx] as f32).collect::<Vec<_>>(),(0..nx*ny).into_par_iter().map(|i|lon[i%nx] as f32).collect::<Vec<_>>())
 }else {
  if shape.len()<2||shape.len()>3{return Err("target coordinates must have shape [y,x] or [time,y,x]".into())}
  let(ny,nx)=(shape[shape.len()-2] as usize,shape[shape.len()-1] as usize);
  if lat.len()<ny*nx||lon.len()<ny*nx{return Err("target coordinates differ".into())}
  (ny,nx,f32s(&lat[..ny*nx]),f32s(&lon[..ny*nx]))
 };
 if nx<2||ny<2||yy.iter().chain(xx.iter()).any(|v|!v.is_finite()){return Err("target grid needs finite coordinates and >=2x2 cells".into())}
 let mut bed=vec![f32::NAN;nx*ny];let mut sources=Vec::new();
 for s in r["sources"].as_array().ok_or("sources must be an array")? {
  let path=Path::new(text(s,"path")?);let bytes=fs::read(path)?;let digest=sha(&bytes);
  if s["sha256"].as_str()!=Some(&digest){return Err("bathymetry SHA256 mismatch".into())}
  if text(s,"datum")?!=text(r,"datum")?{return Err("DEM datums differ: supply an explicit datum transform before mosaic".into())}
  if !["CC0-1.0","public-domain"].contains(&text(s,"licence")?){return Err("source row lacks a verified permissive data licence".into())}
  let units=s["elevation_units"].as_str().unwrap_or("m");
  let scale=match units {"m"=>1.,"ft"=>0.3048,"US survey foot"=>1200./3937.,_=>return Err("elevation_units must be m, ft or US survey foot".into())};
  let values:Vec<f32>=if s["format"].as_str()==Some("geotiff") {
   let raster=static_fields::raster::geotiff::read_band1(path,None,None,1.)?;
   let mut sx:Vec<f64>=xx.par_iter().map(|v|*v as f64).collect();let mut sy:Vec<f64>=yy.par_iter().map(|v|*v as f64).collect();
   static_fields::raster::transform_points(&static_fields::raster::Crs::Geographic,&raster.crs,&mut sx,&mut sy)?;
   let t=raster.transform;let det=t[0]*t[4]-t[1]*t[3];if det.abs()<1e-30{return Err("singular raster affine".into())}
   (0..nx*ny).into_par_iter().map(|i|{let x=((sx[i]-t[2])*t[4]-(sy[i]-t[5])*t[1])/det-0.5;let y=((sy[i]-t[5])*t[0]-(sx[i]-t[2])*t[3])/det-0.5;
    if x<0.||y<0.||x>=raster.nx as f64-1.||y>=raster.ny as f64-1.{f32::NAN}else{sample(&raster.values,raster.nx,x.floor() as usize,y.floor() as usize,x.fract(),y.fract())}}).collect()
  }else{
   let full_lat=read_nc(path,s["latitude"].as_str().unwrap_or("lat"))?;let full_lon=read_nc(path,s["longitude"].as_str().unwrap_or("lon"))?;
   for axis in [&full_lat,&full_lon]{if axis.len()<2||axis.iter().any(|v|!v.is_finite())||axis.windows(3).any(|w|(w[1]-w[0])*(w[2]-w[1])<=0.){return Err("DEM axes must be finite strictly monotone".into())}}
   let bounds=|axis:&[f64],target:&[f32]|->R<(usize,usize)>{let low=target.iter().copied().reduce(f32::min).ok_or("empty target")? as f64;let high=target.iter().copied().reduce(f32::max).ok_or("empty target")? as f64;
    let a=bracket(axis,low.clamp(axis[0].min(axis[axis.len()-1]),axis[0].max(axis[axis.len()-1]))).ok_or("invalid source bounds")?.0;let b=bracket(axis,high.clamp(axis[0].min(axis[axis.len()-1]),axis[0].max(axis[axis.len()-1]))).ok_or("invalid source bounds")?.0;Ok((a.min(b),a.max(b)+2))};
   let (y0,y1)=bounds(&full_lat,&yy)?;let(x0,x1)=bounds(&full_lon,&xx)?;
   let slat=&full_lat[y0..y1];let slon=&full_lon[x0..x1];
   let file=netcrust::open(path)?;let var=s["variable"].as_str().unwrap_or("z");
   if file.variable(var).ok_or("missing DEM elevation")?.shape()!=[full_lat.len(),full_lon.len()]{return Err("DEM must have [latitude,longitude] shape".into())}
   let z=file.read_array_f64_slice(var,&netcrust::NcSliceInfo{selections:vec![netcrust::NcSliceInfoElem::Slice{start:y0 as u64,end:y1 as u64,step:1},netcrust::NcSliceInfoElem::Slice{start:x0 as u64,end:x1 as u64,step:1}]})?;
   (0..nx*ny).into_par_iter().map(|i|{match(bracket(slon,xx[i] as f64),bracket(slat,yy[i] as f64)){(Some((x,fx)),Some((y,fy)))=>sample(z.values(),slon.len(),x,y,fx,fy),_=>f32::NAN}}).collect()

  };
  bed.par_iter_mut().zip(values).for_each(|(b,v)|if v.is_finite(){*b=(v as f64*scale) as f32});sources.push(s.clone());
 }
 if bed.par_iter().any(|v|!v.is_finite()){return Err("bathymetry has uncovered or nodata cells; source coverage must include the full target domain".into())}
 // Ocean connectivity, not just elevation <0: isolated below-datum land stays dry.
 // Atomic frontier expansion uses all Rayon workers; no single-thread flood fill.
 use std::sync::atomic::{AtomicBool,Ordering};
 let marks:Vec<AtomicBool>=(0..nx*ny).map(|_|AtomicBool::new(false)).collect();
 let mut frontier:Vec<usize>=(0..nx*ny).into_par_iter().filter(|&i|bed[i]<0.&&(i%nx==0||i%nx==nx-1||i/nx==0||i/nx==ny-1)).collect();
 for &i in &frontier{marks[i].store(true,Ordering::Relaxed)}
 while !frontier.is_empty(){frontier=frontier.par_iter().flat_map_iter(|&i|neighbours(i,nx,ny).into_iter().filter(|&j|bed[j]<0.&&!marks[j].swap(true,Ordering::Relaxed))).collect();}
 let wet:Vec<f32>=marks.par_iter().map(|v|if v.load(Ordering::Relaxed){1.}else{0.}).collect();
 let coastal:Vec<f32>=(0..nx*ny).into_par_iter().map(|i|if neighbours(i,nx,ny).iter().any(|&j|wet[j]!=wet[i]){1.}else{0.}).collect();
 let depth:Vec<f32>=bed.par_iter().map(|&v|(-v).max(0.)).collect();
 let rotation=|key:&str,default:f32|->R<Vec<f32>>{if let Some(name)=r[key].as_str(){
  let vshape=gf.variable(name).ok_or("missing rotation field")?.shape();
  if vshape.len()<2||vshape.len()>3||vshape[vshape.len()-2..]!=[ny,nx]{return Err("rotation fields must match target [y,x] or [time,y,x] dimensions".into())}
  let values=read_nc(gp,name)?;if values.len()<nx*ny{return Err("incomplete rotation field".into())}
  Ok((0..nx*ny).into_par_iter().map(|i|{let y=i/nx;let x=i%nx;values[(if flipy{ny-1-y}else{y})*nx+if flipx{nx-1-x}else{x}] as f32}).collect())
 }else{Ok(vec![default;nx*ny])}};
 let mut arrays=serde_json::Map::new();let ones=rotation("cosalpha",1.)?;let zeros=rotation("sinalpha",0.)?;validate_rotation(&ones,&zeros)?;
 for(name,v)in[("elevation",&bed),("depth",&depth),("wet",&wet),("coastal",&coastal),("latitude",&yy),("longitude",&xx),("cosalpha",&ones),("sinalpha",&zeros)]{arrays.insert(name.into(),write(out,name,v,&[ny,nx])?);}
 let meanlat=yy.par_iter().map(|v|*v as f64).sum::<f64>()/(nx*ny) as f64;
 let dx=r["dx"].as_f64().unwrap_or_else(||if rect{(lon[1]-lon[0]).abs()*111195.08*meanlat.to_radians().cos()}else{f64::NAN});let dy=r["dy"].as_f64().unwrap_or_else(||if rect{(lat[1]-lat[0]).abs()*111195.08}else{f64::NAN});
 if !dx.is_finite()||!dy.is_finite()||dx<=0.||dy<=0.{return Err("curvilinear target requires positive dx/dy in metres".into())}
 Ok(json!({"schema":"gpuwm.ocean.prepared/v1","nx":nx,"ny":ny,"dx":dx,"dy":dy,"datum":r["datum"],"sources":sources,"arrays":arrays,"wet_cells":wet.iter().filter(|&&v|v>0.).count(),"coastal_cells":coastal.iter().filter(|&&v|v>0.).count(),"grid_geometry":if rect{"geographic rectilinear; dx at mean latitude; latitude retained for solver spherical metrics"}else{"atmosphere target grid; supplied dx/dy and rotation"},"row_order":if rect{"south-to-north, west-to-east"}else{"same as target atmosphere grid"}}))
}
fn extract(r:&Value,out:&Path)->R<Value>{
 let path=Path::new(text(r,"input")?);let f=netcrust::open(path)?;let mut arrays=serde_json::Map::new();
 let lat=read_nc(path,r["latitude"].as_str().unwrap_or("latitude"))?;let lon=read_nc(path,r["longitude"].as_str().unwrap_or("longitude"))?;
 if lat.len()<2||lon.len()<2||lat.iter().chain(lon.iter()).any(|v|!v.is_finite())||[&lat,&lon].iter().any(|a|a.windows(2).any(|w|w[0]==w[1])||a.windows(3).any(|w|(w[1]-w[0])*(w[2]-w[1])<=0.)){return Err("forcing axes must be finite strictly monotone with at least two cells".into())}
 let (ny,nx)=(lat.len(),lon.len());let flipy=lat[1]<lat[0];let flipx=lon[1]<lon[0];
 for (name,var) in r["fields"].as_object().ok_or("fields must map output names to NetCDF variable names")? {
  let var=var.as_str().ok_or("variable name must be a string")?;let shape:Vec<usize>=f.variable(var).ok_or("missing forcing variable")?.shape().iter().map(|v|*v as usize).collect();
  if shape.len()!=3||shape[1..]!=[ny,nx]{return Err("forcing must have [time,latitude,longitude] shape".into())}
  let v=f.read_array_f64(var)?;let words=v.values();if words.par_iter().any(|v|!v.is_finite()||v.abs()>1e20){return Err(format!("invalid forcing values in {var}").into())}
  let ordered:Vec<f32>=(0..words.len()).into_par_iter().map(|i|{let t=i/(ny*nx);let y=i/nx%ny;let x=i%nx;words[t*ny*nx+(if flipy{ny-1-y}else{y})*nx+if flipx{nx-1-x}else{x}] as f32}).collect();
  arrays.insert(name.clone(),write(out,name,&ordered,&shape)?);
 }
 let times=read_nc(path,r["time"].as_str().unwrap_or("time"))?;
 if times.len()<2||arrays.values().any(|row|row["shape"][0].as_u64()!=Some(times.len() as u64)){return Err("forcing time axis and record counts differ or have fewer than two records".into())}
 if times.windows(2).any(|w|w[1]<=w[0])||times.iter().any(|v|!v.is_finite()){return Err("times must be finite increasing epoch seconds".into())}
 Ok(json!({"schema":"gpuwm.ocean.forcing/v1","input":path,"input_sha256":sha(&fs::read(path)?),"arrays":arrays,"times":times,"row_order":"south-to-north, west-to-east","latitude":if flipy{lat.into_iter().rev().collect::<Vec<_>>()}else{lat},"longitude":if flipx{lon.into_iter().rev().collect::<Vec<_>>()}else{lon}}))
}
/// The commit this binary was built from; the release pin step reads it from the bytes.
const SOURCE_REV_STAMP: &str = concat!("GPUWM_BRIDGE_SOURCE_REV=", env!("GPUWM_BRIDGE_SOURCE_REV"));

fn main()->R<()>{
 std::hint::black_box(SOURCE_REV_STAMP);
 if std::env::args().nth(1).as_deref()==Some("--version"){println!("rw_ocean gpuwm.ocean.prepared/v1 GPUWM_BRIDGE_SOURCE_REV={}",env!("GPUWM_BRIDGE_SOURCE_REV"));return Ok(())}
 let start=Instant::now();let p=PathBuf::from(std::env::args().nth(1).ok_or("rw_ocean REQUEST.json")?);let r:Value=serde_json::from_slice(&fs::read(p)?)?;
 let out=Path::new(text(&r,"output")?);fs::create_dir_all(out)?;
 if out.join("manifest.json").exists(){return Err("prepared manifest exists; use a fresh output directory".into())}
 let mut m=match text(&r,"operation")?{"prepare"=>prepare(&r,out)?,"extract"=>extract(&r,out)?,_=>return Err("unknown operation".into())};
 m["tool_source_revision"]=json!(env!("GPUWM_BRIDGE_SOURCE_REV"));m["parallelism"]=json!({"threads":rayon::current_num_threads(),"data_loops":"Rayon","decode":"Rust NetCDF/GeoTIFF"});m["wall_seconds"]=json!(start.elapsed().as_secs_f64());
 fs::write(out.join("manifest.json"),serde_json::to_vec_pretty(&m)?)?;println!("{}",m);Ok(())}

#[cfg(test)]
mod tests {
 use super::*;
 #[test]
 fn wind_rotation_rejects_nonunit_or_nonfinite_stress_directions() {
  assert!(validate_rotation(&[1.,0.6],&[0.,0.8]).is_ok());
  assert!(validate_rotation(&[1.,0.],&[0.,0.]).is_err());
  assert!(validate_rotation(&[f32::NAN],&[0.]).is_err());
  assert!(validate_rotation(&[1.],&[]).is_err());
 }
 #[test]
 fn descending_axes_and_closed_boundaries_keep_the_correct_cell() {
  assert_eq!(bracket(&[30.,20.,10.],30.),Some((0,0.)));
  assert_eq!(bracket(&[30.,20.,10.],10.),Some((1,1.)));
  assert_eq!(bracket(&[30.,20.,10.],15.),Some((1,0.5)));
  assert_eq!(bracket(&[10.,20.,30.],30.),Some((1,1.)));
  assert_eq!(bracket(&[30.,20.,10.],9.),None);
 }
 #[test]
 fn coastal_overlay_interpolation_rejects_nodata_instead_of_erasing_base() {
  assert!((sample(&[0.,2.,4.,6.],2,0,0,0.25,0.5)-2.5).abs()<1.0e-6);
  assert!(sample(&[0.,f64::NAN,4.,6.],2,0,0,0.25,0.5).is_nan());
  assert!(sample(&[0.,-32768.,4.,6.],2,0,0,0.25,0.5).is_nan());
 }
}
