//! Original Apache-2.0 ocean product rows. Numeric input and rendering stay native.
use crate::panel::{layout_path, render_panel_cells, PanelRequest};
use rayon::prelude::*;
use rustwx_render::{Color, ColorScale, DiscreteColorScale, ExtendMode, LegendControls, LegendMode};
use serde::Deserialize;
use serde_json::{json,Value};
use std::{collections::BTreeMap,fs,path::{Path,PathBuf},time::Instant};
type R<T> = Result<T,String>;
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {schema:String,out_dir:PathBuf,domain:String,source_label:String,nx:usize,ny:usize,latitude:PathBuf,longitude:PathBuf,#[serde(default)] wet:Option<PathBuf>,rows:PathBuf,frames:Vec<Frame>}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Frame {valid_time:String,fields:BTreeMap<String,PathBuf>}
fn read(path:&Path,n:usize)->R<Vec<f32>>{
 let b=fs::read(path).map_err(|e|e.to_string())?;
 if b.len()!=n*4{return Err(format!("{} has {} bytes; expected {}",path.display(),b.len(),n*4))}
 let v:Vec<f32>=b.par_chunks_exact(4).map(|x|f32::from_le_bytes(x.try_into().unwrap())).collect();
 if v.par_iter().any(|v|!v.is_finite()){return Err(format!("nonfinite ocean plane {}",path.display()))}Ok(v)
}
fn token(s:&str)->bool{!s.is_empty()&&s.bytes().all(|b|b.is_ascii_alphanumeric()||b==b'-'||b==b'_')}
fn run(path:&Path)->R<()> {
 let start=Instant::now();let r:Request=serde_json::from_slice(&fs::read(path).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;
 if r.schema!="gpuwm.ocean.products/v1"||!token(&r.domain)||r.nx<2||r.ny<2{return Err("invalid ocean product schema, domain or grid".into())}
 let rows:Value=serde_json::from_slice(&fs::read(&r.rows).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;
 let lat=read(&r.latitude,r.nx*r.ny)?;let lon=read(&r.longitude,r.nx*r.ny)?;
 let wet=r.wet.as_ref().map(|p|read(p,r.nx*r.ny)).transpose()?;
 // Preserve existing products on an ordinary repeat. Claim a fresh sibling.
 let mut out=r.out_dir.clone();
 if out.exists()&&fs::read_dir(&out).map_err(|e|e.to_string())?.next().is_some(){let stamp=std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_err(|e|e.to_string())?.as_nanos();out=out.with_file_name(format!("{}-{stamp}",out.file_name().unwrap_or_default().to_string_lossy()));}
 fs::create_dir_all(&out).map_err(|e|e.to_string())?;
 let mut receipt=Vec::new();
 for f in &r.frames {
  if f.valid_time.len()!=20||!f.valid_time.is_ascii()||f.valid_time.as_bytes()[10]!=b'T'||!f.valid_time.bytes().all(|b|b.is_ascii_digit()||b"-:TZ".contains(&b)){return Err("valid_time must be ISO UTC YYYY-MM-DDTHH:MM:SSZ".into())}
  for (slug,path) in &f.fields {
   if !token(slug){return Err("invalid ocean product name".into())}
   let row=&rows["products"][slug];let title=row["title"].as_str().ok_or_else(||format!("unknown ocean product row {slug}"))?;
   let units=row["units"].as_str().ok_or("row units missing")?;
   let low=row["range"][0].as_f64().ok_or("row lower range missing")?;let high=row["range"][1].as_f64().ok_or("row upper range missing")?;
   if !low.is_finite()||!high.is_finite()||high<=low{return Err("invalid product range".into())}
   let factor=row["scale"].as_f64().unwrap_or(1.) as f32;
   let mut values=read(path,r.nx*r.ny)?;values.par_iter_mut().for_each(|v|*v*=factor);
   if let Some(mask)=row["mask"].as_str(){
    let wet=wet.as_ref().ok_or("ocean product mask requires prepared wet plane")?;
    let flooded=match mask{"wet"=>None,"wet-or-inundated"=>Some(read(f.fields.get("inundation").ok_or("water-level mask requires inundation plane")?,r.nx*r.ny)?),_=>return Err(format!("unknown ocean product mask {mask}"))};
    values.par_iter_mut().enumerate().for_each(|(i,v)|{if wet[i]<=0.5&&!flooded.as_ref().is_some_and(|a|a[i]>0.){*v=f32::NAN;}});
   }
   let min=values.par_iter().copied().filter(|v|v.is_finite()).reduce(||f32::INFINITY,f32::min);let max=values.par_iter().copied().filter(|v|v.is_finite()).reduce(||f32::NEG_INFINITY,f32::max);
   let bins=20;let levels=(0..=bins).map(|i|low+(high-low)*i as f64/bins as f64).collect();
   let colors=(0..bins).map(|i|{let q=i as f64/(bins-1) as f64;Color::rgba((35.+215.*q) as u8,(90.+120.*(1.-(2.*q-1.).abs())) as u8,(235.-205.*q) as u8,255)}).collect();
   let scale=ColorScale::Discrete(DiscreteColorScale{levels,colors,extend:ExtendMode::Both,mask_below:None});
   let p=layout_path(&out,&r.domain,slug,&f.valid_time[..10],&format!("{slug}_{}",f.valid_time.replace(':',"")));
   render_panel_cells(PanelRequest {lat_deg:&lat,lon_deg:&lon,projection:None,ny:r.ny,nx:r.nx,values,product_slug:slug.clone(),title:format!("{title}; scale {low} to {high}"),display_units:units.into(),scale,cbar_tick_step:None,legend:LegendControls{mode:LegendMode::Stepped,..Default::default()},render_density:Default::default(),subtitle_left:f.valid_time.clone(),subtitle_center:None,subtitle_right:r.source_label.clone(),width:1200,height:900,contours:vec![],colorbar:true,overlays:None,annotations:None,out_path:p.clone()})?;
   receipt.push(json!({"product":slug,"valid_time":f.valid_time,"path":p,"min":min,"max":max,"units":units,"display_range":[low,high]}));
  }
 }
 let receipt=json!({"schema":"gpuwm.ocean.render-receipt/v1","renderer":"rw_wrfbatch","threads":rayon::current_num_threads(),"wall_seconds":start.elapsed().as_secs_f64(),"products":receipt});
 fs::write(out.join("receipt.json"),serde_json::to_vec_pretty(&receipt).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;println!("{}",receipt);Ok(())
}
pub fn cli(args:&[String])->Option<R<()>>{match args.first().map(String::as_str){Some("--ocean-products")=>Some(args.get(1).ok_or("--ocean-products needs request JSON".into()).and_then(|p|run(Path::new(p)))),Some("--ocean-products-abi")=>Some({println!("gpuwm.ocean.products/v1");Ok(())}),_=>None}}
