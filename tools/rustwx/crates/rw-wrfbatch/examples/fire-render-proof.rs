//! Verify the native geographic field footprint and LFN=0 contour in real PNGs.
use std::path::PathBuf;
use netcrust::File;
use rustwx_render::PanelGeoReference;
use serde_json::json;
use sha2::{Digest,Sha256};

fn run() -> Result<(),Box<dyn std::error::Error>> {
    let args=std::env::args().collect::<Vec<_>>();
    if !(args.len()==6 || (args.len()==7 && args[6]=="--forecast")) {return Err("usage: fire-render-proof NETCDF PERIMETER_PNG ROS_PNG HEAT_PNG RECEIPT_JSON [--forecast]".into());}
    let forecast=args.len()==7;
    let file=File::open(&args[1])?;
    let lat=file.read_array_f64_record_or_all("FXLAT",0)?;
    let lon=file.read_array_f64_record_or_all("FXLONG",0)?;
    let lfn=file.read_array_f64_record_or_all("LFN",0)?;
    let shape=lfn.shape();let (ny,nx)=(shape[shape.len()-2],shape[shape.len()-1]);
    if lat.shape()!=shape || lon.shape()!=shape || lat.values().iter().chain(lon.values()).any(|v|!v.is_finite()) {
        return Err("Fine geographic coordinates have a mismatched shape or nonfinite value".into());
    }
    let mut field_statistics=Vec::new();
    for name in ["LFN","ROS_FRONT","FGRNHFX"] {
        let field=file.read_array_f64_record_or_all(name,0)?;
        if field.shape()!=shape || field.values().iter().any(|v|!v.is_finite()) {
            return Err(format!("{name} has a mismatched fine shape or nonfinite value").into());
        }
        if !forecast && name!="LFN" && field.values().iter().any(|v|*v!=0.) {
            return Err(format!("{name} is not a zero renderer control").into());
        }
        field_statistics.push(json!({"field":name,"minimum":field.values().iter().copied().fold(f64::INFINITY,f64::min),
            "maximum":field.values().iter().copied().fold(f64::NEG_INFINITY,f64::max),
            "positive_cells":field.values().iter().filter(|v|**v>0.).count(),"sum":field.values().iter().sum::<f64>()}));
    }
    let mut images=Vec::new();
    for png in &args[2..5] {
        let path=PathBuf::from(png);let pixels=image::open(&path)?.into_rgb8();
        let reference:PanelGeoReference=serde_json::from_slice(&std::fs::read(path.with_extension("georef.json"))?)?;
        if (reference.image_width_px,reference.image_height_px)!=pixels.dimensions() {
            return Err("Georeference size differs from the written PNG".into());
        }
        let mut bounds=[f64::INFINITY,f64::INFINITY,f64::NEG_INFINITY,f64::NEG_INFINITY];
        for (la,lo) in lat.values().iter().zip(lon.values()) {
            let (x,y)=reference.lonlat_to_pixel(*la,*lo).ok_or("Fine geographic cell projects outside PNG")?;
            bounds[0]=bounds[0].min(x);bounds[1]=bounds[1].min(y);
            bounds[2]=bounds[2].max(x);bounds[3]=bounds[3].max(y);
        }
        let width=bounds[2]-bounds[0];let height=bounds[3]-bounds[1];
        if width<0.6*pixels.width() as f64 || height<0.75*pixels.height() as f64 {
            return Err(format!("Native fine field shrank to {width:.2}x{height:.2} inside PNG {:?}",pixels.dimensions()).into());
        }
        let mut samples=0usize;let mut hits=0usize;
        for j in 0..ny {for i in 0..nx {
            let a=j*nx+i;
            for b in [if i+1<nx {Some(a+1)} else {None},if j+1<ny {Some(a+nx)} else {None}].into_iter().flatten() {
                let av=lfn.values()[a];let bv=lfn.values()[b];
                if (av<0.)==(bv<0.) || av==bv {continue;}
                let fraction=-av/(bv-av);
                let la=lat.values()[a]+fraction*(lat.values()[b]-lat.values()[a]);
                let lo=lon.values()[a]+fraction*(lon.values()[b]-lon.values()[a]);
                let (x,y)=reference.lonlat_to_pixel(la,lo).ok_or("Perimeter crossing leaves PNG")?;
                samples+=1;
                let mut hit=false;
                for py in (y.round() as i64-4)..=(y.round() as i64+4) {
                    for px in (x.round() as i64-4)..=(x.round() as i64+4) {
                        if px<0 || py<0 || px>=pixels.width() as i64 || py>=pixels.height() as i64 {continue;}
                        let rgb=pixels.get_pixel(px as u32,py as u32).0;
                        hit|=(70..=110).contains(&rgb[0]) && rgb[1]<40 && rgb[2]<50;
                    }
                }
                hits+=usize::from(hit);
            }
        }}
        if samples==0 || hits as f64/(samples as f64)<0.95 {
            return Err(format!("Geographic LFN=0 contour found at only {hits}/{samples} expected edge crossings").into());
        }
        images.push(json!({"png":path.file_name().unwrap().to_string_lossy(),"sha256":format!("{:x}",Sha256::digest(std::fs::read(&path)?)),
            "image_size_px":pixels.dimensions(),"fine_field_bounds_px":bounds,"fine_field_size_px":[width,height],
            "lfn_zero_crossings":samples,"contour_pixel_hits":hits,"crossing_search_radius_px":4}));
    }
    let receipt=json!({"schema":"sfire-renderer-native-footprint-proof-v1","scope":if forecast {"coupled forecast fields and geographic perimeter"} else {"captured observed perimeter, zero ROS and heat; no coupled forecast"},
        "input_sha256":format!("{:x}",Sha256::digest(std::fs::read(&args[1])?)),"fine_shape":[ny,nx],"field_statistics":field_statistics,"images":images});
    std::fs::write(&args[5],serde_json::to_vec_pretty(&receipt)?)?;
    println!("{}",serde_json::to_string_pretty(&receipt)?);Ok(())
}
fn main() {if let Err(error)=run() {eprintln!("{error}");std::process::exit(1);}}
