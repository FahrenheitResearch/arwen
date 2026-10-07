//! Refined SFIRE planes retain their own coordinates and zero-level perimeter.
use std::path::{Path,PathBuf};
use netcrust::File;
use rustwx_render::{Color,ColorScale,ContourLayer,DiscreteColorScale,ExtendMode};
use crate::panel::{PanelRequest,render_panel_with_rgba};

pub const PRODUCTS:[(&str,&str,&str,&str);3]=[
    ("fire_perimeter","FIRE_AREA","Fire perimeter", "1"),
    ("fire_ros","ROS_FRONT","Fire front rate of spread", "m/s"),
    ("fire_heat_flux","FGRNHFX","Ground fire sensible heat flux", "W/m2"),
];

pub fn split_products(spec:&str) -> (String,Vec<String>) {
    let mut remaining=Vec::new();let mut fire=Vec::new();
    for token in spec.split(',').map(str::trim).filter(|s|!s.is_empty()) {
        if PRODUCTS.iter().any(|row|row.0==token) {fire.push(token.to_string());}
        else {
            remaining.push(token);
            if matches!(token,"all"|"direct") {fire.extend(PRODUCTS.iter().map(|p|p.0.to_string()));}
        }
    }
    fire.sort();fire.dedup();(remaining.join(","),fire)
}

pub type CatalogRow=(String,&'static str,&'static str,String,&'static str);

/// The coupled fire's output fields; a file with none of them wrote no fire.
const FIRE_OUTPUTS:[&str;4]=["LFN","FIRE_AREA","ROS_FRONT","FGRNHFX"];

/// A domain with no coupled fire: declared IFIRE=0, or (a stock WRF or other
/// non-WOOF wrfout, which declares no IFIRE) no IFIRE attribute at all, and in
/// both cases none of the fire output fields.  Before, an undeclared file was
/// held to be an active fire domain, so `render` with `all` or `direct` failed
/// on every wrfout that never ran a fire.
fn no_coupled_fire(file:&File) -> bool {
    matches!(file.attribute("IFIRE").and_then(|a|a.as_f64()),Some(0.)|None)
        && FIRE_OUTPUTS.iter().all(|name|file.variable(name).is_none())
}

pub fn all_inactive(inputs:&[PathBuf]) -> bool {
    !inputs.is_empty() && inputs.iter().all(|path|File::open(path).ok().is_some_and(|file|no_coupled_fire(&file)))
}

pub fn catalog(inputs:&[PathBuf]) -> Vec<CatalogRow> {
    let inactive=all_inactive(inputs);
    let active=inputs.iter().any(|path|File::open(path).ok().is_some_and(|file|
        file.attribute("IFIRE").and_then(|a|a.as_f64())==Some(2.)));
    PRODUCTS.iter().map(|(slug,field,_,_)| {
        if inactive {return ((*slug).into(),"direct","excluded",
            "inactive-fire-domain: IFIRE=0 has no coupled fire outputs".into(),"inactive-fire-domain");}
        let ready=inputs.iter().any(|path|File::open(path).ok().is_some_and(|file| {
            file.attribute("FIRE_COORDINATE_MODE").and_then(|a|a.as_string().map(str::to_string)).as_deref()==Some("geographic")
                && ["FXLAT","FXLONG","ZSF","LFN",field].iter().all(|name|file.variable(name).is_some())
        }));
        if ready {((*slug).into(),"direct","renderable","refined SFIRE grid with FXLAT/FXLONG and LFN=0 perimeter".into(),"")}
        else {((*slug).into(),"direct","excluded",format!("needs explicit geographic FIRE_COORDINATE_MODE, FXLAT, FXLONG, ZSF, LFN and {field} on the refined grid"),
            if active {"invalid-active-fire-domain"} else {"missing-fire-grid-fields"})}
    }).collect()
}

fn plane(file:&File,name:&str,record:usize,shape:Option<[usize;2]>) -> Result<(Vec<f32>,[usize;2]),String> {
    let array=file.read_array_f64_record_or_all(name,record as u64).map_err(|e|format!("{name}: {e}"))?;
    let dimensions=array.shape();
    let dimensions=if dimensions.len()==3 && dimensions[0]==1 {&dimensions[1..]} else {dimensions};
    if dimensions.len()!=2 || dimensions.contains(&0) {return Err(format!("{name}: expected a nonempty fine-grid plane, got {:?}",array.shape()));}
    let actual=[dimensions[0],dimensions[1]];
    if shape.is_some_and(|expected|expected!=actual) {return Err(format!("{name}: fine-grid shape {actual:?} differs from coordinate shape {shape:?}"));}
    let values=array.values().iter().map(|v|*v as f32).collect();
    Ok((values,actual))
}

fn scale(slug:&str) -> ColorScale {
    let levels=match slug {
        "fire_perimeter"=>vec![0.0,0.01,0.25,0.5,0.75,1.00001],
        "fire_ros"=>vec![0.0,0.05,0.1,0.2,0.4,0.8,1.6,3.2],
        _=>vec![0.0,1000.,5000.,10000.,25000.,50000.,100000.,250000.],
    };
    let colors=if slug=="fire_perimeter" {
        vec![Color::rgba(0,0,0,0),Color::rgba(253,210,96,160),Color::rgba(251,148,42,180),
            Color::rgba(223,72,35,200),Color::rgba(151,33,37,220)]
    } else {
        [[255,245,204],[254,225,128],[253,174,67],[244,109,39],[216,55,36],[156,23,38],[91,10,43]]
            .into_iter().map(|[r,g,b]|Color::rgba(r,g,b,255)).collect()
    };
    ColorScale::Discrete(DiscreteColorScale {levels,colors,extend:ExtendMode::Max,
        mask_below:Some(if slug=="fire_perimeter" {0.01} else {f64::MIN_POSITIVE})})
}

fn context_pixels(values:&[f32],terrain:&[f32],shape:[usize;2],dx:f64,dy:f64,scale:&ColorScale) -> Result<Vec<Color>,String> {
    if !dx.is_finite() || !dy.is_finite() || dx<=0. || dy<=0.
        || terrain.len()!=values.len() || terrain.iter().any(|v|!v.is_finite()) {
        return Err("Fire terrain context needs finite ZSF and positive fine-grid spacing".into());
    }
    let style=rustwx_render::StaticPlotStyle::from_env();
    let map=rustwx_render::build_colormap(scale,rustwx_render::ColormapBuildOptions {
        render_density:style.render_density(Default::default()),legend:Default::default()});
    let [ny,nx]=shape;let mut pixels=Vec::with_capacity(values.len());
    for j in 0..ny {for i in 0..nx {
        let west=i.saturating_sub(1);let east=(i+1).min(nx-1);
        let south=j.saturating_sub(1);let north=(j+1).min(ny-1);
        let gx=if west==east {0.} else {(terrain[j*nx+east] as f64-terrain[j*nx+west] as f64)/((east-west) as f64*dx)};
        let gy=if south==north {0.} else {(terrain[north*nx+i] as f64-terrain[south*nx+i] as f64)/((north-south) as f64*dy)};
        let light=(-0.5*gx+0.5*gy+std::f64::consts::FRAC_1_SQRT_2)/(1.+gx*gx+gy*gy).sqrt();
        let shade=(167.+75.*light.clamp(0.,1.)).round() as u8;
        let background=[shade,shade.saturating_add(3),shade.saturating_sub(4)];
        let fire=map.map(values[j*nx+i] as f64);let a=fire.a as u32;
        let blend=|channel:u8,base:u8|((channel as u32*a+base as u32*(255-a)+127)/255) as u8;
        pixels.push(Color::rgba(blend(fire.r,background[0]),blend(fire.g,background[1]),blend(fire.b,background[2]),255));
    }}
    Ok(pixels)
}

fn graticule(values:&[f32]) -> Vec<f64> {
    let lo=values.iter().copied().fold(f32::INFINITY,f32::min) as f64;
    let hi=values.iter().copied().fold(f32::NEG_INFINITY,f32::max) as f64;
    let raw=(hi-lo)/5.;if raw<=0. || !raw.is_finite() {return vec![];}
    let magnitude=10_f64.powf(raw.log10().floor());
    let step=[1.,2.,5.,10.].into_iter().map(|n|n*magnitude).find(|n|*n>=raw).unwrap();
    let start=(lo/step).ceil();let end=(hi/step).floor();
    (start as i64..=end as i64).map(|n|((n as f64*step)*1e6).round()/1e6).collect()
}

fn graticule_labels(lat:&[f32],lon:&[f32],shape:[usize;2]) -> Vec<crate::annotate::LabelSpec> {
    let [ny,nx]=shape;let mut labels=Vec::new();
    let column=(nx/32).min(nx-1);let row=(ny/32).min(ny-1);
    for value in graticule(lat) {
        let j=(0..ny).min_by(|a,b|((lat[*a*nx+column] as f64)-value).abs()
            .total_cmp(&((lat[*b*nx+column] as f64)-value).abs())).unwrap();
        let index=j*nx+column;
        labels.push(crate::annotate::LabelSpec {lat:lat[index] as f64,lon:lon[index] as f64,
            text:format!("{:.2}°{}",value.abs(),if value<0. {"S"} else {"N"})});
    }
    for value in graticule(lon) {
        let i=(0..nx).min_by(|a,b|((lon[row*nx+*a] as f64)-value).abs()
            .total_cmp(&((lon[row*nx+*b] as f64)-value).abs())).unwrap();
        let index=row*nx+i;
        labels.push(crate::annotate::LabelSpec {lat:lat[index] as f64,lon:lon[index] as f64,
            text:format!("{:.2}°{}",value.abs(),if value<0. {"W"} else {"E"})});
    }
    labels
}

/// `<out>/<domain>/<product>/<valid-day>/<slug>_<stamp>.png`, the layout the
/// weather maps write ([`crate::panel::layout_path`]): a WRF valid time
/// `2025-01-07_18:00:00` files as `20250107/fire_ros_20250107T180000.png`.
/// Every component is built from the frame's time rather than copied from
/// it, so a character Windows refuses in a file name (`<>:"/\|?*` or a
/// control character) cannot reach the path: the time's `-` and `:` are
/// dropped, its `_` becomes `T`, and any other character that is not ASCII
/// alphanumeric becomes `_`.
pub(crate) fn map_path(out_dir:&Path,folder:&str,slug:&str,time:&str) -> PathBuf {
    let date=time.chars().take(10).filter(|c|c.is_ascii_digit()).collect::<String>();
    let stamp=time.trim().chars().filter_map(|c|match c {
        '-'|':'=>None,'_'=>Some('T'),c if c.is_ascii_alphanumeric()=>Some(c),_=>Some('_'),
    }).collect::<String>();
    let product=crate::panel::safe_component(slug,"product");
    out_dir.join(crate::panel::safe_component(folder,"native_grid")).join(&product)
        .join(if date.is_empty() {"undated"} else {&date})
        .join(format!("{product}_{stamp}.png"))
}

pub struct RenderConfig<'a> {
    pub inputs:&'a [PathBuf],pub out_dir:&'a Path,pub frame:Option<usize>,
    pub width:u32,pub height:u32,pub source_label:&'a str,
    pub overlays:Option<&'a crate::annotate::MapOverlays>,
    pub annotations:Option<&'a crate::annotate::PanelAnnotations>,
}

/// Each frame reads its own refined coordinates; the atmospheric store is not
/// used to crop or reinterpret a fire plane as a coarse atmospheric field.
pub type RenderedGeoref=(PathBuf,Option<rustwx_render::PanelGeoReference>,Option<String>);

pub fn render(products:&[String],config:&RenderConfig<'_>) -> Result<(usize,usize,usize,Vec<RenderedGeoref>),String> {
    let mut frames=Vec::new();
    for path in config.inputs {
        let wrf=wrf_core::WrfFile::open(path).map_err(|e|format!("{}: {e}",path.display()))?;
        let times=wrf.times().map_err(|e|e.to_string())?;
        for (record,time) in times.into_iter().enumerate() {frames.push((time,path.clone(),record));}
    }
    frames.sort_by(|a,b|a.0.cmp(&b.0));
    if let Some(index)=config.frame {
        let frame=frames.get(index).ok_or_else(||format!("--frames {index} exceeds {} fire frames",frames.len()))?.clone();
        frames=vec![frame];
    }
    let mut rendered=0;let mut skipped=0;let mut failed=0;let mut georeferences=Vec::new();
    for (time,path,record) in frames {
        let file=File::open(&path).map_err(|e|e.to_string())?;
        let wrf=wrf_core::WrfFile::open(&path).map_err(|e|e.to_string())?;
        let projection=crate::wrf_process::wrf_projection(&wrf);
        let attr=|name:&str|file.attribute(name).and_then(|a|a.as_f64());
        let domain=attr("GRID_ID").map(|n|format!("d{:02}",n as u32)).unwrap_or_else(||"native_grid".into());
        // The run's domain folder, named as the weather products name it
        // (`d01-1km`), so fire and smoke maps of one domain share a folder.
        let folder=crate::domain_naming::domain_folder(&domain,attr("DX"));
        for slug in products {
            let (_,variable,title,units)=PRODUCTS.iter().find(|row|row.0==slug).ok_or_else(||format!("Unknown fire product {slug}"))?;
            if attr("IFIRE")==Some(0.) {
                if FIRE_OUTPUTS.iter().any(|name|file.variable(name).is_some()) {
                    eprintln!("FAILED {slug} {} [frame {record} valid {time}]: IFIRE=0 contradicts active fire output fields",path.display());failed+=1;
                } else {
                    println!("SKIPPED {slug} {} [frame {record} valid {time}]: inactive-fire-domain: IFIRE=0 has no coupled fire outputs",path.display());skipped+=1;
                }
                continue;
            }
            if no_coupled_fire(&file) {
                println!("SKIPPED {slug} {} [frame {record} valid {time}]: inactive-fire-domain: no IFIRE declared and no coupled fire outputs",path.display());skipped+=1;
                continue;
            }
            if ["FXLAT","FXLONG","ZSF","LFN",variable].iter().any(|name|file.variable(name).is_none()) {
                eprintln!("FAILED {slug} {} [frame {record} valid {time}]: active or undeclared fire domain is missing refined fire fields",path.display());failed+=1;continue;
            }
            let result=(|| {
                if file.attribute("FIRE_COORDINATE_MODE").and_then(|a|a.as_string().map(str::to_string)).as_deref()!=Some("geographic") {
                    return Err("Fire coordinates lack explicit geographic provenance; metric or ambiguous FXLAT/FXLONG cannot be plotted as degrees".into());
                }
                let (lat,shape)=plane(&file,"FXLAT",record,None)?;
                let (lon,_)=plane(&file,"FXLONG",record,Some(shape))?;
                if lat.iter().any(|v|!v.is_finite() || !(-90.0..=90.0).contains(v))
                    || lon.iter().any(|v|!v.is_finite() || !(-180.0..=180.0).contains(v)) {
                    return Err("Fine-grid coordinates leave finite geographic degrees".into());
                }
                let (values,_)=plane(&file,variable,record,Some(shape))?;
                let (lfn,_)=plane(&file,"LFN",record,Some(shape))?;
                let (terrain,_)=plane(&file,"ZSF",record,Some(shape))?;
                let fdx=attr("DX").zip(attr("SR_X")).filter(|(_,sr)|*sr>0.).map(|(dx,sr)|((dx as f32)/(sr as f32)) as f64).ok_or("Fire terrain context needs DX and SR_X")?;
                let fdy=attr("DY").zip(attr("SR_Y")).filter(|(_,sr)|*sr>0.).map(|(dy,sr)|((dy as f32)/(sr as f32)) as f64).ok_or("Fire terrain context needs DY and SR_Y")?;
                let colors=scale(slug);let pixels=context_pixels(&values,&terrain,shape,fdx,fdy,&colors)?;
                let mut overlays=config.overlays.cloned().unwrap_or_default();
                overlays.labels.extend(graticule_labels(&lat,&lon,shape));
                let mut contours=Vec::new();
                for coordinates in [&lat,&lon] {contours.push(ContourLayer {data:coordinates.clone(),levels:graticule(coordinates),
                    color:Color::rgba(90,98,106,155),width:1,labels:true,show_extrema:false,pattern:Default::default(),major_every:None,major_width:None});}
                contours.push(ContourLayer {data:lfn,levels:vec![0.0],color:Color::rgba(88,12,25,255),
                    width:3,labels:false,show_extrema:false,pattern:Default::default(),major_every:None,major_width:None});
                let grid_label=match (attr("DX"),attr("SR_X")) { (Some(dx),Some(sr)) if sr>0.=>format!("{} m fire grid",(dx as f32)/(sr as f32)),_=>format!("{} x {} fire grid",shape[1],shape[0])};
                render_panel_with_rgba(PanelRequest {lat_deg:&lat,lon_deg:&lon,projection:projection.as_ref(),ny:shape[0],nx:shape[1],values,
                    product_slug:slug.clone(),title:if *units=="1" {(*title).into()} else {format!("{title} ({units})")},
                    display_units:(*units).into(),scale:colors,cbar_tick_step:None,legend:Default::default(),
                    render_density:Default::default(),subtitle_left:config.source_label.into(),
                    subtitle_center:Some(format!("{domain} | {grid_label}")),subtitle_right:format!("{time} UTC"),
                    width:config.width,height:config.height,contours,colorbar:*units!="1",
                    overlays:Some(&overlays),annotations:config.annotations,
                    out_path:map_path(config.out_dir,&folder,slug,&time)},Some(pixels))
            })();
            match result {
                Ok((output,reference))=>{
                    let reason=reference.is_none().then(||"Native SFIRE panel emitted no map transform".into());
                    georeferences.push((output.clone(),reference,reason));
                    println!("RENDERED {slug} {}",output.display());rendered+=1;
                },
                Err(error)=>{eprintln!("FAILED {slug} {} [frame {record} valid {time}]: {error}",path.display());failed+=1;},
            }
        }
    }
    Ok((rendered,skipped,failed,georeferences))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn declared_inactive_domain_skips_and_active_missing_fields_fail() {
        use netcdf_writer::{AttrValue,NcFormat,NcType,NcWriter,Schema,VarData};
        let nonce=std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
        let folder=std::env::temp_dir().join(format!("sfire-eligibility-{nonce}"));
        std::fs::create_dir(&folder).unwrap();
        // -1: no IFIRE attribute (a stock WRF wrfout); it holds no fire
        // output, so it is inactive like IFIRE=0, not a broken active domain.
        for active in [-1,0,2] {
            let path=folder.join(format!("domain-{active}.nc"));
            let mut schema=Schema::new(NcFormat::Offset64);
            let record=schema.def_dim("Time",0,true).unwrap();
            let chars=schema.def_dim("DateStrLen",19,false).unwrap();
            let rows=schema.def_dim("south_north",2,false).unwrap();
            let cols=schema.def_dim("west_east",2,false).unwrap();
            if active>=0 {schema.put_global_attr("IFIRE",AttrValue::Ints(vec![active])).unwrap();}
            let times=schema.def_var("Times",NcType::Char,&[record,chars]).unwrap();
            let latitude=schema.def_var("XLAT",NcType::Float,&[record,rows,cols]).unwrap();
            let longitude=schema.def_var("XLONG",NcType::Float,&[record,rows,cols]).unwrap();
            let mut writer=NcWriter::create(&path,schema).unwrap();
            writer.write_record(0,times,VarData::Char(b"2026-01-01_00:00:00")).unwrap();
            writer.write_record(0,latitude,VarData::F32(&[35.,35.,35.01,35.01])).unwrap();
            writer.write_record(0,longitude,VarData::F32(&[-120.,-119.99,-120.,-119.99])).unwrap();
            writer.finish().unwrap();
            let inputs=vec![path.clone()];
            assert_eq!(all_inactive(&inputs),active<=0);
            let expected=if active<=0 {"inactive-fire-domain"} else {"invalid-active-fire-domain"};
            assert!(catalog(&inputs).iter().all(|row|row.4==expected));
            let (_,skipped,failed,_)=render(&vec!["fire_ros".into()],&RenderConfig {
                inputs:&inputs,out_dir:&folder,frame:None,width:120,height:100,
                source_label:"control",overlays:None,annotations:None}).unwrap();
            assert_eq!((skipped,failed),if active<=0 {(1,0)} else {(0,1)});
            if let Some(binary)=std::env::var_os("GPUWM_SFIRE_RENDERER_CONTROL_BIN") {
                let output=std::process::Command::new(binary).arg(&path).args(["--products","fire_ros","--store-root"])
                    .arg(&folder).arg("--out-dir").arg(&folder).output().unwrap();
                println!("native renderer IFIRE={active} status={} stdout={} stderr={}",output.status,
                    String::from_utf8_lossy(&output.stdout),String::from_utf8_lossy(&output.stderr));
                assert_eq!(output.status.success(),active<=0);
            }
            println!("deleted owned eligibility control {} bytes {}",path.file_name().unwrap().to_string_lossy(),std::fs::metadata(&path).unwrap().len());
            std::fs::remove_file(path).unwrap();
        }
        std::fs::remove_dir(folder).unwrap();
    }
    #[test]
    fn fire_maps_land_in_the_domain_folder_the_weather_maps_use() {
        use netcdf_writer::{AttrValue,NcFormat,NcType,NcWriter,Schema,VarData};
        let nonce=std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
        let folder=std::env::temp_dir().join(format!("sfire-folder-{nonce}"));
        std::fs::create_dir(&folder).unwrap();
        // WRF's `nocolons` spelling: a name holding `:` cannot be created
        // on Windows (os error 123), which failed this test on windows-2025.
        let path=folder.join("wrfout_d01_2025-01-07_18_00_00");
        let mut schema=Schema::new(NcFormat::Offset64);
        let record=schema.def_dim("Time",0,true).unwrap();
        let chars=schema.def_dim("DateStrLen",19,false).unwrap();
        let rows=schema.def_dim("south_north_subgrid",4,false).unwrap();
        let cols=schema.def_dim("west_east_subgrid",4,false).unwrap();
        let mass_rows=schema.def_dim("south_north",2,false).unwrap();
        let mass_cols=schema.def_dim("west_east",2,false).unwrap();
        schema.put_global_attr("IFIRE",AttrValue::Ints(vec![2])).unwrap();
        schema.put_global_attr("GRID_ID",AttrValue::Ints(vec![1])).unwrap();
        for name in ["DX","DY"] {schema.put_global_attr(name,AttrValue::Floats(vec![1000.0])).unwrap();}
        for name in ["SR_X","SR_Y"] {schema.put_global_attr(name,AttrValue::Ints(vec![10])).unwrap();}
        schema.put_global_attr("FIRE_COORDINATE_MODE",AttrValue::Text("geographic".into())).unwrap();
        let times=schema.def_var("Times",NcType::Char,&[record,chars]).unwrap();
        let mass:Vec<_>=["XLAT","XLONG"].iter()
            .map(|name|schema.def_var(name,NcType::Float,&[record,mass_rows,mass_cols]).unwrap()).collect();
        let planes:Vec<_>=["FXLAT","FXLONG","ZSF","LFN","ROS_FRONT"].iter()
            .map(|name|schema.def_var(name,NcType::Float,&[record,rows,cols]).unwrap()).collect();
        let mut writer=NcWriter::create(&path,schema).unwrap();
        writer.write_record(0,times,VarData::Char(b"2025-01-07_18:00:00")).unwrap();
        writer.write_record(0,mass[0],VarData::F32(&[34.05,34.05,34.06,34.06])).unwrap();
        writer.write_record(0,mass[1],VarData::F32(&[-118.55,-118.54,-118.55,-118.54])).unwrap();
        let lat:Vec<f32>=(0..16).map(|k|34.05+0.001*(k/4) as f32).collect();
        let lon:Vec<f32>=(0..16).map(|k|-118.55+0.001*(k%4) as f32).collect();
        let lfn:Vec<f32>=(0..16).map(|k|if k==5 {-1.0} else {1.0}).collect();
        for (var,values) in planes.iter().zip([lat,lon,vec![100.0;16],lfn,vec![0.2;16]]) {
            writer.write_record(0,*var,VarData::F32(&values)).unwrap();
        }
        writer.finish().unwrap();
        let inputs=vec![path.clone()];
        let (rendered,_,failed,_)=render(&vec!["fire_ros".into()],&RenderConfig {
            inputs:&inputs,out_dir:&folder,frame:None,width:120,height:100,
            source_label:"WOOF",overlays:None,annotations:None}).unwrap();
        assert_eq!((rendered,failed),(1,0));
        // `d01-1km`, the folder the same run's smoke and weather maps take.
        assert!(folder.join("d01-1km").join("fire_ros").join("20250107").join("fire_ros_20250107T180000.png").is_file());
        assert!(!folder.join("d01").exists());
        std::fs::remove_dir_all(folder).unwrap();
    }
    /// A fire map's path holds no character Windows refuses in a file name,
    /// for the WRF valid time spellings and for a malformed one.
    #[test]
    fn fire_map_paths_hold_no_character_windows_refuses() {
        let forbidden=|c:char|matches!(c,'<'|'>'|':'|'"'|'/'|'\\'|'|'|'?'|'*') || c.is_control();
        let root=Path::new("out");
        for (slug,_,_,_) in PRODUCTS {
            for (folder,time) in [("d01-1km","2025-01-07_18:00:00"),("d02-333m","2025-01-07_18_00_00"),
                ("native_grid","2025-01-07T18:00:00Z"),("d01","2025/01/07 18:00:00<>|?*\"\\\u{0}\t")] {
                let path=map_path(root,folder,slug,time);
                let parts:Vec<String>=path.strip_prefix(root).unwrap().components()
                    .map(|part|part.as_os_str().to_string_lossy().into_owned()).collect();
                assert_eq!(parts.len(),4,"{path:?}");
                for part in &parts {assert!(!part.is_empty() && !part.chars().any(forbidden),"{slug} {time:?}: {part:?}");}
            }
        }
        // The spelling the fire maps have always had on Linux.
        assert_eq!(map_path(root,"d01-1km","fire_ros","2025-01-07_18:00:00"),
            root.join("d01-1km").join("fire_ros").join("20250107").join("fire_ros_20250107T180000.png"));
    }
    #[test]
    fn zero_fire_keeps_terrain_context_and_positive_ros_keeps_legend_color() {
        let terrain=vec![0.,100.,0.,100.];let scale=scale("fire_ros");
        let zero=context_pixels(&[0.;4],&terrain,[2,2],50.,50.,&scale).unwrap();
        assert!(zero.iter().all(|pixel|pixel.a==255 && pixel.r>160 && pixel.g>pixel.r));
        let lit=context_pixels(&[1.;4],&terrain,[2,2],50.,50.,&scale).unwrap();
        let legend=rustwx_render::build_colormap(&scale,rustwx_render::ColormapBuildOptions {
            render_density:rustwx_render::StaticPlotStyle::from_env().render_density(Default::default()),legend:Default::default()}).map(1.);
        assert!(lit.iter().all(|pixel|*pixel==Color::rgba(legend.r,legend.g,legend.b,legend.a)));
        let varied=context_pixels(&[0.;4],&[0.,100.,0.,0.],[2,2],50.,50.,&scale).unwrap();
        assert!(varied.windows(2).any(|pair|pair[0]!=pair[1]));
        assert!(context_pixels(&[0.;4],&[f32::NAN;4],[2,2],50.,50.,&scale).is_err());
    }
    #[test]
    fn fire_products_keep_the_refined_lane_beside_atmospheric_products() {
        assert_eq!(split_products("t2m,fire_ros,fire_perimeter"),("t2m".into(),vec!["fire_perimeter".into(),"fire_ros".into()]));
        assert_eq!(split_products("all").1.len(),3);
        assert_eq!(split_products("fire_ros,fire_ros").1.len(),1);
    }
}
