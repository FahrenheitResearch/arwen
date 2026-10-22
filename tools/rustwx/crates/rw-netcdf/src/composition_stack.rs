//! Table-selected composition columns. NetCDF decode, CF time and pressure
//! reconstruction stay in Rust; no dataset or species dispatch lives here.
use serde::{Deserialize, Serialize};
use std::{collections::BTreeSet, fs, io::Write, path::{Path, PathBuf}};
use chrono::DateTime;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    schema: String,
    files: Vec<Input>,
    latitude: String, longitude: String, time: String, level: String,
    variables: Vec<Selection>,
    pressure: Option<Pressure>,
    pressure_levels: Option<PressureLevels>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Selection { key: String, name: String, units: String, group: String }
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Input { path: PathBuf, group: String }
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PressureLevels { output: String, units: String, scale_pa: f64 }
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Pressure {
    thickness: String, surface: String, output: String, order: String,
    top_pressure_pa: Option<f64>, expected_layers: Option<usize>,
}
#[derive(Serialize)]
struct PressureReceipt {
    valid_time: String, reference: &'static str, top_pressure_pa: Option<f64>,
    columns: usize, surface_difference_min_pa: f64, surface_difference_max_pa: f64,
}
#[derive(Clone)]
struct Header {
    path: PathBuf, latitude: Vec<f64>, longitude: Vec<f64>, levels: Vec<f64>,
    times: Vec<String>, dimensions: Vec<String>, present: BTreeSet<String>,
}
#[derive(Serialize)]
struct Output { key: String, valid_time: String, levels: Vec<f64>, file: String }
#[derive(Serialize)]
struct Stack {
    schema: &'static str, latitude: Vec<f64>, longitude: Vec<f64>, outputs: Vec<Output>,
    workers: usize, pressure_reconstruction: Vec<PressureReceipt>,
}
fn checked(file: &netcrust::File, name: &str, units: Option<&str>) -> Result<(Vec<f64>, Vec<usize>), String> {
    if let Some(expected) = units {
        let actual = file.variable(name).and_then(|v| v.attribute("units").and_then(|a|a.as_string()).map(str::to_owned));
        if actual.as_deref() != Some(expected) {
            return Err(format!("{name}: units {actual:?}, expected {expected}"));
        }
    }
    super::compose_source::values(file, name)
}
fn axis(file: &netcrust::File, name: &str) -> Result<(Vec<f64>, String), String> {
    let (values, shape) = checked(file, name, None)?;
    if shape.len()!=1 || values.is_empty() { return Err(format!("{name}: coordinate must be a nonempty 1-D axis")); }
    let var=file.variable(name).ok_or_else(||format!("missing {name}"))?;
    if values.len()>1 {
        let direction=(values[1]-values[0]).signum();
        if direction==0. || values.windows(2).any(|p|(p[1]-p[0]).signum()!=direction) {
            return Err(format!("{name}: repeated or unordered coordinate"));
        }
    }
    Ok((values, var.dimensions()[0].name().to_owned()))
}
fn header(m: &Manifest, input: &Input) -> Result<Header,String> {
    let path=&input.path;
    let f=netcrust::File::open(path).map_err(|e|format!("{}: {e}",path.display()))?;
    let (lat,yd)=axis(&f,&m.latitude)?;
    let (lon,xd)=axis(&f,&m.longitude)?;
    let (levels,zd)=if f.variable(&m.level).is_some() {axis(&f,&m.level)?} else {(Vec::new(),String::new())};
    if let Some(p)=&m.pressure_levels { if !levels.is_empty(){ checked(&f,&m.level,Some(&p.units))?; } }
    let (time,td)=axis(&f,&m.time)?;
    let var=f.variable(&m.time).ok_or("missing time")?;
    let units=var.attribute("units").and_then(|a|a.as_string()).ok_or("time has no CF units")?;
    let calendar=var.attribute("calendar").and_then(|a|a.as_string()).unwrap_or("standard");
    if !["standard","gregorian","proleptic_gregorian"].contains(&calendar) { return Err(format!("unsupported calendar {calendar}")); }
    let reference=super::parse_cf_units(units)?.ok_or("time has no CF reference")?;
    let times=super::decode_times(&reference,&time)?;
    let present=m.variables.iter().filter(|s|s.group==input.group && f.variable(&s.name).is_some()).map(|s|s.key.clone()).collect();
    Ok(Header { path:path.to_owned(),latitude:lat,longitude:lon,levels,times,dimensions:vec![td,zd,yd,xd],present })
}
fn write_f32(path: &Path, values: &[f64]) -> Result<(),String> {
    let mut bytes=Vec::with_capacity(values.len()*4);
    for &v in values {
        let x=v as f32;
        if !x.is_finite() { return Err("value overflows float32".into()); }
        bytes.extend_from_slice(&x.to_le_bytes());
    }
    fs::File::create(path).and_then(|mut f|f.write_all(&bytes)).map_err(|e|e.to_string())
}
fn read_f32(path:&Path)->Result<Vec<f64>,String>{
    let bytes=fs::read(path).map_err(|e|e.to_string())?;
    if bytes.len()%4!=0 { return Err("partial float32 payload".into()); }
    Ok(bytes.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap()) as f64).collect())
}
fn pressure(dp:&[f64], ps:&[f64], nz:usize, order:&str, top:Option<f64>)->Result<(Vec<f64>,PressureReceipt),String>{
    if nz==0 || ps.is_empty() || dp.len()!=nz*ps.len() { return Err("pressure shape mismatch".into()); }
    if order!="top_down" && order!="bottom_up" { return Err("pressure order must be top_down or bottom_up".into()); }
    if top.is_some_and(|p| !p.is_finite() || p<=0.) {return Err("nonpositive declared top pressure".into());}
    let n=ps.len();
    let mut result=vec![0.;dp.len()];
    let mut min_difference=f64::INFINITY;
    let mut max_difference=f64::NEG_INFINITY;
    for i in 0..n {
        if !ps[i].is_finite() || ps[i]<=0. { return Err("nonpositive surface pressure".into()); }
        let mut edge=top.unwrap_or(ps[i]);
        for z in 0..nz {
            let k=if (order=="top_down")==top.is_some() {z} else {nz-1-z};
            let d=dp[k*n+i];
            if !d.is_finite() || d<=0. { return Err("nonpositive layer pressure thickness".into()); }
            // Some products explicitly define edges by DELP integrated from a
            // fixed model top. PS is independently packed/regridded; subtracting
            // DELP from it can produce negative upper layers. Never rescale DELP
            // or clamp those layers: use the declared pressure reference.
            let step=if top.is_some() {d} else {-d};
            result[k*n+i]=edge+step*0.5;
            edge+=step;
            if !edge.is_finite() || result[k*n+i]<=0. || (top.is_none() && edge < -0.1) {
                return Err(format!("layer thickness exceeds surface pressure at column {i}, layer {k}: edge={edge} Pa"));
            }
        }
        if top.is_none() && edge>100. { return Err("incomplete pressure column: more than 100 Pa above supplied levels".into()); }
        let difference=if top.is_some(){edge-ps[i]}else{0.};
        min_difference=min_difference.min(difference);
        max_difference=max_difference.max(difference);
    }
    Ok((result,PressureReceipt {valid_time:String::new(),reference:if top.is_some(){"declared_top"}else{"supplied_surface"},top_pressure_pa:top,columns:n,surface_difference_min_pa:min_difference,surface_difference_max_pa:max_difference}))
}
pub(super) fn run(manifest:&Path,out:&Path,jobs:usize)->Result<(),String>{
    let m:Manifest=serde_json::from_slice(&fs::read(manifest).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;
    if m.schema!="gpuwm-composition-stack-v1" || m.files.is_empty() || m.variables.is_empty() || jobs==0 || jobs>64 {
        return Err("invalid composition manifest or worker count (1..64)".into());
    }
    let mut names=BTreeSet::new();
    for s in &m.variables {
        if s.key.is_empty() || !names.insert(&s.key) { return Err("empty or duplicate selection key".into()); }
    }
    let headers=m.files.iter().map(|p|header(&m,p)).collect::<Result<Vec<_>,_>>()?;
    let first=headers.iter().find(|h|!h.levels.is_empty()).ok_or("no vertical coordinate in composition files")?;
    for h in &headers {
        if h.latitude!=first.latitude || h.longitude!=first.longitude || (!h.levels.is_empty() && h.levels!=first.levels) {
            return Err("composition files have different horizontal or vertical grids".into());
        }
    }
    if out.exists() { return Err(format!("{} already exists",out.display())); }
    fs::create_dir_all(out).map_err(|e|e.to_string())?;
    let mut tasks=Vec::new();
    for (i,h) in headers.iter().enumerate(){
        for (j,s) in m.variables.iter().enumerate(){
            if h.present.contains(&s.key){tasks.push((i,j));}
        }
    }
    let next=std::sync::atomic::AtomicUsize::new(0);
    let slots:Vec<_>=(0..tasks.len()).map(|_|std::sync::Mutex::new(None)).collect();
    let workers=jobs.min(tasks.len()).max(1);
    std::thread::scope(|scope| {
        for _ in 0..workers {
            scope.spawn(||loop {
                let t=next.fetch_add(1,std::sync::atomic::Ordering::Relaxed);
                if t>=tasks.len(){break;}
                let (i,j)=tasks[t]; let h=&headers[i]; let s=&m.variables[j];
                let answer=(||->Result<Vec<Output>,String>{
                    let f=netcrust::File::open(&h.path).map_err(|e|e.to_string())?;
                    let (v,shape)=checked(&f,&s.name,Some(&s.units))?;
                    let dims=f.variable(&s.name).unwrap().dimensions().iter().map(|d|d.name().to_owned()).collect::<Vec<_>>();
                    let surface=shape.len()==3;
                    let wanted=if surface {vec![h.dimensions[0].clone(),h.dimensions[2].clone(),h.dimensions[3].clone()]} else {h.dimensions.clone()};
                    let nz=if surface {1} else {h.levels.len()};
                    let plane=nz*h.latitude.len()*h.longitude.len();
                    let expected=if surface {vec![h.times.len(),h.latitude.len(),h.longitude.len()]} else {vec![h.times.len(),nz,h.latitude.len(),h.longitude.len()]};
                    if dims!=wanted || shape!=expected || v.len()!=plane*h.times.len() {
                        return Err(format!("{}: dimension order/shape mismatch {dims:?} {shape:?}",s.name));
                    }
                    let mut result=Vec::new();
                    for (k,time) in h.times.iter().enumerate(){
                        DateTime::parse_from_rfc3339(time).map_err(|e|e.to_string())?;
                        let filename=format!("{t:04}-{k:04}.f32");
                        write_f32(&out.join(&filename),&v[k*plane..(k+1)*plane])?;
                        result.push(Output {key:s.key.clone(),valid_time:time.clone(),levels:if surface {vec![0.]} else {h.levels.clone()},file:filename});
                    }
                    Ok(result)
                })();
                *slots[t].lock().unwrap()=Some(answer);
            });
        }
    });
    let mut outputs=Vec::new();
    let mut seen=BTreeSet::new();
    for slot in slots {
        for o in slot.into_inner().unwrap().unwrap()? {
            if !seen.insert((o.key.clone(),o.valid_time.clone())) {return Err(format!("duplicate {} at {}",o.key,o.valid_time));}
            outputs.push(o);
        }
    }
    let times=outputs.iter().map(|o|o.valid_time.clone()).collect::<BTreeSet<_>>();
    for time in &times {
        for s in &m.variables {
            if !seen.contains(&(s.key.clone(),time.clone())) {return Err(format!("missing {} at {time}",s.key));}
        }
    }
    if times.is_empty(){return Err("no selected composition data".into());}
    if m.pressure.is_some() && m.pressure_levels.is_some(){return Err("two pressure constructors selected".into());}
    if let Some(p)=&m.pressure_levels {
        if names.contains(&p.output) || !p.scale_pa.is_finite() || p.scale_pa<=0. {return Err("invalid pressure-level constructor".into());}
        let plane=first.latitude.len()*first.longitude.len();
        let values=first.levels.iter().flat_map(|z|std::iter::repeat(*z*p.scale_pa).take(plane)).collect::<Vec<_>>();
        if values.iter().any(|p|!p.is_finite() || *p<=0.){return Err("nonpositive pressure level".into());}
        for (i,time) in times.iter().enumerate(){
            let filename=format!("pressure-{i:04}.f32");
            write_f32(&out.join(&filename),&values)?;
            outputs.push(Output{key:p.output.clone(),valid_time:time.clone(),levels:first.levels.clone(),file:filename});
        }
    }
    let mut pressure_reconstruction=Vec::new();
    if let Some(p)=&m.pressure {
        if p.expected_layers.is_some_and(|n|n!=first.levels.len()) {return Err("pressure constructor requires its declared complete layer count".into());}
        if names.contains(&p.output) {return Err("derived pressure collides with a selected variable".into());}
        for (i,time) in times.iter().enumerate(){
            let dp=outputs.iter().find(|o|o.key==p.thickness && o.valid_time==*time).ok_or("missing pressure thickness")?;
            let ps=outputs.iter().find(|o|o.key==p.surface && o.valid_time==*time).ok_or("missing surface pressure")?;
            if dp.levels!=first.levels || ps.levels!=vec![0.] {return Err("pressure variables have wrong ranks".into());}
            let (values,mut receipt)=pressure(&read_f32(&out.join(&dp.file))?,&read_f32(&out.join(&ps.file))?,first.levels.len(),&p.order,p.top_pressure_pa)?;
            receipt.valid_time=time.clone();
            pressure_reconstruction.push(receipt);
            let filename=format!("pressure-{i:04}.f32");
            write_f32(&out.join(&filename),&values)?;
            outputs.push(Output{key:p.output.clone(),valid_time:time.clone(),levels:first.levels.clone(),file:filename});
        }
    }
    outputs.sort_by(|a,b|(&a.valid_time,&a.key).cmp(&(&b.valid_time,&b.key)));
    let stack=Stack{schema:"gpuwm-composition-stack-v1",latitude:first.latitude.clone(),longitude:first.longitude.clone(),outputs,workers,pressure_reconstruction};
    fs::write(out.join("stack.json"),serde_json::to_vec_pretty(&stack).map_err(|e|e.to_string())?).map_err(|e|e.to_string())
}
#[cfg(test)]
mod tests{
    use super::*;
    #[test]
    fn pressure_uses_each_column_and_both_orders(){
        let x=pressure(&[10000.,20000.,89999.,79999.],&[100000.,100000.],2,"top_down",None).unwrap().0;
        assert_eq!(x,vec![5001.,10001.,55000.5,60000.5]);
        let y=pressure(&[89999.,79999.,10000.,20000.],&[100000.,100000.],2,"bottom_up",None).unwrap().0;
        assert_eq!(y,vec![55000.5,60000.5,5001.,10001.]);
    }
    #[test]
    fn incomplete_or_unphysical_pressure_is_refused(){
        assert!(pressure(&[1000.],&[100000.],1,"top_down",None).is_err());
        assert!(pressure(&[-1.],&[100000.],1,"top_down",None).is_err());
        assert!(pressure(&[200000.],&[100000.],1,"top_down",None).is_err());
    }
    #[test]
    fn declared_top_prevents_independent_surface_roundoff_from_corrupting_upper_layers(){
        let (top,audit)=pressure(&[1.,99999.],&[99965.],2,"top_down",Some(1.)).unwrap();
        assert_eq!(top,vec![1.5,50001.5]);
        assert_eq!(audit.surface_difference_min_pa,36.);
        let (bottom,_)=pressure(&[99999.,1.],&[100036.],2,"bottom_up",Some(1.)).unwrap();
        assert_eq!(bottom,vec![50001.5,1.5]);
        assert!(pressure(&[1.,99999.],&[99965.],2,"top_down",None).is_err());
        assert!(pressure(&[1.,99999.],&[99965.],2,"top_down",Some(0.)).is_err());
    }
    use netcdf_writer::{AttrValue,NcFormat,NcType,NcWriter,Schema,VarData};
    fn fixture(path:&Path, group:&str, times:&[f64], lon0:f64, aerosol_units:&str) {
        let mut schema=Schema::new(NcFormat::Classic);
        let t=schema.def_dim("clock",times.len(),false).unwrap();
        let z=schema.def_dim("height",2,false).unwrap();
        let y=schema.def_dim("row",2,false).unwrap();
        let x=schema.def_dim("column",2,false).unwrap();
        let mut variable=|name,dims:&[usize],units:&str| {
            let id=schema.def_var(name,NcType::Double,dims).unwrap();
            schema.put_var_attr(id,"units",AttrValue::Text(units.into())).unwrap();id
        };
        let time=variable("stamp",&[t],"hours since 2022-03-15 00:00:00");
        let lev=variable("layer",&[z],"hPa");
        let lat=variable("north",&[y],"degrees_north");
        let lon=variable("east",&[x],"degrees_east");
        let specs=if group=="mass" {
            vec![("tracer",vec![t,z,y,x],aerosol_units),("dp",vec![t,z,y,x],"Pa"),("ps",vec![t,y,x],"Pa")]
        } else {vec![("water",vec![t,z,y,x],"kg kg-1")]};
        let vars=specs.iter().map(|(n,d,u)|variable(n,d,u)).collect::<Vec<_>>();
        let mut w=NcWriter::create(path,schema).unwrap();
        for (id,v) in [(time,times.to_vec()),(lev,vec![100.,900.]),(lat,vec![40.,39.]),(lon,vec![lon0,lon0+1.])] {
            w.write_var(id,VarData::F64(&v)).unwrap();
        }
        for ((name,_,_),id) in specs.iter().zip(vars) {
            let mut v=Vec::new();
            for (k,_) in times.iter().enumerate() {
                match *name {
                    "ps"=>v.extend([100000.,90000.,80000.,70000.]),
                    "dp"=>v.extend([10000.,20000.,30000.,40000.,89999.,69999.,49999.,29999.]),
                    "tracer"=>v.extend((0..8).map(|i|1e-9*(1+i+8*k) as f64)),
                    _=>v.extend(vec![0.01+k as f64*0.001;8]),
                }
            }
            w.write_var(id,VarData::F64(&v)).unwrap();
        }
        w.finish().unwrap();
    }
    #[test]
    fn netcdf_multifile_groups_decode_cf_time_and_column_pressure_and_refuse_bad_inputs() {
        let root=std::env::temp_dir().join(format!("composition-stack-{}",std::process::id()));
        fs::create_dir_all(&root).unwrap();
        let a=root.join("mass.nc");let b=root.join("water.nc");
        fixture(&a,"mass",&[0.,3.],-4.,"kg kg-1");
        fixture(&b,"water",&[0.,3.],-4.,"kg kg-1");
        let manifest=root.join("manifest.json");
        let doc=serde_json::json!({"schema":"gpuwm-composition-stack-v1",
            "files":[{"path":a,"group":"mass"},{"path":b,"group":"moisture"}],
            "latitude":"north","longitude":"east","time":"stamp","level":"layer",
            "variables":[
                {"key":"dust","name":"tracer","group":"mass","units":"kg kg-1"},
                {"key":"thickness","name":"dp","group":"mass","units":"Pa"},
                {"key":"surface","name":"ps","group":"mass","units":"Pa"},
                {"key":"humidity","name":"water","group":"moisture","units":"kg kg-1"}],
            "pressure":{"thickness":"thickness","surface":"surface","output":"pressure","order":"top_down"}});
        fs::write(&manifest,serde_json::to_vec(&doc).unwrap()).unwrap();
        let out=root.join("good");run(&manifest,&out,4).unwrap();
        let stack:serde_json::Value=serde_json::from_slice(&fs::read(out.join("stack.json")).unwrap()).unwrap();
        assert_eq!(stack["workers"],4);
        assert_eq!(stack["latitude"],serde_json::json!([40.,39.]));
        assert_eq!(stack["outputs"].as_array().unwrap().len(),10);
        let output=|key:&str,time:&str| stack["outputs"].as_array().unwrap().iter()
            .find(|o|o["key"]==key && o["valid_time"]==time).unwrap()["file"].as_str().unwrap().to_owned();
        assert_eq!(read_f32(&out.join(output("pressure","2022-03-15T00:00:00Z"))).unwrap(),
                   vec![5001.,10001.,15001.,20001.,55000.5,55000.5,55000.5,55000.5]);
        let mass=read_f32(&out.join(output("dust","2022-03-15T03:00:00Z"))).unwrap();
        assert!((mass[0]-9e-9).abs()<1e-15 && (mass[7]-16e-9).abs()<1e-15);
        fixture(&b,"water",&[0.],-4.,"kg kg-1");
        assert!(run(&manifest,&root.join("gap"),4).unwrap_err().contains("missing humidity"));
        fixture(&b,"water",&[0.,3.],-3.,"kg kg-1");
        assert!(run(&manifest,&root.join("grid"),4).unwrap_err().contains("different horizontal"));
        fixture(&b,"water",&[0.,3.],-4.,"kg kg-1");
        fixture(&a,"mass",&[0.,3.],-4.,"wrong");
        assert!(run(&manifest,&root.join("units"),4).unwrap_err().contains("expected kg kg-1"));
        fixture(&a,"mass",&[0.,3.],-4.,"kg kg-1");
        let mut duplicate=doc.clone();duplicate["files"].as_array_mut().unwrap().push(doc["files"][0].clone());
        fs::write(&manifest,serde_json::to_vec(&duplicate).unwrap()).unwrap();
        assert!(run(&manifest,&root.join("duplicate"),4).unwrap_err().contains("duplicate dust"));
        let mut levels=doc.clone();levels.as_object_mut().unwrap().remove("pressure");
        levels["pressure_levels"]=serde_json::json!({"output":"pressure","units":"hPa","scale_pa":100.});
        fs::write(&manifest,serde_json::to_vec(&levels).unwrap()).unwrap();
        let out2=root.join("levels");run(&manifest,&out2,2).unwrap();
        assert_eq!(read_f32(&out2.join("pressure-0000.f32")).unwrap(),vec![10000.,10000.,10000.,10000.,90000.,90000.,90000.,90000.]);
        fs::remove_dir_all(&root).unwrap();
    }

}
