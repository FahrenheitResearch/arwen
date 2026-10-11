// Native MPAS-cell comparisons. All arithmetic remains in Rust.
// Pressure products use log-pressure interpolation without extrapolation.
use std::ffi::{CString, CStr, c_char, c_int};
use std::fs;

#[link(name = "netcdf")]
unsafe extern "C" {
    fn nc_open(path:*const c_char, mode:c_int, id:*mut c_int)->c_int;
    fn nc_close(id:c_int)->c_int;
    fn nc_strerror(code:c_int)->*const c_char;
    fn nc_inq_varid(id:c_int,name:*const c_char,var:*mut c_int)->c_int;
    fn nc_inq_varndims(id:c_int,var:c_int,rank:*mut c_int)->c_int;
    fn nc_inq_vardimid(id:c_int,var:c_int,dims:*mut c_int)->c_int;
    fn nc_inq_dimlen(id:c_int,dim:c_int,len:*mut usize)->c_int;
    fn nc_inq_dimname(id:c_int,dim:c_int,name:*mut c_char)->c_int;
    fn nc_inq_vartype(id:c_int,var:c_int,dtype:*mut c_int)->c_int;
    fn nc_get_var_double(id:c_int,var:c_int,data:*mut f64)->c_int;
    fn nc_get_var_text(id:c_int,var:c_int,data:*mut c_char)->c_int;
    fn nc_inq_attlen(id:c_int,var:c_int,name:*const c_char,len:*mut usize)->c_int;
    fn nc_get_att_double(id:c_int,var:c_int,name:*const c_char,data:*mut f64)->c_int;
}
fn check(code:c_int)->Result<(),String>{
    if code==0{Ok(())}else{Err(unsafe{CStr::from_ptr(nc_strerror(code))}.to_string_lossy().into_owned())}
}
struct File{ id:c_int, path:String }
impl Drop for File{fn drop(&mut self){unsafe{nc_close(self.id);}}}
#[derive(Clone)]
struct Array{ shape:Vec<usize>, values:Vec<f64> }
impl File {
    fn open(path:&str)->Result<Self,String>{let p=CString::new(path).map_err(|e|e.to_string())?;let mut id=0;unsafe{check(nc_open(p.as_ptr(),0,&mut id))?;}Ok(Self{id,path:path.into()})}
    fn variable(&self,name:&str)->Option<c_int>{let name=CString::new(name).ok()?;let mut var=0;if unsafe{nc_inq_varid(self.id,name.as_ptr(),&mut var)}==0{Some(var)}else{None}}
    fn shape(&self,var:c_int)->Result<Vec<usize>,String>{let mut rank=0;unsafe{check(nc_inq_varndims(self.id,var,&mut rank))?;}let mut dims=vec![0;rank as usize];unsafe{check(nc_inq_vardimid(self.id,var,dims.as_mut_ptr()))?;}dims.into_iter().map(|d|{let mut n=0;let mut name=[0 as c_char;257];unsafe{check(nc_inq_dimlen(self.id,d,&mut n))?;check(nc_inq_dimname(self.id,d,name.as_mut_ptr()))?;}let name=unsafe{CStr::from_ptr(name.as_ptr())}.to_string_lossy();if name.eq_ignore_ascii_case("time")&&n!=1{return Err(format!("{} contains {n} time records; exactly one is required",self.path));}Ok(n)}).collect()}
    fn numeric_attribute(&self,var:c_int,name:&str)->Option<Vec<f64>>{let name=CString::new(name).ok()?;let mut n=0;if unsafe{nc_inq_attlen(self.id,var,name.as_ptr(),&mut n)}!=0||n==0{return None;}let mut values=vec![0.;n];if unsafe{nc_get_att_double(self.id,var,name.as_ptr(),values.as_mut_ptr())}!=0{return None;}Some(values)}
    fn array(&self,names:&[&str])->Result<Array,String>{for name in names{if let Some(var)=self.variable(name){let shape=self.shape(var)?;let count:usize=shape.iter().product();let mut values=vec![0.;count];let mut dtype=0;unsafe{check(nc_get_var_double(self.id,var,values.as_mut_ptr()))?;check(nc_inq_vartype(self.id,var,&mut dtype))?;}let mut fills:Vec<f64>=["_FillValue","missing_value"].iter().filter_map(|name|self.numeric_attribute(var,name)).flatten().collect();if dtype==5||dtype==6{fills.push(9.969209968386869e36);}let scale=self.numeric_attribute(var,"scale_factor").map(|v|v[0]).unwrap_or(1.);let offset=self.numeric_attribute(var,"add_offset").map(|v|v[0]).unwrap_or(0.);for value in &mut values{if fills.contains(value){*value=f64::NAN;}else{*value=*value*scale+offset;}}return Ok(Array{shape,values});}}Err(format!("{} missing {}",self.path,names.join("|")))}
    fn timestamp(&self)->Result<String,String>{for name in ["xtime","Times"]{if let Some(var)=self.variable(name){let count:usize=self.shape(var)?.iter().product();let mut text=vec![0u8;count];unsafe{check(nc_get_var_text(self.id,var,text.as_mut_ptr().cast()))?;}return Ok(String::from_utf8_lossy(&text).trim_matches(|c|c=='\0'||c==' ').into());}}let name=std::path::Path::new(&self.path).file_name().unwrap().to_string_lossy();filename_timestamp(&name).ok_or_else(||format!("{} has no xtime/Times or timestamped filename",self.path))}
}
fn filename_timestamp(name:&str)->Option<String>{let b=name.as_bytes();for start in 0..b.len().saturating_sub(18){let s=&b[start..start+19];if s[4]!=b'-'||s[7]!=b'-'||s[10]!=b'_'||![b'.',b':'].contains(&s[13])||![b'.',b':'].contains(&s[16]){continue;}if (0..19).filter(|i|![4,7,10,13,16].contains(i)).any(|i|!s[i].is_ascii_digit()){continue;}let mut v=s.to_vec();v[13]=b':';v[16]=b':';return String::from_utf8(v).ok();}None}
fn surface(history:&File,diag:Option<&File>,names:&[&str],n:usize)->Result<Vec<f64>,String>{let a=history.array(names).or_else(|e|if let Some(d)=diag{d.array(names)}else{Err(e)})?;if a.values.len()!=n{return Err(format!("surface {} has {} values, expected {n}",names[0],a.values.len()));}Ok(a.values)}
fn volume(file:&File,names:&[&str],n:usize)->Result<(Vec<f64>,usize),String>{let a=file.array(names)?;let shape:Vec<usize>=a.shape.into_iter().filter(|&x|x!=1).collect();if shape.len()!=2||shape[0]!=n{return Err(format!("{} expected cell-major nCells,nLevels; got {shape:?}",names[0]));}Ok((a.values,shape[1]))}

fn interpolate(p:&[f64],v:&[f64],target:f64)->f64{
    if p.len()!=v.len()||p.len()<2{return f64::NAN;}
    for k in 0..p.len()-1{
        if !p[k].is_finite()||!p[k+1].is_finite()||p[k]<=0.||p[k+1]<=0.||!v[k].is_finite()||!v[k+1].is_finite(){continue;}
        if (p[k]>=target&&p[k+1]<=target)||(p[k]<=target&&p[k+1]>=target){
            let denom=p[k+1].ln()-p[k].ln();
            if denom==0.{continue;}
            let weight=(target.ln()-p[k].ln())/denom;
            return v[k]+weight*(v[k+1]-v[k]);
        }
    }
    f64::NAN
}
fn coordinate_difference(left:f64,right:f64,longitude:bool)->f64{
    let d=left-right;
    if longitude { d.sin().atan2(d.cos()).abs() } else { d.abs() }
}
fn pressure_product(history:&File,init:&File,n:usize,product:&str)->Result<Vec<f64>,String>{
    let (p,levels)=volume(history,&["pressure"],n)?;
    let target=if product.ends_with("850"){85000.}else{50000.};
    let v=match product{
        "z500"=>{let (z,zlevels)=volume(init,&["zgrid"],n)?;if zlevels!=levels+1{return Err("zgrid interfaces disagree with pressure levels".into());}let mut out=vec![0.;n*levels];for c in 0..n{for k in 0..levels{out[c*levels+k]=0.5*(z[c*zlevels+k]+z[c*zlevels+k+1]);}}out},
        "t850"=>{let (theta,tlevels)=volume(history,&["theta"],n)?;if tlevels!=levels{return Err("theta and pressure levels disagree".into());}theta.iter().zip(&p).map(|(t,p)|t*(p/100000.).powf(287./1004.5)).collect()},
        "u500"|"u850"=>{let(v,k)=volume(history,&["u_zonal","uReconstructZonal"],n)?;if k!=levels{return Err("wind and pressure levels disagree".into());}v},
        "v500"|"v850"=>{let(v,k)=volume(history,&["v_meridional","uReconstructMeridional"],n)?;if k!=levels{return Err("wind and pressure levels disagree".into());}v},
        _=>return Err("unknown pressure product".into())
    };
    Ok((0..n).map(|c|interpolate(&p[c*levels..(c+1)*levels],&v[c*levels..(c+1)*levels],target)).collect())
}
#[derive(Debug)]
struct Statistics{ count:usize, excluded:usize, bias:f64, rms:f64, area_bias:f64, area_rms:f64 }
fn statistics(left:&[f64],right:&[f64],areas:&[f64],mask:&[f64],interior:bool)->Statistics{
    let(mut count,mut eligible)=(0,0);let(mut sum,mut square,mut asum,mut asquare,mut weight)=(0.,0.,0.,0.,0.);
    let levels=left.len()/areas.len();
    for index in 0..left.len(){let c=index/levels;if interior&&mask[c]!=0.{continue;}eligible+=1;if !left[index].is_finite()||!right[index].is_finite(){continue;}let d=left[index]-right[index];let a=areas[c];count+=1;sum+=d;square+=d*d;asum+=a*d;asquare+=a*d*d;weight+=a;}
    Statistics{count,excluded:eligible-count,bias:sum/count as f64,rms:(square/count as f64).sqrt(),area_bias:asum/weight,area_rms:(asquare/weight).sqrt()}
}
fn escaped(s:&str)->String{format!("\"{}\"",s.replace('\\',"\\\\").replace('"',"\\\"").replace('\n',"\\n"))}
fn number(v:f64)->String{if v.is_finite(){format!("{v:.12}")}else{"null".into()}}
fn stat_json(s:&Statistics)->String{format!("{{\"count\":{},\"excluded\":{},\"mean_difference\":{},\"rms_difference\":{},\"area_mean_difference\":{},\"area_rms_difference\":{}}}",s.count,s.excluded,number(s.bias),number(s.rms),number(s.area_bias),number(s.area_rms))}
fn mesh_check(mesh:&str,reference:&str,output:&str)->Result<(),String>{
    if std::path::Path::new(output).exists(){return Err("output exists; use a new output path".into());}
    let left=File::open(mesh)?;let right=File::open(reference)?;let mut fields=Vec::new();
    let mut count=None;
    for name in ["latCell","lonCell","areaCell"]{
        let a=left.array(&[name])?;let b=right.array(&[name])?;
        if a.values.len()!=b.values.len()||a.values.is_empty(){return Err(format!("{name} paired cell counts disagree"));}
        if count.is_some_and(|n|n!=a.values.len()){return Err("mesh field cell counts disagree".into());}count=Some(a.values.len());
        let(mut maximum,mut max_relative)=(0.0_f64,0.0_f64);
        for(&x,&y)in a.values.iter().zip(&b.values){
            if !x.is_finite()||!y.is_finite()||(name=="areaCell"&&(x<=0.||y<=0.)){return Err(format!("invalid {name} cell"));}
            let d=coordinate_difference(x,y,name=="lonCell");maximum=maximum.max(d);
            max_relative=max_relative.max(d/x.abs().max(y.abs()).max(f64::MIN_POSITIVE));
        }
        let qualified=if name=="areaCell"{max_relative<=1e-6}else{maximum<=1e-6};
        fields.push(format!("{{\"field\":{},\"maximum_absolute_difference\":{},\"maximum_relative_difference\":{},\"passes\":{}}}",escaped(name),number(maximum),number(max_relative),qualified));
        if !qualified{return Err(format!("{name} exceeds mesh identity tolerance: absolute {maximum}, relative {max_relative}"));}
    }
    fs::write(output,format!("{{\"schema\":\"native-mpas-mesh-check/v1\",\"mesh\":{},\"reference\":{},\"cells\":{},\"coordinate_tolerance_radians\":1e-6,\"area_relative_tolerance\":1e-6,\"longitude\":\"modulo 2*pi\",\"fields\":[{}],\"status\":\"passed\"}}\n",escaped(mesh),escaped(reference),count.unwrap(),fields.join(","))).map_err(|e|e.to_string())
}
fn run()->Result<(),String>{
    let a:Vec<String>=std::env::args().skip(1).collect();
    if a.len()==4&&a[0]=="--mesh-check"{return mesh_check(&a[1],&a[2],&a[3]);}
    if a.len()!=8&&!(a.len()==10&&a[8]=="--raw"){return Err("usage: native_compare MESH LEFT_HISTORY RIGHT_HISTORY LEFT_INIT RIGHT_INIT LEFT_DIAG|- RIGHT_DIAG|- OUTPUT_JSON [--raw theta,qv,u_zonal,surface_pressure]".into());}
    if std::path::Path::new(&a[7]).exists(){return Err("output exists; use a new output path".into());}
    let mesh=File::open(&a[0])?;let left=File::open(&a[1])?;let right=File::open(&a[2])?;let li=File::open(&a[3])?;let ri=File::open(&a[4])?;
    let ld=if a[5]=="-"{None}else{Some(File::open(&a[5])?)};let rd=if a[6]=="-"{None}else{Some(File::open(&a[6])?)};
    let valid=left.timestamp()?;if right.timestamp()?!=valid{return Err("history valid times do not match".into());}
    for d in [&ld,&rd]{if let Some(d)=d{if d.timestamp()?!=valid{return Err("diagnostic valid time does not match history".into());}}}
    let areas=mesh.array(&["areaCell"])?;let mask=mesh.array(&["bdyMaskCell"])?;let n=areas.values.len();if mask.values.len()!=n{return Err("boundary mask and cell count disagree".into());}if areas.values.iter().any(|a|!a.is_finite()||*a<=0.){return Err("mesh areas must be positive finite values".into());}
    for coords in ["latCell","lonCell"]{let anchor=mesh.array(&[coords])?;for f in [&left,&right,&li,&ri]{if let Ok(c)=f.array(&[coords]){if c.values.len()!=n||c.values.iter().zip(&anchor.values).any(|(x,y)|!x.is_finite()||coordinate_difference(*x,*y,coords=="lonCell")>1e-6){return Err(format!("input {} {coords} differ from supplied mesh",f.path));}}else if f.path==li.path||f.path==ri.path{return Err(format!("initial file {} has no {coords}",f.path));}}}
    let mut rows=Vec::new();
    println!("product,scope,valid_time,samples,excluded,mean_difference,rms_difference,area_mean_difference,area_rms_difference,units");
    let products:Vec<(&str,&str)>=if a.len()==10{a[9].split(',').map(|p|(p,match p{"theta"|"t2m"=>"K","qv"=>"kg/kg","u_zonal"|"v_meridional"|"w"=>"m/s","surface_pressure"|"pressure"|"mslp"=>"Pa","rainc"|"rainnc"|"precipitation"=>"mm",_=>"source units"})).collect()}else{vec![("z500","m"),("t850","K"),("u500","m/s"),("v500","m/s"),("u850","m/s"),("v850","m/s"),("surface_pressure","Pa"),("t2m","K"),("precipitation","mm")]};
    for(product,units)in products{
        let field=|h:&File,i:&File,d:Option<&File>|->Result<Vec<f64>,String>{match product{
            "surface_pressure"=>surface(h,d,&["surface_pressure"],n),
            "t2m"=>surface(h,d,&["t2","t2m"],n),
            "mslp"|"rainnc"|"rainc"=>surface(h,d,&[product],n),
            "precipitation"=>{let nc=surface(h,d,&["rainnc"],n)?;let cu=surface(h,d,&["rainc"],n)?;Ok(nc.iter().zip(cu).map(|(a,b)|a+b).collect())},
            _ if a.len()==10=>{let names:Vec<&str>=match product{"u_zonal"=>vec!["u_zonal","uReconstructZonal"],"v_meridional"=>vec!["v_meridional","uReconstructMeridional"],_=>vec![product]};let(v,_)=volume(h,&names,n)?;Ok(v)},
            _=>pressure_product(h,i,n,product)
        }};
        match(field(&left,&li,ld.as_ref()),field(&right,&ri,rd.as_ref())){
            (Ok(l),Ok(r))=>{if l.len()!=r.len()||l.len()%n!=0{return Err(format!("{product} paired shapes differ or do not cover complete cells"));}let all=statistics(&l,&r,&areas.values,&mask.values,false);let interior=statistics(&l,&r,&areas.values,&mask.values,true);for(scope,s)in[("all",&all),("interior",&interior)]{println!("{product},{scope},{valid},{},{},{},{},{},{},{units}",s.count,s.excluded,number(s.bias),number(s.rms),number(s.area_bias),number(s.area_rms));}rows.push(format!("{{\"product\":{},\"units\":{},\"status\":\"compared\",\"all\":{},\"interior\":{}}}",escaped(product),escaped(units),stat_json(&all),stat_json(&interior)));},
            (l,r)=>{let reason=format!("left={:?}; right={:?}",l.err(),r.err());eprintln!("SKIPPED {product}: {reason}");rows.push(format!("{{\"product\":{},\"status\":\"missing-fields\",\"reason\":{}}}",escaped(product),escaped(&reason)));}
        }
    }
    let result=format!("{{\"schema\":\"native-mpas-comparison/v1\",\"difference\":\"left-minus-right\",\"valid_time\":{},\"valid_time_source\":\"xtime/Times when present; driver timestamped filename otherwise\",\"left\":{},\"right\":{},\"interior\":\"bdyMaskCell == 0\",\"masking\":\"_FillValue, missing_value, default floating NetCDF fill; no range clipping\",\"pressure_interpolation\":\"linear in log(P), no extrapolation; geometric height is interface midpoint; temperature theta*(P/100000)^(287/1004.5)\",\"precipitation\":\"rainc+rainnc, accumulated since model start\",\"products\":[{}]}}\n",escaped(&valid),escaped(&a[1]),escaped(&a[2]),rows.join(",\n"));
    fs::write(&a[7],result).map_err(|e|e.to_string())?;Ok(())
}
fn main(){if let Err(e)=run(){eprintln!("ERROR {e}");std::process::exit(1);}}

#[cfg(test)]
mod tests{
    use super::*;
    #[test]fn longitude_wrap_is_equivalent_but_a_displaced_cell_is_not(){assert!(coordinate_difference(-1.2,-1.2+std::f64::consts::TAU,true)<1e-14);assert!(coordinate_difference(-1.2,-1.19,true)>1e-3);assert!(coordinate_difference(0.5,0.5+std::f64::consts::TAU,false)>6.0);}
    #[test]fn log_pressure_and_no_extrapolation(){let p=[100000.,50000.];let v=[10.,30.];let mid=(100000f64*50000.).sqrt();assert!((interpolate(&p,&v,mid)-20.).abs()<1e-12);assert!(interpolate(&p,&v,110000.).is_nan());assert_eq!(interpolate(&p,&v,50000.),30.);}
    #[test]fn paired_masked_area_statistics(){let s=statistics(&[2.,4.,f64::NAN],&[1.,2.,0.],&[1.,3.,1.],&[0.,1.,0.],false);assert_eq!(s.count,2);assert_eq!(s.excluded,1);assert_eq!(s.bias,1.5);assert!((s.rms-2.5f64.sqrt()).abs()<1e-12);assert_eq!(s.area_bias,1.75);let i=statistics(&[2.,4.,f64::NAN],&[1.,2.,0.],&[1.,3.,1.],&[0.,1.,0.],true);assert_eq!(i.count,1);assert_eq!(i.bias,1.);assert_eq!(i.excluded,1);}
    #[test]fn timestamped_driver_capsule(){assert_eq!(filename_timestamp("cuda-history.2026-10-03_12.00.00.nc").unwrap(),"2026-10-03_12:00:00");assert!(filename_timestamp("history.badtime.nc").is_none());}
}
