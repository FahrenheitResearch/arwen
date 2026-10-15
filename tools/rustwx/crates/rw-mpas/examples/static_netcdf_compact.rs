//! Losslessly compact owned proof containers. All field I/O stays in Rust.
//! This is a verification/storage utility, not an MPAS arithmetic port.
use std::{ffi::{CStr,CString,c_char,c_int,c_void},path::Path};
#[link(name="netcdf")]
unsafe extern "C" {
 fn nc_open(p:*const c_char,mode:c_int,id:*mut c_int)->c_int;
 fn nc_create(p:*const c_char,mode:c_int,id:*mut c_int)->c_int;
 fn nc_close(id:c_int)->c_int;
 fn nc_strerror(code:c_int)->*const c_char;
 fn nc_inq(id:c_int,nd:*mut c_int,nv:*mut c_int,na:*mut c_int,un:*mut c_int)->c_int;
 fn nc_inq_unlimdims(id:c_int,n:*mut c_int,ids:*mut c_int)->c_int;
 fn nc_inq_dim(id:c_int,d:c_int,name:*mut c_char,len:*mut usize)->c_int;
 fn nc_def_dim(id:c_int,name:*const c_char,len:usize,d:*mut c_int)->c_int;
 fn nc_inq_var(id:c_int,v:c_int,name:*mut c_char,t:*mut c_int,n:*mut c_int,ds:*mut c_int,na:*mut c_int)->c_int;
 fn nc_def_var(id:c_int,name:*const c_char,t:c_int,n:c_int,ds:*const c_int,v:*mut c_int)->c_int;
 fn nc_inq_type(id:c_int,t:c_int,name:*mut c_char,size:*mut usize)->c_int;
 fn nc_inq_attname(id:c_int,v:c_int,a:c_int,name:*mut c_char)->c_int;
 fn nc_copy_att(src:c_int,v:c_int,name:*const c_char,dst:c_int,w:c_int)->c_int;
 fn nc_def_var_deflate(id:c_int,v:c_int,shuffle:c_int,deflate:c_int,level:c_int)->c_int;
 fn nc_enddef(id:c_int)->c_int;
 fn nc_get_vara(id:c_int,v:c_int,start:*const usize,count:*const usize,data:*mut c_void)->c_int;
 fn nc_put_vara(id:c_int,v:c_int,start:*const usize,count:*const usize,data:*const c_void)->c_int;
}
fn check(rc:c_int)->Result<(),String>{if rc==0 {Ok(())} else {Err(unsafe{CStr::from_ptr(nc_strerror(rc))}.to_string_lossy().into_owned())}}
struct File(c_int);
impl Drop for File{fn drop(&mut self){unsafe{nc_close(self.0);}}}
fn attributes(src:c_int,v:c_int,dst:c_int,w:c_int,n:c_int)->Result<(),String>{
 for a in 0..n {let mut name=[0 as c_char;257]; unsafe{check(nc_inq_attname(src,v,a,name.as_mut_ptr()))?;check(nc_copy_att(src,v,name.as_ptr(),dst,w))?;}}
 Ok(())
}
fn compact(input:&str,output:&str)->Result<(),String>{
 if Path::new(output).exists(){return Err("destination already exists".into());}
 let a=CString::new(input).map_err(|e|e.to_string())?;let b=CString::new(output).map_err(|e|e.to_string())?;
 let mut src=0;unsafe{check(nc_open(a.as_ptr(),0,&mut src))?;}let src=File(src);
 let mut nd=0;let mut nv=0;let mut na=0;let mut un=0;
 unsafe{check(nc_inq(src.0,&mut nd,&mut nv,&mut na,&mut un))?;check(nc_inq_unlimdims(src.0,&mut un,std::ptr::null_mut()))?;}
 let mut unlimited=vec![0;un as usize];unsafe{check(nc_inq_unlimdims(src.0,&mut un,unlimited.as_mut_ptr()))?;}
 let mut dst=0;unsafe{check(nc_create(b.as_ptr(),0x1000|4,&mut dst))?;}let dst=File(dst); // NETCDF4 | NOCLOBBER
 let mut lengths=vec![0;nd as usize];let mut dimensions=vec![0;nd as usize];
 for d in 0..nd{let mut name=[0 as c_char;257];unsafe{
  check(nc_inq_dim(src.0,d,name.as_mut_ptr(),&mut lengths[d as usize]))?;
  check(nc_def_dim(dst.0,name.as_ptr(),if unlimited.contains(&d){0}else{lengths[d as usize]},&mut dimensions[d as usize]))?;
 }}
 attributes(src.0,-1,dst.0,-1,na)?;
 let mut variables=Vec::new();
 for v in 0..nv{let mut name=[0 as c_char;257];let mut ty=0;let mut rank=0;let mut natts=0;
  unsafe{check(nc_inq_var(src.0,v,name.as_mut_ptr(),&mut ty,&mut rank,std::ptr::null_mut(),&mut natts))?;}
  if !(1..=11).contains(&ty){return Err(format!("unsupported native proof storage type {ty}"));}
  let mut dims=vec![0;rank as usize];unsafe{check(nc_inq_var(src.0,v,std::ptr::null_mut(),std::ptr::null_mut(),std::ptr::null_mut(),dims.as_mut_ptr(),std::ptr::null_mut()))?;}
  let dest_dims:Vec<_>=dims.iter().map(|&d|dimensions[d as usize]).collect();let mut w=0;let mut size=0;
  unsafe{check(nc_def_var(dst.0,name.as_ptr(),ty,rank,dest_dims.as_ptr(),&mut w))?;check(nc_inq_type(src.0,ty,std::ptr::null_mut(),&mut size))?;
   if rank>0 {check(nc_def_var_deflate(dst.0,w,1,1,1))?;}
  }
  attributes(src.0,v,dst.0,w,natts)?;
  variables.push((v,w,size,dims.iter().map(|&d|lengths[d as usize]).collect::<Vec<_>>()));
 }
 unsafe{check(nc_enddef(dst.0))?;}
 for (v,w,size,shape) in variables{
  if shape.contains(&0){continue;}
  let limit=(1024*1024/size).max(1);let mut axis=shape.len();let mut trailing=1usize;
  while axis>0 && trailing<=limit/shape[axis-1]{axis-=1;trailing*=shape[axis];}
  if axis>0 {axis-=1;}
  let mut start=vec![0usize;shape.len()];let mut count=shape.clone();
  for c in count.iter_mut().take(axis){*c=1;}
  if axis<shape.len(){let suffix:usize=shape[axis+1..].iter().product();count[axis]=shape[axis].min((limit/suffix).max(1));}
  let block=if axis<shape.len(){count[axis]}else{1};
  loop{
   if axis<shape.len(){count[axis]=block.min(shape[axis]-start[axis]);}
   let bytes=count.iter().try_fold(size,|n,&d|n.checked_mul(d)).ok_or("container shape overflow")?;
   let mut buffer=vec![0u64;bytes.div_ceil(8)];
   unsafe{check(nc_get_vara(src.0,v,start.as_ptr(),count.as_ptr(),buffer.as_mut_ptr().cast()))?;
    check(nc_put_vara(dst.0,w,start.as_ptr(),count.as_ptr(),buffer.as_ptr().cast()))?;}
   if shape.is_empty(){break;}
   start[axis]+=count[axis];if start[axis]<shape[axis]{continue;}start[axis]=0;
   let mut carried=true;
   for d in (0..axis).rev(){start[d]+=1;if start[d]<shape[d]{carried=false;break;}start[d]=0;}
   if carried{break;}
  }
 }
 drop(dst);drop(src);Ok(())
}

fn verify(input:&str,output:&str)->Result<serde_json::Value,String>{
 let open=|path:&str|->Result<File,String>{let name=CString::new(path).map_err(|e|e.to_string())?;let mut id=0;unsafe{check(nc_open(name.as_ptr(),0,&mut id))?;}Ok(File(id))};
 let (a,b)=(open(input)?,open(output)?);let mut fields=Vec::new();let mut totals=[0,0];
 let describe=|id:c_int|->Result<(c_int,c_int,Vec<(String,usize)>,Vec<c_int>),String>{
  let (mut nd,mut nv,mut na,mut un)=(0,0,0,0);unsafe{check(nc_inq(id,&mut nd,&mut nv,&mut na,std::ptr::null_mut()))?;check(nc_inq_unlimdims(id,&mut un,std::ptr::null_mut()))?;}
  let mut us=vec![0;un as usize];unsafe{check(nc_inq_unlimdims(id,&mut un,us.as_mut_ptr()))?;}
  let mut dims=Vec::new();for d in 0..nd{let mut name=[0 as c_char;257];let mut len=0;unsafe{check(nc_inq_dim(id,d,name.as_mut_ptr(),&mut len))?;}dims.push((unsafe{CStr::from_ptr(name.as_ptr())}.to_string_lossy().into_owned(),len));}
  Ok((nv,na,dims,us))
 };
 let (nv,na,dims,un)=describe(a.0)?;if describe(b.0)?!=(nv,na,dims.clone(),un.clone()){return Err("dimension, variable or attribute inventory changed".into());}
 for v in 0..nv {
  let metadata=|id:c_int|->Result<(String,c_int,Vec<c_int>,c_int,usize),String>{
   let mut name=[0 as c_char;257];let (mut ty,mut n,mut na)=(0,0,0);unsafe{check(nc_inq_var(id,v,name.as_mut_ptr(),&mut ty,&mut n,std::ptr::null_mut(),&mut na))?;}
   let mut ds=vec![0;n as usize];let mut size=0;unsafe{check(nc_inq_var(id,v,std::ptr::null_mut(),std::ptr::null_mut(),std::ptr::null_mut(),ds.as_mut_ptr(),std::ptr::null_mut()))?;check(nc_inq_type(id,ty,std::ptr::null_mut(),&mut size))?;}
   Ok((unsafe{CStr::from_ptr(name.as_ptr())}.to_string_lossy().into_owned(),ty,ds,na,size))
  };
  let meta=metadata(a.0)?;if metadata(b.0)?!=meta{return Err(format!("stored type or field dimensions changed: {}",meta.0));}
  let (name,ty,ds,_,size)=meta;let shape:Vec<_>=ds.iter().map(|&d|dims[d as usize].1).collect();let words=shape.iter().try_fold(1usize,|n,&d|n.checked_mul(d)).ok_or("shape overflow")?;
  let mut differing=0usize;
  if words>0 {
   let limit=(1024*1024/size).max(1);let mut axis=shape.len();let mut suffix=1usize;
   while axis>0 && suffix<=limit/shape[axis-1]{axis-=1;suffix*=shape[axis];}if axis>0{axis-=1;}
   let mut start=vec![0;shape.len()];let mut count=shape.clone();for c in count.iter_mut().take(axis){*c=1;}
   if axis<shape.len(){let trailing:usize=shape[axis+1..].iter().product();count[axis]=shape[axis].min((limit/trailing).max(1));}
   let block=if axis<shape.len(){count[axis]}else{1};
   loop {
    if axis<shape.len(){count[axis]=block.min(shape[axis]-start[axis]);}
    let bytes=count.iter().product::<usize>()*size;let mut x=vec![0u64;bytes.div_ceil(8)];let mut y=x.clone();
    unsafe{check(nc_get_vara(a.0,v,start.as_ptr(),count.as_ptr(),x.as_mut_ptr().cast()))?;check(nc_get_vara(b.0,v,start.as_ptr(),count.as_ptr(),y.as_mut_ptr().cast()))?;}
    let x=unsafe{std::slice::from_raw_parts(x.as_ptr().cast::<u8>(),bytes)};let y=unsafe{std::slice::from_raw_parts(y.as_ptr().cast::<u8>(),bytes)};
    differing+=x.chunks_exact(size).zip(y.chunks_exact(size)).filter(|(a,b)|a!=b).count();
    if shape.is_empty(){break;}start[axis]+=count[axis];if start[axis]<shape[axis]{continue;}start[axis]=0;let mut carried=true;
    for d in (0..axis).rev(){start[d]+=1;if start[d]<shape[d]{carried=false;break;}start[d]=0;}if carried{break;}
   }
  }
  totals[0]+=words;totals[1]+=differing;fields.push(serde_json::json!({"name":name,"stored_nc_type":ty,"n":words,"exact":words-differing,"different_words":differing,"dimensions":ds.iter().map(|&d|dims[d as usize].clone()).collect::<Vec<_>>()}));
 }
 Ok(serde_json::json!({"dimensions_agree":true,"fields":fields,"words":totals[0],"different_words":totals[1],"word_identical":totals[1]==0}))
}
fn main(){let args:Vec<_>=std::env::args().collect();if args.len()==4 && args[1]=="--compare"{
 match verify(&args[2],&args[3]){Ok(v)=>{println!("{}",serde_json::to_string_pretty(&v).unwrap());if v["word_identical"]!=true{std::process::exit(1);}},Err(e)=>{eprintln!("{e}");std::process::exit(1);}}return;}
 if args.len()!=3{eprintln!("usage: static_netcdf_compact INPUT OUTPUT | --compare INPUT OUTPUT");std::process::exit(2);}
 if let Err(e)=compact(&args[1],&args[2]){eprintln!("{e}");std::process::exit(1);}}
