// Normalize MPAS field names and attach static geometry for the Rust converter.
// Uses a typed NetCDF copy; no numeric field arithmetic is performed.
use std::ffi::{CStr,CString,c_char,c_int,c_void};
use std::collections::BTreeMap;
#[link(name="netcdf")]
unsafe extern "C"{
    fn nc_open(path:*const c_char,mode:c_int,id:*mut c_int)->c_int;
    fn nc_close(id:c_int)->c_int;
    fn nc_strerror(code:c_int)->*const c_char;
    fn nc_inq_dimid(id:c_int,name:*const c_char,dim:*mut c_int)->c_int;
    fn nc_inq_dim(id:c_int,dim:c_int,name:*mut c_char,len:*mut usize)->c_int;
    fn nc_def_dim(id:c_int,name:*const c_char,len:usize,dim:*mut c_int)->c_int;
    fn nc_inq_varid(id:c_int,name:*const c_char,var:*mut c_int)->c_int;
    fn nc_rename_var(id:c_int,var:c_int,name:*const c_char)->c_int;
    fn nc_inq_var(id:c_int,var:c_int,name:*mut c_char,dtype:*mut c_int,rank:*mut c_int,dims:*mut c_int,natts:*mut c_int)->c_int;
    fn nc_inq_type(id:c_int,dtype:c_int,name:*mut c_char,size:*mut usize)->c_int;
    fn nc_def_var(id:c_int,name:*const c_char,dtype:c_int,rank:c_int,dims:*const c_int,var:*mut c_int)->c_int;
    fn nc_inq_attname(id:c_int,var:c_int,att:c_int,name:*mut c_char)->c_int;
    fn nc_copy_att(id:c_int,var:c_int,name:*const c_char,output:c_int,output_var:c_int)->c_int;
    fn nc_redef(id:c_int)->c_int;
    fn nc_enddef(id:c_int)->c_int;
    fn nc_get_var(id:c_int,var:c_int,data:*mut c_void)->c_int;
    fn nc_put_var(id:c_int,var:c_int,data:*const c_void)->c_int;
    fn nc_create(path:*const c_char,mode:c_int,id:*mut c_int)->c_int;
    fn nc_inq(id:c_int,nd:*mut c_int,nv:*mut c_int,na:*mut c_int,unlim:*mut c_int)->c_int;
}
fn text(s:&str)->CString{CString::new(s).unwrap()}
fn check(code:c_int)->Result<(),String>{if code==0{Ok(())}else{Err(unsafe{CStr::from_ptr(nc_strerror(code))}.to_string_lossy().into_owned())}}
fn variable(id:c_int,name:&str)->Option<c_int>{let mut v=0;if unsafe{nc_inq_varid(id,text(name).as_ptr(),&mut v)}==0{Some(v)}else{None}}
fn dimension(id:c_int,name:&str)->Option<(c_int,usize)>{let mut d=0;let mut n=0;if unsafe{nc_inq_dimid(id,text(name).as_ptr(),&mut d)}==0&&unsafe{nc_inq_dim(id,d,std::ptr::null_mut(),&mut n)}==0{Some((d,n))}else{None}}
fn repack(input:&str,output:&str)->Result<(),String>{
    if std::path::Path::new(output).exists(){return Err("output exists; use a new owned output path".into());}
    let(mut src,mut dst)=(0,0);
    unsafe{check(nc_open(text(input).as_ptr(),0,&mut src))?;check(nc_create(text(output).as_ptr(),512,&mut dst))?;}
    let(mut nd,mut nv,mut na,mut unlimited)=(0,0,0,0);
    unsafe{check(nc_inq(src,&mut nd,&mut nv,&mut na,&mut unlimited))?;}
    let mut dims=BTreeMap::new();let mut lengths=BTreeMap::new();
    for d in 0..nd{let mut name=[0 as c_char;257];let mut n=0;let mut target=0;unsafe{check(nc_inq_dim(src,d,name.as_mut_ptr(),&mut n))?;check(nc_def_dim(dst,name.as_ptr(),n,&mut target))?;}dims.insert(d,target);lengths.insert(d,n);}
    let mut copies=Vec::new();
    for v in 0..nv{let mut name=[0 as c_char;257];let(mut dtype,mut rank,mut attrs)=(0,0,0);let mut vd=[0;16];unsafe{check(nc_inq_var(src,v,name.as_mut_ptr(),&mut dtype,&mut rank,vd.as_mut_ptr(),&mut attrs))?;}
        if !(1..=6).contains(&dtype){return Err(format!("variable {} has type {dtype} that CDF-2 cannot preserve",unsafe{CStr::from_ptr(name.as_ptr())}.to_string_lossy()));}
        let mapped:Vec<c_int>=vd[..rank as usize].iter().map(|d|dims[d]).collect();let mut target=0;
        unsafe{check(nc_def_var(dst,name.as_ptr(),dtype,rank,mapped.as_ptr(),&mut target))?;}
        for att in 0..attrs{let mut name=[0 as c_char;257];unsafe{check(nc_inq_attname(src,v,att,name.as_mut_ptr()))?;check(nc_copy_att(src,v,name.as_ptr(),dst,target))?;}}
        let mut size=0;unsafe{check(nc_inq_type(src,dtype,std::ptr::null_mut(),&mut size))?;}
        let count:usize=vd[..rank as usize].iter().map(|d|lengths[d]).product();copies.push((v,target,count*size));
    }
    for att in 0..na{let mut name=[0 as c_char;257];unsafe{check(nc_inq_attname(src,-1,att,name.as_mut_ptr()))?;check(nc_copy_att(src,-1,name.as_ptr(),dst,-1))?;}}
    unsafe{check(nc_enddef(dst))?;}
    for(src_var,dst_var,bytes)in copies{let mut data=vec![0u64;bytes.div_ceil(8)];unsafe{check(nc_get_var(src,src_var,data.as_mut_ptr().cast()))?;check(nc_put_var(dst,dst_var,data.as_ptr().cast()))?;}}
    unsafe{check(nc_close(src))?;check(nc_close(dst))?;}Ok(())
}
fn run()->Result<(),String>{
    let a:Vec<String>=std::env::args().skip(1).collect();if a.len()!=3{return Err("usage: normalize_history INPUT INIT OUTPUT".into());}
    let(mut init,mut history)=(0,0);unsafe{check(nc_open(text(&a[1]).as_ptr(),0,&mut init))?;check(nc_open(text(&a[0]).as_ptr(),0,&mut history))?;}
    let n=dimension(history,"nCells").ok_or("history missing nCells")?.1;
    if dimension(init,"nCells").ok_or("init missing nCells")?.1!=n{return Err("history/init cell count differs".into());}
    unsafe{check(nc_close(history))?;}
    repack(&a[0],&a[2])?;
    unsafe{check(nc_open(text(&a[2]).as_ptr(),1,&mut history))?;check(nc_redef(history))?;}
    for(old,new)in[("t2m","t2"),("uReconstructZonal","u_zonal"),("uReconstructMeridional","v_meridional")]{if variable(history,new).is_none(){if let Some(v)=variable(history,old){unsafe{check(nc_rename_var(history,v,text(new).as_ptr()))?;}println!("RENAMED {old} {new}");}}}
    // The converter needs the atmospheric axis even when rendering a surface-only file.
    if dimension(history,"nVertLevels").is_none(){let levels=dimension(init,"nVertLevels").ok_or("init missing nVertLevels")?.1;let mut d=0;unsafe{check(nc_def_dim(history,text("nVertLevels").as_ptr(),levels,&mut d))?;}}
    let mut append=Vec::new();
    for name in ["latCell","lonCell","ter"]{
        if variable(history,name).is_some(){continue;}let Some(v)=variable(init,name)else{continue;};
        let(mut dtype,mut rank,mut natts)=(0,0,0);let mut dims=[0;16];
        unsafe{check(nc_inq_var(init,v,std::ptr::null_mut(),&mut dtype,&mut rank,dims.as_mut_ptr(),&mut natts))?;}
        if rank!=1{return Err(format!("static geometry {name} expected one cell axis"));}
        let mut len=0;unsafe{check(nc_inq_dim(init,dims[0],std::ptr::null_mut(),&mut len))?;}if len!=n{return Err(format!("static geometry {name} size differs"));}
        let target_dim=dimension(history,"nCells").unwrap().0;let mut nv=0;unsafe{check(nc_def_var(history,text(name).as_ptr(),dtype,1,&target_dim,&mut nv))?;}
        for att in 0..natts{let mut aname=[0 as c_char;257];unsafe{check(nc_inq_attname(init,v,att,aname.as_mut_ptr()))?;check(nc_copy_att(init,v,aname.as_ptr(),history,nv))?;}}
        let mut size=0;unsafe{check(nc_inq_type(init,dtype,std::ptr::null_mut(),&mut size))?;}append.push((name,v,nv,n*size));
    }
    unsafe{check(nc_enddef(history))?;}
    for(name,v,nv,bytes)in append{let mut data=vec![0u64;bytes.div_ceil(8)];unsafe{check(nc_get_var(init,v,data.as_mut_ptr().cast()))?;check(nc_put_var(history,nv,data.as_ptr().cast()))?;}println!("ATTACHED {name} bytes={bytes}");}
    unsafe{check(nc_close(init))?;check(nc_close(history))?;}println!("NORMALIZED arithmetic=none");Ok(())
}
fn main(){if let Err(e)=run(){eprintln!("ERROR {e}");std::process::exit(1);}}
