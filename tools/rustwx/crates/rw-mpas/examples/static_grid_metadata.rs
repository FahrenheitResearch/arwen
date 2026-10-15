//! Metadata-only native driver fixture; all stored weather words remain copied.
//! MPAS: mpas_bootstrapping.F:287-298; mpas_atm_advection.F:130-200,822-895.
use std::{env,ffi::CString,fs,os::raw::{c_char,c_int}};
#[link(name="netcdf")]
unsafe extern "C" {
    fn nc_open(path:*const c_char,mode:c_int,id:*mut c_int)->c_int;
    fn nc_redef(id:c_int)->c_int;
    fn nc_put_att_text(id:c_int,var:c_int,name:*const c_char,len:usize,value:*const c_char)->c_int;
    fn nc_enddef(id:c_int)->c_int;
    fn nc_close(id:c_int)->c_int;
}
fn main() {
    let args:Vec<_>=env::args().skip(1).collect();
    assert_eq!(args.len(),4,"static_grid_metadata GRID OUT ATTRIBUTE TEXT");
    assert!(!std::path::Path::new(&args[1]).exists(),"fixture must be new");
    fs::copy(&args[0],&args[1]).unwrap();
    let path=CString::new(args[1].as_str()).unwrap();
    let name=CString::new(args[2].as_str()).unwrap();
    let value=CString::new(args[3].as_str()).unwrap();
    let mut id=0;
    unsafe {
        assert_eq!(nc_open(path.as_ptr(),1,&mut id),0);
        assert_eq!(nc_redef(id),0);
        assert_eq!(nc_put_att_text(id,-1,name.as_ptr(),args[3].len(),value.as_ptr()),0);
        assert_eq!(nc_enddef(id),0);
        assert_eq!(nc_close(id),0);
    }
}
