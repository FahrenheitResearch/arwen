//! Full-driver fixtures for both longitude correction branches of every
//! static climatology selector. MPAS: mpas_init_atm_static.F:694-704,
//! 928-937,1162-1171,2080-2089. All other input words remain copied.
use std::{env, ffi::CString, fs, os::raw::{c_char, c_int}};

#[link(name="netcdf")]
unsafe extern "C" {
    fn nc_open(path:*const c_char, mode:c_int, id:*mut c_int)->c_int;
    fn nc_inq_varid(id:c_int, name:*const c_char, var:*mut c_int)->c_int;
    fn nc_inq_varndims(id:c_int, var:c_int, rank:*mut c_int)->c_int;
    fn nc_inq_vardimid(id:c_int, var:c_int, dims:*mut c_int)->c_int;
    fn nc_inq_dimlen(id:c_int, dim:c_int, len:*mut usize)->c_int;
    fn nc_get_var_double(id:c_int, var:c_int, values:*mut f64)->c_int;
    fn nc_put_var_double(id:c_int, var:c_int, values:*const f64)->c_int;
    fn nc_close(id:c_int)->c_int;
}

fn main() {
    let args:Vec<_>=env::args().skip(1).collect();
    assert_eq!(args.len(),3,"static_grid_longitudes GRID OUT RADIANS");
    let offset:f64=args[2].parse().unwrap();
    assert!(offset.is_finite());
    assert!(!std::path::Path::new(&args[1]).exists(),"fixture must be new");
    fs::copy(&args[0],&args[1]).unwrap();
    let path=CString::new(args[1].as_str()).unwrap();
    let mut id=0;
    unsafe {
        assert_eq!(nc_open(path.as_ptr(),1,&mut id),0);
        for name in ["lonCell","lonVertex"] {
            let name=CString::new(name).unwrap();
            let (mut var,mut rank)=(0,0);
            assert_eq!(nc_inq_varid(id,name.as_ptr(),&mut var),0);
            assert_eq!(nc_inq_varndims(id,var,&mut rank),0);
            let mut dims=vec![0;usize::try_from(rank).unwrap()];
            assert_eq!(nc_inq_vardimid(id,var,dims.as_mut_ptr()),0);
            let mut count=1usize;
            for dim in dims {
                let mut len=0;
                assert_eq!(nc_inq_dimlen(id,dim,&mut len),0);
                count=count.checked_mul(len).unwrap();
            }
            let mut values=vec![0.0;count];
            assert_eq!(nc_get_var_double(id,var,values.as_mut_ptr()),0);
            for value in &mut values {*value+=offset;}
            assert_eq!(nc_put_var_double(id,var,values.as_ptr()),0);
        }
        assert_eq!(nc_close(id),0);
    }
}
