//! Additive C ABI for refined SFIRE static preparation.
use super::{clear_error,guard,register_fieldset,set_error,utf8,with_fieldset,OK,ERR};
use crate::projection::GridSpec;
use crate::sfire::{self,FireStaticRequest,FireStaticLoadRequest,FirePerimeterRequest};
use std::collections::HashMap;
use std::sync::{Mutex,OnceLock};

fn bundle_metadata() -> &'static Mutex<HashMap<u64,String>> {
    static DATA:OnceLock<Mutex<HashMap<u64,String>>>=OnceLock::new();
    DATA.get_or_init(||Mutex::new(HashMap::new()))
}

#[derive(serde::Deserialize)]
struct BoundsRequest {grid_spec:GridSpec,sr_x:usize,sr_y:usize}
#[derive(serde::Deserialize)]
struct CoverageRequest {
    bounds:[f64;4],
    #[serde(alias="resolution_degrees")]
    resolution:f64,
    #[serde(default="geographic_crs")]
    source_crs:String,
    #[serde(default)]
    pixel_edge_offset:f64,
}
fn geographic_crs() -> String {"EPSG:4326".into()}

#[unsafe(no_mangle)]
pub extern "C" fn gpuwm_static_sfire_v1() -> u32 {1}

/// Parse native formatted initializer inputs into a metadata-bound fieldset.
/// # Safety
/// JSON points to readable UTF-8 bytes; out points to a writable u64.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gpuwm_static_sfire_ideal_inputs(json:*const u8,len:usize,out:*mut u64)->i32 {
    guard(ERR,|| {
        clear_error();if out.is_null() {return set_error("Native ideal fieldset output is null");}
        let Some(text)=(unsafe{utf8(json,len)}) else {return set_error("Native ideal input request is not UTF-8");};
        let request:crate::sfire_ideal::IdealInputRequest=match serde_json::from_str(text) {Ok(r)=>r,Err(e)=>return set_error(e.to_string())};
        match crate::sfire_ideal::read_inputs(&request) {
            Err(e)=>set_error(e.to_string()),
            Ok((fields,metadata))=> {
                let text=match serde_json::to_string(&metadata) {Ok(t)=>t,Err(e)=>return set_error(e.to_string())};
                let handle=register_fieldset(fields);bundle_metadata().lock().unwrap().insert(handle,text);
                unsafe{*out=handle};OK
            }
        }
    })
}

/// Export the native default-REAL words directly into caller f32 storage.
/// # Safety
/// Name is readable UTF-8; non-null out has capacity writable f32 words.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gpuwm_static_sfire_ideal_field_f32(handle:u64,name:*const u8,name_len:usize,out:*mut f32,capacity:usize)->i64 {
    guard(-1_i64,|| {
        clear_error();let Some(name)=(unsafe{utf8(name,name_len)}) else {set_error("Native ideal field name is not UTF-8");return -1;};
        match with_fieldset(handle,|set| {
            let field=set.get(name).map_err(|e|e.to_string())?;let data=field.data();
            if out.is_null() {return Ok(data.len() as i64);}
            if capacity<data.len() {return Err("Native ideal f32 output buffer is too small".to_string());}
            for (i,value) in data.iter().enumerate() {unsafe{*out.add(i)=*value as f32};}
            Ok(data.len() as i64)
        }) {
            None=>{set_error("Native ideal fieldset handle is absent");-1},
            Some(Err(e))=>{set_error(e);-1},Some(Ok(n))=>n,
        }
    })
}

/// Write WRF E20.12 debug records from a contiguous default-REAL array.
/// # Safety
/// JSON is readable UTF-8 and values points to value_len readable f32 words.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gpuwm_static_sfire_debug_array(json:*const u8,len:usize,values:*const f32,value_len:usize) -> i32 {
    guard(ERR,|| {
        clear_error();
        let Some(text)=(unsafe{utf8(json,len)}) else {return set_error("SFIRE debug request is not UTF-8");};
        let request:crate::sfire_debug::FireDebugRequest=match serde_json::from_str(text) {Ok(r)=>r,Err(e)=>return set_error(e.to_string())};
        if values.is_null() || value_len>isize::MAX as usize/std::mem::size_of::<f32>() {
            return set_error("SFIRE debug values pointer is null or its length exceeds a readable slice");
        }
        let data=unsafe{std::slice::from_raw_parts(values,value_len)};
        match crate::sfire_debug::write_array(&request,data) {
            Ok(_)=>OK,Err(e)=>set_error(e.to_string()),
        }
    })
}

/// Resolve a complete declared experiment grid chain without Python geometry.
/// A null output probes the required JSON length.
/// # Safety
/// Input is readable UTF-8; a non-null output has capacity writable bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gpuwm_static_sfire_experiment_grids(json:*const u8,len:usize,out:*mut u8,capacity:usize) -> i64 {
    guard(-1_i64,|| {
        clear_error();
        let Some(text)=(unsafe{utf8(json,len)}) else {set_error("Fire experiment grid request is not UTF-8");return -1;};
        let result=(|| {
            let request:sfire::FireGridRequest=serde_json::from_str(text).map_err(|e|e.to_string())?;
            let grids=sfire::experiment_grids(&request).map_err(|e|e.to_string())?;
            serde_json::to_vec(&grids).map_err(|e|e.to_string())
        })();
        match result {
            Err(e)=>{set_error(e);-1},
            Ok(bytes)=> {
                if out.is_null() {return bytes.len() as i64;}
                if capacity<bytes.len() {set_error("Fire experiment grid JSON output buffer is too small");return -1;}
                unsafe{std::ptr::copy_nonoverlapping(bytes.as_ptr(),out,bytes.len())};bytes.len() as i64
            }
        }
    })
}

/// # Safety
/// JSON points to readable UTF-8 bytes and out points to a writable u64.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gpuwm_static_sfire_observed_perimeter(json:*const u8,len:usize,out:*mut u64) -> i32 {
    guard(ERR,|| {
        clear_error();
        if out.is_null() {return set_error("Observed fire fieldset output is null");}
        let Some(text)=(unsafe{utf8(json,len)}) else {return set_error("Observed fire request is not UTF-8");};
        let request:FirePerimeterRequest=match serde_json::from_str(text) {Ok(r)=>r,Err(e)=>return set_error(e.to_string())};
        match sfire::observed_perimeter(&request) {
            Err(e)=>set_error(e.to_string()),
            Ok(fields)=>{unsafe{*out=register_fieldset(fields)};OK}
        }
    })
}

/// # Safety
/// JSON points to readable UTF-8 bytes and out points to a writable u64.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gpuwm_static_sfire_load(json:*const u8,len:usize,out:*mut u64) -> i32 {
    guard(ERR,|| {
        clear_error();
        if out.is_null() {return set_error("SFIRE load fieldset output is null");}
        let Some(text)=(unsafe{utf8(json,len)}) else {return set_error("SFIRE load request is not UTF-8");};
        let request:FireStaticLoadRequest=match serde_json::from_str(text) {Ok(r)=>r,Err(e)=>return set_error(e.to_string())};
        match sfire::load(&request) {
            Err(e)=>set_error(e.to_string()),
            Ok((fields,metadata))=> {
                let text=match serde_json::to_string(&metadata) {Ok(v)=>v,Err(e)=>return set_error(e.to_string())};
                let handle=register_fieldset(fields);
                bundle_metadata().lock().unwrap().insert(handle,text);
                unsafe{*out=handle};OK
            }
        }
    })
}

/// # Safety
/// A non-null output buffer points to capacity writable bytes. Null probes size.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gpuwm_static_sfire_metadata_json(handle:u64,out:*mut u8,capacity:usize) -> i64 {
    guard(-1_i64,|| {
        clear_error();let registry=bundle_metadata().lock().unwrap();
        let Some(text)=registry.get(&handle) else {set_error("SFIRE fieldset has no bundle metadata");return -1;};
        if out.is_null() {return text.len() as i64;}
        if capacity<text.len() {set_error("SFIRE metadata output buffer is too small");return -1;}
        unsafe{std::ptr::copy_nonoverlapping(text.as_ptr(),out,text.len())};text.len() as i64
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn gpuwm_static_sfire_metadata_drop(handle:u64) {
    bundle_metadata().lock().unwrap().remove(&handle);
}

/// # Safety
/// JSON points to readable UTF-8 bytes and out points to four writable f64s.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gpuwm_static_sfire_coverage_bounds(json:*const u8,len:usize,out:*mut f64) -> i32 {
    guard(ERR,|| {
        clear_error();
        if out.is_null() {return set_error("SFIRE coverage output is null");}
        let Some(text)=(unsafe{utf8(json,len)}) else {return set_error("SFIRE coverage request is not UTF-8");};
        let request:CoverageRequest=match serde_json::from_str(text) {Ok(r)=>r,Err(e)=>return set_error(e.to_string())};
        match sfire::projected_coverage_bounds(request.bounds,request.resolution,&request.source_crs,request.pixel_edge_offset) {
            Err(e)=>set_error(e.to_string()),
            Ok(bounds)=>{unsafe{std::ptr::copy_nonoverlapping(bounds.as_ptr(),out,4)};OK}
        }
    })
}

/// # Safety
/// JSON points to readable UTF-8 bytes and out points to four writable f64s.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gpuwm_static_sfire_bounds(json:*const u8,len:usize,out:*mut f64) -> i32 {
    guard(ERR,|| {
        clear_error();
        if out.is_null() {return set_error("SFIRE bounds output is null");}
        let Some(text)=(unsafe{utf8(json,len)}) else {return set_error("SFIRE bounds request is not UTF-8");};
        let request:BoundsRequest=match serde_json::from_str(text) {Ok(r)=>r,Err(e)=>return set_error(e.to_string())};
        match sfire::bounds(&request.grid_spec,request.sr_x,request.sr_y) {
            Err(e)=>set_error(e.to_string()),
            Ok(bounds)=>{unsafe{std::ptr::copy_nonoverlapping(bounds.as_ptr(),out,4)};OK}
        }
    })
}

/// # Safety
/// JSON points to readable UTF-8 bytes and out points to one writable u64.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gpuwm_static_sfire_build(json:*const u8,len:usize,out:*mut u64) -> i32 {
    guard(ERR,|| {
        clear_error();
        if out.is_null() {return set_error("SFIRE fieldset output is null");}
        let Some(text)=(unsafe{utf8(json,len)}) else {return set_error("SFIRE request is not UTF-8");};
        let request:FireStaticRequest=match serde_json::from_str(text) {Ok(r)=>r,Err(e)=>return set_error(e.to_string())};
        match sfire::build(&request) {
            Err(e)=>set_error(e.to_string()),
            Ok(fields)=>{unsafe{*out=register_fieldset(fields)};OK}
        }
    })
}
