//! First-guess terrain blending. MPAS mpas_init_atm_cases.F:7804-7970.
use super::{hinterp::{Slab,InterpContext,Method,Underflow},metfile::MetSlab};

/// Complete regular-latlon and Gaussian branches of blend_bdy_terrain.
/// MPAS v8.4.2 mpas_init_atm_cases.F:7867-7948; mpas_init_atm_llxy.F:1389-1416.
/// Coordinates enter in radians, as in the mesh. No fallback method is used.
pub fn blend_bdy_terrain(field:&MetSlab,lat:&[f32],lon:&[f32],mask:&[i32],terrain:&mut [f32])->Result<(),String> {
    let n=terrain.len();
    if field.field!="SOILHGT" || field.nx<3 || field.ny==0 || field.values.len()!=field.nx*field.ny
        || lat.len()!=n || lon.len()!=n || mask.len()!=n {
        return Err("terrain blending requires SOILHGT and matching cell coordinates/masks".into());
    }
    let slab=Slab::from_met(&field.values,field.nx,field.ny);
    let projection=super::projection::Projection::from_met(field).map_err(|e|e.to_string())?;
    let context=InterpContext {array:&slab,msgval:-1e30,maskval:None,mask:None,
        underflow:Underflow::Preserve,underflow_hits:None};
    const DEG_PER_RAD:f32=180.0/(std::f64::consts::PI as f32);
    for c in 0..n {
        if mask[c]<=0 {continue;}
        let (x,y)=projection.locate(lat[c]*DEG_PER_RAD,lon[c]*DEG_PER_RAD);
        let first_guess=super::hinterp::interp_sequence(x,y,&context,&[Method::FourPoint],0);
        if mask[c]>5 {terrain[c]=first_guess;} else {
            let weight=mask[c] as f32/5.;
            terrain[c]=weight*first_guess+(1.-weight)*terrain[c];
        }
    }
    Ok(())
}
