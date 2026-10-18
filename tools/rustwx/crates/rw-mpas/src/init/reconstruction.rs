//! Complete case-7 two-dimensional wind reconstruction and spherical rotation.
//! MPAS v8.4.2 operators/mpas_vector_reconstruction.F:206-350;
//! core_init_atmosphere/mpas_init_atm_cases.F:5140-5147.

use crate::error::{MpasError,MpasResult};

pub const FIELDS:[&str;5]=["uReconstructX","uReconstructY","uReconstructZ",
    "uReconstructZonal","uReconstructMeridional"];

/// Cell-major, vertical-fastest diagnostic fields in the native stream order.
pub struct ReconstructedWind {pub fields:[Vec<f32>;5]}

/// MPAS v8.4.2 operators/mpas_vector_reconstruction.F:206-350.
pub fn reconstruct(u:&[Vec<f32>],counts:&[usize],edges:&[Vec<usize>],
    coefficients:&[f32],lat:&[f32],lon:&[f32],sphere:bool) -> MpasResult<ReconstructedWind> {
    let nc=counts.len();let nz=u.first().map_or(0,Vec::len);
    let me=edges.first().map_or(0,Vec::len);
    if nc==0 || nz==0 || me==0 || edges.len()!=nc || lat.len()!=nc || lon.len()!=nc
        || coefficients.len()!=nc*me*3 || u.iter().any(|v|v.len()!=nz)
        || edges.iter().any(|v|v.len()!=me) {
        return Err(MpasError::Refusal("wind reconstruction dimensions would change native array strides".into()));
    }
    let mut fields:[Vec<f32>;5]=std::array::from_fn(|_|vec![0.;nc*nz]);
    for c in 0..nc {
        if counts[c]>me {return Err(MpasError::Refusal("wind reconstruction edge count exceeds its cell stencil".into()));}
        for i in 0..counts[c] {
            let e=edges[c][i];
            if e==0 || e>u.len() {return Err(MpasError::Refusal("active wind reconstruction edge is outside the mesh".into()));}
            for k in 0..nz {
                let wind=u[e-1][k];let position=c*nz+k;
                for axis in 0..3 {
                    fields[axis][position]=fields[axis][position]+coefficients[(c*me+i)*3+axis]*wind;
                }
            }
        }
    }
    for c in 0..nc {
        if sphere {
            let clat=lat[c].cos();let slat=lat[c].sin();
            let clon=lon[c].cos();let slon=lon[c].sin();
            for k in 0..nz {
                let p=c*nz+k;
                fields[3][p]=-fields[0][p]*slon+fields[1][p]*clon;
                fields[4][p]=-(fields[0][p]*clon+fields[1][p]*slon)*slat+fields[2][p]*clat;
            }
        } else {
            for k in 0..nz {let p=c*nz+k;fields[3][p]=fields[0][p];fields[4][p]=fields[1][p];}
        }
    }
    Ok(ReconstructedWind {fields})
}
