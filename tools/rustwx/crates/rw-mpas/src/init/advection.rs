//! Static advection and deformation geometry in scalar RKIND order.
//! MPAS v8.4.2 core_init_atmosphere/mpas_atm_advection.F:21-923.
//! The fixed polynomial_order=2 branch is the module's only reachable fit.
use crate::geometry::Real;
use serde::Deserialize;
use std::io::{Read,Write};

pub const ABI_MARKER: &str = "rw_mpas_geometry --protocol hex-advection-v1";
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Request {
    pub width: usize,
    pub counts: Vec<usize>,
    pub edges: Vec<Vec<usize>>,
    pub neighbors: Vec<Vec<i64>>,
    pub owners: Vec<[i64;2]>,
    pub edge_vertices: Vec<[usize;2]>,
    pub cell_vertices: Vec<Vec<usize>>,
    pub cells: Vec<[f64;3]>,
    pub vertices: Vec<[f64;3]>,
    pub angles: Vec<f64>, pub dc: Vec<f64>, pub dv: Vec<f64>,
    pub sphere: bool, pub radius: f64,
    pub stencil: usize,
}
/// Scale decoded unit-sphere fields in the native RKIND load/operation order.
/// The caller passes the mesh configuration radius; it is not inferred from
/// coordinates. MPAS-A v8.4.2 core_init_atmosphere/mpas_init_atm_static.F:281-306.
pub fn scale_native_grid_field(name: &str, values: &[f64], radius: f32) -> Option<Vec<f32>> {
    let factor = match name {
        "xCell"|"yCell"|"zCell"|"xEdge"|"yEdge"|"zEdge"|
        "xVertex"|"yVertex"|"zVertex"|"dcEdge"|"dvEdge"|"nominalMinDc" => radius,
        "areaCell"|"areaTriangle"|"kiteAreasOnVertex" => radius * radius,
        _ => return None,
    };
    Some(values.iter().map(|&v|(v as f32) * factor).collect())
}

/// MPAS mpas_atm_advection.F:403-496: ordered geometric products.
fn dot<T:Real>(a:[T;3],b:[T;3])->T { (a[0]*b[0]+a[1]*b[1])+a[2]*b[2] }
fn sub<T:Real>(a:[T;3],b:[T;3])->[T;3] { [a[0]-b[0],a[1]-b[1],a[2]-b[2]] }
fn limit<T:Real>(v:T,lo:T,hi:T)->T { if v<lo { lo } else if v>hi { hi } else { v } }
/// MPAS mpas_atm_advection.F:506-529 (arc_length).
fn arc<T:Real>(a:[T;3],b:[T;3])->T {
    let r=dot(a,a).sqrt(); let d=sub(b,a); let c=dot(d,d).sqrt();
    r*T::number(2.0)*(c/(T::number(2.0)*r)).asin()
}
/// MPAS mpas_atm_advection.F:403-447 (sphere_angle).
fn angle<T:Real>(a:[T;3],b:[T;3],c:[T;3])->T {
    let sa=arc(b,c); let sb=arc(a,c); let sc=arc(a,b);
    let ab=sub(b,a); let ac=sub(c,a);
    let d=[ab[1]*ac[2]-ab[2]*ac[1],-(ab[0]*ac[2]-ab[2]*ac[0]),ab[0]*ac[1]-ab[1]*ac[0]];
    let s=T::half()*((sa+sb)+sc);
    let sine=limit(((s-sb).sin()*(s-sc).sin())/(sb.sin()*sc.sin()),T::zero(),T::one()).sqrt();
    let v=limit(sine,-T::one(),T::one()).asin();
    if dot(d,a)>=T::zero() { T::number(2.0)*v } else { -T::number(2.0)*v }
}
/// MPAS mpas_atm_advection.F:539-564 (arc_bisect).
fn bisect<T:Real>(a:[T;3],b:[T;3])->Result<[T;3],String> {
    let r=dot(a,a).sqrt(); let c=[T::half()*(a[0]+b[0]),T::half()*(a[1]+b[1]),T::half()*(a[2]+b[2])];
    let d=dot(c,c).sqrt();
    if d==T::zero() { return Err("arc_bisect: diametrically opposite vertices".into()); }
    Ok([r*c[0]/d,r*c[1]/d,r*c[2]/d])
}
/// Scalar MATMUL reduction order for the pinned GNU strict build.
/// MPAS mpas_atm_advection.F:567-614 (MATMUL in poly_fit_2).
fn multiply<T:Real>(a:&[Vec<T>],b:&[Vec<T>])->Vec<Vec<T>> {
    let mut c=vec![vec![T::zero();b[0].len()];a.len()];
    for i in 0..a.len() { for j in 0..b[0].len() { for k in 0..b.len() {
        c[i][j]=c[i][j]+a[i][k]*b[k][j];
    } } } c
}
/// MPAS mpas_atm_advection.F:567-614 (TRANSPOSE in poly_fit_2).
fn transpose<T:Real>(a:&[Vec<T>])->Vec<Vec<T>> {
    (0..a[0].len()).map(|j|a.iter().map(|row|row[j]).collect()).collect()
}
/// ELGS and MIGS, including scaled pivot selection and inverse backsolve.
/// MPAS mpas_atm_advection.F:633-741.
fn inverse<T:Real>(a:&mut [Vec<T>])->Result<Vec<Vec<T>>,String> {
    let n=a.len(); let mut indices:Vec<usize>=(0..n).collect();
    let scales:Vec<T>=a.iter().map(|row|row.iter().fold(T::zero(),|m,&v|if v.abs()>m {v.abs()} else {m})).collect();
    if scales.iter().any(|&v|v==T::zero()) { return Err("zero row in advection polynomial fit".into()); }
    for j in 0..n-1 {
        let mut pi1=T::zero(); let mut slot=j;
        for i in j..n { let pi=a[indices[i]][j].abs()/scales[indices[i]];
            if pi>pi1 { pi1=pi;slot=i; } }
        indices.swap(j,slot); let pivot=indices[j];
        if a[pivot][j]==T::zero() { return Err("singular advection polynomial fit".into()); }
        for i in j+1..n { let row=indices[i]; let pj=a[row][j]/a[pivot][j]; a[row][j]=pj;
            for k in j+1..n { a[row][k]=a[row][k]-pj*a[pivot][k]; } }
    }
    let mut b=vec![vec![T::zero();n];n]; for i in 0..n { b[i][i]=T::one(); }
    for i in 0..n-1 { for j in i+1..n { for k in 0..n {
        let row=indices[j]; b[row][k]=b[row][k]-a[row][i]*b[indices[i]][k];
    } } }
    if a[indices[n-1]][n-1]==T::zero() { return Err("singular final advection pivot".into()); }
    let mut x=vec![vec![T::zero();n];n];
    for i in 0..n {
        x[n-1][i]=b[indices[n-1]][i]/a[indices[n-1]][n-1];
        for j in (0..n-1).rev() {
            x[j][i]=b[indices[j]][i];
            for k in j+1..n { x[j][i]=x[j][i]-a[indices[j]][k]*x[k][i]; }
            x[j][i]=x[j][i]/a[indices[j]][j];
        }
    } Ok(x)
}
/// Whole fixed quadratic least-squares fit; retain all native MATMUL steps.
/// MPAS mpas_atm_advection.F:567-614.
fn fit<T:Real>(xp:&[T],yp:&[T])->Result<Vec<Vec<T>>,String> {
    let m=xp.len()+1;
    let mut a=vec![vec![T::zero();6];m]; a[0][0]=T::one();
    for i in 1..m { let x=xp[i-1];let y=yp[i-1]; a[i]=vec![T::one(),x,y,x*x,x*y,y*y]; }
    let mut w=vec![vec![T::zero();m];m]; for i in 0..m { w[i][i]=T::one(); }
    let h=multiply(&transpose(&w),&w); let at=transpose(&a);let ath=multiply(&at,&h);
    let mut atha=multiply(&ath,&a); let inv=inverse(&mut atha)?;
    Ok(multiply(&inv,&ath))
}
/// Spherical tangent coordinates for center-neighbor or center-vertex stencil.
/// MPAS mpas_atm_advection.F:136-194,826-881.
fn tangent<T:Real>(center:[T;3],points:&[[T;3]],radius:T)->(Vec<T>,Vec<T>,Vec<T>) {
    let pi=T::number(2.0)*T::one().asin();
    let abs=if center[2]==T::one() {pi/T::number(2.0)} else {pi/T::number(2.0)-angle(center,points[0],[T::zero(),T::zero(),T::one()])};
    let mut theta=vec![abs;points.len()]; let mut xp=vec![T::zero();points.len()]; let mut yp=xp.clone();
    for i in 0..points.len() {
        if i>0 {theta[i]=theta[i-1]+angle(center,points[i-1],points[i]);}
        let dl=radius*arc(center,points[i]); xp[i]=theta[i].cos()*dl;yp[i]=theta[i].sin()*dl;
    } (xp,yp,theta)
}
pub const COMPRESSION_MARKER:&str="rw_mpas_geometry --protocol hex-advection-compression-v1";
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CompressionRequest {
    pub width:usize,pub counts:Vec<usize>,pub neighbors:Vec<Vec<i64>>,pub owners:Vec<[i64;2]>,
    pub dc:Vec<f64>,pub dv:Vec<f64>,pub stencil:usize,pub deriv:Vec<Vec<Vec<f64>>>,
}
impl From<&Request> for CompressionRequest {
    fn from(r:&Request)->Self {Self {width:r.width,counts:r.counts.clone(),neighbors:r.neighbors.clone(),
        owners:r.owners.clone(),dc:r.dc.clone(),dv:r.dv.clone(),stencil:r.stencil,deriv:vec![]}}
}
/// MPAS v8.4.2 core_atmosphere/mpas_atm_core.F:1391-1534.
pub fn run_compression(mut input:impl Read,mut output:impl Write)->Result<(),String> {
    let _abi=std::hint::black_box(COMPRESSION_MARKER);
    let mut bytes=Vec::new();input.read_to_end(&mut bytes).map_err(|e|e.to_string())?;
    let r:CompressionRequest=serde_json::from_slice(&bytes).map_err(|e|e.to_string())?;
    let nc=r.counts.len();let ne=r.owners.len();
    if nc==0 || r.neighbors.len()!=nc || r.dc.len()!=ne || r.dv.len()!=ne || r.stencil==0
        || r.deriv.len()!=ne || r.deriv.iter().any(|s|s.len()!=2 || s.iter().any(|v|v.len()!=r.stencil))
        || r.counts.iter().enumerate().any(|(c,&n)|n>=r.stencil || r.neighbors[c].len()<n)
        || r.neighbors.iter().flatten().chain(r.owners.iter().flatten()).any(|&c|c < -1 || c>=nc as i64) {
        return Err("advection compression dimensions or connectivity are inconsistent".into());
    }
    fn process<T:Real>(r:&CompressionRequest,out:&mut impl Write)->Result<(),String> {
        let deriv:Vec<T>=r.deriv.iter().flatten().flatten().map(|&x|T::number(x)).collect();
        let (coef,third,counts,ids)=compress::<T>(r,&deriv)?;
        for v in [&coef,&third] {for &x in v {x.encode(out)?;}}
        for v in [&counts,&ids] {for x in v {out.write_all(&x.to_le_bytes()).map_err(|e|e.to_string())?;}}
        Ok(())
    }
    match r.width {4=>process::<f32>(&r,&mut output),8=>process::<f64>(&r,&mut output),_=>Err("compression width must be 4 or 8".into())}
}
/// Complete initialized fields from atm_initialize_advection_rk and
/// atm_initialize_deformation_weights. MPAS mpas_atm_advection.F:21-923.
/// MPAS v8.4.2 core_atmosphere/mpas_atm_core.F:1391-1534.
/// Retain owner-first insertion order and the last matching stencil slot.
fn compress<T:Real>(r:&CompressionRequest,deriv:&[T])->Result<(Vec<T>,Vec<T>,Vec<i64>,Vec<i64>),String> {
    let ne=r.owners.len();let nc=r.counts.len();let stride=r.stencil;
    let mut coef=vec![T::zero();ne*stride];let mut third=coef.clone();
    let mut counts=vec![0i64;ne];let mut ids=vec![0i64;ne*stride];
    for (e,&[a,b]) in r.owners.iter().enumerate() {
        if a<0 && b<0 {continue;}
        let around=|c:i64|->&[i64] {if c<0 {&[]} else {&r.neighbors[c as usize][..r.counts[c as usize]]}};
        let mut list=vec![a,b];
        for &c in around(a) {if c!=b {list.push(c);}}
        for &c in around(b) {if !list.contains(&c) {list.push(c);}}
        if list.len()>stride || list.len()>20 {return Err("compressed advection stencil exceeds native capacity".into());}
        counts[e]=list.len() as i64;
        for (j,&c) in list.iter().enumerate() {ids[e*stride+j]=if c<0 {(nc+1) as i64} else {c+1};}
        for (side,owner) in [a,b].into_iter().enumerate() {
            for (i,c) in std::iter::once(owner).chain(around(owner).iter().copied()).enumerate() {
                let j=list.iter().rposition(|&v|v==c).unwrap();
                let d=deriv[(e*2+side)*stride+i];
                coef[e*stride+j]=coef[e*stride+j]+d;
                third[e*stride+j]=if side==0 {third[e*stride+j]+d} else {third[e*stride+j]-d};
            }
        }
        let dc=T::number(r.dc[e]);let dv=T::number(r.dv[e]);
        for j in 0..list.len() {
            coef[e*stride+j]=-(dc*dc)*coef[e*stride+j]/T::number(12.0);
            third[e*stride+j]=-(dc*dc)*third[e*stride+j]/T::number(12.0);
        }
        for c in [a,b] {let j=list.iter().rposition(|&v|v==c).unwrap();coef[e*stride+j]=coef[e*stride+j]+T::half();}
        for j in 0..list.len() {coef[e*stride+j]=dv*coef[e*stride+j];third[e*stride+j]=dv*third[e*stride+j];}
    }
    Ok((coef,third,counts,ids))
}
pub struct Fields<T> {
    pub deriv_two:Vec<T>,pub gradient_x:Vec<T>,pub gradient_y:Vec<T>,pub adv_cells:Vec<i64>,
    pub adv_coefs:Vec<T>,pub adv_coefs_3rd:Vec<T>,pub n_adv_cells:Vec<i64>,pub cells_for_edge:Vec<i64>,
}
/// MPAS mpas_atm_advection.F:21-394,744-923: both initialization drivers.
fn compute<T:Real>(r:&Request)->Result<Fields<T>,String> {
    let nc=r.cells.len();let ne=r.owners.len();let max=r.counts.iter().copied().max().unwrap_or(0);
    if nc==0 || ne==0 || r.counts.len()!=nc || r.edges.len()!=nc || r.neighbors.len()!=nc
        || r.cell_vertices.len()!=nc || r.edge_vertices.len()!=ne || r.dc.len()!=ne
        || r.dv.len()!=ne || r.angles.len()!=ne || r.stencil<max+1 || max+1>25 {
        return Err("advection topology dimensions are inconsistent".into());
    }
    let radius=T::number(r.radius);
    if !radius.finite() || radius<=T::zero() { return Err("sphere radius must be positive".into()); }
    let cells:Vec<[T;3]>=r.cells.iter().map(|v|v.map(T::number)).collect();
    let vertices:Vec<[T;3]>=r.vertices.iter().map(|v|v.map(T::number)).collect();
    for c in 0..nc {
        let count=r.counts[c];
        if count<5 || r.edges[c].len()<count || r.neighbors[c].len()<count || r.cell_vertices[c].len()<count {
            return Err(format!("cell {c} lacks a complete quadratic stencil"));
        }
        for j in 0..count { if r.edges[c][j]>=ne || r.neighbors[c][j] < -1 || r.neighbors[c][j]>=nc as i64
            || r.cell_vertices[c][j]>=vertices.len() {return Err("advection connectivity is out of range".into());} }
    }
    for (e,&[a,b]) in r.owners.iter().enumerate() {
        if a < -1 || b < -1 || a>=nc as i64 || b>=nc as i64
            || r.edge_vertices[e].iter().any(|&v|v>=vertices.len()) {return Err("advection edge connectivity is out of range".into());}
    }
    let pi=T::number(2.0)*T::one().asin();
    let mut deriv=vec![T::zero();ne*2*r.stencil];
    let mut adv=vec![0i64;nc*(max+2)];
    let mut gx=vec![T::zero();nc*max];let mut gy=gx.clone();
    for c in 0..nc {
        let count=r.counts[c];adv[c*(max+2)]=(count+1) as i64;
        // MPAS skips a cell whose stencil reaches the garbage/halo cell.
        if r.neighbors[c][..count].iter().any(|&v|v<0) {continue;}
        let mut list=vec![c];list.extend(r.neighbors[c][..count].iter().map(|&v|v as usize));
        let normalized:Vec<[T;3]>=list.iter().map(|&v|cells[v].map(|x|x/radius)).collect();
        let (xp,yp,theta)=if r.sphere {tangent(normalized[0],&normalized[1..],radius)} else {
            let angles:Vec<T>=r.edges[c][..count].iter().map(|&e|if r.owners[e][0]==c as i64 {T::number(r.angles[e])} else {T::number(r.angles[e])-pi}).collect();
            (r.edges[c][..count].iter().zip(&angles).map(|(&e,&a)|T::number(r.dc[e])*a.cos()).collect(),
             r.edges[c][..count].iter().zip(&angles).map(|(&e,&a)|T::number(r.dc[e])*a.sin()).collect(),angles)
        };
        // The planar native branch does not populate advCells beyond count.
        if r.sphere {for (i,&v) in list.iter().enumerate() {adv[c*(max+2)+i+1]=(v+1) as i64;}}
        let b=fit(&xp,&yp)?;
        for j in 0..count {
            let e=r.edges[c][j]; let side=if r.owners[e][0]==c as i64 {0}else{1};
            let theta_e=if r.sphere {
                let [v1,v2]=r.edge_vertices[e];
                let mid=bisect(vertices[v1].map(|x|x/radius),vertices[v2].map(|x|x/radius))?;
                angle(normalized[0],normalized[j+1],mid)+theta[j]
            } else {theta[j]};
            let cos=theta_e.cos();let sin=theta_e.sin();let cs=cos*sin;let c2=cos*cos;let s2=sin*sin;
            for i in 0..count+1 {deriv[(e*2+side)*r.stencil+i]=(T::number(2.0)*c2*b[3][i]+T::number(2.0)*cs*b[4][i])+T::number(2.0)*s2*b[5][i];}
        }
        let (px,py,_)=if r.sphere {
            let center=cells[c].map(|x|x/radius);
            let points:Vec<[T;3]>=r.cell_vertices[c][..count].iter().map(|&v|vertices[v].map(|x|x/radius)).collect();
            tangent(center,&points,radius)
        } else {
            // Reproduce the native center-first indexing of the planar path.
            let mut px=vec![cells[c][0]];let mut py=vec![cells[c][1]];
            for &v in &r.cell_vertices[c][..count-1] {px.push(vertices[v][0]);py.push(vertices[v][1]);}
            (px,py,vec![])
        };
        let mut area=T::zero();let mut normals=vec![T::zero();count];
        for j in 0..count {let next=(j+1)%count;let dx=px[next]-px[j];let dy=py[next]-py[j];
            area=(area+T::number(0.25)*(px[j]+px[next])*(py[next]-py[j]))-T::number(0.25)*(py[j]+py[next])*(px[next]-px[j]);
            normals[j]=dy.atan2(dx)-pi/T::number(2.0);
        }
        if area==T::zero() {return Err(format!("zero tangent-plane area at cell {c}"));}
        for j in 0..count {let next=(j+1)%count;let dx=px[next]-px[j];let dy=py[next]-py[j];let dl=(dx*dx+dy*dy).sqrt();
            gx[c*max+j]=dl*normals[j].cos()/area;gy[c*max+j]=dl*normals[j].sin()/area;
        }
    }
    let (coef,third,counts,ids)=compress::<T>(&CompressionRequest::from(r),&deriv)?;
    Ok(Fields {deriv_two:deriv,gradient_x:gx,gradient_y:gy,adv_cells:adv,
        adv_coefs:coef,adv_coefs_3rd:third,n_adv_cells:counts,cells_for_edge:ids})
}
fn process<T:Real>(r:&Request,out:&mut impl Write)->Result<(),String> {
    let f=compute::<T>(r)?;
    for v in [&f.deriv_two,&f.gradient_x,&f.gradient_y] {for &x in v {x.encode(out)?;}}
    for x in f.adv_cells {out.write_all(&x.to_le_bytes()).map_err(|e|e.to_string())?;}
    for v in [&f.adv_coefs,&f.adv_coefs_3rd] {for &x in v {x.encode(out)?;}}
    for v in [&f.n_adv_cells,&f.cells_for_edge] {for x in v {out.write_all(&x.to_le_bytes()).map_err(|e|e.to_string())?;}}
    Ok(())
}
/// Protocol marshaling only; all geometry calculation remains in Rust.
pub fn run(mut input:impl Read,mut output:impl Write)->Result<(),String> {
    let _abi=std::hint::black_box(ABI_MARKER);
    let mut bytes=Vec::new();input.read_to_end(&mut bytes).map_err(|e|e.to_string())?;
    let r:Request=serde_json::from_slice(&bytes).map_err(|e|e.to_string())?;
    match r.width {4=>process::<f32>(&r,&mut output),8=>process::<f64>(&r,&mut output),_=>Err("advection width must be 4 or 8".into())}
}
