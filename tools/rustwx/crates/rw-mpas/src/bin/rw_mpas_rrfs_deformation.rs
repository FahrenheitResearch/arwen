//! Complete RRFS deformation initialization, including plane and halo branches.
//! RRFSx/MPAS-Model src/core_init_atmosphere/mpas_atm_advection.F:403-447,506-529,743-939
//! at 1d6e370e. Native RKIND and GNU single-precision libm, source-order arithmetic.
use std::io::{self, Read, Write};
#[cfg_attr(unix, link(name = "m"))]
unsafe extern "C" {fn sinf(x:f32)->f32;fn cosf(x:f32)->f32;fn asinf(x:f32)->f32;fn atan2f(y:f32,x:f32)->f32;}
fn real(r:&mut impl Read)->io::Result<f32>{let mut b=[0;4];r.read_exact(&mut b)?;Ok(f32::from_le_bytes(b))}
fn index(r:&mut impl Read)->io::Result<i32>{let mut b=[0;4];r.read_exact(&mut b)?;Ok(i32::from_le_bytes(b))}
fn dim(r:&mut impl Read)->io::Result<usize>{let mut b=[0;8];r.read_exact(&mut b)?;usize::try_from(u64::from_le_bytes(b)).map_err(|_|io::Error::other("dimension overflow"))}
fn floats(r:&mut impl Read,n:usize)->io::Result<Vec<f32>>{(0..n).map(|_|real(r)).collect()}
fn coordinate_field(r:&mut impl Read,n:usize)->io::Result<Vec<f32>>{
 match index(r)? {
  4=>floats(r,n),
  8=>(0..n).map(|_|{let mut b=[0;8];r.read_exact(&mut b)?;Ok(f64::from_le_bytes(b) as f32)}).collect(),
  _=>Err(io::Error::other("coordinate width must be 4 or 8")),
 }
}
fn indices(r:&mut impl Read,n:usize)->io::Result<Vec<i32>>{(0..n).map(|_|index(r)).collect()}
fn arc(a:[f32;3],b:[f32;3])->f32 {
 let cx=b[0]-a[0];let cy=b[1]-a[1];let cz=b[2]-a[2];
 let r=(a[0]*a[0]+a[1]*a[1]+a[2]*a[2]).sqrt();let c=(cx*cx+cy*cy+cz*cz).sqrt();
 r*2.0*unsafe{asinf(c/(2.0*r))}
}
fn angle(a:[f32;3],b:[f32;3],c:[f32;3])->f32 {
 let sa=arc(b,c);let sb=arc(a,c);let sc=arc(a,b);
 let ab=[b[0]-a[0],b[1]-a[1],b[2]-a[2]];let ac=[c[0]-a[0],c[1]-a[1],c[2]-a[2]];
 let dx=ab[1]*ac[2]-ab[2]*ac[1];let dy=-(ab[0]*ac[2]-ab[2]*ac[0]);let dz=ab[0]*ac[1]-ab[1]*ac[0];
 let s=0.5*(sa+sb+sc);
 let ratio=unsafe{(sinf(s-sb)*sinf(s-sc))/(sinf(sb)*sinf(sc))};
 let sine=ratio.max(0.0).min(1.0).sqrt();let v=unsafe{asinf(sine.min(1.0).max(-1.0))};
 if dx*a[0]+dy*a[1]+dz*a[2]>=0.0 {2.0*v}else{-2.0*v}
}
fn run()->io::Result<()> {
 let mut r=io::stdin().lock();let mut w=io::stdout().lock();let mut magic=[0;8];r.read_exact(&mut magic)?;
 if &magic!=b"RRFSDEF2"{return Err(io::Error::other("expected RRFSDEF2 ABI"));}
 let nc=dim(&mut r)?;let ne=dim(&mut r)?;let nv=dim(&mut r)?;let me=dim(&mut r)?;
 let sphere=index(&mut r)?!=0;let radius=real(&mut r)?;
 let counts=indices(&mut r,nc)?;let eoc=indices(&mut r,nc*me)?;let coc=indices(&mut r,nc*me)?;let voc=indices(&mut r,nc*me)?;let coe=indices(&mut r,ne*2)?;
 let xc=coordinate_field(&mut r,nc)?;let yc=coordinate_field(&mut r,nc)?;let zc=coordinate_field(&mut r,nc)?;
 let xv=coordinate_field(&mut r,nv)?;let yv=coordinate_field(&mut r,nv)?;let zv=coordinate_field(&mut r,nv)?;
 let mut defa=vec![0.0f32;nc*me];let mut defb=defa.clone();let mut gx=defa.clone();let mut gy=defa.clone();
 let pi=2.0*unsafe{asinf(1.0)};
 for cell in 0..nc {
  let count=counts[cell] as usize;let row=cell*me;
  if count>me || count>=25{return Err(io::Error::other("native polygon exceeds storage"));}
  // Native tests only cell_list > nCells. Zero (wire -1) is a regional
  // sentinel and is never dereferenced by this geometry routine.
  if coc[row..row+count].iter().any(|&v|v>=nc as i32){continue;}
  let mut xp=[0.0f32;25];let mut yp=xp;let mut theta=xp;
  if sphere {
   let center=[xc[cell]/radius,yc[cell]/radius,zc[cell]/radius];
   let mut vertices=[[0.0f32;3];25];
   for j in 0..count{let v=voc[row+j] as usize;vertices[j]=[xv[v]/radius,yv[v]/radius,zv[v]/radius];}
   theta[0]=if center[2]==1.0{pi/2.0}else{pi/2.0-angle(center,vertices[0],[0.0,0.0,1.0])};
   let mut previous_angle=0.0;
   for j in 0..count {
    let ta=angle(center,vertices[j],vertices[(j+1)%count]);
    let distance=radius*arc(center,vertices[j]);
    if j>0{theta[j]=theta[j-1]+previous_angle;}
    xp[j]=unsafe{cosf(theta[j])}*distance;yp[j]=unsafe{sinf(theta[j])}*distance;
    previous_angle=ta;
   }
  }else{
   xp[0]=xc[cell];yp[0]=yc[cell];
   for j in 0..count{let v=voc[row+j] as usize;xp[j+1]=xv[v];yp[j+1]=yv[v];}
  }
  let mut area=0.0f32;
  for j in 0..count{let next=(j+1)%count;let dx=xp[next]-xp[j];let dy=yp[next]-yp[j];
   area=area+0.25*(xp[j]+xp[next])*(yp[next]-yp[j])-0.25*(yp[j]+yp[next])*(xp[next]-xp[j]);
   theta[j]=unsafe{atan2f(dy,dx)}-pi/2.0;
  }
  for j in 0..count{let next=(j+1)%count;let dx=xp[next]-xp[j];let dy=yp[next]-yp[j];let dl=(dx*dx+dy*dy).sqrt();
   let sin=unsafe{sinf(theta[j])};let cos=unsafe{cosf(theta[j])};let sint2=sin*sin;let cost2=cos*cos;let sc=sin*cos;
   let mut a=dl*(cost2-sint2)/area;let mut b=dl*2.0*sc/area;
   gx[row+j]=dl*cos/area;gy[row+j]=dl*sin/area;
   if coe[(eoc[row+j] as usize)*2]!=cell as i32{a=-a;b=-b;}
   defa[row+j]=a;defb[row+j]=b;
  }
 }
 for array in [defa,defb,gx,gy]{for value in array{w.write_all(&value.to_le_bytes())?;}}
 Ok(())
}
fn trig()->io::Result<()> {
 let mut r=io::stdin().lock();let mut w=io::stdout().lock();let mut magic=[0;8];r.read_exact(&mut magic)?;
 if &magic!=b"RRFSTRG1"{return Err(io::Error::other("expected RRFSTRG1 ABI"));}
 let n=dim(&mut r)?;let values=coordinate_field(&mut r,n)?;
 for cosine in [true,false]{for &x in &values{let y=unsafe{if cosine{cosf(x)}else{sinf(x)}};w.write_all(&y.to_le_bytes())?;}}
 Ok(())
}
fn main(){let action=if std::env::args().nth(1).as_deref()==Some("--edge-trig"){trig()}else{run()};if let Err(e)=action{eprintln!("RRFS deformation: {e}");std::process::exit(1);}}
