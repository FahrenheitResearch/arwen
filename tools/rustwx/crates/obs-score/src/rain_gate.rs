//! Conservative geometry and exact-support regional rain screening kernels.
use super::{call, input, output, size};

#[unsafe(no_mangle)]
pub extern "C" fn gpuwm_rain_gate_abi_version() -> u32 { 1 }

type Point = (f64, f64);
fn area(poly: &[Point]) -> f64 {
    if poly.len() < 3 { return 0.0; }
    // Translate first to avoid cancellation far from the projection origin.
    let origin = poly[0];
    let mut sum = 0.0;
    for i in 1..poly.len()-1 {
        sum += (poly[i].0-origin.0)*(poly[i+1].1-origin.1)
             - (poly[i+1].0-origin.0)*(poly[i].1-origin.1);
    }
    sum.abs() * 0.5
}
fn clip(poly: &[Point], axis: usize, bound: f64, greater: bool) -> Vec<Point> {
    let mut result = Vec::new();
    if poly.is_empty() { return result; }
    let coordinate = |p: Point| if axis == 0 { p.0 } else { p.1 };
    let inside = |p: Point| if greater { coordinate(p) >= bound } else { coordinate(p) <= bound };
    let mut previous = poly[poly.len()-1];
    for &current in poly {
        if inside(previous) != inside(current) {
            let fraction = (bound-coordinate(previous))/(coordinate(current)-coordinate(previous));
            result.push((previous.0+fraction*(current.0-previous.0),
                         previous.1+fraction*(current.1-previous.1)));
        }
        if inside(current) { result.push(current); }
        previous = current;
    }
    result
}
fn overlap(poly: &[Point], x0: f64, x1: f64, y0: f64, y1: f64) -> f64 {
    let p = clip(poly, 0, x0, true);
    let p = clip(&p, 0, x1, false);
    let p = clip(&p, 1, y0, true);
    area(&clip(&p, 1, y1, false))
}
fn edges(values: &[f64]) -> Result<(), String> {
    if values.len()<2 || values.iter().any(|v| !v.is_finite())
        || values.windows(2).any(|v| v[1]<=v[0]) {
        return Err("cell edges must be finite and strictly increasing".into());
    }
    Ok(())
}

/// Spherical Lambert azimuthal equal-area projection. Geometry stays in Rust.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gpuwm_rain_gate_laea(
    lat: *const f64, lon: *const f64, n: usize, lon0: f64, lat0: f64,
    x: *mut f64, y: *mut f64,
) -> i32 { call(|| {
    let lat = unsafe { input(lat,n)? }; let lon = unsafe { input(lon,n)? };
    let x = unsafe { output(x,n)? }; let y = unsafe { output(y,n)? };
    if !lon0.is_finite() || !lat0.is_finite() || lat0.abs()>90.0 {
        return Err("invalid equal-area projection center".into());
    }
    let a = lat0.to_radians(); let b = lon0.to_radians();
    for i in 0..n {
        if !lat[i].is_finite() || !lon[i].is_finite() || lat[i].abs()>90.0 {
            return Err("non-finite or invalid geographic cell geometry".into());
        }
        let p = lat[i].to_radians(); let d = lon[i].to_radians()-b;
        let denom = 1.0+a.sin()*p.sin()+a.cos()*p.cos()*d.cos();
        if denom <= 1e-10 { return Err("grid reaches equal-area antipode".into()); }
        let k = (2.0/denom).sqrt()*6370000.0;
        x[i] = k*p.cos()*d.sin();
        y[i] = k*(a.cos()*p.sin()-a.sin()*p.cos()*d.cos());
    }
    Ok(())
}) }

#[unsafe(no_mangle)]
pub unsafe extern "C" fn gpuwm_rain_gate_corners(
    centers: *const f64, ny: usize, nx: usize, corners: *mut f64,
) -> i32 { call(|| {
    if ny<2 || nx<2 { return Err("corner recovery needs at least 2x2 centers".into()); }
    let centers = unsafe { input(centers,size(ny,nx)?)? };
    let out = unsafe { output(corners,size(ny+1,nx+1)?)? };
    if centers.iter().any(|v| !v.is_finite()) { return Err("invalid cell centers".into()); }
    // Bilinear centers with linear edge extrapolation, no nearest-column fill.
    let sample = |j: isize,i: isize| {
        let j0 = j.clamp(0,ny as isize-2) as usize;
        let i0 = i.clamp(0,nx as isize-2) as usize;
        let fy = j as f64-j0 as f64; let fx = i as f64-i0 as f64;
        centers[j0*nx+i0]*(1.0-fx)*(1.0-fy)
        + centers[j0*nx+i0+1]*fx*(1.0-fy)
        + centers[(j0+1)*nx+i0]*(1.0-fx)*fy
        + centers[(j0+1)*nx+i0+1]*fx*fy
    };
    for j in 0..=ny { for i in 0..=nx {
        out[j*(nx+1)+i] = 0.25*(sample(j as isize-1,i as isize-1)
            +sample(j as isize-1,i as isize)+sample(j as isize,i as isize-1)
            +sample(j as isize,i as isize));
    }}
    Ok(())
}) }

/// Projected source quadrilaterals to destination rectangles by exact clipping.
/// out[3] receipt = matched source integral, destination integral, valid area.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gpuwm_rain_gate_remap(
    field: *const f64, valid: *const u8, sx: *const f64, sy: *const f64,
    sny: usize, snx: usize, dx: *const f64, dy: *const f64,
    dny: usize, dnx: usize, result: *mut f64, result_valid: *mut u8,
    coverage: *mut f64, receipt: *mut f64,
) -> i32 { call(|| {
    let ns = size(sny,snx)?; let nd = size(dny,dnx)?;
    let f = unsafe { input(field,ns)? }; let v = unsafe { input(valid,ns)? };
    let sx = unsafe { input(sx,size(sny+1,snx+1)?)? };
    let sy = unsafe { input(sy,size(sny+1,snx+1)?)? };
    if sx.iter().chain(sy.iter()).any(|x| !x.is_finite()) { return Err("invalid cell corners".into()); }
    let dx = unsafe { input(dx,dnx+1)? }; let dy = unsafe { input(dy,dny+1)? };
    edges(dx)?; edges(dy)?;
    let out = unsafe { output(result,nd)? }; out.fill(0.0);
    let vo = unsafe { output(result_valid,nd)? }; vo.fill(0);
    let covered = unsafe { output(coverage,nd)? }; covered.fill(0.0);
    let receipt = unsafe { output(receipt,3)? };
    let mut observed = vec![0.0;nd]; let mut source_integral = 0.0;
    for j in 0..sny { for i in 0..snx {
        let q = [j*(snx+1)+i,j*(snx+1)+i+1,(j+1)*(snx+1)+i+1,(j+1)*(snx+1)+i];
        let poly = q.map(|k|(sx[k],sy[k]));
        let polyarea = area(&poly);
        if polyarea <= 0.0 { return Err("degenerate projected source cell".into()); }
        let mut sign = 0.0;
        for k in 0..4 {
            let a=poly[k]; let b=poly[(k+1)%4]; let c=poly[(k+2)%4];
            let cross=(b.0-a.0)*(c.1-b.1)-(b.1-a.1)*(c.0-b.0);
            if cross!=0.0 { if sign!=0.0 && sign*cross<0.0 { return Err("non-convex projected source cell".into()); } sign=cross; }
        }
        let xmin=poly.iter().map(|p|p.0).fold(f64::INFINITY,f64::min);
        let xmax=poly.iter().map(|p|p.0).fold(f64::NEG_INFINITY,f64::max);
        let ymin=poly.iter().map(|p|p.1).fold(f64::INFINITY,f64::min);
        let ymax=poly.iter().map(|p|p.1).fold(f64::NEG_INFINITY,f64::max);
        let ia=dx.partition_point(|&x|x<=xmin).saturating_sub(1).min(dnx);
        let ib=dx.partition_point(|&x|x<xmax).min(dnx);
        let ja=dy.partition_point(|&y|y<=ymin).saturating_sub(1).min(dny);
        let jb=dy.partition_point(|&y|y<ymax).min(dny);
        for jj in ja..jb { for ii in ia..ib {
            let overlap=overlap(&poly,dx[ii],dx[ii+1],dy[jj],dy[jj+1]);
            let k=jj*dnx+ii; covered[k]+=overlap;
            if v[j*snx+i]!=0 && f[j*snx+i].is_finite() {
                observed[k]+=overlap; out[k]+=overlap*f[j*snx+i];
                source_integral+=overlap*f[j*snx+i];
            }
        }}
    }}
    let mut dest_integral=0.0; let mut validarea=0.0;
    for j in 0..dny { for i in 0..dnx {
        let k=j*dnx+i; let a=(dx[i+1]-dx[i])*(dy[j+1]-dy[j]);
        // Overlapping cells are invalid geometry, not extra precipitation.
        if covered[k] > a*(1.0+1e-7) { return Err("source cell overlap exceeds destination area".into()); }
        dest_integral+=out[k];
        out[k]/=a;
        vo[k]=u8::from(observed[k]>=a*(1.0-1e-7) && covered[k]<=a*(1.0+1e-7));
        if vo[k]!=0 { validarea+=a; }
        covered[k]/=a;
    }}
    receipt.copy_from_slice(&[source_integral,dest_integral,validarea]);
    Ok(())
}) }

/// mode 0 cumulative mm, 1 rate mm/hour, 2 thresholded echo occupancy.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gpuwm_rain_gate_hour(
    times: *const f64, fields: *const f64, valid: *const u8,
    reset_ids: *const i64, carry: *const f64, nt: usize, n: usize,
    start: f64, end: f64, max_gap: f64, mode: u32,
    result: *mut f64, result_valid: *mut u8, footprint: *mut u8,
    receipt: *mut f64,
) -> i32 { call(|| {
    let t=unsafe{input(times,nt)?}; let f=unsafe{input(fields,size(nt,n)?)?};
    let v=unsafe{input(valid,size(nt,n)?)?}; let ids=unsafe{input(reset_ids,nt)?};
    let carry=unsafe{input(carry,size(nt,n)?)?};
    let out=unsafe{output(result,n)?}; out.fill(0.0);
    let vo=unsafe{output(result_valid,n)?}; vo.fill(1);
    let footprint=unsafe{output(footprint,n)?}; footprint.fill(0);
    let receipt=unsafe{output(receipt,3)?};
    if nt<2 || mode>2 || end<=start || max_gap<=0.0 || !max_gap.is_finite()
      || t.iter().any(|x|!x.is_finite()) || t.windows(2).any(|p|p[1]<=p[0]) {
        return Err("invalid hourly series or mode".into());
    }
    let a=if mode==0 {t.iter().position(|x|(*x-start).abs()<1e-6).ok_or("missing exact hourly start frame")?}
          else {t.iter().rposition(|x|*x<=start).ok_or("missing bracketing hourly start frame")?};
    let b=if mode==0 {t.iter().position(|x|(*x-end).abs()<1e-6).ok_or("missing exact hourly end frame")?}
          else {t.iter().position(|x|*x>=end).ok_or("missing bracketing hourly end frame")?};
    if b<=a { return Err("hour endpoints are out of order".into()); }
    let mut resets=0; let mut largest: f64=0.0;
    for k in a..b {
        let dt=t[k+1]-t[k]; largest=largest.max(dt);
        if dt>max_gap+1e-6 { return Err("missing frame: hourly cadence exceeds declared maximum".into()); }
        let reset=ids[k+1]!=ids[k]; if reset && mode==0 {resets+=1;}
        let left=t[k].max(start); let right=t[k+1].min(end);
        let fa=(left-t[k])/dt; let fb=(right-t[k])/dt;
        for p in 0..n {
            let l=k*n+p; let r=(k+1)*n+p;
            if v[l]==0 || v[r]==0 || !f[l].is_finite() || !f[r].is_finite() { vo[p]=0;continue; }
            let delta=match mode {
                0=>f[r]-f[l]+if reset {carry[r]}else{0.0},
                1=>{let a=f[l]+fa*(f[r]-f[l]);let b=f[l]+fb*(f[r]-f[l]);
                    0.5*(a+b)*(right-left)/3600.0},
                _=>{let l=f64::from(f[l]>=35.0);let r=f64::from(f[r]>=35.0);
                    let a=l+fa*(r-l);let b=l+fb*(r-l);
                    if a>0.0 || b>0.0 {footprint[p]=1;}
                    0.5*(a+b)*(right-left)/(end-start)},
            };
            if !delta.is_finite() || delta < 0.0 || (mode<2 && (f[l]<0.0 || f[r]<0.0)) { vo[p]=0;continue; }
            out[p]+=delta;
        }
    }
    receipt.copy_from_slice(&[(b-a)as f64,resets as f64,largest]);
    Ok(())
}) }

/// Exact physical square neighborhoods, fractional edge-cell weights.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gpuwm_rain_gate_fss(
    model: *const f64, obs: *const f64, valid: *const u8,
    ny: usize, nx: usize, threshold: f64, width: f64, dx: f64,
    receipt: *mut f64, support: *mut u8,
) -> i32 { call(|| {
    let n=size(ny,nx)?; let m=unsafe{input(model,n)?}; let o=unsafe{input(obs,n)?};
    let v=unsafe{input(valid,n)?}; let out=unsafe{output(receipt,5)?};
    let supported=unsafe{output(support,n)?}; supported.fill(0);
    if !threshold.is_finite() || threshold<=0.0 || !width.is_finite() || !dx.is_finite() || width<=0.0 || dx<=0.0 {
        return Err("invalid physical FSS threshold or width".into());
    }
    let half=width/(2.0*dx); let radius=(half+0.5).ceil() as isize;
    let mut weights=Vec::new();
    for offset in -radius..=radius {
        let w=(half.min(offset as f64+0.5)-(-half).max(offset as f64-0.5)).max(0.0);
        if w>1e-13 {weights.push((offset,w));}
    }
    // Two separable passes instead of O(width^2) per center.
    let mut mh=vec![0.0;n]; let mut oh=vec![0.0;n]; let mut vh=vec![true;n];
    for j in 0..ny { for i in 0..nx { for &(offset,w) in &weights {
        let ii=i as isize+offset;
        let k=j*nx+i;
        if ii<0 || ii>=nx as isize {vh[k]=false;continue;}
        let p=j*nx+ii as usize;
        if v[p]==0 || !m[p].is_finite() || !o[p].is_finite() {vh[k]=false;continue;}
        mh[k]+=w*f64::from(m[p]>=threshold); oh[k]+=w*f64::from(o[p]>=threshold);
    }}}
    let normal=(width/dx).powi(2); let mut numerator=0.0; let mut denominator=0.0; let mut count=0;
    let mut wet=0;
    for j in 0..ny { for i in 0..nx {
        let mut good=true; let mut a=0.0; let mut b=0.0;
        for &(offset,w) in &weights {
            let jj=j as isize+offset;
            if jj<0 || jj>=ny as isize {good=false;continue;}
            let p=jj as usize*nx+i;
            if !vh[p] {good=false;continue;}
            a+=w*mh[p]; b+=w*oh[p];
        }
        if good {
            supported[j*nx+i]=1;
            a/=normal; b/=normal; numerator+=(a-b).powi(2); denominator+=a*a+b*b;count+=1;
            if b>0.0 {wet+=1;}
        }
    }}
    out.copy_from_slice(&[numerator,denominator,count as f64,wet as f64,dx*dx*count as f64]);
    Ok(())
}) }

#[unsafe(no_mangle)]
pub unsafe extern "C" fn gpuwm_rain_gate_combine(
    fields: *const f64, nfields: usize, n: usize, mode: u32, result: *mut f64,
) -> i32 { call(|| {
    if nfields==0 || mode>3 {return Err("invalid field combination".into());}
    let f=unsafe{input(fields,size(nfields,n)?)?}; let out=unsafe{output(result,n)?};
    for p in 0..n {
        let mut value=if mode==0 {0.0}else if mode==1 {f64::NEG_INFINITY}else{1.0};
        for k in 0..nfields {
            if !f[k*n+p].is_finite() {value=f64::NAN;break;}
            value=if mode==0 {value+f[k*n+p]}else if mode==1 {value.max(f[k*n+p])}
                  else {if mode==2 && f[k*n+p]!=0.0 && f[k*n+p]!=1.0 {return Err("validity intersection needs binary masks".into());}
                        value*f[k*n+p]};
        }
        out[p]=value;
    }
    Ok(())
}) }

#[unsafe(no_mangle)]
pub unsafe extern "C" fn gpuwm_rain_gate_mask(
    field: *const f64, valid: *const u8, n: usize, threshold: f64, result: *mut u8,
) -> i32 { call(|| {
    let f=unsafe{input(field,n)?};let v=unsafe{input(valid,n)?};let out=unsafe{output(result,n)?};
    if !threshold.is_finite(){return Err("quality threshold must be finite".into());}
    for p in 0..n {out[p]=u8::from(v[p]!=0 && f[p].is_finite() && f[p]>=threshold);}
    Ok(())
}) }

#[unsafe(no_mangle)]
pub unsafe extern "C" fn gpuwm_rain_gate_sum(
    field: *const f64, mask: *const u8, n: usize, result: *mut f64,
) -> i32 { call(|| {
    let f=unsafe{input(field,n)?}; let v=unsafe{input(mask,n)?};
    let out=unsafe{output(result,1)?};
    let mut sum=0.0;
    for i in 0..n { if v[i]!=0 {if !f[i].is_finite(){return Err("non-finite scored value".into());}sum+=f[i];}}
    out[0]=sum; Ok(())
}) }
