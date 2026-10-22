//! Bounded CF regular-grid lake field sampling, parallel across target points.
//! No land donor extrapolation: finite local bilinear donors are renormalized;
//! absent coverage stays NaN so the caller retains its forcing and counts it.
use std::slice;
use crate::parallel::run_ranges;

fn coordinate(axis: &[f64], x: f64) -> Option<(usize, f64)> {
    if axis.len()<2 || !x.is_finite() { return None; }
    let step=axis[1]-axis[0];
    if step==0.0 { return None; }
    let v=(x-axis[0])/step;
    if v<0.0 || v>(axis.len()-1) as f64 { return None; }
    let k=(v.floor() as usize).min(axis.len()-2);
    Some((k,v-k as f64))
}

/// Field-major arrays, arbitrary ascending/descending uniform axes. CF scale
/// and mask decoding precedes this entry in rw_netcdf; SI conversion is here.
#[no_mangle]
pub unsafe extern "C" fn gpuwm_lake_state_sample_f64(
    lat: *const f64, ny: usize, lon: *const f64, nx: usize,
    source: *const f64, gain: f64, offset: f64,
    target_lat: *const f64, target_lon: *const f64, n: usize,
    output: *mut f64, workers: usize) -> i32 {
    if lat.is_null() || lon.is_null() || source.is_null() || target_lat.is_null()
        || target_lon.is_null() || output.is_null() || ny<2 || nx<2
        || nx.checked_mul(ny).is_none() || !gain.is_finite() || !offset.is_finite() {return 1;}
    let la=slice::from_raw_parts(lat,ny); let lo=slice::from_raw_parts(lon,nx);
    for axis in [la,lo] {
        let d=axis[1]-axis[0];
        if d==0.0 || !axis.iter().all(|v|v.is_finite())
            || axis.windows(2).any(|w| ((w[1]-w[0])-d).abs()>d.abs()*1e-5) {return 2;}
    }
    let data=slice::from_raw_parts(source,nx*ny);
    let tla=slice::from_raw_parts(target_lat,n); let tlo=slice::from_raw_parts(target_lon,n);
    let address=output as usize;
    run_ranges(n,workers,|start,stop| {
        for q in start..stop {
            let mut result=f64::NAN;
            // Choose an equivalent longitude inside the actual source extent.
            // The first axis value does not identify its encoding: descending
            // grids crossing zero may start positive and end negative.
            // Every candidate still obeys regional no-extrapolation bounds.
            let longitude=[tlo[q],tlo[q]-360.0,tlo[q]+360.0].into_iter()
                .find_map(|x|coordinate(lo,x));
            if let (Some((j,fy)),Some((i,fx)))=(coordinate(la,tla[q]),longitude) {
                let mut sum=0.0; let mut mass=0.0;
                for (y,x,w) in [(j,i,(1.0-fy)*(1.0-fx)),(j,i+1,(1.0-fy)*fx),
                    (j+1,i,fy*(1.0-fx)),(j+1,i+1,fy*fx)] {
                    let v=data[y*nx+x];
                    if w>0.0 && v.is_finite() {sum+=w*v;mass+=w;}
                }
                if mass>0.0 {result=(sum/mass)*gain+offset;}
            }
            *((address as *mut f64).add(q))=result;
        }
    });
    0
}
/// Normalize source-declared representation noise around physical zero only.
/// Values outside the tolerance remain unchanged for physical range admission.
#[no_mangle]
pub unsafe extern "C" fn gpuwm_lake_state_zero_roundoff_f64(
    values: *mut f64, n: usize, tolerance: f64, workers: usize) -> i32 {
    if values.is_null() || !tolerance.is_finite() || tolerance < 0.0 { return 1; }
    let address=values as usize;
    run_ranges(n,workers,|start,stop| {
        for i in start..stop {
            let value=(address as *mut f64).add(i);
            if (*value).abs() <= tolerance { *value=0.0; }
        }
    });
    0
}
#[cfg(test)]
mod tests {
 use super::*;
 #[test] fn source_zero_roundoff_preserves_material_depths_and_missing() {
  let mut values=[-2.16840486e-19,2.16840486e-19,0.,-0.001,0.001,f64::NAN];
  let rc=unsafe{gpuwm_lake_state_zero_roundoff_f64(values.as_mut_ptr(),values.len(),1e-12,8)};
  assert_eq!(rc,0);assert_eq!(&values[..5],&[0.,0.,0.,-0.001,0.001]);assert!(values[5].is_nan());
  assert_eq!(unsafe{gpuwm_lake_state_zero_roundoff_f64(values.as_mut_ptr(),values.len(),-1.,8)},1);
 }
 #[test] fn descending_longitudes_crossing_zero_accept_both_encodings() {
  let la=[43.,42.];let lo=[1.,0.,-1.];let s=[10.,20.,30.,10.,20.,30.];
  let a=[42.5;4];let b=[-0.5,359.5,2.,362.];let mut out=[0.;4];
  let rc=unsafe{gpuwm_lake_state_sample_f64(la.as_ptr(),2,lo.as_ptr(),3,s.as_ptr(),1.,0.,a.as_ptr(),b.as_ptr(),4,out.as_mut_ptr(),8)};
  assert_eq!(rc,0);assert_eq!(&out[..2],&[25.,25.]);
  assert!(out[2..].iter().all(|v|v.is_nan()));
 }
 #[test] fn coasts_missing_values_descending_axes_and_si_units() {
  let la=[43.,42.];let lo=[-81.,-80.];let s=[10.,f64::NAN,12.,14.];
  let a=[42.5,44.,43.];let b=[-80.5,-80.5,-80.];let mut out=[0.;3];
  let rc=unsafe{gpuwm_lake_state_sample_f64(la.as_ptr(),2,lo.as_ptr(),2,s.as_ptr(),1.,273.15,a.as_ptr(),b.as_ptr(),3,out.as_mut_ptr(),8)};
  assert_eq!(rc,0); assert!((out[0]-285.15).abs()<1e-10);assert!(out[1].is_nan());assert!(out[2].is_nan());
 }
}

/// Decode the nearest standard-met WTMP report in a caller's bounded window.
/// NDBC sentinel temperatures and malformed dates cannot become observations.
/// Output is [report epoch seconds, water temperature K].
#[no_mangle]
pub unsafe extern "C" fn gpuwm_lake_ndbc_temperature_f64(
    text: *const u8, bytes: usize, epoch: i64, tolerance_seconds: i64,
    output: *mut f64) -> i32 {
    if text.is_null() || output.is_null() || tolerance_seconds<0 { return 1; }
    let Ok(input)=std::str::from_utf8(slice::from_raw_parts(text,bytes)) else {return 2;};
    let mut lines=input.lines();
    let Some(header)=lines.next() else {return 2;};
    let names:Vec<_>=header.trim_start_matches('#').split_whitespace().collect();
    let mut positions=Vec::new();
    for key in ["YY","MM","DD","hh","mm","WTMP"] {
        let Some(p)=names.iter().position(|x|*x==key) else {return 2;};
        positions.push(p);
    }
    let mut best:Option<(i64,i64,f64)>=None;
    for line in lines.filter(|line|!line.starts_with('#')) {
        let cells:Vec<_>=line.split_whitespace().collect();
        let values:Option<Vec<f64>>=positions.iter().map(|p|cells.get(*p)?.parse().ok()).collect();
        let Some(v)=values else {continue;};
        if v.iter().any(|v|!v.is_finite()) || !(-5.0..=45.0).contains(&v[5]) {continue;}
        let year=v[0] as i32;
        let year=if year<80 {2000+year} else if year<100 {1900+year} else {year};
        let Some(time)=chrono::NaiveDate::from_ymd_opt(year,v[1] as u32,v[2] as u32)
            .and_then(|d|d.and_hms_opt(v[3] as u32,v[4] as u32,0)) else {continue;};
        let stamp=time.and_utc().timestamp();
        let delta=stamp.abs_diff(epoch);
        if delta<=tolerance_seconds as u64 && best.as_ref().map_or(true,|(d,t,_)|
            delta<*d as u64 || (delta==*d as u64 && stamp<*t)) {
            best=Some((delta as i64,stamp,v[5]+273.15));
        }
    }
    let Some((_,stamp,temp))=best else {return 3;};
    *output=stamp as f64; *output.add(1)=temp; 0
}

#[cfg(test)]
mod buoy_tests {
    use super::*;
    #[test] fn water_not_air_missing_and_time_window() {
        let input=b"#YY MM DD hh mm ATMP WTMP\n2022 11 17 12 00 -4.0 99.0\n2022 11 17 12 10 -5.0 10.5\n2022 11 17 12 20 -6.0 11.0\n";
        let epoch=chrono::NaiveDate::from_ymd_opt(2022,11,17).unwrap().and_hms_opt(12,0,0).unwrap().and_utc().timestamp();
        let mut out=[0.;2];
        assert_eq!(unsafe{gpuwm_lake_ndbc_temperature_f64(input.as_ptr(),input.len(),epoch,900,out.as_mut_ptr())},0);
        assert_eq!(out,[epoch as f64+600.,283.65]);
        assert_eq!(unsafe{gpuwm_lake_ndbc_temperature_f64(input.as_ptr(),input.len(),epoch,300,out.as_mut_ptr())},3);
    }
}

/// Nearest finite wet cell for independent point verification. The caller
/// bounds the matching distance, so a dry coast or another lake is not used.
#[no_mangle]
pub unsafe extern "C" fn gpuwm_lake_state_nearest_f64(
    latitude: *const f64, longitude: *const f64, values: *const f64,
    mask: *const f64, n: usize, target_latitude: f64, target_longitude: f64,
    max_distance_km: f64, output: *mut f64) -> i32 {
    if latitude.is_null() || longitude.is_null() || values.is_null() || mask.is_null()
        || output.is_null() || !target_latitude.is_finite() || !target_longitude.is_finite()
        || !max_distance_km.is_finite() || max_distance_km<0.0 {return 1;}
    let la=slice::from_raw_parts(latitude,n);let lo=slice::from_raw_parts(longitude,n);
    let va=slice::from_raw_parts(values,n);let wet=slice::from_raw_parts(mask,n);
    let mut best:Option<(f64,usize)>=None;
    for k in 0..n {
        if wet[k]!=1.0 || !la[k].is_finite() || !lo[k].is_finite() || !va[k].is_finite() {continue;}
        let dlon=(lo[k]-target_longitude+180.0).rem_euclid(360.0)-180.0;
        let x=dlon.to_radians()*((la[k]+target_latitude)*0.5).to_radians().cos();
        let y=(la[k]-target_latitude).to_radians();
        let distance=6371.0*x.hypot(y);
        if distance<=max_distance_km && best.map_or(true,|(d,_)|distance<d) {best=Some((distance,k));}
    }
    let Some((distance,k))=best else {return 3;};
    *output=k as f64;*output.add(1)=distance;*output.add(2)=va[k];0
}

/// Native wet-cell diagnostic conversion. Land and absent donors stay missing.
#[no_mangle]
pub unsafe extern "C" fn gpuwm_lake_state_diagnostic_f64(
    values:*const f64, mask:*const f64, n:usize, gain:f64, offset:f64,
    output:*mut f64, workers:usize) -> i32 {
    if values.is_null() || mask.is_null() || output.is_null()
       || !gain.is_finite() || !offset.is_finite() {return 1;}
    let v=slice::from_raw_parts(values,n);let m=slice::from_raw_parts(mask,n);
    let address=output as usize;
    run_ranges(n,workers,|a,b| {for i in a..b {
        *((address as *mut f64).add(i))=if m[i]==1. && v[i].is_finite() {v[i]*gain+offset} else {f64::NAN};
    }});0
}

/// Paired finite wet-cell scores: count, bias, MAE, RMSE, maximum absolute error.
#[no_mangle]
pub unsafe extern "C" fn gpuwm_lake_state_scores_f64(
    prediction:*const f64, reference:*const f64, n:usize, output:*mut f64,
    workers:usize) -> i32 {
    use std::sync::Mutex;
    if prediction.is_null() || reference.is_null() || output.is_null() {return 1;}
    let p=slice::from_raw_parts(prediction,n);let r=slice::from_raw_parts(reference,n);
    let parts=Mutex::new(Vec::new());
    run_ranges(n,workers,|a,b| {
        let mut score=[0.;5];for i in a..b {if p[i].is_finite() && r[i].is_finite() {
            let e=p[i]-r[i];score[0]+=1.;score[1]+=e;score[2]+=e.abs();score[3]+=e*e;
            score[4]=f64::max(score[4],e.abs());
        }}parts.lock().unwrap().push((a,score));
    });
    let mut parts=parts.into_inner().unwrap();parts.sort_by_key(|x|x.0);
    let mut s=[0.;5];for (_,v) in parts {for k in 0..4 {s[k]+=v[k];}s[4]=f64::max(s[4],v[4]);}
    if s[0]>0. {for k in 1..4 {s[k]/=s[0];}s[3]=s[3].sqrt();}
    else {s[1..].fill(f64::NAN);}
    slice::from_raw_parts_mut(output,5).copy_from_slice(&s);0
}


/// Geographic request envelope and source intersection, reduced in parallel.
/// Missing coordinates cannot extend an acquisition to the whole globe.
#[no_mangle]
pub unsafe extern "C" fn gpuwm_lake_state_coverage_f64(
    latitude:*const f64, longitude:*const f64, n:usize,
    south:f64, north:f64, west:f64, east:f64, output:*mut f64, workers:usize) -> i32 {
    use std::sync::Mutex;
    if latitude.is_null() || longitude.is_null() || output.is_null()
        || ![south,north,west,east].iter().all(|v|v.is_finite())
        || south>north || west>east {return 1;}
    let la=slice::from_raw_parts(latitude,n);let lo=slice::from_raw_parts(longitude,n);
    let parts=Mutex::new(Vec::new());
    run_ranges(n,workers,|a,b| {
        let mut v=[f64::INFINITY,f64::NEG_INFINITY,f64::INFINITY,f64::NEG_INFINITY,0.];
        for i in a..b {
            if !la[i].is_finite() || !lo[i].is_finite() || la[i].abs()>90. {continue;}
            let x=(lo[i]+180.).rem_euclid(360.)-180.;
            v[0]=v[0].min(la[i]);v[1]=v[1].max(la[i]);
            v[2]=v[2].min(x);v[3]=v[3].max(x);
            if la[i]>=south && la[i]<=north && x>=west && x<=east {v[4]+=1.;}
        }
        parts.lock().unwrap().push(v);
    });
    let mut v=[f64::INFINITY,f64::NEG_INFINITY,f64::INFINITY,f64::NEG_INFINITY,0.];
    for p in parts.into_inner().unwrap() {
        v[0]=v[0].min(p[0]);v[1]=v[1].max(p[1]);v[2]=v[2].min(p[2]);v[3]=v[3].max(p[3]);v[4]+=p[4];
    }
    if !v[0].is_finite() {return 3;}
    slice::from_raw_parts_mut(output,5).copy_from_slice(&v);0
}
