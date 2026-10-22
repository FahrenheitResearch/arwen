//! Native model-level water-vapour integrals, with no isobaric resampling.
//!
//! The trapezoid rule integrates q, q*u, q*v on every native pressure segment,
//! clipping segments analytically at the integration bounds. Endpoint values
//! extend only to the supplied surface and model-top interfaces. Winds must be
//! earth relative. WRF mixing ratio r becomes specific humidity q=r/(1+r).
//! IWV and verification vectors span surface to model top; operational IVT
//! spans min(surface,1000 hPa) to 300 hPa. A finite model lid is explicitly
//! distinct from ERA5's total-atmosphere flux.
use rayon::prelude::*;
use crate::CalcError;

pub const IVT_GRAVITY: f64 = 9.80665;
pub const AR_THRESHOLD: f32 = 250.0;

pub struct IvtInputs<'a> {
    pub cells: usize,
    pub levels: usize,
    pub pressure_pa: &'a [f64],
    pub mixing_ratio: &'a [f64],
    pub eastward_ms: &'a [f64],
    pub northward_ms: &'a [f64],
    pub surface_pressure_pa: &'a [f64],
    pub model_top_pa: f64,
}

#[derive(Debug)]
pub struct IvtFields {
    pub eastward: Vec<f32>,
    pub northward: Vec<f32>,
    pub magnitude: Vec<f32>,
    pub iwv: Vec<f32>,
    pub full_eastward: Vec<f32>,
    pub full_northward: Vec<f32>,
    pub full_magnitude: Vec<f32>,
}

/// Invalid columns return NaN, never a partial sum masquerading as dry air.
/// Pressure decreases strictly upward; surface and model lid bracket it.
pub fn compute_ivt(input: IvtInputs<'_>) -> Result<IvtFields, CalcError> {
    let invalid = |reason| CalcError::InvalidConfig { field: "ivt", reason };
    if input.levels < 2 || input.cells == 0 {
        return Err(invalid("need at least two native levels and one cell"));
    }
    let size = input.cells.checked_mul(input.levels).ok_or_else(||invalid("shape overflow"))?;
    for (field,values,expected) in [
        ("pressure_pa",input.pressure_pa,size), ("mixing_ratio",input.mixing_ratio,size),
        ("eastward_ms",input.eastward_ms,size), ("northward_ms",input.northward_ms,size),
        ("surface_pressure_pa",input.surface_pressure_pa,input.cells),
    ] {
        if values.len()!=expected {
            return Err(CalcError::LengthMismatch {field,expected,actual:values.len()});
        }
    }
    if !input.model_top_pa.is_finite() || input.model_top_pa<0.0 || input.model_top_pa>30000.0 {
        return Err(invalid("model top must be finite and at or above 300 hPa"));
    }
    let rows: Vec<[f32;7]> = (0..input.cells).into_par_iter().map(|cell| {
        let missing = [f32::NAN;7];
        let surface = input.surface_pressure_pa[cell];
        if !surface.is_finite() || surface<=30000.0 { return missing; }
        let sample = |k: usize| {
            let i=k*input.cells+cell;
            let r=input.mixing_ratio[i];
            let q=r/(1.0+r);
            [q,q*input.eastward_ms[i],q*input.northward_ms[i]]
        };
        let first=input.pressure_pa[cell];
        let last=input.pressure_pa[(input.levels-1)*input.cells+cell];
        if first>surface || last<input.model_top_pa { return missing; }
        let mut full=[0.0;3];
        let mut bounded=[0.0;3];
        let mut p0=surface;
        let mut f0=sample(0);
        for k in 0..=input.levels {
            let (p1,f1)=if k<input.levels {
                let i=k*input.cells+cell;
                let r=input.mixing_ratio[i];
                if !r.is_finite() || r<0.0 { return missing; }
                (input.pressure_pa[i],sample(k))
            } else { (input.model_top_pa,sample(input.levels-1)) };
            if !p1.is_finite() || p1<0.0 || p1>p0 || (k>0 && k<input.levels && p1==p0)
                || f0.iter().chain(f1.iter()).any(|v|!v.is_finite()) { return missing; }
            if p0>p1 {
                let hi=p0.min(surface.min(100000.0));
                let lo=p1.max(30000.0);
                for j in 0..3 {
                    full[j]+=0.5*(f0[j]+f1[j])*(p0-p1)/IVT_GRAVITY;
                    if hi>lo {
                        let at=|p:f64| f0[j]+(f1[j]-f0[j])*(p0-p)/(p0-p1);
                        bounded[j]+=0.5*(at(hi)+at(lo))*(hi-lo)/IVT_GRAVITY;
                    }
                }
            }
            p0=p1; f0=f1;
        }
        [bounded[1] as f32,bounded[2] as f32,bounded[1].hypot(bounded[2]) as f32,
            full[0] as f32,full[1] as f32,full[2] as f32,full[1].hypot(full[2]) as f32]
    }).collect();
    let plane=|j:usize| rows.iter().map(|r|r[j]).collect();
    Ok(IvtFields {eastward:plane(0),northward:plane(1),magnitude:plane(2),iwv:plane(3),
        full_eastward:plane(4),full_northward:plane(5),full_magnitude:plane(6)})
}

#[cfg(test)]
mod tests {
    use super::*;
    fn run(p:&[f64],r:&[f64],u:&[f64],v:&[f64],ps:f64)->IvtFields {
        compute_ivt(IvtInputs {cells:1,levels:p.len(),pressure_pa:p,mixing_ratio:r,eastward_ms:u,
            northward_ms:v,surface_pressure_pa:&[ps],model_top_pa:10000.0}).unwrap()
    }
    fn close(a:f32,b:f64) { assert!((a as f64-b).abs()<b.abs().max(1.0)*2e-6,"{a} vs {b}"); }
    #[test] fn ivt_native_irregular_levels_specific_humidity_and_units() {
        let p=[98000.,87777.,43000.,15000.];
        let out=run(&p,&[0.01;4],&[10.;4],&[-20.;4],102000.);
        let q=0.01/1.01; let water=q*70000./IVT_GRAVITY;
        close(out.eastward[0],10.*water); close(out.northward[0],-20.*water);
        close(out.magnitude[0],500_f64.sqrt()*water); close(out.iwv[0],q*92000./IVT_GRAVITY);
        close(out.full_eastward[0],q*92000.*10./IVT_GRAVITY);
    }
    #[test] fn ivt_terrain_clipping_has_no_below_ground_mass() {
        let out=run(&[78000.,45000.,15000.],&[0.01;3],&[10.;3],&[0.;3],80000.);
        close(out.eastward[0],0.01/1.01*10.*50000./IVT_GRAVITY);
    }
    #[test] fn ivt_integrates_vector_before_magnitude_and_clips_native_segments() {
        let out=run(&[100000.,65000.,30000.,10000.],&[0.01;4],&[10.,0.,-10.,-10.],&[0.;4],100000.);
        close(out.magnitude[0],0.);
        let out=run(&[100000.,55000.,10000.],&[0.01;3],&[100.,55.,10.],&[0.;3],100000.);
        close(out.eastward[0],0.01/1.01*65.*70000./IVT_GRAVITY);
    }
    #[test] fn ivt_invalid_columns_do_not_become_zero_or_partial_integrals() {
        for (p,r) in [([90000.,95000.,10000.],[0.01;3]),([90000.,50000.,10000.],[0.01,f64::NAN,0.01]),
            ([90000.,50000.,10000.],[0.01,-0.01,0.01])] {
            assert!(run(&p,&r,&[10.;3],&[0.;3],100000.).magnitude[0].is_nan());
        }
        assert!(run(&[90000.,50000.,10000.],&[0.01;3],&[f64::NAN,10.,10.],&[0.;3],100000.).iwv[0].is_nan());
    }
    #[test] fn ivt_threshold_is_in_flux_units() {
        let speed=250.*IVT_GRAVITY/(70000.*0.01/1.01);
        let out=run(&[95000.,50000.,10000.],&[0.01;3],&[speed;3],&[0.;3],100000.);
        close(out.magnitude[0],250.); assert!(out.magnitude[0]>=AR_THRESHOLD);
    }
}
