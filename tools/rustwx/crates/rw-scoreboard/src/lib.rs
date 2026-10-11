//! Shared native scoring. Readers, campaigns and nightly jobs use this owner.
use serde::{Deserialize, Serialize};

#[derive(Clone, Default, Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct Stats {
    pub count: u64,
    pub sum_error: f64,
    pub sum_absolute_error: f64,
    pub sum_squared_error: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hits: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub misses: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub false_alarms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub correct_negatives: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fractions_squared_error: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fractions_reference: Option<f64>,
}
impl Stats {
    pub fn add(&mut self, forecast: f64, observed: f64) {
        self.add_error(forecast - observed);
    }
    pub fn add_error(&mut self, error: f64) {
        self.count += 1;
        self.sum_error += error;
        self.sum_absolute_error += error.abs();
        self.sum_squared_error += error * error;
    }
    pub fn merge(&mut self, other: &Self) {
        self.count += other.count;
        self.sum_error += other.sum_error;
        self.sum_absolute_error += other.sum_absolute_error;
        self.sum_squared_error += other.sum_squared_error;
        macro_rules! add { ($($field:ident),*) => { $(if let Some(v) = other.$field { *self.$field.get_or_insert(Default::default()) += v; })* }; }
        add!(hits, misses, false_alarms, correct_negatives, fractions_squared_error, fractions_reference);
    }
    pub fn metric(&self, name: &str) -> Option<f64> {
        if self.count == 0 { return None; }
        match name {
            "rmse" => Some((self.sum_squared_error / self.count as f64).sqrt()),
            "mae" => Some(self.sum_absolute_error / self.count as f64),
            "bias" => Some(self.sum_error / self.count as f64),
            "fss" => self.fractions_reference.filter(|v| *v > 0.).map(|d| 1. - self.fractions_squared_error.unwrap_or(0.) / d),
            "ets" => {
                let (h,m,a,c)=(self.hits? as f64,self.misses? as f64,self.false_alarms? as f64,self.correct_negatives? as f64);
                let random=(h+m)*(h+a)/(h+m+a+c);
                let denominator=h+m+a-random;
                (denominator != 0.).then(|| (h-random)/denominator)
            }
            _ => None,
        }
    }
}

pub fn prefix(field: &[f64], ny: usize, nx: usize) -> Vec<f64> {
    let mut out=vec![0.;(ny+1)*(nx+1)];
    for y in 0..ny {
        let mut row=0.;
        for x in 0..nx {
            row+=field[y*nx+x];
            out[(y+1)*(nx+1)+x+1]=out[y*(nx+1)+x+1]+row;
        }
    }
    out
}
pub fn rectangle(sums: &[f64], nx: usize, x0: usize, y0: usize, x1: usize, y1: usize) -> f64 {
    sums[y1*(nx+1)+x1]-sums[y0*(nx+1)+x1]-sums[y1*(nx+1)+x0]+sums[y0*(nx+1)+x0]
}

/// Frozen methods differ in edge support, not in their scoring implementation.
/// `complete` requires full finite boxes; false retains clipped-mask v1 boxes.
pub fn fractions(forecast:&[f64],observed:&[f64],valid:&[bool],nx:usize,ny:usize,threshold:f64,half_widths:&[usize],half_height:usize,complete:bool)->Stats {
    assert_eq!(forecast.len(),nx*ny);assert_eq!(observed.len(),nx*ny);assert_eq!(valid.len(),nx*ny);assert_eq!(half_widths.len(),ny);
    let mask=prefix(&valid.iter().map(|&v|f64::from(v)).collect::<Vec<_>>(),ny,nx);
    let f=prefix(&(0..forecast.len()).map(|i|f64::from(valid[i] && forecast[i]>=threshold)).collect::<Vec<_>>(),ny,nx);
    let o=prefix(&(0..observed.len()).map(|i|f64::from(valid[i] && observed[i]>=threshold)).collect::<Vec<_>>(),ny,nx);
    let(mut n,mut error,mut reference)=(0u64,0.,0.);
    for (y,&half) in half_widths.iter().enumerate() {
        for x in 0..nx {
            if complete {
                if x<half || y<half_height || x.saturating_add(half)>=nx || y.saturating_add(half_height)>=ny {continue;}
            } else if !valid[y*nx+x] {continue;}
            let(x0,y0,x1,y1)=(x.saturating_sub(half),y.saturating_sub(half_height),(x+half+1).min(nx),(y+half_height+1).min(ny));
            let count=rectangle(&mask,nx,x0,y0,x1,y1);
            if count==0. || complete && count!=((x1-x0)*(y1-y0)) as f64 {continue;}
            let fa=rectangle(&f,nx,x0,y0,x1,y1)/count;
            let ob=rectangle(&o,nx,x0,y0,x1,y1)/count;
            error+=(fa-ob).powi(2);reference+=fa*fa+ob*ob;n+=1;
        }
    }
    Stats{count:n,fractions_squared_error:Some(error),fractions_reference:Some(reference),..Stats::default()}
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn pool_sufficient_statistics_not_rmse() {
        let mut a=Stats::default(); a.add(1.,0.);
        let mut b=Stats::default(); b.add(3.,0.); b.add(3.,0.);
        a.merge(&b);
        assert_eq!(a.metric("rmse"),Some((19_f64/3.).sqrt()));
        assert_eq!(Stats::default().metric("rmse"),None);
    }
    #[test]
    fn dry_ets_is_undefined() {
        let s=Stats{count:4,hits:Some(0),misses:Some(0),false_alarms:Some(0),correct_negatives:Some(4),..Stats::default()};
        assert_eq!(s.metric("ets"),None);
    }
    #[test]
    fn both_frozen_neighborhood_policies_keep_their_support() {
        let a=vec![1.;9];let mut mask=vec![true;9];mask[0]=false;
        assert_eq!(fractions(&a,&a,&mask,3,3,0.5,&[1;3],1,true).count,0);
        let s=fractions(&a,&a,&mask,3,3,0.5,&[1;3],1,false);
        assert_eq!(s.count,8);assert_eq!(s.metric("fss"),Some(1.));
    }
    #[test]
    fn compact_float_transport_is_bitwise_for_every_build_target_set() {
        let mut state=107u64;
        for _ in 0..2000 {
            state^=state<<13;state^=state>>7;state^=state<<17;
            let original=f64::from_bits((1023u64<<52)|(state&((1u64<<52)-1)));
            let bytes=serde_json::to_vec(&original).unwrap();
            let decoded:f64=serde_json::from_slice(&bytes).unwrap();
            assert_eq!(original.to_bits(),decoded.to_bits(),"{}",String::from_utf8_lossy(&bytes));
        }
    }
}
