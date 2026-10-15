//! RUC depth-node initialization, WRF v4.6.1 module_soil_pre.F,
//! init_soil_depth_3 and init_soil_3_real. No four-to-nine padding.
use crate::error::{MpasError, MpasResult};
pub const DEPTHS: [f32; 9] = [0., 0.01, 0.04, 0.10, 0.30, 0.60, 1., 1.60, 3.];
pub fn thicknesses() -> Vec<f32> {
    let mut edges = [0.; 9];
    let mut dz = vec![0.; 9];
    edges[1] = (DEPTHS[1] + DEPTHS[0]) * 0.5;
    dz[0] = edges[1];
    for k in 1..8 {
        edges[k] = (DEPTHS[k+1] + DEPTHS[k]) * 0.5;
        dz[k] = edges[k] - edges[k-1];
    }
    dz[8] = DEPTHS[8] - edges[7];
    dz
}
pub fn remap(depths: &[f32], samples: &[f32]) -> MpasResult<Vec<f32>> {
    if depths.len() != samples.len() || depths.len() < 2
        || depths.windows(2).any(|w| w[0] >= w[1])
        || depths.iter().chain(samples).any(|x| !x.is_finite()) {
        return Err(MpasError::Refusal("RUC requires finite samples on increasing source depths".into()));
    }
    DEPTHS.iter().map(|&z| {
        let k = depths.windows(2).position(|w| w[0] <= z && z <= w[1])
            .ok_or_else(|| MpasError::Refusal(format!("RUC depth {z} m has no source bracket")))?;
        Ok((samples[k] * (depths[k+1]-z) + samples[k+1] * (z-depths[k]))
            / (depths[k+1]-depths[k]))
    }).collect()
}
/// One land column from a layer source (WPS `FLAG_SOIL_LAYERS`, e.g. GFS or
/// ERA5), as WRF v4.6.1 `init_soil_3_real` builds its samples before the
/// interpolation: temperature is anchored at 0 m by the skin temperature and
/// at 3 m by the deep temperature; moisture at 0 m by a linear extrapolation
/// off the top two layers (in WRF's own expression order, dividing by the
/// difference of the integer centimetre mid-depths) and at 3 m by repeating
/// the deepest layer. `mid_cm` are the layer mid-depths in centimetres.
pub fn layer_column(mid_cm: &[f32], st: &[f32], sm: &[f32], tsk: f32, tmn: f32)
    -> MpasResult<(Vec<f32>, Vec<f32>)> {
    if mid_cm.len() < 2 || st.len() != mid_cm.len() || sm.len() != mid_cm.len() {
        return Err(MpasError::Refusal("RUC layer source needs at least two layers with temperature and moisture".into()));
    }
    let mut depths = Vec::with_capacity(mid_cm.len() + 2);
    depths.push(0.0f32);
    depths.extend(mid_cm.iter().map(|z| z / 100.0f32));
    depths.push(300.0f32 / 100.0f32);
    let mut t = Vec::with_capacity(depths.len());
    t.push(tsk); t.extend_from_slice(st); t.push(tmn);
    let span = mid_cm[1] - mid_cm[0];
    let surface = (sm[0] - sm[1]) / span * mid_cm[0] + sm[0];
    let mut m = Vec::with_capacity(depths.len());
    m.push(surface); m.extend_from_slice(sm); m.push(sm[sm.len() - 1]);
    Ok((remap(&depths, &t)?, remap(&depths, &m)?))
}

/// One land column from a node source (WPS `FLAG_SOIL_LEVELS`, e.g. HRRR,
/// RAP or RRFS): the nodes are the samples, with no anchors.
pub fn node_column(depth_cm: &[f32], st: &[f32], sm: &[f32]) -> MpasResult<(Vec<f32>, Vec<f32>)> {
    let depths: Vec<f32> = depth_cm.iter().map(|z| z / 100.0f32).collect();
    Ok((remap(&depths, st)?, remap(&depths, sm)?))
}

#[cfg(test)] mod tests {
    use super::*;
    /// Every oracle land column WRF's default path produces (flag_sm_adj = 0
    /// for layers; nodes ignore the flag) must be reproduced bit for bit by
    /// the column builders the init uses. The adjusted and reversed-order
    /// layer experiments are WRF behaviours this init does not select.
    #[test] fn every_default_wrf_oracle_land_column_is_bit_identical() {
        let csv=include_str!("../../../../../../gpuwm/data/ruc/oracle/soil_ingest.csv");
        let mut lines=csv.lines().filter(|line| !line.starts_with('#') && !line.is_empty());
        let names:Vec<_>=lines.next().unwrap().trim_start_matches('\u{feff}').split(',').collect();
        let (mut layers, mut nodes) = (0, 0);
        for line in lines {
            let row:Vec<_>=line.split(',').collect();
            let get=|name:&str| row[names.iter().position(|n|*n==name).unwrap()];
            let num=|name:&str| get(name).trim().parse::<f32>().unwrap();
            if num("landmask") != 1. { continue; }
            let n: usize = get("nlev").parse().unwrap();
            let read=|prefix:&str, count:usize| (1..=count).map(|k|num(&format!("{prefix}_{k}"))).collect::<Vec<_>>();
            let (t, m) = if get("flag_soil_layers") == "1" {
                if get("flag_sm_adj") != "0" { continue; }
                layers += 1;
                layer_column(&read("lev_cm", n), &read("st_src", n), &read("sm_src", n), num("tsk_in"), num("tmn")).unwrap()
            } else {
                nodes += 1;
                node_column(&read("lev_cm", n), &read("st_src", n), &read("sm_src", n)).unwrap()
            };
            for (name, got) in [("tslb", t), ("smois", m)] {
                for (k,(a,b)) in got.into_iter().zip(read(name, 9)).enumerate() {
                    assert_eq!(a.to_bits(),b.to_bits(),"{} {name} level {}: {a} vs WRF {b}",get("experiment"),k+1);
                }
            }
        }
        assert!(layers >= 30 && nodes >= 30, "oracle coverage fell: {layers} layer and {nodes} node columns");
    }
    #[test] fn node_identity_and_layer_interpolation() {
        let samples = [281., 282., 283., 284., 285., 286., 287., 288., 289.];
        assert_eq!(remap(&DEPTHS, &samples).unwrap(), samples);
        let out = remap(&[0., 0.05, 0.25, 0.70, 1.50, 3.], &[280.,281.,285.,294.,310.,340.]).unwrap();
        for (z, value) in DEPTHS.iter().zip(out) { assert!((value-(280.+20.*z)).abs() < 0.0001); }
    }
    #[test] fn refuses_missing_deep_source_and_nan() {
        assert!(remap(&[0.,1.], &[280.,290.]).is_err());
        assert!(remap(&[0.,3.], &[f32::NAN,290.]).is_err());
        assert_eq!(thicknesses(), vec![0.005,0.024999999,0.045,0.13,0.25,0.35,0.49999994,1.,0.70000005]);
    }
    #[test] fn source_node_columns_match_the_retained_wrf_soil_oracle() {
        let csv=include_str!("../../../../../../gpuwm/data/ruc/oracle/soil_ingest.csv");
        let mut lines=csv.lines().filter(|line| !line.starts_with('#') && !line.is_empty());
        let names:Vec<_>=lines.next().unwrap().trim_start_matches('\u{feff}').split(',').collect();
        let mut checked=0;
        for line in lines {
            let row:Vec<_>=line.split(',').collect();
            let get=|name:&str| row[names.iter().position(|n|*n==name).unwrap()];
            if get("flag_soil_levels")!="1" || get("nlev")!="9" || get("landmask").parse::<f32>().unwrap()!=1. {continue;}
            let read=|prefix:&str| (1..=9).map(|k|get(&format!("{prefix}_{k}")).parse::<f32>().unwrap()).collect::<Vec<_>>();
            let depths:Vec<f32>=read("lev_cm").into_iter().map(|z|z/100.).collect();
            for (source,target) in [("st_src","tslb"),("sm_src","smois")] {
                let got=remap(&depths,&read(source)).unwrap();
                for (k,(a,b)) in got.into_iter().zip(read(target)).enumerate() {
                    assert_eq!(a.to_bits(),b.to_bits(),"{} {source} level {}",get("experiment"),k+1);
                }
            }
            checked+=1;
        }
        assert!(checked>=4,"the WRF oracle must exercise multiple source node columns");
    }
}
