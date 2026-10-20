//! Column -> rustwx-sounding `SoundingColumn` contract -> sharppyrs analysis,
//! with the user's analysis options (parcel depths, user parcel, storm
//! motion). All numerics are sharprs/sharppyrs; this module only selects
//! which of their functions run.
//!
//! ECAPE / NCAPE: never offered. `DerivedParams::compute` fills them
//! internally; they are overwritten with NaN here, never serialised, and the
//! two table cells are always blanked by the renderer.

use rustwx_sounding::{SoundingColumn, SoundingMetadata};
use serde_json::{Value, json};
use sharppyrs::sharprs;
use sharprs::params::cape::{ParcelResult, ParcelType as SP, define_parcel};

use crate::store::{RawColumn, Res};

const MS_TO_KT: f64 = 1.943_844_492_440_604_7;
const PRESSURE_EPSILON_HPA: f64 = 0.1;
const HEIGHT_EPSILON_M: f64 = 1.0;

/// Storm-motion choice. `Brm` keeps sharppyrs' default (parcel-based
/// Bunkers right, non-parcel fallback).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum StormMotion {
    Brm,
    Blm,
    Mean06,
    Mean06Npw,
    CorfidiUp,
    CorfidiDown,
    NonParcelBunkers,
    User { dir_deg: f64, spd_kt: f64 },
}

impl StormMotion {
    pub fn parse(s: &str, dir: Option<f64>, spd: Option<f64>) -> Res<Self> {
        Ok(match s {
            "" | "brm" | "bunkers_right" | "bunkers_rm" => StormMotion::Brm,
            "blm" | "bunkers_left" | "bunkers_lm" => StormMotion::Blm,
            "mean06" | "mean_0_6km" | "mean_wind" => StormMotion::Mean06,
            "mean06npw" | "mean_0_6km_npw" | "mean_wind_npw" => StormMotion::Mean06Npw,
            "corfidi_up" | "upshear" => StormMotion::CorfidiUp,
            "corfidi_down" | "corfidi_dn" | "downshear" => StormMotion::CorfidiDown,
            "bunkers_np" | "nonparcel_bunkers" => StormMotion::NonParcelBunkers,
            "user" => StormMotion::User {
                dir_deg: dir.ok_or("sm=user needs sm_dir (deg, direction FROM)")?,
                spd_kt: spd.ok_or("sm=user needs sm_spd (kt)")?,
            },
            other => return Err(format!("unknown sm '{other}'")),
        })
    }
    pub fn label(&self) -> String {
        match self {
            StormMotion::Brm => "Bunkers Right".into(),
            StormMotion::Blm => "Bunkers Left".into(),
            StormMotion::Mean06 => "0-6 km Mean".into(),
            StormMotion::Mean06Npw => "0-6 km Mean (npw)".into(),
            StormMotion::CorfidiUp => "Corfidi Up".into(),
            StormMotion::CorfidiDown => "Corfidi Down".into(),
            StormMotion::NonParcelBunkers => "Bunkers Right (non-parcel)".into(),
            StormMotion::User { .. } => "User storm motion".into(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct AnalysisOpts {
    pub ml_depth_hpa: f64,
    pub mu_depth_hpa: f64,
    /// (pres hPa, T C, Td C), each defaulting to the surface value:
    /// replaces the FCST parcel slot.
    pub user_parcel: Option<(Option<f64>, Option<f64>, Option<f64>)>,
    pub storm_motion: StormMotion,
}

impl Default for AnalysisOpts {
    fn default() -> Self {
        AnalysisOpts {
            ml_depth_hpa: 100.0,
            mu_depth_hpa: 300.0,
            user_parcel: None,
            storm_motion: StormMotion::Brm,
        }
    }
}

pub struct Analysis {
    pub column: SoundingColumn,
    pub prof: sharppyrs::Profile,
    pub derived: sharppyrs::DerivedParams,
    pub notes: Vec<String>,
    /// Some(label) when the storm motion is not the default Bunkers right.
    pub sm_label: Option<String>,
    pub user_parcel: bool,
}

/// Build the bridge column from a raw column. Same convention as rw-ui
/// `skewt::build_sounding_column` (production `wxprofile_sounding_render`):
/// start at the model surface (psfc / orography / 2 m T,Td / 10 m wind),
/// drop levels at or below it (p >= psfc - 0.1 hPa or z <= orog + 1 m),
/// keep p strictly decreasing and z strictly increasing, clamp Td <= T,
/// drop non-finite levels. Omega is carried when every kept level has it.
pub fn to_column(raw: &RawColumn, meta: SoundingMetadata) -> Res<SoundingColumn> {
    let mut c = SoundingColumn {
        pressure_hpa: vec![],
        height_m_msl: vec![],
        temperature_c: vec![],
        dewpoint_c: vec![],
        u_ms: vec![],
        v_ms: vec![],
        omega_pa_s: vec![],
        metadata: meta,
    };
    let s = &raw.sfc;
    let mut omega = vec![];
    let mut push =
        |c: &mut SoundingColumn, p: f64, z: f64, t: f64, td: f64, u: f64, v: f64, om: f64| {
            if ![p, z, t, td, u, v].iter().all(|x| x.is_finite()) {
                return;
            }
            if let (Some(lp), Some(lz)) = (c.pressure_hpa.last(), c.height_m_msl.last())
                && (p >= *lp || z <= *lz)
            {
                return;
            }
            c.pressure_hpa.push(p);
            c.height_m_msl.push(z);
            c.temperature_c.push(t);
            c.dewpoint_c.push(td.min(t));
            c.u_ms.push(u);
            c.v_ms.push(v);
            omega.push(om);
        };
    push(
        &mut c,
        s.psfc_hpa,
        s.orog_m,
        s.t2_c,
        s.td2_c,
        s.u10,
        s.v10,
        f64::NAN,
    );
    if c.pressure_hpa.is_empty() {
        return Err(
            "surface anchor (psfc / orography / 2 m / 10 m) is missing at this point".into(),
        );
    }
    for k in 0..raw.levels() {
        if raw.p[k] >= s.psfc_hpa - PRESSURE_EPSILON_HPA || raw.z[k] <= s.orog_m + HEIGHT_EPSILON_M
        {
            continue;
        }
        push(
            &mut c,
            raw.p[k],
            raw.z[k],
            raw.t[k],
            raw.td[k],
            raw.u[k],
            raw.v[k],
            raw.omega.get(k).copied().unwrap_or(f64::NAN),
        );
    }
    // omega meter: the surface row has no omega; give it the first level's
    // value so the column stays rectangular, and only when the levels have it.
    if omega.iter().skip(1).all(|x| x.is_finite()) && omega.len() > 1 {
        omega[0] = 0.0;
        c.omega_pa_s = omega;
    }
    if c.len() < 2 {
        return Err(format!(
            "only {} usable level(s) above the surface",
            c.len()
        ));
    }
    c.validate().map_err(|e| e.to_string())?;
    Ok(c)
}

fn lift(prof: &sharppyrs::Profile, t: SP) -> ParcelResult {
    let cp = sharppyrs::extras::cape_profile(&prof.inner);
    sharppyrs::extras::parcelx_sharppy(&cp, &define_parcel(&cp, t), None, None)
}

/// Run sharppyrs on the column with the requested analysis options.
pub fn analyze(column: SoundingColumn, opts: &AnalysisOpts) -> Res<Analysis> {
    let wspd: Vec<f64> = column
        .u_ms
        .iter()
        .zip(&column.v_ms)
        .map(|(u, v)| u.hypot(*v) * MS_TO_KT)
        .collect();
    let wdir: Vec<f64> = column
        .u_ms
        .iter()
        .zip(&column.v_ms)
        .map(|(u, v)| (-u).atan2(-v).to_degrees().rem_euclid(360.0))
        .collect();
    let mut prof = sharppyrs::Profile::new(sharppyrs::SoundingData {
        pres: column.pressure_hpa.clone(),
        hght: column.height_m_msl.clone(),
        tmpc: column.temperature_c.clone(),
        dwpc: column.dewpoint_c.clone(),
        wdir,
        wspd,
        omeg: (!column.omega_pa_s.is_empty()).then(|| column.omega_pa_s.clone()),
        latitude: column.metadata.latitude_deg,
        longitude: column.metadata.longitude_deg,
        missing: None,
    })
    .ok_or("sharppyrs rejected the sounding profile")?;
    let mut notes = vec![];
    if (opts.ml_depth_hpa - 100.0).abs() > 1e-9 {
        prof.mlpcl = lift(
            &prof,
            SP::MixedLayer {
                depth_hpa: opts.ml_depth_hpa,
            },
        );
        notes.push(format!(
            "ML parcel uses a {:.0} hPa mixed layer",
            opts.ml_depth_hpa
        ));
    }
    if (opts.mu_depth_hpa - 300.0).abs() > 1e-9 {
        prof.mupcl = lift(
            &prof,
            SP::MostUnstable {
                depth_hpa: opts.mu_depth_hpa,
            },
        );
        notes.push(format!(
            "MU parcel searched over the lowest {:.0} hPa; effective inflow layer stays SHARPpy's (MU 300 hPa)",
            opts.mu_depth_hpa
        ));
    }
    let user_parcel = opts.user_parcel.is_some();
    if let Some((p, t, td)) = opts.user_parcel {
        let sfc = prof.inner.sfc;
        let p = p.unwrap_or(prof.inner.pres[sfc]);
        let t = t.unwrap_or(prof.inner.tmpc[sfc]);
        let td = td.unwrap_or(prof.inner.dwpc[sfc]);
        prof.fcstpcl = lift(
            &prof,
            SP::UserDefined {
                pres: p,
                tmpc: t,
                dwpc: td,
            },
        );
        notes.push(format!(
            "FCST parcel row holds the user parcel {p:.0} hPa {t:.1}/{td:.1} C"
        ));
    }
    let mut sm_label = None;
    if opts.storm_motion != StormMotion::Brm {
        let inner = &prof.inner;
        let sfc_p = inner.sfc_pressure();
        let p6 = inner.pres_at_height(inner.to_msl(6000.0));
        let (lu, lv) = (prof.srwind.2, prof.srwind.3);
        let uv: Option<(f64, f64)> = match opts.storm_motion {
            StormMotion::Blm => Some((lu, lv)),
            StormMotion::Mean06 => sharprs::winds::mean_wind(inner, sfc_p, p6, -1.0, 0.0, 0.0).ok(),
            StormMotion::Mean06Npw => {
                sharprs::winds::mean_wind_npw(inner, sfc_p, p6, -1.0, 0.0, 0.0).ok()
            }
            StormMotion::CorfidiUp => sharprs::winds::corfidi_mcs_motion(inner)
                .ok()
                .map(|c| (c.0, c.1)),
            StormMotion::CorfidiDown => sharprs::winds::corfidi_mcs_motion(inner)
                .ok()
                .map(|c| (c.2, c.3)),
            StormMotion::NonParcelBunkers => sharprs::winds::non_parcel_bunkers_motion(inner)
                .ok()
                .map(|c| (c.0, c.1)),
            StormMotion::User { dir_deg, spd_kt } => {
                let r = dir_deg.to_radians();
                Some((-spd_kt * r.sin(), -spd_kt * r.cos()))
            }
            StormMotion::Brm => None,
        };
        let (u, v) = uv.ok_or("storm motion could not be computed for this profile")?;
        prof.srwind.0 = u;
        prof.srwind.1 = v;
        if prof.ebotm.is_finite() && prof.etopm.is_finite() {
            prof.right_esrh =
                sharppyrs::extras::helicity(&prof.inner, prof.ebotm, prof.etopm, u, v).0;
        }
        sm_label = Some(opts.storm_motion.label());
        notes.push(format!(
            "storm motion {} drives SRH/SR winds and is drawn in the Bunkers-right slot",
            opts.storm_motion.label()
        ));
    }
    let mut derived = sharppyrs::DerivedParams::compute(&prof);
    // Never ECAPE.
    derived.ecape = f64::NAN;
    derived.ncape = f64::NAN;
    Ok(Analysis {
        column,
        prof,
        derived,
        notes,
        sm_label,
        user_parcel,
    })
}

fn f(v: f64) -> Value {
    if v.is_finite() {
        json!((v * 1000.0).round() / 1000.0)
    } else {
        Value::Null
    }
}

fn parcel_json(p: &ParcelResult, _prof: &sharppyrs::Profile) -> Value {
    // sharprs parcel heights (lclhght, lfchght, elhght, mplhght) are already AGL (params/cape.rs)
    let agl = |h: f64| h;
    json!({
        "lpl_hpa": f(p.pres), "lpl_t_c": f(p.tmpc), "lpl_td_c": f(p.dwpc),
        "cape_jkg": f(p.bplus), "cin_jkg": f(p.bminus),
        "lcl_hpa": f(p.lclpres), "lcl_m_agl": f(agl(p.lclhght)),
        "lfc_hpa": f(p.lfcpres), "lfc_m_agl": f(agl(p.lfchght)),
        "el_hpa": f(p.elpres), "el_m_agl": f(agl(p.elhght)),
        "mpl_hpa": f(p.mplpres), "mpl_m_agl": f(agl(p.mplhght)),
        "li500_c": f(p.li5), "li300_c": f(p.li3), "limax_c": f(p.limax),
        "cape_0_3km_jkg": f(p.b3km), "cape_0_6km_jkg": f(p.b6km), "cape_frz_jkg": f(p.bfzl),
        "cap_c": f(p.cap), "cap_hpa": f(p.cappres),
    })
}

fn comp(c: (f64, f64)) -> Value {
    json!({"u_kt": f(c.0), "v_kt": f(c.1), "dir_deg": f((-c.0).atan2(-c.1).to_degrees().rem_euclid(360.0)), "spd_kt": f(c.0.hypot(c.1))})
}
fn vect(c: (f64, f64)) -> Value {
    json!({"dir_deg": f(c.0), "spd_kt": f(c.1)})
}

/// Full JSON of an analysis (no ECAPE / NCAPE anywhere).
pub fn analysis_json(a: &Analysis) -> Value {
    let p = &a.prof;
    let d = &a.derived;
    let inner = &p.inner;
    let levels: Vec<Value> = (0..inner.pres.len())
        .map(|i| {
            json!([
                f(inner.pres[i]),
                f(inner.hght[i]),
                f(inner.to_agl(inner.hght[i])),
                f(inner.tmpc[i]),
                f(inner.dwpc[i]),
                f(inner.wetbulb.get(i).copied().unwrap_or(f64::NAN)),
                f(inner.vtmp.get(i).copied().unwrap_or(f64::NAN)),
                f(inner.thetae.get(i).copied().unwrap_or(f64::NAN)),
                f(inner.relh.get(i).copied().unwrap_or(f64::NAN)),
                f(inner.wdir[i]),
                f(inner.wspd[i]),
                f(inner.omeg.get(i).copied().unwrap_or(f64::NAN))
            ])
        })
        .collect();
    let hodo: Vec<Value> = (0..inner.pres.len())
        .filter(|&i| inner.u[i].is_finite() && inner.v[i].is_finite())
        .map(|i| json!([f(inner.to_agl(inner.hght[i])), f(inner.u[i]), f(inner.v[i])]))
        .collect();
    json!({
        "profile": {
            "columns": ["p_hpa","z_m_msl","z_m_agl","t_c","td_c","tw_c","tv_c","thetae_k","rh_pct","wdir_deg","wspd_kt","omega_pa_s"],
            "levels": levels,
            "surface_hpa": f(inner.sfc_pressure()),
            "elevation_m": f(inner.hght[inner.sfc]),
            "lat": f(p.latitude()),
            "lon": f(a.column.metadata.longitude_deg.unwrap_or(f64::NAN)),
        },
        "parcels": {
            "sb": parcel_json(&p.sfcpcl, p),
            (if a.user_parcel { "user" } else { "fcst" }): parcel_json(&p.fcstpcl, p),
            "mu": parcel_json(&p.mupcl, p),
            "ml": parcel_json(&p.mlpcl, p),
        },
        "effective_inflow": {"bottom_hpa": f(p.ebottom), "top_hpa": f(p.etop), "bottom_m_agl": f(p.ebotm), "top_m_agl": f(p.etopm), "esrh_m2s2": f(p.right_esrh)},
        "storm_motion": {
            "label": a.sm_label.clone().unwrap_or_else(|| "Bunkers Right".into()),
            "right": comp((p.srwind.0, p.srwind.1)),
            "left": comp((p.srwind.2, p.srwind.3)),
            "corfidi_up": comp(d.corfidi_up), "corfidi_down": comp(d.corfidi_dn),
        },
        "hodograph": {"columns": ["z_m_agl","u_kt","v_kt"], "points": hodo},
        "dcape_jkg": f(p.dcape),
        "max_lapse_2_6km": {"c_per_km": f(p.max_lapse_rate_2_6.0), "pbot_hpa": f(p.max_lapse_rate_2_6.1), "ptop_hpa": f(p.max_lapse_rate_2_6.2)},
        "thermo": {
            "pwat_in": f(d.pwat), "k_index": f(d.k_idx), "tei": f(d.tei), "esp": f(d.esp), "mmp": f(d.mmp), "wndg": f(d.wndg),
            "dcp": f(d.dcp), "mburst": f(d.mburst), "sweat": f(d.sweat), "dcape_jkg": f(d.dcape), "downrush_t_f": f(d.drush_f),
            "mean_mixr_gkg": f(d.mean_mixr), "low_rh_pct": f(d.low_rh), "mid_rh_pct": f(d.mid_rh), "total_totals": f(d.totals_totals),
            "convective_t_f": f(d.conv_t_f), "max_t_f": f(d.max_t_f), "thetae_diff_k": f(d.thetae_diff),
            "lapse_0_3km": f(d.lapserate_3km), "lapse_3_6km": f(d.lapserate_3_6km), "lapse_850_500": f(d.lapserate_850_500),
            "lapse_700_500": f(d.lapserate_700_500), "lapse_sfc_500m": f(d.lapserate_sfc_500m), "lapse_sfc_1km": f(d.lapserate_sfc_1km),
            "wbz_m_agl": f(d.wbz_height), "hgz_cape_jkg": f(d.hgz_cape), "cape_0_3km_jkg": f(d.cape_0_3km), "cape_0_6km_jkg": f(d.cape_0_6km),
        },
        "kinematics": {
            "srh_0_500m": f(d.srh500), "srh_0_1km": f(d.srh1km), "srh_0_3km": f(d.srh3km), "esrh": f(d.right_esrh),
            "shear_0_500m": comp(d.sfc_500m_shear), "shear_0_1km": comp(d.sfc_1km_shear), "shear_0_3km": comp(d.sfc_3km_shear),
            "shear_0_6km": comp(d.sfc_6km_shear), "shear_0_8km": comp(d.sfc_8km_shear), "shear_effective": comp(d.eff_shear),
            "ebwd": comp(d.ebwd), "shear_lcl_el": comp(d.lcl_el_shear),
            "mean_wind_0_500m": comp(d.mean_wind_sfc_500m), "mean_wind_0_1km": vect(d.mean_1km), "mean_wind_0_3km": vect(d.mean_3km),
            "mean_wind_0_6km": vect(d.mean_6km), "mean_wind_0_8km": vect(d.mean_8km), "mean_wind_effective": comp(d.mean_eff),
            "mean_wind_ebw": comp(d.mean_ebw), "mean_wind_lcl_el": vect(d.mean_lcl_el),
            "srw_0_500m": comp(d.srw_sfc_500m), "srw_0_1km": vect(d.srw_1km), "srw_0_3km": vect(d.srw_3km), "srw_0_6km": vect(d.srw_6km),
            "srw_0_8km": vect(d.srw_8km), "srw_4_6km": vect(d.srw_4_5km), "srw_effective": comp(d.srw_eff), "srw_ebw": comp(d.srw_ebw),
            "srw_lcl_el": vect(d.srw_lcl_el), "wind_1km": vect(d.wind1km), "wind_6km": vect(d.wind6km),
            "critical_angle_deg": f(d.right_critical_angle), "brn_shear": f(d.brnshear),
        },
        "severe": {
            "stp_cin": f(d.stp_cin), "stp_fixed": f(d.stp_fixed), "scp_right": f(d.right_scp), "scp_left": f(d.left_scp),
            "ship": f(d.ship), "sig_severe": f(d.sig_severe), "ehi_0_1km": f(d.ehi_0_1km), "ehi_0_3km": f(d.ehi_0_3km),
            "vgp": f(d.vgp), "peskov": f(d.peskov), "mcs_index": f(d.mcs_index), "lrghail": f(d.lrghail), "lscp": f(d.lscp),
            "nstp": f(d.nstp), "moshe": f(d.modified_sherbe),
        },
        "notes": a.notes,
    })
}

/// Compact per-member summary row (for spaghetti / plumes tables).
pub fn summary_json(a: &Analysis) -> Value {
    let p = &a.prof;
    let d = &a.derived;
    json!({
        "sbcape": f(p.sfcpcl.bplus), "sbcin": f(p.sfcpcl.bminus), "mlcape": f(p.mlpcl.bplus), "mlcin": f(p.mlpcl.bminus),
        "mucape": f(p.mupcl.bplus), "mllcl_m": f(p.mlpcl.lclhght),
        "srh01": f(d.srh1km), "srh03": f(d.srh3km), "esrh": f(d.right_esrh), "shear06_kt": f(d.sfc_6km_shear.0.hypot(d.sfc_6km_shear.1)),
        "ebwd_kt": f(d.ebwd.0.hypot(d.ebwd.1)), "stp_cin": f(d.stp_cin), "stp_fixed": f(d.stp_fixed), "scp": f(d.right_scp),
        "ship": f(d.ship), "pwat_in": f(d.pwat), "dcape": f(d.dcape), "lapse_700_500": f(d.lapserate_700_500),
    })
}
