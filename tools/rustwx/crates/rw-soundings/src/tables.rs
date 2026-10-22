//! The sharppyrs native board cells as JSON rows, keyed by the stable
//! `native_diagnostic_slot_ids`, so the site can show the tables natively.
//! Values come from the same Profile / DerivedParams the board draws.
//! `severe.ecape` / `severe.ncape` never appear.

use serde_json::{Value, json};
use sharppyrs::DiagnosticTablePanelKind as K;

use crate::analysis::Analysis;

pub const BLOCKED: [&str; 2] = ["severe.ecape", "severe.ncape"];

fn r(v: f64, dp: i32) -> Value {
    if v.is_finite() {
        let m = 10f64.powi(dp);
        json!((v * m).round() / m)
    } else {
        Value::Null
    }
}

fn fmt(v: f64, dp: usize) -> String {
    if v.is_finite() {
        format!("{v:.dp$}")
    } else {
        "--".into()
    }
}

fn dirspd(u: f64, v: f64) -> (f64, f64) {
    ((-u).atan2(-v).to_degrees().rem_euclid(360.0), u.hypot(v))
}

/// Human label for a cell id (also used by /api/options).
pub fn label(id: &str) -> String {
    let parts: Vec<&str> = id.split('.').collect();
    let up = |s: &str| s.to_uppercase();
    match parts.as_slice() {
        ["parcel", p, f] => {
            let pn = match *p {
                "sfc" => "SB",
                "ml" => "ML",
                "fcst" => "FCST",
                "mu" => "MU",
                x => x,
            };
            let fname = match *f {
                "cinh" => "CIN".to_string(),
                x => up(x),
            };
            format!("{pn} {fname}")
        }
        ["kin", layer, what] => {
            let l = match *layer {
                "sfc_500m" => "0-500 m",
                "sfc_1km" => "0-1 km",
                "sfc_3km" => "0-3 km",
                "sfc_6km" => "0-6 km",
                "sfc_8km" => "0-8 km",
                "effective" => "Eff",
                "lcl_el" => "LCL-EL",
                "ebwd" => "EBWD",
                x => x,
            };
            let w = match *what {
                "srh" => "SRH",
                "shear" => "shear",
                "mean_wind" => "mean wind",
                "srw" => "SR wind",
                x => x,
            };
            format!("{l} {w}")
        }
        [_, x] => match *x {
            "pwat" => "PWAT".into(),
            "mean_mixr" => "Mean w".into(),
            "low_rh" => "Low RH".into(),
            "mid_rh" => "Mid RH".into(),
            "dcape" => "DCAPE".into(),
            "downrush_temp" => "Downrush T".into(),
            "k_index" => "K index".into(),
            "total_totals" => "Total totals".into(),
            "convective_temp" => "Conv T".into(),
            "max_temp" => "Max T".into(),
            "cape_0_3km" => "0-3 km CAPE".into(),
            "cape_0_6km" => "0-6 km CAPE".into(),
            "sig_severe" => "SigSvr".into(),
            "sfc_500m" => "0-500 m lapse".into(),
            "sfc_1km" => "0-1 km lapse".into(),
            "sfc_3km" => "0-3 km lapse".into(),
            "850_500" => "850-500 lapse".into(),
            "700_500" => "700-500 lapse".into(),
            "scp_right" => "Supercell comp".into(),
            "stp_effective" => "STP (cin)".into(),
            "stp_fixed" => "STP (fixed)".into(),
            "dcp" => "Derecho comp".into(),
            "brn_shear" => "BRN shear".into(),
            "srw_4_6km" => "4-6 km SR wind".into(),
            "bunkers_right" => "Bunkers right".into(),
            "bunkers_left" => "Bunkers left".into(),
            "corfidi_down" => "Corfidi down".into(),
            "corfidi_up" => "Corfidi up".into(),
            "wind_1km" => "1 km wind".into(),
            "wind_6km" => "6 km wind".into(),
            "ehi_0_1km" => "EHI 0-1 km".into(),
            "ehi_0_3km" => "EHI 0-3 km".into(),
            "mcs" => "MCS index".into(),
            "peskov" => "Peskov".into(),
            "moshe" => "MOSHE".into(),
            "lrghail" => "LRGHAIL".into(),
            "hgz_cape" => "HGZ CAPE".into(),
            "nstp" => "NSTP".into(),
            "lscp" => "LSCP".into(),
            "wbz_height" => "WBZ height".into(),
            x => up(x),
        },
        _ => id.to_string(),
    }
}

fn panel_name(k: K) -> &'static str {
    match k {
        K::Convective => "convective",
        K::Kinematics => "kinematics",
        K::Severe => "severe",
    }
}

/// (value string, units, numeric) for one cell.
fn cell(a: &Analysis, id: &str) -> (String, &'static str, Option<f64>) {
    let p = &a.prof;
    let d = &a.derived;
    let agl = |h: f64| h; // sharprs parcel heights are already AGL
    let num = |v: f64, dp: usize, u: &'static str| (fmt(v, dp), u, v.is_finite().then_some(v));
    let vec_ds = |dir: f64, spd: f64| {
        (
            format!("{}/{}", fmt(dir, 0), fmt(spd, 0)),
            "deg/kt",
            spd.is_finite().then_some(spd),
        )
    };
    let comp = |c: (f64, f64)| {
        let (dd, ss) = dirspd(c.0, c.1);
        vec_ds(dd, ss)
    };
    let mag = |c: (f64, f64)| {
        let m = c.0.hypot(c.1);
        num(m, 0, "kt")
    };
    let parts: Vec<&str> = id.split('.').collect();
    match parts.as_slice() {
        ["parcel", which, f] => {
            let pc = match *which {
                "sfc" => &p.sfcpcl,
                "ml" => &p.mlpcl,
                "fcst" => &p.fcstpcl,
                _ => &p.mupcl,
            };
            match *f {
                "cape" => num(pc.bplus, 0, "J/kg"),
                "cinh" => num(pc.bminus, 0, "J/kg"),
                "lcl" => num(agl(pc.lclhght), 0, "m AGL"),
                "li" => num(pc.li5, 0, "C"),
                "lfc" => num(agl(pc.lfchght), 0, "m AGL"),
                "el" => num(agl(pc.elhght), 0, "m AGL"),
                "mpl" => num(agl(pc.mplhght), 0, "m AGL"),
                _ => ("--".into(), "", None),
            }
        }
        ["kin", layer, what]
            if matches!(
                *layer,
                "sfc_500m"
                    | "sfc_1km"
                    | "sfc_3km"
                    | "effective"
                    | "sfc_6km"
                    | "sfc_8km"
                    | "lcl_el"
                    | "ebwd"
            ) =>
        {
            match (*layer, *what) {
                ("sfc_500m", "srh") => num(d.srh500, 0, "m2/s2"),
                ("sfc_1km", "srh") => num(d.srh1km, 0, "m2/s2"),
                ("sfc_3km", "srh") => num(d.srh3km, 0, "m2/s2"),
                ("effective", "srh") => num(d.right_esrh, 0, "m2/s2"),
                ("sfc_500m", "shear") => mag(d.sfc_500m_shear),
                ("sfc_1km", "shear") => mag(d.sfc_1km_shear),
                ("sfc_3km", "shear") => mag(d.sfc_3km_shear),
                ("effective", "shear") => mag(d.eff_shear),
                ("sfc_6km", "shear") => mag(d.sfc_6km_shear),
                ("sfc_8km", "shear") => mag(d.sfc_8km_shear),
                ("lcl_el", "shear") => mag(d.lcl_el_shear),
                ("ebwd", "shear") => mag(d.ebwd),
                ("sfc_500m", "mean_wind") => comp(d.mean_wind_sfc_500m),
                ("sfc_1km", "mean_wind") => vec_ds(d.mean_1km.0, d.mean_1km.1),
                ("sfc_3km", "mean_wind") => vec_ds(d.mean_3km.0, d.mean_3km.1),
                ("effective", "mean_wind") => comp(d.mean_eff),
                ("sfc_6km", "mean_wind") => vec_ds(d.mean_6km.0, d.mean_6km.1),
                ("sfc_8km", "mean_wind") => vec_ds(d.mean_8km.0, d.mean_8km.1),
                ("lcl_el", "mean_wind") => vec_ds(d.mean_lcl_el.0, d.mean_lcl_el.1),
                ("ebwd", "mean_wind") => comp(d.mean_ebw),
                ("sfc_500m", "srw") => comp(d.srw_sfc_500m),
                ("sfc_1km", "srw") => vec_ds(d.srw_1km.0, d.srw_1km.1),
                ("sfc_3km", "srw") => vec_ds(d.srw_3km.0, d.srw_3km.1),
                ("effective", "srw") => comp(d.srw_eff),
                ("sfc_6km", "srw") => vec_ds(d.srw_6km.0, d.srw_6km.1),
                ("sfc_8km", "srw") => vec_ds(d.srw_8km.0, d.srw_8km.1),
                ("lcl_el", "srw") => vec_ds(d.srw_lcl_el.0, d.srw_lcl_el.1),
                ("ebwd", "srw") => comp(d.srw_ebw),
                _ => ("--".into(), "", None),
            }
        }
        [group, x] => match (*group, *x) {
            ("thermo", "pwat") => num(d.pwat, 2, "in"),
            ("thermo", "mean_mixr") => num(d.mean_mixr, 2, "g/kg"),
            ("thermo", "low_rh") => num(d.low_rh, 0, "%"),
            ("thermo", "mid_rh") => num(d.mid_rh, 0, "%"),
            ("thermo", "dcape") => num(d.dcape, 0, "J/kg"),
            ("thermo", "downrush_temp") => num(d.drush_f, 0, "F"),
            ("thermo", "k_index") => num(d.k_idx, 0, ""),
            ("thermo", "total_totals") => num(d.totals_totals, 0, ""),
            ("thermo", "convective_temp") => num(d.conv_t_f, 0, "F"),
            ("thermo", "max_temp") => num(d.max_t_f, 0, "F"),
            ("thermo", "esp") => num(d.esp, 1, ""),
            ("thermo", "mmp") => num(d.mmp, 2, ""),
            ("thermo", "wndg") => num(d.wndg, 1, ""),
            ("thermo", "tei") => num(d.tei, 0, ""),
            ("thermo", "cape_0_3km") => num(d.cape_0_3km, 0, "J/kg"),
            ("thermo", "cape_0_6km") => num(d.cape_0_6km, 0, "J/kg"),
            ("thermo", "mburst") => num(d.mburst, 0, ""),
            ("thermo", "sig_severe") => num(d.sig_severe, 0, "m3/s3"),
            ("lapse", "sfc_500m") => num(d.lapserate_sfc_500m, 1, "C/km"),
            ("lapse", "sfc_1km") => num(d.lapserate_sfc_1km, 1, "C/km"),
            ("lapse", "sfc_3km") => num(d.lapserate_3km, 1, "C/km"),
            ("lapse", "850_500") => num(d.lapserate_850_500, 1, "C/km"),
            ("lapse", "700_500") => num(d.lapserate_700_500, 1, "C/km"),
            ("composite", "scp_right") => num(d.right_scp, 1, ""),
            ("composite", "stp_effective") => num(d.stp_cin, 1, ""),
            ("composite", "stp_fixed") => num(d.stp_fixed, 1, ""),
            ("composite", "ship") => num(d.ship, 1, ""),
            ("composite", "dcp") => num(d.dcp, 1, ""),
            ("kin", "brn_shear") => num(d.brnshear, 0, "m2/s2"),
            ("kin", "srw_4_6km") => vec_ds(d.srw_4_5km.0, d.srw_4_5km.1),
            ("kin", "bunkers_right") => comp((p.srwind.0, p.srwind.1)),
            ("kin", "bunkers_left") => comp((p.srwind.2, p.srwind.3)),
            ("kin", "corfidi_down") => comp(d.corfidi_dn),
            ("kin", "corfidi_up") => comp(d.corfidi_up),
            ("kin", "wind_1km") => vec_ds(d.wind1km.0, d.wind1km.1),
            ("kin", "wind_6km") => vec_ds(d.wind6km.0, d.wind6km.1),
            ("severe", "ehi_0_1km") => num(d.ehi_0_1km, 1, ""),
            ("severe", "ehi_0_3km") => num(d.ehi_0_3km, 1, ""),
            ("severe", "vgp") => num(d.vgp, 2, ""),
            ("severe", "peskov") => num(d.peskov, 1, ""),
            ("severe", "mcs") => num(d.mcs_index, 1, ""),
            ("severe", "sweat") => num(d.sweat, 0, ""),
            ("severe", "moshe") => num(d.modified_sherbe, 1, ""),
            ("severe", "lrghail") => num(d.lrghail, 1, ""),
            ("severe", "hgz_cape") => num(d.hgz_cape, 0, "J/kg"),
            ("severe", "nstp") => num(d.nstp, 1, ""),
            ("severe", "lscp") => num(d.lscp, 1, ""),
            ("severe", "wbz_height") => num(d.wbz_height, 0, "m AGL"),
            _ => ("--".into(), "", None),
        },
        _ => ("--".into(), "", None),
    }
}

/// Every board cell after blanking (ECAPE cells never present; `blank`
/// cells omitted). The storm-motion relabel follows the rendered board.
pub fn tables_json(a: &Analysis, blank: &[String]) -> Value {
    let mut out = vec![];
    for (k, title) in [
        (K::Convective, "Parcels & thermo"),
        (K::Kinematics, "Kinematics"),
        (K::Severe, "Severe indices"),
    ] {
        let rows: Vec<Value> = sharppyrs::native_diagnostic_slot_ids(k)
            .iter()
            .filter(|id| !BLOCKED.contains(id) && !blank.iter().any(|b| b == *id))
            .map(|id| {
                let (value, units, n) = cell(a, id);
                let lab = if *id == "kin.bunkers_right" { a.sm_label.clone().unwrap_or_else(|| label(id)) } else { label(id) };
                json!({"id": id, "label": lab, "value": value, "units": units, "num": n.map(|x| r(x, 3)).unwrap_or(Value::Null)})
            })
            .collect();
        out.push(json!({"panel": panel_name(k), "title": title, "rows": rows}));
    }
    Value::Array(out)
}

pub fn slot_panel(id: &str) -> Option<K> {
    [K::Convective, K::Kinematics, K::Severe]
        .into_iter()
        .find(|k| sharppyrs::native_diagnostic_slot_ids(*k).contains(&id))
}
