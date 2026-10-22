//! Shared helpers for the public batch sounding plots.
use crate::analysis::Analysis;
use crate::options::DisplayOpts;
use crate::render::{self, Bands};
use crate::store::{self, RawColumn};

pub(crate) fn patches(d: &DisplayOpts, a: &Analysis) -> sharppyrs::NativeDiagnosticPatchBoard {
    let mut p = render::base_patches();
    for (k, id) in &d.blank {
        p.push(sharppyrs::NativeDiagnosticPatch::blank(*k, id.clone()));
    }
    if let Some(label) = &a.sm_label {
        let (u, v) = (a.prof.srwind.0, a.prof.srwind.1);
        let dir = (-u).atan2(-v).to_degrees().rem_euclid(360.0);
        p.push(sharppyrs::NativeDiagnosticPatch::replace(
            sharppyrs::DiagnosticTablePanelKind::Kinematics,
            "kin.bunkers_right",
            sharppyrs::DiagnosticTableRow::new(
                label.clone(),
                format!("{:.0}/{:.0} kt", dir, u.hypot(v)),
            ),
        ));
    }
    sharppyrs::NativeDiagnosticPatchBoard { patches: p }
}

pub(crate) fn bands_from(cols: &[RawColumn], members: &[&Analysis]) -> Bands {
    let nl = cols.iter().map(|c| c.levels()).min().unwrap_or(0);
    let qs = [10.0, 25.0, 50.0, 75.0, 90.0];
    let mut b = Bands::default();
    for k in 0..nl {
        let pv: Vec<f64> = cols
            .iter()
            .map(|c| c.p[k])
            .filter(|x| x.is_finite())
            .collect();
        b.p.push(if pv.is_empty() {
            f64::NAN
        } else {
            pv.iter().sum::<f64>() / pv.len() as f64
        });
        for (qi, q) in qs.iter().enumerate() {
            b.t[qi].push(store::percentile(
                &mut cols.iter().map(|c| c.t[k]).collect(),
                *q,
            ));
            b.td[qi].push(store::percentile(
                &mut cols.iter().map(|c| c.td[k]).collect(),
                *q,
            ));
        }
    }
    // median hodograph on a common 0-10 km AGL ladder
    for i in 0..=40 {
        let z = i as f64 * 250.0;
        let mut us = vec![];
        let mut vs = vec![];
        for a in members {
            let inner = &a.prof.inner;
            let p = inner.pres_at_height(inner.to_msl(z));
            if p.is_finite() {
                let (u, v) = inner.interp_wind(p);
                us.push(u);
                vs.push(v);
            }
        }
        b.hodo_med.push((
            store::percentile(&mut us, 50.0),
            store::percentile(&mut vs, 50.0),
        ));
    }
    b
}

pub(crate) fn hodo_trace(a: &Analysis) -> Vec<(f64, f64)> {
    let inner = &a.prof.inner;
    (0..inner.pres.len())
        .filter(|&i| inner.to_agl(inner.hght[i]) <= 10_000.0)
        .map(|i| (inner.u[i], inner.v[i]))
        .collect()
}
