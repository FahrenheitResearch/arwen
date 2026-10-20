//! Contract conformance: ID scheme, topology, attribute completeness, canonical
//! order. A builder's output must pass with zero errors. Pre-contract NOAA
//! files are checked too; their gaps are reported, not hidden.
use crate::contract::{Fabric, METADATA_KEYS};
use serde::Serialize;
use std::collections::{HashMap, HashSet};

#[derive(Debug, Default, Serialize)]
pub struct Report {
    pub errors: Vec<String>,
    /// Properties a builder must establish but a pre-contract source lacks.
    pub contract_gaps: Vec<String>,
    pub counts: HashMap<&'static str, usize>,
    pub canonical_order: bool,
}
impl Report {
    pub fn conforming(&self) -> bool {
        self.errors.is_empty() && self.contract_gaps.is_empty()
    }
}

/// `<prefix>-<n>`, n a positive decimal integer below 2^53 without leading zeros.
pub fn parse_id(s: &str) -> Option<(&str, i64)> {
    let (p, n) = s.rsplit_once('-')?;
    if n.is_empty() || n.starts_with('0') || !n.bytes().all(|b| b.is_ascii_digit()) || n.len() > 16 {
        return None;
    }
    let v: i64 = n.parse().ok()?;
    (v < (1i64 << 53)).then_some((p, v))
}

/// Canonical sort key: prefix rank, then numeric suffix.
pub fn canonical_key(s: &str) -> (u8, i64) {
    match parse_id(s) {
        Some((p, n)) => (
            match p {
                "wb" => 0,
                "cat" => 1,
                "nex" => 2,
                "tnx" => 3,
                "cnx" => 4,
                _ => 9,
            },
            n,
        ),
        None => (255, i64::MAX),
    }
}

fn ordered<'a>(fids: impl Iterator<Item = (i64, &'a str)>) -> bool {
    let v = fids.collect::<Vec<_>>();
    v.iter().enumerate().all(|(i, (fid, _))| *fid == i as i64 + 1) && v.windows(2).all(|w| canonical_key(w[0].1) < canonical_key(w[1].1))
}

pub fn check(f: &Fabric) -> Report {
    let mut r = Report::default();
    let mut errors = Vec::new();
    let mut err = |m: String| errors.push(m);
    let fp_ids = f.flowpaths.iter().map(|x| x.id.clone().unwrap_or_default()).collect::<Vec<_>>();
    let fp_set = fp_ids.iter().cloned().collect::<HashSet<_>>();
    if fp_set.len() != fp_ids.len() {
        err("duplicate flowpath ids".into());
    }
    let nex = f.nexus.iter().map(|x| (x.id.clone().unwrap_or_default(), x.toid.clone().unwrap_or_default())).collect::<HashMap<_, _>>();
    if nex.len() != f.nexus.len() {
        err("duplicate nexus ids".into());
    }
    let bad = |ids: &mut dyn Iterator<Item = &str>, prefixes: &[&str]| ids.filter(|s| !parse_id(s).is_some_and(|(p, _)| prefixes.contains(&p))).count();
    let n = bad(&mut fp_ids.iter().map(String::as_str), &["wb"]);
    if n > 0 {
        err(format!("{n} flowpath ids are not wb-<n>"));
    }
    let n = bad(&mut f.divides.iter().map(|d| d.divide_id.as_deref().unwrap_or("")), &["cat"]);
    if n > 0 {
        err(format!("{n} divide ids are not cat-<n>"));
    }
    let n = bad(&mut nex.keys().map(String::as_str), &["nex", "tnx", "cnx"]);
    if n > 0 {
        err(format!("{n} nexus ids are not nex/tnx/cnx-<n>"));
    }
    // Pairing: cat-n <-> wb-n.
    let unpaired = f.flowpaths.iter().filter(|x| match (x.id.as_deref().and_then(parse_id), x.divide_id.as_deref().and_then(parse_id)) {
        (Some((_, a)), Some((_, b))) => a != b,
        (_, None) => x.has_divide == Some(true),
        _ => true,
    }).count();
    if unpaired > 0 {
        err(format!("{unpaired} flowpaths whose divide_id is not cat-<same n>"));
    }
    let div_unpaired = f.divides.iter().filter(|d| match (d.divide_id.as_deref().and_then(parse_id), d.id.as_deref().and_then(parse_id)) {
        (Some((_, a)), Some((_, b))) => a != b,
        (_, None) => d.has_flowline == Some(true),
        _ => true,
    }).count();
    if div_unpaired > 0 {
        err(format!("{div_unpaired} divides whose id is not wb-<same n>"));
    }
    // Topology: flowpath -> nexus -> flowpath, acyclic.
    let mut down = vec![None; f.flowpaths.len()];
    let index = fp_ids.iter().enumerate().map(|(i, s)| (s.as_str(), i)).collect::<HashMap<_, _>>();
    let mut dangling = 0;
    let mut nexus_to_missing = 0;
    for (i, x) in f.flowpaths.iter().enumerate() {
        let to = x.toid.as_deref().unwrap_or("");
        match nex.get(to) {
            Some(t) if t.is_empty() => {}
            Some(t) => match index.get(t.as_str()) {
                Some(&j) => down[i] = Some(j),
                None => nexus_to_missing += 1,
            },
            None => dangling += 1,
        }
    }
    if dangling > 0 {
        err(format!("{dangling} flowpaths whose toid is not a nexus in the fabric"));
    }
    // A nexus whose toid is not in the fabric is an exterior outlet (engine
    // downstream index -1): a terminal tnx- with the v2.0.1 sentinel "wb-0",
    // or a subset boundary. Counted, not an error.
    let exterior_outlets = nexus_to_missing;
    let mut indeg = vec![0usize; down.len()];
    for d in down.iter().flatten() {
        indeg[*d] += 1;
    }
    let mut stack = (0..down.len()).filter(|&i| indeg[i] == 0).collect::<Vec<_>>();
    let mut seen = 0;
    while let Some(i) = stack.pop() {
        seen += 1;
        if let Some(j) = down[i] {
            indeg[j] -= 1;
            if indeg[j] == 0 {
                stack.push(j);
            }
        }
    }
    if seen != down.len() {
        err(format!("{} flowpaths are on a cycle", down.len() - seen));
    }
    let hs_bad = f.flowpaths.iter().enumerate().filter(|(i, x)| match (down[*i], x.hydroseq) {
        (Some(j), Some(h)) => f.flowpaths[j].hydroseq.is_none_or(|hd| hd <= h),
        _ => false,
    }).count();
    // Attributes: exactly one finite MC row per flowpath.
    let attr = f.flowpath_attributes.iter().map(|a| a.id.clone().unwrap_or_default()).collect::<HashSet<_>>();
    if attr != fp_set || f.flowpath_attributes.len() != f.flowpaths.len() {
        err("flowpath_attributes is not exactly one row per flowpath".into());
    }
    let nonfinite = f.flowpath_attributes.iter().filter(|a| {
        [a.length_m, a.btm_wdth, a.top_wdth, a.top_wdth_cc, a.n, a.n_cc, a.ch_slp, a.so].iter().any(|v| !v.is_some_and(f64::is_finite))
    }).count();
    if nonfinite > 0 {
        err(format!("{nonfinite} flowpath_attributes rows with a missing MC column"));
    }
    let lake_bad = f.lakes.iter().filter(|l| {
        [l.lk_area, l.lk_mx_e, l.weir_e, l.weir_c, l.weir_l, l.dam_length, l.orifice_e, l.orifice_c, l.orifice_a, l.ifd, l.hl_link].iter().any(|v| !v.is_some_and(f64::is_finite))
    }).count();
    // case (b) carries one such lake (nex-2420985). Builders write complete
    // rows or no row.
    let lake_gap = (lake_bad > 0).then(|| format!("{lake_bad} lakes rows with a missing level-pool column"));
    if !(f.srs.definition.starts_with("PROJCS") || f.srs.definition.starts_with("PROJCRS")) {
        err(format!("fabric CRS {} is not projected", f.srs.srs_id));
    }
    r.errors = errors;
    r.counts.insert("exterior_outlet_flowpaths", exterior_outlets);
    // Builder-only properties. hydroseq: builders write a topological rank,
    // strictly larger downstream (the majority v2.0.1 convention; NOAA files
    // break it on some reaches, e.g. 73 of 6970 edges in case (b)).
    if hs_bad > 0 {
        r.contract_gaps.push(format!("{hs_bad} flowpaths whose downstream hydroseq is not larger"));
    }
    r.contract_gaps.extend(lake_gap);
    match &f.model_attributes {
        None => r.contract_gaps.push("model_attributes table absent".into()),
        Some(m) => {
            let ids = m.iter().map(|x| x.divide_id.clone().unwrap_or_default()).collect::<Vec<_>>();
            let dv = f.divides.iter().map(|d| d.divide_id.clone().unwrap_or_default()).collect::<Vec<_>>();
            if ids != dv {
                r.errors.push("model_attributes rows are not the divides roster in divides order".into());
            }
            let incomplete = m.iter().filter(|x| serde_json::to_value(x).map(|v| v.as_object().unwrap().values().any(|v| v.is_null())).unwrap_or(true)).count();
            if incomplete > 0 {
                r.errors.push(format!("{incomplete} model_attributes rows with NULL"));
            }
        }
    }
    match &f.fabric_metadata {
        None => r.contract_gaps.push("fabric_metadata table absent".into()),
        Some(m) => {
            let keys = m.iter().filter_map(|x| x.key.as_deref()).collect::<HashSet<_>>();
            for k in METADATA_KEYS {
                if !keys.contains(k) {
                    r.errors.push(format!("fabric_metadata lacks {k}"));
                }
            }
        }
    }
    r.canonical_order = ordered(f.flowpaths.iter().map(|x| (x.fid, x.id.as_deref().unwrap_or(""))))
        && ordered(f.divides.iter().map(|x| (x.fid, x.divide_id.as_deref().unwrap_or(""))))
        && ordered(f.nexus.iter().map(|x| (x.fid, x.id.as_deref().unwrap_or(""))))
        && ordered(f.flowpath_attributes.iter().map(|x| (x.fid, x.id.as_deref().unwrap_or(""))));
    if !r.canonical_order {
        r.contract_gaps.push("rows are not in canonical order (fid = rank of <prefix rank, n>)".into());
    }
    r.counts.insert("divides", f.divides.len());
    r.counts.insert("flowpaths", f.flowpaths.len());
    r.counts.insert("nexus", f.nexus.len());
    r.counts.insert("network", f.network.len());
    r.counts.insert("hydrolocations", f.hydrolocations.len());
    r.counts.insert("lakes", f.lakes.len());
    r
}
