//! Topology comparison of a built fabric against a reference fabric (the
//! CONUS NHDPlus-derived NextGen fabric) at the reference's gauges:
//! drainage area at the gauge, reaches upstream, Shreve magnitude (number of
//! sources) of the network thinned to drainage area >= A for several A
//! (invariant to how a builder splits reaches), main-stem length, outlets
//! and cycles. Reference coordinates in EPSG:5070 are inverted on GRS80
//! (Snyder 1987, Albers equal-area conic, eqs. 14-1 to 14-21, 3-16).
use crate::contract::Fabric;
use crate::geo;
use crate::Result;
use serde_json::{json, Value};
use std::collections::{HashMap, HashSet};

pub struct Graph {
    pub ids: Vec<String>,
    pub down: Vec<Option<usize>>,
    pub ups: Vec<Vec<usize>>,
    pub tot: Vec<f64>,
    pub local: Vec<f64>,
    pub len_km: Vec<f64>,
    pub lines: Vec<Vec<(f64, f64)>>,
    pub nexus_xy: HashMap<String, (f64, f64)>,
    pub cycles: usize,
    pub outlets: usize,
}

pub fn graph(f: &Fabric) -> Graph {
    let ids: Vec<String> = f.flowpaths.iter().map(|x| x.id.clone().unwrap_or_default()).collect();
    let idx: HashMap<&str, usize> = ids.iter().enumerate().map(|(i, s)| (s.as_str(), i)).collect();
    let nex: HashMap<&str, &str> = f.nexus.iter().map(|x| (x.id.as_deref().unwrap_or(""), x.toid.as_deref().unwrap_or(""))).collect();
    let down: Vec<Option<usize>> = f.flowpaths.iter().map(|x| nex.get(x.toid.as_deref().unwrap_or("")).and_then(|t| idx.get(t).copied())).collect();
    let mut ups = vec![Vec::new(); ids.len()];
    for (i, d) in down.iter().enumerate() {
        if let Some(d) = d {
            ups[*d].push(i);
        }
    }
    let mut indeg: Vec<usize> = ups.iter().map(|u| u.len()).collect();
    let mut q: Vec<usize> = (0..ids.len()).filter(|&i| indeg[i] == 0).collect();
    let mut seen = 0;
    while let Some(i) = q.pop() {
        seen += 1;
        if let Some(d) = down[i] {
            indeg[d] -= 1;
            if indeg[d] == 0 {
                q.push(d);
            }
        }
    }
    let lines = f.flowpaths.iter().map(|x| x.geom.as_ref().and_then(|g| geo::gpkg_rings(&g.0)).and_then(|r| r.into_iter().next()).unwrap_or_default()).collect();
    let nexus_xy = f.nexus.iter().filter_map(|x| Some((x.id.clone()?, geo::gpkg_rings(&x.geom.as_ref()?.0)?.first()?.first().copied()?))).collect();
    Graph {
        down: down.clone(),
        ups,
        tot: f.flowpaths.iter().map(|x| x.tot_drainage_areasqkm.unwrap_or(f64::NAN)).collect(),
        local: f.flowpaths.iter().map(|x| x.areasqkm.unwrap_or(0.0)).collect(),
        len_km: f.flowpaths.iter().map(|x| x.lengthkm.unwrap_or(0.0)).collect(),
        lines,
        nexus_xy,
        cycles: ids.len() - seen,
        outlets: down.iter().filter(|d| d.is_none()).count(),
        ids,
    }
}

impl Graph {
    pub fn subtree(&self, roots: &[usize]) -> Vec<usize> {
        let mut out = Vec::new();
        let mut st = roots.to_vec();
        let mut seen = HashSet::new();
        while let Some(i) = st.pop() {
            if seen.insert(i) {
                out.push(i);
                st.extend(self.ups[i].iter().copied());
            }
        }
        out
    }
    /// Sources of the subnetwork with drainage area >= a (Shreve magnitude).
    pub fn magnitude(&self, set: &[usize], a: f64) -> usize {
        let s: HashSet<usize> = set.iter().copied().collect();
        set.iter().filter(|&&i| self.tot[i] >= a && !self.ups[i].iter().any(|u| s.contains(u) && self.tot[*u] >= a)).count()
    }
    /// Length (km) of the main stem from `start` up the largest-drainage branch.
    pub fn mainstem_km(&self, start: usize) -> f64 {
        let mut l = 0.0;
        let mut cur = Some(start);
        while let Some(i) = cur {
            l += self.len_km[i];
            cur = self.ups[i].iter().copied().max_by(|a, b| self.tot[*a].total_cmp(&self.tot[*b]));
        }
        l
    }
}

/// EPSG:5070 (NAD83 / Conus Albers) inverse, GRS80.
pub fn albers5070_inverse(x: f64, y: f64) -> (f64, f64) {
    let a = 6_378_137.0;
    let f = 1.0 / 298.257_222_101;
    let e2: f64 = f * (2.0 - f);
    let e = e2.sqrt();
    let qf = |p: f64| {
        let s = p.sin();
        (1.0 - e2) * (s / (1.0 - e2 * s * s) - (1.0 / (2.0 * e)) * ((1.0 - e * s) / (1.0 + e * s)).ln())
    };
    let mf = |p: f64| p.cos() / (1.0 - e2 * p.sin().powi(2)).sqrt();
    let (p0, p1, p2, l0) = (23f64.to_radians(), 29.5f64.to_radians(), 45.5f64.to_radians(), -96f64.to_radians());
    let (m1, m2) = (mf(p1), mf(p2));
    let (q0, q1, q2) = (qf(p0), qf(p1), qf(p2));
    let n = (m1 * m1 - m2 * m2) / (q2 - q1);
    let c = m1 * m1 + n * q1;
    let rho0 = a * (c - n * q0).sqrt() / n;
    let rho = (x * x + (rho0 - y).powi(2)).sqrt();
    let q = (c - rho * rho * n * n / (a * a)) / n;
    let theta = x.atan2(rho0 - y);
    let lon = l0 + theta / n;
    let mut phi = (q / 2.0).asin();
    for _ in 0..20 {
        let s = phi.sin();
        let d = (1.0 - e2 * s * s).powi(2) / (2.0 * phi.cos()) * (q / (1.0 - e2) - s / (1.0 - e2 * s * s) + (1.0 / (2.0 * e)) * ((1.0 - e * s) / (1.0 + e * s)).ln());
        phi += d;
        if d.abs() < 1e-13 {
            break;
        }
    }
    (lon.to_degrees(), phi.to_degrees())
}

/// Planar distance from p to a polyline (same projected CRS), metres.
fn dist_to_line(p: (f64, f64), l: &[(f64, f64)]) -> f64 {
    let mut best = f64::INFINITY;
    for w in l.windows(2) {
        let (a, b) = (w[0], w[1]);
        let (dx, dy) = (b.0 - a.0, b.1 - a.1);
        let t = if dx == 0.0 && dy == 0.0 { 0.0 } else { (((p.0 - a.0) * dx + (p.1 - a.1) * dy) / (dx * dx + dy * dy)).clamp(0.0, 1.0) };
        let (cx, cy) = (a.0 + t * dx, a.1 + t * dy);
        best = best.min(((p.0 - cx).powi(2) + (p.1 - cy).powi(2)).sqrt());
    }
    if l.len() == 1 {
        best = ((p.0 - l[0].0).powi(2) + (p.1 - l[0].1).powi(2)).sqrt();
    }
    best
}

/// Fraction of the line length (from its first vertex) at the point of the
/// line nearest to `p`.
fn along_line(p: (f64, f64), l: &[(f64, f64)]) -> f64 {
    let (mut best, mut at, mut run) = (f64::INFINITY, 0.0, 0.0);
    for w in l.windows(2) {
        let (a, b) = (w[0], w[1]);
        let (dx, dy) = (b.0 - a.0, b.1 - a.1);
        let seg = (dx * dx + dy * dy).sqrt();
        let t = if seg == 0.0 { 0.0 } else { (((p.0 - a.0) * dx + (p.1 - a.1) * dy) / (seg * seg)).clamp(0.0, 1.0) };
        let d = ((p.0 - a.0 - t * dx).powi(2) + (p.1 - a.1 - t * dy).powi(2)).sqrt();
        if d < best {
            best = d;
            at = run + t * seg;
        }
        run += seg;
    }
    if run > 0.0 { at / run } else { 1.0 }
}

pub struct Tolerances {
    pub area_rel: f64,
    pub magnitude_rel: f64,
    pub mainstem_rel: f64,
}

/// `ours` must be in an ARWEN LAEA CRS (lat0/lon0 from its WKT), `reference`
/// in EPSG:5070. Gauges are the reference's Gages hydrolocations (optionally
/// a named subset).
pub fn compare(ours: &Fabric, reference: &Fabric, gauges: &[String], radius_km: f64, thresholds: &[f64], tol: &Tolerances) -> Result<Value> {
    let param = |name: &str| -> Result<f64> {
        let d = &ours.srs.definition;
        let k = d.find(&format!("PARAMETER[\"{name}\",")).ok_or_else(|| format!("fabric CRS has no {name}"))?;
        let rest = &d[k + name.len() + 13..];
        Ok(rest[..rest.find(',').ok_or("bad WKT")?].parse()?)
    };
    let proj = geo::Laea::new(param("Latitude of natural origin")?, param("Longitude of natural origin")?);
    if reference.srs.organization != "EPSG" || reference.srs.organization_coordsys_id != 5070 {
        return Err("reference fabric must be EPSG:5070 (NOAA CONUS fabric)".into());
    }
    // Our domain box (fabric_metadata.domain, a WKT POLYGON of the box).
    let dom = ours.fabric_metadata.iter().flatten().find(|m| m.key.as_deref() == Some("domain")).and_then(|m| m.value.clone()).ok_or("fabric has no fabric_metadata.domain")?;
    let nums: Vec<f64> = dom.replace(['(', ')', ','], " ").split_whitespace().filter_map(|s| s.parse().ok()).collect();
    let (bw, bs, be, bn) = (nums.iter().step_by(2).cloned().fold(f64::MAX, f64::min), nums.iter().skip(1).step_by(2).cloned().fold(f64::MAX, f64::min), nums.iter().step_by(2).cloned().fold(f64::MIN, f64::max), nums.iter().skip(1).step_by(2).cloned().fold(f64::MIN, f64::max));
    let go = graph(ours);
    let gr = graph(reference);
    let ref_idx_by_toid: HashMap<&str, Vec<usize>> = {
        let mut m: HashMap<&str, Vec<usize>> = HashMap::new();
        for (i, x) in reference.flowpaths.iter().enumerate() {
            m.entry(x.toid.as_deref().unwrap_or("")).or_default().push(i);
        }
        m
    };
    let mut seen_uri = HashSet::new();
    let mut rows = Vec::new();
    for h in &reference.hydrolocations {
        let uri = h.hl_uri.clone().unwrap_or_default();
        if h.hl_reference.as_deref() != Some("Gages") || (!gauges.is_empty() && !gauges.contains(&uri)) || !seen_uri.insert(uri.clone()) {
            continue;
        }
        let nex = h.id.clone().unwrap_or_default();
        let Some(roots) = ref_idx_by_toid.get(nex.as_str()) else { continue };
        let Some(&(rx, ry)) = gr.nexus_xy.get(&nex) else { continue };
        let (lon, lat) = albers5070_inverse(rx, ry);
        let ref_da: f64 = roots.iter().map(|&i| gr.tot[i]).sum();
        let ref_sub = gr.subtree(roots);
        let (mut w, mut s, mut e, mut n) = (f64::MAX, f64::MAX, f64::MIN, f64::MIN);
        for &i in &ref_sub {
            for &(x, y) in &gr.lines[i] {
                let (lo, la) = albers5070_inverse(x, y);
                w = w.min(lo);
                e = e.max(lo);
                s = s.min(la);
                n = n.max(la);
            }
        }
        // Snap: reaches of ours within the radius; prefer the one whose
        // drainage interval [tot - local, tot] is closest to the reference area.
        let p = proj.forward(lon, lat);
        let mut best: Option<(f64, f64, usize)> = None;
        for i in 0..go.ids.len() {
            let d = dist_to_line(p, &go.lines[i]);
            if d > radius_km * 1000.0 {
                continue;
            }
            let (lo_a, hi_a) = (go.tot[i] - go.local[i], go.tot[i]);
            let miss = if ref_da < lo_a { (lo_a / ref_da).ln() } else if ref_da > hi_a { (ref_da / hi_a).ln() } else { 0.0 };
            if best.is_none_or(|(bm, bd, _)| miss < bm || (miss == bm && d < bd)) {
                best = Some((miss, d, i));
            }
        }
        let mags: Vec<Value> = thresholds
            .iter()
            .map(|&a| {
                let r = gr.magnitude(&ref_sub, a);
                let o = best.map(|(_, _, i)| go.magnitude(&go.subtree(&[i]), a));
                json!({"min_drainage_km2": a, "reference_sources": r, "ours_sources": o, "rel_diff": o.map(|o| (o as f64 - r as f64) / (r.max(1) as f64))})
            })
            .collect();
        // A gauge is in scope only when its whole reference basin lies inside
        // our box: a box-clipped basin cannot have the same drainage area.
        let in_scope = w >= bw && e <= be && s >= bs && n <= bn;
        if !(lon >= bw && lon <= be && lat >= bs && lat <= bn) {
            continue;
        }
        let row = match best {
            None => json!({"gauge": uri, "nexus": nex, "lon": lon, "lat": lat, "reference_drainage_km2": ref_da, "matched": false, "in_scope": in_scope}),
            Some((miss, d, i)) => {
                let sub = go.subtree(&[i]);
                let ms_ref: f64 = roots.iter().map(|&r| gr.mainstem_km(r)).fold(0.0, f64::max);
                let ms_ours = go.mainstem_km(i);
                let area_rel = miss.exp() - 1.0;
                // Point estimate: local area spread evenly along the reach, read at
                // the gauge position (strict for every network, whatever its reach size).
                let frac = along_line(p, &go.lines[i]);
                let point_da = go.tot[i] - go.local[i] + go.local[i] * frac;
                let point_rel = (point_da - ref_da) / ref_da;
                let mag_ok = mags.iter().all(|m| m["rel_diff"].as_f64().is_some_and(|x| x.abs() <= tol.magnitude_rel) || m["reference_sources"].as_u64().unwrap_or(0) < 5);
                json!({
                    "gauge": uri, "nexus": nex, "lon": lon, "lat": lat, "in_scope": in_scope,
                    "reference_subtree_bbox": [w, s, e, n],
                    "reference_drainage_km2": ref_da,
                    "matched": true, "ours_reach": go.ids[i], "snap_distance_m": d,
                    "ours_drainage_interval_km2": [go.tot[i] - go.local[i], go.tot[i]],
                    "drainage_rel_diff": area_rel,
                    "ours_drainage_at_gauge_km2": point_da,
                    "drainage_at_gauge_rel_diff": point_rel,
                    "reference_reaches_upstream": ref_sub.len(), "ours_reaches_upstream": sub.len(),
                    "magnitude": mags,
                    "mainstem_km": {"reference": ms_ref, "ours": ms_ours, "rel_diff": (ms_ours - ms_ref) / ms_ref},
                    "within_tolerance": {
                        "drainage": area_rel.abs() <= tol.area_rel,
                        "drainage_at_gauge": point_rel.abs() <= tol.area_rel,
                        "magnitude": mag_ok,
                        "mainstem": ((ms_ours - ms_ref) / ms_ref).abs() <= tol.mainstem_rel
                    }
                })
            }
        };
        rows.push(row);
    }
    let scoped: Vec<&Value> = rows.iter().filter(|r| r["in_scope"] == true).collect();
    let pass = |k: &str| scoped.iter().filter(|r| r["within_tolerance"][k] == true).count();
    let summary = json!({
        "gauges_in_box": rows.len(),
        "gauges_with_whole_reference_basin_in_box": scoped.len(),
        "matched": scoped.iter().filter(|r| r["matched"] == true).count(),
        "within_drainage_tolerance": pass("drainage"),
        "within_drainage_at_gauge_tolerance": pass("drainage_at_gauge"),
        "max_abs_drainage_at_gauge_rel_diff": scoped.iter().filter_map(|r| r["drainage_at_gauge_rel_diff"].as_f64()).fold(0.0, |a: f64, x| a.max(x.abs())),
        "within_magnitude_tolerance": pass("magnitude"),
        "within_mainstem_tolerance": pass("mainstem"),
        "max_abs_drainage_rel_diff": scoped.iter().filter_map(|r| r["drainage_rel_diff"].as_f64()).fold(0.0, |a: f64, x| a.max(x.abs())),
    });
    Ok(json!({
        "domain_bbox": [bw, bs, be, bn],
        "summary": summary,
        "ours": {"flowpaths": go.ids.len(), "outlets": go.outlets, "cycles": go.cycles},
        "reference": {"flowpaths": gr.ids.len(), "outlets": gr.outlets, "cycles": gr.cycles},
        "tolerances": {"drainage_rel": tol.area_rel, "magnitude_rel": tol.magnitude_rel, "magnitude_applies_when_reference_sources_at_least": 5, "mainstem_rel": tol.mainstem_rel},
        "gauges": rows
    }))
}
