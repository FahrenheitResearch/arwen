// SPDX-License-Identifier: Apache-2.0
//! Station-to-reach matching against a HYDRO-WORLD contract fabric
//! (arwen.hydro-world.fabric.v1). A gauge is assigned to the nearest flowpath
//! inside the search radius whose drainage area agrees with the gauge's
//! published drainage area within the relative tolerance. The area along a
//! flowpath runs from its inflow area (total minus local) at the upstream end
//! to its total area at the outlet; agreement is checked against that range.
//! Sources that publish no drainage area use the row's declared policy.
use crate::geo::{gpkg_lines, nearest_on_line, Projection};
use crate::world::StationInfo;
use crate::Result;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Rules {
    /// Search radius around the gauge, metres in the fabric CRS.
    pub radius_m: f64,
    /// Relative drainage-area tolerance, e.g. 0.15 = 15 percent.
    pub area_tolerance: f64,
    /// For gauges whose source publishes no drainage area:
    /// `nearest_plausible` (nearest flowpath inside `radius_m` whose area can carry the
    /// gauge's observed mean flow at no more than `max_specific_discharge_m3s_km2`;
    /// without observations it falls back to `largest_within_radius` inside
    /// `no_area_radius_m`), `largest_within_radius`, or `refuse`.
    pub no_area_policy: String,
    /// Radius for the area-less fallback (defaults to `radius_m`): near a confluence
    /// only an area tells two rivers apart, so it is kept to the fabric's accuracy.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub no_area_radius_m: Option<f64>,
    /// Upper bound on mean specific discharge over the fetched window, m3/s per km2.
    /// 0.2 is 17 mm of runoff per day sustained over the whole window.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_specific_discharge_m3s_km2: Option<f64>,
}

#[derive(Clone, Debug)]
pub struct Reach {
    pub id: String,
    pub toid: String,
    pub line: Vec<[f64; 2]>,
    pub area_top_km2: f64,
    pub area_outlet_km2: f64,
    pub order: Option<f64>,
}

#[derive(Clone, Debug, Serialize)]
pub struct Candidate {
    pub flowpath_id: String,
    pub toid: String,
    pub distance_m: f64,
    pub fraction_along: f64,
    pub area_top_km2: f64,
    pub area_outlet_km2: f64,
    pub area_at_point_km2: f64,
    /// Relative distance of the gauge area from the reach's area range (0 inside the range).
    pub range_error: Option<f64>,
    pub point_error: Option<f64>,
}

#[derive(Clone, Debug, Serialize)]
pub struct Match {
    pub station_id: String,
    pub hl_uri: String,
    pub gauge_area_km2: Option<f64>,
    pub drainage_area_origin: String,
    pub projected_xy: [f64; 2],
    pub matched: Option<Candidate>,
    /// Mean observed flow over the fetched window, when observations were given.
    pub mean_flow_m3s: Option<f64>,
    /// Smallest drainage area that can carry that mean flow under the bound.
    pub min_plausible_area_km2: Option<f64>,
    pub rule: String,
    pub reason: Option<String>,
    /// Nearest candidates inside the radius (at most 5), for review.
    pub candidates: Vec<Candidate>,
}

/// Uniform grid over reach segments so each gauge looks only at nearby reaches.
pub struct Index {
    cell: f64,
    buckets: HashMap<(i64, i64), Vec<usize>>,
}
impl Index {
    pub fn new(reaches: &[Reach], cell: f64) -> Index {
        let mut buckets: HashMap<(i64, i64), Vec<usize>> = HashMap::new();
        for (i, r) in reaches.iter().enumerate() {
            let mut seen = std::collections::BTreeSet::new();
            for w in r.line.windows(2).chain(std::iter::once(&r.line[..1]).filter(|_| r.line.len() == 1)) {
                let (x0, x1) = (w[0][0].min(w[w.len() - 1][0]), w[0][0].max(w[w.len() - 1][0]));
                let (y0, y1) = (w[0][1].min(w[w.len() - 1][1]), w[0][1].max(w[w.len() - 1][1]));
                for cx in (x0 / cell).floor() as i64..=(x1 / cell).floor() as i64 {
                    for cy in (y0 / cell).floor() as i64..=(y1 / cell).floor() as i64 {
                        if seen.insert((cx, cy)) {
                            buckets.entry((cx, cy)).or_default().push(i);
                        }
                    }
                }
            }
        }
        Index { cell, buckets }
    }
    pub fn near(&self, p: [f64; 2], radius: f64) -> Vec<usize> {
        let mut out = Vec::new();
        let (cx0, cx1) = (((p[0] - radius) / self.cell).floor() as i64, ((p[0] + radius) / self.cell).floor() as i64);
        let (cy0, cy1) = (((p[1] - radius) / self.cell).floor() as i64, ((p[1] + radius) / self.cell).floor() as i64);
        for cx in cx0..=cx1 {
            for cy in cy0..=cy1 {
                if let Some(v) = self.buckets.get(&(cx, cy)) {
                    out.extend(v);
                }
            }
        }
        out.sort_unstable();
        out.dedup();
        out
    }
}

/// Read the flowpaths of a contract GeoPackage and its CRS.
pub fn load_fabric(path: &str) -> Result<(Vec<Reach>, Projection, i32, hydro_world::contract::Fabric)> {
    let (fabric, _report) = hydro_world::gpkg::read(path)?;
    let projection = Projection::from_wkt(&fabric.srs.definition)?;
    let srs = fabric.srs.srs_id as i32;
    let reaches: Result<Vec<Reach>> = fabric
        .flowpaths
        .par_iter()
        .map(|f| -> Result<Reach> {
            let id = f.id.clone().ok_or("flowpath without id")?;
            let geom = f.geom.as_ref().ok_or_else(|| format!("{id}: no geometry"))?;
            let line: Vec<[f64; 2]> = gpkg_lines(&geom.0)?.into_iter().flatten().collect();
            if line.is_empty() {
                return Err(format!("{id}: empty geometry").into());
            }
            let total = f.tot_drainage_areasqkm.ok_or_else(|| format!("{id}: no tot_drainage_areasqkm"))?;
            let local = f.areasqkm.unwrap_or(0.0).max(0.0);
            Ok(Reach {
                id,
                toid: f.toid.clone().unwrap_or_default(),
                line,
                area_top_km2: (total - local).max(0.0),
                area_outlet_km2: total,
                order: f.order,
            })
        })
        .collect();
    Ok((reaches?, projection, srs, fabric))
}

fn candidate(r: &Reach, p: [f64; 2], gauge_area: Option<f64>) -> Candidate {
    let (d, s) = nearest_on_line(&r.line, p);
    let at = r.area_top_km2 + s * (r.area_outlet_km2 - r.area_top_km2);
    let (range_error, point_error) = match gauge_area {
        Some(g) if g > 0.0 => {
            let range = if g < r.area_top_km2 {
                (r.area_top_km2 - g) / g
            } else if g > r.area_outlet_km2 {
                (g - r.area_outlet_km2) / g
            } else {
                0.0
            };
            (Some(range), Some((at - g).abs() / g))
        }
        _ => (None, None),
    };
    Candidate {
        flowpath_id: r.id.clone(),
        toid: r.toid.clone(),
        distance_m: d,
        fraction_along: s,
        area_top_km2: r.area_top_km2,
        area_outlet_km2: r.area_outlet_km2,
        area_at_point_km2: at,
        range_error,
        point_error,
    }
}

pub fn match_one(s: &StationInfo, reaches: &[Reach], index: &Index, proj: &Projection, rules: &Rules, mean_flow: Option<f64>) -> Match {
    let p = proj.forward(s.longitude, s.latitude);
    let qmax = rules.max_specific_discharge_m3s_km2.unwrap_or(0.2);
    let plausible = rules.no_area_policy == "nearest_plausible" && mean_flow.is_some();
    let min_area = mean_flow.map(|q| q / qmax);
    let radius = match s.drainage_area_km2 {
        Some(_) => rules.radius_m,
        None if plausible => rules.radius_m,
        None => rules.no_area_radius_m.unwrap_or(rules.radius_m),
    };
    let mut near: Vec<Candidate> = index
        .near(p, radius)
        .into_iter()
        .map(|i| candidate(&reaches[i], p, s.drainage_area_km2))
        .filter(|c| c.distance_m <= radius)
        .collect();
    near.sort_by(|a, b| a.distance_m.total_cmp(&b.distance_m).then_with(|| a.flowpath_id.cmp(&b.flowpath_id)));
    let (matched, rule, reason) = match s.drainage_area_km2 {
        Some(_) => {
            let ok = near
                .iter()
                .filter(|c| c.range_error.is_some_and(|e| e <= rules.area_tolerance))
                .min_by(|a, b| {
                    a.distance_m
                        .total_cmp(&b.distance_m)
                        .then_with(|| a.point_error.unwrap_or(f64::MAX).total_cmp(&b.point_error.unwrap_or(f64::MAX)))
                })
                .cloned();
            let reason = ok.is_none().then(|| {
                if near.is_empty() {
                    format!("no flowpath within {} m", rules.radius_m)
                } else {
                    format!(
                        "no flowpath within {} m has drainage area within {:.0} percent of the gauge's {:.1} km2",
                        rules.radius_m,
                        rules.area_tolerance * 100.0,
                        s.drainage_area_km2.unwrap()
                    )
                }
            });
            (ok, format!("nearest within {} m with drainage area within {:.0} percent", rules.radius_m, rules.area_tolerance * 100.0), reason)
        }
        None if plausible => {
            let need = min_area.unwrap();
            let ok = near.iter().find(|c| c.area_outlet_km2 >= need).cloned();
            let reason = ok.is_none().then(|| {
                format!("no flowpath within {radius} m drains the {need:.0} km2 that a mean flow of {:.1} m3/s needs", mean_flow.unwrap())
            });
            (
                ok,
                format!("source publishes no drainage area: nearest within {radius} m draining at least mean flow / {qmax} m3/s/km2"),
                reason,
            )
        }
        None if rules.no_area_policy == "largest_within_radius" || rules.no_area_policy == "nearest_plausible" => {
            let ok = near
                .iter()
                .max_by(|a, b| a.area_at_point_km2.total_cmp(&b.area_at_point_km2).then_with(|| b.distance_m.total_cmp(&a.distance_m)))
                .cloned();
            let reason = ok.is_none().then(|| format!("no flowpath within {radius} m"));
            (ok, format!("source publishes no drainage area: largest drainage area within {radius} m"), reason)
        }
        None => (None, "refuse without a drainage area".into(), Some("source publishes no drainage area and the policy refuses".into())),
    };
    Match {
        station_id: s.station_id.clone(),
        hl_uri: s.hl_uri.clone(),
        gauge_area_km2: s.drainage_area_km2,
        drainage_area_origin: s.drainage_area_origin.clone(),
        projected_xy: p,
        matched,
        mean_flow_m3s: mean_flow,
        min_plausible_area_km2: min_area,
        rule,
        reason,
        candidates: near.into_iter().take(5).collect(),
    }
}

pub fn match_all(stations: &[StationInfo], reaches: &[Reach], proj: &Projection, rules: &Rules, mean_flows: &HashMap<String, f64>) -> Vec<Match> {
    let index = Index::new(reaches, rules.radius_m.max(1000.0));
    stations
        .par_iter()
        .map(|s| match_one(s, reaches, &index, proj, rules, mean_flows.get(&s.station_id).copied()))
        .collect()
}

/// Contract hydrolocation rows (stage 8 input) for matched gauges.
pub fn hydrolocations(matches: &[Match], srs_id: i32) -> Vec<hydro_world::contract::Hydrolocation> {
    matches
        .iter()
        .filter_map(|m| m.matched.as_ref().map(|c| (m, c)))
        .enumerate()
        .map(|(i, (m, c))| hydro_world::contract::Hydrolocation {
            fid: i as i64 + 1,
            geom: Some(hydro_world::contract::Geometry(crate::geo::gpkg_point_bytes(srs_id, m.projected_xy))),
            hl_id: Some(m.hl_uri.clone()),
            hl_reference: Some("Gages".into()),
            hl_link: Some(m.station_id.clone()),
            hl_uri: Some(m.hl_uri.clone()),
            hl_position: Some("outflow".into()),
            id: Some(c.toid.clone()),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    fn reach(id: &str, line: Vec<[f64; 2]>, top: f64, out: f64) -> Reach {
        Reach { id: id.into(), toid: format!("nex-{id}"), line, area_top_km2: top, area_outlet_km2: out, order: None }
    }
    fn station(area: Option<f64>) -> StationInfo {
        StationInfo {
            station_id: "X:1".into(),
            agency: "X".into(),
            source: "s".into(),
            name: None,
            latitude: 0.0,
            longitude: 0.0,
            drainage_area_km2: area,
            drainage_area_origin: "test".into(),
            hl_uri: "Gages-X:1".into(),
            licence: String::new(),
            attribution: String::new(),
        }
    }
    #[test]
    fn nearest_reach_with_agreeing_area_wins_over_nearer_tributary() {
        // Projection centred at (0,0): a gauge at the origin is at x=y=0.
        let proj = Projection::Laea { a: 6_378_137.0, e: 0.08181919084262149, lat0: 0.0, lon0: 0.0, fe: 0.0, fn_: 0.0 };
        let reaches = vec![
            reach("wb-1", vec![[-2000.0, 100.0], [2000.0, 100.0]], 50.0, 60.0), // tributary 100 m away
            reach("wb-2", vec![[-3000.0, -900.0], [3000.0, -900.0]], 9000.0, 11000.0), // main stem 900 m away
        ];
        let flows = HashMap::new();
        let rules = Rules { radius_m: 2000.0, area_tolerance: 0.15, no_area_policy: "largest_within_radius".into(), no_area_radius_m: None, max_specific_discharge_m3s_km2: None };
        let m = match_all(&[station(Some(10_000.0))], &reaches, &proj, &rules, &flows);
        assert_eq!(m[0].matched.as_ref().unwrap().flowpath_id, "wb-2");
        let unmatched = match_all(&[station(Some(500.0))], &reaches, &proj, &rules, &flows);
        assert!(unmatched[0].matched.is_none() && unmatched[0].reason.as_ref().unwrap().contains("15 percent"));
        let no_area = match_all(&[station(None)], &reaches, &proj, &rules, &flows);
        assert_eq!(no_area[0].matched.as_ref().unwrap().flowpath_id, "wb-2");
        let tight = Rules { no_area_radius_m: Some(500.0), ..rules.clone() };
        assert_eq!(match_all(&[station(None)], &reaches, &proj, &tight, &flows)[0].matched.as_ref().unwrap().flowpath_id, "wb-1");
        // Discharge plausibility: 5 m3/s fits the 50-60 km2 tributary (nearest), 800 m3/s does not.
        let plaus = Rules { no_area_policy: "nearest_plausible".into(), ..rules.clone() };
        let q = |v: f64| HashMap::from([("X:1".to_string(), v)]);
        assert_eq!(match_all(&[station(None)], &reaches, &proj, &plaus, &q(5.0))[0].matched.as_ref().unwrap().flowpath_id, "wb-1");
        assert_eq!(match_all(&[station(None)], &reaches, &proj, &plaus, &q(800.0))[0].matched.as_ref().unwrap().flowpath_id, "wb-2");
        assert!(match_all(&[station(None)], &reaches, &proj, &plaus, &q(5000.0))[0].matched.is_none());
        let refuse = Rules { no_area_policy: "refuse".into(), ..rules };
        assert!(match_all(&[station(None)], &reaches, &proj, &refuse, &flows)[0].matched.is_none());
    }
}
