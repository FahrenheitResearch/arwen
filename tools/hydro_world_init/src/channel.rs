//! Channel initial state from the spin-up: steady-state flow accumulated down
//! the fabric network from each catchment's last-day mean runoff, with depth
//! and velocity from Manning's equation on the reach's trapezoid, using the
//! t-route MC geometry convention of the engine kernel (z = 1/ChSlp, ChSlp 0
//! -> z = 1; gpuwm/hydro/kernels/troute_mc.cu). Output rows are
//! (Q m3/s, velocity m/s, depth m) per flowpath, the routing_state triple
//! the engine's MC component takes as initial_state_component.
use crate::Result;
use rayon::prelude::*;
use hydro_world::contract::Fabric;
use std::collections::{BTreeMap, VecDeque};

pub struct ChannelInit {
    pub ids: Vec<String>,
    pub state: Vec<[f64; 3]>,
    pub terminal: usize,
    pub levels: usize,
    pub missing_geometry: usize,
}

fn normal_depth(q: f64, n: f64, bw: f64, cs: f64, s0: f64) -> (f64, f64) {
    if q <= 0.0 {
        return (0.0, 0.0);
    }
    let z = if cs == 0.0 { 1.0 } else { 1.0 / cs };
    let flow = |h: f64| {
        let area = (bw + z * h) * h;
        let wp = bw + 2.0 * h * (1.0 + z * z).sqrt();
        (area, (1.0 / n) * area * (area / wp).powf(2.0 / 3.0) * s0.sqrt())
    };
    let (mut lo, mut hi) = (0.0, 1.0);
    while flow(hi).1 < q && hi < 1.0e4 {
        hi *= 2.0;
    }
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if flow(mid).1 < q { lo = mid } else { hi = mid }
    }
    let h = 0.5 * (lo + hi);
    let (a, _) = flow(h);
    (h, if a > 0.0 { q / a } else { 0.0 })
}

/// `lateral` maps a divide id (cat-n) to its steady inflow in m3/s.
pub fn steady_state(f: &Fabric, lateral: &BTreeMap<String, f64>) -> Result<ChannelInit> {
    let ids: Vec<String> = f
        .flowpaths
        .iter()
        .map(|p| p.id.clone().ok_or("flowpath without id"))
        .collect::<std::result::Result<_, _>>()?;
    let index: BTreeMap<&str, usize> = ids
        .iter()
        .enumerate()
        .map(|(i, s)| (s.as_str(), i))
        .collect();
    let nexus_to: BTreeMap<&str, &str> = f
        .nexus
        .iter()
        .filter_map(|n| Some((n.id.as_deref()?, n.toid.as_deref()?)))
        .collect();
    let n = ids.len();
    let mut down = vec![usize::MAX; n];
    let mut indeg = vec![0usize; n];
    for (i, p) in f.flowpaths.iter().enumerate() {
        let Some(nx) = p.toid.as_deref() else {
            continue;
        };
        let target = nexus_to
            .get(nx)
            .copied()
            .or_else(|| index.contains_key(nx).then_some(nx));
        if let Some(t) = target {
            if let Some(&j) = index.get(t) {
                down[i] = j;
                indeg[j] += 1;
            }
        }
    }
    let mut q = vec![0.0; n];
    for (i, p) in f.flowpaths.iter().enumerate() {
        if let Some(d) = p.divide_id.as_deref() {
            q[i] += lateral.get(d).copied().unwrap_or(0.0);
        }
    }
    // Kahn order: upstream first; level = longest path from a headwater
    let mut level = vec![0usize; n];
    let mut queue: VecDeque<usize> = (0..n).filter(|i| indeg[*i] == 0).collect();
    let mut seen = 0;
    while let Some(i) = queue.pop_front() {
        seen += 1;
        let j = down[i];
        if j != usize::MAX {
            q[j] += q[i];
            level[j] = level[j].max(level[i] + 1);
            indeg[j] -= 1;
            if indeg[j] == 0 {
                queue.push_back(j);
            }
        }
    }
    if seen != n {
        return Err(format!("flowpath network has a cycle ({} of {n} ordered)", seen).into());
    }
    let attrs: BTreeMap<&str, &hydro_world::contract::FlowpathAttributes> = f
        .flowpath_attributes
        .iter()
        .filter_map(|a| Some((a.id.as_deref()?, a)))
        .collect();
    let (state, missing_rows): (Vec<_>, Vec<_>) = ids
        .par_iter()
        .zip(&q)
        .map(|(id, &qq)| match attrs.get(id.as_str()) {
            Some(a) if a.n.is_some() && a.btm_wdth.is_some() && a.so.is_some() => {
                let s0 = a.so.unwrap().max(1e-5);
                let (h, v) = normal_depth(
                    qq,
                    a.n.unwrap(),
                    a.btm_wdth.unwrap(),
                    a.ch_slp.unwrap_or(0.0),
                    s0,
                );
                ([qq, v, h], false)
            }
            _ => ([qq, 0.0, 0.0], true),
        })
        .unzip();
    Ok(ChannelInit {
        terminal: down.iter().filter(|d| **d == usize::MAX).count(),
        levels: level.iter().max().map(|m| m + 1).unwrap_or(0),
        ids,
        state,
        missing_geometry: missing_rows.into_iter().filter(|x| *x).count(),
    })
}

#[cfg(test)]
mod tests {
    #[test]
    fn manning_inverts() {
        let (h, v) = super::normal_depth(10.0, 0.05, 5.0, 0.5, 0.001);
        let z = 2.0;
        let a = (5.0 + z * h) * h;
        let wp = 5.0 + 2.0 * h * (1.0f64 + z * z).sqrt();
        let q = (1.0 / 0.05) * a * (a / wp).powf(2.0 / 3.0) * 0.001f64.sqrt();
        assert!((q - 10.0).abs() < 1e-9 && (v * a - 10.0).abs() < 1e-9);
    }
}
