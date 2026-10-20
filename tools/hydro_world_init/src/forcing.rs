//! Spin-up forcing per catchment and hour from the row's forcing bindings.
//!
//! Temporal kinds (a binding or a single term may set its own):
//! - interval_total: the value stamped t is the total over (t - interval, t]
//! - daily_running_total: running total since 00 UTC (ERA5-Land); de-accumulated
//! - interval_mean_rate: mean rate over (t - interval, t], per second
//! A step [s, s+3600) takes the source interval containing it, scaled to one hour.
use crate::ncread;
use crate::state::LandWeights;
use crate::table::{Forcing, Row};
use crate::Result;
use rayon::prelude::*;
use std::collections::BTreeMap;

/// Catchment means of one variable at every record of every file.
type Series = BTreeMap<i64, Vec<f64>>;

fn read_variables(files: &[String], vars: &[String], lw: &LandWeights) -> Result<BTreeMap<String, Series>> {
    let per_file: Vec<Result<Vec<(String, i64, Vec<f64>)>>> = files
        .par_iter()
        .map(|path| {
            let src = ncread::open(path)?;
            let mut out = Vec::new();
            for (r, t) in src.times.iter().enumerate() {
                for v in vars {
                    let f = ncread::field(&src, v, r)?;
                    let mut c = Vec::with_capacity(lw.catchments());
                    for k in 0..lw.catchments() {
                        c.push(lw.mean(k, &f).ok_or_else(|| format!("{path}: {v} missing over a catchment at record {r}"))?);
                    }
                    out.push((v.clone(), *t, c));
                }
            }
            Ok(out)
        })
        .collect();
    let mut all: BTreeMap<String, Series> = BTreeMap::new();
    for r in per_file {
        for (v, t, c) in r? {
            if all.entry(v.clone()).or_default().insert(t, c).is_some() {
                return Err(format!("forcing variable {v} has two records at {}", crate::time::iso(t)).into());
            }
        }
    }
    Ok(all)
}

fn step_value(series: &Series, temporal: &str, interval: i64, offset: i64, step_start: i64, k: usize) -> Result<f64> {
    // The interval ending at stamp t covers (t + offset, t + offset + interval]
    // with offset = -interval for interval-end stamps.
    // Stamps sit on multiples of the interval: the one whose interval holds
    // step_start is the smallest multiple above step_start - offset - interval.
    let stamp = ((step_start - offset - interval).div_euclid(interval) + 1) * interval;
    let get = |t: i64| -> Result<f64> {
        series.get(&t).map(|v| v[k]).ok_or_else(|| format!("forcing record {} is missing", crate::time::iso(t)).into())
    };
    let hour_share = 3600.0 / interval as f64;
    match temporal {
        "interval_total" => Ok(get(stamp)? * hour_share),
        "interval_mean_rate" => Ok(get(stamp)? * 3600.0),
        "daily_running_total" => {
            if interval != 3600 {
                return Err("daily_running_total needs hourly records".into());
            }
            let hour = stamp.rem_euclid(86_400) / 3600;
            Ok(if hour == 1 { get(stamp)? } else { get(stamp)? - get(stamp - 3600)? })
        }
        t => Err(format!("unknown temporal kind {t}").into()),
    }
}

/// [hour][catchment] series for one binding, in the binding's units x scale,
/// converted to depth per hour.
fn binding_series(b: &Forcing, data: &BTreeMap<String, Series>, start: i64, hours: usize, c: usize) -> Result<Vec<Vec<f64>>> {
    let mut out = vec![vec![0.0; c]; hours];
    for (h, row) in out.iter_mut().enumerate() {
        let s = start + 3600 * h as i64;
        for (k, slot) in row.iter_mut().enumerate() {
            let mut v = 0.0;
            for term in &b.terms {
                let temporal = term.temporal.as_deref().unwrap_or(&b.temporal);
                let scale = term.scale.or(b.scale).unwrap_or(1.0);
                v += term.sign * scale * step_value(&data[&term.variable], temporal, b.interval_seconds, b.time_offset_seconds, s, k)?;
            }
            if let Some(m) = b.clamp_min {
                if v < m {
                    v = m;
                }
            }
            *slot = v;
        }
    }
    Ok(out)
}

pub struct SpinForcing {
    /// metres of liquid water per hour, [hour][catchment]
    pub water_m: Vec<Vec<f64>>,
    /// metres of potential evaporation per hour, [hour][catchment]
    pub pet_m: Vec<Vec<f64>>,
    pub records_read: usize,
}

pub fn load(row: &Row, files: &[String], lw: &LandWeights, start: i64, hours: usize) -> Result<SpinForcing> {
    let water = row.forcing.get("water_input").ok_or("row has no water_input forcing binding")?;
    let pet = row.forcing.get("pet").ok_or("row has no pet forcing binding")?;
    let mut vars: Vec<String> = water.terms.iter().chain(&pet.terms).map(|t| t.variable.clone()).collect();
    vars.sort();
    vars.dedup();
    let data = read_variables(files, &vars, lw)?;
    let c = lw.catchments();
    let records_read = data.values().map(|s| s.len()).sum();
    Ok(SpinForcing {
        water_m: binding_series(water, &data, start, hours, c)?,
        pet_m: binding_series(pet, &data, start, hours, c)?,
        records_read,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn interval_end_stamps_and_running_totals() {
        let mut s: Series = BTreeMap::new();
        let day = 1_672_531_200; // 2023-01-01T00Z
        // running total since 00 UTC: 1 mm per hour
        for h in 1..=24 {
            s.insert(day + 3600 * h, vec![h as f64]);
        }
        // step 00-01 takes stamp 01 (offset -3600)
        assert_eq!(step_value(&s, "interval_total", 3600, -3600, day, 0).unwrap(), 1.0);
        assert_eq!(step_value(&s, "daily_running_total", 3600, -3600, day, 0).unwrap(), 1.0);
        assert_eq!(step_value(&s, "daily_running_total", 3600, -3600, day + 5 * 3600, 0).unwrap(), 1.0);
        // 3-hourly mean rate stamped at interval end covers three hourly steps
        let mut r: Series = BTreeMap::new();
        r.insert(day + 10800, vec![2.0]);
        for h in 0..3 {
            assert_eq!(step_value(&r, "interval_mean_rate", 10800, -10800, day + 3600 * h, 0).unwrap(), 7200.0);
        }
    }
}
