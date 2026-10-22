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

/// Catchment means of one variable at every record of the resident files.
type Series = BTreeMap<i64, Vec<f64>>;

/// The words a forcing record with no value over a catchment fails with; a
/// carried spin-up (carry_to) ends at the last complete day before it.
pub const MISSING: &str = "missing over a catchment";

/// Every record of one file reduced to catchment means: (variable, time, means).
/// Records and variables reduce in parallel.
fn read_file(path: &str, vars: &[String], lw: &LandWeights) -> Result<Vec<(String, i64, Vec<f64>)>> {
    let src = ncread::open(path)?;
    let jobs: Vec<(usize, i64, &String)> = src.times.iter().enumerate().flat_map(|(r, t)| vars.iter().map(move |v| (r, *t, v))).collect();
    jobs.par_iter()
        .map(|&(r, t, v)| -> Result<(String, i64, Vec<f64>)> {
            let f = ncread::field(&src, v, r)?;
            let mut c = Vec::with_capacity(lw.catchments());
            for k in 0..lw.catchments() {
                c.push(lw.mean(k, &f).ok_or_else(|| format!("{path}: {v} {MISSING} at record {r}"))?);
            }
            Ok((v.clone(), t, c))
        })
        .collect()
}

/// The stamp whose source interval holds the hour starting at `step_start`.
/// The interval ending at stamp t covers (t + offset, t + offset + interval]
/// with offset = -interval for interval-end stamps. Stamps sit on multiples
/// of the interval: the one whose interval holds step_start is the smallest
/// multiple above step_start - offset - interval.
fn stamp_for(interval: i64, offset: i64, step_start: i64) -> i64 {
    ((step_start - offset - interval).div_euclid(interval) + 1) * interval
}

fn step_value(series: &Series, temporal: &str, interval: i64, offset: i64, step_start: i64, k: usize) -> Result<f64> {
    let stamp = stamp_for(interval, offset, step_start);
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

/// [hour][catchment] series for one binding over hours `h0..h1`, in the
/// binding units x scale, converted to depth per hour (catchments in parallel;
/// each value is the same sum in the same order as before streaming).
fn binding_series(b: &Forcing, data: &BTreeMap<String, Series>, start: i64, h0: usize, h1: usize, c: usize) -> Result<Vec<Vec<f64>>> {
    (h0..h1)
        .map(|h| {
            let s = start + 3600 * h as i64;
            (0..c)
                .into_par_iter()
                .map(|k| -> Result<f64> {
                    let mut v = 0.0;
                    for term in &b.terms {
                        let temporal = term.temporal.as_deref().unwrap_or(&b.temporal);
                        let scale = term.scale.or(b.scale).unwrap_or(1.0);
                        let series = data.get(&term.variable).ok_or_else(|| format!("forcing variable {} not read", term.variable))?;
                        v += term.sign * scale * step_value(series, temporal, b.interval_seconds, b.time_offset_seconds, s, k)?;
                    }
                    if let Some(m) = b.clamp_min {
                        if v < m {
                            v = m;
                        }
                    }
                    Ok(v)
                })
                .collect()
        })
        .collect()
}

/// Spin-up forcing streamed in hour chunks: only the files the chunk's steps
/// read are resident. Breakage it prevents: the whole 90-day series (every
/// record of every file, then [hour][catchment] water and PET) was held at
/// once, about 70 GB for a 740 k-catchment continental fabric, and the spin-up
/// died. Values are unchanged: the same records, the same per-step sums.
pub struct ForcingStream<'a> {
    water: &'a Forcing,
    pet: &'a Forcing,
    vars: Vec<String>,
    files: Vec<(String, Vec<i64>)>,
    lw: &'a LandWeights,
    start: i64,
    resident: BTreeMap<usize, Vec<(String, i64, Vec<f64>)>>,
    pub records_read: usize,
}

impl<'a> ForcingStream<'a> {
    pub fn open(row: &'a Row, files: &[String], lw: &'a LandWeights, start: i64) -> Result<Self> {
        let water = row.forcing.get("water_input").ok_or("row has no water_input forcing binding")?;
        let pet = row.forcing.get("pet").ok_or("row has no pet forcing binding")?;
        let mut vars: Vec<String> = water.terms.iter().chain(&pet.terms).map(|t| t.variable.clone()).collect();
        vars.sort();
        vars.dedup();
        let times: Vec<Result<Vec<i64>>> = files.par_iter().map(|p| ncread::open(p).map(|s| s.times)).collect();
        let mut index = Vec::with_capacity(files.len());
        let mut seen = std::collections::HashSet::new();
        for (p, t) in files.iter().zip(times) {
            let t = t?;
            for &x in &t {
                if !seen.insert(x) {
                    return Err(format!("forcing variable {} has two records at {}", vars.first().map(String::as_str).unwrap_or("?"), crate::time::iso(x)).into());
                }
            }
            index.push((p.clone(), t));
        }
        Ok(Self { water, pet, vars, files: index, lw, start, resident: BTreeMap::new(), records_read: 0 })
    }

    fn stamps(&self, b: &Forcing, h0: usize, h1: usize, out: &mut std::collections::BTreeSet<i64>) {
        for h in h0..h1 {
            let s = self.start + 3600 * h as i64;
            for term in &b.terms {
                let temporal = term.temporal.as_deref().unwrap_or(&b.temporal);
                let t = stamp_for(b.interval_seconds, b.time_offset_seconds, s);
                out.insert(t);
                if temporal == "daily_running_total" {
                    out.insert(t - 3600);
                }
            }
        }
    }

    /// (water m/h, PET m/h), each [hour - h0][catchment], for hours h0..h1.
    pub fn chunk(&mut self, h0: usize, h1: usize) -> Result<(Vec<Vec<f64>>, Vec<Vec<f64>>)> {
        let mut need = std::collections::BTreeSet::new();
        self.stamps(self.water, h0, h1, &mut need);
        self.stamps(self.pet, h0, h1, &mut need);
        let wanted: Vec<usize> = (0..self.files.len()).filter(|&i| self.files[i].1.iter().any(|t| need.contains(t))).collect();
        self.resident.retain(|i, _| wanted.contains(i));
        let load: Vec<usize> = wanted.iter().copied().filter(|i| !self.resident.contains_key(i)).collect();
        let loaded: Vec<Result<Vec<(String, i64, Vec<f64>)>>> = load.par_iter().map(|&i| read_file(&self.files[i].0, &self.vars, self.lw)).collect();
        for (i, records) in load.into_iter().zip(loaded) {
            let records = records?;
            self.records_read += records.len();
            self.resident.insert(i, records);
        }
        let mut data: BTreeMap<String, Series> = BTreeMap::new();
        for records in self.resident.values() {
            for (v, t, c) in records {
                data.entry(v.clone()).or_default().insert(*t, c.clone());
            }
        }
        let c = self.lw.catchments();
        Ok((binding_series(self.water, &data, self.start, h0, h1, c)?, binding_series(self.pet, &data, self.start, h0, h1, c)?))
    }
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
