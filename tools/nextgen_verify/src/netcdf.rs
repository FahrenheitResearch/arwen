// SPDX-License-Identifier: Apache-2.0
//! Table-described, pure Rust NetCDF transport. No model-specific adapters.
use crate::{timestamp, utc, Observation, Result, Station};
use netcdf_reader::{NcFile, NcType, NcVariable};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, path::Path};

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Layout {
    pub feature: String,
    pub time: String,
    pub flow: String,
    pub epoch: String,
    pub time_unit_seconds: f64,
    pub expected_time_units: String,
}

fn units(v: &NcVariable) -> Result<String> {
    v.attribute("units")
        .and_then(|a| a.value.as_string())
        .ok_or_else(|| format!("{} missing text units", v.name).into())
}

fn clock(value: f64, seconds: f64, epoch: i64) -> Result<i64> {
    let ms = value * seconds * 1000.;
    if !ms.is_finite() || ms.fract() != 0. || ms < i64::MIN as f64 || ms >= i64::MAX as f64 {
        return Err("NetCDF time must resolve exactly to a finite UTC millisecond".into());
    }
    let t = epoch.checked_add(ms as i64).ok_or("NetCDF time overflow")?;
    chrono::DateTime::from_timestamp_millis(t).ok_or("NetCDF time outside UTC range")?;
    Ok(t)
}

pub fn decode(
    path: &Path,
    nc: &Layout,
    layout: &crate::Layout,
    stations: &[Station],
) -> Result<(Vec<Observation>, usize)> {
    if !layout.filters.is_empty()
        || !layout.optional_filters.is_empty()
        || layout.time_series.is_some()
        || stations.iter().any(|s| s.time_series_id.is_some())
    {
        return Err(
            "NetCDF feature-axis import does not accept JSON record filters/time-series selectors"
                .into(),
        );
    }
    if !nc.time_unit_seconds.is_finite() || nc.time_unit_seconds <= 0. {
        return Err("NetCDF time_unit_seconds must be finite and positive".into());
    }
    let file = NcFile::open(path)?;
    let feature = file.variable(&nc.feature)?;
    let time = file.variable(&nc.time)?;
    let flow = file.variable(&nc.flow)?;
    if feature.ndim() != 1
        || time.ndim() != 1
        || feature.dimensions[0].name == time.dimensions[0].name
    {
        return Err("NetCDF requires distinct one-dimensional feature/time coordinates".into());
    }
    if units(time)? != nc.expected_time_units {
        return Err("NetCDF time units differ from the declared table clock".into());
    }
    let unit = units(flow)?;
    let factor = *layout
        .unit_factors
        .get(&unit)
        .ok_or_else(|| format!("unsupported discharge unit {unit}"))?;
    if !factor.is_finite() || factor <= 0. {
        return Err("unit factor must be finite and positive".into());
    }
    let ids: Vec<String> = match feature.dtype() {
        NcType::Int64 => file
            .read_variable::<i64>(&nc.feature)?
            .iter()
            .map(ToString::to_string)
            .collect(),
        NcType::UInt64 => file
            .read_variable::<u64>(&nc.feature)?
            .iter()
            .map(ToString::to_string)
            .collect(),
        NcType::Int => file
            .read_variable::<i32>(&nc.feature)?
            .iter()
            .map(ToString::to_string)
            .collect(),
        NcType::UInt => file
            .read_variable::<u32>(&nc.feature)?
            .iter()
            .map(ToString::to_string)
            .collect(),
        _ => {
            return Err("NetCDF feature IDs must use an integer type; no f64 ID conversion".into())
        }
    };
    let index: BTreeMap<_, _> = ids
        .iter()
        .enumerate()
        .map(|(i, id)| (id.as_str(), i))
        .collect();
    if index.len() != ids.len() {
        return Err("duplicate NetCDF feature IDs".into());
    }
    let epoch = timestamp(&nc.epoch)?;
    let times = file.read_variable_unpacked_masked(&nc.time)?;
    let times: Vec<_> = times
        .iter()
        .map(|v| clock(*v, nc.time_unit_seconds, epoch))
        .collect::<Result<_>>()?;
    if times.is_empty() || ids.is_empty() {
        return Err("empty NetCDF feature/time coordinate".into());
    }
    let dims: Vec<_> = flow.dimensions.iter().map(|d| d.name.as_str()).collect();
    let fdim = feature.dimensions[0].name.as_str();
    let tdim = time.dimensions[0].name.as_str();
    // Coordinate dimension names establish order, including square arrays.
    let time_major = if dims == [tdim, fdim] {
        true
    } else if dims == [fdim, tdim] || (dims == [fdim] && times.len() == 1) {
        false
    } else {
        return Err("NetCDF flow dimensions must be feature,time or time,feature; feature-only requires one time".into());
    };
    let data = file.read_variable_unpacked_masked(&nc.flow)?;
    if data.len()
        != ids
            .len()
            .checked_mul(times.len())
            .ok_or("NetCDF shape overflow")?
    {
        return Err("NetCDF flow/coordinate size mismatch".into());
    }
    let data = data.as_slice().ok_or("non-contiguous NetCDF flow array")?;
    let jobs: Result<Vec<_>> = stations
        .par_iter()
        .map(|s| -> Result<_> {
            let id = s
                .feature_id
                .as_deref()
                .ok_or("NetCDF station lacks an explicit feature_id")?;
            let Some(&r) = index.get(id) else {
                return Ok((Vec::new(), 0));
            };
            let mut rows = Vec::with_capacity(times.len());
            let mut rejected = 0;
            for (t, stamp) in times.iter().enumerate() {
                let value = data[if time_major {
                    t * ids.len() + r
                } else {
                    r * times.len() + t
                }];
                if !value.is_finite() || value < 0. {
                    rejected += 1;
                    continue;
                }
                let flow_m3s = value * factor;
                if !flow_m3s.is_finite() {
                    return Err("nonfinite converted discharge".into());
                }
                rows.push(Observation {
                    station_id: s.station_id.clone(),
                    time: utc(*stamp),
                    flow_m3s,
                    quality: String::new(),
                });
            }
            Ok((rows, rejected))
        })
        .collect();
    let mut rows = Vec::new();
    let mut rejected = 0;
    for (mut r, n) in jobs? {
        rows.append(&mut r);
        rejected += n;
    }
    Ok((rows, rejected))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn declared_clock_keeps_minutes_offsets_and_rejects_submillisecond_or_fill() {
        let epoch = timestamp("1970-01-01T01:00:00+01:00").unwrap();
        assert_eq!(
            clock(61., 60., epoch).unwrap(),
            timestamp("1970-01-01T01:01:00Z").unwrap()
        );
        assert!(clock(f64::NAN, 1., epoch).is_err());
        assert!(clock(0.0001, 1., epoch).is_err());
        assert!(clock(f64::MAX, 60., epoch).is_err());
    }
}
