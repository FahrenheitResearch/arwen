//! From decoded BUFR subsets to neutral table rows: the surface reports
//! (SYNOP, the land templates 307080, 307086 and 307096 with or without
//! the WIGOS prefix 301150; SHIP 308009), and the upper-air soundings
//! (TEMP 309052, and the high-resolution 309056 where a level carries a
//! pressure).  Every template is walked the same way: the header
//! elements by descriptor wherever they sit, the level loop by the
//! replication path the reader tagged each item with, so a centre's
//! additions before or after the standard sequence do not move anything.
//!
//! What a row is (the acceptance contract, `measurement` column):
//! * SYNOP pressure: 010004 (station pressure, Pa) anchored at the
//!   barometer height (007031, else the station height 007030), measured
//!   `station_pressure`; a report with only 010051 (reduced to mean sea
//!   level) is anchored at 0 m as `sea_level_pressure`, the buoy
//!   convention of the table;
//! * SYNOP temperature and dewpoint: the first 012101 / 012103 of the
//!   report (the screen block 302072 / 302052), `screen_temperature_2m`
//!   and `screen_dewpoint_2m`, anchored at the station height; a ship's
//!   are `platform_temperature` / `platform_dewpoint`;
//! * SYNOP wind: the first 011001 / 011002 pair (the mean wind of 302042 /
//!   302059), `anemometer_wind_10m`, written only when the sensor height
//!   (the 007032 in force) is unstated or within 8 to 12 m; a wind at
//!   another height is counted, never reduced here (the neutral log law
//!   is a buoy rule with a buoy label);
//! * TEMP levels: every replication instance with a pressure 007004:
//!   temperature 012101, dewpoint 012103, wind 011001 / 011002 at the
//!   level, the level's time the launch time plus 004086, its position
//!   the station plus the displacements 005015 / 006015, its height
//!   010009 (the ISA height of the pressure when absent, a label only,
//!   as `rw_igra2` stamps it), measured `sonde_level`; the level flagged
//!   surface (008042 bit 1) also writes the station pressure row;
//! * the observation errors are the METAR and radiosonde values of the
//!   table (`rw_obs::table`).
//!
//! What is counted and not written: a report without a time or a
//! position, a variable-direction wind (direction 0 with speed), a value
//! outside the vocabulary's gross bounds, a dewpoint above its
//! temperature, a wind at a sensor height outside 8 to 12 m, a level
//! without a pressure, a template this layer does not read (its
//! descriptors are named in the record).

use std::collections::BTreeMap;

use chrono::{DateTime, Duration, TimeZone, Utc};
use serde::Serialize;

use crate::bufr::{first, fxy, Item, Message, Value};
use crate::table::{
    isa_altitude_m, wind_components, RowProvenance, TableRow, ERROR_DEWPOINT_ALOFT_K, ERROR_DEWPOINT_SURFACE_K,
    ERROR_SURFACE_PRESSURE_PA, ERROR_TEMPERATURE_ALOFT_K, ERROR_TEMPERATURE_SURFACE_K, ERROR_WIND_ALOFT_M_S,
    ERROR_WIND_SURFACE_M_S, GROSS_DEWPOINT_K, GROSS_SURFACE_PRESSURE_PA, GROSS_TEMPERATURE_K, GROSS_WIND_M_S,
    MEAS_ANEMOMETER_WIND_10M, MEAS_PLATFORM_DEWPOINT, MEAS_PLATFORM_TEMPERATURE, MEAS_SCREEN_DEWPOINT_2M,
    MEAS_SCREEN_TEMPERATURE_2M, MEAS_SEA_LEVEL_PRESSURE, MEAS_SONDE_LEVEL, MEAS_STATION_PRESSURE, VAR_DEWPOINT,
    VAR_SURFACE_PRESSURE, VAR_TEMPERATURE, VAR_WIND_U, VAR_WIND_V,
};

pub const SOURCE: &str = "wis2";
const WIND_SENSOR_MIN_M: f64 = 8.0;
const WIND_SENSOR_MAX_M: f64 = 12.0;
/// 008042 extended vertical sounding significance, bit 1 (of 18): surface.
const SOUNDING_SURFACE_BIT: u64 = 1 << 17;

/// What kind of report a subset's template is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum Kind {
    Synop,
    Ship,
    Temp,
    Unhandled,
}

/// The template a message carries, read from its unexpanded descriptors.
pub fn kind_of(descriptors: &[u32]) -> Kind {
    let mut kind = Kind::Unhandled;
    for &d in descriptors {
        match d {
            307080 | 307086 | 307096 | 307079 | 307081 | 307082 | 307083 | 307084 | 307085 | 307089 | 307090
            | 307091 | 307092 | 307093 | 307094 | 307095 | 307101 | 307102 | 307103 => return Kind::Synop,
            308009 | 308010 | 308011 | 308012 | 308013 => return Kind::Ship,
            309050 | 309051 | 309052 | 309053 | 309054 | 309055 | 309056 | 309057 => return Kind::Temp,
            _ => {
                let (f, x, _) = fxy(d);
                // A bare element template of a surface report (a centre's own
                // list) is read as SYNOP when it carries a station pressure or
                // a screen temperature.
                if f == 0 && x == 10 && d == 10004 || d == 12101 {
                    kind = Kind::Synop;
                }
            }
        }
    }
    kind
}

#[derive(Debug, Default, Serialize, Clone)]
pub struct Counters {
    pub subsets: usize,
    pub subsets_synop: usize,
    pub subsets_ship: usize,
    pub subsets_temp: usize,
    pub subsets_unhandled_template: usize,
    pub subsets_without_time: usize,
    pub subsets_without_position: usize,
    pub subsets_without_identifier: usize,
    pub values_out_of_range: usize,
    pub values_dewpoint_above_temperature: usize,
    pub values_wind_variable_direction: usize,
    pub winds_at_other_sensor_heights: usize,
    pub pressures_reduced_only: usize,
    pub sounding_levels: usize,
    pub sounding_levels_without_pressure: usize,
    pub sounding_levels_at_nominal_time: usize,
    pub sounding_heights_isa_stamped: usize,
    pub rows_by_variable: BTreeMap<String, usize>,
    pub unhandled_templates: BTreeMap<String, usize>,
}

fn number(items: &[Item], code: u32) -> Option<f64> {
    first(items, code).and_then(Value::number)
}

fn text(items: &[Item], code: u32) -> Option<String> {
    first(items, code).and_then(Value::text).map(str::to_string)
}

/// The report's identifier: the WIGOS identifier when the four parts are
/// present, else the WMO block and station number, else a ship or
/// mobile station identifier.
pub fn identifier(items: &[Item]) -> Option<String> {
    let series = number(items, 1125);
    let issuer = number(items, 1126);
    let issue = number(items, 1127);
    let local = text(items, 1128);
    if let (Some(s), Some(i), Some(n), Some(l)) = (series, issuer, issue, local.as_deref()) {
        return Some(format!("{}-{}-{}-{}", s as u64, i as u64, n as u64, l.replace(',', "_")));
    }
    if let (Some(block), Some(station)) = (number(items, 1001), number(items, 1002)) {
        return Some(format!("{:02}{:03}", block as u64, station as u64));
    }
    if let Some(ship) = text(items, 1011) {
        return Some(ship.replace(',', "_"));
    }
    if let Some(l) = local {
        return Some(l.replace(',', "_"));
    }
    None
}

/// The report time from 004001 to 004006 (the first of each; a sounding's
/// launch time sits under 008021 = 18 in 301113, the first 004xxx run).
pub fn report_time(items: &[Item]) -> Option<DateTime<Utc>> {
    let year = number(items, 4001)? as i32;
    let month = number(items, 4002)? as u32;
    let day = number(items, 4003)? as u32;
    let hour = number(items, 4004)? as u32;
    let minute = number(items, 4005).unwrap_or(0.0) as u32;
    let second = number(items, 4006).map(|s| s.max(0.0)).unwrap_or(0.0) as u32;
    Utc.with_ymd_and_hms(year, month, day, hour, minute, second.min(59)).single()
}

pub fn position(items: &[Item]) -> Option<(f64, f64)> {
    let lat = number(items, 5001).or_else(|| number(items, 5002))?;
    let lon = number(items, 6001).or_else(|| number(items, 6002))?;
    if !(-90.0..=90.0).contains(&lat) || !(-180.0..=360.0).contains(&lon) {
        return None;
    }
    Some((lat, lon))
}

fn in_bounds(value: f64, bounds: (f64, f64)) -> bool {
    value.is_finite() && bounds.0 <= value && value <= bounds.1
}

struct Emit<'a> {
    provenance: &'a RowProvenance,
    counters: &'a mut Counters,
    rows: &'a mut Vec<TableRow>,
}

impl Emit<'_> {
    #[allow(clippy::too_many_arguments)]
    fn push(&mut self, station: &str, lat: f64, lon: f64, elevation_m: f64, level_pa: Option<f64>,
            when: DateTime<Utc>, variable: &str, value: f64, error: f64, measurement: &'static str,
            nominal: Option<DateTime<Utc>>) {
        let mut provenance = self.provenance.measuring(measurement);
        if let Some(n) = nominal {
            provenance = provenance.nominal(n);
        }
        *self.counters.rows_by_variable.entry(variable.to_string()).or_insert(0) += 1;
        self.rows.push(TableRow {
            source: SOURCE.to_string(),
            station_id: station.to_string(),
            latitude_deg: lat,
            longitude_deg: lon,
            elevation_m,
            level_pa,
            valid_time: when,
            variable: variable.to_string(),
            value,
            error,
            provenance,
        });
    }

    /// A wind pair from a direction and speed, screened: the variable
    /// direction encoding (0 with a speed), the gross bound.
    #[allow(clippy::too_many_arguments)]
    fn wind(&mut self, station: &str, lat: f64, lon: f64, elevation_m: f64, level_pa: Option<f64>,
            when: DateTime<Utc>, direction: f64, speed: f64, error: f64, measurement: &'static str,
            nominal: Option<DateTime<Utc>>) {
        if !in_bounds(speed, GROSS_WIND_M_S) || !(0.0..=360.0).contains(&direction) {
            self.counters.values_out_of_range += 1;
            return;
        }
        if speed > 0.0 && direction == 0.0 {
            self.counters.values_wind_variable_direction += 1;
            return;
        }
        let (u, v) = wind_components(direction, speed);
        self.push(station, lat, lon, elevation_m, level_pa, when, VAR_WIND_U, u, error, measurement, nominal);
        self.push(station, lat, lon, elevation_m, level_pa, when, VAR_WIND_V, v, error, measurement, nominal);
    }
}

/// The sensor height (007032) in force before the first item of `code`:
/// the last 007032 seen before it in template order.
fn sensor_height_before(items: &[Item], code: u32) -> Option<f64> {
    let mut height = None;
    for item in items {
        if item.code == 7032 {
            height = item.value.number();
        }
        if item.code == code {
            return height;
        }
    }
    None
}

/// The first non-missing value of `code`, and whether the element sits in
/// the subset at all.
fn first_present(items: &[Item], code: u32) -> Option<f64> {
    items.iter().filter(|i| i.code == code).find_map(|i| i.value.number())
}

fn surface_rows(items: &[Item], ship: bool, provenance: &RowProvenance, counters: &mut Counters, rows: &mut Vec<TableRow>) {
    let Some(station) = identifier(items) else {
        counters.subsets_without_identifier += 1;
        return;
    };
    let Some(when) = report_time(items) else {
        counters.subsets_without_time += 1;
        return;
    };
    let Some((lat, lon)) = position(items) else {
        counters.subsets_without_position += 1;
        return;
    };
    let station_height = number(items, 7030).filter(|h| h.is_finite() && *h > -500.0 && *h < 9000.0);
    let barometer_height = number(items, 7031).filter(|h| h.is_finite() && *h > -500.0 && *h < 9000.0);
    let ground = station_height.unwrap_or(if ship { 0.0 } else { f64::NAN });
    let mut emit = Emit { provenance, counters, rows };
    // Pressure.
    match (first_present(items, 10004), first_present(items, 10051)) {
        (Some(p), _) => {
            let anchor = barometer_height.or(station_height).unwrap_or(if ship { 0.0 } else { f64::NAN });
            if !anchor.is_finite() {
                emit.counters.subsets_without_position += 1;
            } else if in_bounds(p, GROSS_SURFACE_PRESSURE_PA) {
                emit.push(&station, lat, lon, anchor, None, when, VAR_SURFACE_PRESSURE, p, ERROR_SURFACE_PRESSURE_PA,
                          MEAS_STATION_PRESSURE, None);
            } else {
                emit.counters.values_out_of_range += 1;
            }
        }
        (None, Some(p)) => {
            emit.counters.pressures_reduced_only += 1;
            if in_bounds(p, GROSS_SURFACE_PRESSURE_PA) {
                emit.push(&station, lat, lon, 0.0, None, when, VAR_SURFACE_PRESSURE, p, ERROR_SURFACE_PRESSURE_PA,
                          MEAS_SEA_LEVEL_PRESSURE, None);
            } else {
                emit.counters.values_out_of_range += 1;
            }
        }
        (None, None) => {}
    }
    if !ground.is_finite() {
        // No station height: the temperature, dewpoint and wind have no
        // anchor for the surface operators.
        return;
    }
    let temperature = first_present(items, 12101).or_else(|| first_present(items, 12104));
    let dewpoint = first_present(items, 12103).or_else(|| first_present(items, 12106));
    let (t_label, td_label) = if ship {
        (MEAS_PLATFORM_TEMPERATURE, MEAS_PLATFORM_DEWPOINT)
    } else {
        (MEAS_SCREEN_TEMPERATURE_2M, MEAS_SCREEN_DEWPOINT_2M)
    };
    if let Some(t) = temperature {
        if in_bounds(t, GROSS_TEMPERATURE_K) {
            emit.push(&station, lat, lon, ground, None, when, VAR_TEMPERATURE, t, ERROR_TEMPERATURE_SURFACE_K, t_label, None);
        } else {
            emit.counters.values_out_of_range += 1;
        }
    }
    if let Some(td) = dewpoint {
        if !in_bounds(td, GROSS_DEWPOINT_K) {
            emit.counters.values_out_of_range += 1;
        } else if temperature.is_some_and(|t| td > t + 0.05) {
            emit.counters.values_dewpoint_above_temperature += 1;
        } else {
            emit.push(&station, lat, lon, ground, None, when, VAR_DEWPOINT, td, ERROR_DEWPOINT_SURFACE_K, td_label, None);
        }
    }
    let direction = first_present(items, 11001).or_else(|| first_present(items, 11011));
    let speed = first_present(items, 11002).or_else(|| first_present(items, 11012));
    if let (Some(direction), Some(speed)) = (direction, speed) {
        let height = sensor_height_before(items, 11002).or_else(|| number(items, 7033));
        match height {
            Some(h) if !(WIND_SENSOR_MIN_M..=WIND_SENSOR_MAX_M).contains(&h) => {
                emit.counters.winds_at_other_sensor_heights += 1;
            }
            _ => emit.wind(&station, lat, lon, ground, None, when, direction, speed, ERROR_WIND_SURFACE_M_S,
                           MEAS_ANEMOMETER_WIND_10M, None),
        }
    }
}

/// One level of a sounding: the items of one replication instance.
struct Level<'a> {
    items: Vec<&'a Item>,
}

fn levels_of(items: &[Item]) -> Vec<Level<'_>> {
    // The level loop is the first replication whose instances carry a
    // pressure or a height and a temperature or wind (303054 / 303055):
    // group the replicated items by their outermost path index.
    let mut levels: BTreeMap<u32, Vec<&Item>> = BTreeMap::new();
    for item in items {
        if item.path.is_empty() {
            continue;
        }
        if matches!(item.code, 4086 | 8042 | 7004 | 7009 | 10009 | 5015 | 6015 | 12101 | 12103 | 11001 | 11002 | 7007) {
            levels.entry(item.path[0]).or_default().push(item);
        }
    }
    // Only the instances that look like levels (a pressure or a height and
    // a level time or significance) are kept; the wind-shear loop (011061)
    // never enters because its items are not in the set above.
    levels
        .into_values()
        .filter(|list| list.iter().any(|i| matches!(i.code, 7004 | 7009)) && list.iter().any(|i| matches!(i.code, 4086 | 8042)))
        .map(|items| Level { items })
        .collect()
}

fn level_number(level: &Level<'_>, code: u32) -> Option<f64> {
    level.items.iter().find(|i| i.code == code).and_then(|i| i.value.number())
}

fn sounding_rows(items: &[Item], provenance: &RowProvenance, counters: &mut Counters, rows: &mut Vec<TableRow>) {
    let Some(station) = identifier(items) else {
        counters.subsets_without_identifier += 1;
        return;
    };
    let Some(launch) = report_time(items) else {
        counters.subsets_without_time += 1;
        return;
    };
    let Some((lat0, lon0)) = position(items) else {
        counters.subsets_without_position += 1;
        return;
    };
    let station_height = number(items, 7030).or_else(|| number(items, 7007)).filter(|h| h.is_finite() && *h > -500.0 && *h < 9000.0);
    // The nominal hour: the launch rounded to the synoptic hour it serves.
    let nominal = launch + Duration::minutes(30);
    let nominal = Utc.with_ymd_and_hms(nominal.format("%Y").to_string().parse().unwrap_or(2000),
                                       nominal.format("%m").to_string().parse().unwrap_or(1),
                                       nominal.format("%d").to_string().parse().unwrap_or(1),
                                       nominal.format("%H").to_string().parse().unwrap_or(0), 0, 0).single();
    let mut emit = Emit { provenance, counters, rows };
    for level in levels_of(items) {
        emit.counters.sounding_levels += 1;
        let Some(pressure) = level_number(&level, 7004).filter(|p| p.is_finite() && *p > 0.0) else {
            emit.counters.sounding_levels_without_pressure += 1;
            continue;
        };
        let elapsed = level_number(&level, 4086);
        let when = match elapsed {
            Some(s) if s.is_finite() && (0.0..=36_000.0).contains(&s) => launch + Duration::seconds(s.round() as i64),
            _ => {
                emit.counters.sounding_levels_at_nominal_time += 1;
                launch
            }
        };
        let lat = lat0 + level_number(&level, 5015).unwrap_or(0.0);
        let lon = lon0 + level_number(&level, 6015).unwrap_or(0.0);
        if !(-90.0..=90.0).contains(&lat) {
            emit.counters.values_out_of_range += 1;
            continue;
        }
        let height = match level_number(&level, 10009).filter(|h| h.is_finite()) {
            Some(h) => h,
            None => {
                emit.counters.sounding_heights_isa_stamped += 1;
                isa_altitude_m(pressure)
            }
        };
        let significance = level.items.iter().find(|i| i.code == 8042).and_then(|i| i.value.code()).unwrap_or(0);
        if significance & SOUNDING_SURFACE_BIT != 0 {
            if let Some(ground) = station_height {
                if in_bounds(pressure, GROSS_SURFACE_PRESSURE_PA) {
                    emit.push(&station, lat0, lon0, ground, None, when, VAR_SURFACE_PRESSURE, pressure,
                              ERROR_SURFACE_PRESSURE_PA, MEAS_STATION_PRESSURE, nominal);
                }
            }
        }
        let temperature = level_number(&level, 12101);
        if let Some(t) = temperature {
            if in_bounds(t, GROSS_TEMPERATURE_K) {
                emit.push(&station, lat, lon, height, Some(pressure), when, VAR_TEMPERATURE, t,
                          ERROR_TEMPERATURE_ALOFT_K, MEAS_SONDE_LEVEL, nominal);
            } else {
                emit.counters.values_out_of_range += 1;
            }
        }
        if let Some(td) = level_number(&level, 12103) {
            if !in_bounds(td, GROSS_DEWPOINT_K) {
                emit.counters.values_out_of_range += 1;
            } else if temperature.is_some_and(|t| td > t + 0.05) {
                emit.counters.values_dewpoint_above_temperature += 1;
            } else {
                emit.push(&station, lat, lon, height, Some(pressure), when, VAR_DEWPOINT, td, ERROR_DEWPOINT_ALOFT_K,
                          MEAS_SONDE_LEVEL, nominal);
            }
        }
        if let (Some(direction), Some(speed)) = (level_number(&level, 11001), level_number(&level, 11002)) {
            emit.wind(&station, lat, lon, height, Some(pressure), when, direction, speed, ERROR_WIND_ALOFT_M_S,
                      MEAS_SONDE_LEVEL, nominal);
        }
    }
}

/// The rows of one decoded message, every subset.
pub fn rows_of(message: &Message, provenance: &RowProvenance, counters: &mut Counters) -> Vec<TableRow> {
    let mut rows = Vec::new();
    let kind = kind_of(&message.descriptors);
    for subset in &message.subsets {
        counters.subsets += 1;
        match kind {
            Kind::Synop => {
                counters.subsets_synop += 1;
                surface_rows(subset, false, provenance, counters, &mut rows);
            }
            Kind::Ship => {
                counters.subsets_ship += 1;
                surface_rows(subset, true, provenance, counters, &mut rows);
            }
            Kind::Temp => {
                counters.subsets_temp += 1;
                sounding_rows(subset, provenance, counters, &mut rows);
            }
            Kind::Unhandled => {
                counters.subsets_unhandled_template += 1;
                let key = message.descriptors.iter().map(|d| format!("{d:06}")).collect::<Vec<_>>().join(",");
                *counters.unhandled_templates.entry(key).or_insert(0) += 1;
            }
        }
    }
    rows
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bufr::tests::{message, Encoder};
    use crate::bufr::{decode, Tables};

    fn synop_message() -> Vec<u8> {
        // A bare-element surface template with the WIGOS prefix: identifier,
        // time, position, heights, station pressure and MSLP, the screen
        // block (sensor height, T, Td), the wind block (sensor height,
        // direction, speed).
        let t = Tables::get();
        let descriptors = [301150, 301011, 301012, 301021, 7030, 7031, 10004, 10051, 7032, 12101, 12103, 7032, 11001, 11002];
        let mut e = Encoder::new();
        e.put(0, 4); // series
        e.put(20000, 16); // issuer
        e.put(0, 16); // issue
        e.put_text("72530", 16); // local id, 128 bits
        e.put_number(t.element(4001).unwrap(), 2026.0);
        e.put_number(t.element(4002).unwrap(), 9.0);
        e.put_number(t.element(4003).unwrap(), 6.0);
        e.put_number(t.element(4004).unwrap(), 2.0);
        e.put_number(t.element(4005).unwrap(), 0.0);
        e.put_number(t.element(5001).unwrap(), 41.98);
        e.put_number(t.element(6001).unwrap(), -87.9);
        e.put_number(t.element(7030).unwrap(), 200.5);
        e.put_number(t.element(7031).unwrap(), 201.7);
        e.put_number(t.element(10004).unwrap(), 98910.0);
        e.put_number(t.element(10051).unwrap(), 101300.0);
        e.put_number(t.element(7032).unwrap(), 2.0);
        e.put_number(t.element(12101).unwrap(), 290.15);
        e.put_number(t.element(12103).unwrap(), 285.05);
        e.put_number(t.element(7032).unwrap(), 10.0);
        e.put_number(t.element(11001).unwrap(), 270.0);
        e.put_number(t.element(11002).unwrap(), 5.0);
        message(&descriptors, &e.finish(), 1, false)
    }

    #[test]
    fn a_synop_subset_becomes_station_pressure_screen_values_and_a_10_m_wind() {
        let m = decode(&synop_message(), "synop").unwrap();
        assert_eq!(kind_of(&m.descriptors), Kind::Synop);
        let mut counters = Counters::default();
        let rows = rows_of(&m, &RowProvenance::default(), &mut counters);
        let by: BTreeMap<String, &TableRow> = rows.iter().map(|r| (r.variable.clone(), r)).collect();
        assert_eq!(rows.len(), 5, "{counters:?}");
        let p = by[VAR_SURFACE_PRESSURE];
        assert_eq!(p.station_id, "0-20000-0-72530");
        assert!((p.value - 98910.0).abs() < 1e-6 && p.elevation_m == 201.7);
        assert_eq!(p.provenance.measurement, MEAS_STATION_PRESSURE);
        assert_eq!(p.valid_time, Utc.with_ymd_and_hms(2026, 9, 6, 2, 0, 0).unwrap());
        assert!((p.latitude_deg - 41.98).abs() < 1e-9 && (p.longitude_deg + 87.9).abs() < 1e-9);
        let t = by[VAR_TEMPERATURE];
        assert!((t.value - 290.15).abs() < 1e-9 && t.elevation_m == 200.5 && t.level_pa.is_none());
        assert_eq!(t.provenance.measurement, MEAS_SCREEN_TEMPERATURE_2M);
        assert!((by[VAR_DEWPOINT].value - 285.05).abs() < 1e-9);
        let u = by[VAR_WIND_U];
        assert!((u.value - 5.0).abs() < 1e-9 && by[VAR_WIND_V].value.abs() < 1e-9);
        assert_eq!(u.provenance.measurement, MEAS_ANEMOMETER_WIND_10M);
        assert_eq!(counters.subsets_synop, 1);
    }

    #[test]
    fn a_temp_subset_writes_every_level_at_its_own_time_and_the_surface_pressure() {
        let t = Tables::get();
        // 301001 (block, station), 301011, 301013, 301021, 007030, then two
        // levels: 102000 031001 [ 004086 008042 007004 010009 005015 006015
        // 012101 012103 011001 011002 ] as a delayed replication of 303054.
        let descriptors = [301001, 301011, 301013, 301021, 7030, 101000, 31001, 303054];
        let mut e = Encoder::new();
        e.put_number(t.element(1001).unwrap(), 72.0);
        e.put_number(t.element(1002).unwrap(), 201.0);
        e.put_number(t.element(4001).unwrap(), 2026.0);
        e.put_number(t.element(4002).unwrap(), 9.0);
        e.put_number(t.element(4003).unwrap(), 5.0);
        e.put_number(t.element(4004).unwrap(), 23.0);
        e.put_number(t.element(4005).unwrap(), 31.0);
        e.put_number(t.element(4006).unwrap(), 0.0);
        e.put_number(t.element(5001).unwrap(), 30.0);
        e.put_number(t.element(6001).unwrap(), 20.0);
        e.put_number(t.element(7030).unwrap(), 50.0);
        e.put(2, 8); // two levels
        // surface level
        e.put_number(t.element(4086).unwrap(), 0.0);
        e.put(SOUNDING_SURFACE_BIT, 18);
        e.put_number(t.element(7004).unwrap(), 100500.0);
        e.put_number(t.element(10009).unwrap(), 50.0);
        e.put_number(t.element(5015).unwrap(), 0.0);
        e.put_number(t.element(6015).unwrap(), 0.0);
        e.put_number(t.element(12101).unwrap(), 300.0);
        e.put_number(t.element(12103).unwrap(), 295.0);
        e.put_number(t.element(11001).unwrap(), 180.0);
        e.put_number(t.element(11002).unwrap(), 3.0);
        // 500 hPa, 1500 s later, displaced, no dewpoint, no height (ISA stamp)
        e.put_number(t.element(4086).unwrap(), 1500.0);
        e.put(0, 18);
        e.put_number(t.element(7004).unwrap(), 50000.0);
        e.put_missing(17);
        e.put_number(t.element(5015).unwrap(), 0.1);
        e.put_number(t.element(6015).unwrap(), -0.2);
        e.put_number(t.element(12101).unwrap(), 260.0);
        e.put_missing(16);
        e.put_number(t.element(11001).unwrap(), 270.0);
        e.put_number(t.element(11002).unwrap(), 20.0);
        let m = decode(&message(&descriptors, &e.finish(), 1, false), "temp").unwrap();
        // the template has no 309052 marker: read as a sounding by its level loop
        let mut counters = Counters::default();
        let mut rows = Vec::new();
        sounding_rows(&m.subsets[0], &RowProvenance::default(), &mut counters, &mut rows);
        assert_eq!(counters.sounding_levels, 2, "{counters:?}");
        assert_eq!(counters.sounding_heights_isa_stamped, 1);
        // surface: ps + T + Td + u + v; 500 hPa: T + u + v
        assert_eq!(rows.len(), 8, "{rows:?}");
        let ps = rows.iter().find(|r| r.variable == VAR_SURFACE_PRESSURE).unwrap();
        assert_eq!(ps.station_id, "72201");
        assert!(ps.level_pa.is_none() && ps.elevation_m == 50.0 && (ps.value - 100500.0).abs() < 1e-6);
        assert_eq!(ps.provenance.nominal_time, Some(Utc.with_ymd_and_hms(2026, 9, 6, 0, 0, 0).unwrap()));
        let upper = rows.iter().find(|r| r.level_pa == Some(50000.0) && r.variable == VAR_TEMPERATURE).unwrap();
        assert_eq!(upper.valid_time, Utc.with_ymd_and_hms(2026, 9, 5, 23, 56, 0).unwrap());
        assert!((upper.latitude_deg - 30.1).abs() < 1e-9 && (upper.longitude_deg - 19.8).abs() < 1e-9);
        assert!((upper.elevation_m - isa_altitude_m(50000.0)).abs() < 1e-9);
        assert_eq!(upper.provenance.measurement, MEAS_SONDE_LEVEL);
        assert_eq!(upper.error, ERROR_TEMPERATURE_ALOFT_K);
        let wind = rows.iter().find(|r| r.level_pa == Some(50000.0) && r.variable == VAR_WIND_U).unwrap();
        assert!((wind.value - 20.0).abs() < 1e-9);
    }

    #[test]
    fn the_screens_count_what_they_refuse() {
        let t = Tables::get();
        let descriptors = [301001, 301011, 301012, 301021, 7030, 12101, 12103, 7032, 11001, 11002];
        let mut e = Encoder::new();
        e.put_number(t.element(1001).unwrap(), 10.0);
        e.put_number(t.element(1002).unwrap(), 5.0);
        e.put_number(t.element(4001).unwrap(), 2026.0);
        e.put_number(t.element(4002).unwrap(), 9.0);
        e.put_number(t.element(4003).unwrap(), 6.0);
        e.put_number(t.element(4004).unwrap(), 2.0);
        e.put_number(t.element(4005).unwrap(), 0.0);
        e.put_number(t.element(5001).unwrap(), 10.0);
        e.put_number(t.element(6001).unwrap(), 10.0);
        e.put_number(t.element(7030).unwrap(), 10.0);
        e.put_number(t.element(12101).unwrap(), 280.0);
        e.put_number(t.element(12103).unwrap(), 281.0); // above the temperature
        e.put_number(t.element(7032).unwrap(), 4.0); // a 4 m anemometer
        e.put_number(t.element(11001).unwrap(), 90.0);
        e.put_number(t.element(11002).unwrap(), 2.0);
        let m = decode(&message(&descriptors, &e.finish(), 1, false), "screens").unwrap();
        let mut counters = Counters::default();
        let rows = rows_of(&m, &RowProvenance::default(), &mut counters);
        assert_eq!(rows.len(), 1, "{rows:?}");
        assert_eq!(rows[0].variable, VAR_TEMPERATURE);
        assert_eq!(counters.values_dewpoint_above_temperature, 1);
        assert_eq!(counters.winds_at_other_sensor_heights, 1);
        assert_eq!(kind_of(&[315008]), Kind::Unhandled);
        assert_eq!(kind_of(&[301150, 307096]), Kind::Synop);
        assert_eq!(kind_of(&[308009]), Kind::Ship);
        assert_eq!(kind_of(&[301150, 309052]), Kind::Temp);
    }
}
