//! JMA polar-coordinate radar GRIB2 tars (Japan), every station, with the
//! two things the shared decode does not carry.
//!
//! The gates, the geometry and the tar/GRIB2/run-length walk are BowEcho's
//! `nexrad_io::jma`, reused as it is. This module adds what that decode
//! leaves out and an assimilating consumer cannot do without:
//!
//! 1. **"No Echo" is clear air, not 0 dBZ.** JMA's format description for
//!    this product (GRIB2 Ver. 2.00, table 3) defines level 0 of the level
//!    table as outside the observed range or missing, and level 1 as
//!    "No Echo". The shared decode maps level 0 to `NaN` and level 1 to its
//!    table value, 0.00 dBZ, so every clear-air gate of the network arrives
//!    as a weak echo (on a quiet afternoon that is 99.9 % of all gates).
//!    Here level 1 of a reflectivity sweep becomes [`censor::UNDETECT`] and
//!    level 0 [`censor::NODATA`]. The velocity level table has no such level
//!    (its level 1 is a real 0.0 m/s); velocity "no echo" is level 0.
//! 2. **The Nyquist interval.** The file states no interval. It does state
//!    the transmitted frequency (PDT 4.51022 octets 33-36, kHz), the number
//!    of PRFs (octet 44) and the representative PRFs (octets 45-50, 0.1 Hz),
//!    and every Doppler cut is dual-PRF with a 5:4 or 6:5 pair. The extended
//!    interval is `lambda * f_high * f_low / (4 (f_high - f_low))`. On the
//!    2019-10-12 09Z Hagibis volume (Akita and Sendai Doppler cuts, typhoon
//!    winds) 25-80 % of gates exceed the single-PRF interval and at most
//!    0.005 % exceed the derived extended one, which is the evidence the
//!    stored velocities are dual-PRF unfolded to it. Cuts that declare no PRF
//!    (the long-range lowest cut) get no interval and their velocity is
//!    refused downstream by name; their reflectivity is unaffected.
//!
//! The level table, PRFs and frequency are read by a header-only walk of the
//! same GRIB2 members, aligned cut-for-cut with the shared decode and checked
//! (elevation, gate count, radial count) so a misalignment is an error, never
//! a quietly wrong label.

use super::convert::{FirstGate, from_radar_core};
use super::{SPEED_OF_LIGHT_MS, census_of};
use crate::censor;
use crate::error::{OdimError, Result};
use crate::volume::{Nyquist, PolarVolume};

const TAR_BLOCK: usize = 512;
const PRODUCT_TEMPLATE_RADAR_ELEVATION: u16 = 51022;

/// One station of a JMA tar, by its header alone.
#[derive(Debug, Clone, serde::Serialize)]
pub struct Station {
    pub id: String,
    pub number: u16,
    pub latitude_deg: f64,
    pub longitude_deg: f64,
    pub elevation_m: Option<f32>,
}

/// Every station a JMA tar holds, without decoding a gate.
pub fn stations(bytes: &[u8], name: &str) -> Result<Vec<Station>> {
    let headers = nexrad_io::jma::jma_tar_station_headers(bytes)
        .map_err(|error| OdimError::format(name.to_string(), error))?;
    Ok(headers
        .into_iter()
        .map(|h| Station {
            id: h.id,
            number: h.number,
            latitude_deg: h.latitude_deg,
            longitude_deg: h.longitude_deg,
            elevation_m: h.elevation_m,
        })
        .collect())
}

/// Decode a JMA tar into one [`PolarVolume`] per station (or the one `site`
/// names: a JMA id such as `MURO`, or the station number `47899` /
/// `RS47899`).
pub fn read_jma_tar(bytes: &[u8], name: &str, site: Option<&str>) -> Result<Vec<PolarVolume>> {
    let decoded = nexrad_io::jma::decode_jma_tar_volumes(bytes, site)
        .map_err(|error| OdimError::format(name.to_string(), error))?;
    let metas = sweep_metadata(bytes, name, site)?;
    let mut volumes = Vec::with_capacity(decoded.len());
    for volume in decoded {
        let station = volume.site.id.clone();
        let context = format!("{name} station {station}");
        let meta = metas
            .iter()
            .find(|(id, _)| *id == station)
            .map(|(_, meta)| meta)
            .ok_or_else(|| {
                OdimError::format(context.clone(), "the header walk found no sweeps for it")
            })?;
        if meta.len() != volume.cuts.len() {
            return Err(OdimError::format(
                context,
                format!(
                    "the decode has {} cuts and the header walk {} sweeps",
                    volume.cuts.len(),
                    meta.len()
                ),
            ));
        }
        for (index, (cut, meta)) in volume.cuts.iter().zip(meta).enumerate() {
            let gates = cut
                .moments
                .values()
                .next()
                .map(|grid| grid.gate_range.gate_count)
                .unwrap_or(0);
            if cut.elevation_deg.to_bits() != meta.elevation_deg.to_bits()
                || gates != meta.gates
                || cut.radials.len() != meta.rays
                || cut.moments.len() != 1
            {
                return Err(OdimError::format(
                    context,
                    format!(
                        "cut {} ({} deg, {} gates x {} rays) does not line up with header \
                         sweep ({} deg, {} gates x {} rays)",
                        index + 1,
                        cut.elevation_deg,
                        gates,
                        cut.radials.len(),
                        meta.elevation_deg,
                        meta.gates,
                        meta.rays
                    ),
                ));
            }
        }

        let mut polar = from_radar_core(&volume, name, "JMA-GRIB2", FirstGate::Start)?;
        if polar.sweeps.len() != meta.len() {
            return Err(OdimError::format(
                context,
                format!(
                    "{} sweeps converted from {} cuts",
                    polar.sweeps.len(),
                    meta.len()
                ),
            ));
        }
        let number = polar.source.place.take();
        polar.source.rad = number.clone();
        polar.source.raw = format!(
            "JMA:{station}{}",
            number.map(|n| format!(",{n}")).unwrap_or_default()
        );
        for (sweep, meta) in polar.sweeps.iter_mut().zip(meta) {
            sweep.nyquist = meta.nyquist();
            if meta.is_reflectivity() {
                let no_echo = meta.no_echo.ok_or_else(|| {
                    OdimError::format(
                        format!("{context} sweep {}", sweep.index),
                        "the reflectivity level table has no level 1",
                    )
                })?;
                if meta.no_echo_ambiguous {
                    return Err(OdimError::format(
                        format!("{context} sweep {}", sweep.index),
                        format!(
                            "another level of the table carries the No Echo value {no_echo}, \
                             so No Echo cannot be told from an echo"
                        ),
                    ));
                }
                let no_echo = f64::from(no_echo);
                for moment in &mut sweep.moments {
                    for (value, code) in moment.values.iter_mut().zip(moment.censor.iter_mut()) {
                        if *code == censor::MEASURED && *value == no_echo {
                            *value = f64::NAN;
                            *code = censor::UNDETECT;
                        }
                    }
                    moment.census = census_of(&moment.censor);
                }
            }
        }
        if polar.system.wavelength_cm.is_none() {
            polar.system.wavelength_cm = meta.iter().find_map(SweepMeta::wavelength_cm);
        }
        volumes.push(polar);
    }
    Ok(volumes)
}

/// What the header walk records for one sweep (one GRIB2 data section).
#[derive(Debug, Clone)]
struct SweepMeta {
    elevation_deg: f32,
    gates: usize,
    rays: usize,
    category: u8,
    number: u8,
    frequency_khz: Option<u32>,
    prf_count: u8,
    prfs_hz: Vec<f64>,
    /// Level 1's value, computed exactly as the shared decode computes it.
    no_echo: Option<f32>,
    /// Another level of the table has the same value as level 1.
    no_echo_ambiguous: bool,
}

impl SweepMeta {
    fn is_reflectivity(&self) -> bool {
        (self.category, self.number) == (15, 1)
    }

    fn wavelength_cm(&self) -> Option<f64> {
        self.frequency_khz
            .filter(|khz| *khz > 0)
            .map(|khz| SPEED_OF_LIGHT_MS / (f64::from(khz) * 1.0e3) * 100.0)
    }

    fn nyquist(&self) -> Nyquist {
        let wavelength = self.wavelength_cm();
        let mut distinct: Vec<f64> = Vec::new();
        for prf in &self.prfs_hz {
            if !distinct.iter().any(|seen| (seen - prf).abs() < 1e-6) {
                distinct.push(*prf);
            }
        }
        match (self.prf_count, distinct.as_slice()) {
            (1 | 2, [single]) => Nyquist::establish(None, Some(*single), Some(*single), wavelength),
            (2, [a, b]) => Nyquist::dual_prf_extended(a.max(*b), a.min(*b), wavelength),
            _ => Nyquist::establish(None, None, None, wavelength),
        }
    }
}

/// Header-only walk: per station (archive order, merged by id), its sweeps in
/// the order the shared decode emits cuts.
fn sweep_metadata(
    bytes: &[u8],
    name: &str,
    site: Option<&str>,
) -> Result<Vec<(String, Vec<SweepMeta>)>> {
    let mut stations: Vec<(String, Vec<SweepMeta>)> = Vec::new();
    for (member, data) in tar_members(bytes, name)? {
        let base = member.rsplit('/').next().unwrap_or(&member);
        if !(base.starts_with("Z__C_RJTD_")
            && base.contains("_RDR_JMAGPV")
            && base.to_ascii_lowercase().ends_with(".bin"))
        {
            continue;
        }
        // A member the shared decode skips (corrupt) is skipped here too; if
        // only one of the two walks drops it the alignment check refuses.
        let Ok((station, number, sweeps)) = member_sweeps(data, &member) else {
            continue;
        };
        if let Some(filter) = site
            && !station_matches(&station, number, filter)
        {
            continue;
        }
        match stations.iter_mut().find(|(id, _)| *id == station) {
            Some((_, list)) => list.extend(sweeps),
            None => stations.push((station, sweeps)),
        }
    }
    for (_, sweeps) in &mut stations {
        // The shared decode sorts cuts lowest beam first, stably.
        sweeps.sort_by(|a, b| a.elevation_deg.total_cmp(&b.elevation_deg));
    }
    Ok(stations)
}

fn station_matches(id: &str, number: u16, filter: &str) -> bool {
    let filter = filter.trim();
    if filter.eq_ignore_ascii_case(id) {
        return true;
    }
    let number = number.to_string();
    filter == number
        || filter
            .strip_prefix("RS")
            .or_else(|| filter.strip_prefix("rs"))
            .is_some_and(|rest| rest == number)
}

fn member_sweeps(data: &[u8], member: &str) -> Result<(String, u16, Vec<SweepMeta>)> {
    let err = |detail: String| OdimError::format(member.to_string(), detail);
    if data.len() < 16 || &data[0..4] != b"GRIB" {
        return Err(err("no GRIB indicator".into()));
    }
    let total = usize::try_from(be_u64(data, 8).ok_or_else(|| err("short indicator".into()))?)
        .map_err(|_| err("message length overflow".into()))?;
    let message = data
        .get(..total)
        .ok_or_else(|| err(format!("message of {total} bytes overruns the member")))?;
    let mut position = 16usize;
    let mut grid: Option<(usize, usize)> = None;
    let mut product: Option<&[u8]> = None;
    let mut representation: Option<&[u8]> = None;
    let mut station: Option<(String, u16)> = None;
    let mut sweeps = Vec::new();
    while position + 4 <= message.len() {
        if &message[position..position + 4] == b"7777" {
            break;
        }
        let length =
            be_u32(message, position).ok_or_else(|| err("section length".into()))? as usize;
        let section = message
            .get(position..position + length)
            .filter(|s| s.len() >= 5)
            .ok_or_else(|| err(format!("section at {position} overruns the message")))?;
        match section[4] {
            3 => {
                let gates = be_u32(section, 14).ok_or_else(|| err("grid gates".into()))? as usize;
                let rays = be_u32(section, 18).ok_or_else(|| err("grid rays".into()))? as usize;
                grid = Some((gates, rays));
            }
            4 => product = Some(section),
            5 => representation = Some(section),
            7 => {
                let (gates, rays) = grid.ok_or_else(|| err("data before grid".into()))?;
                let product = product
                    .take()
                    .ok_or_else(|| err("data before product".into()))?;
                let representation = representation
                    .take()
                    .ok_or_else(|| err("data before representation".into()))?;
                if section_u16(product, 7)? != PRODUCT_TEMPLATE_RADAR_ELEVATION {
                    return Err(err("not product template 4.51022".into()));
                }
                if product.len() < 60 {
                    return Err(err("product section too short".into()));
                }
                let id = String::from_utf8_lossy(&product[24..28])
                    .trim_matches(char::from(0))
                    .trim()
                    .to_string();
                let number = section_u16(product, 28)?;
                station.get_or_insert((id, number));
                // Exactly the shared decode's expression, so the f32 values
                // compare bit-for-bit.
                let elevation_deg = signed_magnitude_i16(section_u16(product, 41)?)
                    .map(|value| f32::from(value) / 100.0)
                    .unwrap_or(0.0);
                let frequency = be_u32(product, 32).ok_or_else(|| err("frequency".into()))?;
                let prfs_hz = [44usize, 46, 48]
                    .iter()
                    .filter_map(|offset| be_u16(product, *offset))
                    .filter(|raw| *raw != u16::MAX && *raw != 0)
                    .map(|raw| f64::from(raw) / 10.0)
                    .collect();
                let (no_echo, no_echo_ambiguous) = level_one(representation)?;
                sweeps.push(SweepMeta {
                    elevation_deg,
                    gates,
                    rays,
                    category: product[9],
                    number: product[10],
                    frequency_khz: (frequency != u32::MAX).then_some(frequency),
                    prf_count: product[43],
                    prfs_hz,
                    no_echo,
                    no_echo_ambiguous,
                });
            }
            _ => {}
        }
        position += length;
    }
    let (id, number) = station.ok_or_else(|| err("no radar sweeps".into()))?;
    Ok((id, number, sweeps))
}

/// Level 1 of a template 5.200 level table, as the shared decode computes
/// it, and whether any other level carries the same value.
fn level_one(section: &[u8]) -> Result<(Option<f32>, bool)> {
    let err =
        |detail: &str| OdimError::format("JMA data representation".to_string(), detail.to_string());
    let levels = usize::from(be_u16(section, 14).ok_or_else(|| err("level count"))?);
    let scale = *section.get(16).ok_or_else(|| err("decimal scale"))?;
    let factor = 10_f32.powi(-i32::from(scale));
    let mut values = Vec::with_capacity(levels);
    for index in 0..levels {
        let raw = be_u16(section, 17 + index * 2).ok_or_else(|| err("level table"))?;
        values.push(signed_magnitude_i16(raw).map(|value| f32::from(value) * factor));
    }
    let Some(first) = values.first().copied().flatten() else {
        return Ok((None, false));
    };
    let ambiguous = values[1..]
        .iter()
        .flatten()
        .any(|value| value.to_bits() == first.to_bits());
    Ok((Some(first), ambiguous))
}

fn tar_members<'a>(bytes: &'a [u8], name: &str) -> Result<Vec<(String, &'a [u8])>> {
    let mut members = Vec::new();
    let mut position = 0usize;
    while position + TAR_BLOCK <= bytes.len() {
        let header = &bytes[position..position + TAR_BLOCK];
        if header.iter().all(|byte| *byte == 0) {
            break;
        }
        let end = header[..100].iter().position(|b| *b == 0).unwrap_or(100);
        let member = String::from_utf8_lossy(&header[..end]).trim().to_string();
        let size_text = String::from_utf8_lossy(&header[124..136]);
        let size_text = size_text.trim_matches(|c: char| c == '\0' || c == ' ');
        let size = if size_text.is_empty() {
            0
        } else {
            usize::from_str_radix(size_text, 8).map_err(|_| {
                OdimError::format(name.to_string(), format!("tar member {member} has no size"))
            })?
        };
        let start = position + TAR_BLOCK;
        let data = bytes.get(start..start + size).ok_or_else(|| {
            OdimError::format(
                name.to_string(),
                format!("tar member {member} overruns the archive"),
            )
        })?;
        if matches!(header[156], b'0' | 0) {
            members.push((member, data));
        }
        position = start + size.div_ceil(TAR_BLOCK) * TAR_BLOCK;
    }
    Ok(members)
}

fn signed_magnitude_i16(raw: u16) -> Option<i16> {
    if raw == u16::MAX {
        return None;
    }
    let magnitude = (raw & 0x7fff) as i16;
    Some(if raw & 0x8000 != 0 {
        -magnitude
    } else {
        magnitude
    })
}

fn be_u16(bytes: &[u8], at: usize) -> Option<u16> {
    bytes
        .get(at..at + 2)
        .map(|b| u16::from_be_bytes([b[0], b[1]]))
}

fn section_u16(bytes: &[u8], at: usize) -> Result<u16> {
    be_u16(bytes, at)
        .ok_or_else(|| OdimError::format("JMA product".to_string(), format!("short at {at}")))
}

fn be_u32(bytes: &[u8], at: usize) -> Option<u32> {
    bytes
        .get(at..at + 4)
        .map(|b| u32::from_be_bytes([b[0], b[1], b[2], b[3]]))
}

fn be_u64(bytes: &[u8], at: usize) -> Option<u64> {
    bytes.get(at..at + 8).map(|b| {
        let mut word = [0u8; 8];
        word.copy_from_slice(b);
        u64::from_be_bytes(word)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_dual_prf_interval_is_the_extended_one() {
        // Akita, 2019-10-12: 5.365 GHz, 940/752 Hz (5:4). Single-PRF on the
        // high PRF is 13.1 m/s; the extended interval is four times that.
        let meta = SweepMeta {
            elevation_deg: 2.8,
            gates: 300,
            rays: 512,
            category: 15,
            number: 2,
            frequency_khz: Some(5_365_000),
            prf_count: 2,
            prfs_hz: vec![940.0, 752.0],
            no_echo: Some(0.0),
            no_echo_ambiguous: false,
        };
        let nyquist = meta.nyquist();
        assert_eq!(nyquist.source, crate::volume::NyquistSource::DerivedDualPrf);
        let interval = nyquist.interval_ms.expect("derived");
        assert!((interval - 52.53).abs() < 0.02, "{interval}");
        assert!(nyquist.dual_prf);
    }

    #[test]
    fn a_cut_with_no_prf_gets_no_interval() {
        let meta = SweepMeta {
            elevation_deg: 0.5,
            gates: 500,
            rays: 512,
            category: 15,
            number: 2,
            frequency_khz: Some(5_365_000),
            prf_count: 255,
            prfs_hz: Vec::new(),
            no_echo: Some(0.0),
            no_echo_ambiguous: false,
        };
        assert!(!meta.nyquist().is_usable());
    }

    #[test]
    fn level_one_is_read_as_the_shared_decode_reads_it() {
        // 5.200 section: length(4) number(1) points(4) template(2) bits(1)
        // V(2) M(2) scale(1) then M levels.
        let mut section = vec![0u8, 0, 0, 23, 5, 0, 0, 0, 1, 0, 200, 8, 0, 3, 0, 3, 2];
        for level in [0u16, 16, 48] {
            section.extend_from_slice(&level.to_be_bytes());
        }
        let (first, ambiguous) = level_one(&section).expect("parses");
        assert_eq!(first, Some(0.0));
        assert!(!ambiguous);
    }
}
