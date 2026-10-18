//! BowEcho's shared radar model into this crate's [`PolarVolume`].
//!
//! BowEcho's `nexrad_io` already reads CfRadial 1.x, DORADE and the JMA tar
//! into `radar_core::RadarVolume`; this is the one conversion from that model
//! into the one this crate packs, so those readers are reused as they are.
//!
//! Two things the shared model leaves implicit are made explicit here:
//!
//! - **Why a gate is empty.** A float plane's `NaN` and a raw plane's
//!   `nodata` / `range_folded` value all become [`censor::NODATA`]: the
//!   source did not say the radar looked and found nothing, so nothing here
//!   claims it did. Range folding is folded into `NODATA` rather than minted
//!   as code 2, which the `v3` pack reserves. Readers that *do* know a gate is
//!   clear air (JMA's "No Echo" level) say so after this conversion.
//! - **Where the first gate is.** `first_gate_m` is the gate centre for
//!   CfRadial and DORADE ([`FirstGate::Centre`]) and the start of the first
//!   bin for the JMA grid ([`FirstGate::Start`]); [`crate::Sweep`] stores the
//!   start, so the half bin is taken off here, once.

use chrono::Duration;
use radar_core::{MomentGrid, MomentStorage, MomentType, RadarVolume, Radial};

use super::{SPEED_OF_LIGHT_MS, moment};
use crate::censor;
use crate::error::{OdimError, Result};
use crate::volume::{
    AzimuthSource, Nyquist, NyquistSource, PolarVolume, Site, Source, Sweep, SystemNotes,
};

/// What `GateRange::first_gate_m` means for the format being converted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FirstGate {
    /// The range to the centre of the first gate (CfRadial, DORADE, NEXRAD).
    Centre,
    /// The range to the start of the first gate (the JMA polar grid).
    Start,
}

/// The ODIM spelling for a shared-model moment, so the pack's canonical
/// `REF`/`VEL` claim works for every converted format.
pub fn quantity_name(moment: &MomentType) -> String {
    match moment {
        MomentType::Reflectivity => "DBZH".to_string(),
        MomentType::Velocity => "VRADH".to_string(),
        MomentType::SpectrumWidth => "WRADH".to_string(),
        MomentType::DifferentialReflectivity => "ZDR".to_string(),
        MomentType::CorrelationCoefficient => "RHOHV".to_string(),
        MomentType::DifferentialPhase => "PHIDP".to_string(),
        MomentType::SpecificDifferentialPhase => "KDP".to_string(),
        MomentType::Unknown(name) => match name.trim().to_ascii_uppercase().as_str() {
            // Spellings the shared model leaves unknown but a consumer must
            // not: Gematronik/Leonardo CfRadial (IDEAM Bogota) writes its
            // horizontal velocity as VELH, which would otherwise never claim
            // the pack's VEL token and so never be assimilated.
            "VELH" | "VEL_H" | "VELOCITY" => "VRADH".to_string(),
            "VELV" | "VEL_V" => "VRADV".to_string(),
            "WIDTHH" | "WIDTH_H" => "WRADH".to_string(),
            "WIDTHV" | "WIDTH_V" => "WRADV".to_string(),
            "UH" => "TH".to_string(),
            "UV" => "TV".to_string(),
            _ => name.clone(),
        },
    }
}

/// Convert one decoded volume.
///
/// `name` is the file name for error messages; `label` names the format in
/// the volume's provenance (`conventions`, `source.raw`).
pub fn from_radar_core(
    volume: &RadarVolume,
    name: &str,
    label: &str,
    first_gate: FirstGate,
) -> Result<PolarVolume> {
    let latitude_deg = volume
        .site
        .latitude_deg
        .ok_or_else(|| OdimError::format(name.to_string(), "the file declares no site latitude"))?;
    let longitude_deg = volume.site.longitude_deg.ok_or_else(|| {
        OdimError::format(name.to_string(), "the file declares no site longitude")
    })?;
    // An antenna with no altitude cannot place a single gate in the vertical;
    // refusing is better than assimilating every beam from sea level.
    let height_m = volume.site.elevation_m.ok_or_else(|| {
        OdimError::format(
            name.to_string(),
            "the file declares no antenna altitude, so no gate can be placed in the vertical",
        )
    })?;

    let mut sweeps = Vec::new();
    for (cut_number, cut) in volume.cuts.iter().enumerate() {
        let nrays = cut.radials.len();
        if nrays == 0 || cut.moments.is_empty() {
            continue;
        }
        // A cut whose moments sit on different range axes (NEXRAD's long
        // reflectivity and short velocity) becomes one sweep per axis: a
        // `Sweep` states its range geometry once for all of its moments.
        let mut axes: Vec<((i32, i32), Vec<(&MomentType, &MomentGrid)>)> = Vec::new();
        for (kind, grid) in &cut.moments {
            let key = (grid.gate_range.first_gate_m, grid.gate_range.gate_spacing_m);
            match axes.iter_mut().find(|(axis, _)| *axis == key) {
                Some((_, members)) => members.push((kind, grid)),
                None => axes.push((key, vec![(kind, grid)])),
            }
        }
        for ((first_gate_m, spacing_m), members) in axes {
            if spacing_m <= 0 {
                return Err(OdimError::format(
                    format!("{name} cut {}", cut_number + 1),
                    format!("gate spacing {spacing_m} m"),
                ));
            }
            let spacing = f64::from(spacing_m);
            let range_start_m = match first_gate {
                FirstGate::Centre => f64::from(first_gate_m) - spacing / 2.0,
                FirstGate::Start => f64::from(first_gate_m),
            };
            let mut moments = Vec::new();
            let mut nbins = 0usize;
            for (kind, grid) in members {
                let quantity = quantity_name(kind);
                let (values, codes, bins) = plane(grid, nrays, name, cut_number)?;
                nbins = nbins.max(bins);
                moments.push(moment(
                    &quantity,
                    format!("{label}/cut{}/{quantity}", cut_number + 1),
                    nrays,
                    bins,
                    values,
                    codes,
                ));
            }
            let (start_time, end_time) = cut_times(volume, &cut.radials);
            sweeps.push(Sweep {
                index: sweeps.len() + 1,
                path: format!("{label}/cut{}", cut_number + 1),
                elevation_deg: f64::from(cut.elevation_deg),
                nrays,
                nbins,
                range_scale_m: spacing,
                range_start_m,
                a1gate: None,
                start_time,
                end_time,
                nyquist: sweep_nyquist(&cut.radials),
                azimuth_deg: cut
                    .radials
                    .iter()
                    .map(|r| f64::from(r.azimuth_deg))
                    .collect(),
                azimuth_source: AzimuthSource::Recorded,
                ray_elevation_deg: Some(
                    cut.radials
                        .iter()
                        .map(|r| f64::from(r.elevation_deg))
                        .collect(),
                ),
                ray_prf_hz: None,
                moments,
            });
        }
    }
    if sweeps.is_empty() {
        return Err(OdimError::format(
            name.to_string(),
            "the file decoded to no sweep with both radials and moments",
        ));
    }

    let id = volume.site.id.trim().to_string();
    let wavelength_cm = volume
        .metadata
        .radar_frequency_mhz
        .filter(|mhz| *mhz > 0)
        .map(|mhz| SPEED_OF_LIGHT_MS / (f64::from(mhz) * 1.0e6) * 100.0);
    Ok(PolarVolume {
        object: "PVOL".to_string(),
        conventions: Some(label.to_string()),
        version: volume.metadata.archive_version.clone(),
        source: Source {
            raw: format!("{label}:{id}"),
            nod: (!id.is_empty()).then(|| id.clone()),
            wmo: None,
            wigos: None,
            place: volume.site.name.clone(),
            rad: None,
        },
        nominal_time: Some(volume.volume_time),
        site: Site {
            latitude_deg: f64::from(latitude_deg),
            longitude_deg: f64::from(longitude_deg),
            height_m: f64::from(height_m),
        },
        system: SystemNotes {
            wavelength_cm,
            beamwidth_deg: volume.metadata.beam_width_h_deg.map(f64::from),
            system: None,
            software: None,
            sw_version: None,
        },
        sweeps,
    })
}

/// One moment grid as a full `[ray][bin]` plane with its censor codes.
fn plane(
    grid: &MomentGrid,
    nrays: usize,
    name: &str,
    cut_number: usize,
) -> Result<(Vec<f64>, Vec<u8>, usize)> {
    let bins = grid.gate_range.gate_count;
    let total = nrays
        .checked_mul(bins)
        .ok_or_else(|| OdimError::format(name.to_string(), "gate count overflow"))?;
    let mut values = vec![f64::NAN; total];
    // A ray the grid carries no row for was not collected for this moment.
    let mut codes = vec![censor::NOT_COLLECTED; total];
    let stored = grid.storage.len();
    let identity: Vec<usize>;
    let rows: &[usize] = if grid.radial_indices.is_empty() && stored == total {
        identity = (0..nrays).collect();
        &identity
    } else {
        &grid.radial_indices
    };
    if rows.len().checked_mul(bins) != Some(stored) {
        return Err(OdimError::format(
            format!("{name} cut {}", cut_number + 1),
            format!(
                "{:?} stores {stored} gates for {} rows of {bins}",
                grid.moment,
                rows.len()
            ),
        ));
    }
    for (row, &ray) in rows.iter().enumerate() {
        if ray >= nrays {
            return Err(OdimError::format(
                format!("{name} cut {}", cut_number + 1),
                format!("{:?} row {row} names ray {ray} of {nrays}", grid.moment),
            ));
        }
        for bin in 0..bins {
            let at = row * bins + bin;
            let out = ray * bins + bin;
            let value = match &grid.storage {
                MomentStorage::F32(plane) => {
                    let value = plane[at];
                    value.is_finite().then_some(f64::from(value))
                }
                MomentStorage::U8(plane) => raw_value(grid, u16::from(plane[at])),
                MomentStorage::U16(plane) => raw_value(grid, plane[at]),
            };
            match value {
                Some(value) => {
                    values[out] = value;
                    codes[out] = censor::MEASURED;
                }
                None => {
                    values[out] = f64::NAN;
                    codes[out] = censor::NODATA;
                }
            }
        }
    }
    Ok((values, codes, bins))
}

fn raw_value(grid: &MomentGrid, raw: u16) -> Option<f64> {
    if grid.nodata == Some(raw) || grid.range_folded == Some(raw) {
        return None;
    }
    let value = (f64::from(raw) - f64::from(grid.offset)) / f64::from(grid.scale);
    value.is_finite().then_some(value)
}

/// The sweep's Nyquist interval from its radials.
///
/// One interval for the sweep, declared by the file. Radials that disagree
/// by more than half a percent (a PRF change inside the cut, or a writer that
/// stored each ray's single-PRF interval of a dual-PRF scan) leave the sweep
/// without an interval rather than with one of them: the dealiaser then
/// refuses its velocity by name.
pub fn sweep_nyquist(radials: &[Radial]) -> Nyquist {
    let declared: Vec<f64> = radials
        .iter()
        .filter_map(|radial| radial.nyquist_velocity_mps)
        .filter(|value| value.is_finite() && *value > 0.0)
        .map(f64::from)
        .collect();
    let unavailable = Nyquist {
        interval_ms: None,
        source: NyquistSource::Unavailable,
        high_prf_hz: None,
        low_prf_hz: None,
        dual_prf: false,
    };
    let Some(first) = declared.first().copied() else {
        return unavailable;
    };
    let low = declared.iter().copied().fold(f64::INFINITY, f64::min);
    let high = declared.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if high - low > 0.005 * high {
        return unavailable;
    }
    Nyquist {
        interval_ms: Some(first),
        source: NyquistSource::Declared,
        high_prf_hz: None,
        low_prf_hz: None,
        dual_prf: false,
    }
}

fn cut_times(
    volume: &RadarVolume,
    radials: &[Radial],
) -> (
    Option<chrono::DateTime<chrono::Utc>>,
    Option<chrono::DateTime<chrono::Utc>>,
) {
    if radials.iter().all(|radial| radial.time_offset_ms == 0) {
        return (None, None);
    }
    let first = radials.iter().map(|r| r.time_offset_ms).min().unwrap_or(0);
    let last = radials.iter().map(|r| r.time_offset_ms).max().unwrap_or(0);
    (
        Some(volume.volume_time + Duration::milliseconds(i64::from(first))),
        Some(volume.volume_time + Duration::milliseconds(i64::from(last))),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn horizontal_velocity_spelled_velh_claims_the_velocity_name() {
        assert_eq!(quantity_name(&MomentType::Unknown("VELH".into())), "VRADH");
        assert_eq!(quantity_name(&MomentType::Velocity), "VRADH");
        assert_eq!(quantity_name(&MomentType::Unknown("HMC".into())), "HMC");
    }
}
