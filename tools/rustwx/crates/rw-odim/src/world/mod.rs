//! Polar volumes from the rest of the open world, into the same model.
//!
//! The ODIM decoder in this crate is the seam every assimilated radar crosses:
//! [`PolarVolume`] carries per-gate censor codes and a Nyquist interval with
//! its provenance, and [`crate::pack::build_pack`] turns it into the sweep
//! pack the dealias / superob / LETKF path already reads. Nothing below that
//! seam knows which country a radar is in, so a new network is a new *reader*
//! into `PolarVolume`, never a new path through the analysis.
//!
//! The readers here are chosen by the file's own magic bytes, never by its
//! name or by a network table, so adding a network that publishes one of
//! these formats is table work in `gpuwm/obs/data/radar_archives_world.json`
//! and nothing else:
//!
//! | format | reader | reused | written here |
//! |---|---|---|---|
//! | ODIM_H5 (OPERA, BoM via NCI, FMI, ...) | [`crate::read_volume_with`] | all of it | -- |
//! | JMA polar GRIB2 tar (Japan) | [`jma`] | BowEcho `nexrad_io::jma` (tar walk, GRIB2 sections, run-length decode, geometry) | the "No Echo" level, the Nyquist interval from the PRF pair and transmit frequency, every station of a tar |
//! | CfRadial 1.x (IDEAM Bogota, field campaigns) | [`convert`] | BowEcho `nexrad_io::cfradial` | the conversion into censor codes and a sweep Nyquist |
//! | DORADE sweep files | [`convert`] | BowEcho `nexrad_io::dorade` | the same conversion |
//! | IRIS/Sigmet RAW (Vaisala; IDEAM and many others) | [`iris`] | -- | the whole reader (nothing in BowEcho, rusty-weather or the toolbox reads it) |
//! | Universal Format (KMA, RVP8, NASA NPOL) | [`uf`] | -- | the whole reader (same) |
//!
//! gzip-wrapped files of any of these are expanded first.

use std::io::Read;
use std::path::Path;

use crate::censor::{self, Census};
use crate::error::{OdimError, Result};
use crate::quantity::describe;
use crate::volume::{Moment, PolarVolume};
use crate::{Calibration, DecodeOptions};

pub mod convert;
pub mod iris;
pub mod jma;
pub mod uf;

/// The largest file this module will expand from gzip, bytes. A whole
/// CfRadial volume from a C-band dual-pol radar is about 60 MB; this leaves
/// room for an order of magnitude more without letting a tiny hostile stream
/// claim the machine.
const MAX_GUNZIP_BYTES: u64 = 1 << 30;

/// The formats this module recognises by content.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum Format {
    OdimH5,
    JmaGrib2Tar,
    IrisRaw,
    UniversalFormat,
    Cfradial1,
    Dorade,
}

impl Format {
    /// The token the CLI records print.
    pub fn token(self) -> &'static str {
        match self {
            Format::OdimH5 => "odim-h5",
            Format::JmaGrib2Tar => "jma-grib2-tar",
            Format::IrisRaw => "iris-raw",
            Format::UniversalFormat => "universal-format",
            Format::Cfradial1 => "cfradial1",
            Format::Dorade => "dorade",
        }
    }
}

/// Which reader these bytes belong to, or `None`.
pub fn sniff(bytes: &[u8]) -> Option<Format> {
    if bytes.starts_with(b"\x89HDF\r\n\x1a\n") {
        Some(Format::OdimH5)
    } else if nexrad_io::jma::looks_like_jma_tar_bytes(bytes) {
        Some(Format::JmaGrib2Tar)
    } else if iris::looks_like_iris(bytes) {
        Some(Format::IrisRaw)
    } else if uf::looks_like_uf(bytes) {
        Some(Format::UniversalFormat)
    } else if nexrad_io::netcdf3::looks_like_netcdf3_bytes(bytes) {
        Some(Format::Cfradial1)
    } else if nexrad_io::dorade::looks_like_dorade_bytes(bytes) {
        Some(Format::Dorade)
    } else {
        None
    }
}

/// Sniff a file on disk, expanding gzip first. `None` when unrecognised.
pub fn sniff_path(path: &Path) -> Result<Option<Format>> {
    let raw = std::fs::read(path).map_err(|source| OdimError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let expanded = gunzip_if_needed(&raw, &path.display().to_string())?;
    Ok(sniff(expanded.as_deref().unwrap_or(&raw)))
}

/// What one file yielded.
#[derive(Debug)]
pub struct WorldRead {
    pub format: Format,
    /// One volume per station. Every format but the JMA tar carries one
    /// station per file.
    pub volumes: Vec<PolarVolume>,
}

/// Read one file of any recognised format.
///
/// `site` selects one station from a multi-station container (a JMA tar
/// holds the whole national network) before its gates are decoded; it is
/// ignored by single-station formats, where the file is the station and
/// [`select_site`] checks it afterwards.
pub fn read_path(path: &Path, site: Option<&str>, options: &DecodeOptions) -> Result<WorldRead> {
    let raw = std::fs::read(path).map_err(|source| OdimError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let name = path
        .file_name()
        .map(|name| name.to_string_lossy().to_string())
        .unwrap_or_else(|| path.display().to_string());
    let expanded = gunzip_if_needed(&raw, &name)?;
    let bytes = expanded.as_deref().unwrap_or(&raw);
    let format = sniff(bytes).ok_or_else(|| {
        OdimError::format(
            name.clone(),
            "not a format this decoder reads (ODIM_H5, JMA GRIB2 tar, IRIS RAW, \
             Universal Format, CfRadial 1, DORADE, or any of them gzipped)",
        )
    })?;
    let volumes = match format {
        Format::OdimH5 => {
            if expanded.is_some() {
                return Err(OdimError::format(
                    name,
                    "a gzipped HDF5 file; the HDF5 reader opens files, so expand it first",
                ));
            }
            vec![crate::read_volume_with(path, options)?]
        }
        Format::JmaGrib2Tar => jma::read_jma_tar(bytes, &name, site)?,
        Format::IrisRaw => vec![iris::read_iris(bytes, &name)?],
        Format::UniversalFormat => vec![uf::read_uf(bytes, &name)?],
        Format::Cfradial1 => {
            let volume = nexrad_io::cfradial::decode_cfradial1_volume(bytes)
                .map_err(|error| OdimError::format(name.clone(), error.to_string()))?;
            let mut polar =
                convert::from_radar_core(&volume, &name, "CfRadial-1", convert::FirstGate::Centre)?;
            restore_cfradial_range(&mut polar, bytes, &name)?;
            vec![polar]
        }
        Format::Dorade => {
            let volume = nexrad_io::dorade::decode_dorade_sweep_volume(bytes)
                .map_err(|error| OdimError::format(name.clone(), error.to_string()))?;
            vec![convert::from_radar_core(
                &volume,
                &name,
                "DORADE",
                convert::FirstGate::Centre,
            )?]
        }
    };
    // ODIM applied the quantity filter itself, exactly as the plain ODIM
    // path always has; the other readers decode every moment and drop here.
    let volumes = if format == Format::OdimH5 {
        volumes
    } else {
        volumes
            .into_iter()
            .map(|volume| keep_quantities(volume, options))
            .collect()
    };
    Ok(WorldRead { format, volumes })
}

/// Pick one station out of what a set of files yielded.
///
/// With `site` named, the volumes whose station matches (case-insensitive,
/// against any identifier the file carried) are taken. Without it there must
/// be exactly one station; several are refused with the list, because taking
/// the first would put a radar the caller did not ask for behind a record that
/// looks ordinary.
pub fn select_site(
    volumes: Vec<PolarVolume>,
    site: Option<&str>,
    what: &str,
) -> Result<Vec<PolarVolume>> {
    let mut labels: Vec<String> = volumes.iter().map(|v| v.source.label()).collect();
    labels.sort();
    labels.dedup();
    match site {
        Some(wanted) => {
            let chosen: Vec<PolarVolume> = volumes
                .into_iter()
                .filter(|volume| site_matches(volume, wanted))
                .collect();
            if chosen.is_empty() {
                return Err(OdimError::format(
                    what.to_string(),
                    format!(
                        "no station matches --site {wanted:?}; the files hold {}",
                        labels.join(", ")
                    ),
                ));
            }
            Ok(chosen)
        }
        None if labels.len() <= 1 => Ok(volumes),
        None => Err(OdimError::format(
            what.to_string(),
            format!(
                "the files hold {} stations ({}); pass --site to say which one",
                labels.len(),
                labels.join(", ")
            ),
        )),
    }
}

fn site_matches(volume: &PolarVolume, wanted: &str) -> bool {
    let wanted = wanted.trim();
    let source = &volume.source;
    [
        source.nod.as_deref(),
        source.rad.as_deref(),
        source.wmo.as_deref(),
        source.place.as_deref(),
    ]
    .into_iter()
    .flatten()
    .any(|id| id.eq_ignore_ascii_case(wanted))
}

/// Fold the volumes of one station that arrived in several files (JMA's
/// reflectivity and velocity tars; an IRIS site's surveillance and Doppler
/// tasks) into one volume.
///
/// The files must agree on the antenna to within a metre and a hundredth of
/// a degree; a set that does not is two radars and is refused. Sweeps keep
/// file order and are renumbered 1..n so the pack's sweep indices stay
/// unique; the nominal time is the earliest.
pub fn merge_station(mut parts: Vec<PolarVolume>, what: &str) -> Result<PolarVolume> {
    if parts.is_empty() {
        return Err(OdimError::format(what.to_string(), "no volume to merge"));
    }
    let mut merged = parts.remove(0);
    if parts.is_empty() {
        // One file, one volume: its own sweep numbering stands.
        return Ok(merged);
    }
    for part in parts {
        let a = merged.site;
        let b = part.site;
        if (a.latitude_deg - b.latitude_deg).abs() > 0.01
            || (a.longitude_deg - b.longitude_deg).abs() > 0.01
            || (a.height_m - b.height_m).abs() > 1.0
        {
            return Err(OdimError::format(
                what.to_string(),
                format!(
                    "{} at ({:.4}, {:.4}, {:.1} m) and {} at ({:.4}, {:.4}, {:.1} m) are not one antenna",
                    merged.source.label(),
                    a.latitude_deg,
                    a.longitude_deg,
                    a.height_m,
                    part.source.label(),
                    b.latitude_deg,
                    b.longitude_deg,
                    b.height_m
                ),
            ));
        }
        merged.nominal_time = match (merged.nominal_time, part.nominal_time) {
            (Some(x), Some(y)) => Some(x.min(y)),
            (x, y) => x.or(y),
        };
        if merged.system.wavelength_cm.is_none() {
            merged.system.wavelength_cm = part.system.wavelength_cm;
        }
        merged.sweeps.extend(part.sweeps);
    }
    for (index, sweep) in merged.sweeps.iter_mut().enumerate() {
        sweep.index = index + 1;
    }
    Ok(merged)
}

/// Put CfRadial's own range axis back.
///
/// The shared model stores gate geometry in whole metres, so IDEAM Bogota's
/// 62.5 m gates came back as 63 m with the first centre at 31 m instead of
/// 31.25 m: half a metre per gate, 480 m by the 960th gate, in the direction
/// that lifts every far gate's beam height. The file's `range(range)`
/// variable is exact; it is read here and replaces the rounded axis on every
/// sweep that was built from it (the rounded values must match, so a sweep
/// on some other axis is left alone).
fn restore_cfradial_range(volume: &mut PolarVolume, bytes: &[u8], name: &str) -> Result<()> {
    let file = nexrad_io::netcdf3::Nc3File::open(bytes)
        .map_err(|error| OdimError::format(name.to_string(), error.to_string()))?;
    let Ok(range) = file.read_var("range") else {
        return Ok(());
    };
    let (Some(first), Some(second)) = (range.get_f64(0), range.get_f64(1)) else {
        return Ok(());
    };
    let spacing = second - first;
    let last = range.get_f64(range.len() - 1).unwrap_or(f64::NAN);
    let uniform =
        ((last - first) - spacing * (range.len() - 1) as f64).abs() < 1e-3 * spacing.abs().max(1.0);
    if !(spacing > 0.0 && uniform) {
        return Ok(());
    }
    for sweep in &mut volume.sweeps {
        let rounded_first = (sweep.range_start_m + sweep.range_scale_m / 2.0).round();
        if sweep.range_scale_m == spacing.round() && rounded_first == first.round() {
            sweep.range_scale_m = spacing;
            sweep.range_start_m = first - spacing / 2.0;
        }
    }
    Ok(())
}

fn keep_quantities(mut volume: PolarVolume, options: &DecodeOptions) -> PolarVolume {
    if let Some(wanted) = &options.quantities {
        for sweep in &mut volume.sweeps {
            sweep
                .moments
                .retain(|m| wanted.iter().any(|w| w.eq_ignore_ascii_case(&m.quantity)));
        }
        volume.sweeps.retain(|sweep| !sweep.moments.is_empty());
    }
    volume
}

fn gunzip_if_needed(raw: &[u8], name: &str) -> Result<Option<Vec<u8>>> {
    if !raw.starts_with(&[0x1f, 0x8b]) {
        return Ok(None);
    }
    let mut out = Vec::new();
    flate2::read::MultiGzDecoder::new(raw)
        .take(MAX_GUNZIP_BYTES + 1)
        .read_to_end(&mut out)
        .map_err(|error| OdimError::format(name.to_string(), format!("gzip: {error}")))?;
    if out.len() as u64 > MAX_GUNZIP_BYTES {
        return Err(OdimError::format(
            name.to_string(),
            format!("expands past the {MAX_GUNZIP_BYTES}-byte limit"),
        ));
    }
    Ok(Some(out))
}

/// Count a censor plane.
pub(crate) fn census_of(codes: &[u8]) -> Census {
    let mut census = Census::default();
    for code in codes {
        match *code {
            censor::MEASURED => census.measured += 1,
            censor::UNDETECT => census.undetect += 1,
            censor::NOT_COLLECTED => census.not_collected += 1,
            censor::SENTINEL_AMBIGUOUS => census.sentinel_ambiguous += 1,
            _ => census.nodata += 1,
        }
    }
    census
}

/// A decoded moment from physical values and their censor codes.
pub(crate) fn moment(
    quantity: &str,
    path: String,
    nrays: usize,
    nbins: usize,
    values: Vec<f64>,
    censor: Vec<u8>,
) -> Moment {
    let info = describe(quantity);
    Moment {
        quantity: quantity.to_string(),
        path,
        unit: info.unit.to_string(),
        kind: info.kind,
        // Not an ODIM file: the values are already physical, so the
        // calibration is the identity and declares no sentinels. The
        // reason a gate is not a number is in `censor`.
        calibration: Calibration {
            gain: 1.0,
            offset: 0.0,
            nodata: None,
            undetect: None,
            sentinels_collide: false,
        },
        nrays,
        nbins,
        census: census_of(&censor),
        values,
        censor,
    }
}

/// Circular mean of two azimuths, degrees in `[0, 360)`.
pub(crate) fn mean_azimuth_deg(a: f64, b: f64) -> f64 {
    let mut diff = b - a;
    if diff > 180.0 {
        diff -= 360.0;
    } else if diff < -180.0 {
        diff += 360.0;
    }
    (a + diff / 2.0).rem_euclid(360.0)
}

/// Speed of light, m s-1, for a wavelength from a declared frequency.
pub(crate) const SPEED_OF_LIGHT_MS: f64 = 299_792_458.0;

/// Test helpers other modules' tests build volumes with.
#[cfg(test)]
pub(crate) mod tests_support {
    pub(crate) fn moment(quantity: &str, values: Vec<f64>, censor: Vec<u8>) -> crate::Moment {
        let n = values.len();
        super::moment(quantity, "test".to_string(), 1, n, values, censor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn azimuth_means_wrap_through_north() {
        assert!((mean_azimuth_deg(359.0, 1.0) - 0.0).abs() < 1e-9);
        assert!((mean_azimuth_deg(10.0, 12.0) - 11.0).abs() < 1e-9);
        assert!((mean_azimuth_deg(1.0, 359.0) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn unknown_bytes_are_not_claimed() {
        assert_eq!(
            sniff(b"definitely not a radar file at all, not even close"),
            None
        );
    }
}
