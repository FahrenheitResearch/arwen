//! Universal Format (UF) radar files.
//!
//! The format (Barnes, 1980, "Universal Format for Radar Data", BAMS 61,
//! 1401-1404, and the NCAR/RAF UF document): big-endian 16-bit words, one
//! record per ray. A 45-word mandatory header (site, time, azimuth,
//! elevation, sweep number, fixed angle, the missing-data value), an
//! optional header, a data header listing each field's two-letter name and
//! the word position of its field header, and per field a header (data
//! position, scale factor, range to the first gate in km plus metres to its
//! centre, gate count and spacing, wavelength, PRT, and for velocity fields
//! the Nyquist velocity) followed by the scaled integers. Records may be
//! wrapped in 2- or 4-byte Fortran record-length markers.
//!
//! KMA publishes its raw volumes in UF (API Hub, `rdr_file_down.php?rdr=UF`),
//! as do Vaisala RVP8 sites and NASA's NPOL. Nothing in BowEcho, rusty-weather
//! or the toolbox read it, so the reader is new; it was checked gate-for-gate
//! against Py-ART 2.3's `read_uf` on public UF files.
//!
//! **Censor codes.** UF has one sentinel, the mandatory header's missing-data
//! value. It says the gate holds no value, not that the radar saw clear air,
//! so it becomes [`censor::NODATA`]; UF cannot mint `UNDETECT`.
//!
//! **Nyquist.** Declared per velocity field (field header word 19, scaled
//! like the field). The sweep takes it when every ray agrees.

use chrono::{DateTime, Duration, NaiveDate, Utc};

use super::moment;
use crate::censor;
use crate::error::{OdimError, Result};
use crate::volume::{
    AzimuthSource, Nyquist, NyquistSource, PolarVolume, Site, Source, Sweep, SystemNotes,
};

/// The two-letter UF field names this reader gives an ODIM spelling, so the
/// pack's canonical `REF`/`VEL` claim finds them. Corrected reflectivity
/// (`CZ`) outranks `DZ`; edited velocity (`VE`) outranks raw (`VR`). Any
/// other field keeps its own name and is still packed.
fn quantity(field: &str) -> String {
    match field {
        "CZ" => "DBZH",
        "DZ" => "DBZ",
        "ZT" => "TH",
        "VE" => "VRADH",
        "VR" => "VRAD",
        "SW" => "WRADH",
        "ZD" | "DR" | "ZR" => "ZDR",
        "PH" | "DP" => "PHIDP",
        "RH" => "RHOHV",
        "KD" => "KDP",
        "SQ" => "SQIH",
        other => other,
    }
    .to_string()
}

fn is_velocity(field: &str) -> bool {
    matches!(field, "VE" | "VR" | "VF" | "VT" | "VEL" | "VP")
}

/// UF sweep modes this reader packs: PPI (1) and surveillance (8).
fn is_ppi(mode: i16) -> bool {
    matches!(mode, 1 | 8)
}

/// Where the first record's "UF" sits: 0, or after a 2- or 4-byte marker.
fn padding(bytes: &[u8]) -> Option<usize> {
    [0usize, 4, 2]
        .into_iter()
        .find(|pad| bytes.get(*pad..pad + 2) == Some(b"UF"))
}

/// `true` when these bytes open with a UF record.
pub fn looks_like_uf(bytes: &[u8]) -> bool {
    padding(bytes).is_some_and(|pad| {
        // The record length (mandatory word 2) must fit the file.
        be_i16(bytes, pad + 2)
            .is_some_and(|words| words > 45 && pad + 2 * words as usize <= bytes.len())
    })
}

struct Ray {
    sweep: i16,
    mode: i16,
    fixed_angle: f64,
    azimuth: f64,
    elevation: f64,
    time: DateTime<Utc>,
    fields: Vec<Field>,
}

struct Field {
    name: String,
    first_gate_m: f64,
    spacing_m: f64,
    values: Vec<f64>,
    nyquist: Option<f64>,
    wavelength_cm: Option<f64>,
}

/// Decode a UF file into a polar volume (PPI sweeps only).
pub fn read_uf(bytes: &[u8], name: &str) -> Result<PolarVolume> {
    let err = |detail: String| OdimError::format(name.to_string(), detail);
    let pad = padding(bytes).ok_or_else(|| err("no UF record".into()))?;
    let mut at = 0usize;
    let mut rays: Vec<Ray> = Vec::new();
    let mut site: Option<(String, f64, f64, f64)> = None;
    let mut skipped_rays = 0usize;
    while at + pad + 4 <= bytes.len() {
        let start = at + pad;
        if bytes.get(start..start + 2) != Some(b"UF") {
            break;
        }
        let words = be_i16(bytes, start + 2).unwrap_or(0);
        if words <= 45 {
            return Err(err(format!(
                "record at byte {start} declares {words} words"
            )));
        }
        let length = 2 * words as usize;
        let record = bytes
            .get(start..start + length)
            .ok_or_else(|| err(format!("record at byte {start} overruns the file")))?;
        at = start + length + pad;
        let ray =
            parse_ray(record).map_err(|detail| err(format!("record at byte {start}: {detail}")))?;
        if site.is_none() {
            site = Some(parse_site(record).map_err(|detail| err(detail))?);
        }
        if is_ppi(ray.mode) {
            rays.push(ray);
        } else {
            skipped_rays += 1;
        }
    }
    let (site_name, latitude_deg, longitude_deg, height_m) =
        site.ok_or_else(|| err("no records".into()))?;
    if rays.is_empty() {
        return Err(err(format!(
            "no PPI rays ({skipped_rays} rays of other sweep modes, e.g. RHI, are not packed)"
        )));
    }

    // Consecutive rays with one sweep number are one sweep.
    let mut groups: Vec<Vec<Ray>> = Vec::new();
    for ray in rays {
        match groups.last_mut() {
            Some(group) if group[0].sweep == ray.sweep => group.push(ray),
            _ => groups.push(vec![ray]),
        }
    }

    let mut sweeps = Vec::new();
    let mut wavelength_cm = None;
    let nominal_time = groups[0][0].time;
    for group in &groups {
        // Field names in first-seen order, then one sweep per range axis.
        let mut axes: Vec<((i64, i64), Vec<String>)> = Vec::new();
        for ray in group {
            for field in &ray.fields {
                let key = (
                    (field.first_gate_m * 100.0).round() as i64,
                    (field.spacing_m * 100.0).round() as i64,
                );
                match axes.iter_mut().find(|(axis, _)| *axis == key) {
                    Some((_, names)) => {
                        if !names.contains(&field.name) {
                            names.push(field.name.clone());
                        }
                    }
                    None => axes.push((key, vec![field.name.clone()])),
                }
            }
        }
        let nrays = group.len();
        for ((first_cm, spacing_cm), names) in axes {
            let spacing_m = spacing_cm as f64 / 100.0;
            if spacing_m <= 0.0 {
                continue;
            }
            let first_gate_m = first_cm as f64 / 100.0;
            let mut moments = Vec::new();
            let mut nbins_sweep = 0usize;
            let mut nyquists: Vec<f64> = Vec::new();
            for field_name in &names {
                let on_axis = |field: &&Field| {
                    field.name == *field_name
                        && (
                            (field.first_gate_m * 100.0).round() as i64,
                            (field.spacing_m * 100.0).round() as i64,
                        ) == (first_cm, spacing_cm)
                };
                let nbins = group
                    .iter()
                    .filter_map(|ray| ray.fields.iter().find(on_axis))
                    .map(|field| field.values.len())
                    .max()
                    .unwrap_or(0);
                if nbins == 0 {
                    continue;
                }
                nbins_sweep = nbins_sweep.max(nbins);
                let mut values = vec![f64::NAN; nrays * nbins];
                let mut codes = vec![censor::NOT_COLLECTED; nrays * nbins];
                for (row, ray) in group.iter().enumerate() {
                    let Some(field) = ray.fields.iter().find(on_axis) else {
                        continue;
                    };
                    if wavelength_cm.is_none() {
                        wavelength_cm = field.wavelength_cm;
                    }
                    if is_velocity(field_name)
                        && let Some(nyquist) = field.nyquist
                    {
                        nyquists.push(nyquist);
                    }
                    for bin in 0..nbins {
                        let out = row * nbins + bin;
                        match field.values.get(bin) {
                            Some(value) if value.is_finite() => {
                                values[out] = *value;
                                codes[out] = censor::MEASURED;
                            }
                            _ => codes[out] = censor::NODATA,
                        }
                    }
                }
                let spelled = quantity(field_name);
                moments.push(moment(
                    &spelled,
                    format!("UF/sweep{}/{field_name}", group[0].sweep),
                    nrays,
                    nbins,
                    values,
                    codes,
                ));
            }
            if moments.is_empty() {
                continue;
            }
            sweeps.push(Sweep {
                index: sweeps.len() + 1,
                path: format!("UF/sweep{}", group[0].sweep),
                elevation_deg: group[0].fixed_angle,
                nrays,
                nbins: nbins_sweep,
                range_scale_m: spacing_m,
                range_start_m: first_gate_m - spacing_m / 2.0,
                a1gate: None,
                start_time: group.iter().map(|ray| ray.time).min(),
                end_time: group.iter().map(|ray| ray.time).max(),
                nyquist: agreed_nyquist(&nyquists),
                azimuth_deg: group.iter().map(|ray| ray.azimuth).collect(),
                azimuth_source: AzimuthSource::Recorded,
                ray_elevation_deg: Some(group.iter().map(|ray| ray.elevation).collect()),
                ray_prf_hz: None,
                moments,
            });
        }
    }
    if sweeps.is_empty() {
        return Err(err("no sweep carried a field".into()));
    }
    let label = site_name;
    Ok(PolarVolume {
        object: "PVOL".to_string(),
        conventions: Some("UF".to_string()),
        version: None,
        source: Source {
            raw: format!("UF:{label}"),
            nod: (!label.is_empty()).then(|| label.clone()),
            wmo: None,
            wigos: None,
            place: (!label.is_empty()).then(|| label.clone()),
            rad: None,
        },
        nominal_time: Some(nominal_time),
        site: Site {
            latitude_deg,
            longitude_deg,
            height_m,
        },
        system: SystemNotes {
            wavelength_cm,
            beamwidth_deg: None,
            system: None,
            software: None,
            sw_version: None,
        },
        sweeps,
    })
}

fn agreed_nyquist(values: &[f64]) -> Nyquist {
    let finite: Vec<f64> = values
        .iter()
        .copied()
        .filter(|v| v.is_finite() && *v > 0.0)
        .collect();
    let Some(first) = finite.first().copied() else {
        return Nyquist::establish(None, None, None, None);
    };
    let low = finite.iter().copied().fold(f64::INFINITY, f64::min);
    let high = finite.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if high - low > 0.005 * high {
        return Nyquist::establish(None, None, None, None);
    }
    Nyquist {
        interval_ms: Some(first),
        source: NyquistSource::Declared,
        high_prf_hz: None,
        low_prf_hz: None,
        dual_prf: false,
    }
}

/// Word `n` (1-based, as the UF document numbers them) of a record.
fn word(record: &[u8], n: usize) -> std::result::Result<i16, String> {
    n.checked_sub(1)
        .and_then(|index| be_i16(record, 2 * index))
        .ok_or_else(|| format!("word {n} is past the record's end"))
}

fn text(record: &[u8], n: usize, words: usize) -> String {
    let start = 2 * (n - 1);
    record
        .get(start..start + 2 * words)
        .map(|bytes| {
            String::from_utf8_lossy(bytes)
                .trim_matches(|c: char| c == '\0' || c.is_whitespace())
                .to_string()
        })
        .unwrap_or_default()
}

fn degrees(record: &[u8], n: usize) -> std::result::Result<f64, String> {
    let d = f64::from(word(record, n)?);
    let m = f64::from(word(record, n + 1)?);
    let s = f64::from(word(record, n + 2)?) / 64.0;
    // Each component carries the sign of the whole angle.
    Ok(d + m / 60.0 + s / 3600.0)
}

fn parse_site(record: &[u8]) -> std::result::Result<(String, f64, f64, f64), String> {
    let radar = text(record, 11, 4);
    let site = text(record, 15, 4);
    let label = if !site.is_empty() { site } else { radar };
    Ok((
        label,
        degrees(record, 19)?,
        degrees(record, 22)?,
        f64::from(word(record, 25)?),
    ))
}

fn parse_ray(record: &[u8]) -> std::result::Result<Ray, String> {
    let missing = word(record, 45)?;
    let data_header =
        usize::try_from(word(record, 5)?).map_err(|_| "negative data header position")?;
    let mut year = i32::from(word(record, 26)?);
    if year < 100 {
        year += if year >= 70 { 1900 } else { 2000 };
    }
    let date = NaiveDate::from_ymd_opt(
        year,
        u32::try_from(word(record, 27)?).map_err(|_| "month")?,
        u32::try_from(word(record, 28)?).map_err(|_| "day")?,
    )
    .ok_or("invalid date")?;
    let time = date
        .and_hms_opt(
            u32::try_from(word(record, 29)?).map_err(|_| "hour")?,
            u32::try_from(word(record, 30)?).map_err(|_| "minute")?,
            0,
        )
        .ok_or("invalid time")?
        .and_utc()
        + Duration::seconds(i64::from(word(record, 31)?));
    let fields_in_record =
        usize::try_from(word(record, data_header + 2)?).map_err(|_| "field count")?;
    let mut fields = Vec::with_capacity(fields_in_record);
    for index in 0..fields_in_record {
        let entry = data_header + 3 + 2 * index;
        let name = text(record, entry, 1);
        let header =
            usize::try_from(word(record, entry + 1)?).map_err(|_| "field header position")?;
        let data_at = usize::try_from(word(record, header)?).map_err(|_| "data position")?;
        let scale = f64::from(word(record, header + 1)?);
        if scale == 0.0 {
            return Err(format!("field {name} has scale factor 0"));
        }
        // Field header words 3-4: range to the first gate (km) plus the
        // adjustment to its centre (m), so their sum is the centre. Py-ART
        // 2.3 adds a further half gate (it reads the sum as the start); the
        // UF definition is followed here and the divergence is deliberate.
        let first_gate_m =
            f64::from(word(record, header + 2)?) * 1000.0 + f64::from(word(record, header + 3)?);
        let spacing_m = f64::from(word(record, header + 4)?);
        let gates = usize::try_from(word(record, header + 5)?).map_err(|_| "gate count")?;
        let wavelength_raw = word(record, header + 11)?;
        let mut values = Vec::with_capacity(gates);
        for gate in 0..gates {
            let raw = word(record, data_at + gate)?;
            values.push(if raw == missing {
                f64::NAN
            } else {
                f64::from(raw) / scale
            });
        }
        // Velocity fields carry the Nyquist velocity in word 20 of a
        // 21-word field header (the field-specific block after word 19).
        let nyquist = if is_velocity(&name) && data_at >= header + 21 {
            word(record, header + 19)
                .ok()
                .filter(|raw| *raw != missing && *raw > 0)
                .map(|raw| f64::from(raw) / scale)
        } else {
            None
        };
        fields.push(Field {
            name,
            first_gate_m,
            spacing_m,
            values,
            nyquist,
            wavelength_cm: (wavelength_raw > 0).then(|| f64::from(wavelength_raw) / 64.0),
        });
    }
    Ok(Ray {
        sweep: word(record, 10)?,
        mode: word(record, 35)?,
        fixed_angle: f64::from(word(record, 36)?) / 64.0,
        azimuth: (f64::from(word(record, 33)?) / 64.0).rem_euclid(360.0),
        elevation: f64::from(word(record, 34)?) / 64.0,
        time,
        fields,
    })
}

fn be_i16(bytes: &[u8], at: usize) -> Option<i16> {
    bytes
        .get(at..at + 2)
        .map(|b| i16::from_be_bytes([b[0], b[1]]))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn field_names_take_the_spelling_the_pack_claims() {
        assert_eq!(quantity("CZ"), "DBZH");
        assert_eq!(quantity("DZ"), "DBZ");
        assert_eq!(quantity("VE"), "VRADH");
        assert_eq!(quantity("VR"), "VRAD");
        assert_eq!(quantity("XX"), "XX");
    }

    #[test]
    fn fortran_markers_are_found() {
        let mut bytes = vec![0u8, 0, 0, 100];
        bytes.extend_from_slice(b"UF");
        assert_eq!(padding(&bytes), Some(4));
        assert_eq!(padding(b"UF\0\0"), Some(0));
    }
}
