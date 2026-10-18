//! Vaisala IRIS/Sigmet RAW product files.
//!
//! The format of the IRIS "RAW" ingest product (Vaisala, *IRIS Programmer's
//! Manual*, sections 4.2-4.4 and 4.9): 6144-byte records, record 0 the
//! `product_hdr` (structure 27), record 1 the `ingest_header` (structure 23),
//! then data records each opening with a 12-byte `raw_prod_bhdr`. A sweep's
//! first record carries one 76-byte `ingest_data_header` per recorded data
//! type; the rays follow as one run-length-compressed stream across records,
//! every data type of a ray in turn, each ray starting with a six-word ray
//! header (start/stop azimuth and elevation, bin count, seconds into the
//! sweep).
//!
//! Nothing in BowEcho, rusty-weather or the toolbox read this format, so the
//! reader is new. Its decode table and record walk were checked gate-for-gate
//! against Py-ART 2.3's `read_sigmet` and xradar 0.12's IRIS backend on real
//! IDEAM files (see the crate's `world_real_files` test).
//!
//! **Censor codes.** IRIS reserves raw 0 of every data type for "no data
//! available" (the gate failed the task's thresholds: the radar looked and
//! nothing passed) and raw 255 / 65535 for "area not scanned". They become
//! [`censor::UNDETECT`] and [`censor::NODATA`], the mapping Vaisala's own
//! ODIM export uses. One-byte velocity, width, KDP and the velocity-like
//! types use their full code range for data, so only 0 is censored there.
//!
//! **Nyquist.** The one-byte velocity is a fraction of the unambiguous
//! interval, `lambda * PRF / 4` times `multi_prf_mode_flag + 1` (1, 2, 3, 4
//! for the 1:1, 2:3, 3:4 and 4:5 dual-PRF modes). That interval is the
//! sweep's Nyquist, derived from the file's own wavelength, PRF and mode:
//! [`NyquistSource::DerivedSinglePrf`] or
//! [`NyquistSource::DerivedDualPrf`](crate::volume::NyquistSource).

use chrono::{DateTime, Duration, NaiveDate, Utc};

use super::{mean_azimuth_deg, moment};
use crate::censor;
use crate::error::{OdimError, Result};
use crate::volume::{AzimuthSource, Nyquist, PolarVolume, Site, Source, Sweep, SystemNotes};

const RECORD: usize = 6144;
const PRODUCT_HDR: i16 = 27;
const INGEST_HEADER: i16 = 23;
const BHDR: usize = 12;
const DATA_HEADER: usize = 76;
/// `task_scan_info.antenna_scan_mode` value for an RHI.
const SCAN_MODE_RHI: u16 = 2;

/// Product header (record 0) offsets.
mod product {
    pub const PRF_HZ: usize = 332 + 120; // product_end.prf
    pub const WAVELENGTH_CENTI_CM: usize = 332 + 148; // product_end.wavelength
}

/// Ingest header (record 1) offsets, relative to the record.
mod ingest {
    pub const VOLUME_START: usize = 100;
    pub const SITE_NAME: usize = 162;
    pub const GMT_OFFSET_LOCAL_MIN: usize = 160;
    pub const LATITUDE: usize = 180;
    pub const LONGITUDE: usize = 184;
    pub const HEIGHT_SITE_M: usize = 188;
    pub const HEIGHT_RADAR_M: usize = 190;
    pub const ALTITUDE_RADAR_CM: usize = 200;
    pub const DSP_MASK_WORDS: [usize; 5] = [628, 636, 640, 644, 648];
    pub const MULTI_PRF_MODE: usize = 768;
    pub const RANGE_FIRST_BIN_CM: usize = 1264;
    pub const OUTPUT_BINS: usize = 1274;
    pub const OUTPUT_BIN_STEP_CM: usize = 1280;
    pub const SCAN_MODE: usize = 1424;
    pub const WAVELENGTH_CENTI_CM: usize = 1744;
    pub const BEAMWIDTH_H: usize = 1808;
}

/// `true` when these bytes open with an IRIS product header followed, one
/// record later, by an ingest header.
pub fn looks_like_iris(bytes: &[u8]) -> bool {
    bytes.len() >= 2 * RECORD
        && le_i16(bytes, 0) == Some(PRODUCT_HDR)
        && le_i16(bytes, RECORD) == Some(INGEST_HEADER)
}

/// How one IRIS data type's raw words become a value and a censor code.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Decode {
    /// 1 byte, `(N - 64) / 2` dB; 0 no data, 255 not scanned.
    Db1,
    /// 1 byte, `(N - 128) / 127` of the extended Nyquist; 0 no data.
    Vel1,
    /// 1 byte, `N / 256` of the single-PRF Nyquist; 0 no data.
    Width1,
    /// 1 byte, `(N - 128) / 16` dB; 0 no data, 255 not scanned.
    Zdr1,
    /// 1 byte signed, logarithmic, divided by wavelength (cm).
    Kdp1,
    /// 1 byte, `180 (N - 1) / 254` deg; 0 no data, 255 not scanned.
    Phidp1,
    /// 1 byte, `(N - 1) / 1.68667 - 75` m/s (unfolded velocity); 0 no data.
    Velc1,
    /// 1 byte, `sqrt((N - 1) / 253)`; 0 no data, 255 not scanned.
    Rho1,
    /// 2 bytes, `(N - 32768) / 100`; 0 no data, 65535 not scanned.
    Centi2,
    /// 2 bytes, `N / 100`; 0 no data, 65535 not scanned.
    Width2,
    /// 2 bytes, `(N - 1) / 65533`; 0 no data, 65535 not scanned.
    Rho2,
    /// 2 bytes, `360 (N - 1) / 65534` deg; 0 no data, 65535 not scanned.
    Phidp2,
}

impl Decode {
    fn one_byte(self) -> bool {
        !matches!(
            self,
            Decode::Centi2 | Decode::Width2 | Decode::Rho2 | Decode::Phidp2
        )
    }
}

/// The ODIM spelling and decoder for an IRIS data type, or `None` for types
/// that are not radar moments an assimilation reads (extended headers,
/// hydrometeor class, rain rate, ...). Those are skipped, never guessed.
fn data_type(code: u16) -> Option<(&'static str, Decode)> {
    Some(match code {
        1 => ("TH", Decode::Db1),
        2 => ("DBZH", Decode::Db1),
        3 => ("VRADH", Decode::Vel1),
        4 => ("WRADH", Decode::Width1),
        5 => ("ZDR", Decode::Zdr1),
        7 => ("DBZC", Decode::Db1),
        8 => ("TH", Decode::Centi2),
        9 => ("DBZH", Decode::Centi2),
        10 => ("VRADH", Decode::Centi2),
        11 => ("WRADH", Decode::Width2),
        12 => ("ZDR", Decode::Centi2),
        14 => ("KDP", Decode::Kdp1),
        15 => ("KDP", Decode::Centi2),
        16 => ("PHIDP", Decode::Phidp1),
        17 => ("VRADDH", Decode::Velc1),
        18 => ("SQIH", Decode::Rho1),
        19 => ("RHOHV", Decode::Rho1),
        20 => ("RHOHV", Decode::Rho2),
        21 => ("DBZC", Decode::Centi2),
        22 => ("VRADDH", Decode::Centi2),
        23 => ("SQIH", Decode::Rho2),
        24 => ("PHIDP", Decode::Phidp2),
        57 => ("ZDRC", Decode::Zdr1),
        58 => ("ZDRC", Decode::Centi2),
        71 => ("DBTE", Decode::Db1),
        72 => ("DBTE", Decode::Centi2),
        73 => ("DBZE", Decode::Db1),
        74 => ("DBZE", Decode::Centi2),
        _ => return None,
    })
}

struct Scales {
    /// Unambiguous interval the one-byte velocity is scaled by, m/s.
    extended_nyquist: f64,
    /// Single-PRF interval the one-byte width is scaled by, m/s.
    single_nyquist: f64,
    wavelength_cm: f64,
}

fn decode_gate(kind: Decode, raw: u16, scales: &Scales) -> (f64, u8) {
    let nan = f64::NAN;
    let n = f64::from(raw);
    match kind {
        Decode::Db1 | Decode::Zdr1 | Decode::Phidp1 | Decode::Rho1 => match raw {
            0 => (nan, censor::UNDETECT),
            255 => (nan, censor::NODATA),
            _ => (
                match kind {
                    Decode::Db1 => (n - 64.0) / 2.0,
                    Decode::Zdr1 => (n - 128.0) / 16.0,
                    Decode::Phidp1 => 180.0 * (n - 1.0) / 254.0,
                    _ => ((n - 1.0) / 253.0).sqrt(),
                },
                censor::MEASURED,
            ),
        },
        Decode::Vel1 | Decode::Width1 | Decode::Velc1 => match raw {
            0 => (nan, censor::UNDETECT),
            _ => (
                match kind {
                    Decode::Vel1 => (n - 128.0) / 127.0 * scales.extended_nyquist,
                    Decode::Width1 => n / 256.0 * scales.single_nyquist,
                    _ => (n - 1.0) / (253.0 / 150.0) - 75.0,
                },
                censor::MEASURED,
            ),
        },
        Decode::Kdp1 => {
            let signed = raw as u8 as i8;
            match signed {
                0 => (nan, censor::UNDETECT),
                -1 => (nan, censor::NODATA),
                -128 => (0.0, censor::MEASURED),
                _ => {
                    let d = f64::from(signed);
                    let magnitude = 600f64.powf((127.0 - d.abs()) / 126.0);
                    (
                        -0.25 * d.signum() * magnitude / scales.wavelength_cm,
                        censor::MEASURED,
                    )
                }
            }
        }
        Decode::Centi2 | Decode::Width2 | Decode::Rho2 | Decode::Phidp2 => match raw {
            0 => (nan, censor::UNDETECT),
            u16::MAX => (nan, censor::NODATA),
            _ => (
                match kind {
                    Decode::Centi2 => (n - 32768.0) / 100.0,
                    Decode::Width2 => n / 100.0,
                    Decode::Rho2 => (n - 1.0) / 65533.0,
                    _ => 360.0 * (n - 1.0) / 65534.0,
                },
                censor::MEASURED,
            ),
        },
    }
}

/// Decode one IRIS RAW file into a polar volume.
pub fn read_iris(bytes: &[u8], name: &str) -> Result<PolarVolume> {
    let err = |detail: String| OdimError::format(name.to_string(), detail);
    if !looks_like_iris(bytes) {
        return Err(err("not an IRIS RAW product (structures 27 and 23)".into()));
    }
    let ingest = &bytes[RECORD..2 * RECORD];
    let get16 =
        |at: usize| le_i16(ingest, at).ok_or_else(|| err(format!("ingest header short at {at}")));
    let getu16 =
        |at: usize| le_u16(ingest, at).ok_or_else(|| err(format!("ingest header short at {at}")));
    let get32 =
        |at: usize| le_i32(ingest, at).ok_or_else(|| err(format!("ingest header short at {at}")));
    let getu32 =
        |at: usize| le_u32(ingest, at).ok_or_else(|| err(format!("ingest header short at {at}")));

    let scan_mode = getu16(ingest::SCAN_MODE)?;
    if scan_mode == SCAN_MODE_RHI {
        return Err(err(
            "an RHI task; a sweep pack holds PPI cuts, whose fixed angle is an elevation".into(),
        ));
    }
    let site_name = ascii(&ingest[ingest::SITE_NAME..ingest::SITE_NAME + 16]);
    let latitude_deg = bin4_signed(getu32(ingest::LATITUDE)?);
    let longitude_deg = bin4_signed(getu32(ingest::LONGITUDE)?);
    let altitude_cm = get32(ingest::ALTITUDE_RADAR_CM)?;
    let height_m = if altitude_cm != 0 {
        f64::from(altitude_cm) / 100.0
    } else {
        f64::from(get16(ingest::HEIGHT_SITE_M)?) + f64::from(get16(ingest::HEIGHT_RADAR_M)?)
    };
    let gmt_offset_min = get16(ingest::GMT_OFFSET_LOCAL_MIN)?;

    // Wavelength and PRF: the product header's are the ones the one-byte
    // velocity was scaled with; the task's wavelength is the fallback.
    let mut wavelength_centi_cm = le_i32(bytes, product::WAVELENGTH_CENTI_CM).unwrap_or(0);
    if wavelength_centi_cm <= 0 {
        wavelength_centi_cm = get32(ingest::WAVELENGTH_CENTI_CM)?;
    }
    let prf_hz = f64::from(le_i32(bytes, product::PRF_HZ).unwrap_or(0));
    let multi_prf = getu16(ingest::MULTI_PRF_MODE)?;
    let wavelength_cm = f64::from(wavelength_centi_cm) / 100.0;
    let nyquist = if wavelength_cm > 0.0 && prf_hz > 0.0 && multi_prf <= 3 {
        if multi_prf == 0 {
            Nyquist::establish(None, Some(prf_hz), Some(prf_hz), Some(wavelength_cm))
        } else {
            // N:N+1 dual PRF; the task PRF is the high one.
            let n = f64::from(multi_prf) + 1.0;
            Nyquist::dual_prf_extended(prf_hz, prf_hz * n / (n + 1.0), Some(wavelength_cm))
        }
    } else {
        Nyquist::establish(None, None, None, None)
    };
    let single_nyquist = wavelength_cm / 100.0 * prf_hz / 4.0;
    let scales = Scales {
        extended_nyquist: nyquist.interval_ms.unwrap_or(f64::NAN),
        single_nyquist,
        wavelength_cm,
    };

    let first_bin_cm = get32(ingest::RANGE_FIRST_BIN_CM)?;
    let step_cm = get32(ingest::OUTPUT_BIN_STEP_CM)?;
    let nbins = usize::try_from(get16(ingest::OUTPUT_BINS)?)
        .map_err(|_| err("negative output bin count".into()))?;
    if step_cm <= 0 || nbins == 0 {
        return Err(err(format!("range geometry: {nbins} bins of {step_cm} cm")));
    }
    let range_scale_m = f64::from(step_cm) / 100.0;
    // `range_first_bin` is the centre of the first bin; IRIS writes 0 when
    // the first bin starts at the antenna.
    let range_start_m = if first_bin_cm == 0 {
        0.0
    } else {
        f64::from(first_bin_cm) / 100.0 - range_scale_m / 2.0
    };

    let mut types = Vec::new();
    for (word, offset) in ingest::DSP_MASK_WORDS.iter().enumerate() {
        let mask = getu32(*offset)?;
        for bit in 0..32u16 {
            if mask & (1 << bit) != 0 {
                types.push(word as u16 * 32 + bit);
            }
        }
    }
    if types.is_empty() {
        return Err(err("the task records no data types".into()));
    }

    let volume_start = ymds(ingest, ingest::VOLUME_START, gmt_offset_min)
        .ok_or_else(|| err("unreadable volume start time".into()))?;

    // Group the data records into one contiguous stream per sweep.
    let mut streams: Vec<(i16, usize, Vec<u8>)> = Vec::new();
    let mut record = 2usize;
    while (record + 1) * RECORD <= bytes.len() {
        let block = &bytes[record * RECORD..(record + 1) * RECORD];
        record += 1;
        let sweep = le_i16(block, 2).unwrap_or(0);
        if sweep <= 0 {
            continue;
        }
        match streams.last_mut() {
            Some((current, _, stream)) if *current == sweep => {
                stream.extend_from_slice(&block[BHDR..])
            }
            _ => {
                let first_ray = usize::try_from(le_i16(block, 4).unwrap_or(-1)).unwrap_or(0);
                streams.push((sweep, first_ray, block[BHDR..].to_vec()));
            }
        }
    }
    if streams.is_empty() {
        return Err(err("no data records".into()));
    }

    let mut sweeps = Vec::new();
    for (sweep_number, first_ray, stream) in streams {
        let context = format!("{name} sweep {sweep_number}");
        let headers_len = DATA_HEADER * types.len();
        if stream.len() < headers_len {
            return Err(OdimError::format(
                context,
                "ingest data headers are truncated",
            ));
        }
        let header = &stream[..DATA_HEADER];
        let fixed_angle = bin2_signed(le_u16(header, 34).unwrap_or(0));
        let rays_expected = usize::try_from(le_i16(header, 30).unwrap_or(0)).unwrap_or(0);
        let sweep_start = ymds(header, 12, gmt_offset_min).unwrap_or(volume_start);
        for (index, code) in types.iter().enumerate() {
            let declared = le_u16(&stream[index * DATA_HEADER..], 38).unwrap_or(u16::MAX);
            if declared != *code {
                return Err(OdimError::format(
                    context,
                    format!(
                        "ingest data header {index} declares type {declared}, the task mask says {code}"
                    ),
                ));
            }
        }
        // The first ray's byte offset in the record, as the record header
        // states it; past the data headers in any case.
        let ray_start = first_ray
            .checked_sub(BHDR)
            .filter(|offset| *offset >= headers_len && offset % 2 == 0)
            .unwrap_or(headers_len);
        let mut words = Words {
            bytes: &stream,
            at: ray_start,
        };

        // Decode every ray of every type.
        let mut rays: Vec<Vec<Option<Vec<u16>>>> = Vec::with_capacity(rays_expected);
        for _ in 0..rays_expected {
            if words.exhausted() {
                break;
            }
            let mut per_type = Vec::with_capacity(types.len());
            for code in &types {
                let one_byte = data_type(*code).is_none_or(|(_, kind)| kind.one_byte());
                let capacity = 6 + if one_byte { nbins.div_ceil(2) } else { nbins };
                per_type.push(words.ray(capacity));
            }
            rays.push(per_type);
        }
        // A ray every type reports missing has no azimuth and is dropped.
        let kept: Vec<usize> = (0..rays.len())
            .filter(|ray| {
                rays[*ray]
                    .iter()
                    .any(|data| data.as_ref().is_some_and(|d| d.len() >= 6))
            })
            .collect();
        if kept.is_empty() {
            continue;
        }
        let nrays = kept.len();
        let mut azimuth_deg = Vec::with_capacity(nrays);
        let mut elevation_deg = Vec::with_capacity(nrays);
        let mut seconds = Vec::with_capacity(nrays);
        for ray in &kept {
            let head = rays[*ray]
                .iter()
                .flatten()
                .find(|data| data.len() >= 6)
                .expect("kept rays have a header");
            azimuth_deg.push(mean_azimuth_deg(bin2(head[0]), bin2(head[2])));
            elevation_deg.push((bin2_signed(head[1]) + bin2_signed(head[3])) / 2.0);
            seconds.push(i64::from(head[5]));
        }

        let mut moments = Vec::new();
        for (index, code) in types.iter().enumerate() {
            let Some((quantity, kind)) = data_type(*code) else {
                continue;
            };
            let mut values = vec![f64::NAN; nrays * nbins];
            let mut codes = vec![censor::NODATA; nrays * nbins];
            for (row, ray) in kept.iter().enumerate() {
                let Some(data) = rays[*ray][index].as_ref().filter(|d| d.len() >= 6) else {
                    continue;
                };
                let ray_bins = usize::try_from(data[4] as i16).unwrap_or(0).min(nbins);
                for bin in 0..ray_bins {
                    let raw = if kind.one_byte() {
                        let Some(word) = data.get(6 + bin / 2) else {
                            break;
                        };
                        if bin % 2 == 0 { word & 0xff } else { word >> 8 }
                    } else {
                        let Some(word) = data.get(6 + bin) else { break };
                        *word
                    };
                    let (value, censor_code) = decode_gate(kind, raw, &scales);
                    values[row * nbins + bin] = value;
                    codes[row * nbins + bin] = censor_code;
                }
            }
            moments.push(moment(
                quantity,
                format!("IRIS/sweep{sweep_number}/type{code}"),
                nrays,
                nbins,
                values,
                codes,
            ));
        }
        if moments.is_empty() {
            continue;
        }
        let last = seconds.iter().copied().max().unwrap_or(0);
        sweeps.push(Sweep {
            index: usize::try_from(sweep_number).unwrap_or(sweeps.len() + 1),
            path: format!("IRIS/sweep{sweep_number}"),
            elevation_deg: fixed_angle,
            nrays,
            nbins,
            range_scale_m,
            range_start_m,
            a1gate: None,
            start_time: Some(sweep_start),
            end_time: Some(sweep_start + Duration::seconds(last)),
            nyquist,
            azimuth_deg,
            azimuth_source: AzimuthSource::MeasuredStartStopMean,
            ray_elevation_deg: Some(elevation_deg),
            ray_prf_hz: None,
            moments,
        });
    }
    if sweeps.is_empty() {
        return Err(err("no sweep carried a decodable moment".into()));
    }

    let label = site_name.trim().to_string();
    Ok(PolarVolume {
        object: "PVOL".to_string(),
        conventions: Some("IRIS-RAW".to_string()),
        version: None,
        source: Source {
            raw: format!("IRIS:{label}"),
            nod: (!label.is_empty()).then(|| label.clone()),
            wmo: None,
            wigos: None,
            place: (!label.is_empty()).then(|| label.clone()),
            rad: None,
        },
        nominal_time: Some(volume_start),
        site: Site {
            latitude_deg,
            longitude_deg,
            height_m,
        },
        system: SystemNotes {
            wavelength_cm: (wavelength_cm > 0.0).then_some(wavelength_cm),
            beamwidth_deg: getu32(ingest::BEAMWIDTH_H)
                .ok()
                .filter(|raw| *raw != 0)
                .map(|raw| f64::from(raw) * 360.0 / 4_294_967_296.0),
            system: Some("IRIS".to_string()),
            software: None,
            sw_version: None,
        },
        sweeps,
    })
}

/// The run-length-compressed word stream of one sweep.
struct Words<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl Words<'_> {
    fn exhausted(&self) -> bool {
        self.at + 2 > self.bytes.len()
    }

    fn next(&mut self) -> Option<u16> {
        let word = le_u16(self.bytes, self.at)?;
        self.at += 2;
        Some(word)
    }

    /// One ray, decompressed: `None` when IRIS marks the ray missing (an
    /// end-of-ray code with nothing before it). A code with the high bit set
    /// is followed by that many data words; any other code but 1 stands for
    /// that many zero words; 1 ends the ray.
    fn ray(&mut self, capacity: usize) -> Option<Vec<u16>> {
        let mut out = Vec::with_capacity(capacity);
        let mut first = true;
        loop {
            let code = self.next()?;
            if code == 1 {
                return if first && out.is_empty() {
                    None
                } else {
                    Some(out)
                };
            }
            first = false;
            if code & 0x8000 != 0 {
                for _ in 0..(code & 0x7fff) {
                    let word = self.next()?;
                    if out.len() < capacity {
                        out.push(word);
                    }
                }
            } else {
                let room = capacity.saturating_sub(out.len());
                out.extend(std::iter::repeat_n(0u16, usize::from(code).min(room)));
            }
        }
    }
}

/// IRIS `ymds_time`: seconds since midnight, milliseconds and flags, year,
/// month, day. Bit 11 of the millisecond word marks UTC; a time without it is
/// local and is moved to UTC with the site's declared offset.
fn ymds(block: &[u8], at: usize, gmt_offset_min: i16) -> Option<DateTime<Utc>> {
    let seconds = le_i32(block, at)?;
    let millis_word = le_u16(block, at + 4)?;
    let year = le_i16(block, at + 6)?;
    let month = le_i16(block, at + 8)?;
    let day = le_i16(block, at + 10)?;
    let date = NaiveDate::from_ymd_opt(
        i32::from(year),
        u32::try_from(month).ok()?,
        u32::try_from(day).ok()?,
    )?;
    let mut time = date.and_hms_opt(0, 0, 0)?.and_utc()
        + Duration::seconds(i64::from(seconds))
        + Duration::milliseconds(i64::from(millis_word & 0x3ff));
    if millis_word & 0x800 == 0 {
        time -= Duration::minutes(i64::from(gmt_offset_min));
    }
    Some(time)
}

fn bin2(raw: u16) -> f64 {
    f64::from(raw) * 360.0 / 65536.0
}

fn bin2_signed(raw: u16) -> f64 {
    let angle = bin2(raw);
    if angle > 180.0 { angle - 360.0 } else { angle }
}

fn bin4_signed(raw: u32) -> f64 {
    let angle = f64::from(raw) * 360.0 / 4_294_967_296.0;
    if angle > 180.0 { angle - 360.0 } else { angle }
}

fn ascii(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes)
        .trim_matches(char::from(0))
        .trim()
        .to_string()
}

fn le_i16(bytes: &[u8], at: usize) -> Option<i16> {
    bytes
        .get(at..at + 2)
        .map(|b| i16::from_le_bytes([b[0], b[1]]))
}

fn le_u16(bytes: &[u8], at: usize) -> Option<u16> {
    bytes
        .get(at..at + 2)
        .map(|b| u16::from_le_bytes([b[0], b[1]]))
}

fn le_i32(bytes: &[u8], at: usize) -> Option<i32> {
    bytes
        .get(at..at + 4)
        .map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]))
}

fn le_u32(bytes: &[u8], at: usize) -> Option<u32> {
    bytes
        .get(at..at + 4)
        .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scales() -> Scales {
        Scales {
            extended_nyquist: 6.6625,
            single_nyquist: 6.6625,
            wavelength_cm: 5.33,
        }
    }

    #[test]
    fn reserved_raw_values_are_classified_not_decoded() {
        let s = scales();
        assert_eq!(decode_gate(Decode::Db1, 0, &s).1, censor::UNDETECT);
        assert_eq!(decode_gate(Decode::Db1, 255, &s).1, censor::NODATA);
        assert_eq!(decode_gate(Decode::Db1, 64, &s), (0.0, censor::MEASURED));
        assert_eq!(decode_gate(Decode::Centi2, 0, &s).1, censor::UNDETECT);
        assert_eq!(decode_gate(Decode::Centi2, u16::MAX, &s).1, censor::NODATA);
        // One-byte velocity uses the whole code range for data.
        let (v, code) = decode_gate(Decode::Vel1, 255, &s);
        assert_eq!(code, censor::MEASURED);
        assert!((v - 6.6625).abs() < 1e-9);
    }

    #[test]
    fn a_compressed_ray_expands_runs_of_zeros_and_data() {
        // 2 data words, 3 zeros, 1 data word, end.
        let words: [u16; 8] = [0x8002, 7, 8, 3, 0x8001, 9, 1, 0];
        let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let mut stream = Words {
            bytes: &bytes,
            at: 0,
        };
        assert_eq!(stream.ray(16), Some(vec![7, 8, 0, 0, 0, 9]));
    }

    #[test]
    fn an_immediate_end_code_is_a_missing_ray() {
        let bytes: Vec<u8> = [1u16, 0x8001, 5, 1]
            .iter()
            .flat_map(|w| w.to_le_bytes())
            .collect();
        let mut stream = Words {
            bytes: &bytes,
            at: 0,
        };
        assert_eq!(stream.ray(8), None);
        assert_eq!(stream.ray(8), Some(vec![5]));
    }
}
