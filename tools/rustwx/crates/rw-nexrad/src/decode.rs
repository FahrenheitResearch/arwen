//! Volume -> sweep pack.
//!
//! Everything geometric stays in floating-point degrees and metres exactly
//! as the RDA reported it; the pack is a transcription, not an
//! interpretation.  The only reductions are the three the caller asked for
//! — moment selection, an elevation ceiling, and a range ceiling — and all
//! three are counted into the metadata so a thin pack is never mistaken for
//! a thin volume.

use std::error::Error;
use std::path::Path;

use chrono::{DateTime, Duration, NaiveDate, TimeZone, Utc};
use wx_radar::level2::{Level2File, Level2Sweep, VolSite};
use wx_radar::products::RadarProduct;

use crate::pack::{
    self, ArrayEntry, DecodeParams, Framing, MomentEntry, PackMeta, PayloadBuilder, SiteEntry,
    SweepEntry, VolumeEntry, SWEEPS_SCHEMA, SWEEPS_SCHEMA_CENSOR,
};
use crate::s3::{boxed_error, hex_sha256, iso8601};

/// Volume start time from the Archive-II header: `volume_date` counts days
/// since 1970-01-01 with the epoch day numbered 1, `volume_time` is
/// milliseconds past midnight UTC.
pub fn volume_time(volume_date: u16, volume_time_ms: u32) -> Option<DateTime<Utc>> {
    let epoch = NaiveDate::from_ymd_opt(1970, 1, 1)?;
    let date = epoch.checked_add_signed(Duration::days(volume_date as i64 - 1))?;
    let seconds = (volume_time_ms / 1000) as i64;
    let naive = date.and_hms_opt(0, 0, 0)? + Duration::seconds(seconds);
    Some(Utc.from_utc_datetime(&naive))
}

/// Coordinates for a site, and the honest record of where they came from.
#[derive(Debug, Clone)]
pub struct SiteFix {
    pub id: String,
    pub name: String,
    pub lat_deg: f64,
    pub lon_deg: f64,
    pub alt_m: f64,
    pub source: String,
}

/// The vendored site table's placeholder for "elevation not filled in".
///
/// 130 of its 141 entries carry exactly this value, including KUEX at
/// Hastings (really 602 m) and KTWX at Topeka (really 417 m); the eleven
/// that differ are the ones somebody typed in.  It is a sentinel, not a
/// measurement, and no operational WSR-88D antenna sits at exactly 0.000 m
/// MSL — even KBYX at Key West is a few metres up.
pub const SITE_ELEVATION_UNSET_M: f64 = 0.0;

/// Resolve a site's coordinates: the caller's explicit override wins, then
/// the vendored NEXRAD table.  An unknown site with no override is a hard
/// error — a superob placed at a guessed radar position is worse than no
/// superob at all.
///
/// A site whose *elevation* is the table's unset sentinel is the same hard
/// error, for the same reason and with more consequence.  The antenna
/// height is the ray origin: every gate's height above sea level is
/// computed from it, so a radar placed 600 m too low puts its entire volume
/// 600 m too low in the model column, systematically, in the direction that
/// pushes a mid-level echo into the boundary layer.  Latitude and longitude
/// were the only fields the table's silence was ever noticed in; the
/// altitude column fails identically and had nothing stopping it.
pub fn resolve_site(
    station_id: &str,
    override_fix: Option<(f64, f64, f64)>,
) -> Result<SiteFix, Box<dyn Error>> {
    resolve_site_from(station_id, override_fix, None)
}

/// Resolve a site with the volume's own Message-31 VOL block in hand.
///
/// Precedence: an explicit `--site-latlon` override, then the volume's
/// own VOL block, then the vendored table.  The VOL block outranks the
/// table because it is the radar's own survey riding inside the volume:
/// latitude, longitude, site ground elevation AND the feedhorn height the
/// table never carries, so its `antenna_height_m()` is the beam origin
/// with no downstream addition.  The table's role shrinks to volumes old
/// or damaged enough to carry no VOL block, where its elevation rules
/// (placeholder refused, populated value documented as ground-only)
/// still apply unchanged.
///
/// When both are present and the table disagrees with the volume by more
/// than ~0.05 degrees, the disagreement is recorded in the fix's
/// `source` string — that is how the KPBZ longitude transcription error
/// (17 km) surfaces on real volumes instead of silently misplacing
/// superobs.
pub fn resolve_site_from(
    station_id: &str,
    override_fix: Option<(f64, f64, f64)>,
    vol_site: Option<&VolSite>,
) -> Result<SiteFix, Box<dyn Error>> {
    if let Some((lat, lon, alt)) = override_fix {
        if !(-90.0..=90.0).contains(&lat) || !(-180.0..=360.0).contains(&lon) {
            return Err(boxed_error(format!(
                "--site-latlon out of range: lat {lat}, lon {lon}"
            )));
        }
        return Ok(SiteFix {
            id: station_id.to_string(),
            name: station_id.to_string(),
            lat_deg: lat,
            lon_deg: lon,
            alt_m: alt,
            source: "cli-override".to_string(),
        });
    }
    if let Some(vol) = vol_site {
        let mut source = "message31-vol-block".to_string();
        if let Some(site) = wx_radar::sites::find_site(station_id) {
            let dlat = (site.lat - f64::from(vol.latitude_deg)).abs();
            let dlon = (site.lon - f64::from(vol.longitude_deg)).abs();
            if dlat.max(dlon) > 0.05 {
                source = format!(
                    "message31-vol-block (vendored table disagrees by                      {dlat:.4} deg lat / {dlon:.4} deg lon)"
                );
            }
        }
        return Ok(SiteFix {
            id: station_id.to_string(),
            name: station_id.to_string(),
            lat_deg: f64::from(vol.latitude_deg),
            lon_deg: f64::from(vol.longitude_deg),
            alt_m: vol.antenna_height_m(),
            source,
        });
    }
    match wx_radar::sites::find_site(station_id) {
        Some(site) if site.elevation == SITE_ELEVATION_UNSET_M => Err(boxed_error(format!(
            "the vendored NEXRAD table has no antenna elevation for site {:?} ({}): it carries \
             the unset placeholder {SITE_ELEVATION_UNSET_M} m, as 130 of its 141 entries do. \
             The antenna height is the ray origin, so accepting it would place every gate in \
             the volume too low by the site's real elevation. Volumes whose radials carry a \
             Message-31 VOL block resolve themselves (site plus feedhorn, no table needed), \
             so this refusal means the volume lacked one. Pass --site-latlon \
             LAT,LON,ALT_M with the antenna height above mean sea level (site elevation plus \
             feedhorn height AGL -- nothing downstream adds the feedhorn)",
            site.id, site.name
        ))),
        Some(site) => Ok(SiteFix {
            id: site.id,
            name: site.name,
            lat_deg: site.lat,
            lon_deg: site.lon,
            alt_m: site.elevation,
            source: "wx-radar-site-table".to_string(),
        }),
        None => Err(boxed_error(format!(
            "no coordinates for site {station_id:?} in the vendored NEXRAD table; pass \
             --site-latlon LAT,LON,ALT_M to place it explicitly"
        ))),
    }
}

/// The gate geometry one moment uses across a whole sweep.
#[derive(Debug, Clone, PartialEq)]
pub struct MomentLayout {
    pub product: RadarProduct,
    /// Gates retained after the range ceiling.
    pub gates: usize,
    /// Gates the radials themselves declared, before the ceiling.
    pub declared_gates: usize,
    pub first_gate_range_m: f64,
    pub gate_size_m: f64,
}

/// Agree one sweep's per-moment gate geometry, or name the radial that broke it.
///
/// The pack stores a sweep as a rectangle, so the range axis is a property
/// of the sweep rather than of its first radial.  Taking the geometry from
/// the first radial that carries a moment and copying every later radial
/// into that rectangle is how a gate the RDA placed at 2,375 m gets
/// published at 2,125 m: the disagreement is consumed by the copy and the
/// pack keeps only the canonical axis, so nothing downstream — not the
/// superob, not the receipt, not the grid file — can discover that a value
/// moved.  A silent 250 m displacement of a velocity gate is worse than no
/// gate at all, so the geometry is *agreed* here instead: every radial
/// carrying the moment must declare the same gate count, first-gate range
/// and gate interval, and one that does not refuses the volume by name.
///
/// A radial that simply does not carry the moment is not a disagreement —
/// split cuts legitimately omit one — and its row stays NaN.
///
/// Returns the layouts in first-seen order and the gate samples the range
/// ceiling trimmed off the far end.
pub fn agree_sweep_layouts(
    sweep: &Level2Sweep,
    wanted: &[RadarProduct],
    max_range_km: f64,
) -> Result<(Vec<MomentLayout>, usize), String> {
    // (product, gate_count, first_gate_range, gate_size, radial that set it)
    let mut agreed: Vec<(RadarProduct, u16, u16, u16, usize)> = Vec::new();
    for (row, radial) in sweep.radials.iter().enumerate() {
        for moment in &radial.moments {
            if !wanted.contains(&moment.product) {
                continue;
            }
            if moment.data.len() != moment.gate_count as usize {
                return Err(format!(
                    "corrupt Level-II volume: sweep {} radial {} moment {} declares {} gates \
                     but decoded {} samples",
                    sweep.sweep_index,
                    row,
                    moment.product.short_name(),
                    moment.gate_count,
                    moment.data.len()
                ));
            }
            match agreed.iter().find(|(product, ..)| *product == moment.product) {
                None => agreed.push((
                    moment.product,
                    moment.gate_count,
                    moment.first_gate_range,
                    moment.gate_size,
                    row,
                )),
                Some(&(_, gates, first, size, first_row)) => {
                    if (moment.gate_count, moment.first_gate_range, moment.gate_size)
                        != (gates, first, size)
                    {
                        return Err(format!(
                            "sweep {} moment {} changes gate geometry mid-sweep: radial \
                             {first_row} declares {gates} gates from {first} m by {size} m, \
                             radial {row} declares {} gates from {} m by {} m. The pack stores \
                             a sweep as one rectangle, so the later radial's gates could only \
                             be published at the earlier radial's ranges -- refusing the volume \
                             rather than misplacing them",
                            sweep.sweep_index,
                            moment.product.short_name(),
                            moment.gate_count,
                            moment.first_gate_range,
                            moment.gate_size
                        ));
                    }
                }
            }
        }
    }

    let radial_count = sweep.radials.len();
    let mut layouts = Vec::new();
    let mut trimmed_gates = 0usize;
    for (product, declared, first, size, _) in agreed {
        let gate_size = size as f64;
        let first_m = first as f64;
        let mut gates = declared as usize;
        if gate_size > 0.0 && max_range_km.is_finite() {
            let limit_m = max_range_km * 1000.0;
            let allowed = if limit_m <= first_m {
                0
            } else {
                (((limit_m - first_m) / gate_size).floor() as usize).saturating_add(1)
            };
            if allowed < gates {
                trimmed_gates += (gates - allowed) * radial_count;
                gates = allowed;
            }
        }
        if gates == 0 {
            continue;
        }
        layouts.push(MomentLayout {
            product,
            gates,
            declared_gates: declared as usize,
            first_gate_range_m: first_m,
            gate_size_m: gate_size,
        });
    }
    Ok((layouts, trimmed_gates))
}

pub struct DecodeRequest<'a> {
    pub volume_path: &'a Path,
    /// The message stream to decode: the file's bytes, or their expansion
    /// when the archived key was gzipped.
    pub raw: &'a [u8],
    /// The file exactly as it arrived on disk.  Equal to `raw` unless the
    /// volume was gzip-wrapped.  The pack's `volume.bytes` and
    /// `volume.sha256` are taken over *these*, because they are what the S3
    /// listing stated and what a re-download can be checked against; the
    /// expansion is an interpretation and has no independent authority.
    pub source: &'a [u8],
    pub framing: Framing,
    pub moments: Vec<RadarProduct>,
    pub max_range_km: f64,
    pub max_elevation_deg: f64,
    pub site_override: Option<(f64, f64, f64)>,
    /// Emit a `|u1` censor plane beside every moment plane, and declare the
    /// pack [`crate::pack::SWEEPS_SCHEMA_CENSOR`].
    ///
    /// Off by default, and the default path is byte-identical to the one
    /// that existed before the flag: no extra arrays, no extra metadata
    /// keys, the same schema string.
    pub censor_flags: bool,
}

/// Build the pack metadata and payload for one validated volume.
pub fn build_pack(request: &DecodeRequest<'_>) -> Result<(PackMeta, Vec<u8>), Box<dyn Error>> {
    // Strict: the observation front door refuses a volume that contradicts
    // itself rather than publishing the part of it that happened to decode.
    let file = Level2File::parse_strict(request.raw).map_err(boxed_error)?;
    crate::pack::validate_decoded(&file)?;
    let site = resolve_site_from(&file.station_id, request.site_override, file.vol_site.as_ref())?;
    let valid_time = volume_time(file.volume_date, file.volume_time).ok_or_else(|| {
        boxed_error(format!(
            "corrupt Level-II volume: volume date {} / time {} is not a calendar instant",
            file.volume_date, file.volume_time
        ))
    })?;

    let mut builder = PayloadBuilder::new();
    let mut sweeps = Vec::new();
    let mut dropped_sweeps = 0usize;
    let mut dropped_moments = 0usize;
    let mut trimmed_gates = 0usize;

    for sweep in &file.sweeps {
        if sweep.elevation_angle as f64 > request.max_elevation_deg {
            dropped_sweeps += 1;
            continue;
        }
        if sweep.radials.is_empty() {
            dropped_sweeps += 1;
            continue;
        }
        let radial_count = sweep.radials.len();
        let azimuth: Vec<f32> = sweep.radials.iter().map(|r| r.azimuth).collect();
        let elevation: Vec<f32> = sweep.radials.iter().map(|r| r.elevation).collect();

        // Which moments does this sweep actually carry, and with what
        // geometry?  Every radial carrying a moment must agree on the gate
        // layout; a disagreement refuses the volume rather than publishing
        // the later radial's gates at the earlier radial's ranges.
        let (layouts, trimmed) =
            agree_sweep_layouts(sweep, &request.moments, request.max_range_km)
                .map_err(boxed_error)?;
        trimmed_gates += trimmed;
        let selected: usize = sweep
            .radials
            .iter()
            .flat_map(|r| r.moments.iter())
            .filter(|m| request.moments.contains(&m.product))
            .count();
        let present: usize = sweep.radials.iter().map(|r| r.moments.len()).sum();
        dropped_moments += present.saturating_sub(selected);

        if layouts.is_empty() {
            dropped_sweeps += 1;
            continue;
        }

        let azimuth_key = builder.push_f32(&azimuth, vec![radial_count]);
        let elevation_key = builder.push_f32(&elevation, vec![radial_count]);

        let mut moment_entries = Vec::new();
        for layout in layouts {
            let gates = layout.gates;
            let mut flat = vec![f32::NAN; radial_count * gates];
            // The rectangle starts as NOT_COLLECTED everywhere, which is the
            // truth for a row a split cut never filled.  A radial that does
            // carry the moment overwrites its whole row with the decoder's
            // own codes, so the only cells left saying NOT_COLLECTED are the
            // ones the NaN fill above created.
            let mut censor = if request.censor_flags {
                vec![pack::censor_plane::NOT_COLLECTED; radial_count * gates]
            } else {
                Vec::new()
            };
            for (row, radial) in sweep.radials.iter().enumerate() {
                let Some(moment) = radial.moments.iter().find(|m| m.product == layout.product)
                else {
                    continue;
                };
                // `agree_sweep_layouts` proved every carrying radial declares
                // `declared_gates >= gates` samples, so this is a copy at the
                // agreed ranges, never a trim from an unknown origin.
                flat[row * gates..row * gates + gates].copy_from_slice(&moment.data[..gates]);
                if request.censor_flags {
                    // The decoder keeps `censor` exactly as long as `data`,
                    // so the same span is in bounds for both.
                    censor[row * gates..row * gates + gates]
                        .copy_from_slice(&moment.censor[..gates]);
                }
            }
            let key = builder.push_f32(&flat, vec![radial_count, gates]);
            let censor_key = request
                .censor_flags
                .then(|| builder.push_u8(&censor, vec![radial_count, gates]));
            moment_entries.push(MomentEntry {
                product: layout.product.short_name().to_string(),
                unit: layout.product.unit().to_string(),
                gate_count: gates,
                first_gate_range_m: layout.first_gate_range_m,
                gate_size_m: layout.gate_size_m,
                array: key,
                censor_array: censor_key,
            });
        }

        sweeps.push(SweepEntry {
            sweep_index: sweep.sweep_index,
            elevation_number: sweep.elevation_number,
            elevation_angle_deg: sweep.elevation_angle as f64,
            nyquist_velocity_ms: sweep.nyquist_velocity.map(|v| v as f64),
            nyquist_radials_disagree: sweep.nyquist_radials_disagree,
            start_status: sweep.start_status,
            end_status: sweep.end_status,
            cut_sector: sweep.cut_sector,
            complete: matches!(sweep.start_status, 0 | 3 | 5)
                && matches!(sweep.end_status, 2 | 4),
            radial_count,
            azimuth_array: azimuth_key,
            elevation_array: elevation_key,
            moments: moment_entries,
        });
    }

    if sweeps.is_empty() {
        return Err(boxed_error(format!(
            "volume decoded to {} sweeps but none survived the filter (moments {:?}, \
             max elevation {} deg, max range {} km)",
            file.sweeps.len(),
            request
                .moments
                .iter()
                .map(|p| p.short_name())
                .collect::<Vec<_>>(),
            request.max_elevation_deg,
            request.max_range_km
        )));
    }

    let (payload, arrays) = builder.finish();
    let meta = PackMeta {
        schema: if request.censor_flags {
            SWEEPS_SCHEMA_CENSOR.to_string()
        } else {
            SWEEPS_SCHEMA.to_string()
        },
        status: "READY".to_string(),
        site: SiteEntry {
            id: site.id,
            name: site.name,
            lat_deg: site.lat_deg,
            lon_deg: site.lon_deg,
            alt_m: site.alt_m,
            source: site.source,
        },
        volume: VolumeEntry {
            file: request
                .volume_path
                .file_name()
                .map(|name| name.to_string_lossy().to_string())
                .unwrap_or_else(|| request.volume_path.to_string_lossy().to_string()),
            bytes: request.source.len(),
            sha256: hex_sha256(request.source),
            station_id: file.station_id.clone(),
            valid_time: iso8601(valid_time),
            volume_date: file.volume_date,
            volume_time_ms: file.volume_time,
            framing: request.framing.clone(),
        },
        params: DecodeParams {
            moments: request
                .moments
                .iter()
                .map(|p| p.short_name().to_string())
                .collect(),
            max_range_km: request.max_range_km,
            max_elevation_deg: request.max_elevation_deg,
            censor_flags: request.censor_flags,
        },
        sweeps,
        arrays: arrays as std::collections::BTreeMap<String, ArrayEntry>,
        payload_bytes: payload.len(),
        content_sha256: hex_sha256(&payload),
        dropped_sweeps,
        dropped_moments,
        trimmed_gates,
    };
    Ok((meta, payload))
}

#[cfg(test)]
mod tests {
    use super::*;
    use wx_radar::level2::{MomentData, RadialData};

    fn moment(product: RadarProduct, gate_count: u16, first: u16, size: u16) -> MomentData {
        MomentData {
            product,
            gate_count,
            first_gate_range: first,
            gate_size: size,
            data: (0..gate_count).map(|g| g as f32).collect(),
            censor: vec![wx_radar::level2::censor::MEASURED; gate_count as usize],
        }
    }

    fn sweep_of(rows: Vec<Vec<MomentData>>) -> Level2Sweep {
        Level2Sweep {
            elevation_number: 1,
            elevation_angle: 0.5,
            nyquist_velocity: Some(32.0),
            nyquist_radials_disagree: false,
            sweep_index: 4,
            start_status: 3,
            end_status: 2,
            cut_sector: 0,
            radials: rows
                .into_iter()
                .enumerate()
                .map(|(row, moments)| RadialData {
                    azimuth: row as f32,
                    elevation: 0.5,
                    azimuth_spacing: 1.0,
                    nyquist_velocity: Some(32.0),
                    radial_status: if row == 0 { 3 } else { 1 },
                    moments,
                })
                .collect(),
        }
    }

    const REF: RadarProduct = RadarProduct::Reflectivity;
    const VEL: RadarProduct = RadarProduct::Velocity;

    #[test]
    fn a_radial_that_changes_gate_geometry_refuses_the_sweep() {
        // The audit's own case: 2,125 m and 2,375 m at the same 250 m
        // spacing.  Publishing radial 1's first gate at radial 0's first
        // range displaces every one of its values by 250 m.
        let sweep = sweep_of(vec![
            vec![moment(REF, 3, 2125, 250)],
            vec![moment(REF, 3, 2375, 250)],
        ]);
        let err = agree_sweep_layouts(&sweep, &[REF], f64::INFINITY).unwrap_err();
        assert!(err.contains("changes gate geometry mid-sweep"), "{err}");
        assert!(err.contains("2125") && err.contains("2375"), "{err}");
        assert!(err.contains("radial 0") && err.contains("radial 1"), "{err}");
        assert!(err.contains("sweep 4"), "{err}");

        // A different gate interval and a different gate count are the same
        // refusal, not a pad or a trim.
        for later in [moment(REF, 3, 2125, 1000), moment(REF, 7, 2125, 250)] {
            let sweep = sweep_of(vec![vec![moment(REF, 3, 2125, 250)], vec![later]]);
            assert!(agree_sweep_layouts(&sweep, &[REF], f64::INFINITY)
                .unwrap_err()
                .contains("changes gate geometry mid-sweep"));
        }
    }

    #[test]
    fn agreeing_radials_yield_one_layout_and_a_uniform_range_ceiling() {
        let sweep = sweep_of(vec![
            vec![moment(REF, 10, 2125, 250), moment(VEL, 8, 2125, 250)],
            vec![moment(REF, 10, 2125, 250), moment(VEL, 8, 2125, 250)],
        ]);
        let (layouts, trimmed) = agree_sweep_layouts(&sweep, &[REF, VEL], f64::INFINITY).unwrap();
        assert_eq!(layouts.len(), 2);
        assert_eq!(layouts[0].product, REF);
        assert_eq!(layouts[0].gates, 10);
        assert_eq!(layouts[0].first_gate_range_m, 2125.0);
        assert_eq!(layouts[0].gate_size_m, 250.0);
        assert_eq!(trimmed, 0);

        // Ceiling at 3 km keeps gates centred at 2125, 2375, 2625, 2875 m.
        let (layouts, trimmed) = agree_sweep_layouts(&sweep, &[REF], 3.0).unwrap();
        assert_eq!(layouts[0].gates, 4);
        assert_eq!(layouts[0].declared_gates, 10);
        assert_eq!(trimmed, (10 - 4) * 2);
    }

    #[test]
    fn a_radial_that_simply_omits_the_moment_is_not_a_disagreement() {
        // Split cuts legitimately carry velocity on only part of a sweep.
        let sweep = sweep_of(vec![
            vec![moment(REF, 4, 2125, 250), moment(VEL, 4, 2125, 250)],
            vec![moment(REF, 4, 2125, 250)],
        ]);
        let (layouts, _) = agree_sweep_layouts(&sweep, &[REF, VEL], f64::INFINITY).unwrap();
        assert_eq!(layouts.len(), 2);
        assert_eq!(layouts[1].product, VEL);
    }

    #[test]
    fn a_moment_whose_sample_count_contradicts_its_header_is_refused() {
        let mut short = moment(REF, 6, 2125, 250);
        short.data.truncate(4);
        let sweep = sweep_of(vec![vec![short]]);
        let err = agree_sweep_layouts(&sweep, &[REF], f64::INFINITY).unwrap_err();
        assert!(err.contains("declares 6 gates but decoded 4 samples"), "{err}");
    }

    #[test]
    fn an_unwanted_moment_never_constrains_the_geometry() {
        // VEL is not selected, so its disagreement is irrelevant.
        let sweep = sweep_of(vec![
            vec![moment(REF, 4, 2125, 250), moment(VEL, 4, 2125, 250)],
            vec![moment(REF, 4, 2125, 250), moment(VEL, 9, 9999, 1000)],
        ]);
        let (layouts, _) = agree_sweep_layouts(&sweep, &[REF], f64::INFINITY).unwrap();
        assert_eq!(layouts.len(), 1);
        assert_eq!(layouts[0].product, REF);
    }

    /// A whole Archive-II volume carrying exactly one Message-31 radial.
    ///
    /// `bad_pointer` aims the moment block past the end of its own radial.
    /// The byte is still inside the volume, so the outer LDM framing and
    /// the old whole-volume pointer bound both accept it; only the
    /// per-radial envelope can tell that following it would read the next
    /// radial's bytes as this one's.
    fn one_radial_volume(bad_pointer: bool) -> Vec<u8> {
        one_radial_volume_words(bad_pointer, [100u8, 101, 102, 103])
    }

    /// The same fixture with the four gate words chosen by the caller, so a
    /// test can place raw 0 (below threshold) and raw 1 (range folded)
    /// exactly where it wants them.
    fn one_radial_volume_words(bad_pointer: bool, words: [u8; 4]) -> Vec<u8> {
        fn constant(name: &[u8; 4], lrtup: u16, body: &[u8]) -> Vec<u8> {
            let mut block = Vec::from(&name[..]);
            block.extend_from_slice(&lrtup.to_be_bytes());
            block.extend_from_slice(body);
            block.resize(lrtup as usize, 0);
            block
        }
        // VOL and ELV are the verbatim constant blocks of KTLX
        // 2019-05-20 00:00:34Z -- generic format version 2.0 in 44 bytes,
        // 35.33336 N / 97.27776 W / site 370 m / VCP 32.  A hand-built VOL
        // block used to stand here and it declared version 1.0, a layout
        // nothing has read; `Level2File::parse_vol_block` now refuses that,
        // and this fixture exists to exercise the strict parser rather than
        // to find out what it will tolerate.  See
        // `wx_radar::level2` `REAL_VOL_V2`/`REAL_ELV` for the same bytes and
        // the survey they came from.
        let real_vol_v2: [u8; 44] = [
            0x52, 0x56, 0x4f, 0x4c, 0x00, 0x2c, 0x02, 0x00, 0x42, 0x0d, 0x55, 0x5d, 0xc2, 0xc2,
            0x8e, 0x37, 0x01, 0x72, 0x00, 0x13, 0xc2, 0x34, 0x4a, 0x62, 0x43, 0x81, 0x2e, 0x7c,
            0x43, 0x6f, 0x43, 0x73, 0x3d, 0xe5, 0x4f, 0x09, 0x42, 0x70, 0x00, 0x00, 0x00, 0x20,
            0x00, 0x01,
        ];
        let real_elv: [u8; 12] = [
            0x52, 0x45, 0x4c, 0x56, 0x00, 0x0c, 0xff, 0xf4, 0xc2, 0x2f, 0x40, 0x00,
        ];
        let mut rad_body = vec![0u8; 10];
        rad_body.extend_from_slice(&2384u16.to_be_bytes()); // 23.84 m/s
        let mut moment = Vec::from(&b"DREF"[..]);
        moment.extend_from_slice(&0u32.to_be_bytes());
        moment.extend_from_slice(&4u16.to_be_bytes()); // gates
        moment.extend_from_slice(&2125u16.to_be_bytes());
        moment.extend_from_slice(&250u16.to_be_bytes());
        moment.extend_from_slice(&0u16.to_be_bytes());
        moment.push(0);
        moment.push(0);
        moment.extend_from_slice(&8u16.to_be_bytes());
        moment.extend_from_slice(&2.0f32.to_be_bytes());
        moment.extend_from_slice(&66.0f32.to_be_bytes());
        moment.extend(words);

        let _ = &constant;
        let blocks = vec![
            real_vol_v2.to_vec(),
            real_elv.to_vec(),
            constant(b"RRAD", 28, &rad_body),
            moment,
        ];
        let header_bytes = 32 + 4 * blocks.len();
        let mut pointers = Vec::new();
        let mut running = header_bytes as u32;
        for block in &blocks {
            pointers.push(running);
            running += block.len() as u32;
        }
        if bad_pointer {
            *pointers.last_mut().unwrap() += 512;
        }

        let mut msg31 = Vec::from(&b"KTLX"[..]);
        msg31.extend_from_slice(&0u32.to_be_bytes());
        msg31.extend_from_slice(&20663u16.to_be_bytes());
        msg31.extend_from_slice(&1u16.to_be_bytes());
        msg31.extend_from_slice(&90.0f32.to_be_bytes());
        msg31.push(0); // compression
        msg31.push(0); // spare
        msg31.extend_from_slice(&(running as u16).to_be_bytes()); // radial length
        msg31.push(1);
        msg31.push(3); // radial status: start of volume
        msg31.push(1); // elevation number
        msg31.push(0); // cut sector
        msg31.extend_from_slice(&0.5f32.to_be_bytes());
        msg31.push(0);
        msg31.push(0);
        msg31.extend_from_slice(&(blocks.len() as u16).to_be_bytes());
        for pointer in &pointers {
            msg31.extend_from_slice(&pointer.to_be_bytes());
        }
        for block in &blocks {
            msg31.extend_from_slice(block);
        }

        let mut message = vec![0u8; 12];
        message.extend_from_slice(&(((16 + msg31.len()) / 2) as u16).to_be_bytes());
        message.push(0);
        message.push(31);
        message.extend_from_slice(&[0u8; 12]);
        message.extend_from_slice(&msg31);
        // 512 bytes of slack so the bad pointer still lands in the volume.
        message.resize(message.len() + 1024, 0);

        let mut raw = Vec::from(&b"AR2V0006."[..]);
        raw.resize(24, 0);
        raw[14..16].copy_from_slice(&20663u16.to_be_bytes());
        raw[16..20].copy_from_slice(&72_196_232u32.to_be_bytes());
        raw[20..24].copy_from_slice(b"KTLX");
        // The pre-2016 shape: messages straight after the volume header,
        // no LDM block table.  This fixture used to prepend a length word
        // and store the message uncompressed, which is neither shape any
        // real volume has -- an LDM block is always a bzip2 stream.  Naming
        // it as the unframed layout makes it a volume the archive actually
        // contains, and exercises the `.gz`-era path end to end.
        raw.extend_from_slice(&message);
        raw
    }

    fn pack_request(raw: &[u8]) -> DecodeRequest<'_> {
        DecodeRequest {
            volume_path: Path::new("KTLX20260728_200316_V06"),
            raw,
            source: raw,
            framing: Framing {
                magic: "AR2V0006".to_string(),
                layout: crate::pack::layout::UNCOMPRESSED_MESSAGES.to_string(),
                gzip_wrapped: false,
                block_count: 0,
                bzip2_block_count: 0,
                message_count: 1,
                bytes: raw.len(),
                source_bytes: raw.len(),
            },
            moments: vec![REF],
            max_range_km: 250.0,
            max_elevation_deg: 20.0,
            site_override: Some((35.3331, -97.2778, 370.0)),
            censor_flags: false,
        }
    }

    #[test]
    fn the_pack_builder_reads_a_volume_through_the_strict_parser() {
        // Conforming: the one radial becomes one sweep with four gates.
        let raw = one_radial_volume(false);
        let (meta, _payload) = build_pack(&pack_request(&raw)).unwrap();
        assert_eq!(meta.sweeps.len(), 1);
        assert!((meta.sweeps[0].nyquist_velocity_ms.unwrap() - 23.84).abs() < 1e-4);
        assert_eq!(meta.sweeps[0].moments[0].gate_count, 4);
        assert_eq!(meta.sweeps[0].moments[0].first_gate_range_m, 2125.0);

        // One pointer moved past the end of its radial, still inside the
        // volume: a refusal, not a sweep with three moments instead of four.
        let raw = one_radial_volume(true);
        let err = build_pack(&pack_request(&raw)).unwrap_err().to_string();
        assert!(err.contains("outside its own"), "{err}");
        assert!(err.contains("neighbouring radial"), "{err}");
    }

    /// Raw 0 (below threshold), raw 1 (range folded), and two measurements.
    fn censored_request(raw: &[u8]) -> DecodeRequest<'_> {
        let mut request = pack_request(raw);
        request.censor_flags = true;
        request
    }

    #[test]
    fn the_default_pack_says_nothing_at_all_about_censoring() {
        // The do-no-harm contract, pinned in the metadata rather than
        // asserted in a commit message: with the flag off, a pack has the
        // v1 schema string, no censor array on any moment, and no censoring
        // keys anywhere in its serialized JSON.  The last one is what keeps
        // every already-committed pack digest reproducible.
        let raw = one_radial_volume_words(false, [0, 1, 102, 103]);
        let (meta, payload) = build_pack(&pack_request(&raw)).unwrap();
        assert_eq!(meta.schema, crate::pack::SWEEPS_SCHEMA);
        assert!(!meta.params.censor_flags);
        assert!(meta.sweeps[0].moments[0].censor_array.is_none());
        assert_eq!(meta.arrays.len(), 3); // azimuth, elevation, one moment
        for entry in meta.arrays.values() {
            assert_eq!(entry.dtype, "<f4");
        }
        let json = serde_json::to_string(&meta).unwrap();
        assert!(!json.contains("censor"), "{json}");

        // And the gates themselves are what they always were: raw 0 and
        // raw 1 both NaN, in the same plane, indistinguishable.
        let values: Vec<f32> = payload
            .chunks_exact(4)
            .map(|word| f32::from_le_bytes([word[0], word[1], word[2], word[3]]))
            .collect();
        let gates = &values[values.len() - 4..];
        assert!(gates[0].is_nan() && gates[1].is_nan());
        assert_eq!(gates[2], 18.0);
        assert_eq!(gates[3], 18.5);
    }

    #[test]
    fn the_censor_plane_names_each_gates_reason_and_rides_a_v2_schema() {
        let raw = one_radial_volume_words(false, [0, 1, 102, 103]);
        let (meta, payload) = build_pack(&censored_request(&raw)).unwrap();
        assert_eq!(meta.schema, crate::pack::SWEEPS_SCHEMA_CENSOR);
        assert!(meta.params.censor_flags);

        let moment = &meta.sweeps[0].moments[0];
        let key = moment.censor_array.clone().expect("censor plane");
        let entry = &meta.arrays[&key];
        assert_eq!(entry.dtype, "|u1");
        assert_eq!(entry.shape, vec![1, 4]);
        assert_eq!(entry.bytes, 4);

        // The moment plane is untouched by the flag: same values, same
        // dtype, same shape.  Only the explanation is new.
        let values = &meta.arrays[&moment.array];
        assert_eq!(values.dtype, "<f4");
        assert_eq!(values.shape, vec![1, 4]);

        let codes = &payload[entry.offset..entry.offset + entry.bytes];
        use wx_radar::level2::censor;
        assert_eq!(
            codes,
            [
                censor::BELOW_THRESHOLD,
                censor::RANGE_FOLDED,
                censor::MEASURED,
                censor::MEASURED,
            ]
        );

        // A pack that says v2 and a pack that says v1 both round-trip, and
        // each is refused if its arrays contradict its schema string.
        let bytes = crate::pack::encode_pack(&meta, &payload).unwrap();
        let (read_back, _) = crate::pack::decode_pack(&bytes).unwrap();
        assert_eq!(read_back.schema, crate::pack::SWEEPS_SCHEMA_CENSOR);
        assert_eq!(
            read_back.sweeps[0].moments[0].censor_array.as_deref(),
            Some(key.as_str())
        );

        let mut lying = meta.clone();
        lying.schema = crate::pack::SWEEPS_SCHEMA.to_string();
        lying.params.censor_flags = false;
        let bytes = crate::pack::encode_pack(&lying, &payload).unwrap();
        let err = crate::pack::decode_pack(&bytes).unwrap_err().to_string();
        assert!(err.contains("carries a censor plane"), "{err}");

        let mut stripped = meta.clone();
        stripped.sweeps[0].moments[0].censor_array = None;
        let bytes = crate::pack::encode_pack(&stripped, &payload).unwrap();
        let err = crate::pack::decode_pack(&bytes).unwrap_err().to_string();
        assert!(err.contains("is missing a censor plane"), "{err}");
    }

    #[test]
    fn a_radial_that_never_carried_the_moment_is_not_collected_not_clear() {
        // The third NaN source, and the one that is purely this layer's
        // doing: the pack stores a sweep as a rectangle, so a split cut's
        // empty rows are NaN-filled here.  They must read as "not
        // collected", never as "measured and empty" -- a clear-air builder
        // that mistook them would invent observations over every azimuth
        // the moment was never scanned on.
        let sweep = sweep_of(vec![
            vec![moment(REF, 4, 2125, 250), moment(VEL, 4, 2125, 250)],
            vec![moment(REF, 4, 2125, 250)],
        ]);
        let mut builder = PayloadBuilder::new();
        let (layouts, _) = agree_sweep_layouts(&sweep, &[REF, VEL], f64::INFINITY).unwrap();
        let vel = layouts.iter().find(|l| l.product == VEL).unwrap();
        let mut censor = vec![pack::censor_plane::NOT_COLLECTED; 2 * vel.gates];
        censor[..vel.gates]
            .copy_from_slice(&sweep.radials[0].moments[1].censor[..vel.gates]);
        let key = builder.push_u8(&censor, vec![2, vel.gates]);
        let (payload, arrays) = builder.finish();
        let entry = &arrays[&key];
        let codes = &payload[entry.offset..entry.offset + entry.bytes];
        use wx_radar::level2::censor as code;
        assert!(codes[..vel.gates].iter().all(|c| *c == code::MEASURED));
        assert!(codes[vel.gates..]
            .iter()
            .all(|c| *c == pack::censor_plane::NOT_COLLECTED));
        assert_ne!(pack::censor_plane::NOT_COLLECTED, code::BELOW_THRESHOLD);
    }

    #[test]
    fn volume_time_uses_the_one_based_epoch_day() {
        // volume_date == 1 is 1970-01-01 itself, not 1970-01-02.
        let when = volume_time(1, 0).unwrap();
        assert_eq!(iso8601(when), "1970-01-01T00:00:00Z");
        // 20:03:56 == 72_236_000 ms past midnight.
        let when = volume_time(19498, 72_236_000).unwrap();
        assert_eq!(iso8601(when), "2023-05-20T20:03:56Z");
    }

    #[test]
    fn site_resolution_prefers_the_override_and_refuses_to_guess() {
        let fix = resolve_site("KTLX", None).unwrap();
        assert_eq!(fix.id, "KTLX");
        assert_eq!(fix.source, "wx-radar-site-table");
        assert!((fix.lat_deg - 35.33).abs() < 0.1);
        assert!((fix.alt_m - 370.0).abs() < 1.0);

        let fix = resolve_site("KTLX", Some((1.0, 2.0, 3.0))).unwrap();
        assert_eq!(fix.source, "cli-override");
        assert_eq!((fix.lat_deg, fix.lon_deg, fix.alt_m), (1.0, 2.0, 3.0));

        // KOUN is deliberately absent from the operational table.
        let err = resolve_site("KOUN", None).unwrap_err().to_string();
        assert!(err.contains("--site-latlon"), "{err}");
        assert!(resolve_site("KTLX", Some((999.0, 0.0, 0.0))).is_err());
    }

    #[test]
    fn a_site_with_no_table_elevation_is_refused_rather_than_placed_at_sea_level() {
        // KUEX at Hastings really sits at about 602 m; the table says 0.0,
        // as it does for 130 of its 141 entries.  Accepting that number
        // puts every gate in the volume 600 m too low in the model column.
        let err = resolve_site("KUEX", None).unwrap_err().to_string();
        assert!(err.contains("no antenna elevation"), "{err}");
        assert!(err.contains("KUEX"), "{err}");
        assert!(err.contains("ray origin"), "{err}");
        assert!(err.contains("--site-latlon"), "{err}");
        assert!(err.contains("feedhorn"), "{err}");

        // An explicit override is exactly how the caller supplies it, and
        // it is tagged as the caller's number rather than the table's.
        let fix = resolve_site("KUEX", Some((40.3208, -98.4419, 602.0))).unwrap();
        assert_eq!(fix.source, "cli-override");
        assert_eq!(fix.alt_m, 602.0);
    }

    #[test]
    fn the_volumes_own_vol_block_outranks_the_table_and_needs_no_override() {
        // KUEX with a VOL block: the site that refused above now resolves
        // from the volume itself, with the feedhorn already in the height.
        let vol = VolSite {
            latitude_deg: 40.3208,
            longitude_deg: -98.4419,
            site_height_m: 602,
            feedhorn_height_m: 20,
        };
        let fix = resolve_site_from("KUEX", None, Some(&vol)).unwrap();
        assert_eq!(fix.source, "message31-vol-block");
        assert_eq!(fix.alt_m, 622.0);
        assert!((fix.lat_deg - 40.3208).abs() < 1e-4);

        // The explicit override still outranks the volume: it is the
        // operator saying "I know better than the data", which is the one
        // statement a front door must not overrule.
        let fix = resolve_site_from("KUEX", Some((1.0, 2.0, 3.0)), Some(&vol)).unwrap();
        assert_eq!(fix.source, "cli-override");

        // A populated table entry loses to the volume too -- the table's
        // 370 m KTLX value is ground elevation, the VOL block's sum is the
        // beam origin, and the two differing is the defect, not the tie.
        let vol_ktlx = VolSite {
            latitude_deg: 35.33336,
            longitude_deg: -97.27776,
            site_height_m: 370,
            feedhorn_height_m: 19,
        };
        let fix = resolve_site_from("KTLX", None, Some(&vol_ktlx)).unwrap();
        assert_eq!(fix.source, "message31-vol-block");
        assert_eq!(fix.alt_m, 389.0);
    }

    #[test]
    fn a_table_that_disagrees_with_the_volume_is_named_in_the_source() {
        // The KPBZ longitude was transcribed 0.2 degrees off and shipped
        // that way; a volume's own VOL block is how a table error of that
        // kind surfaces on real data.  The fix takes the volume's numbers
        // and says the table disagreed.
        let vol = VolSite {
            latitude_deg: 40.5317,
            longitude_deg: -80.5183, // deliberately 0.3 deg from the table
            site_height_m: 361,
            feedhorn_height_m: 20,
        };
        let fix = resolve_site_from("KPBZ", None, Some(&vol)).unwrap();
        assert!(fix.source.starts_with("message31-vol-block ("), "{}", fix.source);
        assert!(fix.source.contains("disagrees"), "{}", fix.source);
        assert_eq!(fix.lon_deg, f64::from(vol.longitude_deg));
        assert_eq!(fix.alt_m, 381.0);

        // Within survey noise the source stays clean.
        let close = VolSite {
            latitude_deg: 40.5317,
            longitude_deg: -80.2183,
            site_height_m: 361,
            feedhorn_height_m: 20,
        };
        let fix = resolve_site_from("KPBZ", None, Some(&close)).unwrap();
        assert_eq!(fix.source, "message31-vol-block");
    }

    #[test]
    fn the_table_is_swept_for_every_entry_the_decoder_would_accept() {
        // The sweep the audit asked for, kept as a test rather than a
        // one-off: whatever the vendored table holds, exactly the entries
        // with a real elevation resolve, and every other entry refuses.
        // A table update that fills elevations in makes more sites work
        // without touching this file; one that blanks them fails loudly.
        let mut resolved = 0usize;
        let mut refused = 0usize;
        for (id, _, _, _, alt) in wx_radar::sites::SITES {
            match resolve_site(id, None) {
                Ok(fix) => {
                    resolved += 1;
                    assert_ne!(fix.alt_m, SITE_ELEVATION_UNSET_M, "{id}");
                    assert_eq!(fix.source, "wx-radar-site-table", "{id}");
                }
                Err(error) => {
                    refused += 1;
                    assert_eq!(*alt, SITE_ELEVATION_UNSET_M, "{id}: {error}");
                }
            }
        }
        assert_eq!(resolved + refused, wx_radar::sites::SITES.len());
        assert!(resolved > 0, "no site in the table has an elevation");
        assert!(
            refused > 0,
            "the table gained elevations everywhere -- delete this arm and \
             the placeholder refusal with it"
        );
    }
}
