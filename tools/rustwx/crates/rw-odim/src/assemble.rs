//! Assemble one polar volume out of the many single-sweep files that some
//! national feeds serve.
//!
//! The Netherlands and Romania publish one `PVOL` file per volume, and
//! [`read_volume`](crate::read_volume) is the whole story for them. Germany
//! does not: one Boostedt volume arrives as a **file per (elevation,
//! quantity) pair** — a `SCAN` object holding a single `/dataset1`. Measured
//! on the live feed 2026-08-14, that is **30 files** for one ten-elevation
//! volume, because it serves `TH` beside `DBZH` and `VRADH`, and they mean
//! nothing apart. Assembling them is not a convenience; without it the German
//! half of Europe is unreachable no matter how good the decoder is.
//!
//! # What could go wrong here, and what is done about it
//!
//! The failure this module exists to prevent is **a volume made of two
//! volumes**. Consecutive scans of the same radar produce files whose names
//! differ only in a timestamp; gluing a 0.5-degree cut from 12:00 to a
//! 1.5-degree cut from 12:05 yields a plausible object with a wrong wind
//! field five minutes out of date in half its columns, and nothing downstream
//! can see it. So the nominal time is the grouping key, it is read from
//! `/what/date` + `/what/time` **inside** each file rather than from its
//! name, and a group whose members disagree is refused rather than repaired.
//!
//! The second failure is **merging two antenna passes into one sweep**.
//! It is tempting to key a cut on its elevation, and it is wrong: a Dutch
//! volume carries three sweeps at 0.30 degrees with two different Nyquist
//! intervals, and split-cut strategies (a long-PRT reflectivity pass and a
//! short-PRT Doppler pass at the same angle) are ordinary radar practice.
//! Two files therefore merge into one [`Sweep`] only when they agree on the
//! **whole** cut identity — elevation, ray and bin counts, range scale, range
//! start, and the sweep's own start and end times — *and* their per-ray
//! azimuths match ray for ray. Anything less becomes two sweeps, which is the
//! honest reading: they are two passes.
//!
//! That rule makes the merge ratio a measurement rather than an assumption.
//! Thirty files becoming ten sweeps says the feed records all three moments on
//! one antenna pass; thirty becoming thirty says it does not. Neither is
//! assumed here and [`AssembleReport`] reports which happened. Measured on the
//! real feed it is the first: 30 files, 10 sweeps, 20 merged.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use chrono::{DateTime, Utc};
use sha2::{Digest, Sha256};

use crate::decode::{DecodeOptions, read_volume_with};
use crate::error::{OdimError, Result};
use crate::volume::{PolarVolume, Sweep};

/// Positions closer than this are the same antenna, in degrees.
///
/// ODIM stores `/where/lat` and `/where/lon` as float64 and every file of one
/// volume is written by one radar from one configuration, so the realistic
/// disagreement is zero. The tolerance exists so that a feed that rounds
/// differently between products does not fail the check, and it is tight
/// enough that two different radars can never pass it: 1e-6 degrees is about
/// 0.1 m.
const POSITION_TOLERANCE_DEG: f64 = 1e-6;

/// Antenna heights closer than this are the same antenna, in metres.
const HEIGHT_TOLERANCE_M: f64 = 0.5;

/// Azimuths closer than this are the same ray, in degrees.
///
/// Azimuths are derived per ray from `startazA`/`stopazA` when the file
/// carries them, so two products of one pass can differ in the last bits of a
/// float64 without being different rays. A real ray-to-ray step is at least
/// 0.5 degrees on any operational scan, so this separates the two by three
/// orders of magnitude.
const AZIMUTH_TOLERANCE_DEG: f64 = 1e-6;

/// One file on disk and the identity it contributed to an assembled volume.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MemberFile {
    pub name: String,
    pub bytes: usize,
    pub sha256: String,
}

/// One candidate volume found in a directory: the files that share a nominal
/// time, and what they are.
#[derive(Debug, Clone, serde::Serialize)]
pub struct VolumeGroup {
    /// The grouping key: `/what/date` + `/what/time`, `YYYYmmddTHHMMSSZ`, or
    /// `"unstamped"` for files that declare no nominal time.
    pub stamp: String,
    /// The site label the members agree on, or the first one seen when they
    /// do not (`site_disagreement` then says so).
    pub site: String,
    pub files: Vec<PathBuf>,
    pub bytes: usize,
    /// ODIM `/what/object` values present, sorted and deduplicated. A
    /// single-file group of `["PVOL"]` needs no assembly at all.
    pub objects: Vec<String>,
    /// Quantities the members carry, sorted and deduplicated.
    pub quantities: Vec<String>,
    /// Elevations present, sorted, one entry per distinct cut angle.
    pub elevations_deg: Vec<f64>,
    /// Set when members of one stamp disagree about which radar they are.
    /// The group is still reported — refusing to *describe* a directory would
    /// leave a user with no way to see the problem — but [`assemble`] will
    /// refuse it.
    pub site_disagreement: Option<String>,
}

/// What an assembly did, so a caller can see the merge rather than trust it.
#[derive(Debug, Clone, serde::Serialize)]
pub struct AssembleReport {
    pub files: usize,
    pub sweeps: usize,
    /// Files whose cut identity matched an earlier file's, and whose moments
    /// were therefore folded into that sweep. `files - merged == sweeps`.
    pub merged: usize,
    pub stamp: String,
    pub members: Vec<MemberFile>,
    /// Digest over the member manifest, not over any single file's bytes.
    /// See [`manifest_sha256`].
    pub manifest_sha256: String,
    pub total_bytes: usize,
}

/// The digest an assembled volume carries in place of a file digest.
///
/// An assembled volume has no bytes of its own, so it cannot honestly quote a
/// file `sha256`. This is the SHA-256 of the manifest text — one
/// `"<sha256>  <name>\n"` line per member, sorted by name — which identifies
/// exactly the set of files that went in and changes if any of them changes.
/// It is a different claim from a file digest and is named differently for
/// that reason.
pub fn manifest_sha256(members: &[MemberFile]) -> String {
    let mut sorted: Vec<&MemberFile> = members.iter().collect();
    sorted.sort_by(|a, b| a.name.cmp(&b.name));
    let mut hasher = Sha256::new();
    for member in sorted {
        hasher.update(format!("{}  {}\n", member.sha256, member.name).as_bytes());
    }
    hasher
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn stamp_of(time: Option<DateTime<Utc>>) -> String {
    match time {
        None => "unstamped".to_string(),
        Some(when) => when.format("%Y%m%dT%H%M%SZ").to_string(),
    }
}

fn read_identity(path: &Path) -> Result<MemberFile> {
    let bytes = std::fs::read(path).map_err(|source| OdimError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let mut hasher = Sha256::new();
    hasher.update(&bytes);
    let sha256: String = hasher
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect();
    Ok(MemberFile {
        name: path
            .file_name()
            .map(|name| name.to_string_lossy().to_string())
            .unwrap_or_else(|| path.display().to_string()),
        bytes: bytes.len(),
        sha256,
    })
}

/// Every `.h5`/`.hdf`/`.hdf5` file in `dir`, sorted by name.
///
/// Not recursive: a directory of volumes is flat, and descending would sweep
/// an unrelated tree into a volume.
pub fn odim_files_in(dir: &Path) -> Result<Vec<PathBuf>> {
    let entries = std::fs::read_dir(dir).map_err(|source| OdimError::Io {
        path: dir.to_path_buf(),
        source,
    })?;
    let mut paths = Vec::new();
    for entry in entries {
        let entry = entry.map_err(|source| OdimError::Io {
            path: dir.to_path_buf(),
            source,
        })?;
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let matches = path
            .extension()
            .and_then(|ext| ext.to_str())
            .map(|ext| {
                let ext = ext.to_ascii_lowercase();
                ext == "h5" || ext == "hdf" || ext == "hdf5"
            })
            .unwrap_or(false);
        if matches {
            paths.push(path);
        }
    }
    paths.sort();
    Ok(paths)
}

/// Group the ODIM files in `dir` into candidate volumes by nominal time.
///
/// Geometry only: every file's header is read, no payload is decoded, so
/// surveying a directory of twenty 30 MB scans does not decode 20 million
/// gates to answer a question about timestamps.
pub fn survey_dir(dir: &Path) -> Result<Vec<VolumeGroup>> {
    let paths = odim_files_in(dir)?;
    let options = DecodeOptions::geometry_only();
    let mut groups: BTreeMap<String, VolumeGroup> = BTreeMap::new();

    for path in paths {
        let volume = read_volume_with(&path, &options)?;
        let stamp = stamp_of(volume.nominal_time);
        let bytes = std::fs::metadata(&path)
            .map(|meta| meta.len() as usize)
            .unwrap_or(0);
        let label = volume.source.label();

        let group = groups.entry(stamp.clone()).or_insert_with(|| VolumeGroup {
            stamp: stamp.clone(),
            site: label.clone(),
            files: Vec::new(),
            bytes: 0,
            objects: Vec::new(),
            quantities: Vec::new(),
            elevations_deg: Vec::new(),
            site_disagreement: None,
        });
        if group.site != label && group.site_disagreement.is_none() {
            group.site_disagreement = Some(format!(
                "{} declares site {label:?} while an earlier member declares {:?}",
                path.display(),
                group.site
            ));
        }
        group.files.push(path.clone());
        group.bytes += bytes;
        if !group.objects.contains(&volume.object) {
            group.objects.push(volume.object.clone());
        }
        for quantity in volume.quantities() {
            if !group.quantities.contains(&quantity) {
                group.quantities.push(quantity);
            }
        }
        for sweep in &volume.sweeps {
            if !group
                .elevations_deg
                .iter()
                .any(|elev| (elev - sweep.elevation_deg).abs() < 1e-9)
            {
                group.elevations_deg.push(sweep.elevation_deg);
            }
        }
    }

    let mut out: Vec<VolumeGroup> = groups.into_values().collect();
    for group in &mut out {
        group.objects.sort();
        group.quantities.sort();
        group
            .elevations_deg
            .sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    }
    Ok(out)
}

/// Do two decoded sweeps describe the same antenna pass?
///
/// Every field of the cut identity, then the azimuths ray by ray. Elevation
/// alone is deliberately not enough — see the module docs.
fn same_cut(left: &Sweep, right: &Sweep) -> bool {
    if left.nrays != right.nrays
        || left.nbins != right.nbins
        || (left.elevation_deg - right.elevation_deg).abs() > 1e-9
        || (left.range_scale_m - right.range_scale_m).abs() > 1e-9
        || (left.range_start_m - right.range_start_m).abs() > 1e-9
        || left.start_time != right.start_time
        || left.end_time != right.end_time
    {
        return false;
    }
    if left.azimuth_deg.len() != right.azimuth_deg.len() {
        return false;
    }
    left.azimuth_deg
        .iter()
        .zip(right.azimuth_deg.iter())
        .all(|(a, b)| (a - b).abs() <= AZIMUTH_TOLERANCE_DEG)
}

fn refuse(message: String) -> OdimError {
    OdimError::format("assembling a volume from split files", message)
}

/// Assemble `paths` into one polar volume.
///
/// Every file is decoded with `options` and its sweeps folded into the
/// result. The caller is expected to have chosen a coherent set — normally
/// one [`VolumeGroup`] from [`survey_dir`] — and every disagreement that
/// would make the result a lie is refused here rather than smoothed over:
/// a different radar, a different nominal time, or two files claiming the
/// same quantity on the same cut.
///
/// A single-element `paths` is legal and simply decodes that file, so a
/// caller does not need to branch on whether a country splits its volumes.
pub fn assemble(
    paths: &[PathBuf],
    options: &DecodeOptions,
) -> Result<(PolarVolume, AssembleReport)> {
    let mut decoded = Vec::with_capacity(paths.len());
    for path in paths {
        let identity = read_identity(path)?;
        let volume = read_volume_with(path, options)?;
        decoded.push((volume, identity));
    }
    assemble_decoded(decoded)
}

/// The assembly itself, over volumes that are already decoded.
///
/// Split out from [`assemble`] so that every refusal below can be exercised
/// without three megabytes of real HDF5 on disk. Reading files and deciding
/// whether they are one volume are separate concerns, and only the second one
/// has any judgement in it.
pub fn assemble_decoded(
    decoded: Vec<(PolarVolume, MemberFile)>,
) -> Result<(PolarVolume, AssembleReport)> {
    if decoded.is_empty() {
        return Err(refuse("no files to assemble".to_string()));
    }

    let mut assembled: Option<PolarVolume> = None;
    let mut members: Vec<MemberFile> = Vec::new();
    let mut total_bytes = 0usize;
    let mut merged = 0usize;

    for (volume, identity) in decoded {
        total_bytes += identity.bytes;

        let Some(target) = assembled.as_mut() else {
            members.push(identity);
            assembled = Some(volume);
            continue;
        };

        if volume.nominal_time != target.nominal_time {
            return Err(refuse(format!(
                "{} is stamped {} but the assembly is stamped {}. Two nominal \
                 times are two volumes, and a volume made of two volumes has \
                 a wind field from the wrong minute in some of its columns \
                 with nothing downstream able to see it. Group the files by \
                 nominal time first",
                identity.name,
                stamp_of(volume.nominal_time),
                stamp_of(target.nominal_time)
            )));
        }
        if volume.source.label() != target.source.label() {
            return Err(refuse(format!(
                "{} declares site {:?} but the assembly is site {:?}",
                identity.name,
                volume.source.label(),
                target.source.label()
            )));
        }
        if (volume.site.latitude_deg - target.site.latitude_deg).abs() > POSITION_TOLERANCE_DEG
            || (volume.site.longitude_deg - target.site.longitude_deg).abs()
                > POSITION_TOLERANCE_DEG
            || (volume.site.height_m - target.site.height_m).abs() > HEIGHT_TOLERANCE_M
        {
            return Err(refuse(format!(
                "{} places the antenna at ({:.6}, {:.6}, {:.1} m) but the \
                 assembly places it at ({:.6}, {:.6}, {:.1} m). Two antennas \
                 are two radars",
                identity.name,
                volume.site.latitude_deg,
                volume.site.longitude_deg,
                volume.site.height_m,
                target.site.latitude_deg,
                target.site.longitude_deg,
                target.site.height_m
            )));
        }

        for sweep in volume.sweeps {
            match target.sweeps.iter_mut().find(|held| same_cut(held, &sweep)) {
                None => target.sweeps.push(sweep),
                Some(held) => {
                    for moment in sweep.moments {
                        if held.moments.iter().any(|m| m.quantity == moment.quantity) {
                            return Err(refuse(format!(
                                "{} carries a second {} on the {:.2}-degree cut \
                                 the assembly already holds one for. Two values \
                                 for one gate is not a volume; check the file \
                                 list for a duplicate",
                                identity.name, moment.quantity, held.elevation_deg
                            )));
                        }
                        held.moments.push(moment);
                    }
                    merged += 1;
                }
            }
        }
        members.push(identity);
    }

    let mut volume = assembled.expect("non-empty paths yield a volume");

    // Order by acquisition, then by angle: a consumer quoting "sweep 3"
    // should mean the third cut the antenna made. `index` is rewritten to
    // match, because after assembly the original `/datasetN` numbering is one
    // file's private counter and every member's starts at 1.
    volume.sweeps.sort_by(|a, b| {
        a.start_time
            .cmp(&b.start_time)
            .then_with(|| {
                a.elevation_deg
                    .partial_cmp(&b.elevation_deg)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .then_with(|| a.path.cmp(&b.path))
    });
    for (position, sweep) in volume.sweeps.iter_mut().enumerate() {
        sweep.index = position + 1;
    }

    // An assembled object is a volume whatever its members called themselves.
    volume.object = "PVOL".to_string();

    let report = AssembleReport {
        files: members.len(),
        sweeps: volume.sweeps.len(),
        merged,
        stamp: stamp_of(volume.nominal_time),
        manifest_sha256: manifest_sha256(&members),
        members,
        total_bytes,
    };
    Ok((volume, report))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::volume::{AzimuthSource, Nyquist, NyquistSource};

    fn sweep(elevation: f64, start: Option<DateTime<Utc>>, azimuths: Vec<f64>) -> Sweep {
        Sweep {
            index: 1,
            path: "/dataset1".to_string(),
            elevation_deg: elevation,
            nrays: azimuths.len(),
            nbins: 10,
            range_scale_m: 250.0,
            range_start_m: 0.0,
            a1gate: None,
            start_time: start,
            end_time: start,
            nyquist: Nyquist {
                interval_ms: Some(8.0),
                source: NyquistSource::Declared,
                high_prf_hz: None,
                low_prf_hz: None,
                dual_prf: false,
            },
            azimuth_deg: azimuths,
            azimuth_source: AzimuthSource::NominalFromRayCount { astart_deg: 0.0 },
            ray_elevation_deg: None,
            moments: Vec::new(),
        }
    }

    #[test]
    fn one_cut_is_one_cut() {
        let left = sweep(0.5, None, vec![0.0, 1.0, 2.0]);
        let right = sweep(0.5, None, vec![0.0, 1.0, 2.0]);
        assert!(same_cut(&left, &right));
    }

    #[test]
    fn a_different_start_time_is_a_different_pass() {
        let when = DateTime::from_timestamp(1_760_000_000, 0).map(|t| t.to_utc());
        let later = DateTime::from_timestamp(1_760_000_060, 0).map(|t| t.to_utc());
        let left = sweep(0.5, when, vec![0.0, 1.0]);
        let right = sweep(0.5, later, vec![0.0, 1.0]);
        assert!(
            !same_cut(&left, &right),
            "two passes at one elevation must stay two sweeps"
        );
    }

    #[test]
    fn different_azimuths_are_a_different_pass() {
        let left = sweep(0.5, None, vec![0.0, 1.0, 2.0]);
        let right = sweep(0.5, None, vec![0.0, 1.0, 2.5]);
        assert!(!same_cut(&left, &right));
    }

    #[test]
    fn different_range_geometry_is_a_different_cut() {
        let left = sweep(0.5, None, vec![0.0, 1.0]);
        let mut right = sweep(0.5, None, vec![0.0, 1.0]);
        right.range_scale_m = 500.0;
        assert!(!same_cut(&left, &right));
    }

    #[test]
    fn manifest_digest_is_order_independent_and_content_sensitive() {
        let a = MemberFile {
            name: "a.h5".into(),
            bytes: 1,
            sha256: "aa".into(),
        };
        let b = MemberFile {
            name: "b.h5".into(),
            bytes: 2,
            sha256: "bb".into(),
        };
        let forward = manifest_sha256(&[a.clone(), b.clone()]);
        let backward = manifest_sha256(&[b.clone(), a.clone()]);
        assert_eq!(forward, backward, "file order must not change the identity");

        let changed = MemberFile {
            sha256: "cc".into(),
            ..b
        };
        assert_ne!(
            forward,
            manifest_sha256(&[a, changed]),
            "a changed member must change the identity"
        );
    }

    #[test]
    fn an_empty_list_is_refused() {
        let error = assemble(&[], &DecodeOptions::all()).unwrap_err();
        assert!(error.to_string().contains("no files"));
    }

    // ----------------------------------------------------------- refusals
    //
    // These are the properties the German route rests on, and every one of
    // them is a silent failure if it does not hold: an assembled volume looks
    // exactly as ordinary when it is wrong as when it is right.

    fn moment(quantity: &str) -> crate::volume::Moment {
        crate::volume::Moment {
            quantity: quantity.to_string(),
            path: format!("/dataset1/data1 {quantity}"),
            unit: "dBZ".to_string(),
            kind: crate::quantity::describe(quantity).kind,
            calibration: crate::volume::Calibration {
                gain: 1.0,
                offset: 0.0,
                nodata: Some(255.0),
                undetect: Some(0.0),
                sentinels_collide: false,
            },
            nrays: 2,
            nbins: 10,
            values: vec![0.0; 20],
            censor: vec![0; 20],
            census: crate::censor::Census::default(),
        }
    }

    fn volume(nod: &str, stamp: i64, lat: f64, sweeps: Vec<Sweep>) -> PolarVolume {
        PolarVolume {
            object: "SCAN".to_string(),
            conventions: None,
            version: None,
            source: crate::volume::Source::parse(&format!("NOD:{nod}")),
            nominal_time: DateTime::from_timestamp(stamp, 0).map(|t| t.to_utc()),
            site: crate::volume::Site {
                latitude_deg: lat,
                longitude_deg: 10.0,
                height_m: 124.56,
            },
            system: crate::volume::SystemNotes::default(),
            sweeps,
        }
    }

    fn member(name: &str) -> MemberFile {
        MemberFile {
            name: name.to_string(),
            bytes: 1,
            sha256: format!("{name:0>64}"),
        }
    }

    fn cut(elevation: f64, quantity: &str) -> Sweep {
        let mut held = sweep(elevation, None, vec![0.0, 180.0]);
        held.moments.push(moment(quantity));
        held
    }

    #[test]
    fn two_nominal_times_are_refused_as_two_volumes() {
        let error = assemble_decoded(vec![
            (
                volume("deboo", 1_760_000_000, 54.0, vec![cut(0.5, "DBZH")]),
                member("a.h5"),
            ),
            (
                volume("deboo", 1_760_000_300, 54.0, vec![cut(1.5, "DBZH")]),
                member("b.h5"),
            ),
        ])
        .unwrap_err()
        .to_string();
        assert!(error.contains("b.h5"), "{error}");
        assert!(error.contains("Two nominal"), "{error}");
    }

    #[test]
    fn two_radars_are_refused_however_well_their_times_agree() {
        let error = assemble_decoded(vec![
            (
                volume("deboo", 1_760_000_000, 54.0, vec![cut(0.5, "DBZH")]),
                member("a.h5"),
            ),
            (
                volume("deess", 1_760_000_000, 54.0, vec![cut(1.5, "DBZH")]),
                member("b.h5"),
            ),
        ])
        .unwrap_err()
        .to_string();
        assert!(error.contains("declares site"), "{error}");
    }

    #[test]
    fn one_site_label_over_two_antennas_is_still_refused() {
        // The label can agree while the position does not -- a renamed or
        // relocated antenna. The position is the physical claim, and every
        // gate's height is computed from it.
        let error = assemble_decoded(vec![
            (
                volume("deboo", 1_760_000_000, 54.0, vec![cut(0.5, "DBZH")]),
                member("a.h5"),
            ),
            (
                volume("deboo", 1_760_000_000, 48.0, vec![cut(1.5, "DBZH")]),
                member("b.h5"),
            ),
        ])
        .unwrap_err()
        .to_string();
        assert!(error.contains("Two antennas are two radars"), "{error}");
    }

    #[test]
    fn a_repeated_quantity_on_one_cut_is_refused() {
        let error = assemble_decoded(vec![
            (
                volume("deboo", 1_760_000_000, 54.0, vec![cut(0.5, "DBZH")]),
                member("a.h5"),
            ),
            (
                volume("deboo", 1_760_000_000, 54.0, vec![cut(0.5, "DBZH")]),
                member("a-again.h5"),
            ),
        ])
        .unwrap_err()
        .to_string();
        assert!(error.contains("second DBZH"), "{error}");
    }

    #[test]
    fn moments_of_one_pass_become_one_sweep_and_the_ratio_is_reported() {
        // The German shape: three quantities per elevation, two elevations.
        let files: Vec<(PolarVolume, MemberFile)> = [
            (0.5, "DBZH"),
            (0.5, "TH"),
            (0.5, "VRADH"),
            (1.5, "DBZH"),
            (1.5, "TH"),
            (1.5, "VRADH"),
        ]
        .into_iter()
        .enumerate()
        .map(|(index, (elevation, quantity))| {
            (
                volume("deboo", 1_760_000_000, 54.0, vec![cut(elevation, quantity)]),
                member(&format!("f{index}.h5")),
            )
        })
        .collect();

        let (assembled, report) = assemble_decoded(files).expect("one volume");
        assert_eq!(report.files, 6);
        assert_eq!(report.sweeps, 2);
        assert_eq!(report.merged, 4);
        assert_eq!(report.files - report.merged, report.sweeps);
        assert_eq!(assembled.object, "PVOL", "an assembled object is a volume");
        for (position, held) in assembled.sweeps.iter().enumerate() {
            assert_eq!(held.index, position + 1, "sweeps are renumbered");
            assert_eq!(held.moments.len(), 3);
        }
    }

    #[test]
    fn two_passes_at_one_elevation_stay_two_sweeps() {
        // The other direction of the same rule, and the one that would be
        // wrong if the merge keyed on elevation: a split-cut strategy records
        // reflectivity and Doppler on separate passes at the same angle.
        let first = DateTime::from_timestamp(1_760_000_000, 0).map(|t| t.to_utc());
        let second = DateTime::from_timestamp(1_760_000_030, 0).map(|t| t.to_utc());
        let mut reflectivity = sweep(0.5, first, vec![0.0, 180.0]);
        reflectivity.moments.push(moment("DBZH"));
        let mut doppler = sweep(0.5, second, vec![0.0, 180.0]);
        doppler.moments.push(moment("VRADH"));

        let (assembled, report) = assemble_decoded(vec![
            (
                volume("deboo", 1_760_000_000, 54.0, vec![reflectivity]),
                member("a.h5"),
            ),
            (
                volume("deboo", 1_760_000_000, 54.0, vec![doppler]),
                member("b.h5"),
            ),
        ])
        .expect("one volume, two passes");
        assert_eq!(report.sweeps, 2);
        assert_eq!(report.merged, 0);
        assert_eq!(assembled.sweeps[0].moments.len(), 1);
    }
}
