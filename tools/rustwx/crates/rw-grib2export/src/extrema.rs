//! Extrema over a window (`extrema_interval_seconds`, the front door's
//! `--extrema-interval-seconds`).
//!
//! A WRF-format history carries running extremes under their Registry
//! names (`UP_HELI_MAX`, `WSPD10MAX`, ...): each frame holds the extreme
//! since the previous history write, because the model resets them after
//! every write (WOOF does the same, `gpuwm.core.uh_diag`).  The history
//! does not record its own output interval, so the exporter reads each
//! frame's window off the frame sequence: a frame's running extreme covers
//! (previous frame of its domain, the frame], or (run start, the frame] for
//! the first one.
//!
//! The extreme over the window (T - N, T] is then the maximum (or minimum)
//! of the running extremes of the frames inside it, which is exact only
//! when those frames' own windows tile it: the first one must start at
//! T - N.  Any other case is omitted with its reason, never relabelled:
//! frames spaced wider than N, a window reaching back before the run
//! start, or a frame of the window that this export cannot read.
//!
//! The frame sequence is every frame the request discovered (all files of
//! a run folder, before any `--times` filter), plus every frame an earlier
//! `append` recorded.  Running-extreme planes of recent frames are kept in
//! `<out>/.woof-grib2-extrema/` so a live export (one new frame per
//! append) can build its windows; planes older than the window are pruned
//! as frames arrive.  Pass the whole run folder (or every frame of the
//! window): a hand-picked subset of a finer history would look like a
//! coarser one, and the exporter cannot tell the difference from the files.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use rw_mlexport::inputs::{self, Candidate};

use crate::catalog::{Extreme, EXTREMA, PROCESS_MIN};
use crate::{fail, Result};

/// The cache folder, under the export folder.
pub const CACHE_DIR: &str = ".woof-grib2-extrema";
const INDEX: &str = "frames.json";

/// The window of one frame's extrema rows: (start, end], epoch seconds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Window {
    pub start: i64,
    pub end: i64,
}

/// One frame's extrema planes, by row index into [`EXTREMA`], and the rows
/// it could not write.
#[derive(Debug, Default)]
pub struct FrameExtrema {
    pub window: Option<Window>,
    pub planes: BTreeMap<usize, Vec<f32>>,
    pub omitted: Vec<(&'static str, String)>,
}

/// The window state of one export request.
pub struct Context {
    seconds: i64,
    cache: PathBuf,
    /// Known frame valid times per domain.
    known: BTreeMap<String, BTreeSet<i64>>,
    /// Discovered candidates by (domain, valid time), for frames of a
    /// window that are not in the cache.
    candidates: BTreeMap<(String, i64), Candidate>,
}

fn time_key(t: i64) -> String {
    t.to_string()
}

/// The units a carrier must have (Registry units); other units would
/// rescale the extreme, so they are refused per row.
fn units_ok(carrier: &str, units: &str) -> bool {
    let u = units.replace(' ', "").to_ascii_lowercase();
    if carrier.starts_with("UP_HELI") {
        matches!(u.as_str(), "m2s-2" | "m^2s^-2" | "m2/s2" | "m**2s**-2" | "m2s**-2")
    } else {
        matches!(u.as_str(), "ms-1" | "m/s" | "ms**-1" | "m^1s^-1")
    }
}

/// Read the running-extreme carriers a history has: plane, or the reason
/// it cannot be used.
pub fn read_carriers(path: &Path, n: usize) -> Result<BTreeMap<&'static str, std::result::Result<Vec<f32>, String>>> {
    let file = netcrust::File::open(path).map_err(|e| fail(format!("open {}: {e}", path.display())))?;
    let mut out = BTreeMap::new();
    for e in EXTREMA.iter() {
        let Some(var) = file.variable(e.carrier) else { continue };
        let units = var.attribute("units").and_then(|a| a.as_string().map(str::to_owned)).unwrap_or_default();
        if !units_ok(e.carrier, &units) {
            out.insert(e.carrier, Err(format!("{} has units {units:?}, not the Registry's", e.carrier)));
            continue;
        }
        let v = file.read_f64_first_record_or_all(e.carrier).map_err(|err| fail(format!("read {}: {err}", e.carrier)))?;
        if v.len() != n {
            out.insert(e.carrier, Err(format!("{} has {} cells, the grid {n}", e.carrier, v.len())));
            continue;
        }
        out.insert(e.carrier, Ok(v.into_iter().map(|x| if x.is_finite() && x.abs() < 1.0e30 { x as f32 } else { f32::NAN }).collect()));
    }
    Ok(out)
}

/// Reduce planes with the row's statistical process; a cell missing in any
/// frame is missing.
pub fn reduce(e: &Extreme, planes: &[&[f32]]) -> Vec<f32> {
    let n = planes[0].len();
    (0..n)
        .map(|i| {
            let mut acc = planes[0][i];
            for p in &planes[1..] {
                let x = p[i];
                if acc.is_nan() || x.is_nan() {
                    acc = f32::NAN;
                } else if e.process == PROCESS_MIN {
                    if x < acc {
                        acc = x;
                    }
                } else if x > acc {
                    acc = x;
                }
            }
            acc
        })
        .collect()
}

/// The frames whose running extremes tile (T - N, T], or the reason they
/// do not.  `known` holds the domain's frame times, `start` the run start.
pub fn tiling(known: &BTreeSet<i64>, start: i64, end: i64, seconds: i64) -> std::result::Result<Vec<i64>, String> {
    let w0 = end - seconds;
    if w0 < start {
        return Err(format!(
            "the {seconds} s window ending at this frame begins {} s before the run start, so no history frame holds its first part",
            start - w0
        ));
    }
    let frames: Vec<i64> = known.range(w0 + 1..=end).copied().collect();
    let Some(&first) = frames.first() else {
        return Err("no frame of the window is known".into());
    };
    let own_start = known.range(..first).next_back().copied().unwrap_or(start);
    if own_start < w0 {
        return Err(format!(
            "the frame at the window's start holds the running extreme of the {} s since the previous frame, longer than the {seconds} s window; frames are spaced wider than the window",
            first - own_start
        ));
    }
    Ok(frames)
}

impl Context {
    /// Open the window state of a request: `run` starts a fresh cache,
    /// `append` and `finalize` keep it.
    pub fn new(out: &Path, seconds: u64, fresh: bool, candidates: &[Candidate]) -> Result<Self> {
        let cache = out.join(CACHE_DIR);
        if fresh && cache.exists() {
            std::fs::remove_dir_all(&cache).map_err(|e| fail(format!("clear {}: {e}", cache.display())))?;
        }
        let mut known: BTreeMap<String, BTreeSet<i64>> = std::fs::read_to_string(cache.join(INDEX))
            .ok()
            .and_then(|t| serde_json::from_str(&t).ok())
            .unwrap_or_default();
        let mut by_key = BTreeMap::new();
        for c in candidates {
            if let Some((d, t)) = &c.hint {
                known.entry(d.clone()).or_default().insert(*t);
                by_key.insert((d.clone(), *t), c.clone());
            }
        }
        Ok(Self { seconds: seconds as i64, cache, known, candidates: by_key })
    }

    fn frame_dir(&self, domain: &str, t: i64) -> PathBuf {
        self.cache.join(domain).join(time_key(t))
    }

    fn save_index(&self) -> Result<()> {
        std::fs::create_dir_all(&self.cache)?;
        let tmp = self.cache.join(format!("{INDEX}.part"));
        std::fs::write(&tmp, serde_json::to_vec(&self.known).map_err(|e| fail(e.to_string()))?)?;
        std::fs::rename(&tmp, self.cache.join(INDEX))?;
        Ok(())
    }

    fn store(&self, domain: &str, t: i64, carriers: &BTreeMap<&'static str, std::result::Result<Vec<f32>, String>>) -> Result<()> {
        let dir = self.frame_dir(domain, t);
        std::fs::create_dir_all(&dir)?;
        for (name, plane) in carriers {
            if let Ok(p) = plane {
                let bytes: Vec<u8> = p.iter().flat_map(|x| x.to_le_bytes()).collect();
                std::fs::write(dir.join(format!("{name}.f32")), bytes)?;
            }
        }
        Ok(())
    }

    fn cached(&self, domain: &str, t: i64, carrier: &str, n: usize) -> Option<Vec<f32>> {
        let bytes = std::fs::read(self.frame_dir(domain, t).join(format!("{carrier}.f32"))).ok()?;
        (bytes.len() == 4 * n).then(|| bytes.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect())
    }

    fn prune(&self, domain: &str, before_or_at: i64) {
        let Ok(entries) = std::fs::read_dir(self.cache.join(domain)) else { return };
        for entry in entries.flatten() {
            let keep = entry.file_name().to_string_lossy().parse::<i64>().map_or(true, |t| t > before_or_at);
            if !keep {
                let _ = std::fs::remove_dir_all(entry.path());
            }
        }
    }

    /// The extrema planes of one frame.  `path` is the frame on disk,
    /// `start` the run start, `scratch` where archived frames unpack.
    pub fn frame(&mut self, domain: &str, valid: i64, start: i64, path: &Path, n: usize, scratch: &Path) -> Result<FrameExtrema> {
        let mut out = FrameExtrema::default();
        let current = read_carriers(path, n)?;
        self.known.entry(domain.to_owned()).or_default().insert(valid);
        self.store(domain, valid, &current)?;
        self.save_index()?;
        if current.is_empty() {
            for e in EXTREMA.iter() {
                out.omitted.push((e.row.id, format!("carrier absent: {} (running extremes need nwp_diagnostics output)", e.carrier)));
            }
            return Ok(out);
        }
        let known = self.known.get(domain).cloned().unwrap_or_default();
        let frames = match tiling(&known, start, valid, self.seconds) {
            Ok(f) => f,
            Err(why) => {
                for e in EXTREMA.iter() {
                    out.omitted.push((e.row.id, why.clone()));
                }
                return Ok(out);
            }
        };
        out.window = Some(Window { start: valid - self.seconds, end: valid });
        // Running-extreme planes of every frame in the window.
        let mut per_frame: Vec<BTreeMap<&'static str, std::result::Result<Vec<f32>, String>>> = Vec::new();
        for &t in &frames {
            if t == valid {
                per_frame.push(current.clone());
                continue;
            }
            let mut planes = BTreeMap::new();
            let mut missing = Vec::new();
            for e in EXTREMA.iter() {
                if let Some(p) = self.cached(domain, t, e.carrier, n) {
                    planes.insert(e.carrier, Ok(p));
                } else {
                    missing.push(e.carrier);
                }
            }
            if planes.is_empty() {
                // Not cached: read it from the request's own inputs.
                match self.candidates.get(&(domain.to_owned(), t)) {
                    Some(c) => {
                        let on_disk = inputs::materialize(c, scratch)?;
                        let read = read_carriers(&on_disk.path, n)?;
                        self.store(domain, t, &read)?;
                        planes = read;
                    }
                    None => {
                        let why = format!(
                            "the frame at {} is in the window but neither in this export nor in its window cache",
                            rw_mlexport::times::iso(t)
                        );
                        for e in EXTREMA.iter() {
                            out.omitted.push((e.row.id, why.clone()));
                        }
                        out.window = None;
                        return Ok(out);
                    }
                }
            }
            let _ = missing;
            per_frame.push(planes);
        }
        for (i, e) in EXTREMA.iter().enumerate() {
            let mut planes: Vec<&[f32]> = Vec::new();
            let mut why = None;
            for (k, f) in per_frame.iter().enumerate() {
                match f.get(e.carrier) {
                    Some(Ok(p)) => planes.push(p),
                    Some(Err(reason)) => why = Some(reason.clone()),
                    None => {
                        why = Some(if frames[k] == valid {
                            format!("carrier absent: {}", e.carrier)
                        } else {
                            format!("{} is absent from the frame at {}", e.carrier, rw_mlexport::times::iso(frames[k]))
                        })
                    }
                }
                if why.is_some() {
                    break;
                }
            }
            match why {
                Some(w) => out.omitted.push((e.row.id, w)),
                None => {
                    out.planes.insert(i, reduce(e, &planes));
                }
            }
        }
        self.prune(domain, valid - self.seconds);
        Ok(out)
    }
}

/// Forecast time of the window start and the window length in one unit of
/// code table 4.4 (hours when both are whole hours, else minutes, else
/// seconds): `(unit, start offset, length)`.
pub fn time_range(window: Window, reference: i64) -> std::result::Result<(u8, u32, u32), String> {
    let offset = window.start - reference;
    let length = window.end - window.start;
    if offset < 0 || length <= 0 {
        return Err("the extrema window starts before the reference time".into());
    }
    let (unit, div) = if offset % 3600 == 0 && length % 3600 == 0 {
        (1u8, 3600)
    } else if offset % 60 == 0 && length % 60 == 0 {
        (0, 60)
    } else {
        (13, 1)
    };
    let fit = |v: i64| u32::try_from(v / div).map_err(|_| "the extrema window does not fit GRIB2 octets".to_owned());
    Ok((unit, fit(offset)?, fit(length)?))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn set(v: &[i64]) -> BTreeSet<i64> {
        v.iter().copied().collect()
    }

    #[test]
    fn hourly_window_from_quarter_hour_frames() {
        let k = set(&[0, 900, 1800, 2700, 3600, 4500]);
        assert_eq!(tiling(&k, 0, 3600, 3600).unwrap(), vec![900, 1800, 2700, 3600]);
        assert_eq!(tiling(&k, 0, 4500, 3600).unwrap(), vec![1800, 2700, 3600, 4500]);
        // A window equal to the history interval passes one frame through.
        assert_eq!(tiling(&k, 0, 4500, 900).unwrap(), vec![4500]);
    }

    #[test]
    fn windows_that_cannot_be_built_are_refused_with_the_reason() {
        let k = set(&[0, 3600, 7200]);
        // Hourly frames cannot make a 15 min window.
        assert!(tiling(&k, 0, 7200, 900).unwrap_err().contains("wider than the window"));
        // An hour window at 30 min reaches before the run start.
        assert!(tiling(&set(&[0, 1800]), 0, 1800, 3600).unwrap_err().contains("before the run start"));
        // A 2 h window at 1 h also reaches back before the start.
        assert!(tiling(&k, 0, 3600, 7200).is_err());
        assert_eq!(tiling(&k, 0, 7200, 7200).unwrap(), vec![3600, 7200]);
    }

    #[test]
    fn reduce_takes_max_or_min_and_keeps_missing() {
        let a = [1.0f32, 5.0, f32::NAN, -2.0];
        let b = [3.0f32, 4.0, 1.0, -7.0];
        let mx = reduce(&EXTREMA[0], &[&a, &b]);
        assert_eq!(&mx[..2], &[3.0, 5.0]);
        assert!(mx[2].is_nan());
        assert_eq!(mx[3], -2.0);
        let mn = reduce(&EXTREMA[1], &[&a, &b]);
        assert_eq!(mn[3], -7.0);
        assert_eq!(mn[0], 1.0);
    }

    #[test]
    fn time_range_units() {
        assert_eq!(time_range(Window { start: 7200, end: 10800 }, 0).unwrap(), (1, 2, 1));
        assert_eq!(time_range(Window { start: 900, end: 1800 }, 0).unwrap(), (0, 15, 15));
        assert_eq!(time_range(Window { start: 30, end: 90 }, 0).unwrap(), (13, 30, 60));
        assert!(time_range(Window { start: -60, end: 0 }, 0).is_err());
    }
}
