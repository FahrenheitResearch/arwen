//! Scan cycles: which sweeps of a volume a radar collected in one pass of
//! its scan strategy.
//!
//! A radar repeats its scan strategy: every cycle collects the same cuts
//! again. A file or a merge can hold more than one cycle (JMA's 10-minute
//! tars hold two 5-minute cycles; an ODIM product file can carry a sweep of
//! the cycle before). Formats that hold one scan per file, NEXRAD Level II
//! above all, must not mix them. [`scan_cycles`] finds where one cycle ends
//! and the next begins, and [`split_scan_cycles`] makes one volume of each.
//!
//! Sweeps are taken in the order they were collected ([`collection_order`]),
//! and a sweep begins a new cycle when:
//!
//! - it collects a cut the current cycle already collected: the same sweep
//!   mode, a fixed angle within [`ANGLE_MATCH_TOLERANCE_DEG`], the same range
//!   gates and the same field names ([`CycleBreak::RepeatedCut`]). Two cuts at
//!   one angle with other gates or moments (a long-range surveillance cut and
//!   a Doppler cut, NEXRAD's split cuts) belong to one cycle;
//! - it starts more than [`MAX_SCAN_PAUSE_S`] after every sweep of the current
//!   cycle ended ([`CycleBreak::Pause`]): a radar does not pause that long
//!   within one volume scan;
//! - for a NEXRAD Level II source, one of its radials begins a volume scan
//!   (radial status 3 in the per-ray `nexrad_radial_status` variable,
//!   ICD 2620002 Table XVII-A byte 21; [`CycleBreak::VolumeStart`]). A Level
//!   II volume coverage pattern repeats cuts within one volume by design
//!   (SAILS and MRLE rescans of the lowest cuts), so the repeated-cut rule
//!   does not apply to it.
//!
//! A sweep without rays never begins a cycle; it stays with the sweep
//! stored before it.

use chrono::{DateTime, Duration, Utc};
use std::fmt;

use super::merge::ANGLE_MATCH_TOLERANCE_DEG;
use super::sweep::{RangeCoord, Sweep, SweepMode};
use super::values::ArrayBuf;
use super::volume::{SourceFormat, TimeCoverage, Volume, floor_to_second};

/// Longest pause, in seconds, between the end of one sweep and the start of
/// the next within one scan cycle. Operational volume scans repeat every 4
/// to 10 minutes and run their cuts back to back; a sweep that begins more
/// than four minutes after every sweep of the cycle ended belongs to another
/// cycle. In the corpus the longest pause within a cycle is 129 s (JMA
/// Takayasu's 2019 velocity file, which leaves out the reflectivity-only
/// cuts collected in between and whose rays carry only each sweep's
/// observation start), and the pause before a sweep of another cycle is
/// 442 s (the 90 deg sweep of Hurum's 2026-06-12 14:46 velocity file,
/// collected in the scan before).
pub const MAX_SCAN_PAUSE_S: f64 = 240.0;

/// Range gates of two cuts closer than this, in metres, are the same gates.
const RANGE_MATCH_TOLERANCE_M: f64 = 0.01;

/// Radial status of the first radial of a Level II volume scan
/// (ICD 2620002 Table XVII-A byte 21, low seven bits).
const NEXRAD_BEGINNING_OF_VOLUME: u16 = 3;

/// Why a sweep begins a new scan cycle. Sweep numbers are indices into
/// [`Volume::sweeps`] (for a [`CycleTracker`], the labels its caller gave).
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum CycleBreak {
    /// `sweep` collects the same cut as `earlier`, which the current cycle
    /// already holds.
    RepeatedCut {
        /// The sweep that begins the new cycle.
        sweep: usize,
        /// The earlier sweep of the same cut.
        earlier: usize,
        /// Seconds from the start of `earlier` to the start of `sweep`, when
        /// both have ray times.
        seconds_apart: Option<f64>,
    },
    /// `sweep` starts `seconds` after `previous`, the last sweep of the
    /// current cycle to end, ended: more than [`MAX_SCAN_PAUSE_S`].
    Pause {
        /// The sweep that begins the new cycle.
        sweep: usize,
        /// The sweep of the current cycle that ended last.
        previous: usize,
        /// The pause, in seconds.
        seconds: f64,
    },
    /// A radial of `sweep` begins a Level II volume scan (radial status 3).
    VolumeStart {
        /// The sweep that begins the new cycle.
        sweep: usize,
    },
}

impl CycleBreak {
    /// The sweep that begins the new cycle.
    pub fn sweep(&self) -> usize {
        match self {
            Self::RepeatedCut { sweep, .. }
            | Self::Pause { sweep, .. }
            | Self::VolumeStart { sweep } => *sweep,
        }
    }
}

impl fmt::Display for CycleBreak {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RepeatedCut {
                sweep,
                earlier,
                seconds_apart,
            } => {
                write!(
                    f,
                    "sweep {sweep} collects the cut of sweep {earlier} again (same mode, fixed \
                     angle, gates and fields)"
                )?;
                if let Some(seconds) = seconds_apart {
                    write!(f, " {seconds:.0} s later")?;
                }
                Ok(())
            }
            Self::Pause {
                sweep,
                previous,
                seconds,
            } => write!(
                f,
                "sweep {sweep} starts {seconds:.0} s after sweep {previous} ended (a scan cycle \
                 pauses at most {MAX_SCAN_PAUSE_S:.0} s)"
            ),
            Self::VolumeStart { sweep } => write!(
                f,
                "sweep {sweep} has a radial that begins a volume scan (Level II radial status 3)"
            ),
        }
    }
}

/// The sweeps of one scan cycle.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct ScanCycle {
    /// Indices into [`Volume::sweeps`], in the order they were collected.
    pub sweeps: Vec<usize>,
    /// Why the cycle begins where it does; `None` for the first cycle.
    pub starts_with: Option<CycleBreak>,
}

/// A cut of the current cycle, as [`CycleTracker`] remembers it.
#[derive(Clone, Debug)]
struct TrackedCut {
    label: usize,
    sweep_mode: SweepMode,
    fixed_angle_deg: f32,
    range: RangeCoord,
    fields: Vec<String>,
    start_s: Option<f64>,
}

/// Follows the sweeps of one scan cycle as they are collected, to tell
/// whether the next sweep continues the cycle or begins another (the rules
/// of the module documentation). [`scan_cycles`] runs one over a whole
/// volume; a writer that receives a volume sweep by sweep keeps one across
/// its parts.
#[derive(Clone, Debug, Default)]
pub struct CycleTracker {
    cuts: Vec<TrackedCut>,
    /// Label and end (seconds since 1970) of the sweep that ended last.
    latest_end: Option<(usize, f64)>,
}

impl CycleTracker {
    /// A tracker with no sweeps.
    pub fn new() -> Self {
        Self::default()
    }

    /// Whether no sweep has been added since the tracker was made or cleared.
    pub fn is_empty(&self) -> bool {
        self.cuts.is_empty()
    }

    /// Forget every sweep: the next one begins a cycle.
    pub fn clear(&mut self) {
        self.cuts.clear();
        self.latest_end = None;
    }

    /// Why sweep `index` of `volume`, collected after the sweeps added so
    /// far, begins a new scan cycle; `None` when it continues the current
    /// one (or is the first). `label` names the sweep in the returned break.
    /// A sweep that does not exist or has no rays continues the cycle.
    pub fn check(&self, volume: &Volume, index: usize, label: usize) -> Option<CycleBreak> {
        let sweep = volume.sweeps.get(index).filter(|sweep| sweep.nrays() > 0)?;
        if self.cuts.is_empty() {
            return None;
        }
        let (start, _) = epoch_extent(volume, sweep).unzip();
        if volume.provenance.source_format == SourceFormat::NexradLevel2 {
            if begins_nexrad_volume(sweep) {
                return Some(CycleBreak::VolumeStart { sweep: label });
            }
        } else {
            let fields = field_names(sweep);
            if let Some(earlier) = self.cuts.iter().find(|cut| {
                cut.sweep_mode == sweep.sweep_mode
                    && (cut.fixed_angle_deg - sweep.fixed_angle_deg).abs()
                        <= ANGLE_MATCH_TOLERANCE_DEG
                    && same_range(&cut.range, &sweep.range)
                    && cut.fields == fields
            }) {
                return Some(CycleBreak::RepeatedCut {
                    sweep: label,
                    earlier: earlier.label,
                    seconds_apart: start
                        .zip(earlier.start_s)
                        .map(|(start, earlier)| start - earlier),
                });
            }
        }
        if let (Some(start), Some((previous, end))) = (start, self.latest_end) {
            let seconds = start - end;
            if seconds > MAX_SCAN_PAUSE_S {
                return Some(CycleBreak::Pause {
                    sweep: label,
                    previous,
                    seconds,
                });
            }
        }
        None
    }

    /// Add sweep `index` of `volume` to the current cycle under `label`. A
    /// sweep that does not exist or has no rays is not added.
    pub fn add(&mut self, volume: &Volume, index: usize, label: usize) {
        let Some(sweep) = volume.sweeps.get(index).filter(|sweep| sweep.nrays() > 0) else {
            return;
        };
        let extent = epoch_extent(volume, sweep);
        if let Some((_, end)) = extent
            && self.latest_end.is_none_or(|(_, latest)| end > latest)
        {
            self.latest_end = Some((label, end));
        }
        self.cuts.push(TrackedCut {
            label,
            sweep_mode: sweep.sweep_mode.clone(),
            fixed_angle_deg: sweep.fixed_angle_deg,
            range: sweep.range.clone(),
            fields: field_names(sweep),
            start_s: extent.map(|(start, _)| start),
        });
    }
}

/// Earliest and latest finite ray time of `sweep`, in seconds since 1970.
fn epoch_extent(volume: &Volume, sweep: &Sweep) -> Option<(f64, f64)> {
    let (start, end) = time_extent(sweep)?;
    let base = volume.time_reference.timestamp() as f64
        + f64::from(volume.time_reference.timestamp_subsec_nanos()) * 1e-9;
    Some((base + start, base + end))
}

/// Earliest and latest finite ray time of `sweep` (seconds from the volume's
/// time reference).
pub(super) fn time_extent(sweep: &Sweep) -> Option<(f64, f64)> {
    sweep
        .rays
        .time_s
        .iter()
        .copied()
        .filter(|time| time.is_finite())
        .fold(None, |extent, time| {
            Some(match extent {
                None => (time, time),
                Some((lo, hi)) => (f64::min(lo, time), f64::max(hi, time)),
            })
        })
}

fn field_names(sweep: &Sweep) -> Vec<String> {
    let mut names: Vec<String> = sweep
        .fields
        .iter()
        .map(|field| field.name.as_str().to_owned())
        .collect();
    names.sort_unstable();
    names
}

fn same_range(a: &RangeCoord, b: &RangeCoord) -> bool {
    let close = |x: f64, y: f64| (x - y).abs() <= RANGE_MATCH_TOLERANCE_M;
    match (a, b) {
        (
            RangeCoord::Uniform {
                first_center_m: first_a,
                spacing_m: spacing_a,
                ngates: ngates_a,
            },
            RangeCoord::Uniform {
                first_center_m: first_b,
                spacing_m: spacing_b,
                ngates: ngates_b,
            },
        ) => ngates_a == ngates_b && close(*first_a, *first_b) && close(*spacing_a, *spacing_b),
        (RangeCoord::Explicit { centers_m: a }, RangeCoord::Explicit { centers_m: b }) => {
            a.len() == b.len()
                && a.iter()
                    .zip(b)
                    .all(|(x, y)| close(f64::from(*x), f64::from(*y)))
        }
        _ => false,
    }
}

/// Whether a radial of a Level II sweep begins a volume scan.
fn begins_nexrad_volume(sweep: &Sweep) -> bool {
    let Some(status) = sweep
        .extra_vars
        .iter()
        .find(|var| &*var.name == "nexrad_radial_status" && var.is_per_ray())
    else {
        return false;
    };
    let begins = |code: u16| code & 0x7f == NEXRAD_BEGINNING_OF_VOLUME;
    match &status.values {
        ArrayBuf::U8(codes) => codes.iter().any(|code| begins(u16::from(*code))),
        ArrayBuf::U16(codes) => codes.iter().any(|code| begins(*code)),
        _ => false,
    }
}

/// Indices of `volume`'s sweeps in the order they were collected: by the
/// earliest ray time of each sweep. A sweep without a finite ray time keeps
/// its place after the sweep stored before it; ties keep storage order.
pub fn collection_order(volume: &Volume) -> Vec<usize> {
    let mut keys = Vec::with_capacity(volume.sweeps.len());
    let mut previous = f64::NEG_INFINITY;
    for sweep in &volume.sweeps {
        let key = time_extent(sweep).map_or(previous, |(start, _)| start);
        keys.push(key);
        previous = key;
    }
    let mut order: Vec<usize> = (0..volume.sweeps.len()).collect();
    order.sort_by(|a, b| keys[*a].total_cmp(&keys[*b]));
    order
}

/// The scan cycles of `volume`, in the order they were collected, each with
/// its sweeps in collection order (the rules of the module documentation).
/// A volume of one scan gives one cycle; a volume without sweeps gives none.
pub fn scan_cycles(volume: &Volume) -> Vec<ScanCycle> {
    let mut cycles: Vec<ScanCycle> = Vec::new();
    let mut tracker = CycleTracker::new();
    for index in collection_order(volume) {
        let begins = tracker.check(volume, index, index);
        match (begins, cycles.last_mut()) {
            (None, Some(cycle)) => cycle.sweeps.push(index),
            (begins, _) => {
                tracker.clear();
                cycles.push(ScanCycle {
                    sweeps: vec![index],
                    starts_with: begins,
                });
            }
        }
        tracker.add(volume, index, index);
    }
    cycles
}

/// One volume per scan cycle of `volume` ([`scan_cycles`]), in the order
/// the cycles were collected. Each holds its cycle's sweeps in collection
/// order, numbered from 0 (`sweep_number`; the elevation numbers stay as
/// they were), with the volume-level items of `volume`, its time reference
/// moved to the cycle's first ray (floored to the second, the ray and
/// calibration times shifted to match) and its time coverage set to the
/// cycle's first and last rays. A volume of one scan comes back as one
/// volume, its sweeps in collection order. The sweeps are moved, not
/// copied.
pub fn split_scan_cycles(mut volume: Volume) -> Vec<Volume> {
    let cycles = scan_cycles(&volume);
    let mut sweeps: Vec<Option<Sweep>> = std::mem::take(&mut volume.sweeps)
        .into_iter()
        .map(Some)
        .collect();
    let shell = volume;
    let mut volumes = Vec::with_capacity(cycles.len());
    for cycle in cycles {
        let mut part = shell.clone();
        part.sweeps = cycle
            .sweeps
            .iter()
            .filter_map(|index| sweeps.get_mut(*index).and_then(Option::take))
            .collect();
        for (number, sweep) in part.sweeps.iter_mut().enumerate() {
            sweep.sweep_number = u32::try_from(number).unwrap_or(u32::MAX);
        }
        rebase_to_first_ray(&mut part);
        volumes.push(part);
    }
    volumes
}

/// Move `volume`'s time reference to its first ray (floored to the second)
/// and set its time coverage to its first and last rays.
fn rebase_to_first_ray(volume: &mut Volume) {
    let Some((first, last)) = volume
        .sweeps
        .iter()
        .filter_map(time_extent)
        .reduce(|(lo, hi), (start, end)| (lo.min(start), hi.max(end)))
    else {
        return;
    };
    let Some(first_time) = volume.instant(first) else {
        return;
    };
    let reference: DateTime<Utc> = floor_to_second(first_time);
    // Both are whole seconds; a shift beyond a century is left alone (the
    // offsets would lose their precision).
    let difference = volume.time_reference - reference;
    let shift = if difference.abs() < Duration::days(36_500) {
        difference.num_milliseconds() as f64 / 1000.0
    } else {
        0.0
    };
    if shift != 0.0 {
        for sweep in &mut volume.sweeps {
            sweep.rays.time_s.iter_mut().for_each(|time| *time += shift);
        }
        for calibration in &mut volume.radar_calibration {
            if let Some(time) = calibration.time_s.as_mut() {
                *time += shift;
            }
        }
        volume.time_reference = reference;
    }
    if let (Some(start), Some(end)) = (volume.instant(first + shift), volume.instant(last + shift))
    {
        volume.time_coverage = Some(TimeCoverage { start, end });
    }
}
