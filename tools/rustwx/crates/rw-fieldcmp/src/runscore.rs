//! Distance between two runs of the same case, over a registered metric set.
//!
//! Where [`crate::compare`] reports what each arm looks like frame by frame,
//! this reports how far apart the two arms are as runs: one number per metric
//! class, per domain, per field, pooled over every scored time.  Four classes
//! are computed, and each takes its geometry and its thresholds from the
//! caller rather than knowing a campaign of its own:
//!
//! * **state** -- pooled RMSE of the low-pass-filtered difference over the
//!   domain interior;
//! * **boundary** -- pooled RMSE of the difference between the two arms'
//!   per-interval increments, over the outer frame of cells;
//! * **object** -- the gap between the two arms' first qualifying
//!   composite object, in seconds;
//! * **neighbourhood** -- mean `1 - FSS` at a physical radius.
//!
//! # Reading once
//!
//! The reference reads each frame four times per field: once as the current
//! frame of its own interval and once as the previous frame of the next, on
//! each arm, and opens the file afresh every time.  Over a four-domain
//! seven-field ladder that is 672 decodes of the state fields and 720 file
//! opens for a single pair.
//!
//! This walks the ladder once instead.  Every frame is opened once per arm,
//! every field is decoded once from that open handle, and each decode is
//! reduced immediately to the two things later times need from it: the scored
//! interior sum of squares, which needs only the frame's own two arms, and
//! the outer ring of cells, which is the only part of the raw field the next
//! interval's boundary metric will look at.  The full field is dropped before
//! the next one is read, so the peak is a handful of arrays rather than a
//! ladder of them, and the frames run in parallel.
//!
//! Selecting the ring before differencing rather than after is a
//! rearrangement of the reference's expression, not a change to it: the
//! selection is positional and the arithmetic is element-wise, so the same
//! cells are combined in the same order and the sums land on the same bits.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use rayon::prelude::*;
use serde::Serialize;

use crate::gridops::{
    half_width_cells, minimum_object_cells, odd_width_cells, Field, ShapeError,
};
use crate::stats::{pairwise_sum, pairwise_sum_by, PooledRmse};
use crate::OpenFrame;

/// Why a paired-run score could not be produced.
#[derive(Debug, thiserror::Error)]
pub enum ScoreError {
    #[error("cannot read history frames: {0}")]
    Read(#[from] crate::Error),

    #[error("{0}")]
    Shape(#[from] ShapeError),

    #[error("{context}: {source}")]
    Field {
        context: String,
        #[source]
        source: ShapeError,
    },

    #[error("cannot read the start instant {start:?}: expected YYYY-MM-DD[T ]HH:MM:SS")]
    StartInstant { start: String },

    #[error("{path} is not a {prefix}_{domain}_<valid-time> history frame")]
    FrameName {
        path: PathBuf,
        prefix: String,
        domain: String,
    },

    #[error("{domain} carries two history frames for second {seconds}")]
    DuplicateFrame { domain: String, seconds: i64 },

    #[error("{arm} {domain} frame times {found:?} are not the registered ladder {expected:?}")]
    FrameLadder {
        arm: String,
        domain: String,
        found: Vec<i64>,
        expected: Vec<i64>,
    },

    #[error("{variable} in {path} has shape {shape:?}, which carries no field")]
    NotAField { variable: String, path: PathBuf, shape: Vec<usize> },

    #[error("{variable} in {path} is empty or carries {count} non-finite cells")]
    NotFinite {
        variable: String,
        path: PathBuf,
        count: usize,
    },

    #[error("{domain}/{variable} has shape {left:?} in one arm and {right:?} in the other")]
    ArmShapes {
        domain: String,
        variable: String,
        left: Vec<usize>,
        right: Vec<usize>,
    },

    #[error("{domain}/{variable} changes shape between {earlier}s and {later}s")]
    ShapeInTime {
        domain: String,
        variable: String,
        earlier: i64,
        later: i64,
    },

    #[error("the run duration {run_seconds}s is not a whole number of {cadence_seconds}s intervals")]
    Cadence {
        run_seconds: i64,
        cadence_seconds: i64,
    },

    #[error("a paired-run score needs at least one domain and one field")]
    NothingToScore,

    #[error("{label} is named twice, and its second scoring would silently replace its first")]
    RepeatedLabel { label: String },

    #[error("{metric} produced no samples")]
    NoSamples { metric: String },

    #[error("{metric} is {value}, which is not a distance")]
    NotADistance { metric: String, value: f64 },
}

type Result<T> = std::result::Result<T, ScoreError>;

/// One domain of the ladder: what it is called, how far apart its cells are,
/// and whether it carries the neighbourhood row.
#[derive(Debug, Clone)]
pub struct DomainSpec {
    pub label: String,
    pub dx_m: f64,
    pub score_neighborhood: bool,
}

/// The metric-key spellings the caller's scoreboard expects.
///
/// These are strings rather than constants because a published gate record's
/// key strings are part of what it published: a scorer that renamed them
/// would produce the same numbers under names its own evidence does not use.
#[derive(Debug, Clone)]
pub struct MetricKeys {
    pub state: String,
    pub boundary: String,
    pub object: String,
    pub object_subject: String,
    pub neighborhood: String,
    pub neighborhood_subject: String,
}

impl Default for MetricKeys {
    fn default() -> Self {
        Self {
            state: "low_pass_state_rmse".to_string(),
            boundary: "applied_boundary_increment_error".to_string(),
            object: "storm_object_timing_difference".to_string(),
            object_subject: "first_object".to_string(),
            neighborhood: "neighborhood_fss_distance".to_string(),
            neighborhood_subject: "composite".to_string(),
        }
    }
}

/// Everything a paired-run score needs.
#[derive(Debug, Clone)]
pub struct RunScoreRequest {
    pub left: PathBuf,
    pub right: PathBuf,
    pub frame_prefix: String,
    pub start: String,
    pub run_seconds: i64,
    pub cadence_seconds: i64,
    pub domains: Vec<DomainSpec>,
    pub fields: Vec<String>,
    pub low_pass_width_m: f64,
    pub interior_exclusion_cells: usize,
    pub boundary_width_cells: usize,
    pub composite_variable: String,
    pub threshold: f64,
    pub neighborhood_radius_m: f64,
    pub object_connectivity: u8,
    pub object_min_area_km2: f64,
    pub keys: MetricKeys,
}

/// One pooled-RMSE sample, before pooling.
///
/// A verification instrument that disagrees with its reference by one part in
/// ten thousand million is useless unless it can say *where*: these are the
/// per-interval sums of squares the two pooled numbers are built from, so a
/// disagreement can be localised to one domain, one field and one interval
/// rather than argued about at the metric.
#[derive(Debug, Clone, Serialize)]
pub struct Sample {
    pub domain: String,
    pub field: String,
    pub seconds: i64,
    pub class: &'static str,
    pub sum_of_squares: f64,
    pub count: usize,
}

/// What one domain cost and what geometry it was scored under.
#[derive(Debug, Clone, Serialize)]
pub struct DomainReport {
    pub domain: String,
    pub dx_m: f64,
    pub frames_per_arm: usize,
    pub scored_times: Vec<i64>,
    pub filter_width_cells: usize,
    pub neighborhood_half_width_cells: usize,
    pub minimum_object_cells: usize,
    pub variable_decodes: usize,
    pub frame_opens: usize,
    pub samples: Vec<Sample>,
}

/// A whole paired-run score.
#[derive(Debug, Clone, Serialize)]
pub struct RunScore {
    pub left_directory: String,
    pub right_directory: String,
    pub start: String,
    pub run_seconds: i64,
    pub cadence_seconds: i64,
    pub distances: BTreeMap<String, f64>,
    pub domains: Vec<DomainReport>,
    pub variable_decodes: usize,
    pub frame_opens: usize,
}

/// Score a pair of run directories.  Domains run in sequence and the frames
/// inside a domain run in parallel, so peak memory is set by how many frames
/// are in flight rather than by how long the ladder is.
pub fn score(request: &RunScoreRequest) -> Result<RunScore> {
    if request.domains.is_empty() || request.fields.is_empty() {
        return Err(ScoreError::NothingToScore);
    }
    if request.cadence_seconds <= 0 || request.run_seconds % request.cadence_seconds != 0 {
        return Err(ScoreError::Cadence {
            run_seconds: request.run_seconds,
            cadence_seconds: request.cadence_seconds,
        });
    }
    // Every metric key carries its domain's label, so two domains under one
    // label would quietly overwrite each other's whole row set and the
    // scoreboard would show the second one's numbers under both names.
    for (index, domain) in request.domains.iter().enumerate() {
        if request.domains[..index]
            .iter()
            .any(|earlier| earlier.label == domain.label)
        {
            return Err(ScoreError::RepeatedLabel {
                label: domain.label.clone(),
            });
        }
    }
    let start = parse_instant(&request.start)?;
    let times: Vec<i64> = (0..=request.run_seconds)
        .step_by(request.cadence_seconds as usize)
        .collect();

    let mut distances: BTreeMap<String, f64> = BTreeMap::new();
    let mut reports = Vec::with_capacity(request.domains.len());
    for domain in &request.domains {
        let report = score_domain(request, domain, start, &times, &mut distances)?;
        reports.push(report);
    }

    for (metric, value) in &distances {
        if !value.is_finite() || *value < 0.0 {
            return Err(ScoreError::NotADistance {
                metric: metric.clone(),
                value: *value,
            });
        }
    }

    Ok(RunScore {
        left_directory: request.left.display().to_string(),
        right_directory: request.right.display().to_string(),
        start: request.start.clone(),
        run_seconds: request.run_seconds,
        cadence_seconds: request.cadence_seconds,
        distances,
        variable_decodes: reports.iter().map(|r| r.variable_decodes).sum(),
        frame_opens: reports.iter().map(|r| r.frame_opens).sum(),
        domains: reports,
    })
}

/// Everything one frame of one domain contributes to the domain's score.
///
/// The two heavy inputs -- the pair of full fields -- are gone by the time
/// this exists; what survives is one scalar per field for the interval that
/// ends here and the outer ring of cells the next interval will need.
struct FrameWork {
    seconds: i64,
    shapes: Vec<Vec<usize>>,
    /// Sum of squares and cell count of the low-pass difference, absent on
    /// the first frame because no interval ends there.
    low_pass: Option<Vec<(f64, usize)>>,
    rings: Vec<(Vec<f64>, Vec<f64>)>,
    composite: Option<(Field, Field)>,
    decodes: usize,
}

fn score_domain(
    request: &RunScoreRequest,
    domain: &DomainSpec,
    start: i64,
    times: &[i64],
    distances: &mut BTreeMap<String, f64>,
) -> Result<DomainReport> {
    let left_frames = discover_frames(&request.left, domain, request, start, times, "left")?;
    let right_frames = discover_frames(&request.right, domain, request, start, times, "right")?;
    let filter_width = odd_width_cells(request.low_pass_width_m, domain.dx_m);
    let half_width = half_width_cells(request.neighborhood_radius_m, domain.dx_m);
    let min_cells = minimum_object_cells(request.object_min_area_km2, domain.dx_m);
    let scored: Vec<i64> = times.iter().copied().skip(1).collect();

    let works: Vec<FrameWork> = times
        .par_iter()
        .map(|&seconds| {
            read_frame(
                request,
                domain,
                seconds,
                &left_frames[&seconds],
                &right_frames[&seconds],
                filter_width,
                seconds != times[0],
            )
        })
        .collect::<Result<Vec<_>>>()?;
    let by_time: BTreeMap<i64, &FrameWork> =
        works.iter().map(|work| (work.seconds, work)).collect();

    let mut samples: Vec<Sample> = Vec::new();
    for (index, field) in request.fields.iter().enumerate() {
        let mut state = PooledRmse::new();
        let mut boundary = PooledRmse::new();
        let mut previous = &by_time[&times[0]];
        for seconds in &scored {
            let work = &by_time[seconds];
            if work.shapes[index] != previous.shapes[index] {
                return Err(ScoreError::ShapeInTime {
                    domain: domain.label.clone(),
                    variable: field.clone(),
                    earlier: previous.seconds,
                    later: work.seconds,
                });
            }
            let (sum, count) = work.low_pass.as_ref().expect("scored frame")[index];
            state.push(sum, count);
            samples.push(Sample {
                domain: domain.label.clone(),
                field: field.clone(),
                seconds: *seconds,
                class: "state",
                sum_of_squares: sum,
                count,
            });

            // ((left_now - left_then) - (right_now - right_then)) over the
            // ring, squared and summed in the order the reference's packed
            // temporary holds them.
            let (left_now, right_now) = &work.rings[index];
            let (left_then, right_then) = &previous.rings[index];
            let sum = pairwise_sum_by(left_now.len(), &|cell| {
                let error = (left_now[cell] - left_then[cell]) - (right_now[cell] - right_then[cell]);
                error * error
            });
            boundary.push(sum, left_now.len());
            samples.push(Sample {
                domain: domain.label.clone(),
                field: field.clone(),
                seconds: *seconds,
                class: "boundary",
                sum_of_squares: sum,
                count: left_now.len(),
            });
            previous = work;
        }
        let state_key = format!("{}:{}:{}", request.keys.state, domain.label, field);
        let boundary_key = format!("{}:{}:{}", request.keys.boundary, domain.label, field);
        distances.insert(
            state_key.clone(),
            state.finish().ok_or(ScoreError::NoSamples { metric: state_key })?,
        );
        distances.insert(
            boundary_key.clone(),
            boundary
                .finish()
                .ok_or(ScoreError::NoSamples { metric: boundary_key })?,
        );
    }

    let quiet = (request.run_seconds + request.cadence_seconds) as f64;
    let mut left_first = quiet;
    let mut right_first = quiet;
    let mut found_left = false;
    let mut found_right = false;
    for seconds in &scored {
        let (left, right) = by_time[seconds].composite.as_ref().expect("scored frame");
        if !found_left
            && has_qualifying_object(left, request.threshold, min_cells, request.object_connectivity)
        {
            left_first = *seconds as f64;
            found_left = true;
        }
        if !found_right
            && has_qualifying_object(right, request.threshold, min_cells, request.object_connectivity)
        {
            right_first = *seconds as f64;
            found_right = true;
        }
    }
    distances.insert(
        format!(
            "{}:{}:{}",
            request.keys.object, domain.label, request.keys.object_subject
        ),
        (left_first - right_first).abs(),
    );

    if domain.score_neighborhood {
        let mut per_time = Vec::with_capacity(scored.len());
        for seconds in &scored {
            let (left, right) = by_time[seconds].composite.as_ref().expect("scored frame");
            per_time.push(neighborhood_distance(left, right, request.threshold, half_width)?);
        }
        let key = format!(
            "{}:{}:{}",
            request.keys.neighborhood, domain.label, request.keys.neighborhood_subject
        );
        if per_time.is_empty() {
            return Err(ScoreError::NoSamples { metric: key });
        }
        distances.insert(key, pairwise_sum(&per_time) / per_time.len() as f64);
    }

    Ok(DomainReport {
        domain: domain.label.clone(),
        dx_m: domain.dx_m,
        frames_per_arm: times.len(),
        scored_times: scored,
        filter_width_cells: filter_width,
        neighborhood_half_width_cells: half_width,
        minimum_object_cells: min_cells,
        variable_decodes: works.iter().map(|work| work.decodes).sum(),
        frame_opens: 2 * times.len(),
        samples,
    })
}

/// Read one frame of one domain on both arms and reduce it on the spot.
fn read_frame(
    request: &RunScoreRequest,
    domain: &DomainSpec,
    seconds: i64,
    left_path: &Path,
    right_path: &Path,
    filter_width: usize,
    scored: bool,
) -> Result<FrameWork> {
    let (left, right) = rayon::join(
        || OpenFrame::open(left_path.to_path_buf()),
        || OpenFrame::open(right_path.to_path_buf()),
    );
    let (left, right) = (left?, right?);

    let mut shapes = Vec::with_capacity(request.fields.len());
    let mut low_pass = Vec::with_capacity(request.fields.len());
    let mut rings = Vec::with_capacity(request.fields.len());
    let mut decodes = 0usize;

    for field in &request.fields {
        let (left_field, right_field) = rayon::join(
            || read_field(&left, field),
            || read_field(&right, field),
        );
        let (left_field, right_field) = (left_field?, right_field?);
        decodes += 2;
        if left_field.shape() != right_field.shape() {
            return Err(ScoreError::ArmShapes {
                domain: domain.label.clone(),
                variable: field.clone(),
                left: left_field.shape().to_vec(),
                right: right_field.shape().to_vec(),
            });
        }
        shapes.push(left_field.shape().to_vec());

        if scored {
            let (left_smooth, right_smooth) = rayon::join(
                || left_field.boxcar(filter_width),
                || right_field.boxcar(filter_width),
            );
            let (left_smooth, right_smooth) = (left_smooth?, right_smooth?);
            let difference = Field::new(
                left_smooth.shape().to_vec(),
                left_smooth
                    .values()
                    .iter()
                    .zip(right_smooth.values())
                    .map(|(a, b)| a - b)
                    .collect(),
            )?;
            let interior = difference
                .interior(request.interior_exclusion_cells)
                .map_err(|source| ScoreError::Field {
                    context: format!("{}/{} interior", domain.label, field),
                    source,
                })?;
            let values = interior.values();
            let sum = pairwise_sum_by(values.len(), &|cell| values[cell] * values[cell]);
            low_pass.push((sum, values.len()));
        }

        let (left_ring, right_ring) = rayon::join(
            || left_field.boundary_values(request.boundary_width_cells),
            || right_field.boundary_values(request.boundary_width_cells),
        );
        let context = || format!("{}/{} boundary", domain.label, field);
        rings.push((
            left_ring
                .map_err(|source| ScoreError::Field { context: context(), source })?
                .into_values(),
            right_ring
                .map_err(|source| ScoreError::Field { context: context(), source })?
                .into_values(),
        ));
    }

    let composite = if scored {
        let (left_volume, right_volume) = rayon::join(
            || read_field(&left, &request.composite_variable),
            || read_field(&right, &request.composite_variable),
        );
        let (left_volume, right_volume) = (left_volume?, right_volume?);
        decodes += 2;
        if left_volume.shape() != right_volume.shape() {
            return Err(ScoreError::ArmShapes {
                domain: domain.label.clone(),
                variable: request.composite_variable.clone(),
                left: left_volume.shape().to_vec(),
                right: right_volume.shape().to_vec(),
            });
        }
        Some((left_volume.composite(), right_volume.composite()))
    } else {
        None
    };

    Ok(FrameWork {
        seconds,
        shapes,
        low_pass: if scored { Some(low_pass) } else { None },
        rings,
        composite,
        decodes,
    })
}

/// Read one record of one variable, refusing a field with a gap in it.
///
/// The reference refuses masked cells and non-finite cells separately; this
/// reader promotes on read and checks finiteness, which catches both a fill
/// value that survived as a NaN and an arithmetic that went bad.  A masked
/// cell backed by a finite fill value would pass here and be refused there,
/// so a stream that uses CF masking on the scored fields is outside what this
/// instrument claims to reproduce.
fn read_field(frame: &OpenFrame, variable: &str) -> Result<Field> {
    let (shape, values) = frame.read(variable)?;
    if shape.len() < 2 {
        return Err(ScoreError::NotAField {
            variable: variable.to_string(),
            path: frame.path().to_path_buf(),
            shape,
        });
    }
    let bad = values.iter().filter(|value| !value.is_finite()).count();
    if values.is_empty() || bad > 0 {
        return Err(ScoreError::NotFinite {
            variable: variable.to_string(),
            path: frame.path().to_path_buf(),
            count: bad,
        });
    }
    Ok(Field::new(shape, values)?)
}

/// Does any connected run of at-or-above-threshold cells reach `min_cells`?
///
/// The scan is a flood fill that stops the moment one component is large
/// enough, so a frame with an obvious storm in it costs a fraction of a frame
/// that is quiet everywhere.
fn has_qualifying_object(
    plane: &Field,
    threshold: f64,
    min_cells: usize,
    connectivity: u8,
) -> bool {
    let shape = plane.shape();
    let (rows, columns) = (shape[shape.len() - 2], shape[shape.len() - 1]);
    let values = plane.values();
    let mut seen = vec![false; values.len()];
    let mut stack: Vec<usize> = Vec::new();
    let diagonal = connectivity == 8;
    for origin in 0..values.len() {
        // Spelled as a negated comparison, deliberately: a cell that is not a
        // number must fail the test rather than pass it, which is how the
        // reference's `field >= threshold` treats one.  `< threshold` would
        // admit it.
        #[allow(clippy::neg_cmp_op_on_partial_ord)]
        let inactive = !(values[origin] >= threshold);
        if inactive || seen[origin] {
            continue;
        }
        seen[origin] = true;
        stack.clear();
        stack.push(origin);
        let mut size = 0usize;
        while let Some(cell) = stack.pop() {
            size += 1;
            if size >= min_cells {
                return true;
            }
            let (row, column) = (cell / columns, cell % columns);
            for row_step in -1i64..=1 {
                for column_step in -1i64..=1 {
                    if (row_step == 0 && column_step == 0)
                        || (!diagonal && row_step != 0 && column_step != 0)
                    {
                        continue;
                    }
                    let neighbour_row = row as i64 + row_step;
                    let neighbour_column = column as i64 + column_step;
                    if neighbour_row < 0
                        || neighbour_column < 0
                        || neighbour_row >= rows as i64
                        || neighbour_column >= columns as i64
                    {
                        continue;
                    }
                    let neighbour = neighbour_row as usize * columns + neighbour_column as usize;
                    if values[neighbour] >= threshold && !seen[neighbour] {
                        seen[neighbour] = true;
                        stack.push(neighbour);
                    }
                }
            }
        }
    }
    false
}

/// `1 - FSS` between two planes at one neighbourhood half-width.
///
/// Both fractions being identically zero -- neither arm has an event anywhere
/// -- is agreement, not a division by zero, and scores as a perfect skill of
/// one and a distance of zero.
fn neighborhood_distance(
    left: &Field,
    right: &Field,
    threshold: f64,
    half_width: usize,
) -> Result<f64> {
    let events = |field: &Field| -> std::result::Result<Field, ShapeError> {
        Field::new(
            field.shape().to_vec(),
            field
                .values()
                .iter()
                .map(|&value| if value >= threshold { 1.0 } else { 0.0 })
                .collect(),
        )?
        .boxcar(2 * half_width + 1)
    };
    let (left_fraction, right_fraction) = rayon::join(|| events(left), || events(right));
    let (left_fraction, right_fraction) = (left_fraction?, right_fraction?);
    let (a, b) = (left_fraction.values(), right_fraction.values());
    let numerator = pairwise_sum_by(a.len(), &|cell| {
        let difference = a[cell] - b[cell];
        difference * difference
    });
    let denominator = pairwise_sum_by(a.len(), &|cell| a[cell] * a[cell] + b[cell] * b[cell]);
    let skill = if denominator == 0.0 {
        1.0
    } else {
        1.0 - numerator / denominator
    };
    let distance = 1.0 - skill;
    if !distance.is_finite() {
        return Err(ScoreError::NotADistance {
            metric: "neighborhood".to_string(),
            value: distance,
        });
    }
    Ok(distance.clamp(0.0, 1.0))
}

/// The frames of one domain, keyed by their offset from the start instant.
fn discover_frames(
    directory: &Path,
    domain: &DomainSpec,
    request: &RunScoreRequest,
    start: i64,
    expected: &[i64],
    arm: &str,
) -> Result<BTreeMap<i64, PathBuf>> {
    let leader = format!("{}_{}_", request.frame_prefix, domain.label);
    let entries = std::fs::read_dir(directory).map_err(|source| {
        ScoreError::Read(crate::Error::ListDirectory {
            path: directory.to_path_buf(),
            source,
        })
    })?;
    let mut found: BTreeMap<i64, PathBuf> = BTreeMap::new();
    for entry in entries {
        let entry = entry.map_err(|source| {
            ScoreError::Read(crate::Error::ListDirectory {
                path: directory.to_path_buf(),
                source,
            })
        })?;
        let name = entry.file_name().to_string_lossy().into_owned();
        if !name.starts_with(&leader) {
            continue;
        }
        let path = entry.path();
        let seconds = parse_instant(&name[leader.len()..]).map_err(|_| ScoreError::FrameName {
            path: path.clone(),
            prefix: request.frame_prefix.clone(),
            domain: domain.label.clone(),
        })? - start;
        if found.insert(seconds, path).is_some() {
            return Err(ScoreError::DuplicateFrame {
                domain: domain.label.clone(),
                seconds,
            });
        }
    }
    let times: Vec<i64> = found.keys().copied().collect();
    if times != expected {
        return Err(ScoreError::FrameLadder {
            arm: arm.to_string(),
            domain: domain.label.clone(),
            found: times,
            expected: expected.to_vec(),
        });
    }
    Ok(found)
}

/// Seconds since the epoch of a `YYYY-MM-DD` date and `HH:MM:SS` time, where
/// the two may be joined by `T`, a space or an underscore and the time's own
/// separator may be a colon or an underscore.
///
/// History frames are named with underscores because a colon is not a
/// filename character everywhere, and a registration is written with colons
/// because it is ISO-8601; one parser reads both rather than two agreeing.
fn parse_instant(text: &str) -> Result<i64> {
    let bad = || ScoreError::StartInstant {
        start: text.to_string(),
    };
    let trimmed = text.trim();
    let bytes: Vec<char> = trimmed.chars().collect();
    if bytes.len() < 19 {
        return Err(bad());
    }
    let number = |from: usize, to: usize| -> Result<i64> {
        trimmed
            .get(from..to)
            .ok_or_else(bad)?
            .parse::<i64>()
            .map_err(|_| bad())
    };
    let separator = |at: usize, allowed: &[char]| -> Result<()> {
        if allowed.contains(&bytes[at]) {
            Ok(())
        } else {
            Err(bad())
        }
    };
    separator(4, &['-'])?;
    separator(7, &['-'])?;
    separator(10, &['T', ' ', '_'])?;
    separator(13, &[':', '_'])?;
    separator(16, &[':', '_'])?;
    let (year, month, day) = (number(0, 4)?, number(5, 7)?, number(8, 10)?);
    let (hour, minute, second) = (number(11, 13)?, number(14, 16)?, number(17, 19)?);
    if !(1..=12).contains(&month) || !(1..=31).contains(&day) {
        return Err(bad());
    }
    if !(0..24).contains(&hour) || !(0..60).contains(&minute) || !(0..61).contains(&second) {
        return Err(bad());
    }
    Ok(days_from_civil(year, month, day) * 86_400 + hour * 3_600 + minute * 60 + second)
}

/// Days between the epoch and a proleptic-Gregorian date.
fn days_from_civil(year: i64, month: i64, day: i64) -> i64 {
    let year = if month <= 2 { year - 1 } else { year };
    let era = if year >= 0 { year } else { year - 399 } / 400;
    let year_of_era = year - era * 400;
    let shifted_month = (month + 9) % 12;
    let day_of_year = (153 * shifted_month + 2) / 5 + day - 1;
    let day_of_era = year_of_era * 365 + year_of_era / 4 - year_of_era / 100 + day_of_year;
    era * 146_097 + day_of_era - 719_468
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn instants_parse_from_both_spellings() {
        let iso = parse_instant("1974-04-03T12:00:00").expect("iso");
        let frame = parse_instant("1974-04-03_12_00_00").expect("frame");
        let spaced = parse_instant("1974-04-03 12:00:00").expect("spaced");
        assert_eq!(iso, frame);
        assert_eq!(iso, spaced);
        assert_eq!(parse_instant("1974-04-03_12_05_00").expect("later") - iso, 300);
    }

    #[test]
    fn the_epoch_and_a_leap_day_land_where_they_should() {
        assert_eq!(parse_instant("1970-01-01T00:00:00").expect("epoch"), 0);
        assert_eq!(
            parse_instant("2000-03-01T00:00:00").expect("after leap day")
                - parse_instant("2000-02-28T00:00:00").expect("before leap day"),
            2 * 86_400
        );
    }

    #[test]
    fn a_malformed_instant_is_refused_rather_than_guessed() {
        for text in [
            "1974-04-03",
            "1974-04-03_12_00",
            "1974/04/03_12_00_00",
            "1974-13-03_12_00_00",
            "1974-04-03_25_00_00",
        ] {
            assert!(parse_instant(text).is_err(), "{text} should not parse");
        }
    }

    fn plane(rows: usize, columns: usize, cells: &[(usize, usize)]) -> Field {
        let mut values = vec![0.0f64; rows * columns];
        for &(row, column) in cells {
            values[row * columns + column] = 50.0;
        }
        Field::new(vec![rows, columns], values).expect("shape")
    }

    #[test]
    fn an_object_qualifies_only_once_it_is_big_enough() {
        let field = plane(6, 6, &[(1, 1), (1, 2), (2, 1)]);
        assert!(has_qualifying_object(&field, 40.0, 3, 8));
        assert!(!has_qualifying_object(&field, 40.0, 4, 8));
        // Below threshold there is no object at all, however large the blob.
        assert!(!has_qualifying_object(&field, 60.0, 1, 8));
    }

    #[test]
    fn connectivity_decides_whether_a_diagonal_touch_is_one_object() {
        let field = plane(6, 6, &[(1, 1), (2, 2)]);
        assert!(has_qualifying_object(&field, 40.0, 2, 8));
        assert!(!has_qualifying_object(&field, 40.0, 2, 4));
    }

    #[test]
    fn two_identical_planes_are_at_zero_neighborhood_distance() {
        let field = plane(20, 20, &[(5, 5), (5, 6), (6, 5)]);
        let distance = neighborhood_distance(&field, &field, 40.0, 2).expect("distance");
        assert_eq!(distance, 0.0);
    }

    #[test]
    fn two_empty_planes_agree_rather_than_dividing_by_zero() {
        let field = plane(20, 20, &[]);
        assert_eq!(
            neighborhood_distance(&field, &field, 40.0, 2).expect("distance"),
            0.0
        );
    }

    #[test]
    fn a_displaced_object_scores_worse_the_further_it_moves() {
        let here = plane(40, 40, &[(20, 20)]);
        let near = plane(40, 40, &[(20, 22)]);
        let far = plane(40, 40, &[(20, 34)]);
        let close = neighborhood_distance(&here, &near, 40.0, 5).expect("distance");
        let distant = neighborhood_distance(&here, &far, 40.0, 5).expect("distance");
        assert!(close > 0.0);
        assert!(distant > close, "{distant} should exceed {close}");
        // Beyond the neighbourhood the two share no window at all, which is
        // total disagreement rather than an unbounded number.
        assert_eq!(distant, 1.0);
    }
}
