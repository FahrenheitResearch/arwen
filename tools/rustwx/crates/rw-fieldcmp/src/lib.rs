//! Paired history-stream field comparison.
//!
//! Two directories of matching history frames go in; a metric table and a
//! machine-readable record of the same numbers come out.  For every frame
//! that exists under the same name on both sides the judge reports, per named
//! field, each arm's mean/p10/p50/p99/max and their differences; per named
//! accumulation field, each arm's domain sum and their ratio; and, for a
//! named volume field, the per-column composite's coverage counts at each
//! threshold plus its maximum.
//!
//! The reading discipline is the point of the crate.  Each frame is opened
//! once per arm, each requested variable is read once from that open handle,
//! and each array is reduced to its statistics and dropped before the next
//! one is read.  Frames are compared in parallel and the two arms of a frame
//! are read concurrently, so a comparison is bounded by storage throughput
//! rather than by a single core.  Nothing here is specific to a case, a
//! model configuration, or a physics suite: the field list, the labels, the
//! accumulation list, the composite variable and its thresholds all arrive
//! from the caller, and the defaults name only WRF-convention variables.

pub mod gridops;
pub mod numfmt;
pub mod report;
pub mod runscore;
pub mod stats;

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use rayon::prelude::*;
use serde::Serialize;

use stats::Summary;

/// Errors a comparison can fail with.  A judge that cannot read one side of
/// a pair reports that rather than scoring the side it could read.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("cannot list history frames in {path}: {source}")]
    ListDirectory {
        path: PathBuf,
        source: std::io::Error,
    },

    #[error("no history frames matching {prefix}* are present in both {left} and {right}")]
    NoCommonFrames {
        prefix: String,
        left: PathBuf,
        right: PathBuf,
    },

    #[error("cannot open history frame {path}: {source}")]
    OpenFrame {
        path: PathBuf,
        source: netcrust::Error,
    },

    #[error("cannot read {variable} from {path}: {source}")]
    ReadVariable {
        variable: String,
        path: PathBuf,
        source: netcrust::Error,
    },

    #[error("{variable} in {path} has shape {shape:?}, which is not a single 2-D field")]
    NotAPlane {
        variable: String,
        path: PathBuf,
        shape: Vec<usize>,
    },

    #[error("{variable} in {path} has shape {shape:?}, which is not a single 3-D volume")]
    NotAVolume {
        variable: String,
        path: PathBuf,
        shape: Vec<usize>,
    },

    #[error("{variable} has {left} cells in the left arm and {right} in the right arm")]
    ShapeMismatch {
        variable: String,
        left: usize,
        right: usize,
    },

    #[error("cannot write {path}: {source}")]
    Write {
        path: PathBuf,
        source: std::io::Error,
    },
}

pub type Result<T> = std::result::Result<T, Error>;

/// One named field and the label the table prints for it.
#[derive(Debug, Clone)]
pub struct FieldSpec {
    pub variable: String,
    pub label: Option<String>,
}

impl FieldSpec {
    pub fn new(variable: impl Into<String>) -> Self {
        Self {
            variable: variable.into(),
            label: None,
        }
    }

    pub fn labelled(variable: impl Into<String>, label: impl Into<String>) -> Self {
        Self {
            variable: variable.into(),
            label: Some(label.into()),
        }
    }
}

/// The volume field reduced to a per-column composite, with the coverage
/// thresholds counted on it.
#[derive(Debug, Clone)]
pub struct CompositeSpec {
    pub variable: String,
    pub label: String,
    pub thresholds: Vec<f64>,
}

/// Everything a comparison needs.  Defaults name WRF-convention variables
/// only; every label is the caller's to choose.
#[derive(Debug, Clone)]
pub struct CompareRequest {
    pub left: PathBuf,
    pub right: PathBuf,
    pub left_label: String,
    pub right_label: String,
    pub frame_prefix: String,
    pub fields: Vec<FieldSpec>,
    pub accumulations: Vec<String>,
    pub composite: Option<CompositeSpec>,
}

impl CompareRequest {
    /// A request over two directories with the default field set.
    pub fn new(left: impl Into<PathBuf>, right: impl Into<PathBuf>) -> Self {
        Self {
            left: left.into(),
            right: right.into(),
            left_label: "left".to_string(),
            right_label: "right".to_string(),
            frame_prefix: "wrfout".to_string(),
            fields: default_fields(),
            accumulations: vec!["RAINNC".to_string(), "RAINC".to_string()],
            composite: Some(CompositeSpec {
                variable: "REFL_10CM".to_string(),
                label: "comp-dBZ".to_string(),
                thresholds: vec![20.0, 40.0],
            }),
        }
    }
}

/// The surface fields a WRF-convention history stream carries that a paired
/// comparison scores by default.
pub fn default_fields() -> Vec<FieldSpec> {
    ["RAINNC", "RAINC", "PBLH", "T2", "Q2", "GLW", "OLR", "SWDOWN"]
        .into_iter()
        .map(FieldSpec::new)
        .collect()
}

/// One field's outcome in one frame.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum FieldOutcome {
    /// The variable is absent from at least one arm, so no comparison exists.
    Missing {
        variable: String,
        label: String,
        left_present: bool,
        right_present: bool,
    },
    Present {
        variable: String,
        label: String,
        cells: usize,
        left: SummaryRecord,
        right: SummaryRecord,
        delta: SummaryRecord,
        /// Cells that are not finite, counted per arm.  The reference judge
        /// reports statistics over whatever it read; this count is what tells
        /// a reader whether those statistics mean anything.
        left_nonfinite: usize,
        right_nonfinite: usize,
    },
}

/// A five-number summary in a form that serialises.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct SummaryRecord {
    pub mean: f64,
    pub p10: f64,
    pub p50: f64,
    pub p99: f64,
    pub max: f64,
}

impl From<Summary> for SummaryRecord {
    fn from(summary: Summary) -> Self {
        Self {
            mean: summary.mean,
            p10: summary.p10,
            p50: summary.p50,
            p99: summary.p99,
            max: summary.max,
        }
    }
}

/// One accumulation field's domain sums.
#[derive(Debug, Clone, Serialize)]
pub struct AccumulationOutcome {
    pub variable: String,
    /// Sum accumulated in single precision, which is what the reference judge
    /// reports and what the table prints.
    pub left: f64,
    pub right: f64,
    /// The same sums accumulated in double precision.  These are the accurate
    /// values; they differ from the printed ones by the single-precision
    /// accumulation error, which is why both are carried.
    pub left_f64: f64,
    pub right_f64: f64,
    /// Percent difference of the printed sums, left over right minus one.
    pub ratio_percent: f64,
}

/// The composite field's coverage and peak.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum CompositeOutcome {
    NotRequested,
    Missing {
        variable: String,
        left_present: bool,
        right_present: bool,
    },
    Present {
        variable: String,
        columns: usize,
        levels: usize,
        counts: Vec<CompositeCount>,
        left_max: f64,
        right_max: f64,
    },
}

/// Cells at or above one threshold, per arm.
#[derive(Debug, Clone, Serialize)]
pub struct CompositeCount {
    pub threshold: f64,
    pub left: usize,
    pub right: usize,
}

/// One frame's full comparison.
#[derive(Debug, Clone, Serialize)]
pub struct FrameComparison {
    pub frame: String,
    pub fields: Vec<FieldOutcome>,
    pub accumulations: Vec<AccumulationOutcome>,
    pub composite: CompositeOutcome,
}

/// A whole comparison.
#[derive(Debug, Clone, Serialize)]
pub struct Comparison {
    pub left_label: String,
    pub right_label: String,
    pub left_directory: String,
    pub right_directory: String,
    pub left_frame_count: usize,
    pub right_frame_count: usize,
    /// Printed name of the composite rows, carried here so the table does not
    /// re-derive it per frame.
    pub composite_label: String,
    pub frames: Vec<FrameComparison>,
}

/// Frame names present under `prefix` in a directory, sorted.
fn frame_names(directory: &Path, prefix: &str) -> Result<Vec<String>> {
    let entries = std::fs::read_dir(directory).map_err(|source| Error::ListDirectory {
        path: directory.to_path_buf(),
        source,
    })?;
    let mut names = BTreeSet::new();
    for entry in entries {
        let entry = entry.map_err(|source| Error::ListDirectory {
            path: directory.to_path_buf(),
            source,
        })?;
        let name = entry.file_name().to_string_lossy().into_owned();
        if name.starts_with(prefix) && entry.path().is_file() {
            names.insert(name);
        }
    }
    Ok(names.into_iter().collect())
}

/// A frame opened once, from which every requested variable is read.
pub(crate) struct OpenFrame {
    file: netcrust::File,
    path: PathBuf,
}

impl OpenFrame {
    pub(crate) fn open(path: PathBuf) -> Result<Self> {
        let file = netcrust::File::open(&path).map_err(|source| Error::OpenFrame {
            path: path.clone(),
            source,
        })?;
        Ok(Self { file, path })
    }

    pub(crate) fn has(&self, variable: &str) -> bool {
        self.file.variable(variable).is_some()
    }

    pub(crate) fn path(&self) -> &Path {
        &self.path
    }

    /// Read one record of a variable, with its leading singleton axes
    /// stripped so callers see the field's own rank.
    pub(crate) fn read(&self, variable: &str) -> Result<(Vec<usize>, Vec<f64>)> {
        let array = self
            .file
            .read_array_f64_first_record_or_all(variable)
            .map_err(|source| Error::ReadVariable {
                variable: variable.to_string(),
                path: self.path.clone(),
                source,
            })?;
        let mut shape = array.shape().to_vec();
        while shape.len() > 2 && shape.first() == Some(&1) {
            shape.remove(0);
        }
        Ok((shape, array.into_values()))
    }

    fn read_plane(&self, variable: &str) -> Result<Vec<f64>> {
        let (shape, values) = self.read(variable)?;
        if shape.len() != 2 {
            return Err(Error::NotAPlane {
                variable: variable.to_string(),
                path: self.path.clone(),
                shape,
            });
        }
        Ok(values)
    }

    /// Read a volume and reduce it to its per-column composite, returning
    /// the level count beside it because the table reports what was reduced.
    ///
    /// A volume is by far the largest thing a frame holds, so it is read and
    /// reduced in one call and never stored on the comparison.  It is read
    /// whole rather than level by level: the level-wise variant was measured
    /// on the reference pair and came out about a fifth slower, because the
    /// per-slice overhead in the reader costs more than the extra
    /// parallelism buys.
    ///
    /// Peak memory is therefore one promoted volume -- levels x rows x
    /// columns x 8 bytes -- per frame-arm being read concurrently.  Sizing
    /// the worker pool with `--threads` is what bounds it on a machine where
    /// that product is large.
    fn read_composite(&self, variable: &str) -> Result<(usize, Vec<f64>)> {
        let (shape, values) = self.read(variable)?;
        if shape.len() != 3 {
            return Err(Error::NotAVolume {
                variable: variable.to_string(),
                path: self.path.clone(),
                shape,
            });
        }
        let plane = shape[1] * shape[2];
        Ok((shape[0], stats::column_max(&values, shape[0], plane)))
    }
}

/// One arm's planes for one frame, read once each.
///
/// A field that is both scored and summed -- an accumulation like total
/// precipitation is normally both -- would otherwise be decoded twice per
/// arm per frame, which is the single largest avoidable cost in a paired
/// comparison.  Reading the union up front and indexing it afterwards is
/// what makes "once each" true rather than aspirational.
struct FramePlanes {
    variables: Vec<String>,
    planes: Vec<Vec<f64>>,
}

impl FramePlanes {
    fn read(frame: &OpenFrame, wanted: &[String]) -> Result<Self> {
        let planes = wanted
            .par_iter()
            .map(|variable| frame.read_plane(variable))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            variables: wanted.to_vec(),
            planes,
        })
    }

    fn plane(&self, variable: &str) -> Option<&[f64]> {
        self.variables
            .iter()
            .position(|name| name == variable)
            .map(|index| self.planes[index].as_slice())
    }
}

/// Compare one frame present in both arms.
fn compare_frame(
    request: &CompareRequest,
    frame: &str,
    left: &OpenFrame,
    right: &OpenFrame,
) -> Result<FrameComparison> {
    // Every plane the frame needs, named once: the scored fields and the
    // summed accumulations, minus anything either arm does not carry.
    let mut wanted: Vec<String> = Vec::new();
    for variable in request
        .fields
        .iter()
        .map(|spec| &spec.variable)
        .chain(request.accumulations.iter())
    {
        if !wanted.contains(variable) && left.has(variable) && right.has(variable) {
            wanted.push(variable.clone());
        }
    }

    let (left_planes, right_planes) = rayon::join(
        || FramePlanes::read(left, &wanted),
        || FramePlanes::read(right, &wanted),
    );
    let (left_planes, right_planes) = (left_planes?, right_planes?);

    let mut fields = Vec::with_capacity(request.fields.len());
    for spec in &request.fields {
        let label = spec.label.clone().unwrap_or_else(|| spec.variable.clone());
        let (Some(left_values), Some(right_values)) = (
            left_planes.plane(&spec.variable),
            right_planes.plane(&spec.variable),
        ) else {
            fields.push(FieldOutcome::Missing {
                variable: spec.variable.clone(),
                label,
                left_present: left.has(&spec.variable),
                right_present: right.has(&spec.variable),
            });
            continue;
        };
        if left_values.len() != right_values.len() {
            return Err(Error::ShapeMismatch {
                variable: spec.variable.clone(),
                left: left_values.len(),
                right: right_values.len(),
            });
        }
        // Summarising permutes its input, so each arm summarises a scratch
        // copy and the cached plane stays in file order for the sums.
        let (left_summary, right_summary) = rayon::join(
            || Summary::of(left_values),
            || Summary::of(right_values),
        );
        fields.push(FieldOutcome::Present {
            variable: spec.variable.clone(),
            label,
            cells: left_values.len(),
            left: left_summary.into(),
            right: right_summary.into(),
            delta: left_summary.minus(&right_summary).into(),
            left_nonfinite: stats::nonfinite_count(left_values),
            right_nonfinite: stats::nonfinite_count(right_values),
        });
    }

    let mut accumulations = Vec::with_capacity(request.accumulations.len());
    for variable in &request.accumulations {
        let (Some(left_values), Some(right_values)) =
            (left_planes.plane(variable), right_planes.plane(variable))
        else {
            continue;
        };
        let left_sum = stats::accumulation_sum_f32(left_values) as f64;
        let right_sum = stats::accumulation_sum_f32(right_values) as f64;
        accumulations.push(AccumulationOutcome {
            variable: variable.clone(),
            left: left_sum,
            right: right_sum,
            left_f64: stats::pairwise_sum(left_values),
            right_f64: stats::pairwise_sum(right_values),
            ratio_percent: if right_sum > 0.0 {
                (left_sum / right_sum - 1.0) * 100.0
            } else {
                f64::NAN
            },
        });
    }

    let composite = match &request.composite {
        None => CompositeOutcome::NotRequested,
        Some(spec) => {
            let (in_left, in_right) = (left.has(&spec.variable), right.has(&spec.variable));
            if !in_left || !in_right {
                CompositeOutcome::Missing {
                    variable: spec.variable.clone(),
                    left_present: in_left,
                    right_present: in_right,
                }
            } else {
                let (left_read, right_read) = rayon::join(
                    || left.read_composite(&spec.variable),
                    || right.read_composite(&spec.variable),
                );
                let ((levels, left_plane), (_, right_plane)) = (left_read?, right_read?);
                if left_plane.len() != right_plane.len() {
                    return Err(Error::ShapeMismatch {
                        variable: spec.variable.clone(),
                        left: left_plane.len(),
                        right: right_plane.len(),
                    });
                }
                let counts = spec
                    .thresholds
                    .iter()
                    .map(|&threshold| CompositeCount {
                        threshold,
                        left: left_plane.iter().filter(|&&v| v >= threshold).count(),
                        right: right_plane.iter().filter(|&&v| v >= threshold).count(),
                    })
                    .collect();
                CompositeOutcome::Present {
                    variable: spec.variable.clone(),
                    columns: left_plane.len(),
                    levels,
                    counts,
                    left_max: stats::propagating_max(&left_plane),
                    right_max: stats::propagating_max(&right_plane),
                }
            }
        }
    };

    Ok(FrameComparison {
        frame: frame.to_string(),
        fields,
        accumulations,
        composite,
    })
}

/// Run a comparison.  Frames run in parallel; within a frame the two arms
/// are read concurrently.
pub fn compare(request: &CompareRequest) -> Result<Comparison> {
    let left_names = frame_names(&request.left, &request.frame_prefix)?;
    let right_names = frame_names(&request.right, &request.frame_prefix)?;
    let paired: Vec<String> = left_names
        .iter()
        .filter(|name| right_names.contains(name))
        .cloned()
        .collect();
    if paired.is_empty() {
        return Err(Error::NoCommonFrames {
            prefix: request.frame_prefix.clone(),
            left: request.left.clone(),
            right: request.right.clone(),
        });
    }

    let frames = paired
        .par_iter()
        .map(|frame| {
            let (left, right) = rayon::join(
                || OpenFrame::open(request.left.join(frame)),
                || OpenFrame::open(request.right.join(frame)),
            );
            compare_frame(request, frame, &left?, &right?)
        })
        .collect::<Result<Vec<_>>>()?;

    Ok(Comparison {
        left_label: request.left_label.clone(),
        right_label: request.right_label.clone(),
        left_directory: request.left.display().to_string(),
        right_directory: request.right.display().to_string(),
        left_frame_count: left_names.len(),
        right_frame_count: right_names.len(),
        composite_label: request
            .composite
            .as_ref()
            .map(|spec| spec.label.clone())
            .unwrap_or_default(),
        frames,
    })
}
