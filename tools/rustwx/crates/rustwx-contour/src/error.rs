use std::error::Error;
use std::fmt::{self, Display, Formatter};

#[derive(Debug, Clone, PartialEq)]
pub enum ContourError {
    InvalidGridShape {
        nx: usize,
        ny: usize,
    },
    AxisTooShort {
        axis: &'static str,
        len: usize,
    },
    AxisNotStrictlyMonotonic {
        axis: &'static str,
        index: usize,
        previous: f64,
        current: f64,
    },
    NonFiniteAxisValue {
        axis: &'static str,
        index: usize,
        value: f64,
    },
    ProjectedPointCount {
        expected: usize,
        actual: usize,
    },
    NonFiniteProjectedPoint {
        index: usize,
        x: f64,
        y: f64,
    },
    ValueCount {
        expected: usize,
        actual: usize,
    },
    EmptyContourLevels,
    NeedAtLeastTwoThresholds,
    NonFiniteLevel {
        index: usize,
        value: f64,
    },
    LevelsNotStrictlyAscending {
        index: usize,
        previous: f64,
        current: f64,
    },
}

impl Display for ContourError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidGridShape { nx, ny } => {
                write!(f, "grid shape must be at least 2x2, got {nx}x{ny}")
            }
            Self::AxisTooShort { axis, len } => {
                write!(f, "{axis} axis must contain at least 2 values, got {len}")
            }
            Self::AxisNotStrictlyMonotonic {
                axis,
                index,
                previous,
                current,
            } => write!(
                f,
                "{axis} axis must be strictly monotonic; values at {index} and {} were {previous} and {current}",
                index + 1
            ),
            Self::NonFiniteAxisValue { axis, index, value } => {
                write!(f, "{axis} axis value at {index} was not finite: {value}")
            }
            Self::ProjectedPointCount { expected, actual } => write!(
                f,
                "projected grid point count mismatch: expected {expected}, got {actual}"
            ),
            Self::NonFiniteProjectedPoint { index, x, y } => write!(
                f,
                "projected grid point at {index} was not finite: ({x}, {y})"
            ),
            Self::ValueCount { expected, actual } => {
                write!(
                    f,
                    "field value count mismatch: expected {expected}, got {actual}"
                )
            }
            Self::EmptyContourLevels => write!(f, "at least one contour level is required"),
            Self::NeedAtLeastTwoThresholds => {
                write!(f, "at least two thresholds are required to form bins")
            }
            Self::NonFiniteLevel { index, value } => {
                write!(f, "level at index {index} was not finite: {value}")
            }
            Self::LevelsNotStrictlyAscending {
                index,
                previous,
                current,
            } => write!(
                f,
                "levels must be strictly ascending; values at {index} and {} were {previous} and {current}",
                index + 1
            ),
        }
    }
}

impl Error for ContourError {}
