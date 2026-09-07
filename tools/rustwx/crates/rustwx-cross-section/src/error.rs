use std::error::Error;
use std::fmt::{self, Display, Formatter};

/// Errors returned by cross-section builders, containers, and the lightweight renderer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CrossSectionError {
    TooFewWaypoints,
    DegeneratePath,
    InvalidSampleCount,
    InvalidSpacing,
    EmptyLevels,
    NonMonotonicLevels,
    NonMonotonicDistances,
    InvalidCoordinate,
    InvalidLevelValue,
    ShapeMismatch {
        context: &'static str,
        expected: usize,
        actual: usize,
    },
    InvalidTerrainProfile,
    InvalidRenderDimensions,
    InvalidPlotMargins,
    EmptyColorRamp,
    NoFiniteData,
}

impl Display for CrossSectionError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::TooFewWaypoints => {
                f.write_str("cross-section path requires at least two waypoints")
            }
            Self::DegeneratePath => f.write_str("cross-section path has zero total distance"),
            Self::InvalidSampleCount => {
                f.write_str("cross-section sampling requires at least two points")
            }
            Self::InvalidSpacing => f.write_str("cross-section spacing must be finite and > 0"),
            Self::EmptyLevels => f.write_str("vertical axis requires at least two levels"),
            Self::NonMonotonicLevels => f.write_str("vertical levels must be strictly monotonic"),
            Self::NonMonotonicDistances => {
                f.write_str("distance coordinates must be finite and strictly increasing")
            }
            Self::InvalidCoordinate => {
                f.write_str("geographic coordinates must be finite, with latitude in [-90, 90]")
            }
            Self::InvalidLevelValue => {
                f.write_str("vertical levels must be finite, and positive for log-pressure axes")
            }
            Self::ShapeMismatch {
                context,
                expected,
                actual,
            } => write!(
                f,
                "shape mismatch for {context}: expected {expected} values, found {actual}"
            ),
            Self::InvalidTerrainProfile => {
                f.write_str("terrain profile must align with its distance coordinate")
            }
            Self::InvalidRenderDimensions => {
                f.write_str("render width and height must both be at least 2 pixels")
            }
            Self::InvalidPlotMargins => f.write_str("render margins leave no drawable plot area"),
            Self::EmptyColorRamp => f.write_str("color ramp must contain at least two colors"),
            Self::NoFiniteData => f.write_str("section does not contain any finite data values"),
        }
    }
}

impl Error for CrossSectionError {}
