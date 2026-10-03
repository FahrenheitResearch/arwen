//! The WMO FM301 / CfRadial 2 data model (`docs/design/fm301-model.md`).
//!
//! - [`Volume`] is the root group of an FM301 file: global attributes, the
//!   station location, radar parameters and calibration, and the sweeps.
//! - [`Sweep`] is one `sweep_<n>` group: ray coordinates (`time`, `azimuth`,
//!   `elevation`), one `range` coordinate, per-ray instrument variables and the
//!   dataset variables ([`Field`]).
//! - [`Field`] holds one variable's values row-major `[nrays × ngates]` in the
//!   source's own encoding (`u8`, `u16`, `i8`, `i16`, `i32`, `f32`, `f64`) with its CF
//!   packing. Physical values are computed on demand; decoders never expand raw
//!   storage to floats.
//!
//! The model keeps rays in the source's storage order and each field in its
//! native gate geometry ([`GateMapping`] onto the sweep range). The FM301 view
//! ([`crate::fm301`]) applies ray order, padding and gate repetition when a
//! caller reads a variable.

mod cycles;
mod field;
mod merge;
mod names;
#[cfg(feature = "serde")]
mod serde_checked;
mod sweep;
mod values;
mod volume;

pub use cycles::{
    CycleBreak, CycleTracker, MAX_SCAN_PAUSE_S, ScanCycle, collection_order, scan_cycles,
    split_scan_cycles,
};
pub use field::{
    Coding, Field, FieldAttrs, FieldData, FieldError, FieldParts, FloatCoding, FloatWidth, Gate,
    GateMapping, IntCoding, LevelTable, LinearTransform, PackedInt, RowRef,
};
pub use merge::{
    ANGLE_MATCH_TOLERANCE_DEG, MERGE_TIME_TOLERANCE_S, MergeError, MergeReport, merge_volumes,
};
pub use names::{FieldName, NameInfo, Polarization, PyartNames, Quantity, XradarAttrs};
pub use sweep::{
    FollowMode, GeometryError, Monitoring, PlatformTrack, PolarizationMode, PrtMode, PrtSequence,
    RangeCoord, RayVariables, Rays, Sweep, SweepError, SweepMode,
};
pub use values::{ArrayBuf, AttrValue, ExtraVariable, RayAlignment, Scalar, VariableAttrs};
pub use volume::{
    DecodeStats, GeoreferencingCorrection, GlobalAttrs, InstrumentType, Location, PlatformType,
    PrimaryAxis, Provenance, RadarCalibration, RadarParameters, ScanDefinition, ScanLeg,
    ScanStrategy, SimulationProvenance, SourceFormat, TimeCoverage, Volume, WmoAttrs,
    WmoDataPolicy, floor_to_second,
};
