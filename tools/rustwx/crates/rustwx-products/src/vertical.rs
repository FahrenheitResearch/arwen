//! Source rows for pressure/height maps. A model supplies history carriers;
//! the row selects the shared operator, with no model-specific adapter.
use rustwx_core::{CanonicalField as F, FieldSelector};
#[derive(Clone, Copy)]
pub enum Operator {
    Pressure(usize, u32),
    WindSpeed(u32),
}
#[derive(Clone, Copy)]
pub struct Row {
    pub selector: FieldSelector,
    pub operator: Operator,
}
pub const ROWS: &[Row] = &[
    Row {
        selector: FieldSelector::isobaric(F::SpecificHumidity, 850),
        operator: Operator::Pressure(0, 850),
    },
    Row {
        selector: FieldSelector::isobaric(F::Omega, 500),
        operator: Operator::Pressure(1, 500),
    },
    Row {
        selector: FieldSelector::height_agl(F::WindSpeed, 100),
        operator: Operator::WindSpeed(100),
    },
];
