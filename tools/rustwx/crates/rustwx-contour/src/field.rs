use crate::error::ContourError;
use crate::grid::{CellIndex, Grid2D, GridShape};

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ScalarMetadata {
    pub name: Option<String>,
    pub units: Option<String>,
}

impl ScalarMetadata {
    pub fn named<S: Into<String>>(name: S) -> Self {
        Self {
            name: Some(name.into()),
            units: None,
        }
    }

    pub fn with_units<S: Into<String>>(mut self, units: S) -> Self {
        self.units = Some(units.into());
        self
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ScalarField2D<G> {
    grid: G,
    values: Vec<f64>,
    metadata: ScalarMetadata,
}

impl<G: Grid2D> ScalarField2D<G> {
    pub fn new(grid: G, values: Vec<f64>) -> Result<Self, ContourError> {
        Self::with_metadata(grid, values, ScalarMetadata::default())
    }

    pub fn with_metadata(
        grid: G,
        values: Vec<f64>,
        metadata: ScalarMetadata,
    ) -> Result<Self, ContourError> {
        let expected = grid.shape().node_count();
        if values.len() != expected {
            return Err(ContourError::ValueCount {
                expected,
                actual: values.len(),
            });
        }

        Ok(Self {
            grid,
            values,
            metadata,
        })
    }

    pub fn grid(&self) -> &G {
        &self.grid
    }

    pub fn shape(&self) -> GridShape {
        self.grid.shape()
    }

    pub fn values(&self) -> &[f64] {
        &self.values
    }

    pub fn metadata(&self) -> &ScalarMetadata {
        &self.metadata
    }

    pub fn value_at(&self, column: usize, row: usize) -> Option<f64> {
        if !self.shape().contains_node(column, row) {
            return None;
        }

        Some(self.values[self.shape().node_index(column, row)])
    }

    pub fn cell_values(&self, cell: CellIndex) -> Option<[f64; 4]> {
        let shape = self.shape();
        if !shape.contains_cell(cell.column, cell.row) {
            return None;
        }

        Some([
            self.value_at(cell.column, cell.row)?,
            self.value_at(cell.column + 1, cell.row)?,
            self.value_at(cell.column + 1, cell.row + 1)?,
            self.value_at(cell.column, cell.row + 1)?,
        ])
    }

    pub(crate) fn cell_center_value(&self, cell: CellIndex) -> Option<f64> {
        let values = self.cell_values(cell)?;
        if values.iter().any(|value| !value.is_finite()) {
            return None;
        }

        Some((values[0] + values[1] + values[2] + values[3]) * 0.25)
    }
}
