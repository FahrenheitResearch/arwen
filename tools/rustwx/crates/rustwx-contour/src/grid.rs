use crate::error::ContourError;
use crate::geometry::Point2;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct GridShape {
    pub nx: usize,
    pub ny: usize,
}

impl GridShape {
    pub fn new(nx: usize, ny: usize) -> Result<Self, ContourError> {
        if nx < 2 || ny < 2 {
            return Err(ContourError::InvalidGridShape { nx, ny });
        }
        Ok(Self { nx, ny })
    }

    pub const fn node_count(self) -> usize {
        self.nx * self.ny
    }

    pub const fn cell_columns(self) -> usize {
        self.nx - 1
    }

    pub const fn cell_rows(self) -> usize {
        self.ny - 1
    }

    pub const fn contains_node(self, column: usize, row: usize) -> bool {
        column < self.nx && row < self.ny
    }

    pub const fn contains_cell(self, column: usize, row: usize) -> bool {
        column + 1 < self.nx && row + 1 < self.ny
    }

    pub const fn node_index(self, column: usize, row: usize) -> usize {
        row * self.nx + column
    }

    pub fn cells(self) -> GridCells {
        GridCells {
            shape: self,
            next_column: 0,
            next_row: 0,
        }
    }
}

pub struct GridCells {
    shape: GridShape,
    next_column: usize,
    next_row: usize,
}

impl Iterator for GridCells {
    type Item = CellIndex;

    fn next(&mut self) -> Option<Self::Item> {
        if self.next_row >= self.shape.cell_rows() {
            return None;
        }

        let cell = CellIndex {
            column: self.next_column,
            row: self.next_row,
        };

        self.next_column += 1;
        if self.next_column >= self.shape.cell_columns() {
            self.next_column = 0;
            self.next_row += 1;
        }

        Some(cell)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CellIndex {
    pub column: usize,
    pub row: usize,
}

pub trait Grid2D {
    fn shape(&self) -> GridShape;
    fn point_at(&self, column: usize, row: usize) -> Option<Point2>;

    fn cell_corners(&self, cell: CellIndex) -> Option<[Point2; 4]> {
        let shape = self.shape();
        if !shape.contains_cell(cell.column, cell.row) {
            return None;
        }

        Some([
            self.point_at(cell.column, cell.row)?,
            self.point_at(cell.column + 1, cell.row)?,
            self.point_at(cell.column + 1, cell.row + 1)?,
            self.point_at(cell.column, cell.row + 1)?,
        ])
    }

    fn cell_center(&self, cell: CellIndex) -> Option<Point2> {
        let corners = self.cell_corners(cell)?;
        Some(Point2::new(
            (corners[0].x + corners[1].x + corners[2].x + corners[3].x) * 0.25,
            (corners[0].y + corners[1].y + corners[2].y + corners[3].y) * 0.25,
        ))
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct RectilinearGrid {
    shape: GridShape,
    x: Vec<f64>,
    y: Vec<f64>,
}

impl RectilinearGrid {
    pub fn new(x: Vec<f64>, y: Vec<f64>) -> Result<Self, ContourError> {
        if x.len() < 2 {
            return Err(ContourError::AxisTooShort {
                axis: "x",
                len: x.len(),
            });
        }
        if y.len() < 2 {
            return Err(ContourError::AxisTooShort {
                axis: "y",
                len: y.len(),
            });
        }

        validate_axis("x", &x)?;
        validate_axis("y", &y)?;

        Ok(Self {
            shape: GridShape::new(x.len(), y.len())?,
            x,
            y,
        })
    }

    pub fn x(&self) -> &[f64] {
        &self.x
    }

    pub fn y(&self) -> &[f64] {
        &self.y
    }
}

impl Grid2D for RectilinearGrid {
    fn shape(&self) -> GridShape {
        self.shape
    }

    fn point_at(&self, column: usize, row: usize) -> Option<Point2> {
        if !self.shape.contains_node(column, row) {
            return None;
        }

        Some(Point2::new(self.x[column], self.y[row]))
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ProjectedGrid {
    shape: GridShape,
    points: Vec<Point2>,
}

impl ProjectedGrid {
    pub fn new(shape: GridShape, points: Vec<Point2>) -> Result<Self, ContourError> {
        if points.len() != shape.node_count() {
            return Err(ContourError::ProjectedPointCount {
                expected: shape.node_count(),
                actual: points.len(),
            });
        }

        for (index, point) in points.iter().enumerate() {
            if !point.x.is_finite() || !point.y.is_finite() {
                return Err(ContourError::NonFiniteProjectedPoint {
                    index,
                    x: point.x,
                    y: point.y,
                });
            }
        }

        Ok(Self { shape, points })
    }

    pub fn points(&self) -> &[Point2] {
        &self.points
    }
}

impl Grid2D for ProjectedGrid {
    fn shape(&self) -> GridShape {
        self.shape
    }

    fn point_at(&self, column: usize, row: usize) -> Option<Point2> {
        if !self.shape.contains_node(column, row) {
            return None;
        }

        Some(self.points[self.shape.node_index(column, row)])
    }
}

fn validate_axis(axis: &'static str, values: &[f64]) -> Result<(), ContourError> {
    for (index, value) in values.iter().copied().enumerate() {
        if !value.is_finite() {
            return Err(ContourError::NonFiniteAxisValue { axis, index, value });
        }
    }

    let first_diff = values[1] - values[0];
    if first_diff == 0.0 || !first_diff.is_finite() {
        return Err(ContourError::AxisNotStrictlyMonotonic {
            axis,
            index: 0,
            previous: values[0],
            current: values[1],
        });
    }
    let increasing = first_diff.is_sign_positive();

    for index in 0..values.len() - 1 {
        let previous = values[index];
        let current = values[index + 1];
        let diff = current - previous;
        let monotonic = if increasing { diff > 0.0 } else { diff < 0.0 };
        if !monotonic {
            return Err(ContourError::AxisNotStrictlyMonotonic {
                axis,
                index,
                previous,
                current,
            });
        }
    }

    Ok(())
}
