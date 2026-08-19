//! Shared array and field types.  ALL THREE LANES code against these;
//! changing a shape here needs coordination, adding a method does not.
//!
//! Layout convention (fixed by the Python reference): 2-D arrays are
//! row-major `(south_north, west_east)` = `[j][i]`, matching geo_em's
//! `[j, i]` and `numpy`'s C order, so a `Grid2` buffer is byte-for-byte
//! what `np.ndarray.tobytes()` of the float64 field yields.  3-D stacks
//! carry the leading category/month axis first: `(planes, ny, nx)`.

use std::collections::BTreeMap;

use crate::error::{Result, StaticError};

/// One float64 plane, row-major `(ny, nx)`.
#[derive(Debug, Clone, PartialEq)]
pub struct Grid2 {
    pub ny: usize,
    pub nx: usize,
    pub data: Vec<f64>,
}

impl Grid2 {
    pub fn filled(ny: usize, nx: usize, value: f64) -> Self {
        Grid2 { ny, nx, data: vec![value; ny * nx] }
    }

    #[inline]
    pub fn at(&self, j: usize, i: usize) -> f64 {
        self.data[j * self.nx + i]
    }

    #[inline]
    pub fn set(&mut self, j: usize, i: usize, value: f64) {
        self.data[j * self.nx + i] = value;
    }
}

/// One float64 stack, row-major `(planes, ny, nx)`; `planes` is the
/// category axis (LANDUSEF/SOILCTOP/SOILCBOT) or the month axis
/// (GREENFRAC/LAI12M/ALBEDO12M).
#[derive(Debug, Clone, PartialEq)]
pub struct Stack3 {
    pub planes: usize,
    pub ny: usize,
    pub nx: usize,
    pub data: Vec<f64>,
}

impl Stack3 {
    pub fn filled(planes: usize, ny: usize, nx: usize, value: f64) -> Self {
        Stack3 { planes, ny, nx, data: vec![value; planes * ny * nx] }
    }

    /// Borrow one plane as a slice (`[j*nx+i]` indexed).
    pub fn plane(&self, z: usize) -> &[f64] {
        let n = self.ny * self.nx;
        &self.data[z * n..(z + 1) * n]
    }
}

/// One named static field: a plane or a stack.
#[derive(Debug, Clone, PartialEq)]
pub enum Field {
    Plane(Grid2),
    Stack(Stack3),
}

impl Field {
    /// `(planes, ny, nx)` with `planes == 1` for a plane.  The seam
    /// reports dims through this so the Python side allocates exactly.
    pub fn dims(&self) -> (usize, usize, usize) {
        match self {
            Field::Plane(g) => (1, g.ny, g.nx),
            Field::Stack(s) => (s.planes, s.ny, s.nx),
        }
    }

    pub fn data(&self) -> &[f64] {
        match self {
            Field::Plane(g) => &g.data,
            Field::Stack(s) => &s.data,
        }
    }
}

/// The build result: geo_em-named float64 fields, deterministically
/// ordered (BTreeMap) so seam enumeration and receipts are stable.
#[derive(Debug, Clone, Default)]
pub struct FieldSet {
    pub fields: BTreeMap<String, Field>,
    /// JSON receipts keyed by GEOG field name
    /// (`gpuwm-geog-source-coverage-v1` documents, rendered by lane 2).
    pub coverage_reports: BTreeMap<String, String>,
}

impl FieldSet {
    pub fn get(&self, name: &str) -> Result<&Field> {
        self.fields.get(name).ok_or_else(|| {
            StaticError::Invalid(format!("field set has no field {name:?}"))
        })
    }
}

/// Staggering of a coordinate/field request, WPS registration:
/// mass (i, j), U (i - 0.5, j), V (i, j - 0.5), corner (i - 0.5, j - 0.5).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum Stagger {
    Mass = 0,
    U = 1,
    V = 2,
    Corner = 3,
}

impl Stagger {
    pub fn from_code(code: u32) -> Result<Self> {
        Ok(match code {
            0 => Stagger::Mass,
            1 => Stagger::U,
            2 => Stagger::V,
            3 => Stagger::Corner,
            other => {
                return Err(StaticError::Invalid(format!(
                    "unknown stagger code {other}"
                )))
            }
        })
    }
}
