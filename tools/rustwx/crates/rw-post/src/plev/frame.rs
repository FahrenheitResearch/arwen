//! Group B's view of one frame: the shared column state that Group A builds
//! (specification 3.4, [`crate::state`]) plus the 2D carriers Group B reads.
//!
//! Group B no longer prepares its own pressure, temperature or winds: the
//! hydrostatic interface and mass pressure, interface heights, temperature,
//! virtual temperature, specific humidity and destaggered winds all come
//! from [`ColumnState`], on the CPU and (as device slots) on the GPU.

use crate::state::{ColumnState, InterfaceState, MassState};

#[derive(Debug, Clone, Copy)]
pub struct Frame<'a> {
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    /// Group A's shared column state of this frame.
    pub state: &'a ColumnState,
    /// Surface pressure (Pa) and terrain height (m).
    pub psfc: &'a [f32],
    pub hgt: &'a [f32],
    /// Native grid spacing (m).
    pub dx: f64,
    /// Manifest text: where the state's pressure coordinate came from.
    pub pressure_source: &'a str,
    /// Manifest text: where the vertical-coordinate coefficients came from.
    pub vertical_source: &'a str,
}

impl<'a> Frame<'a> {
    pub fn new(
        state: &'a ColumnState,
        psfc: &'a [f32],
        hgt: &'a [f32],
        dx: f64,
        pressure_source: &'a str,
        vertical_source: &'a str,
    ) -> Self {
        let s = state.shape;
        Self { nx: s.nx, ny: s.ny, nz: s.nz, state, psfc, hgt, dx, pressure_source, vertical_source }
    }

    pub fn ncell(&self) -> usize {
        self.nx * self.ny
    }

    /// A mass-level slot of the shared state, `[k][cell]`.
    pub fn mass(&self, slot: MassState) -> &'a [f32] {
        self.state.mass_slot(slot)
    }

    /// An interface slot of the shared state, `[k][cell]`, ground first.
    pub fn iface(&self, slot: InterfaceState) -> &'a [f32] {
        self.state.interface_slot(slot)
    }

    /// Check the 2D carriers against the state's shape.
    pub fn validate(&self) -> Result<(), String> {
        let (nx, ny, nz) = (self.nx, self.ny, self.nz);
        if nx < 1 || ny < 1 || nz < 2 {
            return Err(format!("grid {nx} x {ny} x {nz} is too small"));
        }
        let n2 = nx * ny;
        for (name, got) in [("PSFC", self.psfc.len()), ("HGT", self.hgt.len())] {
            if got != n2 {
                return Err(format!("{name} has {got} values, expected {n2}"));
            }
        }
        Ok(())
    }
}
