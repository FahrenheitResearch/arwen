//! Sampler coordinates with fixed arithmetic in both precisions.
mod lambert;
mod mercator;
mod polar;

use super::{wrap180, GridSpec, ProjectionKind, DEG_PER_RAD, RAD_PER_DEG};

fn np_pow(x: f64, y: f64) -> f64 {
    if y == 2.0 {
        x * x
    } else {
        libm::pow(x, y)
    }
}
fn np_mod(x: f64, y: f64) -> f64 {
    let r = libm::fmod(x, y);
    if r == 0.0 {
        0.0_f64.copysign(y)
    } else if (r < 0.0) != (y < 0.0) {
        r + y
    } else {
        r
    }
}

#[derive(Debug, Clone)]
pub(crate) enum State {
    Lambert(lambert::LambertState),
    Mercator(mercator::MercatorState),
    Polar(polar::PolarState),
    /// Explicit rows: the transforms are table lookups and plain IEEE
    /// arithmetic (no transcendental call), so the float64 rows state is
    /// already portable and the sampler shares it.
    Rows(super::rows::RowsState),
}

#[derive(Debug, Clone)]
pub(crate) struct SamplingProjection {
    pub spec: GridSpec,
    pub state: State,
    hemi: f64,
}

impl SamplingProjection {
    pub fn new(spec: GridSpec) -> Self {
        let hemi = if spec.truelat1 < 0.0 { -1.0 } else { 1.0 };
        let state = match spec.kind {
            ProjectionKind::Lambert => State::Lambert(lambert::setup(&spec, hemi)),
            ProjectionKind::Mercator => State::Mercator(mercator::setup(&spec)),
            ProjectionKind::Polar => State::Polar(polar::setup(&spec, hemi)),
            ProjectionKind::Rows => State::Rows(super::rows::setup(&spec).expect(
                "a rows spec is validated by ProjectedGrid::new before its \
                 sampling projection is built")),
        };
        Self { spec, state, hemi }
    }
    pub fn ij_to_latlon(&self, x: f64, y: f64) -> (f64, f64) {
        match &self.state {
            State::Lambert(s) => lambert::ij_to_latlon(s, &self.spec, self.hemi, x, y),
            State::Mercator(s) => mercator::ij_to_latlon(s, &self.spec, x, y),
            State::Polar(s) => polar::ij_to_latlon(s, &self.spec, self.hemi, x, y),
            State::Rows(s) => super::rows::ij_to_latlon(s, x, y),
        }
    }
    pub fn latlon_to_ij(&self, lat: f64, lon: f64) -> (f64, f64) {
        match &self.state {
            State::Lambert(s) => lambert::latlon_to_ij(s, &self.spec, self.hemi, lat, lon),
            State::Mercator(s) => mercator::latlon_to_ij(s, &self.spec, lat, lon),
            State::Polar(s) => polar::latlon_to_ij(s, &self.spec, self.hemi, lat, lon),
            State::Rows(s) => super::rows::latlon_to_ij(s, lat, lon),
        }
    }
}
