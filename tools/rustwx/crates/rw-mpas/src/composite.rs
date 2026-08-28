//! One frame built from several model runs at once: the coarse field
//! everywhere, each fine run's data inside the ground that fine run
//! actually resolves.
//!
//! gpuwm addition (VENDOR.md).  The cascade produces two or more forecasts
//! of the same hour at different resolutions -- a coarse global run, and
//! one fine run per placed grid -- and until now they could only be looked
//! at side by side.  The picture the programme is missing is the one that
//! puts them together: coarse everywhere, fine where fine exists, one
//! frame.
//!
//! # Why this is a DATA operation and not a picture operation
//!
//! Compositing two finished PNGs means knowing exactly what each pixel
//! maps to, and then reconciling two different projections, two basemaps
//! and two colour scales.  Compositing the DATA means one resample onto
//! one grid, then one render: one projection, one basemap, one scale, and
//! no registration anywhere.  The seam that remains is a real seam in the
//! data, which is the only seam that should be visible.
//!
//! # Which source owns a point
//!
//! Every source can answer at every point -- the resample is
//! nearest-cell over the whole sphere and never refuses -- so "which
//! source covers this point" has to be decided, not discovered.  Two
//! conditions, both necessary:
//!
//! 1. **The point is inside that source's refined region.**  A placed mesh
//!    is a whole globe that spends its cells unevenly; its background
//!    cells are not what it was built to resolve, and its 6-hour
//!    background is not better than the coarse run's.  The refined region
//!    is [`Window::refined_region`]'s definition and nothing new: cells at
//!    or under twice that mesh's own finest spacing.  Reusing it matters
//!    -- it is the same set of cells `--window mesh` centres a fine core's
//!    own render on, so the composite and the fine render agree about
//!    where the fine data is.
//! 2. **The point is actually near a cell of it.**  Distance from the grid
//!    point to the cell centre it took its value from, against that cell's
//!    own spacing.  This is what makes the rule indifferent to whether a
//!    placed grid is a globally-refined mesh or a limited-area cull: on a
//!    cull there are no cells outside the region at all, so the nearest
//!    one is far away and the test fails on its own, with no flag to set
//!    and no second code path.
//!
//! Where more than one source qualifies, the one with the finest local
//! spacing wins.  That is what makes overlapping placed grids at different
//! resolutions work without a priority list to maintain.
//!
//! # The seam
//!
//! At the edge of a fine region the value changes source, and two
//! different model runs do not agree.  The switch is HARD -- no blending.
//! A blend would put values on the map that neither run produced, in
//! exactly the band a reader is most likely to be studying, and it would
//! hide the one thing worth knowing: where the fine data stops.
//! [`CompositePlan::source_index`] is written into the frame as a field of
//! its own so the boundary can be drawn, and
//! [`CompositePlan::boundary_slots`] finds it.

use crate::error::{MpasError, MpasResult};
use crate::weights::{MeshCoordinates, NearestCellWeights};
use crate::window::Window;

/// The schema of the composite record written into every composed frame.
pub const COMPOSITE_SCHEMA: &str = "mpas-port.render-composite/v1";

/// How much wider than the fine regions a composite window is drawn.
///
/// A window sized to the fine region alone shows no coarse field at all --
/// it is the fine render again, and the composite has nothing to composite
/// against.  At 3x, the fine core spans about a third of the frame and the
/// coarse field it sits in is unmistakably there.
pub const COMPOSITE_CONTEXT_FACTOR: f64 = 3.0;

/// One model run's contribution.
///
/// `spacing_metres` is per CELL of that source's own mesh, not per grid
/// point: the rule asks how finely the mesh resolves the ground under a
/// point, which is a property of the cell the point landed in.
#[derive(Debug, Clone)]
pub struct CompositeSource {
    pub label: String,
    pub weights: NearestCellWeights,
    pub spacing_metres: Vec<f64>,
    /// Per cell: is this cell part of the mesh's refined region?  All
    /// `false` for the base, which is refined nowhere and owns every point
    /// no other source claims.
    pub refined: Vec<bool>,
    /// The mesh's own cell centres, kept so a composite window can be
    /// sized to the refined footprint without re-reading the mesh file.
    pub latitude_degrees: Vec<f64>,
    pub longitude_degrees: Vec<f64>,
}

impl CompositeSource {
    /// A source that contributes only inside its own refined region.
    pub fn overlay(
        label: impl Into<String>,
        mesh: &MeshCoordinates,
        weights: NearestCellWeights,
        spacing_metres: Vec<f64>,
    ) -> MpasResult<Self> {
        let label = label.into();
        if spacing_metres.len() != weights.n_cells {
            return Err(MpasError::Refusal(format!(
                "composite source '{label}' carries {} cell spacing(s) against {} mesh \
                 cell(s); pairing them would read one cell's resolution off another cell",
                spacing_metres.len(),
                weights.n_cells
            )));
        }
        let selected = Window::refined_region(&spacing_metres).map_err(|err| {
            MpasError::Refusal(format!(
                "composite source '{label}' has no refined region, so it would contribute \
                 nothing anywhere and its presence in the source list is a mistake worth \
                 naming rather than a no-op worth ignoring: {err}"
            ))
        })?;
        let mut refined = vec![false; spacing_metres.len()];
        for index in selected {
            refined[index] = true;
        }
        Ok(Self {
            label,
            weights,
            spacing_metres,
            refined,
            latitude_degrees: mesh.latitude_degrees.clone(),
            longitude_degrees: mesh.longitude_degrees.clone(),
        })
    }

    /// The base: it answers wherever nothing finer does.
    pub fn base(
        label: impl Into<String>,
        mesh: &MeshCoordinates,
        weights: NearestCellWeights,
        spacing_metres: Vec<f64>,
    ) -> MpasResult<Self> {
        let label = label.into();
        if spacing_metres.len() != weights.n_cells {
            return Err(MpasError::Refusal(format!(
                "composite base '{label}' carries {} cell spacing(s) against {} mesh cell(s)",
                spacing_metres.len(),
                weights.n_cells
            )));
        }
        let cells = spacing_metres.len();
        Ok(Self {
            label,
            weights,
            spacing_metres,
            refined: vec![false; cells],
            latitude_degrees: mesh.latitude_degrees.clone(),
            longitude_degrees: mesh.longitude_degrees.clone(),
        })
    }

    fn cell_at(&self, slot: usize) -> usize {
        self.weights.cell_index[slot] as usize
    }

    fn spacing_at(&self, slot: usize) -> f64 {
        self.spacing_metres[self.cell_at(slot)]
    }
}

/// Which source owns each grid point, and what resolution that buys.
#[derive(Debug, Clone)]
pub struct CompositePlan {
    /// Source index per grid point, row-major `(ny, nx)`.  `0` is the base.
    pub source_index: Vec<u16>,
    /// The local cell spacing, in metres, of whichever source won.  This
    /// is the resolution the composite actually shows at that point, which
    /// is not the window's `dx` and not any source's nominal spacing.
    pub shown_spacing_metres: Vec<f32>,
    /// Grid points won, per source, in source order.
    pub counts: Vec<usize>,
    pub labels: Vec<String>,
    pub ny: usize,
    pub nx: usize,
}

impl CompositePlan {
    /// Decide the whole grid.
    ///
    /// Every source must have been built against the SAME window, or the
    /// plan would be indexing different grids with one slot number.
    pub fn build(base: &CompositeSource, overlays: &[&CompositeSource]) -> MpasResult<Self> {
        let (ny, nx) = base.weights.shape();
        let points = ny * nx;
        for overlay in overlays {
            let (oy, ox) = overlay.weights.shape();
            if (oy, ox) != (ny, nx) {
                return Err(MpasError::Refusal(format!(
                    "composite source '{}' was resampled onto a {oy}x{ox} window against the \
                     base's {ny}x{nx}; one slot number would mean two different places",
                    overlay.label
                )));
            }
            if overlay.weights.window != base.weights.window {
                return Err(MpasError::Refusal(format!(
                    "composite source '{}' was resampled onto a different window than the \
                     base, though both are {ny}x{nx}. Same shape is not same ground",
                    overlay.label
                )));
            }
        }

        let mut source_index = vec![0u16; points];
        let mut shown_spacing_metres = vec![0.0f32; points];
        let mut counts = vec![0usize; overlays.len() + 1];
        let mut labels = Vec::with_capacity(overlays.len() + 1);
        labels.push(base.label.clone());
        for overlay in overlays {
            labels.push(overlay.label.clone());
        }

        for slot in 0..points {
            let mut winner = 0u16;
            let mut winner_spacing = base.spacing_at(slot);
            for (offset, overlay) in overlays.iter().enumerate() {
                let cell = overlay.cell_at(slot);
                if !overlay.refined[cell] {
                    continue;
                }
                let spacing = overlay.spacing_metres[cell];
                // The point has to be inside a cell of this source, not
                // merely nearest to one that happens to be its closest on
                // the whole sphere.  `distance_km` is the great-circle
                // distance to the cell centre it took its value from.
                if f64::from(overlay.weights.distance_km[slot]) * 1000.0 > spacing {
                    continue;
                }
                if spacing < winner_spacing {
                    winner = offset as u16 + 1;
                    winner_spacing = spacing;
                }
            }
            source_index[slot] = winner;
            shown_spacing_metres[slot] = winner_spacing as f32;
            counts[winner as usize] += 1;
        }

        Ok(Self {
            source_index,
            shown_spacing_metres,
            counts,
            labels,
            ny,
            nx,
        })
    }

    pub fn points(&self) -> usize {
        self.ny * self.nx
    }

    /// The grid points that sit against a change of source: the seam.
    ///
    /// A point is on the boundary when any of its four neighbours came
    /// from a different source.  Published rather than smoothed away --
    /// the composite draws this line so a reader can see where the data
    /// changed hands instead of having to guess from the texture.
    pub fn boundary_slots(&self) -> Vec<usize> {
        let mut out = Vec::new();
        for j in 0..self.ny {
            for i in 0..self.nx {
                let slot = j * self.nx + i;
                let mine = self.source_index[slot];
                let mut edge = false;
                if i > 0 && self.source_index[slot - 1] != mine {
                    edge = true;
                }
                if i + 1 < self.nx && self.source_index[slot + 1] != mine {
                    edge = true;
                }
                if j > 0 && self.source_index[slot - self.nx] != mine {
                    edge = true;
                }
                if j + 1 < self.ny && self.source_index[slot + self.nx] != mine {
                    edge = true;
                }
                if edge {
                    out.push(slot);
                }
            }
        }
        out
    }

    /// A sorted-key JSON object recording the decision, for the frame's
    /// own provenance attribute and for the report.
    pub fn spec_json(&self) -> String {
        let mut sources = Vec::with_capacity(self.labels.len());
        for (index, label) in self.labels.iter().enumerate() {
            sources.push(format!(
                "{{\"grid_points\": {}, \"index\": {index}, \"label\": \"{}\", \"role\": \"{}\"}}",
                self.counts[index],
                json_escape(label),
                if index == 0 { "base" } else { "overlay" }
            ));
        }
        format!(
            "{{\"boundary_points\": {}, \"grid_points\": {}, \"schema\": \"{COMPOSITE_SCHEMA}\", \
             \"selection\": \"{}\", \"sources\": [{}]}}",
            self.boundary_slots().len(),
            self.points(),
            "finest refined source that the point falls inside; base elsewhere",
            sources.join(", ")
        )
    }
}

/// How far apart the two runs are, measured across the seam.
///
/// The composite switches source hard, so at the boundary the value comes
/// from a different model run than it did one grid point earlier.  This
/// measures that step for one field: at every boundary point the fine
/// source won, the fine value against the base value at the same place and
/// hour.  Publishing it is the difference between saying "there is a seam"
/// and saying how big it is.
#[derive(Debug, Clone)]
pub struct SeamStat {
    pub field: String,
    pub points: usize,
    pub mean_abs: f64,
    pub p95_abs: f64,
    pub max_abs: f64,
    /// The field's own spread over the whole frame, for scale.  A step of
    /// 2 m/s means one thing in a field that spans 3 m/s and another in one
    /// that spans 60.
    pub frame_range: f64,
}

impl SeamStat {
    pub fn measure(
        field: &str,
        boundary: &[usize],
        source_index: &[u16],
        composed: &[f32],
        base: &[f32],
        points: usize,
    ) -> Option<Self> {
        let mut diffs: Vec<f64> = Vec::new();
        for &slot in boundary {
            if source_index[slot] == 0 {
                continue;
            }
            let a = composed[slot];
            let b = base[slot];
            if a.is_finite() && b.is_finite() {
                diffs.push(f64::from(a - b).abs());
            }
        }
        if diffs.is_empty() {
            return None;
        }
        diffs.sort_by(f64::total_cmp);
        let mean_abs = diffs.iter().sum::<f64>() / diffs.len() as f64;
        let p95_abs = diffs[((diffs.len() as f64 * 0.95) as usize).min(diffs.len() - 1)];
        let max_abs = *diffs.last().expect("non-empty");
        let mut lo = f64::INFINITY;
        let mut hi = f64::NEG_INFINITY;
        for value in composed.iter().take(points) {
            if value.is_finite() {
                lo = lo.min(f64::from(*value));
                hi = hi.max(f64::from(*value));
            }
        }
        Some(Self {
            field: field.to_string(),
            points: diffs.len(),
            mean_abs,
            p95_abs,
            max_abs,
            frame_range: if hi >= lo { hi - lo } else { 0.0 },
        })
    }

    pub fn json(&self) -> String {
        format!(
            "{{\"field\": \"{}\", \"frame_range\": {:e}, \"max_abs\": {:e}, \"mean_abs\": {:e}, \"p95_abs\": {:e}, \"points\": {}}}",
            json_escape(&self.field),
            self.frame_range,
            self.max_abs,
            self.mean_abs,
            self.p95_abs,
            self.points
        )
    }
}

fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            c if (c as u32) < 0x20 => out.push(' '),
            c => out.push(c),
        }
    }
    out
}

/// The refined cells of every overlay, concatenated, for sizing a window
/// that covers all of them.
pub fn refined_footprint(overlays: &[&CompositeSource]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut lat = Vec::new();
    let mut lon = Vec::new();
    let mut spacing = Vec::new();
    for overlay in overlays {
        for (cell, refined) in overlay.refined.iter().enumerate() {
            if !refined {
                continue;
            }
            lat.push(overlay.latitude_degrees[cell]);
            lon.push(overlay.longitude_degrees[cell]);
            spacing.push(overlay.spacing_metres[cell]);
        }
    }
    (lat, lon, spacing)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::weights::{AngleUnits, MeshCoordinates, build_weights};
    use crate::window::{LatLonWindow, Window};

    /// A small lat-lon window and two meshes over it: a coarse one that
    /// spans the whole box, and a fine one whose cells cluster in the
    /// middle third.
    fn window() -> Window {
        Window::LatLon(LatLonWindow {
            south: 0.0,
            north: 9.0,
            west: 0.0,
            east: 9.0,
            spacing_degrees: 1.0,
            description: "test".to_string(),
        })
    }

    fn mesh(points: &[(f64, f64)]) -> MeshCoordinates {
        MeshCoordinates {
            latitude_degrees: points.iter().map(|p| p.0).collect(),
            longitude_degrees: points.iter().map(|p| p.1).collect(),
            source_path: std::path::PathBuf::from("test"),
            source_sha256: String::new(),
            angle_units: AngleUnits::Declared,
        }
    }

    fn coarse_source() -> CompositeSource {
        let mut points = Vec::new();
        for j in 0..5 {
            for i in 0..5 {
                points.push((j as f64 * 2.25, i as f64 * 2.25));
            }
        }
        let mesh = mesh(&points);
        let weights = build_weights(&mesh, "test", &window()).expect("coarse weights");
        // ~250 km cells: coarse everywhere, refined nowhere.
        let spacing = vec![250_000.0f64; points.len()];
        CompositeSource::base("coarse", &mesh, weights, spacing).expect("coarse source")
    }

    /// A globally-covering mesh whose middle is refined -- the shape a
    /// placed MPAS grid actually has.
    fn placed_source(label: &str, centre: (f64, f64)) -> CompositeSource {
        let mut points = Vec::new();
        let mut spacing = Vec::new();
        // Background, spread over the whole box.
        for j in 0..5 {
            for i in 0..5 {
                points.push((j as f64 * 2.25, i as f64 * 2.25));
                spacing.push(200_000.0f64);
            }
        }
        // A refined patch about `centre`, at a tenth of a degree.
        let mut k = 0.0;
        while k < 20.0 {
            let lat = centre.0 - 1.0 + (k / 10.0);
            let mut m = 0.0;
            while m < 20.0 {
                points.push((lat, centre.1 - 1.0 + (m / 10.0)));
                spacing.push(11_000.0f64);
                m += 1.0;
            }
            k += 1.0;
        }
        let mesh = mesh(&points);
        let weights = build_weights(&mesh, "test", &window()).expect("placed weights");
        CompositeSource::overlay(label, &mesh, weights, spacing).expect("placed source")
    }

    #[test]
    fn the_fine_source_owns_its_refined_region_and_nothing_else() {
        let base = coarse_source();
        let fine = placed_source("s01", (4.5, 4.5));
        let plan = CompositePlan::build(&base, &[&fine]).expect("plan");

        assert!(
            plan.counts[1] > 0,
            "the fine source has to win somewhere, or the composite shows nothing new"
        );
        assert!(
            plan.counts[0] > plan.counts[1],
            "a placed grid covers a small part of the frame; base {} fine {}",
            plan.counts[0],
            plan.counts[1]
        );
        // The corners are far from the refined patch and must stay coarse.
        for slot in [0usize, plan.nx - 1, plan.points() - plan.nx, plan.points() - 1] {
            assert_eq!(
                plan.source_index[slot], 0,
                "grid point {slot} is a corner and cannot be inside the fine region"
            );
        }
    }

    /// The defect this rule exists to prevent: a placed mesh is a whole
    /// globe, so "nearest cell" alone hands it the entire frame even
    /// though its background is no finer than the coarse run's.
    #[test]
    fn a_placed_mesh_does_not_win_the_frame_on_its_background() {
        let base = coarse_source();
        let fine = placed_source("s01", (4.5, 4.5));
        let plan = CompositePlan::build(&base, &[&fine]).expect("plan");
        let fine_share = plan.counts[1] as f64 / plan.points() as f64;
        assert!(
            fine_share < 0.5,
            "the fine source took {:.0} per cent of the frame; its background is 200 km and \
             must not displace the base",
            fine_share * 100.0
        );
    }

    #[test]
    fn two_placed_grids_each_own_their_own_ground() {
        let base = coarse_source();
        let west = placed_source("s01", (4.5, 2.0));
        let east = placed_source("s02", (4.5, 7.0));
        let plan = CompositePlan::build(&base, &[&west, &east]).expect("plan");
        assert_eq!(plan.labels, vec!["coarse", "s01", "s02"]);
        assert!(
            plan.counts[1] > 0 && plan.counts[2] > 0,
            "both placed grids must appear; got {:?}",
            plan.counts
        );
        assert_eq!(
            plan.counts.iter().sum::<usize>(),
            plan.points(),
            "every grid point belongs to exactly one source"
        );
    }

    /// Overlap is decided by resolution, not by order in the source list.
    #[test]
    fn the_finer_of_two_overlapping_grids_wins() {
        let base = coarse_source();
        let coarser = placed_source("s01", (4.5, 4.5));
        let mut finer = placed_source("s02", (4.5, 4.5));
        for (cell, refined) in finer.refined.iter().enumerate() {
            if *refined {
                finer.spacing_metres[cell] = 5_000.0;
            }
        }
        let plan = CompositePlan::build(&base, &[&coarser, &finer]).expect("plan");
        let reversed = CompositePlan::build(&base, &[&finer, &coarser]).expect("plan");
        assert!(
            plan.counts[2] > 0 && plan.counts[1] == 0,
            "the 5 km grid must take the overlap from the 11 km grid; got {:?}",
            plan.counts
        );
        assert_eq!(
            plan.counts[2], reversed.counts[1],
            "which grid wins cannot depend on the order they were listed in"
        );
    }

    #[test]
    fn the_seam_is_found_and_is_a_closed_edge() {
        let base = coarse_source();
        let fine = placed_source("s01", (4.5, 4.5));
        let plan = CompositePlan::build(&base, &[&fine]).expect("plan");
        let boundary = plan.boundary_slots();
        assert!(
            !boundary.is_empty(),
            "a composite with two sources has a seam; publishing none would hide it"
        );
        for slot in &boundary {
            assert!(*slot < plan.points());
        }
    }

    #[test]
    fn a_source_resampled_onto_a_different_window_is_refused() {
        let base = coarse_source();
        let other_window = Window::LatLon(LatLonWindow {
            south: 20.0,
            north: 29.0,
            west: 0.0,
            east: 9.0,
            spacing_degrees: 1.0,
            description: "elsewhere".to_string(),
        });
        let mut points = Vec::new();
        let mut spacing = Vec::new();
        for j in 0..5 {
            for i in 0..5 {
                points.push((20.0 + j as f64 * 2.25, i as f64 * 2.25));
                spacing.push(200_000.0f64);
            }
        }
        points.push((24.5, 4.5));
        spacing.push(11_000.0);
        let mesh = mesh(&points);
        let weights = build_weights(&mesh, "test", &other_window).expect("weights");
        let elsewhere =
            CompositeSource::overlay("elsewhere", &mesh, weights, spacing).expect("source");
        let err = CompositePlan::build(&base, &[&elsewhere]).expect_err("must refuse");
        let text = err.to_string();
        assert!(
            text.contains("different window") || text.contains("two different places"),
            "the refusal has to name the breakage; got {text}"
        );
    }

    #[test]
    fn a_source_with_no_refined_region_is_refused_by_name() {
        let mut points = Vec::new();
        for j in 0..5 {
            for i in 0..5 {
                points.push((j as f64 * 2.25, i as f64 * 2.25));
            }
        }
        let mesh = mesh(&points);
        let weights = build_weights(&mesh, "test", &window()).expect("weights");
        let err = CompositeSource::overlay("flat", &mesh, weights, vec![200_000.0; points.len()])
            .expect_err("a uniform overlay must be refused");
        assert!(
            err.to_string().contains("no refined region"),
            "got {err}"
        );
    }
}
