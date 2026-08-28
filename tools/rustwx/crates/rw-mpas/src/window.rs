//! Render windows: the structured target grids an unstructured frame is
//! gathered onto, and the WRF projection attributes that describe them.
//!
//! Two kinds, because the renderer treats them differently. A Lambert window
//! emulates a WRF domain and gets the renderer's native regional map
//! furniture; a regular latitude/longitude window is the honest shape for a
//! global overview. The Lambert setup and the two transforms below are WRF
//! `module_llxy.F` (`set_lc`, `ijll_lc` and `llij_lc`) transcribed, in f64,
//! so the emitted `XLAT`/`XLONG` agree with the projection attributes a
//! reader recomputes from.
//!
//! Two windows are FIXED boxes -- `focus` and `global` -- and their geometry
//! is frozen: their `weights_sha256` values are recorded in evidence already
//! written, and moving a corner by a metre changes every one of those digests
//! without changing a filename. The third, [`Window::mesh_focus`], is DERIVED
//! from the mesh being converted, which is what lets a mesh refined anywhere
//! render its own fine region without a fourth box being added here.

use std::f64::consts::PI;

use crate::error::{MpasError, MpasResult};

/// WRF's spherical earth. These windows emulate a WRF domain, so they use
/// WRF's radius and not a WGS84 mean radius, or the emitted coordinates would
/// disagree with the projection attributes the renderer reads.
pub const WRF_EARTH_RADIUS_M: f64 = 6_370_000.0;
/// The radius the MPAS meshes carry, used for physical distances only.
pub const MPAS_EARTH_RADIUS_M: f64 = 6_371_229.0;

const RAD: f64 = PI / 180.0;
const DEG: f64 = 180.0 / PI;

/// A cell belongs to the refined region when its spacing is within this
/// factor of the mesh's minimum spacing.
///
/// The rule is stated in the derived window's description string so it
/// travels with the window instead of living only here. 2.0 because a graded
/// mesh's density function is smooth: the finest spacing is attained over a
/// plateau of thousands of cells rather than at one freak cell, and doubling
/// it takes the core plus the innermost ring of the transition -- the part of
/// the mesh that carries structure the background cannot hold. A tighter
/// factor puts the window edge inside the gradient; a looser one walks out
/// into the background, and the derived dx stops being the resolution the
/// mesh actually carries where the window is pointed.
pub const MESH_FOCUS_REFINEMENT_FACTOR: f64 = 2.0;

/// Margin added to each half-extent of the refined region, as a fraction.
///
/// The refined core of a graded mesh is where the storm is; the transition
/// ring around it still resolves the inflow feeding it. Ten percent is one
/// panel border, not a second domain.
pub const MESH_FOCUS_MARGIN_FRACTION: f64 = 0.10;

/// The largest derived target grid, per side.
///
/// The breakage this prevents, named: a mesh refined over a whole hemisphere
/// at 3 km asks for 13,000 points a side, and the converter writes every 3-D
/// field at `(levels, ny, nx)` float32 -- 55 levels over 13,000^2 is 37 GB
/// for ONE field, and a frame carries several, so the run would allocate
/// until the host died and emit nothing. At 2,000 a side that same field is
/// 880 MB, which is the practical ceiling for a wrfout-shaped frame, and the
/// grid still carries 111x the points of the fixed CONUS `focus` box.
///
/// When the clamp binds, `dx_metres` is coarsened so the refined region stays
/// FULLY covered -- cropping would silently drop the part of the mesh the
/// caller asked to see -- and the description records what dx it started from.
pub const MESH_FOCUS_MAX_POINTS_PER_SIDE: usize = 2_000;

/// Below this |centre latitude| the derived window is emitted as lat-lon
/// rather than Lambert.
///
/// A Lambert cone's constant is `sin(|truelat|)`, so it collapses toward the
/// equator, and the projection origin `polej` in `set_lc` carries a `1/cone`
/// and recedes as it does. On the equator itself the cone is a literal 0/0 --
/// the two derived truelats sit symmetric about it, so
/// `log10(cos t1) - log10(cos t2)` is zero and so is its denominator -- and
/// EVERY emitted coordinate is NaN. That is measured, not feared:
/// `the_cone_is_why_a_near_equatorial_centre_falls_back` builds the window at
/// a centre of exactly 0 and finds `cone = 0`, `polej = inf` and no finite
/// coordinate in the grid.
///
/// Where the line sits is a judgement, and it is not a float-precision line:
/// measured, the f64 transcription still round-trips its own centre to
/// 6e-8 m at a centre of 0.5 degrees, where the cone is 0.0087 and the origin
/// sits 161,204 grid units away. What forces a line at all is that this
/// centre is DERIVED -- it is wherever the mesh's refined cells happen to
/// average out -- so a mesh refined symmetrically about the equator lands
/// arbitrarily close to that NaN, and no small number is a safe distance from
/// a singularity.
///
/// 10 degrees is where a Lambert cone is still recognisably a cone: measured,
/// its constant is 0.174 and its origin is 8,008 grid units off the grid,
/// which at 4.5 km is 5.7 earth radii. Below it the map's shape is degrading
/// toward a cylinder anyway, so emitting the cylinder is both better
/// conditioned and the honest answer for a near-equatorial region.
pub const MESH_FOCUS_MIN_LAMBERT_LATITUDE_DEG: f64 = 10.0;

/// Half the span between the two derived true latitudes.
///
/// Bracketing the centre is the standard WRF practice the fixed `focus`
/// window follows (37 N between 30 and 60). The half-span is additionally
/// held to `|centre_lat| / 2` so both truelats stay strictly in the centre's
/// own hemisphere: `set_lc` takes `hemi` from the SIGN of `truelat1`, so a
/// truelat that crosses the equator flips the cone to the wrong pole and the
/// window lands in the opposite hemisphere.
const MESH_FOCUS_TRUELAT_HALF_SPAN_DEG: f64 = 5.0;

/// The northernmost true latitude a derived window will use. `set_lc`
/// divides by `tan((90 - truelat) * pi / 360)`, which is zero at the pole.
const MESH_FOCUS_MAX_ABS_TRUELAT_DEG: f64 = 89.0;

/// How short the mean of the selected cells' unit vectors may be before the
/// centre it names has no direction. A mean this short means the selected
/// cells are spread over the sphere with no middle -- the exact case a
/// uniform global mesh produces, where the mean is the origin.
const MESH_FOCUS_MIN_CENTROID_NORM: f64 = 1.0e-6;

/// How far a selected cell may sit from the derived centre. Past a quarter
/// turn the window has passed the projection's horizon: a Lambert cone sends
/// the antipode to infinity, so the far cells project to a radius with no
/// bound and nx and ny would be sized from it, while a cylindrical window
/// that wide has wrapped the planet and covers the same ground twice.
const MESH_FOCUS_MAX_ANGULAR_RADIUS_RAD: f64 = PI / 2.0;

/// A regular latitude/longitude window; renders under WRF `MAP_PROJ=6`.
#[derive(Debug, Clone, PartialEq)]
pub struct LatLonWindow {
    pub south: f64,
    pub north: f64,
    pub west: f64,
    pub east: f64,
    pub spacing_degrees: f64,
    pub description: String,
}

/// A Lambert conformal window emulating a WRF domain (`MAP_PROJ=1`).
#[derive(Debug, Clone, PartialEq)]
pub struct LambertWindow {
    pub centre_lat: f64,
    pub centre_lon: f64,
    pub dx_metres: f64,
    pub nx: usize,
    pub ny: usize,
    pub truelat1: f64,
    pub truelat2: f64,
    pub stand_lon: f64,
    pub description: String,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Window {
    LatLon(LatLonWindow),
    Lambert(LambertWindow),
}

/// A projection attribute value as it lands in the emitted file.
#[derive(Debug, Clone, PartialEq)]
pub enum ProjAttr {
    Int(i32),
    Float(f64),
    Text(String),
}

impl Window {
    /// The catalogue of FIXED named windows: a coarse global overview plus
    /// one frozen CONUS box.
    ///
    /// Neither geometry may move. Both are inputs to `weights_sha256`, and
    /// those digests are recorded in evidence already written; a corner
    /// nudged by a metre invalidates every one of them silently. A mesh
    /// refined somewhere these boxes do not reach gets
    /// [`Window::mesh_focus`], not a third entry here.
    pub fn named(name: &str) -> MpasResult<Window> {
        match name {
            "global" => Ok(Window::LatLon(LatLonWindow {
                south: -90.0,
                north: 90.0,
                west: -180.0,
                east: 179.75,
                spacing_degrees: 0.25,
                description: "global 0.25 degree overview window".to_string(),
            })),
            "focus" => Ok(Window::Lambert(LambertWindow {
                centre_lat: 37.0,
                centre_lon: -96.0,
                dx_metres: 22_000.0,
                nx: 240,
                ny: 150,
                truelat1: 30.0,
                truelat2: 60.0,
                stand_lon: -96.0,
                description:
                    "CONUS focus window on a 22 km Lambert grid, matched to the x4.163842 refined-region spacing"
                        .to_string(),
            })),
            "mesh" => Err(MpasError::Refusal(
                "render window 'mesh' is derived from the mesh being converted, not served \
                 from this catalogue: build it with Window::mesh_focus once that mesh's cell \
                 coordinates and per-cell spacing have been read. Reaching here means a \
                 caller asked for the derived window without the mesh in hand, and answering \
                 with a fixed box would render a region the mesh may not refine at all"
                    .to_string(),
            )),
            other => Err(MpasError::Refusal(format!(
                "unknown render window '{other}' (known: focus, global, mesh)"
            ))),
        }
    }

    /// The render window a mesh implies: its own refined region, at its own
    /// spacing.
    ///
    /// The defect this exists for, measured: a global variable-resolution
    /// mesh whose refined core sits at 15-20 N, 50-57 W attains 4.53 km
    /// inside a 75 km background, and the only windows on offer were a fixed
    /// CONUS box 5,000 km away -- which contains none of that core and
    /// samples 75-80 km background cells -- and a 0.25 degree global grid,
    /// which is 28 km and cannot resolve a 4.5 km core at all. There was no
    /// way to render that mesh's fine region at its own resolution, and every
    /// future mesh placed somewhere new would have needed another hardcoded
    /// box. This derives the box instead, so a mesh refined anywhere is data
    /// rather than a code path.
    ///
    /// `cell_spacing_m` is the per-cell horizontal spacing in metres, one
    /// value per cell, on the same ordering as the coordinates --
    /// [`crate::weights::read_mesh_cell_spacing`] reads it off a mesh file.
    ///
    /// What it decides, in order:
    ///
    /// - The refined region is every cell within
    ///   [`MESH_FOCUS_REFINEMENT_FACTOR`] of the mesh's minimum spacing.
    /// - The centre is the mean of those cells' unit vectors, re-normalised.
    ///   Not a mean of longitudes: a region straddling the antimeridian
    ///   averages +179 and -179 to 0 and lands the window on the wrong side
    ///   of the planet.
    /// - `dx_metres` is the MEDIAN spacing of the selected cells, so the
    ///   window resolves what the mesh carries there rather than what its
    ///   single finest cell carries.
    /// - `nx`/`ny` cover the selected cells' extent plus
    ///   [`MESH_FOCUS_MARGIN_FRACTION`], clamped by
    ///   [`MESH_FOCUS_MAX_POINTS_PER_SIDE`]; a clamp coarsens `dx_metres`
    ///   instead of cropping.
    /// - Lambert when the centre is far enough from the equator for the cone
    ///   to be conditioned ([`MESH_FOCUS_MIN_LAMBERT_LATITUDE_DEG`]),
    ///   lat-lon when it is not.
    ///
    /// A UNIFORM mesh is refused by name rather than answered. It has no
    /// refined region to find, and the whole-mesh answer is not merely
    /// unhelpful: the unit-vector mean of a quasi-uniform global mesh IS the
    /// origin, so the centre would be float noise pointing in an arbitrary
    /// direction, and the extent would be the whole sphere, which no Lambert
    /// cone represents. `--window global` is the window a uniform mesh has.
    pub fn mesh_focus(
        latitude_degrees: &[f64],
        longitude_degrees: &[f64],
        cell_spacing_m: &[f64],
    ) -> MpasResult<Window> {
        let n = latitude_degrees.len();
        if n == 0 {
            return Err(MpasError::Refusal(
                "a mesh-derived window needs at least one cell; the mesh handed over none, \
                 so there is no refined region to centre on and no spacing to render at"
                    .to_string(),
            ));
        }
        if longitude_degrees.len() != n || cell_spacing_m.len() != n {
            return Err(MpasError::Refusal(format!(
                "a mesh-derived window needs one latitude, longitude and spacing per cell; \
                 got {n} latitude(s), {} longitude(s) and {} spacing(s). Zipping arrays of \
                 different lengths pairs one cell's position with another cell's resolution",
                longitude_degrees.len(),
                cell_spacing_m.len()
            )));
        }
        for i in 0..n {
            if !latitude_degrees[i].is_finite() || !longitude_degrees[i].is_finite() {
                return Err(MpasError::Refusal(format!(
                    "mesh cell {i} sits at latitude {} longitude {}, which is not finite; a \
                     non-finite coordinate poisons the unit-vector mean and every derived \
                     corner with it",
                    latitude_degrees[i], longitude_degrees[i]
                )));
            }
            if latitude_degrees[i].abs() > 90.0 + 1.0e-6 {
                return Err(MpasError::Refusal(format!(
                    "mesh cell {i} sits at latitude {}, outside [-90, 90]",
                    latitude_degrees[i]
                )));
            }
            if !cell_spacing_m[i].is_finite() || cell_spacing_m[i] <= 0.0 {
                return Err(MpasError::Refusal(format!(
                    "mesh cell {i} carries spacing {} m; a spacing has to be positive and \
                     finite or the refinement ratio it is compared against is meaningless",
                    cell_spacing_m[i]
                )));
            }
        }

        // --- the refined region -------------------------------------------
        let min_spacing = cell_spacing_m.iter().copied().fold(f64::INFINITY, f64::min);
        let selected = Self::refined_region(cell_spacing_m)?;
        let median_spacing_m = Self::median_spacing_of(cell_spacing_m, &selected);
        let selection = format!(
            "{} of {n} cells carry spacing at or under {:.2}x the mesh minimum {:.3} km and              are taken as the refined region",
            selected.len(),
            MESH_FOCUS_REFINEMENT_FACTOR,
            min_spacing / 1000.0,
        );

        Self::window_over_selected_cells(
            latitude_degrees,
            longitude_degrees,
            &selected,
            median_spacing_m,
            MESH_FOCUS_MARGIN_FRACTION,
            &selection,
            "mesh-derived focus window",
        )
    }

    /// A window covering every fine region handed over, with room around
    /// them for the coarse field they will be composited into.
    ///
    /// gpuwm addition (VENDOR.md).  `mesh_focus` frames ONE mesh's refined
    /// region tightly, which is right for rendering that core on its own
    /// and wrong for a composite: a tight frame contains no coarse field,
    /// so there is nothing to composite against and the picture is the fine
    /// render again. This takes the refined cells of every source at once
    /// -- so N placed grids get one frame, not N -- renders at the finest
    /// spacing among them, and pulls the frame out to
    /// `context_factor` times the fine footprint.
    ///
    /// The same clamp applies: past 2000 points a side the spacing
    /// coarsens rather than the frame cropping. When N grids are scattered
    /// far apart, that clamp is what a caller will hit, and it is the
    /// honest outcome -- one frame wide enough to hold two grids on
    /// opposite sides of an ocean cannot also resolve 5 km. The
    /// description says so, in the numbers.
    pub fn composite_focus(
        latitude_degrees: &[f64],
        longitude_degrees: &[f64],
        spacing_metres: &[f64],
        context_factor: f64,
    ) -> MpasResult<Window> {
        let n = latitude_degrees.len();
        if n == 0 {
            return Err(MpasError::Refusal(
                "a composite window is sized to the fine regions being composited in; none                  was handed over, so the window would have nothing to centre on"
                    .to_string(),
            ));
        }
        if longitude_degrees.len() != n || spacing_metres.len() != n {
            return Err(MpasError::Refusal(format!(
                "a composite window needs one latitude, longitude and spacing per fine cell;                  got {n}, {} and {}",
                longitude_degrees.len(),
                spacing_metres.len()
            )));
        }
        if !(context_factor.is_finite() && context_factor >= 1.0) {
            return Err(MpasError::Refusal(format!(
                "a context factor of {context_factor} would draw a frame smaller than the                  fine regions it is meant to surround"
            )));
        }
        let selected: Vec<usize> = (0..n).collect();
        let finest = spacing_metres
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min);
        if !finest.is_finite() || finest <= 0.0 {
            return Err(MpasError::Refusal(format!(
                "the finest spacing among the composited regions is {finest} m; a window                  cannot be rendered at it"
            )));
        }
        let selection = format!(
            "{n} fine cell(s) from the composited sources, finest spacing {:.3} km, drawn              into a frame {context_factor:.1}x their own extent so the coarse field they sit              in is visible",
            finest / 1000.0
        );
        Self::window_over_selected_cells(
            latitude_degrees,
            longitude_degrees,
            &selected,
            finest,
            context_factor - 1.0,
            &selection,
            "composite window",
        )
    }

    /// The median spacing of `selected`, which is the resolution a window
    /// over them renders at.
    fn median_spacing_of(cell_spacing_m: &[f64], selected: &[usize]) -> f64 {
        let mut spacings: Vec<f64> = selected.iter().map(|&i| cell_spacing_m[i]).collect();
        let mid = spacings.len() / 2;
        spacings.select_nth_unstable_by(mid, |a, b| a.total_cmp(b));
        spacings[mid]
    }

    /// The cells of one mesh that make up its refined region.
    ///
    /// The rule is the mesh's own: spacing at or under
    /// `MESH_FOCUS_REFINEMENT_FACTOR` times that mesh's minimum. It needs
    /// no cross-mesh comparison and no tunable, which is why the composite
    /// path reuses it rather than inventing a second definition of
    /// "refined" that could disagree with the window a fine core renders
    /// itself on.
    pub fn refined_region(cell_spacing_m: &[f64]) -> MpasResult<Vec<usize>> {
        let n = cell_spacing_m.len();
        if n == 0 {
            return Err(MpasError::Refusal(
                "a refined region needs at least one cell spacing; none was handed over"
                    .to_string(),
            ));
        }
        let min_spacing = cell_spacing_m.iter().copied().fold(f64::INFINITY, f64::min);
        let max_spacing = cell_spacing_m
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        if !min_spacing.is_finite() || min_spacing <= 0.0 {
            return Err(MpasError::Refusal(format!(
                "the mesh minimum spacing is {min_spacing}; a refinement ratio measured                  against it is meaningless"
            )));
        }
        if max_spacing <= MESH_FOCUS_REFINEMENT_FACTOR * min_spacing {
            return Err(MpasError::Refusal(format!(
                "this mesh is uniform to within the refinement factor -- spacing runs                  {:.3} km to {:.3} km, a ratio of {:.3}x against the {MESH_FOCUS_REFINEMENT_FACTOR:.2}x                  that marks a refined region -- so there is no refined region to derive a                  window from. Deriving one anyway would centre the window on the unit-vector                  mean of every cell, which for a quasi-uniform global mesh is the origin and                  therefore points nowhere, and would size it to the whole sphere, which no                  Lambert cone represents. Render a uniform mesh with --window global",
                min_spacing / 1000.0,
                max_spacing / 1000.0,
                max_spacing / min_spacing
            )));
        }
        let threshold = MESH_FOCUS_REFINEMENT_FACTOR * min_spacing;
        let selected: Vec<usize> = (0..n).filter(|&i| cell_spacing_m[i] <= threshold).collect();
        // Unreachable while the minimum is a member of the mesh, but a
        // silently empty selection would divide by zero downstream.
        if selected.is_empty() {
            return Err(MpasError::Refusal(format!(
                "no cell has spacing at or under {:.3} km, though the mesh minimum is {:.3} km",
                threshold / 1000.0,
                min_spacing / 1000.0
            )));
        }
        Ok(selected)
    }

    /// A window centred on, and sized to cover, an explicit set of cells.
    ///
    /// gpuwm addition (VENDOR.md): extracted verbatim from `mesh_focus`
    /// rather than copied, because the composite window has to be built by
    /// the same geometry as a single fine core's own window or the two
    /// frames disagree and the composite shows the fine data in the wrong
    /// place. `margin_fraction` is the one thing that differs: a fine core
    /// wants a tight frame, a composite wants surrounding context.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn window_over_selected_cells(
        latitude_degrees: &[f64],
        longitude_degrees: &[f64],
        selected: &[usize],
        median_spacing_m: f64,
        margin_fraction: f64,
        selection: &str,
        label: &str,
    ) -> MpasResult<Window> {
        // --- the centre, by unit-vector mean ------------------------------
        let mut sum = [0.0f64; 3];
        for &i in selected {
            let v = unit_vector(latitude_degrees[i], longitude_degrees[i]);
            for k in 0..3 {
                sum[k] += v[k];
            }
        }
        let count = selected.len() as f64;
        let mean = [sum[0] / count, sum[1] / count, sum[2] / count];
        let norm = (mean[0] * mean[0] + mean[1] * mean[1] + mean[2] * mean[2]).sqrt();
        if !norm.is_finite() || norm < MESH_FOCUS_MIN_CENTROID_NORM {
            return Err(MpasError::Refusal(format!(
                "the {} selected cells have a mean direction of magnitude {norm:e}, \
                 under the {MESH_FOCUS_MIN_CENTROID_NORM:e} a centre needs to have a \
                 direction at all: they are spread over the sphere with no middle, so any \
                 point this returned would be float noise",
                selected.len()
            )));
        }
        let up = [mean[0] / norm, mean[1] / norm, mean[2] / norm];
        let centre_lat = up[2].clamp(-1.0, 1.0).asin() * DEG;
        let centre_lon = up[1].atan2(up[0]) * DEG;

        // --- the horizon guard --------------------------------------------
        let mut max_angle_rad = 0.0f64;
        for &i in selected {
            let v = unit_vector(latitude_degrees[i], longitude_degrees[i]);
            let dot = (v[0] * up[0] + v[1] * up[1] + v[2] * up[2]).clamp(-1.0, 1.0);
            max_angle_rad = max_angle_rad.max(dot.acos());
        }
        if max_angle_rad > MESH_FOCUS_MAX_ANGULAR_RADIUS_RAD {
            return Err(MpasError::Refusal(format!(
                "the refined region reaches {:.2} degrees of arc from its own centre, past \
                 the {:.2} degree horizon a projected window has. A Lambert cone sends the \
                 antipode to infinity, so the far cells would project to a radius with no \
                 bound and nx and ny would be sized from it; a cylindrical window that wide \
                 has wrapped the planet and is covering the same ground twice. A mesh \
                 refined that widely wants --window global",
                max_angle_rad * DEG,
                MESH_FOCUS_MAX_ANGULAR_RADIUS_RAD * DEG
            )));
        }

        let selection = format!(
            "{selection}; centre {} from their unit-vector mean",
            format_centre(centre_lat, centre_lon)
        );

        // --- lat-lon, when the cone will not stand ------------------------
        //
        // Sized in DEGREES, which is exact for a cylindrical grid: the axes
        // ARE the coordinates, so no proxy geometry sits between the region
        // and the box that has to hold it.
        if centre_lat.abs() < MESH_FOCUS_MIN_LAMBERT_LATITUDE_DEG {
            let mut half_lat_deg = 0.0f64;
            let mut half_lon_deg = 0.0f64;
            for &i in selected {
                half_lat_deg = half_lat_deg.max((latitude_degrees[i] - centre_lat).abs());
                let delta =
                    (longitude_degrees[i] - centre_lon + 180.0).rem_euclid(360.0) - 180.0;
                half_lon_deg = half_lon_deg.max(delta.abs());
            }
            let nominal_degrees = median_spacing_m / (RAD * WRF_EARTH_RADIUS_M);
            half_lat_deg = (half_lat_deg * (1.0 + margin_fraction)).max(nominal_degrees);
            half_lon_deg = (half_lon_deg * (1.0 + margin_fraction)).max(nominal_degrees);

            let span = (MESH_FOCUS_MAX_POINTS_PER_SIDE - 1) as f64;
            let floor_degrees = (2.0 * half_lat_deg / span).max(2.0 * half_lon_deg / span);
            let mut spacing_degrees = nominal_degrees;
            let mut coarsened_from: Option<f64> = None;
            if spacing_degrees < floor_degrees {
                coarsened_from = Some(spacing_degrees);
                // The hair of headroom keeps the ceiling below from rounding
                // one point past the limit the clamp exists to hold.
                spacing_degrees = floor_degrees * (1.0 + 1.0e-12);
            }
            let points = |half: f64| -> usize {
                ((2.0 * half / spacing_degrees).ceil() as usize + 1)
                    .clamp(2, MESH_FOCUS_MAX_POINTS_PER_SIDE)
            };
            let nx = points(half_lon_deg);
            // The latitude axis has to stay inside [-90, 90]; `coordinates`
            // refuses a window that leaves it.
            let mut ny = points(half_lat_deg);
            let room = 2.0 * (90.0 - centre_lat.abs()) / spacing_degrees;
            if (ny - 1) as f64 > room {
                ny = (room.floor() as usize + 1).max(2);
            }
            let south = centre_lat - 0.5 * (ny - 1) as f64 * spacing_degrees;
            let north = south + (ny - 1) as f64 * spacing_degrees;
            let west = centre_lon - 0.5 * (nx - 1) as f64 * spacing_degrees;
            let east_bound = west + (nx - 1) as f64 * spacing_degrees;
            return Ok(Window::LatLon(LatLonWindow {
                south,
                north,
                west,
                east: east_bound,
                spacing_degrees,
                description: format!(
                    "{label} (lat-lon): {selection}; spacing \
                     {spacing_degrees:.6} degrees is their median {:.3} km{}; {nx}x{ny} covers \
                     their {:.3} by {:.3} degree extent including a {:.0} percent margin. The \
                     centre is {:.2} degrees from the equator, under the \
                     {MESH_FOCUS_MIN_LAMBERT_LATITUDE_DEG:.0} a Lambert cone needs: its cone \
                     constant is sin(|latitude|), which is zero on the equator, and the \
                     projection origin carries 1/cone. The longitude axis is allowed past \
                     +/-180 rather than wrapped, because an axis has to increase",
                    median_spacing_m / 1000.0,
                    match coarsened_from {
                        Some(was) => format!(
                            ", coarsened from {was:.6} degrees because that extent needed more \
                             than {MESH_FOCUS_MAX_POINTS_PER_SIDE} points a side"
                        ),
                        None => String::new(),
                    },
                    2.0 * half_lon_deg,
                    2.0 * half_lat_deg,
                    100.0 * margin_fraction,
                    centre_lat.abs()
                ),
            }));
        }

        // --- Lambert ------------------------------------------------------
        let hemi = if centre_lat >= 0.0 { 1.0 } else { -1.0 };
        let abs_lat = centre_lat.abs();
        let half_span = MESH_FOCUS_TRUELAT_HALF_SPAN_DEG.min(0.5 * abs_lat);
        let truelat1 = hemi * (abs_lat - half_span);
        let truelat2 = hemi * (abs_lat + half_span).min(MESH_FOCUS_MAX_ABS_TRUELAT_DEG);

        // The extent is measured ON the projection the window will use, not
        // on a proxy. A proxy is where a sizing goes wrong quietly: a
        // great-circle or azimuthal offset is SHORTER than the Lambert
        // easting across a wide region, so nx would come out too small and
        // the corners the window exists to show would be cropped off it.
        //
        // `dx` cancels out of the measurement -- `set_lc` divides by it once
        // through `rebydx`, so the offsets scale as 1/dx -- and the probe
        // therefore runs at the median spacing and reports metres, which is
        // what the clamp below works in.
        let probe = LambertProjection::set_lc(
            truelat1,
            truelat2,
            centre_lon,
            median_spacing_m,
            centre_lat,
            centre_lon,
            0.0,
            0.0,
        );
        let mut half_width_m = 0.0f64;
        let mut half_height_m = 0.0f64;
        for &i in selected {
            let (gi, gj) = probe.llij(latitude_degrees[i], longitude_degrees[i]);
            if !gi.is_finite() || !gj.is_finite() {
                return Err(MpasError::Refusal(format!(
                    "refined cell {i} at {} projects to ({gi}, {gj}) on the derived Lambert \
                     cone (truelats {truelat1:.3}/{truelat2:.3}); a window cannot be sized \
                     from a coordinate that is not finite",
                    format_centre(latitude_degrees[i], longitude_degrees[i])
                )));
            }
            half_width_m = half_width_m.max(gi.abs() * median_spacing_m);
            half_height_m = half_height_m.max(gj.abs() * median_spacing_m);
        }
        // A one-cell refined region has no extent; floor each half-extent at
        // one dx so the window is a 3x3 rather than a degenerate 2x2.
        half_width_m = (half_width_m * (1.0 + margin_fraction)).max(median_spacing_m);
        half_height_m = (half_height_m * (1.0 + margin_fraction)).max(median_spacing_m);

        // --- the clamp: coarsen, never crop -------------------------------
        let span = (MESH_FOCUS_MAX_POINTS_PER_SIDE - 1) as f64;
        let dx_floor = (2.0 * half_width_m / span).max(2.0 * half_height_m / span);
        let mut dx_metres = median_spacing_m;
        let mut coarsened_from: Option<f64> = None;
        if dx_metres < dx_floor {
            coarsened_from = Some(dx_metres);
            // The hair of headroom keeps the ceiling below from rounding one
            // point past the limit the clamp exists to hold.
            dx_metres = dx_floor * (1.0 + 1.0e-12);
        }
        let points = |half: f64| -> usize {
            ((2.0 * half / dx_metres).ceil() as usize + 1)
                .clamp(2, MESH_FOCUS_MAX_POINTS_PER_SIDE)
        };
        let nx = points(half_width_m);
        let ny = points(half_height_m);

        Ok(Window::Lambert(LambertWindow {
            centre_lat,
            centre_lon,
            dx_metres,
            nx,
            ny,
            truelat1,
            truelat2,
            stand_lon: centre_lon,
            description: format!(
                "{label} (Lambert): {selection}; dx {:.3} km is their \
                 median spacing{}; {nx}x{ny} covers their {:.0} by {:.0} km extent on this \
                 cone including a {:.0} percent margin; truelats {truelat1:.3} and \
                 {truelat2:.3} bracket the centre and stand_lon is the centre longitude",
                dx_metres / 1000.0,
                match coarsened_from {
                    Some(was) => format!(
                        ", coarsened from {:.3} km because that extent needed more than \
                         {MESH_FOCUS_MAX_POINTS_PER_SIDE} points a side",
                        was / 1000.0
                    ),
                    None => String::new(),
                },
                2.0 * half_width_m / 1000.0,
                2.0 * half_height_m / 1000.0,
                100.0 * margin_fraction,
            ),
        }))
    }

    pub fn description(&self) -> &str {
        match self {
            Window::LatLon(w) => &w.description,
            Window::Lambert(w) => &w.description,
        }
    }

    /// The canonical specification, as the weights digest and the emitted
    /// `MPAS_RENDER_WINDOW_SPEC` attribute serialize it: JSON object keys in
    /// sorted order, which is what makes the digest reproducible.
    pub fn spec_json(&self) -> String {
        match self {
            Window::LatLon(w) => format!(
                "{{\"east\": {}, \"kind\": \"latlon\", \"north\": {}, \"south\": {}, \"spacing_degrees\": {}, \"west\": {}}}",
                json_f64(w.east),
                json_f64(w.north),
                json_f64(w.south),
                json_f64(w.spacing_degrees),
                json_f64(w.west)
            ),
            Window::Lambert(w) => format!(
                "{{\"centre_lat\": {}, \"centre_lon\": {}, \"dx_metres\": {}, \"earth_radius_m\": {}, \"kind\": \"lambert\", \"nx\": {}, \"ny\": {}, \"stand_lon\": {}, \"truelat1\": {}, \"truelat2\": {}}}",
                json_f64(w.centre_lat),
                json_f64(w.centre_lon),
                json_f64(w.dx_metres),
                json_f64(WRF_EARTH_RADIUS_M),
                w.nx,
                w.ny,
                json_f64(w.stand_lon),
                json_f64(w.truelat1),
                json_f64(w.truelat2)
            ),
        }
    }

    /// The same specification with no spaces, which is the form the weights
    /// digest hashes (`json.dumps(separators=(",", ":"))`).
    pub fn spec_json_compact(&self) -> String {
        self.spec_json().replace(", ", ",").replace("\": ", "\":")
    }

    /// Target grid shape `(ny, nx)`.
    pub fn shape(&self) -> MpasResult<(usize, usize)> {
        match self {
            Window::Lambert(w) => Ok((w.ny, w.nx)),
            Window::LatLon(w) => {
                let lat = axis(w.south, w.north, w.spacing_degrees, "latitude")?;
                let lon = axis(w.west, w.east, w.spacing_degrees, "longitude")?;
                Ok((lat.len(), lon.len()))
            }
        }
    }

    /// Target latitudes and longitudes, row-major `(ny, nx)`, in degrees.
    pub fn coordinates(&self) -> MpasResult<(Vec<f64>, Vec<f64>)> {
        match self {
            Window::LatLon(w) => {
                let lat = axis(w.south, w.north, w.spacing_degrees, "latitude")?;
                let lon = axis(w.west, w.east, w.spacing_degrees, "longitude")?;
                if lat.iter().any(|v| v.abs() > 90.0) {
                    return Err(MpasError::Refusal(
                        "latitude window leaves [-90, 90]".to_string(),
                    ));
                }
                let mut lat_grid = Vec::with_capacity(lat.len() * lon.len());
                let mut lon_grid = Vec::with_capacity(lat.len() * lon.len());
                for &la in &lat {
                    for &lo in &lon {
                        lat_grid.push(la);
                        lon_grid.push(lo);
                    }
                }
                Ok((lat_grid, lon_grid))
            }
            Window::Lambert(w) => {
                if w.nx < 2 || w.ny < 2 {
                    return Err(MpasError::Refusal(
                        "a Lambert window needs at least 2x2 points".to_string(),
                    ));
                }
                if !w.dx_metres.is_finite() || w.dx_metres <= 0.0 {
                    return Err(MpasError::Refusal(
                        "Lambert dx must be positive and finite".to_string(),
                    ));
                }
                let projection = LambertProjection::set_lc(
                    w.truelat1,
                    w.truelat2,
                    w.stand_lon,
                    w.dx_metres,
                    w.centre_lat,
                    w.centre_lon,
                    0.5 * (w.nx as f64 + 1.0),
                    0.5 * (w.ny as f64 + 1.0),
                );
                let mut lat_grid = Vec::with_capacity(w.nx * w.ny);
                let mut lon_grid = Vec::with_capacity(w.nx * w.ny);
                for j in 1..=w.ny {
                    for i in 1..=w.nx {
                        let (la, lo) = projection.ijll(i as f64, j as f64);
                        lat_grid.push(la);
                        lon_grid.push(lo);
                    }
                }
                Ok((lat_grid, lon_grid))
            }
        }
    }

    /// The WRF global attributes describing this projection, in the order the
    /// emitted file carries them.
    pub fn projection_attributes(&self) -> MpasResult<Vec<(String, ProjAttr)>> {
        match self {
            Window::LatLon(w) => {
                let lat = axis(w.south, w.north, w.spacing_degrees, "latitude")?;
                let lon = axis(w.west, w.east, w.spacing_degrees, "longitude")?;
                let centre_lat = 0.5 * (lat[0] + lat[lat.len() - 1]);
                let centre_lon = 0.5 * (lon[0] + lon[lon.len() - 1]);
                // WRF writes DX/DY in metres even for a lat-lon grid; the
                // nominal spacing at the window centre is the honest number.
                let metres = w.spacing_degrees * RAD * WRF_EARTH_RADIUS_M;
                Ok(vec![
                    ("MAP_PROJ".into(), ProjAttr::Int(6)),
                    (
                        "MAP_PROJ_CHAR".into(),
                        ProjAttr::Text("Cylindrical Equidistant".into()),
                    ),
                    ("DX".into(), ProjAttr::Float(metres)),
                    ("DY".into(), ProjAttr::Float(metres)),
                    ("TRUELAT1".into(), ProjAttr::Float(0.0)),
                    ("TRUELAT2".into(), ProjAttr::Float(0.0)),
                    ("STAND_LON".into(), ProjAttr::Float(centre_lon)),
                    ("CEN_LAT".into(), ProjAttr::Float(centre_lat)),
                    ("CEN_LON".into(), ProjAttr::Float(centre_lon)),
                    ("MOAD_CEN_LAT".into(), ProjAttr::Float(centre_lat)),
                    ("POLE_LAT".into(), ProjAttr::Float(90.0)),
                    ("POLE_LON".into(), ProjAttr::Float(0.0)),
                ])
            }
            Window::Lambert(w) => {
                let (lat, lon) = self.coordinates()?;
                let centre_j = w.ny / 2;
                let centre_i = w.nx / 2;
                let at = centre_j * w.nx + centre_i;
                Ok(vec![
                    ("MAP_PROJ".into(), ProjAttr::Int(1)),
                    (
                        "MAP_PROJ_CHAR".into(),
                        ProjAttr::Text("Lambert Conformal".into()),
                    ),
                    ("DX".into(), ProjAttr::Float(w.dx_metres)),
                    ("DY".into(), ProjAttr::Float(w.dx_metres)),
                    ("TRUELAT1".into(), ProjAttr::Float(w.truelat1)),
                    ("TRUELAT2".into(), ProjAttr::Float(w.truelat2)),
                    ("STAND_LON".into(), ProjAttr::Float(w.stand_lon)),
                    ("CEN_LAT".into(), ProjAttr::Float(lat[at])),
                    ("CEN_LON".into(), ProjAttr::Float(lon[at])),
                    ("MOAD_CEN_LAT".into(), ProjAttr::Float(lat[at])),
                    ("POLE_LAT".into(), ProjAttr::Float(90.0)),
                    ("POLE_LON".into(), ProjAttr::Float(0.0)),
                ])
            }
        }
    }
}

/// The point on the unit sphere a latitude/longitude pair names.
fn unit_vector(lat_degrees: f64, lon_degrees: f64) -> [f64; 3] {
    let lat = lat_degrees * RAD;
    let lon = lon_degrees * RAD;
    let cos_lat = lat.cos();
    [cos_lat * lon.cos(), cos_lat * lon.sin(), lat.sin()]
}

/// A centre as a reader says it out loud: `17.312N 53.897W`.
fn format_centre(lat: f64, lon: f64) -> String {
    format!(
        "{:.3}{} {:.3}{}",
        lat.abs(),
        if lat >= 0.0 { "N" } else { "S" },
        lon.abs(),
        if lon >= 0.0 { "E" } else { "W" }
    )
}

/// Render a float the way `json.dumps` does, so the digest input matches.
fn json_f64(v: f64) -> String {
    if v == v.trunc() && v.abs() < 1e16 {
        format!("{v:.1}")
    } else {
        format!("{v}")
    }
}

fn axis(first: f64, last: f64, step: f64, role: &str) -> MpasResult<Vec<f64>> {
    if !(first.is_finite() && last.is_finite() && step.is_finite()) {
        return Err(MpasError::Refusal(format!(
            "{role} window bounds must be finite"
        )));
    }
    if last <= first || step <= 0.0 {
        return Err(MpasError::Refusal(format!("{role} window must increase")));
    }
    let count = ((last - first) / step).round() as usize + 1;
    let mut values: Vec<f64> = (0..count).map(|k| first + k as f64 * step).collect();
    let end = values[count - 1];
    if (end - last).abs() > 1.0e-9 {
        return Err(MpasError::Refusal(format!(
            "{role} spacing does not exactly partition its bounds"
        )));
    }
    values[count - 1] = last;
    Ok(values)
}

/// WRF `module_llxy.F` Lambert state.
#[derive(Debug, Clone, Copy)]
pub struct LambertProjection {
    cone: f64,
    hemi: f64,
    rebydx: f64,
    polei: f64,
    polej: f64,
    truelat1: f64,
    truelat2: f64,
    stand_lon: f64,
}

fn lambert_cone(truelat1: f64, truelat2: f64) -> f64 {
    if (truelat1 - truelat2).abs() > 0.1 {
        let numerator = (truelat1 * RAD).cos().log10() - (truelat2 * RAD).cos().log10();
        let denominator = ((45.0 - truelat1.abs() / 2.0) * RAD).tan().log10()
            - ((45.0 - truelat2.abs() / 2.0) * RAD).tan().log10();
        numerator / denominator
    } else {
        (truelat1.abs() * RAD).sin()
    }
}

impl LambertProjection {
    /// WRF `set_lc`, transcribed.
    #[allow(clippy::too_many_arguments)]
    pub fn set_lc(
        truelat1: f64,
        truelat2: f64,
        stand_lon: f64,
        dx_metres: f64,
        known_lat: f64,
        known_lon: f64,
        known_i: f64,
        known_j: f64,
    ) -> Self {
        let cone = lambert_cone(truelat1, truelat2);
        let hemi = if truelat1 >= 0.0 { 1.0 } else { -1.0 };
        let rebydx = WRF_EARTH_RADIUS_M / dx_metres;
        let mut delta_lon = known_lon - stand_lon;
        if delta_lon > 180.0 {
            delta_lon -= 360.0;
        }
        if delta_lon < -180.0 {
            delta_lon += 360.0;
        }
        let ctl1r = (truelat1 * RAD).cos();
        let rsw = rebydx * ctl1r / cone
            * (((90.0 * hemi - known_lat) * RAD / 2.0).tan()
                / ((90.0 * hemi - truelat1) * RAD / 2.0).tan())
            .powf(cone);
        let arg = cone * (delta_lon * RAD);
        let polei = hemi * known_i - hemi * rsw * arg.sin();
        let polej = hemi * known_j + rsw * arg.cos();
        LambertProjection {
            cone,
            hemi,
            rebydx,
            polei,
            polej,
            truelat1,
            truelat2,
            stand_lon,
        }
    }

    /// WRF `llij_lc`, transcribed. Returns `(i, j)`, the grid coordinate a
    /// latitude/longitude pair lands on, in the same 1-based convention
    /// [`LambertProjection::ijll`] reads back.
    ///
    /// The inverse alone was enough while every window was a fixed box. A
    /// DERIVED window has to ask the opposite question -- how many points does
    /// it take to cover these cells -- and answering it with a great-circle or
    /// azimuthal proxy undersizes a wide region, because those offsets are
    /// shorter than the Lambert easting they stand in for. Measuring on the
    /// cone the window will actually use makes the coverage exact instead.
    pub fn llij(&self, latitude: f64, longitude: f64) -> (f64, f64) {
        let hemi = self.hemi;
        let mut delta_lon = longitude - self.stand_lon;
        if delta_lon > 180.0 {
            delta_lon -= 360.0;
        }
        if delta_lon < -180.0 {
            delta_lon += 360.0;
        }
        let ctl1r = (self.truelat1 * RAD).cos();
        let rm = self.rebydx * ctl1r / self.cone
            * (((90.0 * hemi - latitude) * RAD / 2.0).tan()
                / ((90.0 * hemi - self.truelat1) * RAD / 2.0).tan())
            .powf(self.cone);
        let arg = self.cone * (delta_lon * RAD);
        let i = self.polei + hemi * rm * arg.sin();
        let j = self.polej - rm * arg.cos();
        (hemi * i, hemi * j)
    }

    /// WRF `ijll_lc`, transcribed. Returns `(latitude, longitude)` in degrees.
    pub fn ijll(&self, i: f64, j: f64) -> (f64, f64) {
        let hemi = self.hemi;
        let chi1 = (90.0 - hemi * self.truelat1) * RAD;
        let chi2 = (90.0 - hemi * self.truelat2) * RAD;
        let inew = hemi * i;
        let jnew = hemi * j;
        let xx = inew - self.polei;
        let yy = self.polej - jnew;
        let radius_squared = xx * xx + yy * yy;
        let radius = radius_squared.sqrt() / self.rebydx;
        let mut longitude = self.stand_lon + DEG * (hemi * xx).atan2(yy) / self.cone;
        longitude = (longitude + 360.0).rem_euclid(360.0);
        let chi = if (chi1 - chi2).abs() < 1.0e-12 {
            2.0 * ((radius / chi1.tan()).powf(1.0 / self.cone) * (chi1 * 0.5).tan()).atan()
        } else {
            2.0 * ((radius * self.cone / chi1.sin()).powf(1.0 / self.cone) * (chi1 * 0.5).tan())
                .atan()
        };
        let mut latitude = (90.0 - chi * DEG) * hemi;
        if radius_squared == 0.0 {
            latitude = hemi * 90.0;
            longitude = self.stand_lon;
        }
        if longitude > 180.0 {
            longitude -= 360.0;
        } else if longitude < -180.0 {
            longitude += 360.0;
        }
        (latitude, longitude)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::Digest as _;

    /// A mesh as this derivation sees one: `(latitude, longitude, spacing)`
    /// per cell, a refined patch inside a coarse background. Nothing else is
    /// read, so a synthetic mesh built this way exercises exactly the code a
    /// real one does.
    ///
    /// `west > east` means the patch crosses the antimeridian and is walked
    /// through it.
    fn patch_in_background(
        south: f64,
        north: f64,
        west: f64,
        east: f64,
        core_spacing_m: f64,
        background_spacing_m: f64,
    ) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let mut lat = Vec::new();
        let mut lon = Vec::new();
        let mut spacing = Vec::new();
        let mut la = -85.0;
        while la <= 85.0 {
            let mut lo = -180.0;
            while lo < 180.0 {
                lat.push(la);
                lon.push(lo);
                spacing.push(background_spacing_m);
                lo += 5.0;
            }
            la += 5.0;
        }
        let down = ((north - south) / 0.2).round() as usize;
        let width = if east >= west { east - west } else { east + 360.0 - west };
        let across = (width / 0.2).round() as usize;
        for j in 0..=down {
            for i in 0..=across {
                lat.push(south + j as f64 * 0.2);
                lon.push(((west + i as f64 * 0.2) + 180.0).rem_euclid(360.0) - 180.0);
                spacing.push(core_spacing_m);
            }
        }
        (lat, lon, spacing)
    }

    /// Great-circle distance from a point to the nearest point of a window's
    /// grid, in km. This is what "covers" has to mean: a bounding box in
    /// latitude and longitude says nothing about a Lambert grid, whose edges
    /// are curved and whose corners are the part that gets cropped.
    fn nearest_grid_point_km(grid: &(Vec<f64>, Vec<f64>), lat: f64, lon: f64) -> f64 {
        let t = unit_vector(lat, lon);
        let mut best = f64::INFINITY;
        for k in 0..grid.0.len() {
            let v = unit_vector(grid.0[k], grid.1[k]);
            let chord = ((v[0] - t[0]).powi(2) + (v[1] - t[1]).powi(2) + (v[2] - t[2]).powi(2))
                .sqrt();
            let km = 2.0 * (chord / 2.0).clamp(0.0, 1.0).asin() * MPAS_EARTH_RADIUS_M / 1000.0;
            if km < best {
                best = km;
            }
        }
        best
    }

    fn coordinate_digest(window: &Window) -> String {
        let (lat, lon) = window.coordinates().unwrap();
        let mut hash = sha2::Sha256::new();
        for v in lat.iter().chain(lon.iter()) {
            hash.update(v.to_le_bytes());
        }
        crate::weights::hex(&hash.finalize())
    }

    /// The platform key a coordinate digest belongs to.
    ///
    /// A digest over `f64` bytes belongs to the C library that produced them,
    /// not to this source. `f64::atan` and `f64::powf` compile to the
    /// platform's libm, IEEE-754 does not require either to be correctly
    /// rounded, and the two implementations this program runs on disagree —
    /// so the key is the (OS, C environment) pair, which is what selects the
    /// libm.
    fn libm_platform_key() -> String {
        let c_env = if cfg!(target_env = "msvc") {
            "msvc"
        } else if cfg!(target_env = "gnu") {
            "gnu"
        } else if cfg!(target_env = "musl") {
            "musl"
        } else {
            "unknown-c-env"
        };
        format!("{}-{c_env}", std::env::consts::OS)
    }

    /// The `focus` window's coordinate digest, PER PLATFORM — measured, not
    /// assumed.
    ///
    /// MEASURED 2026-08-27 (#373), same source, same optimisation level, and
    /// the compiler EXCLUDED as a cause by direct control: rustc 1.94.0 and
    /// rustc 1.97.0 on the SAME Linux box produce the same digest, `-C
    /// opt-level=0` and `-O` on the same Windows box produce the same digest,
    /// and a dependency-free reproduction of `set_lc`/`ijll` reproduces BOTH
    /// rows exactly. What differs is the library call:
    ///
    /// * every value feeding `powf` is bit-identical across the two boxes
    ///   (0 of 36,000 differ), so nothing upstream of the library call moved;
    /// * `powf` returns a different `f64` for 32 of those 36,000 arguments;
    /// * `atan` returns a different `f64` for 1,820 of 36,000 arguments;
    /// * 1,684 of the 36,000 latitudes therefore differ, by at most 4 ULP —
    ///   a largest absolute disagreement of 1.42e-14 degrees, which is
    ///   **1.6 nanometres** on the ground.
    ///
    /// `global` needs no such table: its coordinates come from `axis()`,
    /// which is `first + k * step` and calls no transcendental at all. That
    /// is why it is one value everywhere, and it is asserted as one below.
    ///
    /// ADDING A ROW IS ADDITIVE. A new platform gets its own row measured on
    /// that platform; no existing row is ever edited to make a box agree,
    /// because an edited row silently retires the evidence it was minted for.
    const FROZEN_FOCUS_COORDINATE_DIGESTS: &[(&str, &str)] = &[
        (
            // Windows 11, MSVC CRT. rustc 1.94.0, `cargo test -p rw-mpas`.
            "windows-msvc",
            "34e3521f9778c1a7f0ce283f6ba1ae24453d5c9a16ce0d12c221bb56765e08bc",
        ),
        (
            // weather-node-2, Ubuntu, glibc 2.43. Reproduced under rustc
            // 1.97.0 and rustc 1.94.0, and by the standalone probe.
            "linux-gnu",
            "551a1ff12abd2af6ec611432f3a795369b4fa4a80b46bee08dee0c286c9715b2",
        ),
    ];

    /// The `focus` grid's corner and centre coordinates, in degrees, to a
    /// tolerance that no libm can reach and no real spec change can hide in.
    ///
    /// THE BREAKAGE THIS PREVENTS is the one the digest was always for: a
    /// corner nudged by a metre invalidates every recorded `weights_sha256`
    /// while every filename stays the same. This is the half of that gate
    /// which holds on EVERY platform, including one nobody has measured yet.
    ///
    /// The tolerance is chosen from both sides, measured: the largest
    /// cross-libm disagreement in these coordinates is 1.42e-14 degrees, and
    /// one metre of ground is 9.0e-6 degrees. `1e-9` degrees is 0.11 mm — five
    /// orders of magnitude above the noise this cannot be allowed to trip on,
    /// and four orders below the smallest movement it must catch.
    const FOCUS_GEOMETRY_TOLERANCE_DEG: f64 = 1.0e-9;
    const FROZEN_FOCUS_GEOMETRY: &[(&str, usize, f64, f64)] = &[
        ("first point (j=1, i=1)", 0, 18.955_624_884_995_8, -119.806_558_555_523_42),
        ("last point (j=ny, i=nx)", 35_999, 46.496_632_054_296_91, -59.143_091_166_431_93),
    ];

    /// The two fixed windows may not move, ever. Their `weights_sha256`
    /// values are recorded in evidence already written, and the digest is
    /// taken over the window spec and the nearest-cell index map, so a corner
    /// nudged by a metre invalidates every recorded digest while every
    /// filename stays the same. This pins the spec string the digest hashes
    /// AND the coordinates the index map is built from.
    ///
    /// #373: the `focus` half of that pin is PER PLATFORM and now says so.
    /// See `FROZEN_FOCUS_COORDINATE_DIGESTS` for the measurement. The spec
    /// strings, the shape and the geometry gate are portable and are still
    /// asserted identically everywhere — those are what actually catch a
    /// moved corner. The exact digest is what catches a moved BIT, and a bit
    /// belongs to a libm.
    #[test]
    fn the_two_fixed_windows_are_frozen() {
        let focus = Window::named("focus").unwrap();
        assert_eq!(
            focus.spec_json(),
            "{\"centre_lat\": 37.0, \"centre_lon\": -96.0, \"dx_metres\": 22000.0, \
             \"earth_radius_m\": 6370000.0, \"kind\": \"lambert\", \"nx\": 240, \"ny\": 150, \
             \"stand_lon\": -96.0, \"truelat1\": 30.0, \"truelat2\": 60.0}"
        );
        assert_eq!(
            focus.spec_json_compact(),
            "{\"centre_lat\":37.0,\"centre_lon\":-96.0,\"dx_metres\":22000.0,\
             \"earth_radius_m\":6370000.0,\"kind\":\"lambert\",\"nx\":240,\"ny\":150,\
             \"stand_lon\":-96.0,\"truelat1\":30.0,\"truelat2\":60.0}"
        );
        assert_eq!(focus.shape().unwrap(), (150, 240));

        // The portable half, asserted on every platform including an
        // unmeasured one: the grid is still where it was.
        let (lat, lon) = focus.coordinates().unwrap();
        assert_eq!(lat.len(), 36_000, "the focus grid changed size");
        for (label, index, want_lat, want_lon) in FROZEN_FOCUS_GEOMETRY {
            let dlat = (lat[*index] - want_lat).abs();
            let dlon = (lon[*index] - want_lon).abs();
            assert!(
                dlat < FOCUS_GEOMETRY_TOLERANCE_DEG && dlon < FOCUS_GEOMETRY_TOLERANCE_DEG,
                "the focus window's {label} MOVED: got ({}, {}), recorded ({want_lat}, \
                 {want_lon}), off by ({dlat:.3e}, {dlon:.3e}) degrees against a tolerance of \
                 {FOCUS_GEOMETRY_TOLERANCE_DEG:.0e}. That tolerance is 0.11 mm on the ground \
                 and the largest cross-libm disagreement ever measured here is 1.42e-14 \
                 degrees, so this is not floating-point noise -- the projection or the spec \
                 changed. Every `weights_sha256` recorded in evidence was taken over this \
                 grid and none of them reproduces now, while every filename stays the same.",
                lat[*index],
                lon[*index],
            );
        }

        // The exact half, per platform.
        let key = libm_platform_key();
        let digest = coordinate_digest(&focus);
        match FROZEN_FOCUS_COORDINATE_DIGESTS.iter().find(|(k, _)| *k == key) {
            Some((_, want)) => assert_eq!(
                &digest,
                want,
                "the focus window's coordinate digest MOVED on a platform that has a \
                 measured row ({key}). The geometry gate above passed, so the grid is \
                 where it was to within 0.11 mm and this is a change of BITS, not of \
                 position. Either the Lambert transcription changed, or this box's libm \
                 is not the one the {key} row was measured on. Do NOT edit the row to \
                 make this pass: every `weights_sha256` recorded against the {key} row \
                 was taken over these exact bytes, and editing the row retires that \
                 evidence without saying so. Measure which of `powf`/`atan` moved first."
            ),
            None => panic!(
                "no measured coordinate-digest row for platform {key}, so this box \
                 cannot claim the focus window's byte anchor. This is NOT a failure of \
                 the code: the geometry gate above PASSED, so the grid is correct to \
                 within 0.11 mm. What is missing is a measurement. `f64::atan` and \
                 `f64::powf` compile to the platform's libm, neither is required to be \
                 correctly rounded, and the two libms already measured here disagree \
                 for 1,820 and 32 of 36,000 arguments respectively. The remedy is to \
                 ADD a row to FROZEN_FOCUS_COORDINATE_DIGESTS reading \
                 (\"{key}\", \"{digest}\") -- additive, never editing an existing row -- \
                 and to record in the receipt which OS, C library and libm version \
                 produced it. Refusing here rather than passing is deliberate: an \
                 unmeasured platform has no anchor, and a silent pass would read as one."
            ),
        }

        let global = Window::named("global").unwrap();
        assert_eq!(
            global.spec_json(),
            "{\"east\": 179.75, \"kind\": \"latlon\", \"north\": 90.0, \"south\": -90.0, \
             \"spacing_degrees\": 0.25, \"west\": -180.0}"
        );
        assert_eq!(
            global.spec_json_compact(),
            "{\"east\":179.75,\"kind\":\"latlon\",\"north\":90.0,\"south\":-90.0,\
             \"spacing_degrees\":0.25,\"west\":-180.0}"
        );
        assert_eq!(global.shape().unwrap(), (721, 1440));
        // ONE value, on every platform, and that is a measured fact rather
        // than an assumption: `global` is a lat-lon window, its coordinates
        // come from `axis()` -- `first + k as f64 * step` -- and no
        // transcendental is called anywhere in that path. Measured 2026-08-27
        // on Windows/MSVC and on glibc 2.43: identical. This is the control
        // that proves the `focus` split above is the libm and not the source.
        assert_eq!(
            coordinate_digest(&global),
            "ce6e916e0c01874f7a21296232706eb9855d9422be3e255fc9201352efb5a4d9",
            "the global window's coordinate digest moved. Unlike `focus` this one \
             is platform-independent by construction -- pure `first + k * step`, no \
             libm call -- so a move here is a real change to the window spec or to \
             `axis()`, and every recorded `weights_sha256` over the global grid \
             stops reproducing while every filename stays the same."
        );
    }

    /// The geometry gate, validated in the direction that matters: it must
    /// SEE a corner nudged by a metre.
    ///
    /// A tolerance chosen to sit above floating-point noise is worthless
    /// unless someone proves it still sits below the breakage. This nudges
    /// `focus`'s centre by one metre of latitude (9.0e-6 degrees) — the
    /// movement the frozen test's own docstring names as fatal to every
    /// recorded `weights_sha256` — and asserts the gate's arithmetic rejects
    /// it by a wide margin. Without this test, `FOCUS_GEOMETRY_TOLERANCE_DEG`
    /// could be loosened to 1.0 and nothing would notice.
    #[test]
    fn the_geometry_gate_sees_a_corner_nudged_by_one_metre() {
        const ONE_METRE_OF_LATITUDE_DEG: f64 = 9.0e-6;
        let Window::Lambert(mut spec) = Window::named("focus").unwrap() else {
            unreachable!("focus is a Lambert window");
        };
        spec.centre_lat += ONE_METRE_OF_LATITUDE_DEG;
        let (lat, lon) = Window::Lambert(spec).coordinates().unwrap();

        let mut caught = 0usize;
        for (label, index, want_lat, want_lon) in FROZEN_FOCUS_GEOMETRY {
            let dlat = (lat[*index] - want_lat).abs();
            let dlon = (lon[*index] - want_lon).abs();
            if dlat >= FOCUS_GEOMETRY_TOLERANCE_DEG || dlon >= FOCUS_GEOMETRY_TOLERANCE_DEG {
                caught += 1;
            }
            assert!(
                dlat < 1.0e-3,
                "{label} moved {dlat:.3e} degrees for a one-metre nudge, which is far more \
                 than a metre -- the nudge is not doing what this control claims"
            );
        }
        assert_eq!(
            caught,
            FROZEN_FOCUS_GEOMETRY.len(),
            "the geometry gate did NOT see a one-metre nudge at every pinned point. \
             Its tolerance {FOCUS_GEOMETRY_TOLERANCE_DEG:.0e} degrees is too loose to \
             catch the breakage the frozen window exists to prevent, and the \
             per-platform digest table is then the only thing holding that grid in \
             place -- which leaves an unmeasured platform with no gate at all."
        );
    }

    /// The measurement behind the per-platform `focus` row, kept executable.
    ///
    /// #373 asked whether a frozen digest that disagrees across two boxes is
    /// the compiler or the C library. This holds the answer that settled it:
    /// the `global` window, whose coordinates call no libm function at all,
    /// must digest identically on every platform. If this ever fails while the
    /// `focus` row also fails, the cause is NOT the libm and the per-platform
    /// table above is the wrong explanation -- look at `axis()` or at the
    /// spec. Keeping both in one file is the point: one is the treatment, the
    /// other is the control.
    #[test]
    fn the_transcendental_free_window_needs_no_platform_row() {
        let global = Window::named("global").unwrap();
        let (lat, lon) = global.coordinates().unwrap();
        assert_eq!(lat.len(), 721 * 1440);
        // Every coordinate is exactly representable as `first + k * 0.25`,
        // which is what makes this digest portable. Assert the property, not
        // just the digest, so the reason survives a refactor of `axis()`.
        for (k, v) in lat.iter().take(1440 * 8).enumerate() {
            let row = (k / 1440) as f64;
            assert_eq!(
                *v,
                -90.0 + row * 0.25,
                "global latitude {k} is not exactly -90 + row*0.25; the moment this \
                 path acquires arithmetic that is not a multiply and an add, its \
                 digest becomes platform-specific and needs a table like `focus`'s"
            );
        }
        assert!(lon.iter().all(|v| v.is_finite()));
    }

    /// The measured defect, as a test: neither fixed window can render a
    /// refined core in the tropical Atlantic. `focus` puts no grid point
    /// anywhere near it, and `global` is 6x too coarse to hold it. This is
    /// the breakage the derived window prevents, so it is asserted rather
    /// than asserted about.
    #[test]
    fn neither_fixed_window_can_render_a_tropical_atlantic_core() {
        let focus = Window::named("focus").unwrap().coordinates().unwrap();
        let inside = (0..focus.0.len())
            .filter(|&k| {
                (14.6..=20.0).contains(&focus.0[k]) && (-57.3..=-50.5).contains(&focus.1[k])
            })
            .count();
        assert_eq!(
            inside, 0,
            "the CONUS focus box must contain no part of a 15-20 N, 50-57 W core"
        );

        let global = match Window::named("global").unwrap() {
            Window::LatLon(w) => w.spacing_degrees * RAD * WRF_EARTH_RADIUS_M,
            Window::Lambert(_) => unreachable!("global is a lat-lon window"),
        };
        assert!(
            global / 4_530.0 > 6.0,
            "the global overview is {global:.0} m, only {:.1}x a 4.53 km core",
            global / 4_530.0
        );
    }

    /// The case that forced this: a refined core at roughly 14.6-20.0 N,
    /// 50.5-57.3 W attaining 4.53 km inside a 75 km background.
    #[test]
    fn a_tropical_atlantic_core_derives_a_window_over_itself() {
        let (lat, lon, spacing) =
            patch_in_background(14.6, 20.0, -57.3, -50.5, 4_530.0, 75_000.0);
        let window = Window::mesh_focus(&lat, &lon, &spacing).unwrap();
        let w = match &window {
            Window::Lambert(w) => w.clone(),
            Window::LatLon(_) => panic!("17 N is well inside the Lambert branch"),
        };
        assert!(
            (w.centre_lat - 17.310).abs() < 0.01,
            "centre latitude {} is not the core's",
            w.centre_lat
        );
        assert!(
            (w.centre_lon + 53.900).abs() < 0.01,
            "centre longitude {} is not the core's",
            w.centre_lon
        );
        // The median of a uniform core is the core's own spacing, exactly.
        assert_eq!(w.dx_metres, 4_530.0);
        assert_eq!((w.nx, w.ny), (179, 148));
        // Truelats bracket the centre, both in the northern hemisphere.
        assert!((w.truelat1 - (w.centre_lat - 5.0)).abs() < 1.0e-9);
        assert!((w.truelat2 - (w.centre_lat + 5.0)).abs() < 1.0e-9);
        assert!(w.truelat1 > 0.0 && w.stand_lon == w.centre_lon);

        // Coverage, corner by corner: every corner of the core has a grid
        // point within half a cell of it. A window that merely brackets the
        // core in latitude and longitude would pass a weaker test and still
        // crop the corners.
        let grid = window.coordinates().unwrap();
        for (clat, clon) in [(14.6, -57.3), (14.6, -50.5), (20.0, -57.3), (20.0, -50.5)] {
            let km = nearest_grid_point_km(&grid, clat, clon);
            assert!(
                km <= 4.53,
                "core corner {clat} {clon} is {km:.3} km from the nearest grid point, \
                 further than one cell"
            );
        }
        assert!(window.description().contains("2.00x the mesh minimum 4.530 km"));
    }

    /// A naive mean of longitudes puts a core straddling the antimeridian at
    /// roughly the prime meridian -- half a planet from where it is. The
    /// unit-vector mean puts it on the antimeridian, which is where it is.
    #[test]
    fn a_core_across_the_antimeridian_stays_on_the_antimeridian() {
        let (lat, lon, spacing) = patch_in_background(30.0, 40.0, 175.0, -175.0, 3_000.0, 60_000.0);
        let window = Window::mesh_focus(&lat, &lon, &spacing).unwrap();
        let w = match &window {
            Window::Lambert(w) => w.clone(),
            Window::LatLon(_) => panic!("35 N is well inside the Lambert branch"),
        };
        assert!(
            w.centre_lon.abs() > 179.9,
            "centre longitude {} is not on the antimeridian; an arithmetic mean of these \
             longitudes lands near 0 and that is the bug this guards",
            w.centre_lon
        );
        assert!((w.centre_lat - 35.0).abs() < 0.1, "{}", w.centre_lat);

        // And the grid genuinely straddles the line rather than sitting on
        // one side of it.
        let grid = window.coordinates().unwrap();
        assert!(grid.1.iter().any(|&v| v > 176.0), "no grid point east of 176 E");
        assert!(grid.1.iter().any(|&v| v < -176.0), "no grid point west of 176 W");
        for (clat, clon) in [(30.0, 175.0), (30.0, -175.0), (40.0, 175.0), (40.0, -175.0)] {
            assert!(nearest_grid_point_km(&grid, clat, clon) <= 3.0, "{clat} {clon}");
        }
    }

    /// A core on the equator has no Lambert cone to sit on, so it gets the
    /// cylindrical grid instead -- at its own spacing, not the 0.25 degree
    /// overview's.
    #[test]
    fn a_near_equatorial_core_falls_back_to_lat_lon() {
        let (lat, lon, spacing) = patch_in_background(-2.0, 2.0, -20.0, -14.0, 3_000.0, 60_000.0);
        let window = Window::mesh_focus(&lat, &lon, &spacing).unwrap();
        let w = match &window {
            Window::LatLon(w) => w.clone(),
            Window::Lambert(_) => panic!("a centre on the equator has no conditioned cone"),
        };
        // The derived spacing is the core's own 3 km, in degrees.
        let metres = w.spacing_degrees * RAD * WRF_EARTH_RADIUS_M;
        assert!((metres - 3_000.0).abs() < 1.0, "{metres} m is not the core's 3 km");
        assert!(w.south < -2.0 && w.north > 2.0, "{} to {}", w.south, w.north);
        assert!(w.west < -20.0 && w.east > -14.0, "{} to {}", w.west, w.east);
        assert!(window.description().contains("under the 10 a Lambert cone needs"));
        // The axes must still partition exactly, or `coordinates` refuses.
        let grid = window.coordinates().unwrap();
        for (clat, clon) in [(-2.0, -20.0), (-2.0, -14.0), (2.0, -20.0), (2.0, -14.0)] {
            assert!(nearest_grid_point_km(&grid, clat, clon) <= 3.0, "{clat} {clon}");
        }
    }

    /// Just far enough from the equator to keep the cone, and the fallback
    /// must not fire: the branch is a threshold, not a wide band.
    #[test]
    fn the_lambert_branch_holds_at_its_own_threshold() {
        let below = patch_in_background(8.0, 10.0, -20.0, -18.0, 3_000.0, 60_000.0);
        assert!(
            matches!(
                Window::mesh_focus(&below.0, &below.1, &below.2).unwrap(),
                Window::LatLon(_)
            ),
            "a centre at 9 N is under the threshold and must be cylindrical"
        );
        let above = patch_in_background(11.0, 13.0, -20.0, -18.0, 3_000.0, 60_000.0);
        let window = Window::mesh_focus(&above.0, &above.1, &above.2).unwrap();
        let w = match &window {
            Window::Lambert(w) => w.clone(),
            Window::LatLon(_) => panic!("a centre at 12 N is over the threshold"),
        };
        assert!(w.centre_lat >= MESH_FOCUS_MIN_LAMBERT_LATITUDE_DEG);
        let grid = window.coordinates().unwrap();
        assert!(grid.0.iter().chain(grid.1.iter()).all(|v| v.is_finite()));
    }

    /// The measurement behind [`MESH_FOCUS_MIN_LAMBERT_LATITUDE_DEG`]: on the
    /// equator the cone is zero, the projection origin is infinite and every
    /// coordinate is NaN, while at the threshold the same transcription is
    /// exact. Both halves are asserted, because a threshold that has only
    /// been checked on the good side has not been checked.
    #[test]
    fn the_cone_is_why_a_near_equatorial_centre_falls_back() {
        // On the equator: 0/0.
        let cone = lambert_cone(0.0, 0.0);
        assert_eq!(cone, 0.0, "a cone of zero is what a 1/cone divides by");
        let dead = LambertProjection::set_lc(0.0, 0.0, 0.0, 4_530.0, 0.0, 0.0, 90.0, 74.0);
        assert!(!dead.polej.is_finite(), "polej is {}", dead.polej);
        let (bad_lat, bad_lon) = dead.ijll(90.0, 74.0);
        assert!(
            bad_lat.is_nan() && bad_lon.is_nan(),
            "an equatorial cone must be visibly dead, not quietly wrong: got {bad_lat} {bad_lon}"
        );

        // At the threshold: cone 0.174, origin 8,008 grid units off, and the
        // centre round-trips to well under a millimetre.
        let centre = MESH_FOCUS_MIN_LAMBERT_LATITUDE_DEG;
        let (t1, t2) = (centre - 5.0, centre + 5.0);
        let cone = lambert_cone(t1, t2);
        assert!((cone - 0.1739).abs() < 1.0e-3, "cone is {cone}");
        let live = LambertProjection::set_lc(t1, t2, -53.9, 4_530.0, centre, -53.9, 90.0, 74.0);
        assert!((live.polej - 8_008.4).abs() < 1.0, "polej is {}", live.polej);
        let (lat, lon) = live.ijll(90.0, 74.0);
        let error_m = (lat - centre).abs() * RAD * WRF_EARTH_RADIUS_M;
        assert!(error_m < 1.0e-3, "centre round-trip is off by {error_m:e} m");
        assert!((lon + 53.9).abs() < 1.0e-9, "{lon}");
    }

    /// The forward transform the sizing depends on has to be the inverse of
    /// the one the coordinates come from, or the window is sized on one map
    /// and rendered on another.
    #[test]
    fn the_lambert_transforms_invert_each_other() {
        let projection =
            LambertProjection::set_lc(30.0, 60.0, -96.0, 22_000.0, 37.0, -96.0, 120.5, 75.5);
        for (i, j) in [(1.0, 1.0), (120.5, 75.5), (240.0, 150.0), (17.0, 133.0)] {
            let (lat, lon) = projection.ijll(i, j);
            let (bi, bj) = projection.llij(lat, lon);
            assert!(
                (bi - i).abs() < 1.0e-9 && (bj - j).abs() < 1.0e-9,
                "({i}, {j}) went to {lat} {lon} and came back ({bi}, {bj})"
            );
        }
    }

    /// A pathological mesh must not be able to ask for a target grid that
    /// cannot be written. The clamp coarsens dx; it does not crop, and the
    /// region it was derived from is still fully inside the result.
    #[test]
    fn the_clamp_coarsens_dx_and_still_covers_the_region() {
        let (lat, lon, spacing) =
            patch_in_background(20.0, 45.0, -110.0, -70.0, 1_000.0, 60_000.0);
        let window = Window::mesh_focus(&lat, &lon, &spacing).unwrap();
        let w = match &window {
            Window::Lambert(w) => w.clone(),
            Window::LatLon(_) => panic!("33 N is well inside the Lambert branch"),
        };
        assert!(w.nx <= MESH_FOCUS_MAX_POINTS_PER_SIDE, "nx is {}", w.nx);
        assert!(w.ny <= MESH_FOCUS_MAX_POINTS_PER_SIDE, "ny is {}", w.ny);
        assert_eq!(w.nx, MESH_FOCUS_MAX_POINTS_PER_SIDE, "the clamp must be the binding side");
        assert!(
            w.dx_metres > 1_000.0,
            "dx {} was not coarsened, so the clamp cropped instead",
            w.dx_metres
        );
        assert!(window.description().contains("coarsened from 1.000 km"));

        // Coverage survives the clamp: every corner of the region still has a
        // grid point within a cell of it.
        let grid = window.coordinates().unwrap();
        for (clat, clon) in [(20.0, -110.0), (20.0, -70.0), (45.0, -110.0), (45.0, -70.0)] {
            let km = nearest_grid_point_km(&grid, clat, clon);
            assert!(
                km <= w.dx_metres / 1000.0,
                "corner {clat} {clon} is {km:.3} km out after the clamp"
            );
        }
    }

    /// A uniform mesh has no refined region, and the answer is a refusal that
    /// says so and says what to run instead.
    #[test]
    fn a_uniform_mesh_is_refused_by_name() {
        let lat = vec![10.0, 20.0, 30.0];
        let lon = vec![0.0, 10.0, 20.0];
        let error = Window::mesh_focus(&lat, &lon, &[60_000.0; 3])
            .unwrap_err()
            .to_string();
        assert!(error.contains("uniform"), "{error}");
        assert!(error.contains("no refined region"), "{error}");
        assert!(error.contains("--window global"), "{error}");
        assert!(error.contains("60.000 km"), "{error}");
    }

    /// The refinement factor is a closed boundary: a mesh exactly at the
    /// factor is uniform, a hair past it is not. A boundary nobody pinned is
    /// a boundary that moves.
    #[test]
    fn the_refinement_factor_boundary_is_closed() {
        let lat = vec![10.0, 20.0, 30.0];
        let lon = vec![0.0, 10.0, 20.0];
        assert!(
            Window::mesh_focus(&lat, &lon, &[30_000.0, 60_000.0, 45_000.0]).is_err(),
            "a 2.000x ratio is uniform to within the factor"
        );
        let window =
            Window::mesh_focus(&lat, &lon, &[30_000.0, 60_001.0, 45_000.0]).unwrap();
        assert!(window.description().contains("2 of 3 cells"), "{}", window.description());
    }

    /// Every degenerate input names what it broke.
    #[test]
    fn degenerate_inputs_are_refused_by_name() {
        let empty = Window::mesh_focus(&[], &[], &[]).unwrap_err().to_string();
        assert!(empty.contains("at least one cell"), "{empty}");

        let ragged = Window::mesh_focus(&[1.0, 2.0], &[1.0], &[1.0, 2.0])
            .unwrap_err()
            .to_string();
        assert!(ragged.contains("one latitude, longitude and spacing per cell"), "{ragged}");

        let nan = Window::mesh_focus(&[1.0, f64::NAN], &[1.0, 2.0], &[1.0, 2.0])
            .unwrap_err()
            .to_string();
        assert!(nan.contains("not finite"), "{nan}");

        let off_globe = Window::mesh_focus(&[1.0, 91.0], &[1.0, 2.0], &[1.0, 2.0])
            .unwrap_err()
            .to_string();
        assert!(off_globe.contains("outside [-90, 90]"), "{off_globe}");

        let zero = Window::mesh_focus(&[1.0, 2.0], &[1.0, 2.0], &[0.0, 2.0])
            .unwrap_err()
            .to_string();
        assert!(zero.contains("positive and finite"), "{zero}");

        // Two refined cells on opposite sides of the planet average to the
        // origin, which points nowhere.
        let antipodal = Window::mesh_focus(
            &[45.0, -45.0, 0.0],
            &[0.0, 180.0, 90.0],
            &[2_000.0, 2_000.0, 60_000.0],
        )
        .unwrap_err()
        .to_string();
        assert!(antipodal.contains("mean direction of magnitude"), "{antipodal}");
        assert!(antipodal.contains("no middle"), "{antipodal}");
    }

    /// A refined region wider than a quarter turn has passed the horizon of
    /// any projection a window can be cut on.
    #[test]
    fn a_refined_region_past_the_horizon_is_refused() {
        let error = Window::mesh_focus(
            &[0.0, 0.0, 0.0, 40.0],
            &[0.0, 95.0, -95.0, 10.0],
            &[2_000.0, 2_000.0, 2_000.0, 60_000.0],
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("degrees of arc from its own centre"), "{error}");
        assert!(error.contains("--window global"), "{error}");
    }

    /// A one-cell refined region has no extent of its own. It gets the
    /// smallest window `coordinates` will build rather than a 2x2 with no
    /// centre point.
    #[test]
    fn a_single_refined_cell_still_makes_a_window() {
        let window = Window::mesh_focus(
            &[45.0, 10.0, -10.0],
            &[8.0, 100.0, -100.0],
            &[2_000.0, 60_000.0, 60_000.0],
        )
        .unwrap();
        assert_eq!(window.shape().unwrap(), (3, 3));
        let grid = window.coordinates().unwrap();
        assert!(nearest_grid_point_km(&grid, 45.0, 8.0) < 0.001);
    }

    /// A southern core keeps both truelats in the south, because `set_lc`
    /// takes its hemisphere from the sign of `truelat1` and a truelat over
    /// the equator would put the window in the wrong one.
    #[test]
    fn a_southern_core_keeps_its_truelats_in_the_south() {
        let (lat, lon, spacing) = patch_in_background(-40.0, -30.0, 140.0, 152.0, 4_000.0, 60_000.0);
        let window = Window::mesh_focus(&lat, &lon, &spacing).unwrap();
        let w = match &window {
            Window::Lambert(w) => w.clone(),
            Window::LatLon(_) => panic!("35 S is well inside the Lambert branch"),
        };
        assert!(w.centre_lat < 0.0, "{}", w.centre_lat);
        assert!(w.truelat1 < 0.0 && w.truelat2 < 0.0, "{} {}", w.truelat1, w.truelat2);
        let grid = window.coordinates().unwrap();
        assert!(grid.0.iter().all(|&v| v < 0.0), "a southern window drifted north");
        for (clat, clon) in [(-40.0, 140.0), (-40.0, 152.0), (-30.0, 140.0), (-30.0, 152.0)] {
            assert!(nearest_grid_point_km(&grid, clat, clon) <= 4.0, "{clat} {clon}");
        }
    }

    /// `mesh` is not a catalogue entry, and asking the catalogue for it says
    /// why rather than handing back a box.
    #[test]
    fn the_catalogue_refuses_the_derived_name_and_says_why() {
        let error = Window::named("mesh").unwrap_err().to_string();
        assert!(error.contains("derived from the mesh being converted"), "{error}");
        assert!(error.contains("mesh_focus"), "{error}");
        let unknown = Window::named("atlantic").unwrap_err().to_string();
        assert!(unknown.contains("focus, global, mesh"), "{unknown}");
    }
}
