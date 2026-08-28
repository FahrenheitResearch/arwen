//! The case-7 initial-state builder.
//!
//! Reads a WPS intermediate file, a mesh, a statics capsule and a
//! vertical-metric capsule, and writes an MPAS initial-conditions file: the
//! whole of `init_atm_case_gfs`'s meteorology plus
//! `mpas_atmphys_initialize_real.F`'s soil and surface companions, in Rust.
//!
//! The vertical grid itself is **not** built here — see
//! [`capsule`] for that deferral and the bit-identity assertion that keeps it
//! honest.
//!
//! Indexed loops are used throughout rather than iterator chains: every loop
//! here walks several parallel arrays at once against a Fortran original whose
//! index arithmetic is the thing being reproduced, and rewriting them as zips
//! would make the correspondence unreadable at exactly the places it matters.
#![allow(clippy::needless_range_loop)]

pub mod capsule;
pub mod dynamics;
pub mod emit;
pub mod hinterp;
pub mod metfile;
pub mod soil;
pub mod vinterp;

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use rayon::prelude::*;

use crate::error::{MpasError, MpasResult};
use dynamics::VirtualFactor;
use hinterp::{InterpContext, LatLonProjection, Method, Slab, Underflow};
use metfile::MetSlab;
use soil::DeepMoisture;
use vinterp::{sorted_column, vertical_interp, Extrap, SURFACE_LEVEL_TAG, SURFACE_SENTINEL};

/// Radians to degrees, derived the way `mpas_init_atm_llxy.F` derives it.
///
/// This is **not** the correctly rounded `180/pi`.  The Fortran declares
/// `REAL(KIND=RKIND), PARAMETER :: PI = 3.141592653589793` and then
/// `DEG_PER_RAD = 180./PI`, so in the single-precision build the division is
/// performed on `f32(pi)` and the result lands one ULP *below* the correctly
/// rounded constant: `0x42652ee0` (57.295776367) rather than `0x42652ee1`
/// (57.295780182).
///
/// One ULP here is not cosmetic.  It is the constant that turns a mesh cell's
/// latitude and longitude into a position on the first-guess grid, so a 6.7e-8
/// relative difference displaces every interpolation point by up to 1.22e-4 of
/// a grid cell at longitudes near 360 degrees.  Over most of the mesh that is
/// a sub-ULP change in the interpolated value; at the roughly twenty cells per
/// global mesh whose position lands *within that band of a grid line*, it
/// changes `floor(x)` and therefore selects an entirely different four- or
/// sixteen-point stencil, which moves the value by as much as the field varies
/// between neighbouring source points.
///
/// Writing the constant as a decimal literal picked the accurate value and
/// produced exactly that: a systematic -6.6e-5 grid-cell shift plus a handful
/// of cells sampling the wrong stencil.  It is derived rather than typed here
/// so the port stands where the reference stands.  See
/// `deg_per_rad_is_the_fortran_parameter_not_the_accurate_one`.
pub(crate) const DEG_PER_RAD: f32 = 180.0f32 / std::f32::consts::PI;
const MSGVAL: f32 = -1.0e30;

/// Which first-guess array a met field feeds, and how it is sampled.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Target {
    /// A cell-indexed 3-D field, by level.
    T,
    Rh,
    Sh,
    Z,
    P,
    /// An edge-indexed 3-D wind component, by level.
    U,
    V,
    /// Cell-indexed surface scalars.
    Psfc,
    Pmsl,
    SoilHgt,
    Skintemp,
    Snow,
    Seaice,
    /// Soil temperature / moisture at a fixed first-guess layer.
    SoilT(usize, f32),
    SoilM(usize, f32),
}

struct FieldRule {
    target: Target,
    methods: &'static [Method],
    /// `Some((maskval, masked, fillval))` when the land-sea mask applies.
    mask: Option<(f32, i32, f32)>,
}

const PROFILE_METHODS: &[Method] = &[Method::SixteenPoint, Method::Search];
const SOIL_T_METHODS: &[Method] = &[
    Method::SixteenPoint,
    Method::FourPoint,
    Method::WtAverage4,
    Method::Search,
];
const SOIL_M_METHODS: &[Method] = &[Method::FourPoint, Method::WtAverage4, Method::Search];
const SNOW_METHODS: &[Method] = &[Method::FourPoint, Method::WtAverage4];
const SEAICE_METHODS: &[Method] = &[Method::FourPoint, Method::WtAverage4, Method::Search];

/// The dispatch table out of `init_atm_case_gfs`, one row per field name.
///
/// The soil rows carry the layer index and the layer *thickness in
/// centimetres* the Fortran assigns alongside them.  Only the thickness
/// survives: `init_soil_layers_depth` recomputes the mid-depths cumulatively
/// and overwrites whatever the dispatch put in `zs_fg`.
fn rule_for(field: &str) -> Option<FieldRule> {
    let profile = |t: Target| FieldRule {
        target: t,
        methods: PROFILE_METHODS,
        mask: None,
    };
    let soil_t = |k: usize, dz: f32| FieldRule {
        target: Target::SoilT(k, dz),
        methods: SOIL_T_METHODS,
        mask: Some((0.0, 0, 285.0)),
    };
    let soil_m = |k: usize, dz: f32| FieldRule {
        target: Target::SoilM(k, dz),
        methods: SOIL_M_METHODS,
        mask: Some((0.0, 0, 1.0)),
    };
    Some(match field {
        "TT" => profile(Target::T),
        "RH" => profile(Target::Rh),
        "SPECHUMD" => profile(Target::Sh),
        "GHT" => profile(Target::Z),
        "PRES" | "PRESSURE" => profile(Target::P),
        "UU" => profile(Target::U),
        "VV" => profile(Target::V),
        "PSFC" => profile(Target::Psfc),
        "PMSL" => profile(Target::Pmsl),
        "SOILHGT" => profile(Target::SoilHgt),
        "SKINTEMP" => profile(Target::Skintemp),
        "SNOW" => FieldRule {
            target: Target::Snow,
            methods: SNOW_METHODS,
            mask: None,
        },
        "SEAICE" => FieldRule {
            target: Target::Seaice,
            methods: SEAICE_METHODS,
            mask: Some((1.0, 1, 0.0)),
        },
        "ST000010" => soil_t(0, 10.0),
        "ST010200" => soil_t(1, 190.0),
        "ST010040" => soil_t(1, 30.0),
        "ST040100" => soil_t(2, 60.0),
        "ST100200" => soil_t(3, 100.0),
        "ST000007" => soil_t(0, 7.0),
        "ST007028" => soil_t(1, 21.0),
        "ST028100" => soil_t(2, 72.0),
        "ST100255" => soil_t(3, 155.0),
        "ST100289" => soil_t(3, 189.0),
        "SOILT001" => soil_t(0, 1.0),
        "SOILT002" => soil_t(1, 2.0),
        "SOILT006" => soil_t(2, 6.0),
        "SOILT018" => soil_t(3, 18.0),
        "SOILT054" => soil_t(4, 54.0),
        "SOILT162" => soil_t(5, 162.0),
        "SOILT486" => soil_t(6, 486.0),
        "SOILT999" => soil_t(7, 1458.0),
        "SM000010" => soil_m(0, 10.0),
        "SM010200" => soil_m(1, 190.0),
        "SM010040" => soil_m(1, 30.0),
        "SM040100" => soil_m(2, 60.0),
        "SM100200" => soil_m(3, 100.0),
        "SM000007" => soil_m(0, 7.0),
        "SM007028" => soil_m(1, 21.0),
        "SM028100" => soil_m(2, 72.0),
        "SM100255" => soil_m(3, 155.0),
        "SM100289" => soil_m(3, 189.0),
        "SOILM001" => soil_m(0, 1.0),
        "SOILM002" => soil_m(1, 2.0),
        "SOILM006" => soil_m(2, 6.0),
        "SOILM018" => soil_m(3, 18.0),
        "SOILM054" => soil_m(4, 54.0),
        "SOILM162" => soil_m(5, 162.0),
        "SOILM486" => soil_m(6, 486.0),
        "SOILM999" => soil_m(7, 1458.0),
        _ => return None,
    })
}

/// Whether a field's level participates in the first-guess level table.
fn contributes_a_level(target: Target) -> bool {
    matches!(
        target,
        Target::T | Target::Rh | Target::Sh | Target::Z | Target::P | Target::U | Target::V
    )
}

/// Everything the caller must state.  There are no defaults on the switches
/// that select physics: a wrong guess produces a structurally perfect file
/// with wrong numbers, which is exactly the failure mode this project has
/// already been bitten by once on `--map-source`.
#[derive(Debug, Clone)]
pub struct InitConfig {
    pub met_path: PathBuf,
    pub static_path: PathBuf,
    pub capsule_path: PathBuf,
    /// The native init the capsule's `zgrid` is asserted against.
    pub reference_path: PathBuf,
    pub out_path: PathBuf,
    pub start_time: String,
    pub n_fg_levels: usize,
    pub n_fg_soil_levels: usize,
    pub extrap_airtemp: Extrap,
    pub use_spechumd: bool,
    pub theta_adv_order: i32,
    pub coef_3rd_order: f32,
    pub virtual_factor: VirtualFactor,
    pub deep_moisture: DeepMoisture,
    /// `config_landuse_data`, which becomes the `mminlu` label the physics
    /// reads its land-use table by.  No default: the wrong table silently
    /// renumbers every vegetation category.
    pub landuse_table: String,
    /// `config_frac_seaice`.  With it on the sea-ice threshold is 0.02 and the
    /// fraction is kept; with it off the fraction is collapsed to 0/1 at 0.5
    /// first.  No default: the two give different coastlines.
    pub frac_seaice: bool,
    /// `config_tsk_seaice_threshold`, in kelvin.  MPAS's own default is 100 K,
    /// which never fires; a value near freezing converts every cold water
    /// point to land.
    pub tsk_seaice_threshold: f32,
    /// What `oned` does when its guard products underflow.  Not a namelist
    /// option: it is a property of how the reference was *compiled*.  See
    /// [`Underflow`].  No default, because the two answers differ by several
    /// relative-humidity points at the cells where it fires and both files
    /// look perfectly well formed.
    pub oned_underflow: Underflow,
    pub provenance: String,
    /// Who this file says it is: `model_name`, `core_name`, `version` and
    /// `git_version`, the four lineage attributes `rw_mpas_lbc` refuses a
    /// `--grid` for lacking.  See [`Lineage`].
    pub lineage: Lineage,
}

/// The four global attributes that say which model a file belongs to.
///
/// THE BREAKAGE THIS EXISTS TO PREVENT, MEASURED (2026-08-26): this writer
/// re-emitted the capsule's global attributes and added `file_id`,
/// `parent_id` and `gpuwm_provenance`, and nothing else.  A capsule carries
/// the MESH's lineage -- `on_a_sphere`, `sphere_radius`, `mesh_spec` and so
/// on -- but not the MODEL's, so no init this program produced carried
/// `model_name`, `core_name`, `version` or `git_version`.  `rw_mpas_lbc`
/// reads all four off its `--grid` and refuses rather than invent one, so
/// NO init this program minted could drive a boundary stream, regional or
/// global, and the limited-area chain ran through a hand-written patcher
/// instead.
///
/// The producer cannot know which model will integrate the file, so the
/// caller says.  The default is this engine's own name, which is at least
/// true; it is never `mpas`, because a file carrying this port's numerics
/// claiming MPAS's identity is exactly what the consumer's refusal is for.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Lineage {
    pub model_name: String,
    pub core_name: String,
    pub version: String,
    pub git_version: String,
}

impl Default for Lineage {
    fn default() -> Self {
        let version = env!("CARGO_PKG_VERSION").to_string();
        Lineage {
            model_name: "rw-mpas".to_string(),
            core_name: "atmosphere".to_string(),
            git_version: format!("rw-mpas-{version}"),
            version,
        }
    }
}

/// The receipt: counts, not adjectives.
#[derive(Debug, Default, serde::Serialize)]
pub struct InitReceipt {
    pub met_path: String,
    pub static_path: String,
    pub capsule_path: String,
    pub reference_path: String,
    pub out_path: String,
    pub start_time: String,
    pub n_cells: usize,
    pub n_edges: usize,
    pub n_vert_levels: usize,
    pub met_records_read: usize,
    pub met_records_used: usize,
    pub met_records_ignored: Vec<String>,
    pub first_guess_levels_found: usize,
    pub surface_level_present: bool,
    pub zgrid_values_asserted_identical: usize,
    pub extrap_airtemp: String,
    pub use_spechumd: bool,
    pub virtual_factor: String,
    pub deep_moisture_rule: String,
    /// Which side of the reference's `-ftz` this run stands on.
    pub oned_underflow: String,
    /// How many `oned` calls hit a subnormal guard product, i.e. how many
    /// interpolation points the two `oned_underflow` modes can disagree at.
    /// Counted in both modes.  A non-zero count here is the whole reason the
    /// switch exists.
    pub oned_underflow_sites: u64,
    pub hydrostatic_iterations_max: u32,
    pub hydrostatic_iterations_mean: f64,
    pub hydrostatic_cells_hitting_the_cap: usize,
    pub soil: soil::SoilCounts,
    pub seaice: soil::SeaiceCounts,
    pub emit: emit::EmitLedger,
    /// Named loudly so nobody markets a capsule-fed init as a from-scratch one.
    pub deferrals: Vec<String>,
    pub seconds: f64,
}

struct FirstGuess {
    vert_level: Vec<f32>,
    t: Vec<Vec<f32>>,
    rh: Vec<Vec<f32>>,
    sh: Vec<Vec<f32>>,
    z: Vec<Vec<f32>>,
    p: Vec<Vec<f32>>,
    u: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
    psfc: Vec<f32>,
    pmsl: Vec<f32>,
    soilz: Vec<f32>,
    skintemp: Vec<f32>,
    snow: Vec<f32>,
    xice: Vec<f32>,
    u10: Vec<f32>,
    v10: Vec<f32>,
    st: Vec<Vec<f32>>,
    sm: Vec<Vec<f32>>,
    dzs_cm: Vec<f32>,
}

#[allow(clippy::too_many_lines)]
fn horizontal_pass(
    slabs: &[MetSlab],
    mesh: &capsule::MeshGeometry,
    statics: &capsule::Statics,
    cfg: &InitConfig,
    receipt: &mut InitReceipt,
) -> MpasResult<FirstGuess> {
    let n_cells = mesh.n_cells;
    let n_edges = mesh.n_edges;
    let nfg = cfg.n_fg_levels;
    let nfgs = cfg.n_fg_soil_levels;

    let landsea = slabs
        .iter()
        .find(|s| s.field == "LANDSEA")
        .ok_or_else(|| {
            MpasError::Refusal(
                "the intermediate file carries no LANDSEA field.  Case 7 uses it as the \
                 interpolation mask for every soil and sea-ice field; without it the coastal \
                 values would come from whatever the source grid holds over the other surface \
                 type, silently"
                    .to_string(),
            )
        })?;
    let mask_slab = Slab::from_met(&landsea.values, landsea.nx, landsea.ny);
    let underflow_hits = std::sync::atomic::AtomicU64::new(0);

    let mut fg = FirstGuess {
        vert_level: Vec::new(),
        t: vec![vec![0.0; nfg]; n_cells],
        rh: vec![vec![0.0; nfg]; n_cells],
        sh: vec![vec![0.0; nfg]; n_cells],
        z: vec![vec![0.0; nfg]; n_cells],
        p: vec![vec![0.0; nfg]; n_cells],
        u: vec![vec![0.0; nfg]; n_edges],
        v: vec![vec![0.0; nfg]; n_edges],
        psfc: vec![0.0; n_cells],
        pmsl: vec![0.0; n_cells],
        soilz: vec![0.0; n_cells],
        skintemp: vec![0.0; n_cells],
        snow: vec![0.0; n_cells],
        xice: vec![0.0; n_cells],
        u10: vec![0.0; n_cells],
        v10: vec![0.0; n_cells],
        st: vec![vec![0.0; nfgs]; n_cells],
        sm: vec![vec![0.0; nfgs]; n_cells],
        dzs_cm: vec![0.0; nfgs],
    };

    let cell_lat: Vec<f32> = mesh.lat_cell.iter().map(|v| v * DEG_PER_RAD).collect();
    let cell_lon: Vec<f32> = mesh.lon_cell.iter().map(|v| v * DEG_PER_RAD).collect();
    let edge_lat: Vec<f32> = mesh.lat_edge.iter().map(|v| v * DEG_PER_RAD).collect();
    let edge_lon: Vec<f32> = mesh.lon_edge.iter().map(|v| v * DEG_PER_RAD).collect();

    for s in slabs {
        receipt.met_records_read += 1;
        let Some(rule) = rule_for(&s.field) else {
            if s.field != "LANDSEA" && !receipt.met_records_ignored.contains(&s.field) {
                receipt.met_records_ignored.push(s.field.clone());
            }
            continue;
        };
        receipt.met_records_used += 1;

        // The level table: first sighting wins, and it is capped.
        let level_index = if contributes_a_level(rule.target) {
            match fg.vert_level.iter().position(|&v| v == s.xlvl) {
                Some(i) => i,
                None => {
                    if fg.vert_level.len() >= nfg {
                        return Err(MpasError::Refusal(format!(
                            "the intermediate file holds more than {nfg} distinct levels; raise \
                             config_nfglevels to at least {} and re-run",
                            fg.vert_level.len() + 1
                        )));
                    }
                    fg.vert_level.push(s.xlvl);
                    fg.vert_level.len() - 1
                }
            }
        } else {
            0
        };

        let proj = LatLonProjection {
            lat1: s.start_lat,
            lon1: s.start_lon,
            latinc: s.delta_lat,
            loninc: s.delta_lon,
            knowni: 1.0,
            knownj: 1.0,
            nx: s.nx,
            ny: s.ny,
        };
        let array = Slab::from_met(&s.values, s.nx, s.ny);
        let ctx = InterpContext {
            array: &array,
            msgval: MSGVAL,
            maskval: rule.mask.map(|(m, _, _)| m),
            mask: rule.mask.map(|_| &mask_slab),
            underflow: cfg.oned_underflow,
            underflow_hits: Some(&underflow_hits),
        };

        let at_surface = s.xlvl == SURFACE_LEVEL_TAG;
        let _ = at_surface;
        let on_edges = matches!(rule.target, Target::U | Target::V) && !at_surface;

        let sample = |i: usize, lat: &[f32], lon: &[f32]| -> f32 {
            let (x, y) = proj.locate(lat[i], lon[i]);
            hinterp::interp_sequence(x, y, &ctx, rule.methods, 0)
        };

        if on_edges {
            let values: Vec<f32> = (0..n_edges)
                .into_par_iter()
                .map(|e| sample(e, &edge_lat, &edge_lon))
                .collect();
            let dest = if rule.target == Target::U {
                &mut fg.u
            } else {
                &mut fg.v
            };
            for (e, v) in values.into_iter().enumerate() {
                dest[e][level_index] = v;
            }
            continue;
        }

        let masked_flag = rule.mask.map(|(_, m, _)| m);
        let fillval = rule.mask.map(|(_, _, f)| f).unwrap_or(0.0);
        let values: Vec<f32> = (0..n_cells)
            .into_par_iter()
            .map(|c| {
                if let Some(m) = masked_flag {
                    if statics.landmask[c] == m {
                        return fillval;
                    }
                }
                sample(c, &cell_lat, &cell_lon)
            })
            .collect();

        match rule.target {
            Target::T => {
                // The surface level lands in the same column slot as any
                // other: it is the 200100.0 tag, not a separate array, that
                // makes it the surface later on.
                for (c, v) in values.iter().enumerate() {
                    fg.t[c][level_index] = *v;
                }
            }
            Target::Rh => {
                for (c, v) in values.iter().enumerate() {
                    fg.rh[c][level_index] = *v;
                }
            }
            Target::Sh => {
                for (c, v) in values.iter().enumerate() {
                    fg.sh[c][level_index] = *v;
                }
            }
            Target::Z => {
                for (c, v) in values.iter().enumerate() {
                    fg.z[c][level_index] = *v;
                }
            }
            Target::P => {
                for (c, v) in values.iter().enumerate() {
                    fg.p[c][level_index] = *v;
                }
            }
            Target::U => fg.u10.copy_from_slice(&values),
            Target::V => fg.v10.copy_from_slice(&values),
            Target::Psfc => fg.psfc.copy_from_slice(&values),
            Target::Pmsl => fg.pmsl.copy_from_slice(&values),
            Target::SoilHgt => fg.soilz.copy_from_slice(&values),
            Target::Skintemp => fg.skintemp.copy_from_slice(&values),
            Target::Snow => fg.snow.copy_from_slice(&values),
            Target::Seaice => fg.xice.copy_from_slice(&values),
            Target::SoilT(k, dz) => {
                if k >= nfgs {
                    return Err(MpasError::Refusal(format!(
                        "{} lands on first-guess soil level {} but only {nfgs} were declared",
                        s.field,
                        k + 1
                    )));
                }
                fg.dzs_cm[k] = dz;
                for (c, v) in values.iter().enumerate() {
                    fg.st[c][k] = *v;
                }
            }
            Target::SoilM(k, dz) => {
                if k >= nfgs {
                    return Err(MpasError::Refusal(format!(
                        "{} lands on first-guess soil level {} but only {nfgs} were declared",
                        s.field,
                        k + 1
                    )));
                }
                fg.dzs_cm[k] = dz;
                for (c, v) in values.iter().enumerate() {
                    fg.sm[c][k] = *v;
                }
            }
        }
    }

    receipt.oned_underflow = cfg.oned_underflow.label().to_string();
    receipt.oned_underflow_sites =
        underflow_hits.load(std::sync::atomic::Ordering::Relaxed);
    receipt.first_guess_levels_found = fg.vert_level.len();
    receipt.surface_level_present = fg.vert_level.contains(&SURFACE_LEVEL_TAG);
    if !receipt.surface_level_present {
        return Err(MpasError::Refusal(
            "no first-guess level is tagged 200100.0.  Case 7 identifies the surface level by \
             that tag alone: without it the surface pressure column has no surface, every \
             profile silently keeps a level it should have dropped, and t2m/rh2 stay zero"
                .to_string(),
        ));
    }
    if fg.dzs_cm.iter().any(|&v| v <= 0.0) {
        return Err(MpasError::Refusal(format!(
            "the first-guess soil column is incomplete: layer thicknesses {:?} (cm).  A zero \
             means a soil level this run declared was never delivered by the intermediate file",
            fg.dzs_cm
        )));
    }
    Ok(fg)
}

/// Build one init file.
pub fn build_init(cfg: &InitConfig) -> MpasResult<InitReceipt> {
    let started = std::time::Instant::now();
    let mut receipt = InitReceipt {
        met_path: cfg.met_path.display().to_string(),
        static_path: cfg.static_path.display().to_string(),
        capsule_path: cfg.capsule_path.display().to_string(),
        reference_path: cfg.reference_path.display().to_string(),
        out_path: cfg.out_path.display().to_string(),
        start_time: cfg.start_time.clone(),
        extrap_airtemp: cfg.extrap_airtemp.label().to_string(),
        use_spechumd: cfg.use_spechumd,
        virtual_factor: format!("{:?}", cfg.virtual_factor),
        deep_moisture_rule: format!("{:?}", cfg.deep_moisture),
        deferrals: vec![
            "vertical grid construction (zgrid, zz, fzm, fzp, dzu, rdzw, zb, zb3) is READ from \
             the capsule, not built: terrain smoothing, the hybrid coordinate and the \
             coordinate-surface smoother are not ported this wave"
                .to_string(),
            "statics (terrain, land use, soil category, monthly greenness and albedo) are \
             native-minted capsule data, not minted here"
                .to_string(),
            "uReconstruct{X,Y,Z,Zonal,Meridional} are diagnostic RBF reconstructions and are \
             carried from the capsule, not computed"
                .to_string(),
            "coeffs_reconstruct and edgeNormalVectors are carried from the capsule, so this \
             file is only drop-in for a consumer that pins the SAME grid/static bytes the \
             capsule was built from.  The forecast driver's \
             overlay_exact_init_reconstruction_coefficients refuses an init built against a \
             different grid/static pair by design, and reports it as changed coefficient \
             bytes rather than as a provenance mismatch"
                .to_string(),
        ],
        ..Default::default()
    };

    let mesh = capsule::read_mesh_geometry(&cfg.static_path)?;
    let mut statics = capsule::read_statics(&cfg.static_path, mesh.n_cells)?;
    // The terrain every later step sees is the SMOOTHED one the vertical-grid
    // block wrote back, not the raw static field.  See read_smoothed_terrain.
    statics.ter = capsule::read_smoothed_terrain(&cfg.capsule_path, mesh.n_cells)?;
    let metrics = capsule::read_vertical_metrics(&cfg.capsule_path, mesh.n_cells, mesh.n_edges)?;
    receipt.zgrid_values_asserted_identical =
        capsule::assert_zgrid_identical(&metrics, &cfg.reference_path)?;
    receipt.n_cells = mesh.n_cells;
    receipt.n_edges = mesh.n_edges;
    receipt.n_vert_levels = metrics.n_vert_levels;

    let slabs = metfile::read_met_file(&cfg.met_path)?;
    let mut fg = horizontal_pass(&slabs, &mesh, &statics, cfg, &mut receipt)?;

    let nz = metrics.n_vert_levels;
    let n_cells = mesh.n_cells;
    let n_edges = mesh.n_edges;
    let nfg = fg.vert_level.len();
    let sfc = fg
        .vert_level
        .iter()
        .position(|&v| v == SURFACE_LEVEL_TAG)
        .expect("checked above");

    // Surface fields lifted off the first-guess surface level.
    let t2m: Vec<f32> = (0..n_cells).map(|c| fg.t[c][sfc]).collect();
    let rh2: Vec<f32> = (0..n_cells).map(|c| fg.rh[c][sfc]).collect();

    // Isobaric data carries no 3-D pressure; fill it from the level tags.
    let isobaric = fg
        .p
        .iter()
        .all(|col| col[..nfg].iter().all(|&v| v == 0.0));
    if isobaric {
        for c in 0..n_cells {
            for (k, &lvl) in fg.vert_level.iter().enumerate() {
                fg.p[c][k] = if lvl == SURFACE_LEVEL_TAG {
                    fg.psfc[c]
                } else {
                    lvl
                };
            }
        }
    } else {
        for c in 0..n_cells {
            fg.p[c][sfc] = fg.psfc[c];
            fg.z[c][sfc] = fg.soilz[c];
        }
    }

    // Vertical interpolation, cell by cell.
    let zgrid = &metrics.zgrid;
    let extrap_t = cfg.extrap_airtemp;
    type Column = (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, f32);
    let columns: Vec<MpasResult<Column>> = (0..n_cells)
        .into_par_iter()
        .map(|c| {
            let mut coord = vec![0.0f32; nfg];
            let mut value = vec![0.0f32; nfg];
            let prep = |coord: &mut Vec<f32>, value: &mut Vec<f32>, src: &dyn Fn(usize) -> f32| {
                for k in 0..nfg {
                    coord[k] = if fg.vert_level[k] == SURFACE_LEVEL_TAG {
                        SURFACE_SENTINEL
                    } else {
                        fg.z[c][k]
                    };
                    value[k] = src(k);
                }
                sorted_column(coord, value);
            };

            let mut t = vec![0.0f32; nz];
            let mut relhum = vec![0.0f32; nz];
            let mut spechum = vec![0.0f32; nz];
            let mut pressure = vec![0.0f32; nz];

            prep(&mut coord, &mut value, &|k| fg.t[c][k]);
            for k in 0..nz {
                let target = 0.5 * (zgrid[c][k] + zgrid[c][k + 1]);
                t[k] = vertical_interp(target, &coord[..nfg - 1], &value[..nfg - 1], extrap_t)
                    .map_err(|_| {
                        MpasError::Refusal(format!(
                            "temperature extrapolation above the first-guess column top is not \
                             implemented for the lapse-rate mode; cell {c} level {k} asked for \
                             it.  This is the Fortran's own fatal error, not a port limitation"
                        ))
                    })?;
            }

            prep(&mut coord, &mut value, &|k| fg.rh[c][k]);
            for k in (0..nz).rev() {
                let target = 0.5 * (zgrid[c][k] + zgrid[c][k + 1]);
                relhum[k] =
                    vertical_interp(target, &coord[..nfg - 1], &value[..nfg - 1], Extrap::Constant)
                        .unwrap_or(value[0]);
            }

            prep(&mut coord, &mut value, &|k| fg.sh[c][k].max(0.0));
            for k in (0..nz).rev() {
                let target = 0.5 * (zgrid[c][k] + zgrid[c][k + 1]);
                spechum[k] =
                    vertical_interp(target, &coord[..nfg - 1], &value[..nfg - 1], Extrap::Constant)
                        .unwrap_or(value[0]);
            }

            prep(&mut coord, &mut value, &|k| fg.p[c][k].ln());
            for k in 0..nz {
                let target = 0.5 * (zgrid[c][k] + zgrid[c][k + 1]);
                pressure[k] =
                    vertical_interp(target, &coord[..nfg - 1], &value[..nfg - 1], Extrap::Linear)
                        .unwrap_or(value[nfg - 2])
                        .exp();
            }

            // Surface pressure is the one field that keeps the surface level:
            // the full column, sentinel included, targeted at zgrid(1).
            prep(&mut coord, &mut value, &|k| fg.p[c][k].ln());
            let psfc = vertical_interp(zgrid[c][0], &coord, &value, Extrap::Linear)
                .unwrap_or(value[nfg - 1])
                .exp();

            Ok((t, relhum, spechum, pressure, psfc))
        })
        .collect();

    let mut t_col = Vec::with_capacity(n_cells);
    let mut rh_col = Vec::with_capacity(n_cells);
    let mut sh_col = Vec::with_capacity(n_cells);
    let mut p_col = Vec::with_capacity(n_cells);
    let mut psfc = Vec::with_capacity(n_cells);
    for c in columns {
        let (t, rh, sh, p, ps) = c?;
        t_col.push(t);
        rh_col.push(rh);
        sh_col.push(sh);
        p_col.push(p);
        psfc.push(ps);
    }

    // Edge winds: rotate to the edge normal, then interpolate on the two-cell
    // mean coordinate to the four-point mean target.
    let u_edge: Vec<Vec<f32>> = (0..n_edges)
        .into_par_iter()
        .map(|e| {
            let [c1, c2] = mesh.cells_on_edge[e];
            let c1 = (c1.max(1) - 1).min(n_cells - 1);
            let c2 = (c2.max(1) - 1).min(n_cells - 1);
            let angle = mesh.angle_edge[e];
            let mut coord = vec![0.0f32; nfg];
            let mut value = vec![0.0f32; nfg];
            for k in 0..nfg {
                coord[k] = if fg.vert_level[k] == SURFACE_LEVEL_TAG {
                    SURFACE_SENTINEL
                } else {
                    dynamics::edge_column_coordinate(fg.z[c1][k], fg.z[c2][k])
                };
                value[k] = dynamics::edge_normal_wind(angle, fg.u[e][k], fg.v[e][k]);
            }
            sorted_column(&mut coord, &mut value);
            (0..nz)
                .map(|k| {
                    let target = dynamics::edge_target_height(
                        zgrid[c1][k],
                        zgrid[c1][k + 1],
                        zgrid[c2][k],
                        zgrid[c2][k + 1],
                    );
                    dynamics::interp_edge_column(
                        target,
                        &coord[..nfg - 1],
                        &value[..nfg - 1],
                    )
                })
                .collect()
        })
        .collect();

    // Moisture, then RH over ice, then the column chain.
    let use_sh = cfg.use_spechumd
        && sh_col
            .iter()
            .any(|col| col.iter().any(|&v| v != 0.0));
    let factor = cfg.virtual_factor;
    let states: Vec<dynamics::ColumnState> = (0..n_cells)
        .into_par_iter()
        .map(|c| {
            dynamics::build_column(
                &t_col[c],
                &p_col[c],
                &rh_col[c],
                &sh_col[c],
                use_sh,
                &zgrid[c],
                &metrics.zz[c],
                &metrics.fzm,
                &metrics.fzp,
                &metrics.dzu,
                metrics.rdzw[0],
                factor,
            )
        })
        .collect();

    for c in 0..n_cells {
        dynamics::convert_relhum_wrt_ice(&t_col[c], &mut rh_col[c]);
    }

    let mut it_max = 0u32;
    let mut it_sum = 0u64;
    let mut it_n = 0u64;
    let mut capped = 0usize;
    for s in &states {
        let mut hit = false;
        for &i in &s.hydrostatic_iterations[1..] {
            it_max = it_max.max(i);
            it_sum += i as u64;
            it_n += 1;
            if i >= 30 {
                hit = true;
            }
        }
        if hit {
            capped += 1;
        }
    }
    receipt.hydrostatic_iterations_max = it_max;
    receipt.hydrostatic_iterations_mean = if it_n > 0 {
        it_sum as f64 / it_n as f64
    } else {
        0.0
    };
    receipt.hydrostatic_cells_hitting_the_cap = capped;

    // ru, rw and w.
    let mut ru = vec![vec![0.0f32; nz]; n_edges];
    for e in 0..n_edges {
        let [c1, c2] = mesh.cells_on_edge[e];
        let c1 = (c1.max(1) - 1).min(n_cells - 1);
        let c2 = (c2.max(1) - 1).min(n_cells - 1);
        for k in 0..nz {
            ru[e][k] = u_edge[e][k] * 0.5 * (states[c1].rho_zz[k] + states[c2].rho_zz[k]);
        }
    }
    // w and rw are dimensioned against nVertLevelsP1 in the file, and the
    // case code fills only k = 2 .. nVertLevels; the bottom and the top stay
    // at zero.  Allocating nz here instead would emit a short slab.
    let nzp1 = nz + 1;
    let mut rw = vec![vec![0.0f32; nzp1]; n_cells];
    for c in 0..n_cells {
        for i in 0..mesh.n_edges_on_cell[c] {
            let e = mesh.edges_on_cell[c][i];
            if e == 0 || e > n_edges {
                continue;
            }
            let e = e - 1;
            let side_one = mesh.cells_on_edge[e][0] == c + 1;
            for k in 1..nz {
                let flux = metrics.fzm[k] * ru[e][k] + metrics.fzp[k] * ru[e][k - 1];
                let metric = metrics.fzm[k] * metrics.zz[c][k] + metrics.fzp[k] * metrics.zz[c][k - 1];
                if side_one {
                    rw[c][k] -= metric * metrics.zb[e][0][k] * flux;
                } else {
                    rw[c][k] += metric * metrics.zb[e][1][k] * flux;
                }
                if cfg.theta_adv_order == 3 {
                    let sign = if ru[e][k] >= 0.0 { 1.0f32 } else { -1.0f32 };
                    if side_one {
                        rw[c][k] += sign * cfg.coef_3rd_order * metric * metrics.zb3[e][0][k] * flux;
                    } else {
                        rw[c][k] -= sign * cfg.coef_3rd_order * metric * metrics.zb3[e][1][k] * flux;
                    }
                }
            }
        }
    }
    let mut w = vec![vec![0.0f32; nzp1]; n_cells];
    for c in 0..n_cells {
        for k in 1..nz {
            w[c][k] = rw[c][k]
                / (metrics.fzp[k] * states[c].rho_zz[k - 1] + metrics.fzm[k] * states[c].rho_zz[k]);
        }
    }

    // Soil and the surface companions.
    // SST is a copy of the skin temperature taken in the case code, BEFORE
    // physics_initialize_real adjusts the skin temperature onto the model's
    // terrain.  Copying it afterwards would give every land cell an SST that
    // carries the terrain correction.
    let sst = fg.skintemp.clone();

    let mut skintemp = fg.skintemp.clone();
    let mut tmn = soil::adjust_input_soiltemps(
        &statics.landmask,
        &statics.ter,
        &statics.soiltemp,
        &fg.soilz,
        &mut skintemp,
        &mut fg.st,
    );
    // The reference's own consistency pass runs before the abort check.
    let (flagged, repaired) =
        soil::first_guess_consistency_pass(&statics.landmask, &fg.st, &mut fg.sm);

    let (mut tslb, mut smois, mut sh2o, mut smcrel, mut soil_counts) =
        soil::init_soil_layers_properties(
        &statics.landmask,
        &skintemp,
        &tmn,
        &fg.st,
        &fg.sm,
        &fg.dzs_cm,
        cfg.deep_moisture,
    )?;
    soil_counts.land_moisture_repaired = repaired;
    soil_counts.land_temperature_flagged = flagged;
    receipt.soil = soil_counts;

    let (year, month, day) = parse_ymd(&cfg.start_time)?;
    let mut sfc_albbck = soil::monthly_interp_to_date(&statics.albedo12m, year, month, day);
    for (c, v) in sfc_albbck.iter_mut().enumerate() {
        *v /= 100.0;
        if statics.landmask[c] == 0 {
            *v = 0.08;
        }
    }
    let vegfra = soil::monthly_interp_to_date(&statics.greenfrac, year, month, day);
    let (shdmin, shdmax) = soil::monthly_min_max(&statics.greenfrac);
    let mut snowc: Vec<f32> = fg
        .snow
        .iter()
        .map(|&s| if s >= 10.0 { 1.0 } else { 0.0 })
        .collect();
    let mut snowh: Vec<f32> = fg.snow.iter().map(|&s| s * 5.0 / 1000.0).collect();
    let mut xland: Vec<f32> = statics
        .landmask
        .iter()
        .map(|&m| if m == 1 { 1.0 } else { 2.0 })
        .collect();
    let mut xice: Vec<f32> = fg.xice.iter().map(|&v| v.clamp(0.0, 1.0)).collect();
    let mut seaice: Vec<f32> = vec![0.0; n_cells];
    let mut snow = fg.snow.clone();
    let mut vegfra = vegfra;
    let mut snoalb = statics.snoalb.clone();
    let mut ivgtyp = statics.ivgtyp.clone();
    let mut isltyp = statics.isltyp.clone();

    // The sea-ice initialisation runs last and rewrites what the soil step
    // produced at every converted point.
    receipt.seaice = soil::physics_init_seaice(
        cfg.frac_seaice,
        cfg.tsk_seaice_threshold,
        statics.isice_lu,
        &statics.landmask,
        &skintemp,
        &mut xice,
        &mut seaice,
        &mut snow,
        &mut snowc,
        &mut snowh,
        &mut vegfra,
        &mut snoalb,
        &mut ivgtyp,
        &mut isltyp,
        &mut xland,
        &mut tmn,
        &mut tslb,
        &mut smois,
        &mut sh2o,
        &mut smcrel,
    );
    let q2: Vec<f32> = (0..n_cells)
        .map(|c| dynamics::diagnose_q2(t2m[c], psfc[c], rh2[c]))
        .collect();

    // Assemble the computed set, cell-major with the vertical fastest.
    let flat3 = |f: &dyn Fn(usize, usize) -> f32| -> Vec<f32> {
        let mut out = Vec::with_capacity(n_cells * nz);
        for c in 0..n_cells {
            for k in 0..nz {
                out.push(f(c, k));
            }
        }
        out
    };
    let flat3_edges = |src: &Vec<Vec<f32>>| -> Vec<f32> {
        let mut out = Vec::with_capacity(n_edges * nz);
        for e in 0..n_edges {
            out.extend_from_slice(&src[e][..nz]);
        }
        out
    };
    let flat3p1 = |src: &Vec<Vec<f32>>| -> Vec<f32> {
        let mut out = Vec::with_capacity(n_cells * nzp1);
        for c in 0..n_cells {
            out.extend_from_slice(&src[c][..nzp1]);
        }
        out
    };
    let flat_soil = |src: &Vec<[f32; 4]>| -> Vec<f32> {
        let mut out = Vec::with_capacity(n_cells * 4);
        for c in 0..n_cells {
            out.extend_from_slice(&src[c]);
        }
        out
    };

    let mut computed: BTreeMap<String, emit::Computed> = BTreeMap::new();
    let mut put = |name: &str, v: Vec<f32>| {
        computed.insert(name.to_string(), emit::Computed::Floats(v));
    };
    let mut ints: Vec<(String, Vec<i32>)> = Vec::new();
    let mut put_ints = |name: &str, v: Vec<i32>| ints.push((name.to_string(), v));
    // Only the names the init stream actually carries.  The dycore's working
    // state (pressure, exner, rho_zz, theta_m, ru, rw and the perturbation
    // pair) is rebuilt by the model at start-up and is not in this file; the
    // emitter refuses on any computed name the file has no variable for, so a
    // rename upstream cannot silently drop one of these.
    put("theta", flat3(&|c, k| states[c].theta[k]));
    put("rho", flat3(&|c, k| states[c].rho[k]));
    put("rho_base", flat3(&|c, k| states[c].rho_base[k]));
    put("theta_base", flat3(&|c, k| states[c].theta_base[k]));
    put("qv", flat3(&|c, k| states[c].qv[k]));
    // The other three scalars and the turbulence kinetic energy start at zero;
    // they are computed-as-zero, not carried, so a capsule with a warm start
    // in it cannot leak into a cold one.
    put("qc", vec![0.0; n_cells * nz]);
    put("qr", vec![0.0; n_cells * nz]);
    put("tke", vec![0.0; n_cells * nz]);
    put("relhum", flat3(&|c, k| rh_col[c][k]));
    put("w", flat3p1(&w));
    put("u", flat3_edges(&u_edge));
    put("precipw", states.iter().map(|s| s.precipw).collect());
    put(
        "surface_pressure",
        states.iter().map(|s| s.surface_pressure).collect(),
    );
    put("t2m", t2m.clone());
    put("rh2", rh2.clone());
    put("q2", q2);
    put("u10", fg.u10.clone());
    put("v10", fg.v10.clone());
    put("skintemp", skintemp.clone());
    put("sst", sst);
    put("tmn", tmn);
    put("xice", xice);
    put("seaice", seaice);
    put("snow", snow);
    put("snowc", snowc);
    put("snowh", snowh);
    put("xland", xland);
    put("sfc_albbck", sfc_albbck);
    put("vegfra", vegfra);
    put("snoalb", snoalb);
    put_ints("ivgtyp", ivgtyp);
    put_ints("isltyp", isltyp);
    put("shdmin", shdmin);
    put("shdmax", shdmax);
    put("tslb", flat_soil(&tslb));
    put("smois", flat_soil(&smois));
    put("sh2o", flat_soil(&sh2o));
    // smcrel is computed (identically zero) but has no variable in the init
    // stream; it is dropped here deliberately rather than by accident, and
    // the emitter would have refused if it were passed.
    let _ = &smcrel;
    // Soil layer geometry, which the file carries per cell.
    let zs_m = soil::noah_mid_depths();
    let mut dzs_flat = Vec::with_capacity(n_cells * 4);
    let mut zs_flat = Vec::with_capacity(n_cells * 4);
    for _ in 0..n_cells {
        dzs_flat.extend_from_slice(&soil::NOAH_LAYER_THICKNESS);
        zs_flat.extend_from_slice(&zs_m);
    }
    put("dzs", dzs_flat);
    put("zs", zs_flat);

    for (name, v) in ints {
        computed.insert(name, emit::Computed::Ints(v));
    }

    // The three identity labels.  They are this file's, not the capsule's.
    computed.insert(
        "xtime".to_string(),
        emit::Computed::Text(cfg.start_time.clone()),
    );
    computed.insert(
        "initial_time".to_string(),
        emit::Computed::Text(cfg.start_time.clone()),
    );
    computed.insert(
        "mminlu".to_string(),
        emit::Computed::Text(cfg.landuse_table.clone()),
    );

    let seed = format!(
        "{}|{}|{}|{}",
        cfg.met_path.display(),
        cfg.static_path.display(),
        cfg.start_time,
        cfg.capsule_path.display()
    );
    let file_id = emit::mint_file_id(&seed);
    let parent = capsule_file_id(&cfg.capsule_path).unwrap_or_default();
    receipt.emit = emit::write_init(
        &cfg.out_path,
        &cfg.capsule_path,
        computed,
        &file_id,
        &[parent],
        &cfg.provenance,
        &cfg.lineage,
    )?;

    receipt.seconds = started.elapsed().as_secs_f64();
    Ok(receipt)
}

fn capsule_file_id(path: &Path) -> Option<String> {
    let file = netcrust::File::open(path).ok()?;
    file.attribute("file_id")?.as_string().map(str::to_string)
}

fn parse_ymd(stamp: &str) -> MpasResult<(i32, u32, u32)> {
    let bytes: Vec<char> = stamp.chars().collect();
    if bytes.len() < 10 {
        return Err(MpasError::Refusal(format!(
            "start time '{stamp}' is not YYYY-MM-DD_HH:MM:SS"
        )));
    }
    let take = |from: usize, n: usize| -> MpasResult<i64> {
        let s: String = bytes[from..from + n].iter().collect();
        s.parse::<i64>()
            .map_err(|_| MpasError::Refusal(format!("start time '{stamp}' is not a date")))
    };
    Ok((take(0, 4)? as i32, take(5, 2)? as u32, take(8, 2)? as u32))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_soil_field_name_lands_on_a_declared_layer_and_thickness() {
        for (name, k, dz) in [
            ("ST000010", 0usize, 10.0f32),
            ("ST010040", 1, 30.0),
            ("ST040100", 2, 60.0),
            ("ST100200", 3, 100.0),
            ("ST000007", 0, 7.0),
            ("ST007028", 1, 21.0),
            ("ST028100", 2, 72.0),
            ("ST100289", 3, 189.0),
        ] {
            let rule = rule_for(name).unwrap_or_else(|| panic!("{name} has no rule"));
            match rule.target {
                Target::SoilT(got_k, got_dz) => {
                    assert_eq!(got_k, k, "{name}");
                    assert_eq!(got_dz, dz, "{name}");
                }
                other => panic!("{name} mapped to {other:?}"),
            }
        }
    }

    #[test]
    fn soil_temperature_and_moisture_take_different_method_lists() {
        // Temperature leads with the parabolic sixteen-point; moisture does
        // not.  Swapping them changes coastal soil values.
        let t = rule_for("ST000010").unwrap();
        let m = rule_for("SM000010").unwrap();
        assert_eq!(t.methods[0], Method::SixteenPoint);
        assert_eq!(m.methods[0], Method::FourPoint);
        assert_eq!(t.mask.unwrap().2, 285.0, "water fill for temperature");
        assert_eq!(m.mask.unwrap().2, 1.0, "water fill for moisture");
    }

    #[test]
    fn only_profile_fields_contribute_first_guess_levels() {
        assert!(contributes_a_level(Target::T));
        assert!(contributes_a_level(Target::U));
        assert!(!contributes_a_level(Target::Psfc));
        assert!(!contributes_a_level(Target::SoilT(0, 10.0)));
        assert!(!contributes_a_level(Target::Skintemp));
    }

    #[test]
    fn a_field_the_table_does_not_know_is_ignored_not_guessed() {
        assert!(rule_for("TOTALCLOUD").is_none());
    }

    #[test]
    fn the_start_date_parses_into_its_parts() {
        assert_eq!(parse_ymd("2025-03-14_12:00:00").unwrap(), (2025, 3, 14));
        assert!(parse_ymd("nope").is_err());
    }

    #[test]
    fn deg_per_rad_is_the_fortran_parameter_not_the_accurate_one() {
        // `mpas_init_atm_llxy.F`: PI is a REAL(RKIND) parameter, so in the
        // single-precision build `180./PI` divides by f32(pi) and lands one
        // ULP below the correctly rounded 180/pi.  Pinned by bit pattern
        // because the whole point is that the two neighbouring f32 values
        // print the same to seven digits.
        assert_eq!(
            DEG_PER_RAD.to_bits(),
            0x4265_2ee0,
            "DEG_PER_RAD must be f32(180)/f32(pi) = 57.295776367, \
             not the correctly rounded 57.295780182"
        );
        let accurate = 57.295_78_f32;
        assert_eq!(accurate.to_bits(), 0x4265_2ee1, "the value NOT to use");
        assert_ne!(DEG_PER_RAD, accurate);
        assert_eq!(
            f32::from_bits(DEG_PER_RAD.to_bits() + 1),
            accurate,
            "the two differ by exactly one ULP"
        );
    }

    #[test]
    fn one_ulp_of_deg_per_rad_can_change_which_stencil_a_cell_samples() {
        // The reason the constant is pinned: at a longitude where the two
        // constants straddle a grid line, `floor(x)` differs and the
        // interpolation reads a different pair of source columns.
        let accurate = 57.295_78_f32;
        let proj = LatLonProjection {
            lat1: 90.0,
            lon1: 0.0,
            latinc: -0.25,
            loninc: 0.25,
            knowni: 1.0,
            knownj: 1.0,
            nx: 1440,
            ny: 721,
        };
        // Search a band of longitudes for a radian value whose degree
        // conversion straddles a grid line under the two constants.
        let mut found = None;
        for n in 0..2_000_000u32 {
            let lon_rad = 4.0 + (n as f32) * 1.0e-6;
            let a = lon_rad * DEG_PER_RAD;
            let b = lon_rad * accurate;
            let (xa, _) = proj.locate(0.0, a);
            let (xb, _) = proj.locate(0.0, b);
            if xa.floor() != xb.floor() {
                found = Some((lon_rad, xa, xb));
                break;
            }
        }
        let (lon_rad, xa, xb) = found.expect(
            "the two constants must disagree about a grid line somewhere in \
             4.0..6.0 radians of longitude",
        );
        assert_ne!(xa.floor(), xb.floor(), "lon_rad = {lon_rad}");
    }
}
