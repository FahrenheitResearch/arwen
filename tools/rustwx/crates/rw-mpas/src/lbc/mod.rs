//! The case-9 lateral-boundary-condition stream builder.
//!
//! A transcription of `init_atm_case_lbc` (`mpas_init_atm_cases.F`, MPAS
//! v8.4.1): per boundary time, read one WPS intermediate file, horizontally
//! and vertically interpolate to the regional mesh, run the same base-state +
//! hydrostatic chain the init case runs, and write the `lbc` stream — one
//! CDF-5 file per boundary time carrying FULL-MESH fields (`lbc_u` over every
//! edge, the cell fields over every cell), not a boundary strip.  Which
//! elements the consuming model reads out of them is the model's business.
//!
//! ## What is shared with case 7 and what is not
//! The met reader (`init::metfile`), the horizontal interpolation
//! (`init::hinterp`), the vertical kernel (`init::vinterp`), the mesh/metric
//! reading (`init::capsule`) and the constants (`init::dynamics::constants`,
//! [`crate::init::dynamics::rslf`]) are the case-7 machinery, reused as-is.
//! The pipeline around them is NOT case 7 and is transcribed separately,
//! because `init_atm_case_lbc` genuinely differs line-by-line:
//!
//! * only nine met fields are consumed (`UU VV TT RH SPECHUMD GHT SOILHGT
//!   PRES PRESSURE`); no soil, no snow, no sea ice, no skin temperature, and
//!   therefore no land-sea interpolation mask anywhere;
//! * there is no PSFC: the isobaric branch leaves the surface pressure row at
//!   zero and the pressure column substitutes `1.0` (log = 0) at the surface
//!   before the sentinel sorts it out of the profile;
//! * edge wind columns extrapolate LINEARLY (`extrap = 1`) where case 7 uses
//!   constant — `interp_edge_column` in `init::dynamics` documents the same
//!   difference from the other side;
//! * the first `rho_zz` is formed with NO virtual-temperature factor at all
//!   (case 7 uses `rvord - 1` there); the hydrostatic loop, `theta_m` and the
//!   final recovery all use the literal `1.61`, as in case 7's
//!   `VirtualFactor::ReproduceFortran` arm;
//! * relative humidity is never rewritten with respect to ice, and no surface
//!   or soil companion field is produced.
//!
//! ## The other route through this module
//! Everything above builds a boundary state from an external model's first
//! guess.  [`parent`] builds one from another run's own output instead —
//! a parent's prognostic state, sampled onto the child's cells and edges and
//! remapped onto its levels, which is what a cascade of separate runs needs.
//! Which route runs is decided by a row of the driving-source registry in
//! [`source`], never by a switch naming a model, and both routes emit through
//! the same [`emit`] writer, so the file a consumer reads is the same file
//! either way.
//!
//! ## The garbage cell
//! On a culled regional mesh some outer edges have one adjacent cell; the
//! file stores `0` in `cellsOnEdge` there (measured on the reference cull:
//! 756 such edges).  Native MPAS redirects index 0 to the allocated-but-never
//! -written cell `nCells+1`, whose fields hold their post-allocation default
//! of `0.0`.  This module reproduces that arithmetic — a stored 0 contributes
//! `0.0` to every average it appears in — rather than clamping to a real
//! cell, because the clamp would put a real cell's column where the reference
//! put zeros.

#![allow(clippy::needless_range_loop)]

pub mod compare;
pub mod emit;
pub mod parent;
pub mod source;
pub mod sphere;

use std::path::PathBuf;

use rayon::prelude::*;

use crate::error::{MpasError, MpasResult};
use crate::init::capsule::{self, MeshGeometry, VerticalMetrics};
use crate::init::dynamics::constants::{CP, CV, GRAVITY, P0, RGAS, T0B};
use crate::init::dynamics::rslf;
use crate::init::hinterp::{interp_sequence, InterpContext, LatLonProjection, Method, Slab, Underflow};
use crate::init::metfile::{read_met_file, MetSlab};
use crate::init::vinterp::{
    sorted_column, vertical_interp, Extrap, SURFACE_LEVEL_TAG, SURFACE_SENTINEL,
};
use crate::init::DEG_PER_RAD;

const MSGVAL: f32 = -1.0e30;

/// The literal `1.61` the Fortran uses in the hydrostatic loop, in `theta_m`
/// and in the recovery of `lbc_theta` — not `rvord - 1`.  See
/// `init::dynamics` for the history of that mix; the lbc case carries no
/// switch because `init_atm_case_lbc` has no `rvord - 1` site at all.
const VIRTUAL_161: f32 = 1.61;

/// The nine met fields `init_atm_case_lbc` consumes.  Everything else in the
/// intermediate file is read past and reported, never guessed at.
const LBC_MET_FIELDS: [&str; 9] = [
    "UU", "VV", "TT", "RH", "SPECHUMD", "GHT", "SOILHGT", "PRES", "PRESSURE",
];

/// One boundary time to produce: the valid time and the intermediate file
/// carrying that time's first guess.  Which source model produced the
/// intermediate is invisible here by design — cadence and availability are
/// the caller's registry data, and the file format is the source-agnostic WPS
/// intermediate the arbitrary ingest already emits for every registered
/// source.
#[derive(Debug, Clone)]
pub struct BoundaryInterval {
    /// `YYYY-MM-DD_HH:MM:SS`.
    pub valid_time: String,
    pub met_path: PathBuf,
}

/// Everything the caller must state.  As with `rw_mpas_init`, the switches
/// that select physics have no defaults.
#[derive(Debug, Clone)]
pub struct LbcConfig {
    /// The regional initial-conditions file: mesh geometry, vertical metrics
    /// and the parent identity all come from here, exactly as native case 9
    /// reads its `input` stream.
    pub grid_path: PathBuf,
    pub out_dir: PathBuf,
    /// `config_start_time`: the epoch of the `Time` axis.
    pub start_time: String,
    /// `config_stop_time`, stamped into the header.
    pub stop_time: String,
    pub intervals: Vec<BoundaryInterval>,
    pub n_fg_levels: usize,
    pub extrap_airtemp: Extrap,
    pub use_spechumd: bool,
    pub theta_adv_order: i32,
    pub coef_3rd_order: f32,
    /// `config_fg_interval` in seconds, stamped into the header.  The data in
    /// each file depends only on that file's own valid time; the interval is
    /// registry metadata carried for the consumer.
    pub fg_interval_seconds: i64,
    pub oned_underflow: Underflow,
    /// Overrides for header config attributes that are namelist metadata
    /// rather than producer switches (met prefix, geography table names, ...).
    /// Names must exist in the v8.4.1 table and must not shadow a producer
    /// switch.  See [`emit::config_attributes`].
    pub attr_overrides: std::collections::BTreeMap<String, serde_json::Value>,
    pub provenance: String,
}

/// Receipt for one produced boundary time.
#[derive(Debug, Default, serde::Serialize)]
pub struct IntervalReceipt {
    pub valid_time: String,
    pub met_path: String,
    pub out_path: String,
    pub time_seconds: f64,
    pub met_records_read: usize,
    pub met_records_used: usize,
    pub met_records_ignored: Vec<String>,
    pub first_guess_levels_found: usize,
    /// `isobaric` or `model-level`, the branch the pressure content selected.
    pub pressure_branch: String,
    pub oned_underflow_sites: u64,
    pub hydrostatic_iterations_max: u32,
    pub hydrostatic_cells_hitting_the_cap: usize,
    /// Edges whose second (or first) adjacent cell is the stored-0 garbage
    /// cell; their columns carry the reference's zero contributions.
    pub one_cell_edges: usize,
    pub out_sha256: String,
    pub seconds: f64,
}

/// The whole run's receipt.
#[derive(Debug, Default, serde::Serialize)]
pub struct LbcReceipt {
    pub grid_path: String,
    pub start_time: String,
    pub n_cells: usize,
    pub n_edges: usize,
    pub n_vert_levels: usize,
    pub extrap_airtemp: String,
    pub use_spechumd: bool,
    pub oned_underflow: String,
    pub intervals: Vec<IntervalReceipt>,
    pub seconds: f64,
}

/// First-guess arrays for one boundary time, `[point][level]`.
struct FgLbc {
    vert_level: Vec<f32>,
    t: Vec<Vec<f32>>,
    rh: Vec<Vec<f32>>,
    sh: Vec<Vec<f32>>,
    z: Vec<Vec<f32>>,
    p: Vec<Vec<f32>>,
    u: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
    soilz: Vec<f32>,
}

/// Seconds between two `YYYY-MM-DD_HH:MM:SS` stamps on the proleptic
/// Gregorian calendar (`config_calendar_type = 'gregorian'`).
pub fn seconds_between(start: &str, end: &str) -> MpasResult<i64> {
    Ok(epoch_seconds(end)? - epoch_seconds(start)?)
}

fn epoch_seconds(stamp: &str) -> MpasResult<i64> {
    let bad = || {
        MpasError::Refusal(format!(
            "time stamp '{stamp}' is not YYYY-MM-DD_HH:MM:SS"
        ))
    };
    let b: Vec<char> = stamp.chars().collect();
    if b.len() < 19 {
        return Err(bad());
    }
    let num = |from: usize, n: usize| -> MpasResult<i64> {
        b[from..from + n]
            .iter()
            .collect::<String>()
            .parse::<i64>()
            .map_err(|_| bad())
    };
    let (y, m, d) = (num(0, 4)?, num(5, 2)?, num(8, 2)?);
    let (hh, mm, ss) = (num(11, 2)?, num(14, 2)?, num(17, 2)?);
    if !(1..=12).contains(&m) || !(1..=31).contains(&d) {
        return Err(bad());
    }
    // Days-from-civil (Howard Hinnant's algorithm), proleptic Gregorian.
    let y_adj = if m <= 2 { y - 1 } else { y };
    let era = if y_adj >= 0 { y_adj } else { y_adj - 399 } / 400;
    let yoe = y_adj - era * 400;
    let mp = (m + 9) % 12;
    let doy = (153 * mp + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    let days = era * 146097 + doe - 719468;
    Ok(days * 86400 + hh * 3600 + mm * 60 + ss)
}

/// Build the whole boundary series.
pub fn build_lbc(cfg: &LbcConfig) -> MpasResult<LbcReceipt> {
    let started = std::time::Instant::now();

    if cfg.intervals.is_empty() {
        return Err(MpasError::Refusal(
            "no boundary intervals were given.  Each --interval names one valid time and the \
             intermediate file carrying it; without at least one there is nothing to produce"
                .to_string(),
        ));
    }

    let mesh = capsule::read_mesh_geometry(&cfg.grid_path)?;
    let metrics = capsule::read_vertical_metrics(&cfg.grid_path, mesh.n_cells, mesh.n_edges)?;
    let max_zgrid = metrics
        .zgrid
        .iter()
        .flat_map(|col| col.iter())
        .fold(f32::MIN, |a, &b| a.max(b));
    if max_zgrid == 0.0 {
        // The native check, kept verbatim in spirit: an input stream without
        // vertical grid information interpolates everything to height zero.
        return Err(MpasError::Refusal(format!(
            "the maximum value of zgrid in {} is 0; the grid file carries no vertical grid \
             information, and every column would be interpolated to height zero",
            cfg.grid_path.display()
        )));
    }

    let header = emit::HeaderSource::from_grid(&cfg.grid_path)?;

    let mut receipt = LbcReceipt {
        grid_path: cfg.grid_path.display().to_string(),
        start_time: cfg.start_time.clone(),
        n_cells: mesh.n_cells,
        n_edges: mesh.n_edges,
        n_vert_levels: metrics.n_vert_levels,
        extrap_airtemp: cfg.extrap_airtemp.label().to_string(),
        use_spechumd: cfg.use_spechumd,
        oned_underflow: cfg.oned_underflow.label().to_string(),
        ..Default::default()
    };

    std::fs::create_dir_all(&cfg.out_dir)?;

    for interval in &cfg.intervals {
        let one = build_one_time(cfg, &mesh, &metrics, &header, interval)?;
        receipt.intervals.push(one);
    }

    receipt.seconds = started.elapsed().as_secs_f64();
    Ok(receipt)
}

fn build_one_time(
    cfg: &LbcConfig,
    mesh: &MeshGeometry,
    metrics: &VerticalMetrics,
    header: &emit::HeaderSource,
    interval: &BoundaryInterval,
) -> MpasResult<IntervalReceipt> {
    let started = std::time::Instant::now();
    let n_cells = mesh.n_cells;
    let n_edges = mesh.n_edges;
    let nz = metrics.n_vert_levels;

    let mut receipt = IntervalReceipt {
        valid_time: interval.valid_time.clone(),
        met_path: interval.met_path.display().to_string(),
        ..Default::default()
    };

    let slabs = read_met_file(&interval.met_path)?;
    let mut fg = horizontal_pass(&slabs, mesh, cfg, &mut receipt)?;
    let nfg = fg.vert_level.len();
    if nfg < 2 {
        return Err(MpasError::Refusal(format!(
            "{} delivered {nfg} first-guess level(s) among the fields the lbc case consumes; \
             a column cannot be interpolated from fewer than two",
            interval.met_path.display()
        )));
    }

    // Isobaric vs model-level, decided the way the Fortran decides it: on
    // whether ANY pressure content arrived.
    let isobaric = fg
        .p
        .iter()
        .all(|col| col[..nfg].iter().all(|&v| v == 0.0));
    if isobaric {
        receipt.pressure_branch = "isobaric".to_string();
        for c in 0..n_cells {
            for (k, &lvl) in fg.vert_level.iter().enumerate() {
                if lvl != SURFACE_LEVEL_TAG {
                    fg.p[c][k] = lvl;
                }
            }
        }
    } else {
        receipt.pressure_branch = "model-level".to_string();
        for k in 0..nfg {
            if fg.vert_level[k] == SURFACE_LEVEL_TAG {
                for c in 0..n_cells {
                    fg.z[c][k] = fg.soilz[c];
                }
            }
        }
    }

    // Rotate the first-guess winds onto the edge normal, in place, exactly as
    // the Fortran overwrites its own fg % u.
    for e in 0..n_edges {
        let angle = mesh.angle_edge[e];
        let (sin, cos) = (angle.sin(), angle.cos());
        for k in 0..nfg {
            fg.u[e][k] = cos * fg.u[e][k] + sin * fg.v[e][k];
        }
    }

    // Vertical interpolation on cells: t, relhum, spechum, pressure.
    let zgrid = &metrics.zgrid;
    let extrap_t = cfg.extrap_airtemp;
    type CellCol = (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>);
    let columns: Vec<MpasResult<CellCol>> = (0..n_cells)
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
                            "temperature extrapolation above the first-guess column top under \
                             the lapse-rate mode at cell {c} level {k}.  This is the Fortran's \
                             own fatal error in init_atm_case_lbc, not a port limitation"
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

            // The surface pressure slot is given `1.0` (a value whose log is
            // valid) before the log; the sentinel then sorts it out of the
            // profile.  There is no PSFC in this case.
            prep(&mut coord, &mut value, &|k| {
                if fg.vert_level[k] == SURFACE_LEVEL_TAG {
                    1.0f32.ln()
                } else {
                    fg.p[c][k].ln()
                }
            });
            for k in 0..nz {
                let target = 0.5 * (zgrid[c][k] + zgrid[c][k + 1]);
                pressure[k] =
                    vertical_interp(target, &coord[..nfg - 1], &value[..nfg - 1], Extrap::Linear)
                        .unwrap_or(value[nfg - 2])
                        .exp();
            }

            Ok((t, relhum, spechum, pressure))
        })
        .collect();

    let mut t_col = Vec::with_capacity(n_cells);
    let mut rh_col = Vec::with_capacity(n_cells);
    let mut sh_col = Vec::with_capacity(n_cells);
    let mut p_col = Vec::with_capacity(n_cells);
    for c in columns {
        let (t, rh, sh, p) = c?;
        t_col.push(t);
        rh_col.push(rh);
        sh_col.push(sh);
        p_col.push(p);
    }

    // Edge wind columns: two-cell mean coordinate, four-point mean target,
    // LINEAR extrapolation — and the stored-0 garbage cell contributes zeros.
    let garbage_z = vec![0.0f32; nfg];
    let garbage_zgrid = vec![0.0f32; nz + 1];
    let zfg_of = |c: usize| -> &[f32] {
        if c >= 1 && c <= n_cells {
            &fg.z[c - 1]
        } else {
            &garbage_z
        }
    };
    let zgrid_of = |c: usize| -> &[f32] {
        if c >= 1 && c <= n_cells {
            &zgrid[c - 1]
        } else {
            &garbage_zgrid
        }
    };
    let one_cell_edges = mesh
        .cells_on_edge
        .iter()
        .filter(|&&[c1, c2]| c1 == 0 || c1 > n_cells || c2 == 0 || c2 > n_cells)
        .count();
    receipt.one_cell_edges = one_cell_edges;

    let u_edge: Vec<Vec<f32>> = (0..n_edges)
        .into_par_iter()
        .map(|e| {
            let [c1, c2] = mesh.cells_on_edge[e];
            let z1 = zfg_of(c1);
            let z2 = zfg_of(c2);
            let g1 = zgrid_of(c1);
            let g2 = zgrid_of(c2);
            let mut coord = vec![0.0f32; nfg];
            let mut value = vec![0.0f32; nfg];
            for k in 0..nfg {
                coord[k] = if fg.vert_level[k] == SURFACE_LEVEL_TAG {
                    SURFACE_SENTINEL
                } else {
                    0.5 * (z1[k] + z2[k])
                };
                value[k] = fg.u[e][k];
            }
            sorted_column(&mut coord, &mut value);
            (0..nz)
                .map(|k| {
                    let target = 0.25 * (g1[k] + g1[k + 1] + g2[k] + g2[k + 1]);
                    vertical_interp(target, &coord[..nfg - 1], &value[..nfg - 1], Extrap::Linear)
                        .unwrap_or(value[nfg - 2])
                })
                .collect()
        })
        .collect();

    // Water vapour: the spechum path only when it was asked for AND delivered.
    let sh_all_zero = sh_col
        .iter()
        .all(|col| col.iter().all(|&v| v == 0.0));
    let use_sh = cfg.use_spechumd && !sh_all_zero;
    let qv: Vec<Vec<f32>> = (0..n_cells)
        .into_par_iter()
        .map(|c| {
            (0..nz)
                .map(|k| {
                    if use_sh {
                        sh_col[c][k] / (1.0 - sh_col[c][k])
                    } else {
                        let rs = rslf(p_col[c][k], t_col[c][k]);
                        0.01 * rs * rh_col[c][k]
                    }
                })
                .collect()
        })
        .collect();

    // The dynamics chain, per cell.  `theta` is the reused `t` array of the
    // Fortran: temperature, then potential temperature, then theta_m.
    struct CellState {
        theta_m: Vec<f32>,
        rho_zz: Vec<f32>,
        iterations: Vec<u32>,
    }
    let states: Vec<CellState> = (0..n_cells)
        .into_par_iter()
        .map(|c| {
            let zz = &metrics.zz[c];
            let mut theta = vec![0.0f32; nz];
            let mut exner = vec![0.0f32; nz];
            let mut rho_zz = vec![0.0f32; nz];
            let mut pressure = p_col[c].clone();

            // Exner, theta, and the first rho_zz — with NO virtual factor,
            // which is where this case departs from case 7.
            for k in 0..nz {
                exner[k] = (pressure[k] / P0).powf(RGAS / CP);
                theta[k] = t_col[c][k] * (P0 / pressure[k]).powf(RGAS / CP);
                rho_zz[k] = pressure[k] / RGAS / (exner[k] * theta[k]);
                rho_zz[k] /= 1.0 + qv[c][k];
            }

            // Base state: dry isothermal at t0b.
            let mut ppb = vec![0.0f32; nz];
            let mut rb = vec![0.0f32; nz];
            let mut pp = vec![0.0f32; nz];
            let mut rr = vec![0.0f32; nz];
            for k in 0..nz {
                let ztemp = 0.5 * (zgrid[c][k + 1] + zgrid[c][k]);
                ppb[k] = P0 * (-GRAVITY * ztemp / (RGAS * T0B)).exp();
                let pb = (ppb[k] / P0).powf(RGAS / CP);
                rb[k] = ppb[k] / (RGAS * T0B);
                exner[k] = pb;
                pp[k] = 0.0;
                rr[k] = 0.0;
            }

            // Couple to the vertical metric.
            for k in 0..nz {
                rb[k] /= zz[k];
                rho_zz[k] /= zz[k];
                pp[k] = pressure[k] - ppb[k];
                rr[k] = rho_zz[k] - rb[k];
            }

            // Hydrostatic rebalance: level 1 seeded, then the 30-pass 1e-4 Pa
            // fixed point, all with the literal 1.61.
            let mut iterations = vec![0u32; nz];
            rho_zz[0] = ((pressure[0] / P0).powf(CV / CP)) * (P0 / RGAS)
                / (theta[0] * (1.0 + VIRTUAL_161 * qv[c][0]))
                / zz[0];
            rr[0] = rho_zz[0] - rb[0];
            for k in 1..nz {
                let mut it = 0u32;
                let mut p_check = 2.0 * 0.0001f32;
                while it < 30 && p_check > 0.0001 {
                    p_check = pp[k];
                    pp[k] = pp[k - 1]
                        - (metrics.fzm[k] * rr[k] + metrics.fzp[k] * rr[k - 1])
                            * GRAVITY
                            * metrics.dzu[k]
                        - (metrics.fzm[k] * rho_zz[k] * qv[c][k]
                            + metrics.fzp[k] * rho_zz[k - 1] * qv[c][k - 1])
                            * GRAVITY
                            * metrics.dzu[k];
                    pressure[k] = pp[k] + ppb[k];
                    exner[k] = (pressure[k] / P0).powf(RGAS / CP);
                    rho_zz[k] = pressure[k]
                        / RGAS
                        / (exner[k] * theta[k] * (1.0 + VIRTUAL_161 * qv[c][k]))
                        / zz[k];
                    rr[k] = rho_zz[k] - rb[k];
                    p_check = (p_check - pp[k]).abs();
                    it += 1;
                }
                iterations[k] = it;
            }

            // theta_m.  (The Fortran also decouples rho_p here; nothing this
            // case writes reads it, so it is not carried.)
            let mut theta_m = theta;
            for k in 0..nz {
                theta_m[k] *= 1.0 + VIRTUAL_161 * qv[c][k];
            }

            CellState {
                theta_m,
                rho_zz,
                iterations,
            }
        })
        .collect();

    let mut it_max = 0u32;
    let mut capped = 0usize;
    for s in &states {
        let mut hit = false;
        for &i in &s.iterations[1..] {
            it_max = it_max.max(i);
            if i >= 30 {
                hit = true;
            }
        }
        if hit {
            capped += 1;
        }
    }
    receipt.hydrostatic_iterations_max = it_max;
    receipt.hydrostatic_cells_hitting_the_cap = capped;

    // ru on every edge, garbage cells contributing zero density.
    let rz_of = |c: usize, k: usize| -> f32 {
        if c >= 1 && c <= n_cells {
            states[c - 1].rho_zz[k]
        } else {
            0.0
        }
    };
    let mut ru = vec![vec![0.0f32; nz]; n_edges];
    for e in 0..n_edges {
        let [c1, c2] = mesh.cells_on_edge[e];
        for k in 0..nz {
            ru[e][k] = u_edge[e][k] * 0.5 * (rz_of(c1, k) + rz_of(c2, k));
        }
    }

    // rw, accumulated per cell over its edges in stored order, then w.
    let nzp1 = nz + 1;
    let mut rw = vec![vec![0.0f32; nzp1]; n_cells];
    for c in 0..n_cells {
        for i in 0..mesh.n_edges_on_cell[c] {
            let e = mesh.edges_on_cell[c][i];
            if e == 0 || e > n_edges {
                return Err(MpasError::Refusal(format!(
                    "edgesOnCell names edge {e} at cell {} slot {i}, outside 1..{n_edges}; the \
                     reference cull stores no zeros in valid edgesOnCell slots, so this grid \
                     file is not one this producer understands",
                    c + 1
                )));
            }
            let e = e - 1;
            let side_one = mesh.cells_on_edge[e][0] == c + 1;
            for k in 1..nz {
                let flux = metrics.fzm[k] * ru[e][k] + metrics.fzp[k] * ru[e][k - 1];
                let metric =
                    metrics.fzm[k] * metrics.zz[c][k] + metrics.fzp[k] * metrics.zz[c][k - 1];
                if side_one {
                    rw[c][k] -= metric * metrics.zb[e][0][k] * flux;
                } else {
                    rw[c][k] += metric * metrics.zb[e][1][k] * flux;
                }
                if cfg.theta_adv_order == 3 {
                    let sign = if ru[e][k] >= 0.0 { 1.0f32 } else { -1.0f32 };
                    if side_one {
                        rw[c][k] +=
                            sign * cfg.coef_3rd_order * metric * metrics.zb3[e][0][k] * flux;
                    } else {
                        rw[c][k] -=
                            sign * cfg.coef_3rd_order * metric * metrics.zb3[e][1][k] * flux;
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

    // The lbc fields.
    let mut lbc_qv = Vec::with_capacity(n_cells * nz);
    let mut lbc_rho = Vec::with_capacity(n_cells * nz);
    let mut lbc_theta = Vec::with_capacity(n_cells * nz);
    for c in 0..n_cells {
        for k in 0..nz {
            lbc_qv.push(qv[c][k]);
            lbc_rho.push(states[c].rho_zz[k] * metrics.zz[c][k]);
            lbc_theta.push(states[c].theta_m[k] / (1.0 + VIRTUAL_161 * qv[c][k]));
        }
    }
    let mut lbc_u = Vec::with_capacity(n_edges * nz);
    for e in 0..n_edges {
        lbc_u.extend_from_slice(&u_edge[e][..nz]);
    }
    let mut lbc_w = Vec::with_capacity(n_cells * nzp1);
    for c in 0..n_cells {
        lbc_w.extend_from_slice(&w[c][..nzp1]);
    }

    let time_seconds = seconds_between(&cfg.start_time, &interval.valid_time)?;
    receipt.time_seconds = time_seconds as f64;

    let out_path = cfg.out_dir.join(emit::lbc_file_name(&interval.valid_time));
    // qc and qr are computed-as-zero: `init_atm_case_lbc` zeroes lbc_scalars
    // at entry and only ever fills the qv slice.
    let zeros = vec![0.0f32; n_cells * nz];
    let fields = emit::LbcFields {
        qv: &lbc_qv,
        qc: &zeros,
        qr: &zeros,
        u: &lbc_u,
        w: &lbc_w,
        rho: &lbc_rho,
        theta: &lbc_theta,
    };
    emit::write_lbc_file(
        &out_path,
        cfg,
        header,
        n_cells,
        n_edges,
        nz,
        &interval.valid_time,
        time_seconds as f32,
        &fields,
    )?;
    receipt.out_path = out_path.display().to_string();
    receipt.out_sha256 = crate::sha256_file(&out_path)?;
    receipt.seconds = started.elapsed().as_secs_f64();
    Ok(receipt)
}

fn horizontal_pass(
    slabs: &[MetSlab],
    mesh: &MeshGeometry,
    cfg: &LbcConfig,
    receipt: &mut IntervalReceipt,
) -> MpasResult<FgLbc> {
    let n_cells = mesh.n_cells;
    let n_edges = mesh.n_edges;
    let nfg_cap = cfg.n_fg_levels;

    let mut fg = FgLbc {
        vert_level: Vec::new(),
        t: vec![vec![0.0; nfg_cap]; n_cells],
        rh: vec![vec![0.0; nfg_cap]; n_cells],
        sh: vec![vec![0.0; nfg_cap]; n_cells],
        z: vec![vec![0.0; nfg_cap]; n_cells],
        p: vec![vec![0.0; nfg_cap]; n_cells],
        u: vec![vec![0.0; nfg_cap]; n_edges],
        v: vec![vec![0.0; nfg_cap]; n_edges],
        soilz: vec![0.0; n_cells],
    };

    let cell_lat: Vec<f32> = mesh.lat_cell.iter().map(|v| v * DEG_PER_RAD).collect();
    let cell_lon: Vec<f32> = mesh.lon_cell.iter().map(|v| v * DEG_PER_RAD).collect();
    let edge_lat: Vec<f32> = mesh.lat_edge.iter().map(|v| v * DEG_PER_RAD).collect();
    let edge_lon: Vec<f32> = mesh.lon_edge.iter().map(|v| v * DEG_PER_RAD).collect();

    // Distinct levels among the consumed fields, counted past the cap the way
    // the Fortran keeps counting through its hash table, so the refusal can
    // name the config_nfglevels actually needed.
    let mut distinct_levels: std::collections::BTreeSet<u32> = std::collections::BTreeSet::new();
    let mut too_many = false;

    let underflow_hits = std::sync::atomic::AtomicU64::new(0);
    let methods: [Method; 2] = [Method::SixteenPoint, Method::Search];

    for s in slabs {
        receipt.met_records_read += 1;
        if !LBC_MET_FIELDS.contains(&s.field.as_str()) {
            if !receipt.met_records_ignored.contains(&s.field) {
                receipt.met_records_ignored.push(s.field.clone());
            }
            continue;
        }
        receipt.met_records_used += 1;

        let level_index = if s.field != "SOILHGT" {
            distinct_levels.insert(s.xlvl.to_bits());
            if distinct_levels.len() > nfg_cap {
                too_many = true;
            }
            if too_many {
                continue;
            }
            match fg.vert_level.iter().position(|&v| v == s.xlvl) {
                Some(i) => i,
                None => {
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
            maskval: None,
            mask: None,
            underflow: cfg.oned_underflow,
            underflow_hits: Some(&underflow_hits),
        };
        let sample = |i: usize, lat: &[f32], lon: &[f32]| -> f32 {
            let (x, y) = proj.locate(lat[i], lon[i]);
            interp_sequence(x, y, &ctx, &methods, 0)
        };

        let on_edges = s.field == "UU" || s.field == "VV";
        if on_edges {
            let values: Vec<f32> = (0..n_edges)
                .into_par_iter()
                .map(|e| sample(e, &edge_lat, &edge_lon))
                .collect();
            let dest = if s.field == "UU" { &mut fg.u } else { &mut fg.v };
            for (e, v) in values.into_iter().enumerate() {
                dest[e][level_index] = v;
            }
            continue;
        }

        let values: Vec<f32> = (0..n_cells)
            .into_par_iter()
            .map(|c| sample(c, &cell_lat, &cell_lon))
            .collect();
        match s.field.as_str() {
            "TT" => {
                for (c, v) in values.iter().enumerate() {
                    fg.t[c][level_index] = *v;
                }
            }
            "RH" => {
                for (c, v) in values.iter().enumerate() {
                    fg.rh[c][level_index] = *v;
                }
            }
            "SPECHUMD" => {
                for (c, v) in values.iter().enumerate() {
                    fg.sh[c][level_index] = *v;
                }
            }
            "GHT" => {
                for (c, v) in values.iter().enumerate() {
                    fg.z[c][level_index] = *v;
                }
            }
            "PRES" | "PRESSURE" => {
                for (c, v) in values.iter().enumerate() {
                    fg.p[c][level_index] = *v;
                }
            }
            "SOILHGT" => fg.soilz.copy_from_slice(&values),
            other => unreachable!("{other} passed the lbc field filter"),
        }
    }

    if too_many {
        return Err(MpasError::Refusal(format!(
            "the intermediate file holds {} distinct levels among the fields the lbc case \
             consumes, more than the declared {nfg_cap}; raise --nfglevels to at least {} and \
             re-run",
            distinct_levels.len(),
            distinct_levels.len()
        )));
    }

    receipt.oned_underflow_sites = underflow_hits.load(std::sync::atomic::Ordering::Relaxed);
    receipt.first_guess_levels_found = fg.vert_level.len();
    Ok(fg)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_gregorian_clock_agrees_with_known_intervals() {
        assert_eq!(
            seconds_between("2026-08-12_06:00:00", "2026-08-12_06:00:00").unwrap(),
            0
        );
        assert_eq!(
            seconds_between("2026-08-12_06:00:00", "2026-08-12_09:00:00").unwrap(),
            10800
        );
        assert_eq!(
            seconds_between("2026-08-12_06:00:00", "2026-08-13_06:00:00").unwrap(),
            86400
        );
        // Across the 2028 leap day.
        assert_eq!(
            seconds_between("2028-02-28_00:00:00", "2028-03-01_00:00:00").unwrap(),
            2 * 86400
        );
        // Across a non-leap February.
        assert_eq!(
            seconds_between("2026-02-28_00:00:00", "2026-03-01_00:00:00").unwrap(),
            86400
        );
        assert!(seconds_between("nope", "2026-08-12_09:00:00").is_err());
    }

    #[test]
    fn the_lbc_case_consumes_exactly_nine_met_fields() {
        // The dispatch difference from case 7 in one place: no PSFC, no PMSL,
        // no SKINTEMP, no SNOW, no SEAICE, no soil fields.
        for f in ["PSFC", "PMSL", "SKINTEMP", "SNOW", "SEAICE", "ST000010", "SM000010"] {
            assert!(!LBC_MET_FIELDS.contains(&f), "{f} must not be consumed");
        }
        for f in ["UU", "VV", "TT", "RH", "SPECHUMD", "GHT", "SOILHGT", "PRES", "PRESSURE"] {
            assert!(LBC_MET_FIELDS.contains(&f), "{f} must be consumed");
        }
    }

    #[test]
    fn the_virtual_factor_here_is_the_literal_not_rvord() {
        use crate::init::dynamics::constants::RVORD;
        assert_eq!(VIRTUAL_161, 1.61f32);
        assert_ne!(VIRTUAL_161, RVORD - 1.0, "1.61 is not rvord - 1, and the difference is the point");
    }
}
