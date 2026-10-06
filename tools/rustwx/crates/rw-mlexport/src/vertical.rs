//! Mass levels onto pressure levels, column by column.
//!
//! Inside the column: linear in ln p between the two mass levels that
//! bracket the target, with the renderer's own walk (`rw_isobaric::bracket`).
//!
//! Below the lowest mass level: the ECMWF rule ERA5's pressure levels are
//! filled with (`rw_isobaric::ecmwf_temperature` and `ecmwf_geopotential`);
//! every other field takes the lowest mass level's value.  The formulas run
//! from the lowest mass level down, as NCL and GeoCAT apply them, and the
//! `below_ground` mask marks exactly the points under the surface (target
//! pressure above the surface pressure).
//!
//! Above the top mass level: a history file with P_TOP has model column up
//! to that lid, half a layer above the top mass level; there geopotential is
//! linear in ln p from the top mass level to the lid's own geopotential (the
//! top face of PH + PHB, exact at P_TOP), temperature and every other field
//! are held.  A level above the lid has no model data and is dropped from
//! the dataset before any frame is read (`kept_levels`).  A file without
//! P_TOP (hex frames, global tapes) has its lid at the top mass level; a
//! later frame whose column falls a little short of a kept level holds the
//! top values and extends geopotential hydrostatically.
//!
//! Geopotential is the exception to reading mass levels: a WRF-family model
//! knows it on the interfaces of its layers (PH + PHB), and a layer-mean
//! geopotential paired with the mass-level pressure reads 4 to 6 m high at
//! 500 hPa on an eta grid.  Where the history file states its eta levels
//! (ZNW), geopotential inside the model column is read between interfaces
//! instead ([`Plan::apply_geopotential_between_interfaces`]); below ground
//! the ECMWF rule is unchanged.

use rayon::prelude::*;
use rw_isobaric::{
    bracket, ecmwf_geopotential, ecmwf_temperature, interface_bracket, lerp, ColumnBase,
    InterfaceStencil, ECMWF_RULE,
};

use crate::error::{refuse, Result};

/// Dry-air gas constant for the isothermal extension above the top level.
const RD: f64 = 287.04;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Lid {
    /// The history file states P_TOP (Pa).
    PTop(f64),
    /// No P_TOP: the top mass level is the highest point with data.
    TopMassLevel,
}

/// The below-ground rule a level variable is filled with.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Rule {
    Temperature,
    Geopotential,
    Lowest,
}

impl Rule {
    pub fn parse(text: Option<&str>) -> Result<Rule> {
        match text.unwrap_or("lowest-level") {
            "ecmwf-temperature" => Ok(Rule::Temperature),
            "ecmwf-geopotential" => Ok(Rule::Geopotential),
            "lowest-level" => Ok(Rule::Lowest),
            other => Err(refuse(format!(
                "below-ground rule '{other}' is not one of ecmwf-temperature, ecmwf-geopotential, lowest-level, so the points under the surface could not be filled"
            ))),
        }
    }
}

/// Which pressure levels the dataset carries, decided from the first frame
/// and fixed for the export: a level above the lid would be a whole level
/// of fill, which poisons the per-level normalisation every ML loader
/// computes.  Returns (kept, dropped), each in request order.
pub fn kept_levels(hpa: &[u32], lid: Lid, p: &[f64], nz: usize, cells: usize) -> (Vec<u32>, Vec<u32>) {
    let lid_pa = match lid {
        Lid::PTop(top) => top,
        Lid::TopMassLevel => {
            let top = &p[(nz - 1) * cells..nz * cells];
            top.iter().copied().filter(|v| v.is_finite()).fold(f64::MIN, f64::max)
        }
    };
    hpa.iter().partition(|&&level| f64::from(level) * 100.0 >= lid_pa - 1e-6)
}

/// The per-column facts the rules read, gathered once per frame.
pub struct Columns<'a> {
    pub nz: usize,
    pub cells: usize,
    /// Pressure on mass levels (Pa), `[nz, cells]`.
    pub p: &'a [f64],
    /// Surface pressure (Pa).
    pub psfc: &'a [f64],
    /// Surface geopotential (m2 s-2): the bottom face of PH + PHB.
    pub phi_sfc: &'a [f64],
    /// Temperature of the lowest mass level (K).
    pub t_bot: &'a [f64],
    /// Temperature of the top mass level (K).
    pub t_top: &'a [f64],
    /// Geopotential of the top mass level (m2 s-2).
    pub phi_top: &'a [f64],
    /// Geopotential of the top face (m2 s-2), the lid's own.
    pub phi_lid: &'a [f64],
    pub lid: Lid,
}

/// Where each target level sits in each column, for every variable of the
/// frame: interior (lower level and weight), below the lowest level, in the
/// layer above the top level, or no data.
pub struct Plan {
    pub levels_pa: Vec<f64>,
    pub cells: usize,
    kind: Vec<u8>,
    lower: Vec<u16>,
    weight: Vec<f64>,
}

const INTERIOR: u8 = 0;
const BELOW: u8 = 1;
const ABOVE: u8 = 2;
const NONE: u8 = 3;

impl Plan {
    pub fn new(levels_hpa: &[u32], cols: &Columns<'_>) -> Plan {
        let levels_pa: Vec<f64> = levels_hpa.iter().map(|&h| f64::from(h) * 100.0).collect();
        let (nz, cells) = (cols.nz, cols.cells);
        let nl = levels_pa.len();
        let mut kind = vec![NONE; nl * cells];
        let mut lower = vec![0u16; nl * cells];
        let mut weight = vec![0.0f64; nl * cells];
        // Blocks of columns in parallel, each with one reused column buffer;
        // the results are scattered into level-major storage.
        const BLOCK: usize = 4096;
        let starts: Vec<usize> = (0..cells).step_by(BLOCK).collect();
        let blocks: Vec<(usize, usize, Vec<u8>, Vec<u16>, Vec<f64>)> = starts
            .into_par_iter()
            .map(|start| {
                let end = (start + BLOCK).min(cells);
                let n = end - start;
                let mut b_kind = vec![NONE; nl * n];
                let mut b_lower = vec![0u16; nl * n];
                let mut b_weight = vec![0.0f64; nl * n];
                let mut col = vec![0.0f64; nz];
                for c in start..end {
                    for (k, slot) in col.iter_mut().enumerate() {
                        *slot = cols.p[k * cells + c];
                    }
                    let (p_bot, p_top) = (col[0], col[nz - 1]);
                    for (l, &target) in levels_pa.iter().enumerate() {
                        let i = l * n + (c - start);
                        if !(p_bot.is_finite() && p_top.is_finite()) {
                            continue;
                        }
                        if target > p_bot {
                            b_kind[i] = BELOW;
                        } else if let Some((k, w)) = bracket(&col, target) {
                            b_kind[i] = INTERIOR;
                            b_lower[i] = k as u16;
                            b_weight[i] = w;
                        } else if target < p_top {
                            b_kind[i] = ABOVE;
                        }
                    }
                }
                (start, n, b_kind, b_lower, b_weight)
            })
            .collect();
        for (start, n, b_kind, b_lower, b_weight) in blocks {
            for l in 0..nl {
                let (to, from) = (l * cells + start, l * n);
                kind[to..to + n].copy_from_slice(&b_kind[from..from + n]);
                lower[to..to + n].copy_from_slice(&b_lower[from..from + n]);
                weight[to..to + n].copy_from_slice(&b_weight[from..from + n]);
            }
        }
        Plan { levels_pa, cells, kind, lower, weight }
    }

    /// 1 where the level lies under the surface, else 0, `[levels, cells]`.
    pub fn below_ground(&self, psfc: &[f64]) -> Vec<u8> {
        let cells = self.cells;
        let mut mask = vec![0u8; self.levels_pa.len() * cells];
        for (l, &target) in self.levels_pa.iter().enumerate() {
            for c in 0..cells {
                mask[l * cells + c] = u8::from(target > psfc[c]);
            }
        }
        mask
    }

    /// Put one mass-level field (`[nz, cells]`) on the plan's levels, filling
    /// with `rule` below ground.  Returns `[levels, cells]` as f32.
    pub fn apply(&self, field: &[f64], rule: Rule, cols: &Columns<'_>) -> Vec<f32> {
        let cells = cols.cells;
        let mut out = vec![f32::NAN; self.levels_pa.len() * cells];
        out.par_chunks_mut(cells).enumerate().for_each(|(l, plane)| {
            for (c, slot) in plane.iter_mut().enumerate() {
                *slot = self.mass_level_value(field, rule, cols, l, c) as f32;
            }
        });
        out
    }

    /// Put geopotential on the plan's levels, reading it between the layer
    /// interfaces inside the model column.
    ///
    /// `phi_stag` is PH + PHB on the `nz + 1` interfaces, `phi_mass` its
    /// layer means (what [`Plan::apply`] reads), `stencil` where each
    /// interface sits between the mass levels.  Inside the column (the
    /// plan's interior and the layer under the lid) a level is interpolated
    /// in ln p between the two interfaces that straddle it, each
    /// interface's pressure linear in eta between the mass-level pressures
    /// and the top one at P_TOP when the file states it.  Below ground, and
    /// in a column whose interfaces cannot place the level, the value is
    /// the one [`Plan::apply`] gives.
    pub fn apply_geopotential_between_interfaces(
        &self,
        phi_stag: &[f64],
        phi_mass: &[f64],
        stencil: &InterfaceStencil,
        rule: Rule,
        cols: &Columns<'_>,
    ) -> Vec<f32> {
        let (nz, cells) = (cols.nz, cols.cells);
        let nl = self.levels_pa.len();
        let targets: Vec<f64> = self.levels_pa.iter().map(|p| rw_isobaric::ln(*p)).collect();
        let lid_ln = match cols.lid {
            Lid::PTop(lid) if lid.is_finite() && lid > 0.0 => Some(rw_isobaric::ln(lid)),
            _ => None,
        };
        let mut out = vec![f32::NAN; nl * cells];
        // Blocks of columns in parallel, each column's interface pressures
        // found once for every level; scattered into level-major storage.
        const BLOCK: usize = 4096;
        let starts: Vec<usize> = (0..cells).step_by(BLOCK).collect();
        let blocks: Vec<(usize, usize, Vec<f32>)> = starts
            .into_par_iter()
            .map(|start| {
                let end = (start + BLOCK).min(cells);
                let n = end - start;
                let mut block = vec![f32::NAN; nl * n];
                let mut ln_p = vec![0.0f64; nz + 1];
                for c in start..end {
                    let usable =
                        stencil.interface_ln_pressures(|k| cols.p[k * cells + c], &mut ln_p);
                    // The model lid is exactly P_TOP when the file says so.
                    if let (true, Some(lid)) = (usable, lid_ln) {
                        if lid < ln_p[nz - 1] {
                            ln_p[nz] = lid;
                        }
                    }
                    for (l, &target) in targets.iter().enumerate() {
                        let inside = matches!(self.kind[l * cells + c], INTERIOR | ABOVE);
                        let between = if usable && inside {
                            interface_bracket(&ln_p, target).and_then(|(below, fraction)| {
                                let lower = phi_stag[below * cells + c];
                                let upper = phi_stag[(below + 1) * cells + c];
                                (lower.is_finite() && upper.is_finite())
                                    .then(|| lower + (upper - lower) * fraction)
                            })
                        } else {
                            None
                        };
                        let value = match between {
                            Some(value) => value,
                            None => self.mass_level_value(phi_mass, rule, cols, l, c),
                        };
                        block[l * n + (c - start)] = value as f32;
                    }
                }
                (start, n, block)
            })
            .collect();
        for (start, n, block) in blocks {
            for l in 0..nl {
                let to = l * cells + start;
                out[to..to + n].copy_from_slice(&block[l * n..(l + 1) * n]);
            }
        }
        out
    }

    /// One level of one column from the mass levels, by the plan.
    fn mass_level_value(&self, field: &[f64], rule: Rule, cols: &Columns<'_>, l: usize, c: usize) -> f64 {
        let (nz, cells) = (cols.nz, cols.cells);
        let target = self.levels_pa[l];
        let i = l * cells + c;
        match self.kind[i] {
            INTERIOR => {
                let k = self.lower[i] as usize;
                lerp(field[k * cells + c], field[(k + 1) * cells + c], self.weight[i]).unwrap_or(f64::NAN)
            }
            BELOW => {
                let base = ColumnBase {
                    t_bot: cols.t_bot[c],
                    p_bot: cols.p[c],
                    p_sfc: cols.psfc[c],
                    phi_sfc: cols.phi_sfc[c],
                };
                match rule {
                    Rule::Temperature => ecmwf_temperature(&base, target, &ECMWF_RULE),
                    Rule::Geopotential => ecmwf_geopotential(&base, target, &ECMWF_RULE),
                    Rule::Lowest => field[c],
                }
            }
            ABOVE => {
                let top = field[(nz - 1) * cells + c];
                let p_top = cols.p[(nz - 1) * cells + c];
                match rule {
                    Rule::Geopotential => match cols.lid {
                        Lid::PTop(lid) if lid < p_top => {
                            let w = (target.ln() - p_top.ln()) / (lid.ln() - p_top.ln());
                            top + w * (cols.phi_lid[c] - top)
                        }
                        _ => top - RD * cols.t_top[c] * (target / p_top).ln(),
                    },
                    _ => top,
                }
            }
            _ => f64::NAN,
        }
    }
}

/// Model levels as they are: `[nz, cells]` cut to the selected levels (1 at
/// the bottom), as f32.
pub fn select_model_levels(field: &[f64], cells: usize, levels: &[usize]) -> Vec<f32> {
    let mut out = Vec::with_capacity(levels.len() * cells);
    for &level in levels {
        let k = level - 1;
        out.extend(field[k * cells..(k + 1) * cells].iter().map(|&v| v as f32));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A two-column, five-level atmosphere: column 0 at sea level, column 1
    /// on a 1500 m plateau.
    fn columns() -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let p0 = [100_500.0, 95_000.0, 85_000.0, 50_000.0, 10_000.0];
        let p1 = [84_000.0, 80_000.0, 70_000.0, 45_000.0, 9_800.0];
        let mut p = Vec::new();
        for k in 0..5 {
            p.push(p0[k]);
            p.push(p1[k]);
        }
        let t: Vec<f64> = p.iter().map(|&pk| 288.0 * (pk / 101_325.0f64).powf(0.19)).collect();
        let psfc = vec![101_000.0, 84_400.0];
        let phi_sfc = vec![0.0, 9.80616 * 1500.0];
        // Geopotential rising with -Rd T dlnp (the test only needs monotone).
        let phi: Vec<f64> = p.iter().enumerate().map(|(i, &pk)| {
            let c = i % 2;
            phi_sfc[c] + 287.0 * 260.0 * (psfc[c] / pk).ln()
        }).collect();
        (p, t, psfc, phi_sfc, phi)
    }

    #[test]
    fn interior_targets_match_the_renderer_walk_exactly() {
        let (p, t, psfc, phi_sfc, phi) = columns();
        let t_bot = vec![t[0], t[1]];
        let t_top = vec![t[8], t[9]];
        let phi_top = vec![phi[8], phi[9]];
        let cols = Columns { nz: 5, cells: 2, p: &p, psfc: &psfc, phi_sfc: &phi_sfc, t_bot: &t_bot, t_top: &t_top, phi_top: &phi_top, phi_lid: &phi_top, lid: Lid::TopMassLevel };
        let plan = Plan::new(&[700], &cols);
        let out = plan.apply(&t, Rule::Temperature, &cols);
        for c in 0..2 {
            let col: Vec<f64> = (0..5).map(|k| p[k * 2 + c]).collect();
            let (k, w) = bracket(&col, 70_000.0).unwrap();
            let expected = lerp(t[k * 2 + c], t[(k + 1) * 2 + c], w).unwrap() as f32;
            assert_eq!(out[c], expected);
        }
    }

    #[test]
    fn below_ground_points_take_the_ecmwf_rules_and_the_mask() {
        let (p, t, psfc, phi_sfc, phi) = columns();
        let t_bot = vec![t[0], t[1]];
        let t_top = vec![t[8], t[9]];
        let phi_top = vec![phi[8], phi[9]];
        let cols = Columns { nz: 5, cells: 2, p: &p, psfc: &psfc, phi_sfc: &phi_sfc, t_bot: &t_bot, t_top: &t_top, phi_top: &phi_top, phi_lid: &phi_top, lid: Lid::TopMassLevel };
        let plan = Plan::new(&[1000, 850], &cols);
        let mask = plan.below_ground(&psfc);
        // 1000 hPa: above ground at sea level, under the plateau.
        assert_eq!(&mask[0..2], &[0, 1]);
        // 850 hPa: above the sea-level surface, and under the plateau's
        // (85 000 Pa against a surface pressure of 84 400 Pa).
        assert_eq!(&mask[2..4], &[0, 1]);
        let temperature = plan.apply(&t, Rule::Temperature, &cols);
        let base = ColumnBase { t_bot: t[1], p_bot: p[1], p_sfc: psfc[1], phi_sfc: phi_sfc[1] };
        assert_eq!(temperature[1], ecmwf_temperature(&base, 100_000.0, &ECMWF_RULE) as f32);
        let height = plan.apply(&phi, Rule::Geopotential, &cols);
        assert_eq!(height[1], ecmwf_geopotential(&base, 100_000.0, &ECMWF_RULE) as f32);
        // Other fields hold the lowest level.
        let wind: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let held = plan.apply(&wind, Rule::Lowest, &cols);
        assert_eq!(held[1], 1.0);
        // The sea-level column between the lowest level (100 500 Pa) and the
        // surface (101 000 Pa) is not under ground and not flagged; the
        // 1000 hPa level there is interior.
        assert!(temperature[0].is_finite());
    }

    #[test]
    fn levels_above_the_lid_are_dropped_and_the_layer_under_it_is_filled() {
        let (p, t, psfc, phi_sfc, phi) = columns();
        let (kept, dropped) = kept_levels(&[50, 100, 500], Lid::PTop(9_000.0), &p, 5, 2);
        assert_eq!(kept, vec![100, 500]);
        assert_eq!(dropped, vec![50]);
        let (kept, dropped) = kept_levels(&[50, 98, 100], Lid::TopMassLevel, &p, 5, 2);
        assert_eq!(kept, vec![100]);
        assert_eq!(dropped, vec![50, 98]);
        // 95 hPa is above both columns' top mass levels (10 000 and
        // 9 800 Pa) and under the 9 000 Pa lid.
        let t_bot = vec![t[0], t[1]];
        let t_top = vec![t[8], t[9]];
        let phi_top = vec![phi[8], phi[9]];
        let phi_lid = vec![phi[8] + 1000.0, phi[9] + 1000.0];
        let cols = Columns { nz: 5, cells: 2, p: &p, psfc: &psfc, phi_sfc: &phi_sfc, t_bot: &t_bot, t_top: &t_top, phi_top: &phi_top, phi_lid: &phi_lid, lid: Lid::PTop(9_000.0) };
        let plan = Plan::new(&[95], &cols);
        let z = plan.apply(&phi, Rule::Geopotential, &cols);
        let w = (9_500f64.ln() - 10_000f64.ln()) / (9_000f64.ln() - 10_000f64.ln());
        assert_eq!(z[0], (phi[8] + w * 1000.0) as f32);
        let temperature = plan.apply(&t, Rule::Temperature, &cols);
        assert_eq!(temperature[0], t[8] as f32);
    }

    #[test]
    fn model_level_selection_is_one_based_from_the_bottom() {
        let field: Vec<f64> = (0..10).map(f64::from).collect();
        assert_eq!(select_model_levels(&field, 2, &[1, 5]), vec![0.0, 1.0, 8.0, 9.0]);
    }
}
