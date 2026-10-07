//! `definitions = renderer` (the front door's `--definitions renderer`, and
//! its alias `arwen`): the rows the product renderer draws are computed by
//! the renderer's own diagnostics, so the GRIB2 numbers are the numbers on
//! the maps.
//!
//! The renderer (`rw_wrfbatch`) computes every map field through wrf-core's
//! `getvar`; [`rw_wrfbatch::wrf_process::compute_var`] is that exact call,
//! with the renderer's per-field panic isolation.  This module is a table
//! from catalog ids to `getvar` names, units and a component index, read
//! as data; a row the renderer has no diagnostic for keeps its WOOF
//! (`rw-post`) plane, and the manifest names which definitions produced
//! each field.  GRIB2 identities are the catalog's, so a decoder finds the
//! same parameter under either definitions.
//!
//! Not in the table, so always WOOF's: vector components (the catalog's are
//! grid-relative unless `winds = earth`, the renderer's earth-relative), the
//! pressure-level block, reflectivity at 1 km (the renderer forms it from
//! its own volume, not a `getvar` name), and every row the renderer does
//! not draw.

use std::collections::BTreeMap;
use std::path::Path;

use crate::{fail, Result};

/// One renderer-computed row.
#[derive(Clone, Copy, Debug)]
pub struct RendererRow {
    /// Catalog id the plane replaces.
    pub id: &'static str,
    /// wrf-core `getvar` name, as the renderer asks for it.
    pub var: &'static str,
    /// Component of a stacked output (`cloudfrac` low, mid, high).
    pub split: Option<usize>,
    /// Units asked of wrf-core, so the plane is in the catalog's unit.
    pub units: Option<&'static str>,
    /// Catalog sign convention: CIN is written negative (catalog and NCEP
    /// convention), whatever sign the diagnostic returns.
    pub negative: bool,
}

const fn r(id: &'static str, var: &'static str, units: Option<&'static str>) -> RendererRow {
    RendererRow { id, var, split: None, units, negative: false }
}

const fn cin(id: &'static str, var: &'static str) -> RendererRow {
    RendererRow { id, var, split: None, units: Some("J/kg"), negative: true }
}

const fn part(id: &'static str, var: &'static str, split: usize, units: Option<&'static str>) -> RendererRow {
    RendererRow { id, var, split: Some(split), units, negative: false }
}

/// The renderer's rows.  Units: catalog units; `pw` is mm of water, which
/// is kg m-2.
pub const ROWS: &[RendererRow] = &[
    r("terrain", "terrain", Some("m")),
    r("surface_pressure", "PSFC", Some("Pa")),
    r("t2", "t2", Some("K")),
    r("td2", "dp2m", Some("K")),
    r("rh2", "rh2m", Some("%")),
    r("mslp", "slp", Some("Pa")),
    r("pwat", "pw", Some("mm")),
    r("sbcape", "sbcape", Some("J/kg")),
    cin("sbcin", "sbcin"),
    r("mlcape", "mlcape", Some("J/kg")),
    cin("mlcin", "mlcin"),
    r("mucape", "mucape", Some("J/kg")),
    cin("mucin", "mucin"),
    r("lcl_height", "lcl", Some("m")),
    r("srh_0_1km", "srh1", None),
    r("srh_0_3km", "srh3", None),
    r("bulk_shear_0_1km", "shear_0_1km", Some("m/s")),
    r("bulk_shear_0_6km", "shear_0_6km", Some("m/s")),
    r("stp", "stp_fixed", None),
    r("ehi_0_1km", "ehi", None),
    r("composite_reflectivity", "maxdbz", Some("dBZ")),
    part("low_cloud", "cloudfrac", 0, Some("%")),
    part("mid_cloud", "cloudfrac", 1, Some("%")),
    part("high_cloud", "cloudfrac", 2, Some("%")),
    r("pbl_height", "PBLH", None),
];

/// The renderer's missing-value cleaning (`rw_wrfbatch`'s plane reader):
/// non-finite values, the 1e30 fill and the -9999 family are missing.
fn clean(v: f64) -> f32 {
    if !v.is_finite() || v.abs() >= 1.0e30 || v <= -9998.0 { f32::NAN } else { v as f32 }
}

/// Planes the renderer computed for one frame, by catalog id, and the
/// rows it could not compute with the reason (those keep WOOF's plane).
pub struct Output {
    pub planes: BTreeMap<&'static str, Vec<f32>>,
    pub kept: Vec<(&'static str, String)>,
}

/// Compute the renderer's rows for the history at `path` (one frame).
pub fn compute(path: &Path, n: usize) -> Result<Output> {
    let file = wrf_core::WrfFile::open(path).map_err(|e| fail(format!("the renderer could not open {}: {e}", path.display())))?;
    let mut stacked: BTreeMap<&'static str, std::result::Result<Vec<f64>, String>> = BTreeMap::new();
    let mut planes = BTreeMap::new();
    let mut kept = Vec::new();
    for row in ROWS {
        let out = stacked.entry(row.var).or_insert_with(|| {
            rw_wrfbatch::wrf_process::compute_var(&file, row.var, 0, row.units).map(|o| o.data)
        });
        let data = match out {
            Ok(d) => d,
            Err(e) => {
                kept.push((row.id, format!("renderer {} unavailable ({e})", row.var)));
                continue;
            }
        };
        let k = row.split.unwrap_or(0);
        if data.len() < (k + 1) * n {
            kept.push((row.id, format!("renderer {} returned {} values for {n} cells", row.var, data.len())));
            continue;
        }
        let plane: Vec<f32> = data[k * n..(k + 1) * n]
            .iter()
            .map(|&v| {
                let x = clean(v);
                if row.negative && x.is_finite() { -x.abs() } else { x }
            })
            .collect();
        planes.insert(row.id, plane);
    }
    Ok(Output { planes, kept })
}

/// Manifest text for a row computed by the renderer.
pub fn method(row: &RendererRow) -> String {
    let what = match row.split {
        Some(k) => format!("{}[{k}]", row.var),
        None => row.var.to_owned(),
    };
    let sign = if row.negative { "; written negative (catalog CIN convention)" } else { "" };
    format!("renderer definitions: rw_wrfbatch / wrf-core getvar {what}, the diagnostic the product maps draw{sign}")
}

/// The table row of a catalog id.
pub fn row(id: &str) -> Option<&'static RendererRow> {
    ROWS.iter().find(|r| r.id == id)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_renderer_row_names_a_catalog_row_once() {
        let ids: Vec<&str> = crate::catalog::surface_rows().iter().map(|r| r.id).collect();
        let mut seen = std::collections::BTreeSet::new();
        for row in ROWS {
            assert!(ids.contains(&row.id), "{} is not a catalog row", row.id);
            assert!(seen.insert(row.id), "{} twice", row.id);
            let cat = crate::catalog::surface_rows().into_iter().find(|r| r.id == row.id).unwrap();
            assert!(cat.wind.is_none(), "{}: vector rows stay WOOF's (frame convention)", row.id);
        }
    }

    #[test]
    fn every_getvar_name_exists_in_wrf_core() {
        for row in ROWS {
            let known = wrf_core::variables::VARS.iter().any(|v| v.name == row.var || v.aliases.contains(&row.var));
            let raw = row.var.chars().all(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || c == '_');
            assert!(known || raw, "{} is neither a wrf-core diagnostic nor a raw history variable", row.var);
        }
    }

    #[test]
    fn cleaning_and_cin_sign() {
        assert!(clean(1.0e30).is_nan());
        assert!(clean(-9999.0).is_nan());
        assert_eq!(clean(12.5), 12.5);
        assert!(row("sbcin").unwrap().negative);
        assert!(method(row("sbcin").unwrap()).contains("negative"));
        assert!(method(row("low_cloud").unwrap()).contains("cloudfrac[0]"));
    }
}
