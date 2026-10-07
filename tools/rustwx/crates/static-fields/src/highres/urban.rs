//! Estimate the built urban fraction from land-cover area fractions.
//!
//! LANDUSEF measures each mapped category's area on the model grid. The
//! coefficient for that category is URBPARM's FRC_URB for its urban type.
//! Their weighted sum is an estimate of built area, not an observed
//! impervious-surface product. WRF's dominant-urban mask is retained.

use crate::error::{Result, StaticError};
use crate::types::{Field, FieldSet, Grid2};

pub const ALGORITHM: &str = "landcover-area-urbparm-weighted-v1";

#[derive(Debug, serde::Serialize)]
pub struct UrbanFractionAudit {
    pub schema: &'static str,
    pub algorithm: String,
    pub field: &'static str,
    pub interpretation: String,
    pub dominant_urban_cells: usize,
    pub cells_with_urban_cover: usize,
    pub urban_cover_outside_dominant_cells: usize,
    pub built_fraction_sum: f64,
    pub built_fraction_min: f64,
    pub built_fraction_max: f64,
    pub category_built_fractions: Vec<(usize, f64)>,
    pub source_category_built_fractions: Option<Vec<(usize, f64)>>,
    pub source_covered_urban_cells: usize,
    pub source_blended_urban_cells: usize,
    pub source_uncovered_urban_cells: usize,
}

pub fn estimate(
    fields: &FieldSet,
    category_built_fractions: &[(usize, f64)],
) -> Result<(Grid2, UrbanFractionAudit)> {
    estimate_with_source(fields, category_built_fractions, None, None, None)
}

/// A source can supply distinct built-area classes while LANDUSEF retains
/// the canopy morphology types. Coverage weights use the same blend as
/// the land-cover overlay; uncovered cells retain the table estimate.
pub fn estimate_with_source(
    fields: &FieldSet,
    category_built_fractions: &[(usize, f64)],
    source_category_built_fractions: Option<&[(usize, f64)]>,
    source_algorithm: Option<&str>,
    source_interpretation: Option<&str>,
) -> Result<(Grid2, UrbanFractionAudit)> {
    let invalid = |s: &str| StaticError::Invalid(s.into());
    let luf = match fields.get("LANDUSEF")? {
        Field::Stack(s) => s,
        _ => return Err(invalid("urban fraction needs categorical LANDUSEF")),
    };
    let lu = match fields.get("LU_INDEX")? {
        Field::Plane(s) => s,
        _ => return Err(invalid("urban fraction needs a LU_INDEX plane")),
    };
    let land = match fields.get("LANDMASK")? {
        Field::Plane(s) => s,
        _ => return Err(invalid("urban fraction needs a LANDMASK plane")),
    };
    let n = luf.ny * luf.nx;
    if (lu.ny, lu.nx) != (luf.ny, luf.nx)
        || (land.ny, land.nx) != (luf.ny, luf.nx)
        || luf.data.len() != luf.planes * n
        || lu.data.len() != n || land.data.len() != n
    {
        return Err(invalid("urban fraction field shapes differ: land cover would be assigned to the wrong cells"));
    }
    let mut weights = vec![0.0; luf.planes];
    for &(category, weight) in category_built_fractions {
        if category == 0 || category > luf.planes
            || !weight.is_finite() || !(0.0..=1.0).contains(&weight)
        {
            return Err(invalid("urban category or built fraction is outside the selected LANDUSEF/URBPARM inventory"));
        }
        if weights[category - 1] != 0.0 {
            return Err(invalid("urban category repeats: its built area would be counted twice"));
        }
        weights[category - 1] = weight;
    }
    let source = if let Some(source_weights) = source_category_built_fractions {
        let fractions = match fields.get("URBAN_SOURCEF")? {
            Field::Stack(s) => s,
            _ => return Err(invalid("urban source fraction needs a categorical URBAN_SOURCEF")),
        };
        let coverage = match fields.get("URBAN_SOURCE_WEIGHT")? {
            Field::Plane(s) => s,
            _ => return Err(invalid("urban source fraction needs its coverage weight plane")),
        };
        if (fractions.ny, fractions.nx) != (luf.ny, luf.nx)
            || fractions.data.len() != fractions.planes * n
            || (coverage.ny, coverage.nx) != (luf.ny, luf.nx)
            || coverage.data.len() != n
        {
            return Err(invalid("urban source shapes differ: built area would be assigned to the wrong cells"));
        }
        let mut source_lookup = vec![0.0; fractions.planes];
        let mut seen = std::collections::BTreeSet::new();
        for &(category, weight) in source_weights {
            if category == 0 || category > fractions.planes
                || !weight.is_finite() || !(0.0..=1.0).contains(&weight)
                || !seen.insert(category)
            {
                return Err(invalid("urban source class or built fraction is invalid or repeated"));
            }
            source_lookup[category - 1] = weight;
        }
        if source_algorithm.is_none() || source_interpretation.is_none() {
            return Err(invalid("urban source fractions need their algorithm and interpretation"));
        }
        Some((fractions, coverage, source_lookup))
    } else {
        if source_algorithm.is_some() || source_interpretation.is_some()
            || fields.fields.contains_key("URBAN_SOURCEF")
            || fields.fields.contains_key("URBAN_SOURCE_WEIGHT")
        {
            return Err(invalid("urban source metadata or fields have no class fractions"));
        }
        None
    };
    let mut out = Grid2::filled(luf.ny, luf.nx, 0.0);
    let mut audit = UrbanFractionAudit {
        schema: "gpuwm-highres-urban-fraction-v1",
        algorithm: source_algorithm.unwrap_or(ALGORITHM).into(),
        field: "FRC_URB2D",
        interpretation: source_interpretation.unwrap_or("Estimate: mapped land-cover category area times its URBPARM built fraction, on dominant urban land cells; not observed imperviousness").into(),
        dominant_urban_cells: 0,
        cells_with_urban_cover: 0,
        urban_cover_outside_dominant_cells: 0,
        built_fraction_sum: 0.0,
        built_fraction_min: 1.0,
        built_fraction_max: 0.0,
        category_built_fractions: category_built_fractions.to_vec(),
        source_category_built_fractions: source_category_built_fractions.map(|values| values.to_vec()),
        source_covered_urban_cells: 0,
        source_blended_urban_cells: 0,
        source_uncovered_urban_cells: 0,
    };
    for i in 0..n {
        let category = lu.data[i];
        if !category.is_finite() || category.fract() != 0.0
            || category < 1.0 || category > luf.planes as f64
            || !matches!(land.data[i], 0.0 | 1.0)
        {
            return Err(invalid("urban fraction LU_INDEX or LANDMASK is invalid: its dominant-urban mask is undefined"));
        }
        let dominant = land.data[i] == 1.0 && weights[category as usize - 1] > 0.0;
        let mut value = 0.0;
        let mut total = 0.0;
        let mut cover = 0.0;
        for (k, weight) in weights.iter().enumerate() {
            let fraction = luf.data[k * n + i];
            if !fraction.is_finite() || !(0.0..=1.0 + 1e-6).contains(&fraction) {
                return Err(invalid("urban fraction LANDUSEF contains an invalid area fraction"));
            }
            total += fraction;
            value += fraction * weight;
            if *weight > 0.0 { cover += fraction; }
        }
        if total > 1.0 + 1e-6 {
            return Err(invalid("urban fraction LANDUSEF exceeds one cell of area: built area would be counted twice"));
        }
        if cover > 0.0 { audit.cells_with_urban_cover += 1; }
        if dominant {
            if let Some((fractions, coverage, source_lookup)) = &source {
                let w = coverage.data[i];
                if !w.is_finite() || !(0.0..=1.0).contains(&w) {
                    return Err(invalid("urban source coverage weight is outside 0..1"));
                }
                if w > 0.0 {
                    let mut source_value = 0.0;
                    let mut source_total = 0.0;
                    for (k, coefficient) in source_lookup.iter().enumerate() {
                        let f = fractions.data[k * n + i];
                        if !f.is_finite() || !(0.0..=1.0 + 1e-6).contains(&f) {
                            return Err(invalid("covered urban source contains an invalid area fraction"));
                        }
                        source_total += f;
                        source_value += f * coefficient;
                    }
                    if (source_total - 1.0).abs() > 1e-6 {
                        return Err(invalid("covered urban source fractions do not sum to one cell"));
                    }
                    value = w * source_value + (1.0 - w) * value;
                    if w == 1.0 { audit.source_covered_urban_cells += 1; }
                    else { audit.source_blended_urban_cells += 1; }
                } else { audit.source_uncovered_urban_cells += 1; }
            }
            if value <= 0.0 {
                return Err(invalid("dominant urban land has no urban area fraction: its canopy would silently fall back to a table fraction"));
            }
            out.data[i] = value.min(1.0);
            audit.dominant_urban_cells += 1;
            audit.built_fraction_sum += out.data[i];
            audit.built_fraction_min = audit.built_fraction_min.min(out.data[i]);
            audit.built_fraction_max = audit.built_fraction_max.max(out.data[i]);
        } else if cover > 0.0 {
            audit.urban_cover_outside_dominant_cells += 1;
        }
    }
    if audit.dominant_urban_cells == 0 { audit.built_fraction_min = 0.0; }
    Ok((out, audit))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Stack3;

    fn fields() -> FieldSet {
        let mut fields = FieldSet::default();
        // Two categories, four cells: pure city, mixed city, mixed rural, water.
        fields.fields.insert("LANDUSEF".into(), Field::Stack(Stack3 {
            planes: 3, ny: 1, nx: 4,
            data: vec![0.0, 0.3, 0.7, 0.0, 1.0, 0.7, 0.3, 0.0, 0.0, 0.0, 0.0, 1.0],
        }));
        fields.fields.insert("LU_INDEX".into(), Field::Plane(Grid2 {
            ny: 1, nx: 4, data: vec![2.0, 2.0, 1.0, 3.0],
        }));
        fields.fields.insert("LANDMASK".into(), Field::Plane(Grid2 {
            ny: 1, nx: 4, data: vec![1.0, 1.0, 1.0, 0.0],
        }));
        fields
    }

    #[test]
    fn built_fraction_accounts_for_mixed_cover_and_dominant_mask() {
        let (out, audit) = estimate(&fields(), &[(2, 0.9)]).unwrap();
        assert_eq!(out.data, vec![0.9, 0.7 * 0.9, 0.0, 0.0]);
        assert_eq!(audit.dominant_urban_cells, 2);
        assert_eq!(audit.urban_cover_outside_dominant_cells, 1);
    }

    #[test]
    fn mixed_urban_types_use_their_own_built_coefficients() {
        let (out, _) = estimate(&fields(), &[(1, 0.5), (2, 0.9)]).unwrap();
        assert_eq!(out.data, vec![0.9, 0.3 * 0.5 + 0.7 * 0.9, 0.7 * 0.5 + 0.3 * 0.9, 0.0]);
    }

    #[test]
    fn corrupt_inventory_and_fractions_are_refused() {
        assert!(estimate(&fields(), &[(4, 0.9)]).is_err());
        assert!(estimate(&fields(), &[(2, 0.9), (2, 0.5)]).is_err());
        let mut fields = fields();
        if let Field::Stack(s) = fields.fields.get_mut("LANDUSEF").unwrap() {
            s.data[0] = f64::NAN;
        }
        assert!(estimate(&fields, &[(2, 0.9)]).is_err());
    }

    #[test]
    fn source_fraction_keeps_morphology_mask_and_blends_coverage() {
        let mut f = fields();
        f.fields.insert("URBAN_SOURCEF".into(), Field::Stack(Stack3 {
            planes: 2, ny: 1, nx: 4,
            data: vec![0.0, 0.5, 0.0, 0.0, 1.0, 0.5, 1.0, 1.0],
        }));
        f.fields.insert("URBAN_SOURCE_WEIGHT".into(), Field::Plane(Grid2 {
            ny: 1, nx: 4, data: vec![1.0, 0.5, 1.0, 1.0],
        }));
        let (out, audit) = estimate_with_source(&f, &[(2, 0.9)],
            Some(&[(2, 0.345)]), Some("source-test"), Some("class estimate")).unwrap();
        assert_eq!(out.data, vec![0.345, 0.5 * (0.5 * 0.345) + 0.5 * (0.7 * 0.9), 0.0, 0.0]);
        assert_eq!(audit.source_covered_urban_cells, 1);
        assert_eq!(audit.source_blended_urban_cells, 1);
        assert_eq!(audit.algorithm, "source-test");
        if let Field::Stack(s) = f.fields.get_mut("URBAN_SOURCEF").unwrap() {
            s.data[4] = f64::NAN;
        }
        assert!(estimate_with_source(&f, &[(2, 0.9)], Some(&[(2, 0.345)]),
            Some("source-test"), Some("class estimate")).is_err());
    }
}
