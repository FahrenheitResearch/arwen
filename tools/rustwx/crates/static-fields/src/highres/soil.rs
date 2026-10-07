//! Carry selected WPS soil fractions through a terrain/land-cover overlay.

use crate::error::{Result, StaticError};
use crate::types::{Field, FieldSet, Grid2, Stack3};

pub fn from_baseline(baseline: &FieldSet, landmask: &Grid2)
    -> Result<(FieldSet, serde_json::Value)>
{
    let (ny, nx) = (landmask.ny, landmask.nx);
    let n = ny * nx;
    let mut fields = FieldSet::default();
    let mut audit = serde_json::Map::new();
    for (fraction_name, dominant_name, layer) in [
        ("SOILCTOP", "SCT_DOM", "top_0_30cm"),
        ("SOILCBOT", "SCB_DOM", "bottom_30_100cm"),
    ] {
        let Some(Field::Stack(source)) = baseline.fields.get(fraction_name) else {
            return Err(StaticError::Missing(format!(
                "selected WPS soil requires {fraction_name} category fractions")));
        };
        if (source.planes, source.ny, source.nx) != (16, ny, nx) {
            return Err(StaticError::Invalid(format!(
                "{fraction_name} selected WPS soil shape differs from (16, {ny}, {nx})")));
        }
        if source.data.iter().any(|value| !value.is_finite() || *value < 0.0) {
            return Err(StaticError::Invalid(format!(
                "{fraction_name} selected WPS soil contains invalid fractions")));
        }
        let dominant = crate::fields::dominant_category(source)?;
        let donors: Vec<bool> = (0..n).map(|cell| {
            landmask.data[cell] > 0.5 && dominant.data[cell] != 14.0
                && (0..16).map(|p| source.data[p*n+cell]).sum::<f64>() > 0.0
        }).collect();
        let repair: Vec<bool> = (0..n).map(|cell| {
            landmask.data[cell] > 0.5 && !donors[cell]
        }).collect();
        let repair_count = repair.iter().filter(|flag| **flag).count();
        let mut fractions: Stack3 = source.clone();
        if repair_count > 0 {
            if !donors.iter().any(|flag| *flag) {
                return Err(StaticError::Invalid(format!(
                    "{fraction_name} selected WPS soil has {repair_count} land cells with water or missing soil and no land-soil donor")));
            }
            let (donor_y, donor_x) = super::nearest_donors(&donors, ny, nx)?;
            for cell in 0..n {
                if repair[cell] {
                    let donor = donor_y[cell] as usize * nx + donor_x[cell] as usize;
                    for p in 0..16 {
                        fractions.data[p*n+cell] = source.data[p*n+donor];
                    }
                }
            }
        }
        let mut water_count = 0;
        for cell in 0..n {
            if landmask.data[cell] <= 0.5 {
                water_count += 1;
                for p in 0..16 {
                    fractions.data[p*n+cell] = if p == 13 { 1.0 } else { 0.0 };
                }
            }
        }
        let dominant = crate::fields::dominant_category(&fractions)?;
        fields.fields.insert(fraction_name.into(), Field::Stack(fractions));
        fields.fields.insert(dominant_name.into(), Field::Plane(dominant));
        audit.insert(layer.into(), serde_json::json!({
            "source": "selected WPS geography",
            "retained_land_cells": donors.iter().filter(|flag| **flag).count(),
            "water_soil_land_cells_from_nearest_land": repair_count,
            "water_cells": water_count,
        }));
    }
    Ok((fields, serde_json::Value::Object(audit)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keeps_land_fraction_bits_and_repairs_water_on_new_land() {
        let mut baseline = FieldSet::default();
        let mut source = Stack3 { planes: 16, ny: 1, nx: 3, data: vec![0.; 48] };
        source.data[3] = 0.20000000298023224;
        source.data[6] = 0.7999999970197678;
        source.data[13*3+1] = 1.;
        source.data[13*3+2] = 1.;
        for name in ["SOILCTOP", "SOILCBOT"] {
            baseline.fields.insert(name.into(), Field::Stack(source.clone()));
        }
        let mask = Grid2 { ny: 1, nx: 3, data: vec![1., 1., 0.] };
        let (fields, audit) = from_baseline(&baseline, &mask).unwrap();
        let Field::Stack(result) = &fields.fields["SOILCTOP"] else { panic!() };
        for p in 0..16 {
            assert_eq!(result.data[p*3].to_bits(), source.data[p*3].to_bits());
            assert_eq!(result.data[p*3+1].to_bits(), source.data[p*3].to_bits());
            assert_eq!(result.data[p*3+2], if p == 13 {1.} else {0.});
        }
        assert_eq!(audit["top_0_30cm"]["water_soil_land_cells_from_nearest_land"], 1);
    }

    #[test]
    fn refuses_water_soil_on_land_without_donors() {
        let mut baseline = FieldSet::default();
        let mut source = Stack3 { planes: 16, ny: 1, nx: 1, data: vec![0.; 16] };
        source.data[13] = 1.;
        baseline.fields.insert("SOILCTOP".into(), Field::Stack(source));
        let mask = Grid2 { ny: 1, nx: 1, data: vec![1.] };
        assert!(from_baseline(&baseline, &mask).unwrap_err().to_string().contains("no land-soil donor"));
    }
}
