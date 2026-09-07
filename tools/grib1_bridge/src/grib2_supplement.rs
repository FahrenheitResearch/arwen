//! Canonical optional quantities selected from explicitly declared GRIB donors.
//! Product names and display strings are not quantity or coordinate authority.
use grib_core::grib2::{Grib2Message, GridDefinition};

/// GRIB code-table quantities already expressed in Pa. NCEP's local parameter
/// 192/198 are meaningful only under their originating-center authority (7).
pub fn is_pmsl_pa(message: &Grib2Message) -> bool {
    message.discipline == 0
        && message.product.parameter_category == 3
        && (message.product.parameter_number == 1
            || (matches!(message.product.parameter_number, 192 | 198)
                && message.identification.center_id == 7))
        && message.product.level_type == 101
        && message.product.second_level_type == 255
}

fn forecast_seconds(message: &Grib2Message) -> Option<u64> {
    let multiplier = match message.product.time_range_unit {
        0 => 60,
        1 => 3600,
        2 => 86400,
        10 => 10800,
        11 => 21600,
        12 => 43200,
        13 => 1,
        _ => return None,
    };
    Some(u64::from(message.product.forecast_time) * multiplier)
}

/// Select exactly one instantaneous value at the requested source time. The
/// consumer's actual grid is compared in full, including projection and scan
/// metadata; equal array sizes alone do not establish coordinate alignment.
pub fn select_pmsl<'a>(
    messages: impl Iterator<Item = &'a Grib2Message>,
    cycle: &str,
    forecast_hour: u32,
    grid: &GridDefinition,
) -> Result<&'a Grib2Message, String> {
    let mut matches = messages.filter(|message| {
        is_pmsl_pa(message)
            && message.reference_time.to_string() == cycle
            && forecast_seconds(message) == Some(u64::from(forecast_hour) * 3600)
    });
    let message = matches.next().ok_or_else(|| {
        format!(
            "PMSL donor lacks mean-sea-level pressure in Pa at cycle {cycle} f{forecast_hour:02}"
        )
    })?;
    if matches.next().is_some() {
        return Err(format!(
            "duplicate PMSL donors at cycle {cycle} f{forecast_hour:02}"
        ));
    }
    if !matches!(message.product.template, 0 | 1) {
        return Err(
            "PMSL donor must carry instantaneous absolute pressure (PDT 4.0 or 4.1)".into(),
        );
    }
    if &message.grid != grid {
        return Err("PMSL donor grid does not exactly equal the primary projected grid".into());
    }
    Ok(message)
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::NaiveDateTime;
    use grib_core::grib2::{DataRepresentation, Identification, ProductDefinition};

    fn field() -> Grib2Message {
        Grib2Message {
            discipline: 0,
            identification: Identification {
                center_id: 7,
                ..Default::default()
            },
            reference_time: NaiveDateTime::parse_from_str(
                "2021-12-10 18:00:00",
                "%Y-%m-%d %H:%M:%S",
            )
            .unwrap(),
            grid: GridDefinition {
                template: 30,
                nx: 7,
                ny: 5,
                dx: 3000.0,
                dy: 3000.0,
                lov: 262.5,
                ..Default::default()
            },
            product: ProductDefinition {
                parameter_category: 3,
                parameter_number: 198,
                level_type: 101,
                time_range_unit: 1,
                forecast_time: 1,
                ..Default::default()
            },
            data_rep: DataRepresentation::default(),
            bitmap: None,
            raw_data: vec![],
        }
    }

    #[test]
    fn quantity_authority_is_numeric_and_local_codes_keep_their_origin() {
        let mut value = field();
        assert!(is_pmsl_pa(&value));
        value.product.parameter_number = 192;
        assert!(is_pmsl_pa(&value));
        value.identification.center_id = 98;
        assert!(!is_pmsl_pa(&value));
        value.product.parameter_number = 1;
        assert!(is_pmsl_pa(&value));
        value.product.level_type = 1;
        assert!(!is_pmsl_pa(&value));
        value.product.level_type = 101;
        value.product.second_level_type = 100;
        assert!(!is_pmsl_pa(&value));
    }

    #[test]
    fn selection_binds_time_full_projection_and_single_provider() {
        let value = field();
        let grid = value.grid.clone();
        let cycle = "2021-12-10 18:00:00";
        assert!(select_pmsl([&value].into_iter(), cycle, 1, &grid).is_ok());
        assert!(select_pmsl([&value].into_iter(), cycle, 0, &grid)
            .unwrap_err()
            .contains("lacks"));
        assert!(select_pmsl([&value, &value].into_iter(), cycle, 1, &grid)
            .unwrap_err()
            .contains("duplicate"));
        for alternative in [
            GridDefinition {
                lov: 263.5,
                ..grid.clone()
            },
            GridDefinition {
                scan_mode: 64,
                ..grid.clone()
            },
            GridDefinition {
                dx: 1000.0,
                ..grid.clone()
            },
        ] {
            assert!(select_pmsl([&value].into_iter(), cycle, 1, &alternative)
                .unwrap_err()
                .contains("grid"));
        }
        let mut other = value.clone();
        other.product.template = 8;
        assert!(select_pmsl([&other].into_iter(), cycle, 1, &grid)
            .unwrap_err()
            .contains("instantaneous"));
        other = value.clone();
        other.product.template = 1;
        other.product.ensemble_type = Some(3);
        other.product.perturbation_number = Some(2);
        other.product.num_forecasts_in_ensemble = Some(20);
        assert!(select_pmsl([&other].into_iter(), cycle, 1, &grid).is_ok());
        other.product.template = 5;
        assert!(select_pmsl([&other].into_iter(), cycle, 1, &grid).is_err());
        other = value.clone();
        other.product.time_range_unit = 0;
        other.product.forecast_time = 60;
        assert!(select_pmsl([&other].into_iter(), cycle, 1, &grid).is_ok());
        other.product.forecast_time = 61;
        assert!(select_pmsl([&other].into_iter(), cycle, 1, &grid).is_err());
    }
}
