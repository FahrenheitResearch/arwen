use super::*;
use crate::request::{ProductMaturity, ProductSemanticFlag};

#[test]
fn explicit_ecape_panel_products_have_expected_titles_and_experimental_flags() {
    assert_eq!(WeatherProduct::Sbecape.display_title(), "SBECAPE");
    assert_eq!(WeatherProduct::Mlecin.display_title(), "MLECIN");
    assert!(WeatherProduct::EcapeScpExperimental.is_experimental());
    assert!(WeatherProduct::EcapeEhi01kmExperimental.is_experimental());
    assert!(WeatherProduct::EcapeEhi03kmExperimental.is_experimental());
    assert!(WeatherProduct::EcapeStpExperimental.is_experimental());
    assert!(WeatherProduct::SbEcapeDerivedCapeRatio.is_experimental());
    assert!(WeatherProduct::SbEcapeNativeCapeRatio.is_experimental());
    assert!(!WeatherProduct::Muecape.is_experimental());
    assert_eq!(
        WeatherProduct::EcapeScpExperimental.semantics().maturity,
        ProductMaturity::Experimental
    );
    assert_eq!(
        WeatherProduct::Sbcape.semantics().maturity,
        ProductMaturity::Operational
    );
}

#[test]
fn ecape_panel_defaults_match_requested_operational_layout() {
    assert_eq!(
        ECAPE_SEVERE_PANEL_PRODUCTS,
        [
            WeatherProduct::Sbecape,
            WeatherProduct::Mlecape,
            WeatherProduct::Muecape,
            WeatherProduct::SbEcapeDerivedCapeRatio,
            WeatherProduct::MlEcapeDerivedCapeRatio,
            WeatherProduct::MuEcapeDerivedCapeRatio,
            WeatherProduct::SbEcapeNativeCapeRatio,
            WeatherProduct::MlEcapeNativeCapeRatio,
            WeatherProduct::MuEcapeNativeCapeRatio,
            WeatherProduct::Sbncape,
            WeatherProduct::Sbecin,
            WeatherProduct::Mlecin,
            WeatherProduct::EcapeScpExperimental,
            WeatherProduct::EcapeEhi01kmExperimental,
            WeatherProduct::EcapeEhi03kmExperimental,
            WeatherProduct::EcapeStpExperimental,
        ]
    );
}

#[test]
fn severe_panel_defaults_cover_classic_severe_suite() {
    assert_eq!(
        SEVERE_CLASSIC_PANEL_PRODUCTS,
        [
            WeatherProduct::Sbcape,
            WeatherProduct::Mlcape,
            WeatherProduct::Mucape,
            WeatherProduct::Mlcin,
            WeatherProduct::Srh01km,
            WeatherProduct::Srh03km,
            WeatherProduct::Stp,
            WeatherProduct::Scp,
        ]
    );
}

#[test]
fn product_name_resolution_covers_parcel_explicit_ecape_fields() {
    assert_eq!(
        WeatherProduct::from_product_name("mlecin"),
        Some(WeatherProduct::Mlecin)
    );
    assert_eq!(
        WeatherProduct::from_product_name("ecape_scp"),
        Some(WeatherProduct::EcapeScpExperimental)
    );
    assert_eq!(
        WeatherProduct::from_product_name("sb_ecape_derived_cape_ratio"),
        Some(WeatherProduct::SbEcapeDerivedCapeRatio)
    );
    assert_eq!(
        WeatherProduct::from_product_name("ml_ecape_native_cape_ratio"),
        Some(WeatherProduct::MlEcapeNativeCapeRatio)
    );
    assert_eq!(
        WeatherPreset::from_product_name("mu_ecape_native_cape_ratio"),
        Some(WeatherPreset::EcapeCapeRatio)
    );
    assert_eq!(
        WeatherProduct::from_product_name("ecape_ehi"),
        Some(WeatherProduct::EcapeEhi01kmExperimental)
    );
    assert_eq!(
        WeatherProduct::from_product_name("ecape_ehi_0_1km"),
        Some(WeatherProduct::EcapeEhi01kmExperimental)
    );
    assert_eq!(
        WeatherProduct::from_product_name("ecape_ehi_0_3km"),
        Some(WeatherProduct::EcapeEhi03kmExperimental)
    );
    assert_eq!(
        WeatherProduct::from_product_name("ehi_0_1km"),
        Some(WeatherProduct::Ehi)
    );
    assert_eq!(
        WeatherProduct::from_product_name("ehi_0_3km"),
        Some(WeatherProduct::Ehi)
    );
    assert_eq!(
        WeatherProduct::from_product_name("vtp_mod"),
        Some(WeatherProduct::VtpMod)
    );
    assert_eq!(
        WeatherPreset::from_product_name("ecape_ehi_0_3km"),
        Some(WeatherPreset::Ehi)
    );
    assert_eq!(
        WeatherPreset::from_product_name("ehi_0_1km"),
        Some(WeatherPreset::Ehi)
    );
    assert_eq!(
        WeatherPreset::from_product_name("vtp_mod"),
        Some(WeatherPreset::Stp)
    );
}

#[test]
fn palette_scale_wraps_palette_and_levels_into_discrete_scale() {
    let scale = palette_scale(
        WeatherPalette::Reflectivity,
        vec![5.0, 15.0, 25.0, 35.0],
        ExtendMode::Max,
        Some(5.0),
    );

    assert_eq!(scale.levels, vec![5.0, 15.0, 25.0, 35.0]);
    assert_eq!(scale.extend, ExtendMode::Max);
    assert_eq!(scale.mask_below, Some(5.0));
    assert!(!scale.colors.is_empty());
}

#[test]
fn dewpoint_palette_has_hard_moisture_thresholds() {
    let levels = vec![59.0, 60.0, 69.0, 70.0, 79.0, 80.0, 89.0, 90.0];
    let colors = dewpoint_palette_fahrenheit_for_levels(&levels);

    assert_eq!(colors.len(), levels.len() - 1);
    assert_ne!(colors[0], colors[1], "60F should start a new green band");
    assert_ne!(colors[2], colors[3], "70F should start the blue/slate band");
    assert_ne!(colors[4], colors[5], "80F should start the purple band");
    assert!(colors[0].g > colors[0].r, "50s dewpoints should read green");
    assert!(
        colors[3].b >= colors[3].g,
        "70s dewpoints should tilt blue/slate"
    );
    assert!(
        colors[5].b > colors[5].g,
        "80s dewpoints should tilt purple"
    );
}

#[test]
fn celsius_dewpoint_palette_uses_fahrenheit_thresholds() {
    let levels_c = vec![
        (69.0 - 32.0) * 5.0 / 9.0,
        (70.0 - 32.0) * 5.0 / 9.0,
        (79.0 - 32.0) * 5.0 / 9.0,
        (80.0 - 32.0) * 5.0 / 9.0,
    ];
    let levels_f = vec![69.0, 70.0, 79.0, 80.0];

    assert_eq!(
        dewpoint_palette_celsius_for_levels(&levels_c),
        dewpoint_palette_fahrenheit_for_levels(&levels_f)
    );
}

#[test]
fn derived_product_styles_cover_new_helper_tranche() {
    assert_eq!(
        DerivedProductStyle::from_product_name("lifted_index"),
        Some(DerivedProductStyle::LiftedIndex)
    );
    assert_eq!(
        DerivedProductStyle::from_product_name("temperature_advection_850mb"),
        Some(DerivedProductStyle::TemperatureAdvection850mb)
    );
    assert_eq!(
        DerivedProductStyle::from_product_name("bulk_shear_0_6km"),
        Some(DerivedProductStyle::BulkShear06km)
    );
    assert_eq!(
        DerivedProductStyle::from_product_name("apparent_temperature"),
        Some(DerivedProductStyle::ApparentTemperature)
    );
}

#[test]
fn lifted_index_and_advection_scales_use_diverging_advection_helper() {
    let li = DerivedScalePreset::LiftedIndex.scale();
    let advection = DerivedScalePreset::TemperatureAdvection.scale();

    assert_eq!(li.levels, range_step(-12.0, 14.0, 2.0));
    assert_eq!(advection.levels, range_step(-12.0, 14.0, 2.0));
    assert_eq!(li.extend, ExtendMode::Both);
    assert_eq!(advection.extend, ExtendMode::Both);
    assert_eq!(li.colors.first(), advection.colors.last());
    assert_eq!(li.colors.last(), advection.colors.first());
}

#[test]
fn severe_reference_scales_match_upstream_wrf_runner_bins() {
    assert_eq!(
        WeatherPreset::Cape.scale().levels,
        range_step(0.0, 8100.0, 100.0)
    );
    assert_eq!(
        WeatherPreset::ThreeCape.scale().levels,
        concat_ranges(&[(0.0, 300.0, 5.0), (300.0, 501.0, 20.0)])
    );
    assert_eq!(
        WeatherPreset::Srh.scale().levels,
        range_step(0.0, 1001.0, 10.0)
    );
    assert_eq!(
        WeatherPreset::Stp.scale().levels,
        concat_ranges(&[(0.0, 10.0, 0.1), (10.0, 20.1, 0.5)])
    );
    assert_eq!(
        WeatherPreset::Scp.scale().levels,
        range_step(0.0, 30.0, 0.5)
    );
    assert_eq!(
        WeatherPreset::Ehi.scale().levels,
        concat_ranges(&[(0.0, 2.0, 0.1), (2.0, 20.2, 0.2)])
    );
    assert_eq!(
        WeatherPreset::EcapeCapeRatio.scale().levels,
        range_step(0.0, 1.15, 0.05)
    );
    assert_eq!(
        WeatherPreset::Uh.scale().levels,
        concat_ranges(&[(0.0, 200.0, 5.0), (200.0, 401.0, 10.0)])
    );
    assert_eq!(
        WeatherPreset::LapseRate.scale().levels,
        range_step(3.0, 10.1, 0.1)
    );
    assert_eq!(WeatherPreset::LapseRate.default_tick_step(), Some(1.0));
}

#[test]
fn bulk_shear_and_surface_comfort_have_sane_tick_steps() {
    assert_eq!(DerivedScalePreset::BulkShear.default_tick_step(), Some(5.0));
    assert_eq!(
        DerivedProductStyle::ApparentTemperature.default_tick_step(),
        Some(5.0)
    );
    assert_eq!(
        DerivedProductStyle::TemperatureAdvection700mb.display_title(),
        "700 MB TEMPERATURE ADVECTION"
    );
}

#[test]
fn semantic_flags_stay_narrow_in_render_presets() {
    let severe = WeatherProduct::Scp.semantics();
    assert_eq!(severe.maturity, ProductMaturity::Operational);
    assert!(!severe.has_flag(ProductSemanticFlag::Proxy));

    let ecape_01km = WeatherProduct::EcapeEhi01kmExperimental.semantics();
    assert_eq!(ecape_01km.maturity, ProductMaturity::Experimental);
    assert!(!ecape_01km.has_flag(ProductSemanticFlag::ProofOriented));

    let ecape_03km = WeatherProduct::EcapeEhi03kmExperimental.semantics();
    assert_eq!(ecape_03km.maturity, ProductMaturity::Experimental);
    assert!(!ecape_03km.has_flag(ProductSemanticFlag::ProofOriented));
}
