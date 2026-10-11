use std::{collections::HashMap, fs};
use rustwx_core::{CycleSpec, GridShape, LatLonGrid, ModelId, SelectedField2D, SourceId};
use rustwx_models::{LatestRun, plot_recipe_store_plan};
use rustwx_products::direct::{DirectBatchRequest, render_direct_recipe_from_selected_fields};

#[test]
fn exact_release_note_window_suffix_has_no_hard_coded_dash_join() {
    let field = rustwx_core::Field2D::new(
        rustwx_core::ProductKey::named("snow_kuchera_window"), "mm",
        LatLonGrid::new(GridShape::new(1,1).unwrap(),vec![47.5],vec![-111.3]).unwrap(),
        vec![0.],
    ).unwrap();
    let mut request = rustwx_render::MapRenderRequest::from_core_field(field,
        rustwx_render::ColorScale::Discrete(rustwx_render::DiscreteColorScale {
            levels: vec![0.,1.], colors:vec![rustwx_render::Color::BLACK],
            extend:rustwx_render::ExtendMode::Neither, mask_below:None,
        }));
    request.title=Some("Kuchera snowfall, window total (d02 750 m)".into());
    let annotation: rustwx_products::geographic_overlays::PanelAnnotations = serde_json::from_str(
        r#"{"title_suffix":", 24 h ending Mon 12 Oct 18Z"}"#,
    ).unwrap();
    annotation.apply(&mut request);
    assert_eq!(request.title.as_deref(),Some(
        "Kuchera snowfall, window total (d02 750 m) , 24 h ending Mon 12 Oct 18Z"
    ));
}

#[test]
fn panel_grid_theme_overrides_the_exact_release_note_source_label() {
    let root = std::env::temp_dir().join(format!("render-labels-{}", std::process::id()));
    fs::create_dir_all(&root).unwrap();
    // The release-note label is the one the built-in public theme prints,
    // read from that theme rather than restated here.
    let label = rustwx_render::RenderTheme::builtin_file("woof-light")
        .and_then(|file| file.text)
        .and_then(|text| text.source_label)
        .expect("the woof-light theme names its source label");
    let public = format!("source: {label}");
    let theme = rustwx_render::RenderThemeFile::from_json(r#"{"extends":"woof-light"}"#).unwrap();
    rustwx_render::install_theme(rustwx_render::RenderTheme::from_file_spec(&theme, None).unwrap()).unwrap();
    let latest = LatestRun { model: ModelId::WrfGdex,
        cycle: CycleSpec::new("20261008", 18).unwrap(), source: SourceId::Gdex };
    let mut failures = Vec::new();
    for slug in ["precipitation_type", "cloud_cover_levels"] {
        let plan = plot_recipe_store_plan(slug, latest.model).unwrap();
        let mut fields = HashMap::new();
        for selector in plan.selectors() {
            let grid = LatLonGrid::new(GridShape::new(3, 3).unwrap(),
                vec![46.8,46.8,46.8,47.2,47.2,47.2,47.6,47.6,47.6],
                vec![-112.2,-111.7,-111.2,-112.2,-111.7,-111.2,-112.2,-111.7,-111.2]).unwrap();
            fields.insert(selector, SelectedField2D::new(selector,
                if slug == "precipitation_type" { "1" } else { "%" },
                grid, vec![0.,1.,0.,1.,1.,0.,0.,1.,1.]).unwrap());
        }
        for (layout, mode) in [
            ("fixed", rustwx_render::LayoutMode::Fixed),
            ("auto", rustwx_render::LayoutMode::Auto { class: rustwx_render::SizeClass::Standard, scale: 1.0 }),
        ] {
        rustwx_render::set_layout_mode(mode);
        let mut images = Vec::new();
        for (arm, source) in [("engine", "source: ArWen 2.8.8"), ("public", public.as_str())] {
            let request: DirectBatchRequest = serde_json::from_value(serde_json::json!({
                "model":latest.model, "date_yyyymmdd":"20261008", "cycle_override_utc":18,
                "forecast_hour":96, "source":latest.source, "domain":{"slug":"winter", "bounds":[-112.2,-111.2,46.8,47.6]},
                "out_dir":root.join(arm), "cache_root":root.join("cache"), "use_cache":false,
                "recipe_slugs":[slug], "product_overrides":{}, "output_width":1000, "output_height":800,
                "subtitle_right_override":source
            })).unwrap();
            let rendered = render_direct_recipe_from_selected_fields(&request, &latest, slug,
                &fields, "native_grid", "fixture", "fixture").unwrap();
            images.push(fs::read(rendered.output_path).unwrap());
        }
        if images[0] != images[1] { failures.push(format!("{slug}/{layout}")); }
        }
    }
    fs::remove_dir_all(root).unwrap();
    assert!(failures.is_empty(), "theme must replace source: ArWen 2.8.8 in both panel headers: {failures:?}");
}
