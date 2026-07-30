# rustwx-render

`rustwx-render` is the Rust map-rendering crate for `rustwx`.

It owns the `rustwx` map-rendering engine directly: request types, Weather palettes, contour layers, barb layers, basemap helpers, projection prep, and panel helpers all live in this crate.

## What is implemented

- Weather-backed filled maps
- generic palette-backed filled maps
- built-in derived field helpers for:
  - lifted index
  - 700/850 mb temperature advection
  - 0-1 / 0-6 km bulk shear
  - apparent temperature / heat index / wind chill
- contour overlays
- wind barb overlays
- contour-only canvases for height/wind style products
- projected line overlays
- PNG and image rendering
- multi-panel composition helpers
- optional product semantics metadata on render requests so experimental/proof
  caveats can travel with the final artifact instead of living only in notes

## What this crate expects from callers

- the field values to render
- the grid definition
- projected coordinates if a projected map is being drawn
- any line overlays or contour/barb layers

## Current limits

- no animation/GIF orchestration yet
- no fetch/decode logic here by design
- overlay builders require matching grids; this crate does not remap fields

## Minimal example

```rust
use rustwx_render::{MapRenderRequest, WeatherProduct, save_png};

let request = MapRenderRequest::for_weather_product(field, WeatherProduct::Sbecape);
save_png(&request, "out.png")?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

## Generic filled product example

```rust
use rustwx_render::{ExtendMode, MapRenderRequest, WeatherPalette, save_png};

let mut request = MapRenderRequest::for_palette_fill(
    field,
    WeatherPalette::Temperature,
    vec![-30.0, -20.0, -10.0, 0.0, 10.0, 20.0, 30.0],
    ExtendMode::Both,
);
request.title = Some("700 MB TEMPERATURE".into());
save_png(&request, "700mb_temperature.png")?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

## Derived helper example

```rust
use rustwx_render::{DerivedProductStyle, MapRenderRequest, save_png};

let request = MapRenderRequest::for_derived_product(
    field,
    DerivedProductStyle::TemperatureAdvection850mb,
);
save_png(&request, "850mb_temperature_advection.png")?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

The built-in derived helpers only provide render defaults: title text, color levels, and palette choice. They do not define the science or unit conversion. Product assembly is still responsible for supplying the correct field values and, where needed, converting them into the plotted units those helpers expect.

## Contour-only height and wind example

```rust
use rustwx_render::{ContourStyle, MapRenderRequest, WindBarbStyle, render_image};

let request = MapRenderRequest::contour_only(height_field)
    .with_contour_field(
        &height_field,
        vec![5400.0, 5460.0, 5520.0, 5580.0],
        ContourStyle {
            labels: true,
            show_extrema: true,
            ..Default::default()
        },
    )?
    .with_wind_barbs(
        &u_wind_field,
        &v_wind_field,
        WindBarbStyle {
            stride_x: 6,
            stride_y: 6,
            ..Default::default()
        },
    )?;

let _image = render_image(&request)?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

## Combination panel composition

Build individual `MapRenderRequest`s with any mix of filled fields, contours, and barbs, then pass them into `render_panel_grid(...)` for a composed image canvas.
