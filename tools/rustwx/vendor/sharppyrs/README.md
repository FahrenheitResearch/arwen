# sharppyrs

The SPC-style **Skew-T sounding plot** from
[SHARPpy-Reimagined-vRust](https://github.com/FahrenheitResearch/SHARPpy-Reimagined-vRust)
(built on [SHARPpy](https://github.com/sharppy/SHARPpy)), ported to
**Rust / [egui](https://github.com/emilk/egui)** as an embeddable widget.

Faithful to the original render: log-p/skewed-T background grid, temperature /
dewpoint / wetbulb / virtual-temperature traces, lifted parcel trace with
orange CAPE + blue CIN buoyancy fill, downdraft parcel trace, LCL/LFC/EL
markers, 0/-20/-30 °C levels, effective inflow layer bracket with ESRH, max
2-6 km lapse-rate bracket, HGZ band, height markers, omega meter, and
speed-colored wind barbs — with the bundled Space Grotesk face.

All numerics come from [`sharprs`](https://github.com/FahrenheitResearch/sharprs)
(the pure-Rust SHARPpy engine) — this crate is rendering-only. The handful of
things sharprs doesn't define live in `src/extras.rs` (parcel-based Bunkers
storm motion, the SHARPpy-style forecast-max-temp FCST parcel). The pipeline
is validated against the Python implementation's output (`tests/golden.rs`);
known engine-level deviations (fractionally different virtual temperature /
saturated lifts, FCST parcel layer means, DCAPE trace direction) are
documented in the test file.

## Usage

```rust
// once at startup (optional, for the original's Space Grotesk look):
sharppyrs::install_fonts(&cc.egui_ctx);

// from an existing sharprs profile (e.g. your app's calc engine):
let profile = sharppyrs::Profile::from_sharprs(sharprs_profile);

// ...or from raw sounding data (surface upward):
let profile = sharppyrs::Profile::new(sharppyrs::SoundingData {
    pres,               // hPa
    hght,               // m MSL
    tmpc, dwpc,         // °C
    wdir, wspd,         // deg, kts
    omeg: Some(omeg),   // Pa/s (optional; enables the omega meter)
    latitude: Some(36.7),
    missing: None,      // defaults to -9999.0
}).unwrap();

// in your egui UI:
ui.add(
    sharppyrs::SkewT::new(&profile)
        .title("Forecast profile")
        .parcel(sharppyrs::ParcelType::MostUnstable) // default, like the original
        .style(sharppyrs::SkewTStyle::space_grotesk()),
);
```

`Profile::new` does all the heavy lifting once (parcels, effective layer,
storm motion, DCAPE); the widget itself just paints, so it is cheap to draw
every frame. Colors are configurable through `SkewTStyle`.

## Demo

```
cargo run --example demo
```

renders the bundled example HRRR sounding.

## Tests

- `cargo test --test golden` — numerical port vs. SHARPpy Python golden data
  (`testdata/golden.json`).
- `cargo test --test snapshot` — headless wgpu render to
  `target/skewt_snapshot.png`.

## License & attribution

BSD-3-Clause, like [SHARPpy](https://github.com/sharppy/SHARPpy) (© SHARPpy
contributors) and
[SHARPpy-Reimagined-vRust](https://github.com/FahrenheitResearch/SHARPpy-Reimagined-vRust);
this crate is a derived port of their rendering and `sharptab` algorithms —
please cite Blumberg et al. 2017 (*BAMS*, 98, 1625–1636,
[doi:10.1175/BAMS-D-15-00309.1](https://doi.org/10.1175/BAMS-D-15-00309.1))
when citing this functionality. Numerics come from
[`sharprs`](https://github.com/FahrenheitResearch/sharprs)
(FahrenheitResearch). Space Grotesk (© 2020 Florian Karsten) is bundled under
the SIL Open Font License 1.1 (`assets/fonts/OFL.txt`). The locator basemap
lines are simplified from [Natural Earth](https://www.naturalearthdata.com/)
1:50m data (public domain). Full license texts live in the linked
repositories and `LICENSE` / `assets/fonts/OFL.txt` here.
