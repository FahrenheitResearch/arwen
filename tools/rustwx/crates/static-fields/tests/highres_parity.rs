//! Lane-3 parity: the Rust raster substrate + highres compute against
//! goldens extracted by RUNNING THE REAL PYTHON
//! (`gpuwm.static.highres` + rasterio/pyproj) on real source data —
//! see `tests/fixtures/highres/generate_goldens.py` for exactly how
//! every expected byte was produced.
//!
//! Contract split (docs/dev/static-rust-port.md §3):
//! * byte parity: USDA triangle, SoilGrids depth means, nearest-donor
//!   BFS, both merges (fields AND audit counts AND refusal messages),
//!   tile-id enumeration, GeoTIFF decode of the committed real-data
//!   windows;
//! * instrument validation: CRS transforms <= 1e-6 m against pyproj
//!   on fixture point sets BEFORE any warp result is trusted;
//! * tolerance parity: the warped/mosaicked planes against the
//!   rasterio/GDAL output, with the observed divergence printed and
//!   gated by the caps recorded in the fixtures.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use static_fields::highres;
use static_fields::projection::GridSpec;
use static_fields::raster::warp::{self, Resampling};
use static_fields::raster::{geotiff, transform_points, Crs, Raster};
use static_fields::types::{Field, FieldSet, Grid2, Stack3};

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/highres")
}

fn meta() -> serde_json::Value {
    let raw = std::fs::read_to_string(fixtures().join("meta.json"))
        .expect("fixtures present (run generate_goldens.py)");
    serde_json::from_str(&raw).expect("meta.json parses")
}

fn read_raw(entry: &serde_json::Value) -> Vec<u8> {
    let file = entry["file"].as_str().expect("fixture file name");
    std::fs::read(fixtures().join(file)).expect("fixture file readable")
}

fn read_f64(entry: &serde_json::Value) -> Vec<f64> {
    assert_eq!(entry["dtype"], "float64");
    read_raw(entry)
        .chunks_exact(8)
        .map(|raw| f64::from_le_bytes(raw.try_into().unwrap()))
        .collect()
}

fn read_f32(entry: &serde_json::Value) -> Vec<f32> {
    assert_eq!(entry["dtype"], "float32");
    read_raw(entry)
        .chunks_exact(4)
        .map(|raw| f32::from_le_bytes(raw.try_into().unwrap()))
        .collect()
}

fn read_i16(entry: &serde_json::Value) -> Vec<i16> {
    assert_eq!(entry["dtype"], "int16");
    read_raw(entry)
        .chunks_exact(2)
        .map(|raw| i16::from_le_bytes(raw.try_into().unwrap()))
        .collect()
}

fn read_i32(entry: &serde_json::Value) -> Vec<i32> {
    assert_eq!(entry["dtype"], "int32");
    read_raw(entry)
        .chunks_exact(4)
        .map(|raw| i32::from_le_bytes(raw.try_into().unwrap()))
        .collect()
}

fn read_u8(entry: &serde_json::Value) -> Vec<u8> {
    assert_eq!(entry["dtype"], "uint8");
    read_raw(entry)
}

fn shape(entry: &serde_json::Value) -> Vec<usize> {
    entry["shape"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect()
}

fn grid_spec(value: &serde_json::Value) -> GridSpec {
    serde_json::from_value(value.clone()).expect("grid spec parses")
}

fn crs_for_transform_case(name: &str) -> Crs {
    let lambert = |truelat1: f64, truelat2: f64, stand_lon: f64| {
        Crs::ModelSphere(GridSpec {
            kind: static_fields::projection::ProjectionKind::Lambert,
            ref_lat: 0.0,
            ref_lon: 0.0,
            truelat1,
            truelat2,
            stand_lon,
            dx: 1000.0,
            dy: 1000.0,
            e_we: 10,
            e_sn: 10,
            known_x: 5.0,
            known_y: 5.0,
            moad_cen_lat: 0.0,
            moad_cen_lon: 0.0,
        })
    };
    match name {
        "lcc" => lambert(38.0, 41.0, -84.0),
        "lcc_tangent" => lambert(45.0, 45.0, 10.0),
        "merc" => Crs::ModelSphere(GridSpec {
            kind: static_fields::projection::ProjectionKind::Mercator,
            ref_lat: 0.0,
            ref_lon: -80.0,
            truelat1: 20.0,
            truelat2: 20.0,
            stand_lon: 0.0,
            dx: 1000.0,
            dy: 1000.0,
            e_we: 10,
            e_sn: 10,
            known_x: 5.0,
            known_y: 5.0,
            moad_cen_lat: 0.0,
            moad_cen_lon: 0.0,
        }),
        "polar_n" => Crs::ModelSphere(GridSpec {
            kind: static_fields::projection::ProjectionKind::Polar,
            ref_lat: 0.0,
            ref_lon: 0.0,
            truelat1: 60.0,
            truelat2: 60.0,
            stand_lon: -100.0,
            dx: 1000.0,
            dy: 1000.0,
            e_we: 10,
            e_sn: 10,
            known_x: 5.0,
            known_y: 5.0,
            moad_cen_lat: 0.0,
            moad_cen_lon: 0.0,
        }),
        "polar_s" => Crs::ModelSphere(GridSpec {
            kind: static_fields::projection::ProjectionKind::Polar,
            ref_lat: 0.0,
            ref_lon: 0.0,
            truelat1: -60.0,
            truelat2: -60.0,
            stand_lon: 170.0,
            dx: 1000.0,
            dy: 1000.0,
            e_we: 10,
            e_sn: 10,
            known_x: 5.0,
            known_y: 5.0,
            moad_cen_lat: 0.0,
            moad_cen_lon: 0.0,
        }),
        "igh" => Crs::InterruptedGoodeHomolosine,
        "aea" => Crs::AlbersConusNad83 {
            lat_1: 29.5,
            lat_2: 45.5,
            lat_0: 23.0,
            lon_0: -96.0,
            false_easting: 0.0,
            false_northing: 0.0,
        },
        other => panic!("unknown transform case {other}"),
    }
}

#[test]
fn transforms_match_pyproj_to_a_micron() {
    let meta = meta();
    for case in meta["transform_points"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let crs = crs_for_transform_case(name);
        let lon = read_f64(&case["lon"]);
        let lat = read_f64(&case["lat"]);
        let expect_x = read_f64(&case["x"]);
        let expect_y = read_f64(&case["y"]);

        let mut x = lon.clone();
        let mut y = lat.clone();
        transform_points(&Crs::Geographic, &crs, &mut x, &mut y)
            .expect("forward transform runs");
        let mut worst = 0.0f64;
        for index in 0..x.len() {
            let dx = (x[index] - expect_x[index]).abs();
            let dy = (y[index] - expect_y[index]).abs();
            worst = worst.max(dx).max(dy);
        }
        println!("transform {name}: forward max |delta| = {worst:.3e} m");
        assert!(
            worst <= 1.0e-6,
            "{name}: forward diverges from pyproj by {worst} m"
        );

        // Inverse against pyproj's own round trip.
        let expect_lon = read_f64(&case["lon_back"]);
        let expect_lat = read_f64(&case["lat_back"]);
        let mut ix = expect_x.clone();
        let mut iy = expect_y.clone();
        transform_points(&crs, &Crs::Geographic, &mut ix, &mut iy)
            .expect("inverse transform runs");
        let mut worst_deg = 0.0f64;
        for index in 0..ix.len() {
            worst_deg = worst_deg
                .max((ix[index] - expect_lon[index]).abs())
                .max((iy[index] - expect_lat[index]).abs());
        }
        println!("transform {name}: inverse max |delta| = {worst_deg:.3e} deg");
        assert!(
            worst_deg <= 1.0e-11,
            "{name}: inverse diverges from pyproj by {worst_deg} deg"
        );
    }
}

fn assert_clip_decodes(meta: &serde_json::Value, key: &str) {
    let entry = &meta[key];
    let file = fixtures().join(entry["file"].as_str().unwrap());
    let (raster, _nodata) =
        geotiff::read_band1_raw(&file, None, None).expect("clip decodes");
    let expect_shape = shape(entry);
    assert_eq!(
        (raster.ny, raster.nx),
        (expect_shape[0], expect_shape[1]),
        "{key}: decoded shape"
    );
    assert_eq!(raster.crs, Crs::Geographic, "{key}: CRS");
    let expect_transform: Vec<f64> = entry["transform"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();
    for (mine, theirs) in raster.transform.iter().zip(&expect_transform) {
        assert!(
            (mine - theirs).abs() <= 1e-12,
            "{key}: transform {mine} vs {theirs}"
        );
    }
    let stride = entry["sample_stride"].as_u64().unwrap() as usize;
    let sample = read_f32(&entry["sample"]);
    for (offset, expect) in sample.iter().enumerate() {
        let mine = raster.values[offset * stride] as f32;
        assert!(
            mine.to_bits() == expect.to_bits(),
            "{key}: sample {offset} decodes {mine} vs {expect}"
        );
    }
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    let mut nans = 0usize;
    for value in &raster.values {
        if value.is_nan() {
            nans += 1;
        } else {
            min = min.min(*value);
            max = max.max(*value);
        }
    }
    assert_eq!(nans, entry["nan_count"].as_u64().unwrap() as usize);
    assert_eq!(min, entry["min"].as_f64().unwrap(), "{key}: min");
    assert_eq!(max, entry["max"].as_f64().unwrap(), "{key}: max");
}

#[test]
fn geotiff_decode_matches_rasterio_on_real_windows() {
    let meta = meta();
    // Tiled + deflate + floating-point predictor.
    assert_clip_decodes(&meta, "terrain_clip");
    // Striped, no predictor.
    assert_clip_decodes(&meta, "mosaic_clip_west");
    // Tiled, no predictor.
    assert_clip_decodes(&meta, "mosaic_clip_east");
}

/// The high-resolution DEM cache a real production run filled, if this
/// box has one.  Overridable; the default is COMPOSED from this
/// account's home rather than spelled out, because a written-out
/// default is one developer's absolute path and the release snapshot's
/// machine-path gate refuses to build a tree that ships one.
/// `fixtures/highres/generate_goldens.py` resolves the same root the
/// same way, which is why `meta.json` records the cached tile by NAME.
fn dem_cache() -> PathBuf {
    std::env::var("GPUWM_HIGHRES_DEM_CACHE")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            std::env::var("USERPROFILE")
                .or_else(|_| std::env::var("HOME"))
                .map(PathBuf::from)
                .unwrap_or_default()
                .join("arwen-verify-232")
                .join("hrcache")
                .join("copernicus_dem_glo30")
        })
}

#[test]
fn geotiff_reads_the_real_cached_tile_when_present() {
    let meta = meta();
    let summary = &meta["real_tile_summary"];
    let path = dem_cache().join(summary["path"].as_str().unwrap());
    if !path.is_file() {
        println!("real tile absent on this box; committed clips carry \
                  the decode contract");
        return;
    }
    let observed = highres::sha256_file(&path).expect("hash runs");
    assert_eq!(observed, summary["sha256"].as_str().unwrap(),
               "the cached tile changed since the goldens were cut");
    let mut reader = geotiff::TiffReader::open(&path).expect("tile opens");
    let expect_shape = shape(summary);
    assert_eq!((reader.height, reader.width),
               (expect_shape[0], expect_shape[1]));
    let window: Vec<i64> = summary["window_col_row_w_h"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_i64().unwrap())
        .collect();
    let values = reader
        .read_window_raw(window[0] as usize, window[1] as usize,
                         window[2] as usize, window[3] as usize)
        .expect("window reads");
    let sum: f64 = values.iter().sum();
    let expect_sum = summary["window_sum"].as_f64().unwrap();
    assert!(
        (sum - expect_sum).abs() <= expect_sum.abs() * 1e-9 + 1e-3,
        "window sum {sum} vs rasterio {expect_sum}"
    );
    let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    assert_eq!(min, summary["window_min"].as_f64().unwrap());
    assert_eq!(max, summary["window_max"].as_f64().unwrap());
}

/// Test-local copy of the WPS smoother (`smth_desmth_special`),
/// validated byte-for-byte against the Python's output on the golden
/// plane below; lane 2's `crate::smooth` implementation replaces it at
/// integration (the production path calls `crate::smooth`, not this).
fn local_smth_desmth_special(a: &Grid2, passes: usize) -> Grid2 {
    fn one_pass(a: &Grid2, coef: f64) -> Grid2 {
        let (ny, nx) = (a.ny, a.nx);
        let mut mid = a.clone();
        for j in 0..ny {
            for i in 1..nx.saturating_sub(1) {
                let c = a.data[j * nx + i];
                let l = a.data[j * nx + i - 1];
                let r = a.data[j * nx + i + 1];
                mid.data[j * nx + i] = c + coef * (0.5 * (l + r) - c);
            }
        }
        let mut out = mid.clone();
        for j in 1..ny.saturating_sub(1) {
            for i in 0..nx {
                let c = mid.data[j * nx + i];
                let d = mid.data[(j - 1) * nx + i];
                let u = mid.data[(j + 1) * nx + i];
                out.data[j * nx + i] = c + coef * (0.5 * (d + u) - c);
            }
        }
        out
    }
    let original = a.clone();
    let mut out = a.clone();
    for _ in 0..passes {
        out = one_pass(&out, 0.5);
        out = one_pass(&out, -0.52);
    }
    for index in 0..out.data.len() {
        if original.data[index] >= 0.0 && out.data[index] < 0.0 {
            out.data[index] = original.data[index];
        }
    }
    out
}

#[test]
fn terrain_warp_holds_the_recorded_tolerance() {
    let meta = meta();
    let warp_meta = &meta["terrain_warp"];
    let spec = grid_spec(&warp_meta["grid_spec"]);
    let halo = warp_meta["halo"].as_u64().unwrap() as usize;
    let extended = highres::extended_spec(&spec, halo);

    // Geometry parity first.
    let (_crs, transform, (ny, nx)) =
        highres::raster_geometry(&extended).expect("geometry");
    let expect_transform: Vec<f64> = warp_meta["extended_transform"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();
    for (index, (mine, theirs)) in
        transform.iter().zip(&expect_transform).enumerate()
    {
        let tolerance = if index == 2 || index == 5 { 1e-6 } else { 1e-12 };
        assert!(
            (mine - theirs).abs() <= tolerance,
            "raster geometry [{index}]: {mine} vs {theirs}"
        );
    }
    let expect_shape: Vec<usize> = warp_meta["extended_shape"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();
    assert_eq!((ny, nx), (expect_shape[0], expect_shape[1]));

    // The warp itself, tolerance-gated.
    let clip = fixtures().join("terrain_clip.tif");
    let source =
        geotiff::read_band1(&clip, None, None, 1.0).expect("clip reads");
    let warped = highres::resample_continuous(
        &source,
        &extended,
        Resampling::Average,
    )
    .expect("warp runs");
    let expect = read_f64(&warp_meta["warped"]);
    assert_eq!(warped.data.len(), expect.len());
    let mut max_delta = 0.0f64;
    let mut sum_delta = 0.0f64;
    for (mine, theirs) in warped.data.iter().zip(&expect) {
        assert!(theirs.is_finite() && mine.is_finite(),
                "coverage must be complete on both sides");
        let delta = (mine - theirs).abs();
        max_delta = max_delta.max(delta);
        sum_delta += delta;
    }
    let mean_delta = sum_delta / expect.len() as f64;
    println!(
        "terrain warp vs rasterio average: max |delta| = {max_delta:.3} m, \
         mean |delta| = {mean_delta:.4} m"
    );
    assert!(max_delta
        <= warp_meta["max_abs_delta_cap_m"].as_f64().unwrap());
    assert!(mean_delta
        <= warp_meta["mean_abs_delta_cap_m"].as_f64().unwrap());

    // The smoother copy is byte-exact against the Python on the
    // PYTHON-warped plane (isolating the smoother from the warp).
    let python_warped = Grid2 { ny, nx, data: expect };
    let smoothed = local_smth_desmth_special(&python_warped, 1);
    let expect_smoothed = read_f64(&warp_meta["smoothed"]);
    for (index, (mine, theirs)) in
        smoothed.data.iter().zip(&expect_smoothed).enumerate()
    {
        assert!(
            mine.to_bits() == theirs.to_bits(),
            "smoother diverges at {index}: {mine} vs {theirs}"
        );
    }

    // End-to-end (Rust warp + smoother + crop) stays inside the same
    // physical tolerance against the full Python build.
    let smoothed_mine = local_smth_desmth_special(&warped, 1);
    let expect_hgt = read_f64(&warp_meta["hgt"]);
    let (cny, cnx) = ((spec.e_sn - 1) as usize, (spec.e_we - 1) as usize);
    let mut max_delta = 0.0f64;
    for row in 0..cny {
        for col in 0..cnx {
            let mine =
                smoothed_mine.data[(row + halo) * nx + halo + col];
            let theirs = expect_hgt[row * cnx + col];
            max_delta = max_delta.max((mine - theirs).abs());
        }
    }
    println!("terrain HGT end-to-end max |delta| = {max_delta:.3} m");
    assert!(max_delta
        <= warp_meta["max_abs_delta_cap_m"].as_f64().unwrap());
}

#[test]
fn landcover_fractions_hold_the_recorded_tolerance() {
    let meta = meta();
    let lc = &meta["landcover"];
    let spec = grid_spec(&lc["grid_spec"]);
    let halo = lc["halo"].as_u64().unwrap() as usize;
    let extended = highres::extended_spec(&spec, halo);
    let nodata = lc["nodata"].as_f64();
    let (raw, effective_nodata) = geotiff::read_band1_raw(
        &fixtures().join("landcover.tif"),
        None,
        None,
    )
    .expect("landcover reads");
    assert_eq!(effective_nodata, nodata, "nodata tag decodes");
    match &raw.crs {
        Crs::AlbersConusNad83 { lat_1, lat_2, lat_0, lon_0, .. } => {
            assert_eq!((*lat_1, *lat_2, *lat_0, *lon_0),
                       (29.5, 45.5, 23.0, -96.0));
        }
        other => panic!("landcover CRS decoded as {other:?}"),
    }
    let mapping: BTreeMap<i64, i64> = lc["mapping"]
        .as_array()
        .unwrap()
        .iter()
        .map(|pair| {
            let pair = pair.as_array().unwrap();
            (pair[0].as_i64().unwrap(), pair[1].as_i64().unwrap())
        })
        .collect();
    let fractions = highres::resample_mapped_categories(
        &raw,
        "landcover.tif",
        &extended,
        &mapping,
        21,
        effective_nodata,
    )
    .expect("category warp runs");
    let expect = read_f64(&lc["fractions"]);
    let expect_shape = shape(&lc["fractions"]);
    assert_eq!(
        (fractions.planes, fractions.ny, fractions.nx),
        (expect_shape[0], expect_shape[1], expect_shape[2])
    );
    let cells = fractions.ny * fractions.nx;
    let mut max_l1 = 0.0f64;
    let mut sum_l1 = 0.0f64;
    let mut nan_mismatch = 0usize;
    for cell in 0..cells {
        let mut l1 = 0.0f64;
        for plane in 0..fractions.planes {
            let mine = fractions.data[plane * cells + cell];
            let theirs = expect[plane * cells + cell];
            if mine.is_nan() != theirs.is_nan() {
                nan_mismatch += 1;
                continue;
            }
            if mine.is_finite() {
                l1 += (mine - theirs).abs();
            }
        }
        max_l1 = max_l1.max(l1);
        sum_l1 += l1;
    }
    println!(
        "landcover fractions: max per-cell L1 = {max_l1:.4}, mean = {:.5}, \
         nan mismatches = {nan_mismatch}",
        sum_l1 / cells as f64
    );
    assert_eq!(nan_mismatch, 0, "coverage decisions must agree");
    assert!(max_l1 <= lc["per_cell_l1_cap"].as_f64().unwrap());
    assert!(sum_l1 / cells as f64 <= lc["mean_l1_cap"].as_f64().unwrap());
}

#[test]
fn soil_depth_means_are_byte_parity_and_fractions_hold_tolerance() {
    let meta = meta();
    let soil = &meta["soil"];
    let spec = grid_spec(&soil["grid_spec"]);
    let halo = soil["halo"].as_u64().unwrap() as usize;
    let extended = highres::extended_spec(&spec, halo);
    let src_shape: Vec<usize> = soil["source_shape"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();
    let transform: Vec<f64> = soil["source_transform"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();
    let scale = soil["scale_factor"].as_f64().unwrap();
    let nodata = soil["nodata"].as_f64().unwrap();
    let crs_override = soil["crs_override"].as_str().unwrap();

    for (layer, layer_meta) in soil["layers"].as_object().unwrap() {
        let weights: Vec<(String, f64)> = layer_meta["weights"]
            .as_array()
            .unwrap()
            .iter()
            .map(|pair| {
                let pair = pair.as_array().unwrap();
                (
                    pair[0].as_str().unwrap().to_string(),
                    pair[1].as_f64().unwrap(),
                )
            })
            .collect();
        let mut planes: BTreeMap<String, Vec<Vec<f64>>> = BTreeMap::new();
        for component in ["sand", "silt", "clay"] {
            let mut stack = Vec::new();
            for (depth, _) in &weights {
                let path = fixtures()
                    .join(format!("soil_{component}_{depth}.tif"));
                let spec = highres::BoundRasterSpec {
                    sha256: highres::sha256_file(&path).unwrap(),
                    path: path.clone(),
                    expected_bytes: None,
                    crs_override: Some(crs_override.to_string()),
                    nodata_override: Some(nodata),
                    scale_factor: scale,
                };
                let raster = spec.open().expect("soil window reads");
                assert_eq!((raster.ny, raster.nx),
                           (src_shape[0], src_shape[1]));
                stack.push(raster.values);
            }
            planes.insert(component.into(), stack);
        }
        let weight_values: Vec<f64> =
            weights.iter().map(|(_, w)| *w).collect();
        let (category, valid, raw_total) = highres::soilgrids_categories(
            &planes,
            &weight_values,
            src_shape[0] * src_shape[1],
        )
        .expect("depth means run");

        let expect_category = read_i16(&layer_meta["category"]);
        assert_eq!(category, expect_category, "{layer}: category bytes");
        let expect_valid = read_u8(&layer_meta["valid"]);
        let valid_u8: Vec<u8> =
            valid.iter().map(|ok| *ok as u8).collect();
        assert_eq!(valid_u8, expect_valid, "{layer}: valid mask");
        let expect_total = read_f64(&layer_meta["raw_total"]);
        for (index, (mine, theirs)) in
            raw_total.iter().zip(&expect_total).enumerate()
        {
            let same = (mine.is_nan() && theirs.is_nan())
                || mine.to_bits() == theirs.to_bits();
            assert!(same, "{layer}: raw_total[{index}] {mine} vs {theirs}");
        }

        // Fractions: tolerance parity on the extended grid.
        let carrier = Raster {
            ny: src_shape[0],
            nx: src_shape[1],
            values: Vec::new(),
            transform: [
                transform[0], transform[1], transform[2],
                transform[3], transform[4], transform[5],
            ],
            crs: Crs::parse_override(crs_override).unwrap(),
        };
        let (dst_crs, dst_transform, (dny, dnx)) =
            highres::raster_geometry(&extended).unwrap();
        let fractions = warp::reproject_category_fractions(
            &category,
            &valid,
            &carrier,
            &dst_crs,
            dst_transform,
            dny,
            dnx,
            16,
        )
        .expect("soil fraction warp runs");
        let expect = read_f64(&layer_meta["fractions"]);
        let cells = dny * dnx;
        let mut max_l1 = 0.0f64;
        let mut sum_l1 = 0.0f64;
        let mut nan_mismatch = 0usize;
        for cell in 0..cells {
            let mut l1 = 0.0f64;
            for plane in 0..16 {
                let mine = fractions.data[plane * cells + cell];
                let theirs = expect[plane * cells + cell];
                if mine.is_nan() != theirs.is_nan() {
                    nan_mismatch += 1;
                } else if mine.is_finite() {
                    l1 += (mine - theirs).abs();
                }
            }
            max_l1 = max_l1.max(l1);
            sum_l1 += l1;
        }
        println!(
            "soil {layer}: max per-cell L1 = {max_l1:.4}, mean = {:.5}, \
             nan mismatches = {nan_mismatch}",
            sum_l1 / cells as f64
        );
        assert_eq!(nan_mismatch, 0);
        assert!(max_l1 <= soil["per_cell_l1_cap"].as_f64().unwrap());
        assert!(sum_l1 / cells as f64
            <= soil["mean_l1_cap"].as_f64().unwrap());
    }
}

#[test]
fn usda_triangle_is_byte_parity() {
    let meta = meta();
    let usda = &meta["usda"];
    let mut sand = Vec::new();
    let mut silt = Vec::new();
    let mut clay = Vec::new();
    for s in 0..=100i64 {
        for si in 0..=(100 - s) {
            sand.push(s as f64);
            silt.push(si as f64);
            clay.push((100 - s - si) as f64);
        }
    }
    let categories =
        highres::usda_texture_category(&sand, &silt, &clay).unwrap();
    assert_eq!(categories, read_i16(&usda["categories"]));

    let scale = usda["scale"].as_f64().unwrap();
    let scaled = |v: &[f64]| -> Vec<f64> {
        v.iter().map(|x| x * scale).collect()
    };
    let categories_scaled = highres::usda_texture_category(
        &scaled(&sand),
        &scaled(&silt),
        &scaled(&clay),
    )
    .unwrap();
    assert_eq!(categories_scaled, read_i16(&usda["categories_scaled"]));

    let refusal =
        highres::usda_texture_category(&[0.0], &[0.0], &[0.0]).unwrap_err();
    assert_eq!(
        refusal.to_string(),
        usda["invalid_total_message"].as_str().unwrap()
    );
}

#[test]
fn nearest_donors_are_byte_parity() {
    let meta = meta();
    let donors = &meta["donors"];
    for case in donors["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let mask_shape = shape(&case["mask"]);
        let mask: Vec<bool> = read_u8(&case["mask"])
            .into_iter()
            .map(|v| v != 0)
            .collect();
        let (donor_y, donor_x) =
            highres::nearest_donors(&mask, mask_shape[0], mask_shape[1])
                .expect("donors run");
        assert_eq!(donor_y, read_i32(&case["donor_y"]), "{name}: donor_y");
        assert_eq!(donor_x, read_i32(&case["donor_x"]), "{name}: donor_x");
    }
    let refusal = highres::nearest_donors(&[false; 9], 3, 3).unwrap_err();
    assert_eq!(
        refusal.to_string(),
        donors["empty_message"].as_str().unwrap()
    );
}

fn fieldset_from(entries: &serde_json::Value) -> FieldSet {
    let mut set = FieldSet::default();
    for entry in entries.as_array().unwrap() {
        let name = entry["name"].as_str().unwrap().to_string();
        let planes = entry["planes"].as_u64().unwrap() as usize;
        let ny = entry["ny"].as_u64().unwrap() as usize;
        let nx = entry["nx"].as_u64().unwrap() as usize;
        let data = read_f64(&entry["data"]);
        let field = if planes == 1 {
            Field::Plane(Grid2 { ny, nx, data })
        } else {
            Field::Stack(Stack3 { planes, ny, nx, data })
        };
        set.fields.insert(name, field);
    }
    set
}

fn assert_fieldsets_bit_equal(
    mine: &FieldSet,
    expect: &serde_json::Value,
    label: &str,
) {
    let expect_set = fieldset_from(expect);
    let mine_names: Vec<&String> = mine.fields.keys().collect();
    let expect_names: Vec<&String> = expect_set.fields.keys().collect();
    assert_eq!(mine_names, expect_names, "{label}: field inventory");
    for (name, field) in &expect_set.fields {
        let mine_field = mine.fields.get(name).unwrap();
        assert_eq!(mine_field.dims(), field.dims(), "{label}/{name}: dims");
        for (index, (a, b)) in mine_field
            .data()
            .iter()
            .zip(field.data())
            .enumerate()
        {
            let same = (a.is_nan() && b.is_nan()) || a.to_bits() == b.to_bits();
            assert!(same, "{label}/{name}[{index}]: {a} vs {b}");
        }
    }
}

#[test]
fn full_merge_is_byte_parity_with_python() {
    let meta = meta();
    let case = &meta["merge_full"];
    let baseline = fieldset_from(&case["baseline"]);
    let overrides = fieldset_from(&case["overrides"]);
    let (merged, audit) =
        highres::merge_highres_overrides(&baseline, &overrides)
            .expect("merge runs");
    assert_fieldsets_bit_equal(&merged, &case["merged"], "merge_full");
    let expect_audit = case["audit"].as_object().unwrap();
    assert_eq!(
        audit.newly_land_nearest_climatology_fallback_cells,
        expect_audit["newly_land_nearest_climatology_fallback_cells"]
            .as_u64()
            .unwrap()
    );
    assert_eq!(
        audit.newly_water_masked_cells,
        expect_audit["newly_water_masked_cells"].as_u64().unwrap()
    );
    assert_eq!(
        audit.unchanged_land_water_cells.unwrap(),
        expect_audit["unchanged_land_water_cells"].as_u64().unwrap()
    );
    assert_eq!(
        audit.deep_soil_water_masked_cells.unwrap(),
        expect_audit["deep_soil_water_masked_cells"].as_u64().unwrap()
    );
}

#[test]
fn terrain_merge_is_byte_parity_with_python() {
    let meta = meta();
    let case = &meta["merge_terrain"];
    let baseline = fieldset_from(&case["baseline"]);
    let hgt_shape = shape(&case["hgt_override"]);
    let hgt = Grid2 {
        ny: hgt_shape[0],
        nx: hgt_shape[1],
        data: read_f64(&case["hgt_override"]),
    };
    let (merged, audit) =
        highres::merge_terrain_override(&baseline, &hgt)
            .expect("terrain merge runs");
    assert_fieldsets_bit_equal(&merged, &case["merged"], "merge_terrain");
    let expect_audit = case["audit"].as_object().unwrap();
    assert_eq!(
        audit.terrain_cells_changed.unwrap(),
        expect_audit["terrain_cells_changed"].as_u64().unwrap()
    );
    assert_eq!(
        audit.land_water_cells_unchanged.unwrap(),
        expect_audit["land_water_cells_unchanged"].as_u64().unwrap()
    );
}

#[test]
fn merge_refusal_decisions_and_messages_match_python() {
    let meta = meta();
    let refusals = &meta["merge_refusals"];
    let case = &meta["merge_full"];
    let overrides = fieldset_from(&case["overrides"]);

    let mut dead = fieldset_from(&case["baseline"]);
    if let Some(Field::Plane(grid)) = dead.fields.get_mut("SOILTEMP") {
        grid.data.iter_mut().for_each(|v| *v = 0.0);
    }
    let error = highres::merge_highres_overrides(&dead, &overrides)
        .expect_err("0 K land deep soil must refuse");
    assert_eq!(
        error.to_string(),
        refusals["whole_domain"].as_str().unwrap()
    );

    let mut holed = fieldset_from(&case["baseline"]);
    if let Some(Field::Plane(grid)) = holed.fields.get_mut("SOILTEMP") {
        let nx = grid.nx;
        grid.data[6 * nx + 6] = 0.0;
    }
    let error = highres::merge_highres_overrides(&holed, &overrides)
        .expect_err("single 0 K land cell must refuse");
    assert_eq!(
        error.to_string(),
        refusals["single_cell"].as_str().unwrap()
    );

    let baseline = fieldset_from(&case["baseline"]);
    let error = highres::merge_terrain_override(
        &baseline,
        &Grid2 { ny: 5, nx: 4, data: vec![0.0; 20] },
    )
    .expect_err("shape mismatch must refuse");
    assert_eq!(
        error.to_string(),
        refusals["terrain_shape"].as_str().unwrap()
    );
}

#[test]
fn mosaic_matches_rasterio_merge_within_tolerance() {
    let meta = meta();
    let mosaic_meta = &meta["mosaic"];
    let mut tiles = Vec::new();
    for tile in mosaic_meta["tiles"].as_array().unwrap() {
        let path = fixtures().join(tile.as_str().unwrap());
        let (raster, _) =
            geotiff::read_band1_raw(&path, None, None).expect("tile reads");
        tiles.push(raster);
    }
    let bounds: Vec<f64> = mosaic_meta["bounds_wsen"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();
    let resolution = mosaic_meta["resolution_deg"].as_f64().unwrap();
    let (mosaic, holes) = warp::mosaic(
        &tiles,
        [bounds[0], bounds[1], bounds[2], bounds[3]],
        Some(resolution),
        None,
    )
    .expect("mosaic runs");
    let expect_shape: Vec<usize> = mosaic_meta["shape"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();
    assert_eq!((mosaic.ny, mosaic.nx), (expect_shape[0], expect_shape[1]));
    let expect_transform: Vec<f64> = mosaic_meta["transform"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();
    for (mine, theirs) in mosaic.transform.iter().zip(&expect_transform) {
        assert!((mine - theirs).abs() <= 1e-12, "{mine} vs {theirs}");
    }
    // The Rust mosaic's holes must be a SUBSET of rasterio's (it may
    // additionally cover rasterio's dropped sub-pixel seam columns,
    // never the reverse); values are compared where BOTH covered.
    let python_holes: Vec<bool> = read_u8(&mosaic_meta["hole_mask"])
        .into_iter()
        .map(|v| v != 0)
        .collect();
    let python_hole_count =
        mosaic_meta["holes"].as_u64().unwrap() as usize;
    assert!(holes <= python_hole_count,
            "Rust mosaic has {holes} holes, rasterio {python_hole_count}");
    let expect = read_f32(&mosaic_meta["filled"]);
    let mut exact = 0usize;
    let mut compared = 0usize;
    let mut max_delta = 0.0f64;
    let mut sum_delta = 0.0f64;
    for (index, (mine, theirs)) in
        mosaic.values.iter().zip(&expect).enumerate()
    {
        if mine.is_nan() {
            assert!(python_holes[index],
                    "Rust hole at {index} where rasterio has data");
            continue;
        }
        if python_holes[index] {
            continue; // rasterio's dropped seam column, documented
        }
        compared += 1;
        let mine = *mine as f32;
        if mine.to_bits() == theirs.to_bits() {
            exact += 1;
        } else {
            let delta = (mine - theirs).abs() as f64;
            max_delta = max_delta.max(delta);
            sum_delta += delta;
        }
    }
    let fraction = exact as f64 / compared as f64;
    let mean_delta = sum_delta / compared as f64;
    println!(
        "mosaic vs rasterio.merge: exact fraction = {fraction:.5} of \
         {compared} mutually covered, max |delta| among differing = \
         {max_delta:.2} m, mean over compared = {mean_delta:.3} m, \
         holes {holes} (rasterio {python_hole_count})"
    );
    assert!(fraction
        >= mosaic_meta["exact_fraction_floor"].as_f64().unwrap());
    assert!(max_delta
        <= mosaic_meta["max_abs_delta_cap_m"].as_f64().unwrap());
    assert!(mean_delta
        <= mosaic_meta["mean_abs_delta_cap_m"].as_f64().unwrap());
}

#[test]
fn tile_id_enumeration_is_byte_parity() {
    let meta = meta();
    let tile_meta = &meta["tile_ids"];
    for case in tile_meta["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let bbox: Vec<f64> = case["bbox"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();
        let bbox = [bbox[0], bbox[1], bbox[2], bbox[3]];
        let expect: Vec<String> = case["copernicus"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap().to_string())
            .collect();
        assert_eq!(
            highres::copernicus_dem_tile_ids(bbox).unwrap(),
            expect,
            "{name}: copernicus ids"
        );
        let expect: Vec<String> = case["srtm"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap().to_string())
            .collect();
        assert_eq!(highres::srtm_tile_ids(bbox).unwrap(), expect,
                   "{name}: srtm ids");
        match case["three_dep"].as_array() {
            Some(ids) => {
                let expect: Vec<String> = ids
                    .iter()
                    .map(|v| v.as_str().unwrap().to_string())
                    .collect();
                assert_eq!(
                    highres::three_dep_tile_ids(bbox).unwrap(),
                    expect,
                    "{name}: 3dep ids"
                );
            }
            None => {
                assert!(highres::three_dep_tile_ids(bbox).is_err(),
                        "{name}: 3dep must refuse");
            }
        }
    }
    for (tile, bbox) in tile_meta["tile_bboxes"].as_object().unwrap() {
        let expect: Vec<f64> = bbox
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();
        assert_eq!(
            highres::one_degree_tile_bbox(tile).unwrap().to_vec(),
            expect,
            "{tile}: bbox"
        );
    }
    assert!(highres::one_degree_tile_bbox("X12_00_W105_00").is_err());
}

#[test]
fn geotiff_writer_round_trips_through_its_own_reader() {
    // f32 + predictor 3 and u8 + predictor 2, both tiled, both with a
    // nodata sentinel, geographic and Albers CRS spellings.
    let dir = std::env::temp_dir().join("static_fields_highres_test");
    std::fs::create_dir_all(&dir).unwrap();
    let ny = 300usize;
    let nx = 517usize;
    let mut values = vec![0.0f64; ny * nx];
    for (index, value) in values.iter_mut().enumerate() {
        *value = ((index % 977) as f64) * 0.75 - 100.0;
    }
    values[42] = f64::NAN;
    let raster = Raster {
        ny,
        nx,
        values: values.clone(),
        transform: [0.001, 0.0, 7.25, 0.0, -0.001, 46.75],
        crs: Crs::Geographic,
    };
    let path = dir.join("roundtrip_f32.tif");
    geotiff::write_band1(&path, &raster, geotiff::SampleType::F32,
                         Some(-9999.0))
        .expect("f32 write");
    let (back, nodata) =
        geotiff::read_band1_raw(&path, None, None).expect("f32 read");
    assert_eq!(nodata, Some(-9999.0));
    assert_eq!((back.ny, back.nx), (ny, nx));
    assert_eq!(back.crs, Crs::Geographic);
    for (index, (a, b)) in back.values.iter().zip(&values).enumerate() {
        let expect =
            if b.is_nan() { -9999.0f32 } else { *b as f32 };
        assert!((*a as f32).to_bits() == expect.to_bits(),
                "f32 roundtrip [{index}]: {a} vs {expect}");
    }

    let mut category = vec![0.0f64; ny * nx];
    for (index, value) in category.iter_mut().enumerate() {
        *value = ((index / 53) % 21 + 1) as f64;
    }
    let raster = Raster {
        ny,
        nx,
        values: category.clone(),
        transform: [30.0, 0.0, 100_000.0, 0.0, -30.0, 2_000_000.0],
        crs: Crs::AlbersConusNad83 {
            lat_1: 29.5,
            lat_2: 45.5,
            lat_0: 23.0,
            lon_0: -96.0,
            false_easting: 0.0,
            false_northing: 0.0,
        },
    };
    let path = dir.join("roundtrip_u8.tif");
    geotiff::write_band1(&path, &raster, geotiff::SampleType::U8,
                         Some(250.0))
        .expect("u8 write");
    let (back, nodata) =
        geotiff::read_band1_raw(&path, None, None).expect("u8 read");
    assert_eq!(nodata, Some(250.0));
    for (a, b) in back.values.iter().zip(&category) {
        assert_eq!(*a, *b);
    }
    match &back.crs {
        Crs::AlbersConusNad83 { lat_1, lon_0, .. } => {
            assert_eq!((*lat_1, *lon_0), (29.5, -96.0));
        }
        other => panic!("u8 roundtrip CRS {other:?}"),
    }

    // Determinism: the same raster writes the same bytes.
    let path2 = dir.join("roundtrip_u8_again.tif");
    geotiff::write_band1(&path2, &raster, geotiff::SampleType::U8,
                         Some(250.0))
        .expect("u8 rewrite");
    assert_eq!(std::fs::read(&path).unwrap(),
               std::fs::read(&path2).unwrap(),
               "writer must be byte-deterministic");
}
