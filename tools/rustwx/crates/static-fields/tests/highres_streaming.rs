//! Real TIFF regressions for bounded high-resolution static preparation.
use static_fields::raster::{Crs, Raster, geotiff, warp};
use std::cell::Cell;
use std::path::PathBuf;

struct Scratch(PathBuf);
impl Scratch {
    fn new(label: &str) -> Self {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("target")
            .join(format!("streaming-{label}-{}", std::process::id()));
        std::fs::create_dir_all(&root).unwrap();
        Self(root)
    }
    fn file(&self, name: &str) -> PathBuf {
        self.0.join(name)
    }
}
impl Drop for Scratch {
    fn drop(&mut self) {
        for entry in std::fs::read_dir(&self.0).unwrap() {
            let path = entry.unwrap().path();
            let size = std::fs::metadata(&path).unwrap().len();
            eprintln!(
                "deleted generated test file {} {size} bytes",
                path.display()
            );
            std::fs::remove_file(path).unwrap();
        }
        std::fs::remove_dir(&self.0).unwrap();
    }
}
fn bits(a: &[f64], b: &[f64]) {
    assert_eq!(a.len(), b.len());
    for (index, (a, b)) in a.iter().zip(b).enumerate() {
        assert_eq!(a.to_bits(), b.to_bits(), "sample {index}");
    }
}
fn source() -> Raster {
    let n = 2200;
    Raster {
        ny: n,
        nx: n,
        values: (0..n * n)
            .map(|at| {
                if at % 73 == 0 {
                    f64::NAN
                } else {
                    ((at * 17 % 65521) as f32 * 0.25) as f64
                }
            })
            .collect(),
        transform: [1.0, 0.0, 0.0, 0.0, -1.0, n as f64],
        crs: Crs::Geographic,
    }
}
#[test]
fn real_tiff_source_larger_than_cache_matches_all_dense_continuous_methods() {
    let tmp = Scratch::new("continuous");
    let path = tmp.file("source.tif");
    geotiff::write_band1(&path, &source(), geotiff::SampleType::F32, None).unwrap();
    let dense = geotiff::read_band1(&path, None, None, 1.0).unwrap();
    let mut reader = geotiff::TiffReader::open(&path).unwrap();
    reader.set_cache_budget(8 * 1024 * 1024);
    let peak = Cell::new(0usize);
    let transform = [68.75, 0.0, 0.0, 0.0, -68.75, 2200.0];
    for method in [
        warp::Resampling::Average,
        warp::Resampling::Bilinear,
        warp::Resampling::Nearest,
    ] {
        let expected =
            warp::reproject_continuous(&dense, &Crs::Geographic, transform, 32, 32, method)
                .unwrap();
        let actual = warp::reproject_continuous_windowed(
            &dense,
            &Crs::Geographic,
            transform,
            32,
            32,
            method,
            |c, r, w, h| {
                peak.set(peak.get().max(w * h));
                reader.read_window_raw(c, r, w, h)
            },
        )
        .unwrap();
        bits(&expected.data, &actual.data);
    }
    assert!(dense.nx * dense.ny > 4 * 1024 * 1024);
    assert!(
        peak.get() <= 4 * 1024 * 1024,
        "source windows must split to honor the fixed budget"
    );
}
#[test]
fn real_tiff_category_windows_match_dense_fractions_and_holes() {
    let tmp = Scratch::new("categories");
    let path = tmp.file("source.tif");
    let mut raster = source();
    for (at, value) in raster.values.iter_mut().enumerate() {
        *value = if at % 73 == 0 {
            0.0
        } else {
            (at % 4 + 1) as f64
        };
    }
    geotiff::write_band1(&path, &raster, geotiff::SampleType::U8, Some(0.0)).unwrap();
    let mapped: Vec<i16> = raster.values.iter().map(|v| *v as i16).collect();
    let valid: Vec<bool> = mapped.iter().map(|v| *v > 0).collect();
    let transform = [68.75, 0.0, 0.0, 0.0, -68.75, 2200.0];
    let expected = warp::reproject_category_fractions(
        &mapped,
        &valid,
        &raster,
        &Crs::Geographic,
        transform,
        32,
        32,
        4,
    )
    .unwrap();
    let mut reader = geotiff::TiffReader::open(&path).unwrap();
    reader.set_cache_budget(8 * 1024 * 1024);
    let actual = warp::reproject_categories_windowed(
        &raster,
        &Crs::Geographic,
        transform,
        32,
        32,
        4,
        |c, r, w, h| {
            Ok(reader
                .read_window_u8(c, r, w, h)?
                .into_iter()
                .map(|v| v as i16)
                .collect())
        },
    )
    .unwrap();
    bits(&expected.data, &actual.data);
}
#[test]
fn streamed_mosaic_matches_dense_file_bytes_with_different_tile_resolutions() {
    let tmp = Scratch::new("mosaic");
    let tiles: Vec<Raster> = [(300, 1.0, -0.5, 300.5), (160, 2.0, 298.5, 300.5)]
        .into_iter()
        .map(|(nx, res, x, y)| Raster {
            nx,
            ny: 300,
            values: (0..nx * 300)
                .map(|at| {
                    if at % 41 == 0 {
                        -9999.0
                    } else if at % 137 == 0 {
                        -32767.0
                    } else {
                        (at % 1000) as f64
                    }
                })
                .collect(),
            transform: [res, 0.0, x, 0.0, -res, y],
            crs: Crs::Geographic,
        })
        .collect();
    let paths: Vec<PathBuf> = tiles
        .iter()
        .enumerate()
        .map(|(i, t)| {
            let p = tmp.file(&format!("source-{i}.tif"));
            geotiff::write_band1(&p, t, geotiff::SampleType::F32, Some(-9999.0)).unwrap();
            p
        })
        .collect();
    let masked: Vec<Raster> = paths
        .iter()
        .map(|p| geotiff::read_band1(p, None, None, 1.0).unwrap())
        .collect();
    let bounds = [-1.3, -303.0, 620.2, 302.1];
    for (i, resolution) in [None, Some(1.0)].into_iter().enumerate() {
        let (dense, holes) = warp::mosaic(&masked, bounds, resolution, Some(-32767.0)).unwrap();
        let expected = tmp.file(&format!("dense-{i}.tif"));
        let actual = tmp.file(&format!("streamed-{i}.tif"));
        geotiff::write_band1(&expected, &dense, geotiff::SampleType::F32, None).unwrap();
        let (ny, nx, reached, _) = warp::mosaic_tiffs(
            &paths,
            bounds,
            resolution,
            Some(-32767.0),
            &actual,
            None,
            None,
        )
        .unwrap();
        assert_eq!((ny, nx, reached), (dense.ny, dense.nx, holes));
        assert_eq!(
            std::fs::read(expected).unwrap(),
            std::fs::read(actual).unwrap()
        );
    }
}
#[test]
fn mosaic_source_inventory_does_not_open_all_files_at_once() {
    let tmp = Scratch::new("inventory");
    let path = tmp.file("source.tif");
    let raster = Raster {
        ny: 1,
        nx: 1,
        values: vec![42.0],
        transform: [1.0, 0.0, 0.0, 0.0, -1.0, 1.0],
        crs: Crs::Geographic,
    };
    geotiff::write_band1(&path, &raster, geotiff::SampleType::F32, None).unwrap();
    let paths = vec![path; 1100];
    let output = tmp.file("output.tif");
    let result = warp::mosaic_tiffs(
        &paths,
        [0.0, 0.0, 1.0, 1.0],
        None,
        None,
        &output,
        None,
        None,
    )
    .unwrap();
    assert_eq!((result.0, result.1, result.2), (1, 1, 0));
    assert_eq!(
        geotiff::read_band1(&output, None, None, 1.0)
            .unwrap()
            .values,
        vec![42.0]
    );
}
#[test]
fn large_mosaic_uses_bigtiff_header_without_a_full_plane_allocation() {
    let tmp = Scratch::new("bigtiff");
    let path = tmp.file("large.tif");
    let result = geotiff::write_band1_tiles(
        &path,
        20_000,
        50_000,
        &[1.0, 0.0, 0.0, 0.0, -1.0, 20_000.0],
        &Crs::Geographic,
        geotiff::SampleType::F32,
        None,
        |_, _, _, _| {
            Err(static_fields::StaticError::Invalid(
                "stop after header".into(),
            ))
        },
    );
    assert!(result.is_err());
    let bytes = std::fs::read(&path).unwrap();
    assert_eq!(&bytes[..4], b"II+\0");
    let reader = geotiff::TiffReader::open(&path).unwrap();
    assert_eq!((reader.width, reader.height), (50_000, 20_000));
}

#[test]
fn mosaic_far_edge_prefilter_keeps_containment_rounding() {
    let tmp = Scratch::new("edge-rounding");
    let rx = 1.0 / 1800.0;
    let r = 1.0 / 3600.0;
    let ox = -0.5 * r;
    let raster = Raster {
        ny: 3,
        nx: 1800,
        values: (0..5400).map(|i| (i % 1800) as f64).collect(),
        transform: [rx, 0.0, 8.0 - 0.5 * rx, 0.0, -r, 1.0 + 0.5 * r],
        crs: Crs::Geographic,
    };
    let input = tmp.file("source.tif");
    geotiff::write_band1(&input, &raster, geotiff::SampleType::F32, None).unwrap();
    let source = geotiff::read_band1(&input, None, None, 1.0).unwrap();
    let west = ox + 32399.0 * r;
    let bounds = [west, 1.0 - r, west + r, 1.0];
    let (dense, holes) = warp::mosaic(&[source], bounds, Some(r), None).unwrap();
    assert!(dense.values.iter().any(|v| *v == 1799.0));
    let expected = tmp.file("dense.tif");
    let actual = tmp.file("streamed.tif");
    geotiff::write_band1(&expected, &dense, geotiff::SampleType::F32, None).unwrap();
    let (_, _, actual_holes, _) =
        warp::mosaic_tiffs(&[input], bounds, Some(r), None, &actual, None, None).unwrap();
    assert_eq!(actual_holes, holes);
    assert_eq!(
        std::fs::read(expected).unwrap(),
        std::fs::read(actual).unwrap()
    );
}
#[test]
fn unused_mapping_target_keeps_existing_acceptance_rule() {
    use static_fields::highres::{self, BoundRasterSpec};
    use static_fields::projection::{GridSpec, ProjectionKind};
    use std::collections::BTreeMap;
    let tmp = Scratch::new("unused-mapping");
    let path = tmp.file("source.tif");
    let raster = Raster {
        ny: 8,
        nx: 8,
        values: vec![1.0; 64],
        transform: [0.01, 0.0, -0.04, 0.0, -0.01, 0.04],
        crs: Crs::Geographic,
    };
    geotiff::write_band1(&path, &raster, geotiff::SampleType::U8, None).unwrap();
    let spec = GridSpec {
        kind: ProjectionKind::Mercator,
        ref_lat: 0.0,
        ref_lon: 0.0,
        truelat1: 0.0,
        truelat2: 0.0,
        stand_lon: 0.0,
        dx: 1000.0,
        dy: 1000.0,
        e_we: 3,
        e_sn: 3,
        known_x: 1.5,
        known_y: 1.5,
        moad_cen_lat: 0.0,
        moad_cen_lon: 0.0,
        lat_deg: Vec::new(),
        lon0_deg: 0.0,
        dlon_deg: 0.0,
    };
    let mapping: BTreeMap<i64, i64> = [(1, 1), (255, 22)].into_iter().collect();
    let expected =
        highres::resample_mapped_categories(&raster, "fixture", &spec, &mapping, 21, None).unwrap();
    let source = BoundRasterSpec {
        sha256: highres::sha256_file(&path).unwrap(),
        expected_bytes: Some(std::fs::metadata(&path).unwrap().len()),
        path,
        crs_override: None,
        nodata_override: None,
        scale_factor: 1.0,
    };
    let actual = highres::resample_mapped_categories_bound(&source, &spec, &mapping, 21).unwrap();
    bits(&expected.data, &actual.data);
}
#[test]
fn bounded_four_worker_tiff_warp_matches_serial_with_global_row_indices() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Barrier};
    struct LiveReader {
        reader: geotiff::TiffReader,
        live: Arc<AtomicUsize>,
    }
    impl Drop for LiveReader {
        fn drop(&mut self) {
            self.live.fetch_sub(1, Ordering::SeqCst);
        }
    }
    let tmp = Scratch::new("parallel-continuous");
    let path = tmp.file("source.tif");
    geotiff::write_band1(&path, &source(), geotiff::SampleType::F32, None).unwrap();
    let dense = geotiff::read_band1(&path, None, None, 1.0).unwrap();
    // Each worker starts at a nonzero global row that need not divide 32.
    // The fractional transform also reaches outside the source and over holes.
    let transform = [2200.0 / 71.0, 0.0, -13.5, 0.0, -2200.0 / 103.0, 2211.0];
    let live = Arc::new(AtomicUsize::new(0));
    let peak = Arc::new(AtomicUsize::new(0));
    let largest = Arc::new(AtomicUsize::new(0));
    for method in [
        warp::Resampling::Average,
        warp::Resampling::Bilinear,
        warp::Resampling::Nearest,
    ] {
        let mut serial = geotiff::TiffReader::open(&path).unwrap();
        serial.set_cache_budget(warp::CONTINUOUS_READER_CACHE_BYTES);
        let expected = warp::reproject_continuous_windowed(
            &dense,
            &Crs::Geographic,
            transform,
            103,
            71,
            method,
            |c, r, w, h| serial.read_window_raw(c, r, w, h),
        )
        .unwrap();
        let barrier = Arc::new(Barrier::new(warp::CONTINUOUS_WORKER_CAP));
        let actual = warp::reproject_continuous_windowed_parallel(
            &dense,
            &Crs::Geographic,
            transform,
            103,
            71,
            method,
            48,
            || {
                let mut reader = geotiff::TiffReader::open(&path)?;
                reader.set_cache_budget(warp::CONTINUOUS_READER_CACHE_BYTES);
                let active = live.fetch_add(1, Ordering::SeqCst) + 1;
                peak.fetch_max(active, Ordering::SeqCst);
                barrier.wait();
                Ok(LiveReader {
                    reader,
                    live: live.clone(),
                })
            },
            |reader, c, r, w, h| {
                largest.fetch_max(w * h, Ordering::SeqCst);
                reader.reader.read_window_raw(c, r, w, h)
            },
        )
        .unwrap();
        bits(&expected.data, &actual.data);
        assert_eq!(
            live.load(Ordering::SeqCst),
            0,
            "readers must close after the warp"
        );
    }
    assert_eq!(
        peak.load(Ordering::SeqCst),
        4,
        "48 requested workers must cap at four private readers"
    );
    assert!(largest.load(Ordering::SeqCst) <= 4 * 1024 * 1024);
    assert_eq!(warp::gpuwm_static_continuous_warp_worker_cap(), 4);
    assert_eq!(
        warp::gpuwm_static_continuous_warp_reader_cache_bytes(),
        8 * 1024 * 1024
    );
    assert_eq!(
        warp::gpuwm_static_continuous_warp_source_window_bytes(),
        32 * 1024 * 1024
    );
}

#[test]
fn default_bound_continuous_parallel_path_matches_dense_masked_scaled_source() {
    use static_fields::highres::{self, BoundRasterSpec};
    use static_fields::projection::{GridSpec, ProjectionKind};
    let tmp = Scratch::new("default-parallel");
    let path = tmp.file("source.tif");
    let raster = Raster {
        ny: 513,
        nx: 521,
        values: (0..513 * 521)
            .map(|i| {
                if i % 101 == 0 {
                    -9999.0
                } else {
                    (i % 4001) as f64 * 0.25
                }
            })
            .collect(),
        transform: [0.002, 0.0, -0.521, 0.0, -0.002, 0.513],
        crs: Crs::Geographic,
    };
    geotiff::write_band1(&path, &raster, geotiff::SampleType::F32, Some(-9999.0)).unwrap();
    let spec = GridSpec {
        kind: ProjectionKind::Mercator,
        ref_lat: 0.0,
        ref_lon: 0.0,
        truelat1: 0.0,
        truelat2: 0.0,
        stand_lon: 0.0,
        dx: 1000.0,
        dy: 1000.0,
        e_we: 114,
        e_sn: 98,
        known_x: 57.0,
        known_y: 49.0,
        moad_cen_lat: 0.0,
        moad_cen_lon: 0.0,
        lat_deg: Vec::new(),
        lon0_deg: 0.0,
        dlon_deg: 0.0,
    };
    let bound = BoundRasterSpec {
        sha256: highres::sha256_file(&path).unwrap(),
        expected_bytes: Some(std::fs::metadata(&path).unwrap().len()),
        path,
        crs_override: None,
        nodata_override: Some(-9999.0),
        scale_factor: 0.1,
    };
    let dense = bound.open().unwrap();
    for method in [
        warp::Resampling::Average,
        warp::Resampling::Bilinear,
        warp::Resampling::Nearest,
    ] {
        let expected = highres::resample_continuous(&dense, &spec, method).unwrap();
        let actual = highres::resample_continuous_bound(&bound, &spec, method).unwrap();
        bits(&expected.data, &actual.data);
    }
}
