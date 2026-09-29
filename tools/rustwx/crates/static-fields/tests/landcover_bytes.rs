//! The 8-bit land-cover path.
//!
//! A global 100 m land-cover raster is read as bytes: the window step
//! and the mapped-category step must give exactly what the f64 path
//! gives (the same derived file, the same fractions, the same refusal),
//! and the window margin on a geographic raster must be degrees, not the
//! metre figure read as degrees.

use std::collections::BTreeMap;
use std::path::PathBuf;

use static_fields::highres;
use static_fields::projection::{GridSpec, ProjectionKind};
use static_fields::raster::{geotiff, Crs, Raster};

/// A small geographic byte raster in a MODIS-with-LCZ legend, 0 being
/// the unclassified value.
fn categories(nx: usize, ny: usize, extra: Option<f64>) -> Raster {
    let classes = [0.0, 1.0, 5.0, 12.0, 17.0, 21.0, 51.0, 55.0, 61.0];
    let mut values: Vec<f64> = (0..nx * ny)
        .map(|k| classes[(k / 7 + k % 5) % classes.len()])
        .collect();
    if let Some(value) = extra {
        values[nx * ny / 2] = value;
    }
    Raster {
        ny,
        nx,
        values,
        transform: [0.001, 0.0, 10.0, 0.0, -0.001, 45.3],
        crs: Crs::Geographic,
    }
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir()
        .join(format!("static-fields-landcover-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("scratch directory");
    dir.join(name)
}

fn lcz_mapping() -> BTreeMap<i64, i64> {
    let mut mapping: BTreeMap<i64, i64> = (1..=21).map(|c| (c, c)).collect();
    for lcz in 51..=61 {
        mapping.insert(lcz, 13);
    }
    mapping
}

fn grid() -> GridSpec {
    GridSpec {
        kind: ProjectionKind::Lambert,
        ref_lat: 45.2,
        ref_lon: 10.15,
        truelat1: 30.0,
        truelat2: 60.0,
        stand_lon: 10.15,
        dx: 1000.0,
        dy: 1000.0,
        e_we: 11,
        e_sn: 11,
        known_x: 5.5,
        known_y: 5.5,
        moad_cen_lat: 45.2,
        moad_cen_lon: 10.15,
        lat_deg: Vec::new(),
        lon0_deg: 0.0,
        dlon_deg: 0.0,
    }
}

#[test]
fn a_byte_band_writes_the_same_file_from_bytes_as_from_f64() {
    let raster = categories(300, 200, None);
    let from_f64 = scratch("from_f64.tif");
    let from_bytes = scratch("from_bytes.tif");
    geotiff::write_band1(&from_f64, &raster, geotiff::SampleType::U8, None)
        .expect("f64 write");
    let bytes: Vec<u8> = raster.values.iter().map(|v| *v as u8).collect();
    geotiff::write_band1_u8(
        &from_bytes,
        &bytes,
        raster.ny,
        raster.nx,
        &raster.transform,
        &raster.crs,
        None,
    )
    .expect("byte write");
    assert_eq!(
        std::fs::read(&from_f64).unwrap(),
        std::fs::read(&from_bytes).unwrap()
    );
}

#[test]
fn a_byte_window_reads_what_the_f64_window_reads() {
    let raster = categories(300, 200, None);
    let path = scratch("window.tif");
    geotiff::write_band1(&path, &raster, geotiff::SampleType::U8, None)
        .unwrap();
    let mut reader = geotiff::TiffReader::open(&path).unwrap();
    assert_eq!(reader.sample_type(), geotiff::SampleType::U8);
    assert_eq!(reader.crs, Some(Crs::Geographic));
    // A window crossing tile boundaries in both directions.
    let bytes = reader.read_window_u8(250, 17, 50, 180).unwrap();
    let floats = reader.read_window_raw(250, 17, 50, 180).unwrap();
    assert_eq!(bytes.len(), floats.len());
    for (byte, float) in bytes.iter().zip(&floats) {
        assert_eq!(*byte as f64, *float);
    }
    // Outside the image is refused, not clamped.
    assert!(reader.read_window_u8(260, 0, 50, 10).is_err());

    let wide = Raster {
        values: raster.values.clone(),
        ..raster.clone()
    };
    let float_path = scratch("float.tif");
    geotiff::write_band1(&float_path, &wide, geotiff::SampleType::F32, None)
        .unwrap();
    let mut reader = geotiff::TiffReader::open(&float_path).unwrap();
    let refused = reader.read_window_u8(0, 0, 4, 4).unwrap_err();
    assert!(refused.to_string().contains("8-bit read"), "{refused}");
}

#[test]
fn byte_categories_give_the_f64_fractions_bit_for_bit() {
    let raster = categories(300, 200, None);
    let bytes: Vec<u8> = raster.values.iter().map(|v| *v as u8).collect();
    let carrier = Raster { values: Vec::new(), ..raster.clone() };
    let mapping = lcz_mapping();
    for nodata in [Some(0.0), None, Some(0.5), Some(f64::NAN)] {
        let reference = highres::resample_mapped_categories(
            &raster, "fixture", &grid(), &mapping, 21, nodata,
        );
        let candidate = highres::resample_mapped_categories_u8(
            &bytes, &carrier, "fixture", &grid(), &mapping, 21, nodata,
        );
        match (reference, candidate) {
            (Ok(reference), Ok(candidate)) => {
                assert_eq!(reference.planes, candidate.planes);
                for (a, b) in reference.data.iter().zip(&candidate.data) {
                    assert_eq!(a.to_bits(), b.to_bits(), "nodata {nodata:?}");
                }
                // The LCZ classes arrive as the urban category, inside
                // the 21-category inventory.
                assert_eq!(candidate.planes, 21);
                let cells = reference.ny * reference.nx;
                let urban: f64 = candidate.data[12 * cells..13 * cells]
                    .iter()
                    .filter(|v| v.is_finite())
                    .sum();
                assert!(urban > 0.0);
            }
            (Err(a), Err(b)) => assert_eq!(a.to_string(), b.to_string()),
            (a, b) => panic!("engines disagree: {:?} vs {:?}",
                             a.map(|_| ()), b.map(|_| ())),
        }
    }
}

#[test]
fn an_unmapped_byte_category_is_refused_in_the_f64_words() {
    let raster = categories(300, 200, Some(99.0));
    let bytes: Vec<u8> = raster.values.iter().map(|v| *v as u8).collect();
    let carrier = Raster { values: Vec::new(), ..raster.clone() };
    let reference = highres::resample_mapped_categories(
        &raster, "fixture", &grid(), &lcz_mapping(), 21, Some(0.0),
    )
    .unwrap_err();
    let candidate = highres::resample_mapped_categories_u8(
        &bytes, &carrier, "fixture", &grid(), &lcz_mapping(), 21, Some(0.0),
    )
    .unwrap_err();
    assert_eq!(reference.to_string(), candidate.to_string());
    assert!(candidate.to_string().contains("unmapped categories [99]"));
}

#[test]
fn the_window_margin_on_a_geographic_raster_is_degrees() {
    let (dlon, dlat) = highres::margin_degrees(-1.0, 1.0, 2000.0);
    assert!((dlat - 2000.0 / 111_320.0).abs() < 1e-15);
    assert!(dlon >= dlat && dlon < dlat * 1.001);
    let (dlon, dlat) = highres::margin_degrees(58.0, 60.0, 2000.0);
    assert!((dlon / dlat - 2.0).abs() < 1e-9);
    // Toward a pole the longitude margin stops growing at cos 87.
    let (dlon, dlat) = highres::margin_degrees(80.0, 89.99, 2000.0);
    assert!((dlon - dlat / 87.0f64.to_radians().cos()).abs() < 1e-12);
    // Never the metre figure itself.
    assert!(dlon < 1.0 && dlat < 1.0);
}
