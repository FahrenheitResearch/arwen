//! A SoilGrids tile whose GeoKeys name a user-defined model type
//! (GTModelType 32767, as ISRIC's WCS has written since 2026-10-15)
//! opens, and reads under the IGH `crs_override`; without an override
//! the read is refused by name. The tile is real, fetched from the
//! WCS: set SOILFIX_TILE to its path (the test passes vacuously
//! without it, so a checkout without the fixture is not red).
use static_fields::raster::geotiff::read_band1_raw;
use static_fields::raster::Crs;

#[test]
fn user_defined_model_type_reads_under_override() {
    let Ok(path) = std::env::var("SOILFIX_TILE") else { return };
    let path = std::path::Path::new(&path);
    let (raster, nodata) = read_band1_raw(path, Some(Crs::InterruptedGoodeHomolosine), None)
        .expect("opens and reads under the IGH override");
    assert!(raster.nx > 0 && raster.ny > 0);
    assert!(matches!(raster.crs, Crs::InterruptedGoodeHomolosine));
    let _ = nodata;
    let err = read_band1_raw(path, None, None).err().expect("refused without an override");
    let text = format!("{err}");
    assert!(text.contains("GTModelType 32767"), "{text}");
    assert!(text.contains("no crs_override"), "{text}");
}
