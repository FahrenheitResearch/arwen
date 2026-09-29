//! The seam's shared plumbing is real today even though the lanes are
//! stubs: version probe, error copy discipline, handle registries
//! refusing unknown handles by name.  Everything else is red by design
//! until the lanes land (tests/test_static_rust_parity.py on the
//! Python side is the parity harness).

use static_fields::capi::STATIC_ABI_VERSION;

#[test]
fn abi_version_is_one() {
    assert_eq!(static_fields::capi::gpuwm_static_abi_version(), 1);
    assert_eq!(STATIC_ABI_VERSION, 1);
}

#[test]
fn unknown_fieldset_handle_is_refused_by_name() {
    assert_eq!(
        static_fields::capi::build::gpuwm_static_fieldset_len(9999),
        -1
    );
    let mut buf = vec![0u8; 256];
    let len = unsafe {
        static_fields::capi::gpuwm_static_last_error(buf.as_mut_ptr(), buf.len())
    };
    let message = std::str::from_utf8(&buf[..len.min(buf.len())]).unwrap();
    assert!(message.contains("9999"), "error names the handle: {message}");
}

#[test]
fn grid_new_is_real() {
    // Lane 1 landed: the same call the skeleton refused now yields a
    // live grid handle.
    let spec = br#"{
        "kind": "lambert",
        "ref_lat": 40.0, "ref_lon": -100.0,
        "truelat1": 38.0, "truelat2": 42.0, "stand_lon": -100.0,
        "dx": 3000.0, "dy": 3000.0, "e_we": 100, "e_sn": 100,
        "known_x": 50.0, "known_y": 50.0,
        "moad_cen_lat": 40.0, "moad_cen_lon": -100.0
    }"#;
    let mut handle = 0u64;
    let rc = unsafe {
        static_fields::capi::grid::gpuwm_static_grid_new(
            spec.as_ptr(),
            spec.len(),
            &mut handle,
        )
    };
    assert_eq!(rc, 0, "lane 1's grid_new must accept a valid spec");
    assert_ne!(handle, 0);
    static_fields::capi::grid::gpuwm_static_grid_free(handle);
}

#[test]
fn grid_new_accepts_a_rows_spec() {
    // The global model's Gaussian grid crosses the seam as kind "rows":
    // explicit latitude rows on a uniform longitude ring.  Without that
    // kind every global statics build stopped here with "unknown
    // variant `rows`".  The mass latitudes that come back are the rows
    // that went in, and the columns step by dlon from lon0.
    let spec = br#"{
        "kind": "rows",
        "ref_lat": 0.0, "ref_lon": 0.0,
        "truelat1": 0.0, "truelat2": 0.0, "stand_lon": 0.0,
        "dx": 111194.92664455873, "dy": 111194.92664455873,
        "e_we": 5, "e_sn": 4,
        "known_x": 1.0, "known_y": 1.0,
        "moad_cen_lat": 0.0, "moad_cen_lon": 0.0,
        "lat_deg": [35.5, 36.25, 37.5], "lon0_deg": -110.0, "dlon_deg": 1.0
    }"#;
    let mut handle = 0u64;
    let rc = unsafe {
        static_fields::capi::grid::gpuwm_static_grid_new(
            spec.as_ptr(),
            spec.len(),
            &mut handle,
        )
    };
    let mut buf = vec![0u8; 512];
    let len = unsafe {
        static_fields::capi::gpuwm_static_last_error(buf.as_mut_ptr(), buf.len())
    };
    let message = String::from_utf8_lossy(&buf[..len.min(buf.len())]).into_owned();
    assert_eq!(rc, 0, "grid_new must accept a rows spec: {message}");
    assert_ne!(handle, 0);
    let mut lat = vec![0.0f64; 3 * 4];
    let mut lon = vec![0.0f64; 3 * 4];
    let rc_lat = unsafe {
        static_fields::capi::grid::gpuwm_static_grid_array(
            handle, 0, 0, lat.as_mut_ptr(), lat.len(),
        )
    };
    let rc_lon = unsafe {
        static_fields::capi::grid::gpuwm_static_grid_array(
            handle, 0, 1, lon.as_mut_ptr(), lon.len(),
        )
    };
    static_fields::capi::grid::gpuwm_static_grid_free(handle);
    assert_eq!((rc_lat, rc_lon), (0, 0));
    for (row, want) in [35.5, 36.25, 37.5].iter().enumerate() {
        for col in 0..4 {
            assert_eq!(lat[row * 4 + col], *want, "row {row} col {col}");
            assert_eq!(lon[row * 4 + col], -110.0 + col as f64, "row {row} col {col}");
        }
    }
}
