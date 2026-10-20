//! Exact executed-handle export, checked against existing captured grid corners
//! through independent system PROJ. These are static-field fixtures, not an
//! invented parent forecast or a hydrological event acceptance run.
use serde_json::Value;
use static_fields::capi::{grid::*, hydrofabric::gpuwm_static_grid_hydrofabric_json};
use static_fields::projection::GridSpec;
use std::ffi::{CString, c_char, c_int, c_void};
use std::path::PathBuf;

#[repr(C)]
struct Coord {
    v: [f64; 4],
}
#[cfg_attr(target_os = "linux", link(name = ":libproj.so.25"))]
#[cfg_attr(not(target_os = "linux"), link(name = "proj"))]
unsafe extern "C" {
    fn proj_context_create() -> *mut c_void;
    fn proj_context_destroy(ctx: *mut c_void);
    fn proj_context_set_enable_network(ctx: *mut c_void, enabled: c_int) -> c_int;
    fn proj_create_crs_to_crs(
        ctx: *mut c_void,
        from: *const c_char,
        to: *const c_char,
        area: *mut c_void,
    ) -> *mut c_void;
    fn proj_normalize_for_visualization(ctx: *mut c_void, p: *const c_void) -> *mut c_void;
    fn proj_trans(p: *mut c_void, direction: c_int, coord: Coord) -> Coord;
    fn proj_destroy(p: *mut c_void) -> *mut c_void;
}
struct Projection {
    ctx: *mut c_void,
    p: *mut c_void,
}
impl Projection {
    fn new(crs: &str) -> Self {
        unsafe {
            let ctx = proj_context_create();
            assert!(!ctx.is_null());
            proj_context_set_enable_network(ctx, 0);
            let from = CString::new("+proj=longlat +a=6370000 +b=6370000 +type=crs").unwrap();
            let to = CString::new(crs).unwrap();
            let raw = proj_create_crs_to_crs(ctx, from.as_ptr(), to.as_ptr(), std::ptr::null_mut());
            assert!(
                !raw.is_null(),
                "exported CRS must parse as a projected metre CRS"
            );
            let p = proj_normalize_for_visualization(ctx, raw);
            proj_destroy(raw);
            assert!(!p.is_null());
            Self { ctx, p }
        }
    }
    fn forward(&self, lon: f64, lat: f64) -> [f64; 2] {
        let p = unsafe {
            proj_trans(
                self.p,
                1,
                Coord {
                    v: [lon, lat, 0., 0.],
                },
            )
        };
        [p.v[0], p.v[1]]
    }
}
impl Drop for Projection {
    fn drop(&mut self) {
        unsafe {
            proj_destroy(self.p);
            proj_context_destroy(self.ctx);
        }
    }
}
fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/goldens/lane1")
}
fn fixture(name: &str) -> Value {
    let manifest: Value =
        serde_json::from_slice(&std::fs::read(root().join("manifest.json")).unwrap()).unwrap();
    manifest["cases"][name].clone()
}
fn handle(spec: &Value) -> u64 {
    let spec = serde_json::to_vec(spec).unwrap();
    let mut handle = 0;
    assert_eq!(
        unsafe { gpuwm_static_grid_new(spec.as_ptr(), spec.len(), &mut handle) },
        0
    );
    handle
}
fn export(handle: u64) -> Value {
    let n = unsafe { gpuwm_static_grid_hydrofabric_json(handle, std::ptr::null_mut(), 0) };
    assert!(n > 0);
    let mut short = [0xabu8; 4];
    assert_eq!(
        unsafe { gpuwm_static_grid_hydrofabric_json(handle, short.as_mut_ptr(), 3) },
        n
    );
    assert_eq!(short[3], 0xab, "truncated output must not overrun capacity");
    let mut bytes = vec![0; n as usize];
    assert_eq!(
        unsafe { gpuwm_static_grid_hydrofabric_json(handle, bytes.as_mut_ptr(), bytes.len()) },
        n
    );
    serde_json::from_slice(&bytes).unwrap()
}
fn check_corners(label: &str, contract: &Value, lat: &[f64], lon: &[f64]) {
    assert_eq!(contract["schema"], "arwen.nextgen.parent-grid.v1");
    let g = &contract["grid"];
    assert_eq!(g["row_order"], "south_to_north");
    let nx = g["nx"].as_u64().unwrap() as usize;
    let ny = g["ny"].as_u64().unwrap() as usize;
    let project = Projection::new(g["crs"].as_str().unwrap());
    let mut maximum = 0f64;
    for (x, y) in [(0, 0), (nx, 0), (0, ny), (nx, ny)] {
        let k = y * (nx + 1) + x;
        let p = project.forward(lon[k], lat[k]);
        let expected = [
            g["x0"].as_f64().unwrap() + x as f64 * g["dx"].as_f64().unwrap(),
            g["y0"].as_f64().unwrap() + y as f64 * g["dy"].as_f64().unwrap(),
        ];
        for axis in 0..2 {
            assert!(p[axis].is_finite());
            maximum = maximum.max((p[axis] - expected[axis]).abs());
        }
    }
    println!("{label}: 4 captured/native corners; maximum PROJ coordinate error {maximum:.12e} m");
    // Existing static-fields projection qualification is <=1e-6 metre vs PROJ.
    // This is an instrument check, not a tolerance for routing replay.
    assert!(
        maximum <= 1e-6,
        "{label}: projected footprint/row order mismatch {maximum} m"
    );
}
fn captured(case: &Value, name: &str) -> Vec<f64> {
    let path = root().join(case["arrays"][name]["file"].as_str().unwrap());
    std::fs::read(path)
        .unwrap()
        .chunks_exact(8)
        .map(|b| f64::from_le_bytes(b.try_into().unwrap()))
        .collect()
}

#[test]
fn exported_crs_and_south_first_edges_match_captured_parent_corners() {
    // One existing fixture per actual projected grid family. South/north and
    // sphere/ellipsoid confusion shift these corners by whole grid cells/km.
    for name in ["lam_parent", "merc", "polar"] {
        let case = fixture(name);
        let h = handle(&case["spec"]);
        let contract = export(h);
        check_corners(
            name,
            &contract,
            &captured(&case, "lat_c"),
            &captured(&case, "lon_c"),
        );
        gpuwm_static_grid_free(h);
    }
}

#[test]
fn translated_handle_preserves_executed_extent_and_offsets() {
    let case = fixture("lam_parent");
    let root = handle(&case["spec"]);
    let mut moved = 0;
    assert_eq!(
        unsafe { gpuwm_static_grid_translated(root, 7, -3, 12, 9, &mut moved) },
        0
    );
    let contract = export(moved);
    assert_eq!(contract["grid"]["nx"], 11);
    assert_eq!(contract["grid"]["ny"], 8);
    let mut lat = vec![0.; 12 * 9];
    let mut lon = vec![0.; 12 * 9];
    assert_eq!(
        unsafe { gpuwm_static_grid_array(moved, 3, 0, lat.as_mut_ptr(), lat.len()) },
        0
    );
    assert_eq!(
        unsafe { gpuwm_static_grid_array(moved, 3, 1, lon.as_mut_ptr(), lon.len()) },
        0
    );
    check_corners("translated", &contract, &lat, &lon);
    gpuwm_static_grid_free(moved);
    gpuwm_static_grid_free(root);
}

#[test]
fn mercator_longitude_anchor_is_reference_and_rows_are_not_metres() {
    let mut spec = fixture("merc")["spec"].clone();
    spec["stand_lon"] = serde_json::json!(spec["ref_lon"].as_f64().unwrap() + 17.);
    let h = handle(&spec);
    let contract = export(h);
    let nx = contract["grid"]["nx"].as_u64().unwrap() as usize + 1;
    let ny = contract["grid"]["ny"].as_u64().unwrap() as usize + 1;
    let mut lat = vec![0.; nx * ny];
    let mut lon = vec![0.; nx * ny];
    assert_eq!(
        unsafe { gpuwm_static_grid_array(h, 3, 0, lat.as_mut_ptr(), lat.len()) },
        0
    );
    assert_eq!(
        unsafe { gpuwm_static_grid_array(h, 3, 1, lon.as_mut_ptr(), lon.len()) },
        0
    );
    check_corners("mercator distinct stand_lon", &contract, &lat, &lon);
    gpuwm_static_grid_free(h);
    let mut rows: GridSpec = serde_json::from_value(spec).unwrap();
    rows.kind = static_fields::projection::ProjectionKind::Rows;
    rows.e_we = 5;
    rows.e_sn = 4;
    rows.lat_deg = vec![35.5, 36.25, 37.5];
    rows.lon0_deg = -110.;
    rows.dlon_deg = 1.;
    let h = handle(&serde_json::to_value(rows).unwrap());
    assert_eq!(
        unsafe { gpuwm_static_grid_hydrofabric_json(h, std::ptr::null_mut(), 0) },
        -1
    );
    gpuwm_static_grid_free(h);
    assert_eq!(
        unsafe { gpuwm_static_grid_hydrofabric_json(u64::MAX, std::ptr::null_mut(), 0) },
        -1
    );
}
