pub mod bundle;
pub mod cell_overlap;
pub mod collapse;
pub mod dbf;
pub mod manifest;
pub mod network;
pub mod replay;
pub mod render_map;
pub mod routing_capture;
pub mod source_network;
pub mod source_waterbodies;
pub mod source_geometry;
pub mod source_projection;
pub mod source_divides;
pub mod source_polygons;
pub mod source_lengths;
pub mod sqlite;
pub mod weights;
pub mod world_routing;
pub type Error = Box<dyn std::error::Error + Send + Sync>;
pub type Result<T> = std::result::Result<T, Error>;
pub fn inspect(path: &str) -> Result<serde_json::Value> {
    let db = sqlite::Db::open(path)?;
    let mut out = Vec::new();
    for r in db.query("SELECT table_name,data_type,srs_id,min_x,min_y,max_x,max_y FROM gpkg_contents ORDER BY table_name",None)?{
  let name=r.properties["table_name"].as_str().ok_or("missing table name")?;
  let cols=db.query(&format!("PRAGMA table_info({})",sqlite::ident(name)),None)?;
  let sample=db.query(&format!("SELECT * FROM {} LIMIT 1",sqlite::ident(name)),Some("geom"))?;
  let count=db.query(&format!("SELECT count(*) AS n FROM {}",sqlite::ident(name)),None)?;
  out.push(serde_json::json!({"table":name,"count":count[0].properties["n"],"columns":cols.into_iter().map(|x|x.properties).collect::<Vec<_>>(),"sample":sample.into_iter().map(|x|x.properties).collect::<Vec<_>>(),"metadata":r.properties}));
 }
    Ok(serde_json::json!({"tables":out}))
}
#[no_mangle]
pub extern "C" fn arwen_hydrofabric_abi_version() -> u32 {
    1
}
/// Validate a transported buffer without interpreting it in Python.
/// # Safety
/// expected_sha256 is NUL-terminated; data is readable for length bytes.
#[no_mangle]
pub unsafe extern "C" fn arwen_hydrofabric_verify_memory(
    expected_sha256: *const std::ffi::c_char,
    data: *const u8,
    length: usize,
) -> i32 {
    use sha2::{Digest, Sha256};
    if expected_sha256.is_null() || (data.is_null() && length != 0) {
        return 1;
    }
    let bytes = if length == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(data, length)
    };
    let Ok(expected) = std::ffi::CStr::from_ptr(expected_sha256).to_str() else {
        return 1;
    };
    if format!("{:x}", Sha256::digest(bytes)) == expected {
        0
    } else {
        2
    }
}
pub const SOURCE_REVISION: &str =
    concat!("GPUWM_BRIDGE_SOURCE_REV=", env!("GPUWM_HYDROFABRIC_REV"));
#[no_mangle]
pub extern "C" fn arwen_hydrofabric_source_revision() -> *const std::ffi::c_char {
    concat!(
        "GPUWM_BRIDGE_SOURCE_REV=",
        env!("GPUWM_HYDROFABRIC_REV"),
        "\0"
    )
    .as_ptr()
    .cast()
}
