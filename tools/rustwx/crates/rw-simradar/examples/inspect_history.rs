//! Metadata and field support probe for a saved history, without a GPU.
use bowecho_simradar::WrfFile;
fn main() -> Result<(), String> {
    for path in std::env::args().skip(1) {
        let f = WrfFile::open(&path).map_err(|e| e.to_string())?;
        let lats = f.xlat(0).map_err(|e| e.to_string())?;
        let lons = f.xlong(0).map_err(|e| e.to_string())?;
        let span = |x: &[f64]| {
            (
                x.iter().copied().fold(f64::INFINITY, f64::min),
                x.iter().copied().fold(f64::NEG_INFINITY, f64::max),
            )
        };
        println!(
            "{}",
            serde_json::json!({"path":path,"shape":[f.nz,f.ny,f.nx],"times":f.times().map_err(|e|e.to_string())?,
            "lat":span(&lats),"lon":span(&lons),"dx_m":f.dx,"mp_physics":f.global_attr_i32("MP_PHYSICS").ok()})
        );
    }
    Ok(())
}
