//! The named import must publish actual finite diagnostic planes, and must
//! refuse missing microphysics rather than turn it into zero lightning.
mod stored_plane_fixture;
use rw_wrfbatch::wrf_process::{
    WrfProcessMessage, WrfProcessOptions, required_fields_for_products, spawn_process_paths,
};
use std::path::PathBuf;
struct Scratch(PathBuf);
impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}
#[test]
fn named_heat_fire_import_writes_requested_planes_and_names_missing_ice() {
    let root = Scratch(
        std::env::temp_dir().join(format!(
            "derived-diagnostics-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        )),
    );
    std::fs::create_dir_all(&root.0).unwrap();
    let file =
        stored_plane_fixture::write_with_surface_planes(&root.0, &[("SWDOWN", "W m-2", 500.)]);
    let products = "heat_index_2m,apparent_temperature_2m,wetbulb_2m,fire_weather_composite,wbgt_2m,hot_dry_windy,haines_mid,lightning_threat_3";
    let store = root.0.join("store");
    let task = spawn_process_paths(
        vec![file],
        store.clone(),
        WrfProcessOptions {
            required_fields: required_fields_for_products(products),
            ..Default::default()
        },
    );
    let summary = loop {
        match task.rx.recv().unwrap() {
            WrfProcessMessage::Progress(_) => {}
            WrfProcessMessage::Done(result) => break result.unwrap(),
        }
    };
    let hour = store.join(summary.model).join(summary.run).join("f000.rws");
    let reader = rw_store::reader::HourReader::open(&hour).unwrap();
    for name in products
        .split(',')
        .filter(|name| *name != "lightning_threat_3")
    {
        assert!(
            summary.variables.iter().any(|n| n == name),
            "missing {name}; notes {:?}",
            summary.notes
        );
        let data = reader.read_full_2d(name).unwrap();
        assert_eq!(
            data.len(),
            stored_plane_fixture::NX * stored_plane_fixture::NY
        );
        assert!(data.iter().all(|v| v.is_finite()), "nonfinite {name}");
        if name == "haines_mid" {
            assert!(data.iter().all(|v| (2.0..=6.0).contains(v)));
        }
    }
    assert!(!summary.variables.iter().any(|n| n == "lightning_threat_3"));
    assert!(
        summary
            .notes
            .iter()
            .any(|n| n.contains("LTG1/2/3 unavailable")),
        "{:?}",
        summary.notes
    );
    assert!(
        !summary.variables.iter().any(|n| n == "sbcape"),
        "scoped import computed unused parcels"
    );
}
