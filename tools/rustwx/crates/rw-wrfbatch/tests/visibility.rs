//! The native import must materialize visibility and its low-visibility flag.
mod stored_plane_fixture;
use rw_store::reader::HourReader;
use rw_wrfbatch::wrf_process::{WrfProcessMessage, WrfProcessOptions, spawn_process_paths};

#[test]
fn selected_native_visibility_is_in_metres_and_has_a_strict_flag() {
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let root =
        std::env::temp_dir().join(format!("visibility-import-{}-{nonce}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    let file = stored_plane_fixture::write(&root);
    let store = root.join("store");
    let options = WrfProcessOptions {
        only: vec!["visibility_surface".into(), "fog_flag_surface".into()],
        ..WrfProcessOptions::default()
    };
    assert!(
        options
            .planned_store_fields()
            .contains(&"visibility_surface".to_string())
    );
    assert!(
        options
            .planned_store_fields()
            .contains(&"fog_flag_surface".to_string())
    );
    let task = spawn_process_paths(vec![file], store.clone(), options);
    let summary = loop {
        match task.rx.recv().unwrap() {
            WrfProcessMessage::Progress(_) => (),
            WrfProcessMessage::Done(result) => break result.unwrap(),
        }
    };
    let reader =
        HourReader::open(&store.join(summary.model).join(summary.run).join("f000.rws")).unwrap();
    let visibility = reader.read_full_2d("visibility_surface").unwrap();
    let flags = reader.read_full_2d("fog_flag_surface").unwrap();
    assert_eq!(
        visibility.len(),
        stored_plane_fixture::NX * stored_plane_fixture::NY
    );
    assert!(
        visibility
            .iter()
            .all(|v| v.is_finite() && *v > 0.0 && *v < 400.0)
    );
    assert!(flags.iter().all(|v| *v == 1.0));
    assert_eq!(
        reader
            .meta()
            .variables
            .iter()
            .find(|v| v.name == "visibility_surface")
            .unwrap()
            .units,
        "m"
    );
    assert_eq!(
        reader
            .meta()
            .variables
            .iter()
            .find(|v| v.name == "fog_flag_surface")
            .unwrap()
            .units,
        "1"
    );
    drop(reader);
    std::fs::remove_dir_all(&root).unwrap();
}
