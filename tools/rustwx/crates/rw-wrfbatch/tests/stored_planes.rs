//! The generic `var:<name>` product has to be able to name a 2-D plane a
//! USER added to their wrfout.
//!
//! The concrete breakage these tests prevent: `rw_wrfbatch` imports every
//! wrfout through the full science route (`wrf_process`), and that route
//! only ever wrote the fields its two fixed catalogs name — the wrf-core
//! diagnostic registry and `RAW_EXTRA_CATALOG`. A plane the file carries
//! but no catalog knows about was silently dropped at import, so
//! `--products var:wrf_<name>` refused with
//! `stored 2-D variable "wrf_<name>" does not exist` and the user's own
//! Registry output was unreachable through the renderer. Adding a product
//! per user variable is exactly the per-model-adapter shape the arbitrary
//! acceptance test forbids; ingesting every stored `(Time, south_north,
//! west_east)` plane is metadata-level and costs no new product code.

mod stored_plane_fixture;

use std::path::{Path, PathBuf};

use rw_wrfbatch::wrf_process::{WrfProcessMessage, WrfProcessOptions, spawn_process_paths};

struct Scratch(PathBuf);

impl Scratch {
    fn new(tag: &str) -> Self {
        let dir = std::env::temp_dir().join(format!(
            "rw-wrfbatch-stored-planes-{tag}-{}-{:?}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|value| value.as_nanos())
                .unwrap_or_default()
        ));
        std::fs::create_dir_all(&dir).expect("create scratch dir");
        Self(dir)
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

struct Imported {
    variables: Vec<String>,
    hour: PathBuf,
}

fn import(wrfout: &Path, store_root: &Path, options: WrfProcessOptions) -> Imported {
    let task = spawn_process_paths(
        vec![wrfout.to_path_buf()],
        store_root.to_path_buf(),
        options,
    );
    let summary = loop {
        match task.rx.recv().expect("the WRF processor answers") {
            WrfProcessMessage::Progress(_) => {}
            WrfProcessMessage::Done(result) => {
                break result.expect("the synthetic wrfout imports");
            }
        }
    };
    Imported {
        hour: store_root
            .join(&summary.model)
            .join(&summary.run)
            .join("f000.rws"),
        variables: summary.variables,
    }
}

/// The defect itself: a stored 2-D plane no catalog names must reach the
/// store under `wrf_<sanitized name>`, which is the exact spelling
/// `--products var:...` resolves.
#[test]
fn user_added_two_d_plane_reaches_the_store_by_default() {
    let scratch = Scratch::new("user-plane");
    let wrfout = stored_plane_fixture::write(scratch.path());
    let store_root = scratch.path().join("store");

    let imported = import(&wrfout, &store_root, WrfProcessOptions::default());

    assert!(
        imported
            .variables
            .iter()
            .any(|name| name == stored_plane_fixture::USER_PLANE_STORE_NAME),
        "a bare default import dropped the user's own 2-D plane {:?}; stored: {:?}",
        stored_plane_fixture::USER_PLANE_STORE_NAME,
        imported.variables
    );

    // The VALUES, not just the name: a pass that carried the wrong plane
    // under the right name would render a confidently wrong panel.
    let reader = rw_store::reader::HourReader::open(&imported.hour).expect("open the stored hour");
    let values = reader
        .read_full_2d(stored_plane_fixture::USER_PLANE_STORE_NAME)
        .expect("read the user plane back");
    assert_eq!(
        values.len(),
        stored_plane_fixture::NX * stored_plane_fixture::NY
    );
    for y in 0..stored_plane_fixture::NY {
        for x in 0..stored_plane_fixture::NX {
            assert_eq!(
                values[y * stored_plane_fixture::NX + x],
                stored_plane_fixture::user_plane_value(y, x),
                "user plane value differs at ({y}, {x})"
            );
        }
    }
}

/// The ordinary science suite keeps its own names: the stored-plane pass
/// adds, it never renames or displaces a catalog field. `T2` is the case
/// that matters — it is a canonical `temperature_2m` AND a raw plane, and
/// a stored-plane pass that shadowed it would replace an earth-relative,
/// unit-checked field with a raw one.
#[test]
fn stored_plane_ingest_does_not_displace_the_science_suite() {
    let scratch = Scratch::new("no-shadow");
    let wrfout = stored_plane_fixture::write(scratch.path());
    let store_root = scratch.path().join("store");

    let variables = import(&wrfout, &store_root, WrfProcessOptions::default()).variables;

    assert!(
        variables.iter().any(|name| name == "temperature_2m"),
        "the canonical 2 m temperature vanished; stored: {variables:?}"
    );
    assert!(
        variables.iter().any(|name| name == "wrf_t2"),
        "the diagnostic-registry browse plane vanished; stored: {variables:?}"
    );
    let mut sorted = variables.clone();
    sorted.sort();
    sorted.dedup();
    assert_eq!(
        sorted.len(),
        variables.len(),
        "the stored-plane pass duplicated a store name; stored: {variables:?}"
    );
}

/// `--skip` still reaches the stored planes: they are filterable under the
/// same Raw group and the same token grammar as the rest of the import, so
/// a user who does not want them has a documented way to say so.
#[test]
fn stored_planes_answer_the_skip_filter() {
    let scratch = Scratch::new("skip");
    let wrfout = stored_plane_fixture::write(scratch.path());
    let store_root = scratch.path().join("store");

    let options = WrfProcessOptions {
        skip: vec![stored_plane_fixture::USER_PLANE.to_string()],
        ..WrfProcessOptions::default()
    };
    let variables = import(&wrfout, &store_root, options).variables;

    assert!(
        !variables
            .iter()
            .any(|name| name == stored_plane_fixture::USER_PLANE_STORE_NAME),
        "--skip {} did not reach the stored-plane pass; stored: {variables:?}",
        stored_plane_fixture::USER_PLANE
    );
}

/// Writes the fixture wrfout to the path in `RW_STORED_PLANE_FIXTURE` and
/// leaves it there, so the real `rw_wrfbatch` binary can be pointed at a
/// wrfout carrying a user-added plane. Ignored by default because it is a
/// file-producing helper, not an assertion:
/// `cargo test -p rw-wrfbatch --test stored_planes -- --ignored`.
#[test]
#[ignore = "writes a durable fixture for the artifact proof"]
fn write_durable_fixture() {
    let target = std::env::var("RW_STORED_PLANE_FIXTURE")
        .expect("set RW_STORED_PLANE_FIXTURE to the directory to write into");
    let dir = PathBuf::from(target);
    std::fs::create_dir_all(&dir).expect("create the fixture directory");
    let path = stored_plane_fixture::write(&dir);
    println!("FIXTURE {}", path.display());
}
