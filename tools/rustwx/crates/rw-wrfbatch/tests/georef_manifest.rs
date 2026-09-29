//! The georeference manifest, proven against the ARTIFACT: the real
//! `rw_wrfbatch` binary, a real wrfout import, a real render.
//!
//! The concrete breakage this gate prevents: a rendered PNG that appears
//! in the run's output directory but in NEITHER half of
//! `render-georef.json` -- a silent omission, which is exactly the defect
//! the manifest exists to fix (`rw_wrfbatch` published no geographic
//! transform anywhere, so consumers recovered one by registering
//! coastlines, and a linear registration measurably cannot describe a
//! projected panel).  Every rendered panel must land in `panels` with a
//! transform or in `without_georeference` with a reason naming what was
//! missing.
//!
//! The fixture's nest is REGIONAL, so its panel carries a domain frame
//! and the post-render recentre pass runs.  The passes now REPORT how
//! far they moved the map and the plot rectangle follows it, so a
//! regional panel PUBLISHES its transform -- "fixed means default"
//! demanded exactly this, because a transform published only for global
//! panels left the common case (every WRF nest) showing the defect.
//! This gate asserts the published transform places the fixture's domain
//! centre inside the written image; the earlier version of this gate
//! pinned the interim suppression behaviour and was upgraded when the
//! passes learned to report their offsets.

mod stored_plane_fixture;

use std::path::{Path, PathBuf};
use std::process::Command;

struct Scratch(PathBuf);

impl Scratch {
    fn new(tag: &str) -> Self {
        let dir = std::env::temp_dir().join(format!(
            "rw-wrfbatch-georef-{tag}-{}-{:?}",
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

#[test]
fn a_real_run_writes_the_manifest_and_accounts_for_every_rendered_panel() {
    let scratch = Scratch::new("run");
    let wrfout = stored_plane_fixture::write(scratch.path());
    let store_root = scratch.path().join("store");
    let out_dir = scratch.path().join("out");

    let output = Command::new(env!("CARGO_BIN_EXE_rw_wrfbatch"))
        .arg("--store-root")
        .arg(&store_root)
        .arg("--out-dir")
        .arg(&out_dir)
        .arg("--products")
        .arg(format!(
            "var:{}",
            stored_plane_fixture::USER_PLANE_STORE_NAME
        ))
        .arg(&wrfout)
        .output()
        .expect("launch the built rw_wrfbatch binary");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "rw_wrfbatch failed\nstdout:\n{stdout}\nstderr:\n{stderr}"
    );

    // The pinned event grammar is untouched, and the one NEW line type
    // arrives after FINISHED.
    let rendered_line = stdout
        .lines()
        .find(|line| line.starts_with("RENDERED "))
        .expect("the run must announce its rendered panel");
    let rendered_path = PathBuf::from(
        rendered_line
            .splitn(3, ' ')
            .nth(2)
            .expect("RENDERED <slug> <path>"),
    );
    let finished_at = stdout
        .find("FINISHED ")
        .expect("the run must announce FINISHED");
    let georef_at = stdout
        .find("GEOREF ")
        .expect("the run must announce its georeference manifest");
    assert!(
        georef_at > finished_at,
        "GEOREF must follow FINISHED:\n{stdout}"
    );

    // The manifest itself, beside the PNGs, default-on with no flag.
    let manifest_path = out_dir.join("render-georef.json");
    let manifest: serde_json::Value = serde_json::from_slice(
        &std::fs::read(&manifest_path).expect("a bare run must write render-georef.json"),
    )
    .expect("render-georef.json parses");
    assert_eq!(
        manifest["schema"].as_str(),
        Some("rustwx.render-georef/v1")
    );

    let key = rendered_path
        .strip_prefix(&out_dir)
        .expect("the rendered panel lives under out_dir")
        .to_string_lossy()
        .replace('\\', "/");
    let absences = manifest["without_georeference"].as_array().unwrap();
    let panel = manifest["panels"].as_object().unwrap().get(&key);
    assert!(
        panel.is_some() || absences.iter().any(|entry| entry["path"].as_str() == Some(key.as_str())),
        "rendered panel '{key}' is in neither half of the manifest -- the silent \
         omission the manifest exists to prevent:\n{manifest}"
    );

    // The GEOREF tallies are the manifest's own.
    let georef_line = stdout[georef_at..].lines().next().unwrap();
    let panel_count = manifest["panels"].as_object().unwrap().len();
    assert!(
        georef_line.contains(&format!("panels={panel_count}"))
            && georef_line.contains(&format!("without={}", absences.len())),
        "GEOREF tallies must match the manifest: {georef_line}"
    );

    // A regional nest PUBLISHES: the recentre pass reports its shift, the
    // rectangle follows the map, and nothing about this batch is left in
    // `without_georeference`.
    let panel = panel.unwrap_or_else(|| {
        panic!(
            "the regional panel must publish its transform, not sit in \
             without_georeference:\n{manifest}"
        )
    });
    assert!(
        absences.is_empty(),
        "a regional batch must leave without_georeference empty:\n{manifest}"
    );
    let georeference: rustwx_render::PanelGeoReference =
        serde_json::from_value(panel.clone()).expect("the published transform parses back");
    // The fixture grid spans lat 36.0..36.85, lon -98.0..-96.85; its
    // centre must land on a pixel inside the written image.
    let (px, py) = georeference
        .lonlat_to_pixel(36.4, -97.4)
        .expect("the domain centre must land on a pixel");
    assert!(
        px >= 0.0
            && py >= 0.0
            && px < f64::from(georeference.image_width_px)
            && py < f64::from(georeference.image_height_px),
        "the domain centre must land inside the image: ({px}, {py}) in {}x{}",
        georeference.image_width_px,
        georeference.image_height_px
    );
}

/// Launch one `rw_wrfbatch` batch over `wrfout` into `out_dir`.
fn launch_batch(wrfout: &Path, store_root: &Path, out_dir: &Path) -> std::process::Child {
    Command::new(env!("CARGO_BIN_EXE_rw_wrfbatch"))
        .arg("--store-root")
        .arg(store_root)
        .arg("--out-dir")
        .arg(out_dir)
        .arg("--products")
        .arg(format!(
            "var:{}",
            stored_plane_fixture::USER_PLANE_STORE_NAME
        ))
        .arg(wrfout)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .expect("launch the built rw_wrfbatch binary")
}

/// The manifest keys of every panel one finished batch announced.
fn rendered_keys(child: std::process::Child, out_dir: &Path) -> Vec<String> {
    let output = child.wait_with_output().expect("wait for rw_wrfbatch");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success(),
        "rw_wrfbatch failed\nstdout:\n{stdout}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let keys: Vec<String> = stdout
        .lines()
        .filter(|line| line.starts_with("RENDERED "))
        .map(|line| {
            PathBuf::from(line.splitn(3, ' ').nth(2).expect("RENDERED <slug> <path>"))
                .strip_prefix(out_dir)
                .expect("the rendered panel lives under out_dir")
                .to_string_lossy()
                .replace('\\', "/")
        })
        .collect();
    assert!(!keys.is_empty(), "each batch must render a panel:\n{stdout}");
    keys
}

/// The concrete breakage this gate prevents: the caller renders a run as
/// many invocations into one output directory, and each invocation wrote
/// the manifest from its own batch alone.  A real 6 h two-grid run of 646
/// pictures ended with 60 in the manifest, the last batch's, so a map
/// could place none of the rest.  Every panel of every batch must stay
/// recorded, including two batches that finish at the same moment, and a
/// panel rendered twice must be recorded once.
#[test]
fn every_batch_into_one_out_dir_stays_in_the_manifest() {
    let scratch = Scratch::new("batches");
    let out_dir = scratch.path().join("out");
    let frames: Vec<PathBuf> = (0..3)
        .map(|index| {
            let dir = scratch.path().join(format!("frame{index}"));
            std::fs::create_dir_all(&dir).expect("frame dir");
            stored_plane_fixture::write_rain_frame(&dir, 3_600 * (index + 1), 1.0)
        })
        .collect();
    let store = |tag: &str| scratch.path().join(format!("store-{tag}"));

    // One batch alone, then two together, then the first again.
    let mut expected = rendered_keys(launch_batch(&frames[0], &store("a"), &out_dir), &out_dir);
    let second = launch_batch(&frames[1], &store("b"), &out_dir);
    let third = launch_batch(&frames[2], &store("c"), &out_dir);
    let second_keys = rendered_keys(second, &out_dir);
    expected.extend(second_keys.iter().cloned());
    expected.extend(rendered_keys(third, &out_dir));
    expected.extend(rendered_keys(launch_batch(&frames[0], &store("a2"), &out_dir), &out_dir));
    expected.sort();
    expected.dedup();
    assert!(expected.len() >= 3, "three valid times must be three panels: {expected:?}");

    assert_eq!(
        recorded_keys(&out_dir),
        expected,
        "the manifest must record every panel of every batch exactly once"
    );
    assert!(
        !out_dir.join("render-georef.json.lock").exists(),
        "the merge lock must be released"
    );

    // A picture removed from the folder leaves the record at the next
    // batch, so the map is never offered a panel it cannot open.
    for key in &second_keys {
        std::fs::remove_file(out_dir.join(key)).expect("remove a rendered panel");
    }
    rendered_keys(launch_batch(&frames[2], &store("c2"), &out_dir), &out_dir);
    expected.retain(|key| !second_keys.contains(key));
    assert_eq!(recorded_keys(&out_dir), expected);
}

/// Every path `render-georef.json` records, placed or not, sorted.
fn recorded_keys(out_dir: &Path) -> Vec<String> {
    let manifest: serde_json::Value = serde_json::from_slice(
        &std::fs::read(out_dir.join("render-georef.json")).expect("the manifest exists"),
    )
    .expect("render-georef.json parses");
    let panels = manifest["panels"].as_object().unwrap();
    let absences = manifest["without_georeference"].as_array().unwrap();
    let mut recorded: Vec<String> = panels
        .keys()
        .cloned()
        .chain(absences.iter().map(|entry| entry["path"].as_str().unwrap().to_string()))
        .collect();
    recorded.sort();
    recorded
}
