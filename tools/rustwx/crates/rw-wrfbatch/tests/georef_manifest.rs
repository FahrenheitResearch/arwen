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
