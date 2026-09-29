//! What a local run's default product set asks for, against the renderer
//! itself.
//!
//! The concrete breakages these tests prevent, each measured on a real
//! ArWen run:
//!
//! * `all` expanded to every stored variable as well as every named
//!   product, and an 18 h run published 137 of its 204 product folders as
//!   raw variables named with a hash (`var_wrf_t2_1222df9c491fb635`);
//! * `all` and `windowed` asked an 18 h run for 41 windows that close at
//!   F024 or F048, each skipped on every frame (805 skip lines), with a
//!   reason telling a local-run user to use "a HRRR extended cycle";
//! * every per-frame skip line of a series named the LAST input file, so
//!   an F000 reason was filed against the F018 frame;
//! * the fileless catalog could not say which products a wrfout import
//!   can ever draw, so the default preset asked for three that no wrfout
//!   carries the fields of.

mod stored_plane_fixture;

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

struct Scratch(PathBuf);

impl Scratch {
    fn new(tag: &str) -> Self {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "rw-catalog-keywords-{tag}-{}-{nonce}",
            std::process::id()
        ));
        std::fs::create_dir_all(&path).unwrap();
        Self(path)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn render(root: &Path, tag: &str, products: &str, inputs: &[PathBuf]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_rw_wrfbatch"))
        .args(["--products", products, "--width", "480", "--height", "360"])
        .arg("--store-root")
        .arg(root.join(format!("store-{tag}")))
        .arg("--out-dir")
        .arg(root.join(format!("out-{tag}")))
        .args(inputs)
        .env("GPUWM_NO_LOCAL_GPU", "1")
        .env("CUDA_VISIBLE_DEVICES", "-1")
        .env("RUSTWX_BATCH_RENDER_THREADS", "1")
        .output()
        .unwrap()
}

fn text(output: &Output) -> (String, String) {
    (
        String::from_utf8_lossy(&output.stdout).into_owned(),
        String::from_utf8_lossy(&output.stderr).into_owned(),
    )
}

/// `(slug, path)` of every RENDERED line.
fn rendered(stdout: &str) -> Vec<(String, String)> {
    stdout
        .lines()
        .filter_map(|line| line.strip_prefix("RENDERED "))
        .filter_map(|rest| rest.split_once(' '))
        .map(|(slug, path)| (slug.to_string(), path.to_string()))
        .collect()
}

/// Seven whole-hour frames, F000 to F006, of cumulative rain.
fn six_hour_series(dir: &Path) -> Vec<PathBuf> {
    (0..=6)
        .map(|hour| {
            stored_plane_fixture::write_rain_frame(dir, hour * 3600, (hour * (hour + 1)) as f32)
        })
        .collect()
}

/// A trailing 16-hex-digit hash on a file stem's last token.
fn ends_in_hash(stem: &str) -> bool {
    let tail = stem.rsplit('_').next().unwrap_or("");
    tail.len() == 16 && tail.bytes().all(|byte| byte.is_ascii_hexdigit())
}

#[test]
fn all_draws_the_named_products_and_no_stored_variable() {
    let scratch = Scratch::new("all");
    let wrfout = stored_plane_fixture::write(&scratch.0);
    let output = render(&scratch.0, "all", "all", &[wrfout]);
    let (stdout, stderr) = text(&output);
    assert!(output.status.success(), "{stdout}\n{stderr}");
    let drawn = rendered(&stdout);
    assert!(!drawn.is_empty(), "{stdout}");
    assert!(
        drawn.iter().all(|(slug, _)| !slug.starts_with("var:")),
        "'all' drew a raw variable: {drawn:?}"
    );
    assert!(
        drawn.iter().all(|(_, path)| !path.contains("_var_")),
        "{drawn:?}"
    );
}

#[test]
fn the_variables_keyword_draws_stored_variables_named_without_a_hash() {
    let scratch = Scratch::new("variables");
    let wrfout = stored_plane_fixture::write(&scratch.0);
    let output = render(&scratch.0, "variables", "variables", &[wrfout]);
    let (stdout, stderr) = text(&output);
    assert!(output.status.success(), "{stdout}\n{stderr}");
    let drawn = rendered(&stdout);
    assert!(
        drawn.iter().all(|(slug, _)| slug.starts_with("var:")),
        "'variables' drew a named product: {drawn:?}"
    );
    let user = format!("var:{}", stored_plane_fixture::USER_PLANE_STORE_NAME);
    let (_, path) = drawn
        .iter()
        .find(|(slug, _)| *slug == user)
        .unwrap_or_else(|| panic!("{user} was not drawn: {drawn:?}"));
    let stem = Path::new(path).file_stem().unwrap().to_string_lossy();
    assert!(
        stem.ends_with(&format!(
            "_var_{}",
            stored_plane_fixture::USER_PLANE_STORE_NAME
        )),
        "{stem}"
    );
    for (_, path) in &drawn {
        let stem = Path::new(path).file_stem().unwrap().to_string_lossy();
        assert!(!ends_in_hash(&stem), "a hashed folder name: {stem}");
    }
}

#[test]
fn a_catalog_keyword_leaves_out_windows_the_run_cannot_close() {
    let scratch = Scratch::new("windowed");
    let inputs = six_hour_series(&scratch.0);
    for keyword in ["windowed", "all"] {
        let output = render(&scratch.0, keyword, keyword, &inputs);
        let (stdout, stderr) = text(&output);
        assert!(output.status.success(), "{keyword}: {stdout}\n{stderr}");
        for longer in ["qpf_12h", "qpf_24h", "10m_wind_0_24h_max", "2m_temp_0_48h_max"] {
            assert!(
                !stdout.contains(&format!(" {longer} ")),
                "'{keyword}' asked a 6 h run for {longer}: {stdout}"
            );
        }
        let drawn = rendered(&stdout);
        assert!(
            drawn.iter().any(|(slug, path)| slug == "qpf_6h" && path.contains("_f006_")),
            "'{keyword}' did not draw the 6 h window the run closes: {drawn:?}"
        );
        assert!(!stdout.contains("HRRR"), "{stdout}");
    }
}

#[test]
fn a_named_window_longer_than_the_run_is_refused_in_the_runs_own_terms() {
    let scratch = Scratch::new("named");
    let inputs = six_hour_series(&scratch.0);
    let output = render(&scratch.0, "named", "qpf_24h,qpf_1h", &inputs);
    let (stdout, stderr) = text(&output);
    assert!(output.status.success(), "{stdout}\n{stderr}");
    let skips: Vec<&str> = stdout
        .lines()
        .filter(|line| line.starts_with("SKIPPED qpf_24h "))
        .collect();
    assert_eq!(skips.len(), inputs.len(), "{stdout}");
    for line in &skips {
        assert!(
            line.contains("this run's stored frames end at F006"),
            "{line}"
        );
        assert!(!line.contains("HRRR"), "{line}");
    }
    // A window the run DOES close is refused only where it has not closed
    // yet, and without the run-length clause.
    let early: Vec<&str> = stdout
        .lines()
        .filter(|line| line.starts_with("SKIPPED qpf_1h "))
        .collect();
    assert_eq!(early.len(), 1, "{stdout}");
    assert!(!early[0].contains("stored frames end"), "{}", early[0]);
    assert!(!early[0].contains("HRRR"), "{}", early[0]);
}

#[test]
fn every_skip_line_names_the_file_of_its_own_frame() {
    let scratch = Scratch::new("frames");
    let inputs = six_hour_series(&scratch.0);
    let output = render(&scratch.0, "frames", "qpf_6h", &inputs);
    let (stdout, stderr) = text(&output);
    assert!(output.status.success(), "{stdout}\n{stderr}");
    let skips: Vec<&str> = stdout
        .lines()
        .filter(|line| line.starts_with("SKIPPED qpf_6h "))
        .collect();
    assert_eq!(skips.len(), 6, "{stdout}");
    for (hour, line) in skips.iter().enumerate() {
        let expected = format!(
            "SKIPPED qpf_6h {}: F{hour:03}: ",
            inputs[hour].display()
        );
        assert!(line.starts_with(&expected), "{line}\nexpected {expected}");
    }
}

/// `slug -> (verdict, minimum hour)` from the fileless listing.
fn wrfout_rows() -> std::collections::HashMap<String, (String, String)> {
    let output = Command::new(env!("CARGO_BIN_EXE_rw_wrfbatch"))
        .arg("--list-products")
        .output()
        .unwrap();
    assert!(output.status.success());
    String::from_utf8_lossy(&output.stdout)
        .lines()
        .filter_map(|line| line.strip_prefix("WRFOUT\t"))
        .map(|rest| {
            let fields: Vec<&str> = rest.split('\t').collect();
            assert_eq!(fields.len(), 5, "{rest}");
            (
                fields[0].to_string(),
                (fields[2].to_string(), fields[3].to_string()),
            )
        })
        .collect()
}

#[test]
fn the_fileless_catalog_says_which_products_a_wrfout_can_ever_draw() {
    let rows = wrfout_rows();
    let selectable = rusty_weather::render_all::known_product_slugs();
    assert_eq!(rows.len(), selectable.len(), "one row per selectable slug");
    // The General preset's three that never drew: no wrfout import writes
    // their fields.
    for slug in ["10m_wind_gusts", "precipitation_type", "cloud_cover"] {
        assert_eq!(rows[slug].0, "missing", "{slug}: {:?}", rows[slug]);
    }
    // And the products the retired NEEDS/PLANNED reading wrongly called
    // undrawable, beside the layer cloud panel that does draw.
    for slug in [
        "2m_temperature",
        "500mb_height_winds",
        "850mb_temperature_height_winds",
        "composite_reflectivity",
        "cloud_cover_levels",
        "precipitable_water",
        "sbcape",
    ] {
        assert_eq!(
            rows[slug],
            ("drawable".to_string(), "0".to_string()),
            "{slug}"
        );
    }
    // A window carries the first hour it closes.
    assert_eq!(rows["qpf_1h"], ("drawable".to_string(), "1".to_string()));
    assert_eq!(rows["qpf_24h"], ("drawable".to_string(), "24".to_string()));
    assert_eq!(
        rows["10m_wind_24_48h_max"],
        ("drawable".to_string(), "48".to_string())
    );
}

#[test]
fn every_selector_a_real_import_writes_is_one_the_plan_names() {
    use rw_wrfbatch::wrf_process::{WrfProcessMessage, WrfProcessOptions, spawn_process_paths};

    let scratch = Scratch::new("plan");
    let wrfout = stored_plane_fixture::write(&scratch.0);
    let store = scratch.0.join("store");
    let options = WrfProcessOptions::default().normalized();
    let planned: std::collections::HashSet<String> = options
        .planned_store_selectors()
        .into_iter()
        .map(|selector| selector.key())
        .collect();
    let task = spawn_process_paths(vec![wrfout.clone()], store.clone(), options);
    let summary = loop {
        match task.rx.recv().unwrap() {
            WrfProcessMessage::Progress(_) => {}
            WrfProcessMessage::Done(result) => break result.unwrap(),
        }
    };
    assert_eq!(summary.frame_sources, vec![(0u16, wrfout)]);
    let source = rusty_weather::render_all::StoreFieldSource::open(
        &store,
        &summary.model,
        &summary.run,
        0,
    )
    .unwrap();
    let mut written = 0usize;
    for variable in source.surface_variables() {
        let Ok(selector) =
            serde_json::from_value::<rustwx_core::FieldSelector>(variable.selector.clone())
        else {
            continue;
        };
        written += 1;
        assert!(
            planned.contains(&selector.key()),
            "the import wrote {} under {}, which the plan never names",
            variable.name,
            selector.key()
        );
    }
    assert!(written > 5, "the fixture imported almost nothing: {written}");
}
