//! Named pictures must not pay for diagnostic grids outside their catalog.
mod stored_plane_fixture;

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::process::Command;

struct Scratch(PathBuf);

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn render(root: &Path, tag: &str, products: &str, input: &Path) -> String {
    render_inputs(root, tag, products, &[input.to_path_buf()], false)
}

fn render_inputs(
    root: &Path,
    tag: &str,
    products: &str,
    inputs: &[PathBuf],
    catalog: bool,
) -> String {
    let mut command = Command::new(env!("CARGO_BIN_EXE_rw_wrfbatch"));
    command
        .args(["--products", products, "--width", "480", "--height", "360"])
        .arg("--store-root")
        .arg(root.join(format!("store-{tag}")))
        .arg("--out-dir")
        .arg(root.join(format!("png-{tag}")))
        .args(inputs)
        .env("CUDA_VISIBLE_DEVICES", "")
        .env("RUSTWX_BATCH_RENDER_THREADS", "2");
    if catalog {
        command.arg("--list-products");
    } else if inputs.len() == 2 {
        command.args(["--frames", "1"]);
    }
    let output = command.output().unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout).unwrap()
}

#[test]
fn rainfall_windows_and_their_availability_read_only_the_rainfall_inputs() {
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let scratch = Scratch(
        std::env::temp_dir().join(format!("rainfall-render-{}-{nonce}", std::process::id())),
    );
    std::fs::create_dir_all(&scratch.0).unwrap();
    let inputs = [
        stored_plane_fixture::write_rain_frame(&scratch.0, 0, 0.0),
        stored_plane_fixture::write_rain_frame(&scratch.0, 3600, 3.0),
    ];
    let listing = render_inputs(&scratch.0, "catalog", "qpf_1h,qpf_total", &inputs, true);
    assert!(
        listing
            .lines()
            .any(|line| line.starts_with("PRODUCT\tqpf_1h\twindowed\trenderable\t"))
    );
    assert!(
        !listing.contains("Computing WRF diagnostic"),
        "rainfall availability must not compute parcel columns"
    );
    let full = render_inputs(
        &scratch.0,
        "full",
        "qpf_1h,qpf_total,var:wrf_t2",
        &inputs,
        false,
    );
    let rain = render_inputs(&scratch.0, "rain", "qpf_1h,qpf_total", &inputs, false);
    assert!(!rain.contains("Computing WRF diagnostic"));
    let mut expected = pictures(&full);
    assert!(expected.remove("var:wrf_t2").is_some());
    assert_eq!(expected.len(), 2);
    assert_eq!(pictures(&rain), expected);
}

fn pictures(stdout: &str) -> BTreeMap<String, Vec<u8>> {
    stdout
        .lines()
        .filter_map(|line| line.strip_prefix("RENDERED "))
        .filter_map(|line| line.split_once(' '))
        .map(|(slug, path)| (slug.to_string(), std::fs::read(path).unwrap()))
        .collect()
}

#[test]
fn named_import_avoids_unused_diagnostics_and_preserves_every_named_picture() {
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let scratch =
        Scratch(std::env::temp_dir().join(format!("named-render-{}-{nonce}", std::process::id())));
    std::fs::create_dir_all(&scratch.0).unwrap();
    let input = stored_plane_fixture::write(&scratch.0);
    // A generic variable explicitly requests the complete diagnostic import.
    // Discover the reference set before running the filtered named import.
    let catalog = render_inputs(&scratch.0, "catalog", "var:wrf_t2", &[input.clone()], true);
    let expected_slugs: BTreeSet<String> = catalog
        .lines()
        .filter_map(|line| line.strip_prefix("PRODUCT\t"))
        .filter_map(|line| {
            let fields: Vec<_> = line.split('\t').collect();
            (fields.len() >= 3 && fields[1] != "generic" && fields[2] == "renderable")
                .then(|| fields[0].to_string())
        })
        .collect();
    assert!(
        expected_slugs.len() > 20,
        "the fixture must exercise a complete named gallery"
    );
    let mut products: Vec<_> = expected_slugs.iter().cloned().collect();
    products.push("var:wrf_t2".into());
    let full = render(&scratch.0, "full", &products.join(","), &input);
    let named = render(&scratch.0, "named", "all", &input);
    assert!(full.contains(": stp_effective"));
    assert!(
        !named.contains(": stp_effective"),
        "unused effective-layer calculation delayed named pictures"
    );
    assert!(
        named.contains(": cloudfrac"),
        "canonical cloud charts need the split diagnostic"
    );
    let mut expected = pictures(&full);
    assert!(expected.remove("var:wrf_t2").is_some());
    assert_eq!(
        expected.keys().cloned().collect::<BTreeSet<_>>(),
        expected_slugs
    );
    let actual = pictures(&named);
    assert_eq!(
        actual.keys().cloned().collect::<BTreeSet<_>>(),
        expected_slugs
    );
    assert_eq!(actual, expected);
}
