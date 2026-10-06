//! Both native run-input inventories preserve context and literal run names.
mod stored_plane_fixture;

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

struct Scratch(PathBuf);

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn render(root: &Path, tag: &str, a: &[PathBuf], b: &[PathBuf], inventory: bool,
          frame: &str) -> Output {
    let mut command = Command::new(env!("CARGO_BIN_EXE_rw_wrfbatch"));
    command.args(["--products", "qpf_1h", "--frames", frame, "--width", "480", "--height", "360"])
        .arg("--store-root").arg(root.join(format!("store-{tag}")))
        .arg("--out-dir").arg(root.join(format!("out-{tag}")))
        .args(["--diff-label-a", "A, native", "--diff-label-b", "B, native"])
        .env("GPUWM_NO_LOCAL_GPU", "1").env("CUDA_VISIBLE_DEVICES", "-1")
        .env("RUSTWX_BATCH_RENDER_THREADS", "1");
    if inventory {
        for (name, paths) in [("a.json", a), ("b.json", b)] {
            let ordered: Vec<String> = paths.iter().map(|path| {
                path.strip_prefix(root).unwrap().to_string_lossy().into_owned()
            }).collect();
            std::fs::write(root.join(name), serde_json::to_vec(&ordered).unwrap()).unwrap();
        }
        command.arg("--diff-inputs-json").arg(root.join("b.json"))
            .arg("--inputs-json").arg(root.join("a.json"));
    } else {
        for path in b {
            command.arg("--diff-against").arg(path);
        }
        command.args(a);
    }
    command.output().unwrap()
}

fn picture(output: &Output) -> PathBuf {
    assert!(output.status.success(), "stdout={}\nstderr={}",
            String::from_utf8_lossy(&output.stdout), String::from_utf8_lossy(&output.stderr));
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("a=\"A, native\" b=\"B, native\""), "{stdout}");
    assert!(stdout.contains("DIFFERENCE d01-3km_qpf_1h"), "{stdout}");
    assert!(stdout.contains("defined_cells=432"), "{stdout}");
    assert!(stdout.contains("FINISHED rendered=1 skipped=0 failed=0"), "{stdout}");
    let paths: Vec<PathBuf> = stdout.lines().filter_map(|line| {
        line.strip_prefix("RENDERED qpf_1h_difference ").map(PathBuf::from)
    }).collect();
    assert_eq!(paths.len(), 1, "{stdout}");
    paths[0].clone()
}

#[test]
fn both_ordered_input_files_match_inline_context_and_keep_comma_labels_literal() {
    let stamp = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
    let root = Scratch(std::env::temp_dir().join(format!("rw-difference-inputs-{}-{stamp}", std::process::id())));
    let a_root = root.0.join("a");
    let b_root = root.0.join("b");
    std::fs::create_dir_all(&a_root).unwrap();
    std::fs::create_dir_all(&b_root).unwrap();
    let a = [stored_plane_fixture::write_rain_frame(&a_root, 3600, 3.0),
             stored_plane_fixture::write_rain_frame(&a_root, 0, 0.0)];
    let b = [stored_plane_fixture::write_rain_frame(&b_root, 3600, 1.0),
             stored_plane_fixture::write_rain_frame(&b_root, 0, 0.0)];
    let absent = render(&root.0, "first-window", &a, &b, true, "0");
    assert!(absent.status.success(), "{}", String::from_utf8_lossy(&absent.stderr));
    let stdout = String::from_utf8_lossy(&absent.stdout);
    assert!(stdout.contains("SKIPPED qpf_1h run B:"), "{stdout}");
    assert!(stdout.contains("F000"), "{stdout}");
    assert!(stdout.contains("FINISHED rendered=0 skipped=1 failed=0"), "{stdout}");
    let inline = picture(&render(&root.0, "inline", &a, &b, false, "1"));
    let inventory = picture(&render(&root.0, "inventory", &a, &b, true, "1"));
    assert_eq!(std::fs::read(inline).unwrap(), std::fs::read(inventory).unwrap(),
               "ordered JSON transport changed a native QPF difference pixel");
}
