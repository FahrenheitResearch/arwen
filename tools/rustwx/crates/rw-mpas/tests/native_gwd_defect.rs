//! Strict executable witnesses for MPAS OL row-zero behavior. File/word
//! processing remains in Rust; Python only launches the native executables.
use rw_mpas::native_gwd::{NativeMesh, box_size};
use std::path::PathBuf;

#[test]
#[ignore = "requires both hosts' strict native small-box output witnesses"]
fn small_box_native_defect_word_set() {
    let root = PathBuf::from(std::env::var("MPAS_GWD_LANE_ROOT").unwrap());
    let input = PathBuf::from(std::env::var("MPAS_GWD_DEFECT_INPUT").unwrap());
    let binary = std::env::var("MPAS_GWD_BINARY").unwrap();
    let mode = std::env::var("MPAS_GWD_WITNESS_KIND").unwrap_or_else(|_| "defect".into());
    let kind = mode.as_str();
    let halo = kind == "halo";
    let scaling = if halo { 1.0 } else if kind == "tiny" { 0.001 } else { 0.003 };
    let out = root.join(format!("n1out/{kind}-port.nc"));
    let receipt_path = root.join(format!("n1out/{kind}-port.json"));
    let status = std::process::Command::new(binary).args([
        "--grid", input.to_str().unwrap(), "--out", out.to_str().unwrap(), "--geog",
        root.join("geog-view").to_str().unwrap(), "--native-gwd-static", "--gwd-only",
        "--gwd-cell-scaling", &scaling.to_string(), "--receipt", receipt_path.to_str().unwrap(),
    ]).status().unwrap();
    assert!(status.success());
    let mesh = NativeMesh::read(&input, scaling).unwrap();
    let undefined: Vec<usize> = (0..mesh.lat.len())
        .filter(|&c| box_size(mesh.lat[c], mesh.dc[c]).1 < 4).collect();
    let port_receipt: serde_json::Value = serde_json::from_slice(&std::fs::read(receipt_path).unwrap()).unwrap();
    if undefined.is_empty() {
        assert!(port_receipt.get("native_undefined_ol1_cells").is_none());
        assert!(port_receipt.get("native_undefined_exclusion_status").is_none());
    } else {
        assert_eq!(port_receipt["native_undefined_ol1_cells"], serde_json::json!(undefined));
        assert_eq!(port_receipt["native_undefined_exclusion_status"],
            "PROPOSED FOR OWNER SIGN-OFF: case-specific sign-off required");
    }
    let port = netcrust::File::open(&out).unwrap();
    let mut runs = Vec::new();
    let mut differences = 0usize;
    for host in ["node1", "node2"] {
        for run in ["rank1-a", "rank1-b", "rank4"] {
            let native_path = root.join(format!("oracle/native-{kind}-{host}/{run}/x1.2562.init.nc"));
            let native = netcrust::File::open(&native_path).unwrap();
            let mut fields = Vec::new();
            for name in ["var2d", "con", "oa1", "oa2", "oa3", "oa4", "ol1", "ol2", "ol3", "ol4"] {
                let a = native.read_f64(name).unwrap();
                let b = port.read_f64(name).unwrap();
                assert_eq!(a.len(), b.len());
                let mut changed = Vec::new();
                let mut native_undefined_words = Vec::new();
                for (index, (&x, &y)) in a.iter().zip(&b).enumerate() {
                    let (x, y) = ((x as f32).to_bits(), (y as f32).to_bits());
                    if name == "ol1" && undefined.binary_search(&index).is_ok() {
                        native_undefined_words.push(x);
                    } else if x != y {
                        changed.push(serde_json::json!({"index": index, "native": x, "port": y}));
                    }
                }
                differences += changed.len();
                fields.push(serde_json::json!({"field": name, "words": a.len(),
                    "differences": changed, "native_undefined_words": native_undefined_words}));
            }
            runs.push(serde_json::json!({"host": host, "run": run, "reference": native_path,
                "native_process_count": if run == "rank4" { 4 } else { 1 }, "port_process_count": 1,
                "reference_sha256": rw_mpas::sha256_file(&native_path).unwrap(), "fields": fields}));
        }
    }
    let receipt = serde_json::json!({"schema": "native-gwd-undefined-witness/v1",
        "declaration": if halo { "Native defects reproduced" } else { "Exclusions proposed for owner sign-off" },
        "owner_case_sign_off": false, "port_process_count": 1,
        "whole_field_bitwise": undefined.is_empty() && differences == 0,
        "field": if halo { "var2d" } else { "ol1" }, "indices": undefined,
        "reason": if halo { "mpas_init_atm_gwd.F:218,313-314 reads the absent-neighbour sentinel" } else { "mpas_init_atm_gwd.F:1126-1128 reads row zero when ny<4" },
        "in_bounds_reading": if halo { "zero contributions at absent-neighbour sentinels" } else { "central rows clipped to the allocated box, retaining at least one row" },
        "other_word_differences": differences, "runs": runs});
    std::fs::write(root.join(format!("n1out/native-{kind}-witness.json")), serde_json::to_vec_pretty(&receipt).unwrap()).unwrap();
    drop(port);
    std::fs::remove_file(out).unwrap();
    assert_eq!(differences, 0, "all words outside the declared OL1 set must match native");
}

/// Audit all retained native words after disposable NetCDF outputs are retired.
/// This is proposal evidence, never an approval or a whole-model proof.
#[test]
#[ignore = "requires sealed native witnesses and a new metadata output directory"]
fn sealed_native_defect_witness_process_counts() {
    let source = PathBuf::from(std::env::var("MPAS_GWD_SEALED_DIR").unwrap());
    let output = PathBuf::from(std::env::var("MPAS_GWD_METADATA_DIR").unwrap());
    assert!(!output.exists());
    std::fs::create_dir_all(&output).unwrap();
    for kind in ["defect", "tiny", "halo"] {
        for proof_host in ["node1", "node2"] {
            let path = source.join(format!("native-{kind}-witness-{proof_host}.json"));
            let mut receipt: serde_json::Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
            assert_eq!(receipt["other_word_differences"], 0);
            let indices = receipt["indices"].as_array().unwrap().clone();
            let runs = receipt["runs"].as_array_mut().unwrap();
            assert_eq!(runs.len(), 6);
            let mut ol1_words = Vec::new();
            for run in runs.iter_mut() {
                run["native_process_count"] = serde_json::json!(if run["run"] == "rank4" { 4 } else { 1 });
                run["port_process_count"] = serde_json::json!(1);
                let fields = run["fields"].as_array().unwrap();
                assert_eq!(fields.len(), 10);
                for field in fields { assert!(field["differences"].as_array().unwrap().is_empty()); }
                let field = fields.iter().find(|f| f["field"] == "ol1").unwrap();
                let words = field["native_undefined_words"].as_array().unwrap().clone();
                assert_eq!(words.len(), indices.len());
                ol1_words.push(words);
            }
            let mut pairs = Vec::new();
            for a in 0..runs.len() {
                for b in a + 1..runs.len() {
                    let changed: Vec<_> = ol1_words[a].iter().zip(&ol1_words[b]).zip(&indices)
                        .filter_map(|((x, y), index)| (x != y).then_some(index.clone())).collect();
                    pairs.push(serde_json::json!({"a_host": runs[a]["host"], "a_run": runs[a]["run"],
                        "a_process_count": runs[a]["native_process_count"],
                        "b_host": runs[b]["host"], "b_run": runs[b]["run"],
                        "b_process_count": runs[b]["native_process_count"],
                        "differing_indices": changed, "differing_words": changed.len()}));
                }
            }
            if !indices.is_empty() {
                assert!(pairs.iter().any(|p| p["differing_words"].as_u64().unwrap() > 0),
                    "a proposal needs actual strict-native run-to-run variation");
            }
            receipt["schema"] = serde_json::json!("native-gwd-proposed-exclusions/v2");
            receipt["declaration"] = serde_json::json!(if indices.is_empty() {
                "Native defects reproduced" } else { "Exclusions proposed for owner sign-off" });
            receipt["owner_case_sign_off"] = serde_json::json!(false);
            receipt["whole_field_bitwise"] = serde_json::json!(indices.is_empty());
            receipt["port_process_count"] = serde_json::json!(1);
            receipt["proposal_witness_process_count"] = serde_json::json!(1);
            receipt["native_run_pairs"] = serde_json::json!(pairs);
            receipt["source_witness_sha256"] = serde_json::json!(rw_mpas::sha256_file(&path).unwrap());
            receipt["bounds_witness"] = serde_json::json!("native-small-bounds/ol1.log (pristine get_ol1, one process)");
            std::fs::write(output.join(format!("native-{kind}-witness-{proof_host}.json")),
                serde_json::to_vec_pretty(&receipt).unwrap()).unwrap();
        }
    }
}
