//! Real native-init field replay. Tests use published cases unchanged and
//! execute the Rust command, rather than invoking its computation library.
use std::path::PathBuf;

#[test]
#[ignore = "requires published native cases, geography, and explicit disposable output directory"]
fn published_native_init_gwd_words() {
    let cases = PathBuf::from(std::env::var("MPAS_GWD_CASES").unwrap());
    let geog = std::env::var("MPAS_GWD_GEOG").unwrap();
    let scratch = PathBuf::from(std::env::var("MPAS_GWD_REAL_DIR").unwrap());
    assert!(!scratch.exists(), "scratch must be new");
    std::fs::create_dir_all(&scratch).unwrap();
    let binary = std::env::var("MPAS_GWD_BINARY")
        .unwrap_or_else(|_| env!("CARGO_BIN_EXE_rw_mpas_static").into());
    let names = ["var2d", "con", "oa1", "oa2", "oa3", "oa4", "ol1", "ol2", "ol3", "ol4"];
    let mut decks = Vec::new();
    let mut total_differences = 0usize;
    for mesh in ["x1.2562", "x1.10242", "x1.40962"] {
        let first = cases.join(mesh).join("2023-03-31_12.00.00/init-modis20-v2");
        let grid = first.join(format!("{mesh}.grid.nc"));
        let out = scratch.join(format!("{mesh}.nc"));
        let build = scratch.join(format!("{mesh}-build.json"));
        let run = std::process::Command::new(&binary)
            .args(["--grid", grid.to_str().unwrap(), "--out", out.to_str().unwrap(),
                "--geog", &geog, "--native-gwd-static", "--gwd-only",
                "--receipt", build.to_str().unwrap()])
            .output().unwrap();
        std::fs::write(scratch.join(format!("{mesh}.log")), [&run.stdout[..], &run.stderr[..]].concat()).unwrap();
        assert!(run.status.success(), "{mesh}: {}", String::from_utf8_lossy(&run.stderr));
        let candidate = netcrust::File::open(&out).unwrap();
        for date in ["2023-03-31_12.00.00", "2022-12-22_12.00.00"] {
            let native_directory = cases.join(mesh).join(date).join("init-modis20-v2");
            let native_log = native_directory.join("case-7.model.log");
            let log = std::fs::read_to_string(&native_log).unwrap();
            let native_process_count: usize = log.lines().find_map(|line|
                line.trim().strip_prefix("MPI task count:")
                    .map(|count| count.trim().parse().unwrap())).expect("native MPI count required");
            assert!(native_process_count > 0);
            let reference_path = native_directory.join(format!("{mesh}.init.nc"));
            let reference = netcrust::File::open(&reference_path).unwrap();
            let mut fields = Vec::new();
            for name in names {
                let a = reference.read_f64(name).unwrap();
                let b = candidate.read_f64(name).unwrap();
                assert_eq!(a.len(), b.len());
                let mut samples = Vec::new();
                let mut differences = 0usize;
                for (index, (&x, &y)) in a.iter().zip(&b).enumerate() {
                    let (native, port) = ((x as f32).to_bits(), (y as f32).to_bits());
                    if native != port {
                        differences += 1;
                        if samples.len() < 16 {
                            samples.push(serde_json::json!({"index": index, "native": native, "port": port}));
                        }
                    }
                }
                total_differences += differences;
                fields.push(serde_json::json!({"name": name, "words": a.len(), "differences": differences, "samples": samples}));
            }
            decks.push(serde_json::json!({"mesh": mesh, "date": date, "reference": reference_path,
                "native_process_count": native_process_count, "port_process_count": 1,
                "process_count_evidence": native_log,
                "process_count_evidence_sha256": rw_mpas::sha256_file(&native_log).unwrap(),
                "reference_sha256": rw_mpas::sha256_file(&reference_path).unwrap(),
                "candidate_sha256": rw_mpas::sha256_file(&out).unwrap(), "fields": fields}));
        }
        drop(candidate);
        // Preserve only small receipts and logs, never disposable NetCDF output.
        std::fs::remove_file(out).unwrap();
    }
    let receipt = serde_json::json!({"schema": "native-gwd-real-replay/v2", "front_door": "rw_mpas_static",
        "port_process_count": 1,
        "whole_init_identity": false, "total_differences": total_differences, "decks": decks});
    std::fs::write(scratch.join("comparison.json"), serde_json::to_vec_pretty(&receipt).unwrap()).unwrap();
    assert_eq!(total_differences, 0, "see comparison.json for exact affected words");
}
