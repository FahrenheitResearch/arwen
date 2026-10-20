// SPDX-License-Identifier: Apache-2.0
// Tool integrity tests only: no native model or observed physics claim.
use nextgen_replay::{
    file_sha256, nextgen_raw_alloc, nextgen_raw_free, nextgen_raw_load, nextgen_raw_load_owned,
    nextgen_raw_load_slice, nextgen_raw_read_many, nextgen_raw_sha256, nextgen_raw_store,
    nextgen_raw_write_many, ReadItem, WriteItem,
};
use serde_json::{json, Value};
use std::{ffi::CString, fs, path::Path, process::Command};

fn cpath(path: &Path) -> CString {
    CString::new(path.to_str().unwrap()).unwrap()
}

#[test]
fn packed_slice_preserves_unaligned_words_and_failed_reads() {
    let root = std::path::PathBuf::from(std::env::var("NEXTGEN_TEST_ROOT").unwrap())
        .join(format!("slice-{}", std::process::id()));
    fs::create_dir(&root).unwrap();
    let mut packed = vec![0x5a; 3];
    packed.extend(
        [0u64, 0x8000000000000000, 0x7ff8000000000042]
            .into_iter()
            .flat_map(u64::to_le_bytes),
    );
    packed.push(0xa5);
    let input = root.join("packed.raw");
    fs::write(&input, &packed).unwrap();
    let path = cpath(&input);
    let hash = CString::new(file_sha256(&input).unwrap()).unwrap();
    let bad_hash = CString::new("0".repeat(64)).unwrap();
    let mut error = [0i8; 256];
    let mut output = [0xcc; 24];
    unsafe {
        assert_eq!(
            nextgen_raw_load_slice(
                path.as_ptr(),
                hash.as_ptr(),
                3,
                output.as_mut_ptr(),
                output.len(),
                error.as_mut_ptr(),
                error.len()
            ),
            0
        );
        assert_eq!(output.as_slice(), &packed[3..27]);
        for (seal, offset, bytes) in [
            (bad_hash.as_ptr(), 3, 24),
            (hash.as_ptr(), packed.len(), 1),
            (hash.as_ptr(), usize::MAX, 2),
            (hash.as_ptr(), packed.len() + 1, 0),
        ] {
            output.fill(0xcc);
            assert_eq!(
                nextgen_raw_load_slice(
                    path.as_ptr(),
                    seal,
                    offset,
                    output.as_mut_ptr(),
                    bytes,
                    error.as_mut_ptr(),
                    error.len()
                ),
                2
            );
            assert_eq!(output, [0xcc; 24]);
        }
        assert_eq!(
            nextgen_raw_load_slice(
                path.as_ptr(),
                hash.as_ptr(),
                packed.len(),
                output.as_mut_ptr(),
                0,
                error.as_mut_ptr(),
                error.len()
            ),
            0
        );
        assert_eq!(output, [0xcc; 24]);
        assert_eq!(
            nextgen_raw_load_slice(
                path.as_ptr(),
                hash.as_ptr(),
                0,
                std::ptr::null_mut(),
                0,
                error.as_mut_ptr(),
                error.len()
            ),
            2
        );
    }
}

#[test]
fn owned_host_lifecycle_and_failure_outputs() {
    let root = std::path::PathBuf::from(std::env::var("NEXTGEN_TEST_ROOT").unwrap())
        .join(format!("owned-{}", std::process::id()));
    fs::create_dir(&root).unwrap();
    let words = [0u64, 0x8000000000000000, 0x7ff8000000000042];
    let bytes: Vec<u8> = words.into_iter().flat_map(u64::to_le_bytes).collect();
    let input = root.join("input.raw");
    fs::write(&input, &bytes).unwrap();
    let path = cpath(&input);
    let hash = CString::new(file_sha256(&input).unwrap()).unwrap();
    let bad_hash = CString::new("0".repeat(64)).unwrap();
    let mut error = [0i8; 256];
    let mut pointer = std::ptr::null_mut();
    unsafe {
        assert_eq!(
            nextgen_raw_load_owned(
                path.as_ptr(),
                hash.as_ptr(),
                bytes.len(),
                &mut pointer,
                error.as_mut_ptr(),
                error.len()
            ),
            0
        );
        assert_eq!(std::slice::from_raw_parts(pointer, bytes.len()), bytes);
        assert_eq!(
            nextgen_raw_store(
                cpath(&root.join("output.raw")).as_ptr(),
                pointer,
                bytes.len(),
                error.as_mut_ptr(),
                error.len()
            ),
            0
        );
        assert_eq!(fs::read(root.join("output.raw")).unwrap(), bytes);
        assert_eq!(
            nextgen_raw_free(pointer, error.as_mut_ptr(), error.len()),
            0
        );
        assert_eq!(
            nextgen_raw_free(pointer, error.as_mut_ptr(), error.len()),
            2
        );
        pointer = std::ptr::null_mut();
        assert_eq!(
            nextgen_raw_load_owned(
                path.as_ptr(),
                bad_hash.as_ptr(),
                bytes.len(),
                &mut pointer,
                error.as_mut_ptr(),
                error.len()
            ),
            2
        );
        assert!(pointer.is_null());
        assert_eq!(
            nextgen_raw_load_owned(
                path.as_ptr(),
                hash.as_ptr(),
                bytes.len() + 1,
                &mut pointer,
                error.as_mut_ptr(),
                error.len()
            ),
            2
        );
        assert!(pointer.is_null());
        assert_eq!(
            nextgen_raw_alloc(usize::MAX, &mut pointer, error.as_mut_ptr(), error.len()),
            2
        );
        assert!(pointer.is_null());
        assert_eq!(
            nextgen_raw_alloc(24, &mut pointer, error.as_mut_ptr(), error.len()),
            0
        );
        assert_eq!(std::slice::from_raw_parts(pointer, 24), [0; 24]);
        assert_eq!(
            nextgen_raw_load(
                path.as_ptr(),
                hash.as_ptr(),
                pointer,
                24,
                error.as_mut_ptr(),
                error.len()
            ),
            0
        );
        assert_eq!(std::slice::from_raw_parts(pointer, 24), bytes);
        assert_eq!(
            nextgen_raw_free(pointer, error.as_mut_ptr(), error.len()),
            0
        );
        let mut empty = std::ptr::null_mut();
        let mut other_empty = std::ptr::null_mut();
        assert_eq!(
            nextgen_raw_alloc(0, &mut empty, error.as_mut_ptr(), error.len()),
            0
        );
        assert_eq!(
            nextgen_raw_alloc(0, &mut other_empty, error.as_mut_ptr(), error.len()),
            0
        );
        assert!(!empty.is_null());
        assert_ne!(empty, other_empty);
        assert_eq!(nextgen_raw_free(empty, error.as_mut_ptr(), error.len()), 0);
        assert_eq!(
            nextgen_raw_free(other_empty, error.as_mut_ptr(), error.len()),
            0
        );
        assert_eq!(
            nextgen_raw_free(std::ptr::null_mut(), error.as_mut_ptr(), error.len()),
            0
        );
        assert_eq!(
            nextgen_raw_free(error.as_mut_ptr().cast(), std::ptr::null_mut(), 0),
            2
        );
        assert_eq!(
            nextgen_raw_alloc(1, std::ptr::null_mut(), error.as_mut_ptr(), error.len()),
            2
        );
        assert_eq!(
            nextgen_raw_load_owned(
                path.as_ptr(),
                hash.as_ptr(),
                24,
                std::ptr::null_mut(),
                error.as_mut_ptr(),
                error.len()
            ),
            2
        );
    }
    println!("Owned host ABI: exact NaN/signed-zero load/store; alloc/load/free lifecycle; failure pointers null; invalid frees rejected");
}
fn cli(root: &Path, request: &Value, name: &str, want: i32) -> Option<Value> {
    let input = root.join(format!("{name}.json"));
    let output = root.join(format!("{name}-verdict.json"));
    fs::write(&input, serde_json::to_vec_pretty(request).unwrap()).unwrap();
    let result = Command::new(env!("CARGO_BIN_EXE_nextgen-replay"))
        .args([&input, &output])
        .output()
        .unwrap();
    assert_eq!(
        result.status.code(),
        Some(want),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    println!(
        "{name}: exit={want}, {}{}",
        String::from_utf8_lossy(&result.stdout),
        String::from_utf8_lossy(&result.stderr)
    );
    if output.exists() {
        Some(serde_json::from_slice(&fs::read(output).unwrap()).unwrap())
    } else {
        None
    }
}

#[test]
fn executable_word_identity_and_integrity() {
    let root = std::path::PathBuf::from(
        std::env::var("NEXTGEN_TEST_ROOT")
            .expect("set NEXTGEN_TEST_ROOT to lane data-disk scratch"),
    )
    .join(format!("contract-{}", std::process::id()));
    fs::create_dir(&root).unwrap();
    let expected = root.join("f64-expected.raw");
    let actual = root.join("f64-actual.raw");
    // Preserve NaN payload and signed zero; never compare as floating values.
    let words = [
        0u64,
        0x8000000000000000,
        0x7ff8000000000042,
        0x3ff0000000000000,
    ];
    let bytes: Vec<u8> = words.into_iter().flat_map(u64::to_le_bytes).collect();
    fs::write(&expected, &bytes).unwrap();
    fs::write(&actual, &bytes).unwrap();
    let hash = file_sha256(&expected).unwrap();
    let evidence = root.join("origin.txt");
    fs::write(
        &evidence,
        b"Synthetic raw-word tool integrity fixture; no native physics\n",
    )
    .unwrap();
    let mut request = json!({"schema":"nextgen.raw.compare/v1","unit":"tool-integrity","case":"stored-words", "numerics":"strict",
        "candidate_revision": std::env::var("NEXTGEN_TEST_REVISION").unwrap(),
        "command":[env!("CARGO_BIN_EXE_nextgen-replay"), "REQUEST.json", "VERDICT.json"],
        "provenance":{"origin":"tool-self-test", "sources":[{"name":"original nextgen-replay integrity fixture", "revision":"v1", "licence":"Apache-2.0"}],
                      "evidence":[{"path":evidence,"sha256":file_sha256(&evidence).unwrap()}]},
        "fields":[{"name":"nan-and-zero","dtype":"f64","shape":[2,2],"order":"fortran","units":"1",
                   "expected":{"path":expected,"sha256":hash},"actual":actual}]});
    let verdict = cli(&root, &request, "exact", 0).unwrap();
    assert_eq!(verdict["compared_words"], 4);
    assert_eq!(verdict["different_words"], 0);
    let changed: Vec<u8> = [0u64, 0, 0x7ff8000000000043, 0x3ff0000000000000]
        .into_iter()
        .flat_map(u64::to_le_bytes)
        .collect();
    fs::write(&actual, changed).unwrap();
    let verdict = cli(&root, &request, "signed-zero-and-nan-differ", 1).unwrap();
    assert_eq!(verdict["different_words"], 2);
    assert_eq!(verdict["fields"][0]["first_difference"]["word_index"], 1);
    assert_eq!(
        verdict["fields"][0]["first_difference"]["expected_bits"],
        "0x8000000000000000"
    );
    assert_eq!(
        verdict["fields"][0]["first_difference"]["actual_bits"],
        "0x0000000000000000"
    );
    request["fields"][0]["expected"]["sha256"] = json!("0".repeat(64));
    assert!(cli(&root, &request, "bad-seal", 2).is_none());
    request["fields"][0]["expected"]["sha256"] = json!(hash);
    request["fields"][0]["shape"] = json!([5]);
    assert!(cli(&root, &request, "wrong-shape", 2).is_none());
    request["fields"][0]["shape"] = json!([2, 2]);
    let duplicate = request["fields"][0].clone();
    request["fields"].as_array_mut().unwrap().push(duplicate);
    assert!(cli(&root, &request, "duplicate-field", 2).is_none());
}

#[test]
fn batch_host_staging_preserves_bytes_and_rejects_races() {
    let root = std::path::PathBuf::from(std::env::var("NEXTGEN_TEST_ROOT").unwrap())
        .join(format!("staging-{}", std::process::id()));
    fs::create_dir(&root).unwrap();
    let paths = [cpath(&root.join("a.raw")), cpath(&root.join("b.raw"))];
    let source = [vec![0x42u8; 32], vec![0x81; 17]];
    let writes: Vec<_> = (0..2)
        .map(|i| WriteItem {
            path: paths[i].as_ptr(),
            source: source[i].as_ptr(),
            bytes: source[i].len(),
        })
        .collect();
    let mut error = [0i8; 256];
    unsafe {
        assert_eq!(
            nextgen_raw_write_many(writes.as_ptr(), 2, error.as_mut_ptr(), error.len()),
            0
        );
    }
    let hashes: Vec<_> = ["a.raw", "b.raw"]
        .iter()
        .map(|n| CString::new(file_sha256(&root.join(n)).unwrap()).unwrap())
        .collect();
    let mut dest = [vec![0u8; 32], vec![0; 17]];
    let mut reads: Vec<_> = (0..2)
        .map(|i| ReadItem {
            path: paths[i].as_ptr(),
            sha256: hashes[i].as_ptr(),
            destination: dest[i].as_mut_ptr(),
            bytes: dest[i].len(),
        })
        .collect();
    unsafe {
        assert_eq!(
            nextgen_raw_read_many(reads.as_ptr(), 2, error.as_mut_ptr(), error.len()),
            0
        );
    }
    assert_eq!(dest, source);
    let bad = CString::new("0".repeat(64)).unwrap();
    let mut untouched = [0xabu8; 32];
    unsafe {
        assert_eq!(
            nextgen_raw_load(
                paths[0].as_ptr(),
                bad.as_ptr(),
                untouched.as_mut_ptr(),
                32,
                error.as_mut_ptr(),
                256
            ),
            2
        );
    }
    assert_eq!(untouched, [0xab; 32]);
    reads[1].destination = reads[0].destination;
    unsafe {
        assert_eq!(
            nextgen_raw_read_many(reads.as_ptr(), 2, error.as_mut_ptr(), 256),
            2
        );
    }
    let mut hash_out = [0i8; 65];
    unsafe {
        assert_eq!(
            nextgen_raw_sha256(
                paths[0].as_ptr(),
                hash_out.as_mut_ptr(),
                error.as_mut_ptr(),
                256
            ),
            0
        );
    }
    unsafe {
        assert_eq!(
            std::ffi::CStr::from_ptr(hash_out.as_ptr()),
            hashes[0].as_c_str()
        );
    }
    unsafe {
        assert_eq!(
            nextgen_raw_write_many(writes.as_ptr(), 2, error.as_mut_ptr(), 256),
            2
        );
    }
    assert_eq!(fs::read(root.join("a.raw")).unwrap(), source[0]);
    println!(
        "C ABI batch bytes exact; bad-seal destination untouched; aliasing and overwrite rejected"
    );
}
