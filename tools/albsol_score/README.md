# ALBSOL measurement scorer

This retained measurement scorer fixes the checkpoint fallback exposed by the ALBSOL box run. If history lacks COSZEN, hourly checkpoint reading now selects fields/coszen_ref.npy only when fields/coszen.npy is absent. A present primary member keeps precedence. Malformed present data, archive/read errors, and other missing fields are refused. The exact selected member supplies both scored values and JSON field provenance.

The scorer and standalone test crate compile the same src/checkpoint.rs. Six small stored-ZIP tests include the actual checkpoint metadata member and NPY byte-array representation. They cover an f32 reference, an f64 primary, missing both, malformed primary with valid reference, unrelated missing fields, and preserved shape/dtype/C-order/payload-length validation. No model data is copied into these tests.

The default Cargo paths are relative to this directory's intended engine location, tools/albsol_score. For a full CPU build against another owned engine source tree, run:

```sh
nice -n 10 bash ./run-full-scorer.sh /absolute/path/to/owned/engine /absolute/path/to/owned/cargo-home
```

The setup script verifies the three Rust vendor manifests and tools/rustwx/vendor/crates-io before writing relative Cargo dependency paths and an ignored local .cargo/config.toml. The runner uses the retained lockfile and offline vendor sources, hides CUDA, limits build/test threads to two, and keeps its TMPDIR and CARGO_TARGET_DIR inside this directory. It tests the full scorer before building the release executable, and records receipts/full-build.log and receipts/full-build.done. Dependencies and binaries are not copied into this kit.

The standalone reader proof needs only cached ZIP8.6.0:

```sh
nice -n 10 bash ./run-checkpoint-tests.sh
```

Validation on 2026-10-04: the standalone six tests passed on weather-node-1. The complete scorer's seven release tests passed on weather-node-4, including the six reader tests and the existing checkpoint filename test; the release executable also built successfully. The tested main.rs and checkpoint.rs bytes are pinned in receipts/source-packet.json. The release executable SHA256 is e9dd781fd93c40c3e9f50a8a5b094f8745263ffc4f08cfac7662702a31c6dbd6. Both nodes used CPU only, nice, two threads, owned temporary and target paths, and offline dependencies. Each temporary archive deletion is listed with its size in the direct test logs.

The original scorer is retained with the lane's measurement records. The defect and emergency box patch are recorded in the lane's box-run result and receipts/box-scorer-coszen-fallback.diff. This fix uses the ZIP inventory rather than catching every read error as that emergency patch did. The first full build setup failed to resolve serde_json from an empty index; receipts/full-build-first.log preserves it. Explicit source replacement to the verified engine vendor registry fixed resolution. The successful full build and dependency provenance are retained here. This task did not rerun the scorer against raw forecast outputs; the box receipts supply that real-output evidence.
