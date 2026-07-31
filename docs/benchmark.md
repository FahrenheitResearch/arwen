# Reproducible CPU/CUDA preprocessing benchmark

`tools/benchmark_preprocess_backends.py` measures the same hash-bound GFS
temperature field on the deterministic Rust CPU backend and the CuPy CUDA
backend. It reports serial CPU, parallel CPU, and CUDA timings, verifies that
CPU worker counts are byte-identical, and applies the frozen fieldwise
CPU-versus-CUDA tolerances. It is an evidence tool, not a portable sample:
the accepted real-data fixture is not redistributed by this repository.

## Inputs

Prepare these authorities from one already accepted GFS case:

- a decoded bridge directory containing `gate.tsv`, `decoded-sha256.tsv`, and
  the source snapshots;
- the ordered `HOUR<TAB>PATH` series and its cycle;
- the exact Lambert domain specification used by the accepted case;
- the prepared-cache directory, including `header.json`; and
- the release-built Rust CPU preprocessing library.

The benchmark validates the fixture and prepared-array hashes before timing.
It refuses to overwrite its JSON output.

## Run

Build the CPU library first, then run from the same clean source tree:

```bash
# cd into the crate: cargo finds the vendored-registry replacement in
# tools/grib1_bridge/.cargo/config.toml by walking up from the working
# directory, so a --manifest-path build from the repository root resolves
# against crates.io and cannot build air-gapped.
(cd tools/grib1_bridge && cargo build --release --locked --offline)

python tools/benchmark_preprocess_backends.py \
  --decoded /evidence/gfs/decoded \
  --series /evidence/gfs/gfs-series.tsv \
  --cycle 2026-07-20_00:00:00 \
  --domain-spec /evidence/gfs/domain.json \
  --prepared-cache /evidence/gfs/prepared \
  --cpu-bridge tools/grib1_bridge/target/release/libgpuwm_preprocess_cpu.so \
  --workers 16 --repeats 5 \
  --output /evidence/gfs/backend-benchmark.json
```

Use the platform library name on macOS or Windows. Run on an otherwise idle
host, retain every raw timing sample, and do not compare speedups from
different fixtures or domain shapes.

## Acceptance

The command exits zero only when both CPU worker-count identity checks and
CPU/CUDA numerical parity pass. Retain the output JSON with the exact Git
commit and tree state, Python/CuPy versions, CUDA driver/runtime, GPU and CPU
models, operating system, compiler/Rust versions, command line, and fixture
authorities. Timing alone is never a scientific acceptance result.
