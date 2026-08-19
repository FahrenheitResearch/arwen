# Contributing

Thanks for considering a contribution to ArWen. Two things shape how
this project accepts changes: every WRF-derived mechanism is gated
against WRF v4.6.1 evidence before it ships, and the pipeline fails
closed rather than approximating. Contributions are expected to keep
both properties.

Before investing in an implementation, open an issue for anything
broad -- a new source, physics scheme, projection, or packaging change
-- so scope and the required evidence can be agreed first.

Reporting a failure rather than proposing a change: run `gpuwm report`
in the run directory and attach the zip it writes. It collects the
receipts, the failure, the logs, this install's identity, the card and
the free space, with machine identity redacted by class and a manifest
of what it contains printed before you send it
([reporting a problem](docs/public/REPORTING-A-PROBLEM.md)).

Keep contributions fail closed:

- never infer that a decodable product is a complete initial state;
- never substitute a projection, physics scheme, level, cadence, or
  missing-data policy silently -- refuse loudly or report the
  substitution explicitly;
- bind input and implementation authorities with sizes and SHA-256
  values;
- add a focused failure test for every new accepted control;
- distinguish structural writer tests from unchanged-stock-WRF
  execution evidence; and
- do not commit credentials, machine-specific absolute paths, private
  data, or source files whose redistribution terms are unclear.

Physics and numerics changes need evidence proportional to the claim:
a transcription change needs its oracle gate updated or extended
(never weakened); a claim of WRF agreement needs the measured
comparison, not a plausibility argument. The physics registry's
maturity labels (docs/public/PHYSICS.md) must stay truthful about what
has and has not been verified.

Practicalities:

- Python 3.11+; install with `pip install -e '.[gpu-cu12,render,dev]'`
  (`gpu-cu13` instead on a CUDA-13-only box);
  build the Rust workspace with
  `cargo build --release --locked --offline` from `tools/grib1_bridge`
  (the vendored, locked build is the supported one).
- This repository builds **two** distributions. `pyproject.toml` at the
  root builds `gpuwm`; `gpuwm-data/pyproject.toml` builds `gpuwm-data`,
  which carries the RRTMGP and Thompson table directories because the
  single wheel had reached 103.62 MiB against PyPI's 100 MiB cap. Build
  them with `python -m build --wheel` and `python -m build --wheel
  gpuwm-data`, from a tree with no stale `build/` in either place. They
  share one version string and are cut and uploaded together
  (RELEASE_CHECKLIST.md).
  A checkout needs no `pip install -e gpuwm-data` to READ the tables:
  `gpuwm.data_assets` falls to the sibling directory when `gpuwm` is
  running out of the same tree. Reach those files through
  `data_assets.data_path()` and never by joining onto `gpuwm/data`.
  `gpuwm doctor` is a different question and will still report
  `gpuwm-data: not installed` in such an environment, correctly: it
  reads the installed distribution's metadata, and an installed `gpuwm`
  really is missing a dependency it declares. Install the companion
  (`pip install -e gpuwm-data`) to clear the line.
- Run the focused tests for the changed surface first, then the broad
  CPU suite it touches. Rust changes must pass locked offline tests
  and strict formatting.
- GPU or real-data evidence should include the exact commit, hardware
  and runtime, argv, manifests, output hashes, timings, and a
  non-finite scan.

Contributions authored with AI assistance are welcome under the same
rules as any other: the evidence gates every change equally, and the
submitter is responsible for the claim their change makes.

Security reports: see [SECURITY.md](SECURITY.md). Do not change the
Apache-2.0 project license or any third-party license notice without
explicit maintainer authorization.
