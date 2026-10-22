# Offline interface dependency inventory

`manifest.json` records both interface lockfile hashes, each package checksum
and its declared licence. The visual launcher and terminal interface resolve
registry sources from `tools/rustwx/vendor/crates-io`; no private copy is kept
here. Their existing Cargo.lock versions and bytes are unchanged. Package
licence files and Cargo per-file checksums remain in the shared registry.

For normal builds, change into `tools/arwen-launchpad` or `tools/arwen-tui`
and run `cargo build --release --locked --offline`. Cargo discovers the local
source replacement from the working directory. A cold Cargo home can build
from the checked-in sources without a registry connection.

Add dependencies using the scratch-vendoring procedure and all 16 manifests
listed in `tools/rustwx/VENDOR.md`; existing registry directories must never
be overwritten. That document also records the retained test-only log scrub.
Run `python tools/update_tui_license_notice.py` from the repository root to
refresh this inventory and both compiled-binary notices, then refresh the
Zarr notice, which retains a copy of the bridge notices.
