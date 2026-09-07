# Offline interface dependencies

This directory contains the unmodified Cargo vendor output for the visual
launcher and terminal interface. Both crates use this shared source mirror;
their existing Cargo.lock versions are unchanged. Package license files and
Cargo's per-file checksum records remain included.

`manifest.json` records both lockfile hashes, each package checksum and its
declared license, plus the regeneration command. Run that command from the
repository root using a populated Cargo registry cache. No dependency version
upgrade is part of this addition.

For normal builds, change into `tools/arwen-launchpad` or `tools/arwen-tui`
and run `cargo build --release --locked --offline`. Cargo discovers the local
source replacement from the working directory. Both packages are listed in
the release battery; a cold Cargo home can build them from these checked-in
sources without a registry connection. This says nothing about Python engine
or geographic-data installation, which have their own package checks.
