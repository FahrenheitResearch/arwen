# Simulated radar source and build provenance

WOOF retains its simulation and writer sources in this public source tree.
Neither build depends on publishing an extraction branch.

| Component | Source revision | Licence | Record |
| --- | --- | --- | --- |
| BowEcho reference | `eea07fc0a032d4de439d0bfe2def9098afd8b5a8` | MIT OR Apache-2.0 | `bowecho/SOURCE.json` |
| Simulation extraction | `6b752c0a923d8fcef45d58f165b96a20886964d1` | MIT OR Apache-2.0 | `bowecho/SOURCE.json` |
| recast-radar-tools 0.1.3 | `c206a2495c36341caa2a62ff7be3025320dbb028` | MIT OR Apache-2.0 | `recast-radar-tools/SOURCE.json` |

The extraction record lists the simulation modules moved out of the desktop
application, the generic atmosphere input, the containing-cell lookup fixes,
and the standalone workspace changes. The vendored tree is the build input.
Each `SOURCE.json` records every retained file except itself by byte count and
SHA-256, distinguishes upstream bytes from packaging additions, and names the
retained notice files. `NOTICE` in each subtree scopes the upstream notices.

`crates/rw-simradar/Cargo.toml` resolves BowEcho and the writer crates through
relative `path` dependencies. The writer workspace resolves its own six
crates the same way, and the Rust lockfile records them as local packages
without a registry or Git source. No recast-radar Rust crate needs a crates.io
release or a Git URL.

The rest of the Rust dependency closure is checked in under `crates-io` and
`bowecho-git`. The parent `.cargo/config.toml` replaces registry and pinned
Git sources with those directories. Build from `tools/rustwx`, so Cargo reads
that configuration:

```sh
cargo metadata --locked --offline --format-version 1
cargo build --release --locked --offline -p rw-simradar
```

The public snapshot applies `RELEASE-EXCLUDE.txt` to tracked source files.
The radar vendors, their notices, the lockfile and Cargo source-replacement
configuration are retained. `tests/test_licence_notices_ship.py` checks the
file inventory, local writer dependency paths, source replacements, retained
snapshot paths and both binary-form notice copies. The committed source and
an offline build remain the release proof; the checks do not publish anything.
