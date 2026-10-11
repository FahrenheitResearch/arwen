`rw_globe_pack` is the CPU binary for the `woof-globe/1` viewer files.
It reads NetCDF through the existing Rust reader, derives and quantizes the
fields, aligns the payload, and writes JavaScript atomically. System zlib's
public compression ABI preserves the Python oracle's exact compressed bytes.
No Python field-processing fallback exists in the site caller.

Build from `tools/rustwx` on a permitted box:

```sh
CARGO_BUILD_JOBS=8 cargo build --release --locked -p rw-globe-pack
target/release/rw_globe_pack --abi
```

One JSON request arrives on stdin; one small JSON receipt leaves on stdout.
Errors leave no half-written script and return a nonzero exit status.

```json
{"action":"static","tape":"FILE","out":"DIRECTORY","id":"PACK_ID"}
```

```json
{"action":"frame","tape":"FILE","previous":null,"out":"DIRECTORY","id":"PACK_ID","init":"UTC_ISO_TIME"}
```

`previous` is a previous tape or the small private float64 accumulation
snapshot returned as `acc`. The site keeps only that snapshot and its metadata
until the next frame succeeds, then removes both. Full tapes are removed
after rendering and packing. The whole-pack caller clears its final snapshot.
`metadata` returns a tape's valid time, including a Times-only file.
`verify` returns a regular file's SHA-256 key and byte count.
`install` verifies `path` against `key`, copies to `dest` through a temporary
file, verifies the copy again, then publishes it. `verify-statics` checks an
on-box NPZ against its native provenance hash and requested truncation.
These local-file operations do not access a remote cache. Only C7 does that.

The binary resolves `libz.so.1` on Linux, `libz.1.dylib` on macOS, and
`zlib1.dll` or `zlib.dll` on Windows. `GPUWM_GLOBE_ZLIB` can name an installed
copy explicitly; an absent explicit copy refuses. Linux is the tested target.

The metadata layout is serialized in its own declared order. No workspace-wide
serde_json ordering feature or another binary's JSON serialization is changed.
