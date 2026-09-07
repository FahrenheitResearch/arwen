# ArWen visual launchpad

A local Rust window-in-your-browser for the existing ArWen engine. Opening it starts no download or forecast. It serves only on loopback, uses a random per-process request token and checks Host/Origin. Assets are embedded; there are no CDN or map-service requests.

The first screen offers editable starting configurations, an ArWen TOML import, and WRF `real.exe` / WPS `met_em` inputs. Region controls use an explicitly approximate geographic rectangle; the shared domain wizard fits it. After creation, the map displays sampled **actual projected cell edges** from the common projection implementation. Zoom, pan, and Fit region work offline. No weather echoes are fabricated.

## Build and launch

Requires Rust 1.94 and a working ArWen Python environment plus its native tools/data companion.

```
cd tools/arwen-launchpad
cargo build --release --locked --offline
```

Executable from this crate directory: `target/release/arwen-launchpad` (`.exe` on Windows), unless `--target-dir` is supplied.

```
./target/release/arwen-launchpad --python /path/to/environment/python --source /path/to/arwen --workspace /path/to/forecasts --geog-root /path/to/WPS_GEOG
```

Visit the printed local address. For a desktop shortcut, the packaging launcher should start this process hidden, capture its printed address, then open that URL in the default browser. `launch.ps1` supplies this behavior on Windows when placed beside the executable. It changes no global environment. The configured Python directory is prepended to the process-local PATH and the selected source is passed to descendants; use a matching installed/source identity for a release build.

## Engine integration

- New draft: existing `gpuwm domain` with exact requested controls. Slider spans are conveniences; numeric entry is not clamped to them. Integer-only wizard arguments reject fractional values instead of truncating.
- Review plan: actual `gpuwm go CONFIG --dry-run`; the human summary comes from the same parsed authority, with technical commands/logs collapsed.
- Readiness and system checks: existing `check` / `doctor`.
- Prepare & run: existing `go`; declared inputs and registered fetch routes retain their normal dispatch.
- Prepared folder: existing `sim`, with explicit or existing sibling WPS namelist. Review prepared launch uses `--print-command`.
- WRF / met_em: existing resolvers for inspection and existing `run` external-input door.
- Advanced commands: argument arrays passed directly to allowlisted existing engine verbs, never a shell.

Region and spacing dials support pointer dragging (including outside the dial while held), arrow keys and Page Up/Down; Shift makes a finer adjustment. Each retains an unrestricted exact numeric entry beside it. Dial movement adds decimal increments without rounding away entered precision. Suggested slider spans never clamp the requested value on rerender. Reduced-motion preferences disable effects and animated job scrolling.

The complete TOML remains available. `toml_edit` changes only selected values, preserving comments and unrelated/unknown settings. Scientific controls use the existing shared/per-domain schema. Root timestep fields preserve integer seconds plus an exact numerator/denominator; children retain their parent ratio. Shared surface/soil choices are labeled shared. Compatibility validation remains in the engine and does not silently choose replacement physics. An unfamiliar or currently invalid draft stays editable; saving is separate from engine acceptance. Geometry changes may require corresponding WPS/input authority changes; these are not silently rewritten.

Jobs reuse `arwen-tui/src/job.rs` and `gpuwm.tui_worker`: durable logs, real completion receipts, process-group/Windows Job Object ownership, explicit Stop, and continued execution if the browser closes. Active or unknown process state retains ownership of its configuration. Save detects outside edits, writes a UTF-8 backup, and atomically replaces the file; Save a copy is create-only. Results use the actual engine output folder. Public `gpuwm sim --restart` restores prepared hierarchy checkpoints. The GUI does not yet have a dedicated Resume control.

## Verification

```
cd tools/arwen-launchpad
cargo test --locked --offline
cd ../..
python -m pytest tests/test_launchpad_api.py -q
```

The Python group calls the real domain wizard and common experiment loader, including field scope, fractional seconds, projected footprints, integer refusals, and comment/unknown-option retention. Rust tests cover patch preservation, UTF-8/backup/external-edit handling, request-origin rejection, WPS argument composition, and conservative process ownership. Browser verification should use an isolated test browser against this loopback URL; do not automate a user's unrelated tabs. An end-to-end forecast is a separate explicit action requiring real retained inputs, not a mocked job result.

## Map attribution

The offline map combines Natural Earth land, lakes and boundaries with a GeoNames place catalog. The local map-sources link carries attribution and coverage; [the geography notes](../../docs/OFFLINE-LAUNCHPAD-GEOGRAPHY.md) record hashes and rebuilding instructions. Place selection changes only the requested center. Manual coordinates remain unrestricted, and scientific WPS geography is separate.