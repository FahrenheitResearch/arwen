# Static-fields platform qualification

The lane-1 bit/byte contract is not qualified on Linux. The committed
fixtures were produced with NumPy 2.2.6 on Windows x86-64/UCRT; passing
them on Windows does not establish equality with native Linux NumPy.
The exact comparisons remain required. Native fixtures identify which
differences come from the reference platform and expose the remaining
implementation differences; they do not turn a failed platform green.

## Baseline and Rows attribution

On 2026-09-05, clean `4d13c38e3b8ef12a1b2debbfa024283442889dbe`
reproduced the same ten Linux lane-1 failures, including their first
differing values, seen in the global Rows composition based on
`3e2e7512cc94db06f01e4258c004ece2ad3d2867`. The clean baseline passed
all thirteen original lane-1 tests on Windows. The optional large sweep
was not enabled in those original runs.

The composition changes to this crate were confined to Rows dispatch,
its new projection, the sampling twin, raster dispatch and the highres
test. Its classic Lambert/Mercator/polar arithmetic, `npmath.rs`, NPZ
writer and lane-1 fixtures were unchanged. These existing Linux failures
are therefore not evidence of a Rows regression.

## Native reference results

Fresh authorities run the actual Python product implementation with
`GPUWM_STATIC_PYTHON=1`, set by the extractor before product imports.
NumPy is pinned to 2.2.6 for these independent platform references.
The portable sampler authority below is deliberately pinned from Rust and
bounded separately against every corresponding independent NumPy array.

The profiles tested with Rust 1.94.0 were Windows 11 x86-64,
CPython 3.13.7/MSVC 1944/UCRT, and Ubuntu 24.04 under WSL2 x86-64,
CPython 3.12.3/GCC 13.3.0/glibc 2.39. Both NumPy installations reported
AVX2/FMA3 and AVX512_ICL support. The manifests record the full CPU
feature inventory, Python/compiler/libc identity, byte order, producer
and Python source SHA-256 digests, and array payload SHA-256 digests.
These measurements do not qualify other wheels, CPUs, libc versions or
NumPy versions.

After separating corridor metadata from its numeric probes and enabling
the 400,000-input sweep for each of sin/cos/exp/log:

| Reference and Rust target | Passing | Failing | Result |
|---|---:|---:|---|
| Committed Windows authority / Windows | 14 | 0 | Existing contract preserved; actual large sweep enabled |
| Fresh Windows Python authority / Windows | 13 | 1 | Corridor metadata drift; all numeric and NPZ checks pass |
| Fresh Linux Python authority / Linux | 4 | 10 | Eight grid/probe math checks, corridor metadata, and large exp sweep fail |

The four Linux passes are refusal diagnostics, corridor field cropping,
the small sin/cos/exp/log fixture, and deterministic NPZ bytes. The
larger Linux sweep matches sin, cos and log at all 400,000 inputs each;
exp differs at one of 400,000 inputs. Windows matches all four sweeps.
The sweep retains its existing NaN equivalence rule; finite results are
compared by their exact f32 bit patterns.

Representative remaining differences:

| Check | Linux Rust | Native Linux Python/NumPy |
|---|---|---|
| Lambert cone (truelats 38, 41) | 0.6361509085955668 | 0.6361509085955672 |
| Lambert float32 twin cone | `0x3f22dabb` | `0x3f22daca` |
| C-ABI mass latitude, index 8 | `0x4042df7a38dd2b1d` | `0x4042df7a38dd2b1e` |
| Corridor northeast longitude | -81.27278274740655 | -81.2727827474065 |
| exp sweep index 104439, input 87.68312 | 1.2030826e38 | 1.2030828e38 |

`probe_libm.py` independently explains why merely relabeling fixtures is
insufficient. Its Windows f64 NumPy-versus-libm probes have zero
mismatches. On Linux, f64 tan, atan, asin, acos, log, log10, exp, atan2
and pow disagree on some inputs; e.g. log10 differs at 1746/8000 and
exp at 981/20009. Several float32 functions delegated to scalar libm
also disagree. Fixing this requires a separately reviewed native math
implementation and a fresh exact qualification, not a looser tolerance.

Two findings are distinct from those arithmetic failures:

* NPZ byte 809 differs between the committed Windows fixture and Linux
  Rust because CPython ZIP `create_system` is 0 on Windows and 3 on
  Unix. `src/npz.rs` intentionally uses this native-platform rule.
  Rust is byte-exact against the independently generated native NPZ on
  each tested platform. A Windows ZIP is not the Linux Python authority.
* Current Python corridor geometry adds `frame_grid_id`,
  `ratio_to_frame` and `reference_origin_child_cells`; Rust's geometry
  and the committed manifest predate those keys. Fresh Windows and
  Linux references expose this schema drift. All 264 committed Windows
  payload files still match freshly generated Python files exactly;
  only `corridor.geometry` and `corridor.crop_geometry` differ after
  excluding the newly added provenance. This qualification change does
  not alter production corridor behavior or rewrite committed goldens.

## NumPy's AVX-512 loops (2026-09-29)

The Python grid transforms (`gpuwm/static/lambert.py` and the Mercator
and polar classes in `gpuwm/static/projection.py`) now take tan, atan,
atan2, log, log10, exp, asin, acos and pow from `gpuwm.core.host_libm`,
the C library one element at a time, which is what the Rust crate calls.
The Lambert inverse's atan2 came in with the log1p fix: NumPy 2.5.3's
AVX-512 arctan2 differed from glibc on 32,883 of 400,000 random pairs and
6,096 of 400,000 Lambert-inverse-shaped pairs.
Measured on the Ubuntu 24.04 WSL2 host above with NumPy 2.5.3: the two
`TestLane1GridParity` comparisons in `tests/test_static_rust_parity.py`
failed with NumPy's AVX-512 dispatch on and passed with
`NPY_DISABLE_CPU_FEATURES="X86_V4 AVX512_ICL"`; they now pass both
ways. With those loops disabled, NumPy 2.5.3's float64 exp, log, log10,
tan, atan, asin, acos and pow equal glibc's on every probed input. The
NumPy 2.2.6 authority extraction and the cargo lane-1 comparison were
not re-run, so whether the Linux differences listed above are the same
loops is not measured.

## Reproduce without overwriting another platform's fixtures

Use a Python environment with `numpy==2.2.6` on the target OS. Run the
extractor from the repository root; the output is a separate authority
directory, not the committed Windows directory. For Linux:

```sh
python tools/static_rust_port/extract_lane1_goldens.py --output work/static-platform-authority/linux-numpy-2.2.6
python tools/static_rust_port/gen_npmath_sweep.py work/static-platform-authority/linux-sweep
python tools/static_rust_port/probe_libm.py
GPUWM_STATIC_LANE1_GOLDENS="$PWD/work/static-platform-authority/linux-numpy-2.2.6" GPUWM_NPMATH_SWEEP="$PWD/work/static-platform-authority/linux-sweep" cargo +1.94.0 test --manifest-path tools/rustwx/Cargo.toml -p static-fields --offline --test lane1_goldens -- --nocapture
```

For Windows PowerShell:

```powershell
python tools/static_rust_port/extract_lane1_goldens.py --output work/static-platform-authority/windows-numpy-2.2.6
python tools/static_rust_port/gen_npmath_sweep.py work/static-platform-authority/windows-sweep
python tools/static_rust_port/probe_libm.py
$env:GPUWM_STATIC_LANE1_GOLDENS = (Resolve-Path work/static-platform-authority/windows-numpy-2.2.6).Path
$env:GPUWM_NPMATH_SWEEP = (Resolve-Path work/static-platform-authority/windows-sweep).Path
cargo +1.94.0 test --manifest-path tools/rustwx/Cargo.toml -p static-fields --offline --test lane1_goldens -- --nocapture
```

Those native runs currently fail as documented above. To run the
historical Windows authority, unset `GPUWM_STATIC_LANE1_GOLDENS` on
Windows. Retain `GPUWM_NPMATH_SWEEP` to actually execute the large sweep;
without that variable, the optional test returns without checking data.

The native loader refuses a non-Python backend, mismatched OS or
architecture, non-little-endian arrays, and a different NumPy version.
Every native array requires its manifest digest and is checked before
use. Negative smoke runs exercised all those refusal paths, a modified
payload, and a missing digest. The extractor also rejected NumPy 2.4.3
and refused to overwrite the committed Windows directory from Linux.
Native metadata is a provenance record, not a certificate that another
compiler/libc/CPU profile passes; run the actual comparisons.

## Release disposition

The measured Python parity claim is limited to the Windows numerical
profile above. Linux's failure to reproduce native NumPy bits does not
by itself establish invalid forecast numerics or an unsupported Linux
forecast path. This investigation did not compare forecast output or
find invalid geography. It established small exact differences in real
C-ABI grid arrays and corridor probes. Mixed Python/Rust preparation or
cross-platform reuse therefore produces probes that differ in the last
digits, and the corridor probe gate (`gpuwm/static/corridor.py`,
`grid_probe_drift`) compares them within
`gpuwm.static.grid_identity.GRID_POSITION_TOLERANCE_CELLS` of a cell:
that difference is admitted and a corridor grid placed elsewhere is
still refused.
This result does not establish same-backend preparation/run failure.

The three-key geometry drift affects the Rust library's legacy geometry
and its test contract. Current production Python receipt geometry calls
its own `corridor_geometry`, not Rust `CorridorGeometry::derive`; the
schema mismatch alone does not demonstrate a production receipt defect.

The finite disposition is:

1. Preserve the exact assertions, retain the measured Windows numerical
   qualification, and record native Linux Python parity as open. Do not
   use the inherited ten failures to reject the Rows change or to claim
   invalid forecasts.
2. Resolve the current Python/Rust geometry schema as separate bounded
   API/test maintenance, with independently regenerated Python metadata.
3. Require an exact native oracle run before extending the Python parity
   claim to a Linux profile. Any native transcendental implementation
   change belongs in its own reviewed work item. A platform fixture
   directory, successful build or small kernel pass cannot supply that
   qualification.

## Portable sampling and primary CI

The sampler now uses fixed vendored libm arithmetic in both float32 and
float64, including nest anchors and inverse source pixel coordinates.
The public XLAT, XLONG and MAPFAC path keeps its existing platform
qualification. The portable authority and its reason are declared in
`tools/battery/static_qualification.json`.

Porting the float32 reference alone cannot restore fallback parity: the
float64 sampler longitude also changes interpolated fields. The measured
NumPy mismatch is therefore retained explicitly instead of calling the
portable Rust outputs an independent oracle. `numpy-bounds.json` records
ULP, absolute and changed-value bounds for every portable array against
its original Windows NumPy 2.2.6 array. A required functional test checks
all those bounds. The exact portable payloads are also compared on both
operating systems, and complete statics are compared against Windows
preparations, including a large off-centre nest and real terrain.

The portable twin, translation, full sampling-surface and independent-bound
checks are functional controls on both operating systems. Their exact
CPU gate lines are in `cargo_gates.txt`. None of their failures can become
a platform-tolerated result. The generator is a separate maintenance
example, `repin_portable`, and the qualification target has no ignored test.
It contains 16 tests: seven functional controls and nine platform comparisons.
The Python qualification list contains the two public float64 grid
comparisons and a full WPS_GEOG build against the independent NumPy builder.
`full-build-numpy-bounds.json`, beside the coordinate bounds, records
measured maximum and mean absolute differences for every output field on
the reference parent grid, measured on Linux x86-64 with NumPy 2.2.6.
Nonzero caps round outward to two significant figures. Zero bounds retain
exact terrain, green fraction, categorical and soil
comparisons; the nonzero bounds isolate the changed sampling arithmetic.
The test also compares complete source coverage reports. The missing
WPS_GEOG skip remains limited to this named full-build check. A missing
bridge still fails. Other interpolation and high-resolution primitive
reference checks remain unchanged.

The terrain survey uses the same default Rust build as preparation.
`GPUWM_STATIC_PYTHON` remains a reported diagnostic fallback with a different
sampling contract. An in-memory preparation and its moves may use the Python
sampler when their contracts match in the same process. Prepared moving trees
loaded from disk record the contract; a missing,
old or incompatible contract is refused before integration, with the
command to prepare a new tree. Older sealed corridors retain their own
bytes and must not be validated by rebuilding them with this new arithmetic.

The qualification job retains all comparison logs and counts. Windows
must pass. Linux remains explicitly UNQUALIFIED for the existing public
float64 NumPy differences, but a missing test, an ignored test, a build
failure or a failed portable functional control is an operational failure
on either operating system. No tolerance is added to the exact comparisons.
