# Porting `gpuwm/verify/obs/regrid.py` to Rust — design and parity record

`gpuwm/verify/obs/regrid.py` was the last `scipy` site on a shipped data
path. Every remap plan the observation battery builds went through
`scipy.spatial.cKDTree`, and the apply half moved the field values with
`numpy.add.at`. Drew's Python boundary (2026-08-16) names
"regrid/transform" verbatim, so both halves belong in Rust.

Both halves now run on `tools/rustwx/crates/obs-regrid` by default,
reached through `gpuwm/obs_regrid_bridge.py`.

## What was seeded, and what was not

`crates/rustwx-regrid` in Drew's consolidated workspace
(`%USERPROFILE%\rusty-weather-consolidated`) is the seed, and it supplies
the **shape** rather than the arithmetic:

| Seeded from `rustwx-regrid` | Written here |
|---|---|
| plan-as-data: a remap between two fixed grids is a fixed mapping, built once and applied many | the two operators, which that crate does not carry |
| `apply_into_*` writing a caller-owned buffer (what lets the ctypes seam fill preallocated numpy arrays) | the k-d tree, because gpuwm's grids are scattered curvilinear swaths, not a regular lat/lon spec |
| the bounded-distance validation and its refusal class | the chord-on-the-unit-sphere metric, rather than haversine kilometres |
| the error taxonomy (`error.rs`) | validity as an explicit boolean field remapped with the values, rather than a NaN sentinel + missing policy |

The two operators are genuinely different animals from that crate's:

* `nearest` there is index arithmetic on a regular grid with a
  brute-force fallback; here it is a bounded neighbour search over unit
  vectors, because an observation mosaic and a Lambert domain are not
  the same grid family.
* `cell_average` is **not** `conservative`. It is a REVERSE assignment:
  every source cell is given to its nearest destination centre, then
  each destination cell averages what landed on it. That is what the
  battery's qualification receipt measures the registered operator
  against, and an area-overlap remap would answer a different question.

## The parity contract

**Bitwise float64, on real bytes.** `golden/gen_regrid_goldens.py` runs
the real scipy/numpy path over real staged data — the 2024-05-21 case's
real 400x480 model grid and the real MRMS `MergedReflectivityQCComposite`
pack with its real coverage mask — and the crate's `tests/parity.rs`
compares IEEE bit patterns, not tolerances.

Eight golden cases: both operators obs-to-model, `nearest`
model-to-obs (the direction `battery.py` builds for the precipitation
leg), a bound tight enough to leave 11 687 of 12 000 destination cells
unreachable, a bound-edge pair one ULP apart that must straddle the
flip, and the tie pair below.

Three assumptions the bitwise claim rests on are **measured before a
golden byte is written**, and the generator exits if any fails:

1. **`deg2rad`.** numpy's ufunc is one multiply by the double nearest
   pi/180, which is what Rust's `PI / 180.0` folds to. Checked bitwise
   over 100 003 samples.
2. **Trig.** numpy's float64 `sin`/`cos` are bit-identical to the
   platform libm CPython calls, which is the libm Rust's `f64::sin`/`cos`
   lower to. Checked bitwise over 40 001 samples.
3. **The bound predicate.** scipy squares `distance_upper_bound` and
   accepts on the STRICT predicate `d2 < bound*bound`. Measured over
   6 000 boundary trials: the squared-strict spelling agreed 6 000
   times, `d < bound` agreed 5 508, `d2 <= bound*bound` agreed 4 981.
   Both of the wrong spellings look right until you land on the bound.

Measured full-resolution parity, real case, `max_distance_m = 12 000`:

| operator | source cells | destination cells | bitwise | Rust build | scipy build |
|---|---|---|---|---|---|
| `nearest` | 240 000 | 192 000 | yes | 0.059 s | 0.325 s |
| `cell_average` | 240 000 | 192 000 | yes | 0.072 s | 0.115 s |
| `nearest` | 960 000 | 192 000 | yes | 0.175 s | 0.256 s |
| `cell_average` | 960 000 | 192 000 | yes | 0.201 s | 0.327 s |

Speed is not why this moved, and the table is here so nobody has to
claim it was.

## The one divergence, and why it is not a parity failure

When two source cells are **exactly** equidistant from one destination
cell, `cKDTree` returns whichever the tree traversal reached first.
Measured over 400 duplicate-source trials it returned the lowest flat
index 232 times and some other tied index 168 times; over 400
mirror-symmetric trials, 268 / 132. There is no rule there to be
bit-exact to.

`obs-regrid` **defines** the answer: among exactly tied candidates the
lowest flat source index wins. The pruning rule descends into a box
whose lower-bound distance equals the best so far, rather than cutting
on `>=`, so a tied candidate with a lower index in an unvisited box is
still reachable.

The concrete breakage that rule prevents is in this module's own
docstring: *"Building [the plan] once per case and reusing it across arms
is not only faster, it is what makes the arms comparable: every arm is
remapped by the identical integer array, so no score can differ because
a neighbour search broke a tie differently."* A tie resolved by
traversal order is exactly that defect, one library upgrade away.

Following "never bit-exact to a bug", the exemption ships with a
**perturbed control**. `synthetic_tie_degenerate` is the exact tie and
is exempt from index parity; `synthetic_tie_control` is the same grid
with one ULP of longitude, where the nearest neighbour is a fact of the
geometry, and it is held to full bitwise parity. A test asserts that
exactly one case may be exempt and that the control exists.

## The seam

ctypes onto a cdylib, the same discipline as `netcdf-writer` and
`static-fields`. **Stateless**, unlike `static-fields`: no handle
registry, because a remap plan is two small arrays the Python dataclass
already owns and publishes — the battery reads `source_index` and
`reachable` for its receipt, and the plan crosses into evidence. A
registry would have made the one object in this subsystem that exists to
be inspected opaque.

```
gpuwm_obsregrid_abi_version()   -> u32          (probe, 1)
gpuwm_obsregrid_source_rev()    -> *const c_char (GPUWM_BRIDGE_SOURCE_REV)
gpuwm_obsregrid_last_error(buf, cap) -> usize
gpuwm_obsregrid_build_plan(...) -> i32          (the contract marker)
gpuwm_obsregrid_apply_plan(...) -> i32
```

Arrays cross as raw little-endian f64/i64/u8, exactly `numpy.tobytes()`
of a C-contiguous array; booleans as u8 because that is `numpy.bool_`'s
layout.

## Posture when the library is absent

`static_fields`' posture: default-on Rust, the scipy body reachable as
`GPUWM_OBSREGRID_PYTHON=1`, and every fallback prints one WORKAROUND line
per operation per process. The line names **what the degradation costs**,
not just what it substituted — that exactly-tied neighbours go back to
traversal order, so a remap plan stops being a function of its two grids
alone and arms remapped on different boxes may not be comparable.

The scipy body is kept verbatim as `_build_plan_python` /
`_apply_plan_python` rather than deleted, because the crate's goldens
were extracted from that code and a rewritten reference is no longer the
thing the port was proven against.

## Estate

* `gpuwm.bridge_assets.BUNDLED_ARTIFACTS` — `obs_regrid`, kind
  `library`, env `GPUWM_OBSREGRID_BRIDGE`, so `gpuwm fetch-bridges` (the
  command the refusal leads with) can actually supply it.
* `gpuwm.bridge_assets.LIBRARY_ABI` — `gpuwm_obsregrid_abi_version`, 1.
* `gpuwm.bridges.BRIDGE_ABI_MARKERS` — `gpuwm_obsregrid_build_plan`.
* `crates/obs-regrid/build.rs` — injects `GPUWM_BRIDGE_SOURCE_REV`, and
  `capi.rs` embeds it, so `build_bridge_bundle.py pin` can pin it.
* `gpuwm doctor` — the `observation remap (default battery remap engine)`
  line, `missing` rather than `info`, degradation text naming the tie
  breakage.

## Regenerating the goldens

```
python tools/rustwx/crates/obs-regrid/golden/gen_regrid_goldens.py
cd tools/rustwx && cargo test -p obs-regrid --offline
```

The generator refuses rather than substituting synthetic grids when the
staged case bytes are absent, and names the fetch command.

## Files

- `tools/rustwx/crates/obs-regrid/` — the crate (27 tests: 23 unit, 4 parity)
- `gpuwm/obs_regrid_bridge.py` — the ctypes seam
- `gpuwm/verify/obs/regrid.py` — routes to the seam by default
- `tests/test_obs_regrid_rust_parity.py` — 21 shipping-path tests
- `%USERPROFILE%\Downloads\evidence-gallery\obs-regrid-rust\` — charts and
  the measurement JSON
