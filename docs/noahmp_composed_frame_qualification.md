# Noah-MP composed units: how `sf_surface_physics = 4` is priced

Until 2.7.2 every configuration that selected Noah-MP was refused at plan
review with

    cannot price the local-memory reservation: noahmp_driver, noahmp_energy,
    noahmp_thermal do not compile at this checkout (NVRTC: identifier "r_pow"
    is undefined), so their per-thread local frame has never been measured.
    Refusing to guess.

The refusal was right to refuse and wrong about why.  Those three files (and
`noahmp_glacier.cu`, `noahmp_libm_slab.cu`) are not broken kernels; they are
**fragments**.  glibc 2.39's `powf`/`expf`/`logf` are transcribed once in this
tree, in `noahmp_leaves.cu`, and the fragments borrow that copy, so NVRTC
refuses each one handed over alone -- and the memory estimator was handing them
over alone, because its census of per-thread local frames enumerates the
`*.cu` files that compile standalone.  The units the model actually launches
are the compositions `gpuwm/core/noahmp_kernel_sources.py` builds (leaves +
driver, leaves + energy, ...), and nothing had ever read *their* frames.

This document records what now prices scheme 4, what was measured, on what,
what a card without a reading of its own is charged, and the one case that is
still refused.

## One authority for what NVRTC is handed

`gpuwm/core/noahmp_kernel_sources.py` names the fifteen Noah-MP runtime
translation units and, for each, the ordered `.cu` stems, the common preamble,
the exact option tuple and every exported `__global__` (inherited helpers
included).  `compile_runtime_unit(name, ...)` is the **one** `cp.RawModule`
site for Noah-MP: the driver, energy, thermal and glacier factories, the libm
slab, and `gpuwm.core.kernels.load_module` for a standalone Noah-MP stem all
compile through it.  The frame measurement compiles through it too, so the
source string a forecast hands NVRTC is the string the recorded frame was read
from -- by construction, not by two assemblers happening to agree
(`tests/test_noahmp_frame_provenance.py::test_actual_runtime_factory_compiles_authoritative_source`,
`tests/test_kernel_loader_inert.py::test_the_noahmp_runtime_route_assembles_the_loader_source_byte_for_byte`).

| Unit | Ordered source stems | Exports | Pricing key |
|---|---|---:|---|
| bareflux | bareflux | 3 | noahmp_bareflux |
| driver | leaves + driver | 10 | noahmp_driver_composed |
| energy | leaves + energy | 10 | noahmp_energy_composed |
| fluxprep | fluxprep | 3 | noahmp_fluxprep |
| glacier | leaves + glacier | 9 | noahmp_glacier_composed |
| leaves | leaves | 8 | noahmp_leaves |
| libm_slab | leaves + energy + libm_slab | 14 | noahmp_libm_slab_composed |
| radiation | radiation | 8 | noahmp_radiation |
| sflx | sflx | 2 | noahmp_sflx |
| snow | snow | 10 | noahmp_snow |
| soilwater | soilwater | 7 | noahmp_soilwater |
| thermal | leaves + thermal | 14 | noahmp_thermal_composed |
| vegeflux | vegeflux (C++14, no preamble) | 5 | noahmp_vegeflux_runtime |
| vegprecip | vegprecip | 2 | noahmp_vegprecip |
| water | water | 3 | noahmp_water |

Fifteen units, five compositions, 108 unit/function pairs.
`preflight._LAND_SURFACE_KERNEL_MODULES[4]` names these fifteen pricing keys
and no fragment; a selector row that named a fragment would be refused by name
as a table defect (`kernel_local_frame_bytes`).  No `.cu` file changed.

## What is measured, and how

A per-thread local frame is what NVRTC emits for **one target architecture at
one compiler build**; `gpuwm/core/kernel_frame_recordings.py` has documented
since 2026-08-20 that the same source reads differently across those pairs
(`noahmp_leaves` alone: 272 B on sm_120 / NVRTC 13.0.48, 208 B on sm_120 /
13.3.33, 272 B on sm_120 / 13.0.88).  So the Noah-MP reading is a **row per
compile platform**, `ComposedUnitFrameRecording`, in that module's
`NOAHMP_COMPOSED_FRAME_RECORDINGS`.

The instrument is `python tools/measure_noahmp_frames.py measure --output
<receipt.json>`.  In a fresh process with an empty CuPy cache it compiles each
of the fifteen units through `compile_runtime_unit`, asks every exported kernel
of the loaded module for `CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES`
(`get_function(...).attributes["local_size_bytes"]`, plus registers, shared,
const, max threads, PTX and binary version for the audit trail), records the
maximum per unit, reads the compile platform through
`gpuwm.certify.compile_platform.compile_platform_fingerprint`, and prints the
row.  Nothing is launched and no constant table is uploaded: a frame is a
compile attribute, and the driver takes the reservation it prices at launch.
Each row also binds the `identity_sha256` of every unit it read (ordered
component hashes, preamble, composed source, options, exports), so an edit to
any Noah-MP source, to the preamble or to an option tuple makes the row stop
matching the tree and `tests/test_noahmp_frame_provenance.py::test_shipped_rows_describe_the_units_in_this_tree`
go red until the platform is re-read.

`python tools/measure_noahmp_frames.py verify` is the driver gate: it takes
the same reading and compares it with the tree's row for this platform, exit 0
on exact agreement.  `tests/test_noahmp_frame_provenance.py::test_this_platforms_row_is_what_the_card_compiles_to`
(`ARWEN_TEST_NOAHMP_GPU=1`) runs it under pytest.

### The readings of record

Four rows, one per compile platform, every one read with the instrument above
in a fresh process (fifteen units, 108 exports, zero launches) inside the
environment whose NVRTC it names.

Three were read on the sm_120 reading machine, an NVIDIA GeForce RTX 5070 Ti
(70 SMs x 1,536 threads), Linux, CuPy 14.2.0:

* **sm_120 at NVRTC 13.3.33** (read 2026-09-10) -- the compiler of a venv whose
  `[ctk]` extra resolved in the cuda-toolkit 13.3.x window (13.3.1); the same
  compile platform as the standalone recording `SM120_NVRTC_13_3_33`.
* **sm_120 at NVRTC 13.4.59** (read 2026-09-10) -- the compiler a fresh
  `pip install gpuwm[gpu-cu13]` has installed since 2026-09-09, when
  cuda-toolkit 13.4.1 became the `[ctk]` resolution and pinned
  `nvidia-cuda-nvrtc 13.4.59`; read with that library first on the loader path.
* **sm_120 at NVRTC 12.9.86** (read 2026-09-11) -- the compiler a fresh
  `pip install gpuwm[gpu-cu12]` installs, in a venv built from exactly that
  requirement: `cupy-cuda12x[ctk]>=14.0` resolves cuda-toolkit 12.9.2.0, which
  pins `nvidia-cuda-nvrtc-cu12 12.9.86`. The resolution was re-checked against
  the live index the same day (`resolve --extra gpu-cu12`, exit 0).

The fourth was read on the platform the **packaged desktop runtime** compiles
on, which is a CUDA-12 install on a consumer Ampere card:

* **sm_86 at NVRTC 12.9.86** (read 2026-09-11) -- a Windows development
  desktop, NVIDIA GeForce RTX 3080 (68 SMs x 1,536 threads), read through the
  desktop runtime's own bundled interpreter (CuPy 14.2.0,
  `nvidia-cuda-nvrtc-cu12` 12.9.86, build id CL-36037853) with this tree on
  `sys.path`. Taken three times in independent fresh processes across two
  sittings, byte-identical frames each time; the receipts are
  `evidence/noahmp-composed-frames-20260911/sm86-nvrtc-12.9.86.receipt.json`
  (with a bounded eight-stem standalone calibration pass beside it, every stem
  reproducing the `SM86_NVRTC_13_0_48` row to the byte at this compiler) and
  `evidence/noahmp-composed-frames/noahmp-frames-sm86-nvrtc-12.9.86.json`. The
  sm_120 CUDA-12 receipt is
  `evidence/noahmp-composed-frames/noahmp-frames-sm120-nvrtc-12.9.86.json`.

Every unit identity is the same across all four rows, so all four describe the
units this tree holds. The frames are not the same, and the differences are the
platform's:

| Pricing key | sm_120, 13.3.33 and 13.4.59 | sm_120, 12.9.86 | sm_86, 12.9.86 |
|---|---:|---:|---:|
| noahmp_glacier_composed | 456 | 456 | 456 |
| noahmp_thermal_composed | 368 | 368 | 368 |
| noahmp_driver_composed | 352 | 288 | 304 |
| noahmp_water | 224 | 224 | 224 |
| noahmp_energy_composed | 208 | 272 | 208 |
| noahmp_leaves | 208 | 272 | 208 |
| noahmp_libm_slab_composed | 208 | 272 | 208 |
| noahmp_snow | 200 | 200 | 200 |
| bareflux, fluxprep, radiation, sflx, soilwater, vegeflux_runtime, vegprecip | 0 | 0 | 0 |

Reading that table, which is the calibration the fourth column exists for:

* The 13.3 to 13.4 compiler step moved nothing on sm_120. That is a measured
  fact about those two builds and licenses nothing about a third.
* The CUDA-12 compiler moves `noahmp_leaves` on sm_120, 208 B to 272 B, and
  with it the two units whose maximum IS the leaves frame
  (`noahmp_energy_composed`, `noahmp_libm_slab_composed`).
  `noahmp_driver_composed`, whose maximum is its own kernel, moves the other
  way, 352 B to 288 B. 272 B is not a new number for this architecture: the
  standalone census already reads `noahmp_leaves` 272 B on sm_120 at NVRTC
  13.0.48 and at 13.0.88, and this reading's three standalone stems agree with
  the 13.0.88 standalone row to the byte (leaves 272, snow 200, water 224).
* At that same CUDA-12 build, the sm_86 card reads `noahmp_leaves` back at
  208 B, which is what the standalone `SM86_NVRTC_13_0_48` row reads for that
  architecture, while `noahmp_driver_composed` reads 304 B against sm_120's
  288 B. Architecture and compiler each move frames, which is why a row is a
  reading of the pair and why neither of these two rows licenses the other.

The widest unit is `noahmp_glacier_composed` at 456 B on all four platforms,
under the 1,024 B fresh default stack every one of them reports, so Noah-MP
adds **0 B** to the launch-time local-memory reservation on all four; a Noah-MP
configuration pays whatever the rest of its kernel set pays, exactly as a Noah
one does. The refusal that stood since the 1.8.8 sweep was guarding a term that
costs nothing on any platform yet read.

### What sets the compile platform

The NVRTC half of the platform is **not a property of the machine**.  A CuPy
wheel carries no CUDA headers, so the package's GPU extras name
`cupy-cuda13x[ctk]>=14.0` / `cupy-cuda12x[ctk]>=14.0`; `[ctk]` requires
`cuda-toolkit[...]==13.*` (or `12.*`), and each cuda-toolkit release pins
`nvidia-cuda-nvrtc` to one exact build.  Whatever cuda-toolkit release is
newest on the index the day pip runs is the compiler every fresh install
compiles on -- the same RTX 5070 Ti machine read NVRTC 13.0.88, 13.3.33 and
13.4.59 from three venvs.  That is why one row was not enough: with only the 13.3.33 row,
every install made after 2026-09-09 landed on 13.4.59 and refused Noah-MP on
every card, including the card class this page names, while the README said
it admitted.

So the resolution is declared in the tree and gated:

* `RESOLVED_TOOLCHAIN_PINS` in `gpuwm/core/kernel_frame_recordings.py` records,
  per GPU extra, the requirement, the cuda-toolkit release it resolved to, the
  NVRTC build, the date, whether it is the current resolution, and the
  architectures on which this release admits Noah-MP for that build:
  `gpu-cu13` -> 13.4.59 (current, sm_120), 13.3.33 (the window before it,
  sm_120); `gpu-cu12` -> 12.9.86 (current, sm_120 and sm_86), which is what
  `gpu-cu12` and the `gpu` / `all` aliases install and what the packaged
  desktop runtime carries.  A CUDA-12 card of any other architecture is
  priced from the ceiling over the recorded platforms, with the basis
  stated, until a row is taken on it inside such an environment.
* `tests/test_kernel_frame_recordings.py::test_every_gpu_extra_has_one_current_resolved_toolchain_pin`
  holds each declared requirement byte-for-byte equal to pyproject's extra, and
  `::test_the_recorded_noahmp_platforms_include_what_the_spec_resolves_to`
  asserts that every current pin has a composed row at `(architecture, build)`
  for each architecture it lists, that every row's build is a declared
  resolution and its architecture is listed by that build's pin, and that at
  least one current install lands on a measured platform.  No network: an
  edit to the extra, a re-pin without a reading, a row from an undeclared
  library, or a listed architecture without a row is a red CPU test.
* `python tools/measure_noahmp_frames.py resolve` (network) asks pip for a
  clean dry-run resolution of each current requirement against the live index
  and compares the `nvidia-cuda-nvrtc` version with the declaration: exit 0
  while the index still hands out the declared build, 1 the day a new
  cuda-toolkit release moves it -- which is the day the table needs a row on
  every architecture that pin lists, or every fresh install drops from a
  measured Noah-MP price to the ceiling.  It is run before a cut;
  `tests/test_kernel_frame_recordings.py::test_the_declared_resolution_is_what_the_index_resolves_today`
  runs it under `GPUWM_NETWORK_TESTS=1`.

## What the estimator does with it

`preflight.kernel_local_frame_bytes(exp, profile=...)` prices the fifteen
Noah-MP keys through `gpuwm/core/noahmp_frame_provenance.py`
(`frame_basis_for_profile`), which reads the card's compile platform off
`DeviceLocalMemoryProfile.compile_platform`
(`(device_compute_capability, nvrtc_build)`, set when a present card is read
in-process by `local_memory_profile_from_device` or by the `gpuwm go` probe
subprocess through `profile_from_device_probe`) and answers one of two ways:

* **the card's platform has a row that describes this tree** -- the frames
  are that row's, and the basis reads "Noah-MP local frames measured on this
  card's compile platform sm_86/12.9.86 (NVIDIA GeForce RTX 3080,
  development-desktop, read 2026-09-11)".  The exact reading is preferred
  whenever there is one.
* **it has none** -- the card's platform was not read (an explicit profile
  with no compile platform, `--vram-gib` for a machine elsewhere,
  `GPUWM_NO_LOCAL_GPU`, no runtime, `gpuwm check`'s CPU-only route), or the
  pair has no reading, or its row went stale -- the frames are the
  element-wise **ceiling** over every row that describes this tree, the same
  rule every standalone kernel gets on an unrecorded platform
  (`KERNEL_MAX_LOCAL_SIZE_BYTES` is itself that ceiling), and the basis reads
  "Noah-MP local frames priced from the ceiling over the recorded platforms
  sm_120/13.3.33, sm_120/13.4.59, sm_120/12.9.86, sm_86/12.9.86; not measured
  on this card (no reading exists for its compile platform sm_89 / NVRTC
  12.9.86; a reading of this card's own platform is `python
  tools/measure_noahmp_frames.py measure --output <receipt.json>` run on it)".
  The reason clause names what actually kept this card unread: the pair with
  no row, the row that went stale and the unit that moved it, or that the
  platform was not read at all.

This is the rule since 2026-09-11.  The rule it replaces refused every
unrecorded platform by name, which on 2.7.2's table kept Noah-MP out of reach
of every desktop install (the desktop runtime is a CUDA-12 install; the table
held CUDA-13 rows only) while the standalone kernels beside it were priced
from their ceiling without comment.  A ceiling is never below any reading, so
it is the direction a rail gate can survive; what it is not is a measurement
of the card in front of the user, and the basis says so in words.

Where the basis is stated: `gpuwm check` prints `Noah-MP frame basis: ...`
beside the fit verdict in its summary, `NON-POOL BASIS: ...` under `--explain`
(`non_pool_basis(profile, exp)`, the card sentence followed by the Noah-MP
sentence), carries the same text in the `non_pool_basis` field of `--json`,
and on the CPU-only route (GPU readiness unjudged) puts it in the estimate's
`basis` beside what kept the card unread.  `gpuwm downscale --point` prices a
Noah-MP parent from the profile the sizing probe read under `--auto-vram` and
from the ceiling for `--card` / `--vram-gib` targets.

The profile a Noah-MP configuration is priced on is the one the caller asked
about, never a substitute: an explicit profile is used as given; with no
profile and no declared card the question is about this machine, so this
machine's card is read and priced from its own row when it has one; with a
declared card (`--vram-gib` and no profile) this machine's card is NOT read on
its behalf and the reference profile prices the ceiling.  An absent card keeps
the reference geometry -- the retired `CARD_CLASS_MULTIPROCESSORS` defect is
a declared card priced on a smaller card's SMs, and
`test_an_absent_card_is_priced_on_its_own_geometry_from_the_ceiling` holds it
shut.  The present card is read once per process
(`live_device_local_memory_profile` keeps the first read, which is also the
only one that can measure the bare context).

One refusal remains, and it names the tool: a tree with **no usable Noah-MP
reading at all** -- no row, or every row read from a different source, option
tuple or export set than the tree holds (each row binds the `identity_sha256`
of every unit it read, and a stale row is withdrawn from both the exact match
and the ceiling).  Then there is nothing to price from and `gpuwm check`
refuses, naming the unit that moved and `tools/measure_noahmp_frames.py
measure`; the CPU-only route withholds its portable planning command in that
case because a declared card would refuse for the same reason.
`tests/test_noahmp_frame_provenance.py::test_shipped_rows_describe_the_units_in_this_tree`
makes a stale row a red CPU test the day a Noah-MP source moves, so the
refusal is a development-time signal, not a user-facing state of a release.

The composition-pricing walk (`tests/test_composition_pricing.py`) prices
every loader-accepted composition on one explicit profile -- the reference
geometry carrying the first recorded platform -- so all of its rows, the
scheme-4 rows included, price on every machine and under the switch.  Every
other scheme prices exactly as before -- no device read, the reference profile
for absent and undeclared cards (`test_non_noahmp_pricing_is_untouched`; the
30 non-Noah-MP full estimates and 40 domain inventories in
`tests/fixtures/noahmp_estimator_baseline_snapshots.json` are digests the base
tree `3904ae6e6` produced, and the candidate reproduces them).

### Door proofs of record (2026-09-11)

`evidence/noahmp-composed-frames/door-check-gfs-thompson-rrtmgp-mynn-noahmp.toml`
is the GFS 12 km quickstart domain with Thompson (mp 8), RTE+RRTMGP (4/4),
MYNN (bl 5, sfclay 5) and Noah-MP (lsm 4).  `gpuwm check` on it, from this
tree:

* the sm_120 reading machine (RTX 5070 Ti), the `gpu-cu12` venv (CuPy 14.2.0, NVRTC
  12.9.86): exit 0, "GPU fit estimate: fits with 12.09 GiB headroom", basis
  "Noah-MP local frames measured on this card's compile platform
  sm_120/12.9.86 (NVIDIA GeForce RTX 5070 Ti, the sm_120 reading machine, read
  2026-09-11)"; JSON in `door-check-sm120-nvrtc-12.9.86.json`.
* the development desktop, RTX 3080, through the shipped ArWen 2.7.2 desktop
  runtime's own interpreter (the door a desktop user runs): recorded in
  `door-checks-20260911.md` beside the JSON.

The ceiling case is a unit test rather than a door run, because no card on
hand has an unrecorded platform:
`tests/test_noahmp_frame_provenance.py::test_another_platform_is_priced_from_the_ceiling_and_says_so`
forces `sm_99 / NVRTC 0.0.2` against a recorded `sm_99 / NVRTC 0.0.1` row and
asserts the estimate is made, the frames are the element-wise ceiling, and the
basis sentence names the ceiling and the unread pair; the same file holds the
absent-card, unread-machine, declared-card and stale-row shapes of it, and
`tests/test_preflight.py::test_noahmp_on_an_unread_card_is_priced_from_the_ceiling_and_says_so`
does it against the shipped rows.

## Adding a platform

On the machine with the card, from a checkout, in the environment the
forecast will run in (the NVRTC that resolves is part of the platform):

```bash
python tools/measure_noahmp_frames.py measure --output noahmp-frames.json \
    --box "<machine name>" --platform-family linux
```

Paste the printed `ComposedUnitFrameRecording(...)` into
`NOAHMP_COMPOSED_FRAME_RECORDINGS` with a comment saying what it prices, run
`python tools/measure_noahmp_frames.py verify` (exit 0), and the CPU suites
`tests/test_noahmp_frame_provenance.py` and
`tests/test_kernel_frame_recordings.py`.  If the environment is what a GPU
extra resolves to today (`python tools/measure_noahmp_frames.py resolve` says
so), add or update its entry in `RESOLVED_TOOLCHAIN_PINS` naming the
architecture, so the gate knows the install lands on a recorded platform.  A
reading is never overwritten and never back-filled: a platform that was not
read has no row, and is priced from the ceiling until it has one.

## Scope of the change

The release note for this work is the 2.7.3 entry in `CHANGELOG.md`.

No CUDA source changed and no physics changed.  Nine Python files under
`gpuwm/core` changed (the Noah-MP factories now delegate to one compile site;
`preflight` prices scheme 4 from per-platform rows and carries the compile
platform on the device profile; `kernel_frame_recordings` gained the row type,
the four readings and the resolved-toolchain declaration;
`noahmp_frame_provenance` prices a card from its own row or from the ceiling
over the rows and words the basis; `downscale` hands the
auto-sizer the measured card's profile), so a release statement that the
scientific *sources* are byte-identical to the previous release must exclude
those Python modules and say so; the `.cu` tree is unchanged.
