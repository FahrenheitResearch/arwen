# Changelog

## 2.3.1 (2026-08-14)

2.3.0 was tagged but never published: the RW-WPS staging gate refused the
standalone preprocessing wheel, so nothing reached PyPI. The tag stays where
it is, because tags here are forward-only. Everything 2.3.0 carried ships
here, plus the first fix below.

New:
- Forcing a nest off a streamed parent now moves O(child footprint) per
  parent step, not O(parent). The coupler pulls exactly the two windows
  FORCE reads -- the field and the coupling mass over the child's
  footprint plus an 8-parent-cell halo -- through the store seam, and the
  executor's duplicate projection is gone, so a relocation moves the
  window by rewriting the child's placement and nothing else. Measured on
  the 5070 Ti through `execute_experiment` (256x256x49 streamed parent,
  72x72 child at 3:1, 12 parent steps): bit-identical to the all-resident
  tree in every carrier of both domains, one-way and two-way, the parent
  store bitwise unchanged by nest presence, and the corridor moved
  58.0 MiB where whole-parent pulls would have moved 2352 MiB, a 40.5x
  reduction that grows with the parent. A halo-starved window is the
  negative control, and every run carries the traffic in
  `coupler.force_sync_bytes` as a receipt.
- The second concurrent-nesting shape streams: a RESIDENT parent can
  drive a TILE-STREAMED child, one run, coupled every parent step, one-way
  or two-way. FORCE pulls only the child's boundary frame from its pinned
  host store (four strips, `spec_bdy_width + 8` cells, the child-side
  mirror of the streamed-parent footprint corridor;
  `coupler.force_sync_bytes` is the receipt) and writes the same
  full-perimeter rolling tables the resident path writes; the child's tile
  buffers consume them through per-buffer packed table windows
  (`gpuwm.core.nest_stream`) that re-copy at kernel-launch time whenever
  the FORCE generation moves, so a buffer can never apply a previous
  interval's forcing. Two-way feedback restricts out of the child's store
  into the resident parent. Gated bit-identical to the all-resident tree
  through `execute_experiment` in every carrier of both domains, with
  stale-store, stale-tables and starved-frame negative controls firing in
  both directions.
- Roads are assigned per domain, and the tree is priced together.
  `steppers_for_tree` prices each domain against the budget its
  predecessors left: resident claims, streamed tile working sets, and the
  coupling corridor's slots. Every decision's receipt names its road, its
  claim and the budget spent before it, so a tree that cannot fit refuses
  with the arithmetic instead of dying at the allocation. A pinned tiling
  still asks no planner and probes no card. What remains refused by name:
  a coupling edge with BOTH ends streamed (ungated), a nest that also
  carries a tabulated boundary series, and `[relocation]` targeting a
  streamed child.
- The offline child can be asked to stream. `[tiles]` joins the tables a
  RunConfig TOML may carry -- the `gpuwm downscale` child schema refused
  it as unknown, so the route whose domain most outgrows its card was
  the one route that could not ask -- and the child route wires the seam
  exactly as the prepared single-domain runner does: decide once, hand
  the decision to `make_stepper`, record it. The three observers are
  wired with it (refresh on the history cadence and before the final
  read; a store-aware stability observer), `report.json` gains a `tiles`
  verdict, and `gpuwm downscale --tiles {on,auto}` writes the block into
  the config `--point` derives -- mode only, never a tiling, because the
  plan belongs to the card the run meets and not to the machine that
  derived it. A child that configures nothing is byte-for-byte the run it
  was.
- The last two nesting shapes are reachable from the product front door.
  `[tiles]` is now a PER-DOMAIN surface as well as a tree-wide one: a
  `[[domain]]` row carries its own `tiles = { ... }`, and the override
  replaces rather than merges, so "stream the parent, keep the child
  resident" and its roles-flipped twin are both things a user can write in
  a config instead of shapes that only a test could reach. Both run
  through `gpuwm run <config> --outdir <dir>` with no workaround flags and
  with `feedback = 1`. Run-plan delivers the same surface, and
  `docs/public/TILES.md` documents it.
- The tree decision RESERVES its children before a streamed parent picks
  its tile. A streamed parent's tile search maximises the compute window
  against whatever budget it is shown, so pricing parent-first let it take
  the largest clean tile that fit the whole card (3.94 of 4.00 GiB, 98.5%)
  and its resident child then met "no tile fits in 0.06 GiB". Only the
  order of the arithmetic made that shape unreachable. The reservation
  constrains the TILE, never the VERDICT, so a tree that decided
  all-resident decides all-resident still.

Fixed:
- The standalone RW-WPS preprocessing wheel builds again. The
  nest-streaming work above gave `gpuwm/core/streaming.py`, which that
  wheel stages, four new function-local imports of two modules it
  deliberately does not carry, and the builder's unresolved-import scan
  refused the staging. All four sit behind a
  running forecast, so both modules are recorded as forecast-only in the
  builder's exception table rather than staged. Build tooling only: no
  product source changed and the wheel's behaviour is unchanged; it was
  simply unbuildable at 2.3.0.
- Two-way nest feedback runs under a streamed parent. The executor
  refused `feedback = 1` whenever the parent streamed, claiming nothing
  projects a restriction back into the store; the coupler had carried
  exactly that projection since the same merge, proven bit-identical at
  the coupler level, and the two truths were never in one gate. The
  refusal is gone and the dispatch is mode-blind. Measured through
  `execute_experiment` on a 5070 Ti (256x256x49 parent streamed from a
  pinned host store at tile 128, 72x72 child at 3:1, 12 parent steps,
  radiation/cumulus/PBL/LSM firing on both sides): one-way and two-way
  runs are bit-identical to the all-resident tree in every carrier of
  both domains, the feedback arms demonstrably differ from the one-way
  arms, a stale-coupler negative control diverges, and the parent's store
  is bitwise unchanged by the presence of a one-way nest. What a streamed
  parent genuinely cannot do still refuses by name in the coupler: a
  cross-scheme microphysics edge, and a streamed child.
- `[tiles]` streaming starts on native Windows. Two stacked defects made
  every planner-driven streamed run refuse before the config was read:
  `autoplan.Machine.detect` sourced host RAM only from `/proc/meminfo`
  and the cgroup files, so a Windows box was wrongly told it was
  "containerised with no cgroup memory limit"; and `streaming.decide`
  probed the machine before applying the configured `vram_budget_bytes` /
  `host_budget_bytes` overrides, so the keys whose documented purpose is
  to override the probe were never read where it raised. Both halves are
  fixed default-on: host RAM comes from
  `GlobalMemoryStatusEx` (`ullTotalPhys`) on win32, and the configured
  budgets are applied before the machine is consulted, on every platform.
  Measured on an RTX 5090 at 438x350x49 with `mode = "on"` and nothing
  else, planner-chosen 219x175 x3 buffers: dry and full-physics rungs
  both bit-exact against the resident run, twice per rung on this no-ECC
  card with matching state digests.
- The ERA5 direct door's soil hand-off resolves the orography the GRIB
  carries. `rw-wps --source era5` accepted the wizard's config since the
  `[case_data]` fix, decoded the GRIB and built the statics, then died
  in `preprocess_noah_soil` with "terrain and source_orography must be
  provided together": `gpuwm fetch --source era5` writes the invariant
  geopotential INTO `era5-combined.grib`, so the route declares no
  separate orography artifact and the soil call forwarded a None beside
  a real HGT_M. The horizontal stage had already remapped SOILGEO onto
  the mass grid as SOURCE_OROGRAPHY; `soil_source_orography()` now
  resolves exactly that field once, beside the guard that depends on it.
  The declared artifact still outranks the embedded record.
- The root runtime and the nested child stop dropping the same
  orography. Both gated the pair defensively, so where the direct door
  refused loudly they silently skipped WRF's `adjust_soil_temp_new`
  altogether while their own met fields carried SOURCE_OROGRAPHY. Nests
  feel this hardest: the finest grids carry the sharpest terrain
  disagreement with a ~31 km source. Both now resolve through
  `soil_source_orography`; a source that declares no orography anywhere
  keeps the historical no-adjustment path. The direct door and the
  wizard chain are now bit-identical in TSK and TSLB on the case that
  exposed the gap, and the chain route's land TSLB moves by the
  expected -5.28..+3.70 K (rms 0.46 K).
- gpuwm's own wrfout warm-starts gpuwm's own child. The history writer
  published six fields of a surface source; ISLTYP, TMN and VEGFRA --
  required by `--child-surface-from`'s reader -- were never named, so
  `gpuwm downscale` refused gpuwm output as a child surface source.
  Fixed at the source: five rows join the history inventory behind the
  land-surface-scheme gate the soil rows already sit behind, IVGTYP and
  SEAICE ride along (an absent SEAICE was substituted with zeros, so
  ice-covered columns warm-started ice-free with nothing said), every
  string is transcribed from the pinned WRF v4.6.1 Registry, and
  ISOILWATER joins the global attributes stock WRF writes. A child's own
  history is itself a valid `--child-surface-from` file now.
- A streamed `[tiles]` domain consumes its lateral boundaries on the
  domain's BOUND Davies clock, bit-identical to the resident run.  Every
  production real-data root binds a DomainClock to its external LBC
  mirror (WRF's post-increment dtbc, `dt..T_bdy`); the streamed tile
  hook silently DROPPED that binding when it converted each buffer, so
  the buffers fell back to the retired `0..T-dt` path and every
  clock-bound streamed run forced its boundaries ONE TIMESTEP LATE, a
  constant phase error with no NaN and no refusal.  Measured on the
  offline child at
  t+15 min: 41 of 76 wrfout fields differed (W by 0.0043 m/s on a
  0.174 m/s field); at 438x350x49 with the clock bound, 9/9 carriers,
  max|d| 0.43.  The hook now rebinds the domain's clock onto every buffer
  it converts, and both external attach functions preserve an existing
  binding across re-attachment.  The unbound compatibility semantics
  (direct era5/gfs routes without a DomainClock) are proven unmoved, and
  nested children were never affected.
- A streamed domain's `refresh_state` lands the scattered `diag/*`
  members back on the physics driver, so routes that publish frames off
  the DomainState (`gpuwm.io.wrfout.state_frame`; the offline child)
  emit the OLR the tiles computed instead of the snapshot's zeros --
  measured: streamed OLR max 0.0 against 292.77 W/m2 resident, with the
  production StoreFrame route publishing the same store correctly the
  whole time.  A store diagnostic with no destination on the state is
  now refused rather than silently skipped.
- A tree whose `[tiles]` block cannot run is refused when it is read, not
  when it has been paid for, and the refusal now names the shape that is
  actually unsupported: a coupling edge with BOTH ends streamed, which no
  gate has driven. `mode = "on"` streams every grid unconditionally, so on
  a tree it forces exactly that shape on every edge; it is now refused by
  `build_experiment`, the one load every front door shares, naming the
  mechanism and the three ways out, rather than by the tile builder after
  a fetch and two preparations had already been paid for. `mode = "auto"`
  stays accepted over a tree: streamed children and streamed parents are
  both legal roads now, and if the pricing ever streams both ends of one
  edge the walk refuses at DECISION time with both grids named. The
  earlier admission refused every nested `mode = "on"` config for being
  nested and short-circuited every nest under `auto` to resident without
  asking the planner; both are gone, so the child road is reachable from
  the front doors that carry it and a nest's receipt line carries the
  planner's own arithmetic.
- A 1024x1024 streamed forecast now runs from a bare default command.
  `--stream-init auto` priced the resident road from the prepared cache's
  `state/*` manifest, the serialized prognostics and not a `DomainState`,
  so at 1024x1024x49 it read 4 716 B per column where the state alone
  measures 11 276.5: auto called a 6.45 GiB fit and the resident road died
  with 16 276 726 272 B allocated, with nothing re-routing after the
  refusal. Auto now prices this domain's own columns at the cost the whole
  route was measured at, 15 780 B per column at nz = 49, and takes the
  larger of that and the manifest term. The measured term predicts that
  refusal to 0.4%. 384x384 still prices at 2.17 GiB and still takes the
  resident road; the crossover on a 16 GB card sits near 768x768, and the
  receipt names which term set the price.
- The store-direct road runs the full-state health gate. It used to arm
  only the per-step stability fold -- u, w and theta finiteness plus the
  two Courant terms -- and declare the descriptor gate's per-field bounds
  out of reach. The domain is in the pinned host store, so the gate runs
  there instead, over the same fields under the same `rule_for_field`
  rules: moisture ranges, coupled-mass positivity, the geopotential and
  specific-volume limits, the soil bounds, and `gpu_integer_policy`'s
  exclusions and its refusal of an undeclared integer dtype. Armed at
  `initialized.d01` and `final.d01`, where the resident road runs it, and
  a failure is terminal on both. A store that turned out to be slab-height
  rather than domain-shaped is refused rather than reported as a pass. The
  executor's periodic full-state gate stays unarmed on cost, with the
  measured cost of one whole-store pass in the receipt beside the reason.
- Installing gpuwm no longer DOWNGRADES a user's wrf-rust. The `[render]`
  extra pinned `wrf-rust==0.2.35`, so `pip install 'gpuwm[render]'` (and
  `[all]`, `[all-cu12]`, `[all-cu13]`, which route through it) replaced a
  working newer core with the certified one -- reported from the field
  against 0.2.38. The extra now declares the window `>=0.2.35,<0.3`: the
  floor is what the products are certified against, the ceiling is where
  upstream is free to change the diagnostic surface. The four runtime
  checks that compared `== "0.2.35"` and refused anything else test the
  same window now, sourced from one module (`gpuwm/science_core.py`)
  instead of four copies of a string. Proven at both ends: the resolver
  leaves an existing 0.2.38 alone, and the suites pass on 0.2.35 and
  0.2.38 alike. Score files now report the version that ACTUALLY read the
  run beside the window that was required, rather than naming the floor
  regardless. `__version__` remains recorded and never gated: wrf-rust
  0.2.35 shipped the attribute reading `0.2.34`.
- `[tiles]` streaming no longer runs in silence when only a per-domain
  table asks for it. `streaming_receipt` was keyed on the TREE-WIDE
  `[tiles]` table, so a tree whose tree-wide mode is `off` and whose child
  carries `tiles = { mode = "on" }` streamed that grid and produced an
  EMPTY receipt: no line named the grid that tiled. The receipt is keyed
  on the per-domain DECISIONS now -- the tree-wide table is a default, not
  the answer. Where the tree-wide mode did not ask for what happened, the
  summary names the domains that overrode it. A tree the tree-wide table
  governs keeps its existing receipt bytes exactly.
- Four files reached the release line with CRLF line endings after a lane
  edited them on a Windows worktree, three of them product source
  (`gpuwm/core/streaming.py`, `gpuwm/experiment.py`, `gpuwm/runtime.py`).
  `.gitattributes` says `* -text`, so the flip entered the object database
  and every hash the product takes of those files would have disagreed
  with a Linux clone's. Normalised back to LF.

- The red-on-revert harness graded the tree it was written in rather than
  the tree it was run from: it opened with an absolute path to its lane
  worktree, so a copy carried onto any other checkout kept reverting and
  testing the ORIGINAL source and reporting PROVEN about it. It now
  derives the tree from its own location and refuses if `gpuwm` resolves
  outside it, and its restore leg round-trips bytes instead of writing
  back a CRLF copy of every LF file it touched.

- `gpuwm doctor`'s CUDA advice is installable. A support case on Ubuntu
  read `MISSING cupy (GPU runtime) ... Failed to find CUDA headers` and
  was offered `pip install 'gpuwm[gpu-cu13]'`: the wheel was already
  installed and no gpuwm extra has ever carried a CUDA toolkit, so pip
  reported success and the fault survived. That branch now routes on the
  import's own message -- the header/`CUDA_PATH`/`nvcc` signature gets the
  toolkit remedy (`conda install -c nvidia cuda-toolkit=<the major this
  box's driver serves>`, conda-forge and the NVIDIA pip wheels named
  beside it), a missing shared library keeps the wheel remedy, and a
  message carrying neither prints BOTH labelled by symptom rather than
  guessing. Separately, no remedy names a suffixed NVIDIA package any
  more: NVIDIA has deprecated `-cu12` as well as `-cu13` and both
  spellings install cleanly and supply nothing, so the affected lines now
  print `nvidia-cusolver`/`nvidia-cuda-runtime`/`nvidia-cuda-nvrtc` with
  the detected major in a version pin instead of in the package name.
  `docs/da-nowcast-quickstart.md` is corrected to what the code emits, and
  the tombstone rule now covers install lines in every tracked file under
  `docs/`, so a doc may warn about a tombstone but may not tell anyone to
  install one.

Battery:
- `tests/test_streaming_clock_arming.py` is registered on the GPU pytest
  shard. It shipped with its lane on no list, so until now the test
  existed and no leg ran it.
- The two front-door nesting legs are scripted GPU gates in
  `tools/battery/tiles_gates.txt`: each drives `gpuwm run` twice for the
  corruption screen, then digest-compares against an all-resident control
  of the same tree, and fails when the receipt says nothing streamed.
  Their case files carry no card budget and no machine-specific path.
- Five more files join the CPU stage-1 list, all found on no list at the
  cut: the CUDA-header tombstone and docs guards, the wrf-rust pin guard
  and its two consumers, and the gate that keeps the GPU shard list
  honest, itself an entry on nothing.
- `tests/test_native_wrf_distribution.py` joins
  `tools/battery/always_files.txt`, keeping its stage-1 entry, so every
  lane runs the RW-WPS staging boundary rather than only a cut. It stages
  the wheel by path and AST-walks the result, so it has no import edge to
  the modules whose imports it constrains: on this release's own range, 87
  product files touched, 289 tests selected, this suite not among them. It
  is the gate that burned 1.8.8 and 2.3.0, both times catchable in a lane
  in 18 seconds. `RELEASE_CHECKLIST.md` now makes stage-1 green at the
  stamped tip, before any tag, the first per-cut item.

## 2.2.1 (2026-08-13)

Fixed:
- ERA5 preparation no longer refuses the cache it just wrote. Whenever the
  soil-moisture floor fired, which is the common case on clipped ERA5, the
  prepared cache recorded a receipt the reader did not expect and the run
  stopped on its own output. ERA5 domains of 448x448 and larger could not
  be prepared at all.
- `gpuwm doctor` reports a missing CUDA header tree as a missing header
  tree. It compiles two kernels from a cold cache and says whether the
  CuPy wheel or the toolkit headers are the gap, then prints an external
  toolkit and `CUDA_PATH` as the remedy. The previous advice reinstalled
  the CuPy wheel, which carries the compiler and no headers, so it could
  not fix what it named.
- `gpuwm doctor` chooses CUDA library wheels by the CUDA major the box
  serves. A CUDA 13 box is no longer handed the `cu12` spellings, and
  never the `cu13` ones, which install cleanly and supply nothing.
- The Rust fetch cache stores one copy of each object instead of two. A
  whole-file fetch reaches the cache under two key shapes and stored two
  complete payloads at two paths; they now share one content-addressed
  copy, with the second key a hard link to it (a pointer file where the
  filesystem cannot hard-link). Measured on a 193.6 MB HRRR object: the
  cache fell from 3.00x to 2.00x the payload, and the fetch directory
  plus its cache from 4.00x to 3.00x. The fetch record and the HRRR
  manifest now carry a `dedup` block reporting bytes written, bytes
  deduplicated and reference entries.
- The ERA5 direct door accepts the config `gpuwm domain --source era5`
  writes. The adapter loaded the experiment through a path that split
  off `[fetch]` but refused `[case_data]`, so the wizard's own emission
  was refused by the wizard's own source adapter. The adapter now
  consumes `[case_data]` and uses its declarations to default the
  matching inputs: a declared geog_root or source orography needs no
  flag.
- Every front door that loads only the experiment portion of a config
  now validates and detaches `[case_data]` and `[static]` instead of
  refusing the file they sit in.
- A table that is present but not consumed by a loading path is reported
  as exactly that; the "does not have a table" refusal is reserved for
  tables that are genuinely unknown.
- An ERA5 forcing series that begins before the experiment start hour is
  trimmed loudly instead of refused. A series that does not contain the
  start hour refuses by naming the decoded window, the expected hour,
  and both fixes.
- The ERA5 refusals for a missing source terrain and a missing geography
  root name the exact missing input and every accepted remedy.
- The composed Thompson + Shin-Hong suite no longer dies with a
  non-finite TKE error on real forecast cases. Shin-Hong's SGS TKE
  chain is a diagnostic no tendency reads, and its transcribed WRF
  arithmetic divides by quantities that legitimately reach zero; a
  non-finite confined to that diagnostic is now repaired to the
  scheme's own cold-start floor with a one-line advisory, while a
  non-finite in any consumed output still stops the run exactly as
  before.
- Releases no longer fail on PyPI index propagation lag: the final
  promotion check now retries its PyPI read with the same bounded
  backoff the publish job already uses.

New:
- A wizard-emits/adapter-accepts round-trip gate: for each source (ERA5,
  GFS, HRRR), the config `gpuwm domain` writes must be accepted by that
  source's own adapter through config validation. It runs in the
  stage-1 battery on every cut.

## 2.2.0 (2026-08-13)

New:
- A fast-fix lane, for shipping a correctness fix in under 90 minutes.
  `tools/battery/fastfix.py` takes the range a fix spans and prints the
  test files the change can break, cheapest first, with the reason each
  was selected and an estimated wall time. Selection is by direct
  imports: measured against the alternative, a transitive walk selects
  three quarters of the test estate and finds nothing extra.
  `tools/battery/FASTFIX.md` documents the procedure, including what it
  does not yet cover.
- Repository-scanning gates run unconditionally, from
  `tools/battery/always_files.txt`. Citation checkers, receipt
  regenerators and line-ending gates read the tree instead of importing
  it, so no selector can reach them; running them every time is the only
  correct answer.
- A GPU test leg, `tools/battery/gpu_shard_files.txt`. The dynamics
  core had no automated gate on any list: sixth-order diffusion and its
  boundary faces, both Smagorinsky closures, the open lateral boundary,
  the acoustic solver, the mapped mass closure and the turbulence budget
  all skip on a machine without a card. First run on a 16 GB sm_120
  card: 314 passed, 121 seconds. The file that checks a gpuwm history
  frame with an independent WRF reader now runs too, 18 passed, where it
  had never run anywhere.
- The projected-source horizontal mapping runs in the packaged Rust
  bridge. `gpuwm_indexed_interp_f32` takes the donor as an exact integer
  pair plus its FP32 fraction, which is the shape a projected source
  needs and the reason this route had no Rust boundary at all: it selects
  its donor in FP64, and a coordinate just below an integer can advance
  that donor once it is rounded to FP32. The 4x4 stencil and both `oned`
  sweeps are fused per point across every core, with no intermediate
  array. Measured on a real HRRR CONUS window against the real les.km3
  d02 child (1059x1799 source, 608x488 target, 50 levels): the complete
  `interpolate_hrrr_to_lambert` apply set falls from 32.37 s to 0.301 s
  on 32 workers and to 2.334 s on one, and a single 3-D parabolic apply
  from 3.764 s to 0.027 s.
- The port is pinned to the NumPy operator it replaces, not to a
  tolerance. On that same real window and child, 136,187,136 output
  elements compare bit-identical across every method, both ranks, real
  values and an adversarial arm carrying zeros, negative zeros, the
  1e-20 sentinel itself, subnormal operands, NaN donors and overflowing
  products. `tests/test_hrrr_projected_operator.py` gates the same
  comparison on the raw uint32 view, so a signed zero cannot hide in it.
- The bridge now carries two `oned` missing-value predicates and says
  which authority each serves. The regular-grid entry keeps the CUDA one,
  which flushes subnormal operands; the projected entry uses the host
  IEEE one, which flushes only the product, because that is what the
  NumPy operator on this route does and reproducing it is the whole
  point. They differ only when an operand is subnormal or infinite and
  the host product is not, and both the divergence and the test that
  holds the line are named in `tools/grib1_bridge/src/lib.rs`.
- An install whose bridge predates the new entry keeps the NumPy mirror
  and is told so in one line, once, naming the rebuild. `gpuwm doctor`
  reports the same gap on the library itself rather than calling it
  simply verified, because the ABI integer cannot express an addition.
  The HRRR mapping report records `projected_horizontal_operator`, so a
  receipt can no longer say "cpu" for two implementations an order of
  magnitude apart.
- Legacy RRTMG shortwave runs its FP32 arithmetic 1.62x faster on the
  GPU and returns the same bits. The unit's 679 add/sub/mul sites carried
  a subnormal countermeasure that emulated every FP32 operation through
  FP64: decode a possibly subnormal operand from its bits, do the work in
  double, re-encode the result, about twenty instructions where one would
  do, at FP64's 1:64 rate on GeForce. They now compile to `add.rn.f32`,
  `sub.rn.f32` and `mul.rn.f32` written as inline PTX without the `.ftz`
  modifier, which is the same correctly-rounded binary32 operation with
  gradual underflow that the emulation was computing, and the same idiom
  `dycore.cu`, `acoustic.cu` and `mynn_pbl.cu` already use. Measured on
  an RTX 5070 Ti (sm_120) over 8,450 real fixture columns at 50 layers,
  shortwave kernel time per call falls 609.98 ms to 376.40 ms: the
  dominant `rsw_spcvmc_gpt_b` 511.07 to 371.70, `rsw_taumol_b` 43.38 to
  1.31, `rsw_cldprmc_b` 25.10 to 1.13, `rsw_spc_accum_b` 29.01 to 1.78.
  Every output field is byte-identical to the emulation at 169, 1,690,
  8,450 and 16,900 columns.
- The armor is retained at every one of the 679 sites, not traded away.
  A witness build counts, per source line, how many times each site runs,
  how often it sees or makes a subnormal, and whether the PTX and FP64
  arms ever disagree. Over a batch spanning a full diurnal `coszen`
  ladder from 0.02 to 0.99: 304,204,382 macro calls, 2,166,356 with a
  subnormal operand, 1,659,113 producing a subnormal result, and zero
  mismatches between the two arms. The subnormals are real and they are
  concentrated exactly where the file said they would be, in the
  direct-beam transmittance chain and what consumes it: `rsw_reftra`,
  `rsw_vrtqdr`, `rsw_spcvmc_body` and `rsw_spc_accum_body` hold every one
  of them and carry 79.7% of all calls, while `setcoef`, `taumol`,
  `cldprmc`, `sfluxzen`, `inatm` and `post` witness none.
- `tests/test_rrtmg_sw_rn_identity.py` proves on the CPU that the FP64
  emulation is ordinary IEEE binary32 round-to-nearest, over structured
  operand classes, an exhaustive sweep of the low-order subnormal range,
  a mantissa sweep that lands products on subnormal rounding ties, and a
  Sterbenz cancellation sweep. Two red-on-revert arms drop either half of
  the armor, the operand decode or the result encode, and show the same
  check going red.

Fixed:
- The early-render and time-to-first-plot suite ran none of its tests.
  A module-level skip for an optional package sat two thirds of the way
  down the file, and a module-level skip applies to the whole module, so
  the 21 tests defined above it were skipped as well. The suite is on
  the per-cut list and had reported success while executing nothing
  since the feature shipped. It now collects 36 tests and passes them.
- Two Noah-MP kernel suites had the same defect and four more tests
  that had never executed on any machine, including one whose own
  documentation said it needed no GPU and ran everywhere. Their
  device-free checks now live where they run.
- A new gate refuses the whole class: a module-level skip below a test
  definition fails, naming the file, the line, and how many tests it
  silently takes with it.
- The wheel-contents suite no longer skips when its build dependency is
  absent. Five of its nine assertions were skipping in the release
  environment, and those five are the ones that check that declared data
  files actually reach the wheel. An absent dependency is now a failure
  with a one-line remedy.
- The per-merge test cost falls by about 800 seconds with no assertion
  removed or weakened. The health-descriptor ceiling gate keeps running
  on every merge, over the neighbourhood of the measured peak, and
  re-derives that peak rather than reading it back; the exhaustive sweep
  moves to the per-cut tier. The GPU-marker gate reuses one collection
  instead of spawning several hundred. The registry citation checker
  indexes the tree once instead of walking it per citation.
- Seven test files whose subject or replacement was re-verified are
  retired. Three further candidates are kept and the reasons recorded:
  one rested on a duplicate that does not exist.
- Degrading to the Python fetch transport is no longer silent. The
  missing-backbone branch of the engine selection returned the stdlib
  transport with no message at all, which is the branch every install
  without the bridges bundle takes; it is also the expensive one,
  because that transport has no whole-file mode, so a `--mode
  full-file` request quietly became `.idx` subsetting. Selection now
  emits one `warning:` line naming the measured tax (560 s for one
  419 MB HRRR file against 27-35 s taken whole, roughly 16x) and the
  fix, before any bytes move. Nothing is refused.
- The line is said at selection time rather than by one command, so the
  GFS full-file command, the streamer's preflight and every library
  caller of the front door get it; the HRRR command's own near-duplicate
  is gone. `select_fetch_engine` returns the reason beside the engine;
  `resolve_fetch_engine` keeps its two-value shape for the callers that
  only want the pair.
- The fetch manifest records `engine_selection` beside `engine` --
  `rust`, `python-requested`, or `python-fallback` -- so a receipt
  distinguishes a transport somebody chose from one an install
  inherited. It is not called `transport`, because `--transport`
  already names the host on this front door.
- `gpuwm doctor` prices the missing backbone instead of only naming it:
  the check line carries the same measurement and the receipt field to
  look for.
- The shortwave unit no longer depends on a compiler flag to keep its
  subnormals. `tools/ftz_receipt` measures that every compiler-emitted
  FP32 instruction flushes under CuPy's appended `-ftz=true`, and that
  inline PTX without `.ftz` does not. The module currently escapes the
  append by compiling through `compile_using_nvrtc` rather than
  `RawModule`, and that escape is load-bearing but incidental: it was
  adopted because NVRTC 13 started rejecting the duplicate flag, and a
  build-route change would silently reintroduce the flush. Writing the
  instructions as PTX makes the subnormal contract a property of the
  source instead of a property of the option tuple.
- Streaming works from the product front doors. `[tiles]` had only ever
  run through the tilestream engineering route, and every path a user
  actually takes failed somewhere between admission and the first output
  frame. The shipped default physics suite failed first and hardest: its
  radiation is legacy RRTMG, a plain class whose constructor requires a
  start time and per-column latitude and longitude, and the per-buffer
  twin builder could rebuild only dataclasses and empty constructors. So
  every `[tiles]` forecast of the default suite raised StreamingRefused
  inside TiledRun before taking a step, and the docstring that should
  have caught it asserted legacy RRTMG was a dataclass. The builder now
  carries explicit constructor recipes, audited two ways: a recipe must
  name every constructor parameter, and its twin must agree with the
  domain's adapter on every non-volatile scalar. An adapter with no
  recipe is still refused. Measured on a 16 GB card through the prepared
  single-domain runner, 550x550x49 at 3 km with the default suite: 240
  streamed steps, one forecast hour, health green, no NaN, peak 15828
  MiB.
- The tile-buffer warm-up handed the radiation an atmosphere that did not
  exist. Buffers were built on `make_vertical_coord`'s default stretch
  while the domain's own eta table was imposed on top, so their 3-D base
  state described one atmosphere and their pressure another, and then a
  throwaway step integrated that against the domain's real lateral
  forcing. RTE+RRTMGP's gas tables refused the result at 120.3 K to
  407.3 K, and legacy RRTMG, which has no equivalent validator,
  integrated the same field in silence. The `[160, 355] K` range is the
  tables' own and is unchanged. Buffers are built on the domain's eta
  table now, and the warm-up step is gone: it existed only to allocate
  two lazily-created carriers, which are allocated directly instead.
  Verified with RTE+RRTMGP at 550x550x49, `mode = "on"`: 240 streamed
  steps, health green.
- `[tiles] mode = "on"` was refused at admission by the routes that
  support it. The refusal was written when the prepared routes passed no
  streamed-domain builder; they have passed one since, but the refusal
  outlived the wiring and went on rejecting the one mode that asks for
  streaming unconditionally, with a message asserting the route wired no
  builder while `mode = "auto"` streamed through that very builder.
  `gpuwm go` mirrored it before the download, so the front door most
  users type was the last place that rejected streaming. `gpuwm run`
  still refuses, because it reads `[tiles]` at no point.
- A streamed forecast's final health gate and canonical digest reported
  the initial condition. Under a host store the DomainState is the
  snapshot that filled the store, and neither the whole-field health
  validator nor the trajectory digest has a tile-interior form, so both
  answered for the analysis on a run that had integrated for an hour.
  The runner copies the domain back from the store before they read it,
  and the receipt records how many carriers moved.
- A streamed run could not publish reflectivity. `refl_10cm` is rebuilt
  scratch, correctly absent from the carrier manifest and therefore from
  the store, so each tile computed its own window and the transport had
  nowhere to join them. The sweep refused the first due frame, which on
  an hourly cadence is an hour into a healthy forecast. The slot is
  primed on the domain and on every buffer before the store is sized.
- Legacy RRTMG's ozone grid is carried by a checkpoint and no longer by a
  sweep. The adapter recomputes it from the climatology on every
  radiation call and reads it only from a child domain, which streaming
  refuses outright, so a warmed buffer had it while the prepared domain
  did not and the inventories differed by exactly that key.
- The prepare stage holds one forcing time resident instead of two. The
  loop retained the start time's state for its whole duration while every
  later time was built underneath it. Measured at 800x800x49 with
  Morrison, a 6-hour GFS chain: resident 14.67 GiB to 7.66 GiB, peak
  envelope 23.92 GiB to 15.86 GiB. A 24-hour chain and any ERA5 chain
  still price above 16 GiB; the binding term left is
  `domain_boundary_snapshot`'s full-domain host copy.
- The HRRR route can configure `[tiles]`. It had no way to, and said
  nothing: the block was silently dropped and the run went resident.
  Carried as a runner flag rather than rendered into the published
  authority, which is hash-bound, so that a bundle prepared streamed can
  still be re-run resident. Multi-domain plans that cannot stream are
  refused at config validation, before the fetch, naming every nest and
  the grid that can stream.
- The prepared front door renders its first committed frame. It published
  frames with nothing watching, so reaching a first plot needed an
  external watcher polling `GPUWM_WRITE_COMPLETE`.
- Legacy RRTMG compiles its shortwave CUDA engine once per process rather
  than once per adapter. Streaming builds one adapter per tile buffer, so
  an uncached engine multiplied both the NVRTC compile and the resident
  device tables that streaming exists to save.
- A streamed forecast publishes `OLR`. It had been dropped from every
  frame a streamed run wrote, silently: the field is produced by
  radiation, published to output and read back by nothing, so it is not
  a carrier, so the transport neither gathered nor scattered it and the
  store had nowhere to join the tiles' windows. The frame was written
  short and the run still reported forecast validity PASS with health
  status bits clear, so 73 of a resident run's 74 variables was
  indistinguishable from success. The transport now carries the
  output-only diagnostics the same way it carries reflectivity, which
  leaves the trajectory bit-identical because nothing reads them back,
  and `XKMH`/`XKHH` are carried with them when the eddy-viscosity
  diagnostic is on.
- A streamed frame that cannot publish a field a resident run writes is
  refused instead of written short. The refusal was documented but never
  implemented: the writer took the short field list as its schema and
  the file validated. It now names the missing rows and the one-line
  remedy.
- Carrier provenance on a streamed frame states the radiation that
  actually ran. `GLW` and `SWDOWN` were byte-identical to a resident run
  while the same file's provenance attributes read `unwritten` with no
  update time, because the export path read the driver on the domain
  state and a streamed domain's radiation runs on the tile buffers. The
  ledger itself was always current; only the reader was stale.
- `gpuwm go` and `gpuwm check` price a streamed run as streamed. Every
  memory term described a domain resident in VRAM, so a configuration
  with `[tiles]` was refused by the default front door on the strength of
  a number describing a run that was not going to happen -- and the
  configuration refused was the one streaming exists to make possible.
  The forecast term is now the streamed envelope, the tile working set
  and the pinned store, taken from the same measured model the run
  attaches with. Preprocessing is not streamed and keeps its own term.
- `gpuwm check` no longer tells users that `[tiles] mode = "on"` is
  refused by the forecast routes. It has not been since the routes were
  wired, and the same advisory also claimed the estimator had no model
  of a streamed domain, which is no longer true either.
- `UP_HELI_MAX` is bit-identical under tiling. It was the one field of 75
  a streamed run did not reproduce exactly, by one part in ten billion at
  a single cell -- and that cell sat on a tile seam, which is what named
  the cause. The diagnostic is evaluated at the end of a step from a
  stencil two cells wide, at the one moment when a tile's outermost halo
  has already been consumed by that step, and the halo carried exactly
  two cells of spare margin. A run that emits the diagnostic now widens
  its halo by the diagnostic's own reach, with nothing to configure. The
  forecast is unchanged either way; the tile windows grow by four cells
  on each axis.

## 2.1.1 (2026-08-13)

Fixed:
- The release verifier proves every bundled library through the ABI
  symbol that library itself declares. It used to ask every library
  for the CPU preprocessing library's symbol, which the region global
  dealiasing library does not export, and it had no branch at all for
  a vendored artifact, whose freshness is proved by its contract
  marker rather than by a source stamp. Either defect refuses a
  correct set of release artifacts, and both were reached only while
  preparing 2.1.0, so 2.1.0 was tagged and never published. 2.1.1 is
  the published form of that release and carries the same contents.
  A library artifact that declares no ABI handshake is now refused by
  name instead of being probed with another library's, one table
  declares each handshake for both the verifier and the release
  workflow, and a local probe loads every such library before a tag
  exists.

## 2.1.0 (2026-08-13)

New:
- Correlation-coefficient QC is available at the radar grid-build front
  door as `--cc-qc`, off by default. Its rule is per moment. In
  reflectivity a gate is dropped only where RhoHV and reflectivity are
  both low, so hail cores, the melting layer and tornadic debris survive
  as echo. In velocity there is no reflectivity shield: a low-RhoHV gate
  loses its velocity at every reflectivity, because a scatterer that does
  not move with the air is not a wind observation. Off is byte-identical
  to a build from before the flag existed.
- The debris-signature fringe keeps its velocity. A velocity gate
  survives the mask where RhoHV sits between the debris floor and the
  velocity threshold, reflectivity sits between 30 dBZ and the shield,
  and the gate lies inside the neighbourhood of a clustered velocity
  couplet in the same sweep. All five conditions must hold. Low RhoHV
  alone never qualifies, and the debris core at and above the shield is
  untouched. The exemption is on with `--cc-qc`;
  `--cc-no-tds-fringe-exempt` restores the strict rule.
- The mask's account is per radar, not per file. Every exempted gate is
  counted, and so is every candidate each criterion turned away, so a
  reader can reconstruct the strict-rule total from the receipt without
  rerunning anything. The `cc_qc` provenance key appears only when at
  least one radar's mask ran.
- Radar velocity dealiasing has a new default engine. `--dealias` runs
  the vendored `region-global-dealias` crate
  (`tools/region_global_dealias`), a Rust port of Py-ART's
  `dealias_region_based`, through a C ABI, with its refinement pass on.
  Verified fold-for-fold identical to Py-ART 2.2.5 across 1,355,617
  velocity gates of a real Level-II volume. There it keeps 3,894 more
  velocity cells than the previous `vad-region` engine, which rejects
  both gates of that volume's strongest couplet, and the dealias stage
  falls from 18.3 seconds to 62 ms. `--dealias-engine vad-region` still
  selects the old solver, `--no-dealias-refinement` turns the refinement
  off, and the engine that ran is recorded in `provenance.dealias`.
- Dealiased velocities are bounded by physics on both engines. A gate
  unfolded past `max_speed_ms` (75 m/s) is rejected as
  `speed_out_of_range` and counted, never clamped and never passed: 115
  gates of that volume, worst 114.5 m/s, couplet untouched.
  `gpuwm fetch-bridges` stages the library the default engine needs, and
  `gpuwm doctor` blocks without it.
- `tools/dealias_engine_compare.py` runs both engines, the masking-only
  and bound-lifted arms over one volume, reporting each arm's wall clock,
  gate account, gridded `vr` field, per-cell differences, and what each
  did to the strongest couplet in the data.
- Radial-velocity dealiasing runs about 21x faster on a real volume, and
  decides exactly what it decided before. Its coarse VAD seed search --
  1.13 million wrapped-cost cosines per range band, 1214 bands in a
  KTLX volume -- moved into the existing preprocessing cdylib as
  `gpuwm_dealias_coarse_cost_f64`, which rides a plane-rotation
  recurrence along the speed axis instead of calling a cosine per
  candidate, and evaluates a whole sweep's bands in parallel. Measured
  back to back on one KTLX 2013-05-20 volume: the band-fit stage 25.91 s
  to 1.21 s, the dealiasing cost of the volume 29.97 s to 2.63 s, the
  whole observation front door 32.05 s to 5.00 s, with the gridded
  output sha256 and every dealias count identical to the NumPy path and
  to the unported tree. The kernel never ranks candidates: it shortlists
  them, `_coarse_shortlisted` proves from its own error bound that no
  excluded candidate can be in the answer, and the surviving shortlist is
  ranked by the original NumPy expression -- widening to the exhaustive
  search when the proof cannot be made. Without the native library the
  pure-NumPy search runs as before, and the sweep's stats say which path
  it took.
- Wall-clock instrumentation reaches the observation, verification and
  I/O paths, which had none. `GPUWM_PERF_TIMING=1` turns on a per-stage
  clock inside the dealiasing pipeline, the paired-frame reader and
  comparison, the wrfout frame writer, the restart writer, the static
  builder and the HRRR bridge verification; `GPUWM_PERF_TIMING_OUT=<path>`
  writes the receipt at exit. Off by default, and off costs a
  module-global read per probe. Nested stages are subtracted from their
  parent so a receipt says where the wall clock actually sat.
- `tools/perf_obs_timing.py` prices the observation front door on one real
  radar volume, gridding it with dealiasing off and on so the capability's
  cost is a difference rather than an assertion.
- The paired-frame judge is a Rust program. `rw_fieldcmp` prints the
  metric table from two directories of history frames: per-field
  quantiles and their differences, accumulation sums and their ratio,
  composite coverage at each threshold. `--threads` sizes the pool,
  `--table` and `--json` write it out, and arm labels, fields, composite
  and thresholds all come from the command line. On a paired
  verification set: 2.6 times sooner, all 529 table lines identical to
  the Python judge's.
- The paired-run judge is born Rust. `rw_runscore`, in the same crate as
  the frame judge, scores two run directories against each other over a
  registered metric set: pooled low-pass state RMSE and pooled boundary
  increment error per domain and per field, the first-object timing
  difference per domain, and the mean neighbourhood skill distance on
  whichever domains carry it. The ladder, the spacings, the field list,
  every threshold and every metric-key spelling arrive on the command
  line; the defaults name WRF-convention variables and generic keys, and
  no campaign, case or physics suite is baked in.
- It walks the ladder once. The Python scorer reads every frame four
  times per field, once as the current frame of its own interval and once
  as the previous frame of the next, on each arm, reopening the file every
  time: 672 decodes of the state fields and 720 file opens for a
  four-domain seven-field pair. The Rust one decodes each frame's fields
  once from one open handle, reduces each on the spot to the scored
  interior sum and the outer ring of cells the next interval needs, drops
  the field, and runs the frames in parallel. Same pair: 440 decodes, 56
  opens.
- Measured on a real two-arm WRF pair from the pinned node-1 build, seven
  frames per domain per arm staged on the registered four-domain ladder,
  6.0 GB of history, cache warm: all 61 distances are bit-identical to the
  Python scorer, and the comparison costs 16.7 s against the reference's
  111 s. Three independent timing rounds put it between 6.5x and 7.2x.
- `tools/verify_runscore_parity.py` runs both scorers on one pair and
  reports every metric that differs, with the gap in representable
  doubles rather than a relative epsilon that hides a wrong answer at
  small magnitudes. It times both in the same pass, so the parity claim
  and the speed claim come from the same bytes, and it records how the
  pair was staged. The receipt for the run above is
  `evidence/verify-instrument/paired-run-score-parity.json`.
- What the Rust scorer does not replace: the campaign door also re-hashes
  every history frame against its run artifact before it scores anything.
  That leg is still the Python one's. It is cheaper than it looks on a
  warm cache, about 2 s of the 122 s door on this pair, so the metric
  arithmetic is where the time went and where it was taken from.
- A stored hour's plots render at the same time instead of one after
  another, in a private pool the met kernels' own parallel work nests
  inside. The width is the smallest of the box's physical cores, the
  hour's product count, and what free memory affords at half of what is
  free; a width that memory rather than cores decided says so and names
  the override. `RUSTWX_BATCH_RENDER_THREADS` sets it for this loop,
  `RUSTWX_RENDER_THREADS` for the renderer as a whole. Measured on a
  337-plot two-frame gallery from a local WRF run: 85 s to 9 s on a
  16-core box, every PNG byte-identical to the serial render.
- Both console streams report in catalog order rather than completion
  order, at every width. The progress events, the summary's output list
  and `rw_wrfbatch`'s stdout read as the serial pass did, and so now do
  the advisories a render writes to stderr about itself. Cancellation
  still stops new plots from starting and lets in-flight ones finish,
  now up to one per worker rather than one in total.
- A release gate changed: the battery now tests the Rust workspaces this
  project ships, `tools/rustwx` and `tools/grib1_bridge`, which no cut
  had ever compiled. `tools/battery/cargo_gates.txt` lists them per
  package with a shard and a test floor;
  `tools/battery/run_cargo_gates.py` runs the leg offline under an
  external `CARGO_TARGET_DIR`, reporting COULD NOT RUN rather than a
  pass when cargo is absent.
  `tests/test_cargo_gate_manifest.py` keeps list and members in step, so
  a new crate cannot land ungated.
- The LETKF analysis times itself. `gpuwm.da.letkf.analyze` records its
  setup, its chunk loop and its finish, and splits the chunk loop at the
  phase boundary into the localisation weighting (evaluated at every
  gridpoint) and the batched transform (at the active ones only). The
  five numbers reach every cycle report through the `filter` block of
  `assimilate_radar_grid`'s provenance, beside the host-to-device staging
  and unstaging the device arm pays. There was no wall clock anywhere in
  `gpuwm/da/` before this, so a report could say the analysis was most of
  a DA leg and not say what part of the analysis that was.
- `tools/da_solve_ab.py` A/Bs one LETKF analysis between solve devices on
  byte-identical inputs. Each arm runs in a fresh subprocess, the bundle's
  every input file is re-digested before either arm reads it, and the
  receipt carries per-stage wall clock, per-field increment agreement and
  the observation counts that say the two arms saw the same analysis. It
  judges as well as measures: bitwise-identical increments across two
  devices are reported as a device arm that silently ran the host path,
  not as agreement, and a bundle built by the harness rather than dumped
  from a real leg is labelled synthetic in the verdict.
- `tools/da_cycle_prepared.py --dump-analysis-bundle DIR` copies each
  leg's analysis inputs -- the staged member checkpoints, that leg's
  observation file, the history file its observations were gridded onto --
  into a bundle the A/B replays. A leg whose analysis needs a forward
  operator that files alone cannot rebuild says so and dumps nothing.
- MEASURED, on a synthetic bundle at a radar-sparse shape (86,400
  gridpoints, 6.5% of them active, 10 members, two radars, numpy):
  the localisation weighting is 19.9 s of a 22.6 s chunk loop and the
  batched transform is 2.2 s. The phase that scales with the WHOLE
  domain dominates the phase that scales with the observed part, so a
  faster batched eigensolver -- in any language -- is bidding for 9% of
  this analysis. `evidence/da-solve-ab/host-baseline-sparse.json`.
- MEASURED, same bundle, host against cuda on one RTX 5090, medians of
  three: the analysis is 16.63 s on the host arm and 2.17 s on the device
  arm. The device wins the localisation weighting 14.41 to 0.33 and the
  transform 1.65 to 0.84, and pays 0.27 s to stage. Sharing the card with
  a GPU load taxes the device analysis 2x, to 4.29 s, and leaves the host
  analysis unmoved. The arms agree to 2.6e-15 relative, worst field, with
  different eigensolvers and identical observation counts.
  `evidence/da-solve-ab/README.md`.

Fixed:
- Verifying an HRRR bridge publication hashes its payloads concurrently.
  The check still covers every payload the manifest lists, so what it
  proves is unchanged, but the reads no longer run one file at a time.
  Measured here at 1.18 GB/s serial against 3.51 GB/s on eight workers.
  A single lead's snapshot load re-verified the whole sealed
  publication -- all 25 leads on a 24-hour case -- and that read is what
  made two of six cases take 267 and 280 seconds in a stage the other
  four finished in 10 to 12.
- The Rust judge's summation reproduced the reference's pairwise tree but
  not the buffering around it. The reference's reduction hands its
  summation kernel 8192 elements at a time and adds each buffer's total
  into a running scalar, so an array longer than one buffer is summed as
  a sequence of trees rather than as one tree, and past about a million
  elements the two arrangements differ in the last bit or two. Every
  paired-run metric works at that size. Both `rw_fieldcmp` and
  `rw_runscore` now buffer, in single precision as well as double, with
  the block totals computed concurrently and added in order so the answer
  does not depend on the worker count. The buffer length is a settable
  default in the reference stack, which is recorded where the constant
  lives: a caller that changes it moves the last bit of every large sum
  on both sides.
- Selecting the outer frame of cells from a stack of planes leaves the
  reference holding a column-major array, and its sums walk memory rather
  than logic. The ring is packed cell-first to match. Before this was
  traced, three boundary metrics sat one bit away from the reference with
  no explanation.
- The live bundle smoke counted only the binaries it staged. `fetch_bundle`
  stages every pinned binary and every pinned map asset, so the assertion
  read 9 against a true 47 on the published v2.0.0 bundle. It is derived
  from the manifest now. Nothing offline had ever packed a bundle carrying
  assets, which is why it survived; a hermetic test covers that half now.
  The same test skipped two different states through one branch: a tree
  that has never been cut still skips, a cut tree missing a bundle for a
  supported platform fails.
- Three `rustwx-products` tests drove one process-global disclosure with
  nothing serializing them, so one test's reset landed between another's
  set and its assertion about one run in five. The new cargo leg is what
  made it visible. The product was never involved.
- The whole-report doctor paste fixture did not know
  `tools/region_global_dealias`, so the new region-dealias check believed
  a source checkout was a wheel and printed a `git clone` the paste
  contract refuses.
- The release verifier could not reason about a vendored artifact. It
  required every binary in a bridge bundle to carry this release's
  source-revision stamp, but the region-global dealias library is a
  verbatim upstream crate frozen at a recorded commit, deliberately
  unstamped, and proved instead by the contract marker naming the ABI it
  speaks. The packer already knew that; the verifier did not, and would
  have refused a correct bundle. It asks a vendored artifact for its
  marker now, and refuses one carrying the wrong marker or none, so the
  staleness question is still asked of every binary. Its receipt names
  which binaries each proof covered instead of asserting one blanket
  claim, so the receipt schema is `gpuwm-release-artifact-proof-v2`.
  `tests/test_verify_release_artifacts.py` and the release-snapshot
  suites joined the battery's stage-1 list, which had never carried the
  machinery a cut itself runs.

## 2.0.0 (2026-08-12)

New:
- A streamed sweep's transfers run beside its compute. Each tile buffer
  carries a copy-in and a copy-out stream next to its compute stream,
  ordered by the event chain `tilestream/overlap.py` derives from the
  ring plan. When nothing armed on the run needs a hard sweep barrier,
  the end-of-sweep device synchronization is deferred too: the next
  step's gathers chain on the previous step's scatter events and the
  pipeline stays primed across steps. Anything that reads the store
  drains first, through `TiledRun.store` or `TiledRun.drain()`.
  `overlap="on"` is the default and changes no arithmetic; `"off"`
  keeps the single-stream loop as the reference; `"unchained"` drops
  the chain, wrong on purpose, as the negative control.
- `tilestream/test_overlap.py` gates the event chain without a card. The
  sweep is simulated under the driver's own dependencies and driven by
  adversarial schedules; the full chain must match the monolithic
  reference under every adversary, each wait class cut must fire the
  construction-time checker and either move the digest or be proven
  transitively implied, and the unchained control must move the digest.
  Listed in `tools/battery/tiles_gates.txt` as a cpu gate.
- A domain too large for the card runs as a streamed forecast. The state
  lives in pinned host RAM and cycles through the GPU one tile at a time,
  under the `[tiles]` table. `mode = "auto"` streams only when the domain
  does not fit, and unknown keys in the table are refused by name.
- `docs/public/TILES.md` documents the streamed run. `gpuwm stream` and
  `docs/public/STREAMING.md` remain the HRRR cycle-following feature. Both
  pages, and both `stream` help texts, open by naming the other: the two
  share no configuration and no code path.
- Health is reduced per tile. `health_partial_tile` folds a tile's nan,
  w_max and CFL record after its step and before its interior is
  scattered, so the run loop's gate reduces over the host store instead of
  over a state the sweep never writes.
- `tools/battery/tiles_gates.txt` enumerates the four `[tiles]` gates, the
  shard each runs on, the environment each one's real acceptance needs, and
  the measured card floor for the graph section. Two gates have a rung or
  geometry selector whose default is the weaker setting, so each is listed
  twice. `tests/test_tiles_gate_manifest.py` gates the list.
- Stage 1 gains seven CPU-only suites from this work, each with its reason
  inline: the `[tiles]` option surface, spawn-at-trigger under `[tiles]`,
  the gate list, the gates' green-on-nothing preconditions, the GLW
  declaration contract, the physics allocation inventory, and the
  distribution's import closure.
- `tilestream/` carries the harness, its engineering records and the
  evidence logs behind every figure stated here. None of it is product
  surface and no gate covers it.
- Specified (externally forced) lateral boundaries run through the
  multi-GPU decomposition. `plan_split` takes per-axis periodicity; a
  non-periodic axis clamps edge ranks at the domain, so the real boundary
  sits on the rank's own array edge, and no wrap seam is exchanged there.
- Each forced rank attaches the domain's `LateralBoundaries` windowed to
  its array: the domain's own tables on true edges, inert tables on
  interior seams. The halo is padded by `max(spec_zone, relax_zone)` so
  seam-side boundary fiction stays inside the throwaway ring.
- Forced decompositions that cannot be right are refused by name: forcing
  missing on a specified config, forcing on a periodic config, nested
  forcing, a quarantine-defeating halo, Davies zone mismatch between the
  forcing and the config, and ranks narrower than the relaxation frame.
- The gate battery gains a FORCED rung alongside the periodic one:
  `tilestream.test_forced_gate` runs the specified case at 1, 2 and 4
  ranks against the resident digest, with a poison-seam control that must
  match and a scaled-edge control that must differ. Stage 1 gains
  `tests/test_multigpu_specified_bc.py`, the CPU-hermetic geometry,
  windowing, corner and refusal suite.
- Every radiative field a land-surface scheme reads carries a source and a
  last-producer time, and the check runs immediately before the scheme
  consumes it. Sources are `radiation_scheme`, `declared_constant`,
  `external_array`, `analytic_geometry`, `wrf_compat_zero` and `unwritten`.
  The consumer matrix states what each scheme reads: Noah GLW+SWDOWN, RUC
  GLW+GSW, Noah-MP GLW+SWDOWN+COSZEN, no land surface nothing. A scheme not
  in the matrix refuses rather than defaulting to requiring nothing.
- `surface_radiation_policy` defaults to `"required"` for every run. The
  escape `"wrf_compat_zero"` consumes unsourced carriers at their
  allocation fill, is never selected automatically, labels every carrier it
  admits, appears in the run receipt, and is refused on a resume that
  changed it. It is an experimental forcing, not a valid configuration for
  a real case. A declared constant GLW is an experimental forcing on the
  same terms, not a statement that radiation is available.
- The check counts the consumer's cells before refusing, with the same
  land and sea-ice predicate the schemes dispatch on. A domain with zero
  land and zero ice cells feeds its land-surface scheme nothing, so an
  all-water idealised run with radiation off is admitted. One land or
  ice cell restores the full refusal, and a state whose footprint cannot
  be read runs the check in full.
- Per-carrier provenance in the run receipt and the output metadata.
  The prepared runner's `report.json` carries one row per carrier with
  its source, last producer time and a representative value, and every
  wrfout file carries `GPUWM_SURFACE_RADIATION_POLICY` plus
  `GPUWM_CARRIER_<NAME>_SOURCE` and `GPUWM_CARRIER_<NAME>_LAST_UPDATE`
  globals, stamped per frame from the live contract.

Fixed:
- Classic RRTM+Dudhia (1/1) radiation no longer stalls large nests on
  host dispatch. The RRTM longwave solver re-dispatched its whole CuPy
  op graph once per fixed 512-column chunk and re-uploaded its
  coefficient tables in every chunk -- about 3.2 million kernel
  launches per radiation step on a 500x400 nest, GPU idle 93 percent
  of that wall. The chunk is now sized from free device memory at the
  first eager solve of each column geometry and cached (512-column
  floor, whole-grid cap, explicit `column_chunk` still pins it); the
  tables move to the device once per process. Byte identical across
  chunk sizes, pinned by the existing chunking test; the RTX 3090
  re-measure reproduced the pre-fix trajectory digests bit for bit.
  The 2-domain NSSL steady step fell from 39.0 to 4.2 seconds and
  Morrison from 37.9 to 3.0; a window that took 807 seconds takes 165.
  The reported 16x NSSL-vs-Morrison gap was a benchmark config
  confound (d01-only versus d01+d02 walls); on matched configs NSSL
  costs 1.0-1.4x Morrison before and after this fix.
- The RRTM chunk auto-sizer no longer queries device memory inside CUDA
  graph capture, where `cudaMemGetInfo` is refused. A captured solve
  reuses the eagerly cached chunk; a cold capture takes the 512-column
  floor. Byte identical by chunk invariance, with a bit-exact capture
  test.
- The carrier-contract consumption check runs inside CUDA graph capture,
  where its blocking footprint and finiteness reads are refused. Under
  an active health ledger the footprint read is skipped and the check
  runs in full (fail-closed); the
  finiteness verdict is recorded into the ledger, re-accumulated on
  replay, and raised at the sweep drain.
- A streamed run survives its second output frame. The carrier
  contract's produced-at ledger now rides the domain clock's round trip
  through the tile sweep, the graph stepper's scalar records and the
  streamed checkpoint header. Fresh tile buffers previously kept their
  build-time stamps, so the freshness law refused hour-N consumption as
  stale GLW while radiation ran on every tile. The law is unchanged;
  pre-contract streamed checkpoints take the one-time producer
  refresh.
- The tiles harness and the fixtures behind the conformance, RUC
  runtime, MYJ and MYNN pairing suites declare the shortwave carriers
  the carrier contract made law, the way they declared GLW:
  `declared_swdown_kwargs`, `declared_carrier_kwargs` and
  `declare_offline_gsw` read the contract's own consumer matrix and
  declare the allocation zeros those rungs always ran on, so no gate
  digest moves.
- Both tiles real-case preparers assemble their water temperature before
  soil preprocessing, the mainline ERA5 route's own `assemble_for_route`
  call at full-domain scope so slab seams cannot split a connected body.
  Every streamed ERA5-with-SST run was refused at prepare without it.
  Policy, route and receipt reach the soil router,
  under a structural test.
- The committed GLW no-op digest test skips, naming the condition, when
  its probe subprocess cannot see a CUDA device the test process itself
  holds: visibility withheld from child processes, not a probe verdict.
  Every other nonzero exit still fails.
- The production streamed attach bound lateral boundaries EAGERLY on
  every buffer-tile change: every forcing interval re-validated, re-packed
  and re-uploaded, 27-63 ms per bind, host-blocking, ~0.3 s per step at
  the attribution run's largest arm. `make_tile_hook` now uses the
  single-slot streaming bind the real-case harness already carried the
  digest proof for, measured at ~0.01 ms per bind.
- The restart manifest's KF-expiry guard entered `array.device` as a CUDA
  context whenever the attribute existed. NumPy 2 gives every host array a
  `device`, the string `"cpu"`, so checkpointing host-resident state raised
  a TypeError. The guard now enters only a device that is a context
  manager; the CuPy owning-card entry is unchanged.
- The dycore left the periodic staggered alias slot stale between steps.
- RRTMGP memoised two device arrays without keying them to the card, so a
  second device read the first one's arrays.
- The map-factor division left `gpuwm/core/physics.py` without the alias
  it undoes.
- `plan_split` rejected the 1x1 identity plan.
- The restart inventory path ran a device-blind reduction.
- `kf` allocated `w0avg` at the first due cumulus call, so the carrier set
  changed identity mid-run. It is allocated at construction instead, which
  is why the slice gate now reassembles 246 arrays over 229 carriers where
  it reassembled 245 over 228.
- The streamed real-case route died at its first tile change:
  `tilestream/realcase.py` bound `tile_hook` to the superseded
  three-argument contract. The consolidated gate is periodic by design and
  never calls `tile_hook`, so it held 233 PASS / 0 FAIL throughout.
- The standalone RW-WPS preprocessing wheel staged `gpuwm/experiment.py`
  without `gpuwm.core.streaming`, which its field default requires.
- No `[tiles]` gate reports success over an empty comparison. Two empty
  carrier maps hashed equal and were recorded bit-exact, a reassembly
  verdict of `not bad` passed at zero carriers checked, and three verdict
  lines asserted totality while stating no size. Each now carries a
  declared precondition with a floor of one, and each passing verdict
  states how much it covered.
- The `health` entry in the mp=8 frozen-kernel census described a source
  this tree does not carry: two re-pins landed on opposite sides of a
  merge and only one was recorded. Recomputed over the merged source.
- The gate's Noah rungs declare their downward longwave instead of
  inheriting the constant that `initialize_physics` used to supply in
  silence. Every digest is unchanged and the receipt names an origin.
- A graph negative control that ran out of memory held the memory through
  the controls after it, by way of the caught exception's traceback.
- The wheel shipped `gpuwm/core/streaming.py` and not the `tilestream`
  package it calls, so `[tiles]` mode `auto` and `on` raised
  `ModuleNotFoundError` from a clean install while mode `off` worked.
  Every import is function-local and every gate runs from the repository
  root, which is why nothing caught it. `tilestream` is in the
  distribution now. `tools/les1m_probe.py` rode the same gap.
- `tilestream` resolved only when the working directory was the repository
  root, editable installs included. It resolves from any directory now.
- `tests/test_tiles_distribution.py` gates the property rather than the
  fix: the shipped package set must be closed under importing, and any
  file a shipped module opens beside itself must be declared package data.
  It is on stage 1.
- The physics allocation inventory still recorded `w0avg` against
  `kf.update_trigger_history` after the fix above moved it to
  `kf.ensure_trigger_history`. Stage 1 was green because that suite was
  not on the list. Row corrected, suite added.
- The gate list omitted the seam gate's `GRID=2x2` entry, so a battery run
  only ever split one axis. Both geometries discriminate.
- The card-floor note claimed the consolidated gate needs more than 16 GB,
  and its replacement claimed 16 GB is enough on any idle card. Both were
  wrong. An idle 16 GB RTX 4080 reports 233 checks passed and 0 failed. An
  idle 16 GB RTX 5080 refuses the graph section's capture allocations with
  15.2 GiB free, at 1.1 MB in one run and 201.6 MB in another. The note
  carries the measured matrix and says which cell an operator is reading.
- A graph capture the card would not serve was reported as a failed
  negative control, so the consolidated gate exited 1 on an idle 16 GB RTX
  5080. A row the card cannot run is MACHINE-LIMITED: counted on its own,
  printed where it happened, and listed in the verdict, which reads GATE
  PASSED WITH A COVERAGE HOLE and names every unevaluated row. A control
  that fails for any other reason, the harness included, still fails the
  gate.
- `[tiles]` decisions and refusals were worded as "streaming", which is
  the name of the unrelated `gpuwm stream` feature. They say `[tiles]`.
- The four gate runners printed "Box idle throughout" whenever at most one
  CUDA context was on the box, including when that one was somebody
  else's. The verdict separates this gate's own context from every other
  by pid, reports the split, and says UNKNOWN when nvidia-smi cannot be
  read.
- The consolidated gate then took its start count after its own first
  device call, so every run called the box not idle and its verdict
  provisional, an empty box included. The count is read before anything
  touches a card, as the other three runners already did.
- `tests/test_tiles_distribution.py` routed both load-bearing tests
  through `pytest.importorskip("setuptools")`, and the project virtualenv
  carries no setuptools, so the suite skipped in the environment it runs
  in. Each setuptools measurement now has a tomllib reading of the same
  declaration beside it, and a third test holds the two against each
  other. Reverting the include list to the state that shipped the defect
  turns the file red with setuptools absent and with setuptools present.
- `tests/test_tiles_gate_manifest.py` stayed green when `GRID=2x2` or
  `PHYSICS=1` was deleted from the gate list: entry identity is module
  plus environment, and every other test walks whatever the file says.
  Both rows are pinned by name.
- The check is at the consumer, so it covers direct `initialize_physics`
  callers, restarts and DA cycles, not only configs loaded through a front
  door. The config-load guards stay and are unchanged; they refuse earlier
  and more helpfully, and they are no longer the only line.
- `set_forcing` refuses a radiative carrier on a driver assembled
  without a contract, in the same sentence family as the radiation and
  consumption seams, instead of crashing on an attribute error after
  writing the buffer.
- Zero shortwave at night passes on source and age, never on the value
  looking plausible. The same zero after sunrise with no live producer
  refuses before the land-surface call, naming the carrier, the consumer
  and the fix.
- `gsw` joins the generic forcing setter. RUC reads net shortwave, so an
  offline-forced RUC run could not supply the one shortwave carrier its
  land surface consumes and integrated zeros.
- Radiation-free Noah-MP gets an analytic COSZEN provider on the radiation
  cadence. It was written once at startup, so a twelve-hour run computed
  canopy radiative transfer against the sun angle of its first minute.
- The 300 W m-2 GLW buffer fill can no longer be consumed. It stays as the
  allocation value so healthy trajectories are byte-identical, and its
  source is `unwritten` until a producer writes it. `swdown`'s 0.0 default
  becomes `None` on the same argument: the buffer is still zeros, and the
  contract can now tell a silent default from a declared zero.
- Carrier provenance and age are serialized into the restart driver header,
  two scalars per carrier, so no array key moves. A checkpoint written
  before the contract forces one producer refresh rather than guessing.
  Proven through `write_restart` and `restore_restart` themselves:
  a mid-interval checkpoint resumes with source and age bit-equal to the
  uninterrupted run, a header without the mapping forces the refresh,
  and a resume under a changed policy is refused.

Record correction:
- The 1.6.2 nocturnal dewpoint collapse ran under a silent fixed
  300 W m-2 GLW with shortwave-only radiation. It did not run under
  GLW = 0. Full LW+SW radiation removed the symptom and remains the
  default for real-data runs. The direct mechanism is proven for land and
  shoreline columns. Over open water the mechanism does not apply in a
  single surface call: SFCLAY builds its surface endpoint from TSK and
  PSFC alone (`gpuwm/core/kernels/sfclay.cu:255`) and TSK over water is
  prescribed, so a 120 W m-2 longwave change leaves QSFC, QFX, Q2 and
  TSK bit-identical (`tests/test_open_water_longwave_invariant.py`).
  Open-water attribution beyond one call is not claimed.
- The instrument for this class is
  `gpuwm/core/surface_moisture_ledger.py`, which records the final writer
  of Q2 by name, that writer's own inputs, and its formula re-evaluated in
  FP64 from the published inputs. `tools/surface_moisture_ledger.py`
  decomposes a difference between two arms into named terms and exits
  non-zero on an unexplained remainder above 0.1 g/kg. The decomposition
  is provider-aware: Noah SFCDIAGS rows expand the Noah identity
  (QSFC, QFX flux route, rho times cqs2 route, cross), SFCLAY-family
  rows expand theirs, because the first land A/B put the whole QFX route
  into the remainder under the SFCLAY terms.
- The accounting ran on a two-arm column A/B on node 5 differing only in
  the declared constant GLW, 300 versus 420 W m-2 over land for two
  hours: dQ2 +3.49 g/kg, dTd2 +4.8 K, dominant named term QSFC (skin
  saturation humidity, +5.07 g/kg, partially offset by the flux route),
  unexplained remainder 0.0 across all 1440 column-times, exit 0.
  Corrupting one published Q2 in the arm file flips the tool to exit 1.
  Receipt: `docs/public/receipts/surface-moisture-ledger-accounting.json`.
  This is column-tier evidence generated for the accounting; the
  historical case's data was never staged and no case-tier accounting is
  claimed.

Evidence:
- Ported from a tree whose branch point is byte-identical to this line at
  1.8.7 over all 13,585 shared paths, re-measured at the merge tip before
  anything moved. 229 clean adds, 40 modified files, 12 collisions, 10 of
  them merging with zero conflicts.
- No capacity multiplier against a resident run is stated here, because
  none has been measured. `tilestream/NO-DRY-NUMBERS.md` lists the figures
  that may not be quoted and why.

## 1.9.1 (2026-08-11)

Three defects the twin verification of the 1.9.0 ports caught, each
closed with a class instrument so the pattern cannot ship again.

Fixed:
- Milbrandt-Yau (`mp_physics = 9`) real cases run. Seven scheme-keyed
  tables lacked mp=9 arms: the preflight shape manifest, so an accepted
  configuration could not build its real-case workspace; the acoustic
  moist sum; the held-mixing slot registry; the nest forcing inventory;
  the coupled-scalar allowlist; the reflectivity consume membership; and
  the end-of-run digest's nest-slot membership, where a nested run
  integrated its full window and then reported failure. The last three
  also lacked arms for WDM6, P3 or aerosol-aware Thompson. The three
  hand-copied coupling sets and the digest membership now read one
  shared inventory, and a new stage-1 suite proves for every
  `mp_physics` value the loader accepts that the declared closure
  builds, classifies, couples, consumes its reflectivity and digests.
  WDM6's CCN pair and Milbrandt-Yau's hail-number pair joined the
  restart manifest with it.
- WDM6 and Milbrandt-Yau own the SR roundoff envelope WSM6 already had.
  The WDM6 kernel forms SR with the identical positive-sum expression as
  WSM6, but the validator granted the proven envelope to WSM6 alone, so
  a healthy WDM6 run died on WRF expression-order roundoff, one ULP
  above 1.0, at its first frozen-dominated column. The envelope is now
  keyed on the audited shared-expression family with an exact analytic
  bound per member. Morrison's expression provably cannot exceed 1.0 in
  float32 and keeps the tight range.
- RRTM longwave with Dudhia shortwave (1/1) finishes cleanly. The
  longwave adapter stored `p_top` as a NumPy scalar, which the restart
  classifier treats as an unclassified array, so every 1/1 run reported
  failure in the end-of-run digest after a perfect integration. The
  value is coerced exactly as the legacy-RRTMG adapter coerces its own,
  the composed pair's sub-adapters and the RRTM coefficient tables are
  classified, and a new suite walks every registered radiation callable
  through the restart classifier.

Known:
- The time-zero history frame publishes the declared constant downward
  longwave, 300 W m-2, where WRF writes 0.0 before the first radiation
  call. Recorded here for the surface-radiation carrier contract work;
  the first radiation call overwrites it.

## 1.9.0 (2026-08-10)

Six WRF schemes ported, plus the 1.8.9 recalibration batch and a
coherent water-temperature default on every forcing route.

Assembly notes for this line, including the inherited-red inventory and
the artefact regeneration order, are in
`docs/release-1.9-assembly-notes.md`.

New:
- WRF RRTM longwave (`ra_lw_physics = 1`), as WRF's classic pair with
  Dudhia shortwave (`ra_sw_physics = 1`). Line transcription of
  `module_ra_rrtm.F`; column smoke of the shipped seams only, no oracle
  comparison against the Fortran. This was the last schema-legal selector
  value that reached no accepted run.
- The NSSL 2-moment variant family: `nssl_hail_on`, `nssl_ccn_on`,
  `nssl_density_on` and `nssl_3moment` beside the default lane. Hail-off
  and diagnosed-CCN carry column smoke and treatment proofs, no oracle.
  `nssl_hail_on = 2`, `nssl_density_on = 1` and `nssl_3moment = 1` are
  refused by name.
- The MYJ PBL (`bl_pbl_physics = 2`) with the Eta similarity surface layer
  (`sf_sfclay_physics = 2`), admitted only as the 2/2 pair, which is WRF
  v4.6.1's own rule. Float32 CPU authority transcribed from the byte-frozen
  `module_bl_myjpbl.F`, with a CUDA translation unit agreeing inside a
  stated tolerance. TKE cold-starts at WRF's `epsq2` = 0.2. Declared
  divergence: interface heights are carried above ground, not above sea
  level, which cancels to within 69 ULP over 4.4 km terrain with `KPBL`
  unchanged. No oracle comparison against the Fortran.
- Milbrandt-Yau 2-moment microphysics (`mp_physics = 9`). Graupel and hail
  are separate prognostic categories and all twelve moments transport.
  Column smoke on three seeding layouts, water budget closing to 1.3e-4
  relative or better, plus a mutation control. No oracle. Per-domain
  override only.
- P3 one-category microphysics (`mp_physics = 50`). A 15-step 3-column
  integration stays finite and non-negative, conserves total water against
  surface precipitation to 1e-4 relative, and holds rime mass at or below
  ice mass with rime density in [50, 900]. WRF's
  `p3_lookupTable_1.dat-v5.4_2momI` is packaged verbatim and SHA-256
  validated at load; nothing yet checks that the interpolation of it
  reproduces WRF's. CPU only, no CUDA mirror. Per-domain override only.
  Siblings 51, 52 and 53 are refused by name.
- WDM6 double-moment warm rain (`mp_physics = 16`). CUDA kernel and
  `wdm6init` transcribed line by line from `module_mp_wdm6.F` with
  file:line citations, a float64 coefficient block pinning the kernel's
  baked FP32 literals, and a column smoke asserting WDM6's own bounds,
  water conservation to the surface flux, and CCN activation moving number
  from `nn` into `nc`. No oracle. WDM5 (14) and WDM7 (26) are refused by
  name. Per-domain override only.
- `mp_physics = 9` with RTE+RRTMGP is refused for absent cloud-optics
  coupling. WRF leaves `has_reqc`/`has_reqi`/`has_reqs` at 0 for
  MILBRANDT2MOM and the scheme's effective-radius block is commented out,
  so there are no scheme radii to hand RRTMGP. The refusal names two
  remedies and both are measured to work: `ra_rrtmg_variant =
  'rrtmg_legacy'`, or the Dudhia pair. Milbrandt-Yau under MYNN runs on
  either of them.
- `mp_physics = 50` with RTE+RRTMGP is refused for absent cloud-optics
  coupling. WRF sets `has_reqs` to 0 for P3 and its single ice category
  carries no separate snow species, so there is no snow radius to hand
  RRTMGP. The pairing was previously admitted and failed at the first
  radiation call. The refusal names the same two remedies as the
  Milbrandt-Yau one and both are measured to work.
- The composition walk grows to 9781 combinations, 990 accepted against 18
  registered templates and 8791 refused under 19 distinct rules. Every one
  of the 19 rules carries a demonstrated remedy pair, and every admitted
  value of every axis now reaches an accepted run.
- `water_temperature_policy` in `[case_data]` declares which provider decides
  water temperature: `era5_class_coherent` (the default that silence
  selects), `wrf_compat` (the historical per-cell selector, byte-for-byte,
  for stock-WRF certification and parity batteries), and `external_overlay`
  (the declared-overlay machinery, unchanged, which a declared overlay
  selects on its own).
- An advisory line names the policy, the water-cell count, the connected-body
  count and the per-provider cell counts on any domain the coherent policy
  touches, and states that a declared overlay with an observational analysis
  remains the higher-accuracy option. It warns and never blocks.
- `tools/water_temperature_probe.py` emits the decision surface for one
  domain of a case: mapped SST, mapped SKINTEMP, their difference, the
  validity mask the old selector consumed, the assembled field, the provider
  ids and TSK, under both policies side by side.

Fixed:
- WSM6 is priced at the tier it compiles rather than at a flat row. The
  kernel frame under-priced local memory above 64 levels by 446 MiB on
  4-domain LES configs. WSM6 and WDM6 both price per domain that selects
  them, at 112 and 152 bytes per level, so a 40-level WSM6 child beside a
  100-level parent takes its own bound.
- The WSM6 front-door level bound is bound to the kernel's own ladder
  instead of a transcribed constant, so the two cannot drift apart.
- The decoder door asks the bridge-contract question itself. Resolution
  checked existence only, which let a preparation run against a non-ABI
  stand-in.
- `PSFC` follows the surface switch rather than the existence of a driver.
  wrfout published a fabricated surface pressure for runs with surface
  physics off.
- A 0 K deep soil temperature is refused at ingest. The water fill passed
  every finiteness check and reached the land surface as weather.
- The duplicated periodic staggered face is pinned to face 0, exactly.
- Five NSSL selectors entered the restart identity of every run, including
  runs that select no NSSL scheme. They join `SCHEME_SCOPED_RUN_FIELDS`
  beside WDM6's pair, so an experiment written before these schemes existed
  keeps its fingerprint and its checkpoints stay resumable.
- Out-of-schema `mp_physics` values all recite the accepted menu. The P3
  siblings named their missing physics without it, so a value refusal and a
  composition refusal read the same to anything classifying by message.
- mp18's ring HAILNC accumulated where WRF's is exactly zero. The clipped
  tile guard's slot family did not carry the NSSL hail pair, so the ring
  retained hail the driver never wrote. Both hail-bearing schemes now share
  one slot family. The per-domain allocation estimate rises by 8244 bytes.
- P3 (`mp_physics = 50`) and MYJ (`bl_pbl_physics = 2` with
  `sf_sfclay_physics = 2`) can be run. Both were admitted by the config
  loader and absent from the memory preflight's kernel module tables, so
  `gpuwm check` refused them and a run died on the same call. P3 prices at
  the shared moisture validator, since it is a host transcription that
  launches no kernel of its own; MYJ prices at `myjpbl` and `myjsfc`,
  9,232 and 0 bytes per thread.
- The Milbrandt-Yau RTE+RRTMGP refusal reaches the registry. It lived only
  as prose in an option's extensions, so a launcher offered 320 component
  combinations that the runtime refuses after the prepare. Registry
  constraints gain a conditional kind, `refused_when`, because the coupling
  is a conjunction of a radiation option and an `ra_rrtmg_variant` value
  and none of the three existing kinds can state one.
- Lakes and coastal water no longer initialize from a per-cell mixture of two
  differently-mapped fields on the ERA5 route. The shipped chain chose SST
  where the mapped SST passed a 170..400 K test and ERA5 SKINTEMP everywhere
  else. METGRID.TBL maps SST with `sixteen_pt+four_pt` and `fill_missing=0.`,
  both operators need a fully usable stencil, and an SST analysis is missing
  over land, so every water cell within two source cells of a coastline took
  the fill and flipped to SKINTEMP. Over lakes ERA5 SKINTEMP is the FLake
  model state, several K warmer and quantized on the 0.25 degree cell, so the
  lake came out a quilt of two providers joined along the validity boundary.
- `gpuwm/ingest/water_temperature.py` now assembles the finished field before
  soil preprocessing runs. Surface classes come from the target statics, each
  class is labelled into connected components, and one provider is chosen per
  component: the ERA5 analysis on that component's own donors where they
  cover at least half of it, otherwise the coherent skin field for the whole
  body. Donors are selected by component identity, so a lake cannot take
  ocean water and no water cell can take a land donor. Every water cell
  carries a provider id in `WATER_TEMP_SOURCE`.
- Measured on Lake Erie 1985-05-31 12Z, 3 km nest, 2684-cell lake, in the 18Z
  forecast frame of this run and of the control: intra-lake adjacent-cell P99
  7.32 K to 0.13 K, cells stepping more than 1 K 5.18 % to 0.00 %,
  shore-to-midlake TSK gap +5.21 K to +0.46 K, lake TSK range 284.51..294.86 K
  to 284.49..289.41 K. The stock WRF v4.6.1 oracle at 1 km sits at P99 0.60 K
  and 0.65 % above 1 K, so the default now reaches oracle class. No declared
  file and no network access are involved.
- The guarantee is enforced at the soil preprocessing routers, the one seam
  every forcing route crosses, and not route by route. A route that hands the
  router a raw SST beside its SKINTEMP with no assembled water temperature is
  refused by name and told where the decision belongs. A route added later
  cannot reinstate the per-cell selector by omission.
- Lakes are named by the selected land-use table's own ISLAKE on every route.
  A table with no inland-water class is a declared state the advisory states
  out loud, never a silently empty lake mask.
- The met_em / rw-wps route and the 20CRv3 route that rides it now carry the
  lake mask and the policy into the assembly, so the lake-never-takes-ocean
  guarantee holds there and not only on ERA5.
- The GFS route forwarded no assembled field into soil while printing the
  coherent-policy advisory, so the historical per-cell selector ran behind a
  receipt saying it had not. The route now assembles after its lake skin is
  resolved and the router consumes that field. A test refuses the advisory on
  any route that does not consume what it assembled.
- Water temperature under `era5_class_coherent` interpolates the source
  analysis, so cells where the mapped SST was valid are not bit-identical to
  the previous default. Measured on the reproducing case: mean +0.002 K, max
  0.169 K, 0 of 1955 bit-identical. Unverified on a real coastal ocean domain.
  Declare `wrf_compat` to reproduce an archived run byte for byte.
- The ERA5 direct adapter run with `--static-input` and no `--geog-root`
  cannot resolve a land-use table, because the prebuilt static cache carries
  numeric fields only. That run says so in one line and keeps the historical
  selector. Passing `--geog-root` restores the default treatment.
- Connected-component labels are a pure function of two invariant statics but
  were recomputed for every forcing time, 0.19 s at 550 squared and 0.61 s at
  1000 squared on the launch-to-first-plot path. They are computed once per
  domain and reused.

## 1.8.9 (2026-08-10)

1.8.8 was tagged but never published: the RW-WPS staging gate refused the
standalone preprocessing wheel, so nothing reached PyPI. The tag stays where
it is, because tags here are forward-only. Everything 1.8.8 carried ships
here, plus the first fix below.

New:
- MYNN is selectable for a real forecast. Three shipped suites pair the
  MYNN PBL and MYNN surface layer with RTE+RRTMGP longwave AND shortwave,
  one per land surface the no-radiation MYNN family already covered:
  `wsm6-mynn-mynn-noah-rte-rrtmgp-implemented-unverified-v1`,
  `wsm6-mynn-mynn-ruc-rte-rrtmgp-implemented-unverified-v1` and the expert
  `wsm6-mynn-mynn-noahmp-rte-rrtmgp-expert-only-v1`. Every named MYNN suite
  before this ran shortwave with longwave off, which is refused over a night
  window unless declared, so picking MYNN from a menu meant picking the
  nocturnally invalid class. Each new row is its no-radiation sibling with
  only the radiation block changed (`ra_lw_physics` 0 to 4, `ra_sw_physics` 1
  to 4, `radt` 1.0 to 12.0). MYNN's pinned option identity is unchanged.
- Physics composability is now a measurement.
  `docs/public/receipts/physics-composition-walk.json` records 6536 physics
  combinations pushed through `gpuwm.experiment.build_experiment` one at a
  time: 749 accepted (741 distinct suites against 18 registered templates),
  5787 refused, 15 distinct refusal rules, and zero accepted runs whose
  resolved `RunConfig` differs from what the file asked for. Every admitted
  value of every selector reaches an accepted run except `ra_lw_physics = 1`,
  which is unported and says so. Regenerate with
  `tools/report_physics_composition_walk.py --table`.

Fixed:
- The standalone RW-WPS preprocessing wheel builds again, which is why
  1.8.8 was withdrawn. Two imports added during that line reached modules
  the wheel does not carry, one inside a refusal that fires when a config
  loads: a standalone user who tripped it got an import error instead of
  the sentence written for them. Both are resolved at the source rather
  than waived, and the staging scan that caught them now runs in the
  first-stage battery instead of only at a release cut.
- Two refusals you could not act on. `km_opt = 2` with a PBL scheme on
  offered `km_opt 3/4`, but the branch above refuses 3 for the same reason;
  it names `km_opt = 4` now and says why 3 is not an option. The coupled
  LW/SW adapter rule, the most frequent refusal in the configuration space,
  named no key and no offending value; it opens with `ra_lw_physics=<got>`
  and `ra_sw_physics=<got>` now and lists the admitted pairs. A standing
  check requires every refusal to name a selector and to lead somewhere that
  runs.
- `docs/public/PHYSICS.md` read as though the shipped suites were the only
  suites. It
  says on the page now that the tables are a catalogue and not a gate,
  defines `reachability.state` by quoting the registry, and prints what a
  config naming each registry-unreachable option actually gets: two are
  genuinely unreachable, three are accepted by the loader and merely off the
  menus. The revised MM5 surface layer was one of the three: the page said no
  config could reach it, while `sf_sfclay_physics = 1` is accepted and runs.
  The MYNN rows called the 5/5 PBL/surface-layer pairing mandatory both ways
  when only the surface layer restricts its partner; what is pinned is MYNN's
  twelve `bl_mynn_*`/`icloud_bl` knobs, each with one implemented value.
  `tests/test_physics_md_reachability_claims.py` derives all of it from the
  registry, the walk receipt and `build_experiment`.
- `--materialize-authorities` no longer replaces a physics value your config
  states. It deleted all 26 profile-owned keys and rewrote them from the
  named profile, silently, into a file every later stage binds by hash: a
  config saying WSM6 with no longwave ran as Morrison with RRTMG and every
  downstream check passed. A named profile now SUPPLIES the keys a config is
  silent about and REFUSES the ones it states differently, naming each key,
  both values and both remedies, before the fetch. Silent agreement is
  unchanged.
- `gpuwm go` and `gpuwm run-plan`'s prepared route derive the forwarded
  `--physics-profile` from the whole config, not the root domain alone. The
  wizard's `--ladder` trees declare nests that deliberately depart from the
  root suite, so the root-only derivation composed a stage-1 command the
  refusal above is guaranteed to refuse. A config the profile contradicts
  nowhere carries the assertion end to end; one that says more runs unnamed,
  as its own suite.
- A refusal whose root already IS the named profile no longer offers the
  flag it just refused as the remedy. It says to omit `--physics-profile`,
  and no longer suggests editing an LES nest's resolved-turbulence keys back
  to a PBL suite.
- `ra_rrtmg_variant` is governed under a profile that resolves the RRTMG
  (4, 4) pair without pinning the variant. A config declaring the other
  variant kept it and ran legacy RRTMG under a profile named rte-rrtmgp,
  silently; it refuses by name now. A declaration matching the resolved
  variant is kept where it was written.
- `gpuwm certify` exited 0 on a metrics CSV with a header and no data rows,
  because every row condition is a statement about all rows and is true of
  none. A declared condition, `the_comparison_is_not_empty`, runs before the
  per-row ones and names what it found against what it required, and
  `rederive_verdict` refuses an empty comparison independently. The floor is
  one row and one comparison, so a run shorter than the band's longest lead
  is still a legitimate partial comparison.
- `gpuwm dual-run` reported `capsules are identical field for field` on two
  empty documents, and it is the only detector this project has for the
  silent memory corruption a card with no ECC cannot report. An empty
  comparison is refused by name and by arm, a zero-byte capsule is refused
  with its byte count, and every passing screen states its size
  (`... (71 compared quantities)`), with `compared_count` in the written
  report. A half-empty pair is still a divergence, not a refusal.
- The certification schema version moved with the condition set it declares.
  `the_comparison_is_not_empty` was added while `verdict_schema_version`
  stayed at `1.0.0`, and `rederive_verdict` compares that set for exact
  equality, so a genuine document from the previous release rederived
  `False`: the same silent answer a forged one gets. The version is `1.1.0`,
  `CONDITIONS_BY_SCHEMA_VERSION` carries the older set, an unknown version
  fails closed naming itself, and every refusal carries a reason. The
  minimum-comparison floor applies on every version.
- `docs/public/CERTIFICATION.md` listed seven conditions where the code
  declares eight, and described `dual-run` as two outcomes where there are
  three. Both are corrected, in `docs/public/DETERMINISM.md` as well, and a
  test pins the public condition table to
  `gpuwm.certify.verdict.CONDITIONS`.
- The bundled renderer is checked against a contract, like every other
  bundled binary. `rw_wrfbatch` was asked only whether it started, so two
  builds two days and 4 MB apart, neither from this tree, printed the usage
  line, were reported `verified`, and drew the plots. It answers
  `--abi` now with the handshake `rw_fetch` and `rw_nexrad` have always
  answered, pinned in `gpuwm.rustwx.RENDERER_ABI_MARKER`. `gpuwm doctor`
  reports a mismatch with the rebuild remedy and lets the run proceed,
  matplotlib being the documented fallback; the rust render suite skips a
  foreign engine rather than substituting it.
- `gpuwm render --engine rust` went around that check. Only `--engine auto`
  probed, so the caller who pinned the engine to be sure of the real
  renderer was the one who did not get it. Both forms ask the same
  question now: `auto` falls back with the reason, `rust` refuses with exit
  2. `--list-products` with no wrfout resolves its own engine and prints the
  same refusal instead of a traceback.
- The release battery's stage-1 list gained the other half of its radiation
  gate. It proved a shortwave-on/longwave-off pairing is refused and proved
  nothing about radiation running: forcing the Dudhia heating rate to zero
  left the entire pre-amendment list green (901 passed, 6 skipped).
  `tests/test_wrf_legacy_radiation.py` is listed now; it drives the
  production `PhysicsDriver` radiation seam on the CPU and fails on that
  mutation, and `tests/test_stage1_manifest.py` pins both halves.
- Downward longwave no longer has a silent default. Any run with
  `ra_lw_physics = 0` integrated a fixed 300 W m-2 GLW for its whole
  forecast: `initialize_physics` defaulted `glw = 300.0` and no production
  call site passed anything else. Radiative equilibrium at 300 W m-2 is
  269.7 K; a Gulf-coast October night runs near 410 W m-2, or 291.6 K. That
  deficit craters skin temperature, takes the surface saturation humidity
  with it, and produced the reported nocturnal dewpoint collapse. `glw` has
  NO default now: a longwave scheme owns the field, or the caller types a
  constant or hands over a source array, or nothing reads or publishes it.
  Anything else is refused, naming the selectors, the number it would have
  fabricated, and three remedies. WRF v4.6.1 refuses the same shortwave-only
  pairing outright (`phys/module_radiation_driver.F:2245`).
- Radiation entirely off is covered by the same rule, and is where gpuwm
  deliberately parts from WRF. That suite attached no radiation adapter,
  while Noah, Noah-MP and RUC read downward longwave every surface step, so
  the run integrated an undeclared constructor seed. WRF's answer there is
  0 W m-2, an absent atmosphere rather than a thin one, and a column under
  it cools without bound; gpuwm does not copy that. With a land surface
  attached the pairing is refused at load, and a declared run integrates
  `gpuwm.physics_compat.DECLARED_CONSTANT_GLW_WM2`, 300 W m-2, named in the
  run receipt. With no land surface nothing reads the field and nothing
  fires. A run with a longwave scheme is byte-for-byte what it was.
- A real experiment whose downward longwave would be consumed (a land
  surface with no longwave scheme) or published (shortwave on with no
  longwave scheme, so the GLW row reaches every wrfout frame) refuses at
  config load, at the one loader every front door shares, with its own
  acknowledgement: `constant-downward-longwave-v1`. The load guard and
  `initialize_physics` refuse the SAME selector set, so a config either fails
  at the door or runs. This is deliberately NOT the 1.7.1 nocturnal token,
  which is checked before any physics is inspected and so lifted this
  question too. Every resolved-configuration report names each domain's
  downward-longwave source on a `radiation.downward_longwave` line.
- `configs/era5_wrf_direct_proof.toml`, `configs/gfs_wrf_direct_proof.toml`
  and `configs/gfs_wrf_hierarchy_proof.toml` ship with real radiation (legacy
  RRTMG on both streams, matching their paired stock-WRF namelists, which now
  say the same) and carry no acknowledgement at all.
  `tools/ens_sweep/kdmx_case.toml`, a 04Z-10Z Iowa window that is night end
  to end, moves to `thompson-mp8-ysu-mm5-noah-rrtmg-legacy-v1`. The LES
  records, the battery shape smoke, the init demos and the MYNN no-radiation
  probes keep their physics, their bytes being the provenance of committed
  results, and declare the constant instead.
- A shipped config may not carry an acknowledgement token silently. Each one
  must be answered by a `# JUSTIFY <token>:` block of real length in the
  same file, gated by
  `tests/test_shipped_acknowledgement_justifications.py`.
- `gpuwm enprod` stamped `EXPERIMENTAL (v1.2 ensemble, uncalibrated)` onto
  every ensemble panel, frozen at the release the suite was written for, so a
  1.8.7 plot claimed to come from a 1.2 ensemble. The warning stays and is
  still true; the version comes from the running engine now.
- The wizard no longer writes the acknowledgement itself. `gpuwm domain` wrote
  `acknowledgements = ["asymmetric-radiation-nocturnal-window-v1"]` into the
  emitted `[experiment]` by itself, and every later door reads that line for
  the life of the file. It refuses an asymmetric profile over a night window
  now and names both ways forward; `--ack <id>` is what writes the line.
- The shipped LES configs no longer carry the setting that caused the 1.6.0
  anisotropic-mixing instability. 1.6.0 closed that defect with a criterion
  and left every example config on the wrong side of it: the two 250 m nested
  trees at `mix_upper_bound*(dz_max/dx)^2 = 0.702`, and both 100 m tornado
  trees at 4.23 on d04, 17x the limit and the exact configuration that
  aborted a run at step 5467 with `w = 239.48 m/s`. All four run
  `mix_isotropic = 1` on their LES children now. That key is inside the
  restart fingerprint, so these trees run from t = 0, and none has been
  re-scored: every published number for them was measured on the old bytes.
- The configuration a committed receipt was produced under is archived at
  `configs/frozen/`, pinned by sha256, and it loads. Each of the three
  records there declares the constant downward longwave its run always had,
  so the loader accepts the file, the sha256 gate passes, and reproducing
  one of those runs gives the same physics the archived run had. Those three
  are why `gpuwm check` passes on three more shipped configurations than it
  did. No physics selector moved and no committed receipt was edited, so a
  record now carries two digests: the as-run one its receipts name, and the
  on-disk one a reproduction passes. `configs/frozen/README.md` publishes
  both and says which is which. `docs/public/LES.md` and
  `docs/public/GRAYZONE-NEST.md` say now that their measured numbers belong
  to the archived configuration rather than to the configs those pages name.
- `tests/test_shipped_configs_mixing_stability.py` fails if any config under
  `configs/` arrives on the exposed mixing path, and it runs on every release
  cut; the criterion used to be checked only when somebody loaded a config
  and read stderr. The frozen records are the one exemption, each pinned to
  its content hash.
- The criterion is no longer skipped on a config that writes no eta ladder.
  `km_opt = 3` with `mix_isotropic = 0` at `dx = 100` m and no `eta_levels`
  is the exact shape the criterion exists for, and it used to load and pass
  `gpuwm check` in silence, reported clean. The depths are resolved the way
  the model resolves them now: uniform interfaces from `nz`, with `ztop`
  converted to a pressure by the exact inverse of the relation the depths
  are read with. This reverses a 1.6.0 ruling that skipped such coordinates;
  downstream, a skip was indistinguishable from a pass.
- Where no depth can be produced at all, the criterion says so instead of
  going quiet. A model top above the analytic base state's ~24.6 km ceiling
  has no representable pressure, so such a domain sits on the exposed path
  with no number to judge it by. There is still no ratio, because a
  fabricated depth in a stability criterion is worse than none; an advisory
  fires instead, naming what could not be resolved and the two remedies:
  declare `eta_levels` and `p_top`, or set `mix_isotropic = 1`.
- `gpuwm check` repeats the mixing advisory in its report and under
  `advisories` in `--json`; the 1.6.0 run had been warned at config load,
  hours before it died. The advisory is tiered now: a ratio that inverts the
  2dx mode (above 1/4) reads differently from one that grows it (above 1/2),
  so 0.3 and 4.2 no longer look alike. Still not a refusal.

CONTRACT CHANGE: a config that declares a suite and is materialized under a
different named profile used to succeed and now refuses. Edit the config's
physics to the profile's values, name the profile the config already is, or
omit `--physics-profile` to publish the config's own suite.

CONTRACT CHANGE, prepared-cache identity: a config declaring `radt_minutes`
keeps it now (no profile pins that key; the old materializer deleted it),
which moves `prepared_domain_config_identity` for that key. A prepared tree
built from such a config before this change must be prepared once more. No
shipped config declares `radt_minutes`, and the kept value is physically
inert under every shipped profile.

CONTRACT CHANGE, version identity: running gpuwm from a source checkout
while a DIFFERENT gpuwm version is pip-installed now refuses at config load,
because every receipt would otherwise be stamped with a release that is not
executing. Bind the tree to its own metadata with `pip install -e <tree>`, or
run the installed copy. A borrowed version that AGREES warns one line and
still runs, so ordinary worktrees beside an editable install are unaffected,
and a plain wheel install never sees this at all. The refusal fires at the
one load every front door shares, and it names both versions, both locations
and the one-line fix.

Changed (compatibility):
- **A real config that runs a land-surface model with no radiation is now
  refused at load.** This is a load-time break of files that worked before,
  and it is wider than "configs that set `ra_*_physics = 0`": radiation
  defaults to off, so **any** real config (one with a `[projection]`) that
  selects a land surface and simply OMITS a radiation selector is in the
  class. That is the larger group and the likelier surprise, so an omitting
  config is told `NO RADIATION SELECTOR SET AT ALL` and offered the missing
  line as the first remedy, instead of being quoted two zeros its author
  never wrote. Idealized experiments (no `[projection]`) are not guarded. To
  keep such a run, both claims have to be declared, in one edit, in
  `[experiment]`:

  ```
  acknowledgements = [
    "radiation-off-land-surface-v1",
    "constant-downward-longwave-v1",
  ]
  ```

  Two tokens, because there are two claims: this run has no radiation
  scheme, and the number its surface integrates is one the config declares. The
  first token alone is still refused. Two shipped configs are in the class,
  and each declares the pair in-file with its reason.
- **Two refusal layers, in a fixed order, and an exit code that tells them
  apart.** The declaration guard above answers at config load, before any
  route logic runs: a token-less radiation-off config never becomes an
  experiment at all. Route admission (`gpuwm.hrrr_route_inputs`) answers
  second, and only about configs that loaded. Tools carrying both report it
  the same way `tools/battery_route_preflight.py` does: exit 2 with no
  receipt for the config error, exit 1 with a `REFUSED` receipt naming the
  gate for a route-refused run.
- **The radar-DA storm nowcast's default physics profile changed, and it
  roughly doubles the VRAM plan.** `wsm6-ysu-mm5-noah-no-radiation-v1`
  (shortwave on, longwave off) was the default of a product whose cases are
  overwhelmingly nocturnal; it is now the HRRR route's own
  `thompson-mp8-ysu-mm5-noah-rrtmg-legacy-v1`, because the nowcast's
  background is HRRR. On the shipped 132x132x49 3 km parent that moves the
  machine peak envelope from 3606 MiB to 6993 MiB (+3.3 GiB, 1.94x) and the
  allocation estimate from 618 MiB to 3499 MiB. **A member count or VRAM plan
  measured before this has to be re-measured, not extrapolated.** All six
  surfaces move together, including `tools/da_nest_cost.py`, the route's
  documented VRAM gate, which imports the constant now rather than restating
  it. `--physics-profile` still takes any shipped profile.

## 1.8.7 (2026-08-08)

New:
- The first plot lands while the forecast is still running. The first
  frame a forecast commits is the analysis -- WRF's history alarm is
  true at `t = 0`, so it is written before a single step is integrated
  -- and it used to sit finished on disk until the finalize stage,
  which waits for the last timestep. A run that sets `render_products`
  now renders that frame the moment `output_committed` fires for it, on
  a worker thread, concurrent with the forecast. Default ON where it
  applies and inert everywhere else: a plan naming no products (the
  default) or `none` behaves exactly as before, and the `experiment`
  route has no such option. There is no second switch.
- Every run receipt now carries a time-to-first-plot number.
  `first_products_ready` names the frame, the pictures and
  `seconds_from_plan_accepted` -- wall clock from the instant the plan
  was accepted, which is the instant the person who launched it started
  waiting, to the instant the pictures were readable. It is measured by
  the engine rather than by a stopwatch outside it, and it is emitted
  before the frame is digested and before the receipt is written,
  because both are finalize bookkeeping and hashing a 362 MB history
  frame would have put a second of it inside the number. The
  `completed` event repeats it as `first_products_seconds`, null and
  not zero when a run published nothing early, so comparing two runs
  does not mean scanning two event streams.
- Finalize does not redo the early work, and does not take that on
  trust. The early render leaves `first-products.json` naming the frame
  it read and every PNG it wrote, all by sha256. Finalize drops that
  frame from its own list only when the frame still hashes to the
  recorded digest, every recorded picture is on disk hashing to its
  recorded digest, and `render_products` has not changed. Any other
  answer -- a deleted picture, an edited frame, a different product
  spec, a render still running -- and the frame is rendered again. Six
  mutations are pinned to void the claim. Pictures are published by
  `os.replace` out of a scratch directory, so a reader watching the
  output directory sees a whole PNG or none.
- The release battery's stage-1 curated list ships in the repository,
  at `tools/battery/stage1_files.txt`, gated by
  `tests/test_stage1_manifest.py`. Until now that list existed only as
  an array inside per-assembly scratchpad scripts, propagated by copy
  from one assembly to the next: a lost scratchpad silently lost every
  amendment ever made to it and nothing anywhere failed when it did.
  The file carries each entry's reason inline, the gate checks that
  every path exists and nothing is listed twice, and the two entries
  whose absence caused a real miss are pinned by name against a
  tidy-up. This release is the first assembly to consume it, and the
  first to amend it in the repository rather than in a scratchpad.
- Analysis-hour-first fetch ordering is now a contract rather than an
  accident. Every ladder builder already returned an ascending range
  from the forecast-start lead and every fetch loop already walked it
  in order, so reordering buys zero seconds -- there was nothing to
  reorder. The property is pinned because it is what any future overlap
  work stands on, and pinned alongside it is the property such work
  would have to preserve: `SHA256SUMS`, which preparation binds through
  `--source-manifest-sha256`, is byte-identical whatever order the
  hours land in. That is proven by fetching the same window forwards
  and backwards and comparing bytes, not by reading the `sorted` call
  that makes it true, and a second test keeps order independence from
  becoming order blindness by requiring the receipt to claim only hours
  that completed.

Evidence: measured on a real seven-frame `d01-12km` run, 362 MB
analysis frame, products `refl,t2,wind10,precip`. Early render of the
analysis frame alone, 17.0 s and 4 pictures; the finalize render that
used to do all seven frames, 116.0 s and 28 pictures; the finalize
render of the remaining six, 100.0 s. So the split costs nothing
measurable and the 17 s hides under a running forecast, while the first
plot stops waiting for the last timestep. All four early pictures are
byte-identical to the same frame rendered inside the seven-frame batch.
The renderer never touches the card: it is a separate `python -m
gpuwm.cli render` process driving the CPU-side Rust binary, and it
creates no CUDA context -- `cuCtxGetCurrent` returns
`CUDA_ERROR_NOT_INITIALIZED`, interrogated from the driver rather than
read off the module table. Handing the committed frame to the worker
returns to the writer thread in 0.0007 s, and every telemetry path
swallows into a `warning` so no render can fail a forecast. Why the
larger overlap win was not taken -- preparation is one pass that seals
the initial condition and the boundary tables together, and the sealed
manifest binds the role set, so preparing from a subset is a different
digest against a runner that binds exactly one -- is recorded in
`docs/run-plan.md` rather than left for the next reader to rediscover.

## 1.8.6 (2026-08-08)

New:
- Moving nests run on both prepared routes, driven from `gpuwm
  run-plan`. 1.8.4 landed the statics corridor on `gpuwm go` and the
  tree runner; what was left was the front door. The prepared GFS chain
  already composed `--statics-corridor` on its prepare stage and nobody
  had checked it from this door, so that is now pinned by driving the
  real chain for a follow plan and reading the composed command back.
  The prepared HRRR chain could not do it at all, and now does.
- `gpuwm.hrrr_hierarchy_direct` takes `--statics-corridor`, bare or as
  comma-separated child grid ids, spelled exactly as the GFS door
  spells it. It seals the set into
  `hierarchy-artifacts/statics-corridor/` -- the same relative path the
  GFS bundle uses -- and binds its receipt into `receipt.json` beside
  `artifact_receipt`, exactly as the GFS chain binds it into
  `proof.json`. It is that stage and not `tools/prepare_hrrr_wrf`
  because a corridor is child-resolution statics over the parent
  extent, and the root preparer never sees a child: the hierarchy stage
  requires `--geog-root`, builds `d02..dNN`, and already holds the
  verified GEOG catalog. Without the flag the receipt has no such key
  and the bundle is byte-for-byte what it was.
- The tree runner reads `statics_corridor` out of whichever hierarchy
  document matched its pinned digest and cannot tell the two chains
  apart, which was true by construction in 1.8.4 and is now a
  measurement: the corridor refusal matrix -- no corridor, uncovered
  child, failed verification, accepted -- is parametrized over both the
  GFS `proof.json` and the HRRR `receipt.json`.
- `gpuwm run-plan --resolve` reports the decision instead of leaving a
  caller to guess or launch and find out. A `moving_nest` record names
  the chain, the delivery, the relocating grid and whether a corridor
  will be built, with a matching `automatic_resolutions` entry (scope
  `preparation`, basis `relocation_follow`) whose note names the stage
  that actually carries the flag -- on HRRR that is the hierarchy
  stage, and "the prepare stage" would have sent a reader to a tool
  that refuses it. `moving_nest` is `null` for a config that moves no
  nest.
- `gpuwm run-plan --estimate` prices the corridor. It is the one
  preparation artifact whose size cannot be inferred from the domain
  sizes a caller already has, being parent extent at child resolution:
  a 45x45 nest at `parent_grid_ratio = 3` on a 398x320 root is a
  1194x960 corridor at 889 MB. The new `corridor` block prices every
  child, because run-plan passes the flag bare and the preparation
  reads that as every child domain, and it prices them through the
  preparation's own child selection and the corridor module's own
  arithmetic rather than a second copy of either. `corridor_cost` is
  held equal to a real build's sealed `host_bytes` by a test instead of
  by agreeing today. The corridor adds no GPU residency, so the `vram`
  block and the VRAM gate are unchanged with and without it; that was
  verified by pricing one experiment on both chains and requiring the
  two figures to be equal, not assumed.

Fixed:
- One emitter, one child selection, one predicate. Both chains now call
  `gpuwm.static.corridor.emit_statics_corridor_set`, the child
  selection moved to `validated_corridor_selection` in that module
  (`source_hierarchy` held one copy and run-plan's pricing reached in
  past a leading underscore to read it), and the follow predicate that
  had three copies kept equal by a drift test is now the single
  `config_declares_follow_source` that `gpuwm go`, the printed `rw-wps`
  line and run-plan all call. The tree runner's own inline copy of that
  predicate is gone -- it was the one place that decides whether a
  bundle is accepted, holding its own reading of the sentence every
  preparation door reads from the module.
- A moving nest on a nested HRRR tree no longer fetches, prepares a
  root and builds a hierarchy before dying at the forecast preflight.
  `_FOLLOW_STATICS_DELIVERY` declares per chain where a moving nest's
  statics come from -- corridor, live ingest, or nowhere -- under a
  completeness guard, and `_execute_prepared_route` dispatches through
  the same `_chain_key` the refusal is decided on, so a config cannot
  be judged as one chain and run as the other. With HRRR moving from
  `None` to `statics_corridor` the refusal table is empty: no chain
  this front door dispatches to refuses a moving nest today. The
  refusal machinery and both completeness tests stay for a chain added
  tomorrow that cannot feed one, and the refusal is now called directly
  so that retained machinery is exercised rather than merely retained.
- `docs/run-plan.md` no longer says nested HRRR is refused. It shipped
  in c45e24924 without the doc catching up, which by 1.8.6 directly
  contradicted the section above it.
- `docs/run-plan.md` now says not to touch a checkout a run is reading
  from. The HRRR hierarchy stage samples `git status --short` when it
  publishes and refuses a tree with anything uncommitted in it, which
  is deliberate -- a bundle stamped with a commit it was not built from
  is a false provenance claim -- but the stage runs minutes into a run,
  `git status --short` counts untracked files, and a commit landing in
  the worktree mid-run therefore kills the run.

Evidence: crop-versus-direct bitwise identity re-proven for a corridor
built through the HRRR chain's own grids and catalog, 5 placements x 14
fields, with the instrument validated by a planted single-ULP flip
caught as exactly one differing word. Emission determinism, the
receipt-bytes round trip, a tampered-cache refusal, and the real
`prepare_hrrr_hierarchy` sealing and binding. Live on a real 18Z HRRR
cycle: both moves executed off the sealed corridor, the parent bitwise
unchanged, 42 of 42 fields bit-identical against direct builds, and the
`--estimate` figure equal to the sealed bytes exactly.

## 1.8.5 (2026-08-08)

Fixed:
- HRRR no longer publishes a negative 2 m mixing ratio over high
  terrain. A nested 12-3 km tree over eastern Colorado cleared both
  preparation stages on 1.8.4 and was then refused by the tree
  forecast's own input check, `prepared near-surface surface_qv is
  outside the physical range 0.0..0.2`. The guard was right and the
  data was WRF-faithful, which is why nothing upstream caught it.
  HRRR's GRIB2 packing quantises 2 m specific humidity to 1e-5, so
  over source orography of 2557..3535 m it decodes to exactly zero
  beside neighbours three orders of magnitude larger; METGRID.TBL
  routes SPECHUMD through the overshooting `sixteen_pt` operator,
  which undershoots such a stencil below zero, and
  `module_initialize_real.F` converts it with `qv_gc = sh_gc/(1 -
  sh_gc)` and no floor before assigning `grid%q2` verbatim. Two child
  cells came out at -1.79e-05 and -1.48e-05. Nested GFS completed the
  same placement only because its lane builds the same quantity
  through `_saturation_mixing_ratio`, which floors at WRF's
  `qv_min_value` inline; the specific-humidity lane had no floor at
  all, and that asymmetry is the whole defect, since one guard written
  against the floored lane's contract was applied to both. The
  published `surface_qv` is now floored at that same `qv_min_value`,
  at the publication point and nowhere else: `sfcprs2`, `integ_moist`
  and the vertical-interpolation pseudo-level are uses real.exe is
  defined for and keep the raw value, so no prognostic field moves on
  any lane, including the explicit `use_sh_qv=True` lane whose
  prognostic qv is the interpolated surface value. Cells at or above
  the floor are byte-untouched, so no existing artifact changes; 1 of
  63 arrays moved in the refusing tree's cache and a control tree was
  byte-identical across all 270. The divergence is recorded on
  `RealInitResult.surface_moisture_floor` and printed, like the
  soil-moisture floor beside it. Single-domain HRRR carried the
  identical latent defect on the same lane behind the same guard, and
  was unexposed only because the shipped cell runs over flat Oklahoma;
  a 3 km root over the same Colorado terrain reproduces both negative
  cells. The near-surface refusal now names the offending cell count
  and the observed extremes, because reading them back off the old
  message cost a full re-run of a two-stage preparation.
- The two `km_opt` registry citations point at the LES-closure
  docstrings again. 1.8.4 shipped with `tests/test_physics_registry.py`
  red: both anchors in `gpuwm/core/dycore.py` were 29 lines stale
  because the solver perf work added the cached `couple_momentum`
  kernel well above them. The code moved, not the claim, and the two
  want opposite fixes, so it was checked before repointing: both
  docstrings are byte-identical between 1.8.3 and 1.8.4, as is the
  whole of `dycore.py` from the first anchor to end of file. Only the
  line numbers move, 824 to 853 and 914 to 943, in the registry
  warning text and in the checker's paired RESOLVED rows, which must
  move together or the checker fails from the other side. The citation
  checker now reports 85 citations with 0 failing, and the curated
  first-stage battery carries that test so the next drift is caught by
  a release gate rather than by a sweep.

## 1.8.4 (2026-08-08)

New:
- `gpuwm run-plan` runs nested HRRR end to end, the last route that
  refused. An HRRR tree is not a GFS tree with a different source:
  rw-wps is not on the path at all, the root preparation is
  `tools.prepare_hrrr_wrf`, and a third stage builds d02..dNN from the
  sealed d01 before the same `gpuwm-prepared-tree-forecast` the GFS
  tree route already drives takes over. The branch is taken after the
  root preparation, which both HRRR depths share, so neither fetch nor
  that preparation is duplicated. The hierarchy writes `receipt.json`
  where rw-wps writes `proof.json`, and the document resolver added
  for the GFS tree matches on schema rather than filename, so the
  receipt relay needed no new machinery. Render is now one helper
  shared by both HRRR arms, so `render_products` -- including `none`
  -- means the same thing however the forecast was produced.
- Moving nests run on the prepared routes. A prepared tree is
  digest-bound and carries no ingest inputs, so a `[relocation]`
  follow source had nothing to rebuild child statics from and the tree
  runner refused it outright. `--statics-corridor` closes the gap at
  preparation time, when the geography source is still on hand:
  parent-extent child-resolution statics per child, built through the
  same `build_static` the domain statics use, written
  byte-deterministically and sealed with a SHA-256 receipt. Corridor
  crops are bitwise the direct footprint build, pinned by a
  crop-vs-direct identity test across the build's gcell, categorical,
  interpolation and smoother paths with a planted one-ULP
  perturbation proving the instrument fires. At forecast time a
  follow source is accepted only over a verified corridor (receipt
  against proof, cache digest, geometry and grid-arithmetic probes);
  a corridor-less bundle keeps the refusal and now names the flag,
  and a failed verification refuses loudly rather than running a
  silently static nest. Opt-in throughout: without the flag the
  bundle is byte-for-byte what it was. `gpuwm go` and run-plan pass
  it whenever the config declares a follow source, and the printed
  rw-wps chain derives it from the same predicate, so the pasted and
  the driven chain cannot drift on it.
- `prepare_mapped_wrf` accepts a prebuilt native static cache
  (`--static-input` / `--static-receipt`). It was the last direct
  adapter with no such bypass: it called `build_static` every cycle
  and then serialized the exact artifact it had no way to read back.
  The receipt is verified against the resolved cache and the target
  grid before the load, and the proof records `root_static_provider`
  as `prebuilt-hash-bound-cache`.
- The solver step and the preprocessing that feeds it are faster, with
  every output byte unchanged. The surveyed arm took the
  representative full-physics trace from 41.345 to 37.514
  ms/model-step, and a fused coupled-scalar update, a fused
  `couple_momentum`, a Morrison active-span substep sweep and a
  chunked state gate landed on top of it, for about 13% off the step
  in total. `validate_full_state` on a seeded 250x200x49 mp10 state
  went 5.25 to 0.52 ms per check. `build_static` against the
  reference WPS_GEOG tree at d01 251x201 went 18.701 to 5.359 s
  (3.49x), which pays on every path a static cache cannot serve: the
  cold first cycle, geometry changes, moving and spawned nests, and
  child domains. The byte proofs: forecast SHA-256 unchanged on the
  seeded mp10 and dry lanes, ValidationReports bitwise identical
  across a 14-case corruption-injection battery, and all 14 static
  field digests identical in every arm.

Fixed:
- Nested GFS preparation no longer refuses on inland water. An
  ordinary 12-3 km tree died 15.9 s into its prepare stage with 73
  columns disagreeing between land mask and soil category, and the 73
  were reservoirs and rivers scattered across the whole child rather
  than an edge artifact, so this refused essentially any CONUS child
  holding inland water, moving nest or not. The soil-temperature
  lookup was an inline chain that did not know the GFS spelling; it is
  now one table of per-source spellings read by both reconciler call
  sites, the nested child and the root case. The SST argument had no
  fallback and the GFS lane carries no SST field, so WRF's second arm
  could never fire; it now takes real.exe's own precedence, SST then
  the skin temperature that `module_initialize_real.F` substitutes
  where SST has no valid support. The reconciler's soil-temperature
  and SST reads also went straight to `np.asarray` on values that can
  arrive device-resident, which CuPy refuses; both are marshalled to
  the host now, like the rest of that module. That refusal was
  unreachable until a column actually disagreed, so it sat behind the
  reconciler's own hot path instead of failing at setup.
- A chain stage no longer imports from the caller's directory. Every
  prepared-chain stage is spawned as `python -m MODULE`, and `-m`
  prepends the current directory to `sys.path` ahead of the installed
  package, so a chain started inside a source checkout imported that
  checkout; a live run was hijacked exactly that way. The stage cwd is
  deliberately the caller's directory so a relative `--out` means what
  the person typing it meant, so moving it was not the fix.
  `PYTHONSAFEPATH` separates the two: the child still resolves paths
  against the caller's directory and no longer imports from it.
- Progress telemetry reports real numbers on every prepared route.
  `speed_x` was null and `wall_seconds` 0.0 on all 181 progress events
  of a completed run, because the baselines were armed on the stage
  transition and every prepared chain opens the forecast stage itself
  before handing the runner over. They arm on the first progress call
  now, which is true however the stage was opened.
- A completed tree run no longer summarizes itself as zero. The
  summary knew only the single-domain runner's `progress.json` and
  `report.json`; the tree runner writes a certification capsule and
  neither of those, so a run that finished 10800 model seconds and
  wrote 17 frames reported `completed_seconds` 0.0, and the heartbeat
  fed from it published a model time of zero beside an `outer_step` of
  180. Completed seconds now come from the observer, which saw every
  step on either route, and the capsule is read for frames and named
  in the summary. Where no receipt states a status, status stays null
  and `status_basis` says why.
- `gpuwm run-plan --resolve` no longer reports an authored table as a
  schema default. It decided "the author did not type this" by looking
  only inside the `[experiment]` table, but several fields are
  authored as their own top-level table, so a config declaring
  `[relocation]` was reported with the schema's `enabled: false`. For
  a moving-nest plan that reads exactly backwards. A top-level table
  whose name is a field name now counts as the author spelling that
  field, taken from the document rather than from a second list.
- `relocation_receipts.json` carries one run-end summary row. On a
  leg-walking route the executor returns once per leg, so a live
  12-move run left two byte-identical summary rows in a 25-row list.
  The superseded row is dropped and the new one appended, so the
  summary stays last and current.

## 1.8.3 (2026-08-08)

New:
- `gpuwm run-plan` runs nested GFS end to end. A multi-domain config on
  the `prepared` route keeps the same preparation stages (rw-wps builds
  the whole hierarchy in one call) and dispatches the forecast to
  `gpuwm-prepared-tree-forecast`, with run-plan owning the receipt
  relay the manual chain needed a person for: the sha256 of the
  hierarchy document rw-wps left in the prepared root, matched on
  schema against the tree runner's own table, plus the experiment
  config's own digest. The interactive `gpuwm go` command keeps its
  tree refusal; only run-plan, which dispatches to the tree runner,
  drives trees. Multi-domain HRRR remains refused with its by-hand
  chain named.
- `gpuwm run-plan --estimate` reports `peak_envelope_bytes` beside the
  pool request. That is the tree-aware figure: it adds a per-nest term
  and, on WDDM, the measured footprint floor. A tree priced on the
  pool request alone reads as fitting a card it does not fit.
- `model_progress` events carry each grid's own clock on a domain tree
  (a `domains` list of `{domain, model_seconds}`). Absent on a single
  domain, where the root is the tree, so existing consumers see the
  stream they always saw.

Fixed:
- Native HRRR prepared runs complete on Windows. The prepared-cache
  identity keyed source digests by `str(path.relative_to(...))`, which
  is backslashed on Windows, and the reader looks names up by
  forward-slash constants, so the forecast handoff always failed with
  "decode identity omits [...]" on a cache that was in fact complete.
  Every producer now writes POSIX keys and the validator reads both
  spellings, so caches already sealed on Windows under 1.8.2 stay
  restorable. A dict carrying one path under both spellings is refused
  rather than merged.
- `sealed_extension_fingerprint` no longer differs between Windows and
  Linux for identical code: the tree runner's runtime source identity
  used OS-native separators in its keys. Linux fingerprints do not
  move; Windows converges onto the existing value.

## 1.8.2 (2026-08-08)

1.8.1 was tagged but never published: its release workflow's own test
job failed on the runner, so PyPI received nothing and the release was
withdrawn to a draft. The tag stays where it is, because tags here are
forward-only. Everything 1.8.1 carried ships in this release, plus the
fix below.

Fixed:
- `gpuwm doctor`'s printed report no longer depends on whether the box
  running the tests has an NVIDIA driver. The report suite substituted a
  stand-in for `sys` that carried two attributes, and the CUDA-major
  remedy added in 1.8.1 reads a third (`sys.platform`). Whether anything
  reached it depended on `GPUWM_NO_LOCAL_GPU`, so the gap was
  unreachable on a developer box and fatal on a driverless runner. The
  stand-in now delegates every attribute it was not deliberately given
  to the real module, and the report test pins the driver arrangement
  itself instead of inheriting the host's.

## 1.8.1 (2026-08-07, tagged but not published)

New:
- `gpuwm run-plan` reaches HRRR. The `prepared` route takes
  `source = "hrrr"` for a single domain, so the credential-free machine
  path is no longer GFS-only. 1.8.0 shipped this route explicitly scoped
  to GFS and said so; the preparation side is what had to change, not
  run-plan.
- A plan chooses what to render. The `render_products` run option takes
  the product names a run should draw instead of drawing the default
  set, so a fleet controller asking for four fields does not pay for the
  whole catalog.
- `gpuwm run-plan --catalog` prints the render catalog as one JSON
  document and runs nothing, the same shape as `--resolve`,
  `--estimate` and `--probe`. A front end offering a product picker can
  populate it from the build rather than from a list it maintains.
- `gpuwm version` says which code is actually running: the import path,
  whether it resolves inside a checkout or a site-packages install, the
  distribution version, and whether pip would move it. An install whose
  behaviour does not match its version is the first thing to rule out in
  a bug report, and it was the one thing no command reported.
- The GPU install extra names its CUDA major. CuPy ships one wheel per
  CUDA major and the wheel must match the major the box serves: a
  `cupy-cuda12x` wheel on a CUDA-13-only box imports cleanly, compiles
  kernels, and then dies at the first cuBLAS load. A pip extra cannot
  detect a CUDA major, so this does not pretend to choose one:
  `gpuwm[gpu-cu12]` and `gpuwm[gpu-cu13]` name it, with `[all-cu12]` and
  `[all-cu13]` beside them. `[gpu]` and `[all]` still resolve to cu12,
  as aliases rather than a third policy, because every install that
  already works names them.
- `gpuwm doctor` reads the box's CUDA major straight off the driver,
  with or without CuPy installed, and prints the extra that matches it.

Fixed:
- A supervisor guard refusal reads as a refusal. A run refused by a
  physics or configuration guard surfaced as a crash capsule, so the
  headline a user saw was a stack trace rather than the sentence the
  guard wrote for them.
- A `[relocation.follow]` cadence that the reflectivity stash cannot
  serve is refused when the config loads, not at the first cadence that
  needs the fallback. The tracker's composite-reflectivity fallback
  reads a plane the microphysics stashes at history cadence, so a follow
  cadence off that lattice ran fine until the first evaluation where
  updraft helicity was under threshold and then refused mid-run.
- A two-phase DA run that stops early still writes the receipt its
  verifier needs, so the run can be graded instead of being unscoreable.
- A domain tree that cannot be resolved is left to its own validator
  rather than being reported by the tracker, which is not the component
  that knows what is wrong with it.
- An island domain whose source land-sea mask rounds all its land away
  now initializes from the real land fraction instead of refusing. WPS's
  `make_zero_or_one` binarizes the mask, so a sub-grid island whose land
  fraction never reaches 0.5 anywhere in an all-ocean crop left the
  binarized donor set empty over the whole domain, and the nest took a
  METGRID-fill `TSK = 0.0`. The fall back to the discarded fraction fires
  only when that donor set is empty across the entire crop, and it says
  so in a per-domain receipt. The nonphysical-TSK refusal that used to be
  the only symptom now names the offending cells and the fill that
  produced them. The CONUS reference domain is byte-identical across this
  change.

## 1.8.0 (2026-08-07)

New:
- Storm-following moving nests. A `[relocation]` table, default off,
  moves a child domain by a whole number of parent cells at cycle
  boundaries. A moving nest here is a sequence of static nests joined by
  a re-grid, not WRF's continuous per-step motion; those namelist keys
  stay refused and now point at the discrete mechanism instead of
  saying "non-goal". The overlap between the outgoing and incoming
  footprints is transplanted bitwise, because a whole-parent-cell shift
  leaves every overlapped cell's donor index and sub-cell offset
  unchanged: measured 0 mismatches over 18,714,960 cells across 25
  restart-contract fields on an RTX 5090, with the treatment proven by
  56 % of those same cells differing from a cold start. A restart across
  a move promises nothing, by ruling, and the relocation bounds stay out
  of the restart fingerprint.
- `[relocation.follow]` decides the WHEN and the WHERE from the running
  model's own fields. Updraft helicity (`up_heli_max`, WRF's
  UP_HELI_MAX, reset each history interval) is the primary vote, because
  rotation is what a tornado nest exists to follow; column-max
  reflectivity is the fallback for the window before a mesocyclone
  exists. Three configured guards keep a jittering centroid from
  spending the run on relocation spin-up: a dead band below
  `min_shift_cells`, a per-event clamp at `max_shift_cells`, and a
  cooldown that burns on executed moves rather than proposals. Every
  cadence appends a receipt carrying the centroid evidence and the
  decision, so a run's move/hold history is auditable afterwards. Every
  follow key is required and unknown keys refuse.
- A relocated nest rebuilds its statics for the new footprint, at nest
  resolution, from the nest's own static source rather than inheriting
  parent-interpolated terrain: over-land storm-following at 2 km lives
  on resolved terrain. The footprint grid comes from an exact
  whole-cell translation of the reference grid, so a cell two placements
  share evaluates to bitwise-identical coordinates and therefore
  bitwise-identical statics; the preparer asserts that equality on every
  move rather than assuming it, which is what lets the bitwise overlap
  transplant survive a statics rebuild. The atmosphere on the fresh
  strip is adjusted to the rebuilt terrain by the same sequence a t = 0
  real child runs, and land state moves by landmask-aware donor fill.
- Off-centre nest placement is proven placement-independent, in both
  directions. The parent-to-child exchange geometry carries no centred
  placement assumption: force tables are exact on linear fields at every
  quadrant and at the minimum legal edge margins for ratios 3 and 5, the
  residual tables are identical between centred and off-centre
  placements, and the whole force transaction is translation-equivariant
  at the bit level. The one genuine concentric-nest assumption in the
  tree was an LES case module's own placement instrument, and it now
  uses the runner's resolver. A relocated nest is off-centre by
  definition, so this is the moving nest's standing regression bed.
- Spawn-at-trigger nests. A `[[domain]]` can declare a `spawn` block --
  `trigger = "uh" | "reflectivity" | "time"` with a threshold and a
  model-time window -- and stay dormant until the trigger fires, at
  which point it materializes at the tracker-chosen position and follows
  the storm from then on. The nest is pre-declared, not mid-run
  allocated: the memory plan prices it exactly as if it existed, so
  `gpuwm check` refuses honestly at planning time and prints one
  advisory line per dormant nest naming what its declaration costs.
  Spawning is activation, not allocation. A declared-but-never-triggered
  nest costs its reserved VRAM for the whole run and zero compute; that
  is the contract, not a leak.
- The fired nest gets own-grid statics for its footprint, an atmosphere
  interpolated from the CURRENT parent -- so it is born inside the storm
  its trigger saw, not inside a stale analysis -- and terrain adoption
  byte-for-byte the real-data child adjustment sequence. Placement is
  the storm-core weighted centroid around the loudest qualifying cell in
  the search box, so with two storms in one box it lands on the stronger
  storm rather than between them, and a watch ignores signal inside
  another active nest's footprint.
- A nest born mid-run takes its land surface state the way WRF gives one
  to a nest with no `wrfinput` of its own: parent interpolation through
  the mask-aware operator WRF names per field in its Registry, so a
  mixed land/water cell averages only the corners matching the child's
  own land-use class. That is the operator WRF runs at every nest birth
  and at every moving-nest leading edge. ArWen keeps the half of
  `fine_input_stream = 2` it can have, the own-grid statics, which is
  what makes the mask load-bearing: the child's categories are resolved
  at the child's dx and disagree with the parent's wherever finer
  terrain resolves a coast, lake or island the parent smoothed away. The
  receipt counts every branch per mask family and publishes the
  island/lake fallbacks. This lifted the real-data spawn refusal on the
  run route; routes that neither reserve nor watch still refuse a spawn
  declaration by name rather than dropping it.
- `gpuwm run-plan`, a machine interface. One versioned JSON plan in, one
  append-only JSONL event stream out, so a GUI, a scheduler or a fleet
  controller drives gpuwm as a subprocess and never parses a line of
  human output. The plan is an envelope over the existing config system,
  not a second config format: it resolves through the same loader
  `gpuwm run` uses and executes through the same runtime. Unknown
  top-level keys, unknown run options, an unknown route and an unknown
  schema are all refused rather than ignored, and `fetch.args` is
  validated by building gpuwm's real fetch parser.
- `config.intent` submits a shape instead of a config -- a point or
  polygon, a ladder, a source, a cycle, a card budget -- and the `gpuwm
  domain` wizard writes the complete TOML, the same wizard with the same
  refusals and the same emitted bytes a person gets from the CLI. Every
  intent key is a wizard flag, one to one. The generated TOML is carried
  verbatim on the `resolved_plan` event, because the caller never typed
  it and it is the one thing they cannot look up.
- Every value the pipeline chose on its own appears in
  `automatic_resolutions` with its basis, one entry each, before the run
  starts. The per-domain time step is the flagship case: it changes the
  model's answer, nobody writes it, and until now it appeared only
  inside a printed report. `cycle: "latest"` is resolved to a concrete
  cycle before the fetch runs and the resolved value is what gets
  recorded, so a plan does not archive a question whose answer changes
  every six hours.
- `--resolve`, `--estimate` and `--probe` each print exactly one JSON
  document and run nothing. They work before anything is downloaded, and
  they carry the full declared-input inventory instead of skipping the
  existence check silently. `--probe` reads the device inventory through
  NVML only, creating no CUDA context, so it is safe to poll on a busy
  card. Where this package has no measured number the field is `null`
  with its basis stated, rather than an invented one under gpuwm's name.
- A run-plan run is reattachable. `run-manifest.json` is written before
  any work starts and names every stream a consumer may want, including
  the two run-plan does not own. run-plan publishes no progress state of
  its own: the supervisor stays the only writer of `run-progress.json`,
  the run-plan observer composes with the existing heartbeat rather than
  replacing it, and a run-plan run leaves exactly the heartbeat a `gpuwm
  run` leaves. `sequence` on the event stream is monotonic and dense, so
  a gap means a lost line and the reader refuses it; a torn final line is
  refused rather than trimmed.
- `gpuwm domain --history-interval` and `--nest-history-interval` make
  output cadence a first-class control. They were bare literals with no
  knob. A cadence must be a whole number of seconds and a whole number
  of that domain's own time steps, judged against the exact rational
  `dt`, and the wizard round-trips its emitted bytes through the real
  loader before writing, so a bad value is refused with the loader's own
  sentence and no file lands.
- The HRRR route stages microphysics lookup tables for every profile
  whose microphysics reads them, at profile-binding time -- before the
  fetch and before preprocessing -- through the packaged table ladder,
  SHA-256 checked. It resolved tables for one profile behind two
  environment variables before, which is why its default had to be a
  suite that needs none. The resolved set is recorded in the physics
  receipt as `microphysics_table_authority`.
- `gpuwm domain --source hrrr` defaults to
  `thompson-mp8-ysu-mm5-noah-rrtmg-legacy-v1`: Thompson microphysics,
  RRTMG longwave and shortwave, no cumulus at 3 km. That is the
  operational HRRR composition (NOAA/GSL CCPP `HRRR_suite`); gpuwm
  diverges on YSU rather than MYNN-EDMF and Noah rather than RUC.
- The wizard offers both legacy-RRTMG Thompson suites. It offered
  neither, which is why no full-radiation suite could be an HRRR
  default.
- A sizing refusal names a lighter `--physics-profile` beside the
  shallower ladder and the bigger card.

Fixed:
- No door emits an asymmetric radiation pairing (shortwave on, longwave
  off) as a default. This was corrected to exclude HRRR in 1.7.1; it is
  now true everywhere, and the claim's test covers all three sources
  instead of two.
- The HRRR route no longer writes a nocturnal experiment declaration on
  the operator's behalf for its own default. A declaration by silence
  declares nothing. Explicitly selecting an asymmetric suite for a night
  window still writes it, which is what it always should have meant.
- A bare interactive `hrrr` session emitted a config its own route
  refused (`d01 cu_physics=1`, Kain-Fritsch on a convection-permitting
  grid). Its covering test checked registry reachability and maturity,
  both true of that suite, and never ran the route's physics gate; it
  does now.
- An HRRR config carrying a legacy-RRTMG suite could not be emitted at
  all: the emission round-trip re-imported its namelists without the
  RRTMG variant, resolved 4/4 to RTE+RRTMGP, and reported the
  difference as though the emitted files were wrong. It inherits the
  variant from the authoritative experiment, as it already did for
  acknowledgements.
- A wizard header called every `ra_*_physics = 4` suite RTE+RRTMGP,
  including the legacy-RRTMG ones.
- An installed-but-unloadable CuPy no longer kills the command line. An
  absent CuPy raises `ImportError` and was always survived; the
  documented rental trap -- a `cupy-cuda12x` wheel on a CUDA-13 box, a
  missing NVRTC DLL -- raises `RuntimeError` or `OSError` from deep
  inside the import, so `gpuwm` died on exactly the installs
  `run-plan --probe` exists to diagnose. It is deferred, not swallowed:
  the original error is kept and named at the use site, with the
  wheel/CUDA-major mismatch called out, because installed-and-broken has
  a different fix from absent and must not get the "install it"
  sentence.
- A `prepared` run-plan run that succeeded reported `failed`. The
  chain's own exit convention was read as the run's.
- Nothing but JSONL reaches stdout on a run-plan run. The resolved-config
  report, the wizard's sizing table, warnings and the feedback advisory
  all go to stderr, enforced by binding the real stdout for the event
  stream and redirecting `sys.stdout` for the whole run, rather than by
  asking every printer to behave.
- `gpuwm go` runs its forecast in process, so the documented chain is
  observable end to end instead of going dark inside a subprocess.
- Output cadence no longer steers a storm-following nest. The tracker and
  the spawn watch read updraft helicity, and they had been reading WRF's
  `UP_HELI_MAX` diagnostic, which is zeroed by the history writer. Its
  accumulation window was therefore the output interval, so changing
  `history_interval_s` changed the peak and the centroid the tracker saw,
  and the nest went somewhere else. Each consumer now owns a window
  folded from the same cal_helicity columns in the same pass and reset by
  that consumer at every evaluation, accepted or held, so the window is
  exactly "the strongest rotation since I last looked". Two windows, not
  one, because relocation evaluates on cadence boundaries and spawn on
  leg boundaries: a shared window would let whichever consumer read first
  blind the other. Thresholds keep their meaning, and the shipped demo
  configs are unaffected because their cadences were already aligned.
  The windows are not restart-carried: a restart starts them empty and
  the first evaluation after it may under-read, the same posture already
  ruled for a restart across a move or a spawn.
- The offline child route refuses `ra_rrtmg_variant='rrtmg_legacy'` with
  `o3input = 2` instead of silently using the wrong ozone. That pairing
  means "take ozone interpolated from the parent", and this route has no
  resident parent; it was building the radiation adapter without one,
  which made it evaluate a fresh climatology on the child's own latitudes
  and then report `"ozone_routing": "root-climatology"` for a nested
  domain. `o3input = 2` is the default and `gpuwm downscale` copies it
  from the parent, so this was the ordinary path. The refusal names both
  remedies.

## 1.7.1 (2026-08-06)

Fixed:
- The wizard's real-case default physics is
  `morrison-mp10-ysu-mm5-noah-kf-rte-rrtmgp-v1` on the gfs/era5 doors:
  the registry's user-ready `wrf-matched-run` template with both
  radiation streams on, the profile FIRST-LIGHT's worked example and
  the interactive session already used. It replaces the unshipped
  "product default suite". The HRRR door keeps its route-constrained
  WSM6 default. Neither the gfs nor the era5 door emits an asymmetric
  radiation pairing (shortwave on, longwave off) as a default. (The
  HRRR door was fixed in the following release; see Unreleased.)
- A real experiment whose window includes local night refuses to load
  with `ra_sw_physics > 0` and `ra_lw_physics == 0`, naming the
  physics, the matched profile, and both remedies. A field report
  proved the failure: a wizard-emitted 48 h case bound
  `thompson-mp8-ysu-mm5-noah-validation-v1` (longwave OFF, shortwave
  Dudhia), shortwave heated the surface by day, nothing balanced the
  surface's upward longwave at night, skin temperature cratered, and
  2 m dewpoints read in the 50s F inside a 70s airmass. The guard
  lives in the one config load every front door shares
  (`gpuwm.experiment.build_experiment`), so `gpuwm run`, `gpuwm go`,
  `gpuwm check`, both prepared runners, the DA drivers and the
  wizard's own sizing loop all refuse identically. Existing configs
  carrying the pairing refuse at load with the remedy named.
- Asymmetric pairings stay selectable, loudly: a daylight-only window
  loads unguarded, and a night window runs as a declared experiment
  with `acknowledgements = ["asymmetric-radiation-nocturnal-window-v1"]`
  in `[experiment]`. The wizard writes that declaration itself when an
  asymmetric profile is selected explicitly for a night window, so the
  file it emits still loads everywhere.
- The native HRRR route writes the same declaration. Its default suite
  is asymmetric by route constraint -- the HRRR root preparer stages
  no microphysics tables for the full-radiation profiles, and eight of
  the thirteen profiles it accepts run Dudhia shortwave with longwave
  off -- and it builds its experiment in code rather than from a config
  file, so it declares a night window itself and the experiment
  authority it publishes carries the line. Five full-radiation profiles
  remain selectable on that route for a nocturnally valid HRRR run.
- Every wizard-emitted config header now states whether its suite is
  nocturnally valid, and `docs/public/PHYSICS.md` carries a
  per-profile nocturnal-validity table.

## 1.7.0 (2026-08-06)

New:

- Radar data assimilation for nowcasting, as a product surface. A
  WoFS-style ensemble Kalman filter cycles an ensemble against live
  observations and then runs free forecasts from the analysis.
  `gpuwm-da-nowcast run --da full` selects the certified full-stack
  configuration in one flag: radial velocity, reflectivity, clear-air
  echo, velocity dealiasing, surface observations and GOES cloud water
  path. `--without <stream>` subtracts one stream from that preset, so
  a denial experiment is one word rather than a hand-assembled flag
  list, and `--da vr` is radial velocity alone. The default stays
  `custom`, meaning the individual flags remain the whole story for
  anyone already driving them.
- Every assimilated stream is counted. The cycle records how many
  observations each stream actually contributed, and a full-stack claim
  is refused when a stream contributed nothing, so a run cannot report
  a configuration it did not perform.
- Velocity dealiasing on the radial-velocity path. Folded gates above
  0.8 * Nyquist carry a mesocyclone's couplet, and masking them removes
  the signal the analysis is for. Gates the unfolder cannot resolve are
  still dropped and counted. The same unfolding is applied to the
  verification composites, so a run is graded against a truth field
  built the way it was fed. Needs the new `[dealias]` extra.
- GOES cloud water path is assimilated beside radar, from the model's
  own column integral, and an observed-clear column can remove cloud
  the model invented.
- Verification against MRMS. `--truth mrms` grades a run against the
  national mosaic rather than against a composite assembled from the
  same radars that fed it, which is the honest grader for a domain
  wider than one radar's reach.
- HRRR is the default background for the nowcast route.
- Adjusted initial conditions: `[[perturbation.bubbles]]` in the
  experiment TOML adds warm theta bubbles to a real-data initial state
  at prepare time -- WRF's idealized cosine-squared bubble, placed in
  geographic coordinates, with optional `rh_preserve` qv adjustment.
  Applied per domain through each domain's own init, so a bubble
  inside a nest arrives on the nest's own grid. An absent block is
  byte-inert; refusals are loud (out-of-domain center, nonpositive or
  oversized amplitude, a bubble that touches zero cells); what was
  actually written lands in `initial-perturbation.json` per domain.

- Warm bubbles. A `[perturbation]` block applies theta bubbles to the
  initial state, on the single-domain route and on the prepared
  domain-tree route. An absent block changes nothing at all, including
  the restart fingerprint.
- High-resolution static geography for US interior domains: 10 m
  terrain, 30 m land cover and 250 m soils, fetched as on-demand tiles
  with recorded provenance. Off by default. Coastal domains refuse, or
  fall back to the 30 arc second baseline when asked to.
- The render vocabulary grows from 55 products to 164, with proper
  colortable resolution: an operational table where the product has
  one, then a curated table, then a generic full-finite-range style.
  Any stored 2-D plane with no named product renders as `var:<name>`.

Fixed:
- The turbulence selector `km_opt` is a registry component carrying an
  option for every value it accepts, including 0. A plan selecting the
  SASE closure, which requires `km_opt = 0`, previously failed to
  resolve the component at all.
- `km_opt = 3` is admitted with the PBL off only. Its vertical exchange
  pair is applied by a PBL-off-gated operator, so with a PBL scheme on
  only the horizontal half of the closure ran, under the name of the
  3-D one.
- The registry and `validate_run_config` now agree about every one of
  the 46,080 registered component combinations. They disagreed about
  3,944 of them: 3,360 where the registry refused `km_opt = 3` under a
  PBL and the loader did not, 512 where the SASE closure was paired
  with the MYNN surface layer that WRF admits only under the MYNN PBL
  or none, and 72 dry SASE plans the registry called launchable and the
  loader refused.
- An enabled `[static.highres]` block on a route that cannot apply it
  refuses by name instead of silently producing the baseline.
- The native-HRRR static verifier accepts a whole-second rational-clock
  payload, which it had been refusing outright.
- The nested-domain child soil reconciler reads the native lane's soil
  temperature field.

- Large-eddy runs with WRF's anisotropic mixing lengths no longer go
  unstable and abort. With `mix_isotropic = 0` the mixing lengths are
  per-axis, so on a grid whose vertical spacing is far finer than its
  horizontal spacing the vertical exchange coefficient is applied
  through a horizontal operator and amplifies the 2dx vertical-velocity
  mode until the run fails. The condition is now stated as a criterion
  on `mix_upper_bound * (dz_max / dx)^2`, a configuration that violates
  it warns by name at setup and says which of the two knobs to move,
  and `mix_isotropic = 1` mixes on one length instead of three.
- CUDA kernel compiler version is part of the certified numeric
  fingerprint. A silent NVRTC downgrade moved the CUDA column's
  measured distance from WRF on two fields with no source change, and
  nothing in the capsule recorded it. The capsule now carries the full
  four-part NVRTC build, its internal changelist, and the SHA-256 of
  the loaded NVRTC library; the shinhong parity baselines and the
  off-path byte pin are recorded per NVRTC build rather than as one
  row.
- `--surface-obs` without a sigma carried the file and assimilated
  nothing.
- The dealiasing account is a per-radar list, so a multi-radar run
  reports each radar's unfolded and dropped gate counts instead of one
  collapsed number.
- scipy is declared in the new `[dealias]` extra, which the velocity
  dealiaser needs. It was imported but undeclared.

## 1.6.3 (2026-08-06)

New:

- An optional high-resolution water-temperature overlay for the ERA5
  route. ERA5's water-surface temperature is a coarse analysis, and
  over lakes and along coasts it paints through to the near-surface
  fields as blocky dewpoint and temperature artifacts. This is a
  well-known WRF + ERA5 limitation, not something specific to this
  model's ingest; docs/water-temperature-overlay.md cites the community
  threads and their remedies. The overlay is the high-resolution
  substitution remedy implemented natively: a user-supplied gridded
  water-temperature analysis replaces ERA5 SST and SKINTEMP over water
  source cells before horizontal interpolation, on the direct adapter
  route and on `gpuwm run`, with per-snapshot replacement counts
  reported. Off by default; configured on nothing, this code never
  runs.

Fixed:

- Certification capsules on CUDA-13 boxes no longer report the CuPy
  version as unavailable. The capsule pinned the literal distribution
  name cupy-cuda12x; it now records whichever CuPy distribution pip
  actually installed (cupy, cupy-cuda12x, cupy-cuda13x).

## 1.6.2 (2026-08-06)

New:

- A `[gpu-cu13]` install extra for CUDA-13-only machines. The existing
  `[gpu]` extra stays on CuPy's CUDA-12 wheel, which a CUDA-13-only box
  imports cleanly and then fails at its first cuBLAS load, deep inside
  a run. `gpuwm doctor` now performs that first load on purpose, in an
  isolated subprocess probe, and when it fails it reports the installed
  wheel's CUDA major, the box's, and the exact pip commands that put
  the matching wheel on. The load is the judgment, not the version
  numbers: a newer driver serving an older wheel is a working install
  and is never refused. Under `GPUWM_NO_LOCAL_GPU` the probe does not
  run and the report says the pairing went unjudged.

Fixed:

- Shoreline land cells next to water no longer initialize far below
  wilting-point soil moisture on the ERA5 route. Ingest built the soil
  column with a different soil category than the land model then
  integrated: the ingest/land-surface category reconciliation ran after
  the soil moisture floor had been applied, so a coastal cell whose
  category changed kept a floor that belonged to neither category. One
  rulebook now resolves the reconciled category before the soil column
  is built, on the root domain and on nest initialization, and the
  floor reads the reconciled category's own air-dry value. On the
  domain this was verified on, every affected cell initializes at its
  category's air-dry value and no other cell moved.
- The demo/gallery renderer's shapefile reader (pyshp) is declared in
  the `[render]` extra. It was imported but undeclared, so a fresh
  install that followed the quickstart exactly could still crash at the
  first basemap it drew.

## 1.6.1 (2026-08-05)

Fixed:

- ERA5-forced runs on coastal domains could still die at step 0 with a
  non-finite tendency, which 1.6.0 did not fix. Where the land mask and
  the soil category disagree, an ordinary occurrence at a coastline, a
  land cell carries the water soil category. That category has no
  air-dry value, so 1.6.0's air-dry floor skipped exactly those cells
  and left soil moisture at 0.0, which Noah's thermal conductivity then
  divides by. On the reported domain 192 cells were in that state, and
  are now none: a land cell whose category has no air-dry value takes
  WRF v4.6.1's own 0.005 constant, and real land keeps its category's
  air-dry value as before. A run also refuses at the start, by name,
  when a required forcing variable is absent.

## 1.6.0 (2026-08-05)

New:

- LES completion program, packages P1 to P4. Every verdict below was
  scored against acceptance bands registered before the runs existed.
  - Moist LES verification (P1): matched cloud-topped boundary layer
    comparisons against a pristine-source WRF oracle, with the
    saturation mutation control and both negative controls fired. 29 of
    32 metric-arm comparisons landed inside the registered bands; the
    two findings behind the three misses are published beside the
    passes, and no band was widened.
  - Vertical levels (P2): the LES level bound is now a compiled tier
    ladder. Every configuration at or below 128 levels compiles the
    kernel it always did, proven bit-identical; 160-level and 192-level
    runs carry a full device acceptance battery including restart
    bit-identity; above 256 the solver refuses loudly before any launch.
  - Inflow seeding (P3): an LES nest child can seed inflow turbulence
    with cell-block boundary perturbations, per domain, default off,
    and the off path is gated byte-identical to a build without the
    mechanism. The measured need is on the record: the shipped nested
    child's turbulence development fetch is larger than its own domain.
  - Gray-zone chain (P4): the scale-aware parent chain campaign ran end
    to end and ships with its findings section as the status of record.
    The gray-zone nest configuration ships unblessed and says so.
- Observation scoring instrument, first light. A registered reflectivity
  skill battery (fractions skill score against radar composites, frozen
  parameters, a measured persistence floor, and a wrong-day negative
  control measured on real observations) plus the scoring command line.
  Exactly one end-to-end case is scored: model S_refl 0.4618, above the
  persistence floor of 0.1598 and below the registered useful-skill line
  of 0.5249. This is first light for the instrument. No campaign ran and
  none is claimed.
- Radar data assimilation foundation, experimental. Ensemble
  initial-condition perturbations, radar observation operators,
  hot-start reflectivity increments, a batched LETKF gated by a twin
  experiment, a hydrometeor positivity policy, ensemble products, and a
  production NEXRAD Level II pipeline from archive bytes to filter-ready
  observation files. Every entry point stamps status experimental and
  nothing is wired into a default forecast route.
- WoFS at Home (WaH), a radar-DA nowcast that runs on one card. It is
  demo-grade, and that label is a measurement rather than modesty; the
  paragraph below it is what the label means. One command takes a
  WSR-88D site from Level II bytes through ensemble analysis to a
  rendered forecast gallery. An interactive launcher draws the domain as
  a box on a map instead of writing one. A daemon cycles the whole thing
  continuously and says on the page when it stops and why. The
  observation route reads the real-time chunk feed, which publishes
  bytes as they are collected rather than when a volume closes, and
  falls back to the archive when the live prefix is short; a finished
  volume is byte-identical on the two routes. Nested forecasts ride the
  same path, and the analysis carries its own Jacobi eigensolver so it
  does not need cuSOLVER present to solve.
  Exactly one case is verified. KDMX, ten members plus a never-analysed
  control, 3 km spacing, six 15-minute cycles and a 90-minute free
  forecast, 506 s for the whole three-hour exercise on one RTX 5090. One
  radar. Radial velocity only, which is the weaker half of radar DA,
  because reflectivity is what places and maintains storms. Two analysis
  variables, u and v. Fractions skill score 0.72 to 0.77 against 0.24 to
  0.34 for a cold start that never saw an observation: an internally
  controlled measurement against doing nothing, and not a comparison
  against persistence, optical-flow nowcasting, WRFDA, or the
  operational HRRR. One draw per arm, so no error bar. No dual-run byte
  comparison was made for any DA run, so the standing no-ECC corruption
  screen has not been applied to any of these numbers. No velocity
  dealiasing: fold risk is masked and counted, not unwrapped.
  [docs/da-nowcast-quickstart.md](docs/da-nowcast-quickstart.md) is the
  path a stranger can follow, and
  [docs/da-vs-wofs.md](docs/da-vs-wofs.md) sets the system beside
  Warn-on-Forecast line by line, including every line where it is
  smaller.
- Grell-Freitas cumulus, `cu_physics = 3`. A scale-aware deep and shallow
  convection scheme, selectable per domain on the tree route. Bitwise
  against WRF v4.6.1 at the whole-driver boundary over a committed
  216-column oracle: 208 columns word for word on the float32 CPU
  authority and on CUDA, the remaining 8 bounded at 34 ULP by the
  driver's own mixed-precision constants. Never a default and no
  template selects it. Not verified in any real-case forecast: no scored
  comparison against observations has been run with it, so the registry
  label is implemented-unverified with scientific evidence none.
  Selecting it costs 3.20 GiB of device local-memory backing store,
  priced by `gpuwm check` under NON-POOL, which today rules out
  four-domain configurations on a 32 GB card.
- OLR renders as synthetic satellite infrared in the matplotlib fallback
  renderer.

Fixed:

- ERA5-forced runs could die at step 0 with a non-finite tendency. An
  ERA5 soil-moisture value a hair below zero, from GRIB packing or
  interpolation undershoot, was clipped to exactly 0.0 on a land point.
  Noah's thermal conductivity divides by soil moisture three times, so
  one such cell produced NaN conductivity, then NaN ground heat flux,
  then NaN sensible heat flux, and the run failed blaming the PBL
  scheme. Each land layer below its soil category's air-dry value is now
  floored to that value before the derived liquid-water split, on every
  Noah-geometry source. Water, sea ice and healthy land are byte
  untouched: on a mixed land, water, ice and frozen domain all ten state
  arrays hash identically with and without the fix. Where WRF v4.6.1
  differs it is recorded rather than imitated: WRF resets a whole column
  to a constant on a top-layer trigger, this floors per layer at the
  category value, and on the case that was reported 109 of 142,644 land
  cell-layers differ as a result.
- The refusal that run produced named the wrong scheme. A non-finite
  surface heat flux handed to YSU leaves as a non-finite tendency, and
  the message named only that tendency, pointing at the
  boundary-layer scheme when the defect was in the surface layer feeding
  it. 1,474,560 fuzzed columns of finite input never produce a
  non-finite tendency at all, so when one appears an input is the cause.
  The refusal now names the degenerate input it is an image of. Which
  tendency carries it depends on surface stability: over a stable
  surface the heat flux reaches the potential temperature tendency and
  nothing else, which is the shape that run reported; over an unstable
  surface it also sets the convective velocity scale, so it reaches the
  momentum and moisture tendencies too and the refusal names the wind
  tendency instead. The named input is the heat flux either way.
- Two silent wrong answers on sm_120, where FP32 subnormals are flushed
  in all arithmetic. A friction velocity of 1e-13 is an ordinary float32
  whose cube is subnormal and flushes to zero, turning a quotient into
  NaN and laundering it into an exchange coefficient of 1000 m2/s where
  the float64 authority says 131; validation saw nothing. And a
  roughness length small enough to overflow the surface-layer
  logarithm's quotient sent an infinity into every downstream similarity
  quantity. Both now have defined answers, proven bit-exact on the
  healthy path under pinned fixtures and an adversarial sweep. Where WRF
  v4.6.1 leaves the degenerate case undefined, the defined behaviour is
  documented as a divergence rather than reproduced.
- A failure capsule can be read without the tree that produced it. It
  now embeds its own configuration, namelist and vtable text, size
  capped. A first-step failure is also labelled as stepping rather than
  as writer initialization, which had sent one real user-bug diagnosis
  down the wrong path.
- Twelve defects in the observation emitters and scoring path, found
  while hardening the instrument: among them frame selection under radar
  outages, a frozen station table, an archive manifest that describes
  the archive rather than one pull, and the WRF comparison arm being
  told to produce the reflectivity it is scored on.
- Release receipts record exactly the 16 bytes that are a device UUID;
  they previously carried three neighbouring bytes that can change when
  the adapter re-enumerates.
- The release snapshot builder answers --help instead of running.
- `gpuwm doctor` checks for the NEXRAD front door by name, and that
  binary now ships in the bridge bundle. A machine without it previously
  got a green report while being unable to ingest a single radar
  observation, which is every DA route, live and archived. Absent blocks
  and says what it blocks. Stale says to rebuild rather than to
  re-point, because every such binary reports the same version string,
  so pointing at another copy fails identically.
- The NoahMP column-slab preflight priced its device transient by CuPy
  pool growth, which reads zero whenever the pool already holds a block
  big enough to serve the request. It now reads allocation demand at the
  allocator boundary. No gate was widened: the negative controls at
  1,024 and at 16,384 columns both still fail.
- 91 restart gates that need no device were being skipped by a
  whole-module GPU mark, and a collection that crashed under the marker
  leak gate made that gate pass vacuously instead of failing.
- One scorer of record for the skill battery. The two scorers already
  shared the kernel, but each carried its own copy of the constants, and
  the copies disagreed where it matters: the same named half-width is a
  27 km box at 3 km spacing and a 13.5 km box at 1.5 km. The constants
  are now derived in one place, proven inert against the committed
  composites, and held there by a guard that fails when a literal is put
  back.
- A sweep receipt described its scored field as one member when the
  published figure is the ensemble mean. The label is corrected and no
  number was recomputed.

High-resolution terrain statics (hash-bound raster overrides for
topography) are present and unchanged in this release.

## 1.5.2 (2026-08-03)

ArWen gains its first scale-aware boundary-layer scheme: Shin-Hong 2015,
`bl_pbl_physics = 11`. A namelist asking for it now gets it — native
import, no substitution to YSU — and the scheme's subgrid/resolved
partition adapts continuously with the grid spacing, which is what the
1 km-to-100 m "gray zone" between mesoscale and LES resolutions needs.

The evidence ships with the claim. The CPU reference implementation is
bitwise against WRF v4.6.1's own module — max ULP 0, every output
column, both terrain-flux arms — and the CUDA port holds the heating
tendency bitwise on top of it. Beyond WRF-fidelity, the partition
itself was measured: a 3200 m → 100 m grid-spacing ladder, six
independent seeds per rung, scored against acceptance bands registered
before any run existed, drawn from the published similarity envelope.
Every gated rung landed in its band, and at 100 m the scheme reproduces
standard behaviour to within measured noise — scale-awareness in the
gray zone without corruption at the LES end. The full ladder — every
seed, every band, the determinism digests — is at
[docs/public/receipts/grayzone/](docs/public/receipts/grayzone/README.md),
and [PHYSICS.md](docs/public/PHYSICS.md) states exactly what is proven
and what is not. The honest label stays **implemented-unverified**: no
matched free-running forecast comparison has been run with it, and the
ladder is one idealized case on one card.

Surface-layer pairing rules mirror WRF's own — Shin-Hong runs with
`sf_sfclay_physics` 1 or 91, and a namelist pairing it with anything
else is refused by name with the values that would work, citing the
line of WRF that enforces the same thing.

One small fix rides along: the note printed when a GFS fetch domain
crosses 0° longitude now says it is informational — the full-band
widening is handled, the only cost is download size, and the run
continues unchanged. Field reports were reading the old wording as an
error.

## 1.5.1 (2026-08-03)

A hardening release. The published wheel's first week in the field — a
1 km nest on an RTX PRO 6000, cross-architecture runs on an RTX 4090, a
5060 Ti, and real HRRR forecasts on an RTX 5070 Ti — sent back every
rough edge fixed here, and one thing the 1.5.0 notes had to stop
claiming is now true.

GFS gained the full-file transport HRRR has had all along: `gpuwm fetch
--source gfs --mode full-file` takes the whole `pgrb2.0p25` objects
from the AWS S3 archive — no NOMADS rate governor, no spatial crop, and
the archive's reach is years where NOMADS keeps about ten days —
through the Rust backbone's parallel range GETs or the stdlib
transport. The raw objects differ from the NOMADS crops in row order
*and* packing; both differences are certified in `gfs_grib2_bridge`
against committed matched pairs of the same fields from the same cycle,
including a bitmap-carrying SOILW pair that pins complex-packing
missing-value semantics cell for cell, bit for bit. Everything outside
that proven envelope still refuses by name, the NOMADS crop remains the
default transport unchanged, and `gpuwm doctor --source gfs` now
reports the GFS route — decoder and byte transports — the way it
reports HRRR's.

The wizard now emits what the runner accepts: a bare `gpuwm domain`
produces one 12 km domain — the shape `gpuwm go` runs — instead of the
deepest nest tree that fits the card; nest trees are explicit opt-in,
and refusals name the tree runner with a complete remedy. Sizing a card
that is not in the machine is priced against the conservative measured
reference profile instead of a per-class discount, and is labelled an
estimate everywhere it appears. **Migration:** a config sized near a
card's ceiling that previously certified as fitting can now be honestly
refused — re-run the wizard to get a grid the estimate stands behind.

First-run polish, from the first cross-architecture field run of the
published wheel: with `GPUWM_CASE_DATA_ROOT` unset, the case-data root
now follows the platform — the XDG data directory on Linux and macOS
instead of a literal `~/Downloads`, which stays the Windows default
unchanged. `gpuwm fetch-geog` announces the whole bill — datasets still
needed, download bytes, unpacked size — before the first byte moves.
The first forecast on a machine says up front that it is compiling GPU
kernels for the local card and what that costs (typically 1–3 minutes,
cached for every later run), where that time used to pass under a stale
status and read as a hang. `pip show gpuwm` answers `Apache-2.0` — the
wheel carries the SPDX license expression; building from source now
needs setuptools ≥ 77. Rendered PNGs are named `arwen_*`, and `--pair`
reads both spellings so older render directories still pair.

From the 5070 Ti findings: `gpuwm update` exists — print-only, it names
the exact upgrade command for your install and what stays preserved
across upgrades; the generated HRRR chain prints the two experimental
Thompson-aerosol exports before preparation instead of leaving them to
be discovered from a refusal; and the benchmarks grew up — an existing
`--outdir` is refused in a sentence rather than a traceback
(`--allow-existing` reuses without clobbering), the GPU name is decoded
text instead of a bytes repr, provenance resolves from the installed
wheel instead of demanding a git checkout, the aerosol-aware scheme is
selectable with its fixed 362 MiB table cost priced into the estimate,
and the cold first step is reported apart from the warmed rate so short
runs stop extrapolating misleadingly.

wrfout frames now carry `OLR` — the top-of-atmosphere outgoing longwave
the longwave scheme was already computing — with WRF's registry
metadata, present exactly when the attached longwave scheme produces it
and absent with radiation off. It is output-only and rebuilt rather
than restart-carried, so checkpoints on disk stay loadable; a resumed
run reports zeros until its next radiation call — a deliberate,
documented divergence from WRF's restart handling.

The LES comparison receipts ship with the repository — now true: forty
curated receipts and an index at
[docs/public/receipts/les/](docs/public/receipts/les/README.md) back
every WRF-comparison, realisation-spread and determinism number
[LES.md](docs/public/LES.md) cites, byte-identical to the originals
except for relativized machine paths, with every omission stated and
justified. The CBL receipt's VRAM figure splits into `vram_pool_gib`
(screened) and `vram_device_gib` (environmental), so the documented
dual-run corruption screen stops false-positiving on a shared card. And
the single-card determinism known-limit is retired: the same seed is
bit-identical across three cards — two of them different sm_120
silicon — and seed-equivalent against sm_89, receipt and claim boundary
in the same directory.

Bookkeeping regenerated on the assembled line: the flush-to-zero claim
census and route registers re-keyed after 1.5.0's line drift, and one
stale kernel frame pin corrected to its measured value — re-measured
under two NVRTC majors and offline nvcc before it moved. The NVRTC
widening boundary in the Thompson control tests moves to 13.1, where
exact agreement was actually measured.

## 1.5.0 (2026-08-03)

This release makes the model usable at the scales storm work actually
happens at. Two large-eddy-simulation closures are now selectable —
`km_opt = 3` (3-D Smagorinsky) and `km_opt = 2` (1.5-order prognostic
TKE), both per-domain, so a coarse parent running a PBL scheme can carry
a PBL-off LES child in one tree. Both closures were run head-to-head
against an independently built WRF v4.6.1 `em_les` reference on the same
case, same grid, same instrument, across repeated realisations. A nested 250 m LES
configuration is included so the capability is reachable without
hand-assembly, and [LES.md](docs/public/LES.md) states plainly what is
proven and what is not.

Aerosol-aware Thompson microphysics (`mp_physics = 28`) ships with WRF's
own `CCN_ACTIVATE.BIN` table, so a clean install validates and runs it
with no extra download. Its honest label is **implemented-unverified**:
every device kernel is pinned bitwise against an instrumented WRF oracle
column by column, a matched free-running forecast comparison has been
run and published with its limits stated, and a t=0 read-back digest
backs the initial-state numbers — but the matched comparison did not
meet the bar to raise the maturity label, and the pages say so; a
third, pre-registered distribution gate across independent runs reached
the same verdict by its own declared rule
([validation/mp28-distribution-gate.md](docs/public/validation/mp28-distribution-gate.md)).
Against
the 22 pinned oracle fixture columns, 17 of 22 clear a flat bitwise
gate and 18 of 22 clear it with the three named allowances applied; the
residuals are itemized, with mechanisms, in the evidence page. Start at
[validation/mp28-column-evidence.md](docs/public/validation/mp28-column-evidence.md).

A third turbulence option, the SASE closure (`bl_pbl_physics = 900`), is
selectable and GPU-resident, verified against the GABLS1 stable
boundary-layer intercomparison. It is **EXPERIMENTAL and not
WRF-verified**, is selected run-wide only (never per nest), and the
configuration reference says exactly where it is and is not admitted.
With it comes `km_opt = 0` as a deliberate research control — no
horizontal mixing operator at all — behind a written acknowledgement so
it cannot be reached by a mis-set switch.

The domain wizard stops leaving hardware on the table: its internal
search ceiling used to size 64, 96 and 180 GiB cards identically, and
now a larger card buys a larger domain, with an explicit warning in the
one case the search bound rather than memory decides. Sizing below 4 km
with a cumulus scheme active earns a printed advisory (double-counted
convection, and the per-domain switch that fixes it); 4–10 km earns a
softer note; neither refuses. HRRR-driven runs got a matching cleanup:
the wizard derives one coverage envelope from the grid itself and clamps
the emitted fetch area inward so every emitted hint passes the same
validator that will judge it, the coverage fitter reserves the
soil-donor search margin instead of discovering it missing later, and
sub-hour forcing windows use one endpoint convention end to end. One
tightening to note: a hand-edited fetch area that names ground outside
the real grid — including clamps that previous releases accepted — is
now refused; re-emit the area with this release's wizard, which
produces a valid box unaided.

Receipts and diagnostics got more honest. The VRAM receipt now reports a
true high-water mark from a 50 ms watcher, labeled as what it is, with a
separate pool-peak entry — not a post-trim boundary value that
understated the peak. Sea-level pressure is terrain-robust: the
below-terrain extrapolation is smoothed with a terrain-keyed filter, so
high-terrain MSLP fields stop ringing. A first run stages the WPS_GEOG
datasets through the installer with byte accounting, and `gpuwm go`
prints what each stage holds and what a failed stage left on disk. An
interrupt anywhere in the `go` chain — including the pre-download memory
gate — exits with one sentence and code 130.

The release workflow accepts both cut motions again: a re-run of a
partially published cut adopts the assets it already uploaded by content
contract, the post-upload index proof rides out registry lag with
bounded backoff, and the cut refuses bridge binaries that are not
provably built from the release commit.

## 1.4.1 (2026-08-02)

A correction and operations release. No new physics scheme and no kernel
change: every forecast 1.4.0 could produce, 1.4.1 produces identically.

### Corrections

Two statements in 1.4.0's documentation were wrong and are corrected here.
FP32 subnormal flushing on sm_120 was described as a hardware property no
compile flag could change; it follows the compile options actually emitted, and
the same source built without an appended `-ftz=true` keeps full IEEE
subnormals on the same card. The CPU reference binary was described as an Intel
`ifx` build; it is GCC 15.2.0. Neither the countermeasures nor the reference
stream change.

### Fixed

- `gpuwm dual-run` returned nonzero on a byte-identical pair, so the
  determinism screen the documentation describes could not be passed.
- `--products wind10` returned a sea-level pressure analysis on the rust
  engine and a wind map on the matplotlib engine. A standalone
  `10m_wind_speed_and_direction` product was added: speed fill with barbs and
  streamlines, 0-60 kt, unmasked.
- A `pip install` shipped no map assets, so plots drew weather over a blank
  frame. The assets now travel with the renderer, and a renderer that cannot
  find them fails with a message instead of reporting success.
- The domain wizard emitted configurations that `gpuwm check` then rejected.
  It sized against a reserve the verifier does not use and against nameplate
  card capacity rather than usable capacity.
- The VRAM model had no intercept and priced preprocessing on the root domain
  only, so it under-predicted on multi-domain trees.
- An unrecognized `[case_data]` key was dropped silently; it is refused now.
- `gpuwm render --list-products` no longer requires a `wrfout` first.
- Every `wrfout` now records what it was initialized from, so a plot separated
  from its run directory still carries its provenance.
- Restart and preflight refusals exit with a message rather than a traceback,
  and a `run_seconds` off the root-domain step grid is refused at config time.
- Thompson (mp=8) moves closer to WRF v4.6.1; sparse HRRR cloud analyses no
  longer fail a mass-loss gate they should have passed.
- No ArWen or WRF quantity can reach a label reading "wind gust": WRF v4.6.1
  defines no gust. Operational GRIB gust fields keep the name, because that is
  the issuing centre's own field.

### Added

- `gpuwm multi-run PLAN.toml` — one independent forecast per physical GPU.
- `gpuwm stream PLAN.toml` — sealed hourly HRRR forecast streaming.
- `gpuwm report` — a single redacted file describing a failure.

### Running long forecasts on consumer cards

Consumer GPUs do not correct memory errors. On one such card, six 6-hour runs
from byte-identical inputs each ended differently, and two duplicate 1-hour
runs of one configuration wrote byte-different output. Run a long forecast
twice and compare the results byte for byte; a single run on hardware without
ECC is not by itself a reproducible result.

### Known limits

- The VRAM envelope carries no RTX 5090 datapoint.
- `10m_wind_1h_max` read from a `wrfout` store is a lower bound on the true
  maximum, not the maximum.
- Upgrading ArWen does not upgrade a bridge staged by an earlier version.
  Re-stage it with `gpuwm fetch-bridges`.
- Checkpointing is not reachable on the single-domain prepared route, and
  three of the configuration changes `gpuwm run --restart` documents as
  supported do not take effect.

## 1.4.0 (2026-08-01)

This release removes the two things that most often stopped a run that
was going to be fine. Any physics suite the engine implements now runs:
the whitelist of named suites is gone, and verification status is a
label reported on the receipt rather than a gate. And 33 user-facing
refusals became one-line warnings that continue -- a hard refusal is
now reserved for the cases where the run would be garbage.

Preprocessing no longer grows with the length of the run. Ingest used
to hold every forcing time resident at once; it streams now, so its
peak is FLAT in forcing-time count where it used to climb. Measured
against a real 1.3.1 install on identical inputs and the same idle
card: **3.98 -> 4.00 GiB across 2 and 5 forcing times on 1.4.0, where
1.3.1 went 4.80 -> 6.33 GiB.** That is the property that matters on a
small card -- a 24 h window costs the same preprocessing VRAM as a 2 h
one -- and it is what the headline should have said. On the longer
windows the streaming change was developed against (CONUS 12 km
414x330x49, nine 3-hourly GFS times, RTX 5090) the same change is a
14.93 -> 4.56 GiB reduction, but ~3x is a property of a nine-time
window, not of the release: at the 2-12 h windows a first run actually
uses it is 1.2x-1.6x. `gpuwm check` and the `gpuwm domain` fit loop
also price ingest as its own phase -- so a domain that cannot fit is
refused before anything is downloaded rather than after.

A run may now start at a forecast lead instead of only at a cycle's
analysis: `start_time` may be cycle + K hours, the initial condition
comes from f{K}, and the GRIB bridge enforces instantaneous-only
fields for that path.

The HRRR route works end to end from a wheel: the wizard emits the
complete input set for the nested route, per-domain history cadence is
supported, sizing is coverage-aware and refuses an off-grid domain
before the download, transport prefers S3, ocean-edge domains prepare,
and one cell of snow-depth interpolation overshoot is repaired instead
of losing a preparation.

Nests inherit the physics profile's `epssm` instead of a hardcoded
value. That one line was a first-minute blow-up on sub-kilometre nests
over steep terrain.

The externalized Thompson tables stage outside the install, so a wheel
upgrade no longer deletes what you just downloaded, and a missing
table is a named preflight refusal instead of a mid-forecast
traceback. `jsonschema` is a declared dependency, and
certification-capsule emission can no longer cost you a completed
forecast. `gpuwm doctor` now resolves the exact execution paths a run
will use, per data route.

A substantial part of what follows is fixes for issues reported by an
external tester, who ran the published wheel with no git checkout
anywhere near it.

### Performance

- The full-physics step is faster on the same hardware and the same
  inputs, from a profile-first pass that removed per-step scalar
  device-to-host transfers, batched output validation, and reused
  scratch buffers in place. The representative trace, its method and
  its receipts are in [PERF-SURVEY.md](PERF-SURVEY.md). No numerical
  result changed: the pass is measured against a bit-identical base.

### The HRRR route, and the nest that blew up

- **The HRRR hierarchy prints the digest its own chain asks for.** The
  emitted multi-domain chain ends in `--preparation-receipt-sha256
  <printed by the hierarchy>`, and the forecast runner checks that
  digest against the hierarchy's `receipt.json` -- but the hierarchy
  printed only timings and the three WRF file digests, so the one
  placeholder in that chain no stage filled. Walking the printed route
  end to end on a two-domain run is what found it: the chain could not
  be completed without hashing a file by hand. It is printed now, taken
  from the published receipt rather than from the in-memory payload, so
  it is a digest of the bytes the next stage reads. In the same pass the
  single-domain block stopped pointing at FIRST-LIGHT section 3a for its
  forecast stage: 3a is the GFS route, which that same block tells the
  reader not to use, and it never mentions the benchmark runner. It now
  says the arguments come from what the preparation wrapper prints,
  which is where they actually come from.
- **Every nest gets the physics profile's `epssm`.** `gpuwm domain`
  wrote `epssm = 0.1` on every nest while the root took 0.5 from the
  profile -- stripping the vertical-acoustic off-centering exactly where
  nest terrain is steepest. On a wizard 3 km -> 750 m ladder over steep
  terrain, w grew -15 -> -289 -> -976 m/s to non-finite in seven
  acoustic substeps at the child's single steepest cell (34.6 degrees;
  the 3 km parent smooths the same peak to 15.7 and survived). The
  reported nested-forecast blow-up was that one line; the PBL guard was
  the messenger. The value is now written per domain in the emitted
  TOML, so it is visible where it is set and stays overridable there.
- **`gpuwm domain --source hrrr` emits every input the HRRR routes
  read.** It used to emit one of five, and the `namelist.wps` it did
  emit was missing `&share/interval_seconds` -- the one key the
  domain-tree route's first gate requires. The wizard now writes the
  target-domain document, the native `namelist.input` and its stock-WRF
  twin beside the config, all derived from the emitted experiment and
  re-imported through the real importer before they are left on disk;
  binds a route-compatible physics profile when none is named (and
  refuses an incompatible one at emission, naming the switches); and
  prints the actual HRRR chain with every value it knows already bound.
  A multi-domain HRRR emission used to be told to prepare with `rw-wps`,
  which is the GFS front door.
- **An HRRR domain is sized against HRRR's grid, not only your card.**
  HRRR's native grid is finite and the interpolation halo needs real
  source cells outside the target on every side, so a ladder sized
  purely against VRAM could be a legal, well-sized experiment that no
  HRRR fetch can force -- found by running the wizard's own output: a
  3 km root whose halo ran nine rows off the top of the grid, discovered
  by the root preparation after the download. The fit loop now asks the
  source's own window function about every candidate, a point HRRR
  cannot force at any size is refused before anything is written, and a
  domain the grid stopped rather than the card says which bound it hit.
- **One cell of interpolation overshoot no longer loses a preparation.**
  Snow water and snow depth are physically non-negative, so a negative
  mapped value is the horizontal operator overshooting across the snow
  line, not data. A nested HRRR preparation died on one cell of 88 844
  at -4.9 cm beside a 44.5 m maximum. It is repaired at zero, on the
  same terms soil moisture already was; an overshoot beyond what a
  bounded stencil can produce still refuses, now with the count, the
  most negative value and the field maximum in the sentence.
- **A coastal HRRR domain prepares.** Any domain whose 5-cell boundary
  strip was entirely water died in soil mapping with a message naming
  neither the strip nor the domain -- and the four strips are mapped
  independently, so one Pacific-facing edge killed the whole
  preparation. A water target needs no land donor: its soil is the
  target-water fill. The land-window diagnostics are recorded as absent
  with the reason, the finiteness and physical-range checks now run in
  every case, and the refusal that remains (target land with no
  reachable source land) names the grid, the count and the remedy.
- **A frozen soil node no longer discards a prepared hierarchy.** The
  stock-WRF export's frozen-soil refusal is typed as
  `StockWrfExportUnsupported`, and the HRRR hierarchy asks for an
  optional export like every other route, so a state gpuwm integrates
  but the WRF file format cannot represent is recorded as a refused
  export manifest instead of destroying a complete, verified
  preparation.
- **`--transport auto` is paired with the byte mode it runs.** It probed
  NOMADS first; both hosts serve identical bytes, but S3 is several
  times faster for the whole-file default (a measured three-hour window:
  54 minutes over NOMADS, about three over S3). `auto` now prefers S3
  and falls back to NOMADS only when S3 cannot serve the window yet;
  `--wait-for`, which is what NOMADS's head start is for, keeps its
  NOMADS-first polling. When NOMADS is chosen with `--mode full-file`
  the cost is named before the first byte moves, and `gpuwm fetch` says
  who chose the byte mode -- the backbone cannot tell a typed flag from
  a caller's default, and used to call both an "operator override".
- **`--pipeline-workers` is refused at parse time.** `prepare_hrrr_wrf`
  advertised 1..64 while the decoder it spawns accepts 1..13, so an
  impossible request cost a geometry receipt and a static build before
  anything refused it.

### Front doors, packaging and documentation

- `gpuwm fetch-tables` stages outside the install. It used to write its
  two downloads *inside* site-packages, so the 315 MiB a wheel user had
  just paid for was deleted by the next `pip install --upgrade` or venv
  rebuild -- and the nested-GFS forecast that followed died on a bare
  `FileNotFoundError` naming a path in site-packages, after the fetch
  and preprocessing had already been paid for. Staging now lands in
  `~/.gpuwm/tables/thompson`, beside `~/.gpuwm/bridges`, and the staged
  root is completed from the package so it is whole; a complete
  packaged root (every clone, and every wheel staged before this
  change) still answers first and is left alone. The gap itself is
  refused at `--materialize-authorities` -- step 2 of the six the
  documentation prints, before any download -- in one sentence naming
  the table and `gpuwm fetch-tables`, and the domain-tree runner's
  preflight says the same thing rather than raising.
- `gpuwm fetch --author-front-door-manifest` resolves the bridge itself
  when `--bridge` is omitted, through the same `gpuwm.bridges` ladder
  `gpuwm go` has always used. FIRST-LIGHT's stage-by-stage route
  printed `tools/grib1_bridge/target/release/gfs_grib2_bridge`, which
  exists only in a checkout, so a wheel user following the documented
  long form was refused after paying for the fetch. That page now
  documents the resolved default, and its step 5/step 6 pair names the
  `--outdir` rw-wps actually prints (`<output-root>-forecast`) instead
  of a third path neither stage produces.
- `gpuwm adapt --skeleton` names its own gaps, and `--descriptor`
  refuses the unfilled scaffold as a scaffold. The placeholders are
  deliberate, but the reader met them as
  `descriptor.adapt.model_top_pa must be a positive finite number` --
  a type error about one field, with three more waiting one run apart,
  and two (`name`, `target.name`) that passed every validator and would
  have shipped an adapter called `REPLACE_WITH_ADAPTER_NAME`. One
  function now reports them, so the skeleton's list and the authoring
  gate's list cannot drift.
- `rw-wps --namelist-support-report` refuses a missing namelist in one
  sentence. It is step one of `docs/migrating-from-wps.md` and answered
  the commonest possible mistake with a five-frame traceback out of
  `pathlib`, while `gpuwm import-namelist` answered the identical
  condition cleanly; both now go through one function.
- `gpuwm downscale --card` accepts the tiers `gpuwm domain` accepts.
  It hardcoded `16gb|24gb|32gb` and rejected `12gb` -- a tier the
  product advertises and this command already maps -- so two front
  doors quoted different tier lists for one concept.
- The wizard's oversized-footprint advisory measures latitude too, and
  names the flags that matter. A Linux `--card 24gb --ladder 12` sized
  a 144x88 degree domain behind a hemispheric fetch box; the advisory
  saw only longitude, and offered `--card 12gb` rather than saying what
  the printed `--area` is or why shrinking it alone would starve the
  domain it feeds. Still an advisory, still never a refusal.
- DATA.md prices HRRR at what it costs. The documented ~0.4 GB per
  forecast hour was one object in a subset mode that is no longer the
  default; the default pulls the whole `wrfnat` **and** `wrfprs` pair,
  measured 704 MB + 427 MB, so ~1.1 GB an hour and ~21 GB for f00..f18.
  Both modes are now priced separately. `docs/native_source_adapters.md`
  says that its `share/configs/...` paths are distribution-root paths
  and names the checkout's `configs/` equivalents; FIRST-LIGHT's
  custom-ladder examples no longer hide the required `--cycle` and
  `--out` behind an ellipsis -- nor does HARDWARE.md's short version --
  and the page no longer offers `gpuwm domain --explain` as a way to
  list profiles, which is a usage error rather than a listing.

### Physics freedom, memory and initialization

- 33 user-facing refusals are one-line warnings that continue. The
  config loader, preflight, the environment-production path and
  certification-capsule emission now say what they found and carry on,
  because none of those findings was ever evidence that the run would
  be wrong. `jsonschema` is declared in `pyproject.toml` instead of
  assumed, and a missing one costs a capsule rather than a finished
  forecast. The README quickstart is two commands.
- Any physics suite the engine implements runs. The whitelist that
  admitted a fixed list of named suites is gone, and verification
  status -- `readiness`, `warning_only` -- is reported metadata on the
  receipt rather than a gate: an unverified combination is labelled,
  not refused. `export_prepared_wrf` gains an `experiment_config_suite`
  contract, expert-tuple governance is applied uniformly behind
  `--ack`, and the Thompson table checks are keyed on `mp_physics == 8`
  rather than on a suite name. An unacknowledged expert tuple warns and
  runs, with both accepted spellings of the acknowledgement in the line
  and `acknowledged: false` on the receipt; a caller who *names* an
  expert `--physics-profile` asked for that gate and still meets it.
- Preprocessing no longer peaks at roughly double the forecast. The
  initialization loop held every forcing time's horizontally
  interpolated analysis and every time's complete `DomainState`
  resident until the last one existed, purely so the lateral-boundary
  builder could see them all at once; only the first time is used
  downstream. It now streams: `StateBoundaryFrames` takes each state's
  four `spec_bdy_width` perimeter frames as it is built and the state
  is released immediately. On a measured CONUS 12 km case (414x330x49,
  nine 3-hourly GFS times, RTX 5090) the preprocessing peak fell from
  **14.93 GiB to 4.56 GiB**, below the same case's 7.10 GiB forecast
  peak, with the prepared cache's 451 arrays and both wrfinput_d01 and
  wrfbdy_d01 byte-identical to the pre-change run. Applies to the GFS,
  ERA5, mapped-source and config-driven (`gpuwm run`) ingest paths.

  **The 3.3x in that sentence is a property of a nine-forcing-time
  window, not of the release.** The reduction a user sees is whatever
  the streaming saves at *their* window length, and the honest general
  statement is the shape rather than the ratio: 1.3.1's preprocessing
  peak GROWS with the number of forcing times and 1.4.0's does not.
  Measured on one card against a real 1.3.1 install, identical GRIB2
  inputs, identical WPS_GEOG tree, identical config (242x194 @ 12 km +
  480x384 @ 3 km):

  | forcing times | window | 1.3.1 peak | 1.4.0 peak | reduction |
  |---|---|---|---|---|
  | 2 | 2 h | 4.80 GiB | **3.98 GiB** | 1.20x |
  | 5 | 12 h | 6.33 GiB | **4.00 GiB** | 1.58x |

  1.3.1 climbs ~0.51 GiB per extra forcing time; 1.4.0 is flat to
  0.02 GiB. Extrapolating 1.3.1's measured slope, the ratio reaches 3x
  at about 13 forcing times (a ~36 h window). For the 2-12 h windows a
  first run actually uses, expect 1.2x-1.6x -- and expect it not to get
  worse as you lengthen the run, which is the part that is new.
- `gpuwm check` and `gpuwm domain` price **every phase**, not just the
  forecast. The report carries an INGEST line and a one-sentence
  binding-phase verdict, the wizard's fit loop sizes against the
  largest phase, and a config whose forcing product this estimator has
  not modelled (HRRR's native-hybrid-level lane) is reported NOT PRICED
  rather than scored as free. Previously a domain sized to "fit your
  card" could pass `gpuwm check` and then run out of memory during
  preprocessing, a phase nothing had estimated.
- `gpuwm go` runs that check **before the fetch stage**. A binding
  phase that cannot fit the card's free VRAM is refused with its
  number and the re-sizing command, with nothing downloaded; over
  budget but inside free VRAM warns and proceeds. `--no-memory-gate`
  skips it.
- GFS: a run may be initialized from any forecast lead the fetch
  carries, not only from the cycle's f000 analysis. `start_time` may be
  `cycle + K` for any K in the fetched forecast-hour set: the initial
  condition comes from f{K}, the lateral boundaries from f{K+i}, and
  the model clock starts at `start_time`. `gpuwm fetch
  --forecast-start-hour K` plans the window (`--hours` stays its
  length, so `--forecast-start-hour 174 --hours 66` fetches
  f174..f240), the same flag on `--author-front-door-manifest` cuts a
  front-door manifest to that tail of an existing fetch instead of
  re-downloading it, `gpuwm domain --forecast-start-hour K` emits the
  config, and `gpuwm go` carries the lead through its chain. Before
  this, `rw-wps --source gfs` refused any experiment whose start_time
  was not the cycle -- so reaching a window deep in a forecast meant
  integrating to it. Admission warns rather than blocks, and every
  receipt records cycle, lead and start together
  (`gpuwm-gfs-initial-condition-provenance-v1`); a forecast lead is
  never relabelled as an analysis. A run at lead 0 is unchanged, proved
  by a byte-identical prepared cache, wrfinput/wrfbdy, final state and
  history frames against the base commit.
- Docs: added an explicit resolved-scale disclaimer -- the 500 m nest
  and the STP/UH severe suite are tornadic-supercell environment and
  mesocyclone-proxy diagnostics, not resolved tornado dynamics -- to
  README Limits and to VERIFICATION.md section 6 Not-claimed.

### The HRRR route on a pip install

A field report ran the published wheel -- no git checkout anywhere near
it -- through the HRRR to nested-d02 route and hit five failures in a
row that `gpuwm doctor` had just called "no gaps". Every one of them is
a resolution the report never performed.

- `gpuwm fetch --source hrrr` takes whole objects by default
  (`--mode full-file` through the Rust backbone) instead of `.idx`
  record subsets. The old `auto` default resolved to `idx-subset`
  against every healthy host, because its probe rule only takes the
  whole file when the index cannot carry the selection; one measured
  419 MB file cost 560 s that way against 27-35 s taken whole.
  Subsetting is the opt-in bandwidth saver it was always documented as,
  it says in one line what it costs when chosen, and an install without
  the backbone is told before the download rather than after it.
- The HRRR preparation wrapper resolves its decoder through the same
  ladder `gpuwm doctor` reports -- environment override, checkout
  build, `libexec/bridges`, `~/.gpuwm/bridges` -- instead of a cargo
  workspace under `site-packages` that no wheel has. A green doctor
  line and a runnable preparation are now the same claim.
- A run binds its identity from the installed wheel (distribution
  version plus the digests pip wrote into `RECORD`) when it is not a
  git checkout, instead of running `git rev-parse` with the working
  directory in `site-packages` and exiting 128. A venv nested inside an
  unrelated repository is treated as "not a checkout of this tree"
  rather than being bound to that repository's commit.
- The sealed-runtime manifest is validated against its whole schema at
  the first read, by one validator every consumer calls, and the
  refusal names every missing field at once. A minimal hand-authored
  document used to satisfy the identity gate, run a preparation for
  minutes, and die at a third consumer on `contract.platform.backends`.
- The stock-WRF export no longer takes a successful preparation down
  with it. `--skip-stock-wrf-export` never attempts it;
  `--require-stock-wrf-export` makes a refusal fatal; by default a
  refusal is one warning line and the prepared cache and its PASS
  report stand. The converter's own named refusals are unchanged.
- `gpuwm doctor` resolves the paths a run will use: the exact decoder
  per data route, the byte transport `gpuwm fetch` will pick, the
  identity path a receipt will bind, the complete manifest schema, and
  whether the route entry points import from a directory that is not a
  repository. `--source SOURCE` narrows the report to one route; every
  route is in the default estate, because behind a flag these would
  have been invisible to the person this was written for.

### One d01 across the HRRR nested route

The same field report's other half. Seven contract contradictions on the
HRRR to nested-d02 route, six of them the same shape: two parts of this
route answering one question differently, each needing a hand
workaround.

- The single-domain root preparer reads **d01's** column entry of every
  WRF per-domain array, not the last one. `gpuwm domain`'s own emitted
  ladder writes `diff_6th_factor = 0.08, 0.10`; the preparer read 0.10
  -- d02's value, while preparing d01 -- and refused with a sentence
  about the number, never saying which domain it had looked at. A
  refusal on a column that is not uniform now says so and names the fix.
- `num_land_cat` and `fractional_seaice` no longer hit the unmapped-key
  refusal, which they did despite gpuwm's own stock-WRF namelist writer
  emitting both and a shipped hierarchy config carrying both.
  `num_land_cat` is validated against the one land-use identity gpuwm
  builds -- the 21-category MODIS set its static builder stamps and
  every wrfout carries -- and recorded; `fractional_seaice` is reported
  with the branch each initialization route actually runs. Neither is
  silently dropped, and a genuinely unmapped key still refuses by name.
- The hierarchy's run-length ceiling is the sealed root preparation's
  own forcing inventory. It was pinned at 43200 s with a sentence about
  "the native f00..f12 forcing horizon", while its own docstring
  promised not to pin a duration and while the source window, the fetch
  and the forcing-hour check below it all handled f00..f24.
- Per-domain history cadence is supported: a 3 km parent written hourly
  beside a 1 km child written every 15 minutes -- the ladder's whole
  point, and already implemented by the experiment loader, the clock and
  the tree runner -- was refused after preparation as a drift.
- `moist_cq` and `top_lid` have one authority: the shipped physics
  profile. WRF's namelist has no `moist_cq` key and gpuwm's `top_lid`
  default is deliberately not WRF's Registry default, so something must
  decide them; the profiles decided one way and the importer invented
  another, opposite for every WSM6/Kessler/MYNN/RUC/Noah-MP suite, so a
  public root and a public hierarchy built from the *same* namelist
  could not produce matching d01 identities. A test asserts every
  shipped profile agrees with the single answer.
- Prepared-cache identity asks the partition gpuwm already publishes
  twice instead of keeping a third opinion, through one normalization
  shared by the hierarchy's root binding and the tree preflight. Write
  cadences, a trajectory-inert diagnostic toggle, and a `cudt_minutes`
  that is dead at `cu_physics = 0` and spelled differently by the
  importer and the profiles were all compared by exact equality.
- `write_prepared_cache` accepts a read-only mapping. It crashed on the
  `MappingProxyType` hydrometeor record that `restore_prepared_cache`
  deliberately hands back, with "Object of type mappingproxy is not JSON
  serializable" from inside a nested publication, naming no field. A set
  still refuses, having no canonical order.

Every refusal removed here has a negative control beside it: a garbage
namelist key, a USGS category count, an active `cudt_minutes`, a real
trajectory difference and a run longer than the sealed forcing window
all still fail closed.

### Fixed

- `gpuwm doctor` no longer hangs forever probing a present-but-invalid
  executable on Windows. `subprocess.run`'s timeout bounds the wait and
  not `CreateProcess`, so a file with a corrupt image header could make
  the loader raise a modal dialog inside that call which no timeout
  reaches and no hidden-window session dismisses; a release battery
  froze there twice. Every probe in the package -- the bridges, the
  renderer, the fetch backbone -- now reads the ELF/PE header first and
  reports "present but not a native executable" from the bytes, with a
  fail-fast process error mode behind it for the loader dialogs a
  header cannot predict (a missing DLL). The verdict for a genuine
  bridge is unchanged.

### Sending one file when something breaks

`gpuwm report` collects, from a run directory, everything that explains
a failure -- the receipts, the typed failure with its traceback, the
supervisor's worker logs, the resolved config, this install's provenance
through the same resolver a run receipt uses, the Python/CuPy/driver
versions, the card and its memory, and the free space on every volume
involved -- into one plain zip to attach to an issue. Run with no
arguments it reads the directory it is standing in; `--dry-run` prints
the manifest and writes nothing.

It is anonymous by construction. Usernames, home-directory prefixes,
hostnames, IP and MAC addresses, e-mail addresses, credential-shaped
strings and every environment variable outside an allowlist are replaced
by class placeholders in path names, in log text and inside JSON
receipts, and the manifest counts what went by class. Coordinates,
dates, grid shapes, physics choices and SHA-256 digests are kept: a
chosen domain is scientific content, not identity, and a bundle whose
digests were shredded could not be matched to a receipt.

It reads only what ArWen writes. Collection is by allowlist rather
than by sweep, a deny-set refuses dot-files, private-configuration
directories, credential names and key-shaped suffixes before anything
opens or lists them, paths are resolved before they are tested so a
symlink cannot carry a target past the check, and the command refuses
outright in a directory holding nothing this product wrote. Refusals
are counted by class and never named, because a file name can be the
secret. Redaction is the second line of defence, not the first.

It is built for the failure it exists for. The archive is assembled in
memory and written once, relocating to the temporary or home volume if
the first location refuses, so a full disk costs a relocation rather
than the bundle; a missing artifact is a named line in the manifest with
the route that would have written it, never an exception; and when a
volume is nearly full the manifest says so beside the note that an empty
`pipeline producer exited 1:` message is a known erasure rather than
additional information. Model output and input data are never copied --
`wrfout`, restarts and NPZ caches are inventoried by name and size.

## 1.3.1 (2026-07-31)

Four commands take a fresh machine to a rendered forecast: `pip
install gpuwm[gpu,render]`, `gpuwm setup`, `gpuwm domain`, `gpuwm go`.
Physics combination admission follows WRF v4.6.1's own compatibility
verdicts, the run-time parameter ledger is closed out with every
unimplemented parameter refusing by name, polar stereographic and
Mercator projections run end to end, and mixed microphysics across a
nest edge runs under a named transition policy.

### The four-command chain

- Both forecast runners live in the installed package. `python -m
  gpuwm.prepared_single_domain_forecast` and `python -m
  gpuwm.prepared_domain_tree_forecast` work from any directory, with
  console scripts `gpuwm-prepared-forecast` and
  `gpuwm-prepared-tree-forecast`; the `tools/` scripts a checkout
  carries delegate to the same module objects, so no second copy
  exists to drift. Every command the CLI prints is one a pip install
  can execute. In 1.3.0 the chain's last two stages were spelled as
  relative script paths only a git checkout has.
- `gpuwm go <config>` runs the whole single-domain GFS chain as six
  stages in one invocation -- authority, fetch, manifest, prepare,
  forecast, render -- and prints a heartbeat every 20 seconds with
  the stage's elapsed time and, where the stage publishes one, its
  model seconds. Rendering is the sixth stage rather than a command
  to paste; a missing `render` extra is named instead of skipped.
- A refused stage speaks in sentences. A reused output directory is
  refused up front with the directory named, why it is create-only,
  and both ways out; in 1.3.0 this surfaced as a raw
  `FileExistsError` traceback from a flag the caller never typed.
- `gpuwm go` drives one domain from GFS. A multi-domain tree is
  refused up front and the refusal names
  `python -m gpuwm.prepared_domain_tree_forecast`.

### The wizard composes with `go`

- `gpuwm domain` at a bare prompt asks its questions and emits a
  config `gpuwm go` accepts: one 12 km domain on the model-validated
  `morrison-mp10-ysu-mm5-noah-kf-rte-rrtmgp-v1` profile, with both
  supplied defaults stated in the session. This changes the
  bare-session emission -- in 1.3.0 it was a four-domain ladder on
  the product default suite, a shape the single-domain chain refuses
  twice over. Pass `--ladder` for nests and `--physics-profile` for
  another suite; a session with any explicit flag keeps the flag
  path's own defaults.
- The closing next-steps block branches on source: a GFS emission
  ends at `gpuwm go`, an ERA5 emission keeps `gpuwm run`, and HRRR
  and multi-domain trees get their route named. In 1.3.0 the block
  ended at `gpuwm run` for every source, and `gpuwm run` refuses GFS
  configs by design.
- A domain sized to fill the card prints an advisory when its fetch
  box exceeds 90 degrees of longitude, naming `--vram-gib` for a
  smaller first run. Sizing itself is unchanged, and continental
  domains under the threshold pass without comment.

### WRF v4.6.1 combination admission

The combination authority is a 2,400-cell transcription of WRF
v4.6.1's own PBL, surface-layer, and land-surface admission: 1,080
cells legal, 360 legal with WRF's silent reconfiguration named, 600
fatal, 360 not expressible in WRF. Three pairings 1.3.0 refused are
admitted because WRF admits them -- PBL off with the MYNN surface
layer, and MYNN PBL with either MM5 surface layer, each across the
four routed land models -- and refusals cite the matrix cell that
produced them. An edge policy is not a loophole: a PBL/surface-layer
pairing the matrix calls fatal refuses whatever the nest declares.

### Run-time parameters

The declared WRF namelist parameter set stands at 158: 104
implemented, and each of the 54 others refuses by name with the
blocking dependency cited. Five parameters whose runtime branches
already existed are wired end to end: `isfflx`, `o3input`,
`use_mp_re`, `seaice_albedo_default` (replacing a live-path
literal), and `rdmaxalb`.

### Projections

Polar stereographic and Mercator domains run end to end -- wizard
emission, WPS namelist, static-field preparation, map factors, and
earth-relative wind rotation -- verified against WPS geogrid
references for both families. A bare interactive session at high
latitude emits `map_proj = "polar"` with a matching namelist.

### Nest edges

All 20 ordered pairs of ported microphysics schemes run across a
parent-child edge under an explicit `nest_microphysics_transition`
policy, and a mixed edge without one refuses naming the setting.
Each policy's mass and number handling is a registry row with its
WRF citation, not a code path a config reaches silently.

### Byte-stable checkouts

`.gitattributes` declares `* -text`: a clone materializes the
committed bytes on every platform. That immediately exposed one
shipped defect, fixed here -- the packaged GFS Vtable's hash
contract named the file's CRLF form, which only a Windows checkout
produced. The contract names the committed bytes, and
`tests/test_line_ending_stability.py` holds every hashed-contract
file to its blob.

### Fixed

- `gpuwm go`'s render stage handed the render CLI the wrfout
  directory rather than its frames, failing after a successful
  forecast. Found driving this release's own acceptance chain; the
  frames are enumerated, and `--dry-run` still prints the glob.
- `--materialize-authorities` refused a reused output directory with
  a raw traceback; a missing base input did the same. Both are
  sentences.
- The packaged GFS Vtable hash pin, above.

The four certified profiles are byte-identical to 1.3.0: their
pinned two-step trajectory hashes did not move, and the registry
regenerates to the tracked bytes.
## 1.3.0 (2026-07-31)

Ten couplings that in 1.2.x produced silently non-WRF forcing are
implemented from WRF v4.6.1 source transcription, the MYNN surface
layer pairs with the RUC and Noah-MP land models, HRRR analyses
initialize NSSL-2 with their hydrometeors retained, and a
hand-authored physics tuple outside the registry's ratified set -- one
that is otherwise complete and executable -- can be run with one short
acknowledgement. The command surface answers first: `gpuwm doctor` and
the domain wizard print one line per finding with the full prose one
flag away, and `gpuwm setup` stages a fresh install in one command.

### Changed forecasts

Configurations using MYNN, RUC, Noah-MP, or NSSL-2 produce different
trajectories than 1.2.x where their forcing or initialization follows
WRF in 1.3.0:

- MYNN receives the real snow mixing ratio when the microphysics
  carries one (WRF `FLAG_QS`); its boundary-layer cloud fields
  (`QC_BL`/`QI_BL`/`CLDFRA_BL`) are consumed by all three radiation
  implementations with WRF's `icloud_bl=1` merge semantics.
- RUC runs WRF-ARW's per-species precipitation partition (rain, snow,
  graupel reach the soil model separately), the lake bypass, the
  fractional sea-ice deblend/reblend, and radiation-time `GSW`.
- Noah-MP receives all six WRF precipitation rates and the radiation
  driver's carried `COSZEN` instead of a per-call recomputation;
  glacier columns are refused before the forecast starts rather than
  mid-run.
- NSSL-2 initialized from a native HRRR analysis starts from the five
  retained analyzed species whichever radiation implementation is
  selected, RTE included; in 1.2.x those fields started from zero.
- NSSL-2 with legacy RRTMG activates ice and snow cloud optics, as WRF
  does; the 1.2.x exclusion was not WRF-faithful.
- Coastal native-HRRR domains where the analysis landmask disagrees
  with RUC's soil-category requirements are reconciled from the
  evidence in the analysis, or refused before the run starts with the
  points named; in 1.2.x this route crashed inside the RUC soil model.

The four certified profiles (WSM6+Dudhia, Thompson+Dudhia,
Morrison+RTE, NSSL+RTE on Noah/YSU/MM5) are byte-identical to 1.2.x:
their two-step trajectory hashes are pinned in the suite and did not
move through any of these changes.

### MYNN surface pairings

In 1.3.0, `sf_sfclay_physics = 5` pairs with `sf_surface_physics = 3`
(RUC) and `4` (Noah-MP), following the write-back ownership sequence
of WRF's surface driver: which component owns the fluxes, exchange
coefficients, and 2-m diagnostics at each point in the step is
transcribed from `phys/module_surface_driver.F`, tested against
independent source transcription, and proven with short GPU
integrations of both pairings. The WRF namelist importer admits a
complete MYNN 5/5 stack (`bl_pbl_physics = 5` with
`sf_sfclay_physics = 5`) paired with any ported microphysics scheme,
where 1.2.x admitted it only with WSM6. The MYNN half-suite refusals
(MYNN surface without MYNN PBL, and the reverse) remain.

### HRRR analyzed hydrometeors

In 1.3.0, native HRRR preparation retains analyzed QC/QR/QI/QS/QG for
NSSL-2 (in 1.2.x only WSM6/Thompson/Morrison), retains QC/QR for
Kessler under WRF real.exe's Registry policy with the frozen species
accounted in a receipt, and refuses microphysics-off with a statement
of what cannot be represented. The preparation receipt binds decoded
source to initialized state per species -- byte hashes, nonzero masks,
extrema -- and states its own evidentiary strength: a species with no
source mass is reported VACUOUS, not proven. An independent check
exists outside this repository: WRF's own real.exe, run on the same
source bytes, initializes the five species with nonzero-cell counts
within 0.06% of this route's.

### Expert acknowledgement for unratified tuples

A hand-authored domain tree whose resolved physics tuple falls outside
the registry's declared reachability -- but is otherwise complete and
executable -- refuses with the exact line to add, instead of silently
running or silently failing at first call. Acknowledging is one
repeatable flag, `--ack expert-tuple-v1`, or one configuration line,
`acknowledgements = ["expert-tuple-v1"]` under `[experiment]`. The
receipt records which IDs were used and how they were supplied. The
acknowledgement is a provenance gate, not a bypass: component
availability, half-suite coupling, soil and land-model compatibility,
geometry, vertical bounds, and runtime-budget refusals all still
apply, and an acknowledged tuple that fails one of them refuses with
that check's own message. The stock-WRF export is layered the same
way: a physics tuple outside the exporter's stock coverage refuses
the export by name while the forecast itself prepares, and
`--no-stock-wrf-export` skips the export with a NOT_REQUESTED
receipt. Registry-ratified tuples run without acknowledgement, and
two joined the ratified set: Noah-MP on GFS as an expert template
(the runner already advertised it), and NSSL-2 with legacy RRTMG as a
fixed profile at validation-candidate maturity, buildable end to end
on both the GFS and native HRRR routes -- the runner builds its
configuration from the profile's complete declared switch inventory,
and a profile switch without a declared forwarding home is a named
refusal.

### Refusals moved to where they can be read

Scheme vertical limits (MYNN needs nz >= 5, Kain-Fritsch 8..128, WSM6
2..80, the radiation implementations' layer caps, and the rest) are
checked at configuration validation and preparation, naming the
component and both bounds -- in 1.2.x a first-timestep abort.
Microphysics-off plans on real-data routes are decided by the plan
validator, which states what a moist-but-scheme-less state holds,
instead of being deferred to source initialization.

### A command surface that answers first

`gpuwm doctor` prints one line per finding -- status, subject, the one
command that closes it -- and folds repeats that share a remedy; the
full evidence and pasteable remedy blocks are unchanged under
`--explain` (byte-identical to 1.2.x's output, held by a golden
fixture). The domain wizard ends with a numbered three-command block
and nothing after it. `gpuwm setup` runs `fetch-bridges` and
`fetch-tables` with one status line each; geography stays opt-in
(`--with-geog`, size printed first). `pip install gpuwm[all]` installs
the GPU and render extras together. Scripts that parsed doctor's
default text should use `--json`, which is unchanged and complete. The
HRRR decoder's air-gapped build invokes Cargo from the crate-local
directory so the vendored-source configuration is discovered; building
the bridge without network access works from a bare checkout, where in
1.2.x it could fail to find the vendored crates.

### Fixed

- 1.2.3 cannot complete a prepared GFS single-domain forecast: its
  fetch writes front-door manifests carrying pressure-ladder
  provenance (`pressure_levels_hpa`, `top_pressure_pa`) that its own
  runner refuses as an identity mismatch. The 1.3.0 runner compares
  manifest identity strictly on model, product, and cycle; validates
  the declared ladder through the fetch lane's own validator against
  the experiment's model top; and records
  `input.source_manifest_identity` in the run report. An unknown
  `source` field is still a refusal, never a silently dropped key.

### Evidence and its edges

Every reopened coupling passed preparation, a real GPU integration,
and its receipts on GFS and/or native HRRR data, and matched WRF
v4.6.1 reference runs are banked for trajectory comparison. The
reference runs' retained forcing bytes match the ArWen-side inputs by
hash, with two recorded exceptions: the WRF-side HRRR preparation
added the MSLMA field, fetched by byte range from the same archive
objects and hash-recorded; and the WPS static-geography trees on the
two preparation hosts are not identical. Four edges are stated rather
than implied:

- The Dudhia-shortwave radiation selector (`ra_lw_physics = 0`,
  `ra_sw_physics = 1`) has no exact WRF counterpart -- WRF's longwave
  selector has no "off" -- so its reference runs bracket the
  configuration between RRTM-longwave and radiation-off.
- Trajectory comparison against WRF is incomplete, so no combination
  is described as WRF-equivalent in this release. Maturity labels
  state what each combination has passed.
- HRRR with Kessler: the preparation retention policy above exists,
  but no registry template or profile admits the pairing in 1.3.0;
  its front-door admission is deferred.
- Checkpoint/restart content sealing (tracked internally as K-02)
  remains open and out of scope for 1.3.0.

## 1.2.3 (2026-07-30)

The 1.2.2 workflow stopped at its test gate on two defects the ubuntu
runner exposed: a doctor test that asserted the complete-tree Thompson
table state on a checkout where the externalized 243 MiB asset is
legitimately absent, and a remedy-contract test that flipped the shell
flag without re-deriving the build hints frozen from it at import, so
POSIX remedies were judged by PowerShell rules. Both tests now assert
each platform's own truth, negative-controlled, run against the public
byte-mirror under Python 3.11 and 3.13 on both shell arms. No runtime
code change.

## 1.2.2 (2026-07-30)

The 1.2.1 release stopped in its workflow before any wheel reached
PyPI: the release tooling wrote pyproject.toml with a UTF-8 byte-order
mark, which pip's TOML parser refuses at line 1. Rewritten without the
mark; the release tooling now writes both files byte-order-mark-free.
No code change.

## 1.2.1 (2026-07-30)

The 1.2.0 release workflow stopped at its test gate before any wheel
reached PyPI: `gpuwm/core/preflight.py` carried an f-string expression
broken across a line break, which Python 3.12 accepts and the supported
3.11 floor rejects. 1.2.1 is 1.2.0 with that statement rewritten in
3.11-compatible form and the whole tree compile-checked under 3.11.
No other change.

## 1.2.0 (2026-07-30)

`pip install gpuwm` followed by `gpuwm fetch-bridges` puts the compiled
GRIB decoders, the CPU preprocessing library, the fetch backbone and the
batch renderer onto a machine with no clone and no Rust toolchain.
History files carry the variable names, NetCDF types, axes, precipitation
accumulators and physics-selector globals a WRF reader expects, which
changes what a v1.1.x file looked like. `gpuwm fetch --source gfs`
accepts the model top the case asks for instead of stopping at 100 hPa.
The fetch and cache state machines take a single-writer lock, publish
atomically, verify cached bytes where they are used, and pace NOMADS from
the Python transport as well as the Rust one. Two documentation pages
state where the determinism claim and the `gpuwm adapt` trust boundary
stop.

### Breaking: what a reader of v1.1.x output must change

- **43 Noah-MP fields are written under WRF's external names.** The
  history writer upper-cased the internal symbol, so `tvxy` went out as
  `TVXY`. WRF's external name is the Registry `dname` column upper-cased
  (`tools/gen_wrf_io.c:331-334`), which makes those fields `TV`, `TG`,
  `ISNOW`, `TSNO`, `ZSNSO`, `SNICE`, `SNLIQ`, `PGS`, `T2V`, `Q2B`,
  `RUNSF` and so on. Two fields keep the symbol spelling, `qsnowxy` and
  `qrainxy`, because that is what their `dname` is. Anything keyed to the
  old `*XY` spellings reads a v1.2.0 file and finds nothing; anything
  keyed to WRF's names reads it and finds the field.
- **`EL_PBL`, `EXCH_H` and `EXCH_M` are written on `bottom_top_stag`,
  one level taller.** The Registry declares them `Z`-staggered, and WRF
  writes them at `nz+1` with the top interface left at the Registry cold
  value of zero, which is what its own PBL driver leaves there. A script
  that hard-coded `nz` levels for these three MYNN fields reads `nz+1`.
- **`--force-refetch` moves every regular file in `--out` aside.** The
  help text has always described a whole-directory replacement; the
  implementation quarantined per forecast hour, so a shorter forced
  window left earlier hours canonical and unlisted, and an interrupt
  mid-sweep left a manifest claiming bytes that had been replaced. The
  sweep is now receipts first (`fetch-manifest.json`, `SHA256SUMS`, the
  series) and payloads second, over every regular file in the directory,
  including index files, selector files, stale parts and files the
  operator put there. Nothing is deleted -- everything is renamed aside,
  files an earlier quarantine already set aside are left alone, and
  subdirectories are untouched. A directory that was half a fetch and
  half someone's notes comes back with the notes renamed.
- **Four integer fields are written as `NC_INT` with `FieldType 106`.**
  `KTOP_PLUME` and `KPBL` (MYNN), `ISNOW` and `PGS` (Noah-MP) are
  `state integer` rows in the Registry and are allocated as 32-bit
  integers in the model; they had been shipping as float32 with
  `FieldType 104`. Values were already integral, so the change is to type
  and metadata, not to content -- but a reader that assumed float32 for
  every variable except `ITIMESTEP` sees a different dtype.
- **Projection globals are single precision.** `DX`, `DY`, `TRUELAT1`,
  `TRUELAT2`, `STAND_LON`, `CEN_LAT`, `CEN_LON`, `MOAD_CEN_LAT`,
  `POLE_LAT` and `POLE_LON` are written `NC_FLOAT`, which is what stock
  WRF writes; they had been `NC_DOUBLE`. Both consumers tolerate either,
  but a byte-level comparison against a v1.1.x file differs.
- **A fractional history or restart cadence is refused at config
  admission.** `history_interval_s` and `restart_interval_s` must be a
  whole number of seconds. A quarter-second cadence on a quarter-second
  step divides evenly into steps and was accepted, and then wrote three
  distinct model instants onto one file name, because the filename and
  the `Times` string carry whole seconds. Sub-second history remains
  unsupported; it is refused rather than silently collapsed. The writers
  refuse a `valid_time` with a nonzero microsecond and a `Times` value
  longer than 19 characters for the same reason.

### Prebuilt Rust artifacts arrive as a download

- **Added: `gpuwm fetch-bridges`.** A pip install has never been able to
  read a GRIB file. The wheel ships no compiled Rust, so the five
  decoders, the CPU preprocessing library, the `rw_fetch` backbone and
  the `rw_wrfbatch` renderer had exactly one route onto a machine: clone
  the repository, install a Rust toolchain, and run `cargo build` in two
  workspaces. The artifacts are published as one release bundle per
  platform, and this command downloads the one matching this OS and
  architecture, verifies every artifact against the size + SHA-256 pins
  packaged in the wheel, and stages it into `~/.gpuwm/bridges` -- the
  directory the resolver already searched. Bundles are published for
  Windows x86-64 and Linux x86-64; the platform check asks what this box
  can execute and nothing else, and anywhere without a bundle keeps the
  build-from-source route, which works everywhere.
- **Every byte is checked three ways before it is installed**: the exact
  size, the SHA-256 pin, and -- for the decoders that declare one -- the
  contract marker `gpuwm doctor` looks for in an already-built binary.
  A staged file that fails any of the three is deleted rather than
  installed, and members are read out of the archive by their exact
  pinned filename, so nothing can be written anywhere the pins did not
  name. An interrupted download resumes against the partial, or restarts
  when the server ignores the range rather than appending to bytes it did
  not extend. `--from DIR` stages the same bundle -- or the loose
  artifacts -- from a local directory, offline, under identical checks.
- **A stale artifact is replaced, not refused.** This is the one place
  the contract differs from `gpuwm fetch-tables`, deliberately: a physics
  table that does not match its pin is the operator's file and is never
  overwritten, while a bridge executable that does not match is a build
  from a different release. Leaving that one in place is exactly the skew that
  made 1.1.0 preparations fail with a message blaming a file gpuwm had
  just written correctly. The replacement still happens only after the
  new bytes pass all three checks.
- **`gpuwm doctor` offers it first, and only when it is true.** Every
  Rust-artifact remedy on a wheel install leads with
  `gpuwm fetch-bridges` and keeps the clone-and-build block beneath it,
  commented out -- printed because it is the only route on a platform
  with no bundle, commented because a reader who pastes the whole report
  must not also compile what the line above already staged. The offer
  appears only when the pins this wheel carries name a bundle for this
  platform, so the report never advertises a command that would refuse.
- **Pins are generated during the release cut**, by
  `tools/build_bridge_bundle.py`, from the exact bytes the release
  uploads, before the wheel is built -- the bundles are compiled from the
  commit being released, so they cannot be pinned any earlier. The
  release workflow builds both platforms with
  `cargo build --release --locked`, uploads the bundles and their
  manifest to the release *before* publishing the wheel that points at
  them, and re-reads its own pins as the last step before PyPI.
  `RELEASE_CHECKLIST.md` carries the same sequence for a cut driven by
  hand. A tree that has not been through a cut declares no platform,
  which is what an unpinned document says instead of carrying a hash
  nobody computed.

### Scheme output a WRF reader can read

- **Every emitted scheme field has a record**, in
  `gpuwm/io/wrf_output_schema.py`: 85 fields across MYNN, Noah-MP and
  RUC, each carrying WRF's external name, its NetCDF type and
  `FieldType`, its stagger, and its description and units transcribed
  from the pinned v4.6.1 Registry with the Registry line cited. Where
  WRF's own strings are visibly wrong they are transcribed as they are,
  because a gpuwm file that disagreed with a WRF file about the same
  field is the failure this table exists to prevent. The writer refuses
  to publish a field whose Registry stagger contradicts the axis the
  dimension table gave it, refuses a float payload for a field WRF
  declares integer, and refuses a scheme field that has no record rather
  than shipping it without metadata.
- **The six precipitation accumulators WRF always writes are always
  written.** `RAINC`, `RAINSH`, `RAINNC`, `SNOWNC`, `GRAUPELNC` and
  `HAILNC` are core per-domain history fields in `Registry.EM_COMMON`
  with no package gate, so WRF writes all six whatever the physics
  selection is, filling them with zeros when nothing produced the
  quantity. gpuwm emitted each one only when the scheme that fills it was
  routed, so `RAINC` was absent from every `cu_physics=0` run, `HAILNC`
  from everything but NSSL2, and `RAINSH` from every file it had ever
  written. Every precipitation recipe in wrf-python and wrf-rust reads
  `RAINC + RAINNC` unconditionally, which is why a wrfout that omits
  `RAINC` fails an entire product family. The six are emitted always,
  zero-filled when the producer is absent. `RAINSH` is always zero and
  the code says why: gpuwm implements no shallow cumulus, and zero is
  what WRF writes for `shcu_physics=0`.
- **Eleven physics-selector globals are stamped into every history
  file.** `MP_PHYSICS`, `RA_LW_PHYSICS`, `RA_SW_PHYSICS`,
  `SF_SFCLAY_PHYSICS`, `SF_URBAN_PHYSICS`, `SF_SURFACE_PHYSICS`,
  `SF_SURFACE_MOSAIC`, `SF_OCEAN_PHYSICS`, `BL_PBL_PHYSICS`,
  `CU_PHYSICS` and `SHCU_PHYSICS`, as `NC_INT`, every one of them present
  in stock WRF output and cited to its Registry line. Four are constants
  because gpuwm has no such option to select, which makes WRF's "off"
  value the true one -- and it is what makes the accumulator zeros
  legible, since `SHCU_PHYSICS=0` beside `RAINSH` at zero answers a
  question neither answers alone. The radiation pair is resolved through
  `gpuwm.config.radiation_scheme_ids` rather than copied from the config,
  because gpuwm's `-1/-1` is a legacy sentinel meaning "use the aggregate
  spelling" and is not a WRF scheme id. The globals are written only when
  the writer was given a resolved run configuration, so an idealized
  caller receives none.
- **Identity in the files themselves.** wrfout carries a `GPUWM_VERSION`
  global; the restart header records the producing distribution, its
  version, and the restart format version. `TITLE` remains the caller's
  configured output title, which is why it never identified the producer.
  This is producer identity only: nothing in this release hashes or
  otherwise binds checkpoint or history *contents*.
- **A restart clock must be a real, finite, non-negative number.**
  Elapsed seconds are checked on write and on restore for every format --
  refusing NaN, both infinities, negatives, booleans and numeric strings
  -- and the header is written with `allow_nan=False`, so it cannot
  express the value at all. The idealized `Times` helper rolls over the
  calendar instead of emitting `0001-01-32` past the end of a month.
- **A checkpoint is fsynced before it becomes visible.** The standalone
  restart temporary and the feedback receipt are flushed and fsynced
  before the atomic rename, and the receipt stages under a unique name.
  Funnelling every publisher through one durability helper is recorded
  and not done.

### The GFS route follows the case's model top

- **`gpuwm fetch --source gfs --p-top-pa PA`** names the model top the
  fetched atmosphere must reach. The certified 21-level ladder is
  extended upward along whatever the live index publishes until a level
  sits at or above the requested top, which is the condition
  `gpuwm.vertical_contract` enforces at ingest. Requesting 5000 Pa adds
  the 70 and 50 hPa levels and nothing else. `--all-levels` takes the
  whole published ladder. An unsatisfiable request is refused by name,
  stating the deepest top the source offers.
- **The route stopped at 10000 Pa, with no flag and no receipt.** Every
  ArWen GFS run was capped at a 100 hPa model top by a hardcoded level
  list, the fetch manifest never recorded which ladder was taken, and the
  user met the consequence three steps later at ingest ("source
  atmosphere stops at 10000 Pa but requested p_top is 5000 Pa") with no
  mention of the fetch that chose it. The ladder and the source top are
  recorded in the fetch receipt and in the front-door manifest, the
  record-count bar is derived from the request rather than fixed at 124,
  the decoder derives its ladder from the source and reports what it
  decoded, and the vertical contract reads the declared ladder instead of
  a constant. A decode whose ladder the input manifest does not declare
  is refused.
- **Extending, never replacing.** Every level a certified run already
  used is present at every requested top, which is what lets the bridge
  and the front door check "is this the certified ladder, extended
  upward?" rather than trusting a number. Whole GRIB objects remain the
  default transfer shape; level subsetting stays an opt-in bandwidth
  saver and is no longer a ceiling on the model top.

### Fetch and cache state machines

- **One writer per output directory.** `gpuwm fetch` takes an
  OS-enforced lock -- a Windows byte-range lock or a POSIX `flock` -- on
  a file kept outside the output tree, keyed on the resolved target path.
  The CLI holds it across the prior-request guard and the transfer it
  authorises, and `fetch_gfs`, `fetch_hrrr`, `fetch_geog` and the table
  stager take it themselves, so a direct library caller gets the same
  contract. A second process queues and then refuses, naming the holder's
  pid. The lock is an OS lock because the kernel releases it when the
  holder dies: a crashed fetch must not leave a directory permanently
  unfetchable. `GPUWM_FETCH_LOCK_TIMEOUT_S` bounds the wait, default
  600 s, `0` to fail fast; `GPUWM_FETCH_LOCK_ROOT` moves the lock root.
- **Receipts describe only what finished.** An HRRR wait-timeout no
  longer lists a half-fetched hour in `files` and `SHA256SUMS` while
  `forecast_hours` omits it; the partial product stays on disk unclaimed
  and is re-verified under the ordinary bars on the next run. HRRR
  publishes a manifest after every completed hour, as GFS already did, so
  a kill after hour zero leaves a usable receipt. Assembly happens inside
  the unique staging directory, so a canonical `.part` is never created,
  and a legacy one left by an older release is swept by force.
- **Nothing is published under a name another writer could be using.**
  `atomic_write_text`/`atomic_write_bytes` stage under a name unique per
  process and per call, fsync, then rename, and fsync the containing
  directory where the platform has one -- Windows exposes no directory
  handle through the standard library, so there the guarantee is that the
  rename is atomic, not that it is durable across power loss. Quarantine
  proves `<name>.rejected-<stamp>` free before renaming, because
  nanosecond stamps collide inside a tick and the rename destroyed the
  older evidence. Table staging names are unique per writer and the
  stage-verify-install sequence runs under the table-root lock.
- **The raw download cache verifies at use.** Entries carry a sidecar
  recording the exact key, the byte count and a checksum; a read
  re-checks all three and renames a failing entry aside so the next
  request refetches it. Payloads land by atomic rename from a per-call
  staging name. A range response is validated before it is adopted: 206,
  a matching `Content-Range`, and the exact byte count asked for. A
  failed quarantine leaves the file in place and reports, rather than
  deleting the evidence.
- **The NOMADS governor fails closed, and paces the Python transport
  too.** A governor state file that exists but does not parse is read as
  "a request just happened" rather than as an empty state, and a request
  whose shared record fails to land still waits out the gap locally --
  both strictly stricter than before, in Rust and in Python. The GFS CGI
  transport, the HRRR index and range transports and the availability
  probes `--wait-for` polls all route through a Python governor speaking
  the same protocol over the same state files as the Rust client, so a
  Rust fetch and a Python fetch on one node pace each other. The NOMADS
  range pool narrows to one worker, since the governor serialises those
  requests anyway.
- **Geography reuse compares the tile corpus, not just the index.** File
  count and total bytes are checked against the extraction receipt, which
  catches missing, truncated, extra and added tiles; an install with no
  receipt keeps the index-only bar and says so. A same-size mutation of a
  tile's contents still passes, and closing that needs a content digest
  of a multi-gigabyte tree, which is recorded as a deliberate
  verification mode rather than something to run on every command that
  opens WPS_GEOG. Each archive's provenance entry is published as it
  lands, before the verified archive is removed, and publication re-reads
  and merges the canonical manifest under the geography-root lock instead
  of overwriting it. A resume is bound by sidecar to the first response's
  URL, ETag and Last-Modified, is sent with `If-Range`, and refuses a 206
  that does not start at the requested offset.

### Two pages that say where a claim stops

- **`docs/public/DETERMINISM.md`.** Consumer cards have no ECC, and
  running the forecast twice and comparing bytes is what stands in for
  it. The page states that as a transient-fault screen inside a fixed
  numerical environment and not as an ECC replacement, because equality
  cannot detect a fault that is identical in both runs. It names the
  seven undetected fault classes, the pin set that "fixed environment"
  actually means, the three mechanisms that make the pin set necessary
  (library-owned reduction order, FMA contraction and CUDA math
  functions, FP32 subnormal flushing), what each compared surface covers
  and omits, and the six known improvements that are recorded and not
  shipped -- the largest being that no fail-closed comparator command
  exists in this release. Linked from the README, `HARDWARE.md`, and
  `VERIFICATION.md`.
- **`docs/adapt-validation-contract.md`.** `gpuwm adapt` proves the
  emitted files implement your descriptor and that your GRIB files
  satisfy it. It does not prove the descriptor is a correct physical
  reading of them. The page gives every input dimension in two columns
  -- validated for you, trusted from your declaration -- and a
  self-check the reader can run for each trusted row. Wired into the
  adapt parser's description and epilog and into both of the command's
  completion messages, so it is found at the point of use.

### Input and checkpoint identity

- **`--directory-input-hash content`** (also `GPUWM_DIRECTORY_INPUT_HASH`)
  binds a declared directory input -- in practice the static geography
  tree -- by each file's SHA-256 instead of its mtime. The default stays
  `inventory`, which is cheap enough to run before every launch on a
  multi-GB tree but has two known modes that matter to a dual-run
  comparison: a byte-identical copy staged separately compares
  different, and a change that preserves path, size, and mtime compares
  equal. Every recorded hash carries the algorithm that produced it, and
  the `inventory` record layout is unchanged, so digests from earlier
  releases still compare equal.
- **Checkpoint discovery no longer ties.** Sets are ordered by valid
  time, then nanosecond mtime, then set id. Two sets at one model
  instant with tied second-resolution mtimes previously fell back on
  filesystem discovery order, which made the choice of resume point a
  property of the filesystem rather than of the run.

### WRF-Runner interoperability

Verified against namelist pairs generated by WRF-Runner (the
collaborator's workflow tool, branch New-PC-Updates) running its own
code, and against its plotting pipeline and viewer consuming gpuwm
history files unchanged.

- **`gpuwm import-namelist` no longer leaks a traceback on an unported
  selector.** A runner-generated pair carrying `ra_lw_physics=1` (the
  unported WRF RRTM longwave) crashed the CLI with a stack trace where
  every neighbouring refusal printed one actionable line; the
  `NotImplementedError` refusals from `validate_run_config` land on the
  uniform CLI refusal boundary -- message on stderr, exit 2, no partial
  output file.
- **The namelist support report classifies what WRF-Runner namelists
  actually carry, and gates two things it used to wave through.**
  `io_form_auxinput2`, `override_restart_timers`, `iofields_filename`
  and `ignore_iofields_warning` classify as runtime-only instead of
  eight lines of `UNCLASSIFIED_NAMELIST_SETTING` noise;
  `sf_surface_mosaic`/`usemonalb`/`rdlai2d` are state-relevant and
  value-gated to their WRF defaults with the exact selector named.
  Two new fail-closed codes: `NEST_INPUT_STREAM_UNSUPPORTED`
  (`fine_input_stream` nonzero -- WRF's delayed-nest-start pattern needs
  a per-nest input file RW-WPS does not produce) and
  `FDDA_INPUT_NOT_PRODUCED` (`grid_fdda` active -- `wrffdda_d0N` is a
  real.exe product, and the report previously classified the request
  runtime-only while blessing an export that cannot feed it).

### Release plumbing

- **The publish workflow runs tests.** The release path had no test gate
  -- a cut built a wheel and shipped it. It runs the
  packaging-and-contract suite first (what ships in the wheel, what
  doctor promises, what both fetch commands verify before installing),
  and runs it on pull requests too, because a gate whose first execution
  is a release cut is a gate that ambushes the cut. It is not the whole
  suite: the model's own tests need a CUDA GPU and staged case data, and
  five test modules import CuPy at module scope, so collecting everything
  on a GPU-less runner fails before a single test runs.
- **The job that writes release assets is not the job that publishes to
  PyPI.** Uploading the bridge bundles needs `contents: write`, and
  Trusted Publishing needs `id-token: write`; they are separate jobs, so
  neither credential is held by a job that has the other. The ordering is
  unchanged and is now a property of the job graph: the assets job
  uploads the bundles and writes the pins computed from those exact
  bytes, and only then does the publish job build the wheel around them
  and push it.
- **The RW-WPS standalone project stops reaching the forecast side.**
  `gpuwm/resume.py` is no longer staged into it -- nothing that wheel
  ships imports it, no entry point exposes it, and its own lookups are a
  module that wheel forbids and one it does not stage. `gpuwm/doctor.py`
  stays, because a preprocessing install is exactly the one that needs
  to be told which bridge is missing, and it reads its nine dataset
  names from `gpuwm.geog_assets` rather than reaching through the domain
  wizard into the CLI.
- **Both direct-proof descriptors validate against the physics
  registry again.** `configs/gfs_wrf_direct_proof.toml` and
  `configs/era5_wrf_direct_proof.toml` resolved through the legacy
  aggregate radiation spelling, which then demanded `ra_physics = 4`;
  naming `ra_lw_physics = 0` and `ra_sw_physics = 1` in each resolves
  them to `dudhia-shortwave`, and in each `radt` moves 12.0 to 1.0 and
  `diff_6th_factor` moves 0.12 to 0.08 -- the values the profile they
  identify as declares. The two descriptors are deliberately equal apart
  from source, start, and run length, and a test binds that equality.
- **The vendored `wx-core` is 0.3.10** and publishes a capability probe
  -- what this build's NOMADS governor is configured to do, read from the
  places the pacing code reads -- so a consumer can demonstrate the
  pacing rather than infer it from where the crate lives. Two copies of
  that crate shared version 0.3.9 and only one carried the governor; a
  dependency-graph reorder can no longer swap in the governorless copy
  under the same version string.

## Unreleased — physics reopening battery fixes

### The documented HRRR front door was dead, and the suite could not see it

- **Fixed:** `tools/prepare_hrrr_wrf.py` demanded receipt schema
  `gpuwm-hrrr-microphysics-initialization-v1` while
  `tools/hrrr_single_domain_benchmark.py` had moved to `v2`, so a fully
  successful preparation still raised `RuntimeError: HRRR preparation
  omitted deterministic cold-start evidence` -- for all seven profiles,
  taking the `gpuwm.wrf_direct` stock-WRF export and every
  `--root-preparation` tree with it. The consumer now validates the
  **fields**: the QC/QR/QI/QS/QG partition, the per-species
  decoded-source and initialized-state fingerprints, and the retention
  claim each species makes about itself.
- **Older schemas are deliberately not accepted.** `main` runs the
  producer itself and reads back the `report.json` that invocation just
  wrote, so an older receipt cannot reach this consumer and accepting one
  would widen the gate for a case that cannot occur.
- **The test shape was the defect.** `tests/test_prepare_hrrr_wrf.py`
  built the receipt it wanted instead of consuming the producer's, so the
  two were never bound to one schema constant. It now runs the real
  producer -- decoded native fields through `initialize_real`, then
  through the same `_initial_hrrr_microphysics_receipt` the benchmark
  binds into `report.json` -- so a future schema bump breaks the consumer
  the moment it lands.

### A receipt now states its own evidentiary strength

- The analyzed-hydrometeor retention gate is one-sided (`source nonzero
  > 0 and live nonzero == 0 -> raise`), so on a cloud-free analysis it
  cannot fire and a green receipt proves nothing. Each species is now
  recorded `PROVEN` or `VACUOUS` with the counts behind the verdict, and
  the domain carries a `retention_evidence_summary`. No threshold is
  invented for "negligible" beyond exact zero; the numbers an acceptance
  harness would need to impose one are published instead. Schema
  `gpuwm-hrrr-microphysics-initialization-v3`.

### RUC no longer meets a land column carrying water soil

- **Fixed:** a land column with `SOILTYP = 14` reached the RUC cold start
  and produced a non-finite `MAVAIL`, killing native HRRR RUC on its
  first surface call with zero model time advanced. WRF never lets that
  column exist: `dyn_em/module_initialize_real.F:3608-3650` reconciles it
  at `real.exe` time -- to land (`IVGTYP 5`, `ISLTYP 8`) from its soil
  temperature, to water from its SST, and to
  `wrf_error_fatal('mismatch_landmask_ivgtyp')` with neither -- and RUC
  assumes that without re-verifying it (`soilvegin`,
  `phys/module_sf_ruclsm.F:6973-6984`, has no `else` for `isltyp == 14`,
  so the column keeps zeroed soil parameters and `:913` evaluates
  `0./0.`). `gpuwm/core/landuse.py`, the transcription of that same cold
  start, now performs the reconciliation, which fixes every source at one
  seam rather than clamping a NaN where WRF has no clamp.

### MYNN is reachable with something other than WSM6

- `gpuwm import-namelist` had no mapping for `bl_pbl_physics = 5`, so the
  MYNN suite was expressible only through the one fixed profile that pins
  `mp_physics = 6`. The readiness authority already admitted the coupled
  5/5 pair and the dispatch already ran it; only the importer lagged. No
  gate is widened -- the half-suite, MYNN+RUC and MYNN+Noah-MP refusals
  are unchanged and pinned by tests.

### The air-gapped Rust build

- **Fixed, and the cause was not a missing crate.** Cargo finds the
  crates.io -> `vendor/crates-io` replacement in the crate's own
  `.cargo/config.toml` by walking up from the **working directory**,
  never from `--manifest-path`, so a build driven from the repository
  root bypasses the vendored registry and resolves against crates.io.
  `tools/prepare_hrrr_wrf.py`'s decoder auto-build,
  `tools/prepare_hrrr_500_native.sh` and `docs/benchmark.md` now `cd`
  into the crate as README/`install.sh`/`install.ps1`/CONTRIBUTING always
  did. A new test refuses any shipped build site that reaches a vendored
  workspace through `--manifest-path`, and checks every locked external
  package is actually vendored.

## 1.1.2 (2026-07-30)

### A saturated soil cell is packing, not corruption

- **Fixed, and this is why 1.1.2 exists:** a pip user's GFS preparation
  died with `RuntimeError: GFS Rust bridge failed: GFS_SM010040 value
  1.0000000019073487 outside [0,1]`. Nothing was wrong with their data.
  GRIB2 reconstructs every value as `(R + X * 2^E) * 10^-D`, so the
  representable values sit on a grid of spacing `2^E * 10^-D` and the
  encoder rounds onto it. A cell whose soil is physically **saturated**
  is encoded at exactly 1.0 and decodes one step above it -- and that
  reported number is exactly one step of that record's own grid
  (`E = -19`, `D = 3`, one step = 1.9073486328125e-9), which is the whole
  argument. It is a statement about the packing, not about the data.
- **How it is fixed, and how far:** every bound now declares what it
  **is**. A *physical* bound is a saturating limit real cells sit on;
  a value past it by no more than the tolerance derived from that
  record's own quantum is clamped onto the bound, counted, and
  published. A *sanity* bound is a slack plausibility range no real
  value approaches; it is offered nothing and refuses exactly as before.
  The offer cannot be talked upward by the record: it is the quantum,
  raised only to the round-off floor of the bound's own magnitude -- so
  a field with a slack ceiling cannot buy extra room at its physical
  floor -- and capped at 1e-4 of the field's declared range, so integer
  packing (one step = the whole range) or template 5.4's placeholder
  scale slots cannot widen the gate. `1.05` still refuses, and now says
  by how much and against what offer.
- **The exposure was never one field.** In the GFS bridge: `RH`, `RH2`,
  `SNOW` and `SNOWH` at zero; `XICE` and all four `GFS_SM` soil layers
  at both ends; and `LANDSEA`, whose 0/1 codes were matched against a
  fixed 1e-9 that the same grid can overshoot. In the HRRR bridge the
  identical shape: every field declared non-negative (a hydrometeor
  mixing ratio is exactly zero across most of a domain) and the
  `LANDSEA`/`XICE`/`SOILW` unit fractions. `Q2` keeps its exact ceiling
  -- one kg/kg of water vapour is an impossibility, not a limit real
  cells sit on, so no quantization argument applies to it.
- **The clamping is auditable, not invisible.** `gate.tsv` gains
  `bound_clamp_total`, `bound_clamp_max_excursion` and
  `bound_clamp_fields` (per field: count and worst excursion); both
  bridges' inventory manifests gain `clamped` and `max_excursion`
  columns. A run with no clamps says so with zeros and an empty field
  list, which is a stronger statement than silence.
- **And the refusal explains itself.** `GFS Rust bridge failed:
  <stderr>` was true and unreadable on its own -- a soil value of 1.05
  means nothing to someone who did not write the decoder, and the
  obvious reading, that the range is too tight, is the one action that
  must not be taken. An out-of-range refusal now carries what the number
  is, that a bound-kissing value is already clamped against the record's
  own packing step, and that a value past *that* points at the
  downloaded bytes: re-fetch and re-run. Every remedy line is a comment,
  so the block pastes whole. Failures that are not bound refusals gain
  nothing.

### Three surfaces that were telling users something untrue

- **Fixed:** `gpuwm.__version__` was a hand-typed `"0.1.1"` that four
  releases walked past. It is read from the installed distribution's
  metadata now, because two surfaces quote it to say which release is
  speaking: the prepared-cache provenance refusal, which told a 1.1.1
  user *"this is gpuwm 0.1.1"* -- a sentence whose entire job is to name
  the release -- and `rw-wps --version`. The test pins it to the
  metadata **and** rejects a release-shaped literal in the module, since
  a constant that matches today's install passes the first check and
  rots at the next cut, which is exactly what happened. Cache content
  digests are unaffected: the writer's version stamp was already outside
  the hashed basis.
- **Fixed:** `gpuwm doctor` announced *"NO basemap assets found"* on
  installs where `rw_wrfbatch` was drawing the coastlines it said were
  missing. Doctor probed one path -- the checkout's own
  `tools/rustwx/assets/basemap` -- while the renderer walks its own
  candidate list. Doctor now walks the same list in the same order: the
  two `RUSTWX_*` overrides, then `assets/basemap` (and the macOS
  `Resources` layout) under each of the first eight ancestors of the
  **executable's** directory -- which is how a build at
  `tools/rustwx/target/release` reaches the crate's assets two levels up
  -- then the working-directory walk, then the crate path it used to
  check alone. Doctor already resolved the renderer in order to report
  on it, so it had the missing fact all along. The warning survives for
  the case it was written for: nothing found anywhere still says so, and
  still names the environment variable.
- **Fixed:** the 20CRv3 authoring step printed nothing at all. The GFS
  route ends by printing the whole front-door command with its digest
  filled in, and every mapped authoring step prints an `AUTHORED` line;
  a user who had just watched a 20CRv3 manifest be written still had to
  locate it and compute its SHA-256 by hand. It now prints the
  `AUTHORED` line and a `next:` block: the half it knows
  (`--source-manifest` and `--source-manifest-sha256`, bound and exact)
  as a pasteable fragment, and the half it cannot know named in
  comments. It does not print a whole command, because 20CRv3 authoring
  deliberately **refuses** `--wps-namelist`, `--geog-root`,
  `--experiment-config`, `--output-root` and the two GRIB2 tool paths --
  those values do not exist in that process, and a command with
  placeholders in it fails when pasted.

### The same class, everywhere it existed

- **Fixed:** an audit reproduced the reported `1.0000000019` refusal on
  the HRRR and mapped routes, not only GFS. The strict `[0,1]` gates on
  HRRR source `LANDSEA`/`SOILW`, on the mapped soil output, on the HRRR
  soil nodes and on declarative mapped soil moisture all read a
  saturated cell as corruption. The bridges derive their tolerance from
  a record's own packing parameters; by these seams the packing is gone
  and only an array remains, so the head-room there is the round-off the
  pipeline demonstrably carries -- a few float32 ulps of the bound's own
  magnitude, the same constant the HRRR soil report already used for its
  convex-hull comparison. It moves cells that are AT a bound back onto
  it and leaves everything else untouched, so each existing refusal
  still sees, and still refuses, exactly what it did before.
- **Fixed:** mapped land fraction gained the check it never had. It was
  tested for finiteness and then thresholded at 0.5, so a mis-scaled
  unit transform delivering `2.0` was read as land without complaint. It
  is a fraction and is admitted as one. (The netCDF test fixture's own
  land fraction walked to 1.25 -- its helper adds 0.25 per time step to
  every variable -- and is corrected to a physical value.)
- **Fixed:** the stale version was not only printed, it was **sealed
  in**. The standalone RW-WPS wheel's pyproject carried a hardcoded
  `0.1.1` while `_installed_record_receipt` refuses unless that
  distribution's version equals `gpuwm.__version__`; the two agreed by
  coincidence. The moment the constant started telling the truth, a
  hardcoded version there would have failed the very seal it feeds. It
  is stamped from the package now, and a test proves a freshly sealed
  native contract, a prepared-cache writer stamp and that wheel all
  carry the distribution version.

### Commands that survive being pasted

- **Fixed:** a successful HRRR fetch printed a front-door command ending
  in a literal `...`, which its own consumer rejects with `unrecognized
  arguments: ...`. A GFS fetch without `--author-front-door-manifest`
  printed a template carrying `GFS_GRIB2_BRIDGE_EXE`, `NAMELIST_WPS` and
  `EXPERIMENT_TOML` and called it "next". Both now print the bound half
  as a real command and name the rest in comments.
- **Fixed:** the materialized GFS front-door command, the wizard's
  `next:` and `check:` lines, and both prepared-forecast commands
  interpolated paths bare, so a perfectly valid `--out`, config or
  `--outdir` containing a space split into two arguments the moment the
  command was pasted. They render POSIX display form and quote when a
  shell would split, exactly as `rw-wps --dry-run` already did. The new
  test shell-parses the printed line back to argv through a path with a
  space in it -- the old checks were lexical and could not see this.
- **Fixed:** the missing-CuPy remedy put a parenthesised alternative on
  the same physical line as the command it followed. Separate command
  and comment lines, which is doctor's form.
- **Fixed:** `--force-refetch` said it moves "every existing file in
  `--out`" aside. It moves the files that fetch would write; manifests,
  forecast hours you did not request and unrelated files stay. The
  behaviour was right and the scope word was not.

### Pages that described a build we no longer ship

- **Fixed:** the physics route table was labelled "state of play in
  v1.0.1" and claimed the GFS/HRRR door could prepare only YSU + MM5
  surface layer + Noah. v1.1.0 removed that coupling. The table now
  lists the shipped routes, names the one deliberate withdrawal (GFS +
  RUC, whose initialization the GFS route cannot supply), and points at
  `rw-wps --show-physics-registry` as the authority it summarizes.
  `DATA.md` repeated the same claim and now agrees.
- **Fixed:** README, `CONFIGURATION.md` and `PHYSICS.md` said two-way
  feedback is absent, `0 only`, or rejected at load, while 1.1.1 runs
  and stamps an experimental feedback path. They describe what ships,
  with its limits: experimental, stamped in run provenance, refused by
  one-way consumers, and feeding back dynamic state only where WRF also
  feeds back hundreds of masked land-surface fields.
- **Fixed:** `HARDWARE.md` kept the 3 GiB reserve for 12 and 16 GiB
  cards that v1.1.0 replaced with a flat 4 GiB, and said `gpuwm check`
  "still warns rather than blocks" after it began exiting 4 on an
  exceeded budget. Both corrected, with the distinction spelled out: a
  script reading the exit status is blocked, a person reading the output
  is advised, and nothing prevents a later `gpuwm run`.
- **Fixed:** the announcement draft said every physics scheme is "gated
  three ways". The registry says otherwise for MYNN, Noah, Noah-MP and
  RUC, and says so in warnings it prints on request. The draft now
  states which gates each option has passed and points at the registry.
- **Fixed:** README promised that every doctor gap "prints the exact
  command that fixes it". Doctor's own contract -- every remedy LINE is
  a command or a `#` comment -- is the accurate one, and one gap
  (`GPUWM_CASE_DATA_ROOT`) needs a path only the user knows.
- **Fixed:** `gpuwm domain --help` offered "any land point on earth"
  while the parser refuses a pole and the South Pole is land. The help
  now says what the gate does. The gate is unchanged.

### The last front door that said nothing

- **Fixed:** three sources reach the prepared-forecast runners -- GFS,
  ERA5 and 20CRv3 -- and two of them ended a successful preparation by
  printing a complete, hash-bound run command. ERA5 dumped its proof
  document and stopped, so a user who had just prepared a bundle
  reconstructed three SHA-256 values by hand, one of them findable only
  by grepping the JSON. It routes through the same shared printer now.
  The mapped route stays deliberately silent: `mapped` is not a
  `--source` either runner accepts, so a `next:` there would lead
  straight to a refusal -- the GDAS dead end this project already
  decided against.

### Guidance that still named a withdrawn gate

- **Fixed:** the RUC physics template's own warning said it is "OFFERED
  FOR ... gfs" while the route matrix withdrew GFS in v1.1.1 -- a
  GFS-initialised RUC forecast prepares and then cannot take its first
  step. The admission tests checked the matrix and never the prose
  against it, so the template kept telling users what the gate had
  stopped offering. The warning now names exactly the sources the matrix
  reaches (HRRR and ERA5) and explains the withdrawal, and a test
  asserts prose-and-matrix agreement so the two cannot drift apart
  again.
- **Tightened:** the pasteability guard added earlier this release now
  runs over BOTH prepared next-command branches -- single-domain and the
  multi-domain hierarchy -- with the config, namelist and output all in
  a directory whose name contains a space, shell-parsed back to argv.
  And the README's doctor claim ("a remedy whose every line is a command
  or a `#` comment") is now bound to doctor's real behaviour by a test
  that proves a genuine comment-only gap exists and rejects the old
  "every gap prints the exact command" overclaim.

### Carried, not fixed here

- Ordinary missing-file arguments still produce raw tracebacks on
  several public CLIs rather than a sentence. The fix pattern exists in
  this codebase; the surface is wide enough to want its own pass.
- Install and remedy guidance still points at mutable `main` rather than
  the released tag. Pinning it is a release-process decision, not a
  patch-lane one.

## 1.1.1 (2026-07-30)

### The GFS front door stops applying a single-domain gate to nests

- **Fixed, and this is why 1.1.1 exists:** 1.1.0 made the GFS front door
  apply the prepared **single-domain** forecast runner's physics profile
  whitelist to **every** configuration, including multi-domain ones. A
  `max_dom = 2` config that prepared cleanly on 1.0.1 was refused with a
  raw `ValueError` traceback minutes after 1.1.0 shipped. Three things
  followed from one mis-scoped call. The wizard's own note beside every
  emitted config -- *"the multi-domain (domain-tree) runner has no such
  whitelist and runs the suite above as written"* -- became false. The
  product's **default** suite (Thompson MP8 + Kain-Fritsch + RTE+RRTMGP)
  is deliberately not in that whitelist, so the config `gpuwm domain`
  emits when nobody names a profile could not pass the GFS front door at
  all. And the refusal arrived as a stack trace. The whitelist now gates
  single-domain preparation only, which is the boundary the wizard was
  describing all along.
- **Unchanged:** a single-domain config still meets the whitelist
  exactly as it did in 1.1.0, defaulting to the WSM6 profile when the
  caller names none, and an explicit `--physics-profile` is still
  enforced on either route -- a gate you asked for is not a gate to
  drop. On a domain tree it binds the root, because the wizard's own
  nested emission of a shipped profile turns cumulus off on the inner
  domains. Nothing the multi-domain path refused before 1.1 is accepted
  now.
- **New:** the multi-domain preparation records what each domain
  selected. `gpuwm-gfs-native-hierarchy-proof-v1` -> **`-v2`**, carrying
  a `gpuwm-front-door-physics-selection-multi-domain-v1` receipt: one
  selector set per domain (a child chooses its own cumulus and radiation
  cadence), the registry's semantic SHA-256, and the registry's names
  for each selection where it has them. Where it does not -- the
  committed two-domain descriptor selects the legacy aggregate radiation
  spelling with radiation off, which the registry has no option for --
  the receipt carries the blocker text instead of a guess, and the run
  proceeds. Naming is provenance here, not permission: requiring the
  registry to name a tree's physics would have been the same regression
  in better clothes. Both prepared-forecast runners accept v1 and v2 as
  distinct schemas and never promote v1 by inference, the rule the
  direct proof's v2 already lives under.
- **Fixed:** every refusal `python -m gpuwm.gfs_direct` makes now reaches
  you as one sentence and exit code **2** -- what the 20CRv3 door already
  costs for the same class of refusal -- instead of a traceback. That
  covers the two node-8 tracebacks on this path: the physics scoping
  above, and a manifest hash-bound to a namelist that was since
  re-pointed. The gates are untouched; only how they arrive is.

### RUC on the GFS route refuses at preparation, not mid-forecast

- **Fixed:** the RUC land-surface template was selectable through the
  GFS front door and could not complete a forecast. It prepared cleanly
  -- proof PASS, `land_surface: ruc-lsm`, nine soil layers, 339 MB of
  prepared state -- and then died 2.8 s into integration with `mavail
  must be finite`, having advanced no model time. The fail-closed guard
  did its job and the partial output was labelled
  `PARTIAL_NOT_RUN_PASS`, so nothing wrong was produced; what was wrong
  was spending the whole preparation to reach a refusal. The GFS route
  now refuses the pairing **before** any GRIB is decoded, with a
  registry-cited blocker that names what was observed. `gpuwm domain`
  refuses to emit the pairing at all, so the config never gets written.
- **Scoped to the evidence, deliberately:** RUC on the **ERA5** and
  **HRRR** routes is unchanged and still offered. It was not exercised
  by the run that found this, and withdrawing it on the inference that
  it shares the defect would refuse a path nobody has shown to be
  broken. Completing the GFS route's RUC land/soil initialisation is a
  v1.2 item, and the registry, the runner's capability declaration, and
  the front door now say the same thing about which sources offer it.
- **Mechanism worth knowing about:** the registry has always published
  which templates each route offers each source, but until now only plan
  validation and the GUI read that declaration -- nothing consulted it
  before a preparation ran. That gap is what let "selectable" and
  "usable" drift apart. Preparation now enforces the same declaration.

### Upgrading no longer invalidates what you already prepared

- **Fixed, and it affected every user with a prepared tree:** 1.1.0 gave
  every domain an optional per-domain `start_time` for staggered nest
  starts. The prepared-cache identity is compared by strict equality, so
  a header written by 1.0.1 -- before the field existed -- could never
  match again, and **every 1.0.1-era prepared tree became unrunnable
  under 1.1.0**, refused with `d01 cache domain config differs from
  experiment`. That sentence names the user's experiment TOML, which was
  innocent; the cause was a package upgrade. A field diff of a real
  preserved tree found exactly one added key and zero value differences
  among the eleven shared keys and ~110 `run` fields.
- **How it is fixed, and how narrowly:** a field the header does not
  carry, whose live value means the feature is not in use, describes the
  same prepared state as a header written before the field existed --
  for `start_time`, a domain whose start is the experiment's start,
  i.e. no delayed start. That case is accepted and the tolerated field
  names are recorded in the run's provenance. Everything else still
  refuses: a `start_time` that is genuinely late, any value that
  differs, any field the header carries and this build does not (a cache
  from a *newer* gpuwm), and every source/static/namelist/bridge digest,
  which are hashes of bytes and were never relaxed.
- **New:** cache headers now stamp the gpuwm that wrote them, outside
  the hashed content basis so that every existing cache's digest keeps
  verifying exactly as before. Residual mismatches now refuse with the
  honest cause -- `prepared by <version>, this is <version>; these
  identity fields differ: [...]` -- instead of pointing at a
  configuration file for a package difference.

### `gpuwm doctor` catches a bridge that predates the contract

- **Fixed:** the wheel ships no Rust, so upgrading the Python half
  leaves yesterday's bridge binaries in place. 1.1.0 changed the GFS
  series file from two columns to three, and `gpuwm doctor` reported
  every 1.0.1-era bridge `ok` because it only asked whether they
  launched -- after which each preparation died with `series line 1 must
  be HOUR<TAB>GRIB2`, blaming the series file gpuwm had just written
  correctly. Each bridge now declares a marker of the contract it
  speaks, and doctor reports a bridge that lacks it as **MISSING** with
  the rebuild remedy. The check is static, so it works on the binaries
  already on disk. `gpuwm.native_wrf_distribution` has applied this
  mechanism before sealing a distribution since before 1.1; there is now
  one table, shared, so the two surfaces cannot drift again.

### `gpuwm doctor`'s bootstrap wires what it builds

- **Fixed, pip installs:** pasting the whole report on a pip-only
  machine and running every line of it left doctor still reporting six
  MISSING bridges. Every line was honest -- the wiring step was offered
  as two `#` alternatives, copy into `~/.gpuwm/bridges` **or** set
  `GPUWM_<X>_BRIDGE`, because it genuinely is a choice -- but a choice
  printed entirely in comments means "run all the commands" does not
  close the gap the commands were for. The copy is now the printed
  **command**, correct for the shell you are in (`mkdir -p` / `cp`;
  `New-Item -ItemType Directory -Force` / `Copy-Item`, because Windows
  PowerShell 5.1 has neither of the first pair), and the environment
  variable is the `#` alternative beneath it. The destination is the
  default bridge directory spelled out in full rather than through
  `$HOME`, so the path you paste is the path gpuwm searches even where
  the two disagree.

### `gpuwm adapt` stops emitting a descriptor its own battery rejects

- **Fixed:** `gpuwm adapt --skeleton` gave four 3-D fields their
  **surface** counterpart's selector in addition to their own --
  `air_temperature` claimed the 2 m row that `air_temperature_2m` also
  claimed, and the same for relative humidity and both wind components
  -- so filling in the scaffold honestly and running the battery got
  `Vtable line 11 is assigned more than once`. The prefix rule that
  correctly collects `soil_temperature_0_0.1m` under `soil_temperature`
  was swallowing `air_temperature_2m` under `air_temperature`; it now
  takes the **longest** matching canonical name, which separates the
  four pairs without a hand-maintained exception list and leaves the
  four-layer soil collection intact.
- **Fixed:** that refusal named the Vtable and a line of it that was
  perfectly fine. Both claims live in the **descriptor**, so the message
  now names the descriptor and both fields that claimed the row -- which
  matters most when, as here, the tool itself wrote the descriptor.
- **Fixed, pip installs:** `configs/Vtable.GFS.rw-wps` -- the worked
  example the adapt flow is documented against -- lived outside any
  package, so the wheel never carried it and the documented command
  named a path only a checkout has. It ships beside the 20CRv3
  authorities now, under the same recursive package-data glob and the
  same byte-verified contract, and `gpuwm adapt --help` prints its
  resolved path on the install you are running. `--vtable` stays
  required and is deliberately never defaulted: this command adapts
  arbitrary sources, and quietly reaching for a GFS Vtable would
  mis-map every other product.

### `feedback = 1` says where it is supported, before you build

- **Fixed, guidance only -- no gate changed:** `feedback = 1` is a legal
  schema value, so a config could be authored, pass `gpuwm check` with a
  clean exit 0 and output identical to its one-way twin, and only then
  be refused at preparation -- after a 26 s hierarchy build -- by the
  prepared-hierarchy route, which supports static one-way nests only.
  `gpuwm check` now prints an advisory (and reports it under
  `advisories` in `--json`) naming the restriction: it changes no exit
  code and blocks nothing. And the preparation refusal now names the
  route that *does* run experimental two-way feedback -- the native
  experiment-runner route, `gpuwm run` -- instead of stating only what
  it will not do.

### Refusals

- **Fixed:** a `--preparation-receipt-sha256` that matched nothing
  printed `proof.json digest differs` once per accepted (file, schema)
  pair -- three identical lines, four once the GFS hierarchy proof grew
  a v2 -- and named neither the file it read nor the digest it found.
  One note per file now, naming the resolved path and the observed
  SHA-256 (or the schema it carries), plus the list of schemas that
  would have been accepted.
- **Fixed:** `rw-wps --source rap` printed
  `status=adapter_mapping_required: adapter_mapping_required` -- the
  reason fell back to the status value already on the line, and a
  doubled token reads as a truncated message. An adapter with nothing to
  add now says nothing rather than saying it twice. Adapters that do
  have something to say (`gdas`'s notes, the composition family's
  requirement) are unchanged, as is the paragraph underneath that
  explains the refusal.

## 1.1.0 (2026-07-30)

### `gpuwm check` tells the truth about VRAM

- **Fixed:** `gpuwm check` printed "observed peak envelope 12.98 GiB
  exceeds the WDDM budget 11.64 GiB" and exited **0**, so every script
  wrapping it read green out of a report whose own prose said the run
  might not fit. That case now exits **4** -- nonzero, and distinct from
  1 because no gate failed and the levers differ. A harder verdict still
  wins: 1 (a leg FAILED), 2 (nothing evaluable) and 3 (`--alloc`
  aborted) all outrank it. The warning line names the code it will exit
  with, so the reader of the text and the reader of `echo $?` learn the
  same thing.
- **Fixed:** the wizard's `--card 16gb` tier reconstructed a *notional*
  free-VRAM figure of 16.68 GiB -- more than a 16 GB card physically
  has -- because `--budget-gib` plus the reserve is arithmetic that never
  saw the card. `gpuwm check` gained `--vram-gib`, a ceiling and never a
  source: a declared free figure is now clamped to the smaller of the
  named card's capacity and a live NVML reading of it, reported as
  `CAPPED` in text and `free_bytes_capped_to_physical_bytes` in `--json`.
  Without `--vram-gib` nothing is clamped, because `--budget-gib` is how
  you size for a machine that is not this one.
- **Fixed:** `gpuwm check` sized its estimate and its observed-peak
  envelope without knowing the card, so on a 12 GiB card it applied the
  32 GiB machine's pool constants and the 1.75 Windows envelope factor
  while the wizard applied the small-card constants and 1.45 -- the two
  surfaces disagreed by 6.9 GiB on the same config. The wizard now passes
  its card size to `check`, and `check` sizes for the card it is told
  about.
- **Fixed:** the wizard's flat small-card reserve was 3.0 GiB, but the
  reserve policy `gpuwm check` actually applies charges 3.5-3.6 GiB on
  those same 12 and 16 GiB cards -- so the wizard sized layouts against a
  budget the preflight would never grant, and certified them anyway
  because of the exit-code bug above. The 12 and 16 GiB tiers now reserve
  4.0 GiB, the figure the 24 GiB tier already used.
- **Changed, and you will see it:** put together, the four fixes above
  mean **`gpuwm domain` emits smaller domains on the 12 and 16 GiB tiers
  than 1.0.1 did** for the same request. Nothing about your card
  changed; what changed is that the wizard and the preflight now size
  against the same card, the same reserve and the same envelope factor,
  and the answer they agree on is the smaller one. 1.0.1's larger
  domains on those tiers were sized against a budget the preflight would
  not have granted and a free-VRAM figure the card did not have -- and
  the preflight said so and exited 0 anyway. The 24 and 32 GiB tiers are
  unchanged. If you have a 1.0.1 domain that ran, it still runs; it is
  the *suggested* size that moved.

### Remedies

- **Fixed, pip installs:** `gpuwm render`'s matplotlib-fallback notice
  still ended `Build it with: cd tools/rustwx; cargo build ...` -- a
  directory a wheel install does not carry. The 1.0.1 remedy contract
  fixed that everywhere except here, because this call site assembled
  its own string. It routes through the install-aware machinery now, in
  a new one-line form: the `cargo` one-liner where the crate exists, and
  a pointer to `gpuwm doctor` where it does not, because the honest
  answer there is a whole bootstrap and this notice is contractually one
  physical line.
- **Fixed:** `bridges.sources_present()` answered for
  `tools/grib1_bridge` no matter which crate it was asked about, so a
  tree carrying one crate and not the other could be handed a `cd` into
  the missing one. It now takes the crate it is asked about;
  `install_aware_build_hint` passes it through.
- **Fixed:** `gpuwm doctor` says its remedy blocks run "as printed, in
  the order printed", and they did not. A `cd` into a crate never came
  back, so the block after it resolved its relative paths somewhere
  else; the repeated `git clone` a pip-only machine gets once per gap
  errored on every repeat; and two remedies ended with prose fused onto
  the end of a command line, which the shell hands to `cargo` as
  arguments. Every block now returns to the directory it started in,
  the clone carries a note telling you to skip it when the directory is
  already there, and every physical line of every remedy is either a
  command to run or a `#` comment. The claim is now enforced by a test
  that pastes the *whole* report as one sequence, in both shells,
  twice.
- **Fixed:** continuation lines of a multi-line remedy were printed at
  whatever indentation they were composed with -- 0, 2 or 4 spaces --
  so a block could start under the `remedy:` label and then jump to
  column 0. The whole block lines up now.

### The rust renderer says what is wrong

- **Fixed:** `rw_wrfbatch` answered an unknown `--products` slug, a file
  that is not a wrfout, a path that does not exist, and a bare
  `--list-products` with the *same* line -- its usage string -- and exit
  1. Three of the four never named the thing that was wrong. Each now
  names the problem: an unknown product names the token, the five group
  keywords and where to get the full list; an unusable input names the
  path and why, in the same wording matplotlib's engine uses
  (`<path>: unreadable wrfout (...)`).
- **Fixed:** a `--products` typo used to cost a full wrfout import
  before anything checked it, and then reported `No supported WRF files
  selected` -- about a file, not about the typo. The command line is
  checked before any file is opened.
- **Fixed:** the usage line was printed *after* the message, and
  `gpuwm render` reports the renderer's last line as the cause -- so
  every argument mistake reached you as a usage string. Usage first,
  reason last, as argparse does it.
- **Changed:** `rw_wrfbatch` exits **2** on a bad command line, matching
  what the matplotlib engine costs for the same mistake. `gpuwm render`
  still exits 1 either way, so scripts wrapping the front door see no
  change.
- **New:** `rw_wrfbatch --list-products` with no store, no output
  directory and no input prints the product vocabulary you may pass to
  `--products`. The store-aware listing -- which of those the frames you
  imported can actually render, and why not -- is unchanged and still
  needs a store.

### Guidance catches up with the gates

Every gate in this group was already right; the guidance around them
was not, and three of the four surfaced as tracebacks.

- **Fixed:** the GFS front door's own suggested `--outdir` was
  `<prepared_root>/forecast` -- a *child* of the preparation, while both
  prepared-forecast runners declare `--prepared-root` a protected input
  and refuse any output directory overlapping it. The front door's own
  next-command was therefore refused, as a raw traceback. The suggestion
  is now a genuine sibling, `<prepared_root>-forecast`, and the
  pasteable test no longer just looks at the printed text: it runs the
  runner's actual guard over the printed `--outdir`.
- **Fixed:** both runners' protected-input refusals reach the user as
  one sentence naming the problem and a directory that works, and exit
  2, instead of a traceback out of `claim_output_directory`.
- **Fixed:** the 20CRv3 front door rejected the wizard's default TOML
  with a raw `unknown table(s)/top-level key(s) ['case_data']`
  traceback. It now checks config compatibility *before* decoding any
  GRIB2 and refuses in one sentence that names both the incompatibility
  (`[case_data]` declares the ERA5 config-driven run path, which this
  door does not read) and a supported route to a config it accepts.
- **Fixed:** the 20CRv3 front door printed no run instruction at all,
  while the GFS door printed a complete hash-bound command -- after the
  1932 hindcast had just run end to end. Both doors now use the same
  printer, with `--source 20crv3`; every digest is resolved, so the
  pasteable contract holds without a holed command.
- `gpuwm domain` still has no `--source 20crv3` option. DATA.md's 20CRv3
  section now states the supported config route explicitly
  (`gpuwm domain --source gfs`, which emits no `[case_data]` table), and
  the front door's refusal names the same route.

### Any-combo front-door physics

- The single-domain GFS and HRRR front doors carry the *selected*
  registered physics profile all the way through configuration
  materialization, preparation, native WRF export, and content-hashed
  provenance. The exporter no longer applies a stock
  `bl_pbl_physics=1 / sf_sfclay_physics=91 / sf_surface_physics=2`
  identity gate when a front-door profile is supplied: it resolves each
  selector through the physics registry and checks the component's
  `implemented` declaration, required settings, pairings, and runtime
  readiness rails, failing closed with a registry JSON pointer and the
  published blocker when a component is unavailable.
- Three profiles that were declared but unreachable are now reachable on
  the prepared single-domain route: WSM6 + MYNN 5/5 + Noah, WSM6 +
  YSU/MM5 + RUC LSM (with its required nine soil layers), and WSM6 +
  YSU/MM5 + Noah-MP behind the registry-owned
  `noahmp-host-column-throughput-v1` expert acknowledgement. **This
  closes the 1.0.1 known issue** where the registry marked MYNN, RUC and
  Noah-MP `reachability: template` while `--physics-profile` rejected
  them: the two surfaces now agree, and they agree by making the schemes
  preparable rather than by hiding them.
- Provenance: `gpuwm-gfs-direct-wrf-proof-v2` ->
  `-v3` and `gpuwm-native-direct-wrf-export-v2` -> `-v3`, both carrying a
  `gpuwm-front-door-physics-selection-v1` receipt (profile, registry
  semantic SHA-256, resolved components, selectors, complete runtime
  settings, maturity, accepted acknowledgements). Preflight accepts v2
  and v3 as distinct schemas and fails closed on anything else; exact
  proof-file SHA-256 verification is unchanged for both.
- Stock-profile export identity, stated exactly: at integration time the
  pre-change exporter (`c16aed0e` `gpuwm/wrf_direct.py`) and this one
  were each run against *one* identical fully specified stock-profile
  prepared cache, static cache, geometry receipt, valid time and
  boundary cadence, and the two artifacts came out byte-identical --
  949,606-byte `wrfinput_d01` and 539,836-byte `wrfbdy_d01`, matching
  SHA-256 both sides. That comparison is **not** reproducible from this
  tree: it needed the archived base exporter and a real prepared cache,
  neither of which the release contains, and no committed fixture
  renders a stock export. The four digests and the exact identity inputs
  are recorded in the internal ledger
  `PRODUCT-V11-ANYCOMBO-20260730.md`. What the release *does* gate is
  the surrounding contract -- v2 proofs keep their historical shape, v3
  fails closed on absent, mismatched or future-schema physics receipts,
  and callers that omit a profile keep the legacy stock-only exporter --
  and that is tested (`tests/test_prepared_single_domain_forecast.py`).

### `gpuwm adapt`: arbitrary but verified GRIB2 sources

- New `gpuwm adapt` turns a WPS Vtable, an explicit
  `rw-wps.descriptor.v1`, and the caller's own GRIB2 files into a
  create-only runnable mapped adapter bundle. Its machine status is
  `runnable_mapping_not_stock_wrf_certified` -- it never widens or
  inherits a stock-WRF certification gate.
- Publication is gated on a battery that runs before anything is
  written: Vtable compilation into exact numeric GRIB2 selectors, a
  record inventory at every valid time (complete pressure sets, bounded
  soil selectors, no duplicates or member mixing, uniform cadence equal
  to the declared boundary interval), exact target units/axes/location,
  an *executable* decode of the selected records through the real GRIB2
  decoder, soil selector parity with contiguous surface-down coverage
  and no synthesized layers, one shared GDT 0 grid at scan `0x40`, an
  explicit source top at or above the model top, and stable before/after
  identity for every input and both decoder executables.
- Documented in `docs/arbitrary-verified-adapters.md`, including the GDT
  boundary and the runnable-versus-certified definition.

### GDAS

- `gfs_grib2_bridge` no longer infers forecast process 96 from
  `hour > 0`. The generic series contract is now
  `HOUR<TAB>GRIB2[<TAB>FORECAST_PROCESS_ID]`: a two-column legacy row
  declares analysis process 81 only, and ID 96 must be declared
  explicitly and stays inside the certified `{81, 96}` capability set.
  `gpuwm fetch` writes the per-row declaration. Certified against a real
  NOMADS proof corpus -- f000/f003/f006/f009 of `gdas.20260729/12/atmos`,
  124 messages each, frozen at centre 7, tables 2/1, PDT 4.0, GDT 3.0 /
  shape 6 / 0.25 degree / scan `0x40`, DRT 5.0 -- committed under
  `tests/fixtures/gdas-process-id/`. Each forecast sample is also
  required to fail under the undeclared analysis-only ID-81 policy. The
  corpus includes the **endpoint** of the certified span, so f009 rests
  on committed bytes rather than on the ladder constant.
- With the bridge re-certified, `gpuwm fetch --source gdas` serves the
  full **f000..f009** span again rather than 1.0.1's analysis-only
  window. The 1.0.1 scoping existed because the bridge was certified
  only against the analysis tag; that is the re-certification event it
  named, and it has now happened with real samples. Past f009 still
  refuses up front and says why.
- **What GDAS still is not:** a front door. `rw-wps --source gdas` has
  no ingest route and refuses. Every public surface now says so in the
  same words -- `docs/public/DATA.md`'s opening routing summary as well
  as its GDAS section, the README feature table, the `fetch` subcommand
  summary, and `fetch --help`'s `--hours` and `--cadence` text, which
  1.0.1 left describing GDAS as analysis-only. A test pins that help
  text to the registry's own `max_forecast_hour` so the two cannot drift
  apart again.
- The opt-in live GDAS smoke (`GPUWM_NETWORK_TESTS=1`) now fetches the
  hours it asserts about. It downloaded f000 alone, asserted the series
  had exactly one row, then required that row to carry both declared
  process IDs -- unsatisfiable by construction, so enabling it could
  only ever fail. It fetches the whole f000..f009 ladder from a live
  cycle and checks the census, the record bar and the declared process
  ID on every hour.

### Nests: delayed starts and sub-hour forcing

- `DomainConfig` carries an optional per-domain `start_time`. The root
  must still equal `[experiment].start_time` and an omitted child value
  inherits it, so existing configurations are unchanged; a child may
  start later than its parent, never before it and never outside the
  experiment run. `gpuwm import-namelist` reads real-shaped per-domain
  WRF `start_year..start_second` columns and writes a child `start_time`
  only where it differs from the root.
- The scheduler keeps a delayed child dormant while the parent advances,
  then at its seam runs the ordinary nest initialization against live
  parent state and begins normal stepping and output -- no child history
  before that point, and the parent's schedule and history are
  untouched. Restart headers carry `domain_start_time`,
  `domain_start_ticks` and `domain_lifecycle`; a checkpoint taken before
  the seam restores as `NOT_STARTED` and activates exactly once.
- **The whole-hour forcing refusal is gone**, because it was never a
  numerical requirement -- it came from representing the forcing
  inventory as integer hours. Generic mapped hierarchies now use exact
  `forcing_offsets_seconds` when the cadence is sub-hour. The contract
  that IS enforced: integer-second boundary interval, an exact integer
  number of root steps (root Davies/LBC state resets at a top-of-step
  seam), and every delayed child start on both an exact parent-step
  boundary and an exact global forcing seam. Refusals report the
  offending ratio, e.g. `cadence/dt = 31/6`.

### Feedback: experimental two-way nest restriction

- `[experiment].feedback` accepts `1` as an **experimental** capability
  (default remains `0`; `smooth_option` remains restricted to zero
  because parent smoothing is unimplemented). Launch prints the
  experimental warning, the run writes `feedback-provenance.json`, and
  every per-domain wrfout carries `GPUWM_FEEDBACK`,
  `GPUWM_FEEDBACK_VALUE`, `GPUWM_FEEDBACK_STOCK_WRF_CERTIFIED = 0` and
  `GPUWM_FEEDBACK_CERTIFICATION`. Nothing is stamped at `feedback = 0`.
- `feedback = 0` byte identity, and how to re-run it. At integration
  time an archived pre-feedback tree (`e7bf4d88`) and the lane tip
  (`80561c6e`) each ran the same two-domain deterministic schedule at
  `feedback = 0`, and every serialized state byte matched -- whole-tree
  SHA-256 `727ac476e0ebbf97a89be350a151c593e2b9447cc5d9b14fe0436d5f89e47557`,
  with the per-domain digests recorded in the internal ledger
  `PRODUCT-V11-FEEDBACK-20260730.md`. The release cannot re-run the base
  half of that comparison -- it does not contain the base tree -- but it
  no longer has to take the result on trust either: that pre-change
  digest is now frozen in
  `tests/test_feedback.py::test_feedback_zero_output_is_pinned_and_costs_nothing`,
  which re-derives it from the shipped code and additionally proves the
  dormant path is free (the same run with the feedback call path removed
  entirely produces the identical digest, and the coupler records zero
  transactions).
- The transaction is schedule-owned and three-phase -- restrict `MU`
  first, couple each remaining child prognostic into the existing nest
  scratch arena, call WRF's own `copy_fcn` transliteration over the
  exact registration, uncouple into the parent only on the feedback
  rectangle, refresh parent diagnostics -- at synchronized exact-integer
  clocks, including at a delayed child's activation seam. Restart
  continuation is SHA-identical to an uninterrupted run.
- Operator classes were verified against tagged WRF v4.6.1 source: mass
  (`copy_fcn`, 16-point average at ratio 4, odd centered path at ratio
  3), U on x-faces and V on y-faces all **match**. The masked/integer
  class is a **documented divergence**: ArWen's authoritative
  `nest_field_kinds` inventory contains no masked or integer feedback
  fields, so stock WRF's `LU_INDEX`/`TSK`/`TSLB`/`SMOIS`/... restriction
  has no ArWen counterpart. The `copy_fcnm`/`copy_fcni` kernels remain
  present and exact for a future explicit inventory extension.
- Every one-way consumer fails closed on a feedback-modified parent:
  offline-child/downscale, the static source-hierarchy exporter, RW-WPS
  stock export, and the explicitly-one-way prepared-tree runner.
- `tools/compare_feedback_signature.py` (schema
  `gpuwm-feedback-signature-comparison-v2`) compares a four-run
  ArWen/WRF feedback-on/off matrix over the parent overlap with the
  child specified zone excluded, annotating every row with operator
  class, WRF routine, stagger, stencil and source-point count. It is
  signature comparison, not bit or amplitude certification, and the
  thresholds are synthetic defaults until the real reference pair is
  run. The experimental label stands until then.

### Hygiene

- **Fixed, a fabricated CFL:** the shared one-readback health reduction
  now computes `max(|w_upper| / dz_cell)` on device, with `dz_cell`
  taken from that same cell's live `(php + phb)` geopotential faces.
  The old formula combined global extrema, so 100 m/s aloft over its own
  1,000 m layer reported CFL 100 against an unrelated 10 m surface
  layer; the co-located term reports 1 and passes, while 11 m/s over the
  actual 10 m first layer reports 11 and fails. Both single-domain
  runners share one threshold predicate: exactly 10 passes, the first
  representable FP32 value above 10 fails, and the independent 150 m/s
  `w_max` and non-finite guards are unchanged.
- **Fixed, interrupted fetches:** the GFS/GDAS fetch loop atomically
  rewrites its series and fetch manifest after every verified hour, so
  every published prefix contains only files that passed the full GRIB2
  envelope walk, the resolved record-count bar, and SHA-256.
  `KeyboardInterrupt` is now a traceback-free operational error that
  names each verified file (hour, bytes, digest), separately names any
  `.part` or otherwise unverified payload still on disk, and prints the
  exact `gpuwm fetch` command to resume the original
  cycle/window/cadence/area/output. A first-hour interrupt publishes an
  empty request-identity manifest rather than recreating the old
  manifestless-resume trap.
- **Now loud:** a requested box that crosses the prime meridian cannot
  be one NOMADS request (the CGI accepts a single `[0,360]` interval),
  so it falls back to the full longitude band. That fallback was already
  correct and silent; before any bytes move it now names the requested
  box, the actual band being sent, and `360 / requested_width` as the
  exact longitude-span amplification -- with the compressed-byte factor
  explicitly labelled data-dependent. A 20-degree UK-style box reports
  18x; an equal-width dateline box stays narrow and prints nothing. The
  same area and amplification are recorded in the fetch manifest. No
  unverified two-request stitcher was introduced.

## 1.0.1 (2026-07-30)

- **Documented, not new:** the 20CRv3 ensemble-member route. The
  adapter has shipped and is runnable, but no front-facing document
  mentioned it, so nobody could find it. README and DATA now describe
  it in the registry's own terms -- runnable and experimental, not yet
  accepted by unchanged stock WRF, one exact member bound by a
  filename-plus-hash manifest, paired three-hourly pressure/surface
  analyses, one-way Lambert nests through `max_dom = 4` -- plus what the
  registry does not say: **there is no fetch route and the inputs are
  not self-serve.** Every-member
  20CRv3 GRIB2 generally needs access to the NOAA-CIRES-DOE archive
  holdings; the publicly downloadable mean/spread NetCDF products are a
  different family and are not inputs to this route.

### Known issues

- The physics registry marks the MYNN, RUC and Noah-MP components
  `reachability: template` while `--physics-profile` rejects them as
  invalid choices. Both surfaces are telling the truth about different
  things -- the schemes exist and are declared, but the prepared
  single-domain runner has no profile row for them -- so they are
  visible without yet being preparable on that route. Decoupling the
  two surfaces is scheduled for the next release; until then, the
  profile list in `gpuwm domain --help` is the authority on what the
  prepared route will accept.

- **Fixed, pip installs:** `gpuwm doctor`'s six bridge remedies told a
  pip user to `cd tools/grib1_bridge`, a directory a wheel install does
  not contain, and named no repository anywhere -- so the one machine
  state where *nothing* can decode GRIB got a remedy that reads as a
  broken package. Every Rust remedy is now aware of which install it is
  printing on: in a checkout it stays the single `cargo build` line and
  names the real output path; on a pip install it prints the whole
  bootstrap: the Rust install and the PATH activation that makes cargo
  usable in the shell already running (rustup only edits the login
  profile), then `git clone`, then the build -- about two minutes end
  to end. Every emitted line is either a command spelled for the shell
  this platform actually has -- Windows PowerShell 5.1 cannot parse
  `&&`, so it gets `;` and separate lines -- or a `#` comment; nothing
  is prose fused onto a command, which is what `install Rust: winget
  ... (or https://rustup.rs)` was. The renderer and fetch-backbone
  remedies share the same builder, so they no longer carry `<clone>`
  placeholders, and the pip-extra and `fetch-geog` hints put their
  explanation on `#` lines instead of after the command. README's
  install section says plainly that a pip-only install refuses every
  real data source until the bridges are built. Doctor's closing line
  claims only what it can prove: every remedy line is a command to run
  as printed, in order, or a `#` comment.
- `gpuwm doctor` reads the render extra's version from installed
  package metadata rather than the module's `__version__` attribute,
  which lagged a release (0.2.34 reported on a machine with wrf-rust
  0.2.35 installed).

- **Fixed:** a platform gpuwm has never measured now gets the
  *conservative* VRAM accounting, not the optimistic one. Detection
  recognized Windows (with Cygwin and MSYS) and gave everything else
  the Linux envelope -- so Darwin, a BSD, or any future platform name
  was priced with the numbers that omit 4.12 GiB of fixed pool
  constants, on no evidence at all. Those numbers are three runs on two
  Linux cards, not a default. An unrecognized platform now takes the
  Windows envelope and the wizard prints one line naming the platform
  and saying the accounting may size a smaller domain than the machine
  can run. WSL and Linux containers report `linux` and are unaffected.
- **Fixed:** `rw_fetch`'s probe receipt recorded an absent `.idx` (404)
  as one that arrived and failed validation, so the printed reason said
  the index was malformed when it simply was not published yet. The
  transport half of the index read now tags its own failures instead of
  the caller re-deriving them from prose -- the old prefix match could
  never fire, because the error text it looked for is always preceded
  by the error type's own `HTTP error: `. Transport choice was already
  correct (both classes take the full file); this is receipt fidelity.

- **Fixed:** the front door's printed next-command was not pasteable.
  It filled in the three SHA-256 values -- the whole point of printing
  it -- and then asked for `--physics-profile <the profile this config
  was materialized for>` and `--outdir OUTPUT_DIR`. The profile is now
  resolved by asking the same table the runner's own guard asks, and
  the outdir names a real directory beside the preparation, so the
  command runs exactly as printed. The multi-domain branch printed a
  runner name and a fragment; it now prints that runner's whole
  command, digest and all. A config bound to no shipped profile gets
  prose saying so, never a command that cannot run -- and the same
  treatment reached `gpuwm fetch --author-front-door-manifest`, whose
  `rw-wps` line carried `--output-root OUTPUT_DIR` and `--geog-root
  WPS_GEOG_DIR`.

- **Fixed:** an HRRR inventory-change refusal on the Rust backbone said
  "Nothing was downloaded" when the payload was already on disk. That
  transport can only report a record census after it has written the
  object, so the tripwire necessarily fires late -- and it left an
  unverified GRIB plus its `.idx` in a directory with no manifest,
  which the next ordinary run also refused, for a different and
  confusing reason. The refusal now quarantines both files
  (`*.inventory-change-<ns>`, nothing deleted) *before* raising, and
  says what is on disk and where. Refusals that genuinely precede any
  transfer -- the GFS route derives its bar from the live index first --
  still say nothing was downloaded, because there it is true.
- **Fixed:** a completed HRRR resume erased the record that an
  inventory change had been accepted. Re-running the command on a
  directory whose files are all present downloads nothing, resolves no
  record bars, and republished the manifest with `record_bars: []` --
  losing `inventory_change_accepted: true` from the fetch that really
  did accept it. The prior manifest's bars now seed the run, and any
  product actually re-fetched replaces its own entry.

- **No shipped file carries one machine's absolute paths any more, and
  the release build now refuses to let one back in.** 51 lines across
  25 tracked files still held a developer's WSL/Windows/POSIX home --
  executable defaults in the WRF-oracle build scripts, a committed
  `nvidia-smi` process capture, oracle provenance notes, a hardcoded
  scratch directory. Files a stranger can use were parameterized
  (`$WRF_TREE`, `$WRF_SOURCE_ROOT`, `$GPUWM_REPO`, `RW_STORE_ROOT`,
  `SNOW_PROBE_SCRATCH` -- all fail-closed with a message naming what to
  set); the campaign ensemble harness, which needs a privately-built
  WRF oracle and a private reference bundle and so cannot run outside
  the campaign at all, is excluded from the release instead. The
  snapshot builder now scans every staged text file for the *shape* of
  a per-user absolute path and FAILs the build on any hit, because an
  exclusion manifest cannot name the file that grows one next.

- **Fixed, data loss:** `gpuwm render` silently overwrote one domain
  with another. Two nests of one run share model, init cycle, forecast
  hour, and (on the whole-hour axis) everything else the filename
  carried, so rendering two domains at one lead into one directory left
  ONE PNG -- with no error, no warning, and exit code 0. Every output
  filename now carries a domain + resolution token
  (`..._d02-3km_composite_reflectivity_...`; sub-kilometre nests as
  `_d05-111m_`) on both engines, read from the file's own `GRID_ID` and
  `DX`. Both engines degrade by the same three rules: identity plus a
  usable `DX` gives `d02-3km`; identity without one gives a bare `d02`;
  and a file whose domain identity cannot be established keeps the
  generic `native_grid` slug rather than being labelled `d01` on no
  evidence. On the rust engine, which imports several inputs into one
  store and renders them as one run, inputs that disagree get no token
  at all. The matplotlib engine renders each input separately, so it
  instead refuses -- naming both files and the remedy -- when two
  inputs it could not identify would write the same output name, rather
  than letting the second overwrite the first.
- `gpuwm render --pair` keys on the domain token as well as the product
  slug, so a directory holding several nests cannot pair a 3 km panel
  against a 333 m one. Sheet names gain the domain
  (`d02-3km_sbcape-pair.png`); the panel labels are unchanged.
- Plot subtitles gain the grid spacing the file declares -- `Î”x 3 km`
  at and above a kilometre, `Î”x 111 m` below it -- on both engines and
  on all four rust render lanes (direct, derived, heavy, windowed).
- Locally-imported runs are labelled **ArWen** instead of the GDEX
  fetch source inherited from the `wrf` store-model identity, which
  this lane never fetched from. New `gpuwm render --source-label TEXT`
  renames it for rendering stock-WRF files honestly.
- The matplotlib fallback notice is now genuinely one line, and carries
  the three things needed to act on it: the engine actually in use, the
  products available against the rust catalog, and the exact
  `cargo build --release --locked --offline` command. The multi-line
  remedy stays in `gpuwm doctor`, where it can be read at leisure.
- **Windows cards at or below 12 GiB are now sized as an EXPERIMENTAL
  tier instead of being refused.** The refusal was never "your card is
  too small": at the wizard's smallest layout, 4.12 GiB of a 5.38 GiB
  projection was grid-independent constants measured on one 32 GiB
  RTX 5090 running campaign-scale forecasts, so no ladder could fit and
  no smaller grid could help. Small Windows cards are now priced like
  Linux -- itemized alloc estimate under the 1.45 envelope -- plus a
  single reduced 1.5 GiB fixed reserve, and every sizing prints an
  honest pioneer warning: the accounting is extrapolated from one much
  larger machine, the worst case is paging or a clean out-of-memory
  failure (neither corrupts a forecast), and please report your
  measured peak. Windows cards of 16 GiB and up are unchanged, and so
  is `gpuwm check`, which warns rather than blocks as before.
- FIRST-LIGHT, README, and HARDWARE named `run-progress.json` as *the*
  progress file. It is the config-driven `gpuwm run` route's file only:
  the domain-tree runner writes `<outdir>/evidence/progress.json` and
  the single-domain runners write `<outdir>/progress.json`. All three
  documents now state the truth for every route.
- The single-domain CFL safety gate now reduces
  `max(dt*|w_upper|/dz_cell)` with velocity and live geopotential
  thickness from the same cell. It no longer combines a global
  upper-level updraft with the unrelated thinnest surface layer.
  Genuine thin-layer violations, non-finite geometry, the CFL 10
  threshold, and the independent 150 m/s vertical-speed guard remain
  fail-closed.
- GFS/GDAS fetches now atomically refresh their series and fetch
  manifest after every verified hour. Ctrl-C reports the exact
  digest-bound prefix and any unverified partial file on disk, exits
  without a Python traceback, and prints the exact resume command; a
  good completed hour is never discarded merely because a later hour
  was interrupted.
- A GFS/GDAS box crossing 0 degrees longitude no longer silently widens
  to NOMADS' full `0..360` band. The verification-preserving full-band
  fallback now prints and manifests the requested box, fetched band,
  and exact longitude-span amplification (while labelling compressed
  bytes as data-dependent). Splitting remains deferred because
  concatenating two 124-record grids is not one geometry-valid
  124-record product.

- New `gpuwm fetch-geog`: downloads and stages the nine WPS_GEOG
  static datasets the static builder opens (~1.3 GB download, ~16 GB
  unpacked) -- previously an entirely manual step and the launch-day
  pilot's #1 finding. Default source is the ArWen Hugging Face mirror
  (CDN bandwidth); `--source ncar` fetches upstream; both are verified
  against packaged size + SHA-256 pins (NCAR publishes no checksums;
  the pins were computed from UCAR's bytes on 2026-07-29). Resumable
  (HTTP Range), idempotent, safe extraction, per-dataset WPS `index`
  validation (doctor's own bar), and a local
  `geog-fetch-manifest.json` audit record. `gpuwm doctor` and the
  wizard now print this command as the WPS_GEOG remedy; DATA.md
  documents both routes, the exact manual URLs, and per-dataset
  provenance/attribution.
- The WPS_GEOG mirror lives at
  `huggingface.co/datasets/deepguess/wps-geog-arwen`.
- New `rw_fetch`, a Rust download backbone built from the vendored
  `tools/rustwx` workspace and driven by `gpuwm fetch --engine rust`
  (HRRR today; `auto` uses it when built, the Python transport stays
  the always-available fallback). It brings 16 MiB parallel range GETs,
  `.idx` range coalescing, a cross-process NOMADS rate governor
  (2.5 s minimum request gap plus a node-wide cooldown, shared by every
  process on the machine), and a URL+range disk cache (`--cache-dir`).
  `gpuwm doctor` probes it, including an exact fetch-record ABI check
  so a stale binary fails before a download rather than after one.
- New `gpuwm fetch --mode auto|full-file|idx-subset`: the byte
  transport, chosen by **probe** rather than by any time constant. The
  last indexed message's declared length is read from its own header
  and one byte past it is requested; an index that provably ends where
  the object ends earns a range subset, and an index that is absent,
  malformed, short, or unprovable gets the whole file. `idx-subset`
  refuses rather than silently degrading. Full-file HRRR hours feed
  `hrrr_grib2_bridge` unchanged -- it selects by field identity, not by
  file size.
- Record-count bars are now **derived from the live provider
  inventory**, with the certified counts (124 GFS, 561 + 18 HRRR) kept
  as a tripwire: agreement is silent, disagreement names both numbers
  and refuses until `--accept-inventory-change` makes the live count
  the bar and records the acceptance in the fetch manifest. Never
  silently adapt, never mystery-break.
- NOMADS and AWS do **not** publish identical HRRR `.idx` field names:
  NOMADS says `CLWMR` where S3 says `CLMR`. gpuwm treats that as an
  alias (one role, either spelling) instead of hard-failing
  `--transport auto`, and an inventory a host publishes that gpuwm
  genuinely does not recognise now falls back to the other host with an
  explanation rather than ending the run. DATA.md's byte-identical
  claim is corrected: the GRIB files are identical, the indexes are
  not.
- `gfs_grib2_bridge` accepts both published row orders -- the
  grib-filter crop's south-to-north `0x40` and the raw archive's
  north-to-south `0x00` -- and normalizes to one on decode, with the
  gate receipt recording `source_scan_mode` and whether a flip
  happened. Proved bit-identical (no tolerance) against a committed
  matched pair in `tests/fixtures/gfs-scan-order/`. Every other scan
  mode stays fail-closed, and so does GRIB2 template 5.3, which is what
  actually still blocks the raw S3 archive.
- New `gpuwm fetch --source gdas`, **scoped to analysis init**: the GFS
  assimilation cycle through the certified GFS container -- same grid,
  codes, 124-record census, centre and tables -- so it reuses the
  certified mapping, bridge and front door with a source tag. The
  accepted window is `--hours 0`, the f000 analysis, because that is
  what has been verified end to end: NCEP tags GDAS forecast hours with
  a different forecast generating process than the analysis, and the
  fail-closed `gfs_grib2_bridge` is certified against the analysis tag.
  A request past f000 -- on the CLI or in a config `[fetch]` table --
  refuses up front, says why, and points at `--source gfs` (certified
  through f384) rather than downloading files the bridge would reject.
  f000 is an analysis-quality initial state at identical cost and
  through identical machinery; no comparative forecast-impact claim is
  made, because none has been measured.

## 1.0.0 -- first public release (2026-07-29)

ArWen: an independent, GPU-native implementation of a WRF-ARW-class
regional model. Not affiliated with or endorsed by NCAR/UCAR. Research
and educational tool; never a substitute for official warnings.

### Worldwide projections

- Mercator, polar stereographic (both poles), and southern-hemisphere
  Lambert conformal grids now run end to end: `gpuwm domain`
  auto-selects the projection from the point latitude (below 25
  degrees absolute Mercator, 25-60 hemisphere-correct Lambert, above
  60 polar stereographic; `--projection` overrides), the config schema
  gains `[projection] map_proj = "lambert" | "mercator" | "polar"`
  (the `[shared]` integer keeps the WRF convention 1=lambert, 2=polar,
  3=mercator and must agree), and namelist import/emission, the
  WPS_GEOG static build, ERA5/GFS ingest, and the native WRF direct
  export (`MAP_PROJ`/`MAP_PROJ_CHAR` derived per projection) all
  follow.
- Antimeridian-crossing domains are supported: `gpuwm fetch --area`
  reads a longitude pair spanning more than 180 degrees as the
  complementary box crossing 180E, `--point` boxes wrap across the
  seam, and GFS antimeridian crops decode onto a continuous axis.
- New wizard ladder `12` emits a single 12 km domain
  (`restart_interval_s = 0`, the portable prepared-forecast contract).
- `gpuwm check` on a config without `[case_data]` (the GFS/HRRR wizard
  emissions) now prints that the input preflight is not applicable and
  certifies the memory preflight (exit 0) instead of refusing.
- Maturity, stated plainly: the new projections (Mercator, polar
  stereographic, southern-hemisphere Lambert) are oracle-verified and
  smoke-run verified -- transcription gates at binary64 against a
  Fortran oracle built from the pinned WRF v4.6.1
  `share/module_llxy.F` (tools/llxy_wrf461_oracle, fixtures in
  tests/data/llxy_oracle, gates in tests/test_projection_oracle.py
  with measured max-ULP ceilings) plus short GPU smoke integrations --
  NOT matched-run verified. The deep matched-run validation (the 1974
  reference family, geo_em byte-level gates) remains
  northern-hemisphere Lambert only.
- Genuine limits that remain: domains containing or touching a pole
  are refused; forcing footprints wider than 180 degrees of longitude
  are refused; latitude-longitude (cylindrical) and rotated grids fail
  closed; HRRR remains CONUS Lambert only (worldwide points use GFS or
  ERA5, both global).

### The model

- WRF-ARW-class compressible nonhydrostatic core (RK3, split-explicit
  acoustics) in FP32 on CUDA; one-way static Lambert-conformal nests
  with WRF-recurrent boundary-clock semantics.
- Physics transcribed from WRF v4.6.1 (`d66e442f`) with per-option
  maturity labels (docs/public/PHYSICS.md): Kessler, WSM6, Thompson
  (model-validated; WRF tables SHA-256-pinned -- the two largest are
  published as release assets and staged by `gpuwm fetch-tables`,
  which install runs automatically), Morrison
  2-moment, NSSL 2-moment (validation-candidate) microphysics; YSU and
  MYNN PBL; MM5 and MYNN surface layers; Noah, Noah-MP, and RUC land
  surface; RTE+RRTMGP (default) and legacy-RRTMG (verification tier)
  radiation; Kain-Fritsch cumulus.
- Verification against WRF v4.6.1 at three levels -- component ULP
  oracles, t=0 initialization parity, matched 6 h four-domain forecast
  to 500 m -- with published decay tables and explicit non-claims
  (docs/public/VERIFICATION.md). Measured: 6 h on a 250x200x49 domain
  with full physics in 3.6 min on one RTX 5090.

### The product surface (all new in this release)

- `gpuwm fetch` -- GFS (NOMADS subsets), HRRR (AWS byte-range), and
  ERA5 (CDS request templates + validation); resumable, manifested,
  refuses changed requests instead of silently reusing files.
- `gpuwm domain` -- point + GPU tier to a sized experiment TOML via
  the real VRAM estimator (16/24/32 GiB tiers, measured 1.75 peak
  envelope; docs/public/HARDWARE.md).
- `gpuwm check` -- input preflight (decode envelopes, coverage, geog
  tiles, table hashes) plus itemized VRAM preflight with `--alloc`
  device verification.
- `gpuwm run` / `resume` -- supervised forecasts, atomic
  `run-progress.json`, failure capsules, restart checkpoints with
  fail-closed identity checks.
- `gpuwm render` -- composite reflectivity, T2, 10 m wind, and
  accumulated precipitation PNGs via the `wrf-rust` package.
- `UP_HELI_MAX` -- WRF's 2-5 km updraft-helicity running-max diagnostic
  (`nwp_diagnostics = 1`; wizard configs enable it), oracle-pinned to
  WRF v4.6.1 `cal_helicity` at max ULP 0, trajectory-inert by test,
  restart-carried, reset each history frame; unlocks the renderer's UH
  product family.
- `gpuwm downscale` -- offline finer-nest re-runs from archived ArWen
  or stock-WRF history (ndown-class), explicit boundary-cadence
  contract, measured cadence-cost table (docs/public/DOWNSCALE.md).
- `gpuwm import-namelist` -- WRF namelist pair to experiment TOML with
  a structured substitution report and a one-sweep missing-key census.
- `gpuwm doctor` -- whole-estate diagnosis (CuPy, bridges, tables,
  data roots) with copy-pasteable remedies.
- `rw-wps` -- the native preprocessor: HRRR/GFS/ERA5/mapped sources to
  `wrfinput_d0N`/`wrfbdy_d01` directly (no WPS, no `real.exe`);
  unchanged stock WRF v4.6.1 has accepted and integrated its outputs,
  serial and MPI, nests through d06, within documented boundaries
  (docs/public/WRF-INTEROP.md).

### Packaging and platforms

- Git-checkout install (Windows and POSIX) with one vendored, locked,
  offline Rust build; `[gpu]` and `[render]` extras; the pip wheel
  documented honestly as needing the same bridge build.
- `gpuwm fetch-tables` -- stages the externalized table assets (the
  two largest Thompson tables: freezeH2O.dat, 243 MiB, over GitHub's
  blob limit and PyPI's per-file cap; qr_acr_qg_V4.dat, 71 MiB,
  excluded from the wheel/sdist for the same cap but still in the
  repository) from the version-pinned release assets, verified against
  the packaged SHA-256 pins before an atomic install; refuses
  mismatched bytes and never overwrites an existing file.  The install
  scripts run it automatically; `--from DIR` covers offline installs;
  `gpuwm doctor` prints it as the exact remedy while a table is
  missing.
- Sealed Linux runtime archive and CPU-only Windows x86-64 archive
  with deterministic, hash-manifested builders.
- Uniform CLI refusal contract: documented refusals exit 2 with a
  one-line message, never a traceback.

### Known limits (stated in full in README and VERIFICATION)

Lambert conformal, Mercator, and polar stereographic projections
(pole-containing domains, footprints wider than 180 degrees of
longitude, and lat-lon/rotated grids refused; non-Lambert and
southern-hemisphere grids at oracle + smoke maturity, not matched-run
verified); one-way static nests; FP32; no data assimilation; ERA5
drives the config-driven GPU loop (GFS/HRRR feed the native
preprocessor front door; HRRR is CONUS-only); one case deeply
validated, component evidence elsewhere.

## Pre-1.0 development

Pre-release development history, internal milestone evidence, and
per-change hashes are recorded in PROVENANCE.md and the focused status
documents rather than duplicated here.
