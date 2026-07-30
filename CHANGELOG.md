# Changelog

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
- Plot subtitles gain the grid spacing the file declares -- `Δx 3 km`
  at and above a kilometre, `Δx 111 m` below it -- on both engines and
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
