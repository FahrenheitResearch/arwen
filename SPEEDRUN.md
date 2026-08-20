# The gpuwm speedrun

An internal bench surface. It is a leaderboard because a leaderboard makes
people care about seconds, and every second on this clock is a real user's
second. It is an instrument because the rules below make a dishonest record
impossible rather than merely against the rules.

This page carries numbers on purpose. Release-facing pages do not: those carry
capability statements, and the seconds live here.

---

## The one rule everything else follows from

**THE CLOCK STARTS WHEN THE BYTES ARE STAGED.**

Fetching is not part of a run. Nobody benchmarks their ISP. The measured thing
is **staged inputs to finished products**: preparation, forecast, render.

Off the clock:

- downloading the course's input bytes (the capsule pins them by sha256, so
  *which* bytes were used is still fixed; *when they arrived* is not timed)
- asking the card its name, the driver its version, NVRTC its build
- reading the kernel cache to decide which class the record is in
- `gpuwm go`'s own fetch stage, which with `--data-dir` revalidates the staged
  bytes and downloads nothing; its wall is recorded under
  `clock.off_clock.fetch` and subtracted from the process wall **by name**

On the clock: authority, front-door manifest, preparation, forecast — **kernel
compile included** — and render, plus whatever orchestration wall is left over
after those are attributed.

---

## Running one

```
# 1. stage the bytes.  OFF the clock; do it whenever, however.
gpuwm speedrun --list                       # prints the exact fetch line per course
gpuwm fetch --source gfs --cycle ... --out /somewhere/staged

# 2. run the course.  This is the clock.
gpuwm speedrun regional-12km-6h --staged /somewhere/staged --out speedrun

# 3. the capsule is written beside the run; verify it anywhere
gpuwm speedrun --verify speedrun/regional-12km-6h/run-.../speedrun-capsule.json
```

Other doors:

| command | what it does |
|---|---|
| `gpuwm speedrun --list [--json]` | the courses, their product sets, their digests |
| `gpuwm speedrun --verify CAPSULE` | check the seal and the evidence; print the record line |
| `gpuwm speedrun --compare A B` | rank two records, or refuse by name if they are not the same work |
| `gpuwm speedrun --determinism A B` | the dual-run byte screen (the only thing that may set a determinism claim) |
| `gpuwm speedrun --leaderboard CAPSULE...` | emit the tables below, one per comparability class |

---

## The courses

Courses are **table data** — a row in `gpuwm/speedrun_courses.json` plus two
asset files under `configs/speedrun/`. Adding one is table work; there is no
per-course code path, and the test suite fails if one appears.

### `regional-12km-6h` — Regional 12 km, six hours

One 246 x 198 x 49 Lambert domain at 12 km over the continental interior, six
forecast hours against three-hourly global boundaries, hourly history (7 frames).
This is the shape a first-time user's `gpuwm go` actually runs. GFS input,
three staged files.

### `nested-12km-3km-3h` — Nested 12 km parent, 3 km child, three hours

A two-domain tree: a 122 x 98 x 49 parent at 12 km carrying a 240 x 192 x 49
child at 3 km (ratio 4), three forecast hours. Exercises the domain-tree runner,
nest initialisation and the per-nest render layout. GFS input, two staged files.

Both are sized to finish on a **10 GiB** card and both run the same physics
suite (`morrison-mp10-ysu-mm5-noah-kf-rte-rrtmgp-v1`) and the same eight-product
set (`standard-8/v1`):

```
composite_reflectivity   2m_temperature   2m_dewpoint   mslp_10m_winds
total_qpf   500mb_height_winds   850mb_temperature_height_winds   sbcape
```

The product set is named and fixed rather than "the whole catalog", because the
catalog grows: a record set against `--products all` in one release is not
against the same work as one set in the next.

---

## What makes a record

A record **is** its capsule — `speedrun-capsule.json`, written by the run,
content-sealed. There is no separate submission form and no field a human
types. The capsule carries:

- the course id, the course row's digest, the product set and its digest
- gpuwm version, git commit, `git describe`, whether the worktree was clean
- the experiment config and namelist digests, and the exact argv
- card name, driver version, CUDA runtime, compute capability, total and free VRAM
- CuPy version, NVRTC build and build id (the numerical stack's identity)
- the sha256 of every staged input byte
- wall clock, broken down per stage, plus the kernel-compile seconds named separately
- the sha256 of every product file written, and the digest of the whole product set

## What makes a record invalid

`gpuwm speedrun --verify` prints `VOID` and exits non-zero, naming every reason.
A VOID record is still worth keeping — it is a defect report — but it cannot be
ranked and `--compare` refuses it.

1. **Broken seal.** The capsule's body does not hash to its own
   `seal.body_sha256`. Some field moved since the run wrote it. A record whose
   numbers can be retyped is not a record.
2. **No frames.** `evidence.wrfout_frames` is zero. The fastest possible
   forecast is one that writes nothing.
3. **A missing product.** The course declares eight products; the record
   rendered seven. That is less work under the same name, and the refusal names
   the missing product.
4. **An empty product file.** A zero-byte PNG, or one with no digest. A blank
   picture is not a rendered product.
5. **Validity not PASS.** The forecast's own `report.json` verdict — the health,
   stability and input-identity gates. A NaN-filled sprint to a blank chart is
   excluded structurally, not by etiquette.
6. **Unattributed time.** The named stages must account for at least 90 % of the
   record's own wall (`gpuwm.stage_timing`). A record whose largest item is the
   unattributed remainder cannot answer the question it exists to answer.
7. **An unscreened determinism claim.** See below.

## What may be compared with what

Two records are comparable when **all four** of these agree:

```
course_id   course_sha256   product_set_sha256   compile_mode
```

They are hashed into one `comparability.key`, and the key is inside the sealed
body. `gpuwm speedrun --compare` checks the seals **first**, then the key, so
the two guards close each other:

- an honest capsule of a different course refuses the comparison **by name**,
  printing which of the four fields differ and what each says;
- a capsule edited to make its key match fails the seal and never reaches the
  comparison at all.

That is the difference between discouraging an incomparable comparison and
preventing one. There is no `--force`.

## Cold cache and warm cache are different records

The first GPU run on a machine pays roughly a minute of NVRTC kernel
compilation. Including it silently makes a warm box look fast; excluding it
silently reports a number no first-time user will ever see. So:

- **the compile is always inside the clock.** The capsule says so in
  `compile.included_in_clock`, on every record, always `true`.
- **the seconds are named separately**, read from the forecast's own
  `progress.jsonl` (`phase kernel_compile`, measured as step 1's wall minus the
  median of the steps after it). If the run could not measure it, the capsule
  says *not measured* rather than inventing a number.
- **`compile_mode` is part of the comparability key.** A cold record and a warm
  record are never compared.

The door reads the kernel cache **before the clock starts** and refuses a
mismatch with what the course declared — so which class a record is in cannot be
decided after seeing the time. The cache census counts entries *for this card*:
CuPy keys its cache by target architecture, so a cache full of another card's
binaries is warm on disk and cold in every way that costs time.

`--cold-cache-dir DIR` empties a directory and points `CUPY_CACHE_DIR` at it, so
a cold record can be set on a machine whose own cache is warm. It never touches
the inherited cache.

## Determinism is a screen, not a claim

None of these cards carry ECC. A capsule's `determinism` block starts at
`not_screened` with a `null` claim, and only `gpuwm speedrun --determinism A B`
— given two capsules from two runs of one course on one machine — may set it.
The screen compares the product-set digests byte for byte. A capsule that
claims determinism without the screen is VOID.

There is no `--dual-run` shorthand. Run the course twice and pass both
capsules; the second arm is a real second run, not a replay.

---

## Records

Set with `gpuwm speedrun`, capsules committed under `evidence/speedrun/`. One
table per comparability class; a record cannot appear in a table it is not
comparable to. Regenerate with:

```
gpuwm speedrun --leaderboard evidence/speedrun/*.json
```

Every cell below is read off a sealed capsule. Nothing here was typed.

### `regional-12km-6h` — cold kernel cache

comparability key `ee8b62e82436602eff5bb87e58453a72cfb8e74945dc92a60405137f95b6804b`

| # | wall | prepare | forecast | render | of which kernel compile | card | version | commit | determinism | capsule |
|---|------|---------|----------|--------|-------------------------|------|---------|--------|-------------|---------|
| 1 | 2:06 | 0:09 | 1:47 | 0:09 | 0:35 | RTX 5070 Ti | 2.5.0 | `86b87e6fa` | dual-run identical | `f24e12dfe4d6f9dd` |
| 2 | 2:06 | 0:09 | 1:48 | 0:09 | 0:35 | RTX 5070 Ti | 2.5.0 | `86b87e6fa` | dual-run identical | `4a925b01b5abc566` |
| 3 | 4:48 | 0:13 | 3:24 | 1:11 | 1:16 | RTX 3080 | 2.5.0 | `86b87e6fa` | not screened | `e4e88b504d21b455` |

Rows 1 and 2 are the two arms of the dual-run screen, kept as separate records
rather than collapsed: each is a real run, and the screen is what the pair is
FOR. Both wrote the same product set,
`6e8d53e9d84d5cd24a4e7f26925e980365562b5971ad31c4dcc46cf1c0cfe9e3` — the
no-ECC corruption screen, passed. The screen is
`evidence/speedrun/node1-regional-determinism-screen.json`.

### `nested-12km-3km-3h` — cold kernel cache

comparability key `1261532718836cfdc1c42b68e204a891cd8ad4fa508db8793e333af941247e83`

| # | wall | prepare | forecast | render | of which kernel compile | card | version | commit | determinism | capsule |
|---|------|---------|----------|--------|-------------------------|------|---------|--------|-------------|---------|
| 1 | 1:51 | 0:07 | 1:35 | 0:08 | 0:33 | RTX 5070 Ti | 2.5.0 | `86b87e6fa` | not screened | `fa855251e925fef7` |
| 2 | 5:00 | 0:11 | 3:25 | 1:21 | 1:22 | RTX 3080 | 2.5.0 | `86b87e6fa` | not screened | `6b18c37d2c96b998` |

### What the two tables may not say to each other

Nothing. `regional-12km-6h` and `nested-12km-3km-3h` have different
comparability keys and `gpuwm speedrun --compare` refuses the pair by name,
printing the two `course_id`s and the two `course_sha256`s. The 3080 rows are
also not evidence about the 5070 Ti rows in any direction the leaderboard
states: they are two cards under two operating systems, one of which was a
loaded desktop, and both are simply records of the same course.

### Conditions on both boxes

- **Cold cache, forced.** Both boxes ran with `--cold-cache-dir`, so every
  record paid the NVRTC compile for real: 33–35 s on the 5070 Ti, 76–82 s on
  the 3080. It is inside every wall above.
- **CUDA 13 on both.** CuPy 14.0.1/runtime 13000 (3080) and 14.1.1/13020
  (5070 Ti) are outside the certified CUDA 12.x family, so preprocessing ran
  on the deterministic parallel CPU backend on both. The capsules record it
  (`device.cuda_runtime_in_certified_range: false`); the records are
  comparable to each other and would not be comparable to a 12.x record's
  numbers without saying so.
- **The 3080 is a loaded desktop.** ~4.7 GiB of its 10 GiB was in use by
  browsers and the compositor when the courses were sized, which is why both
  courses fit a measured 4.58 GiB budget with headroom rather than filling the
  card.

---

## Submitting

1. Stage the course's bytes (the fetch line is in `gpuwm speedrun --list`).
2. `gpuwm speedrun <course> --staged DIR`.
3. `gpuwm speedrun --verify <the capsule it wrote>` — it must print `VALID`.
4. Add the capsule under `evidence/speedrun/` and regenerate the table with
   `gpuwm speedrun --leaderboard evidence/speedrun/*.json`.

The capsule is the submission. Nothing else is read.
