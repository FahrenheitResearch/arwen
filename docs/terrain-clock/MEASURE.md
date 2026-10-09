# Terrain clock, step 1: measurement (2026-10-07)

This is step 1 of the plan in section 6 of REVIEW.md: measure first. No engine code or default changed. The candidate rows are in
`terrain_clock_map_candidate-2026-10-07.json`, which the engine does not read.

## What was run

- **Tool.** `tools/terrain_clock_probe.py sweep` (new on this lane). It uses the same bell ridge, `step()`, 49-level ladder, etac
  survey, lid and held bound as the shipped map's probe.
  - Each wind from 20 to 60 m/s walks the rungs 6.5, 6.0, 5.5, 5.0, 4.5, 4.0, 3.5 and 3.0 s/km. It starts from the rung the
    weaker wind held.
  - Every run lasts three hours on a domain the flow cannot wrap, unless it stops first.
  - Each run records held or stopped, peak w, the WRF-form vertical Courant number (the quantity w_damping acts on above 1),
    the cells w_damping acted on, and the geometric |w| dt/dz over the lowest four levels.
- **Rows.** 236 rows: 59 ridges × 4 and 6 substeps × 2 settings.
  - Spacings 2, 3 and 12 km, at crests 3000 and 4500 m.
  - Ridge slopes from 0.09 to 0.40, including new rows with grid slopes 0.089 to 0.10.
  - The new 12 km rows include grid slope 0.0915, the exact reading on NCAR's 12 km grid.
  - At 12 km a bell ridge cannot read steeper than 0.187 (4.5 km crest) or 0.125 (3 km crest), so that is the highest 12 km slope.
- **Settings.**
  - `generated`: epssm 0.5, sixth-order filter 0.12, the map's own.
  - `ncar`: epssm 0.1, no sixth-order filter, damp_opt 3, w_damping 1, the NCAR v4.4 namelist as WOOF imports it.
- **Floor.** The lowest rung is 3.0 s/km, which is the default clock's halved step on both NCAR grids. "none" means no rung down to
  3.0 s/km held.
- **Where it ran.** Box S card 2 (RTX 5090, backfill behind the DA lane) ran all 236 rows from 02:02 to 02:28Z. Box G card 0
  (RTX PRO 6000) ran 180 of them, from the other end of the list, from 02:32Z until the sweep was complete.

## Checks on the measuring tool

- **S and G agree exactly.** On the 180 rows run on both boxes, all entries are identical. In the 1403 runs both boxes made,
  every held or stopped result matches and peak w matches to the bit (`compare-boxes.txt`).
- **The new tool matches the shipped probe.** The shipped probe (5bd9a0085, unmodified) and the lane probe give the same peak w
  to the bit (34.655418 and 35.112014 m/s), with and without Courant recording (`validate/`).
- **The generated arm reproduces the shipped map.** At 12 km it reproduces the map's 4.5 km crest rows at 0.1402, 0.1719,
  0.1856 and 0.1869 at every wind from 30 to 60 m/s, and the 0.0893 row from 40 to 60 m/s (`tables.md`, last table).

## Findings

### 1. The map's held bound is set off by finite mountain waves on gentle ridges at 2 and 3 km

The held bound is 4 × wind × slope + 20 m/s. In three-hour runs on slopes 0.09 to 0.25 at 40 to 60 m/s it stops runs at every
rung down to 3.0 s/km. At the stop, w is only 1.00 to 1.09 times the bound (536 runs), and the WRF vertical Courant number falls as the step
shortens.

Every such cell at 2 and 3 km was re-run at 6.0 and 3.0 s/km with `fixed --criterion blowup`, which stops only at non-finite w
or w above 10 times the bound (`recheck-summary.txt`):
- **2 km, all 28 cells:** finite for three hours at both steps. Peak w is 0.9 to 1.67 times the bound. 26 are steady. Two NCAR
  30 m/s cells were still growing in hour 3, at 1.07 to 1.09 times the bound.
- **3 km, 26 cells:**
  - The gentle cells (slope 0.09 to 0.15) are finite.
  - 11 cells on slope 0.30 to 0.40 at 40 to 60 m/s ran away within 54 to 162 s at 6.0 s/km. All 11 ran finite at 3.0 s/km.

The shipped map never saw this at 2 km because its entries are half-hour runs. The shipped probe reproduces it: 3 km, 3000 m
crest, slope 0.1, 40 m/s held 8.0 s/km for three hours (34.7 against a bound of 35.9) and stopped at 6.0 s/km after 4842 s (36.2).
So this bound is not monotone in the step there.

**Consequence.** `read_map` reads every gentler row as well, so these nulls cap every steeper reading at 40 m/s and above. That
is why the 2.5 km readings below return none.

### 2. At 12 km the stops are real

- Every 12 km cell whose 6.0 s/km run stopped near the bound ran away when re-run at 6.0 s/km for three hours. That is 36 of 36
  generated cells and 30 of 30 NCAR cells, within 144 to 4248 s.
- Every one of them ran finite at 3.0 s/km.

### 3. NCAR's settings are not less stable than the generated ones on these ridges

This goes against what REVIEW.md expected. NCAR's settings hold the same or longer steps at 12 km, for example:

| 12 km, 4.5 km crest, slope 0.0915, 4 substeps | 20 m/s | 30 | 40 | 50 | 60 |
|---|---|---|---|---|---|
| generated | 6.5 | 5.5 | 5.0 | 4.5 | 4.0 |
| ncar | 6.5 | 6.0 | 5.5 | 5.5 | 5.0 |

At 2 and 3 km the two arms differ only cell by cell.

### 4. The probe is far harsher than the real NCAR runs

- In runs the probe counts as held, the peak WRF vertical Courant number is 0.34 to 3.13, and above 1 in most of them, so w_damping is acting.
- The real NCAR runs peaked at 0.52 (12 km) and 0.57 (2.5 km), and w_damping never acted.
- The geometric near-ground |w| dt/dz in held probe runs is 4 to 70, because the probe's lowest layers are thin. It is not
  comparable with the review's 1.15 to 1.69 from wrfout snapshots.

## What the candidate rows say about NCAR's steps

These are the engine's own `read_map`, run over the candidate rows alone (`readings.txt`).

**12 km**, slope 0.0915 row, 4.5 km crest. The face-local crest of 3398 m still reads the 4500 m row.

| Wind read | generated, 4 substeps | ncar, 4 substeps |
|---|---|---|
| 48.4 m/s domain-wide (50 column) | refuses: 4.5 s/km | refuses: 5.5 s/km |
| 11 m/s face, start (20 column) | admits 72 s | admits 72 s |
| 25 m/s face, f012 (30 column) | refuses: 5.5 | admits 72 s, with no margin (6.0 held, 6.5 ran away) |
| 57.5 m/s, 2.3x face margin (60 column) | refuses: 4.0 | refuses: 5.0 |

**2.5 km**, read from the 2 and 3 km rows:
- **Domain-wide** (52.4 m/s, 4.5 km crest): none, at both settings.
- **Face-local, 15 m/s** (crest 2590 m reads the 3000 m rows): admits 15 s.
- **Face-local, 34.5 m/s** (the 2.3x margin): none. That none comes only from the finite-wave stops on the gentle rows (finding 1).

So with the 2.3x wind margin the review asked for, the local reading alone admits neither of NCAR's steps on these rows.

## For step 2 (decisions this measurement raises)

1. **The held bound needs a decision before any row is merged.** As measured, it refuses finite steady waves at 2 and 3 km.
   - The blow-up criterion separates the two cases on every cell re-checked.
   - Changing the criterion that defines "held" is a change to the map's measurement, so it should be decided explicitly, not
     slipped in.
2. **Merging these rows into the shipped map changes 3 km entries.** 3 km rows were tried there to 13.33 s/km over half an hour.
   The candidate's three-hour entries are capped at 6.5 s/km, and on the gentle rows they are none.
3. **NCAR's 72 s at 12 km stands or falls on the wind margin** (row 0.0915, 30 column: held at 6.0, ran away at 6.5). The margin
   has to be set from measurement, as REVIEW.md section 6 says.

## Files

All of these are relative to the retained `CLOCK-CHECK-NCAR-2026-10-06/fix/` receipt folder:
- `rows/S`, `rows/G`: every row with every run.
- `recheck/`, `recheck-summary.txt`
- `validate/`
- `tables.md`, `readings.txt`, `compare-boxes.txt`
- `logs/S`, `logs/G`
- `scripts/`, including the S venv note.
- `1007-CLOCK-probe-entries.png`: an analysis chart, not a weather field.
