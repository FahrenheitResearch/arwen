# A fine nest over the radar-DA free forecast

EXPERIMENTAL. Nothing here is on a default route.

The nowcast assimilates and forecasts on a single coarse domain.
Interpolating its final output to a finer mesh produces an unbalanced
field that spends its first 15–30 minutes adjusting — exactly the window
a nowcast exists to serve. This runs a real one-way nest over the
free-forecast legs instead, so the fine field is produced by the model
and takes lateral forcing from the parent at every parent step.

## What you set

Everything about the child except these keys is derived. `dx`, `dy` and
`dt` come off the parent through the ratio chain and are never typed.

```
--nest-half-width-km 60      # or --nest-nx / --nest-ny in child cells
--nest-ratio 3               # space AND time; 3 off 3 km is 1 km at 5 s
--nest-members 0             # default: the control trajectory only
--free-legs 6                # required; the nest covers the free legs
```

Optional: `--nest-i-parent-start` / `--nest-j-parent-start` (default is
centred in the parent), `--nest-history-interval-s` (default is the
parent's), `--nest-acknowledge` (deliberate acceptance of an
admissibility refusal).

Price it before running it:

```
python -m tools.da_nest_cost --parent-leg-seconds 34.5 --trajectories 11
```

## What the code permitted, and what it did not

gpuwm has three ways to give a run a child domain. None of them is
"introduce a nest mid-run", and the one that looks closest is the wrong
one:

- **Whole-run live nest.** `build_experiment` initialises every child
  eagerly at t=0. Proven, but it would carry the nest through the DA
  cycling as well.
- **Delayed per-domain `start_time`.** Implemented, and it is the trap.
  `on_domain_start` activates a child by calling `initialize_child`,
  which resolves its source snapshot at the child's own start time
  (`nest_init._initial_snapshot`). That **cold-starts the nest from the
  raw analysis** valid at the free-forecast start. Every radar increment
  the cycling bought would live only in the parent and reach the nest
  through its lateral boundaries alone — strictly worse than the
  post-hoc interpolation this work replaces, because it also throws away
  the assimilation. There is no flag that makes the delayed start
  inherit the parent's live state.
- **`gpuwm downscale` / offline child.** Inherits the parent's evolved
  state, but forces the child from archived *history* frames rather than
  every parent step, which is the specific weakness being avoided.

The route taken is the one the DA driver's own shape makes available.
`tools/da_cycle_prepared.py` rebuilds the entire model from scratch at
every leg and joins legs through host snapshots, so **a leg is already a
fresh model whose clock is placed at the leg boundary**. A nested leg is
therefore not a mid-run introduction at all: it is an ordinary whole-run
two-domain model whose run happens to be one leg long, and whose parent
state at t=0 of that leg is the *analysed* state. The child is built from
that parent by full SINT (`nest_init.parent_only_init`), so it inherits
every increment the cycling produced, and is forced laterally by the
parent every parent step for the rest of the leg.

## What that costs, stated plainly

`parent_only_init` is the idealized `input_from_file=F` branch: it
returns `static_fields=None` and `soil=None`, and
`runtime.prepare_child_case` refuses a full-physics child without them.
The nowcast profile runs Noah, MM5 surface layer and YSU, so a land state
is mandatory. It is supplied by replicating the parent's land column onto
the child with WRF's own nest-down nearest-donor mapping
(`interp_fcni`/`interp_fcnm`), reusing the donor maps
`NestRegistration` already builds for SINT.

The consequence is a terrain and land policy identical to the one the
offline-child contract already registers as `sint-parent-inherited`:

> **The nest refines the atmosphere, not the surface.**

It buys finer dynamics, finer microphysics and a shorter acoustic step
over the analysed storm. It does not buy fine orography, because on this
route no fine static data exists to buy it with. Fine terrain needs a
per-domain prepared cache and the real-data child initialiser — a much
larger piece of work, and one that reopens the balance problem, because
real fine terrain under a SINT-inherited atmosphere is precisely the
mismatch WRF's `blend_terrain` / `adjust_tempqv` / `press_adj` sequence
exists to repair.

Keeping terrain, base state and land all inherited from the same parent
under the same operator is what makes the child's initial state
**balanced**, which is the whole point of the exercise.

Vertical resolution is shared tree-wide: a 1 km nest inherits the
parent's eta ladder. Often that, not `dx`, is what limits the result.

## Admissibility is enforced, not conventional

`gpuwm.da.nested_forecast.validate_nest_admissibility` refuses, rather
than advises:

| condition | outcome |
| --- | --- |
| `cu_physics != 0` below the convection-permitting spacing | refusal |
| column PBL below the gray-zone floor | refusal unless acknowledged |
| `spec_exp != 0` on a nested domain | refusal |
| `nested=False` or `specified=True` on the child | refusal |
| child `nz` differs from the parent's | refusal |
| child `mp_physics` differs from the parent's | refusal |

The two spacing thresholds are **imported** from
`gpuwm.domain_wizard` rather than restated, so the two surfaces cannot
drift. They exist there only as advisories on the config-authoring path;
an advisory printed during authoring cannot police a domain this module
derives at runtime. Exactly 1.000 km does not trip the gray-zone floor —
it is a strict inequality, matching the wizard.

## One-way, and proved

`feedback` is pinned to 0 and is *not* inherited from the parent
experiment. The parent-inertness claim is a ratchet in the same spirit as
the N3/N4/N5 `ancestor_inertness` nest gates: the parent's whole
restart-serialised state is compared bit for bit with and without the
nest attached, and the comparison is paired with a liveness assertion so
it cannot pass by the nest quietly doing nothing.

Receipt: `evidence/da-nested-forecast/parent-inertness-receipt.json`
(dual-run, both runs identical; this box has no ECC).

## Cost

Nest work scales as (child columns / parent columns) × ratio, times the
number of nested trajectories. Against the shipped 132×132×49 3 km
nowcast with a measured 34.5 s parent leg over 11 trajectories:

| half-width | child | Δ machine peak | control-only | all 11 trajectories |
| --- | --- | --- | --- | --- |
| 30 km | 60×60 | +59 MiB | +1.9 s/leg | +21.4 s/leg |
| 45 km | 90×90 | +108 MiB | +4.4 s/leg | +48.1 s/leg |
| 60 km | 120×120 | +176 MiB | +7.8 s/leg | +85.5 s/leg |
| 90 km | 180×180 | +735 MiB | +17.5 s/leg | +192.5 s/leg |

That is why `--nest-members` defaults to 0. A 60 km half-width 1 km nest
on the control alone adds about 23 % to a leg and under 200 MiB; the same
nest on the whole ensemble costs nine minutes over six free legs and the
nowcast starts to stop being a nowcast. The parent carries the ensemble;
the nest carries the detail.

VRAM here is **computed** by `gpuwm.core.preflight`, not measured. The DA
driver hand-builds its `ExperimentState` and runs no VRAM gate of its
own; `tools/da_nest_cost.py` is the surface that fills that gap.

## Rendering

`tools/da_cycle_prepared.py` writes the nest's leg-end composite beside
the parent's, as `composites/legNN_<name>_dNN.npz`, carrying the
placement a renderer needs. The adoption line in
`tools/da_nowcast_render.py`:

```python
nest = nest_panel(cycle_dir, leg, name, gallery.lat, gallery.lon)
if nest is not None:
    ...pcolormesh(nest["lon"], nest["lat"], nest["refl_colmax"])...
```

`nest_panel` returns `None` for the legs the nest did not cover and for
any case rendered from a run that had no nest. Child geolocation is
derived from the parent's own lat/lon by inverting WRF's nest-down
pickup, so a panel cannot drift from the grid the model ran on. Draw the
nested panel **beside** the parent's, never instead of it: a fine view of
a forecast is only readable next to the forecast it refines.

## Not yet done

- No full nested nowcast has been run end to end. The driver path is
  wired and its pieces are individually proved on the GPU; a live-fire
  nested cycle is the next evidence to collect.
- Fine orography for the nest (see above) is a separate work item.
- The nest is not assimilated into. Observations constrain the parent;
  the nest inherits the analysis and forecasts from it.
