# 1.9 assembly notes

The things a release line has to carry that are not changelog entries: what
is inherited red, what a re-run must regenerate and in what order, and the
record of anything the forward-only rule stops us from tidying.

`CHANGELOG.md` says what changed for a user. This file says what the next
person assembling on top of this line needs to know.

## Inherited reds on this line

MEASURED at `7f0304bb9`, the last commit before the seven 1.9 lane merges,
serially on a quiet card. A red in this list is inherited: it is not
charged to any 1.9 lane and clearing it is separate work. A red NOT in this
list, on a suite this line touches, is this line's.

| Suite / node | Note |
| --- | --- |
| `tests/test_pd_advection.py::test_nested_pd_lbc_is_folded_once_before_advection` | The only survivor of the pre-1.9 list (task #157, measured at `4152fcb31`). Tracked as a 1.8.10 rider, task #161. |
| `tests/test_ruc_admission.py` | Measured red at `7f0304bb9`. |
| `tests/test_prepare_hrrr_wrf.py` public-wrapper node | Measured red at `7f0304bb9`. |
| `tests/test_forecast_offset_init.py` | Measured red at `7f0304bb9`. |
| `tests/test_hrrr_two_domain_forecast.py` | Measured red at `7f0304bb9`. |
| `tests/test_stream.py` (three nodes) | Measured red at `7f0304bb9`. |
| `tests/test_interrupt_contract.py` | Measured red at `7f0304bb9` by the gate2 cut round. |
| `tests/test_mynn_surface_driver_gpu.py` | Measured red at `7f0304bb9` by the gate2 cut round. |
| `tests/test_mynn_surface_gpu.py` | Measured red at `7f0304bb9` by the gate2 cut round. |
| `tests/test_mynn_surface_water_gpu.py` | Measured red at `7f0304bb9` by the gate2 cut round. |
| `tests/test_nssl2_nucond_diagnostics.py` | Measured red at `7f0304bb9` by the gate2 cut round. |
| `tests/test_nssl2_qvexcess.py` | Measured red at `7f0304bb9` by the gate2 cut round. |
| `tests/test_preflight.py::test_alloc_preflight_n0_four_domain` | Measured red at `7f0304bb9` by the gate2 cut round. |
| `tests/test_preflight.py::test_the_recorded_local_frames_match_the_driver` | Measured red at `7f0304bb9` by the gate2 cut round. |
| `tests/test_sase_gpu.py` | Measured red at `7f0304bb9` by the gate2 cut round. |
| `tests/test_shinhong_runtime.py` | Measured red at `7f0304bb9` by the gate2 cut round. |
| `tests/test_shinhong_wrf461_parity.py` | Measured red at `7f0304bb9` by the gate2 cut round. |
| `tests/test_source_adapters.py` | Measured red at `7f0304bb9` by the gate2 cut round. |
| `tests/test_thompson_aerosol_adapter.py` | Measured red at `7f0304bb9` by the gate2 cut round. Supersedes the earlier local reading below that had it green: the two measurements were taken in different venues, and the base-red charge sheet uses this one. |
| `tests/test_thompson_aerosol_state_gpu.py` | Measured red at `7f0304bb9` by the gate2 cut round. |

The fourteen rows naming the gate2 cut round were added when that round
measured the base itself and found them red there; they were missing
from the first version of this table, so the charge sheet for the
merged tip was incomplete until they landed.

Removed from the pre-1.9 list, both GREEN on this line and therefore no
longer inherited: `tests/test_ftz_route_inventory.py` and
`tests/test_render_rust.py`. Of the remaining two members of task
#157's five, `test_chain_stage_env.py` is not red here;
`test_thompson_aerosol_adapter.py` was recorded not-red in the first
local measurement but is in the gate2 base-red list above, which is the
reading the charge sheet uses.

A stale inherited-red list is worse than none: it launders a real
regression as a known one. This table is re-measured at each assembly
against the line's own base, never carried forward unchecked, because a
blocker measured on a lane's own base is not a blocker on the release line
(the three false 1.4.1 blockers).

## Regeneration order

Derived artefacts read each other, so the order is not a matter of taste.
Regenerating out of order is how `gpuwm/verify/ftz_claim_sites.json` came
into the 1.9 gate stale with 18 pure line-drift misses: `aabffda28`
regenerated the census and `1b4bcbdb1` edited `CHANGELOG.md` afterwards, so
every claim below the edit moved by the number of lines the edit added.

1. `python -m tools.build_registry` — `gpuwm/physics_registry_v2.json`.
2. `python -m tools.report_registry_ground_truth` — reads the registry.
3. `python -m tools.report_template_evidence_consistency` — reads the registry.
4. `python -m tools.report_physics_composition_walk` — reads the loader,
   which reads the registry.
5. Every remaining prose and code edit of the assembly, including
   `CHANGELOG.md`, `docs/public/PHYSICS.md` and this file.
6. LAST, after step 5 is final: `python -m tools.ftz_receipt.claim_census`.
   The census records file:line for every FTZ claim site in the tree, so
   ANY text edit anywhere above it invalidates it. Verify with
   `python -m tools.ftz_receipt.claim_census --check` (rc 0).

Steps 1 to 4 are idempotent; re-running them after step 5 is safe and is
the cheap way to prove nothing in step 5 moved them.

## Commits without the `Co-Authored-By` trailer

Nine commits on the P3 lane, merged at `82ce29df7`, carry no
`Co-Authored-By: Claude Fable 5` trailer. Forward-only commits are
absolute on this project, so they are RECORDED here and not rebased:

- `e1e5e67ec` fix(p3): mp=50 could not take a dycore step, and its rime pair never moved
- `252e9c6bf` fix(p3): an mp=50 run could not write a restart at all, and would have lost its rime
- `90ad651aa` test(p3): the dycore tier that would have caught all three blockers
- `f2d835270` docs(p3): publish the transport/restart/output decisions, and repin what moved
- `aefbe5681` chore(p3): pin the absent-mass slot name across its two spellings
- `f190291de` fix(p3): select the absent-mass plane by presence, so the next port cannot regress
- `e20ee9b37` test(p3): pin the mp=50 history inventory, including what it must NOT publish
- `deae4d6e3` docs(p3): the wind A/B never proved transport, so stop saying it did
- `18683af74` test(p3): pin calc_cq's one-ice-category arm, which nothing covered

Regenerate the list with:

```
git log --format='%h|%s|%(trailers:key=Co-Authored-By,valueonly)' \
    7f0304bb9..HEAD | awk -F'|' '$3=="" {print $1" "$2}'
```

## Deferred hygiene

`gpuwm/core/rrtm_taumol.py` is stored with CRLF line endings. That matches
the base's existing pattern rather than diverging from it, and the
`* -text` guard in `.gitattributes` holds, so the source-hash identity is
stable and nothing is broken. Normalising it is a hygiene pass of its own,
worth doing with the other CRLF files in one commit rather than as a rider
here, where it would put an unrelated whole-file diff in a release cut.

## Proof of life for the two resurrected schemes

Pricing is not running.  The 1.9 gate reopened `mp_physics = 50` and the
MYJ pair by giving the preflight the kernel-module rows they lacked, and a
row that makes `gpuwm check` return 0 proves only that the check returns 0.
Both schemes were therefore STEPPED on a real card and their output read
back off the file a renderer opens.

RTX 5080 16 GiB, CUDA 13.1, node 4, under its mutex.  Tree content
identical to this commit's parent: the node checkout's tree object is
`d47b9139b126d13587ff925ac80d65832e408db4`, the same tree
`2fbea9dcd` carries.  CuPy 14.1.1 (cupy-cuda13x) passed the frontdoor
smoke, import plus one cuBLAS matmul, before anything was trusted.

`gpuwm check`, on the node, exit 0 for all three standing configs:
`configs/p3_mp50_shared.toml` and `configs/p3_mp50_domain.toml` (26 kernel
modules selected, 3.87 GiB forecast envelope) and `configs/myj_pbl.toml`
(27 modules, 2.65 GiB).

P3, `mp_physics = 50`, WK82 quarter-circle supercell, 100x100x40 at
dx = 1 km, dt = 6 s, 300 steps, 4 frames.  Read back from
`wrfout_p3_mp50_wk82.nc`: every field finite, W spanning
-10.17 to +43.31 m/s (a supercell updraft), QRAIN to 1.334e-2 kg/kg and
RAINNC to 16.25 mm.  P3's OWN prognostics, all cold-started at exact zero
and all nonzero at the end, which is P3 having run and nothing else:
QICE 1.544e-2 kg/kg, QIR (rime mass) 1.400e-2, QIB (rime volume)
1.911e-5, QNICE 7.41e6 per kg.

MEASURED COST, worth knowing before anyone plans a P3 campaign: that
integration took 30 min 36 s of wall time at 100% of ONE core with the
GPU essentially idle.  P3 is a host float32 transcription
(gpuwm/core/p3.py), so its per-step cost is CPU-bound and does not
benefit from the card.  The MYJ arm, which is CUDA, ran the same 300
steps in 3.4 s.

MYJ, `bl_pbl_physics = 2` with `sf_sfclay_physics = 2` over Noah, moist
bubble at 48x48x40, dx = 2 km, dt = 6 s, 300 steps, 4 frames, no NaN.
MYJ's own carried state moved off its cold start: TKE_MYJ 0.20 (the
scheme's epsq2) to 1.704, EL_MYJ 0 to 172.65 m, EXCH_H to 365.43, PBLH to
1106.44 m, UST to 0.809.

TWO REFUSALS FIRED ON THE WAY, both correctly, and both are recorded
because a proof that had to be argued past a guard is worth reading:
`initialize_physics` refused a shortwave-only suite under Noah because
nothing computes GLW (the 1.8.9 fix), and RTE+RRTMGP refused scalar
radiation geometry because it places the sun per column.  The arm takes
the first remedy the GLW message lists, RTE+RRTMGP on both streams, so
the surface MYJ reads is driven by computed radiation.

Gallery: the `1.9-scheme-proof` folder in the operator's local
deliverables directory, case folder -> `d01` -> product, 16 PNGs from
the Rust renderer (rw_wrfbatch built from this tree), 2 m temperature
and total QPF at all four frames of each arm.  No ECAPE products.  The
wrfouts are beside them under `wrfout/`.

Two gaps the render surfaced, neither a release blocker and both filed
here rather than fixed in a gate commit: `WrfoutWriter` called directly
writes no START_DATE/SIMULATION_START_DATE (the production runtime
supplies them through `global_attrs`), and an idealized case carries no
XLAT/XLONG, which a map renderer requires.  The proof files were stamped
with a run origin matching their own Times and with an equirectangular
map at the 38 N / 97 W reference the MYJ arm already gave the radiation
adapter.  Metadata only; no field byte was touched.
