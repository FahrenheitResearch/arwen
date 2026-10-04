# Unchanged source recipe execution

`OrdinaryRecipeExecution` binds input-ensemble, time-lagged, multi-model and
control recipes to their already prepared ordinary source heads. It uses no
physical capture, field recentering or second native source preparation. Original
scalar and nested runners retain their clocks, physics, statics, boundary streams
and complete checkpoint authorities.

Inputs are keyed by canonical `SourceTrajectory.identity`. Each input needs its
concrete `PostedSourcePreparation` specification from the actual acquisition
handoff. The specification's `physical_root` is unused and may remain absent.
The constructor verifies source/member grammar, cycle, original configuration
files, immutable prepared head and its actual input plan. It reruns the original
head preflight and compares original reader/configuration/domain authority before
publishing an immutable `ordinary-roster.json` descriptor. No future payload or
source seal is required for admission.

Each configured file must match a named role digest captured by the original
head or loader. The descriptor records the matching role names for configuration,
WPS, static, geometry and decoder files. A file hash asserted only by the
acquisition specification is insufficient. Other authored native options must
match the original typed clock, history interval, vertical pressure top, physics
profile or recorded preprocessing backend. Worker counts and export bookkeeping
retain their original execution semantics. An initialization override without a
captured role or value is rejected by its option name before a descriptor is
published.

Construct the owner with the full `SourceRecipe`, trajectory-keyed acquisition
specifications and already head-preflighted scalar or tree inputs, plus a separate
descriptor `root`. Pass it as `source_execution` to `PreparedEnsembleSession`.
The existing runner initializes fresh member state and physics from each selected
original input. `member_indices` selects replay members while retaining the full
recipe identity. An existing descriptor cannot be replaced with another roster.

The Session interface is the same as the changed-field posted owner:

| Method | Behavior |
| --- | --- |
| `planning_inputs(member_id)` | Rechecks the original authorities and returns the same prepared input object |
| `run_member(member_id, forecast=...)` | Runs the original forecast callback once, then requires the original source seal |
| `stochastic_member_binding(member_id)` | Supplies original global ID, seed, recipe, trajectory and head identities |
| `member_order`, `member_metadata` | Retain stable original indices and unsigned 64-bit seeds |
| `receipt()`, `require_complete()` | Record completed or failed members and every bound original source seal |

Sparse selection retains the complete original recipe descriptor and seeds.
Only selected unchanged trajectories need prepared heads. Changed recentered
fields continue through `PostedRecipeExecution` and the physical provider,
including its complete donor population. Different source inputs may select
different original physics suites. Same-trajectory members can select different
already prepared configurations with the additive `member_sources` mapping:

```python
from gpuwm.ensemble.ordinary_execution import OrdinaryMemberSource

member_sources = {3: OrdinaryMemberSource(specification=variant_spec,
                                        inputs=variant_head_inputs)}
```

Keys are original recipe member IDs. Each record requires its own concrete
acquisition specification and original head-preflighted input. Its trajectory
must match the member, and its forecast window must match the complete recipe.
The descriptor records the selected member's own head/configuration authorities;
the stochastic binding retains the original trajectory, global ID and seed.
Unselected replay variants need no head or preflight. An absent or empty mapping
preserves the default descriptor and stochastic binding bytes. This owner never
changes a cache-bound configuration to introduce another suite.

CPU orchestration tests cover distinct source heads, nested inputs, sparse IDs,
different physics, source and file-role mutations, typed backend controls,
source-seal failures and missing future files. They also exercise Session member
dispatch, explicit same-source physics variants and both original reader entry
points. Those tests control the native
reader and forecast seams; they do not qualify numerical source or forecast
identity. An actual source/nested forecast gate remains required. The inactive
Session and ordinary N=1 writer are unchanged.
