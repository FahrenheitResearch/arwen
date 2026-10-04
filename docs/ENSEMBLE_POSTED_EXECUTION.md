# Posted ensemble execution

`gpuwm.ensemble.posted_execution.PostedRecipeExecution` connects a committed
`PostedPreparationFactory` to `PreparedEnsembleSession(source_execution=...)`.
It keeps the complete provider recipe and donor population. Optional
`member_indices` selects original replay IDs without recomputing the mean.

| Interface | Contract |
| --- | --- |
| `source_inputs` | Ordinary head-preflighted inputs keyed by canonical source trajectory identity. Every ordinary source head and provider descriptor is immutable and verified. |
| `planning_inputs(member_id)` | The original source configuration and prepared head for admission. This does not initialize members or wait for future raw fields. |
| `run_member(member_id, forecast=callback, observer=...)` | Calls the original forecast with actual member inputs. Unchanged trajectories reuse their ordinary source tree. Recentered members use the original native producer and start-first head callback, then consume later streamed boundary segments. |
| `stochastic_member_binding(member_id)` | Immutable original ID, seed, recipe, trajectory, provider descriptor and source head authority. The actual native member head is included once published. |
| `receipt()` | Original member order, source heads, current preparation/forecast/seal status and pending member IDs. |
| `require_complete()` | Requires every selected original forecast and its ordinary/provider source verification to finish. |

The Session uses ordinary member execution for this input owner. Its existing
card packing, execution-only tile admission, physics, adaptive clock, products
and optional member-file policy remain authoritative. The configured stochastic
receipt is updated from the actual member head before initialization; public
doors refuse active stochastic controls with the calibration reason, so that
receipt records the all-off configuration. Complete
sealed `PreparedMemberInput` and config-driven `RuntimeMemberInputs` remain
separate contracts.

The run receipt stores this lifecycle under `posted_source_execution` in
`ensemble-run.json`. A published head is sufficient to begin a forecast. A
complete result still requires every consumed endpoint and final source seal.
This contract does not certify numerical forecast identity or calibration;
those require receipts from the actual source and forecast artifacts.
