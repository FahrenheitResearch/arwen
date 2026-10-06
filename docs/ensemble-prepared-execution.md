# Prepared ensemble execution interfaces

`gpuwm.ensemble.prepared_execution.make_automatic_prepared_executor` returns `run(inputs, runner_options=...)`. The factory takes the original prepared runner, request, collector and output directory. It optionally takes a `PreparedMemberRoster`, input provider, card bootstrap factory and initialization callback factory.

The first invocation goes through the original initializer and its `ensemble_bootstrap` handoff. A qualified native handoff plans the whole requested roster before allocating a native member. An unqualified handoff returns control to the original run and produces `AutomaticPreparedResult(mode="ordinary_first")`. Its original first-member receipt is cached by the session. That member is not prepared or forecast twice. A native result has `mode="native_complete"` and covers the full roster.

Native execution owns the existing common-input fixed-clock single-root suite. Release qualification is recorded against the actual source and artifacts. Distinct member atmosphere, soil or boundary trajectories use their original prepared members until their native physics initialization is qualified. `PreparedMemberRoster.select` retains global member IDs, seeds, complete boundaries, source cycles and donor-population receipts. Local product slots are separate from source identities. A source sequence is never replaced with copies of its first member.

`plan_initialized_member_execution` uses the initialized state's allocation declarations, boundary plans, physics banks, radiation call envelope, original device context/local-memory profile, allocator margin and actual output/health plans. It samples free memory after bootstrap. The unchanged original bootstrap remains live through all waves and its measured bytes are recorded. A full-roster bounded product replay is priced even when a resident wave is smaller. Exact precipitation endpoints remain host float32 snapshots; their device working surfaces and uploads are included in the diagnostic plan.

`execute_initialized_member_execution` runs those packs through the actual native forecast. Each card requires its own original initialized source. Each wave gets fresh ordinary `DomainClock` construction and separate state/driver container shells. Native word copies read the retained source. No arbitrary GPU driver is deep-copied, no member clock is minimized, and no physics history is reset by hand. Wave allocations use a private CUDA pool; completed dead owners and only that pool's cached blocks are retired at the wave boundary. Live source tables and collector buffers remain owned and priced.

`ordinary_member_execution_inputs` supplies the real ordinary tile admission path when a resident member does not fit. It changes only execution controls: an off road gains auto selection; existing auto, on and pinned tile choices remain selected. Its card budget excludes named product, health, stochastic and allocator reservations. The helper checks that restart identity is unchanged and preserves the original physics, clocks, source head, initial arrays and complete boundaries.

`initialize_callback_factory(member_id, seed, prepared_member)` returns the original model initialization callback used before restart validation and health checks. It can install the member's admitted stochastic owner for internal qualification; public doors refuse active stochastic controls with the calibration reason. SPP switches are selected before driver construction. Active stochastic physics currently uses the original member path, while its packed coupling remains unqualified.

## Surface-state members

`gpuwm ensemble CONFIG`, `gpuwm go CONFIG` and `gpuwm run CONFIG` accept the `surface-state` recipe. Add these tables to a config that declares its `[fetch]` source and cycle:

```toml
[ensemble]
members = 8
recipe = "surface-state"
base_seed = 73

[ensemble.perturbation]
kind = "surface-state"
soil_moisture_scale = [0.8, 1.2]
sst_offset_k = [-1.0, 1.0]
```

The recipe prepares the base trajectory once, including every nested domain. Each member initializes its own model through the ordinary runner, then applies its seeded surface realization before the first forecast step. A member uses the same factors on its parent and nests. Preparation caches and source files retain their original bytes. Nested stepping, feedback, physics and adaptive clocks remain the ordinary member's. The executor prints the concrete reason when a native pack cannot run the selected configuration and continues through the ordinary member runner.

`soil_moisture_scale` is dimensionless; `sst_offset_k` is a temperature difference in K. Each accepts a scalar or `[minimum, maximum]`. Scalars set fixed values. Intervals draw one value per member and option using a deterministic GPU integer seed transform and FP32 rounding. An omitted scale is 1 and an omitted offset is 0. A shared-source ensemble with more than one member needs at least one nonconstant interval. Surface options can also accompany `time-lagged` or `multi-model` members, whose source trajectories already differ.

Soil scaling affects land cells with `landmask > 0.5` and `xice == 0`, including every soil level. It clips total moisture to the existing initial-input range 0..1 m3 m-3 and applies the same effective scale to liquid water and RUC frozen water. Noah and Noah-MP keep their existing air-dry SMCDRY floor, with the ordinary 0.005 m3 m-3 fallback where the land soil category has no positive floor; this prevents tiny scale factors from restoring the zero-moisture conductivity failure. The land scheme retains its own category saturation treatment. RUC moisture availability is recomputed from the perturbed top soil and its existing dry/reference parameters. Soil temperatures and SMCREL remain the initialized values. Noah publishes that relative-moisture diagnostic during its land call; RUC does not consume it. SST offsets affect open ocean cells with `landmask < 0.5`, `lakemask < 0.5` and `xice == 0`; land, lakes and sea ice keep their initialized temperatures. RUC's ocean skin, saved skin and SST carriers receive the same offset. An offset that would move an ocean carrier outside 170..400 K refuses before field mutation.

The ensemble receipt records each member seed, named options, exact realized FP32 words, affected domain/field inventories and surface hashes before and after application. Replaying that member alone uses its recorded descriptor and seed through `surface_initialization_callback`; it does not draw a replacement seed or reapply initial perturbations to an advanced domain. These are specified surface sensitivity members. Their amplitudes and probabilities are not calibrated forecast uncertainty.

## Stochastic streaming

Public forecast and native-input doors recognise SPPT, SKEBS and SPP selectors and refuse active ones (exit 2) with the calibration reason until their spread amplitudes are calibrated against observations. This section describes the internal implementation; it does not make an uncalibrated run available.

`StochasticSweepLease` advances the original full-domain provider once. Its two-dimensional SPPT, SKEBS and SPP words are fenced and copied into bounded per-buffer or per-rank windows. Mass and face components retain their own exact global coordinates, including the original spectral extra edge. Each local first-RK nonmicrophysics sum uses the original transformation arithmetic. Microphysics terms remain outside that transformation. `complete_timestep` commits once after every compute window succeeds. The original full-domain binding owns spectra and restart state; there are no full-domain three-dimensional rate banks.

Streamed checkpoints include that binding's typed header and exact complex64 spectrum words through the resident codec. Validation precedes every store mutation. Application restores the owner once after carrier copies. The disabled path adds no stochastic payload. Active graph capture chooses the original window execution family and records that choice.

CPU strategy, lifetime, source identity, endpoint, restart atomicity and licence-notice checks pass. The actual periodic resident-versus-tiled stochastic gate is in `tests/test_ensemble_stochastic_streaming_gpu.py`; it has not been run in this lane. Native whole-wave and multi-wave forecast identity and memory calibration remain release GPU gates.
