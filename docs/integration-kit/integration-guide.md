# Integrating ArWen 2.7

## Choose the boundary

| Task | Interface | Result |
|---|---|---|
| Discover data sources | `gpuwm sources --json` | `gpuwm.run-plan.sources.v1` |
| Inspect a source | `gpuwm sources SOURCE --json` | Source capabilities and restrictions |
| Discover plot products | `gpuwm run-plan --catalog` | `gpuwm.run-plan.catalog.v1` |
| Discover physics choices | `gpuwm run-plan --physics-profiles` | `gpuwm.run-plan.physics-profiles.v1` |
| Read device inventory | `gpuwm run-plan --probe --no-readiness` | `gpuwm.run-plan.probe.v1` |
| Resolve a plan | `gpuwm run-plan PLAN.json --resolve` | `gpuwm.run-plan.resolved.v1` |
| Estimate its resources | `gpuwm run-plan PLAN.json --estimate` | `gpuwm.run-plan.estimate.v1` |
| Execute the reviewed plan | `gpuwm run-plan PLAN.json` | Manifest, heartbeat, events, committed output |
| Manage a remote job | `gpuwm remote ACTION ...` | Versioned remote result |
| Read maps in a native client | Native RWS/RWG and companion receipt | Exact fields, geometry, units, and time |

Run `gpuwm remote --help` for the complete option syntax of this release. Remote operations include probe, list, status, logs, review-plan, start-plan, stop, resume, artifact-index, sync-artifacts, sync-processed-frame, and the compact viewer transfer. Use argument arrays, not shell command concatenation.

`--no-readiness` limits the inventory probe to NVML. A full readiness probe executes runtime checks and can create a CUDA context. Do not use the latter as a frequent background poll on a busy forecast computer.

Treat schema identifiers as required. Inspect both the subprocess exit code and the returned document: a catalog can describe an unavailable native capability. Show its reason in the UI. Preserve unknown response fields when forwarding documents; never silently discard an unsupported input option. Version numbers alone do not establish source or binary compatibility.

## Discover before constructing controls

Populate model/source choices, products, and physics profiles from the installed catalogs. HRRR remains a regular source. The special HRRR maximum demonstration is outside this release interface.

A model's forecast times, ensemble members, geographic footprint, archive range, and native cadence differ. Use the source's declared availability rather than a universal list of hourly slots. An unavailable source/time pairing should remain a clear refusal. For an own forecast, simulation, hindcast, or future forecast, distinguish the requested valid period from the initialization and input coverage used to produce it.

The UI can display an easy default and disclose additional choices. A review should show the resolved domain geometry, input source and initialization, duration and output cadence, physics, selected computer, memory estimate, output path, and automatic resolutions. If those inputs change, invalidate the old review. Start the exact plan/configuration that was reviewed.

## A plan is an envelope over a configuration

`examples/prepared-plan.json` uses an existing experiment TOML. Place a real `forecast.toml` alongside it, change `output_root`, then run:

```text
gpuwm run-plan prepared-plan.json --resolve
gpuwm run-plan prepared-plan.json --estimate
```

These calls do not start the integration. A successful review does not guarantee that remote resources or source inputs will remain available. Retain the plan and configuration hashes and revalidate changes before starting.

Execution is a separate, explicit operation:

```text
gpuwm run-plan prepared-plan.json
```

Use the shipped configuration loaders and run-plan routes for validation. Do not reproduce domain fitting, memory arithmetic, source preparation, physics compatibility, or integration in a UI client. Discover the supported intent routes from the source catalog instead of inventing configuration keys.

## Reconnect without owning the simulation process

Read `run-manifest.json` first. It identifies the run, producer, and paths of the durable streams. Read `run-progress.json` for current state, then replay `events.jsonl` for history and tail its new complete lines. The heartbeat is the current-state authority; the event stream provides history. After a crash, the last event alone cannot prove the process remains alive.

Plan events use `gpuwm.run-plan.event.v1`, a monotonic `sequence`, an emission timestamp, and an `event` tag. Ignore an incomplete final line until more bytes arrive. Reject a changed run manifest and reconnect deliberately. Execution stdout may also contain stage output; do not feed every stdout line to an unconditional JSON parser. Query modes return a JSON document on stdout and diagnostics on stderr.

A committed frame is identified by its producing run, domain, sequence, and exact valid UTC. Use the commit's authority, source checksum, and initialization/lead relationship. A filename's ordinal or an integer forecast hour cannot represent every subhourly output. A moved domain can have new geometry at each time.

Opening a result pins a viewer session to that run. Editing a new forecast draft, receiving a status update, or discovering another run must not replace the result the person is inspecting. Following the newest time is an explicit player state; pausing should preserve the chosen UTC as new output arrives.

## Remote execution and desktop control

The CLI uses the user's existing OpenSSH configuration and durable remote job workspace. Keep host identity and the selected target attached to review, start, status, and result operations. A target change invalidates the prior review. Use the remote `review-plan`/`start-plan` path for the same reviewed plan; do not silently substitute the local computer.

The desktop's current internal bridge is `arwen.companion-handoff.v1`. It supplies a session ID, runtime/cwd, selected configuration identity, status path, and a request directory. `src/bridge.rs` in the companion repository and `tools/arwen-tui` in the engine repository are the implementation references. This is a release-coupled desktop integration seam, not a separately stable public service. Prefer the versioned CLI for independent applications.

If extending that bridge, retain request IDs, session/target binding, asynchronous responses, cancellation, and stale-result rejection. A launch acknowledgement may open the newly started run; background status must not navigate the application. Never package a user's handoff, private target configuration, SSH material, or credentials into a distributable prototype.

## Maps, loops, and full analysis

The compact viewer derives selected two-dimensional products from a committed WRF frame on the producing computer. It transfers individually checksummed native members and retains their original values and geometry. The GUI presents those fields using the existing product recipes, color tables, contours, barbs, units, and legends.

The compact native processing contract is `arwen.wrf-process-request.v2` with profile `viewer-2d-v1`; its result uses `arwen.wrf-process-result.v2`. The remote selected-frame response uses `arwen.remote-processed-frame.v2`. It includes the native result, source and publication checksums, run manifest/commit, exact time, member inventory, and local cache paths. This viewer contract is release-coupled; validate it against the installed engine before depending on it in another client.

A native reader must validate the receipt and member checksums and hold the shared cache lease for as long as any reader, map, or frame clone references the object's files. Cache paths are owned local objects, not arbitrary paths supplied by a remote server. The GUI checks run, domain, UTC, source authority, path containment, and lease identity before opening a transferred result. Reopening an evicted frame must request it again.

Playback selects actual available times. It buffers for an unready frame and supports an explicit UTC loop range. Prefetch is bounded and begins on an explicit Play request. Keep the displayed timestamp tied to the actual displayed frame. Avoid taking over the user's camera when a time changes; following a moving domain is a separate choice.

Compact map data does not contain a full three-dimensional sounding volume. Full soundings and 3D fields are a separate explicit analysis preparation. The same remote `sync_processed_frame_v2` action accepts `profile: "full-science-v1"` for the explicitly selected sequence; it invokes the retained `arwen.wrf-process-request.v1` native processor and returns a native v1 result inside the v2 transfer envelope. Do not send prefetch sequences for a single-frame analysis request. The legacy bulk full-store transfer is not the desktop's analysis path. Do not fabricate missing vertical profiles from two-dimensional maps or silently generate the full history merely because someone opened a run.

## What a new front end should test

Exercise source/date changes, exact subhourly frames, unavailable inputs, target changes after review, reconnect after the client closes, stale responses, native processing failure, cache eviction, long loops, moved domains, and a paused cursor while a running forecast appends output. Verify scientific values and geometry as well as the screenshot. A polished empty state must explain how to continue; an error must identify the failing operation and offer an appropriate retry.

The examples in this kit query and inspect. They do not start, stop, resume, or delete runs. Add those mutations only at clear user actions in your application, using the actual reviewed plan and selected target.
