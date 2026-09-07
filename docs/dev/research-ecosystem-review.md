# Research ecosystem integration review

Review date: 2026-09-06. Native source is the active research candidate based on `9c727d3c52d1`. The historical MCP v2 adapter remains clean at `ddaca492f2359a928a034fbfc1a36b35e0221259`; the separately reviewed `2.0.0.dev1` extension lives in `work/arwen-mcp-v2-research`. This is a development integration audit, not final candidate qualification.

## Development closures following the initial audit

Native abbreviated help now exposes Research and Scenario. Research parser options have explanatory help, `research attributes [--json]` exposes the actual closed native registry without a device probe, and TUI capture help names `research:ID`. The generated CLI reference was refreshed. The combined first-use, documentation parity, package-data, research compiler and TUI CLI checks passed **101 tests** (`work/research-ecosystem-audit/cli-attributes-discovery-parity.xml`).

The isolated `work/arwen-mcp-v2-research` adapter worktree extends the frozen baseline as development version `2.0.0.dev1`, with **32 typed tools**. It adds native catalog/attribute discovery, reviewed hardware measurement/estimation, reviewed native research creation, and reviewed native downscale plan/run. Measurement is GPU-class work only after launch; declared-capacity creation remains CPU work. Catalog/hardware-policy hashes, polygon files and saved plot settings are review-bound. Local forecast/resume uses a saved plot sidecar unless the caller explicitly selects products or a preset. Native downscale dry-run is a durable job because it can write a derived configuration; that file stays inside a fresh reserved bundle.

Final development checks passed **151 adapter tests with one Windows symlink capability skip** (`work/research-ecosystem-audit/mcp-research-review-closed-full.xml`). Independent review reproduced and closed two races: saved plots appearing during General-preset discovery, and input files changing during native catalog revalidation. The planner now carries the initial sidecar absence and rechecks every local authority after native queries. Independent mutation repros confirm refusal before review or dispatch (`work/mcp-independent-review-fixed/independent-fix-review.json`).

Actual MCP SDK dispatch against a separate source/dependency overlay discovered 114 configurations and five attributes, executed declared-8-GiB hardware estimation and research creation as two CPU jobs, and retained four selected diagnostics in an unlaunched native Go review. One authorized GPU-class hardware job measured the primary RTX 3080 at 9.9995 GiB total and 7.75 GiB free, while independently retaining the requested class-16 geometry policy. The job ended successfully with confirmed termination and its GPU lease released. A separate native downscale dry-run succeeded against explicitly synthetic CPU archive fixtures and wrote its child TOML inside the reserved bundle. Evidence is under workspace-level `work/research-ecosystem-audit/native-sdk-20260906T094634Z/receipt.json` and `native-downscale-sdk-20260906T094821Z/receipt.json`. No forecast was started by these adapter checks.

The adapter wheel and sdist were built as `2.0.0.dev1`; all 13 wheel Python modules match source, and the sdist includes the new regression tests. A fresh wheel installation completed actual stdio MCP initialization, 32-tool discovery, native catalog/attribute discovery, hardware/research reviews, and rejection of arbitrary arguments (`work/research-ecosystem-audit/installed-stdio-20260906T094957Z/receipt.json`). Durable-job checks used explicit host-lifetime ownership; the native engine remains an editable source/dependency overlay. Final engine-artifact pairing, installed broker-host independence, actual forecast and two-platform adapter qualification remain separate release gates. The integration handoff is recorded in the same audit directory.

The initial audit below records the frozen adapter's capabilities. Its source and historical release artifacts were not changed. Public README/guide links and command examples remain a documentation follow-up unless separately completed by the release owner.

## Initial findings against the frozen adapter

| Priority | Finding | Consequence and required completion |
|---|---|---|
| P1 | Frozen MCP has no research catalog, hardware-class discovery, or research creation tool. | Native research workspaces cannot be discovered and created entirely through this adapter. Add a versioned, typed native-command adapter surface before claiming end-to-end MCP parity. |
| P1 | Existing MCP forecast planning selects its installed default plot preset unless products are explicitly supplied; it does not consume a research workspace's `.arwen-plots.json`. | A valid research experiment can execute while losing its recommended diagnostics. Clients must currently supply explicit products. A future research planner should bind and forward the reviewed product selection. |
| P1 | Generated CLI reference is stale. | Three existing documentation parity tests fail. Regenerate the page after parser help wording is complete. |
| P2 | Abbreviated `gpuwm --help`, README, and incoming public links do not expose the research workflow. | Add the native entry point, public guide link, command examples, and hardware-selection explanation. `--help-all` already discovers the registered command. |
| P2 | Attribute-following settings are accepted through existing scientific TOML, but are not discoverable or authored through frozen MCP. | Expose the native closed attribute registry before promising a discoverable MCP attribute editor; keep engine admission and configuration validation authoritative. |

## Actual frozen MCP surface

The installed release adapter was interrogated through `build_server()` and the MCP SDK's `list_tools()` / `list_resources()` methods. It exposes **27 tools and zero resources**. Calling `arwen_capabilities(section="research")` is rejected by its Literal schema. Supplying arbitrary `argv` is rejected as an unsupported argument. These checks dispatched no engine commands and performed no GPU operations.

The relevant authority is `work/arwen-mcp-v2/src/arwen_mcp_v2/tools.py`: `arwen_capabilities` supports only the renderer catalog, physics profiles, sources, and their combined result. `arwen_plan_domain` takes typed point, declared VRAM, source, cycle, ladder, physics, hours, and output fields. It cannot select a research recipe, hardware class, polygon, custom geometry, source-specific recipe method, or scenario. There is no generic filesystem or arbitrary command escape hatch. Server instructions also explicitly exclude named workflow templates from this initial adapter.

Read-only SDK inventory and schema evidence are in the workspace-level `work/research-ecosystem-audit/frozen-mcp-inventory.json`; the reproducer is `sdk_inventory.py`. Source git status was clean after inspection.

## Existing supported path

An externally created research `experiment.toml` can use the existing workflow:

1. Query `arwen_sources` / `arwen_list_products` as needed and supply actual admitted native source inputs.
2. Call `arwen_plan_prep` with the experiment and its source/WPS/manifest/geography files, using `statics_corridor=true` when the scientific configuration requires that native path.
3. Review the returned plan with `arwen_review`, then execute it with `arwen_launch`.
4. Call `arwen_plan_forecast` with the same experiment, the prepared root when applicable, and the research recipe's **explicit products**; review and launch the returned plan.
5. Inspect jobs through `arwen_job_status`, `arwen_job_logs`, and `arwen_job_result`. Resume and render through their existing typed planners.

`plans.py::_authorities` binds complete scientific configuration bytes, explicit input files, known companion files, and existing paths discovered recursively inside TOML. This preserves attribute-following fields without adapter-specific scientific rewrites and includes source-specific paths such as a named ERA5 Vtable. Native admission still decides whether a prepared archive, moving geometry, source, or scenario is valid. The adapter does not create missing parent archives or scenario state. It also does not bind the research receipt or plot sidecar today.

Existing domain and forecast tools are therefore usable for an already authored, admissible experiment, but are not a complete research creation or attribute-editing interface. Frozen MCP also has no native `downscale` planner; existing parent-archive creation paths remain a separate integration gap.

## Minimal next adapter surface

Keep the adapter thin and version the change separately from the frozen adapter and its historical evidence:

- A read-only research catalog tool should call native `research catalog --json` and return native recipe, method, diagnostics, and hardware-profile metadata. Bind the selected catalog identity and engine identity in subsequent plans. The measured response is approximately 279 KB, within the existing query output bound; clients may benefit from typed family/recipe filtering.
- A hardware tool should distinguish declared capacity from actual local measurement. Native `research hardware` is the authority. If measurement initializes CUDA, execute it with the existing GPU resource/lease path; do not classify an automatic hardware measurement as an unrestricted CPU query.
- A typed research creation planner should accept recipe ID, reviewed geography, cycle, supported source, explicit hardware selection, and a new output directory; invoke native `research create` through the existing review/token/job machinery. Preserve output reservation, input hashes, engine identity, and launch-time revalidation. Do not add arbitrary arguments or copy the domain planner into MCP.
- Bind the compiled scientific configuration and research products after creation, and pass explicit products to forecast/render planning. Add a typed parent-downscale path if the promised workflow includes creating workspaces from archived parents.
- Attribute discovery should expose the native closed registry (including aggregate, vertical-window, threshold, polarity, and unit contracts). Current `research catalog` does not contain that registry. Attribute editing can initially remain ordinary native TOML editing; a later typed creation/editing interface must invoke engine validation and preserve review-bound file bytes.

No MCP code or launcher was modified in this review. Final adapter tests, actual SDK discovery against the final installed engine, package pairing, and physical hardware tests remain release work.

## Native CLI and documentation evidence

Actual source commands executed successfully: `gpuwm --help`, `gpuwm --help-all`, `gpuwm research --help`, `gpuwm research create --help`, `gpuwm research hardware --help`, and `gpuwm research catalog --json`. The abbreviated help contains no research entry. Exhaustive help and registered research command help do contain it. Several research options currently have no explanatory help text, so regeneration alone will produce weak reference entries.

The combined existing `test_docs_extras_agree_with_code.py` and package-data checks reported **43 passed, 3 failed** in 7.44 seconds. All failures are documentation parity: missing research command sections/positionals, missing `downscale --auto-vram`, missing three TUI snapshot flags, and the resulting stale generated page. A rendered preview, without editing the public page, is saved at workspace-level `work/research-ecosystem-audit/CLI-OPTIONS.rendered-preview.md`. Actual command output and `cli-help-audit.json` are in the same audit directory.

The public research guide documents all 114 configurations and their scientific limits, but currently lacks command examples and incoming links from README or other public documents. The practical introduction should show catalog discovery, declared/automatic hardware selection, research creation, how recommended diagnostics are retained, and explicit continuation from an existing scenario or parent archive. Public text should distinguish creating/planning an experiment from running and qualifying it.

## Package boundary

`pyproject.toml` includes the recursive `gpuwm` package-data glob `data/**/*`. Both `gpuwm/data/tui/research-workspaces.json` and `research-hardware-profiles.json` are within that declaration; native Python modules are included by package discovery. Existing package-data coverage tests pass against actual setuptools discovery. `MANIFEST.in` does not prune these files.

This verifies source packaging declarations, not a built final wheel or sdist. Inspect the final artifacts and installed package data after the candidate is frozen. The prepared MCP service's engine package-file manifest must then be regenerated against that exact final engine build; an older engine/adapter receipt is historical and cannot qualify the new research code.
