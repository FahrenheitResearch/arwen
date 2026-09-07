# ArWen 2.7 terminal workspace: user manual

The terminal workspace is a keyboard-and-mouse interface to ArWen's configuration, preparation, forecast, and plotting commands. It keeps the complete TOML configuration editable and shows the command before a launch. Opening the interface, browsing a catalog, or editing a draft does not start a forecast.

Use this manual for the interface. Use the [CLI user manual](CLI-USER-MANUAL.md) for shell commands and automation.

## Contents

1. [Install and open the workspace](#1-install-and-open-the-workspace)
2. [Find your way around](#2-find-your-way-around)
3. [Create and launch a first forecast](#3-create-and-launch-a-first-forecast)
4. [Edit, undo, save, and repeat](#4-edit-undo-save-and-repeat)
5. [Choose a source and UTC cycle](#5-choose-a-source-and-utc-cycle)
6. [Shape domains and manage memory](#6-shape-domains-and-manage-memory)
7. [Start a research workspace](#7-start-a-research-workspace)
8. [Open a historical-case catalog](#8-open-a-historical-case-catalog)
9. [Create a controlled scenario](#9-create-a-controlled-scenario)
10. [Use existing inputs and prepared bundles](#10-use-existing-inputs-and-prepared-bundles)
11. [Downscale an archived forecast](#11-downscale-an-archived-forecast)
12. [Choose plots and render saved output](#12-choose-plots-and-render-saved-output)
13. [Read progress, results, and failures](#13-read-progress-results-and-failures)
14. [Stop, quit, and resume](#14-stop-quit-and-resume)
15. [Run on a Linux node](#15-run-on-a-linux-node)
16. [Key reference](#16-key-reference)
17. [Troubleshooting](#17-troubleshooting)

## 1. Install and open the workspace

Use a Windows or Linux terminal and the Python environment containing the ArWen version you intend to run. Python 3.11 or newer is required. Local integration requires a compatible NVIDIA GPU, driver, and CuPy installation. A machine without a GPU can still edit configurations, estimate a declared target, prepare supported inputs on the CPU, render existing history, and control a prepared Linux node.

For an installed release whose native tools are not yet staged:

```text
gpuwm version --offline
gpuwm setup
gpuwm doctor
gpuwm tui
```

`setup` stages the release's native tools, including the terminal executable, and physics tables. Static geography is a separate download: use `gpuwm setup --with-geog` or `gpuwm fetch-geog` when you need it. Read the size printed by the command and allow additional space for forcing files and forecast output.

For a developer checkout, run `bash install.sh` on POSIX or `.\install.ps1` in PowerShell from the repository root. These scripts create or reuse `.venv`, install the matching local data companion and main package, and build the terminal workspace as well as the other requested native tools. Activate that environment before running `gpuwm tui`; see the [CLI installation instructions](CLI-USER-MANUAL.md#2-install-and-check-the-environment).

Open a known configuration directly:

```text
gpuwm tui --config configs/forecast.toml --output runs
```

Paths containing spaces must be quoted in your shell. Within an interface path field, type or paste the path itself without adding shell quotes. The selected Python controls every local engine job; **F9 Python** changes it and **F10 Check installation** checks it.

For a comfortable layout, use at least 65 columns by 20 rows and enlarge the terminal for long descriptions. **Ctrl+Q** and **F1** remain available when the terminal is too small for editing. A nonempty `NO_COLOR` uses your terminal's default colors while retaining bold, underline, and visible selection markers.

## 2. Find your way around

Home provides entry points for a new forecast, existing inputs, continuation, fitting, research, and case catalogs. The overview shows the selected configuration, output location, target, and available actions.

| Area | Use it for |
|---|---|
| Home | Choose the kind of task you want to perform. |
| Overview | Review the open configuration and select Check, Plan, Run, or an existing preparation. |
| Settings | Edit the entire TOML draft. |
| Domains | Review existing domain geometry, following, and lifecycle controls. |
| Research | Browse weather questions and starting configurations. |
| Scenario | Review initial-state warm bubbles in an open configuration. |
| Plots | Select requested pictures and save the selection beside the configuration. |
| Nodes | Select Local computer or an existing Linux installation. |
| Logs / Details | Read progress, failure reasons, saved log paths, and completion status. |

Click a tab or use the shortcuts in the [key reference](#16-key-reference). While editing TOML, ordinary letters remain text. Use **Esc**, a tab click, or a Ctrl shortcut to leave the editor.

In a guide, **Enter** advances, **Shift+Tab** goes back, **Ctrl+U** clears the current answer, and **Ctrl+A** opens all settings. The ordinary route does not require advanced arguments. To supply them, open **All settings → Advanced CLI options**, set it to `on`, and enter the requested list of quoted argument strings. Setting it back to `off` removes those arguments from the request.

**F1 / ?** opens help over a form. **Esc** closes help and returns to the same answers. Leaving an ordinary setup with **Esc** retains its answers for this session; reopening the same task resumes them. Retained guide answers are not a saved configuration.

## 3. Create and launch a first forecast

You need staged geography, enough disk space, an admitted source/time window, and a suitable local GPU or configured node before integration.

1. On Home, choose **Choose a forecast mode** or press **N**. Choose the relevant workflow and its new-forecast action.
2. Enter a center as `latitude,longitude`, for example `35.3,-97.5`, or choose a local GeoJSON footprint. West and south coordinates are negative.
3. Review the source. The interface displays the installed engine's recommendation; select a source appropriate to your location and period.
4. Choose the **Start cycle (UTC)** and duration. **F3** opens the calendar from the cycle field.
5. Choose a **new configuration filename**. Optional settings include root spacing, nest ratios, vertical levels, output cadence, physics, and target GPU capacity.
6. Open the settings summary and review every value. Confirm the displayed creation command. Successful creation opens the new TOML; no forecast has started.
7. Save any subsequent edits with **Ctrl+S**. Use **F5 Check** to inspect configuration and resource admission, then **F6 Review launch plan**.
8. Read the selected route, output location, and requested plots. Use **F7 Prepare and run forecast**, review its exact command, and confirm when ready.

Check the target in the header before launching. With **Local computer** selected, Check and Plan operate locally. With a node selected, the corresponding actions inspect or review that remote installation and job.

A point lets the fitter choose a centered rectangle within the memory budget. A GeoJSON footprint asks it to retain the entire required area. Smaller grid spacing provides finer sampling; it does not independently enlarge coverage. Inspect the generated extent and every nest before preparing data.

## 4. Edit, undo, save, and repeat

Open Settings with **E** outside the editor, or click its tab. You can edit settings beyond those exposed by the forms.

| Action | Effect |
|---|---|
| Ctrl+Z Undo | Restore the previous draft edit. |
| Ctrl+Y Redo | Reapply an undone edit. |
| Ctrl+U Discard, in Settings | Restore the saved configuration; this discard can itself be undone. |
| Ctrl+S Save | Validate TOML syntax, save the draft, and preserve a backup of the previous file. |
| F12 Save As | Write the full draft to a new filename in an existing folder. It can preserve an unfinished draft. |

Save As refuses an existing file. An unfinished draft saved this way still needs correction before Check or Run. If the original file changed outside the interface, ordinary Save refuses to replace those changes and retains your draft; use Save As to preserve your work separately.

Saving a file currently used by an active command is refused. You can continue editing a draft and use Save As for a separate experiment. Moving a configuration to another folder may change how its relative input paths resolve: review those paths and run Check again.

To repeat the same forecast, reuse the saved configuration and select an ordinary output parent. Go and Sim normally create a fresh timestamped run folder. To create a variation, use Save As or start the guide again. New, Research, Fit, and Tiles choose an unused default filename, including a numeric suffix when needed; an explicitly chosen occupied filename is still refused. Keep generated companion files with their configuration.

Changing a forecast's science can invalidate an existing preparation or checkpoint. A successful Save is not preparation validation. Use Check and Plan after changes, and prepare again when the engine says the existing bundle no longer matches.

## 5. Choose a source and UTC cycle

The source determines geographic coverage, initialization cycles, forcing cadence, available lead times, and acquisition requirements. A name in the registry does not mean every stage supports every product from that source. Use the source details and the launch plan.

In a source or physics question, **F2** opens the relevant catalog. In a cycle question, **F3** opens the source-aware calendar:

| Calendar control | Action |
|---|---|
| Arrows / Tab, or a day/hour click | Select a date and a permitted UTC hour. |
| PgUp / PgDn | Change month. |
| Ctrl+PgUp / Ctrl+PgDn | Change year. |
| Y or F2; click the year | Enter a year directly, then press Enter. |
| F4 | Ask for the latest complete or latest expected cycle, as labelled. |
| Enter / Use date | Keep the chosen cycle. |
| Esc | Cancel the calendar/lookup and retain the guide's previous value. |

You can type an exact `YYYY-MM-DDTHH` UTC cycle. Convert a local event time to UTC before entering it; the calendar does not interpret it as your computer's local time.

For sources with public object probes, **Latest complete** checks the requested window rather than just the presence of an analysis file. ERA5 uses **Latest expected**, based on publication delay and the whole requested analysis window, and shows account and recent-data guidance. An expected date is not proof that your account has retrieved the data. Archive bounds and publication estimates are guidance; the acquisition stage still verifies the actual files.

Changing the source does not silently rewrite an explicitly chosen physics suite. If the combination is refused, choose a supported combination deliberately and review it again.

## 6. Shape domains and manage memory

**D / Ctrl+D Domains** opens controls for the configuration's existing domains. Review grid size, parent placement, refinement and time-step ratios, delayed starts, following, triggered activation, retirement, and re-arm where present.

Edit fields, then use **F2 Apply to draft**. This changes the reviewed settings in the TOML draft; use **Ctrl+S** to save, followed by Check and Plan. Disabling a policy removes its policy table when applied. Cancel leaves the draft unchanged. If changing a signal makes another field incompatible, the form names it and asks you to clear it explicitly with **Ctrl+U**.

Tracking needs an appropriate diagnostic, threshold units, and room for movement. For example, UH following requires a reflectivity fallback threshold. A moving feature can leave a fixed fine domain, and a moving nest is limited by its parent and prepared coverage. Inspect the actual domain spans and movement limits in Details.

### Fit a starter

Choose **Fit starter TOML** on Home, or **Fit domain** after a confirmed GPU memory refusal. Supply the complete starter, location or footprint, intended start time, and a new output filename. Empty duration preserves the starter's duration. Empty GPU capacity measures this computer; a supplied capacity is a planning estimate for that target.

Fitting preserves the starter's scientific choices while changing the fitted layout. A required footprint or research minimum span can make the request impossible to fit. Review the new layout rather than assuming every card can hold the same area.

### Enable tile streaming

After a confirmed GPU memory refusal, **Tile streaming** or **Ctrl+T** opens a reviewed copy of the current configuration:

- `auto` streams only when the planner determines that resident execution does not fit.
- `on` requests streaming and lets the planner choose tile dimensions.

The full domain uses system RAM while the GPU processes tiles. The engine checks both budgets. This workflow retains area, resolution, timing, and physics and writes a new configuration after validation. It does not choose an arbitrary tile size or start integration merely by opening the form.

| Choose | When |
|---|---|
| Fit domain / Ctrl+F | You can revise the fitted extent while retaining the starter's scientific settings. |
| Tile streaming / Ctrl+T | You need to retain the configured extent and have enough system RAM. |
| Read the original memory error | The shortage is source decoding or another host-RAM stage; GPU tiling does not remove that cost. |

Memory recovery is offered only for a recognized refusal tied to the current saved configuration and a validated command result. Save an unsaved draft first. Entering a larger capacity does not add memory; a machine with other GPU users may have much less free memory than its nominal capacity. If Check labels an estimate **INCOMPLETE**, only the evaluated gates were assessed; do not read a fitting estimate as complete device or forecast validation.

## 7. Start a research workspace

Press **W Research**. Browse the family, question, and starting configuration; the catalog contains 12 families, 38 questions, and 114 configurations. Type to filter, including a displayed method such as `archived-parent downscale`. **F4 Details** opens the full scrollable description.

1. Read the research question, required inputs, comparison, limitations, and **Further analysis** disclosures.
2. Select a configuration whose method matches the work you intend to do.
3. Enter the study location, source, UTC cycle, duration, and new filename.
4. Start with hardware profile `auto`, or `8` for an 8 GiB target. Use the bare number, not `8gb`. Leave capacity blank to measure the current machine.
5. Review the exact command and create the configuration. Inspect its fitted coverage, nests, tracking, scenario settings, and plots before preparation.

The hardware profile selects a level of resource detail; it does not declare free memory or guarantee a fit. Larger profiles are marked **fit not checked**. If one cannot preserve the question's minimum study area, retry the same question with `auto` or the suggested smaller profile. The engine still checks the required area and budget.

An archived-parent method needs real archived history. A recipe requiring an existing scenario needs that state. The guide does not replace these inputs with a new ordinary analysis. **Ctrl+D** opens the selected family's archived-parent guide; **Ctrl+N** opens its family starter.

Creating a research configuration checks packaged diagnostic capability metadata and does not require launching the renderer. Actual plotting still requires the renderer and suitable saved fields and time windows. Catalog membership and configuration admission do not establish forecast skill or validate the proposed research question.

The current ArWen history route does not provide gust magnitude, visibility distance, precipitation-type categories, DCAPE, or several derived near-surface comfort/moisture quantities. Context plots are labelled as context. The two lapse-rate alternatives use **plain temperature**, not virtual temperature. Read Further analysis rather than treating a related field as the requested quantity. See the [CLI diagnostic guidance](CLI-USER-MANUAL.md#11-render-existing-forecast-output).

## 8. Open a historical-case catalog

Press **K Cases** or choose **Browse case catalog** on Home. Supply a JSON, TOML, or supported ZIP catalog. The interface reads catalog data; it does not run scripts included by the catalog author. A checkout's offered synthetic example is labelled as such.

Search the list and use **PgUp/PgDn** for additional pages. Open a case, use **Tab** to select tier, source, output, and VRAM fields, and use **Left/Right** to choose a listed tier or initialization. **F3** previews the full selection and notes. **Review creation** shows the exact command.

Creation preserves catalog provenance and the selected source-specific geometry and cadence. It opens editable TOML without starting a forecast. If the catalog changed after review or the destination already exists, creation refuses; refresh the preview or choose a new path. Selecting a named historical event is not evidence that its data are available or that the resulting simulation reproduces the event.

`GPUWM_TUI_CASE_CATALOG` can supply the initial catalog path. Opening a catalog does not overwrite it.

## 9. Create a controlled scenario

Open the base configuration and press **I Scenario**. The Scenario form edits initial-state warm bubbles:

| Field | Meaning |
|---|---|
| Latitude / longitude | Bubble center in degrees; west is negative. |
| Center height | Height above ground in metres. |
| Horizontal radius | Radius in kilometres. |
| Vertical half-depth | Distance from center to the top or bottom, in metres. |
| Peak theta increase | Potential-temperature perturbation greater than 0 and at most 10 K; warm bubbles only. |
| Preserve relative humidity | Adjust water vapor inside the bubble to retain RH, or leave vapor unchanged. |

Review the bubble list and apply it to the draft. **F12** saves a separate scenario file. Check the result before preparing or running it. The engine validates placement, dimensions, and whether the perturbation actually reaches cells. This is an explicit hypothetical change to the initial state; label its plots and compare it with an unmodified control using the same other settings.

## 10. Use existing inputs and prepared bundles

On Home choose **Open existing**, then select what you actually have:

| Input | What to supply |
|---|---|
| ArWen configuration | The complete `.toml` and its referenced files. |
| WRF `real.exe` output | `wrfinput_d0*`, `wrfbdy_d01`, and the producing `namelist.input`. |
| WPS `met_em` output | `met_em.d0*.nc` and the producing `namelist.input`. |
| Prepared bundle | The finished native preparation directory, its experiment TOML, and the exact WPS namelist when required. |

The producing namelist is authoritative. An unrelated physics preset cannot repair mismatched input metadata. For a namelist-to-TOML translation and substitution review, use `gpuwm import-namelist` as described in the [CLI manual](CLI-USER-MANUAL.md#9-use-wrf-and-wps-inputs).

To run a preparation with an open configuration, choose its directory with **F4** and review **F8 Run existing preparation**. The Continue menu also offers a prepared-bundle guide with fields for its configuration and WPS authority. A downloads directory, loose GRIB file, or incomplete preparation is not a prepared bundle.

This action runs `gpuwm sim`. With selected plots, Sim draws the **first committed history frame only** while integration continues. Use Plot history afterward for the rest of the saved timeline.

## 11. Downscale an archived forecast

Open **Domains → Downscale an archived forecast**, or the corresponding Research action. You need actual parent history covering the intended window, parent physics evidence, and suitable child initialization data.

1. Select parent history and, if necessary, its domain ID.
2. Supply the parent's ArWen restart or producing stock-WRF `namelist.input` as appropriate.
3. For an ArWen parent, supply a child center for derived geometry. Alternatively provide the required standalone child RunConfig and parent placement indices.
4. Review refinement, duration, history cadence, and the acceptable parent boundary interval. A maximum interval and **Accept parent archive cadence** are mutually exclusive choices.
5. Supply a child-grid surface warm-start file when surface physics requires one. A parent-grid file is not a substitute.
6. Choose a **new** output directory and action `plan`. Planning validates and may write the derived TOML and report; it does not run a forecast.
7. Read the plan, then select action `run` with another unused output directory and review the launch.

Point sizing measures the local GPU by default in this guide. Entering explicit capacity, child dimensions, or a child configuration disables automatic sizing. The parent namelist's domain column and the history's domain ID are different selectors; set both correctly when needed.

## 12. Choose plots and render saved output

**B / Ctrl+P Plots** opens plot settings. The initial General selection requests 25 plots. Other presets provide starting selections for severe, tropical, winter, rain, and wind studies.

Use **Customize** to load the installed renderer's catalog, type to search, and press **Space** or click a checkbox to toggle a product. **F4** reviews the selection. **All** requests the available catalog; **None** skips pictures. **Edit list** accepts advanced comma-separated selectors such as `var:SNOWH` when that field is available.

**Save plots** writes `<configuration>.arwen-plots.json` beside the TOML. Reopening restores that exact selection. Save As carries an explicitly saved choice unless the destination already has plot settings. Changed or invalid plot settings are reported rather than silently replaced with defaults.

Plot choices do not change saved history fields, output cadence, physics, or grid settings. A requested picture can therefore remain unavailable even when its name is in the catalog.

To draw an existing forecast, choose **Plot history**, supply a real wrfout file or folder, review the discovered files, and choose products, frame selection, image size, source label, and output. The guide uses `render --series`: compatible files provide one timeline while separate runs, domains, grids, and lifecycle episodes remain separate.

Time-window products need suitable history. One-hour rainfall needs its supporting times; six-hour rainfall needs a six-hour window; 24-hour temperature extrema need the full required day. More frequent saved frames do not automatically satisfy every native window convention. Read the renderer's availability and skip reasons. A snapshot wind maximum is not a gust, and a cloud-layer panel is not total cloud fraction or visibility.

## 13. Read progress, results, and failures

Starting a local command leaves the interface responsive. The header retains **FAILED**, **INTERRUPTED**, or **COMPLETED** after it exits. Read the specific stage and outcome; configuration creation, preparation, integration, and rendering are different operations.

Click the status summary or press **Ctrl+L Details** to see the reason and exact saved log path. **Home/End**, page keys, scroll buttons, and the wheel navigate long details.

| Details action | Result |
|---|---|
| C Copy details | Copy the full details, including text outside the viewport and the saved log path. |
| Y Copy log | Copy the complete saved log, up to 16 MiB; an unavailable or oversized file produces an error. |
| L View log | Return to recent command output. |

The live views retain up to 1,000 recent lines / 128 KiB; the saved `job.log` holds the fuller record. Local interface logs live under the selected output directory in `.arwen-tui/<job-id>`. A missing or inconsistent completion record does not become a successful result.

For model results, use the run directory printed by the command. Go and Sim normally create `run-...` subdirectories containing history and reports. PNGs are normally separated by domain, product, and valid day. Read the forecast's report and health verdict as well as the interface completion state; a picture alone does not establish a usable forecast.

On Windows, copying uses the native Unicode clipboard. In Windows Terminal, hold **Shift** while dragging to select text when the interface captures mouse events. On Linux, clipboard copying needs `wl-copy`, `xclip`, or `xsel` and the corresponding desktop session. If copying fails, open the saved log path shown in Details.

## 14. Stop, quit, and resume

### Local work

**Ctrl+C** or **X Stop** opens a stop review for an active local command. Confirming Stop keeps partial output but does not promise a new checkpoint.

On Linux, Stop first allows five seconds for interruption cleanup, then forces termination; a second confirmed Stop forces it immediately. On Windows, Stop terminates the command's owned process job. Wait for the reported result before starting another forecast on the same GPU.

**Ctrl+Q** opens Quit from every screen. Ordinary **Q** also quits outside text entry. If local work is active, Quit offers **Stay** or **Stop and quit** and waits for worker termination. Ordinary Quit offers no keep-running choice; use a durable node job for unattended work.

### Continue from a checkpoint

Choose **Continue forecast**. For a config-driven run, supply the original configuration, its existing forecast/checkpoint directory, and either `latest` or an actual `gpuwmrst_*.npz` checkpoint. The engine chooses a complete valid set; a log folder alone is insufficient.

Checkpoint writing must have been enabled in the configuration and reached successfully. A forced stop may leave only an earlier valid checkpoint, or none. Prepared single-domain and hierarchy restarts require the matching bundle and configuration, plus the exact WPS namelist where required; use the explicit restart route described in the [CLI manual](CLI-USER-MANUAL.md#12-continue-or-branch-a-forecast). Do not rename a history file to make it look like a restart.

## 15. Run on a Linux node

Nodes control an **existing** Linux ArWen installation through SSH. The interface does not install ArWen remotely or upload your forcing, geography, or configuration automatically.

### Add and select a node

1. Ensure OpenSSH can reach the node using your existing alias or `user@host`, key/agent, and known-host entry.
2. Open **R / Ctrl+R Nodes**, then **N Add**.
3. Enter an absolute remote Python path, existing remote workspace, and existing remote configuration.
4. If used, enter an existing remote output parent. Each interface launch creates a new child directory. Geography, prepared-bundle, and WPS paths are also remote paths.
5. Identity and SSH-config fields refer to files on **this** computer.
6. **Save** the profile, then **P Connect** to inspect the actual remote installation. Confirm the selected target in the header.
7. Use **S Start** to validate inputs and review a launch. Confirm the exact configuration, output directory, and products before starting.

Supply a prepared bundle to reuse validated inputs without fetching or preparing another copy. A single-domain bundle may also need its exact WPS namelist. Both computers need compatible `gpuwm remote` support.

### Reconnect and manage jobs

Remote jobs persist when the interface closes, SSH disconnects, or the local computer restarts. Open **J Jobs**, select a recorded job, and view its state and bounded logs. In a job, **R Refresh**, **X Stop**, and **C Resume** perform their labelled actions; Stop and Resume require review. An empty Jobs list offers a first Start review.

A connection failure leaves the last known state visible. It does not establish that the remote job stopped. If starting a job produces an uncertain result, refresh Jobs before starting another; an uncertain launch is not automatically retried. Resume selects a validated checkpoint and a new output directory. Inputs changed since review are refused.

Choose **L Local computer** to return to local work. **Delete** reviews removal of a saved profile; removing it does not stop remote jobs or remove remote files.

### Where profiles live

| Platform | Shared profile file |
|---|---|
| Windows | `%APPDATA%/ArWen/nodes.json` |
| Linux with XDG configuration | `$XDG_CONFIG_HOME/arwen/nodes.json` |
| Linux default | `~/.config/arwen/nodes.json` |

Profiles and their node preferences are shared across launch folders. Existing `.arwen-nodes.json` profiles are imported once with the original file preserved. Profiles store paths and reconnect IDs, not passwords or private-key contents. SSH retains strict known-host verification.

## 16. Key reference

Keys are contextual. Use Ctrl forms while editing where an ordinary letter would insert text.

| Key | Action |
|---|---|
| Ctrl+Q | Quit anywhere; review active local work. |
| Ctrl+C | Review Stop for active local work. |
| F1 / ? | Help; Esc returns to the previous form. |
| H / V / E / L | Home / Overview / Settings / Logs, outside text entry. |
| W / I / K | Research / Scenario / Cases, outside text entry. |
| Ctrl+O | Open a configuration by path. |
| Ctrl+S / F12 | Save / Save As. |
| Ctrl+Z / Ctrl+Y | Undo / Redo in Settings. |
| Ctrl+U | Clear a guide field; undoable Discard in Settings. |
| D / Ctrl+D | Domains. |
| B / Ctrl+P | Plot choices. |
| R / Ctrl+R | Nodes. |
| G / Ctrl+G | Geography directory. |
| F3 | Output directory; UTC calendar while in a cycle question. |
| F4 / F8 | Select existing preparation / review its launch. |
| F5 / F6 / F7 | Check / Plan / review Prepare and run; selected-node actions use that target. |
| F9 / F10 | Python / installation check. |
| Ctrl+L | Command details and saved-log actions. |
| Ctrl+F / Ctrl+T | Fit / Tiles after an eligible memory refusal. |
| Enter / Shift+Tab | Next / previous in a guide. |
| Ctrl+A | All guide settings. |
| F2 | Apply in domain forms; catalog in applicable questions; year entry in the calendar. |
| End | Follow the latest log output. |

## 17. Troubleshooting

| Symptom | Next action |
|---|---|
| `gpuwm tui` cannot find the terminal executable | Use the installed release's `gpuwm fetch-bridges`, or build `tools/arwen-tui` from its own directory with `cargo build --release --locked --offline`. Check the selected Python. |
| Commands use the wrong installation | Read `gpuwm version --offline`; use F9 to select the intended environment and F10 to inspect it. |
| New/Research says a destination exists | Choose a new filename and stem; companion files also occupy the destination. |
| Save reports an external change | Keep the draft, use F12 to save a separate copy, then compare with the changed file. |
| Check refuses memory | Read GPU versus host-RAM requirements; use Fit or Tiles only when the reported stage supports that remedy. |
| A research profile cannot retain its minimum area | Retry the same question with Auto or the suggested smaller profile and the same actual capacity. |
| Latest cannot find a complete cycle | Read source coverage and horizon details, choose an explicit UTC cycle, or retry when the source has published the required window. |
| Preparation requests a missing donor such as PMSL | Supply real compatible data through the supported preparation options; see the CLI supplement instructions. Do not substitute a different pressure quantity. |
| A plot is missing | Check its actual fields, available times, and skip reason. Saving a plot selection does not add fields to history. |
| A local command fails after the view changes | The latest failure remains in the header; open Ctrl+L and its saved log. |
| A node disappears during a run | Reconnect and refresh Jobs. Do not infer termination from an SSH failure. |
| The screen is too small or a form seems trapped | Resize the terminal; F1 and Ctrl+Q still work. |

For a shareable diagnostic bundle, use `gpuwm report RUN_DIR --dry-run` in a terminal to inspect what it would collect, then `gpuwm report RUN_DIR` to write the bundle. Review the resulting archive before sharing it.
