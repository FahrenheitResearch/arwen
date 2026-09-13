# The desktop package launcher, and what the cut tooling should stop doing

`tools/release/launch_arwen.py` is the launcher every ArWen desktop package
ships. One file, two runtimes, two platforms. This page says what it decides,
what a package has to record for it, and which steps in the release drivers
exist only because the file used to serve one package shape.

## What the launcher decides

Read in order, all of it before anything is started:

1. `ARWEN-DESKTOP.json` beside the launcher. A folder holding no readable one
   is refused with the sentence the bootstrap uses, `Extract the complete ArWen
   desktop package`; a document that does not carry schema
   `arwen.desktop-package.v1` and status `READY` or `READY_FOR_ACCEPTANCE` is a
   package that never finished assembly, and the refusal says to use a released
   one. The `platform` has to be a row this launcher holds. The row names the
   controller, the application, the payload those two load, the family of
   computer they run on (`sys.platform`, matched by prefix), the package's own
   runtime, the start script and the profile folder; a new platform is a row in
   `PACKAGE_PLATFORMS` and nothing else.
2. The runtime the package carries, if any: `runtime/python.exe` on Windows,
   `runtime/bin/python3` on Linux. Its presence is what selects the route.
   * **Carried runtime.** That interpreter is the one the package runs on. A
     `--python` is refused, and the refusal names the runtime it would replace
     and the one change that starts the package: the same command without the
     flag. `ARWEN_PYTHON` in the environment is not refused: it is ignored, with
     one line saying so. The engine has to import from inside the package.
     A carried runtime has to be a real interpreter binary inside the package.
     Every `launch_files` row is resolved strictly and required to land under the
     package root, so a Linux `runtime/` built as a virtual environment is refused
     with `A packaged component resolves outside this installation:
     runtime/bin/python3`, its `bin/python3` being a link to the system
     interpreter. Copy a standalone runtime in; do not create a virtual
     environment there.
   * **No carried runtime.** `--python`, else `ARWEN_PYTHON`, else the
     interpreter running the launcher. The engine is verified by running
     `RUNTIME_PROBE` inside that interpreter against `ENGINE-PYTHON.json`:
     `engine_version`, every `python_files` digest, and the installed terminal's
     digest against `tui_sha256`. Its `engine_source_revision` has to equal the
     package manifest's.
3. Every `launch_files` row, by digest, resolved inside the package. A package
   has to record what the launcher starts and what those programs then load: the
   controller, the application, the row's payload (`arwen-weather.exe`,
   `maplibre-native-c.dll` and `vcruntime140.dll` on Windows, `arwen-weather` and
   `libmaplibre-native-c.so` on Linux), and either the carried runtime and its
   `runtime/ARWEN-RUNTIME.json` or `ENGINE-PYTHON.json`. A row that is absent is
   a file nothing verifies, and the package then fails in the interface, on the
   first map or the first forecast, instead of at the door. Both shipped 1.0.3
   packages record every one of these.
4. The profile the launcher's own settings, cache and runs live in:
   `--state-dir`, else `%LOCALAPPDATA%\ArWenDesktop\v1` on Windows
   and `$XDG_STATE_HOME/ArWenDesktop/v1` (`~/.local/state`) on Linux.
5. The folders chosen in the application, out of
   `<state>/appdata/ArWenCompanion/preferences.json`: `data_folder` gives the
   cache, `forecast_output_folder` gives `--output`, and `saved_run_folders`
   plus the profile's own `runs` folder are passed as `--saved-runs`, once each
   and never the output folder itself. A chosen folder that cannot be created
   is named with the reason and the way out, and the profile's own folder holds.

Every refusal above fires before the controller starts. The controller then
starts in the directory the caller stands in: it reads its own working directory
once and starts every engine job, node operation and file browse there, and the
desktop bootstrap starts the same controller for the same package in the package
folder, which is where the caller of the package's own start script stands. The
isolated graphical preview instead starts its children in its separate data
root, keeping relative writes away from the immutable executable package. Its
Windows bootstrap owns that working directory and environment directly; it
uses this Python launcher only for package verification. Both owners preserve
the terminal preferences described below. The
launcher names no directory of its own for it. The children keep the `TERM` and
`COLORTERM` the caller's terminal gives them and are handed `xterm-256color` and
`truecolor` only when it gives none, so a 256-colour terminal is not told it is
truecolor and a package started from the application still has a terminal the
controller can draw in. `NO_COLOR` travels untouched, empty value included: it is
an instruction from the user rather than a description of the window, and the
controller implements it with a monochrome theme, so a launcher that removed it
made a package print colour the user had asked it not to. After that the
launcher is the process that outlives the controller: it saves the terminal's
attributes, restores them and the alternate screen after a kill, and prints one
line naming the signal and the controller log.

`--verify-launcher` runs all of it and prints one JSON line instead of starting
anything: `status`, `platform`, `runtime_source` (`bundled` or `installed`),
`source_revision`, `engine_version`, `runtime` (what the selected interpreter
holds), `python`, `command`, `state_directory`, `cache_directory`,
`output_directory`, `cds_credentials_configured`, `bytecode_writes_disabled`,
`checked_components`, `application_opened`, `forecast_started`. The Windows
desktop bootstrap reads `status` from the last line of that output and keeps the
whole record in its `verification.json`.

Help text names the engine version out of the package manifest. No version
string is written into this file.

## What the cut tooling should stop doing

The release drivers live outside this repository, so this is the recommendation
they should be changed to follow at the next cut.

* **Stop lifting hunks into a copied-forward Linux launcher.**
  `linux-desktop/prepare_stage_273.py` carries `apply_terminal_restore()` and
  `apply_output_folders()`, which cut the terminal-restore block and the folder
  helpers out of this file by string index and splice them into the previous
  release's staged launcher, plus `IMPORTS_OLD`, `OLD_CACHE`, `OUTPUT_OLD`,
  `TAIL_OLD` and their matching `verify_*` assertions. All of it exists because
  the committed file was Windows-and-bundled-only. Delete both appliers, both
  verifiers and the six baseline literals, and copy
  `tools/release/launch_arwen.py` into the stage unchanged.
* **Stop carrying a hand-maintained Windows launcher.** The
  `launch_arwen-carried-forward.py` kept beside the Windows desktop drivers is
  this file's unbundled route. The Windows packaging step should copy the
  committed file the same way the Linux stage does.
* **Verify by digest, not by marker.** What replaces `verify_terminal_restore`
  and `verify_output_folders` is one assertion: the launcher staged into each
  package is byte-identical to `tools/release/launch_arwen.py` at the released
  revision. Record that digest in the stage's qualification document. A marker
  list cannot tell a lifted copy from a stale one; a digest can.
* **Ship the start scripts from here too.** `Start ArWen.cmd`,
  `Start ArWen Terminal.cmd`, `Start ArWen.sh` and `Start ArWen Terminal.sh` are
  beside the launcher in `tools/release/`. Each prefers the runtime the package
  carries and otherwise passes the caller's `--python` or `ARWEN_PYTHON`
  through, so one pair per platform serves both package shapes. With no runtime,
  no `ARWEN_PYTHON` and no Python on the computer, each one prints a sentence
  naming ArWen and the way out rather than the shell's own message about a
  missing command. The two `.sh` scripts are committed executable, so the stage
  should carry each file's mode out of the tree instead of forcing `0755` for the
  names it happens to know; a script the archive marks unreadable to the shell is
  a package the user cannot start.
* **Keep the package manifest complete.** The launcher requires the payload rows
  of its platform table, so a driver that renames or moves the weather program or
  the map library has to move the row with it. Asserting the same set where the
  manifest is built turns a missing payload into a build defect instead of a
  refusal the user meets.
