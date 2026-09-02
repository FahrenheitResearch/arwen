"""The engine tool tier: every door published gpuwm owns, as MCP tools.

Registered on every server start.  Each tool is a thin shell over the
real CLI (:mod:`gpuwm.mcp.doors`) or a reader of receipts a door
already wrote; the split between synchronous tools and jobs follows
wall clock, not importance -- a door that answers in seconds is called
inline, anything that can run for minutes (fetch, prep, forecast,
render) launches through :mod:`gpuwm.mcp.jobs` and returns a job id.

Nothing here deletes anything, and no tool invents an answer a door
did not give: refusals are the door's own sentences, verbatim.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from gpuwm.mcp.doors import (ArwenRefusal, TREE_ROOT, door_json, run_door)
from gpuwm.mcp.jobs import MAX_EVENT_BYTES, JobManager


def _clean_args(extra_args: list[str] | None) -> list[str]:
    return [str(a) for a in (extra_args or [])]


def register(server: Any, manager: JobManager) -> list[str]:
    """Register the engine tier on ``server``; returns the tool names."""

    names: list[str] = []

    def tool(fn):
        names.append(fn.__name__)
        server.tool(name=fn.__name__)(fn)
        return fn

    # -- synchronous doors ----------------------------------------------

    @tool
    def arwen_doctor(source: str | None = None,
                     timeout_s: float = 600.0) -> dict[str, Any]:
        """Probe this install's real estate (gpuwm doctor --json).

        Runs the actual diagnostics -- subprocess imports, bridge
        executions, table hashes -- and returns the checks with the
        door's own exit code (0 healthy, 1 when something blocking is
        broken).  Pass `source` (e.g. "gfs") to also explain one data
        route's resolution.
        """

        args = ["doctor", "--json"]
        if source:
            args += ["--source", source]
        checks, exit_code = door_json(args, timeout_s=timeout_s)
        return {"exit_code": exit_code, "healthy": exit_code == 0,
                "checks": checks}

    @tool
    def arwen_sources(id: str | None = None,
                      timeout_s: float = 120.0) -> dict[str, Any]:
        """The source registry (gpuwm run-plan --sources), or one row.

        With no `id`: every registered source, what each row declares,
        and which run-plan route can drive it.  With `id` (a registry id
        or alias, e.g. "hrrr"): that one row in full (gpuwm sources ID
        --json).
        """

        if id:
            document, exit_code = door_json(["sources", id, "--json"],
                                            timeout_s=timeout_s)
        else:
            document, exit_code = door_json(["run-plan", "--sources"],
                                            timeout_s=timeout_s)
        return {"exit_code": exit_code, "registry": document}

    @tool
    def arwen_capabilities(section: str = "both",
                           timeout_s: float = 300.0) -> dict[str, Any]:
        """What this install can render and which physics it can prepare.

        `section` is "catalog" (the renderer's product catalog, gpuwm
        run-plan --catalog), "physics-profiles" (every source crossed
        with every shipped physics suite and why refused pairings
        refuse, gpuwm run-plan --physics-profiles), or "both".
        """

        if section not in ("catalog", "physics-profiles", "both"):
            raise ArwenRefusal(
                f"section {section!r} is not one of 'catalog', "
                "'physics-profiles', 'both', so there is no door to ask.")
        out: dict[str, Any] = {}
        if section in ("catalog", "both"):
            out["catalog"], _ = door_json(["run-plan", "--catalog"],
                                          timeout_s=timeout_s)
        if section in ("physics-profiles", "both"):
            out["physics_profiles"], _ = door_json(
                ["run-plan", "--physics-profiles"], timeout_s=timeout_s)
        return out

    @tool
    def arwen_plan_domain(out_toml: str,
                          point: str | None = None,
                          polygon_geojson: str | None = None,
                          card: str | None = None,
                          vram_gib: float | None = None,
                          hours: int | None = None,
                          source: str | None = None,
                          cycle: str | None = None,
                          ladder: str | None = None,
                          physics_profile: str | None = None,
                          name: str | None = None,
                          acks: list[str] | None = None,
                          extra_args: list[str] | None = None,
                          timeout_s: float = 600.0) -> dict[str, Any]:
        """Size a domain and write the experiment TOML (gpuwm domain).

        The domain wizard's non-interactive form: give it a `point`
        ("LAT,LON") or a `polygon_geojson` path plus a GPU budget
        (`card` like "12gb"/"32gb", or `vram_gib`), and it fits the
        domain ladder with the real VRAM estimator and emits the config
        at `out_toml`.  With no local card visible you MUST pass `card`
        or `vram_gib` -- the wizard refuses otherwise, and that refusal
        is returned verbatim.  `cycle` is "YYYY-MM-DDTHH" or "latest"
        (needs network).
        """

        args = ["domain", "--out", out_toml]
        if point:
            args += [f"--point={point}"]
        if polygon_geojson:
            args += ["--polygon", polygon_geojson]
        if card:
            args += ["--card", card]
        if vram_gib is not None:
            args += ["--vram-gib", str(vram_gib)]
        if hours is not None:
            args += ["--hours", str(hours)]
        if source:
            args += ["--source", source]
        if cycle:
            args += ["--cycle", cycle]
        if ladder:
            args += ["--ladder", ladder]
        if physics_profile:
            args += ["--physics-profile", physics_profile]
        if name:
            args += ["--name", name]
        for ack in acks or []:
            args += ["--ack", ack]
        args += _clean_args(extra_args)
        proc = run_door(args, timeout_s=timeout_s)
        emitted = Path(out_toml)
        if not emitted.is_absolute():
            emitted = TREE_ROOT / emitted
        return {"exit_code": proc.returncode,
                "config_path": str(emitted),
                "config_written": emitted.is_file(),
                "summary": proc.stdout.splitlines()}

    @tool
    def arwen_estimate(config: str,
                       budget_gib: float | None = None,
                       vram_gib: float | None = None,
                       column_chunk: int | None = None,
                       timeout_s: float = 300.0) -> dict[str, Any]:
        """Price a config's VRAM before anything runs (gpuwm check --json).

        `config` is an experiment TOML path.  With no card visible,
        pass `budget_gib` (free VRAM minus reserve) and `vram_gib` (the
        card's physical total) to size for a card that is not in this
        machine.  Returns the estimator's own JSON report.
        """

        args = ["check", config, "--json"]
        if budget_gib is not None:
            args += ["--budget-gib", str(budget_gib)]
        if vram_gib is not None:
            args += ["--vram-gib", str(vram_gib)]
        if column_chunk is not None:
            args += ["--column-chunk", str(column_chunk)]
        report, exit_code = door_json(args, timeout_s=timeout_s)
        return {"exit_code": exit_code, "fits": exit_code == 0,
                "report": report}

    @tool
    def arwen_list_products(wrfout: str, engine: str | None = None,
                            timeout_s: float = 300.0) -> dict[str, Any]:
        """What is renderable from one wrfout, and why or why not.

        Runs `gpuwm render --list-products WRFOUT`: the engine's product
        catalog with per-file availability.  `engine` is "auto" (the
        default), "rust", or "matplotlib".
        """

        args = ["render", wrfout, "--list-products"]
        if engine:
            args += ["--engine", engine]
        proc = run_door(args, timeout_s=timeout_s)
        return {"exit_code": proc.returncode,
                "listing": proc.stdout.splitlines()}

    @tool
    def arwen_run_summary(run_dir: str) -> dict[str, Any]:
        """A run directory's receipts and progress, read into one JSON.

        Reads what the doors already wrote -- report.json (the run's
        own validity verdict; its `status` field is the answer to "is
        this run usable"), the preparation's proof.json/receipt.json,
        events.jsonl / progress.jsonl counts and last records, and the
        output inventory (wrfout frames, PNG count).  Reads only; runs
        nothing.
        """

        root = Path(run_dir)
        if not root.is_dir():
            raise ArwenRefusal(
                f"{run_dir} is not a directory, so there are no receipts "
                "to read; pass a run folder (a `gpuwm go` --outdir, a "
                "run-.../ stamp folder, or a `gpuwm sim` --outdir).")

        def _first(*names: str) -> tuple[str, dict] | None:
            for name in names:
                for candidate in ([root / name]
                                  + sorted(root.rglob(name))[:4]):
                    if candidate.is_file():
                        try:
                            return str(candidate), json.loads(
                                candidate.read_text(encoding="utf-8"))
                        except (OSError, json.JSONDecodeError):
                            continue
            return None

        def _jsonl_tail(name: str) -> dict | None:
            found = sorted(root.rglob(name))
            if not found:
                return None
            path = found[-1]
            try:
                lines = [ln for ln in path.read_text(
                    encoding="utf-8", errors="replace").splitlines() if ln]
            except OSError:
                return None
            last = None
            for candidate in reversed(lines):
                try:
                    last = json.loads(candidate)
                    break
                except json.JSONDecodeError:
                    continue
            return {"path": str(path), "records": len(lines),
                    "last": last}

        report = _first("report.json")
        proof = _first("proof.json", "receipt.json",
                       "public-wrapper-result.json")
        wrfout = sorted(str(p) for p in root.rglob("wrfout_d*"))
        pngs = sum(1 for _ in root.rglob("*.png"))
        summary: dict[str, Any] = {
            "run_dir": str(root),
            "report": {"path": report[0], "body": report[1]}
            if report else None,
            "status": (report[1].get("status")
                       if report else None),
            "preparation_document": {"path": proof[0],
                                     "schema": proof[1].get("schema")}
            if proof else None,
            "events": _jsonl_tail("events.jsonl"),
            "progress": _jsonl_tail("progress.jsonl"),
            "wrfout_count": len(wrfout),
            "wrfout_files": wrfout[:24],
            "png_count": pngs,
        }
        return summary

    # -- jobs: the launch doors -----------------------------------------

    @tool
    def arwen_fetch(source: str, out_dir: str,
                    cycle: str | None = None,
                    hours: int | None = None,
                    area: str | None = None,
                    point: str | None = None,
                    radius_km: float | None = None,
                    forecast_start_hour: int | None = None,
                    extra_args: list[str] | None = None) -> dict[str, Any]:
        """Download one cycle's forcing data as a background job.

        Launches `gpuwm fetch --source ... --out ...` detached and
        returns {job_id}; follow it with job_status / job_events
        (stream "stdout") / job_result.  `area` is
        "LAT0,LON0,LAT1,LON1", or use `point` ("LAT,LON") with
        `radius_km`.  `cycle` is "YYYY-MM-DDTHH" or "latest".
        """

        args = ["fetch", "--source", source, "--out", out_dir]
        if cycle:
            args += ["--cycle", cycle]
        if hours is not None:
            args += ["--hours", str(hours)]
        if area:
            args += [f"--area={area}"]
        if point:
            args += [f"--point={point}"]
        if radius_km is not None:
            args += ["--radius-km", str(radius_km)]
        if forecast_start_hour is not None:
            args += ["--forecast-start-hour", str(forecast_start_hour)]
        args += _clean_args(extra_args)
        return manager.launch(
            "fetch", [*_engine_prefix(), *args], cwd=TREE_ROOT, gpu=False,
            outputs={"outdir": out_dir})

    @tool
    def arwen_prep(output_root: str,
                   source: str | None = None,
                   wps_namelist: str | None = None,
                   namelist_input: str | None = None,
                   experiment_config: str | None = None,
                   geog_root: str | None = None,
                   inputs: list[str] | None = None,
                   valid_time: str | None = None,
                   preprocess_backend: str | None = None,
                   extra_args: list[str] | None = None) -> dict[str, Any]:
        """Preprocess your files into a prepared tree, as a job.

        Launches `gpuwm prep` (the same program as rw-wps) detached and
        returns {job_id}.  It downloads nothing: you supply the source
        adapter, input files, namelists, experiment TOML and geog root.
        The prep surface is wide; anything not surfaced here rides
        `extra_args` verbatim (e.g. ["--gfs-series", "series.tsv",
        "--cycle", "2026-07-29_18:00:00"]).  With `preprocess_backend`
        "cpu" the job takes no GPU lock; any other backend holds the
        one-job-per-card lock until it exits.
        """

        args = ["prep", "--output-root", output_root]
        if source:
            args += ["--source", source]
        if wps_namelist:
            args += ["--wps-namelist", wps_namelist]
        if namelist_input:
            args += ["--namelist-input", namelist_input]
        if experiment_config:
            args += ["--experiment-config", experiment_config]
        if geog_root:
            args += ["--geog-root", geog_root]
        for path in inputs or []:
            args += ["--input", path]
        if valid_time:
            args += ["--valid-time", valid_time]
        if preprocess_backend:
            args += ["--preprocess-backend", preprocess_backend]
        args += _clean_args(extra_args)
        gpu = preprocess_backend != "cpu"
        return manager.launch(
            "prep", [*_engine_prefix(), *args], cwd=TREE_ROOT, gpu=gpu,
            outputs={"outdir": output_root})

    @tool
    def arwen_forecast(config: str,
                       mode: str = "go",
                       outdir: str | None = None,
                       data_dir: str | None = None,
                       products: str | None = None,
                       dry_run: bool = False,
                       extra_args: list[str] | None = None,
                       timeout_s: float = 300.0) -> dict[str, Any]:
        """Run a forecast as a job -- the whole chain, or the model alone.

        Two spellings of the same forecast: mode "go" runs the composed
        chain (authority, fetch, manifest, prep, forecast, render) on a
        wizard-emitted config; mode "run" integrates the config-driven
        case directly into `outdir` with no fetching or rendering.
        Both take the one-GPU-job lock; a second GPU launch is refused
        naming the running job.  `dry_run` (mode "go") answers
        synchronously with the six commands the chain would run and
        launches nothing.  `products` is go's render product list
        ("all", "none", or comma-separated slugs).
        """

        if mode not in ("go", "run"):
            raise ArwenRefusal(
                f"mode {mode!r} is not 'go' or 'run', so there is no door "
                "to launch.")
        if mode == "go":
            args = ["go", config]
            if outdir:
                args += ["--outdir", outdir]
            if data_dir:
                args += ["--data-dir", data_dir]
            if products:
                args += ["--products", products]
            if dry_run:
                args += ["--dry-run"]
        else:
            if dry_run:
                raise ArwenRefusal(
                    "dry_run belongs to mode 'go' (`gpuwm run` has no "
                    "--dry-run), so this call would launch a real "
                    "integration you asked not to run.")
            args = ["run", config]
            if outdir:
                args += ["--outdir", outdir]
        args += _clean_args(extra_args)
        if dry_run:
            proc = run_door(args, timeout_s=timeout_s)
            return {"exit_code": proc.returncode,
                    "commands": proc.stdout.splitlines()}
        declared_out = outdir or ""
        return manager.launch(
            mode, [*_engine_prefix(), *args], cwd=TREE_ROOT, gpu=True,
            outputs={"outdir": declared_out} if declared_out else {})

    @tool
    def arwen_render(out_dir: str,
                     wrfout: list[str] | None = None,
                     products: str | None = None,
                     engine: str | None = None,
                     timeidx: str | None = None,
                     pair: list[str] | None = None,
                     pair_labels: list[str] | None = None,
                     extra_args: list[str] | None = None) -> dict[str, Any]:
        """Render product PNGs from wrfout frames, as a job.

        Launches `gpuwm render` detached and returns {job_id}.  Pass
        `wrfout` file paths (or omit them and pass `pair` -- two
        rendered PNG directories -- to compose side-by-side comparison
        sheets, optionally labeled with `pair_labels`).  `products` is
        a comma-separated list or "all"; `engine` is "auto" (refuses
        rather than degrading weather fields), "rust", or the
        "matplotlib" workaround by name.
        """

        if not wrfout and not pair:
            raise ArwenRefusal(
                "neither wrfout files nor a pair of rendered directories "
                "was given, so there is nothing to draw.")
        args = ["render", *[str(p) for p in (wrfout or [])],
                "--out", out_dir]
        if products:
            args += ["--products", products]
        if engine:
            args += ["--engine", engine]
        if timeidx:
            args += ["--timeidx", str(timeidx)]
        if pair:
            if len(pair) != 2:
                raise ArwenRefusal(
                    f"--pair takes exactly two rendered directories, got "
                    f"{len(pair)}, so there is nothing to compose.")
            args += ["--pair", *pair]
        if pair_labels:
            args += ["--pair-labels", *pair_labels]
        args += _clean_args(extra_args)
        return manager.launch(
            "render", [*_engine_prefix(), *args], cwd=TREE_ROOT, gpu=False,
            outputs={"outdir": out_dir})

    @tool
    def arwen_cells(out_dir: str, wrfout: list[str],
                    profile: str | None = None,
                    ladder: str | None = None,
                    titan: str | None = None,
                    extra_args: list[str] | None = None) -> dict[str, Any]:
        """Storm cells over a wrfout series, as a job.

        Launches `gpuwm cells analyze` detached: the series is exported
        onto a height ladder, the titan storm-cell engine identifies and
        tracks the cells, and the catalog (one row per cell per frame:
        titan's id/track/age/area/echo tops/trend joined to ArWen's peak
        updraft, cloud top/base, freezing and supercooled levels,
        supercooled liquid water) lands under
        `<out_dir>/<domain>/cells/<first-valid-day>/`.  Returns
        {job_id}; follow it with job_status, then read the catalog with
        arwen_cells_catalog(out_dir).  `profile` is a titan threshold
        profile (research, severe, legacy, operational); `ladder` is
        BOTTOM:TOP:STEP metres; `titan` names the binary when the
        resolution ladder does not find one.
        """

        if not wrfout:
            raise ArwenRefusal(
                "no wrfout files were given, so there is no history to "
                "find a cell in.")
        args = ["cells", "analyze", *[str(p) for p in wrfout],
                "--out", out_dir]
        if profile:
            args += ["--profile", profile]
        if ladder:
            args += ["--ladder", ladder]
        if titan:
            args += ["--titan", titan]
        args += _clean_args(extra_args)
        return manager.launch(
            "cells", [*_engine_prefix(), *args], cwd=TREE_ROOT, gpu=False,
            outputs={"outdir": out_dir})

    @tool
    def arwen_cells_catalog(out_dir: str, track_id: int | None = None,
                            max_rows: int = 500) -> dict[str, Any]:
        """The cell catalog a `gpuwm cells analyze` run wrote, as JSON.

        Reads `catalog.json` under `out_dir` (the case folder, or the
        series folder itself) -- the column table with units and
        provenance, the receipt, and the rows, optionally one track's
        and capped at `max_rows` newest-first.  Reads only; runs
        nothing.
        """

        root = Path(out_dir)
        candidates = ([root / "catalog.json"] if (root / "catalog.json").is_file()
                      else sorted(root.rglob("catalog.json")))
        if not candidates:
            raise ArwenRefusal(
                f"no catalog.json under {out_dir}; run arwen_cells on the "
                f"series first (or pass the folder it wrote).")
        document = json.loads(candidates[-1].read_text(encoding="utf-8"))
        rows = document.get("rows", [])
        if track_id is not None:
            rows = [row for row in rows if row.get("track_id") == track_id]
        total = len(rows)
        rows = sorted(rows, key=lambda row: row.get("timestamp_ms", 0),
                      reverse=True)[:max(int(max_rows), 0)]
        for row in rows:
            row.pop("geometry", None)
        return {"catalog": str(candidates[-1]), "schema": document.get("schema"),
                "columns": document.get("columns"),
                "receipt": document.get("receipt"),
                "rows_total": total, "rows": rows}

    # -- jobs: the follow tools -----------------------------------------

    @tool
    def job_status(job_id: str) -> dict[str, Any]:
        """One job's state: running, exited, cancelled, or lost.

        Derived from the on-disk receipt and pid liveness, so it
        answers the same after a server restart.  "lost" means the
        wrapper died without writing a result (machine restart or an
        outright kill); the job directory's logs are the record.
        """

        return manager.status(job_id)

    @tool
    def job_events(job_id: str, stream: str = "stdout",
                   cursor: int = 0,
                   max_bytes: int = MAX_EVENT_BYTES) -> dict[str, Any]:
        """Tail one of a job's streams incrementally from a byte cursor.

        `stream` is "stdout", "stderr", "events" (the run's own
        events.jsonl, once the door creates it), or "progress"
        (progress.jsonl, one record per model step).  Pass the returned
        `next_cursor` back in to read only what is new; `eof` true
        means the job has exited and the stream is fully read.
        """

        return manager.events(job_id, stream=stream, cursor=cursor,
                              max_bytes=max_bytes)

    @tool
    def job_result(job_id: str) -> dict[str, Any]:
        """A finished job's outcome; refuses while it still runs.

        `ok` is exit 0.  Exit 2 carries the door's refusal sentence
        verbatim in `refusal`; other nonzero exits carry a stderr tail.
        """

        return manager.result(job_id)

    @tool
    def job_cancel(job_id: str) -> dict[str, Any]:
        """Stop a running job's process tree; deletes nothing.

        The job directory keeps its receipt and logs, the result is
        recorded as cancelled, and a held GPU lock is released.
        """

        return manager.cancel(job_id)

    @tool
    def job_list() -> dict[str, Any]:
        """Every job under this server's jobs root, with current state."""

        return manager.list()

    return names


def _engine_prefix() -> list[str]:
    """The subprocess argv prefix for a job running this tree's CLI."""

    import sys
    return [sys.executable, "-m", "gpuwm.cli"]
