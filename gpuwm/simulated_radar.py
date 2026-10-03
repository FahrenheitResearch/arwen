"""Simulated radar: the ``gpuwm simulated-radar`` door and live history orchestration.

The native Rust bridge reads history, samples beams, writes radar formats and
renders PPI images. This module passes paths and options only.
"""
from __future__ import annotations

import argparse
import functools
import json
from pathlib import Path
import queue
import subprocess
import sys
import threading

from gpuwm.simulated_radar_config import (
    FIELDS, FORMATS, MANIFEST_SCHEMA, OFF, REFUSAL_LABEL, REQUEST_SCHEMA,
    SCAN_STRATEGIES, RadarSite, SimulatedRadarOptions, declared_key_rows,
    door_refusals, load_options, require_admitted, scene_shapes,
    validate_history_selection,
)

@functools.lru_cache(maxsize=8)
def _site_volume_bytes(config_json: str) -> int:
    from gpuwm import bridges, rustwx

    # Pricing has no side effects: a stale staged artifact is reported by
    # the admission, never re-fetched by a disk estimate.
    with bridges.inspection_only():
        estimate = rustwx.estimate_simulated_radar(
            (), outdir=Path.cwd(), config=json.loads(config_json))
    return int(estimate["output"]["output_bytes_per_site_volume_upper_bound"])


def output_projection(experiment, frames_by_grid: dict) -> dict:
    """Disk bytes [simulated_radar] writes: one volume per site per frame.

    ``frames_by_grid`` is each grid's committed history frame count, the
    frames the radar scans. The per-site-volume figure is the native
    writer bound (``rw_simradar --estimate``). Automatic sites are chosen
    from each grid's coverage when its first history lands, so their count
    is not known here and the radar is named as unpriced instead of
    guessed.
    """
    options = experiment.simulated_radar
    if not options.enabled:
        return {"bytes": 0}
    if isinstance(options.sites, str):
        return {"bytes": None, "unpriced": (
            "simulated radar (sites = \"auto\" are chosen from domain coverage "
            "when the first history lands; list the sites to price their volumes "
            "before the run)")}
    try:
        per_volume = _site_volume_bytes(json.dumps(options.to_mapping(), sort_keys=True))
    except Exception as error:  # noqa: BLE001 - the admission names the cause
        first = str(error).splitlines()[0] if str(error) else type(error).__name__
        return {"bytes": None, "unpriced": f"simulated radar ({first})"}
    sites = len(options.sites)
    volumes = sites * sum(int(frames) for frames in frames_by_grid.values())
    return {"bytes": per_volume * volumes, "per_site_volume_bytes": per_volume,
            "sites": sites, "volumes": volumes,
            "basis": "native writer upper bound per site-volume (rw_simradar --estimate)"}


def run_with_radar(run, *args, radar_options, radar_outdir, progress_callback=None, **kwargs):
    """Give a synchronous history runner the same asynchronous radar path."""
    if not radar_options.enabled:
        return run(*args, progress_callback=progress_callback, **kwargs)
    with LiveSimulatedRadar(radar_options, radar_outdir) as radar:
        return run(*args, progress_callback=RadarProgress(progress_callback, radar), **kwargs)


class LiveSimulatedRadar:
    """Serialize radar work beside forecasting without retaining field arrays.

    A bounded queue holds file paths only. Native failures are stored because
    history landing observers are best effort; drain/close raise them on the
    forecast thread so a requested radar product cannot silently disappear.

    With ``timing = "scan"`` a history waits for its successor on the same
    grid, and the pair publishes the earlier one's volume once
    (``volume_paths``); the last history of each grid publishes at
    :meth:`close` with no successor. A forecast that stops early calls
    :meth:`cancel`, which drops queued histories, terminates the running
    ``rw_simradar`` and names on stderr every history left without a volume.

    ``runner`` is called as ``runner(paths, *, outdir, config, volume_paths,
    started)``; the default resolves and admits ``rw_simradar`` here, so a
    missing or stale binary refuses before the forecast's first step.
    """

    def __init__(self, options, outdir, *, runner=None):
        self.options = options
        self.outdir = Path(outdir)
        if runner is None:
            from gpuwm.rustwx import require_simulated_radar_binary, simulate_radar
            runner = functools.partial(simulate_radar, binary=require_simulated_radar_binary())
        self._runner = runner
        self._queue = queue.Queue(maxsize=64)
        self._failure = None
        self._closed = False
        self._cancelled = threading.Event()
        self._lock = threading.Lock()
        self._process = None
        self._current = None
        self._held = {}
        self._unfinished = []
        self._reported = False
        self._thread = threading.Thread(target=self._work, name="simulated-radar", daemon=True)
        self._thread.start()

    def output_committed(self, *, domain, valid_time, path):
        if self._cancelled.is_set():
            # A frame the stopping forecast still committed: named, not scanned.
            self._unfinished.append((int(domain), str(Path(path).resolve())))
            return
        self.check()
        if self._closed:
            raise RuntimeError("simulated radar received history after close")
        self._queue.put((int(domain), str(Path(path).resolve())))

    def _started(self, process):
        with self._lock:
            self._process = process
            stop = self._cancelled.is_set()
        if stop:
            process.terminate()

    def _run(self, paths, volume_paths):
        try:
            self._runner(paths, outdir=self.outdir, config=self.options.to_mapping(),
                         volume_paths=volume_paths, started=self._started)
        finally:
            with self._lock:
                self._process = None

    def _scan(self, item):
        domain, path = item
        if self.options.timing != "scan":
            self._run([path], None)
            return
        previous = self._held.get(domain)
        self._held[domain] = path
        if previous is not None:
            # The successor bounds the earlier scan: publish it once, now.
            self._run([previous, path], [previous])

    def _work(self):
        while True:
            item = self._queue.get()
            try:
                if item is None:
                    if not self._cancelled.is_set() and self._failure is None:
                        # The last history of each grid has no successor.
                        for domain in sorted(self._held):
                            if self._cancelled.is_set():
                                break
                            self._current = (domain, self._held.pop(domain))
                            self._run([self._current[1]], [self._current[1]])
                            self._current = None
                    return
                if self._cancelled.is_set():
                    self._unfinished.append(item)
                    continue
                if self._failure is not None:
                    continue
                self._current = item
                self._scan(item)
                self._current = None
            except BaseException as error:
                if self._cancelled.is_set():
                    if self._current is not None:
                        self._unfinished.append(self._current)
                elif self._failure is None:
                    self._failure = error
                self._current = None
                if item is None:
                    return
            finally:
                self._queue.task_done()

    def check(self):
        if self._failure is not None:
            raise RuntimeError(f"simulated radar failed: {self._failure}") from self._failure

    def drain(self):
        self._queue.join()
        self.check()

    def close(self):
        """Finish every queued scan and raise a stored radar failure.

        After :meth:`cancel` it scans nothing and names, on stderr, every
        committed history left without a volume, including frames the
        stopping forecast committed after the cancel.
        """
        if self._cancelled.is_set():
            self._report_unfinished()
            return
        if not self._closed:
            self._closed = True
            self._queue.put(None)
            try:
                self._thread.join()
            except BaseException:
                # Ctrl-C while the last scans run stops them too.
                self.cancel()
                self._report_unfinished()
                raise
        self.check()

    def cancel(self):
        """Stop radar for a forecast that is stopping.

        Queued histories are not scanned and the running ``rw_simradar`` is
        terminated (killed if it has not exited 10 s later), so neither a
        failed forecast nor Ctrl-C waits for radar work nobody will use and
        no native process outlives the run. :meth:`close` then names what
        was left.
        """
        if self._cancelled.is_set():
            return
        self._closed = True
        self._cancelled.set()
        with self._lock:
            process = self._process
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        while True:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                break
            if item is not None:
                self._unfinished.append(item)
            self._queue.task_done()
        self._queue.put(None)
        self._thread.join(timeout=30)

    def unfinished(self) -> list[tuple[int, str]]:
        """Committed histories a cancelled run left without a radar volume."""
        seen, result = set(), []
        for item in [*self._unfinished, *sorted(self._held.items())]:
            if item not in seen:
                seen.add(item)
                result.append(item)
        return result

    def _report_unfinished(self):
        left = self.unfinished()
        if not left or self._reported:
            return
        self._reported = True
        names = ", ".join(f"d{domain:02d} {Path(path).name}" for domain, path in left)
        print(f"simulated radar: the forecast stopped, so {len(left)} committed "
              f"history file(s) have no radar volume: {names}. "
              f"{self.outdir / 'radar' / 'manifest.json'} lists the volumes that "
              "were completed; `gpuwm simulated-radar HISTORY ... --config CONFIG.toml "
              f"--outdir {self.outdir}` scans the rest.", file=sys.stderr)

    def __enter__(self):
        return self

    def __exit__(self, kind, error, traceback):
        if kind is not None:
            self.cancel()
        self.close()
        return False


class RadarProgress:
    """Preserve existing progress hooks and add the radar landing consumer."""

    def __init__(self, observer, radar):
        self.observer = observer
        self.radar = radar

    def __call__(self, *args, **kwargs):
        self.radar.check()
        if self.observer is not None:
            return self.observer(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.observer, name)

    def output_committed(self, **kwargs):
        self.radar.output_committed(**kwargs)
        callback = getattr(self.observer, "output_committed", None)
        if callback is not None:
            callback(**kwargs)


def register_cli(subparsers):
    parser = subparsers.add_parser("simulated-radar", help="make radar volumes and PPI images from saved forecast history")
    parser.add_argument("history", nargs="*", type=Path, help="history files or run directories")
    parser.add_argument("--config", type=Path, help="TOML with a [simulated_radar] table")
    parser.add_argument("--outdir", type=Path, default=Path("."), help="run root receiving radar/manifest.json")
    parser.add_argument("--sites", help="auto or comma-separated radar IDs")
    parser.add_argument("--formats", help="comma-separated output formats")
    parser.add_argument("--timing", choices=("history", "scan"),
                        help="history scans each saved snapshot; scan lets rays use neighboring "
                             "history times (overrides the --config table; default history)")
    parser.add_argument("--input-kind", choices=("wrf", "native-columns"), default="wrf",
                        help="full WRF histories or native-atmosphere.columns/v1 transports")
    parser.add_argument("--describe", action="store_true", help="inspect installed native capabilities, accepted inputs and routes as JSON")
    parser.add_argument("--estimate", action="store_true",
                        help="report native scan dimensions and memory admission without generating radar")
    parser.set_defaults(func=main)


def main(args):
    if args.describe:
        from gpuwm.simulated_radar_capabilities import describe
        print(json.dumps(describe(), indent=2))
        return 0
    options = load_options(args.config).to_mapping() if args.config else {"enabled": True}
    options["enabled"] = True
    for key in ("sites", "formats"):
        value = getattr(args, key)
        if value is not None:
            options[key] = "auto" if key == "sites" and value == "auto" else value.split(",")
    if args.timing:
        options["timing"] = args.timing
    config = SimulatedRadarOptions.from_mapping(options)
    paths = []
    for path in args.history:
        if path.is_dir():
            if args.input_kind == "native-columns":
                paths.extend(sorted(path.glob("*.nc")))
            else:
                from gpuwm.io.wrfout import iter_wrfout_files
                paths.extend(iter_wrfout_files(path, include_temporaries=False))
        elif path.is_file():
            paths.append(path)
        else:
            raise ValueError(f"history input does not exist: {path}")
    paths = list(dict.fromkeys(p.resolve() for p in paths))
    if args.estimate:
        if paths and args.input_kind == "native-columns":
            raise ValueError("--estimate reads full canonical scene headers; native-columns must first "
                             "be converted with the shared adapter, or omit histories for geometry-only estimation")
        from gpuwm.rustwx import estimate_simulated_radar
        print(json.dumps(estimate_simulated_radar(paths, outdir=args.outdir,
                                                  config=config.to_mapping()), indent=2))
        return 0
    if not paths:
        raise ValueError("simulated-radar needs at least one committed history file")
    if args.input_kind == "native-columns":
        import hashlib
        from gpuwm.rustwx import canonical_radar_scene
        # Distinct source directories may use the same filename. Isolate each
        # conversion so a later source cannot overwrite an earlier scene.
        paths = [canonical_radar_scene(
            path, outdir=args.outdir / "radar" / "native-scenes" /
            hashlib.sha256(str(path).encode("utf-8")).hexdigest()[:20]) for path in paths]
    from gpuwm.rustwx import simulate_radar
    result = simulate_radar(paths, outdir=args.outdir, config=config.to_mapping())
    print(json.dumps(result, indent=2))
    return 0
