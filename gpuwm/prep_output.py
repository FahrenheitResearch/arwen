"""Human preparation progress, with complete diagnostics beside the bundle."""

from __future__ import annotations

import contextlib
import io
import os
from pathlib import Path
import shlex
import sys
import tempfile
import threading
import time

from gpuwm.explain import explain_enabled, split
from gpuwm.command_output import AdapterOutputError, DiagnosticLog, text_chunks

HEARTBEAT_SECONDS = 20.0
TAIL_SIZE = 32768


def failure_summary(diagnostic):
    """Keep a bounded refusal paragraph, omitting Python stack frames."""
    import re

    lines = split(diagnostic)[0].rstrip().splitlines()
    # A traceback's final exception can itself have several lines. Keep
    # all of that message, including its cause and next action.
    tracebacks = [i for i, line in enumerate(lines)
                  if line.startswith("Traceback (most recent call last):")]
    if tracebacks:
        start = tracebacks[-1] + 1
        for index in range(start, len(lines)):
            if re.match(r"^[\w.]+(?::|$)", lines[index]):
                lines = lines[index:]
                break
        else:
            lines = []
    return "\n".join(lines[-8:])


def shell_command(words):
    """Display argv for PowerShell on Windows and a POSIX shell elsewhere."""
    if os.name == "nt":
        # A relative name beginning with # or @ is a comment/splat in
        # PowerShell, even when it contains no whitespace.
        def quote(word):
            word = str(word)
            if word and all(c.isalnum() or c in "_./\\:=-" for c in word):
                return word
            return "'" + word.replace("'", "''") + "'"
        quoted = [quote(word) for word in words]
        prefix = "& " if quoted and quoted[0].startswith("'") else ""
        return prefix + " ".join(quoted)
    return shlex.join(map(str, words))


def forecast_command(args):
    """Use the existing schema-aware sim boundary; never ask users for hashes."""
    from gpuwm import stage_cli

    root = Path(args.output_root)
    bundle = stage_cli.resolve_bundle(root)
    config = root / stage_cli.PREPARED_EXPERIMENT_CONFIG
    if not config.is_file():
        config = getattr(args, "experiment_config", None)
    if config is None or not Path(config).is_file():
        raise stage_cli.StageRefusal("The prepared experiment TOML is missing.")
    wps = root / stage_cli.PREPARED_WPS_NAMELIST
    if not wps.is_file():
        wps = getattr(args, "wps_namelist", None)
    outdir = root.with_name(root.name + "-forecast")
    stage_cli.sim_command(bundle, experiment_config=Path(config),
                          wps_namelist=wps, outdir=outdir)
    words = ["gpuwm", "sim", str(root), "--experiment-config", str(config)]
    if bundle["layout"] == "single":
        words.extend(("--wps-namelist", str(wps)))
    words.extend(("--outdir", str(outdir)))
    return shell_command(words)


def run_preparation(args, launch):
    """Keep log failures distinct from preparer execution and argv retries."""
    try:
        return _run_preparation(args, launch)
    except (AdapterOutputError, BrokenPipeError) as error:
        try:
            print(f"prep: diagnostic output failed: {error}", file=sys.stderr)
        except (OSError, ValueError):
            pass  # A closed error pipe cannot carry its own refusal.
        return 74


def _run_preparation(args, launch):
    """Wrap only a validated, executing preparation; preserve its exit code."""
    from gpuwm import source_cli, stage_cli

    root = Path(args.output_root)
    terminal, errors = sys.stdout, sys.stderr
    try:
        root.parent.mkdir(parents=True, exist_ok=True)
        descriptor, name = tempfile.mkstemp(prefix=root.name[:48] + "-prep-", suffix=".log",
                                           dir=root.parent)
    except OSError as error:
        print(f"prep: cannot write a log beside {root}: {error}. "
              "Choose a writable --output-root.", file=errors)
        return 73
    log_path = Path(name)
    explain = explain_enabled(args)
    tails = {"stdout": "", "stderr": ""}
    lock = threading.RLock()
    started = time.monotonic()
    finished = threading.Event()
    from gpuwm.prep_progress import PrepProgress
    progress = PrepProgress()

    class Output(io.TextIOBase):
        def __init__(self, destination, channel, log):
            self.destination, self.channel, self.log = destination, channel, log
            self.pending = ""

        def write(self, text):
            with lock:
                for chunk in text_chunks(text):
                    self.log.write(chunk)
                    tails[self.channel] = (tails[self.channel] + chunk)[-TAIL_SIZE:]
                    if explain:
                        self.destination.write(chunk)
                    else:
                        self.pending += chunk
                        while "\n" in self.pending:
                            line, self.pending = self.pending.split("\n", 1)
                            message = progress.line(line)
                            if message is not None:
                                print(f"prep: {message}", file=terminal, flush=True)
                            elif line.lstrip().lower().startswith(("warning:", "note:")):
                                print(split(line.strip())[0], file=errors, flush=True)
                        self.pending = self.pending[-TAIL_SIZE:]
            return len(text)

        def flush(self):
            with lock:
                if not self.log.closed:
                    self.log.flush()
                if not self.destination.closed:
                    self.destination.flush()

    def heartbeat():
        while not finished.wait(HEARTBEAT_SECONDS):
            print(f"prep: {progress.label} ({time.monotonic() - started:.0f} s total)",
                  file=terminal, flush=True)

    print(f"prep: preparing {root}\nDetails: {log_path}", file=terminal, flush=True)
    with DiagnosticLog(os.fdopen(descriptor, "w", encoding="utf-8", buffering=1)) as log:
        worker = threading.Thread(target=heartbeat, name="prep-progress", daemon=True)
        worker.start()
        try:
            with contextlib.redirect_stdout(Output(terminal, "stdout", log)), \
                    contextlib.redirect_stderr(Output(errors, "stderr", log)), \
                    source_cli.redirect_adapter_output(sys.stdout, sys.stderr):
                code = launch()
                if log.failure is not None:
                    code = 74
        finally:
            finished.set()
            worker.join()
        if code:
            diagnostic = tails["stderr"] or tails["stdout"]
            summary = failure_summary(diagnostic)
            if not explain and summary:
                print(summary, file=errors)
            print(f"prep: failed (exit {code}). Details: {log_path}", file=errors)
            return code
        print(f"prep: complete ({time.monotonic() - started:.1f} s).", file=terminal)
        try:
            command = forecast_command(args)
        except (stage_cli.StageRefusal, OSError, ValueError) as error:
            log.write(f"Forecast handoff: {error}\n")
            import textwrap
            reason = textwrap.shorten(" ".join(split(str(error))[0].split()), width=500)
            print(f"prep: cannot start the forecast: {reason}\nDetails: {log_path}",
                  file=errors)
        else:
            print(f"Run the forecast:\n  {command}", file=terminal)
    return code
