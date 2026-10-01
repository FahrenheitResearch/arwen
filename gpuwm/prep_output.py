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
# From the render layout, which the standalone preparation wheel stages,
# and at import rather than inside forecast_command: that call runs while
# the preparation's output is still held.
from gpuwm.render_layout import DEFAULT_RENDER_PRODUCTS

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


#: Characters a PowerShell word may carry unquoted.  Anything else is
#: quoted: # starts a comment, @ a splat, $ a variable, a comma builds an
#: array, and ~ is expanded for a native command on some hosts.
_POWERSHELL_BARE = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_./\\:=-")

#: What PowerShell reads as a single quote: the ASCII one and the four
#: typographic ones.  Each is doubled inside a single-quoted string to
#: stand for itself, so a folder named with a curly apostrophe does not
#: end the string early.
_POWERSHELL_SINGLE_QUOTES = "'‘’‚‛"


def host_shell() -> str:
    """The shell a line printed on this machine is pasted into.

    PowerShell on Windows, the shell a Windows terminal opens and the one
    the install guide uses there; a POSIX shell everywhere else.
    """

    return "powershell" if os.name == "nt" else "posix"


def _powershell_word(word) -> str:
    word = str(word)
    # A word PowerShell could read as a number is quoted too: an
    # unquoted 0x10, 1kb or 1.10 reaches a native program as 16, 1024 or
    # 1.1.  "--" alone is the end-of-parameters token, and a dash word
    # holding a colon is a parameter that takes the next word as its
    # value.
    dash = word[:1] == "-"
    if (word and word != "--" and set(word) <= _POWERSHELL_BARE
            and not word[0].isdigit() and word[0] != "."
            and not (dash and (":" in word or word[1:2].isdigit()
                               or word[1:2] == "."))):
        return word
    for quote in _POWERSHELL_SINGLE_QUOTES:
        word = word.replace(quote, quote + quote)
    return "'" + word + "'"


def shell_command(words, *, shell: str | None = None) -> str:
    """Display argv as one line for PowerShell on Windows and a POSIX shell elsewhere.

    ``shell`` names the other one explicitly (``"powershell"`` or
    ``"posix"``); omitted, it is :func:`host_shell`.  In PowerShell a
    quoted program is a string, not a command, so a line whose program
    needs quoting starts with the call operator ``&``.
    """

    shell = host_shell() if shell is None else shell
    if shell == "powershell":
        quoted = [_powershell_word(word) for word in words]
        prefix = "& " if quoted and quoted[0].startswith("'") else ""
        return prefix + " ".join(quoted)
    if shell != "posix":
        raise ValueError(f"no command-line spelling for shell {shell!r}")
    return shlex.join(map(str, words))


def forecast_command(args):
    """Use the existing schema-aware sim boundary; never ask users for hashes.

    The line draws pictures: ``--render-products`` with the same default
    set ``gpuwm go`` draws, so each output frame of every grid is drawn
    as it lands while the forecast runs.  Without it the printed line ran
    the forecast and drew nothing, although both runners draw frames as
    they land when asked.
    """
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
                          wps_namelist=wps, outdir=outdir,
                          render_products=DEFAULT_RENDER_PRODUCTS)
    words = ["gpuwm", "sim", str(root), "--experiment-config", str(config)]
    if bundle["layout"] == "single":
        words.extend(("--wps-namelist", str(wps)))
    words.extend(("--outdir", str(outdir),
                  "--render-products", DEFAULT_RENDER_PRODUCTS))
    return shell_command(words)


def preparation_only_handoff(root, missing):
    """What a finished preparation says where no forecast is installed."""

    return (f"prep: this installation prepares inputs only ({', '.join(missing)} "
            f"not installed), so it prints no forecast line.  The prepared tree "
            f"is {root}; `gpuwm sim` in a full gpuwm installation runs it.")


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
    from gpuwm.prep_progress import PrepProgress, step_record
    from gpuwm.progress import PREP_EVENT_PARENT_ENV, relay_prep_record
    progress = PrepProgress()
    # A parent that reads step records off this program's output (a `gpuwm go`
    # stage, which runs `python -m gpuwm.source_cli` on the GFS chain) is told
    # each step line as it is written.  Kept to the log, the steps never left
    # this process, and the run page of a plain GFS run showed none of them.
    to_parent = os.environ.get(PREP_EVENT_PARENT_ENV) == "1"

    class Output(io.TextIOBase):
        def __init__(self, destination, channel, log):
            self.destination, self.channel, self.log = destination, channel, log
            self.pending = ""

        def write(self, text):
            steps = []
            with lock:
                for chunk in text_chunks(text):
                    self.log.write(chunk)
                    tails[self.channel] = (tails[self.channel] + chunk)[-TAIL_SIZE:]
                    if explain:
                        self.destination.write(chunk)
                    self.pending += chunk
                    while "\n" in self.pending:
                        line, self.pending = self.pending.split("\n", 1)
                        record = step_record(line)
                        if record is not None:
                            steps.append(record)
                            if to_parent and not explain:
                                # Under --explain the line was passed on whole above.
                                print(line, file=errors, flush=True)
                        if explain:
                            continue
                        message = None if record is None else progress.event(record)
                        if message is not None:
                            print(f"prep: {message}", file=terminal, flush=True)
                        elif record is None and line.lstrip().lower().startswith(("warning:", "note:")):
                            print(split(line.strip())[0], file=errors, flush=True)
                    self.pending = self.pending[-TAIL_SIZE:]
            # The preparer is its own program, so its steps reached no
            # listener in this process: a run that hosts this preparation
            # (the staged route of `gpuwm run-plan`) hears each one here
            # and puts it on the run's stream.  Said outside the lock, so a
            # listener that writes cannot wait on this output.
            for record in steps:
                relay_prep_record(record)
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
        # The standalone RW-WPS package carries no forecast runner, and
        # resolving the bundle imports both: every finished preparation
        # there ended in an ImportError after "prep: complete".
        missing = stage_cli.missing_forecast_runners()
        if missing:
            print(preparation_only_handoff(root, missing), file=terminal)
            return code
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
