"""``gpuwm setup``: the staging steps a wheel install still needs.

``pip install gpuwm`` lands the Python half and nothing else.  The wheel
ships no compiled Rust -- every GRIB decode goes through the fail-closed
bridges -- and the two largest Thompson tables are published as release
assets because they exceed PyPI's per-file cap.  Both gaps already had a
one-command fix; what they did not have was ONE command.  A first
install therefore read as a list of chores whose order and necessity the
reader had to infer, which is the half of the install complaint that no
amount of message layering fixes.

So this is a wrapper, deliberately thin.  Each step is
:mod:`gpuwm.bridge_assets` or :mod:`gpuwm.table_assets` doing exactly
what it does when invoked directly -- same verification, same pins, same
refusals, same exit codes -- and this module adds only sequencing, one
line of status per step, and a closing doctor summary.  Nothing new
downloads anything, so there is no second implementation to drift.

**Static geography is opt in.**  ``gpuwm fetch-geog`` unpacks about
16 GB, which is not a thing to start on someone's behalf because they
typed a command called ``setup``.  ``--with-geog`` asks for it, and
prints the size before the first byte moves.

**Re-run safe**, because every step is: staged artifacts that match
their pins are verified and skipped, so a second ``gpuwm setup`` is a
verification pass.  The wrapper adds no state of its own -- no marker
file, no "already done" short circuit -- precisely so that a re-run
after a partial or corrupted first run does the checking rather than
trusting a flag.
"""

from __future__ import annotations

import argparse
import contextlib
import io

from gpuwm.explain import explain_enabled

#: One setup step: its label and the module whose ``register_cli``
#: declares the subcommand this step IS.  Ordered so that a reader who
#: stops after step one still has the half that unblocks every data
#: route: the decoders, then the tables.
_STEPS: tuple[tuple[str, str], ...] = (
    ("bridges", "gpuwm.bridge_assets"),
    ("tables", "gpuwm.table_assets"),
)


def step_namespace(module_name: str) -> argparse.Namespace:
    """The exact defaults ``gpuwm <that subcommand>`` would run with.

    Read out of the real registrar rather than transcribed here.  A
    transcribed copy is a second declaration of every default -- the
    geog mirror choice, the archive retention, the dataset list -- and
    the failure mode is silent: setup would keep running yesterday's
    defaults while the documented command ran today's, and no test that
    exercises either one alone would notice.
    """

    from importlib import import_module

    parser = argparse.ArgumentParser(prog="gpuwm")
    subparsers = parser.add_subparsers(dest="command")
    import_module(module_name).register_cli(subparsers)
    [name] = list(subparsers.choices)
    return parser.parse_args([name])


def _run_step(module_name: str, overrides: dict) -> tuple[int, str]:
    """Run one step's real handler; return its exit code and its output.

    The output is captured rather than streamed so the wrapper can print
    one line per step by default and still hand over every captured word
    when the step fails or when ``--explain`` asks.  Capturing is not
    discarding: a failing step replays its full text, because the reason
    a step refused is the one thing a setup command must never swallow.

    A step that RAISES rather than returning a code is handled here, in
    the one place that holds its captured output.  Letting the exception
    escape would have thrown away everything the step had already
    printed before it died -- which, for a download that got most of the
    way through, is the whole diagnosis.
    """

    namespace = step_namespace(module_name)
    for key, value in overrides.items():
        setattr(namespace, key, value)
    buffer = io.StringIO()
    try:
        with contextlib.redirect_stdout(buffer):
            code = namespace.func(namespace)
    except Exception as error:
        return 2, buffer.getvalue() + f"{type(error).__name__}: {error}\n"
    return code, buffer.getvalue()


def _replay(text: str) -> None:
    """Print a step's captured output, indented under its status line."""

    for line in text.splitlines():
        print(f"    {line}" if line.strip() else "")


#: Printed before ``--with-geog`` starts, because 16 GB is a decision.
#: The numbers are the ones :data:`gpuwm.doctor.GEOG_HINT` already
#: publishes, so a reader cannot be told two different sizes.
GEOG_SIZE_NOTICE = (
    "setup: --with-geog will download the nine WPS_GEOG datasets: "
    "~1.3 GB compressed, ~16 GB unpacked.  Resumable and re-run safe; "
    "omit --with-geog to skip it and run `gpuwm fetch-geog` later.")


def setup_main(args) -> int:
    """Run every staging step, then report the estate.

    Returns the first failing step's exit code, or doctor's own verdict
    when every step succeeded.  "First failing" rather than "last"
    because the earlier step is the one further upstream, and its
    refusal is usually why the later one had nothing to do.
    """

    explain = explain_enabled(args)
    steps = list(_STEPS)
    if getattr(args, "with_geog", False):
        print(GEOG_SIZE_NOTICE)
        steps.append(("geog", "gpuwm.geog_assets"))

    failed = 0
    for label, module_name in steps:
        overrides = {}
        if getattr(args, "from_dir", None) is not None and label != "geog":
            overrides["from_dir"] = args.from_dir
        code, output = _run_step(module_name, overrides)
        if code == 0:
            print(f"  ok      {label}")
            if explain:
                _replay(output)
        else:
            # Never swallow a refusal.  A failed step prints everything
            # it said, whether or not --explain was asked for: a
            # wrapper that hides why a step refused is worse than no
            # wrapper, because the reader now has to work out which
            # underlying command to re-run before they can even read
            # the error.
            print(f"  FAILED  {label} (exit {code})")
            _replay(output)
            failed = failed or code

    # Every step still runs.  They are independent -- staging tables
    # does not need the decoders, and both are idempotent -- so
    # stopping at the first refusal would leave a second, unrelated gap
    # undiscovered until the reader re-ran and hit it next.
    #
    # Close with the estate report either way, because after a partial
    # setup "where do I stand" is exactly the question, and it is the
    # same function `gpuwm doctor` runs rather than a summary of it.
    from gpuwm.doctor import (blocking_gaps, collect_checks, format_brief,
                              format_report)

    checks = collect_checks()
    print(format_report(checks) if explain else format_brief(checks))
    if failed:
        if not blocking_gaps(checks):
            # Field case: a checkout install has no published bundle
            # pins, so fetch-bridges refuses -- while the bridges it
            # would have fetched are already built and verified (the
            # doctor brief above says so).  A step that could not run
            # against an estate that is already complete is a note,
            # not a failure.
            print("setup: a step above refused, but the estate is "
                  "already complete (no blocking gaps); nothing was "
                  "missing.")
            return 0
        print("setup: a step above refused; its own output is quoted "
              "under it.  The remaining steps still ran.")
        return failed
    # Every step succeeded: setup did its whole job.  Only a broken or
    # integrity-suspect finding fails the exit; a documented optional
    # absence (cupy on a base install, the render extra, a WPS_GEOG
    # tree nobody opted into) reports in the brief above and exits 0.
    return 1 if blocking_gaps(checks) else 0


def register_cli(subparsers) -> None:
    parser = subparsers.add_parser(
        "setup",
        help="stage what a pip install still needs: the prebuilt Rust "
             "artifacts (gpuwm fetch-bridges) then the externalized "
             "physics tables (gpuwm fetch-tables), then print the doctor "
             "summary.  Idempotent; static geography is opt-in via "
             "--with-geog")
    parser.add_argument(
        "--with-geog", action="store_true", dest="with_geog",
        help="also stage the WPS_GEOG static geography (~1.3 GB "
             "compressed, ~16 GB unpacked); the size is printed before "
             "anything downloads")
    parser.add_argument(
        "--from", dest="from_dir", metavar="DIR", default=None,
        help="stage the bridge and table artifacts from a local "
             "directory instead of downloading (offline installs); "
             "verification is identical.  Does not apply to --with-geog, "
             "which has its own --source")
    parser.set_defaults(func=setup_main)
    return parser


__all__ = ["GEOG_SIZE_NOTICE", "register_cli", "setup_main"]
