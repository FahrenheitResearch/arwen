"""The daemon's argv builders must satisfy the signatures they call.

Every stage of the continuous nowcast daemon is a subprocess whose
command line is built by a function that lives in ``tools.da_nowcast``.
A keyword that stops matching is not a lint: it is a ``TypeError`` on the
daemon's FIRST observation stage, hours after somebody launched it and
walked away.  That break shipped once -- ``obs_cmd`` grew a required
``dealias`` argument and the daemon's call site did not -- and it
escaped every existing test because the three tests that name ``Daemon``
read its source as TEXT rather than binding the call.

So this file binds.  It reads each call site out of the daemon's own AST
and offers the keyword names to the real :func:`inspect.signature` of the
function being called.  A missing required keyword-only argument, or a
keyword the callee does not accept, fails here rather than in the field.

Pure tests: no network, no GPU, no subprocess execution.  Sites are
synthetic ids -- real station names never enter the tree's generic code
or its fixtures (standing owner rule).
"""
from __future__ import annotations

import ast
import inspect
from datetime import datetime, timezone
from pathlib import Path

import pytest

from tools import da_nowcast_auto
from tools.da_nowcast import (DEFAULT_DEALIAS_ENGINE, DealiasChoice,
                              cycle_cmd, obs_cmd, render_cmd)
from tools.da_nowcast_auto import (bootstrap_cmd, build_parser, loop_argv,
                                   obs_argv_kwargs)

#: The argv builders the daemon calls, by the name it calls them under.
BUILDERS = {"obs_cmd": obs_cmd, "cycle_cmd": cycle_cmd,
            "render_cmd": render_cmd, "bootstrap_cmd": bootstrap_cmd}

SITE = "XTST"


def daemon_args(*extra: str):
    """A parsed ``loop`` namespace, with as few flags as a person needs."""

    return build_parser().parse_args(
        ["loop", "--site", SITE, "--out", "run", *extra])


def call_sites(name: str) -> list[ast.Call]:
    """Every call to ``name`` in the daemon's module, from its own AST."""

    tree = ast.parse(Path(da_nowcast_auto.__file__).read_text(
        encoding="utf-8"))
    found = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == name):
            found.append(node)
    return found


# ---------------------------------------------------------------------------
# the contract: every call site binds
# ---------------------------------------------------------------------------
class TestArgvContract:
    @pytest.mark.parametrize("name", sorted(BUILDERS))
    def test_every_daemon_call_site_binds_to_the_real_signature(self, name):
        signature = inspect.signature(BUILDERS[name])
        sites = call_sites(name)
        assert sites, (
            f"no call to {name} found in the daemon; either the stage was "
            "removed or this contract test is watching the wrong module")
        for site in sites:
            assert not site.args, (
                f"{name} is keyword-only by design; a positional call at "
                f"line {site.lineno} cannot be checked against its "
                "signature")
            keywords = [kw.arg for kw in site.keywords]
            if keywords == [None]:
                # ``f(**builder(...))``: the keywords are not readable
                # from the AST, so a live test binds the builder's own
                # mapping instead.  Only a NAMED builder qualifies -- an
                # arbitrary ** expression would let a call site opt out
                # of this contract entirely.
                spread = site.keywords[0].value
                assert (isinstance(spread, ast.Call)
                        and isinstance(spread.func, ast.Name)
                        and spread.func.id.endswith("_kwargs")), (
                    f"{name} is called with an unnamed ** at line "
                    f"{site.lineno}; the contract can be read neither "
                    "from the call site nor from a kwargs builder")
                continue
            assert None not in keywords, (
                f"{name} mixes ** with explicit keywords at line "
                f"{site.lineno}; the contract cannot be read from the "
                "call site")
            # ``bind`` -- not ``bind_partial`` -- because the whole point
            # is to catch a REQUIRED keyword-only argument the call site
            # never learned about.
            signature.bind(**{key: object() for key in keywords})

    def test_daemon_obs_argv_satisfies_obs_cmd_signature(self):
        """The break that shipped: ``obs_cmd`` wants a dealias choice."""

        def build(*extra: str) -> list[str]:
            kwargs = obs_argv_kwargs(
                site=SITE, valid=datetime(2026, 8, 5, 4, 0,
                                          tzinfo=timezone.utc),
                grid_wrfout=Path("wrfout"), out_nc=Path("obs.nc"),
                work_dir=Path("vols"), args=daemon_args(*extra))
            # Binds, and then actually builds -- a signature that binds
            # but a value the builder rejects is the same outage one
            # frame later.
            inspect.signature(obs_cmd).bind(**kwargs)
            assert isinstance(kwargs["dealias"], DealiasChoice)
            return obs_cmd(**kwargs)

        # Both arms, because "the flag did nothing" and "the flag worked"
        # are otherwise indistinguishable.  Off is the shipped default,
        # matching the front door's own ``run`` with no --dealias.
        assert "--dealias" not in build()
        on = build("--dealias")
        assert on[on.index("--dealias-engine") + 1] == DEFAULT_DEALIAS_ENGINE
        assert "--dealias-refinement" in on

    def test_the_daemon_call_site_is_the_one_that_was_bound(self):
        """The kwargs builder is what the stage uses, not a parallel copy."""

        sites = call_sites("obs_cmd")
        assert len(sites) == 1
        keywords = [kw.arg for kw in sites[0].keywords]
        assert keywords == [None], (
            "the obs stage must build its kwargs through obs_argv_kwargs "
            "so the contract test above binds the SAME mapping the daemon "
            f"passes; observed keywords {keywords!r}")


# ---------------------------------------------------------------------------
# the choice has to survive the daemon re-launching itself
# ---------------------------------------------------------------------------
class TestEpochReexec:
    def test_epoch_reexec_preserves_dealias_choice(self, tmp_path):
        """A half-wired flag degrades the daemon at the first epoch roll.

        The daemon does not fork across epochs; it REBUILDS its own
        command line.  A dealias choice that reached the obs stage but
        not :func:`loop_argv` would unfold velocity until the first roll
        and then quietly stop, with every gallery after it still saying
        the run was dealiased because the first one was.
        """

        args = daemon_args("--dealias", "--dealias-engine", "vad-region",
                           "--no-dealias-refinement")
        chosen = DealiasChoice.from_args(args)
        assert chosen == DealiasChoice(on=True, engine="vad-region",
                                       refinement=False)

        argv = loop_argv(args, tmp_path)
        assert argv[3] == "loop"
        rolled = build_parser().parse_args(argv[3:])
        assert DealiasChoice.from_args(rolled) == chosen

    def test_the_shipped_default_survives_the_roll_too(self, tmp_path):
        args = daemon_args()
        rolled = build_parser().parse_args(loop_argv(args, tmp_path)[3:])
        assert DealiasChoice.from_args(rolled) == DealiasChoice.from_args(
            args)
        assert DealiasChoice.from_args(rolled).engine == \
            DEFAULT_DEALIAS_ENGINE

    def test_bootstrap_argv_carries_the_choice(self, tmp_path):
        """The front door builds the georeference forecast's own obs."""

        args = daemon_args("--dealias", "--dealias-engine", "vad-region",
                           "--no-dealias-refinement")
        argv = bootstrap_cmd(site=SITE, out=tmp_path, args=args)
        assert "--dealias" in argv
        assert argv[argv.index("--dealias-engine") + 1] == "vad-region"
        assert "--no-dealias-refinement" in argv
