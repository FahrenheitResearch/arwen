"""The ArWen MCP server: build, register the tools, serve stdio.

:func:`build_server` assembles a FastMCP server carrying the engine
tool tier (:mod:`gpuwm.mcp.engine_tools` -- the CLI's doors, wrapped).
:func:`main` is the stdio entry point behind ``arwen-mcp`` and
``python -m gpuwm.mcp``.

stdout belongs to the protocol: nothing in this process may print to
it.  Diagnostics go to stderr, and every subprocess a tool starts gets
its own captured pipes or log files.
"""

from __future__ import annotations

import argparse
import sys

from gpuwm.mcp import require_sdk
from gpuwm.mcp.jobs import JobManager

_INSTRUCTIONS = """\
ArWen (gpuwm) as tools: plan, price, fetch, prep, forecast, render.

Every tool shells out to the real `gpuwm` command line or reads a
receipt a door already wrote; refusals are the CLI's own one-sentence
refusals, verbatim, so a refused call tells you exactly what to change.

Long work is a JOB: arwen_fetch / arwen_prep / arwen_forecast /
arwen_render return {job_id} immediately.  Follow with job_status,
tail output with job_events (pass next_cursor back in; streams:
stdout, stderr, events, progress), collect with job_result, stop with
job_cancel, enumerate with job_list.  Jobs survive a server restart:
state lives on disk.

One GPU job runs at a time: a second GPU launch is refused with a
sentence naming the running job id -- wait, or cancel it.

A typical forecast: arwen_plan_domain (point + budget -> config TOML),
arwen_estimate (does it fit), arwen_forecast mode "go" (fetch through
render), then arwen_run_summary on the run folder and arwen_render for
more products.
"""


def build_server(manager: JobManager | None = None):
    """Assemble the server; returns ``(server, tool_names_by_tier)``."""

    require_sdk()
    from mcp.server.fastmcp import FastMCP

    server = FastMCP(name="arwen", instructions=_INSTRUCTIONS)
    manager = manager or JobManager()

    from gpuwm.mcp import engine_tools
    tiers = {"engine": engine_tools.register(server, manager)}
    return server, tiers


def build_parser() -> argparse.ArgumentParser:
    """The (deliberately empty) command surface, for the docs generator.

    The server is configured by environment, not flags -- the transport
    is stdio and the jobs directory is ``GPUWM_MCP_JOBS_DIR`` (default
    ``~/.gpuwm/mcp-jobs``) -- so the parser exists to say exactly that
    on ``--help`` and to refuse stray arguments the argparse way, and
    so the CLI reference page can read this door like every other one.
    """

    return argparse.ArgumentParser(
        prog="arwen-mcp",
        description="ArWen's doors served as MCP tools over stdio, for "
                    "an LLM agent (same program: `python -m gpuwm.mcp`). "
                    "Configuration is by environment, not flags: "
                    "GPUWM_MCP_JOBS_DIR is the job-receipt root (default "
                    "~/.gpuwm/mcp-jobs) and the GPUWM_* variables are "
                    "inherited by every door a tool runs. Needs the "
                    "[mcp] extra: pip install gpuwm[mcp].")


def main(argv: list[str] | None = None) -> int:
    """Serve MCP over stdio until the client disconnects."""

    build_parser().parse_args(argv)
    try:
        server, tiers = build_server()
    except ModuleNotFoundError as error:
        print(str(error), file=sys.stderr)
        return 2
    print(f"arwen-mcp: serving {sum(map(len, tiers.values()))} tools "
          f"(engine {len(tiers['engine'])}) over stdio", file=sys.stderr)
    server.run(transport="stdio")
    return 0
