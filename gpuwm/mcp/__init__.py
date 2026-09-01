"""ArWen MCP server: the CLI's doors, spoken over the Model Context Protocol.

``arwen-mcp`` (also ``python -m gpuwm.mcp``) is a stdio MCP server that
lets an LLM agent drive ArWen -- plan, price, fetch, prep, forecast,
render -- as tools.  It is a THIN shell over the real command line:
every tool either shells out to ``python -m gpuwm.cli ...`` (the same
artifact a person runs, so verification stays against the artifact) or
reads a receipt a door already wrote.  No physics, no sizing, no
routing logic lives here; the CLI owns all of it, and the one thing
this layer adds is the ASYNC JOB PATTERN (:mod:`gpuwm.mcp.jobs`) so a
forecast that runs for an hour does not hold a tool call open.

Refusals travel verbatim: the CLI's contract is one sentence at exit 2,
and that sentence IS the tool error the driving agent sees, so the
agent can self-correct off the same words a person reads.

One tool tier: :mod:`gpuwm.mcp.engine_tools`, everything the published
gpuwm release owns.  It works from a bare wheel.

The MCP SDK itself is an optional extra (``pip install gpuwm[mcp]``);
:func:`require_sdk` is the one import guard, shared by the entry point
and the server module.
"""

from __future__ import annotations

#: The one-sentence remedy for a missing SDK, shared so the entry point
#: and any embedder print the same words.
SDK_REMEDY = (
    "the MCP Python SDK is not installed, so this server cannot speak "
    "the protocol; install it with `pip install gpuwm[mcp]` (which "
    "provides mcp>=1.26,<2 -- this server speaks the 1.x FastMCP "
    "surface, and 2.x renamed it).")


def require_sdk() -> None:
    """Refuse with the remedy sentence when the ``mcp`` SDK is absent.

    Raised as ``ModuleNotFoundError`` so an embedder can distinguish the
    install gap from every other failure; the message carries the whole
    remedy, because this is the first thing a bare-wheel user hits.
    """

    try:
        import mcp  # noqa: F401  -- presence probe only
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(f"arwen-mcp: {SDK_REMEDY}") from error
