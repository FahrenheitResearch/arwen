# The ArWen MCP server (`arwen-mcp`)

This page is for agents and the people wiring them up.

`arwen-mcp` (also `python -m gpuwm.mcp`) is a stdio [Model Context
Protocol] server that exposes ArWen's doors as tools, so an LLM agent
-- Claude Code today, a local-LLM harness tomorrow -- can plan, price,
fetch, prep, forecast and render without ever parsing prose.  It is a
thin shell over the real command line: every tool shells out to
`python -m gpuwm.cli ...` (verify against the artifact -- the server
re-implements nothing a door owns) or reads a receipt a door already
wrote, and every documented refusal reaches the agent as the CLI's own
one-sentence refusal, verbatim, so a refused call says exactly what to
change.

## Install and register

The MCP SDK is the `[mcp]` extra; the base wheel is unchanged:

```
pip install gpuwm[mcp]
```

The extra resolves `mcp>=1.26,<2`.  The ceiling is exclusive and
deliberate: this server speaks the 1.x FastMCP surface, 2.x renamed it,
and a floor with no ceiling resolves to whichever SDK is newest -- which
is by definition the one this server has never run on.  If you pin the
SDK yourself, pin it inside that range.

Claude Code registration -- put this in the project's `.mcp.json`
(committed copy: `examples/mcp/claude-code.mcp.json`):

```json
{
  "mcpServers": {
    "arwen": {
      "command": "arwen-mcp",
      "args": [],
      "env": {
        "GPUWM_MCP_JOBS_DIR": "${HOME}/.gpuwm/mcp-jobs"
      }
    }
  }
}
```

From a checkout, use `"command": "python", "args": ["-m", "gpuwm.mcp"]`
with `"cwd"` at the repository root instead.  Running `arwen-mcp`
without the SDK prints the remedy sentence and exits 2; it never
tracebacks.

`GPUWM_MCP_JOBS_DIR` is where job receipts and logs live (default
`~/.gpuwm/mcp-jobs`).  The server inherits the shell's GPUWM
environment: set `GPUWM_NO_LOCAL_GPU=1` on a box whose card is not
authorized, exactly as you would for pytest.

## The tools

The server wraps the CLI doors -- everything published gpuwm owns:

| tool | what it does |
|---|---|
| `arwen_doctor` | `gpuwm doctor --json`: probe the real estate; optionally explain one source's resolution |
| `arwen_sources` | `gpuwm run-plan --sources` (whole registry) or `gpuwm sources ID --json` (one row) |
| `arwen_capabilities` | `gpuwm run-plan --catalog` and/or `--physics-profiles`: what renders, what prepares, why refused pairings refuse |
| `arwen_plan_domain` | the domain wizard, non-interactive: point/polygon + GPU budget -> experiment TOML + summary |
| `arwen_estimate` | `gpuwm check CONFIG --json`: price VRAM before anything runs (pass `budget_gib`/`vram_gib` to size for an absent card) |
| `arwen_fetch` | JOB: `gpuwm fetch` -- one cycle's forcing data |
| `arwen_prep` | JOB: `gpuwm prep` -- your files into a prepared tree (wide surface; `extra_args` passes any prep flag verbatim) |
| `arwen_forecast` | JOB: `gpuwm go` (composed chain) or `gpuwm run` (the model alone) -- both spellings; `dry_run` prints go's six commands synchronously |
| `arwen_render` | JOB: `gpuwm render` -- PNGs from wrfout frames; `--pair` comparison sheets supported |
| `arwen_list_products` | `gpuwm render --list-products`: per-file product availability, with reasons |
| `arwen_run_summary` | read a run folder's receipts (report.json status verdict, proof/receipt, events/progress JSONL, output inventory) into one JSON |
| `job_status` / `job_events` / `job_result` / `job_cancel` / `job_list` | the job pattern, below |

## The job pattern

Anything long runs detached; no tool call is ever held open by a
forecast.

1. A launch tool (`arwen_fetch`, `arwen_prep`, `arwen_forecast`,
   `arwen_render`) validates, takes the GPU lock
   if the job touches the card, spawns the real CLI as a detached
   subprocess, writes a receipt, and returns `{job_id}` immediately.
2. `job_status job_id` -- `running` / `exited` / `cancelled` / `lost`,
   with pids, exit code, declared outputs.
3. `job_events job_id` -- incremental tail with a byte cursor: pass the
   returned `next_cursor` back in to read only what is new.  Streams:
   `stdout`, `stderr`, `events` (the run's own `events.jsonl`),
   `progress` (`progress.jsonl`, one record per model step).
4. `job_result job_id` -- refuses while running; when done: `ok`, exit
   code, stdout tail, and on exit 2 the door's refusal sentence
   verbatim in `refusal`.
5. `job_cancel job_id` -- terminates the process tree, records the
   cancellation, releases the GPU lock.  Deletes nothing, ever.
6. `job_list` -- every job under the jobs root with current state.

Each job owns a directory under `GPUWM_MCP_JOBS_DIR`: `receipt.json`
(argv, cwd, env additions, pids, declared outputs), `stdout.log`,
`stderr.log`, `started.json`, `result.json`.  State is derived from
those files plus pid liveness, so a server restart loses nothing --
jobs launched before the restart still answer `job_status`, and a
wrapper that died without a result reports `lost` rather than lying.

**GPU arbitration:** one GPU job at a time per card, held by a
lockfile the server owns.  A second GPU launch is REFUSED with a
sentence naming the running job id -- never silently queued: a launch
tool that queues has not launched.  A lock whose holder process is
gone releases itself at the next launch.

## What refusals look like

The CLI's contract is one sentence at exit 2, and the server's job is
to not touch it.  A refused synchronous tool errors with the door's
sentence; a refused job carries it in `job_result.refusal`.  Example:

```
gpuwm check: C:\...\nope.toml does not exist; pass the experiment
.toml that `gpuwm domain` wrote.
```

If a sentence does not tell you enough, run the same door in a shell
with `--explain` -- the receipt in the job directory records the exact
argv.

## Boundaries

- v1 has no tool that deletes anything.
- One server per machine is the v1 assumption (the GPU lock is the
  server's own file, not a system-wide arbiter).
- The server is unauthenticated stdio for a local agent; do not put it
  on a socket.

[Model Context Protocol]: https://modelcontextprotocol.io
