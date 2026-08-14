"""Red-on-revert: undo each fix in place, run its test, restore.

Every entry is (label, file, old, new, pytest -k selector).  A fix whose
test still passes with the fix reverted is a test that is not testing it.

THE TREE IS THE ONE THIS FILE LIVES IN.  This module used to open with an
absolute path to the lane worktree it was written in.  Two things were
wrong with that, and both are the kind this repository has been bitten by
before.  It pinned a user's scratchpad directory into a committed tool,
so the harness ran nowhere else; and, worse, when the tool was carried
onto another tree it kept grading the ORIGINAL one -- reporting PROVEN
about source the caller was not cutting, which is precisely the
wrong-tree failure the battery's PYTHONPATH discipline exists to stop.
``parents[2]`` is tools/reach223/ -> tools/ -> the worktree root, so the
harness now grades whatever checkout it was copied into, and the binding
is printed below rather than assumed.

BYTES, NOT TEXT.  The revert/restore cycle reads and writes through
``read_bytes``/``write_bytes``.  ``write_text`` applies newline
translation, so on Windows the "restore" leg rewrote every LF file it
touched as CRLF -- the harness left the tree dirtier than it found it,
and `.gitattributes` (`* -text`) would have carried those bytes straight
into the object database, moving the source hashes the runners bind.
Observed here, at the 2.3.0 round-2 assembly, on this very file's own
first run.
"""
import pathlib
import subprocess
import sys

WT = pathlib.Path(__file__).resolve().parents[2]

CASES = [
    # 1. joint budget decision (mission 1)
    ("joint-reservation", "gpuwm/core/streaming.py",
     "reservations = (_tree_reservations(nodes, total_budget)\n"
     "                    if consults_planner else [(0, [])] * len(nodes))",
     "reservations = [(0, [])] * len(nodes)",
     "reserv or reserves or streamed_parent or child"),
    # 2. per-domain [tiles] surface (mission 1)
    ("per-domain-tiles", "gpuwm/experiment.py",
     "spawn=spawn_cfg, tiles=domain_tiles)",
     "spawn=spawn_cfg)",
     "per_domain or opts_out or inverse_shape"),
    # 3. gpuwm run wires the streamed builder (mission 2)
    ("run-route-builders", "gpuwm/runtime.py",
     "builders=_streaming.builders_for_tree(model, exp.tiles),",
     "",
     "run_route"),
    # 4. review finding 2: host budget to the probe
    ("host-budget-probe", "gpuwm/core/streaming.py",
     "machine0 = autoplan.Machine.detect(\n"
     "            host_bytes=(None if options.host_budget_bytes is None\n"
     "                        else int(options.host_budget_bytes)))",
     "machine0 = autoplan.Machine.detect()",
     "host_budget_to_the_probe"),
    # 5. review finding 5: the host ledger
    ("host-ledger", "gpuwm/core/streaming.py",
     "host_spent += host_claim", "pass",
     "siblings"),
    # 6. review finding 1: state-less clock policy
    ("state-less-clock", "gpuwm/core/streaming.py",
     "if state is None and external_clock is DERIVE_CLOCK:",
     "if False:",
     "clock_policy"),
    # 7. the 2.3.0 round-2 rider: streaming_receipt keyed on the per-domain
    #    DECISIONS rather than on the tree-wide [tiles] table.  Reverting it
    #    restores the early `return {}`, and a tree whose tree-wide mode is
    #    "off" but whose child carries tiles = {mode = "on"} goes back to
    #    streaming in silence -- no receipt line names the grid that tiled.
    ("per-domain-receipt", "gpuwm/core/streaming.py",
     "    if not decisions:\n"
     "        return {}\n"
     "    if options is None:\n"
     "        options = OFF\n",
     "    if options is None or not options.enabled or not decisions:\n"
     "        return {}\n",
     "per_domain_table_streamed or tree_wide_mode_was_overridden or "
     "moves_no_shipped_receipt_byte or "
     "unconfigured_run_still_contributes_no_receipt"),
]


def run(selector):
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_streaming.py", "-q",
         "-k", selector],
        cwd=WT, capture_output=True, text=True,
        env={**__import__("os").environ, "PYTHONSAFEPATH": "1",
             "PYTHONPATH": str(WT)})
    tail = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
    return proc.returncode, tail


print(f"tree under test: {WT}")
_probe = subprocess.run(
    [sys.executable, "-c", "import gpuwm;print(gpuwm.__file__)"],
    cwd=WT, capture_output=True, text=True,
    env={**__import__("os").environ, "PYTHONSAFEPATH": "1",
         "PYTHONPATH": str(WT)})
print(f"binding:        {_probe.stdout.strip()}")
if not _probe.stdout.strip().startswith(str(WT)):
    sys.exit(f"REFUSING: gpuwm resolves outside {WT}")

for label, rel, old, new, selector in CASES:
    path = WT / rel
    # Bytes, not text: `write_text` would translate newlines and the
    # restore leg would hand back a CRLF copy of an LF file.
    raw = path.read_bytes()
    text = raw.decode("utf-8")
    if text.count(old) != 1:
        print(f"SKIP  {label}: anchor count {text.count(old)}")
        continue
    green_rc, green_tail = run(selector)
    path.write_bytes(text.replace(old, new).encode("utf-8"))
    try:
        red_rc, red_tail = run(selector)
    finally:
        path.write_bytes(raw)
    assert path.read_bytes() == raw, f"{rel} not restored byte-for-byte"
    verdict = "PROVEN" if (green_rc == 0 and red_rc != 0) else "**WEAK**"
    print(f"{verdict}  {label}")
    print(f"        green: {green_tail}")
    print(f"        red:   {red_tail}")
