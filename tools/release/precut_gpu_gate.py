"""Run the GPU pin set on the release node's card before export.

THE BREAKAGE THIS PREVENTS: a GPU pin red on the release line shipped twice.
tools/release/precut_gate.py runs the contract set with -m "not gpu" under
GPUWM_NO_LOCAL_GPU=1, so before this stage existed no test that needs a card
ran before a cut. The phase-2 step pin (tests/test_coriolis_map.py) shipped red
in 2.7.4 and 2.7.5 and the mp=8 freeze (tests/test_mp8_frozen.py) shipped red
in 2.7.4 and 2.7.5, and nothing in the release procedure could see either.

This stage archives, ships and identity-checks the candidate with precut_gate's
own functions (no second copy of that machinery), runs the set that
tests/gpu_pin_set.txt declares on the node's card WITHOUT GPUWM_NO_LOCAL_GPU and
with -n 0 (pins are captured serially; two processes on one card are a discarded
reading), records the card name, driver, compute capability, CUDA runtime, cupy
version and the NVRTC build that compiled the kernels beside the commit, names
every test that skipped with its reason, refuses a run whose skips say the card
was absent, and refuses to pass on an early exit through
precut_gate.completed_result. The same file is read by tests/test_gpu_pin_set.py,
so the set the gate runs and the set the tree declares cannot drift apart.

The card is the stage's instrument, so the stage never reads it shared. Before
the run it waits, bounded, for the card to be free of other compute processes;
while the pins run it samples the card every two seconds (its own process and
that process's children excepted) and once more after they end; the sample the
probe saw, the sample the run started on, the samples during and the sample
after all go in the receipt, and a run during which another process held the
card is a discarded reading (status ERROR naming the process), never a PASS and
never a FAIL. The probe that names the card also compiles one kernel, so an
interpreter whose cupy has no toolkit refuses here, naming the toolkit, instead
of recording every pin red with "Failed to find CUDA headers".

Receipts use schema arwen.precut-gpu-gate.v1 and the precut-gpu-gate-<commit>.json
name; a repeated run for one commit writes precut-gpu-gate-<commit>.<n>.json and
names the receipt it follows. A PASS applies to the named commit on the named
card under the named NVRTC build only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time

if __package__:
    from . import precut_gate as base
else:  # run as a script: tools/release/precut_gpu_gate.py
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from tools.release import precut_gate as base

SCOPE = "GPU pin tests on the release node's card before export"
SCHEMA = "arwen.precut-gpu-gate.v1"
PIN_SET = "tests/gpu_pin_set.txt"
GateError = base.GateError
completed_result = base.completed_result

#: A skip whose reason matches one of these means the stage ran with no card,
#: which proves nothing; the run is refused rather than recorded as a pass.
NO_CARD_REASONS = ("no CUDA GPU", "GPUWM_NO_LOCAL_GPU", "device verification",
                   "needs CUDA", "no CUDA device")

#: How long the stage waits for another compute process to leave the card
#: before refusing, and how often it looks: the release procedure's own
#: bound for a shared card.
IDLE_WAIT_SECONDS = 540
IDLE_POLL_SECONDS = 30
#: How often the card is sampled while the pins run.
SAMPLE_SECONDS = 2

#: Runs on the shipped tree through the identity probe.  The NVRTC build comes
#: from the tree's own gpuwm.certify.compile_platform, the fingerprint the
#: per-compiler pin rows are keyed by (a candidate without it records cupy's
#: coarse two-part version and says so).  The kernel it compiles is the
#: toolkit check: getDeviceProperties needs no headers and no NVRTC, so a
#: card that answers is not yet a card that can run a pin.
CARD_PROBE = r'''
import json, cupy
r = cupy.cuda.runtime
p = r.getDeviceProperties(0)
n = p['name']
n = n.decode() if isinstance(n, bytes) else str(n)
card = {'name': n, 'compute_capability': str(p['major']) + '.' + str(p['minor']),
        'cupy': cupy.__version__, 'cuda_runtime': r.runtimeGetVersion(),
        'cuda_driver': r.driverGetVersion(), 'device_count': r.getDeviceCount()}
try:
    from gpuwm.certify.compile_platform import compile_platform_fingerprint
    fp = compile_platform_fingerprint()
    card['nvrtc'] = fp['nvrtc_build']
    card['nvrtc_build_id'] = fp['nvrtc_build_id']
    card['nvrtc_library_sha256'] = fp['nvrtc_library_sha256']
except Exception:
    try:
        card['nvrtc'] = '.'.join(str(v) for v in cupy.cuda.nvrtc.getVersion()) + ' (coarse)'
    except Exception:
        card['nvrtc'] = 'unresolved'
try:
    kernel = cupy.ElementwiseKernel('float32 x', 'float32 y', 'y = 2 * x', 'arwen_card_probe')
    y = kernel(cupy.arange(4, dtype=cupy.float32))
    assert float(y.sum()) == 12.0, y
    card['compiles'] = True
except Exception as error:
    card['compiles'] = False
    print('ARWEN-CARD-NO-COMPILE ' + type(error).__name__ + ': ' + str(error).replace('\n', ' ')[:600])
print('ARWEN-CARD ' + json.dumps(card))
'''
SMI_DRIVER = "nvidia-smi --query-gpu=name,driver_version --format=csv,noheader"
SMI_APPS = "nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader"


GATE_TREE = Path(__file__).resolve().parents[2]


def pin_set(tree: Path) -> list[str]:
    """The declared set: one pytest node id per line; ``#`` comments and blank
    lines are ignored. Every named file must be in the candidate."""
    return pin_set_with_source(tree)[0]


def gate_tree_commit() -> str:
    """The gate's own tree, named by what it holds and not by where it sits.

    A path here said which worktree on which machine the copy came from,
    which is both a machine path in a committed receipt and the less
    useful of the two answers: a reader checking WHICH set ran wants the
    commit, and the commit is the same wherever the tree is checked out.
    """
    result = base.run(["git", "-C", str(GATE_TREE), "rev-parse", "--verify",
                       "HEAD^{commit}"])
    head = result.stdout.strip() if result.returncode == 0 else ""
    return head if re.fullmatch(r"[0-9a-f]{40}", head) else "commit unavailable"


def pin_set_with_source(tree: Path) -> tuple[list[str], str]:
    """``(entries, source)``: the candidate's own tests/gpu_pin_set.txt, or,
    for a candidate that predates the file, the gate's own copy, named by the
    commit it sits at so the receipt says which set it ran."""
    path, source = tree / PIN_SET, "candidate"
    if not path.is_file():
        path, source = GATE_TREE / PIN_SET, f"gate tree {gate_tree_commit()}"
    if not path.is_file():
        raise GateError(f"{PIN_SET} is missing from the candidate and from the gate's own tree. Restore the declared GPU pin set and run again.")
    entries = []
    for line in path.read_text(encoding="utf-8").splitlines():
        entry = line.split("#", 1)[0].strip()
        if entry:
            entries.append(entry)
    if not entries:
        raise GateError(f"{PIN_SET} declares no pin tests. Restore its entries and run again.")
    for entry in entries:
        file = entry.split("::", 1)[0]
        if not re.fullmatch(r"tests/test_[a-z0-9_]+\.py", file) or not (tree / file).is_file():
            raise GateError(f"{PIN_SET} names {entry!r}, whose file is not a test file in the candidate. Correct the entry and run again.")
    return entries, source


def pin_files(entries: list[str]) -> list[str]:
    return sorted({entry.split("::", 1)[0] for entry in entries})


def _apps(block: str) -> list[str]:
    """nvidia-smi's compute-app rows, one process per row: the LAST row seen
    for a pid, because a sampled process's memory changes between samples."""
    rows: dict[str, str] = {}
    for line in block.splitlines():
        line = line.strip()
        if line:
            rows[line.split(",", 1)[0].strip()] = line
    return list(rows.values())


def card_report(node: str, python: str, remote: str, cuda_path: str | None) -> dict:
    """What the node's card is, before any pin runs on it: refused when cupy
    sees no device or cannot compile a kernel with the toolkit it found."""
    prefix = base.remote_prefix(remote, python, cuda_path=cuda_path, local_gpu=True)
    command = (prefix + f"{shlex.quote(python)} -c {shlex.quote(CARD_PROBE)}; "
               f"echo ARWEN-SMI; {SMI_DRIVER}; echo ARWEN-APPS; {SMI_APPS}; true")
    result = base.run_remote(node, command)
    text = result.stdout + result.stderr
    match = re.search(r"^ARWEN-CARD (\{.*\})$", text, re.M)
    if match is None:
        raise GateError("The node reports no usable CUDA device through cupy, so the card stage cannot run. "
                        f"Repair the node's card or its cupy and run again.\n{text[-2000:]}")
    card = json.loads(match.group(1))
    smi = text.split("ARWEN-SMI", 1)[1] if "ARWEN-SMI" in text else ""
    driver_lines, apps_lines = smi, ""
    if "ARWEN-APPS" in smi:
        driver_lines, apps_lines = smi.split("ARWEN-APPS", 1)
    drivers = [line.strip() for line in driver_lines.splitlines() if "," in line]
    card["driver"] = drivers[0].split(",")[-1].strip() if drivers else None
    card["other_compute_processes"] = _apps(apps_lines)
    if card.get("device_count", 0) < 1:
        raise GateError("The node reports zero CUDA devices. Repair the node's card and run again.")
    if not card.get("compiles"):
        failure = re.search(r"^ARWEN-CARD-NO-COMPILE (.*)$", text, re.M)
        raise GateError(f"The node's interpreter sees {card['name']} but cannot compile a CUDA kernel, so every pin "
                        "would fail for the toolkit and not the code: "
                        f"{failure.group(1) if failure else 'the probe kernel did not run'}. Point --cuda-path at a "
                        "toolkit that carries include/ and lib/libnvrtc (the NVRTC build the pins are keyed by) and run again.")
    return card


def other_processes(node: str) -> list[str]:
    """The compute processes on the node's card right now."""
    result = base.run_remote(node, f"echo ARWEN-APPS; {SMI_APPS}; true")
    text = result.stdout + result.stderr
    if "ARWEN-APPS" not in text:
        raise GateError("nvidia-smi did not answer on the node, so the card's occupancy cannot be read. "
                        f"Repair the node's driver and run again.\n{text[-1000:]}")
    return _apps(text.split("ARWEN-APPS", 1)[1])


def wait_for_idle_card(node: str, first: list[str]) -> tuple[list[str], int]:
    """Wait, bounded by IDLE_WAIT_SECONDS, for the card to hold no other
    compute process; ``(the final sample, seconds waited)``.  A card still
    shared at the bound refuses the stage naming the processes: a pin
    captured on a shared card is a discarded reading."""
    apps, waited = first, 0
    while apps and waited < IDLE_WAIT_SECONDS:
        time.sleep(IDLE_POLL_SECONDS)
        waited += IDLE_POLL_SECONDS
        apps = other_processes(node)
    if apps:
        raise GateError(f"Another compute process held the card for {waited} s after the probe, so a pin captured now "
                        f"would be a discarded reading: {'; '.join(apps)}. Wait for the card to be free and run again.")
    return apps, waited


def sampled_test_command(python: str, report_path: str, entries: list[str], apps_path: str) -> str:
    """precut_gate's serial pytest invocation, run in the background on the
    node with the card sampled every SAMPLE_SECONDS until it ends (the pytest
    process and its children excepted), then sampled once more; the samples
    follow the test output between ARWEN-APPS-DURING, ARWEN-APPS-AFTER and
    ARWEN-APPS-END, and the command exits with pytest's own status."""
    quote = shlex.quote
    test = base.remote_test_command(python, report_path, entries, workers=0, markers=None)
    awk = ("'BEGIN{n=split(own,a,\" \");for(i=1;i<=n;i++)o[a[i]]=1}"
           "{p=$0;sub(/,.*/,\"\",p);gsub(/ /,\"\",p);if(!(p in o))print}'")
    return (f"ARWEN_APPS={quote(apps_path)}; : > \"$ARWEN_APPS\"; {test} & ARWEN_PID=$!; "
            "while kill -0 \"$ARWEN_PID\" 2>/dev/null; do "
            "own=\"$ARWEN_PID $(pgrep -P \"$ARWEN_PID\" | tr '\\n' ' ')\"; "
            f"{SMI_APPS} 2>/dev/null | awk -v own=\"$own\" {awk} >> \"$ARWEN_APPS\"; "
            f"sleep {SAMPLE_SECONDS}; done; "
            "wait \"$ARWEN_PID\"; ARWEN_RC=$?; "
            "echo ARWEN-APPS-DURING; sort -u -- \"$ARWEN_APPS\"; "
            f"echo ARWEN-APPS-AFTER; {SMI_APPS} 2>/dev/null; echo ARWEN-APPS-END; exit \"$ARWEN_RC\"")


def card_samples(text: str) -> tuple[list[str], list[str]]:
    """``(during, after)`` from the sampled run's output; a record with no
    sampler blocks is a run whose occupancy is unknown, and is refused."""
    if "ARWEN-APPS-DURING" not in text or "ARWEN-APPS-END" not in text:
        raise GateError("The card sampler's record is missing from the pin run's output, so whether another process "
                        "shared the card is unknown and the reading is discarded. Run again.")
    block = text.split("ARWEN-APPS-DURING", 1)[1].split("ARWEN-APPS-END", 1)[0]
    during, after = block.split("ARWEN-APPS-AFTER", 1) if "ARWEN-APPS-AFTER" in block else (block, "")
    return _apps(during), _apps(after)


def refuse_cardless_skips(skips: list[dict]) -> None:
    cardless = [s for s in skips if any(reason in s["reason"] for reason in NO_CARD_REASONS)]
    if cardless:
        names = ", ".join(s["test"] for s in cardless)
        raise GateError("These pin tests skipped because the run saw no card, so the card stage proved nothing: "
                        f"{names}. Run on the node's card and run again.")


def main(argv=None, *, defaults=None) -> int:
    defaults = defaults or {}
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    base.add_arguments(parser, defaults)
    args = parser.parse_args(argv)
    record = {"schema": SCHEMA, "scope": SCOPE, "full_release_qualification": False,
              "status": "ERROR", "pin_set": PIN_SET}
    started = time.perf_counter()
    evidence = None
    try:
        node = base.validated_node(args.node)
        if not args.remote_dir or not args.evidence_dir:
            raise GateError("The scratch or evidence directory is missing. Supply --remote-dir and --evidence-dir and run again.")
        remote_base = base.remote_directory(args.remote_dir)
        tree = args.tree.resolve()
        head = base.candidate_commit(tree)
        record["candidate"] = head
        evidence, previous = base.evidence_path(args.evidence_dir, f"precut-gpu-gate-{head[:9]}")
        if previous is not None:
            record["previous_receipt"] = previous
        entries, record["pin_set_source"] = pin_set_with_source(tree)
        files = pin_files(entries)
        record["tests"] = entries
        record["files"] = files
        remote, attempt = base.ship_candidate(tree, head, node, remote_base)
        record["remote_directory"] = base.record_name(remote)
        python, cuda_path = base.remote_path(args.python), base.remote_path(args.cuda_path)
        card = record["card"] = card_report(node, python, remote, cuda_path)
        card["other_compute_processes_at_probe"] = list(card["other_compute_processes"])
        card["other_compute_processes"], card["idle_wait_seconds"] = \
            wait_for_idle_card(node, card["other_compute_processes"])
        report_path = remote + "/gpu-pins-" + attempt + ".xml"
        record["structured_report"] = base.record_name(report_path)
        command = (base.remote_prefix(remote, python, cuda_path=cuda_path, local_gpu=True)
                   + sampled_test_command(python, report_path, entries, remote + "/gpu-apps-" + attempt + ".txt"))
        result = base.run_remote(node, command)
        record["process_exit"] = result.returncode
        text = result.stdout + result.stderr
        record["output_tail"] = text[-8000:]
        if result.returncode not in (0, 1):
            raise GateError(f"The pin process exited {result.returncode}. Resolve the execution failure and run the complete check again.")
        card["other_compute_processes_during"], card["other_compute_processes_after"] = card_samples(text)
        shared = card["other_compute_processes_during"] + card["other_compute_processes_after"]
        if shared:
            raise GateError("Another compute process held the card while the pins ran, so this run is a discarded "
                            f"reading, not a pass and not a failure: {'; '.join(shared)}. Wait for the card to be "
                            "free and run again.")
        xml = base.read_report(node, report_path)
        record["report_sha256"] = hashlib.sha256(xml.encode("utf-8")).hexdigest()
        record.update(completed_result(xml, result.returncode, files))
        record["skipped"] = base.skipped_cases(xml)
        refuse_cardless_skips(record["skipped"])
        record["status"] = "FAIL" if record["failures"] else "PASS"
        rc = 1 if record["failures"] else 0
    except (GateError, OSError, subprocess.SubprocessError) as error:
        record["error"] = str(error)
        rc = 2
    return base.finish(record, evidence, started, rc, "precut gpu gate")


if __name__ == "__main__":
    raise SystemExit(main())
