"""A stopped or failed DA run leaves one recovery boundary, nothing else.

CPU only.  THE BREAKAGE THIS PREVENTS: a CONUS DA run stopped by SIGTERM on
box E left 188 GB of staged restart sets, packed job directories and
recovery generations behind, and the next run died on a full disk.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
from pathlib import Path

from tools.da_cycle_prepared import RecoveryRing, StagedRestarts


def _generation(slot: Path, seconds: float, *, manifest=True):
    slot.mkdir(parents=True)
    (slot / "restart_m000.npz").write_bytes(b"x" * 1000)
    if manifest:
        (slot / "ensemble-manifest.json").write_text(
            json.dumps({"elapsed_seconds": seconds}))


def test_a_stopped_run_keeps_only_its_newest_complete_generation(tmp_path):
    ring = tmp_path / "packed-recovery"
    _generation(ring / "slot00", 7200.0)
    _generation(ring / "slot01", 3600.0)
    _generation(ring / "slot02", 10800.0, manifest=False)   # interrupted
    receipt = RecoveryRing(ring).clear()
    assert sorted(p.name for p in ring.glob("slot*")) == ["slot00"]
    assert receipt["kept"] == "slot00"
    assert sorted(r["slot"] for r in receipt["removed"]) == ["slot01", "slot02"]
    assert json.loads((ring / "recovery-cleanup.json").read_text()) == receipt
    assert RecoveryRing(tmp_path / "absent").clear() is None


def test_the_stage_clear_removes_the_packed_job_directories(tmp_path):
    stage = StagedRestarts(tmp_path / "stage", default=True)
    member = tmp_path / "stage" / "packed" / "leg002" / "m029" / "stage"
    member.mkdir(parents=True)
    (member / "gpuwmrst.npz").write_bytes(b"y" * 5000)
    receipt = stage.clear()
    assert not (tmp_path / "stage" / "packed").exists()
    assert receipt["packed_job_bytes_removed"] == 5000


def test_sigterm_unwinds_through_the_door_cleanup(tmp_path):
    if os.name != "posix":
        return
    code = f"""
import os, signal, sys, time
from pathlib import Path
import tools.da_cycle_prepared as d
marker = Path({str(tmp_path)!r}) / "cleared"
class Probe:
    def clear(self):
        marker.write_text("yes")
def cycle(stages):
    stages.append(Probe())
    os.kill(os.getpid(), signal.SIGTERM)
    time.sleep(30)
d.cycle = cycle
sys.exit(d.main())
"""
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, "-c", code], cwd=root,
                            env=dict(os.environ, PYTHONPATH=str(root)),
                            timeout=60)
    assert result.returncode == 128 + signal.SIGTERM
    assert (tmp_path / "cleared").read_text() == "yes"


def test_the_stage_clear_keeps_the_job_logs_aside(tmp_path):
    """A failed member's or control's log is the only record of its error:
    the clear moves every packed job log aside, leg/job layout kept."""
    stage = StagedRestarts(tmp_path / "stage", default=True,
                           logs_dir=tmp_path / "packed-job-logs")
    leg = tmp_path / "stage" / "packed" / "leg002"
    (leg / "wave").mkdir(parents=True)
    (leg / "wave" / "member-029.log").write_text("OSError: No space left")
    (leg / "control.log").write_text("control ok")
    (leg / "m029" / "stage").mkdir(parents=True)
    (leg / "m029" / "stage" / "gpuwmrst.npz").write_bytes(b"z" * 10)
    receipt = stage.clear()
    assert not (tmp_path / "stage" / "packed").exists()
    kept = tmp_path / "packed-job-logs" / "leg002"
    assert (kept / "wave" / "member-029.log").read_text() == "OSError: No space left"
    assert (kept / "control.log").read_text() == "control ok"
    assert receipt["packed_job_logs_kept"] == 2


def test_a_landed_generation_removes_the_other_ring_slots(tmp_path):
    """THE BREAKAGE: the ring kept both slots for the whole run, two complete
    32-member CONUS generations (76 GB each) besides the stage and the
    issuance generation; box L could not host a cycled arm.  Once a
    generation lands, the ring keeps it alone."""
    from tools.da_cycle_prepared import prune_recovery_ring

    ring = tmp_path / "packed-recovery"
    for name in ("slot00", "slot01"):
        (ring / name / "0").mkdir(parents=True)
        (ring / name / "0" / "r.npz").write_bytes(b"x" * 10)
    (ring / "recovery-cleanup.json").write_text("{}")
    removed = prune_recovery_ring(ring / "slot01")
    assert removed == [{"slot": "slot00", "bytes": 10}]
    assert (ring / "slot01" / "0" / "r.npz").exists()
    assert not (ring / "slot00").exists()
    assert (ring / "recovery-cleanup.json").exists()
    assert prune_recovery_ring(ring / "slot01") == []
