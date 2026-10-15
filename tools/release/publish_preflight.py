"""Pre-push and pre-tag checks, kept outside the shared cut implementation."""
import os
from pathlib import Path

from tools.release import pypi_capacity
from tools.workflow_rehearsals import require_receipt


def check(cut, packet, public_tree, commit):
    receipt = Path(os.environ.get("ARWEN_WORKFLOW_REHEARSAL", str(cut / "evidence/workflow-rehearsals.json")))
    require_receipt(receipt, public_tree, commit)
    return pypi_capacity.check_packet(packet, fetch=pypi_capacity.fetch_project,
        out=cut / "evidence/pypi-capacity.json")
