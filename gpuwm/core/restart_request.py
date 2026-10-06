"""Worker checkpoint requests, consumed only after a durable step-boundary write."""
from __future__ import annotations

import os
from pathlib import Path

RESTART_REQUEST_ENV = "GPUWM_RESTART_REQUEST_FILE"


class RestartRequest:
    """A worker-owned file asks the existing restart writer to run once.

    The writer acknowledges by removing the request after it returns. A
    failed write leaves it in place and does not advertise a checkpoint.
    No environment setting means no filesystem work on the ordinary path.
    """

    def __init__(self):
        value = os.environ.get(RESTART_REQUEST_ENV)
        self.path = Path(value) if value else None

    def pending(self) -> bool:
        return self.path is not None and self.path.is_file()

    def acknowledge(self) -> None:
        if self.path is not None:
            self.path.unlink(missing_ok=True)
