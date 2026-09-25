"""Per-card GPU pins: which committed capture describes THIS card.

A bitwise pin of a GPU result is a property of the card as well as of the
code.  Measured 2026-09-17 on node-1 (receipts under
tests/data/receipts/pin-gates/): the phase-2 step builders at one commit
differ between an RTX 5070 Ti (compute capability 12.0) and an RTX 4090
(8.9) in 25 of 27 entries, while an RTX 3080 (8.6) and the 4090 agree bit
for bit, and every other npz pin in the tree (the sina=0 Coriolis pin, the
advection periodic pin, the diff6 base capture, all captured in July 2026
on a box nobody recorded) differs from a 4090 run by ULPs in 3 to 29
percent of its words.  So each pin keeps one file per card, keyed by the
device name cupy reports, and a card with no file SKIPS the bitwise
comparison with ``skip_reason()`` rather than failing for the card and not
the code.  tools/release/precut_gpu_gate.py runs the pins on the release
node's card and names every such skip in its receipt.

Adding a card to a pin is a capture on that card by
``tools/recapture_card_pins.py --pin NAME --write`` (the phase-2 pin by
``tools/recapture_phase2_pin.py --write``), a row in ``PINS`` below, and
the reading the tool prints (against the reference card's file, or the
original capture when no card has one yet) recorded in the ledger of the
test that reads the pin.

The original files (``ORIGINALS``) are the captures the pins were first
made with, on cards nobody recorded; they stay committed as the historical
record and as the file a first per-card capture is read against.
"""

from __future__ import annotations

from pathlib import Path

PIN_DIR = Path(__file__).resolve().parent / "data"

#: pin name -> {device name -> file under tests/data}.
PINS = {
    "phase2_step_regression": {
        "NVIDIA GeForce RTX 5070 Ti": "phase2_step_regression.npz",
        "NVIDIA GeForce RTX 4090": "phase2_step_regression.rtx4090.npz",
    },
    "coriolis_map_sina0_pin": {
        "NVIDIA GeForce RTX 4090": "coriolis_map_sina0_pin.rtx4090.npz",
    },
    "advection_periodic_regression": {
        "NVIDIA GeForce RTX 4090": "advection_periodic_regression.rtx4090.npz",
    },
    "diff6_base_4d2ce99": {
        "NVIDIA GeForce RTX 4090": "diff6_base_4d2ce99.rtx4090.npz",
    },
}

#: The file each pin was first captured into, card unrecorded.
ORIGINALS = {
    "phase2_step_regression": "phase2_step_regression.npz",
    "coriolis_map_sina0_pin": "coriolis_map_sina0_pin.npz",
    "advection_periodic_regression": "advection_periodic_regression.npz",
    "diff6_base_4d2ce99": "diff6_base_4d2ce99.npz",
}

#: The card whose file a new card's first reading is taken against, when
#: that pin has one for it; otherwise the original capture is the reference.
REFERENCE_CARD = "NVIDIA GeForce RTX 5070 Ti"


def device_name() -> str:
    """The device name cupy reports for device 0, as ``PINS`` keys it."""
    import cupy as cp
    name = cp.cuda.runtime.getDeviceProperties(0)["name"]
    return name.decode() if isinstance(name, bytes) else str(name)


def device_compute_capability() -> str:
    import cupy as cp
    props = cp.cuda.runtime.getDeviceProperties(0)
    return f"{props['major']}.{props['minor']}"


def declared_path(pin: str, name: str | None = None) -> Path | None:
    """Where ``name``'s capture of ``pin`` lives by ``PINS``, whether or not
    the file exists."""
    name = device_name() if name is None else name
    file = PINS[pin].get(name)
    return None if file is None else PIN_DIR / file


def path(pin: str, name: str | None = None) -> Path | None:
    """The COMMITTED capture of ``pin`` for ``name`` (default: this device),
    or None.  A row whose file is not in the tree counts as no capture: the
    row declares the file's name, the file carries the pin."""
    declared = declared_path(pin, name)
    return declared if declared is not None and declared.is_file() else None


def original_path(pin: str) -> Path:
    return PIN_DIR / ORIGINALS[pin]


def reference_path(pin: str) -> Path:
    """What a first capture on a new card is read against: the reference
    card's file when the pin has one, else the original capture."""
    return path(pin, REFERENCE_CARD) or original_path(pin)


def committed_cards(pin: str) -> list[str]:
    return sorted(card for card in PINS[pin] if path(pin, card) is not None)


def skip_reason(pin: str, name: str | None = None) -> str | None:
    """Why the bitwise comparison of ``pin`` cannot run on this card, or None."""
    name = device_name() if name is None else name
    if path(pin, name) is not None:
        return None
    return (f"the {pin} capture is per card and none is committed for "
            f"{name!r} (compute capability {device_compute_capability()}); "
            f"captures exist for {committed_cards(pin)}.  Capture one on this "
            f"card with tools/recapture_card_pins.py --pin {pin} --write (the "
            "phase-2 pin with tools/recapture_phase2_pin.py --write), add its "
            "row to tests/_card_pins.py PINS and its reading to the test's "
            "ledger.")
