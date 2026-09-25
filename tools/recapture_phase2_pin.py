"""Recapture the phase-2 step pin for THIS card, with its readings.

The capture pins ``PIN_STEPS`` full dycore steps of the four
``tests/_phase2_pin.py`` builders, compared bitwise by
``tests/test_coriolis_map.py::test_msf_one_f_zero_bitwise_phase2``.  Any
change to the dry dynamics' arithmetic moves it, and the rule for moving it
is a reading beside the new pin: which entries moved, by how much, against
what.  Earlier recaptures were done by a one-shot generator that was not
kept; this is that generator, kept.

The capture is also a property of the CARD (measured 2026-09-17: an RTX
5070 Ti and an RTX 4090 differ in 25 of 27 entries at one commit, an RTX
3080 and the 4090 agree bit for bit), so ``tests/_phase2_pin.py`` keeps one
file per card in ``PIN_FILES`` and this tool reads and writes the file for
the card it runs on.

Run on the card the pin is graded on:

    python tools/recapture_phase2_pin.py                # readings only
    python tools/recapture_phase2_pin.py --write --readings out.json
    python tools/recapture_phase2_pin.py --write --new-card FILE.npz \\
        --readings out.json                              # a card with no file

Without ``--write`` nothing is touched.  With it, every entry of every case
is replaced by the current tree's answer on this card (the file describes
one tip on one card, not a mixture), and the readings JSON carries, per
entry, the maximum absolute difference, the RMS of the entry it replaced,
their ratio, and the count of differing words, plus the card, its compute
capability, the driver and the commit, which is what the ledger in
``tests/test_coriolis_map.py`` must then quote.  A card with no committed
file is refused ``--write`` unless ``--new-card`` names the file to create
under tests/data; its readings are taken against the reference card's file
(``REFERENCE_CARD``) or ``--against CARD``, and the ``PIN_FILES`` row to
add is printed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))


def capture() -> dict[str, np.ndarray]:
    import cupy as cp
    import _phase2_pin as pin
    from gpuwm.core.dycore import run_steps

    entries: dict[str, np.ndarray] = {}
    for case, build in pin.CASES.items():
        s, cfg = build()
        if case == "dry_flat":                 # as the test exercises it
            s.set_map_coriolis(msft=np.ones((cfg.ny, cfg.nx)),
                               f=np.zeros((cfg.ny, cfg.nx)))
        run_steps(s, cfg, n=pin.PIN_STEPS)
        fields = pin.PIN_FIELDS + (("qv", "qc", "qr")
                                   if s.qv is not None else ())
        for f in fields:
            entries[f"{case}/{f}"] = cp.asnumpy(getattr(s, f)).copy()
    return entries


def readings(old: dict[str, np.ndarray], new: dict[str, np.ndarray]) -> dict:
    out = {}
    for key in sorted(set(old) | set(new)):
        if key not in old or key not in new:
            out[key] = {"status": "added" if key in new else "dropped"}
            continue
        a, b = old[key], new[key]
        if a.shape != b.shape or a.dtype != b.dtype:
            out[key] = {"status": "reshaped", "old": [list(a.shape), str(a.dtype)],
                        "new": [list(b.shape), str(b.dtype)]}
            continue
        diff = np.abs(a.astype(np.float64) - b.astype(np.float64))
        rms = float(np.sqrt(np.mean(a.astype(np.float64) ** 2)))
        differing = int((a.view(np.uint32) != b.view(np.uint32)).sum())
        out[key] = {
            "status": "moved" if differing else "held",
            "max_abs_difference": float(diff.max()),
            "rms_of_old_entry": rms,
            "max_abs_over_rms": float(diff.max() / rms) if rms > 0 else None,
            "differing_words": differing, "words": int(a.size),
        }
    return out


def _card() -> dict:
    import cupy as cp
    import _phase2_pin as pin
    def _query(command):
        try:
            done = subprocess.run(command, capture_output=True, text=True)
        except OSError:
            return None
        return done.stdout.strip() if done.returncode == 0 else None

    return {"card": pin.device_name(),
            "compute_capability": pin.device_compute_capability(),
            "driver": _query(["nvidia-smi", "--query-gpu=driver_version",
                              "--format=csv,noheader"]),
            "cupy": cp.__version__,
            "cuda_runtime": cp.cuda.runtime.runtimeGetVersion(),
            **_nvrtc(),
            "commit": _query(["git", "-C", str(ROOT), "rev-parse", "HEAD"])}


def _nvrtc() -> dict:
    """The NVRTC build that compiles this process's kernels, the key the
    per-compiler pin rows use (gpuwm.certify.compile_platform): a pin is a
    property of the compiled image, so the capture records the compiler
    beside the card."""
    try:
        from gpuwm.certify.compile_platform import compile_platform_fingerprint
        fingerprint = compile_platform_fingerprint()
    except Exception as error:  # a tree without the fingerprint module
        return {"nvrtc": f"unresolved ({type(error).__name__})"}
    return {"nvrtc": fingerprint["nvrtc_build"],
            "nvrtc_build_id": fingerprint["nvrtc_build_id"],
            "nvrtc_library_sha256": fingerprint["nvrtc_library_sha256"]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--write", action="store_true",
                        help="replace this card's capture with the current tree's answer")
    parser.add_argument("--readings", type=Path,
                        help="write the per-entry readings as JSON")
    parser.add_argument("--against", metavar="CARD",
                        help="take the readings against CARD's committed file "
                             "instead of this card's own")
    parser.add_argument("--new-card", metavar="FILE", type=str,
                        help="this card has no committed file: with --write, "
                             "create tests/data/FILE and print the PIN_FILES row")
    args = parser.parse_args()
    import _phase2_pin as pin
    card = _card()
    name = card["card"]
    own = pin.pin_path(name)
    if args.against is not None:
        old_path = pin.pin_path(args.against)
        if old_path is None:
            parser.error(f"--against {args.against!r} names no card in PIN_FILES "
                         f"({sorted(pin.PIN_FILES)})")
    elif own is not None:
        old_path = own
    else:
        old_path = pin.pin_path(pin.REFERENCE_CARD)
    print(f"card {name!r} (compute capability {card['compute_capability']}, "
          f"driver {card['driver']}), commit {card['commit']}")
    print(f"readings against {old_path.relative_to(ROOT)}"
          + ("" if old_path == own else
             f" ({args.against or pin.REFERENCE_CARD}; this card has no committed file)"))
    with np.load(old_path) as data:
        old = {k: data[k] for k in data.files}
    new = capture()
    report = readings(old, new)
    moved = [k for k, v in report.items() if v.get("status") != "held"]
    for key in sorted(report):
        v = report[key]
        if v.get("status") in ("moved", "held"):
            print(f"{key:18s} {v['status']:5s} max_abs={v['max_abs_difference']:.3e} "
                  f"rms_old={v['rms_of_old_entry']:.3e} ratio="
                  f"{(v['max_abs_over_rms'] or 0.0):.3e} "
                  f"differing={v['differing_words']}/{v['words']}")
        else:
            print(f"{key:18s} {v}")
    print(f"{len(moved)} of {len(report)} entries moved")
    target = own
    if args.write and target is None:
        target = pin.declared_pin_path(name)
        if target is None and not args.new_card:
            print(f"no committed capture and no PIN_FILES row for {name!r}; "
                  "pass --new-card FILE to create one (and add its row with "
                  "its reading)", file=sys.stderr)
            return 2
        if target is None:
            target = pin.PIN_DIR / args.new_card
    if args.readings is not None:
        args.readings.parent.mkdir(parents=True, exist_ok=True)
        args.readings.write_text(json.dumps(
            {"pin": str((target or old_path).relative_to(ROOT)),
             "against": str(old_path.relative_to(ROOT)), **card,
             "entries": report, "moved": len(moved), "total": len(report)},
            indent=2) + "\n", encoding="utf-8")
    if args.write:
        np.savez(target, **new)
        print(f"wrote {target}")
        if name not in pin.PIN_FILES:
            print(f"add to tests/_phase2_pin.py PIN_FILES: {name!r}: {target.name!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
