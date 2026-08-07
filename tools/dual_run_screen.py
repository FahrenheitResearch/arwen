"""The dual-run byte comparison, on a no-ECC card.

The 5090 has no ECC, so a bit that flips in VRAM produces a plausible
forecast and nothing says otherwise. Running the identical configuration
twice and comparing bytes is the standing corruption detector; a
mismatch is a corruption finding that stops the claim, full stop.

The screen this reproduces is the one the shipped 250 m demonstration
committed (`docs/superpowers/receipts/les/inflow-fetch/dual_run_screen.txt`)
and reports the same lines. That script lived in a measurement-wave
scratch directory and was rewritten from scratch each campaign, which is
how the same misclassification could be re-made twice; this is that
re-implementation, in the tree, with a test.

Three comparisons, because they fail differently:

* **wrfout bytes.** The forecast itself. Byte-for-byte over every frame.
* **final-state digests.** The run receipts' own state digests, which
  cover arrays the history files do not carry.
* **non-environmental receipt scalars.** The receipt blocks with
  `ENVIRONMENTAL_FIELDS` excluded -- wall times, device names, host
  paths and the like legitimately differ between two runs and are not
  evidence of anything. The partition is the one
  `gpuwm.verify.cases.convective_boundary_layer` already publishes, so
  the exclusion list is not invented here.

An exclusion is never a silent drop. Every excluded key whose two arms
DISAGREE is printed, with the reason it was excluded, above the verdict
line -- so a reader sees exactly what the screen chose not to count and
can overrule it. An exclusion nobody can see is an exclusion nobody can
audit, and the difference between that and a hidden corruption is a
reader's trust.

Exit 0 PASS, 1 MISMATCH, 2 the screen could not be evaluated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def _digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _wrfouts(root: Path) -> dict[str, Path]:
    return {p.relative_to(root).as_posix(): p
            for p in sorted(root.rglob("wrfout_d*"))
            if p.is_file()}


def _receipts(root: Path) -> dict[str, dict]:
    out = {}
    for p in sorted(root.rglob("*receipt*.json")):
        try:
            out[p.relative_to(root).as_posix()] = json.loads(
                p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
    return out


def _environmental(repo: Path) -> frozenset[str]:
    sys.path.insert(0, str(repo))
    try:
        from gpuwm.verify.cases.convective_boundary_layer import (
            ENVIRONMENTAL_FIELDS)
        return frozenset(ENVIRONMENTAL_FIELDS)
    except Exception:                                    # noqa: BLE001
        # Fail LOUD, not open: an empty exclusion set would make every
        # wall-clock difference look like corruption, and a guessed one
        # would hide a real difference behind a name nobody registered.
        raise SystemExit(
            "could not import ENVIRONMENTAL_FIELDS from the tree; the "
            "screen refuses to invent its own exclusion list")


#: Run-local metadata, not science: absolute paths differ by run
#: directory, checkpoint bookkeeping names a run's own files, and the
#: background poller's sample COUNT differs by wall-clock jitter.
#: Content identity is carried by the digest keys, compared separately.
_RUN_LOCAL_LEAVES = frozenset({"path", "samples", "last_checkpoint"})

#: The block of a run receipt that holds memory telemetry.  Scoping the
#: rule below to this block is the point: an ``_observed`` suffix or a
#: ``peak_bytes`` leaf anywhere else in a receipt keeps being screened.
_MEMORY_BLOCK = "memory"

#: Leaf names inside a ``memory`` block that record an OBSERVATION of
#: the host's allocator rather than a property of the forecast.
#:
#: ``peak_bytes`` is a maximum over an asynchronously sampled series --
#: a background thread polls the allocator every 50 ms -- so its value
#: depends on where the poller's samples landed relative to transient
#: allocations, not only on which allocations happened.  Two runs whose
#: allocation trajectories are identical can and do report different
#: high-water marks; the tree already classifies the device-wide figure
#: as environmental for the neighbouring reason (co-tenant processes),
#: and the sampled peak of even the in-process pool carries the same
#: defect from the sampling itself.
#:
#: What is NOT here matters as much.  ``preflight_alloc_estimate_bytes``
#: is computed from the configuration, ``interval_seconds`` and
#: ``mechanism`` and ``scope`` are constants, and an instantaneous pool
#: reading is a property of the run -- all four stay screened, so the
#: memory block does not fall out of the comparison wholesale.
_OBSERVED_MEMORY_LEAF_SUFFIX = "_observed"
_OBSERVED_MEMORY_LEAVES = frozenset({"peak_bytes"})


def exclusion_reason(key: str, environmental: frozenset[str]) -> str | None:
    """Why this flattened receipt key is not compared, or ``None``.

    ``None`` means the key is on the determinism-comparison surface and
    a disagreement there is a corruption finding.  Every other return is
    a reason a reader can weigh, and the caller prints it beside the two
    values whenever an excluded key actually disagrees.
    """

    parts = key.split(".")
    leaf = parts[-1].split("[")[0]
    if leaf in environmental:
        return "registered ENVIRONMENTAL_FIELDS"
    if leaf in _RUN_LOCAL_LEAVES:
        return "run-local metadata"
    if "timing_seconds" in key:
        # Wall-clock timings differ between arms whenever the host is
        # under different load; they carry no state.
        return "wall-clock timing"
    if _MEMORY_BLOCK in parts[:-1] and (
            leaf.endswith(_OBSERVED_MEMORY_LEAF_SUFFIX)
            or leaf in _OBSERVED_MEMORY_LEAVES):
        return "sampled memory high-water"
    return None


def _flatten(doc, prefix="", into=None):
    into = {} if into is None else into
    if isinstance(doc, dict):
        for key, value in doc.items():
            _flatten(value, f"{prefix}.{key}" if prefix else str(key), into)
    elif isinstance(doc, list):
        for index, value in enumerate(doc):
            _flatten(value, f"{prefix}[{index}]", into)
    else:
        into[prefix] = doc
    return into


def screen(run_a: Path, run_b: Path, repo: Path) -> tuple[bool, list[str]]:
    environmental = _environmental(repo)
    lines: list[str] = []
    ok = True

    a_files, b_files = _wrfouts(run_a), _wrfouts(run_b)
    lines.append(f"frames: A={len(a_files)} B={len(b_files)}")
    if not a_files:
        return False, lines + ["no wrfout files in A; nothing to compare"]
    if set(a_files) != set(b_files):
        ok = False
        lines.append(f"file sets differ: only in A "
                     f"{sorted(set(a_files) - set(b_files))}, only in B "
                     f"{sorted(set(b_files) - set(a_files))}")

    shared = sorted(set(a_files) & set(b_files))
    differing = [name for name in shared
                 if _digest(a_files[name]) != _digest(b_files[name])]
    lines.append(f"wrfout files compared: {len(shared)}; byte-identical: "
                 f"{len(shared) - len(differing)}; differing: "
                 f"{len(differing)}")
    if differing:
        ok = False
        for name in differing[:10]:
            lines.append(f"  DIFFERS: {name}")

    a_receipts, b_receipts = _receipts(run_a), _receipts(run_b)
    shared_receipts = sorted(set(a_receipts) & set(b_receipts))

    digest_keys = 0
    digest_diff = 0
    scalar_diff = 0
    excluded: list[str] = []
    reason_counts: dict[str, int] = {}
    for name in shared_receipts:
        flat_a = _flatten(a_receipts[name])
        flat_b = _flatten(b_receipts[name])
        for key in sorted(set(flat_a) & set(flat_b)):
            reason = exclusion_reason(key, environmental)
            if reason is not None:
                # Excluded, and SHOWN when it disagrees. A screen that
                # silently drops a differing key is indistinguishable
                # from one that never looked at it.
                if flat_a[key] != flat_b[key]:
                    reason_counts[reason] = reason_counts.get(reason, 0) + 1
                    excluded.append(f"  EXCLUDED [{reason}]: {name}:{key} "
                                    f"{flat_a[key]!r} != {flat_b[key]!r}")
                continue
            if "digest" in key or "sha256" in key:
                digest_keys += 1
                if flat_a[key] != flat_b[key]:
                    digest_diff += 1
                    ok = False
                    lines.append(f"  DIGEST DIFFERS: {name}:{key}")
            elif flat_a[key] != flat_b[key]:
                scalar_diff += 1
                ok = False
                if scalar_diff <= 10:
                    lines.append(f"  SCALAR DIFFERS: {name}:{key} "
                                 f"{flat_a[key]!r} != {flat_b[key]!r}")

    lines.append(f"final-state digests compared: {digest_keys}; differing: "
                 f"{digest_diff}")
    lines.append(f"non-environmental scalar blocks differing: {scalar_diff}")
    lines.append(f"excluded as environmental, and differing: {len(excluded)}")
    if reason_counts:
        lines.append("excluded by reason: " + ", ".join(
            f"{reason} {count}"
            for reason, count in sorted(reason_counts.items())))
    lines.extend(excluded)
    lines.append(f"DUAL-RUN SCREEN: {'PASS (bit-identical)' if ok else
                                    'MISMATCH -- CORRUPTION FINDING'}")
    return ok, lines


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-a", type=Path, required=True)
    ap.add_argument("--run-b", type=Path, required=True)
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    for root in (args.run_a, args.run_b):
        if not root.is_dir():
            print(f"no such run directory: {root}", file=sys.stderr)
            return 2

    ok, lines = screen(args.run_a, args.run_b, args.repo)
    text = "\n".join(lines) + "\n"
    print(text, end="")
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8", newline="\n")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
