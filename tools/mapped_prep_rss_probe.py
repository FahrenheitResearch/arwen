"""Measure WHICH structure a multi-valid-time mapped prep keeps resident.

Peak RSS alone says a preparation is expensive; it does not say what is
holding the bytes.  This probe drives the real composed-source decode --
:func:`gpuwm.mapped_composition.decode_composed_source`, the same call
``gpuwm prep`` makes -- with an RSS sampler running, then prices every
structure the call leaves behind against the number of valid times:

* the canonical frames the decode returns,
* the regular-source snapshots packed from them,
* how much of the snapshot array bytes are VIEWS onto frame arrays and
  how much is freshly allocated (the zero-filled hydrometeors are),

so a rise in the sampled trace can be attributed to an object instead of
guessed at.  Nothing here is a mock: the authorities are the packaged
profile's own documents, the bytes are whatever files are named, and the
decode is the shipped route.

    python tools/mapped_prep_rss_probe.py \
        --profile <packaged-profile-name> \
        --input-manifest inputs.json \
        --input FILE [--input FILE ...] \
        --supplement FILE [--supplement FILE ...] \
        --scratch DIR

``--input-manifest`` is the document a real ``gpuwm prep
--author-input-manifest`` run already wrote for these same files; the
probe verifies it exactly as the route does.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _rss_bytes() -> int:
    """This process's resident set, or 0 where the OS will not say."""

    try:
        with open("/proc/self/status", encoding="ascii") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    try:  # Windows and macOS: psutil if the environment has it.
        import psutil  # type: ignore

        return int(psutil.Process().memory_info().rss)
    except Exception:  # pragma: no cover - probe-only fallback
        return 0


def _hwm_bytes() -> int:
    try:
        with open("/proc/self/status", encoding="ascii") as handle:
            for line in handle:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return 0


class _Sampler:
    """Background RSS sampler with a phase label per sample."""

    def __init__(self, interval: float = 0.1) -> None:
        self.interval = interval
        self.rows: list[tuple[float, int, str]] = []
        self.phase = "start"
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._started = time.perf_counter()

    def _run(self) -> None:
        while not self._stop.is_set():
            self.rows.append(
                (time.perf_counter() - self._started, _rss_bytes(), self.phase))
            self._stop.wait(self.interval)

    def __enter__(self) -> "_Sampler":
        self._thread.start()
        return self

    def __exit__(self, *_exc) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)

    def peak(self, phase: str | None = None) -> int:
        rows = [row for row in self.rows if phase is None or row[2] == phase]
        return max((row[1] for row in rows), default=0)


def _gib(value: int) -> float:
    return round(value / (1024.0 ** 3), 3)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_owner(array) -> int:
    """Identity of the buffer an array reads, following the view chain."""

    base = array
    while getattr(base, "base", None) is not None:
        base = base.base
    return id(base)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mapped_prep_rss_probe",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--profile", required=True,
                        help="packaged profile name to decode through")
    parser.add_argument("--input-manifest", required=True,
                        help="the sealed composition-inputs manifest")
    parser.add_argument("--input", action="append", default=[], required=True,
                        help="one primary source file (repeatable)")
    parser.add_argument("--supplement", action="append", default=[],
                        help="one supplement file for the profile's data role")
    parser.add_argument("--scratch", default=None,
                        help="directory to place the engine compose scratch in")
    parser.add_argument("--trace", default=None,
                        help="write the sampled RSS trace here as TSV")
    parser.add_argument("--json", default=None,
                        help="write the measured report here")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)

    from gpuwm.mapped_composition import decode_composed_source
    from gpuwm.source_authorities import (packaged_authorities,
                                          packaged_contributing_mappings,
                                          packaged_profile)

    profile = packaged_profile(args.profile)
    authorities = packaged_authorities(args.profile)
    contributing = packaged_contributing_mappings(args.profile)
    data_role = str(profile["data_role"])
    provenance_role = str(profile["provenance_role"])

    manifest = Path(args.input_manifest).resolve()
    primary = [Path(value).resolve() for value in args.input]
    supplements = {data_role: [Path(v).resolve() for v in args.supplement]}

    report: dict[str, object] = {
        "profile": args.profile,
        "input_count": len(primary),
        "supplement_count": len(args.supplement),
    }

    with _Sampler() as sampler:
        baseline = _rss_bytes()
        sampler.phase = "decode"
        started = time.perf_counter()
        bundle = decode_composed_source(
            authorities["composition"],
            authorities["mapping"],
            primary,
            supplements,
            {provenance_role: authorities["provenance"]},
            input_manifest=manifest,
            input_manifest_sha256=_sha256(manifest),
            contributing_mappings=contributing,
            scratch_destination=args.scratch,
        )
        decode_seconds = time.perf_counter() - started
        sampler.phase = "after-decode"
        time.sleep(0.3)
        after_decode = _rss_bytes()

        frame_bytes = sum(
            int(field.values.nbytes)
            for frame in bundle.frames for field in frame.fields.values())
        frame_owners = {
            _array_owner(field.values): int(field.values.nbytes)
            for frame in bundle.frames for field in frame.fields.values()}

        sampler.phase = "snapshots"
        started = time.perf_counter()
        snapshots = bundle.regular_snapshots()
        snapshot_seconds = time.perf_counter() - started
        sampler.phase = "after-snapshots"
        time.sleep(0.3)
        after_snapshots = _rss_bytes()

        shared = 0
        fresh = 0
        for snapshot in snapshots:
            for array in snapshot.fields.values():
                owner = _array_owner(array)
                if owner in frame_owners:
                    shared += int(array.nbytes)
                else:
                    fresh += int(array.nbytes)

        report.update({
            "valid_times": len(bundle.frames),
            "grid": {
                "ny": int(bundle.frames[0].latitude.size),
                "nx": int(bundle.frames[0].longitude.size),
                "vertical": int(bundle.frames[0].vertical_values.size),
            },
            "seconds": {
                "decode_composed_source": round(decode_seconds, 2),
                "regular_snapshots": round(snapshot_seconds, 2),
            },
            "rss_gib": {
                "baseline": _gib(baseline),
                "after_decode": _gib(after_decode),
                "after_snapshots": _gib(after_snapshots),
                "peak_during_decode": _gib(sampler.peak("decode")),
                "peak_during_snapshots": _gib(sampler.peak("snapshots")),
                "process_high_water": _gib(_hwm_bytes()),
            },
            "array_gib": {
                "frames_total": _gib(frame_bytes),
                "frames_per_valid_time": _gib(
                    frame_bytes // max(1, len(bundle.frames))),
                "snapshots_sharing_frame_buffers": _gib(shared),
                "snapshots_freshly_allocated": _gib(fresh),
                "snapshots_fresh_per_valid_time": _gib(
                    fresh // max(1, len(snapshots))),
            },
        })

    if args.trace:
        with open(args.trace, "w", encoding="ascii") as handle:
            for stamp, rss, phase in sampler.rows:
                handle.write(f"{stamp:.2f}\t{rss}\t{phase}\n")
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.json:
        Path(args.json).write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
