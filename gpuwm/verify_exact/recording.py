"""The recording layout, for stock WRF recordings and for WOOF's own replays.

A recording root (see ``docs/dev/verify-exact.md``) holds::

    runs.csv                          one row per (build, case, combo) run
    MANIFEST.json                     unreliable_fields, among other things
    per-run-summary.csv               optional: clean_reference per run
    provenance/sweep/combos.json      the combo list the runs were made from
    inputs/<case>/<combo>/            wrfinput_d01, wrfbdy_d01 (the bytes WRF read)
    <build>/<case>/<combo>/           result.json, namelist.input, hashes.json,
                                      stepNNNN.json (+ stepNNNN.bin when kept)

A run directory's ``hashes.json`` holds ``fields[name] = {dtype, shape,
sha256: [one per frame]}`` and ``steps_per_frame``; frame ``f`` is model
step ``f * steps_per_frame``.  WOOF's replays are written in exactly this
form (:func:`write_run_record`), so either side reads through
:func:`load_run_record`.

Raw arrays may live outside the recording: ``overlays`` are further roots
with the same ``<build>/<case>/<combo>/stepNNNN.{json,bin}`` layout holding
any subset of fields (``tools/verify_exact_raw_subset.py`` cuts one).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from gpuwm.verify_exact.compare import FieldMeta, RunRecord

#: Builds whose recordings carry the Thompson table-index fix.  A run that
#: has one is scored against it in place of the stock build of the same
#: arithmetic, because stock 4.6.1's Thompson answer comes from an
#: out-of-bounds read (RECORD.md, fix 4).
FIXED_VARIANT_SUFFIX = "-tfix"

#: Statuses whose frames are WRF's own answer up to ``last_good_step``.
SCORABLE_STATUSES = ("OK", "CRASH", "NON-FINITE")


def _read_index(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))["fields"]


def _load_raw(directories: Sequence[Path], step: int) -> dict[str, np.ndarray] | None:
    """Raw arrays at ``step`` from the first directory that kept each field."""

    found: dict[str, np.ndarray] = {}
    for directory in directories:
        index = directory / f"step{step:04d}.json"
        blob = directory / f"step{step:04d}.bin"
        if not (index.is_file() and blob.is_file()):
            continue
        raw = blob.read_bytes()
        for entry in _read_index(index):
            if entry["name"] in found:
                continue
            dtype = np.dtype(entry["dtype"])
            count = int(entry["nbytes"]) // dtype.itemsize
            found[entry["name"]] = np.frombuffer(
                raw, dtype=dtype, count=count, offset=int(entry["offset"])
            ).reshape(entry["shape"])
    return found or None


def load_run_record(directory: Path, *, overlays: Sequence[Path] = (),
                    label: str = "") -> RunRecord:
    """Read one run directory's digests, with lazy raw-array access."""

    document = json.loads((directory / "hashes.json").read_text(encoding="utf-8"))
    per_frame = int(document.get("steps_per_frame", 1))
    fields: dict[str, FieldMeta] = {}
    digests: dict[int, dict[str, str | None]] = {}
    for name, entry in document["fields"].items():
        fields[name] = FieldMeta(name, str(entry["dtype"]), tuple(int(s) for s in entry["shape"]))
        for frame, digest in enumerate(entry["sha256"]):
            row = digests.setdefault(frame * per_frame, {})
            if digest is not None:  # null: the run did not write the field at this frame
                row[name] = digest
    places = [directory, *overlays]
    raw_steps = sorted({int(p.stem[4:]) for d in places if d.is_dir()
                        for p in d.glob("step*.bin")})
    cache: dict[int, Mapping[str, np.ndarray] | None] = {}

    def raw(step: int):
        if step not in cache:
            cache[step] = _load_raw(places, step)
        return cache[step]

    return RunRecord(fields, digests, raw, tuple(raw_steps), label)


def write_run_record(directory: Path, *, fields: Mapping[str, FieldMeta],
                     frames: Mapping[int, Mapping[str, str | None]], steps_per_frame: int,
                     raw: Mapping[int, Mapping[str, np.ndarray]] | None = None,
                     extra: Mapping[str, object] | None = None) -> None:
    """Write a run in the recording layout (``hashes.json`` + kept arrays).

    ``frames`` maps model step to ``{field: digest}``; steps must be whole
    multiples of ``steps_per_frame`` starting at 0.  A field missing at a
    frame is written as ``null`` and read back as absent.
    """

    directory.mkdir(parents=True, exist_ok=True)
    steps = sorted(frames)
    for step in steps:
        if step % steps_per_frame:
            raise ValueError(f"step {step} is not a multiple of {steps_per_frame}")
    count = (steps[-1] // steps_per_frame + 1) if steps else 0
    document = {
        "steps_per_frame": steps_per_frame, "frames": count,
        "last_step": steps[-1] if steps else -1,
        "fields": {
            name: {"dtype": meta.dtype, "shape": list(meta.shape),
                   "sha256": [frames.get(f * steps_per_frame, {}).get(name)
                              for f in range(count)]}
            for name, meta in fields.items()},
        **dict(extra or {}),
    }
    (directory / "hashes.json").write_text(json.dumps(document), encoding="utf-8")
    for step, arrays in sorted((raw or {}).items()):
        index, offset = [], 0
        with (directory / f"step{step:04d}.bin").open("wb") as stream:
            for name, values in arrays.items():
                meta = fields[name]
                array = np.ascontiguousarray(values, dtype=np.dtype(meta.dtype).newbyteorder("<"))
                payload = array.tobytes()
                stream.write(payload)
                index.append({"name": name, "dtype": array.dtype.str, "shape": list(array.shape),
                              "offset": offset, "nbytes": len(payload)})
                offset += len(payload)
        (directory / f"step{step:04d}.json").write_text(
            json.dumps({"step": step, "fields": index}), encoding="utf-8")


@dataclass(frozen=True)
class ReferenceRun:
    """One build's recording of one run."""

    build: str
    case: str
    combo: str
    status: str
    steps_requested: int
    steps_per_frame: int
    last_good_step: int
    designated: bool
    clean: str
    detail: str
    directory: Path
    #: Steps at which the recording kept raw arrays (runs.csv ``raw_steps``),
    #: whether or not this copy of the recording still holds them.
    raw_steps: tuple[int, ...] = ()

    @property
    def scorable(self) -> bool:
        return (self.status in SCORABLE_STATUSES
                and (self.directory / "hashes.json").is_file())

    @property
    def fixed_variant(self) -> bool:
        return self.build.endswith(FIXED_VARIANT_SUFFIX)


@dataclass
class ReplayRun:
    """One (case, combo) WOOF must replay, with every reference recorded for it."""

    case: str
    combo: str
    combo_record: Mapping[str, object]
    inputs: Path | None
    namelist: Path | None
    references: dict[str, ReferenceRun] = field(default_factory=dict)

    @property
    def key(self) -> str:
        return f"{self.combo}/{self.case}"

    @property
    def stratum(self) -> str:
        return str(self.combo_record.get("stratum", ""))

    @property
    def steps(self) -> int:
        refs = list(self.references.values())
        if refs:
            return max(r.steps_requested for r in refs)
        overrides = self.combo_record.get("case_overrides") or {}
        return int(overrides.get("steps", 20))

    @property
    def steps_per_frame(self) -> int:
        refs = list(self.references.values())
        return refs[0].steps_per_frame if refs else 1

    def scorable(self) -> dict[str, ReferenceRun]:
        return {b: r for b, r in self.references.items() if r.scorable}

    def referee(self, arithmetic: str) -> ReferenceRun | None:
        """The build a PASS/FAIL is decided against.

        ``arithmetic`` is ``scalar`` or ``strict``.  The Thompson-fixed
        variant of that build wins whenever it was recorded: it differs
        from stock only where stock read its table out of bounds, so where
        the read never happened the two are the same words, and where it
        did only the fixed build is a sound reference.
        """

        scorable = self.scorable()
        fixed = scorable.get(arithmetic + FIXED_VARIANT_SUFFIX)
        return fixed if fixed is not None else scorable.get(arithmetic)

    def referee_sound(self, referee: ReferenceRun) -> bool:
        """Whether the recording vouches for ``referee`` as a 0 ULP reference."""

        return self.referee_soundness(referee) == "sound"

    def referee_soundness(self, referee: ReferenceRun) -> str:
        """``sound``, or why ``referee`` cannot decide a PASS.

        ``clean_reference`` (per-run-summary.csv) is ``yes`` for a sound
        stock run and ``yes, with the Thompson fix build`` for one sound
        only against the fixed build.  A run with no entry is unknown, and
        unknown is not sound: the recording has not vouched for it.  A
        referee that did not finish (CRASH, NON-FINITE) is never sound,
        whatever the column says; RECORD.md's first condition is that the
        run is OK.
        """

        if referee.status != "OK":
            return (f"incomplete: the {referee.build} referee ended {referee.status} "
                    f"after step {referee.last_good_step}")
        clean = referee.clean.strip().lower()
        if not clean:
            return f"unknown: no clean_reference entry for {referee.combo}/{referee.case}"
        if clean == "yes" or (clean.startswith("yes") and referee.fixed_variant):
            return "sound"
        return f"unsound: clean_reference is {referee.clean.strip()!r} for {referee.build}"

    def wants_repeat(self) -> bool:
        return "repeat_run_0ulp" in (self.combo_record.get("checks") or ())


def _truthy(text: str) -> bool:
    return str(text).strip().lower() in ("true", "1", "yes")


class Recording:
    """A recording root, read once."""

    def __init__(self, root: Path, *, combos_file: Path | None = None,
                 summary_file: Path | None = None, overlays: Sequence[Path] = ()):
        self.root = Path(root)
        self.overlays = tuple(Path(p) for p in overlays)
        manifest_path = self.root / "MANIFEST.json"
        manifest = (json.loads(manifest_path.read_text(encoding="utf-8"))
                    if manifest_path.is_file() else {})
        #: Fields the recording's own repeat test found to hold uninitialised
        #: memory; never compared (RECORD.md: RQIBLTEN).
        self.unreliable_fields = tuple(manifest.get("unreliable_fields") or ())
        combos_path = combos_file or self.root / "provenance" / "sweep" / "combos.json"
        if not combos_path.is_file():
            raise ValueError(f"{self.root}: no combo list at {combos_path}; pass --combos-file")
        document = json.loads(combos_path.read_text(encoding="utf-8"))
        self.combos = {str(c["id"]): c for c in document["combos"]}
        summary_path = summary_file or self.root / "per-run-summary.csv"
        self.summary = {}
        if summary_path.is_file():
            with summary_path.open(newline="", encoding="utf-8") as stream:
                for row in csv.DictReader(stream):
                    self.summary[(row["case"], row["combo"])] = row
        self.runs = self._runs()

    def _runs(self) -> dict[tuple[str, str], ReplayRun]:
        table = self.root / "runs.csv"
        if not table.is_file():
            raise ValueError(f"{self.root}: no runs.csv; this is not a recording root")
        runs: dict[tuple[str, str], ReplayRun] = {}
        with table.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        for row in rows:
            case, combo = row["case"], row["combo"]
            key = (case, combo)
            if key not in runs:
                inputs = self.root / "inputs" / case / combo
                runs[key] = ReplayRun(
                    case, combo, self.combos.get(combo, {"id": combo}),
                    inputs if (inputs / "wrfinput_d01").is_file() else None, None)
            directory = self.root / row["build"] / case / combo
            summary = self.summary.get(key, {})
            ref = ReferenceRun(
                build=row["build"], case=case, combo=combo, status=row["status"],
                steps_requested=int(row["steps_requested"] or 20),
                steps_per_frame=int(row["steps_per_frame"] or 1),
                last_good_step=int(row["last_good_step"] or 0),
                designated=_truthy(row.get("stock461_is_designated_reference", "")),
                clean=summary.get("clean_reference", ""),
                detail=(row.get("refusal") or row.get("crash") or "")[:300],
                directory=directory,
                # runs.csv lists the requested raw steps; only those on a
                # history frame were stored (a 7.5 s combo with a frame
                # every 2 steps lists 1 and 5 but stored neither).
                raw_steps=tuple(int(s) for s in (row.get("raw_steps") or "").split()
                                if s.isdigit() and int(s) > 0
                                and int(s) % int(row["steps_per_frame"] or 1) == 0))
            runs[key].references[row["build"]] = ref
            if runs[key].namelist is None and (directory / "namelist.input").is_file():
                runs[key].namelist = directory / "namelist.input"
        return runs

    def select(self, *, combos: Iterable[str] = (), cases: Iterable[str] = (),
               strata: Iterable[str] = ()) -> list[ReplayRun]:
        combos, cases, strata = set(combos), set(cases), set(strata)
        chosen = []
        for (case, combo), run in sorted(self.runs.items(), key=lambda kv: (kv[0][1], kv[0][0])):
            if combos and combo not in combos:
                continue
            if cases and case not in cases:
                continue
            if strata and run.stratum not in strata:
                continue
            chosen.append(run)
        return chosen

    def reference_record(self, ref: ReferenceRun):
        overlays = [o / ref.build / ref.case / ref.combo for o in self.overlays]
        return load_run_record(ref.directory, overlays=overlays, label=ref.build)
