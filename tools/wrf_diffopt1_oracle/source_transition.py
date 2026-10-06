"""Write the measured source transition for the merged coordinate receipt.

``merged-gpu.json`` and its archive are the original capture of the 88
coordinate cases and stay immutable.  When an accepted change moves a source
that capture compiled or drives (``smag2d.cu``, ``diff_opt1.cu``,
``common.cuh``, ``constants.py``, ``dycore.py``, the loader), the receipt is
carried forward only by a fresh capture of the CURRENT tree that reproduces
every one of the original float32 words.  This tool compares such a capture
(``capture.py FIXTURE PREFIX --mode all``) with the original archive, word
for word, and writes ``measured-source-transition.json``, which
``tests/test_diff_opt1_wrf471.py`` seals by its SHA-256.

It refuses, and writes nothing, when any word differs, when the case or
field inventory moved, or when the capture was taken against a different
native archive: a moved word is a numerical change that needs its own
reading, not a transition.

    python tools/wrf_diffopt1_oracle/capture.py tests/data/wrf471_diff_opt1 OUT/merged-now --mode all
    python tools/wrf_diffopt1_oracle/source_transition.py tests/data/wrf471_diff_opt1 OUT/merged-now
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
from pathlib import Path

import numpy as np

SCHEMA = "gpuwm-diffopt1-measured-source-transition-v1"

#: Every input of the coordinate capture whose bytes the transition seals.
RAW_SOURCE_INPUTS = (
    "gpuwm/core/kernels/smag2d.cu",
    "gpuwm/core/kernels/diff_opt1.cu",
    "gpuwm/core/kernels/common.cuh",
    "gpuwm/core/kernels/__init__.py",
    "gpuwm/core/constants.py",
    "gpuwm/core/dycore.py",
    "gpuwm/core/kernels/rrtmg_aer3.cu",
    "gpuwm/core/kernels/rrtmg_smoke_manifest.cu",
    "gpuwm/core/kernels/rrtmg_legacy_prep.cu",
)

STAGED_FIXTURE_AND_TOOLS = (
    "tools/wrf_diffopt1_oracle/capture.py",
    "tools/wrf_diffopt1_oracle/model_case.py",
    "tests/data/wrf471_diff_opt1/wrf471.npz",
    "tests/data/wrf471_diff_opt1/wrf471.json",
    "tests/data/wrf471_diff_opt1/merged-gpu.npz",
    "tests/data/wrf471_diff_opt1/merged-gpu.json",
)

MODULES = ("smag2d", "diff_opt1")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def transition(fixture: Path, capture_prefix: Path, root: Path) -> dict:
    import cupy as cp
    from gpuwm.core.kernels import module_options, module_source

    receipt = json.loads((fixture / "merged-gpu.json").read_text())
    captured = json.loads(capture_prefix.with_suffix(".json").read_text())
    if captured.get("mode") != "all":
        raise SystemExit("the capture must be capture.py --mode all")
    if captured["native_archive_sha256"] != receipt["native_archive_sha256"]:
        raise SystemExit("the capture ran against a different native archive")
    if receipt["archive_sha256"] != _sha(fixture / "merged-gpu.npz"):
        raise SystemExit("merged-gpu.npz no longer matches its own receipt")
    current_sources = {name: hashlib.sha256(module_source(name).encode()).hexdigest()
                       for name in MODULES}
    if captured["module_source_sha256"] != current_sources:
        raise SystemExit("the capture was not taken on this tree's sources")
    rows, words, different = [], 0, 0
    with np.load(fixture / "merged-gpu.npz") as prior, \
            np.load(capture_prefix.with_suffix(".npz")) as current:
        if set(prior.files) != set(current.files):
            raise SystemExit(
                f"field inventory moved: {sorted(set(prior.files) ^ set(current.files))}")
        for name in sorted(prior.files):
            old, new = prior[name], current[name]
            if old.dtype != np.float32 or new.dtype != np.float32 or old.shape != new.shape:
                raise SystemExit(f"{name}: dtype or shape moved")
            moved = int(np.count_nonzero(old.view("u4") != new.view("u4")))
            rows.append({"name": name, "words": int(old.size), "different_words": moved,
                         "prior_array_sha256": hashlib.sha256(old.tobytes()).hexdigest(),
                         "current_array_sha256": hashlib.sha256(new.tobytes()).hexdigest()})
            words += int(old.size)
            different += moved
    if different:
        raise SystemExit(f"{different} of {words} words moved; this is a numerical "
                         "change, not a source transition")
    cases = len(receipt["cases"])
    fields = sum(len(row["fields"]) for row in receipt["cases"])
    return {
        "schema": SCHEMA,
        "measured_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "case_count": cases,
        "field_count": fields,
        "words": words,
        "different_words": different,
        "current_module_source_sha256": current_sources,
        "prior_module_source_sha256": receipt["module_source_sha256"],
        "original_production_archive_sha256": receipt["archive_sha256"],
        "native_archive_sha256": receipt["native_archive_sha256"],
        # Equal words make the original archive the production archive
        # still; the fresh capture's own zip bytes carry a timestamp.
        "new_capture_archive_sha256": receipt["archive_sha256"],
        "capture_file_sha256": _sha(capture_prefix.with_suffix(".npz")),
        "raw_source_inputs": [{"path": path, "sha256": _sha(root / path)}
                              for path in RAW_SOURCE_INPUTS],
        "staged_fixture_and_tools": {
            path: {"bytes": (root / path).stat().st_size, "sha256": _sha(root / path)}
            for path in STAGED_FIXTURE_AND_TOOLS},
        "module_options": {name: list(module_options(name)) for name in MODULES},
        "device": captured["device"],
        "nvrtc_version": list(cp.cuda.nvrtc.getVersion()),
        "cupy_version": cp.__version__,
        "different_by_field": rows,
        "forecast_runs": 0,
        "claim": ("source-matched coordinate capture against immutable original GPU "
                  "arrays; no observation skill claim"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("fixture", type=Path)
    parser.add_argument("capture_prefix", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    body = (json.dumps(transition(args.fixture, args.capture_prefix, root), indent=2)
            + "\n").encode("utf-8")
    out = args.fixture / "measured-source-transition.json"
    out.write_bytes(body)
    print(json.dumps({"written": str(out), "sha256": hashlib.sha256(body).hexdigest()}))


if __name__ == "__main__":
    main()
