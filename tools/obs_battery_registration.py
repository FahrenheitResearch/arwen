#!/usr/bin/env python3
"""Emit one version of the observation battery's registration document.

Spec section 9.2 makes the registration a *version series*, not a file that
gets edited: "Changes after first data are a new registered version beside
the old, with both reported -- never an edit."  This tool emits one version.
It never rewrites another.

Two things it computes rather than accepts, because both are facts about this
machine rather than choices:

* **the reader's content pin.**  A distribution version is a label somebody
  typed into packaging metadata -- and on this box that label disagrees with
  the module's own ``__version__`` attribute, which is an upstream note and
  not a gate.  What actually answers a ``getvar`` call is a built artifact,
  so the pin is the science core's **source tree commit** and the **SHA-256
  of the compiled extension**, recorded beside the distribution version and
  inside the hashed parameter block.
* **the registration digest**, from :mod:`gpuwm.verify.obs.registration`'s
  own canonical hash, over the whole parameter block.

The case set and the arm set are transcribed from the battery spec and
carried as data; this tool names no case in its own code.

    python tools/obs_battery_registration.py --version v1 \\
        --out docs/public/receipts/obsbattery/registration/...json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from gpuwm.verify.obs import model_source, registration

#: The seven case days of the spec's section 1.2 menu, in menu order, with
#: the phenomenon each was selected for.  Data, transcribed once.
CASE_DAYS: tuple[tuple[str, str, str], ...] = (
    ("B-01", "2021-12-10", "nocturnal long-track supercell outbreak, mid-South"),
    ("B-02", "2023-03-31", "large mixed-mode outbreak, mid-Mississippi Valley"),
    ("B-03", "2024-04-27", "nocturnal supercell outbreak, central/southern Oklahoma"),
    ("B-04", "2024-05-21", "daytime cyclic supercell, southwest/central Iowa"),
    ("B-05", "2024-07-15", "derecho with embedded QLCS tornadoes, n. Illinois"),
    ("B-06", "2025-03-14", "high-shear early-spring outbreak, MO/AR/MS"),
    ("B-07", "2025-05-16", "violent-outbreak day, St. Louis metro / s. Illinois / KY"),
)

#: Which case day carries a committed entry receipt, and is therefore the one
#: whose init instant is fixed rather than defaulted.
SHAKEDOWN_DAY = "2024-05-21"

#: The arms of the spec's section 6.1 table.
ARMS: tuple[dict[str, object], ...] = (
    {"arm_id": "F", "engine": "arwen", "runs": 2,
     "role": "faithful; the dual-run pair is the standing byte screen"},
    {"arm_id": "P-L3", "engine": "arwen", "runs": 2,
     "role": "patched: Shin-Hong scale-aware PBL"},
    {"arm_id": "P-L4", "engine": "arwen", "runs": 2,
     "role": "patched: 6th-order damping exempted from moist scalars"},
    {"arm_id": "P-all", "engine": "arwen", "runs": 2,
     "role": "full patchset v1; reported, NEVER a promotion basis"},
    {"arm_id": "P-L5", "engine": "arwen", "runs": 2, "conditional": True,
     "role": "SASE overlay; scored on zero cases until its entry gate passes"},
    {"arm_id": "W", "engine": "wrf-v4.6.1", "runs": 1,
     "role": "stock control on the pinned campaign build, never rebuilt"},
    {"arm_id": "W-twin", "engine": "wrf-v4.6.1", "runs": 1,
     "role": "perturbed control; prices chaos at each lead"},
    {"arm_id": "O1", "engine": "none", "runs": 0,
     "role": "HRRR-operational context row; never a promotion input"},
    {"arm_id": "persistence", "engine": "none", "runs": 0,
     "role": "zero-skill floor; never a promotion input"},
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(directory: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(directory), *arguments],
        capture_output=True, text=True, check=False)
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


def reader_pin() -> dict[str, object]:
    """The science core, pinned by content rather than by packaging."""
    from importlib import metadata

    core = model_source.require_science_core()
    module = Path(core.__file__).resolve()
    package = module.parent
    artifacts = []
    for path in sorted(package.iterdir()):
        if path.suffix.lower() in {".pyd", ".so", ".dylib"} or path.name == "__init__.py":
            artifacts.append({"name": path.name, "bytes": path.stat().st_size,
                              "sha256": _sha256(path)})
    tree = package
    while tree != tree.parent and not (tree / ".git").exists():
        tree = tree.parent
    has_git = (tree / ".git").exists()
    return {
        "science_core": "wrf-rust",
        "distribution_version": metadata.version("wrf-rust"),
        "pinned_distribution_version": model_source.PINNED_WRF_RUST_VERSION,
        "module_version_attribute": str(getattr(core, "__version__", "")),
        "module_version_attribute_note": (
            "the module's own __version__ attribute lags the distribution "
            "metadata on this install; that is an upstream packaging note and "
            "is NOT a gate -- the content pins below are"),
        "source_tree": str(tree) if has_git else "",
        "source_tree_commit": _git(tree, "rev-parse", "HEAD") if has_git else "",
        "source_tree_clean": (
            _git(tree, "status", "--short", "--untracked-files=no") == ""
            if has_git else False),
        "artifacts": artifacts,
        "pin_rule": (
            "the reader is pinned by CONTENT: the source tree commit the "
            "editable install resolves to, plus the SHA-256 of the built "
            "extension that answers every diagnostic call. A distribution "
            "version alone cannot detect a rebuilt artifact under an "
            "unchanged label."),
    }


def build(*, version: str, evaluator_commit: str, ratification_reference: str,
          delegation_reference: str, overrule_window: str,
          include_reader: bool, supersedes: str, basis: str) -> dict:
    cases = []
    for case_id, day, phenomenon in CASE_DAYS:
        fixed = day == SHAKEDOWN_DAY
        cases.append({
            "case_id": f"case-{day.replace('-', '')}",
            "menu_row": case_id,
            "case_day": day,
            "phenomenon": phenomenon,
            "init_time": f"{day}T12:00:00",
            "init_time_status": (
                "fixed by its committed entry receipt" if fixed else
                "section 1.3 default (cycle nearest 12 UTC), pending this "
                "case's own entry receipt, which may register 15 or 18 UTC "
                "for a nocturnal case"),
            "entry_receipt": (
                "docs/public/receipts/obsbattery/B5-CASE-ENTRY-20240521.md"
                if fixed else ""),
            "shakedown": fixed,
        })
    promotion = registration.promotion_parameters(
        ratification_reference=ratification_reference,
        delegation_reference=delegation_reference,
        overrule_window=overrule_window)
    document = registration.make_registration(
        evaluator_commit=evaluator_commit,
        reflectivity=registration.reflectivity_parameters(),
        surface=registration.surface_parameters(),
        precipitation=registration.precipitation_parameters(),
        promotion=promotion,
        cases=cases,
        arms=[dict(arm) for arm in ARMS],
        twin={
            "rung": "rung-1-one-ulp-theta",
            "perturbation": ("one documented FP ULP on the wrfinput theta "
                             "field (tools/n5s/perturb_ulp.py)"),
            "escalation": ("a twin band of exactly zero on the primary score "
                           "escalates to rung 2 (seeded FP32 noise, 0.01 K, "
                           "interior theta) as ONE re-registered motion "
                           "before any campaign case is scored, never per "
                           "case"),
            "band_statistic": ("median over cases of the paired absolute "
                               "difference; one pair per case, stated as "
                               "exactly that"),
        },
        reader=reader_pin() if include_reader else None)
    document["registration_version"] = str(version)
    document["supersedes"] = str(supersedes)
    document["ratification_basis"] = str(basis)
    return document


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True)
    parser.add_argument("--evaluator-commit", required=True)
    parser.add_argument("--ratification-reference", default="")
    parser.add_argument("--delegation-reference", default="")
    parser.add_argument("--overrule-window", default="")
    parser.add_argument("--supersedes", default="")
    parser.add_argument("--basis", default="")
    parser.add_argument("--no-reader-pin", action="store_true")
    parser.add_argument("--out", required=True, type=Path)
    arguments = parser.parse_args()

    document = build(
        version=arguments.version,
        evaluator_commit=arguments.evaluator_commit,
        ratification_reference=arguments.ratification_reference,
        delegation_reference=arguments.delegation_reference,
        overrule_window=arguments.overrule_window,
        include_reader=not arguments.no_reader_pin,
        supersedes=arguments.supersedes, basis=arguments.basis)
    registration.validate_registration(document)
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    arguments.out.write_text(
        json.dumps(document, indent=1, sort_keys=True) + "\n",
        encoding="utf-8")
    print(f"{arguments.version}: rule_status={document['rule_status']} "
          f"registration_sha256={document['registration_sha256']}")
    print(f"  -> {arguments.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
