"""Write gpuwm/data/wrf_namelist/physics_options_v4.7.1.json from a WRF tree.

The option VALUES come from WRF's own Registry ``package`` lines (every
``Registry/*`` file, so registry.sbm's spectral-bin schemes are included);
the LABELS come from WRF's ``run/README.namelist`` ``= N, label`` rows.  A
value with a package and no README label is labelled with its package name.

Usage: python tools/wrf_physics_options/extract.py WRF_SOURCE_DIR OUT_JSON

The published table (docs/public/WRF-OPTIONS.md) is generated from this JSON
by tools/build_wrf_option_coverage.py, which asks the importer about every
row; nothing here decides what gpuwm runs.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

#: The scheme selectors whose values WRF declares as Registry packages.
SCHEME_SELECTORS = (
    "mp_physics", "cu_physics", "shcu_physics", "bl_pbl_physics",
    "sf_sfclay_physics", "sf_surface_physics", "sf_urban_physics",
    "sf_ocean_physics", "ra_lw_physics", "ra_sw_physics",
)

#: On/off and enumerated &physics switches with no Registry package.  Each
#: value is WRF's own (Registry.EM_COMMON rconfig description or
#: README.namelist); only the non-default values are listed.
SWITCHES = {
    "sst_update": [(1, "time-varying sea-surface temperature, sea ice, albedo and vegetation from wrflowinp")],
    "sst_skin": [(1, "skin sea-surface temperature (diurnal warm layer and cool skin)")],
    "tmn_update": [(1, "deep-soil temperature updated from the skin temperature (lagday)")],
    "sf_lake_physics": [(1, "CLM lake model")],
    "windfarm_opt": [(1, "Fitch wind farm"), (2, "Ma et al. (2022) wind farm with subgrid wakes")],
    "cu_rad_feedback": [(True, "cumulus sub-grid cloud fed to radiation")],
    "icloud": [(3, "Thompson-Eidhammer cloud fraction (cal_cldfra3)")],
}

#: Labels for package values WRF's README.namelist does not list.
LABEL_FALLBACK = {
    "omlscheme": "1-D ocean mixed layer",
    "pwp3dscheme": "3-D Price-Weller-Pinkel ocean",
}

_PACKAGE = re.compile(r"^package\s+(\S+)\s+([a-z_0-9]+)==(-?\d+)\b")


def _packages(root: Path) -> dict[str, dict[int, str]]:
    out: dict[str, dict[int, str]] = {key: {} for key in SCHEME_SELECTORS}
    for path in sorted((root / "Registry").iterdir()):
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            match = _PACKAGE.match(line)
            if match and match.group(2) in out:
                out[match.group(2)].setdefault(int(match.group(3)), match.group(1))
    return out


def _labels(readme: str, key: str) -> dict[int, str]:
    lines = readme.splitlines()
    head = re.compile(rf"^ {re.escape(key)}\b")
    row = re.compile(r"^\s+=\s*(-?\d+)\s*[,:]\s*(.*\S)\s*$")
    out: dict[int, str] = {}
    for index, line in enumerate(lines):
        if not head.match(line):
            continue
        first = re.search(r"=\s*(-?\d+)\s*[,:]\s*(.*\S)\s*$", line)
        if first:
            out.setdefault(int(first.group(1)), first.group(2))
        for follow in lines[index + 1:]:
            if follow.strip() and not follow.startswith(" " * 20):
                break
            match = row.match(follow)
            if match:
                out.setdefault(int(match.group(1)), match.group(2))
        if out:
            break
    return out


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: list[str]) -> int:
    root, out = Path(argv[1]), Path(argv[2])
    readme_path = root / "run" / "README.namelist"
    readme = readme_path.read_text(encoding="utf-8", errors="replace")
    packages = _packages(root)
    selectors = {}
    for key in SCHEME_SELECTORS:
        labels = _labels(readme, key)
        selectors[key] = [
            {"value": value, "package": package,
             "label": labels.get(value, LABEL_FALLBACK.get(package, package))}
            # -1 is WRF's "not set" sentinel (mpnotset), not an option.
            for value, package in sorted(packages[key].items()) if value >= 0]
    switches = {key: [{"value": value, "label": label} for value, label in rows]
                for key, rows in SWITCHES.items()}
    sources = [{"file": str(p.relative_to(root)).replace("\\", "/"), "sha256": _sha(p)}
               for p in [readme_path, *sorted((root / "Registry").iterdir())]
               if p.is_file()]
    document = {"schema": "gpuwm-wrf-physics-options-v1",
                "wrf": {"tag": "v4.7.1",
                        "commit": "f52c197ed39d12e087d02c50f412d90d418f6186",
                        "url": "https://github.com/wrf-model/WRF"},
                "read_from": sources,
                "selectors": selectors, "switches": switches}
    out.write_text(json.dumps(document, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
