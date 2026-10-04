#!/usr/bin/env python3
"""Build scalar_pblmix fixtures from pinned, unmodified WRF routines.

Usage: python build_scalar_pblmix.py module_pbl_driver.F BUILD_DIR
The compiler and generated sources stay in BUILD_DIR. This CPU-only harness
does not need a WRF build or CUDA. WRF's requested public-domain notice is
copied into the extraction from licenses/LICENSE-WRF-public-domain.txt.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import subprocess

import numpy as np

SOURCE_SHA256 = "90336e30296991fb397ffde87649a4bd20eaa2b7dc6e90639b043810c8420b56"


def build(source: Path, output: Path) -> None:
    original = source.read_bytes()
    source_sha = hashlib.sha256(original).hexdigest()
    if source_sha != SOURCE_SHA256:
        raise ValueError("WRF source differs from the v4.6.1 oracle pin")
    output.mkdir(parents=True, exist_ok=True)
    source_text = original.decode("utf-8")
    routines = []
    for name in ("diff4d", "diff", "invert"):
        match = re.search(
            rf"(?im)^\s*subroutine\s+{name}\s*\(.*?^\s*end subroutine\s+{name}\s*$",
            source_text, re.DOTALL,
        )
        if match is None:
            raise ValueError(f"missing WRF routine {name}")
        routines.append(match.group(0))
    root = Path(__file__).resolve().parents[2]
    notice = (root / "licenses/LICENSE-WRF-public-domain.txt").read_text()
    extracted = output / "scalar_pblmix_wrf.F90"
    extracted.write_text(
        "\n".join("! " + line for line in notice.splitlines()) + "\n"
        "module module_state_description\n"
        "integer, parameter :: P_QNS=5,P_QNR=6,P_QNG=7,P_QT=8\n"
        "integer, parameter :: P_QNH=9,P_QVOLG=10,P_QKE_ADV=11\n"
        "end module\nmodule scalar_pblmix_wrf_oracle\ncontains\n"
        + "\n".join(routines) + "\nend module\n", encoding="utf-8")
    harness = Path(__file__).with_name("run_scalar_pblmix.F90")
    compiler = subprocess.check_output(["gfortran", "--version"], text=True)
    command = ["gfortran", "-O0", "-ffree-form", "-ffree-line-length-none",
               "-fno-fast-math", "-ffp-contract=off", str(extracted.resolve()),
               str(harness.resolve()), "-o", "run_scalar_pblmix"]
    subprocess.run(command, cwd=output, check=True)
    csv_path = output / "scalar_pblmix.csv"
    subprocess.run([str((output / "run_scalar_pblmix").resolve()),
                    str(csv_path.resolve())], cwd=output, check=True)
    with csv_path.open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 8 * 11 * 50
    data = {}
    for key in ("dt", "qn", "dz", "rho", "exch_h", "tendency"):
        data[key] = np.asarray([row[key] for row in rows], np.float32).reshape(8, 11, 50)
    # Excluded precipitating scalars and advected TKE retain their sentinel.
    assert np.all(data["tendency"][:, 4:] == np.float32(-765.25))
    np.savez(output / "scalar_pblmix_wrf461.npz", **data)
    receipt = {
        "source": "WRF v4.6.1 phys/module_pbl_driver.F",
        "source_url": "https://raw.githubusercontent.com/wrf-model/WRF/v4.6.1/phys/module_pbl_driver.F",
        "source_sha256": source_sha,
        "extracted_routines": ["diff4d", "diff", "invert"],
        "compiler": compiler.splitlines()[0],
        "flags": command[1:6],
        "cases": ["zero_diffusivity", "zero_scalar", "surface_jump", "top_jump",
                  "weak_diffusion", "strong_diffusion", "long_step", "short_step"],
        "active_scalars": ["qnc", "qni", "qnwfa", "qnifa"],
        "excluded_scalars": ["qns", "qnr", "qng", "qt", "qnh", "qvolg", "qke_adv"],
        "levels": 50,
        "rows": len(rows),
        "harness_sha256": hashlib.sha256(harness.read_bytes()).hexdigest(),
        "csv_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
        "fixture_sha256": hashlib.sha256((output / "scalar_pblmix_wrf461.npz").read_bytes()).hexdigest(),
    }
    (output / "scalar_pblmix_wrf461.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    build(args.source, args.output)
