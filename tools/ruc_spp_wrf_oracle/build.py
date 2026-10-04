#!/usr/bin/env python3
"""Build current RUC plus the exact historical hydraulic SPP operator.

The enabled oracle is explicitly an overlay, not unmodified WRF v4.6.1.
The source hashes prevent silently testing a different upstream routine.
Run under nice on a CPU node. No CUDA is used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess

CURRENT_SHA = "3265f810d08dcbddfaf198371dc7f652e78e8d3a788f703a515c555a3bbb2a12"
HISTORICAL_SHA = "834833a79a7a57a4e436a038f3019ce2b5a998141cc5bb9cd39cf2d83ec4d786"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def driver(source: str, stage: str) -> str:
    source = source.replace("  character(len=1024) :: output_path", """  character(len=1024) :: output_path, argument
  integer :: spp_mode, pattern_kind
  real :: amplitude""")
    source = source.replace("  call ruclsm_soilvegparm", """  call get_command_argument(2, argument)
  read(argument, *) spp_mode
  call get_command_argument(3, argument)
  read(argument, *) amplitude
  call get_command_argument(4, argument)
  read(argument, *) pattern_kind

  call ruclsm_soilvegparm""", 1)
    source = source.replace("  rstochcol = 0.0", "  rstochcol = amplitude")
    source = source.replace(f"    call {stage}(0,", f"""    do k = 1, nzs
      rstochcol(k) = amplitude
      if (pattern_kind == 1) rstochcol(k) = amplitude * real(k - 5) / 4.0
    end do
    call {stage}(spp_mode,""", 1)
    source = re.sub(r"(  write\(unit, '\(A\)'\) 'case,[^\n]*)'", r"\1,rstochcol,fieldcol_sf'", source, count=1)
    tail = "          infiltrp, smf" if stage == "soil" else "          infiltrp"
    source = source.replace(tail + "\n", tail + ", rstochcol(k), fieldcol_sf(k)\n", 1)
    return source


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("current", type=Path)
    parser.add_argument("historical", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if sha(args.current) != CURRENT_SHA or sha(args.historical) != HISTORICAL_SHA:
        raise SystemExit("RUC source hash differs from the pinned official versions")
    current = args.current.read_text(encoding="utf-8")
    historical = args.historical.read_text(encoding="utf-8")
    routine = historical.split("SUBROUTINE SOILPROP", 1)[1].split("END SUBROUTINE SOILPROP", 1)[0]
    block = re.search(r"(?m)^\s*if \(spp_lsm==1\) then[^\n]*\n.*?^\s*ENDIF", routine, re.S | re.M)
    if block is None or "fieldcol_sf(k)=hydro(k)*rstochcol(k)" not in block.group():
        raise SystemExit("historical hydraulic operator not found")
    operator = block.group().strip() + "\n"
    if "hydro(k)=hydro(k)*(1+rstochcol(k))" not in operator:
        raise SystemExit("historical hydraulic multiplier changed")
    boundaries = list(re.finditer(r"(?im)^\s*end subroutine soilprop\s*$", current))
    if len(boundaries) != 1:
        raise SystemExit("current SOILPROP boundary is ambiguous")
    boundary = boundaries[0].start()
    overlay = current[:boundary] + "\n" + operator + current[boundary:]
    repository = Path(__file__).resolve().parents[2]
    fixture = repository / "tools" / "ruc_wrf461_oracle"
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    records = []
    commands = []

    def run(command, cwd):
        commands.append({"cwd": str(cwd), "argv": list(map(str, command))})
        subprocess.run(list(map(str, command)), cwd=cwd, check=True)

    for variant, text in (("current", current), ("hydraulic_overlay", overlay)):
        target = output / variant
        target.mkdir(exist_ok=True)
        (target / "module_sf_ruclsm.F").write_text(text, encoding="utf-8")
        for table in ("SOILPARM.TBL", "VEGPARM.TBL", "GENPARM.TBL"):
            shutil.copyfile(repository / "gpuwm" / "data" / "noah_tables" / table, target / table)
        flags = ["gfortran", "-O0", "-cpp", "-ffree-form", "-ffree-line-length-none", "-fallow-argument-mismatch"]
        run(flags + ["-c", fixture / "stub_wrf.F90"], target)
        run(flags + ["-c", "-DEM_CORE=0", "-Dwrf_chem=0", "module_sf_ruclsm.F"], target)
        for stage in ("soil", "snowsoil"):
            modified_driver = driver((fixture / f"run_{stage}.F90").read_text(encoding="utf-8"), stage)
            (target / f"run_{stage}.F90").write_text(modified_driver, encoding="utf-8")
            run(flags + ["-o", f"run_{stage}", "stub_wrf.o", "module_sf_ruclsm.o", f"run_{stage}.F90"], target)
            cases = [("off", 0, .3, 0), ("zero", 1, 0., 0),
                     ("minus09", 1, -.9, 0), ("minus03", 1, -.3, 0),
                     ("plus03", 1, .3, 0), ("plus09", 1, .9, 0),
                     ("depth", 1, .9, 1)]
            for label, mode, amplitude, kind in cases:
                filename = f"{stage}-{label}.csv"
                run([target / f"run_{stage}", filename, mode, amplitude, kind], target)
                records.append({"variant": variant, "stage": stage, "label": label,
                                "mode": mode, "file": f"{variant}/{filename}",
                                "sha256": sha(target / filename)})
    manifest = {"operator_id": "wrf-v3.9.1-ruc-hydraulic-spp-v1",
                "current_sha256": CURRENT_SHA, "historical_sha256": HISTORICAL_SHA,
                "overlay_sha256": sha(output / "hydraulic_overlay" / "module_sf_ruclsm.F"),
                "operator": operator,
                "compiler": subprocess.check_output(["gfortran", "--version"], text=True).splitlines()[0],
                "table_sha256": {name: sha(output / "current" / name) for name in
                                 ("SOILPARM.TBL", "VEGPARM.TBL", "GENPARM.TBL")},
                "records": records, "commands": commands}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"built": len(records), "manifest": str(output / "manifest.json")}))


if __name__ == "__main__":
    main()
