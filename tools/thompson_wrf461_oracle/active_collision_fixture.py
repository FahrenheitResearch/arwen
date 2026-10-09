"""Generate small active-collision columns through the complete classic driver.

The existing column driver supplies the call ABI and CSV writer. Only its
four-level input is replaced. The pinned microphysics source is unmodified,
unless ``--corrected-racg`` asks for the reference WOOF's classic Thompson
is graded on: the same source with only its eight rain-graupel table reads
changed to MIN(idx_bg(k),dimNRHG)
(tools/thompson_mp8_column_oracle/corrected_mp28.py), because WOOF reads the
one slab WRF builds where WRF v4.6.1 subscripts it at idx_bg1=5.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

SOURCE_SHA256 = "fabf19e2a9073cff886e882b187080bfdf089d3fd40c0fce1d19bc93b1e5e802"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wrf-root", type=Path, required=True)
    parser.add_argument("--table-root", type=Path, required=True)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--corrected-racg", action="store_true")
    args = parser.parse_args()
    from gpuwm.core.thompson_contract import load_validated_classic_tables

    tables = load_validated_classic_tables(args.table_root)
    source = args.wrf_root / "phys/module_mp_thompson.F"
    assert hashlib.sha256(source.read_bytes()).hexdigest() == SOURCE_SHA256
    if args.output.exists():
        parser.error("Output already exists; retain it and use a new output path")
    args.build_dir.mkdir(parents=True, exist_ok=False)
    build = args.build_dir.resolve()
    variant = None
    if args.corrected_racg:
        from tools.thompson_mp8_column_oracle.corrected_mp28 import (
            NEW, OLD, corrected_source)
        corrected = corrected_source(source.read_bytes())
        source = build / "module_mp_thompson.F"
        source.write_bytes(corrected)
        variant = {"oracle_source_variant": "corrected-racg",
                   "old": OLD, "new": NEW, "replacement_count": 8,
                   "corrected_source_sha256":
                       hashlib.sha256(corrected).hexdigest()}
    support = Path(__file__).resolve().parent
    original_driver = (support / "run_column.F90").read_text(encoding="utf-8")
    driver = original_driver.replace("nz = 24", "nz = 4")
    driver = driver.replace(
        "character(len=32) :: scenario",
        "character(len=512) :: input_path\n  character(len=32) :: scenario")
    override = """  call get_command_argument(3, input_path)
  open(newunit=ios, file=trim(input_path), status='old', action='read')
  read(ios,*) dt
  do k=1,nz
    read(ios,*) p(1,k,1), th(1,k,1), pii(1,k,1), dz(1,k,1), &
      qv(1,k,1), qc(1,k,1), qr(1,k,1), qi(1,k,1), qs(1,k,1), &
      qg(1,k,1), ni(1,k,1), nr(1,k,1), w(1,k,1)
    do j=1,ny
      do i=1,nx
        p(i,k,j)=p(1,k,1)
        th(i,k,j)=th(1,k,1)
        pii(i,k,j)=pii(1,k,1)
        dz(i,k,j)=dz(1,k,1)
        qv(i,k,j)=qv(1,k,1)
        qc(i,k,j)=qc(1,k,1)
        qr(i,k,j)=qr(1,k,1)
        qi(i,k,j)=qi(1,k,1)
        qs(i,k,j)=qs(1,k,1)
        qg(i,k,j)=qg(1,k,1)
        ni(i,k,j)=ni(1,k,1)
        nr(i,k,j)=nr(1,k,1)
        w(i,k,j)=w(1,k,1)
      enddo
    enddo
  enddo
  close(ios)
"""
    needle = "  qv_before = qv(1,:,1)"
    assert driver.count(needle) == 1
    driver = driver.replace(needle, override + needle)
    driver_path = build / "run_column.F90"
    driver_path.write_text(driver, encoding="utf-8")
    for name in ("qr_acr_qg_V4.dat", "qr_acr_qsV2.dat", "freezeH2O.dat"):
        (build / name).symlink_to((args.table_root / name).resolve())
    flags = ["-O2", "-fno-tree-vectorize", "-ffp-contract=off",
             "-ffree-form", "-ffree-line-length-none"]
    inputs = [support / "stub_wrf.F90", args.wrf_root / "phys/module_mp_radar.F",
              source, driver_path]
    for path in inputs:
        subprocess.run(["gfortran", "-c", *flags, "-cpp", "-DWRF_CHEM=0",
                        str(path.resolve())], cwd=build, check=True)
    subprocess.run(["gfortran", *flags, "-o", "run_column", "stub_wrf.o",
                    "module_mp_radar.o", "module_mp_thompson.o", "run_column.o"],
                   cwd=build, check=True)
    result = {
        "scope": "Complete classic column calls with active warm collisions; no forecast or all-solid energy claim.",
        "reference_commit": "d66e442fccc04111067e29274c9f9eaccc3cef28",
        "source_sha256": SOURCE_SHA256,
        "support_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
        "driver_parent_sha256": hashlib.sha256(original_driver.encode()).hexdigest(),
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "compiler": subprocess.check_output(["gfortran", "--version"], text=True).splitlines()[0],
        "flags": flags,
        "table_identity": tables.identity,
        "cases": [],
    }
    if variant is not None:
        result["source_variant"] = variant
    pressure = np.float32(80000.)
    temperature = np.float32(274.15)
    pii = np.power(pressure / np.float32(100000.), np.float32(2. / 7.))
    theta = np.float32(temperature / pii)
    for name, qv in (("saturated", np.float32(0.0051506985910236835)),
                     ("subsaturated", np.float32(0.004120558965951204)),
                     ("supersaturated", np.float32(0.0051506985910236835 * 1.1))):
        output = build / name
        output.mkdir()
        input_path = output / "input.txt"
        rows = [[pressure, theta, pii, dz, qv, np.float32(.001), np.float32(.0003),
                 0., np.float32(.0002), np.float32(.0002), 0., 30000., 0.]
                for dz in (200., 350., 125., 500.)]
        input_path.write_text("10.0\n" + "\n".join(
            " ".join(str(float(v)) for v in row) for row in rows) + "\n", encoding="ascii")
        with (output / "run.log").open("w", encoding="utf-8") as log:
            subprocess.run([str(build / "run_column"), "warm", str(output), str(input_path)],
                           cwd=build, stdout=log, stderr=subprocess.STDOUT, check=True)
        with (output / "warm-column.csv").open(newline="") as stream:
            columns = list(csv.DictReader(stream))
        with (output / "warm-surface.csv").open(newline="") as stream:
            surface = next(csv.DictReader(stream))
        result["cases"].append({
            "name": name, "dt_s": 10.,
            "before": [{k: float(v) for k, v in row.items() if k != "phase"}
                       for row in columns if row["phase"] == "before"],
            "after": [{k: float(v) for k, v in row.items() if k != "phase"}
                      for row in columns if row["phase"] == "after"],
            "surface": {k: float(v) for k, v in surface.items() if k != "scenario"},
        })
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
