"""Which committed aerosol fixtures does a GPU comparison actually read?

A fixture that is hash-pinned but never compared is coverage a reader will
assume they have.  The mp=8 deck has one: `mixed-surface.csv` is pinned by
`tests/test_thompson_oracle_provenance.py` and consumed by no GPU
comparison, so perturbing every field in it by 1% passes the entire mp=8
tier.  This script asks the same question of the 50 aerosol fixtures the way
that one was answered -- by perturbation, not by reading imports.

For each CSV: multiply every numeric field by 1.01 (zeros become 1e-30), run
the mp=28 device tier with -x, restore the original bytes, and record whether
anything failed.  The restore is asserted byte-for-byte before moving on, and
runs in a `finally`, so an interrupted sweep cannot leave a perturbed fixture
behind.

MEASURED on 2026-08-02, one RTX 5090, against the committed deck:
**50 of 50 consumed, none unconsumed.**  The aerosol deck has no equivalent
of the mp=8 gap.

Usage (needs a CUDA device and a staged CCN_ACTIVATE.BIN):

    python tools/thompson_wrf461_oracle/measure_aero_fixture_coverage.py

Takes roughly 45 minutes: one full device-tier run per fixture.
"""
import csv, io, pathlib, shutil, subprocess, sys, tempfile

ROOT = pathlib.Path(__file__).resolve().parents[2]
ORACLE = ROOT / "gpuwm" / "data" / "thompson" / "oracle-aero"
PY_ = sys.executable
SEL = ["tests/test_thompson_aerosol_adapter.py",
       "tests/test_thompson_aerosol_gpu.py",
       "tests/test_thompson_aerosol_state_gpu.py",
       "tests/test_thompson_aerosol_sat_gpu.py",
       "tests/test_thompson_aerosol_warm_gpu.py",
       "tests/test_thompson_aerosol_cold_gpu.py",
       "tests/test_thompson_aerosol_sed_gpu.py",
       "tests/test_thompson_aerosol_device_helpers.py",
       "tests/test_thompson_aerosol_contract.py"]


def perturb(path):
    rows = list(csv.reader(path.open(newline="", encoding="ascii")))
    out = io.StringIO()
    w = csv.writer(out, lineterminator="\n")
    for i, row in enumerate(rows):
        if i == 0:
            w.writerow(row); continue
        new = []
        for cell in row:
            try:
                v = float(cell)
            except ValueError:
                new.append(cell); continue
            new.append(repr(v * 1.01) if v else "1.0e-30")
        w.writerow(new)
    return out.getvalue()


names = sorted(p.name for p in ORACLE.glob("*.csv"))
unconsumed = []
for name in names:
    path = ORACLE / name
    original = path.read_bytes()
    backup = tempfile.NamedTemporaryFile(delete=False)
    backup.write(original); backup.close()
    try:
        path.write_text(perturb(path), encoding="ascii", newline="")
        r = subprocess.run([PY_, "-m", "pytest", "-q", "-x", "-p",
                            "no:randomly", *SEL],
                           cwd=str(ROOT), capture_output=True,
                           text=True, timeout=900)
        caught = r.returncode != 0
    finally:
        path.write_bytes(original)
        assert path.read_bytes() == original, name
        pathlib.Path(backup.name).unlink()
    print(f"{'CONSUMED ' if caught else 'UNCONSUMED'}  {name}", flush=True)
    if not caught:
        unconsumed.append(name)
print()
print(f"{len(names) - len(unconsumed)} of {len(names)} consumed")
print("UNCONSUMED:", unconsumed)
