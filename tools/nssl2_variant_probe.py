"""Column smoke over the four ported NSSL option-18 variant modes.

For each regime below, every ported variant is advanced through the shipped
``run_nssl2_production_step`` seam, and each variant is then compared against
the DEFAULT mode run on THE SAME inputs.  Comparing on identical inputs is
what makes the delta a treatment proof rather than an input difference: an
arm whose deltas are all exactly 0.0 did not actually take its variant path,
which is the failure mode that has voided A/B screens in this project before.

Fields a variant's Registry package would never have allocated are zeroed on
input (that is what their absence means at this seam) and are then required
to come back exactly zero.  The prognostic CCN field is different: with
``nssl_ccn_on=0`` WRF skips the store entirely
(``module_mp_nssl_2mom.F:3283``), so the invariant there is "byte-identical
to the caller's input", not "zero".

Two regimes run, because one is not enough:

* ``moist-column`` -- the seeded profile the mp18 digest baseline uses.  It
  exercises the CCN paths but never reaches the graupel-to-hail conversion:
  the wet-growth diameter stays at its 0.1501 m ceiling and the upper gamma
  tail above it is empty, so the hail arms are legitimately identical there.
* ``riming-updraft`` -- large low-concentration graupel in supercooled cloud
  water with a strong updraft, which is the regime the conversion exists for.
  This is the regime that actually tests the hail switch.

This is a self-consistency and treatment smoke.  It is NOT an oracle
comparison: no WRF Fortran run is involved anywhere in this file.

Usage:
    python tools/nssl2_variant_probe.py [--json OUT.json]
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

# PYTHONSAFEPATH=1 keeps the script directory off sys.path, which is the
# behaviour we want for gpuwm itself; the sibling probe is an explicit,
# path-qualified import rather than an ambient one.
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from nssl2_mp18_digest_probe import (  # noqa: E402
    _OUTPUT_PRECIPITATION as _OUTPUT_PRECIPITATION, _run_one, _seeded_fields,
)


#: (label, selector kwargs) for every mode with a ported numerical path.
#: The deprecated WRF scheme IDs are the spellings users reach for;
#: share/module_check_a_mundo.F:3382-3421 rewrites them onto these flags.
PROBE_MODES = (
    ("mp18-default", {}),
    ("mp17-no-ccn", {"nssl_ccn_on": 0}),
    ("mp18-no-hail", {"nssl_hail_on": 0}),
    ("mp22-no-hail-no-ccn", {"nssl_hail_on": 0, "nssl_ccn_on": 0}),
)


_REGISTRY_ORDER = (
    "qv", "qc", "qr", "qi", "qs", "qg", "qh",
    "qndrop", "qnr", "qni", "qns", "qng", "qnh", "qnn", "qvolg", "qvolh",
)


def moist_column(nz: int, ny: int, nx: int):
    """The seeded profile behind evidence/nssl2-variants/mp18-digest-baseline."""
    return _seeded_fields(nz, ny, nx)


def riming_updraft(nz: int, ny: int, nx: int):
    """Large graupel riming in supercooled water under a strong updraft.

    Graupel number is set low enough (order 100 m-3) that the mean-volume
    diameter reaches the centimetre scale, which is where the default
    ihlcnh=3 conversion at module_mp_nssl_2mom.F:19860 has any upper tail
    above the wet-growth diameter to convert.
    """
    shape = (nz, ny, nx)
    column = np.linspace(0.0, 1.0, nz, dtype=np.float32)[:, None, None]
    pressure = np.broadcast_to(
        70000.0 - 40000.0 * column, shape).astype(np.float32)
    exner = ((pressure / 100000.0) ** (287.04 / 1004.0)).astype(np.float32)
    temperature = np.full(shape, 263.0, dtype=np.float32)
    theta = (temperature / exner).astype(np.float32)
    rho = (pressure / (287.04 * temperature)).astype(np.float32)

    fields = {
        "qv": np.full(shape, 2.0e-3, dtype=np.float32),
        "qc": np.full(shape, 5.0e-3, dtype=np.float32),
        "qr": np.full(shape, 1.0e-3, dtype=np.float32),
        "qi": np.full(shape, 2.0e-4, dtype=np.float32),
        "qs": np.full(shape, 5.0e-4, dtype=np.float32),
        "qg": np.full(shape, 6.0e-3, dtype=np.float32),
        "qh": np.zeros(shape, dtype=np.float32),
    }
    fields["qndrop"] = (fields["qc"] * 1.0e11).astype(np.float32)
    fields["qnr"] = (fields["qr"] * 1.0e7).astype(np.float32)
    fields["qni"] = (fields["qi"] * 1.0e9).astype(np.float32)
    fields["qns"] = (fields["qs"] * 5.0e7).astype(np.float32)
    fields["qng"] = np.full(
        shape, 100.0 / float(rho.mean()), dtype=np.float32)
    fields["qnh"] = np.zeros(shape, dtype=np.float32)
    fields["qnn"] = np.full(shape, 4.0e8, dtype=np.float32)
    fields["qvolg"] = (fields["qg"] / 500.0).astype(np.float32)
    fields["qvolh"] = np.zeros(shape, dtype=np.float32)

    environment = {
        "theta": theta,
        "rho": rho,
        "pressure": pressure,
        "exner": exner,
        "temperature": temperature,
        "w": np.full((nz + 1, ny, nx), 20.0, dtype=np.float32),
        "dz": np.full(shape, 250.0, dtype=np.float32),
    }
    return fields, environment


REGIMES = (
    ("moist-column", moist_column),
    ("riming-updraft", riming_updraft),
)



def run_probe(steps: int = 3, nz: int = 40, ny: int = 4, nx: int = 4):
    import gpuwm
    from gpuwm.core.nssl2_contract import (
        DEFAULT_MODE,
        pinned_zero_fields,
        require_ported_nssl2_mode,
        resolve_nssl2_mode,
    )

    print(f"gpuwm.__file__ = {gpuwm.__file__}")

    report: dict[str, object] = {
        "steps": steps, "shape": [nz, ny, nx], "regimes": {}}

    for regime_label, builder in REGIMES:
        regime_entry: dict[str, object] = {}
        for label, selectors in PROBE_MODES:
            mode = require_ported_nssl2_mode(resolve_nssl2_mode(**selectors))
            pinned = pinned_zero_fields(mode)

            fields, environment = builder(nz, ny, nx)
            # A field the variant's Registry packages never allocate starts
            # at zero.  CCN is the exception: it exists in the caller's
            # arrays either way, WRF simply stops reading and writing it.
            for name in pinned:
                if name != "qnn":
                    fields[name] = np.zeros_like(fields[name])

            outputs, inputs = _run_one(mode, fields, environment, steps)

            # Same inputs, default mode: the control arm for this variant.
            control_fields, control_environment = builder(nz, ny, nx)
            for name in pinned:
                if name != "qnn":
                    control_fields[name] = np.zeros_like(
                        control_fields[name])
            control, _ = _run_one(
                DEFAULT_MODE, control_fields, control_environment, steps)

            entry: dict[str, object] = {
                "selectors": selectors,
                "resolved": {
                    "two_moment": mode.two_moment,
                    "hail": mode.hail,
                    "predicted_ccn": mode.predicted_ccn,
                    "density_moments": mode.density_moments,
                    "sixth_moments": mode.sixth_moments,
                },
                "pinned_zero_fields": list(pinned),
            }
            entry["nonfinite_fields"] = sorted(
                name for name, value in outputs.items()
                if not np.all(np.isfinite(value)))

            violations = {}
            for name in pinned:
                if name == "qnn" or name not in outputs:
                    continue
                peak = float(np.max(np.abs(outputs[name])))
                if peak != 0.0:
                    violations[name] = peak
            entry["pinned_zero_violations"] = violations

            if not mode.predicted_ccn:
                entry["qnn_untouched"] = bool(
                    np.array_equal(outputs["qnn"], inputs["qnn"]))

            deltas = {
                name: float(np.max(np.abs(
                    outputs[name].astype(np.float64)
                    - control[name].astype(np.float64))))
                for name in sorted(outputs)
            }
            entry["max_abs_delta_vs_same_input_default"] = deltas
            entry["fields_changed"] = sorted(
                name for name, value in deltas.items() if value != 0.0)
            regime_entry[label] = entry
        report["regimes"][regime_label] = regime_entry
    return report


def _summarize(report) -> int:
    status = 0
    for regime_label, modes in report["regimes"].items():
        print(f"\n################ regime: {regime_label} ################")
        for label, entry in modes.items():
            print(f"\n=== {label} ===")
            print(f"  resolved           : {entry['resolved']}")
            print(f"  pinned-zero fields : {entry['pinned_zero_fields']}")
            if entry["nonfinite_fields"]:
                print(f"  NON-FINITE         : {entry['nonfinite_fields']}")
                status = 1
            else:
                print("  finite             : yes (all output arrays)")
            if entry["pinned_zero_violations"]:
                print("  PINNED-ZERO BROKEN : "
                      f"{entry['pinned_zero_violations']}")
                status = 1
            elif [n for n in entry["pinned_zero_fields"] if n != "qnn"]:
                print("  pinned-zero held   : yes (stayed exactly 0.0)")
            if "qnn_untouched" in entry:
                print(f"  qnn left untouched : {entry['qnn_untouched']}")
                if not entry["qnn_untouched"]:
                    status = 1
            changed = entry["fields_changed"]
            deltas = entry["max_abs_delta_vs_same_input_default"]
            if label == "mp18-default":
                if changed:
                    print("  SELF-CONTROL FAIL  : the default arm differs "
                          "from its own control run")
                    status = 1
                else:
                    print("  self-control       : identical (determinism ok)")
            elif changed:
                top = sorted(changed, key=lambda k: -deltas[k])[:5]
                print(f"  treatment proof    : {len(changed)} fields differ; "
                      f"largest {[(k, deltas[k]) for k in top]}")
            else:
                print("  treatment: no delta on this regime "
                      "(see module docstring for why that can be correct)")
    return status


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", default=None)
    parser.add_argument("--steps", type=int, default=3)
    args = parser.parse_args()

    report = run_probe(steps=args.steps)
    status = _summarize(report)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=1, sort_keys=True)
        print(f"\nwrote {args.json}")
    print(f"\nstatus = {status}")
    return status


if __name__ == "__main__":
    sys.exit(main())
