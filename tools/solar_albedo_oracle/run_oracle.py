"""Generate branch-complete sun-angle and fractional-seaice oracle fixtures.

Usage: python run_oracle.py BUILD_DIRECTORY OUTPUT.npz
"""
from __future__ import annotations

import ctypes
import json
from pathlib import Path
import sys

import numpy as np

F = np.float32


def solar_inputs(seed=286, n=2048):
    rng = np.random.default_rng(seed)
    fields = {
        "albedo": rng.uniform(0.03, 0.99, n).astype(F),
        "albbck": rng.uniform(0.03, 0.85, n).astype(F),
        "xland": rng.choice(np.array([1, 1.4999999, 1.5, 2], F), n),
        "snow": rng.choice(np.array([0, 0, 0, 1e-30, 0.001, 20], F), n),
        "xice": rng.choice(np.array([0, 0, 0, 1e-30, 0.02, 0.5, 1], F), n),
        "ivgtyp": rng.integers(1, 22, n, dtype=np.int32),
        "albsol": rng.uniform(0.03, 1.1, n).astype(F),
        "albbcksol": rng.uniform(0.03, 1.1, n).astype(F),
    }
    coszen = rng.uniform(-0.2, 1.0, (3, n)).astype(F)
    # Every MODIS class at low sun, half-height sun and overhead sun.
    for call, cosine in enumerate([F(1e-7), F(0.5), F(1.0)]):
        coszen[call, :21] = cosine
    fields["ivgtyp"][:21] = np.arange(1, 22, dtype=np.int32)
    fields["xland"][:21] = F(1)
    fields["snow"][:21] = F(0)
    fields["xice"][:21] = F(0)
    fields["albbck"][:21] = F(0.17)
    # Explicit cap, night, exact land threshold, snow and tiny ice masks.
    fields["xland"][21:28] = np.array([1, 1, 1, 1.5, 1, 1, 2], F)
    fields["snow"][21:28] = np.array([0, 0, 0, 0, 1e-30, 0, 0], F)
    fields["xice"][21:28] = np.array([0, 0, 0, 0, 0, 1e-30, 0], F)
    fields["albbck"][21] = F(0.8)
    fields["albedo"][22:28] = F(0.97)
    coszen[:, 21:28] = np.array([1e-7, 0, -0.1, 0.5, 0.5, 0.5, 0.5], F)
    return fields, coszen


def main():
    build = Path(sys.argv[1]).resolve()
    output = Path(sys.argv[2]).resolve()
    library = ctypes.CDLL(str(build / "libsolar_albedo.so"))
    pointer = np.ctypeslib.ndpointer(dtype=F, flags="C_CONTIGUOUS")
    ipointer = np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS")
    library.solar.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int,
                             pointer, pointer, pointer, pointer, pointer,
                             ipointer, pointer, pointer, pointer]
    library.ice_pre.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_float,
                               ctypes.c_float, *([pointer] * 6)]
    library.ice_post.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_float,
                                pointer, pointer, pointer]
    fields, cosine = solar_inputs()
    fixture = {f"in_{key}": value.copy() for key, value in fields.items()}
    fixture["coszen"] = cosine
    albsol = []
    albbcksol = []
    for call in range(cosine.shape[0]):
        # Changing the original ALBEDO later must not refresh the carried
        # values on snow, water, ice or night columns.
        if call:
            fields["albedo"][...] = F(0.123)
        library.solar(fields["albsol"].size, 1 if call == 0 else 2, 1,
                      *(fields[key] for key in (
                          "albedo", "albbck", "xland", "snow", "xice",
                          "ivgtyp", "albsol", "albbcksol")), cosine[call])
        albsol.append(fields["albsol"].copy())
        albbcksol.append(fields["albbcksol"].copy())
    fixture["out_albsol"] = np.stack(albsol)
    fixture["out_albbcksol"] = np.stack(albbcksol)
    xice = np.array([0, 1e-30, 0.019999999, 0.02, 0.020000001,
                     0.1, 0.49999997, 0.5, 0.50000006, 1, 1.0000001], F)
    fixture["ice_xice"] = xice
    rng = np.random.default_rng(287)
    ice_inputs = {
        "albsol": rng.uniform(0.07, 0.65, xice.size).astype(F),
        "albbcksol": rng.uniform(0.15, 0.5, xice.size).astype(F),
        "emiss": rng.uniform(0.90, 0.99, xice.size).astype(F),
        "tsk": rng.uniform(260, 280, xice.size).astype(F),
        "tsk_save": rng.uniform(260, 280, xice.size).astype(F),
    }
    fixture.update({f"ice_in_{key}": value.copy()
                    for key, value in ice_inputs.items()})
    for fractional in (0, 1):
        threshold = F(0.02 if fractional else 0.5)
        state = {key: value.copy() for key, value in ice_inputs.items()}
        library.ice_pre(xice.size, fractional, threshold, F(0.65), xice,
                        *(state[key] for key in (
                            "albsol", "albbcksol", "emiss", "tsk", "tsk_save")))
        for key, value in state.items():
            fixture[f"ice_{fractional}_pre_{key}"] = value.copy()
        library.ice_post(xice.size, fractional, threshold, xice,
                         state["albsol"], state["emiss"])
        for key in ("albsol", "emiss"):
            fixture[f"ice_{fractional}_post_{key}"] = state[key].copy()
    fixture["receipt"] = np.array((build / "receipt.json").read_text(
        encoding="utf-8"))
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **fixture)
    print(json.dumps({"fixture": output.name, "columns": cosine.shape[1],
                      "radiation_calls": cosine.shape[0],
                      "ice_columns": xice.size, "bytes": output.stat().st_size}))


if __name__ == "__main__":
    main()
