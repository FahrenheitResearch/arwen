"""Grade all defined radiation words over free night-to-dawn trajectories."""
from pathlib import Path
import argparse
import copy
import csv
import hashlib
import inspect
import json
import struct
import subprocess

import numpy as np

from gpuwm.core.noahmp_radiation import RadiationOptions, albedo
from gpuwm.core.noahmp_glacier import _radiation_glacier, _snow_age_glacier

F = np.float32
ROOT = Path(__file__).resolve().parents[2]
SCALARS = "tau0 grain_growth extra_growth dirt_soot swemx".split()
VECTORS = "albsat albdry alblak rhol rhos taul taus".split()
DYNAMIC = "dt cosz elai esai tg tv snowh fsno fwet sneqvo sneqv qsnow fveg".split()
OUTPUT_SCALARS = "fage albold tauss fsun bgap wgap".split()
OUTPUT_VECTORS = "albgrd albgri albd albi fabd fabi ftdd ftid ftii frevd frevi fregd fregi albsnd albsni".split()


def initial():
    with (ROOT / "tests/data/oracles/noahmp/noahmp-radiation-albedo.csv").open() as stream:
        row = next(csv.DictReader(stream))
    def f(key):
        return np.array(int(row[key], 16), dtype=np.uint32).view(F)[()]
    def vec(key):
        return (f(key + "1"), f(key + "2"))
    kw = {key: f(key) for key in SCALARS + ["xl", "betads", "betais"] + DYNAMIC}
    kw.update({key: vec(key) for key in VECTORS + ["omegas"]})
    kw.update(smc=tuple(f("smc" + str(i)) for i in range(1, 5)),
              albold=f("albold_in"), tauss=f("tauss_in"), fage=f("fage_in"),
              vegtyp=11, ist=1, ice=0, nsoil=4)
    # Runtime callers define these otherwise undefined upstream outputs.
    for key in "frevd frevi fregd fregi".split():
        kw[key + "_in"] = (F(0), F(0))
    return kw


def pack(kw, glacier):
    values = [kw[key] for key in SCALARS]
    for key in VECTORS:
        values.extend(kw[key])
    values.extend([kw["xl"], *kw["omegas"], kw["betads"], kw["betais"]])
    values.extend(kw[key] for key in DYNAMIC)
    values.extend([*kw["smc"], kw["albold"], kw["tauss"], kw["fage"]])
    for key in "frevd frevi fregd fregi".split():
        values.extend(kw[key + "_in"])
    if glacier:
        values[44:48] = ([F(0)] * 4 if kw["cosz"] <= 0 else
                          [F(40), F(45), F(10), F(15)])
    assert len(values) == 52
    return struct.pack("<i", int(glacier)) + np.array(values, F).tobytes()


def python_row(kw, glacier, generation):
    if glacier:
        args = [kw[key] for key in "dt tg sneqvo sneqv cosz qsnow".split()]
        solar = ((F(0), F(0)), (F(0), F(0))) if kw["cosz"] <= 0 else ((F(40), F(45)), (F(10), F(15)))
        extra = ({"noahmp_snow_age": generation} if
                 "noahmp_snow_age" in inspect.signature(_radiation_glacier).parameters else {})
        got = _radiation_glacier(*args, *solar, kw["albold"], kw["tauss"], **extra)
        _, leaf_fage = _snow_age_glacier(kw["dt"], kw["tg"], kw["sneqvo"], kw["sneqv"], kw["tauss"])
        return np.array([*got, leaf_fage, *([0] * 35)], F)
    extra = ({"noahmp_snow_age": generation} if
             "noahmp_snow_age" in inspect.signature(RadiationOptions).parameters else {})
    got = albedo(**kw, options=RadiationOptions(3, 2, **extra))
    values = [got[key] for key in OUTPUT_SCALARS]
    for key in OUTPUT_VECTORS:
        values.extend(got[key])
    return np.array([*values, *([0] * 5)], F)


def run(binary, generation, work, *, plant=False):
    if plant:
        import gpuwm.core.noahmp_radiation as radiation
        original = radiation.snow_age
        calls = 0
        def one_bit_term(*args):
            nonlocal calls
            tauss, fage = original(*args)
            calls += 1
            if calls == 2:
                tauss = np.array(tauss, F).view(np.uint32)
                tauss[...] ^= 1
                tauss = tauss.view(F)[()]
            return tauss, fage
        # Perturb the aging term before ALBEDO consumes it, not the grade.
        radiation.snow_age = one_bit_term
    cases = []
    for glacier in (False, True):
        for tg in (250, 265, 273.16, 280):
            for snow in (0, 1.e-20, 50, 800, float(np.nextafter(F(800), F(np.inf)))):
                for snowfall in (0, 0.001):
                    kw = initial()
                    kw.update(tg=F(tg), tv=F(tg), dt=F(300), sneqv=F(snow),
                              sneqvo=F(snow), tauss=F(0.8), albold=F(0.7),
                              qsnow=F(snowfall), fage=F(-9))
                    cases.append((glacier, snowfall, kw))
    work.mkdir(parents=True, exist_ok=True)
    reference_cases = copy.deepcopy(cases)
    diffs, words, excluded, hashes = [], 0, 0, []
    for step, cosz in enumerate((-0.3, -0.0, 0.0, 0.0002, 0.1, 0.6)):
        records = []
        for collection in (cases, reference_cases):
            for glacier, snowfall, kw in collection:
                kw.update(cosz=F(cosz), sneqvo=kw["sneqv"],
                          sneqv=F(kw["sneqv"] + F(F(snowfall) * kw["dt"])))
        for glacier, snowfall, kw in reference_cases:
            records.append(pack(kw, glacier))
        inputs, outputs = work / "input.bin", work / "output.bin"
        inputs.write_bytes(b"".join(records))
        subprocess.run([str(binary), str(inputs), str(outputs)], check=True)
        reference = np.frombuffer(outputs.read_bytes(), F).reshape(len(cases), 41)
        actual = np.stack([python_row(kw, glacier, generation) for glacier, _, kw in cases])
        for index, (glacier, _snowfall, kw) in enumerate(cases):
            width = 6 if glacier else 36
            mask = np.arange(width)
            if not glacier and cosz <= 0:
                mask = mask[(mask < 24) | (mask >= 32)]
                excluded += 8
            a, b = actual[index, mask].view(np.uint32), reference[index, mask].view(np.uint32)
            words += len(mask)
            for word in np.flatnonzero(a != b):
                diffs.append({"step": step, "case": index, "glacier": glacier,
                              "word": int(mask[word]), "actual": int(a[word]), "reference": int(b[word])})
            # Each implementation continues from its own aging/albedo state.
            kw["albold"] = actual[index, 0 if glacier else 1]
            kw["tauss"] = actual[index, 1 if glacier else 2]
            if not glacier:
                kw["fage"] = actual[index, 0]
            rkw = reference_cases[index][2]
            rkw["albold"] = reference[index, 0 if glacier else 1]
            rkw["tauss"] = reference[index, 1 if glacier else 2]
            if not glacier:
                rkw["fage"] = reference[index, 0]
        hashes.append({"step": step, "actual_sha256": hashlib.sha256(actual.tobytes()).hexdigest(),
                       "reference_sha256": hashlib.sha256(reference.tobytes()).hexdigest()})
    receipt = {"generation": generation, "cases": len(cases), "steps": 6,
               "compared_defined_words": words, "excluded_undefined_words": excluded,
               "differing_words": len(diffs), "differences": diffs[:100], "hashes": hashes,
               "planted_one_bit": plant,
               "boundary": "ALBEDO, RADIATION_GLACIER and the standalone glacier FAGE leaf; not the full surface column"}
    (work / "comparison.json").write_text(json.dumps(receipt, indent=2))
    print(json.dumps({k: v for k, v in receipt.items() if k not in ("hashes", "differences")}, indent=2))
    return len(diffs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("binary", type=Path)
    parser.add_argument("generation", choices=("wrf_461", "wrf_470"))
    parser.add_argument("work", type=Path)
    parser.add_argument("--plant", action="store_true")
    args = parser.parse_args()
    raise SystemExit(bool(run(args.binary, args.generation, args.work, plant=args.plant)))
