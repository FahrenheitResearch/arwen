"""Collaborator-scale verification battery for the NOAHMP_GLACIER port.

Five mandated proofs on the rented card, at the scale of the MPAS
collaborator's real x4 case:

P1 ADMISSION+CADENCE: a synthetic surface with exactly 3,010 active
   land-glacier columns (landmask=1, IVGTYP=15, xice=0, first at column
   30) plus a 4,643-column category-15 fractional sea-ice family
   carrying native xland=1 on landmask=0 (1,067 of them with xice in
   [0.02, 0.5)) initializes through run_mpas_column_batch and
   integrates a 2 h phase-1/phase-2 cadence sequence with finite
   outputs.  The classification receipt is quoted at thresholds 0.02
   AND 0.5: the sea-ice count moves by exactly the 1,067-column band.
P2 RESTART: export at the step-30 boundary with glacier columns active,
   restore into a fresh seam, continue 30 steps; the split run's final
   export and caller state are bitwise identical to the uninterrupted
   60-step run.
P3 READ-ONLY: every phase-1 input array is byte-identical after the
   call, checked on step 1 (radiation due) and step 6 (radiation due
   again) with glacier columns present.
P4 BOTH DIRECTIONS:
   a. the guard is intact: with the glacier dispatch disabled the
      named refusal fires (guard_noahmp_glacier_columns directly, and
      the runtime dispatch through the seam);
   b. one bit flipped in one glacier state array inside a restart
      payload is DETECTED as a divergence from the clean twin.

The receipt is written as JSON; stdout carries the quoted lines the
report needs.  Exit 0 only if every proof passed.
"""

import argparse
import datetime
import json
import pathlib
import sys
import time

import numpy as np

NZ, NCOL, DT = 20, 8192, 120.0
START = datetime.datetime(2021, 6, 1, 18, 0)
ISICE = 15
N_GLACIER = 3010          # the collaborator's land-glacier count
FIRST_GLACIER = 30        # their first offending column
N_BAND = 1067             # fractional sea ice, xice in [0.02, 0.5)
N_HIGH = 3576             # fractional sea ice, xice in [0.5, 1.0)
N_WATER = 300
STEPS_FULL = 60           # 2 h at dt=120 s
STEPS_SPLIT = 30

THRESHOLD = 0.02          # the collaborator's value
WRF_DEFAULT = 0.5


def build_surface(seed=176):
    rng = np.random.default_rng(seed)
    f32 = np.float32
    landmask = np.ones(NCOL, f32)
    xland = np.ones(NCOL, f32)
    ivg = np.full(NCOL, 10, np.int32)
    xice = np.zeros(NCOL, f32)
    tsk = np.full(NCOL, 285.0, f32)
    tmn = np.full(NCOL, 280.0, f32)
    snow = np.zeros(NCOL, f32)
    snow_depth = np.zeros(NCOL, f32)

    pool = np.arange(FIRST_GLACIER + 1, NCOL)
    rng.shuffle(pool)
    glacier_idx = np.sort(np.concatenate(
        ([FIRST_GLACIER], pool[:N_GLACIER - 1])))
    rest = pool[N_GLACIER - 1:]
    band_idx = np.sort(rest[:N_BAND])
    high_idx = np.sort(rest[N_BAND:N_BAND + N_HIGH])
    water_idx = np.sort(
        rest[N_BAND + N_HIGH:N_BAND + N_HIGH + N_WATER])

    # the collaborator glacier signature: landmask=1, IVGTYP=15, xice=0
    ivg[glacier_idx] = ISICE
    tsk[glacier_idx] = rng.uniform(255.0, 270.0, N_GLACIER)
    tmn[glacier_idx] = rng.uniform(258.0, 264.0, N_GLACIER)
    snowy = glacier_idx[rng.random(N_GLACIER) < 0.5]
    snow[snowy] = rng.uniform(20.0, 150.0, snowy.size)
    snow_depth[snowy] = snow[snowy] / 200.0

    # category-15 fractional sea ice with NATIVE xland=1 on landmask=0
    family = np.concatenate([band_idx, high_idx])
    landmask[family] = 0.0
    xland[family] = 1.0
    ivg[family] = ISICE
    xice[band_idx] = rng.uniform(0.02, 0.4999, N_BAND)
    xice[high_idx] = rng.uniform(0.5, 0.9999, N_HIGH)
    tsk[family] = 265.0
    tmn[family] = 264.0

    landmask[water_idx] = 0.0
    xland[water_idx] = 2.0
    ivg[water_idx] = 17
    tsk[water_idx] = 300.0
    tmn[water_idx] = 285.0

    first = int(np.flatnonzero(
        (landmask == 1.0) & (ivg == ISICE) & (xice == 0.0))[0])
    assert first == FIRST_GLACIER, first
    return dict(landmask=landmask, xland=xland, ivgtyp=ivg, xice=xice,
                tsk=tsk, tmn=tmn, snow=snow, snow_depth=snow_depth)


def build_fields(cp, seed=0):
    rng = np.random.default_rng(seed)
    z_iface = np.linspace(0.0, 16000.0, NZ + 1)
    z_mid = 0.5 * (z_iface[:-1] + z_iface[1:])
    p_iface = 101325.0 * np.exp(-z_iface / 8000.0)
    p_mid = 101325.0 * np.exp(-z_mid / 8000.0)
    theta = 300.0 + 25.0 * z_mid / 16000.0
    exner = (p_mid / 1.0e5) ** (287.0 / 1004.0)
    rho_dry = p_mid / (287.0 * theta * exner)
    qv = 0.014 * np.exp(-z_mid / 3000.0)
    qc = np.where(z_mid < 4000.0, 8.0e-4, 0.0)

    def cols(profile, jitter=1.0e-3):
        base = np.repeat(profile[:, None], NCOL, axis=1)
        base *= 1.0 + jitter * rng.standard_normal(base.shape)
        return np.ascontiguousarray(base, dtype=np.float32)

    fields = {
        "u": cols(np.full(NZ, 5.0)),
        "v": cols(np.full(NZ, -3.0)),
        "theta": cols(theta),
        "pressure": cols(p_mid, jitter=0.0),
        "pressure_interface": cols(p_iface, jitter=0.0),
        "z_interface": np.ascontiguousarray(
            np.repeat(z_iface[:, None], NCOL, axis=1), dtype=np.float32),
        "w": cols(0.05 * np.sin(np.linspace(0, np.pi, NZ + 1)),
                  jitter=0.0),
        "rho_dry": cols(rho_dry, jitter=0.0),
        "qv": cols(qv),
        "qc": cols(qc),
        "qr": cols(np.full(NZ, 1.0e-5)),
        "qi": np.zeros((NZ, NCOL), dtype=np.float32),
        "qs": np.zeros((NZ, NCOL), dtype=np.float32),
        "qg": np.zeros((NZ, NCOL), dtype=np.float32),
    }
    return {name: cp.asarray(value) for name, value in fields.items()}


def make_seam(surface, *, threshold, native_xland):
    from gpuwm.core import mpas_column_batch as mcb
    kwargs = dict(surface)
    if not native_xland:
        kwargs.pop("xland")
    return mcb.run_mpas_column_batch(
        n_levels=NZ, n_columns=NCOL, dt=DT,
        radiation_seconds=600.0, surface_pbl_seconds=120.0,
        cumulus_seconds=None, cumulus_scheme=None,
        start_time=START,
        latitude_deg=np.full(NCOL, 62.0),
        longitude_deg=np.full(NCOL, -45.0),
        terrain_height_m=np.zeros(NCOL),
        z_interface_nominal_m=np.linspace(0.0, 16000.0, NZ + 1),
        p_top_pa=float(101325.0 * np.exp(-2.0)), dx_m=15000.0,
        xice_threshold=threshold, **kwargs)


def phase1(seam, fields):
    return seam.run_phase1(
        dt=DT, u=fields["u"], v=fields["v"], theta=fields["theta"],
        pressure=fields["pressure"],
        pressure_interface=fields["pressure_interface"],
        z_interface=fields["z_interface"], w=fields["w"],
        rho_dry=fields["rho_dry"], qv=fields["qv"], qc=fields["qc"],
        qr=fields["qr"], qi=fields["qi"], qs=fields["qs"],
        qg=fields["qg"])


def phase2(seam, fields):
    return seam.run_phase2(
        theta=fields["theta"], qv=fields["qv"], qc=fields["qc"],
        qr=fields["qr"], qi=fields["qi"], qs=fields["qs"],
        qg=fields["qg"], pressure=fields["pressure"],
        rho_dry=fields["rho_dry"], z_interface=fields["z_interface"])


def payload_copy(payload):
    return {
        "identity": dict(payload["identity"]),
        "arrays": {key: np.array(value, copy=True)
                   for key, value in payload["arrays"].items()},
        "scalars": json.loads(json.dumps(payload["scalars"])),
    }


def payloads_equal(a, b):
    """(equal, diff_report) over two export_state payloads, bitwise."""
    diffs = []
    if set(a["arrays"]) != set(b["arrays"]):
        diffs.append("array key sets differ")
    for key in sorted(set(a["arrays"]) & set(b["arrays"])):
        left, right = a["arrays"][key], b["arrays"][key]
        if left.shape != right.shape or left.dtype != right.dtype:
            diffs.append(f"{key}: shape/dtype")
        elif left.tobytes() != right.tobytes():
            count = int((np.asarray(left).view(np.uint8) !=
                         np.asarray(right).view(np.uint8)).sum())
            diffs.append(f"{key}: {count} differing bytes")
    if a["scalars"] != b["scalars"]:
        diffs.append(f"scalars differ: {a['scalars']} != {b['scalars']}")
    return (not diffs), diffs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    import cupy as cp
    from gpuwm.core.noahmp_runtime import (
        NoahmpGlacierColumnError, NoahmpRuntimeParameters,
        guard_noahmp_glacier_columns)

    receipt = {
        "schema": "noahmp-glacier-verify-v1",
        "device": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
        "cupy": cp.__version__,
        "grid": {"nz": NZ, "ncol": NCOL, "dt_s": DT,
                 "steps_full": STEPS_FULL, "hours": STEPS_FULL * DT / 3600.0},
        "proofs": {},
    }
    failures = []

    def record(name, ok, detail):
        receipt["proofs"][name] = {"ok": bool(ok), **detail}
        line = "PASS" if ok else "FAIL"
        print(f"[{name}] {line}")
        if not ok:
            failures.append(name)

    surface = build_surface()

    # ---- P1: classification receipts at 0.02 and 0.5 -------------------
    t0 = time.time()
    seam = make_seam(surface, threshold=THRESHOLD, native_xland=True)
    rec02 = dict(seam.surface_classification)
    seam50 = make_seam(surface, threshold=WRF_DEFAULT, native_xland=True)
    rec50 = dict(seam50.surface_classification)
    del seam50
    derived = make_seam(surface, threshold=THRESHOLD, native_xland=False)
    rec_derived = dict(derived.surface_classification)
    del derived
    cp.get_default_memory_pool().free_all_blocks()
    delta = int(rec02["sea_ice_columns"]) - int(rec50["sea_ice_columns"])
    print("receipt @0.02 native :",
          json.dumps(rec02, sort_keys=True, default=str))
    print("receipt @0.50 native :",
          json.dumps(rec50, sort_keys=True, default=str))
    print("receipt @0.02 derived:",
          json.dumps(rec_derived, sort_keys=True, default=str))
    print(f"sea-ice delta 0.02 vs 0.5: {delta}")
    ok = (rec02["xland_source"] == "native"
          and rec02["glacier_columns"] == N_GLACIER
          and rec02["sea_ice_columns"] == N_BAND + N_HIGH
          and rec50["sea_ice_columns"] == N_HIGH
          and delta == N_BAND
          and rec_derived["xland_source"] == "derived-from-landmask")
    record("P1-classification", ok, {
        "receipt_0.02_native": rec02, "receipt_0.5_native": rec50,
        "receipt_0.02_derived": rec_derived, "sea_ice_delta": int(delta),
        "expected_delta": N_BAND})

    # ---- P1 cadence + P3 read-only over the 2 h sequence ---------------
    fields = build_fields(cp)
    finite_all = True
    census_bad = None
    readonly = {}
    for step in range(1, STEPS_FULL + 1):
        if step in (1, 6):
            before = {name: value.copy() for name, value in fields.items()}
        out = phase1(seam, fields)
        if step in (1, 6):
            same = all(bool(cp.array_equal(
                fields[name].view(cp.uint32)
                if fields[name].dtype == cp.float32 else fields[name],
                before[name].view(cp.uint32)
                if before[name].dtype == cp.float32 else before[name]))
                for name in fields)
            readonly[f"step{step}"] = bool(same)
        for name in ("du", "dv", "dtheta", "dqv", "dqc", "h_diabatic"):
            if not bool(cp.isfinite(getattr(out, name)).all()):
                finite_all = False
        census = dict(seam.last_noahmp_census)
        if census.get("glacier") != N_GLACIER:
            census_bad = (step, census)
        phase2(seam, fields)
    tracked = ("tsk", "tgxy", "snow", "snowh", "tslb", "tsnoxy",
               "smois", "sh2o", "snicexy", "snliqxy", "zsnsoxy")
    state_finite = {name: bool(cp.isfinite(
        seam._driver.fields[name]).all()) for name in tracked}
    census = dict(seam.last_noahmp_census)
    print("final census:", json.dumps(
        {key: (value if isinstance(value, (int, str)) else str(value))
         for key, value in census.items()}, sort_keys=True))
    print(f"2 h integration wall: {time.time() - t0:.1f} s")
    record("P1-cadence-finite", finite_all and census_bad is None
           and all(state_finite.values())
           and "noahmp-glacier" in str(census.get("glacier_path", "")), {
        "steps": STEPS_FULL, "finite_outputs": finite_all,
        "census_final": {key: str(value) for key, value in census.items()},
        "census_anomaly": (None if census_bad is None
                           else str(census_bad)),
        "glacier_state_finite": state_finite})
    record("P3-read-only", all(readonly.values()) and len(readonly) == 2,
           {"checks": readonly})
    export_a = seam.export_state()
    fields_a_end = {name: cp.asnumpy(fields[name])
                    for name in ("theta", "qv", "qc", "qr", "qi", "qs",
                                 "qg")}
    del seam
    cp.get_default_memory_pool().free_all_blocks()

    # ---- P2: split run, restore mid-sequence, bit-identical ------------
    seam_b = make_seam(surface, threshold=THRESHOLD, native_xland=True)
    fields_b = build_fields(cp)
    for _ in range(STEPS_SPLIT):
        phase1(seam_b, fields_b)
        phase2(seam_b, fields_b)
    payload = seam_b.export_state()
    saved = payload_copy(payload)          # for P4b
    del seam_b
    cp.get_default_memory_pool().free_all_blocks()

    seam_c = make_seam(surface, threshold=THRESHOLD, native_xland=True)
    seam_c.restore_state(payload)
    for _ in range(STEPS_SPLIT, STEPS_FULL):
        phase1(seam_c, fields_b)
        phase2(seam_c, fields_b)
    export_c = seam_c.export_state()
    equal, diffs = payloads_equal(export_a, export_c)
    caller_equal = all(
        cp.asnumpy(fields_b[name]).tobytes() == fields_a_end[name].tobytes()
        for name in fields_a_end)
    print(f"restart split-vs-unbroken: payload_equal={equal} "
          f"caller_state_equal={caller_equal} "
          f"({len(export_a['arrays'])} arrays compared)")
    if diffs:
        print("  diffs:", diffs[:10])
    record("P2-restart-bitwise", equal and caller_equal, {
        "arrays_compared": len(export_a["arrays"]),
        "payload_equal": equal, "caller_state_equal": caller_equal,
        "diffs": diffs[:20]})
    del seam_c
    cp.get_default_memory_pool().free_all_blocks()

    # ---- P4a: the guard is intact, named refusal both ways -------------
    seam_d = make_seam(surface, threshold=THRESHOLD, native_xland=True)
    params_off = NoahmpRuntimeParameters(
        xice_threshold=THRESHOLD, glacier_path=False)
    guard_msg = None
    try:
        guard_noahmp_glacier_columns(seam_d._driver.fields, params_off)
    except NoahmpGlacierColumnError as error:
        guard_msg = str(error)
    print("guard refusal:", guard_msg)
    seam_d._driver.noahmp_params.glacier_path = False
    fields_d = build_fields(cp)
    runtime_msg = None
    try:
        phase1(seam_d, fields_d)
    except NoahmpGlacierColumnError as error:
        runtime_msg = str(error)
    print("runtime refusal:", runtime_msg)
    ok = (guard_msg is not None and "glacier guard" in guard_msg
          and "i=30" in guard_msg
          and runtime_msg is not None
          and "glacier_path=False" in runtime_msg
          and "NOAHMP_SFLX is not a substitute" in runtime_msg)
    record("P4a-guard-intact", ok, {
        "guard_message": guard_msg, "runtime_message": runtime_msg})
    del seam_d, fields_d
    cp.get_default_memory_pool().free_all_blocks()

    # ---- P4b: corrupt one glacier state array in the blob --> detected -
    glacier_cols = set(np.flatnonzero(
        (surface["landmask"] == 1.0)
        & (surface["ivgtyp"] == ISICE)).tolist())
    glacier_keys = sorted(
        key for key in saved["arrays"]
        if any(tag in key for tag in
               ("snice", "snliq", "tsno", "zsnso", "tgxy")))
    target_key = None
    target_index = None
    for key in glacier_keys:
        array = saved["arrays"][key]
        if array.shape[-1] != NCOL or array.dtype.itemsize != 4:
            continue
        for flat in np.flatnonzero(array):
            if int(flat) % NCOL in glacier_cols:
                target_key = key
                target_index = int(flat)
                break
        if target_key is not None:
            break
    assert target_key is not None, "no nonzero glacier state array found"
    corrupt = payload_copy(saved)
    view = corrupt["arrays"][target_key].reshape(-1)
    original_word = view.view(np.uint32)[target_index]
    view.view(np.uint32)[target_index] ^= np.uint32(1)
    print(f"corrupted {target_key}[flat {target_index}]: "
          f"0x{int(original_word):08x} -> "
          f"0x{int(view.view(np.uint32)[target_index]):08x} (1-bit flip)")

    seam_clean = make_seam(surface, threshold=THRESHOLD, native_xland=True)
    seam_clean.restore_state(payload_copy(saved))
    seam_corrupt = make_seam(surface, threshold=THRESHOLD,
                             native_xland=True)
    seam_corrupt.restore_state(corrupt)
    fields_clean = build_fields(cp)
    fields_corrupt = build_fields(cp)
    detected_step = None
    detected_detail = None
    for step in range(1, STEPS_SPLIT + 1):
        out_clean = phase1(seam_clean, fields_clean)
        out_corrupt = phase1(seam_corrupt, fields_corrupt)
        for name in ("du", "dv", "dtheta", "dqv"):
            n_diff = int((getattr(out_clean, name) !=
                          getattr(out_corrupt, name)).sum())
            if n_diff:
                detected_step = step
                detected_detail = f"{name}: {n_diff} elements differ"
                break
        if detected_step is None:
            for name in ("tsk", "tgxy", "snow", "snicexy"):
                left = seam_clean._driver.fields[name]
                right = seam_corrupt._driver.fields[name]
                n_diff = int((left != right).sum())
                if n_diff:
                    detected_step = step
                    detected_detail = (f"driver {name}: {n_diff} "
                                       "elements differ")
                    break
        if detected_step is not None:
            break
        phase2(seam_clean, fields_clean)
        phase2(seam_corrupt, fields_corrupt)
    print(f"corruption detection: step={detected_step} "
          f"({detected_detail})")
    record("P4b-corruption-detected", detected_step is not None, {
        "corrupted_key": target_key, "flat_index": target_index,
        "detected_at_step": detected_step, "detail": detected_detail,
        "note": ("restore_state accepts the payload silently (values are "
                 "not checksummed); detection is the dual-run divergence "
                 "instrument, the same detector the pre-port proofs "
                 "validated on fields/tsk")})

    receipt["status"] = "GO" if not failures else "RED"
    receipt["failures"] = failures
    path = out_dir / "glacier_verify.json"
    path.write_text(json.dumps(receipt, indent=1, default=str))
    print(json.dumps({"status": receipt["status"],
                      "failures": failures, "receipt": str(path)}))
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
