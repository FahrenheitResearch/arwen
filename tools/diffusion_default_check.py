"""Mechanical diffusion gate, with the production launchers and sealed WRF words.

Run with GPUWM_WRF_EXACT unset for the default arm, or set to 1 for strict.
Each scheme runs in a fresh process so oracle helper imports cannot collide.
The km3 seal holds full-field hashes rather than reference arrays; its gate
reports differing arrays, and reports zero differing words only if all match.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "tests"), str(ROOT)]
SCHEMES = ("km4", "km1", "km2", "km3", "diff1", "diff6", "advance_w", "w_damp")


def test_module(name):
    path = ROOT / "tests" / f"{name}.py"
    if not path.is_file():
        raise RuntimeError(
            f"checkout-only diffusion verification needs its declared test helper {path}; "
            "without that helper and its sealed WRF fixtures this command cannot compare "
            "the production words. Run from the complete selected source delivery.")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def digest(a):
    return hashlib.sha256(np.ascontiguousarray(a, np.float32).tobytes()).hexdigest()


def worker(scheme, out):
    from gpuwm.verify.diffusion_oracle import word_comparison
    from gpuwm import wrf_exact
    rows = {}
    excluded = 0

    def compare(key, got, want):
        rows[key] = {**word_comparison(got, want), "gpu_sha256": digest(got),
                     "wrf_sha256": digest(want)}

    if scheme == "km4":
        path = out.with_suffix(".detail.json")
        subprocess.run([sys.executable, str(ROOT / "tools/smag2d_wrf461_oracle/oracle.py"),
                        "compare", "--cases", str(ROOT / "tests/data/smag2d_wrf461"),
                        "--reference", "wrfctl", "--receipt", str(path)], check=True)
        data = json.loads(path.read_text())
        assert data["case_count"] == 27
        rows = {f"{case}/{field}": value for case, fields in data["cases"].items()
                for field, value in fields.items() if not field.startswith("_")}
    elif scheme == "km1":
        m = test_module("test_km1_wrf461_oracle")
        try:
            from tools.wrf_diffusion_oracle import km1_oracle
        except ImportError as error:
            raise RuntimeError(
                "checkout-only diffusion verification needs "
                "tools/wrf_diffusion_oracle/km1_oracle.py from the selected source delivery; "
                "without its WRF oracle driver no km_opt=1 reference can be compared") from error
        for entry in m._manifest()["cases"]:
            arrays, meta, reference = m._case(entry)
            for arm in km1_oracle.ARMS:
                got = km1_oracle.woof_outputs(arrays, meta, arm)
                for field in km1_oracle.OUTPUTS + ("coef_kmh", "coef_kmv", "coef_khh", "coef_khv"):
                    compare(f"{entry['name']}/{arm[0]}/{field}", got[field], reference[f"{arm[0]}__{field}"])
    elif scheme == "km2":
        m = test_module("test_tke_km2_wrf461_column_oracle")
        rows = m._tools_module("check_fixture").check(m.FIXTURE)
        rows = {field: {"words": r["words"], "different_words": r["different"],
                        "max_ulp": r["max_ulp"]} for field, r in rows.items()}
    elif scheme == "km3":
        m = test_module("test_km3_wrf461_column_oracle")
        tool = m._tool("compare")
        for name, row in m._digests()["cases"].items():
            refdir = os.environ.get("GPUWM_DIFFUSION_KM3_FIXTURES")
            if refdir:
                with np.load(Path(refdir) / f"{name}.npz") as data:
                    reference = {key: data[key] for key in data.files if key.startswith("wrf__")}
            for iso in (0, 1):
                got = tool.engine_case(m._inputs(name, row), row["meta"], iso)
                for key, value in got.items():
                    arm, field = key.split("__")
                    refkey = (tool.reference_key(arm, field, iso) if arm == "coef" else
                              f"wrf__iso{iso}__h_{field}" if arm == "h_chain" else
                              f"wrf__iso{iso}__v{arm[-1]}_wcontrol_{field}")
                    expected = row["wrf_sha256"][refkey]
                    actual = digest(value)
                    if refdir:
                        assert digest(reference[refkey]) == expected, (name, refkey, "rebuilt WRF seal moved")
                        compare(f"{name}/{iso}/{key}", value, reference[refkey])
                        continue
                    rows[f"{name}/{iso}/{key}"] = {"words": int(value.size),
                        "different_words": 0 if actual == expected else None,
                        "different_arrays": int(actual != expected),
                        "gpu_sha256": actual, "wrf_sha256": expected}
    elif scheme == "diff1":
        path = out.with_suffix(".detail.json")
        subprocess.run([sys.executable, str(ROOT / "tools/diffopt1_wrf461_oracle/oracle.py"),
                        "replay", "--sealed", str(ROOT / "tests/data/diffopt1_wrf461"),
                        "--json", str(path)], check=True)
        data = json.loads(path.read_text())
        rows = {f"{case}/{field}": {"words": row["words"], "different_words": row["ctl"]}
                for case, fields in data["cases"].items() for field, row in fields.items()}
    elif scheme == "diff6":
        m = test_module("test_diff6_wrf461_column_oracle")
        meta, fixture = m.oracle()
        for case in meta["cases"]:
            values = m.inputs(fixture, case["case"])
            got, want = m.product(case, values), values["reference"]
            skip = m.v_open_x_rows(case)
            if skip is not None:
                excluded += int(want[:, skip].size)
                keep = np.ones(want.shape[1], dtype=bool)
                keep[skip] = False
                got, want = got[:, keep], want[:, keep]
            compare(str(case["case"]), got, want)
    else:
        m = test_module("test_upper_damping_wrf461_oracle")
        from tools.upper_damping_wrf461_oracle import columns, run_oracle
        from tools.smallstep_wrf471_oracle import vertical_workspace
        if not m.FIXTURE.exists():
            raise FileNotFoundError(m.FIXTURE)
        fx = m._fixture()
        for map_name in m.MAPS:
            raw = m._raw(fx, map_name)
            if scheme == "advance_w":
                module, workspace = run_oracle._vertical_module()
                edge, _ = columns.edge_zdamp(raw)
                for case, metadata in run_oracle.advance_w_cases(edge):
                    context = patch.object(vertical_workspace, "workspace_source", workspace) if workspace else __import__("contextlib").nullcontext()
                    with context:
                        got = module.vertical_port_outputs(raw, metadata)
                    for field in ("w", "ph"):
                        compare(f"{map_name}/{case}/{field}", got[field], fx[f"{map_name}/{case}/{field}_wrf"])
            else:
                for dt in (run_oracle.DT_HRRR, 30.0):
                    inp = dict(ww=fx[f"w_damp/{map_name}/dt{int(dt)}/ww"], rw_t=fx[f"w_damp/{map_name}/dt{int(dt)}/rw_t"],
                               w=raw["W"], mub=raw["MUB"], mup=raw["MU"], u=raw["U"], v=raw["V"],
                               msfu=raw["MAPFAC_U"], msfv=raw["MAPFAC_V"], c1f=raw["C1F"], c2f=raw["C2F"], rdnw=raw["RDNW"])
                    for case, crit, ieva in run_oracle.W_DAMP_CASES:
                        key = f"w_damp/{map_name}/dt{int(dt)}/{case}"
                        rw, vmax, hmax, _ = run_oracle._woof_w_damp(inp, dt, 3000.0, 3000.0, crit, ieva)
                        compare(key + "/rw", rw, fx[key + "/rw_wrf"])
                        compare(key + "/cfl", np.array([vmax, hmax], np.float32), fx[key + "/max_cfl_wrf"])
    unknown = any(row["different_words"] is None for row in rows.values())
    result = {"scheme": scheme, "strict": wrf_exact.ENABLED, "rows": rows,
              "engine_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "words": sum(r["words"] for r in rows.values()),
              "different_words": None if unknown else sum(r["different_words"] for r in rows.values()),
              "different_arrays": sum(r.get("different_arrays", int(bool(r["different_words"]))) for r in rows.values()),
              "excluded_wrf_defect_words": excluded}
    from gpuwm.certify.kernel_manifest import kernel_manifest
    result["compile_manifest"] = kernel_manifest()
    result["pass"] = not unknown and result["different_words"] == 0
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("rows", "compile_manifest")}), flush=True)
    return int(not result["pass"])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--scheme", choices=SCHEMES)
    a = p.parse_args()
    if a.scheme:
        return worker(a.scheme, a.out)
    a.out.mkdir(parents=True, exist_ok=True)
    summary = {}
    for scheme in SCHEMES:
        target = a.out / f"{scheme}.json"
        process = subprocess.run([sys.executable, __file__, "--scheme", scheme, "--out", str(target)])
        summary[scheme] = {"exit_code": process.returncode}
        if target.exists():
            summary[scheme].update({k: v for k, v in json.loads(target.read_text()).items()
                                   if k not in ("rows", "compile_manifest")})
    (a.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return int(any(r["exit_code"] != 0 for r in summary.values()))


if __name__ == "__main__":
    raise SystemExit(main())
