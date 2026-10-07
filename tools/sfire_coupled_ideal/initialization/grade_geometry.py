"""Compare actual initializer device outputs against compiled source blocks."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np


def load(root, name):
    root = Path(root)
    receipt = json.loads((root / "receipt.json").read_text())
    path = root / (name + ".npz")
    if hashlib.sha256(path.read_bytes()).hexdigest() != receipt["cases"][name]["sha256"]:
        raise ValueError("native ideal initialization corpus hash differs")
    with np.load(path, allow_pickle=False) as f:
        return dict(f)


def replay(root, name):
    import cupy as cp
    from gpuwm.core import sfire_ideal as port
    from tools.sfire_wrf471_oracle.fixture import words
    expected = load(root, name)
    if "/vertical_" in name:
        stretch = int(expected["stretch"])
        result = port.build_vertical(int(expected["nz"]), p_top=expected["p_top"],
            dx=expected["dx"], dy=expected["dy"], stretch_grd=stretch > 0,
            stretch_hyp=stretch == 2, z_grd_scale=expected["scale"],
            eta_levels=expected["eta_in"] if stretch == 3 else None,
            hybrid_opt=int(expected["hybrid"]), etac=float(expected["etac"]))
        actual = {key: cp.asnumpy(result[key]) for key in port.VERTICAL_FIELDS}
        actual["scalars"] = cp.asnumpy(cp.stack([result[k] for k in port.VERTICAL_SCALARS]))
    elif "/mountain_" in name:
        args = dict(kind=int(expected["kind"]), height=375., start_x=float(expected["xs"]),
            start_y=float(expected["ys"]), end_x=float(expected["xe"]), end_y=float(expected["ye"]))
        actual = {key: cp.asnumpy(port.mountain(expected[key].shape, expected[x], expected[y], **args))
                  for key, x, y in (("height", "dx", "dy"), ("zsf", "fdx", "fdy"))}
    elif "/gradient_" in name:
        gx, gy = port.terrain_gradient(expected["zsf"], expected["fdx"], expected["fdy"])
        actual = dict(gx=cp.asnumpy(gx), gy=cp.asnumpy(gy))
    elif name.endswith("/coordinates"):
        x, y = port.coordinates(expected["x"].shape, expected["dx"], expected["dy"])
        actual = dict(x=cp.asnumpy(x), y=cp.asnumpy(y))
    elif name.endswith("/interpolation_edges"):
        from gpuwm.core.sfire_coupler import _pad
        from gpuwm.core.sfire_atm import interpolate_2d
        a = _pad(expected["coarse"], expected["coarse"].shape, linear=True)
        actual = dict(corrected=cp.asnumpy(interpolate_2d(a, expected["corrected"].shape, 3, 4,
            coarse_origin=(1.,1.), fine_origin=(1.,1.5))))
        assert np.count_nonzero(expected["original_zero"] != expected["original_sentinel"]) > 0
        assert np.all(expected["corrected"] != -999.)
    else:
        result = port.soil(expected["tsk"], expected["tmn"], scheme=int(expected["scheme"]), layers=int(expected["layers"]))
        actual = {key: cp.asnumpy(value) for key, value in result.items()}
        if int(expected["scheme"]) == 3:
            assert np.count_nonzero(expected["original_dzs"] != expected["dzs"]) == 1
    return {key: words(value, expected[key]) for key, value in actual.items()}


def grade(root, destination):
    import cupy as cp
    receipt = json.loads((Path(root) / "receipt.json").read_text())
    cases = {name: replay(root, name) for name in receipt["cases"]}
    entries = [value for case in cases.values() for value in case.values()]
    summary = dict(cases=len(cases), words=sum(row["words"] for row in entries),
        different_words=sum(row["different_words"] for row in entries),
        max_ulp=max(row["max_ulp"] for row in entries))
    result = dict(summary=summary, device=cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)["name"].decode(),
        native_receipt_sha256=hashlib.sha256((Path(root)/"receipt.json").read_bytes()).hexdigest(),
        source_sha256={name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in
            ("gpuwm/core/sfire_ideal.py", "gpuwm/core/kernels/sfire_ideal.cu")}, cases=cases)
    Path(destination).write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(summary))
    print(json.dumps({name: {key: row for key, row in case.items() if row["different_words"]}
                      for name,case in cases.items() if any(row["different_words"] for row in case.values())}))
    if summary["different_words"]:
        raise SystemExit(1)


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("root")
    parser.add_argument("destination")
    args=parser.parse_args()
    grade(args.root,args.destination)
