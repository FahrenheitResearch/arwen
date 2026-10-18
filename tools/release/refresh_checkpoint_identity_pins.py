"""Record a checkpoint identity change after a byte comparison to the base.

The historical array and format pins stay fixed. Only the current digest
pair is generated, and only when the driver identity and its derived
fingerprint are the entire header difference.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
BASE_DRIVER = "gpuwm-physics-driver-v3-kf-phase-energy-pre-mp-expiry"
BASE_DIGESTS = {
    "d01": "8b2e93ccad34802fb47061530dd962af456cda2c8035eec7da4cbd52ed53fddb",
    "d02": "7b8abced0192f5bd55370d208e16c76d667bf2baa2292ef59d3af0d482d698a6",
}


def capture(tree, output):
    sys.path[:0] = [str(tree / "tests"), str(tree)]
    import pytest
    import test_restart as tests

    patch = pytest.MonkeyPatch()
    try:
        model, start = tests._sealed_tree_fixture(
            patch, forcing_count=2, run_seconds=3600., payload_seed=31,
            run_overrides=tests._PINNED_UNDER_MEASURED_CLOCK)
        tests.restart.write_tree_restart(
            output, model, start + tests.timedelta(seconds=3600))
        return {
            p.name.split("_")[1]: tests._canonical_member_digest(p)
            for p in output.glob("*.npz")
        }
    finally:
        patch.undo()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-tree", type=Path)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--capture-tree", type=Path)
    parser.add_argument("--capture-out", type=Path)
    args = parser.parse_args()
    assert os.environ.get("GPUWM_NO_LOCAL_GPU") == "1"
    assert os.environ.get("GPUWM_WRF_EXACT") == "0"
    if args.capture_tree:
        args.capture_out.mkdir()
        result = capture(args.capture_tree.resolve(), args.capture_out)
        (args.capture_out / "digests.json").write_text(json.dumps(result))
        return
    assert args.baseline_tree and args.receipt
    revision = subprocess.check_output(
        ["git", "-C", str(args.baseline_tree), "rev-parse", "HEAD"], text=True).strip()
    assert revision == "cd336e62d6e6b6312c9636e0fdfff84801909b6a"
    from gpuwm.io import restart
    import numpy as np

    rows, deleted = [], []
    with tempfile.TemporaryDirectory(prefix="checkpoint-identity-", dir=os.environ["TMPDIR"]) as tmp:
        temp = Path(tmp)
        for arm, tree in (("base", args.baseline_tree.resolve()), ("tip", ROOT)):
            env = dict(os.environ, GPUWM_NO_LOCAL_GPU="1", GPUWM_WRF_EXACT="0",
                       CUDA_VISIBLE_DEVICES="-1", PYTHONPATH=str(tree) + ":" + str(tree / "tests"))
            subprocess.run([sys.executable, str(Path(__file__).resolve()),
                            "--capture-tree", str(tree), "--capture-out", str(temp / arm)],
                           cwd=tree, env=env, check=True)
        old = json.loads((temp / "base/digests.json").read_text())
        new = json.loads((temp / "tip/digests.json").read_text())
        assert old == BASE_DIGESTS and set(new) == set(old)
        for grid in sorted(old):
            before = next((temp / "base").glob("gpuwmrst_" + grid + "_*.npz"))
            after = next((temp / "tip").glob("gpuwmrst_" + grid + "_*.npz"))
            with np.load(before, allow_pickle=False) as a, np.load(after, allow_pickle=False) as b:
                assert set(a.files) == set(b.files)
                h0 = json.loads(a[restart._HEADER_KEY].tobytes())
                h1 = json.loads(b[restart._HEADER_KEY].tobytes())
                for header in (h0, h1):
                    for key in ("created", "producer", "checkpoint_set_id", "written_mode"):
                        header.pop(key, None)
                assert h0["physics_setup"]["algorithms"]["physics_driver"] == BASE_DRIVER
                assert h1["physics_setup"]["algorithms"]["physics_driver"] == restart.PHYSICS_DRIVER_ALGORITHM_IDENTITY
                expected = json.loads(json.dumps(h0))
                expected["physics_setup"]["algorithms"]["physics_driver"] = restart.PHYSICS_DRIVER_ALGORITHM_IDENTITY
                expected["physics_setup_fingerprint"] = restart._json_sha256(expected["physics_setup"])
                assert expected == h1, "the identity is not the entire checkpoint header change"
                for name in sorted(set(a.files) - {restart._HEADER_KEY}):
                    x, y = a[name], b[name]
                    assert x.shape == y.shape and x.dtype == y.dtype, name
                    xb, yb = x.tobytes(), y.tobytes()
                    sha0, sha1 = hashlib.sha256(xb).hexdigest(), hashlib.sha256(yb).hexdigest()
                    assert xb == yb, name
                    rows.append(dict(grid=grid, name=name, values=int(x.size),
                                     sha256_before=sha0, sha256_after=sha1,
                                     byte_compare_performed=True, differing_values=0))
        text = (ROOT / "tests/test_restart.py").read_text()
        for grid, label in (("d01", "ROOT"), ("d02", "CHILD")):
            pattern = r'(_LIFECYCLE_FREE_' + label + r'_DIGEST\s*=\s*\\?\s*)"[0-9a-f]{64}"'
            text, count = re.subn(pattern, lambda match: match[1] + '"' + new[grid] + '"', text)
            assert count == 1
        if args.write:
            (ROOT / "tests/test_restart.py").write_text(text)
        for p in temp.rglob("*"):
            if p.is_file():
                deleted.append(dict(path=str(p), bytes=p.stat().st_size))
    receipt = dict(baseline_revision=revision, driver_before=BASE_DRIVER,
                   driver_after=restart.PHYSICS_DRIVER_ALGORITHM_IDENTITY,
                   current_digests_before=old, current_digests_after=new,
                   fields_compared=len(rows), values_compared=sum(r["values"] for r in rows),
                   differing_values=0, rows=rows, deleted=deleted, pins_written=args.write)
    args.receipt.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({k: receipt[k] for k in ("fields_compared", "values_compared", "differing_values", "pins_written")}))


if __name__ == "__main__":
    main()
