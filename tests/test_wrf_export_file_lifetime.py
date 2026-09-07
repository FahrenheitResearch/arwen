"""Long boundary exports must release old mappings without invalidating views."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from gpuwm.wrf_direct import _array_sha256, _canonical


def test_boundary_export_releases_completed_intervals(tmp_path):
    # 24 intervals * 48 arrays crosses the common 1024-descriptor ceiling.
    # Values differ by interval, so closing/reusing a mapping incorrectly is
    # observable as data corruption as well as a resource failure.
    arrays = {}
    for interval in range(24):
        for logical in ("u", "v", "phi", "theta", "mu", "qv"):
            for side in ("west", "east", "south", "north"):
                for kind in ("value", "tendency"):
                    value = np.full((1 if logical == "mu" else 2, 2, 2),
                                    interval + 1, dtype=np.float32)
                    filename = f"{len(arrays)}.npy"
                    np.save(tmp_path / filename, value)
                    arrays[f"lbc/{interval}/{logical}/{side}/{kind}"] = {
                        "file": filename, "shape": list(value.shape),
                        "dtype": str(value.dtype), "sha256": _array_sha256(value),
                    }
    header = {
        "schema": "gpuwm-prepared-real-cache-v1", "status": "READY",
        "identity": {}, "metadata": {}, "arrays": arrays,
        "payload_bytes": sum(int(np.prod(s["shape"])) * 4 for s in arrays.values()),
    }
    basis = {name: header[name] for name in (
        "schema", "identity", "metadata", "arrays", "payload_bytes")}
    header["content_sha256"] = hashlib.sha256(_canonical(basis).encode()).hexdigest()
    (tmp_path / "header.json").write_text(json.dumps(header), encoding="utf-8")
    code = r'''
import gc
from pathlib import Path
import sys
import weakref
import numpy as np
from gpuwm.wrf_direct import PreparedCache, _wrfbdy_fields
try:
    import resource
except ImportError:
    pass  # Windows still checks live mappings and retained-view correctness.
else:
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    limit = 128 if hard == resource.RLIM_INFINITY else min(128, hard)
    resource.setrlimit(resource.RLIMIT_NOFILE, (limit, hard))
opened = []
load = np.load
def observed_load(*args, **kwargs):
    value = load(*args, **kwargs)
    opened.append(weakref.ref(value))
    return value
np.load = observed_load
cache = PreparedCache(Path(sys.argv[1]))
first = cache.array("lbc/0/u/west/value")
assert cache.array("lbc/0/u/west/value") is first
retained_view = first[:, :, 0]
del first
for interval in range(24):
    fields = _wrfbdy_fields(cache, interval)
    for field in fields.values():
        np.testing.assert_array_equal(field, interval + 1)
    del field
    gc.collect()
    live = sum(ref() is not None for ref in opened)
    assert live <= len(fields) + 1, (interval, live)
np.testing.assert_array_equal(retained_view, 1)
del fields, retained_view
gc.collect()
assert not any(ref() is not None for ref in opened)
print("24 intervals exported; live views preserved; all mappings released")
'''
    completed = subprocess.run(
        [sys.executable, "-B", "-c", code, str(tmp_path)],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True, text=True, timeout=90)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "all mappings released" in completed.stdout
