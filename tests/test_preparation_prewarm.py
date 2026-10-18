"""The cold preparation cache must be populated beside fetch, not at the root."""
import os
import subprocess
import sys

import pytest

from conftest import requires_gpu


@requires_gpu
@pytest.mark.gpu
def test_preparation_prewarm_caches_root_initialize_kernels(tmp_path):
    cache = tmp_path / "cupy"
    cache.mkdir()
    env = dict(os.environ, CUPY_CACHE_DIR=str(cache))
    script = """
from concurrent.futures import ThreadPoolExecutor
from gpuwm.ingest.preparation_prewarm import preparation_jobs, warm_preparation_job
jobs = preparation_jobs()
with ThreadPoolExecutor(4) as pool:
    list(pool.map(warm_preparation_job, [options for _, options in jobs]))
"""
    assert not list(cache.glob("*.cubin"))
    subprocess.run([sys.executable, "-c", script], env=env, check=True)
    warmed = {path.name for path in cache.glob("*.cubin")}
    assert warmed, "real preparation paths compiled no kernels into the fresh cache"
    # New interpreter: no CuPy memoized function/module can serve this root.
    subprocess.run([sys.executable, "-c", script], env=env, check=True)
    after = {path.name for path in cache.glob("*.cubin")}
    assert after == warmed, sorted(after - warmed)
    print(f"preparation cubins: 0 -> {len(warmed)}; second root_initialize: +{len(after - warmed)}")
