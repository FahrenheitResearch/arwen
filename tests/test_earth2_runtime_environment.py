"""Runtime setup must precede optional imports that can initialize CUDA."""
import builtins
import importlib.metadata
import json
import os

import pytest

from gpuwm import earth2_runner


@pytest.mark.parametrize("caller,row,expected", [
    (None, {}, "expandable_segments:True"),
    ("max_split_size_mb:128", {}, "max_split_size_mb:128"),
    (None, {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:False"},
     "expandable_segments:False"),
])
def test_allocator_is_configured_before_optional_imports(monkeypatch, tmp_path,
                                                        caller, row, expected):
    # Exercise the real entry point through its first optional import. No
    # Earth2Studio/Torch installation or CUDA device is needed for this check.
    monkeypatch.setenv("GPUWM_NO_LOCAL_GPU", "0")
    if caller is None:
        monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
    else:
        monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", caller)
    monkeypatch.setattr(importlib.metadata, "version", lambda _: "0.19.0")
    request = tmp_path / "request.json"
    request.write_text(json.dumps({"pinned_version": "0.19.0", "runtime_env": row,
                                   "weight_cache": str(tmp_path / "weights"),
                                   "data_cache": str(tmp_path / "data")}))
    original_import = builtins.__import__
    observed = {}

    class BeforeOptionalImports(Exception):
        pass

    def intercept(name, *args, **kwargs):
        if name in ("numpy", "torch") or name.startswith("earth2studio"):
            observed.update(first_import=name,
                            allocator=os.environ.get("PYTORCH_CUDA_ALLOC_CONF"))
            raise BeforeOptionalImports
        return original_import(name, *args, **kwargs)

    with monkeypatch.context() as scope:
        scope.setattr(builtins, "__import__", intercept)
        with pytest.raises(BeforeOptionalImports):
            earth2_runner.main(["earth2_runner.py", str(request)])
    assert observed == {"first_import": "numpy", "allocator": expected}
