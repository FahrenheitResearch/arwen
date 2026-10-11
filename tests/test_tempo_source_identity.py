"""A header correction must not replay the cached older CUDA kernel."""
from gpuwm.core.tempo_runtime import _source_fingerprint


def test_included_header_and_numeric_policy_changes_invalidate_the_cache(tmp_path):
    headers = tmp_path / "tempo"
    headers.mkdir()
    policy = tmp_path / "glibc_flt64.cuh"
    policy.write_text("policy one")
    source = headers / "tempo_column.cuh"
    source.write_text("column one")
    first = _source_fingerprint(headers)
    assert _source_fingerprint(headers) == first
    source.write_text("column two")
    second = _source_fingerprint(headers)
    assert second != first
    policy.write_text("policy two")
    assert _source_fingerprint(headers) not in (first, second)
