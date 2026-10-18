"""A bundle may reuse source decodes, but never across changed files or bundle calls."""
from pathlib import Path

import pytest
from gpuwm import netcdf_bridge
from gpuwm.io import analysis_bundle as B


def test_input_reuse_does_not_cache_outputs_or_cross_bundle_calls(tmp_path, monkeypatch):
    source=tmp_path/'source';source.write_bytes(b'background')
    target=tmp_path/'target';target.write_bytes(b'analysis')
    calls=[]
    def opened(path):
        calls.append(Path(path));return {'bytes':Path(path).read_bytes()}
    monkeypatch.setattr(netcdf_bridge,'open_dataset',opened)
    @B._reuse_input_reads
    def bundle(*,background):
        first=B._open(background)
        assert B._open(background) is first
        assert B._open(target) is not B._open(target)
        return first
    first=bundle(background=source)
    assert calls.count(source)==1 and calls.count(target)==2
    assert bundle(background=source) is not first
    assert calls.count(source)==2
    assert B._RESIDENT_INPUTS.get() is None


def test_replaced_input_is_not_served_from_cached_decode(tmp_path,monkeypatch):
    source=tmp_path/'source';source.write_bytes(b'old')
    monkeypatch.setattr(netcdf_bridge,'open_dataset',lambda p:Path(p).read_bytes())
    @B._reuse_input_reads
    def bundle(*,background):
        assert B._open(background)==b'old'
        fresh=tmp_path/'fresh';fresh.write_bytes(b'new');fresh.replace(background)
        assert B._open(background)==b'new'
    bundle(background=source)


def test_failed_bundle_releases_resident_sources(tmp_path):
    source=tmp_path/'source';source.touch()
    @B._reuse_input_reads
    def bundle(*,background):
        assert B._RESIDENT_INPUTS.get() is not None
        raise RuntimeError('fixture failure')
    with pytest.raises(RuntimeError,match='fixture failure'):
        bundle(background=source)
    assert B._RESIDENT_INPUTS.get() is None


@pytest.mark.parametrize("keyword", [False, True])
def test_output_reuse_invalidates_balanced_replacement(tmp_path, monkeypatch, keyword):
    calls=[]
    def opened(path):
        calls.append(Path(path));return {"bytes": Path(path).read_bytes()}
    monkeypatch.setattr(netcdf_bridge, 'open_dataset', opened)
    target=tmp_path/B.ANALYSIS
    @B._reuse_input_reads
    def bundle(out, *, background):
        target.write_bytes(b'producer')
        first=B._open(target)
        assert B._open(target) is first
        fresh=tmp_path/'balanced';fresh.write_bytes(b'balanced');fresh.replace(target)
        final=B._open(target)
        assert final is not first and final['bytes']==b'balanced'
        assert B._open(target) is final
        return final
    final=bundle(out=tmp_path,background=tmp_path/'source') if keyword else bundle(tmp_path,background=tmp_path/'source')
    assert len(calls)==2
    assert B._open(target) is not final
    assert B._RESIDENT_INPUTS.get() is None
