"""Artifact-backed composition-cache admission and atomic-publication controls."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
import os
from pathlib import Path
import shutil
import threading
from types import SimpleNamespace

import pytest
import gpuwm
from gpuwm.chem_table import catalog
import gpuwm.chem_source_init as source
import gpuwm.chem_boundary_remap as remap
import gpuwm.grib2_stack as stack
from gpuwm.ingest.cpu_backend import CPU_BRIDGE_ENV, resolve_cpu_bridge

ROOT = Path(gpuwm.__file__).resolve().parents[1]


@pytest.fixture
def case(tmp_path, monkeypatch):
    library = resolve_cpu_bridge()
    selected = tmp_path / library.name
    shutil.copyfile(library, selected)
    monkeypatch.setenv(CPU_BRIDGE_ENV, str(selected))
    row = catalog().sources['cams-global']
    variable = next(key for key, spec in row.variables.items()
                    if spec['group'] == 'model_level' and spec['selector'].get('constituent_type') is not None)
    fixture = ROOT / 'tests/data/chem_cams'
    original = fixture / 'SYNTHETIC-cams-ml-20250730T00.grib2'
    replacement = fixture / 'SYNTHETIC-cams-sl-20250730T00.grib2'
    raw = tmp_path / 'same-path.grib2'
    shutil.copyfile(original, raw)
    calls = []
    native = stack.stack_grib2

    def counted(path, select, output):
        calls.append((resolve_cpu_bridge(), Path(output)))
        return native(path, select, output)

    monkeypatch.setattr(stack, 'stack_grib2', counted)
    source._DECODED.clear()
    yield SimpleNamespace(row=row, variable=variable, raw=raw, original=original,
                          replacement=replacement, library=selected, calls=calls,
                          files={'model_level': raw}, variables=(variable,), root=tmp_path / 'cache')
    source._DECODED.clear()


def frames(case, *, row=None, root=None):
    return source._frames(row or case.row, case.files, case.variables, root or case.root)


def alternate_library(case, monkeypatch):
    alternate = case.library.with_name('alternate-' + case.library.name)
    shutil.copyfile(case.library, alternate)
    # A valid shared object's opaque tail changes its exact byte identity.
    with alternate.open('ab') as stream:
        stream.write(b'cache-authority-alternate-fixture')
    monkeypatch.setenv(CPU_BRIDGE_ENV, str(alternate))
    return alternate


def test_unchanged_memory_and_disk_hits_do_not_decode_again(case):
    first = frames(case)
    assert frames(case) is first
    source._DECODED.clear()
    again = frames(case)
    assert len(case.calls) == 1
    assert [field.file for frame in again.frames.values() for field in frame.values()] == [
        field.file for frame in first.frames.values() for field in frame.values()]


@pytest.mark.parametrize('clear_memory', [False, True])
def test_selected_artifact_byte_change_invalidates_memory_and_disk(case, monkeypatch, clear_memory):
    first = frames(case)
    if clear_memory:
        source._DECODED.clear()
    alternate = alternate_library(case, monkeypatch)
    second = frames(case)
    assert second is not first and len(case.calls) == 2
    assert case.calls[-1][0] == alternate.resolve()


@pytest.mark.parametrize('clear_memory', [False, True])
def test_changed_selector_refuses_instead_of_reusing_old_fields(case, clear_memory):
    frames(case)
    if clear_memory:
        source._DECODED.clear()
    variables = dict(case.row.variables)
    spec = dict(variables[case.variable])
    spec['selector'] = dict(spec['selector'], constituent_type=65534)
    variables[case.variable] = spec
    changed = replace(case.row, variables=variables)
    with pytest.raises(ValueError, match='empty'):
        frames(case, row=changed)
    assert len(case.calls) == 2
    assert not list(case.root.rglob('*.stage-*'))


def test_same_raw_path_replaced_is_hashed_before_memory_hit(case):
    frames(case)
    shutil.copyfile(case.replacement, case.raw)
    with pytest.raises(ValueError, match='empty'):
        frames(case)
    assert len(case.calls) == 2


def test_new_cache_root_does_not_return_cleaned_first_roots_lazy_files(case):
    first = frames(case)
    shutil.rmtree(case.root)
    second_root = case.root.with_name('second-cache')
    second = frames(case, root=second_root)
    assert second is not first and len(case.calls) == 2
    assert all(second_root in field.file.parents and field.file.is_file()
               for frame in second.frames.values() for field in frame.values())


def test_memory_hit_still_checks_lazy_field_integrity(case):
    first = frames(case)
    field = next(iter(next(iter(first.frames.values())).values()))
    field.file.write_bytes(b'x' * field.file.stat().st_size)
    with pytest.raises(ValueError, match='sha256 mismatch'):
        frames(case)
    assert len(case.calls) == 1


def test_outer_authority_is_carried_into_inner_admission(case, monkeypatch):
    decode = source.decode_source_files

    def changed_before_inner(*args, **kwargs):
        shutil.copyfile(case.replacement, case.raw)
        return decode(*args, **kwargs)

    monkeypatch.setattr(source, 'decode_source_files', changed_before_inner)
    with pytest.raises(remap.BoundaryRemapError, match='changed before decode'):
        frames(case)
    assert not source._DECODED and not case.calls


def test_outer_postdecode_check_prevents_wrong_memory_admission(case, monkeypatch):
    decode = source.decode_source_files

    def changed_after_inner(*args, **kwargs):
        result = decode(*args, **kwargs)
        shutil.copyfile(case.replacement, case.raw)
        return result

    monkeypatch.setattr(source, 'decode_source_files', changed_after_inner)
    with pytest.raises(remap.BoundaryRemapError, match='changed during decode'):
        frames(case)
    assert not source._DECODED
    shutil.copyfile(case.original, case.raw)
    monkeypatch.setattr(source, 'decode_source_files', decode)
    frames(case)
    assert len(case.calls) == 1  # correctly published original disk payload reuses


def test_raw_change_during_staging_publishes_nothing(case, monkeypatch):
    native = stack.stack_grib2

    def replace_after_decode(*args):
        result = native(*args)
        shutil.copyfile(case.replacement, case.raw)
        return result

    monkeypatch.setattr(stack, 'stack_grib2', replace_after_decode)
    with pytest.raises(remap.BoundaryRemapError, match='changed during decode'):
        frames(case)
    assert not source._DECODED and not list(case.root.rglob('stack.json'))
    assert not list(case.root.rglob('*.stage-*'))


def test_complete_selector_metadata_is_owned_and_canonical(case):
    select = remap.grib2_selection(case.row, case.variables)
    select[0]['levels'] = [1, 2]
    identity = stack.stack_cache_identity(case.raw, select)
    select[0]['levels'].append(3)
    assert identity['specification']['select'][0]['levels'] == [1, 2]
    original = remap.grib2_selection(case.row, case.variables)
    base = stack.stack_cache_digest(stack.stack_cache_identity(case.raw, original))
    reordered = [dict(reversed(list(original[0].items())))]
    assert stack.stack_cache_digest(stack.stack_cache_identity(case.raw, reordered)) == base
    for name, value in dict(key='other', discipline=7, category=1, parameter=99,
                            constituent_type=65534, aerosol_type=3, level_type=1, levels=[1]).items():
        changed = [dict(original[0], **{name: value})]
        assert stack.stack_cache_digest(stack.stack_cache_identity(case.raw, changed)) != base, name


def test_two_same_process_decodes_have_unique_staging_and_atomic_publication(case, monkeypatch):
    native = stack.stack_grib2
    barrier = threading.Barrier(2)
    staging = []

    def together(path, select, output):
        staging.append(Path(output))
        barrier.wait(timeout=10)
        return native(path, select, output)

    monkeypatch.setattr(stack, 'stack_grib2', together)
    select = remap.grib2_selection(case.row, case.variables)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(remap._stack_once, case.raw, select, case.root / 'same-output') for _ in range(2)]
        results = [future.result(timeout=30) for future in futures]
    assert len(set(staging)) == 2
    assert [field.file for field in results[0]] == [field.file for field in results[1]]
    assert len(list(case.root.rglob('stack.json'))) == 1
    assert not list(case.root.glob('*.stage-*'))


def test_unsupported_format_keeps_named_refusal_without_decoder_dependency(monkeypatch, tmp_path):
    monkeypatch.setattr('gpuwm.ingest.cpu_backend.resolve_cpu_bridge',
                        lambda *args: pytest.fail('unsupported format must not resolve a decoder'))
    row = SimpleNamespace(name='unsupported', format='netcdf', variables={})
    with pytest.raises(remap.BoundaryRemapError, match='unsupported.*netcdf.*decodes GRIB2'):
        source._frames(row, {}, (), tmp_path)


def test_inactive_chemistry_has_no_decoder_dependency(monkeypatch):
    monkeypatch.setattr('gpuwm.ingest.cpu_backend.resolve_cpu_bridge',
                        lambda *args: pytest.fail('inactive chemistry must not resolve a decoder'))
    assert source.fill_boundary_sources(SimpleNamespace(chem=None), SimpleNamespace(),
                                       valid_time=None, latlon=None, pressure=None) == []
