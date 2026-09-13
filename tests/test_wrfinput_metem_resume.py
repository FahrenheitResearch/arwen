"""A WRF-input run claims its output directory before it claims a card.

Resume-into-the-same-directory was blocked on these two doors by a bare
`outdir.mkdir(exist_ok=False)` inside the worker child, which fired as
`[Errno 17] File exists` AFTER select_gpu, GPUFileLock and
preflight_exclusive_gpu had reserved a GPU. The directory decision happens
at plan review. Each resumed attempt writes into its own generation, so
replaying an older checkpoint preserves frames named by earlier receipts.
"""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from gpuwm.filesystem_paths import canonical_path


class _Lock:
    def __init__(self, *args, **kwargs):
        raise AssertionError('the GPU was locked for a refused --outdir')

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def _no_gpu(monkeypatch):
    """Make any GPU work in this process an outright test failure."""

    import subprocess

    from gpuwm import go_cli, supervisor

    monkeypatch.setattr(go_cli, 'render_extra_missing', lambda: None)
    monkeypatch.setattr(supervisor, 'select_gpu', lambda uuid=None: (_ for _ in ()).throw(
        AssertionError('a card was selected for a refused --outdir')))
    monkeypatch.setattr(supervisor, 'GPUFileLock', _Lock)
    monkeypatch.setattr(supervisor, 'preflight_exclusive_gpu', lambda *a, **k: (_ for _ in ()).throw(
        AssertionError('the GPU preflight ran for a refused --outdir')))
    monkeypatch.setattr(subprocess, 'run', lambda *a, **k: (_ for _ in ()).throw(
        AssertionError('the worker child was launched for a refused --outdir')))


def _resumable(outdir: Path) -> Path:
    """A directory holding one finished attempt and its checkpoint."""

    (outdir / 'wrfout').mkdir(parents=True)
    (outdir / 'wrfout' / 'wrfout_d01_2026-05-17_18_00_00').write_bytes(b'CDF')
    (outdir / 'evidence').mkdir()
    (outdir / 'evidence' / 'run-receipt.json').write_text(
        json.dumps({'status': 'INTERRUPTED'}), encoding='utf-8')
    checkpoint = outdir / 'restart_d01_2026-05-17_19_00_00.gpuwmrst'
    checkpoint.write_bytes(b'RST')
    return checkpoint


def test_a_non_empty_outdir_is_refused_before_the_card(tmp_path, monkeypatch, capsys):
    from gpuwm import wrfinput_door
    from gpuwm.wrfinput_forecast import run_wrf_forecast

    _no_gpu(monkeypatch)
    monkeypatch.setattr(wrfinput_door, 'resolve_wrfinput_run',
                        lambda *args, **kwargs: SimpleNamespace())
    outdir = tmp_path / 'out'
    (outdir / 'wrfout').mkdir(parents=True)
    (outdir / 'wrfout' / 'wrfout_d01_2026-05-17_18_00_00').write_bytes(b'CDF')
    assert run_wrf_forecast(tmp_path / 'wrf', outdir) == 2
    said = capsys.readouterr().err
    assert str(outdir.resolve()) in said
    assert '--outdir' in said and 'remove the old directory' in said


def _adoption_door(monkeypatch, recorded):
    """The wrfinput worker body with everything but the claim replaced."""

    import datetime

    from gpuwm import go_cli, prepared_domain_tree_forecast, wrfinput_door, wrfinput_forecast

    monkeypatch.setattr(go_cli, 'render_extra_missing', lambda: None)
    monkeypatch.setattr(
        wrfinput_door, 'resolve_wrfinput_run',
        lambda directory, **kwargs: SimpleNamespace(
            substitution_report=SimpleNamespace(substitutions=())))
    inputs = SimpleNamespace(experiment=SimpleNamespace(
        start_time=datetime.datetime(2026, 5, 17, 18, tzinfo=datetime.timezone.utc)))
    def prepare(run, directory, **kwargs):
        inputs.prepared_root = Path(directory)
        return inputs

    monkeypatch.setattr(wrfinput_forecast, 'prepare_wrf_run', prepare)
    monkeypatch.setattr(wrfinput_forecast, 'WrfInitialization', lambda inputs: object())

    def run_prepared_tree(inputs, *, output_directory, **kwargs):
        recorded['outdir'] = canonical_path(output_directory)
        checkpoint = kwargs.get('restart')
        recorded['restart'] = None if checkpoint is None else canonical_path(checkpoint)
        # The runner's own exclusive mkdir, which a resumed run has to
        # find free (prepared_domain_tree_forecast.py:1957).
        (Path(output_directory) / 'evidence').mkdir()
        return {'status': 'ok'}

    monkeypatch.setattr(prepared_domain_tree_forecast, 'run_prepared_tree',
                        run_prepared_tree)


def test_a_resume_adopts_its_own_directory(tmp_path, monkeypatch):
    from gpuwm.wrfinput_forecast import run_wrf_forecast

    recorded = {}
    _adoption_door(monkeypatch, recorded)
    outdir = tmp_path / 'out'
    checkpoint = _resumable(outdir)
    assert run_wrf_forecast(tmp_path / 'wrf', outdir, restart=checkpoint,
                            exclusive_gpu=False, render_products='none') == 0
    assert recorded['outdir'] == outdir.resolve() / 'segment-001'
    assert recorded['restart'] == checkpoint
    kept = outdir / 'evidence' / 'run-receipt.json'
    assert json.loads(kept.read_text(encoding='utf-8'))['status'] == 'INTERRUPTED'
    assert (outdir / 'segment-001' / 'evidence').is_dir()
    assert (outdir / 'wrfout' / 'wrfout_d01_2026-05-17_18_00_00').exists()


def test_a_precreated_empty_output_directory_keeps_unrelated_data(tmp_path, monkeypatch):
    from gpuwm.wrfinput_forecast import run_wrf_forecast

    recorded = {}
    _adoption_door(monkeypatch, recorded)
    outdir = tmp_path / 'out'
    outdir.mkdir()
    sibling = tmp_path / 'observations.bin'
    sibling.write_bytes(b'preserve this observation')
    assert run_wrf_forecast(tmp_path / 'wrf', outdir, exclusive_gpu=False,
                            render_products='none') == 0
    assert recorded['outdir'] == outdir.resolve()
    assert sibling.read_bytes() == b'preserve this observation'
    assert (outdir / 'evidence').is_dir()


def test_a_metem_resume_adopts_its_own_directory(tmp_path, monkeypatch):
    from gpuwm import go_cli, metem_door, metem_forecast, prepared_domain_tree_forecast
    from test_metem_forecast import _launcher_stub_run

    recorded = {}
    run = _launcher_stub_run()
    run.substitution_report = SimpleNamespace(substitutions=())
    monkeypatch.setattr(go_cli, 'render_extra_missing', lambda: None)
    monkeypatch.setattr(
        metem_door, 'resolve_metem_run',
        lambda directory, **kwargs: run)
    import datetime

    inputs = SimpleNamespace(experiment=SimpleNamespace(
        start_time=datetime.datetime(2026, 5, 17, 18, tzinfo=datetime.timezone.utc)))
    def prepare(run, directory, **kwargs):
        inputs.prepared_root = Path(directory)
        return inputs

    monkeypatch.setattr(metem_forecast, 'prepare_metem_run', prepare)
    monkeypatch.setattr(metem_forecast, 'MetemInitialization', lambda inputs: object())

    def run_prepared_tree(inputs, *, output_directory, **kwargs):
        recorded['outdir'] = canonical_path(output_directory)
        (Path(output_directory) / 'evidence').mkdir()
        return {'status': 'ok'}

    monkeypatch.setattr(prepared_domain_tree_forecast, 'run_prepared_tree',
                        run_prepared_tree)
    outdir = tmp_path / 'out'
    checkpoint = _resumable(outdir)
    assert metem_forecast.run_metem_forecast(
        tmp_path / 'met', outdir, restart=checkpoint, exclusive_gpu=False,
        render_products='none') == 0
    assert recorded['outdir'] == outdir.resolve() / 'segment-001'
    assert (outdir / 'evidence' / 'run-receipt.json').exists()
    assert (outdir / 'segment-001' / 'evidence').is_dir()


def test_a_second_resume_keeps_the_second_attempts_receipt_too(tmp_path, monkeypatch):
    from gpuwm.stage_reuse import claim_run_output

    outdir = tmp_path / 'out'
    checkpoint = _resumable(outdir)
    for expected in ('segment-001', 'segment-002'):
        with claim_run_output(outdir, flag='--outdir', protected_roots=(),
                              resume=checkpoint) as claim:
            generation = claim.path
            assert generation == outdir.resolve() / expected
            assert (outdir / 'evidence' / 'run-receipt.json').exists()
            (generation / 'evidence').mkdir()
            (generation / 'evidence' / 'run-receipt.json').write_text('{}', encoding='utf-8')


def test_receipt_number_width_does_not_limit_resume_count(tmp_path):
    from gpuwm.stage_reuse import _next_segment

    for ordinal in range(1, 1000):
        (tmp_path / f'segment-{ordinal:03d}').mkdir()
    assert _next_segment(tmp_path) == tmp_path / 'segment-1000'


def test_the_prepared_input_tree_is_reused_when_its_sources_still_match(tmp_path):
    from gpuwm.stage_reuse import prepared_inputs_reusable

    directory = tmp_path / 'input'
    recorded = {'namelist_input': 'aa', 'wrfbdy': 'bb'}
    assert prepared_inputs_reusable(directory, receipt='wrf-import.json',
                                    source_files=recorded) is False
    directory.mkdir()
    (directory / 'wrf-import.json').write_text(
        json.dumps({'schema': 'gpuwm-wrf-input-import-v1',
                    'source_files': recorded}), encoding='utf-8')
    assert prepared_inputs_reusable(directory, receipt='wrf-import.json',
                                    source_files=recorded) is True
    # A flat manifest is the met_em spelling of the same document.
    (directory / 'source.json').write_text(json.dumps(recorded), encoding='utf-8')
    assert prepared_inputs_reusable(directory, receipt='source.json',
                                    source_files=recorded) is True
    with pytest.raises(ValueError) as refused:
        prepared_inputs_reusable(directory, receipt='wrf-import.json',
                                 source_files={**recorded, 'wrfbdy': 'cc'})
    assert 'wrfbdy' in str(refused.value) and '--outdir' in str(refused.value)
    with pytest.raises(ValueError) as unnamed:
        prepared_inputs_reusable(directory, receipt='metgrid-import.json',
                                 source_files=recorded)
    assert 'metgrid-import.json' in str(unnamed.value)


def test_a_run_with_no_checkpoint_is_still_refused_by_the_shared_claim(tmp_path):
    """The narrowing holds: only a resume adopts; a new run is refused."""

    from gpuwm.prepared_single_domain_forecast import claim_output_directory
    from gpuwm.stage_reuse import claim_run_output

    outdir = tmp_path / 'out'
    _resumable(outdir)
    with pytest.raises(FileExistsError):
        claim_run_output(outdir, flag='--outdir', protected_roots=())
    with pytest.raises(FileExistsError):
        claim_output_directory(outdir)
    # A checkpoint belonging to ANOTHER run does not adopt this one.
    elsewhere = tmp_path / 'other' / 'restart_d01.gpuwmrst'
    elsewhere.parent.mkdir()
    elsewhere.write_bytes(b'RST')
    with pytest.raises(FileExistsError):
        claim_run_output(outdir, flag='--outdir', protected_roots=(),
                         resume=elsewhere)


def test_an_outdir_inside_the_input_tree_is_refused_on_a_resume_too(tmp_path):
    """One configuration, one answer, whatever is left on disk.

    The refusal used to be reached only on the way to moving a previous
    `evidence/` aside, so a resume whose receipt had been cleaned away
    was accepted and the forecast wrote into its own input tree.
    """

    import shutil

    from gpuwm.stage_reuse import claim_run_output

    inputs = tmp_path / 'wrf'
    outdir = inputs / 'out'
    checkpoint = _resumable(outdir)
    with pytest.raises(ValueError, match='protected input tree'):
        claim_run_output(outdir, flag='--outdir', protected_roots=(inputs,),
                         resume=checkpoint)
    # The previous attempt's receipt cleaned away: same answer.
    shutil.rmtree(outdir / 'evidence')
    with pytest.raises(ValueError, match='protected input tree'):
        claim_run_output(outdir, flag='--outdir', protected_roots=(inputs,),
                         resume=checkpoint)
    assert not (outdir / 'segment-001').exists()
    # Nothing under the claimed directory at all: same answer.
    bare = inputs / 'bare'
    bare.mkdir()
    with pytest.raises(ValueError, match='protected input tree'):
        claim_run_output(bare, flag='--outdir', protected_roots=(inputs,),
                         resume=bare / 'restart_d01.gpuwmrst')


def test_a_resume_into_a_directory_that_is_gone_creates_it(tmp_path):
    """The adopted directory is created, not merely resolved.

    The runner's own `evidence.mkdir()` has no parents=True, so handing
    it back a name that is not on disk turns the resume into a raw
    errno at the runner instead of a claim at plan review.
    """

    from gpuwm.stage_reuse import claim_run_output

    outdir = tmp_path / 'out'
    checkpoint = outdir / 'restart_d01_2026-05-17_19_00_00.gpuwmrst'
    with claim_run_output(outdir, flag='--outdir', protected_roots=(),
                          resume=checkpoint) as claim:
        assert claim.path == outdir.resolve()
        assert claim.path.is_dir()


def test_the_met_em_inputs_are_digested_once_for_both_readers(tmp_path, monkeypatch):
    """The reuse question and the source manifest share ONE pass.

    met_em files are the gigabytes on this door, and the reuse question
    arrived with a second full pass over them beside the one the source
    manifest already took.  One pass is also the stronger statement:
    the digest the manifest records is the very digest the reuse
    question was answered from, not a second reading of the same file.
    """

    import tomllib
    from types import SimpleNamespace

    from test_namelist_import import _pair

    from gpuwm import metem_door, metem_forecast, stage_reuse
    from gpuwm.experiment import build_experiment
    from gpuwm.ingest import preprocess_backend
    from gpuwm.namelist_import import import_namelists, parse_namelist_text
    from gpuwm.static import projection

    text, _ = import_namelists(*_pair(tmp_path), metgrid_initialization=True)
    exp = build_experiment(tomllib.loads(text), source='digest count fixture')
    met = tmp_path / 'met_em.d01.2026-05-17_18_00_00.nc'
    met.write_bytes(b'MET')
    run = SimpleNamespace(
        toml_text=text, experiment=exp,
        namelist_input=tmp_path / 'namelist.input', paths={1: (met,)},
        interval_seconds=3600.0,
        controls=parse_namelist_text(
            (tmp_path / 'namelist.input').read_text(encoding='utf-8')))
    # Everything this preparation does AFTER the digests, replaced: the
    # subject here is how many times each input is read, and the domain
    # loop below needs real met_em payloads to run at all.
    monkeypatch.setattr(metem_forecast, 'resolve_metem_vertical',
                        lambda run, text, **kwargs: (text, 'explicit', None))
    monkeypatch.setattr(metem_door, 'metgrid_memory_admission',
                        lambda run, resolved: {})
    monkeypatch.setattr(preprocess_backend, 'resolve_preprocess_backend',
                        lambda backend, **kwargs: SimpleNamespace(
                            receipt=lambda: {'backend': 'test'}))

    class _Stop(Exception):
        """Raised where the digests are finished and the reading starts."""

    monkeypatch.setattr(projection, 'grids_from_projection_config',
                        lambda exp: (_ for _ in ()).throw(_Stop()))
    digested = []
    real_sha = metem_forecast._sha
    monkeypatch.setattr(metem_forecast, '_sha',
                        lambda path: digested.append(Path(path)) or real_sha(path))
    asked = {}
    real_reusable = stage_reuse.prepared_inputs_reusable

    def reusable(directory, **kwargs):
        asked.update(kwargs['source_files'])
        return real_reusable(directory, **kwargs)

    monkeypatch.setattr(stage_reuse, 'prepared_inputs_reusable', reusable)
    outdir = tmp_path / 'prepared'
    with pytest.raises(_Stop):
        metem_forecast.prepare_metem_run(run, outdir)
    assert digested.count(met) == 1
    recorded = json.loads((outdir / 'source.json').read_text(encoding='utf-8'))
    assert asked['met_em_d01_0'] == recorded['met_em_d01_0']
