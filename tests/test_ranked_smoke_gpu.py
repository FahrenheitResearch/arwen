"""The release gate uses real per-rank kernels, source decode and disk I/O."""
import pytest


def test_two_thread_ranks_smoke_seam_radiation_history_restart(tmp_path):
    cp = pytest.importorskip('cupy')
    if cp.cuda.runtime.getDeviceCount() < 1:
        pytest.skip('requires one CUDA card; two ranks share it')
    from tilestream.ranks_smoke_gate import run
    result = run(nsteps=20, devices=[0], directory=tmp_path)
    assert result['completed_steps'] == 20
    assert result['resident_fires']['radiation'] >= 2
    assert result['state_differing'] == result['history_differing'] == []
    assert result['restart_roundtrip'] and result['smoke_processes']['history_disk_roundtrip']
    assert all(v > 0 for v in result['smoke_processes']['seam_emission_ug_m2'])
    assert all(result['source_receipts'])
    prepared = result['prepared_store']
    assert prepared['status'] == 'LOADED' and prepared['slabs'] > 1
    assert prepared['initial_carriers_word_identical']
    assert not prepared['missing']
    assert 'chemdiag_fire_ebu' in prepared['chem_carriers']
    assert 'chem_smoke' in prepared['chem_carriers']
