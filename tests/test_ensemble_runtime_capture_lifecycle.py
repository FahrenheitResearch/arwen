"""The labelled 36-second fixture observes actual member progress, not 48 seconds."""
from test_ensemble_wrf_seed_labels_runtime_gpu import _Words


def test_labelled_final_capture_uses_its_complete_forecast_and_original_member_clock():
    words = _Words.__new__(_Words)
    words.pure_member = 0
    words.owners = {(0, 1): object(), (1, 1): object()}
    snapshots = []
    words.snapshot = lambda *args, **kwargs: snapshots.append((args, kwargs))
    words.progress(phase="post-d01-sync", model_elapsed_seconds=24.)
    assert snapshots == [] and set(words.owners) == {(0, 1), (1, 1)}
    # The ensemble's displayed average is 18 seconds when member zero finishes.
    words.progress(phase="post-d01-sync", member_id=0, model_elapsed_seconds=18.,
                   member_model_elapsed_seconds=36.)
    assert snapshots == [(('final',), {"member": 0, "grid_id": 1})]
    assert set(words.owners) == {(1, 1)}
    words.pure_member = 1
    words.progress(phase="post-d01-sync", model_elapsed_seconds=36.)
    assert snapshots[-1] == (('final',), {"member": 1, "grid_id": 1})
    assert words.owners == {}
