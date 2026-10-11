"""Runtime measurements cannot change a prepared head's scientific binding."""
import json
from gpuwm.ingest.preparation_evidence import scientific_preprocessing, write_runtime_preparation
from gpuwm.ingest.boundary_stream import head_sha256


def test_worker_headroom_varies_without_changing_scientific_preprocessing_or_head(tmp_path):
    before = {"backend": "cpu", "bridge": {"sha256": "a" * 64},
              "vertical_interpolation": [{"levels": 50}],
              "parallelism": {"memory_worker_limit": 7000}, "workers": 8}
    after = {**before, "parallelism": {"memory_worker_limit": 6800}, "workers": 4}
    assert scientific_preprocessing(before) == scientific_preprocessing(after)
    assert head_sha256({"basis": {"proof_head": {"preprocessing": scientific_preprocessing(before)}}}) == \
        head_sha256({"basis": {"proof_head": {"preprocessing": scientific_preprocessing(after)}}})
    snapshot = scientific_preprocessing(before)
    before["vertical_interpolation"].append({"levels": 60})
    assert snapshot["vertical_interpolation"] == [{"levels": 50}]
    path = write_runtime_preparation(tmp_path, preprocessing=after,
        parallelism={"memory_worker_limit": 6800}, forcing_stage_timings=[{"total_seconds": 1.2}])
    actual = json.loads(path.read_text())
    assert actual["preprocessing_measurements"]["parallelism"]["memory_worker_limit"] == 6800
    assert actual["forcing_stage_timings"] == [{"total_seconds": 1.2}]
    assert not path.with_suffix(".json.pending").exists()


def test_decoder_and_vertical_identity_are_never_treated_as_runtime_measurements():
    base = {"backend": "cpu", "bridge": {"sha256": "a" * 64}, "vertical_interpolation": [{"levels": 50}]}
    for changed in ({**base, "backend": "cuda"}, {**base, "bridge": {"sha256": "b" * 64}},
                    {**base, "vertical_interpolation": [{"levels": 60}]}):
        assert scientific_preprocessing(base) != scientific_preprocessing(changed)
