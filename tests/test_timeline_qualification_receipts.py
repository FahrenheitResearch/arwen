"""The comparison view retains complete scientific evidence metadata."""

from tools.timeline.verify_qualification import science_evidence


def test_all_scientific_evidence_is_retained_with_original_nesting():
    final = {"schema": "gpuwm-canonical-state-v2", "sha256": "a" * 64,
             "inventory_sha256": "b" * 64, "scalar_sha256": "c" * 64,
             "members": [{"name": "state/qv", "sha256": "d" * 64}]}
    document = {"final_state_digest": final, "timing_seconds": {"total": 9},
                "prepared": {"content_sha256": "e" * 64, "path": "/volatile"},
                "details": [{"backend": "cuda", "wall_seconds": 2},
                            {"boundary_fingerprint": {"sha256": "f" * 64}}]}
    assert science_evidence(document) == {
        "final_state_digest": final,
        "prepared": {"content_sha256": "e" * 64},
        "details": [{"backend": "cuda"}, {"boundary_fingerprint": {"sha256": "f" * 64}}]}


def test_unknown_bookkeeping_cannot_invent_scientific_evidence():
    assert science_evidence({"wall_seconds": 2, "source": "/run", "pid": 3}) == {}
