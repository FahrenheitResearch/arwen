import pytest

from gpuwm.ingest.hrrr import aerosol_fill_receipt


def test_completed_fill_counts_are_carried_by_the_prepare_reader():
    gate = {"aerosol_missing_policy": "nearest_neighbor+four_pt+average_4pt;fill_missing=0;native_grid_coordinates"}
    for name in ("QNWFA", "QNIFA"):
        for stage, count in {"masked": 5, "nearest_neighbor": 0,
                             "four_pt": 0, "average_4pt": 3, "zero": 2}.items():
            gate[f"aerosol_{name}_{stage}"] = str(count)
    receipt = aerosol_fill_receipt(gate)
    assert receipt["fields"]["QNWFA"]["zero"] == 2
    assert receipt["fields"]["QNIFA"]["average_4pt"] == 3
    gate["aerosol_QNWFA_masked"] = "6"
    with pytest.raises(ValueError, match="do not sum"):
        aerosol_fill_receipt(gate)
    gate["aerosol_QNWFA_masked"] = "-1"
    with pytest.raises(ValueError, match="negative"):
        aerosol_fill_receipt(gate)
    assert aerosol_fill_receipt({}) is None


def test_python_boundary_uses_original_mask_and_finite_neighbor_donors():
    import numpy as np
    from gpuwm.ingest.cpu_backend import CpuPreprocessBackend

    native = CpuPreprocessBackend()
    source = np.array([[[0., 2.], [4., 8.]]], dtype=np.float32)
    valid = np.array([[[0, 1], [1, 1]]], dtype=np.uint8)
    for workers in (1, 3):
        values, counts = native.missing_value_chain(source, valid,
            np.array([0.25, 0.25, 0.]), np.array([0.25, 0.75, 0.]), workers=workers)
        np.testing.assert_array_equal(values, np.array([[14. / 3., 2., 0.]], dtype=np.float32))
        np.testing.assert_array_equal(counts, [[1, 0, 1, 1]])


def test_python_boundary_average_preserves_wps_negative_zero():
    import numpy as np
    from gpuwm.ingest.cpu_backend import CpuPreprocessBackend

    native = CpuPreprocessBackend()
    source = np.array([[[9., -0.], [-0., -0.]]], dtype=np.float32)
    valid = np.array([[[0, 1], [1, 1]]], dtype=np.uint8)
    # Edge averages duplicate corners; the interior has three valid donors.
    yy = np.array([0., 0.25, 0.25, 0.25, 0.])
    xx = np.array([0.25, 0., 0.25, 0.75, 0.])
    for workers in (1, 3):
        values, counts = native.missing_value_chain(source, valid, yy, xx,
                                                     workers=workers)
        np.testing.assert_array_equal(values.view(np.uint32),
                                      [[0x80000000] * 4 + [0]])
        np.testing.assert_array_equal(counts, [[1, 0, 3, 1]])


def test_a_source_bitmap_is_required_and_carried_through_the_live_loader(tmp_path):
    import numpy as np
    from test_hrrr_masked_aerosol_pair import _publication
    from gpuwm.ingest.hrrr import load_hrrr_pipeline_ready_window

    root = tmp_path / "bridge"
    _publication(root, {"atmosphere_selected_per_time": "661",
        "optional_hybrid_fields": "QNWFA,QNIFA",
        "optional_hybrid_units": "QNWFA=kg-1,QNIFA=kg-1",
        "aerosol_missing_policy": "source_mask_preserved_for_target_mapping"})
    for name in ("QNWFA", "QNIFA"):
        np.full((50, 2, 3), 0.5, dtype="<f4").tofile(root / "atmosphere-f00" / f"{name}.f32le")
    with pytest.raises(ValueError, match="QNWFA source bitmap is missing"):
        load_hrrr_pipeline_ready_window(root, 0)
    for name in ("QNWFA", "QNIFA"):
        mask = np.ones((50, 2, 3), dtype=np.uint8)
        mask[0, 0, 0] = 0
        mask.tofile(root / "atmosphere-f00" / f"{name}.mask")
    loaded = load_hrrr_pipeline_ready_window(root, 0)
    assert loaded.fields["QNWFA_SOURCE_MASK"][0, 0, 0] == 0
    assert loaded.fields["QNIFA_SOURCE_MASK"][0, 0, 1] == 1
    (root / "atmosphere-f00" / "QNWFA.mask").write_bytes(b"\x01")
    with pytest.raises(ValueError, match="bitmap has the wrong shape"):
        load_hrrr_pipeline_ready_window(root, 0)
