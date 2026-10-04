"""Device proof of bounded physical recentering, not a forecast skill test."""
import os
import numpy as np
import pytest

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(bool(os.environ.get("GPUWM_NO_LOCAL_GPU")), reason="local GPU disabled")]
cp = pytest.importorskip("cupy")

from gpuwm.ensemble.recentered import FieldBounds, recenter_field


def call(base, donors, selected, bounds):
    return recenter_field(base, donors, donor_ids=("p01", "p02", "p03"), selected_ids=selected,
                          bounds=bounds, mapped_grid_sha256="a" * 64)[0]


def test_single_batch_reorder_partition_and_bounds():
    base = cp.asarray([[280, 290], [299, 251]], dtype=cp.float32)
    donors = cp.asarray([[[260, 290], [290, 251]], [[280, 290], [300, 252]], [[300, 320], [310, 253]]], dtype=cp.float32)
    bound = FieldBounds("K", 4.0, 250.0, 300.0)
    ids = ("p01", "p02", "p03")
    before = (base.get().tobytes(), donors.get().tobytes())
    all_members = call(base, donors, ids, bound).get()
    for i, name in enumerate(ids):
        assert call(base, donors, (name,), bound).get().tobytes() == all_members[i:i+1].tobytes()
    assert call(base, donors, ids[::-1], bound).get().tobytes() == all_members[::-1].tobytes()
    np.testing.assert_array_equal(all_members.mean(axis=0), base.get())
    assert np.max(np.abs(all_members - base.get())) <= 4
    assert all_members.min() >= 250 and all_members.max() <= 300
    assert before == (base.get().tobytes(), donors.get().tobytes())
    # Independent exact arithmetic fixture: first cell anomalies -20/0/+20,
    # second cell -10/-10/+20; common bound gives -2/-2/+4 there.
    np.testing.assert_array_equal(all_members[:, 0], [[276, 288], [280, 288], [284, 294]])


def test_zero_amplitude_preserves_signed_zero_words():
    base = cp.asarray([[0., -0.]], dtype=cp.float32)
    donors = cp.asarray([[[-1., 1.]], [[2., 3.]], [[4., 5.]]], dtype=cp.float32)
    out = call(base, donors, ("p01", "p02"), FieldBounds("m s-1", 10, -100, 100, amplitude=0)).get()
    assert out[0].tobytes() == base.get().tobytes() == out[1].tobytes()


def test_decimal_bounds_hold_after_final_float32_rounding():
    base = cp.asarray([280.0], dtype=cp.float32)
    donors = cp.asarray([[279.0], [280.0], [281.0]], dtype=cp.float32)
    out = call(base, donors, ("p01", "p02", "p03"), FieldBounds("K", 0.01, 279.995, 280.009)).get()
    assert out.astype(np.float64).min() >= 279.995
    assert out.astype(np.float64).max() <= 280.009
    assert np.abs(out.astype(np.float64) - 280.0).max() <= 0.01


def test_invalid_source_fails_before_transform():
    base = cp.asarray([280.], dtype=cp.float32)
    donors = cp.asarray([[280.], [np.nan], [290.]], dtype=cp.float32)
    with pytest.raises(ValueError, match="non-finite"):
        call(base, donors, ("p01",), FieldBounds("K", 10, 180, 340))


@pytest.mark.parametrize("count", [1, 4, 20, 30])
def test_full_population_member_byte_identity_at_awkward_tails(count):
    shape = (7, 11, 17)
    cells = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    base_host = 270 + (cells % 51) * np.float32(0.25)
    donors_host = np.stack([base_host + (i-14.5) * np.float32(0.5) + (cells % (i+3)) * np.float32(0.125)
                            for i in range(30)]).astype(np.float32)
    ids = tuple(f"p{i+1:02d}" for i in range(30))
    selected = ids[:count][::-1]
    base, donors = cp.asarray(base_host), cp.asarray(donors_host)
    bound = FieldBounds("K", 5, 250, 310, amplitude=0.8)
    def transform(roster):
        return recenter_field(base, donors, donor_ids=ids, selected_ids=roster, bounds=bound,
                              mapped_grid_sha256="b"*64)[0].get()
    batched = transform(selected)
    for row, member in zip(batched, selected):
        assert row.tobytes() == transform((member,))[0].tobytes()
    # Independent FP64 reference over the complete fixed population.
    anomalies = donors_host.astype(np.float64) - donors_host.astype(np.float64).mean(axis=0)
    scale = np.minimum(0.8, 5 / np.max(np.abs(anomalies), axis=0))
    expected = (base_host + scale * anomalies[[ids.index(member) for member in selected]]).astype(np.float32)
    np.testing.assert_array_equal(batched, expected)
