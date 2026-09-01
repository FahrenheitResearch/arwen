"""The regrid is the only new arithmetic in the external-baseline scorer.

Everything else it does -- the FSS, the constants, the truth construction --
is imported from code that is already tested.  What is new is the map from
somebody else's projection onto the verification grid, and the whole point
of scoring an external model is that the map must not quietly do the
scoring for us.  So these tests pin the two properties the comparison
rests on: nearest-donor sampling really does carry donor values across
unchanged, and every method really does collapse to the identity when the
two grids coincide.
"""

import numpy as np
import pytest

from tools.da_external_baseline_score import METHODS, regrid


@pytest.fixture
def field():
    rng = np.random.default_rng(20260805)
    return rng.uniform(-10.0, 65.0, size=(24, 31))


@pytest.mark.parametrize("method", METHODS)
def test_coincident_grids_are_the_identity(field, method):
    """No mixing when every target point sits on a source point.

    A regrid that shifted or smoothed a field it was handed at exactly its
    own grid points would be corrupting the comparison everywhere, and the
    corruption would be invisible in the score.
    """

    y, x = np.meshgrid(np.arange(3.0, 20.0), np.arange(2.0, 27.0),
                       indexing="ij")
    out = regrid(field, x, y, method=method)
    expected = field[y.astype(int), x.astype(int)]
    assert np.allclose(out, expected, atol=1e-5)


def test_nearest_never_invents_a_value(field):
    """Every output value is some input value, exactly.

    This is the property that lets the primary method be called
    amplitude-preserving: nearest cannot create a reflectivity that was not
    in the source field, so it cannot move area across a threshold.
    """

    rng = np.random.default_rng(7)
    x = rng.uniform(1.0, field.shape[1] - 3.0, size=(40, 40))
    y = rng.uniform(1.0, field.shape[0] - 3.0, size=(40, 40))
    out = regrid(field, x, y, method="nearest")
    assert np.isin(out, field).all()
    assert out.max() <= field.max()
    assert out.min() >= field.min()


def test_nearest_picks_the_nearest_donor(field):
    """Half-cell offsets round to the neighbouring donor, not the floor."""

    x = np.array([[5.4, 5.6]])
    y = np.array([[7.4, 7.6]])
    out = regrid(field, x, y, method="nearest")
    assert out[0, 0] == field[7, 5]
    assert out[0, 1] == field[8, 6]


def test_bilinear_midpoint_is_the_four_point_mean(field):
    """The mixing method mixes, and mixes the way it says it does."""

    x = np.array([[9.5]])
    y = np.array([[11.5]])
    out = regrid(field, x, y, method="bilinear")
    expected = field[11:13, 9:11].mean()
    assert out[0, 0] == pytest.approx(expected)


def test_unknown_method_is_refused(field):
    with pytest.raises(ValueError, match="unknown method"):
        regrid(field, np.array([[1.0]]), np.array([[1.0]]), method="cubic")


def test_frame_valid_key_reads_the_stamp_out_of_the_name():
    """Frame pairing is by valid time in the filename, not by sort order."""

    from pathlib import Path

    from tools.da_external_baseline_score import frame_valid_key

    name = Path("refc_t05z_wrfsubh_valid20260805T061500Z_lead01h15m.npy")
    assert frame_valid_key(name) == "2026-08-05T06:15:00Z"


def test_frame_valid_key_refuses_an_unstamped_name():
    from pathlib import Path

    from tools.da_external_baseline_score import frame_valid_key

    with pytest.raises(SystemExit, match="no valid"):
        frame_valid_key(Path("refc_frame_003.npy"))
