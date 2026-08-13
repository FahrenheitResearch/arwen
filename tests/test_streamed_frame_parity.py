"""The parity instrument, tested against known answers in BOTH directions.

An equality checker that always answers "equal" passes every parity gate
ever written, and this project has produced confident wrong results that
way more than once.  So these tests do not merely check that two identical
files compare equal: they seed FOUR corruptions, one for each class of
defect the 2.2.0 cut actually found or could have found, and require the
instrument to catch each one.

  * a dropped variable      -- the OLR defect, exactly
  * an added variable       -- its mirror, which is equally not parity
  * one flipped float bit   -- the UP_HELI_MAX class, 1 cell, ~1 ULP
  * one changed attribute   -- the carrier-provenance defect

The files here are written with netCDF4 directly rather than by running a
forecast: what is under test is the COMPARISON, and a fixture that needed
a card could not run in the CPU battery where regressions get caught.
"""
from __future__ import annotations

import numpy as np
import pytest

from tools import streamed_frame_parity as parity


netCDF4 = pytest.importorskip("netCDF4")


def _write(path, *, fields=None, attrs=None):
    """One small wrfout-shaped file."""
    fields = {"T": np.arange(24, dtype="f4").reshape(2, 3, 4),
              "OLR": np.linspace(90.0, 320.0, 12, dtype="f4").reshape(3, 4),
              "UP_HELI_MAX": np.full((3, 4), 0.0015428888, dtype="f4"),
              } if fields is None else fields
    attrs = {"TITLE": "gpuwm test",
             "GPUWM_CARRIER_GLW_SOURCE": "radiation_scheme",
             "GPUWM_CARRIER_GLW_LAST_UPDATE": "2880.0",
             } if attrs is None else attrs
    with netCDF4.Dataset(str(path), "w") as ds:
        ds.createDimension("z", 2)
        ds.createDimension("y", 3)
        ds.createDimension("x", 4)
        dims = {1: ("x",), 2: ("y", "x"), 3: ("z", "y", "x")}
        for name, value in fields.items():
            var = ds.createVariable(name, value.dtype, dims[value.ndim])
            var[:] = value
        for key, value in attrs.items():
            setattr(ds, key, value)
    return path


def _default_fields():
    return {"T": np.arange(24, dtype="f4").reshape(2, 3, 4),
            "OLR": np.linspace(90.0, 320.0, 12, dtype="f4").reshape(3, 4),
            "UP_HELI_MAX": np.full((3, 4), 0.0015428888, dtype="f4")}


def test_two_identical_frames_are_parity(tmp_path):
    """The positive direction, and the one that must not be the only one."""
    a = _write(tmp_path / "a.nc")
    b = _write(tmp_path / "b.nc")

    result = parity.compare(a, b)

    assert result["parity"] is True
    assert result["field_set_equal"] is True
    assert result["identical_fields"] == 3
    assert result["differing_fields"] == []
    assert result["attrs_differing"] == []
    assert parity.main([str(a), str(b)]) == 0


def test_a_dropped_field_is_caught_and_named(tmp_path):
    """THE OLR DEFECT.  73 of 74 variables, every value that survived equal.

    This is the case every gate the release had was blind to, so it is
    asserted hardest: the instrument must fail, must say WHICH field, and
    must not be talked out of it by the other 2 fields being identical.
    """
    fields = _default_fields()
    a = _write(tmp_path / "a.nc")
    fields.pop("OLR")
    b = _write(tmp_path / "b.nc", fields=fields)

    result = parity.compare(a, b)

    assert result["parity"] is False
    assert result["field_set_equal"] is False
    assert result["missing_from_streamed"] == ["OLR"]
    assert result["extra_in_streamed"] == []
    # The surviving fields really are identical -- which is exactly why a
    # value-only comparison passed this file.
    assert result["differing_fields"] == []
    assert result["identical_fields"] == 2
    assert "MISSING" in parity.render(result)
    assert parity.main([str(a), str(b)]) == 1


def test_an_invented_field_is_caught_too(tmp_path):
    """The mirror direction: extra rows are not parity either."""
    fields = _default_fields()
    a = _write(tmp_path / "a.nc")
    fields["QVAPOR"] = np.zeros((3, 4), dtype="f4")
    b = _write(tmp_path / "b.nc", fields=fields)

    result = parity.compare(a, b)

    assert result["parity"] is False
    assert result["extra_in_streamed"] == ["QVAPOR"]
    assert result["missing_from_streamed"] == []


def test_one_flipped_bit_in_one_cell_is_caught(tmp_path):
    """THE UP_HELI_MAX CLASS: 1 cell of 1.16e-10, everything else equal.

    Seeded as a genuine 1-ULP step rather than an arbitrary epsilon, so a
    tolerance creeping into the comparison would fail this test rather
    than hide behind it.
    """
    fields = _default_fields()
    a = _write(tmp_path / "a.nc")
    nudged = fields["UP_HELI_MAX"].copy()
    nudged[1, 2] = np.nextafter(nudged[1, 2], np.float32(0.0))
    fields["UP_HELI_MAX"] = nudged
    b = _write(tmp_path / "b.nc", fields=fields)

    result = parity.compare(a, b)

    assert result["parity"] is False
    assert result["field_set_equal"] is True
    assert [row["name"] for row in result["differing_fields"]] \
        == ["UP_HELI_MAX"]
    row = result["differing_fields"][0]
    assert row["ndiff"] == 1
    assert 0.0 < row["maxabs"] < 1e-9
    assert result["identical_fields"] == 2


def test_a_changed_provenance_attribute_is_caught(tmp_path):
    """THE CARRIER DEFECT: every array equal, the metadata denying it."""
    a = _write(tmp_path / "a.nc")
    b = _write(tmp_path / "b.nc",
               attrs={"TITLE": "gpuwm test",
                      "GPUWM_CARRIER_GLW_SOURCE": "unwritten",
                      "GPUWM_CARRIER_GLW_LAST_UPDATE": "-1.0"})

    result = parity.compare(a, b)

    assert result["parity"] is False
    assert result["field_set_equal"] is True
    assert result["differing_fields"] == []
    names = [row["name"] for row in result["attrs_differing"]]
    assert names == ["GPUWM_CARRIER_GLW_LAST_UPDATE",
                     "GPUWM_CARRIER_GLW_SOURCE"]


def test_an_allowed_attribute_is_the_only_way_to_excuse_one(tmp_path):
    """Never defaulted: excusing an attribute is an explicit claim.

    A default allowlist grows until the check means nothing, so the only
    way past an attribute difference is to name it at the call site.
    """
    a = _write(tmp_path / "a.nc")
    b = _write(tmp_path / "b.nc",
               attrs={"TITLE": "gpuwm test (other run)",
                      "GPUWM_CARRIER_GLW_SOURCE": "radiation_scheme",
                      "GPUWM_CARRIER_GLW_LAST_UPDATE": "2880.0"})

    assert parity.compare(a, b)["parity"] is False
    assert parity.compare(a, b, allow_attrs=["TITLE"])["parity"] is True
    # And the receipt records what was excused, so a reader of the JSON can
    # see the claim rather than infer it from a pass.
    assert parity.compare(a, b, allow_attrs=["TITLE"])["allowed_attrs"] \
        == ["TITLE"]


# ---------------------------------------------------------------------------
# the refusal that makes a short frame impossible to publish by accident
# ---------------------------------------------------------------------------

def _plan_with_an_unavailable_row():
    """A frame plan that can serve everything except one driver diagnostic."""
    from tilestream import output as tsout

    return tsout.FramePlan(
        order=("T", "OLR"),
        source={"T": tsout.SOURCE_CARRIER,
                "OLR": tsout.SOURCE_DRIVER_DIAGNOSTIC},
        origin={"T": "state/thp", "OLR": "driver.olr"},
        shape={"T": (2, 3, 4), "OLR": (3, 4)},
        dtype={"T": np.dtype("f4"), "OLR": np.dtype("f4")})


class _Setup:
    """The DomainSetup surface StoreFrame reads, and nothing more."""

    thb = np.zeros((2, 1, 1), dtype="f4")
    pb = np.zeros((2, 1, 1), dtype="f4")
    phb = np.zeros((3, 1, 1), dtype="f4")

    def output_statics(self, nz, ny, nx):
        return {}


class _Cfg:
    nz, ny, nx = 2, 3, 4
    dx = dy = 3000.0


def test_a_plan_that_cannot_publish_a_resident_row_is_refused_by_default():
    """THE RAISE THAT WAS DOCUMENTED AND ABSENT.

    ``StoreFrame``'s contract said it raised rather than publishing a frame
    missing rows a resident run writes.  It did not: ``fields()`` skipped
    them with a comment, the writer took the short dict as its schema, and
    every streamed run up to 2.2.0 published a wrfout without ``OLR`` while
    reporting validity PASS.  Default-on, because a refusal you have to opt
    into is exactly the workaround this project has ruled against.
    """
    from tilestream import output as tsout

    store = {"state/thp": np.zeros((2, 3, 4), dtype="f4")}
    with pytest.raises(tsout.ShortFrameRefused) as excinfo:
        tsout.StoreFrame(_plan_with_an_unavailable_row(), store,
                         _Setup(), _Cfg())

    message = str(excinfo.value)
    assert "OLR" in message                       # names the row
    assert "driver.olr" in message                # names where it comes from
    assert "diagnostic_inventory" in message      # names the remedy


def test_the_opt_out_exists_for_the_benches_that_price_the_trade():
    """And it has to be asked for explicitly, at the call site.

    Two harnesses publish a carrier-only frame on purpose
    (``tilestream.endure``, ``tilestream.realcase_wrfout``); both now say so
    in the constructor call rather than inheriting the old silence.
    """
    from tilestream import output as tsout

    store = {"state/thp": np.zeros((2, 3, 4), dtype="f4")}
    frame = tsout.StoreFrame(_plan_with_an_unavailable_row(), store,
                             _Setup(), _Cfg(), require_complete=False)

    fields = frame.fields()
    assert "T" in fields
    assert "OLR" not in fields          # the short frame, knowingly published


def test_every_non_test_frame_builder_states_which_it_wants():
    """A new caller must not inherit a short frame by saying nothing.

    Source-scanned rather than exercised because the alternative needs a
    card and a store per call site.  The product route is the one that must
    NOT opt out, so it is asserted by absence.
    """
    import inspect

    from gpuwm.core import streaming
    from tilestream import endure, realcase_wrfout

    for module in (endure, realcase_wrfout):
        src = inspect.getsource(module)
        assert "require_complete=False" in src, module.__name__

    product = inspect.getsource(streaming.StreamedDomain.history_fields)
    assert "StoreFrame(" in product
    assert "require_complete" not in product
