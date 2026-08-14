"""The European radar composite route: wrapper, descriptor, source class.

Three layers, each tested where it can be tested honestly:

* the front-door descriptor and the argument construction, against the
  wrapper alone -- no binary, no socket;
* the pack round trip, against bytes this file builds the way the Rust
  writer does;
* the built binary's own ``--abi`` and its refusals, when a checkout has it.

The decode itself is the Rust side's and is tested there, against the
transcribed geometry of a live frame. What this file must not do is restate
that decode in Python and then assert the restatement agrees with itself.
"""

from __future__ import annotations

import functools
import hashlib
import json
import struct
import subprocess

import numpy as np
import pytest

from gpuwm.obs import frontdoor, obspack, opera, sources


# --------------------------------------------------------------- fixtures

def build_pack(meta: dict, arrays: dict[str, np.ndarray]) -> bytes:
    """Assemble a GPWMOBS1 pack the way the Rust writer does."""

    payload = b""
    index = {}
    for key, array in arrays.items():
        dtype = "<u1" if array.dtype == np.uint8 else "<f8"
        data = array.astype(np.dtype(dtype)).tobytes(order="C")
        index[key] = {"dtype": dtype, "shape": list(array.shape),
                      "offset": len(payload), "bytes": len(data)}
        payload += data
    meta = dict(meta)
    meta["arrays"] = index
    meta["payload_bytes"] = len(payload)
    meta["content_sha256"] = hashlib.sha256(payload).hexdigest()
    blob = json.dumps(meta).encode()
    header = (obspack.PACK_MAGIC
              + struct.pack("<II", obspack.PACK_VERSION, len(blob))
              + struct.pack("<Q", len(payload)))
    header += b"\0" * (obspack.PACK_HEADER_BYTES - len(header))
    return header + blob + payload


def opera_grid_pack(valid_time="2026-08-12T19:30:00", *, values=None,
                    observed_fraction=None) -> bytes:
    values = np.array([[10.0, 20.0], [30.0, 40.0]]) if values is None else values
    meta = {
        "schema": obspack.GRID_SCHEMA,
        "status": "READY",
        "quantity": sources.QUANTITY_COMPOSITE_REFLECTIVITY,
        "units": "dBZ",
        "valid_time": valid_time,
        "provenance": {
            "source": opera.SOURCE_LABEL, "product": opera.PRODUCT_LABEL,
            "uri": "frame.h5", "sha256": "b" * 64,
            "fetched_at": "2026-08-12T19:45:39",
            "is_stub": False, "stub_reason": "",
        },
    }
    if observed_fraction is not None:
        meta["sentinels"] = {"observed_fraction": observed_fraction}
    return build_pack(meta, {
        "values": values,
        "valid": np.ones(values.shape, dtype=np.uint8),
    })


def geo_pack(shape=(2, 2)) -> bytes:
    latitude = np.full(shape, 50.0)
    longitude = np.full(shape, 9.0)
    meta = {"schema": obspack.GEO_SCHEMA, "status": "READY",
            "source_product": opera.PRODUCT_LABEL}
    return build_pack(meta, {"latitude": latitude, "longitude": longitude})


def _tree(tmp_path, times):
    packs = []
    for index, when in enumerate(times):
        path = tmp_path / f"frame{index}.obspack"
        path.write_bytes(opera_grid_pack(valid_time=when))
        packs.append(path)
    geo = tmp_path / "grid.obspack"
    geo.write_bytes(geo_pack())
    return packs, geo


class _Recorder:
    """Stands in for the binary: records the argv the wrapper would run.

    Patched onto :class:`~gpuwm.obs.frontdoor.FrontDoor` rather than onto the
    ``OPERA`` instance, which is a frozen dataclass and refuses attribute
    assignment -- the same immutability that stops a caller repointing a
    front door at another binary at run time.
    """

    def __init__(self, schema):
        self.schema = schema
        self.calls = []

    def __get__(self, instance, owner=None):
        # A plain object set on a class is not a descriptor, so attribute
        # lookup would hand it back unbound and the door would never reach
        # __call__. Binding here is what lets the recorder see which front
        # door was driven.
        if instance is None:
            return self
        return functools.partial(self.__call__, instance)

    def __call__(self, door, subcommand, arguments, *, schema):
        self.calls.append((door.name, subcommand, list(arguments), schema))
        return {"schema": schema, "status": "READY"}


# -------------------------------------------------------- the front door

def test_the_registry_carries_the_european_composite_door():
    assert frontdoor.FRONT_DOORS["opera"] is frontdoor.OPERA
    assert frontdoor.OPERA.name == "rw_opera"
    assert frontdoor.OPERA.env_var == "GPUWM_RW_OPERA"


def test_the_abi_marker_pins_the_distinctions_a_wrong_build_would_lose():
    """What the marker is for, stated as the three failures it catches.

    A build that stopped separating the two sentinels, or that placed cells
    with a spherical inversion, or that wrote some other pack schema, would
    all still print plausible JSON. The marker is what makes each of them a
    probe-time refusal instead.
    """

    marker = frontdoor.OPERA.abi_marker
    assert "no_coverage" in marker and "no_echo" in marker
    assert "corner_check" in marker
    assert obspack.GRID_SCHEMA in marker
    assert sources.QUANTITY_COMPOSITE_REFLECTIVITY in marker
    assert marker.startswith(opera.FETCH_SCHEMA)


def test_the_composite_serves_the_same_quantity_and_units_as_the_us_grader():
    """The point of the route: one consumer, two continents.

    If these ever diverged, a scorer would need to know which network it was
    reading before it could read it, and the whole reason to publish the
    European composite through the existing pack schema would be gone.
    """

    european = sources.OperaCompositeSource.__init__
    assert european is not None
    marker = frontdoor.OPERA.abi_marker
    for token in ("composite_reflectivity", "dBZ"):
        assert token in marker
        assert token in frontdoor.MRMS.abi_marker


# ------------------------------------------------------ the wrapper calls

def test_the_window_subcommands_pass_the_window_through(monkeypatch):
    recorder = _Recorder(opera.COVERAGE_SCHEMA)
    monkeypatch.setattr(frontdoor.FrontDoor, "run", recorder)
    opera.run_coverage(start="2026-08-12T19:00:00Z", end="2026-08-12T19:30:00Z",
                       limit=3)
    name, subcommand, arguments, schema = recorder.calls[0]
    assert name == "rw_opera"
    assert subcommand == "coverage"
    assert arguments == ["--start", "2026-08-12T19:00:00Z",
                         "--end", "2026-08-12T19:30:00Z", "--limit", "3"]
    assert schema == opera.COVERAGE_SCHEMA


def test_decode_forwards_the_box_and_the_no_echo_value(monkeypatch, tmp_path):
    recorder = _Recorder(opera.DECODE_SCHEMA)
    monkeypatch.setattr(frontdoor.FrontDoor, "run", recorder)
    opera.run_decode(file=tmp_path / "f.h5", out=tmp_path / "f.obspack",
                     bbox=(3.0, 47.0, 16.0, 55.5), no_echo_dbz=-32.0)
    _name, _subcommand, arguments, _schema = recorder.calls[0]
    assert "--bbox" in arguments
    assert arguments[arguments.index("--bbox") + 1] == "3,47,16,55.5"
    assert arguments[arguments.index("--no-echo-dbz") + 1] == "-32"


def test_an_omitted_box_stays_omitted_rather_than_becoming_a_default(
        monkeypatch, tmp_path):
    """No implicit region.

    A default box would silently decide which part of Europe a case scored
    over, and the decision would live in a wrapper nobody reads.
    """

    recorder = _Recorder(opera.DECODE_SCHEMA)
    monkeypatch.setattr(frontdoor.FrontDoor, "run", recorder)
    opera.run_decode(file=tmp_path / "f.h5", out=tmp_path / "f.obspack")
    _name, _subcommand, arguments, _schema = recorder.calls[0]
    assert "--bbox" not in arguments
    assert "--no-echo-dbz" not in arguments


def test_the_geometry_command_never_carries_a_reflectivity_knob(
        monkeypatch, tmp_path):
    recorder = _Recorder(opera.GRID_SCHEMA)
    monkeypatch.setattr(frontdoor.FrontDoor, "run", recorder)
    opera.run_grid(file=tmp_path / "f.h5", out=tmp_path / "g.obspack",
                   bbox=(3.0, 47.0, 16.0, 55.5))
    _name, _subcommand, arguments, _schema = recorder.calls[0]
    assert "--no-echo-dbz" not in arguments


# ---------------------------------------------------------- the gap, named

def test_polar_volume_support_names_the_route_that_serves_them():
    """The gap this used to pin is closed, and the answer must say where.

    For most of this pipeline's life ``polar_volume_support()`` returned
    ``False`` and named the vendored composite decoder as the reason, so a
    caller planning a velocity-assimilating cycle could tell "Europe has no
    radial velocity" (untrue) from "nothing here can read it" (true).  The
    second is no longer true either: ``gpuwm.obs.odim`` drives ``rw_odim``,
    which reads polar volumes and writes the sweep pack the superob layer
    already consumes.

    Flipping the boolean is not enough on its own and would arguably be
    worse than leaving it, because this module still cannot serve a polar
    volume -- ``rw_opera`` decodes the 2-D composite and nothing else.  The
    answer therefore has to point at the other half by name, or a caller
    reading ``True`` here would go looking for a subcommand on this route
    that does not exist.
    """

    supported, reason = opera.polar_volume_support()
    assert supported is True
    assert "gpuwm.obs.odim" in reason
    assert "rw_odim" in reason
    # The user-facing route, not just the module: a caller who cannot reach
    # it from a command line has been told about a capability they cannot use.
    assert "gpuwm obs radar" in reason


def test_the_composite_route_still_refuses_to_claim_the_polar_product():
    """``rw_opera`` must not grow a polar leg, and its wrapper must not fake one.

    One binary with two products and two geometries is how a composite gets
    decoded through polar code, or the reverse.  The pairing is the design:
    this module's subcommands stay composite-only and the polar ones live in
    :mod:`gpuwm.obs.odim`.
    """

    assert not hasattr(opera, "run_pack")
    assert not hasattr(opera, "run_nyquist")
    assert opera.PACK_SCHEMA == "gpuwm-obs.obs-grid.v1"

    from gpuwm.obs import odim
    assert odim.SWEEP_PACK_SCHEMA == "gpuwm-obs.radar-sweeps.v3"
    assert odim.ODIM is not opera.OPERA
    assert odim.ODIM.name == "rw_odim"
    assert opera.OPERA.name == "rw_opera"


# ----------------------------------------------------------- the source

def test_the_source_reads_a_pack_and_reports_the_quantity(tmp_path):
    packs, geo = _tree(tmp_path, ["2026-08-12T19:25:00", "2026-08-12T19:30:00"])
    source = sources.OperaCompositeSource(packs, geo)
    assert source.quantity() == sources.QUANTITY_COMPOSITE_REFLECTIVITY
    assert source.valid_times() == ("2026-08-12T19:25:00",
                                    "2026-08-12T19:30:00")


def test_the_matching_window_is_the_published_cadence(tmp_path):
    """Five minutes, not the two-minute value the US composite uses.

    A window inherited from the other network would refuse frames that are
    merely the far side of one European interval.
    """

    packs, geo = _tree(tmp_path, ["2026-08-12T19:30:00"])
    source = sources.OperaCompositeSource(packs, geo)
    assert source._match.total_seconds() == opera.CADENCE_SECONDS
    # 4 minutes away is inside the window; 6 is not.
    assert source._nearest(
        sources._parse_time("2026-08-12T19:34:00")).pack_path == packs[0]
    with pytest.raises(LookupError, match="refusing rather than reaching"):
        source._nearest(sources._parse_time("2026-08-12T19:36:00"))


def test_a_pack_of_another_quantity_is_refused_by_its_own_metadata(tmp_path):
    path = tmp_path / "wrong.obspack"
    meta_values = np.array([[1.0, 2.0], [3.0, 4.0]])
    raw = build_pack({
        "schema": obspack.GRID_SCHEMA, "status": "READY",
        "quantity": sources.QUANTITY_PRECIPITATION_ACCUMULATION,
        "units": "mm", "valid_time": "2026-08-12T19:30:00",
        "provenance": {"source": "opera", "product": "x", "uri": "u",
                       "sha256": "c" * 64, "fetched_at": "t",
                       "is_stub": False, "stub_reason": ""},
    }, {"values": meta_values,
        "valid": np.ones(meta_values.shape, dtype=np.uint8)})
    path.write_bytes(raw)
    geo = tmp_path / "grid.obspack"
    geo.write_bytes(geo_pack())
    with pytest.raises(ValueError, match="this source serves"):
        sources.OperaCompositeSource([path], geo)


def test_the_coverage_floor_reads_the_front_doors_observed_fraction(tmp_path):
    """The floor screens on the fraction the front door measured.

    That number is only meaningful because the decoder counts a no-echo cell
    as observed. Were the two sentinels collapsed, a clear-air frame would
    report a few per cent coverage and every floor would reject it.
    """

    path = tmp_path / "sparse.obspack"
    path.write_bytes(opera_grid_pack(observed_fraction=0.02))
    geo = tmp_path / "grid.obspack"
    geo.write_bytes(geo_pack())
    source = sources.OperaCompositeSource([path], geo,
                                          minimum_observed_fraction=0.5)
    assert source._frames[0].observed_fraction == pytest.approx(0.02)


# ------------------------------------------------- the binary, when built

def _binary():
    try:
        return frontdoor.OPERA.find()
    except FileNotFoundError:
        return None


@pytest.mark.skipif(_binary() is None,
                    reason="rw_opera is not built in this checkout")
def test_the_built_binary_prints_the_abi_this_wrapper_pins():
    printed = subprocess.run([str(_binary()), "--abi"], capture_output=True,
                             text=True, timeout=30)
    assert printed.returncode == 0
    assert printed.stdout.strip() == frontdoor.OPERA.abi_marker


@pytest.mark.skipif(_binary() is None,
                    reason="rw_opera is not built in this checkout")
def test_the_built_binary_refuses_a_file_that_is_not_an_odim_frame(tmp_path):
    junk = tmp_path / "not-a-frame.h5"
    junk.write_bytes(b"this is not HDF5")
    result = subprocess.run(
        [str(_binary()), "decode", "--file", str(junk),
         "--out", str(tmp_path / "out.obspack")],
        capture_output=True, text=True, timeout=60)
    assert result.returncode != 0
    assert not (tmp_path / "out.obspack").exists()
    assert "HDF5" in result.stderr


@pytest.mark.skipif(_binary() is None,
                    reason="rw_opera is not built in this checkout")
def test_the_built_binary_refuses_an_antimeridian_box(tmp_path):
    result = subprocess.run(
        [str(_binary()), "decode", "--file", str(tmp_path / "x.h5"),
         "--out", str(tmp_path / "out.obspack"), "--bbox", "170,-10,-170,10"],
        capture_output=True, text=True, timeout=60)
    assert result.returncode != 0
    assert "antimeridian" in result.stderr
