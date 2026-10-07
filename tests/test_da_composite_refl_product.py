"""DA composite frames carry one reflectivity product, and say which.

The breakage this pins, named: the CONUS first-light DA members' leg-end
frames carried the DA forward operator's reflectivity while free-forecast
history frames carried the model's own REFL_10CM; on one identical control
state at 20Z the two gave 1,739 against 1,368 cells at 35 dBZ or more, so a
proof run scored on history frames could not be compared with the campaign
scored on leg frames.  The member leg now drains the native field at every
history alarm and the leg-end frame uses it; every frame records its product.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def test_the_handler_records_native_reflectivity_at_every_alarm(monkeypatch):
    import gpuwm.prepared_single_domain_forecast as psf
    from tools import da_member_leg

    field = np.zeros((3, 2, 2), dtype=np.float32)
    field[1, 0, 1] = 42.0
    monkeypatch.setattr(psf, "_consume_due_native_refl_10cm",
                        lambda state, ticks, consumer, **kw: field)
    args = types.SimpleNamespace(rain_history=False,
                                 rain_forecast_start_seconds=10800.0)
    context = types.SimpleNamespace(args=args, out=Path("unused"))
    native = {}
    write = da_member_leg.rain_history_handler(
        context=context, name="0", drivers={1: None}, native=native)
    assert write is not None
    node = types.SimpleNamespace(
        state=None, cfg=types.SimpleNamespace(grid_id=1),
        clock=types.SimpleNamespace(elapsed_seconds=3600.0, tick_den=1,
                                    spec=None))
    write(None, node, 3600)
    ticks, colmax = native[1]
    assert ticks == 3600
    assert colmax.shape == (2, 2) and colmax[0, 1] == 42.0


def test_no_handler_without_history_or_composites():
    from tools import da_member_leg

    args = types.SimpleNamespace(rain_history=False)
    context = types.SimpleNamespace(args=args)
    assert da_member_leg.rain_history_handler(
        context=context, name="0", drivers={}) is None


def test_the_products_are_named():
    from tools import da_cycle_prepared as door

    assert door.REFL_PRODUCT_NATIVE == "native_refl_10cm"
    assert door.REFL_PRODUCT_OPERATOR == "da_forward_operator"


def test_the_first_frame_after_an_analysis_is_written_once(monkeypatch, tmp_path):
    """THE BREAKAGE: the 9 km CONUS cycle showed 4.8x MRMS's echo area two
    minutes after an analysis, and nothing on disk said whether the analysis
    or the model put it there.  An analysed leg now writes its first native
    frame (first{NN}_{member}.npz) once, retained by the rain history or
    not."""
    import gpuwm.prepared_single_domain_forecast as psf
    from tools import da_member_leg
    import tools.da_cycle_prepared as door

    field = np.full((3, 2, 2), 5.0, dtype=np.float32)
    monkeypatch.setattr(psf, "_consume_due_native_refl_10cm",
                        lambda state, ticks, consumer, **kw: field)
    written = []
    monkeypatch.setattr(door, "_write_rain_composite",
                        lambda path, **kw: written.append(
                            (Path(path).name, kw["elapsed_seconds"])))
    args = types.SimpleNamespace(rain_history=False,
                                 rain_forecast_start_seconds=10800.0)
    context = types.SimpleNamespace(
        args=args, out=tmp_path, t_start=3600.0,
        inputs=types.SimpleNamespace(experiment=None))
    write = da_member_leg.rain_history_handler(
        context=context, name="4", drivers={1: None}, native={},
        first_frame="first01_4.npz")
    for elapsed in (3720.0, 3840.0):
        node = types.SimpleNamespace(
            state=None, grid=None,
            cfg=types.SimpleNamespace(grid_id=1, run=None),
            clock=types.SimpleNamespace(elapsed_seconds=elapsed, tick_den=1,
                                        spec=None))
        write(None, node, int(elapsed))
    assert written == [("first01_4.npz", 3720.0)]
