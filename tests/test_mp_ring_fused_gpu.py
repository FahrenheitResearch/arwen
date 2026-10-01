"""The one-launch microphysics ring guard equals the slice-copy guard bit for bit.

``gpuwm.core.microphysics._capture_spec_zone_ring`` / ``_restore_spec_zone_ring``
move every ring section through ``mp_ring_copy`` (kernels/microphysics_validation.cu)
in one launch each way.
The reference below is the slice-copy body they replaced, kept verbatim, and
both run on the same inputs: random float32 bits including NaN payloads,
negative zero and denormal bit patterns, 3-D and 2-D fields, spec zones 1 to 5,
a degenerate domain whose ring is the whole grid, and a float64 field that
must take the slice-copy fallback.
"""
import numpy as np
import pytest

from conftest import requires_gpu


class _State:
    """The slice of DomainState the ring guard reads: named fields, scratch
    slots created on demand, existing scratch lookup and h_diabatic."""

    def __init__(self, cp, fields, surface, h_diabatic):
        self._cp = cp
        self._scratch = dict(surface)
        for name, arr in fields.items():
            setattr(self, name, arr)
        self.h_diabatic = h_diabatic

    def scratch(self, shape, name, dtype=None):
        buf = self._scratch.get(name)
        if buf is None or buf.shape != tuple(shape):
            buf = self._cp.zeros(shape, dtype=dtype or np.float32)
            self._scratch[name] = buf
        return buf

    def existing_scratch(self, name):
        return self._scratch.get(name)


def _reference_capture(state, slices, state_fields, slots):
    saved, captured = [], set()

    def snap(arr, key):
        for index, slc in enumerate(slices):
            part = arr[slc]
            if part.size == 0:
                continue
            buf = state.scratch(part.shape, f"mp_ring_save_{key}_{index}")
            buf[...] = part
            saved.append((arr, slc, buf))

    for name in state_fields:
        arr = getattr(state, name, None)
        if arr is not None:
            snap(arr, name)
    for slot in slots:
        arr = state.existing_scratch(slot)
        if arr is not None:
            snap(arr, slot)
            captured.add(slot)
    return saved, captured


def _reference_restore(state, slices, saved, captured, slots):
    for arr, slc, buf in saved:
        arr[slc] = buf
    for slot in slots:
        if slot in captured:
            continue
        arr = state.existing_scratch(slot)
        if arr is None:
            continue
        for slc in slices:
            arr[slc] = 0
    if state.h_diabatic is not None:
        for slc in slices:
            state.h_diabatic[slc] = 0


def _random_bits(rng, shape, dtype=np.float32):
    if dtype == np.float32:
        words = rng.integers(0, 2**32, size=shape, dtype=np.uint64).astype(np.uint32)
        flat = words.reshape(-1)
        flat[:7] = [0x7FC00001, 0xFFC12345, 0x80000000, 0x00000001,
                    0x007FFFFF, 0x7F800000, 0xFF800000][: flat.size]
        return words.view(np.float32)
    return rng.standard_normal(shape).astype(dtype)


@requires_gpu
@pytest.mark.parametrize("nz,ny,nx,sz", [
    (7, 11, 13, 1), (5, 12, 9, 2), (3, 20, 17, 5), (4, 3, 4, 2), (2, 2, 2, 1),
])
def test_fused_ring_guard_matches_slice_copies(nz, ny, nx, sz):
    import cupy as cp

    from gpuwm.core import microphysics as mp
    from gpuwm.core.physics_inventory import spec_zone_ring_slices

    slices = spec_zone_ring_slices(ny, nx, sz)
    names = ("qv", "qc", "qr", "nr", "qvolg", "odd64")
    slots = ("mp_rainnc", "mp_rainncv", "mp_sr", "refl_10cm", "mp_hailnc")
    arms = {}
    for arm in ("reference", "fused"):
        rng = np.random.default_rng(20260930 + nz * 1000 + ny * 10 + nx + sz)
        fields = {n: cp.asarray(_random_bits(rng, (nz, ny, nx),
                                             np.float64 if n == "odd64" else np.float32))
                  for n in names}
        surface = {"mp_rainnc": cp.asarray(_random_bits(rng, (ny, nx))),
                   "mp_rainncv": cp.asarray(_random_bits(rng, (ny, nx))),
                   "mp_sr": cp.asarray(_random_bits(rng, (ny, nx))),
                   "refl_10cm": cp.asarray(_random_bits(rng, (nz, ny, nx)))}
        state = _State(cp, fields, surface,
                       cp.asarray(_random_bits(rng, (nz, ny, nx))))
        if arm == "reference":
            saved, captured = _reference_capture(state, slices, names, slots)
        else:
            saved, captured = _fused_capture(mp, state, slices, names, slots)
        # A scheme call writes everywhere, ring included, and creates a slot.
        for n in names:
            arr = getattr(state, n)
            arr[...] = cp.asarray(_random_bits(rng, arr.shape, arr.dtype.type))
        for slot in ("mp_rainnc", "mp_rainncv", "mp_sr", "refl_10cm"):
            arr = state.existing_scratch(slot)
            arr[...] = cp.asarray(_random_bits(rng, arr.shape))
        state._scratch["mp_hailnc"] = cp.asarray(_random_bits(rng, (ny, nx)))
        state.h_diabatic[...] = cp.asarray(_random_bits(rng, (nz, ny, nx)))
        if arm == "reference":
            _reference_restore(state, slices, saved, captured, slots)
        else:
            _fused_restore(mp, state, slices, saved, captured, slots)
        cp.cuda.runtime.deviceSynchronize()
        arrays = {n: cp.asnumpy(getattr(state, n)) for n in names}
        arrays.update({s: cp.asnumpy(state.existing_scratch(s)) for s in slots})
        arrays["h_diabatic"] = cp.asnumpy(state.h_diabatic)
        arrays.update({f"save{i}": cp.asnumpy(b) for i, (_, _, b) in enumerate(saved)})
        arms[arm] = arrays
    assert arms["reference"].keys() == arms["fused"].keys()
    for key, ref in arms["reference"].items():
        got = arms["fused"][key]
        assert got.dtype == ref.dtype and got.shape == ref.shape, key
        assert got.tobytes() == ref.tobytes(), f"{key} differs"


def _fused_capture(mp, state, slices, names, slots):
    """Run the shipped capture with this test's field and slot families."""
    saved_names, saved_slots = mp._RING_STATE_FIELDS, (mp._RING_SURFACE_SLOTS, mp._RING_VOLUME_SLOTS)
    try:
        mp._RING_STATE_FIELDS = names
        mp._RING_SURFACE_SLOTS, mp._RING_VOLUME_SLOTS = tuple(slots), ()
        return mp._capture_spec_zone_ring(state, slices)
    finally:
        mp._RING_STATE_FIELDS = saved_names
        mp._RING_SURFACE_SLOTS, mp._RING_VOLUME_SLOTS = saved_slots


def _fused_restore(mp, state, slices, saved, captured, slots):
    saved_slots = (mp._RING_SURFACE_SLOTS, mp._RING_VOLUME_SLOTS)
    try:
        mp._RING_SURFACE_SLOTS, mp._RING_VOLUME_SLOTS = tuple(slots), ()
        mp._restore_spec_zone_ring(state, slices, saved, captured)
    finally:
        mp._RING_SURFACE_SLOTS, mp._RING_VOLUME_SLOTS = saved_slots


@requires_gpu
def test_fused_ring_guard_is_one_launch_each_way(monkeypatch):
    import cupy as cp

    from gpuwm.core import microphysics as mp
    from gpuwm.core.physics_inventory import spec_zone_ring_slices

    launches = []
    real = mp._launch_ring_rows

    def counting(rows, *, direction):
        launches.append((direction, len(rows)))
        return real(rows, direction=direction)

    monkeypatch.setattr(mp, "_launch_ring_rows", counting)
    nz, ny, nx, sz = 6, 10, 12, 1
    slices = spec_zone_ring_slices(ny, nx, sz)
    rng = np.random.default_rng(7)
    fields = {n: cp.asarray(_random_bits(rng, (nz, ny, nx))) for n in ("qv", "qc")}
    state = _State(cp, fields, {"mp_rainnc": cp.asarray(_random_bits(rng, (ny, nx)))},
                   cp.asarray(_random_bits(rng, (nz, ny, nx))))
    saved, captured = _fused_capture(mp, state, slices, ("qv", "qc"), ("mp_rainnc",))
    _fused_restore(mp, state, slices, saved, captured, ("mp_rainnc",))
    # 3 arrays x 4 ring edges gathered in one launch; the same 12 scattered
    # plus h_diabatic's 4 edges zeroed in one launch.
    assert launches == [(0, 12), (1, 16)]


@requires_gpu
@pytest.mark.parametrize("warm", [False, True])
def test_ring_guard_inside_a_cuda_graph_capture(warm):
    """The tiled runner's --graph path captures whole steps, microphysics
    included.  With its descriptor table not resident yet, the guard must
    not upload inside the capture (a host transfer fails it) and falls back
    to slice copies; with it resident, the one-launch kernel is captured.
    Either way the replayed graph writes what the slice-copy guard writes."""
    import cupy as cp

    from gpuwm.core import microphysics as mp
    from gpuwm.core.physics_inventory import spec_zone_ring_slices

    nz, ny, nx, sz = 5, 9, 11, 2
    slices = spec_zone_ring_slices(ny, nx, sz)
    names, slots = ("qv", "qr"), ("mp_rainnc", "mp_sr")
    rng = np.random.default_rng(11)
    start = {n: _random_bits(rng, (nz, ny, nx)) for n in names}
    start.update({s: _random_bits(rng, (ny, nx)) for s in slots})
    start["h_diabatic"] = _random_bits(rng, (nz, ny, nx))
    after = {k: _random_bits(rng, v.shape) for k, v in start.items()}

    def build():
        fields = {n: cp.asarray(start[n]) for n in names}
        surface = {s: cp.asarray(start[s]) for s in slots}
        return _State(cp, fields, surface, cp.asarray(start["h_diabatic"]))

    reference = build()
    saved, captured = _reference_capture(reference, slices, names, slots)
    for k, v in after.items():
        target = (reference.h_diabatic if k == "h_diabatic" else
                  getattr(reference, k, None) if k in names
                  else reference.existing_scratch(k))
        target[...] = cp.asarray(v)
    _reference_restore(reference, slices, saved, captured, slots)

    state = build()
    # Allocate the save slots (and, warm, the tables) outside the capture.
    _fused_capture(mp, state, slices, names, slots)
    for k in list(mp._RING_TABLES):
        if not warm:
            del mp._RING_TABLES[k]
    for k, v in start.items():
        target = (state.h_diabatic if k == "h_diabatic" else
                  getattr(state, k, None) if k in names
                  else state.existing_scratch(k))
        target[...] = cp.asarray(v)
    writes = {k: cp.asarray(v) for k, v in after.items()}
    cp.cuda.runtime.deviceSynchronize()
    stream = cp.cuda.Stream(non_blocking=True)
    with stream:
        stream.begin_capture()
        saved2, captured2 = _fused_capture(mp, state, slices, names, slots)
        for k, v in writes.items():
            target = (state.h_diabatic if k == "h_diabatic" else
                      getattr(state, k, None) if k in names
                      else state.existing_scratch(k))
            target[...] = v
        _fused_restore(mp, state, slices, saved2, captured2, slots)
        graph = stream.end_capture()
    graph.launch(stream)
    stream.synchronize()
    for k in start:
        got = (state.h_diabatic if k == "h_diabatic" else
               getattr(state, k, None) if k in names else state.existing_scratch(k))
        ref = (reference.h_diabatic if k == "h_diabatic" else
               getattr(reference, k, None) if k in names
               else reference.existing_scratch(k))
        assert cp.asnumpy(got).tobytes() == cp.asnumpy(ref).tobytes(), k
