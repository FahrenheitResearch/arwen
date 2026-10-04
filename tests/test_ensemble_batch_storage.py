"""Storage admission must not silently broadcast or underprice members."""
import numpy as np
import pytest

from gpuwm.ensemble.batch_storage import BatchArraySpec, BatchMemoryPlan, BatchStorage


def plan():
    return BatchMemoryPlan((BatchArraySpec("u", (3, 5, 8), "member"),
                            BatchArraySpec("mu", (5, 7), "member"),
                            BatchArraySpec("map", (5, 7), "shared")),
                           reserved_bytes=1024)


def test_admission_accounts_for_each_allocation_and_exact_fit():
    p = plan()
    # u=1920->2048, mu=560->1024, map=140->512, reserve=1024.
    assert p.required_bytes(4) == 4608
    assert p.largest_that_fits(4608, max_members=50) == 4
    assert p.largest_that_fits(4607, max_members=50) == 3
    assert p.largest_that_fits(0, max_members=50) == 0
    with pytest.raises(MemoryError, match="at most 3 members fit"):
        p.admit(4, available_bytes=4607)


def test_member_mutations_cannot_cross_slabs():
    p = plan()
    s = BatchStorage(p, 4, array_module=np, available_bytes=p.required_bytes(4))
    assert s.arrays["u"].shape == (4, 3, 5, 8)
    for member in range(4):
        s.member_view("u", member)[...] = member + 1
    for member in range(4):
        np.testing.assert_array_equal(s.arrays["u"][member], member + 1)
    assert s.payload_bytes == 4 * (480 + 140) + 140
    assert s.pointer_stride_bytes("u") == 480
    assert s.pointer_stride_bytes("map") == 0
    with pytest.raises(ValueError, match="read-only"):
        s.member_view("map", 1)[...] = 3


def test_shared_base_admission_compares_bytes_including_signed_zero():
    p = plan()
    s = BatchStorage(p, 4, array_module=np, available_bytes=p.required_bytes(4))
    fields = [np.zeros((5, 7), np.float32) for _ in range(4)]
    s.verify_shared_inputs("map", fields)
    fields[2][0, 0] = -0.0
    with pytest.raises(ValueError, match="differs in member 2"):
        s.verify_shared_inputs("map", fields)
    with pytest.raises(ValueError, match="one prepared input"):
        s.verify_shared_inputs("map", fields[:2])


@pytest.mark.parametrize("bad", [True, 0, -1, 2.5])
def test_invalid_member_count_never_reaches_allocator(bad):
    with pytest.raises((ValueError, TypeError)):
        BatchStorage(plan(), bad, array_module=np, available_bytes=10000)


def test_footprint_is_monotonic_and_matches_allocated_payload():
    p = plan()
    footprints = [p.required_bytes(n) for n in range(1, 51)]
    assert footprints == sorted(footprints)
    for n in (1, 4, 10, 20, 40):
        s = BatchStorage(p, n, array_module=np, available_bytes=p.required_bytes(n))
        assert s.payload_bytes == sum(r["payload_bytes"] for r in p.inventory(n))
        assert p.required_bytes(n) >= s.payload_bytes + p.reserved_bytes


def test_duplicate_names_and_implicit_ownership_are_rejected():
    a = BatchArraySpec("u", (2, 3), "member")
    with pytest.raises(ValueError, match="duplicate"):
        BatchMemoryPlan((a, a), reserved_bytes=0)
    with pytest.raises(ValueError, match="ownership"):
        BatchArraySpec("u", (2, 3), "auto")


@pytest.mark.parametrize("dtype", ["V0", "S10", "U5", "datetime64[D]", "object"])
def test_nondevice_dtypes_cannot_pass_zero_or_incorrect_payload_admission(dtype):
    with pytest.raises(TypeError, match="numeric scalar"):
        BatchArraySpec("field", (2, 3), "member", dtype=dtype)
