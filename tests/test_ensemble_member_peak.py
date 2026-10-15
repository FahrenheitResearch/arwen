"""A member's receipt retains transient allocations after they are retired."""
from types import SimpleNamespace
from gpuwm.ensemble.member_stream import MemberCudaScope


def test_member_peak_retains_transient_and_reserved_high_water():
    state = {'live': 0, 'reserved': 0}
    pointer = object()
    def malloc(size):
        state.update(live=size, reserved=max(state['reserved'],size*2))
        return pointer
    pool = SimpleNamespace(malloc=malloc,used_bytes=lambda: state['live'],
                           total_bytes=lambda: state['reserved'])
    scope = MemberCudaScope(device_id=2,member_id=7,
        stream=SimpleNamespace(ptr=12,priority=-2),pool=pool)
    assert scope.malloc(1000) is pointer
    assert scope.malloc(20) is pointer
    record = scope.receipt()
    assert record['pool_live_peak_bytes'] == 1000
    assert record['pool_reserved_peak_bytes'] == 2000
    assert record['pool_peak_sampling'] == 'after_every_successful_member_allocation'
