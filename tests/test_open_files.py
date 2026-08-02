"""The open-file budget a preparation asks for before it starts.

The defect these cover, measured on the shipped 1.4.1 build (RTX 4080,
Ubuntu 24.04, soft RLIMIT_NOFILE 1024, hard 1048576): a 120 h GFS
series is 41 forcing times x 24 decoded arrays = 984 descriptors held
at once, the process peaked at 1023, and preprocessing died with

    rw-wps --source gfs: [Errno 24] Too many open files: '.../decoded/f000'

The identical command with the soft limit raised to the hard limit
completed at rc 0.  So the route the README calls Supported had a hard
ceiling on forecast length, and it announced itself as an IO fault
against a path that had already been read successfully.
"""

from __future__ import annotations

import sys

import pytest

from gpuwm.open_files import (DEFAULT_ARRAYS_PER_FORCING_TIME,
                              DESCRIPTOR_OVERHEAD,
                              ensure_descriptor_headroom,
                              required_descriptors)


def test_the_requirement_is_linear_in_forcing_times():
    """It is the number that made this a ceiling rather than a glitch."""

    assert required_descriptors(0) == DESCRIPTOR_OVERHEAD
    # The measured case: 120 h at cadence 3 is 41 forcing times.
    assert required_descriptors(41) == 41 * 24 + DESCRIPTOR_OVERHEAD
    assert required_descriptors(41) > 1024, \
        "the measured failure must be on the wrong side of a 1024 limit"
    # A 48 h series is not, which is why the ceiling looked like 48 h.
    assert required_descriptors(17) < 1024
    assert DEFAULT_ARRAYS_PER_FORCING_TIME == 24


class _FakeResource:
    """A settable RLIMIT_NOFILE, standing in for one process's limits."""

    RLIMIT_NOFILE = 7
    RLIM_INFINITY = -1

    def __init__(self, soft, hard, *, settable=True):
        self.soft, self.hard, self.settable = soft, hard, settable
        self.sets = []

    def getrlimit(self, _which):
        return (self.soft, self.hard)

    def setrlimit(self, _which, limits):
        if not self.settable:
            raise OSError("refused")
        soft, hard = limits
        if soft > self.hard:
            raise ValueError("soft above hard")
        self.sets.append((soft, hard))
        self.soft, self.hard = soft, hard


@pytest.fixture
def fake_resource(monkeypatch):
    def install(soft, hard, *, settable=True):
        fake = _FakeResource(soft, hard, settable=settable)
        monkeypatch.setitem(sys.modules, "resource", fake)
        return fake
    return install


def test_even_a_short_series_takes_the_whole_hard_limit(fake_resource):
    """The raise is UNCONDITIONAL, and this is why.

    A 60 h / 21-forcing-time series on a 60x48 grid exhausted a 1024
    limit, while the per-forcing-time model puts it at 632.  So the
    model is a lower bound on the requirement, not the requirement, and
    a raise gated on it would have left that case broken -- which is
    what the first cut of this function did.
    """

    fake = fake_resource(1024, 1048576)
    assert ensure_descriptor_headroom(required_descriptors(17),
                                      what="a short series") is None
    assert fake.sets == [(1048576, 1048576)]


def test_the_measured_failure_is_raised_rather_than_refused(fake_resource):
    """1024 soft / 1048576 hard is the ordinary Linux shape, and on it
    the 120 h series must simply be allowed to run."""

    fake = fake_resource(1024, 1048576)
    needed = required_descriptors(41)
    assert ensure_descriptor_headroom(needed, what="41 forcing times") is None
    # The WHOLE hard limit, not the modelled `needed`: the model is a
    # lower bound, and the measured peak was clamped by the limit it hit,
    # so its true requirement is unknown from above.
    assert fake.sets == [(1048576, 1048576)]
    assert fake.soft >= needed
    # The hard limit is somebody's deliberate setting and is not touched.
    assert fake.hard == 1048576


def test_a_hard_limit_that_cannot_cover_it_refuses_with_the_number(
        fake_resource):
    """The arm that must not become a silent success.

    A deployment that lowered its HARD limit made a decision; it gets
    told what that decision costs, before the decode, with the command
    that changes it -- rather than an [Errno 24] partway through naming
    a file that is not the problem.
    """

    fake = fake_resource(256, 512)
    needed = required_descriptors(41)
    message = ensure_descriptor_headroom(needed, what="41 forcing times")
    assert message is not None
    assert str(needed) in message
    assert "ulimit -n" in message
    # Names what it started with, what it got, and what it needed.
    assert "512" in message and "256" in message
    assert "linear in the number of forcing times" in message
    # It still took everything it could get, so the refusal is honest
    # about being a genuine shortfall rather than an unasked question.
    assert fake.soft == 512


def test_a_setrlimit_the_platform_refuses_is_reported_not_raised(
        fake_resource):
    """An OSError from setrlimit becomes the same sentence, not a
    traceback out of a preparation."""

    fake_resource(256, 1048576, settable=False)
    message = ensure_descriptor_headroom(required_descriptors(41),
                                         what="41 forcing times")
    assert message is not None
    assert "ulimit -n" in message


def test_the_scratch_tree_cleanup_cannot_replace_the_real_failure():
    """Both measured EMFILE failures named an innocent path.

    Descriptor exhaustion also stops `shutil.rmtree` -- it needs one per
    directory it walks -- so TemporaryDirectory's cleanup raised on the
    first directory it reached and its exception REPLACED the body's.
    Every report therefore said `decoded/f000`, which each run had
    already read successfully.
    """

    import inspect

    from gpuwm import gfs_direct

    source = inspect.getsource(gfs_direct.prepare_gfs_wrf)
    assert "ignore_cleanup_errors=True" in source


def test_a_platform_ceiling_below_the_hard_limit_falls_back_to_needed(
        fake_resource):
    """macOS shape: the hard limit is huge but setrlimit refuses it.

    Taking the whole hard limit is the first choice, not the only one --
    the fallback asks for exactly what the work needs, which is what
    lets a platform with an OPEN_MAX ceiling still run the route.
    """

    class Ceilinged(_FakeResource):
        def setrlimit(self, which, limits):
            if limits[0] > 4096:
                raise ValueError("above OPEN_MAX")
            return super().setrlimit(which, limits)

    fake = Ceilinged(1024, 1048576)
    monkey = pytest.MonkeyPatch()
    monkey.setitem(sys.modules, "resource", fake)
    try:
        needed = required_descriptors(41)
        assert ensure_descriptor_headroom(
            needed, what="41 forcing times") is None
        assert fake.sets == [(needed, 1048576)]
    finally:
        monkey.undo()


def test_an_unlimited_soft_limit_is_left_alone(fake_resource):
    fake = fake_resource(_FakeResource.RLIM_INFINITY, _FakeResource.RLIM_INFINITY)
    assert ensure_descriptor_headroom(10 ** 9, what="anything") is None
    assert fake.sets == []


def test_a_platform_without_the_limit_has_no_opinion(monkeypatch):
    """Windows has no RLIMIT_NOFILE and no `resource` module.

    That is not a degraded case to warn about: the limit being raised
    here does not exist there, so the answer is None and the caller
    proceeds exactly as it did.
    """

    import builtins

    real_import = builtins.__import__

    def no_resource(name, *args, **kwargs):
        if name == "resource":
            raise ImportError("no resource module on this platform")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_resource)
    monkeypatch.delitem(sys.modules, "resource", raising=False)
    assert ensure_descriptor_headroom(10 ** 9, what="anything") is None


def test_the_gfs_preparation_asks_before_it_decodes():
    """Pin the wiring: the ask happens on the module that decodes, with
    the field count that module actually writes -- not a copy of 24."""

    import inspect

    from gpuwm import gfs_direct

    source = inspect.getsource(gfs_direct)
    assert "ensure_descriptor_headroom(" in source
    assert "len(_THREE_D) + len(_TWO_D)" in source
    # And that count is the one the failure was made of.
    assert (len(gfs_direct._THREE_D) + len(gfs_direct._TWO_D)
            == DEFAULT_ARRAYS_PER_FORCING_TIME)
    # Before the decode, not after: the ask must precede the subprocess
    # that produces the files.
    head, _, tail = source.partition("ensure_descriptor_headroom(")
    assert "gpuwm-gfs-bridge-" in tail, \
        "the descriptor ask must come before the bridge scratch tree"
