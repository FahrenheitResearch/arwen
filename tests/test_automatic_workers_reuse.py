"""automatic_workers() reuses its answer for a few seconds.

The forecast's host math asks it on every main-thread call; each answer reads
affinity, the cgroup quota files and the memory files.  A thread count never
changes an element, so the reuse window moves wall time only.
"""
from gpuwm.ingest import cpu_backend, preparation_workers


def test_answer_is_reused_inside_the_window_and_refreshed_after(monkeypatch):
    calls = []
    monkeypatch.setattr(preparation_workers, "effective_workers",
                        lambda requested=None: calls.append(requested) or 7)
    clock = [100.0]
    monkeypatch.setattr(cpu_backend.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(cpu_backend, "_AUTOMATIC_WORKERS", {})
    monkeypatch.delenv(preparation_workers.PREPARATION_THREADS_ENV, raising=False)

    assert cpu_backend.automatic_workers() == 7
    assert cpu_backend.automatic_workers() == 7
    assert len(calls) == 1
    clock[0] += cpu_backend.AUTOMATIC_WORKERS_REUSE_SECONDS + 0.01
    assert cpu_backend.automatic_workers() == 7
    assert len(calls) == 2


def test_a_changed_thread_request_is_its_own_answer(monkeypatch):
    monkeypatch.setattr(preparation_workers, "effective_workers",
                        lambda requested=None: 3 if requested is None else int(requested))
    monkeypatch.setattr(cpu_backend, "_AUTOMATIC_WORKERS", {})
    monkeypatch.delenv(preparation_workers.PREPARATION_THREADS_ENV, raising=False)
    assert cpu_backend.automatic_workers() == 3
    monkeypatch.setenv(preparation_workers.PREPARATION_THREADS_ENV, "5")
    assert cpu_backend.automatic_workers() == 5
