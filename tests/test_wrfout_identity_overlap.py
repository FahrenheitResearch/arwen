"""Native output completion overlaps hashing without retaining frame arrays."""
import hashlib
import inspect
import os
from pathlib import Path
from threading import Event, Thread
import weakref

import netCDF4
import numpy as np
import pytest

from gpuwm import output_identity, runtime
from gpuwm.io.wrfout import PerDomainWrfoutWriters, WrfoutWriter
from test_wrfout import _manual_async_writer, _queue_cpu_ticket


def _supersede(replacement, path):
    """Replace the address while retaining the writer's open descriptor.

    Windows cannot replace or unlink a held target directly. Rename it
    aside first; the caller removes that test-owned copy after the writer
    closes its descriptor, before checking the writer's own recovery.
    """
    aside = path.with_name(path.name + ".superseded") if os.name == "nt" else None
    if aside is not None:
        path.rename(aside)
    replacement.replace(path)
    return aside


def _inventory(paths, writers):
    if "completed_records" in inspect.signature(runtime._frame_records).parameters:
        return runtime._frame_records(
            paths, completed_records=getattr(writers, "completed_records", ()))
    return runtime._frame_records(paths)


def test_native_completion_preserves_bytes_and_avoids_final_payload_read(
        tmp_path, monkeypatch):
    fields = {"T": np.full((1, 1, 1), 271.5, np.float32)}
    reference = tmp_path / "reference"
    with WrfoutWriter(reference, nx=1, ny=1, nz=1, dx=1., dy=1.,
                      title="test", soil_layers=4, field_schema=fields) as tape:
        assert tape.engine == "rust"
        tape.write_frame("1974-04-03_12:00:00", fields)
    path = tmp_path / "asynchronous"
    writer = _manual_async_writer(Event())
    try:
        _queue_cpu_ticket(writer, path, fields=fields)
        writer.close()
        assert path.read_bytes() == reference.read_bytes()
        with netCDF4.Dataset(path) as dataset:
            assert dataset.data_model == "NETCDF3_64BIT_OFFSET"
        expected = hashlib.sha256(path.read_bytes()).hexdigest()
        opening = Path.open
        reads = []

        class Reading:
            def __init__(self, stream):
                self.stream = stream
            def __enter__(self):
                return self
            def __exit__(self, *args):
                return self.stream.__exit__(*args)
            def fileno(self):
                return self.stream.fileno()
            def read(self, size=-1):
                value = self.stream.read(size)
                reads.append(len(value))
                return value

        def open_file(candidate, mode="r", *args, **kwargs):
            stream = opening(candidate, mode, *args, **kwargs)
            return Reading(stream) if candidate == path and mode == "rb" else stream

        monkeypatch.setattr(Path, "open", open_file)
        assert _inventory(writer.paths, writer) == [{
            "path": str(path.resolve()), "bytes": path.stat().st_size,
            "sha256": expected}]
        assert sum(reads) == (path.stat().st_size if os.name == "nt" else 0)
    finally:
        writer.close()


def test_staging_is_released_while_identity_waits_and_callback_waits_for_hash(
        tmp_path, monkeypatch):
    hashing, release, landed, consumed = Event(), Event(), Event(), Event()
    capture = getattr(output_identity, "completed_file_record", None)

    def delayed(*args, **kwargs):
        hashing.set()
        release.wait()
        return capture(*args, **kwargs)

    monkeypatch.setattr(output_identity, "completed_file_record", delayed, raising=False)
    writer = _manual_async_writer(Event())
    writer.landing_observer = lambda **kwargs: landed.set()
    owner = np.full((1, 1, 1), 272., np.float32)
    reference = weakref.ref(owner)
    path = tmp_path / "frame"
    ticket = _queue_cpu_ticket(writer, path, fields={"T": owner.view()},
                               pinned_refs=(owner,))
    del owner, ticket
    waiter = None
    try:
        # A baseline that never hashes signals its normal landing instead.
        # Observe either event, so that negative control cannot hang.
        while not hashing.wait(.05) and not landed.is_set():
            writer._raise_failure()
            assert writer._thread.is_alive()
        assert hashing.is_set(), "completion never started the output hash"
        assert path.is_file()
        assert reference() is None, "hashing retained a weather array"
        assert writer.pending == 1
        assert not landed.is_set()

        def drain_staging():
            writer.drain_staging()
            consumed.set()

        waiter = Thread(target=drain_staging)
        waiter.start()
        assert consumed.wait(2), "staging drain waited for payload hashing"
        assert not landed.is_set()
        assert writer._queue.maxsize == 1
    finally:
        release.set()
        if waiter is not None:
            waiter.join()
        writer.close()
    assert landed.is_set()
    assert len(writer.paths) == len(writer.completed_records) == 1


def test_hash_failure_preserves_file_and_prevents_completion(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise OSError("injected checksum read failure")

    monkeypatch.setattr(output_identity, "completed_file_record", fail, raising=False)
    writer = _manual_async_writer(Event())
    landed = []
    writer.landing_observer = lambda **kwargs: landed.append(kwargs)
    path = tmp_path / "frame"
    _queue_cpu_ticket(writer, path)
    with pytest.raises(RuntimeError, match="wrfout writer failed") as error:
        writer.close()
    assert isinstance(error.value.__cause__, OSError)
    assert "checksum read failure" in str(error.value.__cause__)
    assert path.is_file()
    assert writer.paths == [] and not landed
    assert not writer._thread.is_alive()
    assert writer.pending == 0


def test_retired_domain_and_member_paths_keep_all_completed_records(tmp_path):
    writers = [_manual_async_writer(Event()) for _ in range(2)]
    paths = [tmp_path / f"member-{index}" / "d01" / "frame"
             for index in range(2)]
    group = object.__new__(PerDomainWrfoutWriters)
    group._writers = {index + 1: writer for index, writer in enumerate(writers)}
    group._archived_paths = []
    group._metadata_by_grid_id = {1: {}, 2: {}}
    group._episode_by_grid_id = {1: 1, 2: 2}
    group.last_durable_wrfout = None
    try:
        for index, (writer, path) in enumerate(zip(writers, paths, strict=True)):
            _queue_cpu_ticket(writer, path, fields={
                "T": np.full((1, 1, 1), index + 1., np.float32)})
        group.drain()
        group.remove_domain(1)
        records = _inventory(group.paths, group)
        assert len(records) == 2
        assert {row["path"] for row in records} == {str(p.resolve()) for p in paths}
        assert len({row["sha256"] for row in records}) == 2
    finally:
        for writer in writers:
            writer.close()


def test_cancellation_stops_identity_before_completion_and_keeps_durable_bytes(
        tmp_path, monkeypatch):
    started, release, abort = Event(), Event(), Event()
    capture = output_identity.completed_file_record

    def delayed(*args, **kwargs):
        started.set()
        release.wait()
        return capture(*args, **kwargs)

    monkeypatch.setattr(output_identity, "completed_file_record", delayed)
    writer = _manual_async_writer(abort)
    landed = []
    writer.landing_observer = lambda **kwargs: landed.append(kwargs)
    path = tmp_path / "frame"
    _queue_cpu_ticket(writer, path)
    try:
        assert started.wait(2)
        durable = path.read_bytes()
        abort.set()
    finally:
        release.set()
        with pytest.raises(RuntimeError, match="wrfout writer failed") as error:
            writer.close()
    assert isinstance(error.value.__cause__, InterruptedError)
    assert path.read_bytes() == durable
    assert not landed and not writer.paths and not writer.completed_records


def test_another_domain_can_publish_while_first_domain_hash_waits(tmp_path, monkeypatch):
    started, release, second_landed = Event(), Event(), Event()
    capture = output_identity.completed_file_record
    first_path, second_path = tmp_path / "first", tmp_path / "second"

    def delayed(path, **kwargs):
        if path == first_path:
            started.set()
            release.wait()
        return capture(path, **kwargs)

    monkeypatch.setattr(output_identity, "completed_file_record", delayed)
    first, second = _manual_async_writer(Event()), _manual_async_writer(Event())
    second.landing_observer = lambda **kwargs: second_landed.set()
    try:
        _queue_cpu_ticket(first, first_path)
        assert started.wait(2)
        _queue_cpu_ticket(second, second_path)
        assert second_landed.wait(2), "payload hashing held the global NetCDF lock"
        assert first_path.is_file() and second_path.is_file()
        assert first.pending == 1 and not first.paths
    finally:
        release.set()
        first.close()
        second.close()


@pytest.mark.parametrize("seam", ("revision-capture", "rename-return"))
@pytest.mark.parametrize("replacement_kind", ("different-bytes", "same-bytes"))
def test_publication_stays_bound_to_validated_native_file(
        tmp_path, monkeypatch, seam, replacement_kind):
    import gpuwm.io.wrfout as wrfout

    path = tmp_path / "frame"
    original_bytes = []
    superseded = []
    revision = output_identity.publication_revision
    publish = wrfout.replace_file_with_retry

    def replace_final():
        before = path.read_bytes()
        original_bytes.append(before)
        replacement = tmp_path / "replacement"
        replacement.write_bytes(before if replacement_kind == "same-bytes"
                                else b"not the native writer output")
        superseded.append(_supersede(replacement, path))

    def capture(candidate, *args, **kwargs):
        if Path(candidate) == path and seam == "revision-capture":
            replace_final()
        return revision(candidate, *args, **kwargs)

    def rename(source, destination, *args, **kwargs):
        result = publish(source, destination, *args, **kwargs)
        if Path(destination) == path and seam == "rename-return":
            replace_final()
        return result

    monkeypatch.setattr(output_identity, "publication_revision", capture)
    monkeypatch.setattr(wrfout, "replace_file_with_retry", rename)
    fields = {"T": np.full((1, 1, 1), 271.5, np.float32)}
    with pytest.raises(RuntimeError, match="changed"):
        with WrfoutWriter(path, nx=1, ny=1, nz=1, dx=1., dy=1., title="test",
                          soil_layers=4, field_schema=fields) as writer:
            assert writer.engine == "rust"
            writer.write_frame("2026-09-12_12:00:00", fields)
        output_identity.completed_file_record(path, published=writer.publication_revision)
    assert len(original_bytes) == 1
    for aside in superseded:
        if aside is not None:
            aside.unlink()
    # The replacement is left alone. The original validated native bytes
    # must survive in an owned recovery location even after its name moved.
    assert any(candidate.read_bytes() == original_bytes[0]
               for candidate in tmp_path.rglob("*")
               if candidate.is_file() and candidate != path)


@pytest.mark.parametrize("replacement_kind", ("different-bytes", "same-bytes"))
def test_async_hash_keeps_the_published_descriptor_and_original_bytes(
        tmp_path, monkeypatch, replacement_kind):
    path = tmp_path / "frame"
    original_bytes = []
    superseded = []
    capture = output_identity.completed_file_record

    def replace_before_hash(candidate, **kwargs):
        original_bytes.append(path.read_bytes())
        replacement = tmp_path / "replacement"
        replacement.write_bytes(original_bytes[0] if replacement_kind == "same-bytes"
                                else b"not the native writer output")
        superseded.append(_supersede(replacement, path))
        return capture(candidate, **kwargs)

    monkeypatch.setattr(output_identity, "completed_file_record", replace_before_hash)
    writer = _manual_async_writer(Event())
    landed = []
    writer.landing_observer = lambda **kwargs: landed.append(kwargs)
    _queue_cpu_ticket(writer, path)
    with pytest.raises(RuntimeError, match="wrfout writer failed") as error:
        writer.close()
    assert "changed" in str(error.value.__cause__)
    assert not writer.completed_records and not writer.paths and not landed
    assert not writer._thread.is_alive() and writer.pending == 0
    for aside in superseded:
        if aside is not None:
            aside.unlink()
    assert any(candidate.read_bytes() == original_bytes[0]
               for candidate in tmp_path.rglob("*")
               if candidate.is_file() and candidate != path)
