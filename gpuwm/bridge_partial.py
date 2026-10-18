"""Fetch selected pinned ZIP members with HTTP ranges, without a full bundle.

The archive hash cannot cover omitted bytes. Each installed file still passes
its packaged size, SHA-256 and ABI pin before replacing an existing artifact.
"""
from __future__ import annotations

import io
from pathlib import Path
import tempfile
import zipfile
from urllib.request import Request


class RangeReader(io.RawIOBase):
    def __init__(self, url, size, opener):
        self.url, self.size, self.opener, self.position = url, size, opener, 0
        self.transferred = 0
        # The first network error, kept because zipfile reports any OSError
        # while it looks for the central directory as "File is not a zip
        # file", which named a refused connection or an HTTP 404 as a
        # damaged bundle (2.9.1 verify of the on-demand doors).
        self.network_error = None

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=io.SEEK_SET):
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self.position, io.SEEK_END: self.size}[whence]
        position = base + offset
        if position < 0:
            raise ValueError("negative ZIP seek")
        self.position = position
        return position

    def read(self, size=-1):
        from .bridge_assets import BridgeAssetError
        count = min(self.size - self.position, size if size >= 0 else self.size)
        if count <= 0:
            return b""
        start, end = self.position, self.position + count - 1
        request = Request(self.url, headers={"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"})
        try:
            response_cm = self.opener(request)
        except OSError as error:
            if self.network_error is None:
                self.network_error = error
            raise
        with response_cm as response:
            if response.status != 206 or response.headers.get("Content-Range") != f"bytes {start}-{end}/{self.size}":
                raise BridgeAssetError("partial fetch requires exact HTTP byte ranges; server refused or bundle size changed. Use fetch-bridges without --partial for a full download")
            data = response.read(count + 1)
        if len(data) != count:
            raise BridgeAssetError(f"partial fetch received {len(data)} bytes, expected {count}")
        self.position += count
        self.transferred += count
        return data


def fetch_partial(pins, bundle, dest: Path, artifacts=None, *, progress=print, urlopen_fn=None):
    from . import bridge_assets as assets
    opener = urlopen_fn or assets._default_urlopen
    wanted = set(artifacts or [pin.artifact for pin in bundle.binaries])
    known = {pin.artifact for pin in bundle.binaries}
    unknown = wanted - known
    if unknown:
        raise assets.BridgeAssetError("partial fetch names unpinned artifacts: " + ", ".join(sorted(unknown)))
    # The five compatibility launchers cannot run without the shared binary.
    if wanted & {"rw_ensbatch", "rw_compare", "rw_verify", "rw_scorecard", "rw_nowcast_frames"}:
        wanted.add("rw_wrfbatch")
    selected = [pin for pin in bundle.binaries if pin.artifact in wanted]
    missing = [pin for pin in selected if not assets.matches_pin(dest / pin.filename, pin)]
    if not missing:
        return []
    reader = RangeReader(assets.bundle_url(pins, bundle), bundle.bytes, opener)
    installed = []
    dest.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.TemporaryDirectory(prefix=".partial-", dir=dest) as work, zipfile.ZipFile(reader) as archive:
            names = archive.namelist()
            for pin in missing:
                if names.count(pin.filename) != 1:
                    raise assets.BridgeAssetError(f"partial fetch needs one ZIP member {pin.filename}")
                info = archive.getinfo(pin.filename)
                if info.file_size != pin.bytes or info.compress_size > pin.bytes + 65536:
                    raise assets.BridgeAssetError(f"{pin.filename}: partial ZIP member size differs from pin")
                temp = Path(work) / pin.filename
                with archive.open(info) as stream:
                    payload = stream.read(pin.bytes + 1)
                temp.write_bytes(payload)
                assets._verify_binary(temp, pin)
                final = dest / pin.filename
                assets._install(temp, final, pin)
                installed.append(final)
                progress(f"gpuwm fetch-bridges: staged {pin.filename} by partial download")
    except (OSError, ValueError, zipfile.BadZipFile, RuntimeError) as error:
        if reader.network_error is not None:
            raise assets.BridgeAssetError(
                f"partial fetch failed: the release bundle could not be read "
                f"from {assets.bundle_url(pins, bundle)}: "
                f"{reader.network_error}") from error
        raise assets.BridgeAssetError(f"partial fetch failed: {error}") from error
    progress(f"gpuwm fetch-bridges: fetched {reader.transferred} of {bundle.bytes} bundle bytes")
    return installed
