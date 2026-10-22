"""Fast transport tests for environment credentials, ranges and receipts."""
import hashlib
import io
import json
from urllib.error import HTTPError

import pytest

from gpuwm import hf_fetch as hf


@pytest.fixture(autouse=True)
def isolated_box_governor(monkeypatch, tmp_path):
    monkeypatch.setenv("GPUWM_HF_GOVERNOR", str(tmp_path / ".hf-box.lock"))


class Response(io.BytesIO):
    def __init__(self, data, status=200, headers=None):
        super().__init__(data)
        self.status = status
        self.headers = headers or {}


def test_gated_401_uses_no_token_file(monkeypatch, tmp_path):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    seen = []

    class Opener:
        def open(self, request, timeout):
            seen.append(request)
            raise HTTPError(request.full_url, 401, "Unauthorized", {}, io.BytesIO())

    monkeypatch.setattr(hf, "build_opener", lambda *a: Opener())
    with pytest.raises(hf.FetchError, match=r"org/private is gated or private \(HTTP 401\) and HF_TOKEN is not set in the environment"):
        hf.fetch(hf.Target("org/private"), cache_root=tmp_path)
    assert len(seen) == 1
    assert not seen[0].has_header("Authorization")


def test_resume_and_receipt_second_fetch_downloads_nothing(monkeypatch, tmp_path):
    # Chunk the payload to exercise concurrent large-file handling and resume.
    content = b"generic-table-package"
    monkeypatch.setattr(hf, "CHUNK_BYTES", 8)
    sha = "a" * 40
    digest = hashlib.sha256(content).hexdigest()
    config = b'{}'
    blob = hashlib.sha1(b"blob 2\0" + config).hexdigest()
    metadata = {"sha": sha, "siblings": [{"rfilename": "weights.bin", "size": len(content),
                "lfs": {"sha256": digest}}, {"rfilename": "config.json", "size": len(config),
                "blobId": blob}]}
    target = hf.Target("org/package", sha, files=(
        {"path": "weights.bin", "size": len(content), "sha256": digest},
        {"path": "config.json", "size": len(config), "sha256": None}))
    ranges = []

    class Opener:
        def open(self, request, timeout):
            if "/api/" in request.full_url:
                return Response(json.dumps(metadata).encode())
            assert f"/resolve/{sha}/" in request.full_url
            value = request.get_header("Range")
            ranges.append(value)
            start, end = map(int, value.removeprefix("bytes=").split("-"))
            payload = config if request.full_url.endswith("/config.json") else content
            return Response(payload[start:end + 1], 206,
                            {"Content-Range": f"bytes {start}-{end}/{len(payload)}"})

    monkeypatch.setattr(hf, "build_opener", lambda *a: Opener())
    package = tmp_path / "org/package" / sha
    package.mkdir(parents=True)
    (package / "weights.bin.chunk-0.part").write_bytes(content[:3])
    first = hf.fetch(target, cache_root=tmp_path, progress=lambda _: None)
    assert first["bytes"] == len(content) + len(config) - 3
    assert "bytes=3-7" in ranges
    assert (package / "weights.bin").read_bytes() == content
    files = {entry["path"]: entry for entry in first["files"]}
    assert files["weights.bin"]["sha256"] == digest
    assert files["config.json"]["git_blob"] == blob
    assert (package / "config.json").read_bytes() == config
    assert not list(package.glob("*.part"))
    ranges.clear()
    second = hf.fetch(target, cache_root=tmp_path, progress=lambda _: None)
    assert second["bytes"] == 0
    assert ranges == []
    # Corruption of an equal-sized cache entry is caught by SHA256.
    (package / "weights.bin").write_bytes(b"x" * len(content))
    repaired = hf.fetch(target, cache_root=tmp_path, progress=lambda _: None)
    assert repaired["bytes"] == len(content)
    assert (package / "weights.bin").read_bytes() == content


def test_dataset_url_and_environment_alias(monkeypatch):
    target = hf.target_for("hf://datasets/org/weather@v1/nested/data.parquet")
    assert target == hf.Target("org/weather", "v1", repo_type="dataset", prefix="nested/data.parquet")
    from gpuwm import earth2_run
    record = {"path": "nested/data.parquet", "size": 2, "sha256": None}
    monkeypatch.setattr(earth2_run, "model_table", lambda: {"models": {
        "generic": {"licence_class": "permissive", "weights": {"host": "huggingface", "repository": "org/weather",
                    "revision": "b" * 40, "files": [record], "repo_type": "dataset"}}}})
    assert hf.target_for("generic") == hf.Target("org/weather", "b" * 40, (record,), "dataset")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", "unit-test-only")
    transport = hf._Transport("org/weather", 2)
    assert transport.token == "unit-test-only"


def test_restricted_row_refuses_without_accept_licence(monkeypatch, tmp_path, capsys):
    """`gpuwm hf-fetch pangu` downloaded 2.36 GB of CC BY-NC-SA weights with rc 0."""
    from gpuwm import cli
    fetched = []
    monkeypatch.setattr(hf, "_fetch_locked", lambda target, *a: fetched.append(target) or {"repo": target.repository})
    assert cli.main(["hf-fetch", "pangu", "--cache", str(tmp_path)]) == 2
    assert "--accept-licence 'CC BY-NC-SA 4.0'" in capsys.readouterr().err
    with pytest.raises(ValueError, match=r"--accept-licence 'CC BY-NC-SA 4.0'"):
        hf.fetch("pangu", cache_root=tmp_path)
    assert not fetched
    assert cli.main(["hf-fetch", "pangu", "--cache", str(tmp_path), "--accept-licence", "CC BY-NC-SA 4.0"]) == 0
    assert len(fetched) == 1


def test_gated_repo_refuses_at_exit_2_and_leaves_no_part_files(monkeypatch, tmp_path, capsys):
    # HF serves gated-repo metadata and its README anonymously; only the
    # gated files answer 401. The refusal must come before any download.
    readme = b"# card\n"
    metadata = {"sha": "c" * 40, "gated": "manual", "private": False, "siblings": [
        {"rfilename": "README.md", "size": len(readme),
         "blobId": hashlib.sha1(b"blob %d\0" % len(readme) + readme).hexdigest()},
        {"rfilename": "sub/model.safetensors", "size": 4, "lfs": {"sha256": "d" * 64}}]}
    resolves = []

    class Opener:
        def open(self, request, timeout):
            if "/api/" in request.full_url:
                return Response(json.dumps(metadata).encode())
            resolves.append(request.full_url)
            if request.full_url.endswith("/README.md"):
                return Response(readme, 206, {"Content-Range": f"bytes 0-{len(readme) - 1}/{len(readme)}"})
            raise HTTPError(request.full_url, 401, "Unauthorized", {}, io.BytesIO())

    monkeypatch.setattr(hf, "build_opener", lambda *a: Opener())
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    import argparse
    args = argparse.Namespace(target="hf://org/gated", cache=str(tmp_path), connections=None)
    assert hf.cli(args) == 2
    err = capsys.readouterr().err
    assert err.startswith("gpuwm hf-fetch: org/gated is gated or private") and "Traceback" not in err
    assert resolves == []
    assert not list(tmp_path.rglob("*.part")) and not (tmp_path / "org").exists()
    # A token without access refuses on the first gated file: partials this
    # call created are removed, not kept for a resume that cannot happen.
    monkeypatch.setenv("HF_TOKEN", "unit-test-only")
    assert hf.cli(args) == 2
    assert "the environment token has no access" in capsys.readouterr().err
    assert resolves
    assert not list(tmp_path.rglob("*.part")) and not (tmp_path / "org").exists()
