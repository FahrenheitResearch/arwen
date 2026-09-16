"""A CDS credential refusal names the file, its state, its source and the
way out, and ``python -m gpuwm`` reaches the door the key panel calls.

THE BREAKAGE THIS COVERS: an ERA5 fetch on the shipped desktop refused
with one fixed sentence whether the credential file was missing, sat in a
different home folder, was saved by Notepad as ``.cdsapirc.txt``, was
written by a PowerShell redirect as UTF-16, carried a byte-order mark,
held only a key line, or held the retired ``UID:KEY`` pair; and the
product's own key panel could not save anything because the module it
runs, ``python -m gpuwm``, did not exist.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

import gpuwm
from gpuwm import cds_credentials as cds
from gpuwm import era5_acquisition

TOKEN = "0f0f0f0f-1111-2222-3333-THROWAWAYTOK"
CURRENT = f"url: {cds.DEFAULT_URL}\nkey: {TOKEN}\n"
LEGACY = f"url: https://cds.climate.copernicus.eu/api/v2\nkey: 12345:{TOKEN}\n"


@pytest.fixture
def clean_environment(monkeypatch):
    for name in ("CDSAPI_RC", "CDSAPI_KEY", "CDSAPI_URL"):
        monkeypatch.delenv(name, raising=False)


def write(path: Path, text: str, *, encoding="utf-8", newline="\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding=encoding, newline=newline) as stream:
        stream.write(text)
    return path


def use_home(monkeypatch, home: Path) -> Path:
    home.mkdir(parents=True, exist_ok=True)
    text = str(home)
    monkeypatch.setenv("USERPROFILE", text)
    monkeypatch.setenv("HOME", text)
    if os.name == "nt":
        monkeypatch.setenv("HOMEDRIVE", text[:2])
        monkeypatch.setenv("HOMEPATH", text[2:])
    return home


class MissingConfiguration(Exception):
    """What cdsapi raises when its file has no usable url and key."""


def refusal_for(path: Path) -> str:
    return cds.client_refusal(MissingConfiguration(f"Missing/incomplete configuration file: {path}"))


# -- the door the key panel calls --------------------------------------------

def test_python_dash_m_gpuwm_reaches_the_cds_credentials_door(tmp_path, clean_environment):
    """The terminal's key panel runs exactly this command; it used to exit 1
    with 'No module named gpuwm.__main__' on every install."""
    credential = write(tmp_path / "private" / ".cdsapirc", CURRENT)
    env = {name: value for name, value in os.environ.items()
           if not name.upper().startswith(("PYTHON", "GPUWM_", "CDSAPI"))}
    env.update(CDSAPI_RC=str(credential), PYTHONSAFEPATH="1", PYTHONDONTWRITEBYTECODE="1",
               PYTHONPATH=str(Path(gpuwm.__file__).resolve().parent.parent))
    done = subprocess.run([sys.executable, "-m", "gpuwm", "cds-credentials", "--json"],
                          cwd=tmp_path, env=env, capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr
    status = json.loads(done.stdout)
    assert status["schema"] == cds.SCHEMA and status["configured"] and status["source"] == "file"
    assert status["path"] == str(credential.absolute())
    assert TOKEN not in done.stdout


# -- the refusal carries its evidence -----------------------------------------

def test_a_missing_file_is_named_with_the_home_it_was_looked_for_in(tmp_path, monkeypatch, clean_environment):
    home = use_home(monkeypatch, tmp_path / "home-b")
    write(tmp_path / "home-a" / ".cdsapirc", CURRENT)
    sentence = refusal_for(home / ".cdsapirc")
    assert "MissingConfiguration: Missing/incomplete configuration file" in sentence
    assert f"Credential file {home / '.cdsapirc'}" in sentence and "is missing" in sentence
    assert f"home folder {home}" in sentence
    assert "the client reads the file" in sentence
    assert "Next: No credential file exists at" in sentence and "CDS key panel" in sentence
    assert TOKEN not in sentence


def test_a_notepad_dot_txt_sibling_is_named_as_the_way_out(tmp_path, monkeypatch, clean_environment):
    home = use_home(monkeypatch, tmp_path / "home")
    sibling = write(home / ".cdsapirc.txt", CURRENT)
    report = cds.inspect()
    assert report["problem"] == "missing" and report["sibling"] == str(sibling)
    sentence = refusal_for(home / ".cdsapirc")
    assert str(sibling) in sentence and "rename it" in sentence


@pytest.mark.parametrize("encoding, newline, named", [
    ("utf-16", "\r\n", "UTF-16, which is what a PowerShell redirect writes"),
    ("utf-8-sig", "\n", "byte-order mark"),
])
def test_an_encoding_the_client_cannot_read_is_named(tmp_path, monkeypatch, clean_environment,
                                                     encoding, newline, named):
    credential = write(tmp_path / ".cdsapirc", CURRENT, encoding=encoding, newline=newline)
    monkeypatch.setenv("CDSAPI_RC", str(credential))
    report = cds.inspect()
    assert report["problem"] == "encoding" and report["exists"]
    sentence = refusal_for(credential)
    assert "(chosen by CDSAPI_RC)" in sentence and "exists (" in sentence
    assert named in sentence and "re-save the file as UTF-8" in sentence
    assert TOKEN not in sentence
    assert not cds.status()["configured"] and cds.status()["problem"] == "encoding"


def test_windows_line_endings_are_not_a_problem(tmp_path, monkeypatch, clean_environment):
    credential = write(tmp_path / ".cdsapirc", CURRENT, newline="\r\n")
    monkeypatch.setenv("CDSAPI_RC", str(credential))
    report = cds.inspect()
    assert report["problem"] is None and report["key_shape"] == "personal-access-token"
    assert cds.status()["configured"]


def test_a_file_with_only_a_key_line_names_the_missing_url(tmp_path, monkeypatch, clean_environment):
    credential = write(tmp_path / ".cdsapirc", f"key: {TOKEN}\n")
    monkeypatch.setenv("CDSAPI_RC", str(credential))
    sentence = refusal_for(credential)
    assert "url line absent, key line present" in sentence
    assert f"add 'url: {cds.DEFAULT_URL}'" in sentence
    assert TOKEN not in sentence


def test_the_retired_uid_key_pair_is_named_before_and_after_the_request(tmp_path, monkeypatch, clean_environment):
    credential = write(tmp_path / ".cdsapirc", LEGACY)
    monkeypatch.setenv("CDSAPI_RC", str(credential))
    report = cds.inspect()
    assert report["key_shape"] == "legacy-uid-key" and report["url_shape"] == "retired-v2"
    assert report["problem"] == "legacy-key"
    status = cds.status()
    assert not status["configured"] and status["url"] == cds.DEFAULT_URL and status["problem"] == "legacy-key"

    class Response:
        status_code = 404

    class HTTPError(Exception):
        response = Response()

    sentence = cds.retrieval_refusal(HTTPError(f"404 Client Error for url with key 12345:{TOKEN}"))
    assert sentence.startswith("ERA5 retrieval failed at CDS: HTTP 404, HTTPError: 404 Client Error")
    assert "retired 'UID:KEY' shape" in sentence and "personal access token" in sentence
    assert TOKEN not in sentence and "12345:" not in sentence


def test_environment_credentials_are_named_as_the_source(tmp_path, monkeypatch, clean_environment):
    monkeypatch.setenv("CDSAPI_RC", str(tmp_path / "absent"))
    monkeypatch.setenv("CDSAPI_URL", cds.DEFAULT_URL)
    monkeypatch.setenv("CDSAPI_KEY", TOKEN)
    report = cds.inspect()
    assert report["problem"] is None
    sentence = cds.client_refusal(RuntimeError("client construction failed"))
    assert "uses CDSAPI_URL and CDSAPI_KEY from the environment, not the file" in sentence
    assert "is missing" in sentence and "reinstall cdsapi" in sentence
    assert TOKEN not in sentence


def test_a_rejected_token_names_the_http_status_and_the_licences(tmp_path, monkeypatch, clean_environment):
    credential = write(tmp_path / ".cdsapirc", CURRENT)
    monkeypatch.setenv("CDSAPI_RC", str(credential))

    class Response:
        status_code = 401

    class HTTPError(Exception):
        response = Response()

    sentence = cds.retrieval_refusal(HTTPError("401 Client Error: Unauthorized"))
    assert "HTTP 401, HTTPError: 401 Client Error: Unauthorized" in sentence
    assert "CDS rejected the token" in sentence and "dataset licences" in sentence
    assert "exists (" in sentence and "url line present, key line present" in sentence


# -- redaction ----------------------------------------------------------------

def test_redaction_masks_known_secrets_and_token_shapes_but_keeps_the_path(tmp_path):
    path = tmp_path / ".cdsapirc"
    text = (f"file {path}: key: {TOKEN}; uuid 01234567-89ab-cdef-0123-456789abcdef; "
            "pair 98765:0123456789abcdef; Bearer abc.def; nothing else")
    masked = cds.redact(text, TOKEN, "")
    assert str(path) in masked
    for secret in (TOKEN, "01234567-89ab-cdef-0123-456789abcdef", "98765:0123456789abcdef", "abc.def"):
        assert secret not in masked
    assert "nothing else" in masked


def test_the_file_key_is_masked_in_the_clients_message_even_when_unknown_to_the_caller(
        tmp_path, monkeypatch, clean_environment):
    credential = write(tmp_path / ".cdsapirc", CURRENT)
    monkeypatch.setenv("CDSAPI_RC", str(credential))
    sentence = cds.client_refusal(RuntimeError(f"rejected token {TOKEN} at startup"))
    assert TOKEN not in sentence and "[redacted]" in sentence


# -- the real client, when it is installed --------------------------------------

def test_the_real_client_refusal_carries_the_evidence(tmp_path, monkeypatch, clean_environment):
    pytest.importorskip("cdsapi")
    missing = tmp_path / "nowhere" / ".cdsapirc"
    monkeypatch.setenv("CDSAPI_RC", str(missing))
    with pytest.raises(ValueError) as refusal:
        era5_acquisition._client(lambda message: None)
    sentence = str(refusal.value)
    assert sentence.startswith("Cannot initialize the CDS client: cdsapi raised Exception: "
                               "Missing/incomplete configuration file")
    assert f"Credential file {missing} (chosen by CDSAPI_RC) is missing" in sentence
    assert "Next: No credential file exists at" in sentence


# -- the panel's save writes a file the client reads ----------------------------

@pytest.fixture
def plain_save(monkeypatch):
    monkeypatch.setattr(cds, "_private_windows_file", lambda path: None)


def test_saving_a_new_token_over_a_retired_file_moves_to_the_current_endpoint(
        tmp_path, monkeypatch, clean_environment, plain_save):
    credential = write(tmp_path / ".cdsapirc", LEGACY)
    monkeypatch.setenv("CDSAPI_RC", str(credential))
    result = cds.save({"key": "new-personal-access-token"})
    assert result["configured"] and result["url"] == cds.DEFAULT_URL and result["problem"] is None
    assert credential.read_bytes() == f"url: {cds.DEFAULT_URL}\nkey: new-personal-access-token\n".encode()


def test_saving_a_retired_uid_key_pair_is_refused_by_name(tmp_path, monkeypatch, clean_environment, plain_save):
    credential = tmp_path / ".cdsapirc"
    monkeypatch.setenv("CDSAPI_RC", str(credential))
    with pytest.raises(ValueError, match="UID:KEY"):
        cds.save({"key": "12345:abcdef"})
    assert not credential.exists()


def test_saving_over_an_unreadable_encoding_writes_plain_utf8(tmp_path, monkeypatch, clean_environment, plain_save):
    credential = write(tmp_path / ".cdsapirc", CURRENT, encoding="utf-16", newline="\r\n")
    monkeypatch.setenv("CDSAPI_RC", str(credential))
    assert cds.status()["problem"] == "encoding"
    result = cds.save({"key": "fresh-token"})
    assert result["configured"] and cds.inspect()["encoding"] == "utf-8"
    assert credential.read_bytes() == f"url: {cds.DEFAULT_URL}\nkey: fresh-token\n".encode()


def test_the_retired_endpoint_is_refused_by_name():
    with pytest.raises(ValueError, match="switched off in 2024"):
        cds._endpoint("https://cds.climate.copernicus.eu/api/v2")
    assert cds._endpoint(cds.DEFAULT_URL + "/") == cds.DEFAULT_URL
