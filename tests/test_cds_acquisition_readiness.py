"""Credential readiness can leave the fetch computer; secrets never can."""
import importlib.util
import json

import pytest

from gpuwm import cds_credentials as cds


@pytest.fixture
def client_present(monkeypatch):
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object())


def test_private_file_readiness_returns_no_secret_path_or_endpoint(tmp_path, monkeypatch, client_present):
    token = "NEVER-EMIT-THIS-TOKEN"
    path = tmp_path / "PRIVATE-LOCATION.cdsapirc"
    path.write_text(f"url: https://example.invalid/PRIVATE-ENDPOINT\nkey: {token}\n")
    monkeypatch.setenv("CDSAPI_RC", str(path))
    monkeypatch.delenv("CDSAPI_URL", raising=False)
    monkeypatch.delenv("CDSAPI_KEY", raising=False)
    result = cds.acquisition_readiness()
    assert result["ready"] and result["configured"] and result["client_available"]
    assert not result["authentication_checked"] and result["credential_source"] == "file"
    rendered = json.dumps(result)
    for secret in (token, str(path), "PRIVATE-LOCATION", "PRIVATE-ENDPOINT", "example.invalid"):
        assert secret not in rendered
    assert set(result) == {"schema", "ready", "configured", "client_available",
                           "authentication_checked", "credential_source", "message"}


def test_environment_readiness_returns_only_the_source_kind(tmp_path, monkeypatch, client_present):
    monkeypatch.setenv("CDSAPI_RC", str(tmp_path / "absent"))
    monkeypatch.setenv("CDSAPI_URL", "https://example.invalid/api")
    monkeypatch.setenv("CDSAPI_KEY", "ENV-SECRET")
    result = cds.acquisition_readiness()
    assert result["ready"] and result["credential_source"] == "environment"
    assert "ENV-SECRET" not in json.dumps(result)


@pytest.mark.parametrize("content", ["", "key: secret-only\n", "url: not-a-url\nkey: secret\n"])
def test_missing_or_unusable_profile_has_a_plain_remote_remedy(tmp_path, monkeypatch, client_present, content):
    path = tmp_path / "bad-profile"
    path.write_text(content)
    monkeypatch.setenv("CDSAPI_RC", str(path))
    monkeypatch.delenv("CDSAPI_URL", raising=False)
    monkeypatch.delenv("CDSAPI_KEY", raising=False)
    result = cds.acquisition_readiness()
    assert not result["ready"] and not result["configured"]
    assert "selected computer" in result["message"] and "CDSAPI_RC" in result["message"]
    assert "secret" not in json.dumps(result)


def test_missing_client_and_configuration_are_both_explained(monkeypatch):
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    monkeypatch.setattr(cds, "status", lambda: {"configured": False, "source": "missing"})
    result = cds.acquisition_readiness()
    assert not result["ready"] and not result["client_available"]
    assert "cdsapi>=0.7.7" in result["message"] and "~/.cdsapirc" in result["message"]


def test_raw_parser_exceptions_do_not_escape(monkeypatch, client_present):
    def bad_status():
        raise ValueError("SECRET-FROM-BROKEN-PARSER")
    monkeypatch.setattr(cds, "status", bad_status)
    result = cds.acquisition_readiness()
    assert not result["ready"] and "SECRET-FROM-BROKEN-PARSER" not in json.dumps(result)
