"""Local CDS credential status and explicit, private credential updates.

Status never returns a key. Updates accept secrets on stdin, never argv, and
do not contact CDS or start an acquisition.
"""
from __future__ import annotations

import csv
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from urllib.parse import urlsplit

SCHEMA = "arwen.cds-credentials.v1"
DEFAULT_URL = "https://cds.climate.copernicus.eu/api"
_LIMIT = 16384


def _path() -> Path:
    from gpuwm.fetch import cds_credentials_path
    return cds_credentials_path().absolute()


def _endpoint(value) -> str:
    if not isinstance(value, str):
        raise ValueError("Enter an HTTPS CDS API endpoint.")
    value = value.strip().rstrip("/")
    try:
        parts = urlsplit(value)
        valid = (parts.scheme == "https" and parts.hostname
                 and not parts.username and not parts.password
                 and not parts.query and not parts.fragment
                 and not any(character.isspace() for character in value))
    except ValueError:
        valid = False
    if not valid:
        raise ValueError("Enter an HTTPS CDS API endpoint without a key or password in its URL.")
    return value


def _profile(path: Path) -> dict:
    try:
        with path.open("r", encoding="utf-8") as stream:
            content = stream.read(_LIMIT + 1)
        if len(content) > _LIMIT:
            return {}
        # Match cdsapi.read_config: it reads plain colon-delimited lines,
        # not YAML scalars, so quotes would become part of the token/URL.
        profile = {}
        for line in content.splitlines():
            key, separator, value = line.strip().partition(":")
            if separator and key in ("url", "key", "verify"):
                profile[key] = value.strip()
        return profile
    except Exception:
        # Parser exceptions can contain the credential line itself.
        return {}


def status() -> dict:
    path = _path()
    profile = _profile(path)
    overrides = [name for name in ("CDSAPI_KEY", "CDSAPI_URL")
                 if os.environ.get(name) is not None]
    key = os.environ.get("CDSAPI_KEY", profile.get("key"))
    endpoint = os.environ.get("CDSAPI_URL", profile.get("url"))
    configured = isinstance(key, str) and bool(key.strip())
    try:
        url = _endpoint(endpoint)
    except ValueError:
        url = DEFAULT_URL
        configured = False
    return {"schema": SCHEMA, "configured": configured, "path": str(path),
            "source": "environment" if overrides else "file" if configured else "missing",
            "url": url, "editable": not overrides,
            "environment_overrides": overrides,
            "message": ("Configured locally; authentication is checked when downloading."
                        if configured else "No usable CDS credentials are configured.")}


def acquisition_readiness() -> dict:
    """Safe local readiness for a requested CDS acquisition, without contact.

    Call on the computer that will fetch the data. The returned contract
    contains neither credential values nor paths, endpoints, or raw errors;
    file/environment status remains available through the explicit settings
    command. Authentication and dataset access are checked only by CDS during
    a requested download. A verified cached acquisition needs no such gate.
    """
    from importlib.util import find_spec

    try:
        client_available = find_spec("cdsapi") is not None
    except Exception:
        client_available = False
    try:
        configured_status = status()
        configured = bool(configured_status["configured"])
        source = configured_status["source"]
        if source not in ("environment", "file", "missing"):
            source = "missing"
    except Exception:
        configured = False
        source = "missing"
    problems = []
    if not client_available:
        problems.append("Install cdsapi>=0.7.7 in the selected computer's ArWen Python environment.")
    if not configured:
        problems.append("Configure CDS credentials on the selected computer in ~/.cdsapirc, "
                        "or choose its private credential file with CDSAPI_RC. "
                        "Credentials saved only on the local computer do not configure an SSH node.")
    return {"schema": "arwen.cds-acquisition-readiness.v1",
            "ready": configured and client_available, "configured": configured,
            "client_available": client_available, "authentication_checked": False,
            "credential_source": source,
            "message": " ".join(problems) if problems else
                "CDS credentials and the client are configured on this computer; "
                "authentication and dataset access are checked when downloading."}


def _hidden_process_options() -> dict:
    return {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}


def _wsl_location(path: Path):
    text = str(path).replace("/", "\\")
    if text.lower().startswith("\\\\?\\unc\\"):
        text = "\\\\" + text[8:]
    parts = text.split("\\")
    if (len(parts) >= 5 and parts[:2] == ["", ""]
            and parts[2].lower() in ("wsl.localhost", "wsl$")):
        return parts[3], "/" + "/".join(parts[4:])
    return None


_WSL_SAVE = r'''
import json, os, pathlib, sys, tempfile
request = json.load(sys.stdin)
path = pathlib.Path(request["path"]).expanduser().resolve()
path.parent.mkdir(parents=True, exist_ok=True)
descriptor, temporary = tempfile.mkstemp(prefix=".cdsapirc-", dir=path.parent)
try:
    os.fchmod(descriptor, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
        stream.write(request["content"])
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
finally:
    if os.path.exists(temporary):
        os.unlink(temporary)
'''


def _private_windows_file(path: Path) -> None:
    result = subprocess.run(["whoami", "/user", "/fo", "csv", "/nh"],
        capture_output=True, text=True, check=False, **_hidden_process_options())
    try:
        row = next(csv.reader(io.StringIO(result.stdout.strip())))
        sid = row[1]
        if result.returncode or not sid.startswith("S-1-"):
            raise ValueError
    except (ValueError, IndexError, StopIteration):
        raise ValueError("Cannot determine the Windows account for private credential storage.") from None
    result = subprocess.run(["icacls", str(path), "/inheritance:r", "/grant:r", f"*{sid}:(F)"],
        capture_output=True, text=True, check=False, **_hidden_process_options())
    if result.returncode:
        raise ValueError("Cannot restrict the credential file to the current Windows account.")


def save(request: dict) -> dict:
    if not isinstance(request, dict):
        raise ValueError("Credential input must be an object.")
    current = status()
    if not current["editable"]:
        raise ValueError("CDS credentials are overridden by environment variables. Change those variables and reopen ArWen.")
    path = _path()
    prior = _profile(path)
    key = request.get("key", "")
    if not isinstance(key, str):
        raise ValueError("Enter a CDS personal access token.")
    key = key.strip() or prior.get("key", "")
    if (not isinstance(key, str) or not key or len(key) > 8192
            or any(character in key for character in ("\r", "\n", "\0"))):
        raise ValueError("Enter a CDS personal access token on one line.")
    url = _endpoint(request.get("url") or prior.get("url") or DEFAULT_URL)
    body = f"url: {url}\nkey: {key}\n"
    if "verify" in prior:
        body += f"verify: {prior['verify']}\n"
    location = _wsl_location(path) if os.name == "nt" else None
    if location is not None:
        distribution, linux_path = location
        result = subprocess.run(
            ["wsl.exe", "--distribution", distribution, "--exec", "python3", "-c", _WSL_SAVE],
            input=json.dumps({"path": linux_path, "content": body}),
            capture_output=True, text=True, check=False, **_hidden_process_options())
        if result.returncode:
            raise ValueError("Cannot save the CDS credential file in WSL. Check that the displayed distribution and home folder are available.")
    else:
        path = path.resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(prefix=".cdsapirc-", dir=path.parent)
        try:
            if os.name == "nt":
                _private_windows_file(Path(temporary))
            else:
                os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
                descriptor = None
                stream.write(body)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            if descriptor is not None:
                os.close(descriptor)
            if os.path.exists(temporary):
                os.unlink(temporary)
    result = status()
    result["message"] = "CDS credentials saved. Authentication will be checked when downloading."
    return result


def credentials_main(args) -> int:
    try:
        if args.save:
            raw = sys.stdin.read(_LIMIT + 1)
            if len(raw) > _LIMIT:
                raise ValueError("Credential input is too long.")
            try:
                request = json.loads(raw)
            except (ValueError, TypeError):
                raise ValueError("Credential input is not valid JSON.") from None
            result = save(request)
        else:
            result = status()
    except ValueError:
        raise
    except Exception:
        raise ValueError("Could not access the displayed CDS credential file. No credential values were logged.") from None
    if args.json:
        print(json.dumps(result))
    else:
        print(result["message"])
        print(f"Source: {result['source']}\nFile: {result['path']}\nEndpoint: {result['url']}")
    return 0


def register_cli(subparsers) -> None:
    parser = subparsers.add_parser("cds-credentials", help="show or update local CDS credentials without revealing the key")
    parser.add_argument("--json", action="store_true", help="return safe credential status as JSON")
    parser.add_argument("--save", action="store_true", help="read endpoint and key as JSON from stdin and save privately")
    parser.set_defaults(func=credentials_main)
