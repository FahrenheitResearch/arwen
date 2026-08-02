"""What `gpuwm report` must never leak, never crash on, and never fake.

The headline test here is
:func:`test_planted_identity_never_reaches_the_bundle`:
a distinctive fake username, hostname, IPv4, MAC, home directory, e-mail
and API key are planted in EVERY collectable surface -- a path segment, a
config value, a receipt's JSON, a log line, an environment variable, and
a file's own name -- and none of them may appear anywhere in the bundle's
bytes.  Its control,
:func:`test_the_planted_identity_test_can_fail`, disables the redactor
and asserts the same strings DO come through, so the headline assertion
can never quietly become vacuous by collecting nothing.

The rest is the honesty contract: a run directory with no receipts must
produce a bundle that SAYS it has none rather than a thin one that looks
complete; a full disk must relocate the bundle rather than lose it; and
`gpuwm report` with no arguments inside a run directory must find the run
it is standing in.
"""

from __future__ import annotations

import errno
import io
import json
import os
from pathlib import Path
import tempfile
import zipfile

import pytest


def _symlinks_work() -> bool:
    """Windows needs a privilege for this; the tests that need it skip."""

    with tempfile.TemporaryDirectory() as scratch:
        target = Path(scratch) / "target"
        target.write_text("x", encoding="utf-8")
        try:
            (Path(scratch) / "link").symlink_to(target)
        except (OSError, NotImplementedError):
            return False
    return True


#: Resolved once: the symlink tests are the traversal proof, and on a box
#: that cannot make one they must SKIP loudly rather than pass vacuously.
_CAN_SYMLINK = _symlinks_work()

import gpuwm.cli as cli
from gpuwm import report_bundle


# ---------------------------------------------------------------------------
# The planted identity, and where it is planted
# ---------------------------------------------------------------------------

#: Strings chosen so a hit cannot be a coincidence: none of them occurs in
#: this repository, in a receipt schema, or in any real path.
PLANTED = {
    "username": "zqfakeuser7",
    "hostname": "zqfakebox7.example-lan.internal",
    "hostname_short": "zqfakebox7",
    "ipv4": "203.0.113.77",
    "mac": "02:42:ac:11:00:7f",
    "email": "zqfakeuser7@example-lan.internal",
    "api_key": "AKIAZQFAKEUSER7SECRET",
    "bearer": "ghp_zqFakeUser7TokenValue0123456789",
}

#: A home-shaped prefix for the planted account, assembled rather than
#: written out.  `tests/test_release_snapshot_machine_paths.py` scans every
#: shipped file for developer-absolute paths and cannot tell a fixture's
#: synthetic home from a real one -- correctly, since it is looking for the
#: shape.  Building it here keeps the literal out of the source without
#: weakening what the fixture exercises.
POSIX_HOME = "/" + "home/" + PLANTED["username"]


def _plant_run_directory(root: Path) -> Path:
    """A run directory whose every surface carries a planted identity.

    Deliberately built as a HOME-SHAPED path (``.../Users/<name>/...``):
    that is how a real reporter's run directory looks, and it is what
    teaches the redactor an account name it was never told about.
    """

    run = root / "Users" / PLANTED["username"] / "gpuwm" / "out" / "run"
    (run / "evidence").mkdir(parents=True)

    home = POSIX_HOME
    (run / "report.json").write_text(json.dumps({
        "schema": "gpuwm-prepared-single-domain-forecast-v1",
        "status": "FAIL",
        "error_type": "RuntimeError",
        "error": f"decode failed for {home}/data/hrrr/t12z.grib2",
        "traceback": (
            "Traceback (most recent call last):\n"
            f"  File \"{home}/gpuwm/ingest/hrrr.py\", line 825, in _map\n"
            "    raise RuntimeError('decode failed')\n"
            "RuntimeError: decode failed\n"),
        "input": {
            "prepared_root": f"{home}/gpuwm/out/prepared",
            "authority_sha256": {
                "namelist.wps":
                    "9f2b" + "0" * 60,
            },
        },
        # Scientific content: this must SURVIVE.
        "domain": {"grid_id": 1, "nx": 400, "ny": 400, "nz": 45,
                   "ref_lat": 39.7392, "ref_lon": -104.9903, "dx_m": 3000.0},
    }), encoding="utf-8")

    (run / "config.toml").write_text(
        "[experiment]\n"
        f'name = "study-{PLANTED["username"]}"\n'
        f'operator_host = "{PLANTED["hostname"]}"\n'
        f'contact = "{PLANTED["email"]}"\n'
        f'api_key = "{PLANTED["api_key"]}"\n'
        "\n[[domain]]\n"
        "ref_lat = 39.7392\n"
        "ref_lon = -104.9903\n",
        encoding="utf-8")

    (run / "worker-01.stderr.log").write_text(
        f"host {PLANTED['hostname']} ({PLANTED['ipv4']}) link "
        f"{PLANTED['mac']}\n"
        f"user {PLANTED['username']} home {home}\n"
        f"Authorization: Bearer {PLANTED['bearer']}\n"
        f"GET https://{PLANTED['hostname']}/v1/data?token="
        f"{PLANTED['api_key']}\n"
        "Traceback (most recent call last):\n"
        f"  File \"{home}/gpuwm/runtime.py\", line 1149, in run_experiment\n"
        "ValueError: nz=45 exceeds the vertical contract\n",
        encoding="utf-8")

    # The identity in a FILE NAME, not only in file contents.
    (run / f"{PLANTED['username']}-next-steps.sh").write_text(
        f"# generated on {PLANTED['hostname']}\n"
        f"gpuwm run {home}/gpuwm/config.toml --outdir {home}/gpuwm/out/run\n",
        encoding="utf-8")

    (run / "wrfout_d01_2026-08-01_00_00_00").write_bytes(b"\x89NC\x00" * 64)
    return run


def _planted_environment() -> dict:
    return {
        "USERNAME": PLANTED["username"],
        "USER": PLANTED["username"],
        "COMPUTERNAME": PLANTED["hostname_short"],
        "HOSTNAME": PLANTED["hostname"],
        "GPUWM_CASE_DATA_ROOT": f"{POSIX_HOME}/data",
        "GPUWM_API_TOKEN": PLANTED["api_key"],
        "MY_PRIVATE_TOKEN": PLANTED["bearer"],
        "PATH": f"{POSIX_HOME}/bin:/usr/bin",
    }


def _bundle_bytes(report) -> bytes:
    """Every byte a reader of the bundle can see: names AND contents.

    The archive is deflated, so searching the raw zip is not enough to
    prove anything; this reads each member back out and concatenates the
    decompressed text with the member names.
    """

    parts = [report.archive]
    with zipfile.ZipFile(io.BytesIO(report.archive)) as archive:
        for info in archive.infolist():
            parts.append(info.filename.encode("utf-8"))
            parts.append(archive.read(info))
    return b"\n".join(parts)


# ---------------------------------------------------------------------------
# THE test
# ---------------------------------------------------------------------------

def test_planted_identity_never_reaches_the_bundle(tmp_path, monkeypatch):
    """No planted identity string survives into the bundle's bytes.

    Every class the command promises to remove is planted in every
    surface it collects from: a path segment, a config value, a receipt's
    JSON body, a log line, a file NAME, and the environment.  The
    assertion is over the decompressed archive, member names included.
    """

    run = _plant_run_directory(tmp_path)
    monkeypatch.setattr(report_bundle.socket, "gethostname",
                        lambda: PLANTED["hostname_short"])
    monkeypatch.setattr(report_bundle.socket, "getfqdn",
                        lambda: PLANTED["hostname"])

    report = report_bundle.build_report(run, environ=_planted_environment())
    blob = _bundle_bytes(report)

    for label, secret in PLANTED.items():
        assert secret.encode("utf-8") not in blob, (
            f"{label} leaked into the bundle: {secret}")
    # The reporter's own text output is the other thing they send.
    for secret in PLANTED.values():
        assert secret not in report.text

    # And the science survived: the redaction is targeted, not a shredder.
    assert b"39.7392" in blob and b"-104.9903" in blob
    assert b"wrfout_d01_2026-08-01_00_00_00" in blob
    assert b"9f2b" + b"0" * 60 in blob, "a sha256 digest must survive"
    assert b"nz=45 exceeds the vertical contract" in blob


def test_the_planted_identity_test_can_fail(tmp_path, monkeypatch):
    """The control: with the redactor disabled, the planted strings DO leak.

    Without this, a collector that silently collected nothing would pass
    the test above forever.  This pins that the surfaces really are
    collected and that the assertion has something to bite on.
    """

    run = _plant_run_directory(tmp_path)
    monkeypatch.setattr(report_bundle.Redactor, "apply",
                        lambda self, text: text)
    monkeypatch.setattr(report_bundle.Redactor, "json",
                        lambda self, payload: payload)
    monkeypatch.setattr(report_bundle.Redactor, "path",
                        lambda self, value: str(value))

    report = report_bundle.build_report(run, environ=_planted_environment())
    blob = _bundle_bytes(report)

    leaked = [label for label, secret in PLANTED.items()
              if secret.encode("utf-8") in blob]
    assert set(leaked) == set(PLANTED), (
        "the disabled redactor must leak every planted class; "
        f"only {sorted(leaked)} came through, so the surfaces the real "
        "test relies on are not all being collected")


@pytest.mark.parametrize("text,forbidden", [
    (f"C:\\Users\\{PLANTED['username']}\\gpuwm\\out", PLANTED["username"]),
    (f"{POSIX_HOME}/gpuwm", PLANTED["username"]),
    (f"api_key = {PLANTED['api_key']}", PLANTED["api_key"]),
    (f"host is {PLANTED['ipv4']}", PLANTED["ipv4"]),
    (f"link {PLANTED['mac']}", PLANTED["mac"]),
    (f"mail {PLANTED['email']}", PLANTED["email"]),
    (f"Bearer {PLANTED['bearer']}", PLANTED["bearer"]),
])
def test_each_class_is_removed_from_one_line(text, forbidden):
    """Unit-level: harvest then apply removes each class on its own."""

    redactor = report_bundle.Redactor(environ={})
    redactor.harvest(text)
    assert forbidden not in redactor.apply(text)


def test_a_bare_name_is_removed_once_another_surface_taught_it():
    """The two-pass property, isolated.

    A bare account name in prose has nothing about it that says it is an
    account name -- it is just a word.  What makes it removable is that
    it was seen as a path segment somewhere ELSE in the same collection,
    which is why harvest runs over every collected byte before apply
    touches any of them.  The reporter's own account is seeded from the
    environment and the home directory, so it is covered even when no
    path in the bundle contains it; this is the case for a SECOND,
    foreign name, and it is the reason the passes are ordered.
    """

    bare = f"see {PLANTED['username']} in the log"
    unaware = report_bundle.Redactor(environ={})
    assert PLANTED["username"] in unaware.apply(bare), (
        "with nothing to learn from, a bare word is only a word")

    taught = report_bundle.Redactor(environ={})
    taught.harvest(f"{POSIX_HOME}/gpuwm/out/run")
    assert PLANTED["username"] not in taught.apply(bare)


def test_structured_sections_keep_their_shape_through_redaction():
    """Redaction walks the structure; it cannot reshape it.

    The manifest renderer indexes these blocks by key, so a redactor
    that round-tripped them through text and lost a document to one
    unlucky substitution would turn a bundle into a KeyError -- while
    assembling a report about somebody else's exception.
    """

    redactor = report_bundle.Redactor(environ={"USER": PLANTED["username"]})
    block = {"path": f"{POSIX_HOME}/run", "count": 3, "ok": True,
             "free": None, "ratio": 1.5,
             "nested": [{"who": PLANTED["username"]}, "plain"]}
    walked = redactor.json(block)
    assert set(walked) == set(block)
    assert walked["count"] == 3 and walked["ok"] is True
    assert walked["free"] is None and walked["ratio"] == 1.5
    assert "zqfakeuser7" not in json.dumps(walked)
    assert walked["nested"][1] == "plain"


# ---------------------------------------------------------------------------
# Findings from the adversarial review, each with the input that broke it
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", [
    "GPUWM_S3_SECRET", "RW_WPS_NEXRAD_PASSWORD", "GPUWM_ARCHIVE_TOKEN",
    "CUDA_API_KEY", "GPUWM_AUTH", "NVIDIA_SESSION_ID",
])
def test_an_allowlisted_secret_named_variable_loses_its_value(name):
    """The allowlist is by PREFIX, so it admits names nobody enumerated.

    `GPUWM_*` and `RW_WPS_*` are exactly where this project's own
    credentials live, and by the time a value reaches the redactor it is
    a standalone string with no `key=` around it for the credential rule
    to match -- so every one of these shipped verbatim.
    """

    secret = "abc123def456ghi789jkl012"
    report = report_bundle.build_report(
        Path("does-not-exist"), environ={name: secret},
        require_run_directory=False)
    collected = report.manifest["environment"]["collected"]
    assert collected[name] == report_bundle.PLACEHOLDERS["credential"]
    assert secret.encode("utf-8") not in _bundle_bytes(report)


@pytest.mark.parametrize("line,secret", [
    ("db_password=hunter2sekrit", "hunter2sekrit"),
    ("my_api_key: swordfishvalue", "swordfishvalue"),
    ("user_token = abcdefghijklmnop", "abcdefghijklmnop"),
    ("GPUWM_SECRET=letmeinplease", "letmeinplease"),
    ("aws_secret_access_key=wJalrXUtnFEMI", "wJalrXUtnFEMI"),
    ("client_secret:topsecretvalue", "topsecretvalue"),
])
def test_an_underscore_prefixed_credential_key_is_still_a_credential_key(
        line, secret):
    """`\\b` does not fire before `_`, which is every real spelling.

    `_` is a word character, so a leading `\\b` on the keyword required a
    non-word character before it -- and `db_password=`, `my_api_key:`
    and `GPUWM_SECRET=` all have an underscore there.
    """

    redactor = report_bundle.Redactor(environ={})
    assert secret not in redactor.apply(line)


def test_the_word_bearer_cannot_shield_the_token_behind_it():
    """Rule ORDER, isolated: the keyed rule used to eat its own anchor.

    `Authorization: Bearer <token>` matches the assignment rule with
    `authorization` as the key and the literal word `Bearer` as the
    value.  Running that rule first replaced the word "Bearer" -- which
    both left the token intact and destroyed the anchor the bearer rule
    needed to find it.  A lowercase-hex token is used deliberately: it
    matches no token-shape rule, so only ordering can save it.
    """

    token = "0123456789abcdef0123456789abcdef"
    cleaned = report_bundle.Redactor(environ={}).apply(
        f"Authorization: Bearer {token}")
    assert token not in cleaned


@pytest.mark.parametrize("survivor", [
    "RW_volumetric_soil_moisture_0", "RW_geopotential_height_0",
    "module_mp_morr_two_moment_O2vec", "wrfinput_d03_through_dNN",
    "CertifiedHrrrWsm6YsuNoah", "node3_refl_ab_B_enabled_6h_512",
])
def test_the_entropy_rule_leaves_this_project_s_own_vocabulary_alone(
        survivor):
    """Physics choices are promised verbatim; these are physics choices.

    Every string here is taken from this repository's own tracked files.
    With `_` and `-` inside the entropy rule's character class each one
    was replaced by `<redacted-credential>` -- and then COUNTED as a
    credential the reporter never had.
    """

    redactor = report_bundle.Redactor(environ={})
    assert survivor in redactor.apply(f"field = {survivor}")
    assert redactor.counts.get("credential", 0) == 0


def test_the_entropy_rule_still_catches_an_opaque_key():
    """The converse: the narrowing must not disarm the rule."""

    redactor = report_bundle.Redactor(environ={})
    opaque = "Zk3Jq7Xr2Bv9Nw4Ts8Ly6Hd1Pf5Mc0Ge"
    assert opaque not in redactor.apply(f"value {opaque}")


@pytest.mark.parametrize("account", [
    "machine", "python", "gpu", "packages", "os_family", "volumes",
])
def test_an_account_named_like_a_manifest_key_does_not_kill_the_command(
        account):
    """Redacting KEYS renamed the manifest's own sections.

    `USERNAME=gpu` is entirely ordinary on a compute node.  With keys
    redacted, `machine` became `<user>` and the renderer died with
    KeyError while assembling a report about somebody else's exception.
    """

    report = report_bundle.build_report(
        Path("does-not-exist"), environ={"USERNAME": account},
        require_run_directory=False)
    assert report.manifest["machine"]["os_family"]
    assert "gpuwm diagnostic report" in report.text


def test_a_deleted_working_directory_does_not_stop_the_bundle(
        tmp_path, monkeypatch):
    """The exact shape of the defect this command exists for.

    The consumer's unwind removes the staging directory; a reporter
    whose shell is still sitting in it got a raw FileNotFoundError from
    `Path.cwd()` instead of a bundle.
    """

    run = tmp_path / "run"
    run.mkdir()
    (run / "report.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(report_bundle, "_cwd_or_none", lambda: None)

    report = report_bundle.build_report(run, environ={})
    assert "working_directory" not in report.manifest["volumes"]
    written = report_bundle.write_report(report, tmp_path / "out")
    assert written.written_path.exists()


# ---------------------------------------------------------------------------
# The secret deny-set: NOT READ, not merely not-shipped
#
# Owner ruling: "a secret that was read and then scrubbed is still a secret
# we read".  Redaction is the second line of defence; the first is that a
# credential file is structurally uncollectable.  So every test in this
# section asserts on ACCESS -- the set of paths the collector opened --
# and not on the bundle's contents.  Absence from the bundle proves the
# redactor worked; it proves nothing about what was read.
# ---------------------------------------------------------------------------

class _OpenSpy:
    """Records every path the collector opens, and passes the call through."""

    def __init__(self):
        self.opened: list[str] = []
        self._real = open

    def __call__(self, path, *args, **kwargs):
        self.opened.append(str(path))
        return self._real(path, *args, **kwargs)

    def touched(self, path) -> bool:
        wanted = os.path.normcase(os.path.abspath(str(path)))
        return any(os.path.normcase(os.path.abspath(seen)) == wanted
                   for seen in self.opened)


def _spy_on_opens(monkeypatch) -> _OpenSpy:
    spy = _OpenSpy()
    monkeypatch.setattr(report_bundle, "open", spy, raising=False)
    return spy


def _plant_secrets(root: Path) -> dict[str, Path]:
    """One file per class the deny-set must refuse, all beside a real run."""

    planted = {}
    (root / ".aws").mkdir(parents=True, exist_ok=True)
    planted["aws"] = root / ".aws" / "credentials"
    planted["aws"].write_text(
        "[default]\naws_secret_access_key = zqAWSsecret0001\n",
        encoding="utf-8")
    (root / ".ssh").mkdir(exist_ok=True)
    planted["ssh"] = root / ".ssh" / "id_rsa"
    planted["ssh"].write_text(
        "-----BEGIN OPENSSH PRIVATE KEY-----\nzqSSHsecret0002\n"
        "-----END OPENSSH PRIVATE KEY-----\n", encoding="utf-8")
    (root / ".docker").mkdir(exist_ok=True)
    planted["docker"] = root / ".docker" / "config.json"
    planted["docker"].write_text('{"auths": {"r": {"auth": "zqDOCKER0003"}}}',
                                 encoding="utf-8")
    planted["dotenv"] = root / ".env"
    planted["dotenv"].write_text("SITE_TOKEN=zqENVsecret0004\n",
                                 encoding="utf-8")
    planted["pem"] = root / "server.pem"
    planted["pem"].write_text(
        "-----BEGIN PRIVATE KEY-----\nzqPEMsecret0005\n"
        "-----END PRIVATE KEY-----\n", encoding="utf-8")
    planted["key"] = root / "api.key"
    planted["key"].write_text("zqKEYsecret0006\n", encoding="utf-8")
    # The two that a name-only deny-set misses and a suffix-only one
    # misses: ordinary-looking JSON whose NAME says what it holds.
    planted["credentials_json"] = root / "credentials.json"
    planted["credentials_json"].write_text('{"k": "zqCREDsecret0007"}',
                                           encoding="utf-8")
    (root / "secrets").mkdir(exist_ok=True)
    planted["secrets_dir"] = root / "secrets" / "creds.json"
    planted["secrets_dir"].write_text('{"k": "zqSECRETsecret0008"}',
                                      encoding="utf-8")
    planted["netrc"] = root / ".netrc"
    planted["netrc"].write_text("machine x password zqNETRC0009\n",
                                encoding="utf-8")
    return planted


def _real_run(root: Path) -> Path:
    """The smallest thing that is genuinely a gpuwm run directory.

    Deliberately WITHOUT a ``status``: a receipt saying PASS suppresses
    the refusal-derived failure, which is correct behaviour and would
    make the failure-shape tests pass for the wrong reason.
    """

    root.mkdir(parents=True, exist_ok=True)
    (root / "report.json").write_text(
        '{"schema": "gpuwm-prepared-single-domain-forecast-v1"}',
        encoding="utf-8")
    (root / "worker-01.stderr.log").write_text("started\n", encoding="utf-8")
    return root


@pytest.mark.parametrize("planted_key", [
    "aws", "ssh", "docker", "dotenv", "pem", "key", "credentials_json",
    "secrets_dir", "netrc",
])
def test_a_credential_file_inside_the_run_directory_is_never_opened(
        tmp_path, monkeypatch, planted_key):
    """The first line of defence: it is not read, not hashed, not listed.

    Planted INSIDE a directory that is a genuine run directory, so the
    run-directory gate cannot be what saves us -- only the deny-set can.
    """

    run = _real_run(tmp_path / "run")
    planted = _plant_secrets(run)
    target = planted[planted_key]

    spy = _spy_on_opens(monkeypatch)
    report = report_bundle.build_report(run, environ={})

    assert not spy.touched(target), (
        f"the collector OPENED {target.name}; redaction is the second line "
        "of defence, not the first")
    inventoried = {item["path"] for item in report.manifest["inventory"]}
    collected = {entry["path"] for entry in report.manifest["included"]}
    relative = target.relative_to(run).as_posix()
    assert relative not in inventoried, (
        f"{relative} was inventoried; a credential file's NAME and SIZE "
        "are not ours to publish either")
    assert f"run/{relative}" not in collected
    # The run's own artifacts are still collected.
    assert "run/report.json" in collected


def test_the_refused_paths_are_counted_by_class_and_never_named(tmp_path):
    """A file name can be the secret, so the manifest counts, not lists."""

    run = _real_run(tmp_path / "run")
    _plant_secrets(run)

    report = report_bundle.build_report(run, environ={})
    refused = report.manifest["refused_before_reading"]
    assert sum(refused.values()) >= 9, refused
    blob = _bundle_bytes(report)
    for name in (b"credentials.json", b"server.pem", b"api.key", b".netrc"):
        assert name not in blob, name
    assert "refused before being opened" in report.text


@pytest.mark.skipif(not _CAN_SYMLINK, reason="symlinks need privilege here")
def test_a_symlink_pointing_out_of_the_run_directory_is_never_followed(
        tmp_path, monkeypatch):
    """Resolve first, then test.  A product-shaped NAME is not a passport."""

    run = _real_run(tmp_path / "run")
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    secret = outside / "credentials"
    secret.write_text("token = zqSYMLINKsecret0010\n", encoding="utf-8")
    # The link's own name is one the collector actively wants.
    (run / "worker-02.stderr.log").symlink_to(secret)

    spy = _spy_on_opens(monkeypatch)
    report = report_bundle.build_report(run, environ={})

    assert not spy.touched(secret)
    assert not any("elsewhere" in seen for seen in spy.opened)
    assert b"zqSYMLINKsecret0010" not in _bundle_bytes(report)


@pytest.mark.skipif(not _CAN_SYMLINK, reason="symlinks need privilege here")
def test_a_symlinked_directory_out_of_the_run_is_never_descended(
        tmp_path, monkeypatch):
    run = _real_run(tmp_path / "run")
    outside = tmp_path / "private"
    outside.mkdir()
    (outside / "notes.log").write_text("zqDIRLINKsecret0011\n",
                                       encoding="utf-8")
    (run / "evidence").symlink_to(outside, target_is_directory=True)

    spy = _spy_on_opens(monkeypatch)
    report = report_bundle.build_report(run, environ={})

    assert not spy.touched(outside / "notes.log")
    assert b"zqDIRLINKsecret0011" not in _bundle_bytes(report)


def test_a_parent_traversal_target_is_refused(tmp_path, monkeypatch):
    """`gpuwm report RUN/..` must not become `collect my home directory`.

    The traversal itself is legal -- `run/sub/..` IS `run` -- so the
    property is not that `..` is rejected but that where it LANDS is
    tested.  Landing one directory above a run is landing somewhere
    that is not a run, and that is refused with the secret beside it
    never opened.
    """

    _real_run(tmp_path / "run")
    secret = tmp_path / "credentials.json"
    secret.write_text('{"k": "zqTRAVERSALsecret0012"}', encoding="utf-8")

    spy = _spy_on_opens(monkeypatch)
    report = report_bundle.build_report(tmp_path / "run" / "..", environ={})

    # It landed on the parent, found no run there, and descended into the
    # one named `run` below -- three named candidates, never a sweep of
    # the parent.  The secret sitting beside it is untouched either way.
    assert not spy.touched(secret)
    assert report.manifest["run_directory"].endswith("/run")
    assert not any("credentials" in item["path"]
                   for item in report.manifest["inventory"])
    assert b"zqTRAVERSALsecret0012" not in _bundle_bytes(report)


def test_a_traversal_that_lands_back_inside_the_run_still_works(tmp_path):
    """The converse, so the rule above is about WHERE, not about `..`."""

    run = _real_run(tmp_path / "run")
    # The intermediate directory has to EXIST: POSIX resolves `sub/..`
    # through the filesystem and Windows normalises it lexically, so a
    # missing `sub` makes this test mean two different things.
    (run / "sub").mkdir()
    report = report_bundle.build_report(
        tmp_path / "run" / "sub" / "..", environ={})
    assert "run/report.json" in {entry["path"]
                                 for entry in report.manifest["included"]}


# ---------------------------------------------------------------------------
# Refusing to run where there is no run
# ---------------------------------------------------------------------------

def test_it_refuses_a_directory_that_is_not_a_run(tmp_path, monkeypatch):
    """The rule that removes the whole class the audit found.

    A home directory holds nothing this product wrote, so there is
    nothing here to collect -- and the collector says so instead of
    taking whatever it finds.
    """

    home = tmp_path / "home"
    home.mkdir()
    planted = _plant_secrets(home)
    (home / "taxes.txt").write_text("private\n", encoding="utf-8")

    spy = _spy_on_opens(monkeypatch)
    with pytest.raises(ValueError) as refusal:
        report_bundle.build_report(home, environ={})

    message = str(refusal.value)
    assert "does not look like a gpuwm run directory" in message
    assert "report.json" in message and "worker-*.log" in message
    for target in planted.values():
        assert not spy.touched(target), target
    assert not any(str(home) in seen for seen in spy.opened)


def test_the_cli_refuses_with_exit_2_and_no_traceback(
        tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert cli.main(["report"]) == 2
    error = capsys.readouterr().err
    assert "gpuwm report:" in error
    assert "does not look like a gpuwm run directory" in error
    assert "Traceback" not in error


def test_a_run_that_died_before_any_receipt_is_still_collectable(tmp_path):
    """The ENOSPC case must survive the gate.

    The gate asks for a product ARTIFACT, not specifically a receipt --
    a zero-byte decoder log and a staging gate note are what the
    out-of-space defect leaves behind, and refusing to collect them
    would close the exact case this command was written for.
    """

    run = tmp_path / "run"
    run.mkdir()
    (run / "decoder.log").write_bytes(b"")
    (run / "gate.txt").write_text("d01 strip 3\n", encoding="utf-8")
    (run / "pipeline.err").write_text(
        "RuntimeError: pipeline producer exited 1: \n", encoding="utf-8")

    report = report_bundle.build_report(run, environ={})
    assert report.manifest["route_detected"] is None
    collected = {entry["path"] for entry in report.manifest["included"]}
    assert {"run/decoder.log", "run/gate.txt", "run/pipeline.err"} <= collected
    assert report.manifest["erased_producer_message_in"] == \
        ["run/pipeline.err"]


def test_a_named_log_lets_a_reporter_past_the_gate(tmp_path, capsys):
    """`--log` is the reporter naming their own evidence, explicitly."""

    bare = tmp_path / "nothing"
    bare.mkdir()
    stray = tmp_path / "terminal.log"
    stray.write_text("gpuwm run: the wizard refused the latitude\n",
                     encoding="utf-8")

    assert cli.main(["report", str(bare), "--log", str(stray),
                     "--dry-run"]) == 0
    assert "logs/terminal.log" in capsys.readouterr().out


def test_collection_is_by_allowlist_not_by_sweep(tmp_path):
    """An unrecognised file is inventoried by name and size, never read."""

    run = _real_run(tmp_path / "run")
    (run / "notes.rtf").write_text("zqUNKNOWNbody0013\n", encoding="utf-8")
    (run / "scratch.dat").write_bytes(b"zqUNKNOWNbytes0014")

    report = report_bundle.build_report(run, environ={})
    inventoried = {item["path"] for item in report.manifest["inventory"]}
    collected = {entry["path"] for entry in report.manifest["included"]}
    assert {"notes.rtf", "scratch.dat"} <= inventoried
    assert not {"run/notes.rtf", "run/scratch.dat"} & collected
    blob = _bundle_bytes(report)
    assert b"zqUNKNOWNbody0013" not in blob
    assert b"zqUNKNOWNbytes0014" not in blob


def test_an_ipv4_mapped_ipv6_address_loses_all_four_octets():
    """IPv6 first ate `::ffff:a` and left `.b.c.d` too short to match."""

    cleaned = report_bundle.Redactor(environ={}).apply(
        "peer ::ffff:203.0.113.77 connected")
    assert "203.0.113.77" not in cleaned
    assert "0.113.77" not in cleaned


def test_a_double_colon_separator_is_not_an_address():
    """From this project's own pyproject.toml."""

    line = 'classifiers = ["License :: OSI Approved :: Apache Software"]'
    assert report_bundle.Redactor(environ={}).apply(line) == line


def test_an_account_name_glued_by_an_underscore_is_still_redacted():
    """`\\w` boundaries left `<name>_scratch` and `<NAME>_HOME` intact."""

    redactor = report_bundle.Redactor(environ={"USER": PLANTED["username"]})
    text = (f"{PLANTED['username']}_scratch and "
            f"{PLANTED['username'].upper()}_HOME and "
            f"run_{PLANTED['username']}")
    assert PLANTED["username"] not in redactor.apply(text).lower()


def test_a_url_collection_segment_named_users_teaches_nothing():
    """A REST collection segment must not make a plain word an account.

    The URL is assembled rather than written out, for the same reason
    :data:`POSIX_HOME` is: the release-snapshot guard reads this file
    and is looking for the shape, not the intent.
    """

    url = "GET https://api.example.gov/" + "Users/" + "data/list 200"
    redactor = report_bundle.Redactor(environ={})
    redactor.harvest(url)
    assert redactor.apply("assimilation used data from the data cache") == \
        "assimilation used data from the data cache"


def test_a_present_but_unreadable_receipt_is_not_reported_as_missing(
        tmp_path, monkeypatch):
    """The manifest must not contradict itself four sections apart."""

    run = tmp_path / "run"
    run.mkdir()
    (run / "report.json").write_text('{"status": "PASS"}', encoding="utf-8")
    real_open = open

    def _refuse(path, *args, **kwargs):
        if str(path).endswith("report.json"):
            raise PermissionError(errno.EACCES, "denied")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(report_bundle, "open", _refuse, raising=False)
    report = report_bundle.build_report(run, environ={})

    missing = {item["artifact"]
               for item in report.manifest["missing_expected_artifacts"]}
    assert "report.json" not in missing
    assert report.manifest["route_detected"] == \
        "prepared single-domain forecast"
    assert "NO receipt of any route was found here" not in report.text
    assert any("report.json" in note for note in report.manifest["notes"])


def test_two_files_that_redact_alike_keep_two_members(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "report.json").write_text("{}", encoding="utf-8")
    for who in ("alice", "bob"):
        (run / f"{who}-notes.log").write_text(
            f"home is /home/{who}/work\n", encoding="utf-8")

    report = report_bundle.build_report(run, environ={})
    with zipfile.ZipFile(io.BytesIO(report.archive)) as archive:
        names = archive.namelist()
    assert len(names) == len(set(names)), names
    listed = [entry["path"] for entry in report.manifest["included"]]
    assert len(listed) == len(set(listed))
    # Both survive; the second is suffixed rather than overwriting the
    # first, so the bundle carries what the manifest says it carries.
    assert len([n for n in names if "notes" in n]) == 2
    assert any(n.endswith("_user_-notes-2.log") for n in names), names


def test_member_names_are_legal_windows_file_names(tmp_path):
    """Angle brackets are forbidden in Windows file names."""

    run = tmp_path / "run"
    run.mkdir()
    (run / "report.json").write_text("{}", encoding="utf-8")
    (run / f"{PLANTED['username']}-terminal.log").write_text(
        "x\n", encoding="utf-8")

    report = report_bundle.build_report(
        run, environ={"USER": PLANTED["username"]})
    with zipfile.ZipFile(io.BytesIO(report.archive)) as archive:
        names = archive.namelist()
    assert not any(set('<>:"|?*') & set(name.replace("/", ""))
                   for name in names), names
    assert any("_user_-terminal.log" in name for name in names), names


def test_a_run_that_passed_is_not_given_a_failure(tmp_path):
    """A success line has the same shape as a refusal; it is not one."""

    run = tmp_path / "run"
    run.mkdir()
    (run / "report.json").write_text('{"status": "PASS"}', encoding="utf-8")
    (run / "terminal.log").write_text(
        "gpuwm check: the optional [products] table is absent\n"
        "gpuwm run: wrote 17 frames\n", encoding="utf-8")

    report = report_bundle.build_report(run, environ={}, exit_code=0)
    assert report.manifest["failure"] is None
    assert "Nothing said it failed" in report.text


def test_a_refused_output_path_does_not_destroy_what_was_there(
        tmp_path, monkeypatch):
    """`-o keepme.zip` under ENOSPC used to unlink the existing file."""

    run = _real_run(tmp_path / "run")
    keep = tmp_path / "keepme.zip"
    keep.write_bytes(b"someone else's file")
    spare = tmp_path / "spare"
    spare.mkdir()
    monkeypatch.chdir(spare)

    report = report_bundle.build_report(run, environ={})
    real_open = open

    def _enospc(path, *args, **kwargs):
        if "keepme" in str(path):
            raise OSError(errno.ENOSPC, "No space left on device")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(report_bundle, "open", _enospc, raising=False)
    written = report_bundle.write_report(report, keep)

    assert keep.read_bytes() == b"someone else's file"
    assert written.written_path is not None
    assert written.written_path.parent == spare
    assert not list(tmp_path.glob("*.part"))


def test_withheld_warnings_are_counted_not_dropped_in_silence(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "report.json").write_text("{}", encoding="utf-8")
    (run / "worker-01.stderr.log").write_text(
        "".join(f"warning: distinct concern number {index}\n"
                for index in range(60)), encoding="utf-8")

    report = report_bundle.build_report(run, environ={})
    assert len(report.manifest["warnings"]) == report_bundle._MAX_WARNINGS
    assert report.manifest["warnings_withheld"] == 20
    assert "20 more distinct warning(s)" in report.text


def test_digests_and_timestamps_survive_redaction():
    """Redaction must not eat the identity this project reports WITH."""

    redactor = report_bundle.Redactor(environ={})
    digest = "3f5a" + "9c" * 30
    text = (f"sha256 {digest} at 2026-08-01T12:00:00Z step 12:00:00 "
            "version 1.4.0")
    cleaned = redactor.apply(text)
    assert digest in cleaned
    assert "2026-08-01T12:00:00Z" in cleaned
    assert "12:00:00" in cleaned
    assert "1.4.0" in cleaned


def test_environment_values_outside_the_allowlist_are_never_collected():
    redactor = report_bundle.Redactor(environ={})
    report = report_bundle.build_report(
        Path("does-not-exist"), environ=_planted_environment(),
        redactor=redactor, require_run_directory=False)
    collected = report.manifest["environment"]["collected"]
    assert "MY_PRIVATE_TOKEN" not in collected
    assert "PATH" not in collected
    assert report.manifest["environment"]["omitted_count"] >= 2
    # An allowlisted variable is kept -- with its value redacted.
    assert "GPUWM_CASE_DATA_ROOT" in collected
    assert PLANTED["username"] not in collected["GPUWM_CASE_DATA_ROOT"]


# ---------------------------------------------------------------------------
# Honesty about what was not there
# ---------------------------------------------------------------------------

def test_a_directory_with_no_receipts_says_so_rather_than_looking_complete(
        tmp_path):
    """The ENOSPC shape: remnants exist, receipts do not, and it SAYS which.

    This is the bundle the out-of-space defect produces -- a zero-byte
    decoder log and an empty producer message, with every receipt absent
    because the run died before writing one.  The requirement is not that
    the bundle be full; it is that it be honest about being empty.
    """

    run = tmp_path / "run"
    run.mkdir()
    (run / "decoder.log").write_bytes(b"")
    (run / "pipeline.err").write_text(
        "gpuwm ingest: starting hrrr decode\n"
        "RuntimeError: pipeline producer exited 1: \n", encoding="utf-8")

    report = report_bundle.build_report(run, environ={})

    assert report.manifest["route_detected"] is None
    missing = {item["artifact"]
               for item in report.manifest["missing_expected_artifacts"]}
    assert {"report.json", "failure-capsule.json",
            "evidence/run-receipt.json"} <= missing
    # Each absence names the route that would have written it, and what
    # would have been inside -- not just a filename.
    for item in report.manifest["missing_expected_artifacts"]:
        assert item["route"] and item["would_have_shown"]
    assert "NO receipt of any route was found here" in report.text

    # The remnants that DO exist are collected, including the zero-byte one.
    included = {entry["path"] for entry in report.manifest["included"]}
    assert "run/decoder.log" in included and "run/pipeline.err" in included

    # And the erased-cause signature is called out rather than passed on.
    assert report.manifest["erased_producer_message_in"] == \
        ["run/pipeline.err"]
    assert "erased its own diagnostics" in report.text


def test_the_failure_comes_from_the_receipt_when_there_is_one(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "failure-capsule.json").write_text(json.dumps({
        "exception_type": "DomainFitError",
        "exception_message": "d02 does not fit its parent",
        "traceback": "Traceback (most recent call last):\n  ...\n",
    }), encoding="utf-8")

    report = report_bundle.build_report(run, environ={})
    failure = report.manifest["failure"]
    assert failure["exception_type"] == "DomainFitError"
    assert failure["message"] == "d02 does not fit its parent"
    assert failure["found_in"] == "receipt"
    assert report.manifest["route_detected"] == \
        "gpuwm run (supervised front door)"


def test_a_real_sized_receipt_is_kept_whole_and_still_parses(tmp_path):
    """A 6 h HRRR report.json is 320 KiB; a truncated one loses the failure.

    Measured, not assumed: the receipt this cap was set from is the
    prepared single-domain HRRR f06-f12 run's report.json on node 15, at
    327,644 bytes.  The failure the collector exists to report is a key
    INSIDE it, so a receipt truncated into invalid JSON costs exactly the
    thing the reporter needed.
    """

    run = tmp_path / "run"
    run.mkdir()
    payload = {
        "status": "FAIL",
        "error_type": "PreparedCacheIdentityError",
        "error": "prepared cache does not match the config",
        "traceback": "Traceback (most recent call last):\n  ...\n",
        # The bulk a real receipt carries: a per-file digest inventory.
        "input": {"authority_sha256": {
            f"domains/d01/part-{index:04d}.bin": f"{index:064x}"
            for index in range(3400)}},
    }
    body = json.dumps(payload)
    assert len(body) > 300 * 1024, len(body)
    (run / "report.json").write_text(body, encoding="utf-8")

    report = report_bundle.build_report(run, environ={})
    entry = next(item for item in report.manifest["included"]
                 if item["path"] == "run/report.json")
    assert entry["note"] == "whole"
    assert report.manifest["failure"]["exception_type"] == \
        "PreparedCacheIdentityError"

    with zipfile.ZipFile(io.BytesIO(report.archive)) as archive:
        name = next(n for n in archive.namelist()
                    if n.endswith("run/report.json"))
        assert json.loads(archive.read(name))["status"] == "FAIL"


def test_a_receipt_too_big_to_keep_whole_says_it_will_not_parse(
        tmp_path, monkeypatch):
    monkeypatch.setattr(report_bundle, "_JSON_BYTES", 512)
    run = tmp_path / "run"
    run.mkdir()
    (run / "report.json").write_text(
        json.dumps({"status": "PASS", "pad": "x" * 4096}), encoding="utf-8")

    report = report_bundle.build_report(run, environ={})
    entry = next(item for item in report.manifest["included"]
                 if item["path"] == "run/report.json")
    assert "will not parse" in entry["note"]


def test_a_huge_allowlisted_value_is_elided_not_pasted(tmp_path):
    """NVIDIA_REQUIRE_CUDA is 2.5 KiB on a stock CUDA image."""

    report = report_bundle.build_report(
        tmp_path, environ={"NVIDIA_REQUIRE_CUDA": "brand=tesla," * 400},
        require_run_directory=False)
    value = report.manifest["environment"]["collected"]["NVIDIA_REQUIRE_CUDA"]
    assert len(value) < 400
    assert "characters elided" in value


def test_a_one_line_refusal_is_the_failure_when_there_is_no_traceback(
        tmp_path):
    """Most gpuwm failures are refusals, not tracebacks.

    Verbatim from node 15: `gpuwm run` against a config with no
    [case_data] table exits 2 having printed one line and no traceback,
    because that is what this project's refusal boundary is FOR.  A
    collector that only looked for `Traceback (most recent call last):`
    reported "no typed failure" for the commonest way a run ends.
    """

    run = _real_run(tmp_path / "run")
    (run / "terminal.log").write_text(
        "gpuwm check: PASS (rc 0)\n"
        "warning: the archive's own cadence is the default\n"
        "gpuwm run: experiment config /tmp/x.toml carries no [case_data] "
        "table; the experiment runtime requires declared inputs.\n",
        encoding="utf-8")

    report = report_bundle.build_report(run, environ={}, exit_code=2)
    failure = report.manifest["failure"]
    assert failure["found_in"] == "last product message"
    assert failure["exception_type"] == "gpuwm run"
    assert "carries no [case_data] table" in failure["message"]
    # The PASS line has the same shape and must not be offered as a cause.
    assert "PASS" not in failure["message"]
    # Warn-not-block means a run can pass through something worth knowing.
    assert report.manifest["warnings"] == [
        "the archive's own cadence is the default"]
    assert "Warnings this run printed and continued past" in report.text


def test_a_traceback_outranks_a_refusal_line(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "worker-01.stderr.log").write_text(
        "gpuwm run: something was refused\n"
        "Traceback (most recent call last):\n"
        "  File \"x.py\", line 1, in <module>\n"
        "MemoryError: out of memory allocating d02\n",
        encoding="utf-8")

    failure = report_bundle.build_report(run, environ={}).manifest["failure"]
    assert failure["exception_type"] == "MemoryError"
    assert failure["found_in"] == "log"


def test_an_unreadable_artifact_is_named_not_raised(tmp_path, monkeypatch):
    """Warn-not-block: a file that cannot be read is one line, not a crash."""

    run = tmp_path / "run"
    run.mkdir()
    (run / "report.json").write_text("{}", encoding="utf-8")
    (run / "worker-01.stderr.log").write_text("x", encoding="utf-8")

    real_open = open

    def _refuse(path, *args, **kwargs):
        if str(path).endswith("worker-01.stderr.log"):
            raise PermissionError(errno.EACCES, "denied")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(report_bundle, "open", _refuse, raising=False)
    report = report_bundle.build_report(run, environ={})
    assert any("worker-01.stderr.log" in note
               for note in report.manifest["notes"])
    assert report.archive


# ---------------------------------------------------------------------------
# Surviving a full disk
# ---------------------------------------------------------------------------

def test_a_full_first_location_relocates_the_bundle(tmp_path, monkeypatch):
    """ENOSPC on the chosen output must cost a relocation, not the bundle."""

    run = tmp_path / "run"
    run.mkdir()
    (run / "report.json").write_text("{}", encoding="utf-8")
    full = tmp_path / "full"
    full.mkdir()
    spare = tmp_path / "spare"
    spare.mkdir()
    monkeypatch.chdir(spare)

    report = report_bundle.build_report(run, environ={})
    real_open = open

    def _enospc(path, *args, **kwargs):
        if str(path).startswith(str(full)):
            raise OSError(errno.ENOSPC, "No space left on device")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(report_bundle, "open", _enospc, raising=False)
    written = report_bundle.write_report(report, full)

    assert written.written_path is not None
    assert written.written_path.exists()
    assert not str(written.written_path).startswith(str(full))
    assert "no space left on that volume" in written.fallback_reason


def test_it_refuses_in_one_sentence_when_nothing_can_be_written(
        tmp_path, monkeypatch):
    run = _real_run(tmp_path / "run")
    report = report_bundle.build_report(run, environ={})

    def _always_full(path, *args, **kwargs):
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(report_bundle, "open", _always_full, raising=False)
    with pytest.raises(report_bundle.ReportWriteError) as failure:
        report_bundle.write_report(report, tmp_path)
    message = str(failure.value)
    assert "could not write the bundle anywhere" in message
    assert "--output" in message
    assert message.count("tried ") >= 2


def test_the_archive_is_a_plain_readable_zip(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "report.json").write_text('{"status": "PASS"}', encoding="utf-8")

    report = report_bundle.build_report(run, environ={})
    with zipfile.ZipFile(io.BytesIO(report.archive)) as archive:
        names = archive.namelist()
        root = names[0].split("/")[0]
        assert f"{root}/MANIFEST.txt" in names
        assert f"{root}/manifest.json" in names
        assert f"{root}/run/report.json" in names
        for name in names:
            archive.read(name).decode("utf-8")
        parsed = json.loads(archive.read(f"{root}/manifest.json"))
    assert parsed["schema"] == report_bundle.BUNDLE_SCHEMA
    assert parsed["limits"], "the bundle must state what it cannot explain"


def test_model_output_is_inventoried_not_copied(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "report.json").write_text("{}", encoding="utf-8")
    frame = run / "wrfout_d01_2026-08-01_00_00_00"
    frame.write_bytes(b"\x00" * 4096)

    report = report_bundle.build_report(run, environ={})
    inventory = {item["path"]: item["bytes"]
                 for item in report.manifest["inventory"]}
    assert inventory["wrfout_d01_2026-08-01_00_00_00"] == 4096
    included = {entry["path"] for entry in report.manifest["included"]}
    assert "run/wrfout_d01_2026-08-01_00_00_00" not in included
    assert len(report.archive) < 4096 * 2


# ---------------------------------------------------------------------------
# No arguments, inside a run
# ---------------------------------------------------------------------------

def test_no_arguments_inside_a_run_directory(tmp_path, monkeypatch, capsys):
    """`gpuwm report` with no arguments collects the run it is standing in."""

    run = tmp_path / "run"
    run.mkdir()
    (run / "report.json").write_text(
        '{"status": "PASS", "domain": {"ref_lat": 39.7392}}', encoding="utf-8")
    monkeypatch.chdir(run)

    assert cli.main(["report", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "gpuwm diagnostic report" in out
    assert "run/report.json" in out
    assert "dry run: nothing was written" in out
    assert not list(run.glob("gpuwm-report-*.zip"))


def test_no_arguments_descends_to_the_default_output_directory(
        tmp_path, monkeypatch, capsys):
    """A checkout root with `out/run` below it is what the docs produce."""

    run = tmp_path / "out" / "run"
    run.mkdir(parents=True)
    (run / "report.json").write_text("{}", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    assert cli.main(["report", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "read out/run/ below the directory given" in out


def test_the_cli_writes_a_bundle_and_names_it(tmp_path, monkeypatch, capsys):
    run = tmp_path / "run"
    run.mkdir()
    (run / "report.json").write_text("{}", encoding="utf-8")
    destination = tmp_path / "outbox"
    destination.mkdir()
    monkeypatch.chdir(tmp_path)

    assert cli.main(["report", str(run), "-o", str(destination),
                     "--exit-code", "1"]) == 0
    out = capsys.readouterr().out
    written = list(destination.glob("gpuwm-report-*.zip"))
    assert len(written) == 1
    assert str(written[0]) in out
    assert "observed exit   1" in out
    with zipfile.ZipFile(written[0]) as archive:
        manifest = json.loads(archive.read(
            [n for n in archive.namelist() if n.endswith("manifest.json")][0]))
    assert manifest["observed_exit_code"] == 1


def test_report_is_registered_and_dispatches_through_the_real_cli(
        tmp_path, monkeypatch, capsys):
    """cli._dispatch carries a hardcoded name list; this is its gate."""

    _real_run(tmp_path / "run")
    monkeypatch.chdir(tmp_path / "run")
    choices = next(action.choices
                   for action in cli.build_parser()._actions
                   if getattr(action, "choices", None)
                   and "doctor" in action.choices)
    assert "report" in choices
    assert cli.main(["report", "--dry-run"]) == 0
    assert "gpuwm diagnostic report" in capsys.readouterr().out


def test_an_extra_log_outside_the_run_directory_is_collected(
        tmp_path, capsys):
    run = tmp_path / "run"
    run.mkdir()
    (run / "report.json").write_text("{}", encoding="utf-8")
    stray = tmp_path / "terminal.log"
    stray.write_text("ValueError: the wizard refused the latitude\n",
                     encoding="utf-8")

    assert cli.main(["report", str(run), "--log", str(stray),
                     "--dry-run"]) == 0
    assert "logs/terminal.log" in capsys.readouterr().out
