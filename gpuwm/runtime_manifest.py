"""One schema validator for the sealed-runtime manifest, and one answer
to "how does THIS install know what it is".

Two things used to be decided independently in six places each, and both
cost a field user a run.

**The manifest.**  ``GPUWM_NATIVE_DISTRIBUTION_MANIFEST`` names the
``gpuwm-native-wrf-runtime-v1`` document a sealed runtime archive writes
beside its artifacts.  Every consumer validated the part it was about to
read and no more: the identity helper checked ``schema``/``status``/
``artifact.gpuwm_version``/``source.worktree_clean``, the preprocessing
selector checked ``contract.platform.backends``, the decoder binder
checked ``payload``.  A hand-authored minimal manifest therefore passed
the first gate, ran a preparation for several minutes, and died at the
third with "sealed native distribution has no valid preprocessing
backend inventory" -- a true sentence about a document that was already
invalid when the run started.  :func:`validate_manifest` is the whole
required schema in one place, and it names EVERY defect at once, so a
manifest is accepted or refused before any work begins.

**The identity.**  A run's provenance receipt asks the same question a
manifest answers, for installs that have no manifest.  The old answer
was ``git rev-parse`` with the working directory set to the directory
containing the ``gpuwm`` package -- which is a checkout for a developer
and ``site-packages`` for everyone else.  On every pip install that is
``CalledProcessError: returned non-zero exit status 128``, and on a venv
nested inside some unrelated repository it is worse: it succeeds and
binds a stranger's commit.  :func:`provenance` resolves the three real
cases in order -- sealed manifest, genuine checkout of THIS tree,
installed wheel -- and the wheel case is answered from the distribution
metadata and its ``RECORD`` digests, which is exactly the identity pip
installed.

Nothing here imports numpy, cupy, or any ingest module: ``gpuwm doctor``
runs it on a base install to report which path an install has *before*
anything else runs.
"""

from __future__ import annotations

import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import subprocess

from gpuwm import DISTRIBUTION_NAME, __version__

#: The sealed-runtime manifest schema.  Declared here rather than in
#: :mod:`gpuwm.native_wrf_distribution` so the validator is importable
#: without pulling the distribution builder's dependencies; that module
#: re-exports the name it has always published.
RUNTIME_SCHEMA = "gpuwm-native-wrf-runtime-v1"

#: The environment variable a sealed runtime archive's launcher exports.
MANIFEST_ENV = "GPUWM_NATIVE_DISTRIBUTION_MANIFEST"

#: Preprocessing backends a manifest may declare.  Mirrors the
#: ``contract.platform.backends`` inventory
#: :func:`gpuwm.native_wrf_distribution.distribution_contract` writes.
KNOWN_BACKENDS = ("cpu", "cuda")

#: The identity paths :func:`provenance` can report, in the order it
#: tries them.
IDENTITY_SOURCES = (
    "gpuwm-native-distribution-manifest",
    "git",
    "installed-editable-source",
    "installed-wheel-record",
)

#: Distributions that may publish the ``gpuwm`` package.  ``rw-wps`` is
#: the preprocessing-only wheel; it ships the same package directory
#: under a different distribution name, so the identity lookup has to
#: recognise both rather than assuming the public one.
_CANDIDATE_DISTRIBUTIONS = (DISTRIBUTION_NAME, "rw-wps")

_GIT_TIMEOUT_S = 30

#: Appended to every unbindable-identity refusal.  An editable install
#: legitimately has no wheel identity, so "reinstall it with pip" is the
#: wrong advice on its own: what such an install needs is a readable
#: checkout to bind instead.
_EDITABLE_REMEDY = (
    ".  If this is an editable install (pip install -e), its identity "
    "binds to the source tree instead -- and every way of reading that "
    "tree has been tried here: git on PATH, git at its default install "
    "location, and a direct read of the tree's own .git files.  What is "
    "missing is a source tree with a readable .git beside the gpuwm "
    "package")


class ManifestError(ValueError):
    """A sealed-runtime manifest that does not meet the required schema."""


class IdentityError(RuntimeError):
    """This install can prove neither a checkout nor a wheel identity."""


# ---------------------------------------------------------------------------
# The manifest: one required schema, every defect named at once
# ---------------------------------------------------------------------------

def _record_defects(payload: object) -> list[str]:
    """Defects of the per-artifact ``payload`` inventory."""

    if not isinstance(payload, dict) or not payload:
        return ["payload: not a non-empty object of per-artifact records"]
    defects = []
    for relative in sorted(payload)[:5]:
        record = payload[relative]
        if not isinstance(record, dict):
            defects.append(f"payload[{relative}]: not an object")
            continue
        if not isinstance(record.get("sha256"), str):
            defects.append(f"payload[{relative}].sha256: not a string")
        size = record.get("bytes")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            defects.append(
                f"payload[{relative}].bytes: not a non-negative integer")
    return defects


def manifest_defects(payload: object, *,
                     gpuwm_version: str | None = None) -> list[str]:
    """Every way ``payload`` falls short of the required schema.

    An empty list means the document carries the complete inventory every
    consumer reads.  The list is deliberately exhaustive rather than
    first-failure: someone authoring or repairing a manifest should learn
    all of it in one pass, not discover the next missing key after each
    re-run.
    """

    expected_version = __version__ if gpuwm_version is None else gpuwm_version
    if not isinstance(payload, dict):
        return ["the manifest is not a JSON object"]
    defects: list[str] = []
    if payload.get("schema") != RUNTIME_SCHEMA:
        defects.append(
            f"schema: {payload.get('schema')!r} is not {RUNTIME_SCHEMA!r}")
    if payload.get("status") != "READY":
        defects.append(f"status: {payload.get('status')!r} is not 'READY'")

    artifact = payload.get("artifact")
    if not isinstance(artifact, dict):
        defects.append("artifact: missing or not an object")
    elif artifact.get("gpuwm_version") != expected_version:
        defects.append(
            f"artifact.gpuwm_version: {artifact.get('gpuwm_version')!r} is "
            f"not this install's {expected_version!r}")

    source = payload.get("source")
    if not isinstance(source, dict):
        defects.append("source: missing or not an object")
    else:
        for key in ("commit", "tree"):
            if not isinstance(source.get(key), str) or not source[key]:
                defects.append(f"source.{key}: missing or not a string")
        if not source.get("worktree_clean"):
            defects.append("source.worktree_clean: not true")

    contract = payload.get("contract")
    if not isinstance(contract, dict):
        defects.append("contract: missing or not an object")
    else:
        selected = contract.get("platform")
        if not isinstance(selected, dict):
            defects.append("contract.platform: missing or not an object")
        else:
            backends = selected.get("backends")
            if (not isinstance(backends, list) or not backends
                    or any(item not in KNOWN_BACKENDS for item in backends)
                    or len(set(backends)) != len(backends)):
                defects.append(
                    "contract.platform.backends: not a non-empty list of "
                    f"unique names from {list(KNOWN_BACKENDS)} "
                    f"(found {backends!r})")

    defects.extend(_record_defects(payload.get("payload")))
    return defects


def validate_manifest(payload: object, *, path: Path | None = None,
                      gpuwm_version: str | None = None) -> dict:
    """Return ``payload`` when it is a complete manifest; else refuse.

    The refusal names the document and every defect, on its own line, so
    the reader is never sent back for a second run to learn the second
    problem.
    """

    defects = manifest_defects(payload, gpuwm_version=gpuwm_version)
    if not defects:
        return payload  # type: ignore[return-value]
    subject = f"{path}: " if path is not None else ""
    raise ManifestError(
        f"{subject}not a complete {RUNTIME_SCHEMA} document -- "
        f"{len(defects)} problem(s):\n  "
        + "\n  ".join(defects)
        + f"\n  # only a sealed runtime archive's installer writes this "
        f"document.\n  # On a source clone or a pip install, unset "
        f"{MANIFEST_ENV}: those\n  # installs bind their identity from "
        "git or from the installed wheel.")


def load_manifest(path: Path, *,
                  gpuwm_version: str | None = None) -> dict:
    """Read and fully validate one manifest document."""

    path = Path(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeDecodeError) as error:
        raise ManifestError(
            f"{path}: {MANIFEST_ENV} does not name a readable JSON "
            f"document: {error}") from error
    return validate_manifest(payload, path=path, gpuwm_version=gpuwm_version)


def manifest_from_environment(
        *, gpuwm_version: str | None = None) -> tuple[Path, dict] | None:
    """``(path, manifest)`` when the sealed runtime bound one, else None.

    Every consumer that used to re-implement the environment lookup and
    its own partial predicate calls this instead, so an invalid document
    is refused identically wherever it is first read.
    """

    raw = os.environ.get(MANIFEST_ENV)
    if not raw:
        return None
    path = Path(raw).resolve()
    return path, load_manifest(path, gpuwm_version=gpuwm_version)


# ---------------------------------------------------------------------------
# The identity: manifest, then a genuine checkout, then the wheel
# ---------------------------------------------------------------------------

def git_checkout_root(root: Path) -> Path | None:
    """The git top level that owns ``root`` -- only when it IS ``root``.

    ``None`` for a pip install (no repository at all) and, just as
    importantly, for a venv created inside somebody else's repository:
    that repository's history is not this code's provenance, and binding
    it would be a receipt that names the wrong tree rather than an
    honest absence.  Never raises: "is this a checkout" is a question,
    not a failure.
    """

    root = Path(root)
    executable = _git_exe()
    if executable is None:
        return None
    try:
        completed = subprocess.run(
            [executable, "-C", str(root), "rev-parse", "--show-toplevel"],
            capture_output=True, text=True, timeout=_GIT_TIMEOUT_S,
            check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    try:
        top_level = Path(completed.stdout.strip()).resolve()
        return top_level if top_level == root.resolve() else None
    except OSError:
        return None


def _git_exe() -> str | None:
    """The git this process can launch, resolved in ONE place.

    Every git question in this module and in ``gpuwm.provenance`` goes
    through :func:`gpuwm.provenance.git_executable`, so a box where git
    is installed but absent from a composed ``PATH`` answers the same
    here as it does there.  Two resolutions would be two answers to one
    question, which is the shape of the defect this replaced.
    """

    try:
        from gpuwm.provenance import git_executable

        return git_executable()
    except Exception:                                   # noqa: BLE001
        return None


def _git(root: Path, *arguments: str) -> str:
    executable = _git_exe()
    if executable is None:
        raise OSError("git cannot be executed in this process")
    return subprocess.check_output(
        [executable, "-C", str(root), *arguments], text=True,
        stderr=subprocess.DEVNULL, timeout=_GIT_TIMEOUT_S).strip()


def installed_distribution(package: str = "gpuwm"):
    """The distribution that installed ``package`` here, or None.

    Resolved by asking each candidate distribution to locate the package
    file that is actually imported, rather than by taking the first name
    that has metadata: two distributions publish this package directory,
    and a stale ``.dist-info`` for the other one must not be mistaken
    for the identity of the code that is running.
    """

    anchor = (Path(__file__).resolve().parent / "__init__.py")
    for name in _CANDIDATE_DISTRIBUTIONS:
        try:
            dist = metadata.distribution(name)
        except metadata.PackageNotFoundError:
            continue
        try:
            located = Path(dist.locate_file(f"{package}/__init__.py")).resolve()
        except (OSError, ValueError):
            continue
        if located == anchor:
            return dist
    return None


def _is_compiled_bytecode(item) -> bool:
    """True for a ``__pycache__``/``.pyc``/``.pyo`` RECORD entry."""

    text = str(item).replace("\\", "/")
    return text.endswith((".pyc", ".pyo")) or "__pycache__/" in text


def wheel_record_identity(*, verify: bool = False) -> dict[str, object]:
    """This install's identity, read from the wheel pip actually wrote.

    ``RECORD`` carries pip's own digest of every installed file; the
    aggregate over ``name\\0digest`` lines is therefore a stable name for
    the exact artifact set on disk, obtainable without re-reading a
    single payload byte.  ``verify=True`` additionally re-hashes each
    recorded file, which is an integrity check rather than an identity
    one and is left to the caller that wants to pay for it.
    """

    dist = installed_distribution()
    if dist is None:
        raise IdentityError(
            "no installed distribution provides this gpuwm package "
            f"(looked for {', '.join(_CANDIDATE_DISTRIBUTIONS)}), and it is "
            "not a git checkout either, so this run has no provenance to "
            "bind" + _EDITABLE_REMEDY)
    recorded: list[tuple[str, str]] = []
    unhashed = 0
    verified = 0
    for item in dist.files or ():
        if _is_compiled_bytecode(item):
            # Bytecode is DERIVED from the source this identity already
            # binds, and it materializes lazily: `__pycache__` entries
            # appear as modules are first imported, so counting them
            # makes this identity a function of how far the process has
            # got rather than of what is installed.
            #
            # Measured on a fresh `pip install gpuwm[all]`: the count
            # went 109 -> 358 across one process that imported the
            # package tree.  The forecast runner snapshots this identity
            # at the start of a run and compares it at the END, so a
            # long first run died on `RuntimeError: forecast runtime
            # implementation changed during run` with the physics
            # complete and the frames on disk -- reproducibly, on the
            # shipped 1.4.0 wheel, whenever bytecode was not already
            # warm.  That is the new-user path exactly.
            continue
        digest = item.hash
        if digest is None:
            unhashed += 1
            continue
        if digest.mode != "sha256":
            raise IdentityError(
                f"unsupported wheel RECORD digest {digest.mode!r} for {item}")
        recorded.append((str(item).replace("\\", "/"), digest.value))
        if verify:
            path = Path(dist.locate_file(item))
            if not path.is_file():
                raise IdentityError(f"installed wheel file is missing: {path}")
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            encoded = _urlsafe_b64_nopad(bytes.fromhex(actual))
            if encoded != digest.value:
                raise IdentityError(
                    f"installed wheel RECORD mismatch for {item}")
            verified += 1
    if not recorded:
        raise IdentityError(
            f"installed distribution {dist.metadata['Name']} has no hashed "
            "RECORD entries, so its identity cannot be bound; reinstall it "
            "with pip" + _EDITABLE_REMEDY)
    aggregate = hashlib.sha256()
    for name, value in sorted(recorded):
        aggregate.update(name.encode("utf-8") + b"\0" + value.encode("ascii")
                         + b"\n")
    identity: dict[str, object] = {
        "distribution_name": dist.metadata["Name"],
        "distribution_version": dist.version,
        "record_file_count": len(recorded),
        "record_unhashed_count": unhashed,
        "record_aggregate_sha256": aggregate.hexdigest(),
    }
    if verify:
        identity["record_verified_file_count"] = verified
    return identity


def _urlsafe_b64_nopad(raw: bytes) -> str:
    import base64

    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


# ---------------------------------------------------------------------------
# The editable install: the identity is the SOURCE TREE, not a RECORD
# ---------------------------------------------------------------------------

def _is_editable_install(dist) -> bool:
    """Does PEP 610 say pip wired this distribution to a source tree?"""

    if dist is None:
        return False
    try:
        from gpuwm.provenance import editable_root

        return editable_root(dist) is not None
    except Exception:                                   # noqa: BLE001
        return False


def source_tree_identity(*, distribution=None) -> dict[str, object] | None:
    """This install's identity when pip wired it to a source tree.

    An editable install has no wheel identity to bind and never will.
    ``pip install -e`` writes a RECORD whose rows describe a *redirect*
    -- a ``.pth`` and a finder module, or nothing hashed at all -- so
    :func:`wheel_record_identity` either refuses it outright ("no hashed
    RECORD entries") or, worse, succeeds and binds the digest of two
    shim files that say nothing whatsoever about the code that is about
    to run.  Both outcomes were reached in the field; the first killed a
    prepare stage before a byte of the user's data was read.

    What an editable install CAN prove is the source tree it points at,
    which is precisely the identity
    :func:`gpuwm.provenance.describe_provenance` already binds into the
    run-plan receipt.  This resolves it through that module's own
    helpers -- :func:`gpuwm.provenance.editable_root` for the tree and
    :func:`gpuwm.provenance.git_identity` for its commit, branch and
    dirt -- so the run manifest and the run-plan receipt state ONE
    identity in two documents rather than two answers that can drift.

    Returns ``None`` -- never raises -- when no source tree can be
    resolved or git cannot read it.  That is the genuinely unbindable
    case, and :func:`provenance` keeps refusing it.
    """

    from gpuwm.provenance import (editable_root, git_identity,
                                  providing_distribution)

    package = Path(__file__).resolve().parent
    dist = distribution if distribution is not None else installed_distribution()
    if dist is None:
        # ``locate_file`` cannot answer for a PEP 660 editable and
        # answers confidently wrong, so `installed_distribution()`
        # returns None for every one of them -- including the one
        # serving this very import.  `providing_distribution` matches
        # the way an editable install is actually wired.
        try:
            dist = providing_distribution(package)
        except Exception:                               # noqa: BLE001
            dist = None
    declared = None
    if dist is not None:
        try:
            declared = editable_root(dist)
        except Exception:                               # noqa: BLE001
            declared = None

    roots: list[Path] = []
    # The declared root first, deliberately.  Every caller of
    # :func:`provenance` hands it the directory the running package
    # sits in, and reaching the editable branch means
    # :func:`git_checkout_root` has ALREADY refused that directory --
    # so `direct_url.json` is the one thing the caller did not already
    # have.  The running directory stays as the fallback for the
    # editable shapes that leave no `direct_url.json` at all (a legacy
    # `develop` install, or an `.egg-info` beside the source).
    if declared is not None:
        roots.append(declared)
    # `git_identity` refuses any directory that is not the top level of
    # its OWN repository, so neither candidate can bind a stranger's
    # commit from a venv nested in someone else's checkout.
    if package.parent not in roots:
        roots.append(package.parent)

    for candidate in roots:
        try:
            git = git_identity(candidate)
        except Exception:                               # noqa: BLE001
            git = None
        if git is None:
            continue
        return {
            "distribution_name": (
                dist.metadata["Name"] if dist is not None else None),
            "distribution_version": (
                dist.version if dist is not None else None),
            "install_kind": "editable" if dist is not None else "source-tree",
            "source_root": str(candidate),
            "git": dict(git),
        }
    return None


def _git_or_none(root: Path, *arguments: str) -> str | None:
    try:
        return _git(root, *arguments)
    except (OSError, subprocess.SubprocessError):
        return None


def _editable_provenance(identity: dict[str, object]) -> dict[str, object]:
    """The receipt shape every consumer reads, from a source identity."""

    root = Path(str(identity["source_root"]))
    git = identity["git"]
    status = _git_or_none(root, "status", "--short")
    document = {
        "identity_source": "installed-editable-source",
        "git_commit": git["commit_full"],               # type: ignore[index]
        "git_tree": _git_or_none(root, "rev-parse", "HEAD^{tree}"),
        # An editable tree is ALLOWED to be dirty -- a developer's is,
        # by definition -- but a receipt must never claim a clean
        # identity from one, so the lines are reported here and the
        # count under ``installed_editable.git.dirty_files``.
        "git_status_short": None if status is None else status.splitlines(),
        "distribution_manifest_sha256": None,
        "installed_wheel": None,
        "installed_editable": identity,
    }
    # A ``git_status_short`` of None reads as "no lines" to a careless
    # consumer, which is the same mistake as claiming clean.  When the
    # commit was read straight out of ``.git`` because git could not be
    # launched, the receipt says so in as many words rather than
    # leaving the absence to be interpreted.
    unknown = (git.get("dirty_unknown_reason")
               if isinstance(git, dict) else None)
    if unknown:
        document["git_status_unknown_reason"] = unknown
    return document


def provenance(root: Path, *,
               gpuwm_version: str | None = None) -> dict[str, object]:
    """What this install is, by the first of three paths that applies.

    ``root`` is the directory that contains the ``gpuwm`` package: a
    checkout root for a developer, ``site-packages`` for a wheel.  The
    returned mapping always carries ``identity_source``, ``git_commit``,
    ``git_tree`` and ``git_status_short`` -- the shape every existing
    receipt consumer reads -- plus whichever of
    ``distribution_manifest_sha256`` / ``installed_wheel`` /
    ``installed_editable`` the resolved path earned.  It raises only
    when NO path can answer, which is a broken install rather than an
    unusual one.

    The paths, in order: a sealed manifest, a genuine checkout of THIS
    tree, an editable install's source tree, then the installed wheel's
    RECORD.  The editable path comes before the wheel because an
    editable install's RECORD describes the redirect pip wrote, not the
    code -- binding it would be a receipt naming two shim files.  A
    plain wheel install never reaches that branch and so pays for no
    extra git subprocess.
    """

    bound = manifest_from_environment(gpuwm_version=gpuwm_version)
    if bound is not None:
        manifest_path, manifest = bound
        source = manifest["source"]
        return {
            "identity_source": "gpuwm-native-distribution-manifest",
            "git_commit": str(source["commit"]),
            "git_tree": str(source["tree"]),
            "git_status_short": [],
            "distribution_manifest_sha256": hashlib.sha256(
                manifest_path.read_bytes()).hexdigest(),
            "installed_wheel": None,
            "installed_editable": None,
        }
    root = Path(root)
    if git_checkout_root(root) is not None:
        try:
            return {
                "identity_source": "git",
                "git_commit": _git(root, "rev-parse", "HEAD"),
                "git_tree": _git(root, "rev-parse", "HEAD^{tree}"),
                "git_status_short": _git(
                    root, "status", "--short").splitlines(),
                "distribution_manifest_sha256": None,
                "installed_wheel": None,
                "installed_editable": None,
            }
        except (OSError, subprocess.SubprocessError):
            # A checkout with no commits yet, or a git that died between
            # the two calls.  Fall through to the wheel rather than
            # failing a run over a provenance nicety.
            pass
    if _is_editable_install(installed_distribution()):
        source = source_tree_identity()
        if source is not None:
            return _editable_provenance(source)
    try:
        wheel = wheel_record_identity()
    except IdentityError:
        # No wheel identity.  Before refusing, ask the question the
        # wheel path cannot: is this an install pip wired to a source
        # tree?  Every editable shape lands here -- the PEP 660 one that
        # `installed_distribution()` cannot see at all, and the legacy
        # one whose RECORD carries no digests.
        source = source_tree_identity()
        if source is None:
            raise
        return _editable_provenance(source)
    return {
        "identity_source": "installed-wheel-record",
        "git_commit": None,
        "git_tree": None,
        "git_status_short": None,
        "distribution_manifest_sha256": None,
        "installed_wheel": wheel,
        "installed_editable": None,
    }


__all__ = [
    "IDENTITY_SOURCES", "IdentityError", "KNOWN_BACKENDS", "MANIFEST_ENV",
    "ManifestError", "RUNTIME_SCHEMA", "git_checkout_root",
    "installed_distribution", "load_manifest", "manifest_defects",
    "manifest_from_environment", "provenance", "source_tree_identity",
    "validate_manifest", "wheel_record_identity",
]
