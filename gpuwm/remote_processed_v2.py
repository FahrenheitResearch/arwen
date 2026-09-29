"""Compact native viewer fields derived on demand from committed WRF history.

Only explicitly selected or prefetched loop frames enter a bounded CPU queue.
Original WRF history is retained. Transfer streams individual native members;
there is no full-history conversion or duplicate portable archive.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import shutil
import subprocess
import sys
import time
import tomllib
from gpuwm import remote_artifacts as ra, remote_processed as legacy
from gpuwm.remote_artifact_cache import Lease, _owned_directory
SCHEMA = "arwen.remote-processed-frame.v2"
QUEUE_SCHEMA = "arwen.native-store-queue.v2"
PROFILE = "viewer-2d-v1"
SCIENCE_PROFILE = "full-science-v1"
DEFAULT_CACHE_BYTES = 2 * 1024**3
MAX_MEMBER_BYTES = 4 * 1024**3
MAX_PUBLICATION_BYTES = 8 * 1024**3
MAX_METADATA_BYTES = 512 * 1024
MAX_QUEUED = 32
MAX_PREFETCH = 8
MAX_ENTRIES = 100_000
#: The node's own viewer bound, as the node's viewer profile states it: a
#: request naming more than this many products fails the whole frame there.
NODE_PRODUCT_LIMIT = 128
#: The node's own per-product spelling bound, from the same profile.
MAX_SELECTOR_CHARS = 128
#: A selection also has to fit inside the fixed RPC envelope the worker reads,
#: with room left for the request built around it.
MAX_SELECTION_BYTES = 96 * 1024
#: How long one asked catalog answers for. The renderer's vocabulary changes
#: only when the node's renderer is replaced, and the question costs a process.
CATALOG_SECONDS = 300
SLUG = re.compile(r"[a-z0-9][a-z0-9_-]{0,95}\Z")
#: A first-class product selector. The plain slug is one spelling; the
#: renderer's own vocabulary also has colon-bearing families (`var:<field>`,
#: `mesh:<variable>`) that name a field rather
#: than a catalog entry, so a character class without a colon refused the
#: renderer's own spellings before the node ever saw them. The node's catalog
#: decides what it can serve; this grammar only refuses a spelling no node
#: could parse.
SELECTOR = re.compile(r"[a-z0-9][a-z0-9_-]{0,95}(?::[A-Za-z0-9][A-Za-z0-9_.,:=~@+/-]*)?\Z")
#: The renderer's own selector families, named in a refusal so a reader is told
#: what a selector may be rather than only that theirs was not one.
SELECTOR_FAMILIES = ("var:<stored 2-D variable>", "mesh:<history variable>")
SECTION_PREFIX = "xsec:"
SECTION_FAMILY = "xsec:<fill>[/<overlay>...]"
#: What a viewer says for a run that asked only for cross-sections. Without a
#: selection of its own, such a run read as an empty product list, which is the
#: node's default map set, so the viewer prepared and showed maps the run never
#: asked for and said nothing about the sections it did ask for.
NO_MAP_PRODUCTS_NOTE = ("This run asked only for cross-section pictures, which need a line this "
                        "viewer cannot take, so it has no map products to show here. The run draws "
                        "its sections itself.")



class Backpressure(ValueError):
    """The cache cannot admit another publication while readers retain it."""


def _root(workspace):
    return _owned_directory(Path(workspace) / ".arwen-processed-v2")


def _directory(root, job):
    from gpuwm.remote_worker import JOB_ID
    if not isinstance(job, str) or not JOB_ID.fullmatch(job):
        raise ValueError("Invalid native viewer job identity")
    return _owned_directory(root / job)


_CATALOG = {}


def node_catalog(*, now=None, include_sections=False):
    """This node's own product vocabulary, asked rather than transcribed.

    `rw_wrfbatch --list-products` is the renderer's own answer and this tree
    already has one reader for it, so the viewer keeps no second copy of the
    catalog. A node that cannot answer says so in this document and refuses
    nothing: the catalog names what a reader may ask for, and whether a named
    selector can actually be served is decided on the node when the frame is
    derived. Section products are offered only to a caller with a line input.
    """
    moment = time.monotonic() if now is None else now
    cached = _CATALOG.get("value")
    if cached is not None and moment - _CATALOG.get("at", 0) < CATALOG_SECONDS:
        return _catalog_selection(cached, include_sections)
    document = {"schema": "arwen.node-product-catalog.v1", "products": None, "count": None,
                "product_limit": NODE_PRODUCT_LIMIT,
                "product_limit_basis": "the node's own viewer profile bound on named products",
                "selector_families": [*SELECTOR_FAMILIES, SECTION_FAMILY],
                "source": None, "error": None}
    try:
        from gpuwm.runplan import render_catalog
        answered = render_catalog()
        rows = answered.get("products")
        if isinstance(rows, list):
            names = [str(row.get("name")) for row in rows if isinstance(row, dict) and row.get("name")]
            document.update(products=names, count=len(names),
                            source=answered.get("source") or "the node renderer's own --list-products",
                            group_keywords=answered.get("group_keywords") or [])
        else:
            document["error"] = str(answered.get("error") or "this node's renderer published no catalog")
    except Exception as error:  # noqa: BLE001 - an unreadable catalog is stated, never raised.
        document["error"] = f"{type(error).__name__}: {error}"[:1000]
    _CATALOG.update(value=document, at=moment)
    return _catalog_selection(document, include_sections)


def _catalog_selection(document, include_sections):
    if include_sections:
        return dict(document)
    result = {**document, "selector_families": list(SELECTOR_FAMILIES)}
    if isinstance(document.get("products"), list):
        names = [name for name in document["products"] if not name.startswith(SECTION_PREFIX)]
        result.update(products=names, count=len(names))
    return result


def _refuse_sections(products):
    sections = [item for item in products or []
                if isinstance(item, str) and item.startswith(SECTION_PREFIX)]
    if sections:
        raise ValueError("Viewer products " + ", ".join(repr(item) for item in sections)
                         + " need a cross-section line. This viewer has no line input, so the "
                           "renderer cannot locate the slice; choose a map product or draw the "
                           "section through the forecast door.")


def _catalog_note():
    """One clause naming the catalog door, with what a renderer here answered.

    The same sentence is read on a node and on a desktop, so it says which
    renderer answered rather than claiming the selected node's catalog from a
    machine that may hold a different one.
    """
    catalog = node_catalog()
    if catalog.get("count"):
        return (f" The renderer this check could ask publishes {catalog['count']} selectable "
                "products; `gpuwm remote list-products` prints the selected node's own.")
    return " `gpuwm remote list-products` prints what the selected node's renderer serves."


def _products(value):
    if not isinstance(value, list) or not value:
        raise ValueError("Viewer products must name at least one canonical product slug; send an "
                         "empty selection to take the node's own default set instead.")
    _refuse_sections(value)
    if len(value) > NODE_PRODUCT_LIMIT:
        raise ValueError(f"This request names {len(value)} viewer products and the node's viewer "
                         f"profile accepts at most {NODE_PRODUCT_LIMIT} named products, so the node "
                         "would refuse the whole frame rather than any one product. Ask for fewer "
                         "products, or send an empty selection to take the node's default set."
                         + _catalog_note())
    measured = len(ra._encoded(value))
    if measured > MAX_SELECTION_BYTES:
        from gpuwm.remote_worker import MAX_BYTES
        raise ValueError(f"This product selection is {measured} bytes and a selection may use at "
                         f"most {MAX_SELECTION_BYTES} of the node's {MAX_BYTES} byte request "
                         "envelope, because the rest of the envelope carries the request built "
                         "around it. Ask for fewer products.")
    for item in value:
        if not isinstance(item, str) or not SELECTOR.fullmatch(item) or len(item) > MAX_SELECTOR_CHARS:
            raise ValueError(f"Viewer products contain an invalid product slug: {str(item)[:120]!r}. "
                             "A selector is a catalog slug, or one of the renderer's own families "
                             f"({', '.join(SELECTOR_FAMILIES)}), of at most "
                             f"{MAX_SELECTOR_CHARS} characters.")
    return sorted(set(value))


def _cache_bytes(value):
    """The local viewer cache budget, in bytes.

    Any positive whole number of bytes is a budget: whether a frame fits it
    is decided where the frame is admitted, against the frame's measured
    size, and a frame larger than the budget is refused there by name. A
    budget of zero or less holds no frame at all, so every admission would
    refuse; it is refused here instead, once.
    """
    if type(value) is not int or value <= 0:
        raise ValueError("Viewer cache size must be a positive whole number of bytes; a budget "
                         "of zero or less can hold no viewer frame")
    return value


def _entry_path(root, job, sequence, selection):
    return _owned_directory(_directory(root, job) / "entries") / f"{sequence:012d}-{selection['selection_id']}.json"


def _entry(root, job, event, authority, selection):
    path = _entry_path(root, job, event["sequence"], selection)
    if not path.exists():
        return None
    value, _ = ra._raw(path, MAX_METADATA_BYTES)
    if (value.get("schema") != SCHEMA or value.get("job_id") != job
            or value.get("sequence") != event["sequence"] or value.get("domain") != event["domain"]
            or value.get("commit_sha256") != authority["sha256"]
            or value.get("selection_id") != selection["selection_id"]):
        raise ValueError("Viewer publication disagrees with its committed source or product selection")
    return value


def _available(entry):
    if entry is None or entry.get("state") != "ready":
        return False
    # Eviction can remove a published member between these filesystem checks.
    # A missing member is a cache miss; permission and metadata errors propagate.
    try:
        directory = Path(entry["object_root"])
        if not directory.is_dir() or directory.is_symlink():
            return False
        return all(Path(member["path"]).is_file() and not Path(member["path"]).is_symlink()
                   and Path(member["path"]).stat().st_size == member["bytes"] for member in entry["members"])
    except (FileNotFoundError, NotADirectoryError):
        return False


def _entry_state(entry):
    return "queued" if entry is None else ("ready" if _available(entry) else "evicted") if entry["state"] == "ready" else entry["state"]


def _expected_run(request, bound):
    expected = request.get("expected_run_id")
    if expected is not None and (not isinstance(expected, str) or not expected or len(expected) > 256):
        raise ValueError("Expected run identity is invalid")
    if expected is not None and bound is not None and bound[2]["run_id"] != expected:
        raise ValueError("Selected native run changed before the viewer request")


def _initialization(record):
    path = record.get("snapshot_config")
    if path is None:
        return None
    path = Path(path)
    if not path.is_absolute() or path.is_symlink():
        raise ValueError("Saved forecast initialization has no owned configuration path")
    with path.open("rb") as stream:
        payload = stream.read(128 * 1024 + 1)
    # The snapshot is the document the run loads; its own recorded digest is
    # what binds it, and the source file's digest is a different number here.
    if len(payload) > 128 * 1024 or record.get("snapshot_sha256") != ra._sha(payload):
        raise ValueError("Saved forecast configuration changed before viewer processing")
    value = tomllib.loads(payload.decode("utf-8"))["experiment"]["start_time"]
    if hasattr(value, "isoformat"):
        value = value.isoformat()
    return ra._timestamp(value) // 1000


def _native_members(result, store_root):
    if result.get("schema") not in ("arwen.wrf-process-result.v1", "arwen.wrf-process-result.v2"):
        raise ValueError("Native viewer processor did not publish a supported result schema")
    rows = result.get("members") or result.get("files")
    if not isinstance(rows, list) or not 1 <= len(rows) <= legacy.MAX_FILES:
        raise ValueError("Native viewer member inventory is invalid")
    members, seen, total = [], set(), 0
    grid_hash = result.get("frame", {}).get("grid_sha256")
    if not ra.HEX.fullmatch(str(grid_hash)):
        raise ValueError("Native viewer frame lacks its geographic grid SHA-256")
    for index, row in enumerate(rows):
        path = ra._inside(row.get("path"), store_root)
        size = path.stat().st_size
        if (path.suffix not in (".rws", ".rwg", ".json") or type(row.get("bytes")) is not int
                or row["bytes"] != size or not 0 < size <= MAX_MEMBER_BYTES
                or not ra.HEX.fullmatch(str(row.get("sha256"))) or ra._file_sha(path) != row["sha256"]):
            raise ValueError("Native viewer member size or digest is invalid")
        relative = path.relative_to(store_root).as_posix()
        key = row.get("key", f"member-{index}")
        if not isinstance(key, str) or not SLUG.fullmatch(key) or key in seen:
            raise ValueError("Native viewer member key is invalid or repeated")
        seen.add(key); total += size
        if row.get("grid_sha256", grid_hash) != grid_hash:
            raise ValueError("Native viewer member belongs to another geographic grid")
        members.append({"key": key, "relative_path": relative, "path": str(path), "bytes": size,
                        "sha256": row["sha256"], "grid_sha256": grid_hash,
                        "kind": row.get("kind", "rws" if path.suffix == ".rws" else "rwg" if path.suffix == ".rwg" else "metadata")})
    if total > MAX_PUBLICATION_BYTES or not any(member["relative_path"].endswith(".rws") for member in members):
        raise ValueError("Native viewer publication is oversized or has no field store")
    return members


def _check_time(result, record, event):
    identity = result["frame"]["identity"]
    lead, valid = identity.get("lead_seconds"), identity.get("valid_unix")
    if type(lead) is not int or lead < 0 or type(valid) is not int:
        raise ValueError("Native viewer frame needs exact integer UTC/lead seconds")
    initialization = valid - lead
    configured = _initialization(record)
    if (configured is not None and configured != initialization
            or result.get("initialization_unix", initialization) != initialization):
        raise ValueError("Native viewer initialization does not match this forecast")
    return initialization


def _convert(root, record, bound, event, authority, selection):
    from gpuwm.render import require_renderer
    from gpuwm.rustwx import renderer_env
    source = ra._inside(event.get("path"), bound[0]); before = ra._stamp(source)
    if not 0 < before[2] <= 16 * 1024**3 or event.get("size_bytes", before[2]) != before[2]:
        raise ValueError("Committed native WRF size changed or exceeds its processing bound")
    digest = ra._file_sha(source)
    if ra._stamp(source) != before:
        raise ValueError("Committed WRF changed while reading its source identity")
    directory = _owned_directory(_directory(root, record["id"]) / "objects" / (f"{event['sequence']:012d}-" + secrets.token_hex(12)))
    store_root = _owned_directory(directory / "native")
    request_path, result_path = directory / "request.json", directory / "result.json"
    request = {"schema": "arwen.wrf-process-request.v2" if selection["profile"] == PROFILE else "arwen.wrf-process-request.v1",
               "path": str(source), "source_sha256": digest, "case_id": bound[2]["run_id"],
               "domain": f"d{event['domain']:02d}", "valid_utc": ra.datetime.fromtimestamp(ra._timestamp(event["valid_time"]) / 1000,
                    ra.timezone.utc).isoformat().replace("+00:00", "Z"), "store_root": str(store_root), "heavy_ecape": False}
    configured = _initialization(record)
    if configured is not None:
        request["lead_seconds"] = ra._timestamp(event["valid_time"]) // 1000 - configured
        if request["lead_seconds"] < 0:
            raise ValueError("Committed output predates the saved forecast initialization")
    if selection["profile"] == PROFILE:
        request.update(profile=PROFILE, products=selection["products"])
    legacy._write(request_path, request)
    with (directory / "native.log").open("ab", buffering=0) as log:
        # Every call of the renderer gets one environment (renderer_env), so
        # an installed renderer is always handed the map files it draws with.
        process = subprocess.run([str(require_renderer()), "--process-request", str(request_path), "--process-result", str(result_path)],
                                 stdin=subprocess.DEVNULL, stdout=log, stderr=log, timeout=3600, check=False,
                                 env=renderer_env())
    if process.returncode != 0:
        raise ValueError(f"Native viewer derivation exited {process.returncode}; see {directory / 'native.log'}")
    if ra._stamp(source) != before:
        raise ValueError("Committed WRF changed during native viewer derivation")
    result, _ = ra._raw(result_path, MAX_METADATA_BYTES)
    legacy._check_identity(result, bound[2]["run_id"], event, digest)
    initialization = _check_time(result, record, event)
    if selection["profile"] == PROFILE:
        if result.get("schema") != "arwen.wrf-process-result.v2" or result.get("profile") != PROFILE:
            raise ValueError("Installed native processor needs the compact viewer v2 upgrade")
        statuses = result.get("products")
        # The node's returned set is the authority: a named selection must be
        # covered by it, and a node-default selection is whatever it answered.
        returned = sorted(row.get("slug", "") for row in statuses) if isinstance(statuses, list) else None
        if (returned is None or not returned
                or any(type(row.get("available")) is not bool or not isinstance(row.get("source_fields"), list)
                       or not isinstance(row.get("missing_reasons"), list) for row in statuses)
                or not set(selection["products"]).issubset(returned)):
            raise ValueError("Native viewer product capabilities do not match the selected products")
    members = _native_members(result, store_root)
    value = {"schema": SCHEMA, "state": "ready", "job_id": record["id"], "run_id": bound[2]["run_id"],
             "sequence": event["sequence"], "domain": event["domain"], "valid_time": event["valid_time"],
             "initialization_unix": initialization, "lead_seconds": result["frame"]["identity"]["lead_seconds"],
             "source_sha256": digest, "source_stamp": list(before), "source_path": str(source),
             "commit_sha256": authority["sha256"], **selection, "products": result.get("products", []),
             "frame": result["frame"], "native_result": result, "members": members,
             "object_root": str(directory), "native_store_root": str(store_root), "bytes": sum(row["bytes"] for row in members),
             "published_unix_ms": int(time.time() * 1000)}
    value["publication_sha256"] = ra._sha(ra._encoded(value))
    return value


def _remove_owned_tree(path, root):
    path, root = Path(path), Path(root).resolve(strict=True)
    if path.is_symlink() or not path.exists():
        if path.is_symlink():
            raise ValueError("Viewer cache cleanup refuses symlinks")
        return
    resolved = path.resolve(strict=True)
    if resolved == root or not resolved.is_relative_to(root) or any(item.is_symlink() for item in resolved.rglob("*")):
        raise ValueError("Viewer cache cleanup is outside its owned object directory")
    shutil.rmtree(resolved)


def _prune(root, job, limit, *, incoming=0, protected=()):
    directory = _directory(root, job)
    entries = []
    for count, path in enumerate(sorted((directory / "entries").glob("*.json"))):
        if count >= MAX_ENTRIES or path.is_symlink():
            raise ValueError("Viewer publication catalog exceeds its metadata bound")
        entry, _ = ra._raw(path, MAX_METADATA_BYTES)
        if _available(entry):
            entries.append((entry["published_unix_ms"], path, entry))
    used = sum(entry["bytes"] for _when, _path, entry in entries)
    if incoming > limit:
        raise Backpressure(f"This viewer artifact is {incoming} bytes; its selected cache budget is {limit}. Increase the cache budget or choose fewer products.")
    for _when, path, entry in sorted(entries):
        if used + incoming <= limit:
            break
        if path.name in protected:
            continue
        with Lease(directory / (path.stem + ".lock")) as lease:
            if lease.file is None:
                continue
            try:
                _remove_owned_tree(Path(entry["object_root"]), directory / "objects")
            except PermissionError:
                continue
            used -= entry["bytes"]
    if used + incoming > limit:
        raise Backpressure("Viewer cache is retained by active readers; derivation waits for cache space")
    return used


def index_metadata(workspace, job):
    path = _directory(_root(workspace), job) / "status.json"
    return ra._raw(path, 64 * 1024)[0] if path.exists() else {"schema": QUEUE_SCHEMA, "job_id": job, "state": "idle", "backlog": 0, "queued": 0, "active": None}


def stream(request, workspace, output):
    base = {"schema", "action", "workspace", "job", "domain", "sequence", "profile", "products"}
    digests = {"expected_publication_sha256", "expected_member_sha256", "expected_commit_sha256", "expected_manifest_sha256"}
    allowed = base | digests | {"member_key", "expected_run_id"}
    if set(request) - allowed or not (base | digests | {"member_key"}).issubset(request) or request.get("action") != "stream-processed-member-v2":
        raise ValueError("Invalid native viewer member stream request")
    if any(not ra.HEX.fullmatch(str(request[key])) for key in digests):
        raise ValueError("Invalid native viewer member authority digest")
    selection = selection_for(request)
    directory = _directory(_root(workspace), request["job"])
    path = _entry_path(_root(workspace), request["job"], ra._sequence(request["sequence"]), selection)
    with Lease(directory / (path.stem + ".lock"), timeout=5) as lease:
        if lease.file is None:
            raise ValueError("Native viewer publication is busy; retry its member transfer")
        query = {key: value for key, value in request.items() if key in base or key == "expected_run_id"}
        query["action"] = "processed-frame-v2"
        value = catalog(query, workspace, start=False)
        if (value["waiting"] or value["publication_sha256"] != request["expected_publication_sha256"]
                or value["commit"]["sha256"] != request["expected_commit_sha256"]
                or value["run_manifest"]["sha256"] != request["expected_manifest_sha256"]):
            raise ValueError("Native viewer authority changed before member transfer")
        member = next((row for row in value["members"] if row["key"] == request["member_key"]), None)
        if member is None or member["sha256"] != request["expected_member_sha256"]:
            raise ValueError("Native viewer member does not match this publication")
        path = ra._inside(member["path"], Path(value["native_store_root"]))
        before = ra._stamp(path); copied = 0; digest = hashlib.sha256()
        with path.open("rb") as source:
            while block := source.read(1024 * 1024):
                copied += len(block)
                if copied > member["bytes"]:
                    raise ValueError("Native viewer member grew during transfer")
                digest.update(block); output.write(block)
        output.flush()
        if copied != member["bytes"] or digest.hexdigest() != member["sha256"] or ra._stamp(path) != before:
            raise ValueError("Native viewer member changed during transfer")


def stream_main():
    from gpuwm import remote_worker as rw
    try:
        # One ownership provider answers the platform question at every door:
        # this stream serves a job whose ownership is established the same way.
        rw._ownership_provider()
        payload = sys.stdin.buffer.read(rw.MAX_BYTES + 1)
        if len(payload) > rw.MAX_BYTES:
            raise ValueError("Native viewer stream request exceeds its metadata limit")
        request = json.loads(payload)
        if not isinstance(request, dict) or request.get("schema") != "gpuwm.remote.request.v1":
            raise ValueError("Invalid native viewer stream schema")
        stream(request, rw._workspace(request), sys.stdout.buffer)
        return 0
    except (OSError, ValueError, KeyError) as error:
        print("remote native viewer: " + str(error)[:4000], file=sys.stderr)
        return 2


def _selection(profile=PROFILE, products=None):
    """A selection is either named products or the node's own default set."""
    if profile not in (PROFILE, SCIENCE_PROFILE):
        raise ValueError("Unknown native viewer processing profile")
    _refuse_sections(products)
    if profile == SCIENCE_PROFILE:
        value, token = {"profile": profile, "products": []}, None
    elif products is None or not list(products):
        # An empty or absent selection is a stable token for "whatever the
        # node's own viewer profile defaults to", never a transcribed list.
        value, token = {"profile": profile, "products": []}, True
    else:
        value, token = {"profile": profile, "products": _products(products)}, None
    identity = value if token is None else {**value, "node_default_products": token}
    return {**value, "selection_id": ra._sha(ra._encoded(identity))}


def selection_for(request):
    """The one selection a request carries, read the same way at every door.

    The catalog door and the member stream door both ask this, so one request
    cannot yield two selection identities and read one entry under a lease
    taken for the other.
    """
    return _selection(request.get("profile", PROFILE), request.get("products"))


def job_selection(record):
    """The product selection this job's own run asked for.

    The background map preparer and the plot gallery both read the render
    selection the run was started with through this one function, so one job
    never derives two product sets under two publication identities.

    The identity is the map terms' own, so a record that also names sections
    shares it with the same maps asked for alone. A record that names only
    sections has no map product at all: it gets a selection that says so
    (:func:`has_map_products`), never the node's default set.
    """
    terms = selectors(record.get("products"))
    maps = [item for item in terms if not item.startswith(SECTION_PREFIX)]
    if terms and not maps:
        value = {"profile": PROFILE, "products": [], "map_products": False}
        return {**value, "note": NO_MAP_PRODUCTS_NOTE, "selection_id": ra._sha(ra._encoded(value))}
    return _selection(PROFILE, maps)


def has_map_products(selection):
    """Whether a selection names maps to derive; a sections-only run's does not."""
    return selection.get("map_products") is not False


def map_selectors(spec):
    """Keep the run's map products for viewers with no section-line input."""
    return [item for item in selectors(spec) if not item.startswith(SECTION_PREFIX)]


def selectors(spec):
    """A recorded render selector string as a product list.

    `all` and `none` are the renderer's group vocabulary rather than named
    products, and the node's viewer profile takes named products only, so they
    resolve to the node's own default set and `selection_basis` says so. The
    string is read with the engine's own tokenizer (`product_spec_terms` in
    `gpuwm.rustwx`), so a section's level list, and the term that closes it,
    stay one selector (`xsec:QCLOUD=0.01,0.1/wa`) instead of pieces that
    `SELECTOR` refuses.
    """
    if spec is None:
        return []
    if isinstance(spec, list):
        return [str(item).strip() for item in spec if str(item).strip()]
    text = str(spec).strip()
    if not text or text.casefold() in ("all", "none"):
        return []
    from gpuwm.rustwx import product_spec_terms
    return product_spec_terms(text)


def selection_basis(profile, products):
    """Say where a selection's product set came from, in one sentence."""
    if profile == SCIENCE_PROFILE:
        return "the full-science profile derives a volume rather than a named product set"
    if products is None or not list(products):
        return "the node's own viewer profile default set, resolved on the node"
    return "the products this request named"


def selection_estimate(root, job, selection):
    """Price a selection from this job's own published frames; never refuse.

    The basis is the largest bytes-per-product ratio any frame of this job has
    actually published. On the first frame there is no recorded basis at all,
    which is said rather than treated as a reason to refuse.
    """
    named, published = len(selection["products"]), 0
    ratio = None
    try:
        paths = sorted((_directory(root, job) / "entries").glob("*.json"))
    except (OSError, ValueError):
        paths = []
    for path in paths[:MAX_ENTRIES]:
        try:
            entry, _ = ra._raw(path, MAX_METADATA_BYTES)
        except (OSError, ValueError):
            continue
        rows = entry.get("products")
        if entry.get("state") != "ready" or not isinstance(rows, list) or not rows:
            continue
        if type(entry.get("bytes")) is not int:
            continue
        if entry.get("selection_id") == selection["selection_id"]:
            # A node-default selection has no product count until the node has
            # answered one; its own published frames are where that count is.
            published = max(published, len(rows))
        measured = entry["bytes"] / len(rows)
        ratio = measured if ratio is None or measured > ratio else ratio
    named = named or published or None
    if ratio is None or named is None:
        return {"products": named, "estimated_bytes": None, "warn": False,
                "basis": "no frame has been published for this job yet, so this selection has no "
                         "recorded basis to be priced against; it is run and stated."}
    estimate = int(ratio * named)
    return {"products": named, "estimated_bytes": estimate,
            "cache_budget_bytes": DEFAULT_CACHE_BYTES, "warn": estimate > DEFAULT_CACHE_BYTES,
            "basis": "the largest bytes per product any frame of this job has published"}


def _queue(root, job):
    path = _directory(root, job) / "queue.json"
    if not path.exists():
        return []
    value, _ = ra._raw(path, 64 * 1024)
    rows = value.get("requests")
    if value.get("schema") != QUEUE_SCHEMA or not isinstance(rows, list) or len(rows) > MAX_QUEUED:
        raise ValueError("Compact viewer queue exceeds its bound or has an invalid schema")
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"profile", "products", "selection_id", "domain", "sequence", "run_id", "commit_sha256"}:
            raise ValueError("Compact viewer queue contains unsupported request fields")
        ra._domain(row.get("domain")); ra._sequence(row.get("sequence"))
        if (_selection(row.get("profile"), row.get("products"))["selection_id"] != row.get("selection_id")
                or not ra.HEX.fullmatch(str(row.get("commit_sha256")))
                or not isinstance(row.get("run_id"), str) or not row["run_id"]):
            raise ValueError("Compact viewer queue lost its committed source identity")
    return rows


def _save_queue(root, job, rows):
    legacy._write(_directory(root, job) / "queue.json", {"schema": QUEUE_SCHEMA, "requests": rows})


def _launch_worker(root, workspace):
    with Lease(root / "worker.lock") as lease:
        if lease.file is None:
            return
        from gpuwm.remote_worker import TOKEN_ENV
        environment = dict(os.environ)
        environment.pop(TOKEN_ENV, None)
        environment.update(GPUWM_NO_LOCAL_GPU="1", CUDA_VISIBLE_DEVICES="-1", RAYON_NUM_THREADS="2", OMP_NUM_THREADS="2")
        with (root / "worker.log").open("ab", buffering=0) as log:
            subprocess.Popen([sys.executable, "-I", "-m", "gpuwm.remote_processed_v2", "--workspace", str(workspace)],
                cwd=str(workspace), env=environment, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                start_new_session=True, close_fds=True)


def ensure(workspace, job, requests, *, start=True):
    """Only a viewer's selected/explicit prefetch frames enter this queue."""
    root = _root(workspace)
    with Lease(root / "schedule.lock", timeout=3) as lease:
        if lease.file is None:
            raise ValueError("The compact viewer queue is being updated; retry this selection")
        keys = {(row["sequence"], row["selection_id"]) for row in requests}
        previous = [row for row in _queue(root, job) if (row["sequence"], row["selection_id"]) not in keys]
        # The current frame is first, then its explicit ordered loop prefetch.
        # An abandoned selection cannot accumulate an unbounded work backlog.
        _save_queue(root, job, (requests + previous)[:MAX_QUEUED])
        if start:
            _launch_worker(root, workspace)


def _finish(root, job, selected):
    with Lease(root / "schedule.lock", timeout=3) as lease:
        if lease.file is None:
            raise ValueError("The compact viewer queue is being updated")
        _save_queue(root, job, [row for row in _queue(root, job)
                    if (row["sequence"], row["selection_id"]) != (selected["sequence"], selected["selection_id"])])


def _progress(root, job, simulation_state, committed, *, active=None, error=None):
    ready = failed = used = 0
    for count, path in enumerate((_directory(root, job) / "entries").glob("*.json")):
        if count >= MAX_ENTRIES or path.is_symlink():
            raise ValueError("Compact viewer catalog is excessive or contains a symlink")
        entry, _ = ra._raw(path, MAX_METADATA_BYTES)
        if _available(entry):
            ready += 1; used += entry["bytes"]
        elif entry.get("state") == "failed":
            failed += 1
    backlog = len(_queue(root, job))
    value = {"schema": QUEUE_SCHEMA, "job_id": job, "simulation_state": simulation_state,
             "committed": committed, "ready": ready, "failed": failed, "backlog": backlog,
             "queued": max(0, backlog - (active is not None)), "active": active,
             "cache_bytes": used, "cache_limit_bytes": DEFAULT_CACHE_BYTES,
             "state": "deriving" if active else "failed" if error else "queued" if backlog else "idle",
             "done": not backlog and active is None}
    if error is not None:
        value["error"] = str(error)[:2000]
        if isinstance(error, Backpressure):
            value["state"] = "backpressure"
    legacy._write(_directory(root, job) / "status.json", value)
    return value


def _work_job(workspace, job, *, completion=None):
    root = _root(workspace); directory = _directory(root, job)
    requests = _queue(root, job)
    if not requests:
        return False
    selected = requests[0]
    record, state, bound, commits = legacy._job(
        workspace, job, **({"completion": True} if completion is not None else {}))
    if completion is not None:
        # Revalidate before the conversion, the entry publication and the
        # queue mutation that follow, exactly as background preparation does.
        completion.validate(record, state, bound, commits)
    if bound is None:
        raise ValueError("Selected viewer frame lost its native producer manifest")
    _expected_run({"expected_run_id": selected["run_id"]}, bound)
    match = next(((event, authority) for event, authority in commits
                  if event["sequence"] == selected["sequence"] and event["domain"] == selected["domain"]), None)
    if match is None or match[1]["sha256"] != selected["commit_sha256"]:
        raise ValueError("Selected viewer frame lost its exact native output commit")
    event, authority = match
    entry = _entry(root, job, event, authority, selected)
    processor = legacy._processor_identity()
    if _available(entry) or entry is not None and entry.get("state") == "failed" and entry.get("processor") == processor:
        _finish(root, job, selected)
        _progress(root, job, state["state"], len(commits))
        return bool(_queue(root, job))
    path = _entry_path(root, job, event["sequence"], selected)
    error = None
    with Lease(directory / (path.stem + ".lock")) as lease:
        if lease.file is None:
            return True
        _progress(root, job, state["state"], len(commits), active={"domain": event["domain"], "sequence": event["sequence"], "profile": selected["profile"]})
        objects = _owned_directory(directory / "objects")
        before_objects = {path.name for path in objects.iterdir()}
        try:
            _prune(root, job, DEFAULT_CACHE_BYTES)
            entry = _convert(root, record, bound, event, authority, selected)
            _prune(root, job, DEFAULT_CACHE_BYTES, incoming=entry["bytes"])
            legacy._write(path, {**entry, "processor": processor})
        except Exception as failure:
            error = failure
            legacy._write(path, {"schema": SCHEMA, "job_id": job, "state": "backpressure" if isinstance(failure, Backpressure) else "failed", "domain": event["domain"],
                "sequence": event["sequence"], "commit_sha256": authority["sha256"], **selected,
                "processor": processor, "error": str(failure)[:2000]})
            # Only this single workspace worker can create native objects.
            # Cleanup is restricted to new objects from this failed attempt.
            for created in objects.iterdir():
                if created.name not in before_objects:
                    _remove_owned_tree(created, objects)
        _finish(root, job, selected)
    try:
        _record, state, _bound, commits = legacy._job(
            workspace, job, **({"completion": True} if completion is not None else {}))
    except (ra.ProducerCompletionPending, ra.ProducerCompletionUnprovable):
        # The conversion straddled the runner exit. This frame is finished and
        # its queue entry retired; the progress receipt keeps the state this
        # pass already validated rather than discarding the completed work.
        pass
    _progress(root, job, state["state"], len(commits), error=error)
    return bool(_queue(root, job))


def catalog(request, workspace, *, start=True):
    allowed = {"schema", "action", "workspace", "job", "domain", "sequence", "profile", "products", "expected_run_id", "prefetch_sequences"}
    if set(request) - allowed:
        raise ValueError("Unsupported compact viewer frame request fields")
    job, domain = request.get("job"), ra._domain(request.get("domain", 1))
    sequence = ra._sequence(request["sequence"]) if request.get("sequence") is not None else None
    prefetch = request.get("prefetch_sequences", [])
    if not isinstance(prefetch, list) or len(prefetch) > MAX_PREFETCH:
        raise ValueError("Explicit loop prefetch must contain at most eight committed frame sequences")
    prefetch = list(dict.fromkeys(ra._sequence(value) for value in prefetch))
    selection = selection_for(request)
    root = _root(workspace)
    completing = False
    try:
        record, state, bound, commits = legacy._job_completing(workspace, job)
    except ra.ProducerCompletionPending:
        # The interactive door waits with the watcher for the seconds a
        # settling wrapper owns, instead of refusing. Nothing is published and
        # no frame authority is returned until that wrapper settles.
        completing, record, state, bound, commits = True, None, {"state": "running"}, None, []
    _expected_run(request, bound)
    selected = [(event, authority) for event, authority in commits
                if event["domain"] == domain and (sequence is None or event["sequence"] == sequence)]
    value = {"schema": SCHEMA, "job_id": job, "domain": domain, "sequence": sequence,
             "waiting": True, "state": "waiting_for_output", "profile": selection["profile"],
             "selection_products": selection["products"], "products": [],
             "selection_estimate": selection_estimate(root, job, selection),
             "selection_basis": selection_basis(request.get("profile", PROFILE), request.get("products")),
             "processing": index_metadata(workspace, job)}
    if completing:
        value["producer_completing"] = True
    if bound is None:
        return value
    _producer, manifest_path, manifest, manifest_bytes, _started, binding = bound
    value.update(run_id=manifest["run_id"], run_manifest=ra._authority(manifest_path, manifest_bytes),
                 remote_output_root=record["outdir"], run_root=str(ra.run_root(record)), remote_pid=manifest["pid"])
    if binding is not None:
        value["producer_binding"] = binding
    if not selected:
        return value
    event, authority = selected[-1]
    value.update(sequence=event["sequence"], commit=authority, valid_time=event["valid_time"])
    entry = _entry(root, job, event, authority, selection)
    entry_state = _entry_state(entry)
    wanted = [event["sequence"], *(seq for seq in prefetch if seq != event["sequence"])]
    pending = []
    for seq in wanted:
        found = next(((ev, auth) for ev, auth in commits if ev["sequence"] == seq and ev["domain"] == domain), None)
        if found is None:
            raise ValueError("Loop prefetch includes a frame outside the selected committed domain")
        if not _available(_entry(root, job, *found, selection)):
            pending.append({**selection, "domain": domain, "sequence": seq, "run_id": manifest["run_id"], "commit_sha256": found[1]["sha256"]})
    if start and pending:
        ensure(workspace, job, pending)
    value["state"] = entry_state
    value["processing"] = {**index_metadata(workspace, job), "simulation_state": state["state"]}
    if entry_state != "ready":
        active = value["processing"].get("active") or {}
        if entry_state == "queued" and active.get("domain") == domain and active.get("sequence") == event["sequence"]:
            value["state"] = "deriving"
        if entry is not None and entry.get("error"):
            value["error"] = entry["error"]
        elif value["processing"].get("state") == "failed":
            value.update(state="failed", error=value["processing"].get("error", "Native viewer worker could not complete"))
        return ra._bounded(value)
    source = ra._inside(entry["source_path"], bound[0])
    if list(ra._stamp(source)) != entry["source_stamp"]:
        raise ValueError("Committed WRF changed after the compact viewer publication")
    legacy._check_identity(entry["native_result"], manifest["run_id"], event, entry["source_sha256"])
    _check_time(entry["native_result"], record, event)
    value.update(waiting=False, source_sha256=entry["source_sha256"], frame=entry["frame"],
                 initialization_unix=entry["initialization_unix"], lead_seconds=entry["lead_seconds"],
                 publication_sha256=entry["publication_sha256"], selection_id=entry["selection_id"],
                 native_result=entry["native_result"], native_store_root=entry["native_store_root"],
                 products=entry["products"], members=entry["members"], bytes=entry["bytes"])
    return ra._bounded(value)


def worker(workspace, *, cancel=None):
    root = _root(workspace)
    with Lease(root / "worker.lock", timeout=3) as lease:
        if lease.file is None:
            return 0
        if hasattr(os, "nice"):
            os.nice(10)
        waits = {}
        while True:
            pending = False
            # Bound work per job, without letting historical directories hide
            # queued work later in the same workspace.
            directories = sorted(path for path in root.iterdir() if path.is_dir() and not path.is_symlink())
            for directory in directories:
                if not (directory / "queue.json").exists():
                    continue
                job = directory.name
                completion = waits.setdefault(job, ra.CompletionWait(workspace, job, cancel, time))
                try:
                    # Cancellation and the deadline are answered on every pass,
                    # not only on the passes that revalidate.
                    completion.check()
                    if not completion.due():
                        # A job awaiting its wrapper is revalidated on its own
                        # interval; the rest of the workspace keeps its pace.
                        pending = True
                        continue
                    completion.begin()
                    try:
                        pending |= _work_job(workspace, job, completion=completion)
                    except ra.ProducerCompletionPending as event:
                        # The runner exited and its wrapper has not settled. The
                        # queue and its entries are left exactly as they are.
                        completion.pending(event)
                        pending = True
                    except ra.ProducerCompletionUnprovable as unprovable:
                        # This job's runner-exit window cannot be proved. A
                        # terminal receipt here would empty the user's queued
                        # frames for a job that is still running, so the queue
                        # is left exactly as it is and the receipt says why
                        # this job is still pending.
                        completion.unprovable(unprovable)  # Raises if this wait held proof.
                        legacy._write(directory / "status.json", {"schema": QUEUE_SCHEMA,
                            "job_id": job, "state": "waiting_for_producer_completion",
                            "done": False, "error": str(unprovable)[:2000]})
                        pending = True
                except ra.ProducerCompletionCancelled as cancelled:
                    # Cancellation is not a failure: the receipt stays
                    # non-terminal and the user's queued frames survive it.
                    legacy._write(directory / "status.json", {"schema": QUEUE_SCHEMA, "job_id": job,
                        "state": "cancelled", "done": False, "error": str(cancelled)[:2000]})
                    return 2
                except Exception as error:
                    legacy._write(directory / "status.json", {"schema": QUEUE_SCHEMA, "job_id": job,
                        "state": "failed", "done": True, "error": str(error)[:2000]})
                    with Lease(root / "schedule.lock", timeout=3) as schedule:
                        if schedule.file is not None:
                            _save_queue(root, job, [])
            if not pending:
                # Release worker ownership while enqueue is excluded: a request
                # arriving at worker exit must start a successor, never strand.
                with Lease(root / "schedule.lock", timeout=3) as schedule:
                    latest = [path for path in root.iterdir() if path.is_dir() and not path.is_symlink()]
                    if schedule.file is not None and not any(_queue(root, path.name) for path in latest if (path / "queue.json").exists()):
                        lease.close()
                        return 0
            time.sleep(.1)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True)
    args = parser.parse_args(argv)
    from gpuwm.remote_worker import _workspace
    return worker(_workspace({"workspace": args.workspace}), cancel=ra.cancel_on_shutdown())


if __name__ == "__main__":
    raise SystemExit(main())
