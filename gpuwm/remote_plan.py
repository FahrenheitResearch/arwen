"""Manifest-bound desktop plans for the existing durable SSH job controller.

Only selected small inputs are staged. Missing ERA5 forcing stays declared and
is acquired by run-plan's existing fetch owner after a reviewed launch.
"""
from __future__ import annotations

import base64
import copy
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import secrets
import stat
import time
import tomllib

BUNDLE_SCHEMA = "gpuwm.remote.plan-bundle.v1"
BLOB_BUNDLE_SCHEMA = "gpuwm.remote.plan-bundle.v2"
MAX_FILES = 20
MAX_INPUT_BYTES = 64 * 1024
MAX_SINGLE_BYTES = 48 * 1024
ID = re.compile(r"[a-f0-9]{32}\Z")
SHA = re.compile(r"[a-f0-9]{64}\Z")


def _sha(payload):
    return hashlib.sha256(payload).hexdigest()


def _encoded(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                      separators=(",", ":")).encode("utf-8")


def _remote_path(value, label):
    if (not isinstance(value, str) or not PurePosixPath(value).is_absolute()
            or "\\" in value or any(ord(c) < 32 for c in value)
            or ".." in PurePosixPath(value).parts):
        raise ValueError(f"{label} must be an absolute node path without traversal")
    return str(PurePosixPath(value))


def _name(value):
    if (not isinstance(value, str) or not value or len(value) > 256
            or "\\" in value or any(ord(c) < 32 for c in value)
            or PurePosixPath(value).is_absolute()
            or any(p in ("", ".", "..") for p in value.split("/"))):
        raise ValueError("staged input name must stay inside its bundle")
    return value


def _assert_relocated(value, allowed, field="configuration"):
    """Never leave an unhandled desktop authority path to fail on Linux later."""
    if isinstance(value, dict):
        for key, item in value.items():
            _assert_relocated(item, allowed, field + "." + str(key))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _assert_relocated(item, allowed, f"{field}[{index}]")
    elif isinstance(value, str):
        windows = bool(re.match(r"^[A-Za-z]:[\\/]", value)) or value.startswith("\\\\")
        absolute = value.startswith("/")
        permitted = any(value == root or value.startswith(root.rstrip("/") + "/") for root in allowed)
        if windows or absolute and not permitted:
            filename = value.replace("\\", "/").rsplit("/", 1)[-1]
            raise ValueError(f"Selected setting '{field}' still names local input '{filename}'. "
                             "That authority is outside the selected staging manifest; transfer it "
                             "explicitly and use the existing remote-input route rather than assuming a node path.")


def _read_selected(path, role):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Selected {role} input '{path.name}' is missing or is not a regular file. "
                         "Save its companion file, or stage the declared input on the node first.")
    size = path.stat().st_size
    if size > MAX_SINGLE_BYTES:
        raise ValueError(f"Selected {role} input '{path.name}' is {size:,} bytes; automatic companion "
                         f"staging is limited to {MAX_SINGLE_BYTES:,} bytes per file. Transfer this "
                         "specific input to an owned node folder and use the existing remote-input "
                         "route, or select an unfetched case with a matching saved acquisition recipe.")
    with path.open("rb") as stream:
        payload = stream.read(MAX_SINGLE_BYTES + 1)
    if len(payload) != size:
        raise ValueError(f"Selected {role} input '{path.name}' changed while staging; review again")
    return payload


def _wps_bytes(payload, geog, stage_file, rewrites, *, source):
    from gpuwm.namelist_import import parse_namelist_text
    from gpuwm.wps_domain_ids import domain_ids_from_wps_text, with_domain_ids
    original_text = payload.decode("utf-8-sig")
    tables = parse_namelist_text(original_text)
    maximum = tables.get("share", {}).get("max_dom", [1])
    if len(maximum) != 1:
        raise ValueError("WPS max_dom must declare one domain count")
    domain_ids = domain_ids_from_wps_text(original_text, maximum[0])
    if geog is not None:
        table = tables.setdefault("geogrid", {})
        rewrites.append({"field": "namelist.geogrid.geog_data_path",
                         "before": table.get("geog_data_path"), "after": geog})
        table["geog_data_path"] = [geog]
    for section, key, filename, destination in (
        ("geogrid", "opt_geogrid_tbl_path", "GEOGRID.TBL", "tables/geogrid"),
        ("metgrid", "opt_metgrid_tbl_path", "METGRID.TBL", "tables/metgrid"),
    ):
        if key in tables.get(section, {}):
            values = tables[section][key]
            if len(values) != 1:
                raise ValueError(f"WPS {key} must name one selected table directory for remote staging")
            table_path = Path(values[0])
            if not table_path.is_absolute():
                table_path = source.parent / table_path
            stage_file(table_path / filename, destination + "/" + filename,
                       "WPS authority table")
            tables[section][key] = [destination]

    def value(item):
        if isinstance(item, str):
            return "'" + item.replace("'", "''") + "'"
        if isinstance(item, bool):
            return ".true." if item else ".false."
        return repr(item)

    text = "\n".join("&" + section + "\n" + "\n".join(
        " " + key + " = " + ", ".join(value(v) for v in values) + ","
        for key, values in table.items()) + "\n/" for section, table in tables.items()) + "\n"
    if parse_namelist_text(text) != tables:
        raise ValueError("Selected WPS authority did not round-trip during path relocation")
    return with_domain_ids(text, domain_ids).encode("utf-8")


def build_bundle(plan_path, *, workspace, outdir, geog_root=None,
                 expected_plan_sha256, expected_config_sha256):
    """Run on the client; return only explicitly selected, bounded input bytes."""
    from gpuwm.case_data import forcing_has_glob, resolved_case_data_paths, same_case_data_path
    from gpuwm.experiment import build_experiment_from_config_tables
    from gpuwm.hrrr_route_inputs import route_input_paths
    from gpuwm.runplan import load_plan
    from gpuwm.toml_document import emit_experiment_toml

    workspace = _remote_path(workspace, "node workspace")
    outdir = _remote_path(outdir, "node output directory")
    geog = None if geog_root is None else _remote_path(geog_root, "node geography directory")
    source_plan = Path(plan_path).resolve(strict=True)
    plan_bytes = _read_selected(source_plan, "run-plan")
    if _sha(plan_bytes) != expected_plan_sha256:
        raise ValueError("The saved local run plan changed after review")
    parsed = load_plan(source_plan)
    if parsed.config_path is None:
        raise ValueError("Remote map review needs a saved TOML configuration; save the generated case first")
    config = parsed.config_path.resolve(strict=True)
    config_bytes = _read_selected(config, "configuration")
    if _sha(config_bytes) != expected_config_sha256:
        raise ValueError("The saved local configuration changed after review")
    plan = json.loads(plan_bytes)
    raw = tomllib.loads(config_bytes.decode("utf-8-sig"))
    build_experiment_from_config_tables(raw, source=str(config), base_dir=config.parent)
    if plan.get("fetch") is not None:
        raise ValueError("This plan has a separate explicit fetch argv. Remote map staging currently "
                         "uses the saved configuration's acquisition recipe; review this custom "
                         "plan through the existing remote-input route instead of dropping its arguments.")
    identifier = secrets.token_hex(16)
    data_cache_key = None
    if isinstance(raw.get("fetch"), dict) and raw["fetch"].get("source") and raw["fetch"].get("cycle"):
        from gpuwm.go_cli import managed_download_key
        data_cache_key = managed_download_key(raw["fetch"])
    remote_data = str(PurePosixPath(workspace) / ".arwen-plan-data" / (data_cache_key or identifier))
    remote_inputs = str(PurePosixPath(workspace) / ".arwen-plan-inputs" / identifier)
    files = {}
    blobs = []
    originals = {str(source_plan): _sha(plan_bytes), str(config): _sha(config_bytes)}
    rewrites = []
    expected_downloads = []

    def stage_file(path, name, role, *, transform=None, allow_blob=False, placement="inputs"):
        path = Path(path)
        if not path.is_absolute():
            path = config.parent / path
        name = _name(name)
        if placement == "data" or allow_blob and path.is_file() and path.stat().st_size > MAX_SINGLE_BYTES:
            from gpuwm.remote_input_transfer import describe, MAX_BLOBS, MAX_BUNDLE_BLOB_BYTES
            if transform is not None:
                raise ValueError("Only unchanged raw inputs may use the binary transfer")
            identity = describe(path, role)
            record = {**identity, "name": name, "role": role, "placement": placement}
            if any(item["name"] == name and item["placement"] == placement for item in blobs):
                raise ValueError(f"Selected raw inputs collide at '{name}'")
            blobs.append(record)
            if len(blobs) > MAX_BLOBS or sum(item["size"] for item in blobs) > MAX_BUNDLE_BLOB_BYTES:
                raise ValueError("Selected raw inputs exceed the sixty-four-file / 64 GiB staging limit")
            destination = str(PurePosixPath(remote_data if placement == "data" else remote_inputs) / name)
            rewrites.append({"field": role, "before": identity["source_path"], "after": destination,
                             "basis": "exact selected raw bytes through verified SHA-256 object transfer"})
            return destination
        content = _read_selected(path, role)
        originals[str(path.resolve())] = _sha(content)
        if transform:
            content = transform(content)
        if name in files and files[name]["payload"] != content:
            raise ValueError(f"Selected inputs collide at '{name}'; no files were staged")
        files[name] = {"role": role, "payload": content}
        rewrites.append({"field": role, "before": str(path.resolve()), "after": name,
                         "basis": "selected manifest-bound companion input"})
        return name

    hints = raw.get("fetch")
    if isinstance(hints, dict) and hints.get("out") is not None:
        rewrites.append({"field": "fetch.out", "before": hints["out"], "after": remote_data})
        original_fetch_out = Path(hints["out"]).expanduser()
        if not original_fetch_out.is_absolute():
            original_fetch_out = config.parent / original_fetch_out
        hints["out"] = remote_data
    else:
        original_fetch_out = None

    wps_paths = set()
    if "case_data" in raw:
        data = resolved_case_data_paths(raw["case_data"], base_dir=config.parent, source=str(config))
        if geog is None:
            raise ValueError("This saved case requires geography. Set 'Remote geography folder' "
                             "on the selected node before reviewing its map configuration.")
        rewrites.append({"field": "case_data.geog_root", "before": data.get("geog_root"), "after": geog})
        data["geog_root"] = geog
        entries = data.get("forcing", [])
        entries = entries if isinstance(entries, list) else [entries]
        if not entries or len(entries) > 64:
            raise ValueError("Remote staging requires one to sixty-four selected forcing entries")
        if any(forcing_has_glob(path) for path in entries):
            import glob
            expanded = []
            for pattern in entries:
                if forcing_has_glob(pattern):
                    matches = []
                    for match in glob.iglob(pattern):
                        matches.append(match)
                        if len(matches) > 64:
                            raise ValueError("The selected forcing pattern exceeds sixty-four files")
                    matches.sort()
                    if not matches:
                        raise ValueError(f"Declared forcing pattern '{Path(pattern).name}' has no selected local files or registered acquisition binding")
                    expanded.extend(matches)
                else:
                    expanded.append(pattern)
                if len(expanded) > 64:
                    raise ValueError("The selected forcing pattern exceeds sixty-four files")
            entries = list(dict.fromkeys(expanded))
        managed = False
        if isinstance(hints, dict) and hints.get("source") == "era5" and original_fetch_out is not None:
            from gpuwm.fetch import ERA5_COMBINED_NAMES
            provider = hints.get("era5_provider", "cds")
            expected = ERA5_COMBINED_NAMES.get(provider)
            managed = bool(expected and len(entries) == 1 and
                           same_case_data_path(entries[0], original_fetch_out / expected))
        missing = [path for path in entries if not Path(path).is_file()]
        if missing:
            if not managed or len(missing) != len(entries):
                raise ValueError(f"Declared forcing '{Path(missing[0]).name}' is not available locally "
                                 "and is not the exact output of this case's saved ERA5 acquisition. "
                                 "Correct the selected forcing/acquisition binding or transfer that input to the node.")
            destination = str(PurePosixPath(remote_data) / Path(entries[0]).name)
            data["forcing"] = [destination]
            expected_downloads.append({"role": "forcing", "path": destination,
                                       "source": hints["source"], "recipe": copy.deepcopy(hints)})
            rewrites.append({"field": "case_data.forcing", "before": entries, "after": [destination],
                             "basis": "same declared acquisition output on selected node"})
        else:
            receipt = None if not managed else original_fetch_out / "era5-acquisition.json"
            if managed and (hints.get("era5_product") == "ensemble_members" or receipt.is_file()):
                from gpuwm.remote_input_transfer import describe
                identity = describe(entries[0])
                if receipt.is_symlink() or not receipt.is_file() or receipt.stat().st_size > 1024 * 1024:
                    raise ValueError("Selected cached EDA forcing requires its complete bounded era5-acquisition.json receipt")
                acquisition = json.loads(receipt.read_bytes())
                artifact = acquisition.get("artifact", {})
                if (acquisition.get("schema") != "arwen.era5-acquisition.v1" or acquisition.get("status") != "validated"
                        or artifact.get("name") != Path(entries[0]).name or artifact.get("bytes") != identity["size"]
                        or artifact.get("sha256") != identity["sha256"]):
                    raise ValueError("Selected ERA5 acquisition receipt does not bind the exact cached forcing bytes")
                if hints.get("era5_product") == "ensemble_members":
                    from gpuwm.era5_member import check_member, validate_selection
                    member = validate_selection(product_type=hints["era5_product"], member=hints.get("member"),
                                                cadence=hints.get("cadence", 6), provider=hints.get("era5_provider", "cds"))
                    if acquisition.get("request", {}).get("member") != member or acquisition.get("member_selection", {}).get("member") != member:
                        raise ValueError("Selected cached EDA receipt belongs to a different native ensemble member")
                    check_member(entries[0], member)
                data["forcing"] = [stage_file(entries[0], Path(entries[0]).name, "forcing", placement="data")]
                stage_file(receipt, receipt.name, "ERA5 acquisition receipt", placement="data")
            else:
                data["forcing"] = [stage_file(path, f"forcing/{i:02d}-{Path(path).name}", "forcing", allow_blob=True)
                                   for i, path in enumerate(entries)]
        for key, destination in (("vtable", "inputs/Vtable"), ("wps_namelist", "case.namelist.wps"),
                                 ("water_temperature_overlay", "inputs/water-temperature-overlay.nc")):
            if data.get(key) is not None:
                original = data[key]
                transform = None
                if key == "wps_namelist":
                    wps_paths.add(Path(original).resolve())
                    transform = lambda b, p=Path(original): _wps_bytes(b, geog, stage_file, rewrites, source=p)
                data[key] = stage_file(original, destination, key, transform=transform, allow_blob=key=="water_temperature_overlay")
        if data.get("source_orography") is not None:
            value = data["source_orography"]
            if isinstance(value, dict):
                data["source_orography"] = {
                    key: stage_file(path, f"inputs/orography-{key}.nc", "source orography", allow_blob=True)
                    for key, path in value.items()}
            else:
                data["source_orography"] = stage_file(value, "inputs/orography.nc", "source orography", allow_blob=True)
        raw["case_data"] = data

    for role, path in route_input_paths(config).items():
        if not path.exists() or path.resolve() in wps_paths:
            continue
        suffix = path.name[len(config.stem):]
        transform = (lambda b, p=path: _wps_bytes(b, geog, stage_file, rewrites, source=p)) if role == "wps_namelist" else None
        stage_file(path, "case" + suffix, role, transform=transform)
    if "static" in raw and isinstance(raw["static"].get("highres"), dict):
        table = raw["static"]["highres"]
        if table.get("cache_root") is not None:
            rewrites.append({"field": "static.highres.cache_root", "before": table["cache_root"],
                             "after": remote_data + "/static"})
            table["cache_root"] = remote_data + "/static"
    options = plan.setdefault("run_options", {})
    if "case_data" in raw and options.get("data_dir") is not None:
        raise ValueError("This saved plan names [case_data].forcing directly, so run_options.data_dir "
                         "is unused. Omit that run option; the saved fetch.out and forcing paths "
                         "already bind acquisition to the selected node cache.")
    for key in ("prepared_root", "restart", "wps_namelist"):
        if options.get(key) is not None:
            raise ValueError(f"This plan names existing '{key}' input. Use the selected node's existing "
                             "prepared/resume route; automatic map staging cannot transfer that large authority safely.")
    for key, replacement in (("geog_root", geog), ("data_dir", remote_data)):
        if options.get(key) is not None or (key == "geog_root" and geog is not None) or (key == "data_dir" and data_cache_key is not None and "case_data" not in raw):
            if replacement is None:
                raise ValueError("Set the selected node's geography folder before relocating this plan")
            rewrites.append({"field": "run_options." + key, "before": options.get(key), "after": replacement})
            options[key] = replacement
    plan["config"] = {"path": "case.toml"}
    rewrites.append({"field": "output_root", "before": plan.get("output_root"), "after": outdir})
    plan["output_root"] = outdir
    allowed_paths = [remote_data, remote_inputs, outdir] + ([] if geog is None else [geog])
    _assert_relocated(raw, allowed_paths)
    _assert_relocated(plan, allowed_paths, "plan")
    files["case.toml"] = {"role": "configuration", "payload": emit_experiment_toml(raw).encode("utf-8")}
    files["plan.json"] = {"role": "run-plan", "payload": _encoded(plan)}
    if len(files) > MAX_FILES or sum(len(v["payload"]) for v in files.values()) > MAX_INPUT_BYTES:
        raise ValueError("Selected configuration and companion files exceed the 64 KiB staging bundle. "
                         "Stage this exact input set on the node and use its existing remote-input route.")
    for original, digest in originals.items():
        if _sha(_read_selected(Path(original), "selected")) != digest:
            raise ValueError(f"Selected input '{Path(original).name}' changed while preparing the manifest")
    bundle = {"schema": BUNDLE_SCHEMA, "id": identifier, "workspace": workspace, "outdir": outdir,
              "geog_root": geog, "source": {"plan_path": str(source_plan), "plan_sha256": expected_plan_sha256,
                  "config_path": str(config), "config_sha256": expected_config_sha256},
              "source_inputs": originals, "rewrites": rewrites, "expected_downloads": expected_downloads,
              "files": [{"name": name, "role": value["role"], "size": len(value["payload"]),
                         "sha256": _sha(value["payload"]),
                         "data": base64.b64encode(value["payload"]).decode("ascii")}
                        for name, value in sorted(files.items())]}
    if blobs or data_cache_key is not None:
        from gpuwm.remote_input_transfer import source_blobs, verify_sources
        bundle.update(schema=BLOB_BUNDLE_SCHEMA, blobs=blobs, data_cache_key=data_cache_key)
        verify_sources(source_blobs(bundle))
    bundle["sha256"] = _sha(_encoded(bundle))
    if len(_encoded(bundle)) > 120 * 1024:
        raise ValueError("Selected staging manifest exceeds the bounded SSH request size")
    return bundle


def _private_root(workspace, name=".arwen-plan-inputs"):
    root = workspace / name
    try:
        root.mkdir(mode=0o700)
    except FileExistsError:
        pass
    info = root.lstat()
    if root.is_symlink() or not root.is_dir() or (hasattr(os, "getuid") and
            (info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) & 0o077)):
        raise ValueError("Remote plan inputs need a private owned directory without a symlink")
    return root


def stage(bundle, workspace):
    """Worker operation. Publish a complete immutable small-input manifest."""
    from gpuwm import remote_worker as rw
    if not isinstance(bundle, dict) or bundle.get("schema") not in (BUNDLE_SCHEMA, BLOB_BUNDLE_SCHEMA):
        raise ValueError("unsupported staged plan bundle")
    expected_keys = {"schema", "id", "workspace", "outdir", "geog_root", "source", "source_inputs",
                     "rewrites", "expected_downloads", "files", "sha256"}
    if bundle["schema"] == BLOB_BUNDLE_SCHEMA:
        expected_keys.update({"blobs", "data_cache_key"})
        if bundle["data_cache_key"] is not None and (not isinstance(bundle["data_cache_key"], str) or not SHA.fullmatch(bundle["data_cache_key"])):
            raise ValueError("Invalid managed acquisition cache key")
    if set(bundle) != expected_keys or not ID.fullmatch(bundle.get("id", "")):
        raise ValueError("staged plan bundle has invalid identity or fields")
    digest = bundle.get("sha256")
    unsigned = {key: value for key, value in bundle.items() if key != "sha256"}
    if not isinstance(digest, str) or not SHA.fullmatch(digest) or _sha(_encoded(unsigned)) != digest:
        raise ValueError("staged plan manifest SHA-256 does not match")
    if Path(bundle["workspace"]).resolve() != workspace:
        raise ValueError("staged plan belongs to a different remote workspace")
    entries = bundle["files"]
    if not isinstance(entries, list) or not 2 <= len(entries) <= MAX_FILES:
        raise ValueError("invalid staged input count")
    decoded = {}
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {"name", "role", "size", "sha256", "data"}:
            raise ValueError("invalid staged input record")
        name = _name(entry["name"])
        if name in decoded:
            raise ValueError("duplicate staged input name")
        payload = base64.b64decode(entry["data"], validate=True)
        if (len(payload) != entry["size"] or len(payload) > MAX_SINGLE_BYTES
                or _sha(payload) != entry["sha256"]):
            raise ValueError(f"Staged input '{name}' failed its size or SHA-256 check")
        decoded[name] = payload
    if not {"case.toml", "plan.json"} <= decoded.keys() or sum(map(len, decoded.values())) > MAX_INPUT_BYTES:
        raise ValueError("staged plan is incomplete or too large")
    blob_paths = _validated_blobs(bundle, workspace, require=True)
    if any(placement == "inputs" and name in decoded for placement, name, _path, _entry in blob_paths):
        raise ValueError("Raw input collides with a selected configuration companion")
    root = _private_root(workspace)
    destination = root / bundle["id"]
    if destination.exists() or destination.is_symlink():
        saved, _ = read_bundle(workspace, bundle["id"], digest)
        if saved != bundle:
            raise ValueError("staged bundle ID was already used with different inputs")
        return {"bundle_id": bundle["id"], "bundle_sha256": digest}
    destination.mkdir(mode=0o700)
    data = _data_directory(bundle, workspace)
    for name, payload in decoded.items():
        path = destination / name
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    from gpuwm.fetch_guard import hold
    with hold("fetch-out", data, timeout_s=20, progress=lambda *_: None):
        for placement, name, source, _entry in blob_paths:
            target = (destination if placement == "inputs" else data) / name
            target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            if target.exists() or target.is_symlink():
                if target.is_symlink() or not target.is_file() or target.stat().st_size != _entry["size"] or rw._file_sha(target) != _entry["sha256"]:
                    raise ValueError(f"Existing managed input '{name}' differs from the selected raw bytes; no file was replaced")
            else:
                os.link(source, target)
    rw._write(destination / "bundle.json", bundle)
    return {"bundle_id": bundle["id"], "bundle_sha256": digest}


def _data_directory(bundle, workspace):
    key = bundle.get("data_cache_key") or bundle["id"]
    if not isinstance(key, str) or not (SHA.fullmatch(key) or ID.fullmatch(key)):
        raise ValueError("Invalid owned acquisition directory identity")
    path = _private_root(workspace, ".arwen-plan-data") / key
    path.mkdir(mode=0o700, exist_ok=True)
    info = path.lstat()
    if path.is_symlink() or not path.is_dir() or (hasattr(os, "getuid") and (info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) & 0o077)):
        raise ValueError("Managed acquisition cache must remain private, owned and without symlinks")
    return path


def _validated_blobs(bundle, workspace, *, require):
    from gpuwm.remote_input_transfer import object_path, MAX_BLOBS, MAX_BUNDLE_BLOB_BYTES
    blobs = bundle.get("blobs", [])
    if not isinstance(blobs, list) or len(blobs) > MAX_BLOBS:
        raise ValueError("Staged raw-input inventory is invalid")
    total, seen, result, cached = 0, set(), [], {}
    for entry in blobs:
        if not isinstance(entry, dict) or set(entry) != {"source_path", "size", "sha256", "name", "role", "placement"}:
            raise ValueError("Staged raw-input descriptor is invalid")
        placement, name = entry["placement"], _name(entry["name"])
        if placement not in ("inputs", "data") or (placement, name) in seen:
            raise ValueError("Staged raw inputs have an invalid or duplicate destination")
        if not isinstance(entry["source_path"], str) or not isinstance(entry["role"], str):
            raise ValueError("Staged raw input lacks its selected source identity")
        seen.add((placement, name))
        identity = (entry["sha256"], entry["size"])
        source = cached.get(identity)
        if source is None:
            source = object_path(workspace, {key: entry[key] for key in ("size", "sha256")}, require=require)
            cached[identity] = source
        total += entry["size"]
        if total > MAX_BUNDLE_BLOB_BYTES:
            raise ValueError("Staged raw inputs exceed the 64 GiB limit")
        result.append((placement, name, source, entry))
    return result


def read_bundle(workspace, identifier, expected):
    from gpuwm import remote_worker as rw
    if not isinstance(identifier, str) or not ID.fullmatch(identifier):
        raise ValueError("invalid remote plan bundle ID")
    root = _private_root(workspace)
    directory = root / identifier
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("the reviewed remote plan bundle is unavailable; review again")
    if hasattr(os, "getuid") and (directory.stat().st_uid != os.getuid()
            or stat.S_IMODE(directory.stat().st_mode) & 0o077):
        raise ValueError("the reviewed plan bundle is no longer private and owned")
    bundle = rw._json(directory / "bundle.json")
    if bundle.get("id") != identifier or bundle.get("sha256") != expected:
        raise ValueError("remote plan bundle identity changed; review again")
    unsigned = {key: value for key, value in bundle.items() if key != "sha256"}
    if _sha(_encoded(unsigned)) != expected:
        raise ValueError("remote plan manifest changed; review again")
    for entry in bundle["files"]:
        path = directory / _name(entry["name"])
        if (path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(directory.resolve())
                or any(parent.is_symlink() for parent in path.parents if parent != directory and directory in parent.parents)
                or rw._file_sha(path) != entry["sha256"]):
            raise ValueError(f"Reviewed remote input '{entry['name']}' changed; review again")
    for placement, name, source, entry in _validated_blobs(bundle, workspace, require=True):
        base = directory if placement == "inputs" else _data_directory(bundle, workspace)
        path = base / name
        if (base.is_symlink() or path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(base.resolve())
                or any(parent.is_symlink() for parent in path.parents if parent != base and base in parent.parents)
                or path.stat().st_size != entry["size"] or (not os.path.samefile(source, path) and rw._file_sha(path) != entry["sha256"])):
            raise ValueError(f"Reviewed raw input '{name}' changed; review again")
    return bundle, directory


def hardware_probe(*, sizing=None, measure_sizing=True):
    from gpuwm.runplan import probe_environment
    from gpuwm.target_hardware import sizing_from_probe, host_memory_snapshot
    if measure_sizing and sizing is None:
        from gpuwm.core.preflight import device_memory_probe_subprocess
        sizing = sizing_from_probe(device_memory_probe_subprocess())
    return {"measured_unix_ms": int(time.time() * 1000),
            **probe_environment(readiness=False), "sizing": sizing, "host_memory": host_memory_snapshot()}


def memory_review(config, *, experiment=None, cadence=None):
    """The existing launch gate, on this node, including the ingest phase."""
    from gpuwm.go_cli import memory_gate
    gate = memory_gate({"config": str(config), "cadence": cadence}, experiment=experiment)
    phases = gate["phases"]
    forecast = phases.forecast
    from gpuwm.target_hardware import sizing_from_probe
    sizing = sizing_from_probe(gate.get("device_probe"))
    road = getattr(phases, "tree_road", None)
    road_json = None if road is None else road.to_json()
    options = getattr(experiment, "tiles", None)
    if options is None:
        from gpuwm.core.streaming import StreamingOptions
        options = StreamingOptions.from_mapping(tomllib.loads(Path(config).read_text()).get("tiles"), source=str(config))
    # Read the already-priced gate result. A second probe or a separate
    # estimate could describe a different device or auto-tiling decision.
    breakdown = {
        "schema": "arwen.memory-breakdown.v1",
        "binding_phase": phases.binding_phase,
        "peak_envelope_bytes": int(phases.peak_envelope_bytes),
        "forecast_envelope_bytes": int(phases.forecast_envelope_bytes),
        "ingest_envelope_bytes": (None if phases.ingest_envelope_bytes is None
                                  else int(phases.ingest_envelope_bytes)),
        "streamed_forecast": bool(phases.streamed_forecast),
        "resident_forecast_envelope_bytes": int(getattr(phases, "resident_forecast_envelope_bytes", None) or forecast.peak_envelope_bytes),
        "tree_road": road_json,
        "forecast": {
            "allocation_scope": ("resident_reference" if phases.streamed_forecast else "resident_execution"),
            **{key: int(getattr(forecast, key)) for key in (
                "resident_bytes", "workspace_bytes", "transient_peak_bytes", "subtotal_bytes",
                "alloc_estimate_bytes", "non_pool_device_bytes", "peak_envelope_bytes",
                "scratch_arena_bytes", "scratch_arena_saved_bytes",
                "dycore_state_workspace_bytes", "dycore_state_saved_bytes")},
            "terms": forecast.peak_envelope_terms(),
        },
        "domains": [{"grid_id": domain.grid_id,
                     "resident_bytes": int(domain.resident_bytes),
                     "transient_bytes": int(domain.transient_bytes),
                     "by_category": {key: int(domain.category_bytes(key)) for key in
                         ("state", "physics", "scratch", "lbc", "nest", "diagnostic", "sase", "transient")}}
                    for domain in forecast.domains],
    }
    return {"measured": gate.get("free_bytes") is not None,
            "execution": {"schema": "arwen.execution-memory.v1", "configured_mode": options.mode,
                          "configured_tiles": options.to_mapping(), "streamed_forecast": bool(phases.streamed_forecast),
                          "selected_forecast_envelope_bytes": int(phases.forecast_envelope_bytes),
                          "resident_reference_bytes": int(getattr(phases, "resident_forecast_envelope_bytes", None) or forecast.peak_envelope_bytes),
                          "tree_road": road_json, "planner_refusal": None if road is None else road.refusal},
            "free_bytes": gate.get("free_bytes"), "budget_bytes": gate.get("budget_bytes"),
            "peak_envelope_bytes": int(phases.peak_envelope_bytes),
            "refuse": bool(gate["refuse"]), "warn": bool(gate["warn"]),
            "verdict": gate["verdict"], "probe_reason": gate.get("probe_reason"),
            "ingest_priced": bool(phases.ingest_priced), "breakdown": breakdown, "sizing": sizing,
            "measured_unix_ms": int(time.time() * 1000)}


def review(request, workspace):
    from gpuwm import remote_worker as rw
    from gpuwm.runplan import load_plan, resolve_plan
    bundle, directory = read_bundle(workspace, request.get("bundle_id"), request.get("expected_bundle_sha256"))
    outdir = rw._absolute(bundle["outdir"], "reviewed output directory")
    if outdir.exists() or outdir.is_symlink() or not outdir.parent.is_dir():
        raise ValueError("Reviewed output must be a new directory beneath an existing node output folder")
    if bundle["geog_root"] is not None and not Path(bundle["geog_root"]).is_dir():
        raise ValueError("The selected node geography folder is unavailable; correct its node profile")
    plan_path, config = directory / "plan.json", directory / "case.toml"
    resolution, experiment, _data = resolve_plan(load_plan(plan_path), require_inputs=False)
    raw = tomllib.loads(config.read_text())
    acquisition_readiness = None
    if any(item.get("source") == "era5" and item.get("recipe", {}).get("era5_provider", "cds") == "cds"
           for item in bundle["expected_downloads"]):
        from gpuwm.cds_credentials import acquisition_readiness as cds_readiness
        acquisition_readiness = cds_readiness()
        if not acquisition_readiness["ready"]:
            raise ValueError("Selected node cannot acquire ERA5 from CDS: " + acquisition_readiness["message"])
    memory = memory_review(config, experiment=experiment, cadence=raw.get("fetch", {}).get("cadence"))
    hashes = {entry["name"]: entry["sha256"] for entry in bundle["files"]}
    hashes.update({entry["placement"] + "/" + entry["name"]: entry["sha256"] for entry in bundle.get("blobs", [])})
    wps = hashes.get("case.namelist.wps")
    result = {"bundle_id": bundle["id"], "bundle_sha256": bundle["sha256"],
              "plan": str(plan_path), "plan_sha256": hashes["plan.json"],
              "config": str(config), "config_sha256": hashes["case.toml"],
              "wps_sha256": wps, "input_sha256": _sha(_encoded(hashes)),
              "outdir": str(outdir), "geog_root": bundle["geog_root"], "runtime": rw.runtime(),
              "probe": hardware_probe(sizing=memory.get("sizing"), measure_sizing=False),
              "resolution": resolution, "memory": memory,
              "source": bundle["source"], "source_inputs": bundle["source_inputs"], "path_rewrites": bundle["rewrites"],
              "expected_downloads": bundle["expected_downloads"]}
    if acquisition_readiness is not None:
        result["acquisition_readiness"] = acquisition_readiness
    if bundle.get("blobs"):
        from gpuwm.remote_input_transfer import source_blobs
        result["source_blobs"] = source_blobs(bundle)
    return result, bundle, directory


def launch(request, workspace):
    from gpuwm import remote_worker as rw
    review_value, bundle, directory = review(request, workspace)
    if bundle.get("blobs"):
        expected = _sha(_encoded(review_value["source_blobs"]))
        if request.get("expected_source_blobs_sha256") != expected:
            raise ValueError("The selected large local inputs need fresh verification by the matching TUI before launch")
    for name in ("plan", "config", "input"):
        expected = request.get(f"expected_{name}_sha256")
        if expected is None or expected != review_value[f"{name}_sha256"]:
            raise ValueError(f"remote {name} changed after review; review again")
    memory = review_value["memory"]
    if not memory["measured"] or memory["refuse"]:
        raise ValueError("Remote memory review is not launch-ready: " + memory["verdict"])
    snapshots = {entry["name"]: (directory / entry["name"]).read_bytes() for entry in bundle["files"]}
    sources = {str(directory / name): payload for name, payload in snapshots.items()}
    review_value.update({"argv": [os.sys.executable, "-I", "-u", "-m", "gpuwm.cli", "run-plan", str(directory / "plan.json")],
                         "cwd": str(directory), "geog_root": bundle["geog_root"], "products": None,
                         "prepared_root": None, "wps_namelist": None, "parent_job": None,
                         "checkpoint": None, "inputs": {path: _sha(payload) for path, payload in sources.items()}})
    review_value["external_inputs"] = {
        str((directory if entry["placement"] == "inputs" else _data_directory(bundle, workspace)) / entry["name"]): entry["sha256"]
        for entry in bundle.get("blobs", [])}
    return rw._launch_review(request, workspace, review_value, sources, snapshots)
