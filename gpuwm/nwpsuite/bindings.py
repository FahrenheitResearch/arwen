"""Materialize planner instructions without putting engine choices in the loop.

Bindings are a separate deployment document, tied to the resolved suite hash.
They supply prepared authority and existing CPU doors, never measured admission.
"""
from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
import re
import sys
import tomllib

from gpuwm.cycle.ledger import write_receipt_atomically
from .prepare import _sha256 as _file_hash

# Row ids carry hyphens (cam-det), so artifact names such as
# {artifacts.cam-det.12.0.run_plan} do too; without "-" those stayed literal in
# the launched command and the forecast door was handed a brace string.
TOKEN = re.compile(r"\{([a-z_][a-z0-9_.-]*)\}")
#: Where a row's prepare stage publishes each cycle's prepared authority. It is
#: attempt-independent, so the forecast finds the last successful preparation.
PREPARED_RECEIPT = "{root}/prepared/%s/{cycle_compact}/suite-prepared.json"
#: Gaps a deployment closes by binding a row; all of them are named together.
UNBOUND = ("suite_execution_unbound", "suite_launcher_missing", "suite_prepare_base_missing", "suite_prepared_lbc",
           "suite_lagged_unbound", "suite_da_background")
#: The prepare door's bundle is registered as history, so retention removes it
#: with the forecast output instead of letting prepared cycles fill the disk.
PREPARED_GLOBS = ["prepared/" + "/".join(["*"] * depth) for depth in range(1, 9)]
BOX_KEYS = {"launcher_prefix", "health_argv"}
RUN_OPTIONS = {"prepared_root", "wps_namelist", "render_products", "verify_visuals", "keep_checkpoints"}
ROW_KEYS = {"bindings", "source_bindings", "source_receipt", "base_config", "run_options",
            "readiness_argv", "readiness_timeout", "receipts", "events", "receipt_globs",
            "artifact_globs", "door_kind", "cwd", "resume_argv", "degrade", "da_argv", "cpu_stages"}


class SourcePending(OSError):
    """The prepared source has not published its atomic authority receipt yet."""


def runtime_id(job):
    phase = job.get("phase", "forecast")
    member = int(job.get("member", 0))
    return job["row"] + (".da" if phase == "da-shadow" else f".member-{member}" if member else "")


def bind_plan(plan, document=None):
    """Keep one row per trajectory and one launch definition per UTC hour."""
    plan = deepcopy(plan)
    document = document or {}
    if document:
        if document.get("schema") != "gpuwm.suite.bindings.v1":
            raise ValueError("suite_bindings_schema: expected gpuwm.suite.bindings.v1")
        if document.get("suite_sha256") != plan.get("suite_sha256"):
            raise ValueError("suite_bindings_hash: deployment belongs to a different resolved suite")
        unknown = set(document) - {"schema", "suite_sha256", "boxes", "rows", "on_state_change"}
        if unknown:
            raise ValueError(f"suite_bindings_unknown: {sorted(unknown)}")
        if any(not isinstance(document.get(key, {}), dict) for key in ("rows", "boxes")):
            raise ValueError("suite_bindings_shape: rows and boxes must be keyed objects")
    rows = {r["id"]: r for r in plan["rows"]}
    box_ids = {b["id"] for b in plan["boxes"]}
    if set(document.get("rows", {})) - rows.keys() or set(document.get("boxes", {})) - box_ids:
        raise ValueError("suite_bindings_identity: unknown row or box")
    for box in plan["boxes"]:
        supplied = document.get("boxes", {}).get(box["id"], {})
        if set(supplied) - BOX_KEYS:
            raise ValueError("suite_box_binding: only launcher_prefix and health_argv may supplement inventory")
        box.update(supplied)
        if box.get("host", "local") not in {"local", "localhost", "127.0.0.1"}:
            raise ValueError(f"suite_transport_unbound: {box['id']} is remote; local launch tokens and PID identities cannot control another host")
    launches = {}
    job_ids = {j["id"]: runtime_id(j) for j in plan.get("jobs", [])}
    unbound = {}
    for job in plan.get("jobs", []):
        try:
            rid = runtime_id(job)
            policy = deepcopy(document.get("rows", {}).get(job["row"], {}))
            if set(policy) - ROW_KEYS:
                raise ValueError(f"suite_row_binding: {job['row']}: unknown {sorted(set(policy) - ROW_KEYS)}")
            if set(policy.get("run_options", {})) - RUN_OPTIONS:
                raise ValueError("suite_run_options: prepared and output controls only; forecast changes must pass planner admission")
            settings = deepcopy(job.get("settings", rows[job["row"]].get("settings", {})))
            row = dict(settings, **policy)
            row.pop("cpu_stages", None)
            cycle = settings.get("cycle", {})
            row.update(id=rid, suite_row=job["row"], member=job.get("member", 0), box=job["box"],
                       cards=len(job["card_ids"]), card_ids=job["card_ids"], env=job.get("env", {}),
                       every=cycle.get("every", "6h"),
                       # Planner times are block-relative; runtime cycle is this job's hour.
                       deadline=job["deadline_s"] - job.get("ready_offset_s", 0),
                       start_offset_s=max(0, job["start_s"] - job.get("ready_offset_s", 0)),
                       runtime_seconds=job["end_s"] - job["start_s"],
                       dependencies=[job_ids[d] for d in job.get("dependencies", [])],
                       argv=job["argv"], materialization=job)
            row.setdefault("sources", job.get("source_selection", {}).get("ic", []))
            if row["cards"] and not next(b for b in plan["boxes"] if b["id"] == row["box"]).get("launcher_prefix"):
                unbound["box " + row["box"]] = [f"suite_launcher_missing: {row['box']} GPU reservations require a box queue prefix"]
            row.setdefault("bindings", {})
            row["bindings"].setdefault("prepared.outdir", "{out}")
            row["bindings"].update({"prepared." + k: v for k, v in settings.get("prepared", {}).items()
                                    if "prepared." + k not in row["bindings"]})
            if job.get("phase") == "da-shadow":
                row.update(kind="da", argv=policy.get("da_argv", []))
                background = job.get("background_dependency", {})
                if background.get("row"):
                    parent = rows.get(background["row"])
                    if not parent:
                        raise ValueError(f"suite_da_background: {job['id']}: missing previous-cycle row")
                    period = parent.get("settings", {}).get("cycle", {}).get("every", "6h")
                    row["previous_dependencies"] = [{"row": runtime_id(j), "every": period}
                        for j in plan["jobs"] if j["row"] == background["row"] and
                        j["phase"] == "forecast" and j["cycle_hour"] == job["cycle_hour"]]
                    row["previous_dependencies"] = list({x["row"]: x for x in row["previous_dependencies"]}.values())
                    if not row["previous_dependencies"]:
                        raise ValueError(f"suite_da_background: {job['id']}: the background row runs at no matching cycle hour")
            if not row["argv"]:
                raise ValueError(f"suite_execution_unbound: {job['id']} lacks a door command")
            generated = set()
            prepare = None
            if job.get("config_edits"):
                # The forecast's own fetch source is the selected initial source
                # unless the deployment maps it to another fetch door source.
                row["bindings"].setdefault("selected.fetch.source", "{selected.ic}")
                if not row.get("run_options", {}).get("prepared_root"):
                    # Automatic preparation: a zero-card prepare stage fetches and
                    # prepares this cycle ahead of the forecast reservation, so the
                    # forecast never prepares while it holds queued cards.
                    prepare = auto_prepare(job, row, rid, policy, plan)
                    row.update(auto_prepared=True, source_receipt=PREPARED_RECEIPT % rid,
                               dependencies=row["dependencies"] + [prepare["id"]])
                    row.pop("base_config", None)
                elif not row.get("base_config"):
                    raise ValueError(f"suite_execution_unbound: {job['id']} binds prepared_root without the base_config it was prepared from")
                generated = {"derived_config", "prepared.root", "prepared.wps_namelist",
                             *(b for b in job.get("bindings", []) if b.endswith((".config", ".run_plan")))}
                if row["argv"][:2] == ["gpuwm", "go"]:
                    # The go door forecasts a prepared bundle only when it is
                    # named; with --cycle and no bundle it fetched and prepared
                    # inside the queued forecast while holding its cards.
                    if not prepare and not row.get("run_options", {}).get("wps_namelist"):
                        raise ValueError(f"suite_execution_unbound: {job['id']} binds prepared_root without its wps_namelist; the go door binds both")
                    argv = list(row["argv"])
                    if "--cycle" in argv:
                        del argv[argv.index("--cycle"):argv.index("--cycle") + 2]
                    row["argv"] = argv[:3] + ["--prepared-root", "{prepared.root}",
                                              "--wps-namelist", "{prepared.wps_namelist}"] + argv[3:]
                    # The derived config sits inside the prepared bundle, and
                    # gpuwm go writes beside its config by default, so with no
                    # output directory it refused the forecast as overlapping
                    # its own prepared input (2.9.1 verify, live on a card).
                    if "--outdir" not in row["argv"]:
                        row["argv"] += ["--outdir", "{out}"]
            if job.get("manifest_selection"):
                # The public one-member manifest format must be supplied by its door.
                # Do not copy member zero into every distinct trajectory.
                selected = policy.get("bindings", {}).get(f"artifacts.{job['row']}.{job['cycle_hour']:02d}.{job['member']}.manifest")
                if not selected:
                    raise ValueError(f"suite_execution_unbound: {job['id']} needs its own one-member manifest")
            available = {"root", "row", "cycle", "cycle_hour", "cycle_compact", "start_time", "attempt", "out", "work", "cards", "source", "member",
                         "selected.ic", *row["bindings"], *generated}
            available.update(k for values in row.get("source_bindings", {}).values() for k in values)
            required = set(job.get("bindings", [])) | {b for value in row["argv"] for b in TOKEN.findall(value)}
            missing = required - available
            if missing:
                raise ValueError(f"suite_execution_unbound: {job['id']}: {sorted(missing)}")
            for source in row["sources"]:
                source_available = available - {k for values in row.get("source_bindings", {}).values() for k in values}
                source_available.update(row["bindings"])
                source_available.update(row.get("source_bindings", {}).get(source, {}))
                if required - source_available:
                    raise ValueError(f"suite_execution_unbound: {job['id']} source {source}: {sorted(required - source_available)}")
            if row["sources"] and job.get("source_selection", {}).get("requires_prepared_source_receipt") and not row.get("source_receipt"):
                raise ValueError(f"suite_execution_unbound: {job['id']} needs source_receipt; prevents launching another source or cycle")
            hour = str(job["cycle_hour"])
            if prepare is not None:
                stage = launches.setdefault(prepare["id"], dict(prepare, variants={}, cycle_hours=[]))
                if hour in stage["variants"]:
                    raise ValueError(f"suite_reservation_duplicate: {prepare['id']} {hour}")
                stage["variants"][hour] = prepare
                stage["cycle_hours"].append(job["cycle_hour"])
            target = launches.setdefault(rid, dict(row, variants={}, cycle_hours=[]))
            if target["suite_row"] != job["row"]:
                raise ValueError(f"suite_row_identity: generated member id collides with {rid}")
            if hour in target["variants"]:
                raise ValueError(f"suite_reservation_duplicate: {job['id']}")
            target["variants"][hour] = row
            target["cycle_hours"].append(job["cycle_hour"])
        except ValueError as error:
            # Name every row a deployment still has to bind, not only the first.
            if not str(error).startswith(UNBOUND):
                raise
            # One line per row and gap, not one per cycle hour and member.
            message = str(error).replace(job["id"], job["row"] + ("" if job.get("phase", "forecast") == "forecast" else " " + job["phase"]))
            if message not in unbound.setdefault(job["row"], []):
                unbound[job["row"]].append(message)
    bound = list(launches.values())
    for rid, summary in rows.items():
        try:
            trajectories = [r for r in bound if r.get("suite_row") == rid and r.get("kind") not in {"da", "prepare"}]
            policy = document.get("rows", {}).get(rid, {})
            previous = [r["id"] for r in trajectories]
            settings = summary.get("settings", {})
            lagged = settings.get("members", {}).get("lagged", 0) if isinstance(settings.get("members"), dict) else 0
            reuse = []
            # A disabled row has no trajectories and so nothing to assemble.
            if lagged and trajectories:
                parent = rows.get(settings.get("from"))
                if not parent:
                    raise ValueError(f"suite_lagged_unbound: {rid} has no parent receipt row")
                parent_rows = [r for r in bound if r.get("suite_row") == parent["id"] and
                               r.get("materialization", {}).get("phase") == "forecast"]
                if len(parent_rows) != 1:
                    raise ValueError(f"suite_lagged_unbound: {rid} needs one deterministic hourly parent; prevents duplicating a roster")
                period = cadence(parent_rows[0])
                reuse = [{"row": parent_rows[0]["id"], "offset_s": index * period} for index in range(1, lagged + 1)]
                if not any(s.get("kind") == "products" for s in policy.get("cpu_stages", [])):
                    raise ValueError(f"suite_lagged_unbound: {rid} needs a CPU receipt assembly consumer")
            for stage in policy.get("cpu_stages", []):
                stage = deepcopy(stage)
                if stage.get("kind") not in {"products", "verify"} or not stage.get("argv"):
                    raise ValueError("suite_cpu_stage: only existing products and verify doors with argv are admitted")
                stage.update(id=rid + "." + stage["kind"], suite_row=rid, cards=0,
                             box=summary["box"], dependencies=previous,
                             every=min((cadence(r) for r in trajectories), default=21600),
                             cycle_hours=sorted({h for r in trajectories for h in r["cycle_hours"]}))
                if reuse and stage["kind"] == "products":
                    if not any("{lagged_manifest}" in token for token in stage["argv"]):
                        raise ValueError(f"suite_lagged_unbound: {rid} products argv must consume {{lagged_manifest}}; prevents dropping reused members")
                    stage["reuse"] = reuse
                bound.append(stage)
                previous = [stage["id"]]
            for kind in ("products", "verify"):
                if (settings.get(kind) or plan.get("defaults", {}).get(kind)) and trajectories and not any(
                        r.get("suite_row") == rid and r.get("kind") == kind for r in bound):
                    raise ValueError(f"suite_execution_unbound: {rid} lacks requested {kind} CPU door")
        except ValueError as error:
            if not str(error).startswith(UNBOUND):
                raise
            if str(error) not in unbound.setdefault(rid, []):
                unbound[rid].append(str(error))
    if unbound:
        # Name every row a deployment still has to bind, not only the first.
        raise ValueError(" | ".join(message for messages in unbound.values() for message in messages))
    plan.update(rows=bound, execution_ready=True,
                retention=plan.get("defaults", {}).get("retention", plan.get("retention", {})))
    if document.get("on_state_change"):
        plan["on_state_change"] = document["on_state_change"]
    return plan


def auto_prepare(job, row, rid, policy, plan):
    """The zero-card prepare stage for one derived-config forecast reservation.

    Its configuration is derived from the row's base (the deployment's
    base_config, else the suite recipe) with the planner's own config edits,
    and ``gpuwm go --prepare-only`` prepares it on the CPU. The forecast
    depends on it and binds the published bundle from its receipt.
    """
    from .registry import ROOT
    base = policy.get("base_config") or str(ROOT / job.get("recipe_reference", ""))
    if not policy.get("base_config") and not Path(base).is_file():
        raise ValueError(f"suite_prepare_base_missing: {job['id']}: {base} is not installed; "
                         "name a base_config for this row in the deployment bindings")
    lbc = job.get("source_selection", {}).get("lbc", [])
    for source in row.get("sources", []):
        fetch = row.get("source_bindings", {}).get(source, {}).get(
            "selected.fetch.source", row["bindings"].get("selected.fetch.source"))
        fetch = source if fetch == "{selected.ic}" else fetch
        if lbc and fetch not in lbc:
            raise ValueError(f"suite_prepared_lbc: {job['id']}: gpuwm go prepares initial and boundary conditions "
                             f"from its one [fetch] source ({fetch}), which is absent from the row's boundary ladder "
                             f"{lbc}; prevents a forecast forced by boundaries the suite did not name")
    stage = {k: row[k] for k in ("suite_row", "member", "box", "every", "deadline", "sources", "bindings",
                                  "source_bindings", "env") if k in row}
    stage.update(id=rid + ".prepare", kind="prepare", cards=0, card_ids=[], env={}, start_offset_s=0,
                 runtime_seconds=0, dependencies=[], base_config=base, door_kind="command",
                 priority=row.get("priority", settings_priority(job)),
                 receipts=[PREPARED_RECEIPT % rid], artifact_globs={"history": PREPARED_GLOBS},
                 materialization=dict(job, phase="prepare"),
                 argv=[sys.executable, "-m", "gpuwm.nwpsuite.prepare", "--config", "{derived_config}",
                       "--outdir", "{work}/prepared", "--receipt", PREPARED_RECEIPT % rid,
                       "--suite-sha256", str(plan.get("suite_sha256")), "--row", row["suite_row"],
                       "--cycle", "{cycle}", "--source", "{selected.ic}",
                       *(["--go-forecast"] if row["argv"][:2] == ["gpuwm", "go"] else [])])
    stage["bindings"] = dict(stage.get("bindings", {}))
    return stage


def settings_priority(job):
    return job.get("settings", {}).get("priority", 2)


def row_at(row, cycle):
    from .run import utc
    variants = row.get("variants")
    return variants[str(utc(cycle).hour)] if variants else row


def is_due(row, cycle):
    from .run import utc
    from .retention import duration
    moment = utc(cycle)
    if "variants" in row:
        row = row["variants"].get(str(moment.hour))
        if row is None:
            return False
    return (int(moment.timestamp()) % int(duration(row.get("every", "6h"))) == 0 and
            ("cycle_hours" not in row or moment.hour in row["cycle_hours"]))


def cadence(row):
    from .retention import duration
    return min(duration(v.get("every", "6h")) for v in row.get("variants", {"default": row}).values())


def resolve_context(row, context):
    from .run import expand, utc
    context = dict(context, member=row.get("member", 0), cycle_hour=utc(context["cycle"]).strftime("%Y-%m-%dT%H"),
                   lagged_manifest=str(Path(context["work"]) / "lagged-manifest.json"),
                   start_time=context["cycle"].replace("T", "_").removesuffix("Z"),
                   **{"selected.ic": context["source"]})
    values = dict(row.get("bindings", {}), **row.get("source_bindings", {}).get(context["source"], {}))
    while values:
        progress = False
        for key, value in list(values.items()):
            if not isinstance(value, str) or set(TOKEN.findall(value)) <= context.keys():
                context[key] = expand([value], context)[0] if isinstance(value, str) else value
                del values[key]
                progress = True
        if not progress:
            raise ValueError(f"suite_binding_cycle: unresolved {sorted(values)}")
    return context


def materialize(row, context, plan, *, check_source=True):
    """CPU work before queue admission; existing loaders validate generated files."""
    from .run import expand, stamp, utc
    job = row.get("materialization", {})
    if row.get("reuse"):
        from .ledger import SuiteLedger
        state = SuiteLedger(context["root"]).state()
        members = []
        for dependency in row["reuse"]:
            previous = stamp(utc(context["cycle"]).timestamp() - dependency["offset_s"])
            entry = state["jobs"].get((dependency["row"], previous), {})
            receipts = entry.get("result", {}).get("receipts", [])
            if entry.get("phase") != "completed" or not receipts:
                raise ValueError(f"suite_lagged_receipt_missing: {dependency['row']} {previous}; prior runs are reused, never relaunched")
            for receipt in receipts:
                if _file_hash(receipt["path"]) != receipt["sha256"]:
                    raise ValueError(f"suite_lagged_receipt_changed: {receipt['path']}")
            members.append(dict(row=dependency["row"], cycle=previous, work=entry.get("work"), receipts=receipts,
                                suite_hash=entry["suite_hash"], offset_s=dependency["offset_s"]))
        manifest = dict(schema="gpuwm.suite.lagged.v1", cycle=context["cycle"], members=members)
        path = Path(context["lagged_manifest"])
        if path.exists() and json.loads(path.read_text()) != manifest:
            raise ValueError("suite_artifact_changed: reused members changed under a launch token")
        write_receipt_atomically(path, manifest)
        SuiteLedger(context["root"]).append("reuse", row=row["id"], cycle=context["cycle"],
            suite_hash=plan["suite_hash"], members=members, manifest=str(path))
    receipt_path = row.get("source_receipt")
    receipt = None
    if row.get("auto_prepared") and not check_source:
        # A readiness probe runs before this cycle's prepare stage has
        # published anything; the forecast's own launch checks the receipt.
        return context
    if receipt_path and check_source:
        path = Path(expand([receipt_path], context)[0])
        if not path.exists():
            raise SourcePending(f"source_behind: prepared authority not published: {path}")
        receipt = json.loads(path.read_text())
        if not receipt.get("cycle"):
            raise ValueError("suite_prepared_identity: receipt has no cycle")
        if (receipt.get("schema") != "gpuwm.suite.prepared.v1" or
            stamp(receipt.get("cycle")) != context["cycle"] or
            receipt.get("source") != context["source"] or
            receipt.get("suite_sha256") != plan.get("suite_sha256") or
            receipt.get("row") != row.get("suite_row", row["id"])):
            raise ValueError("suite_prepared_identity: receipt differs in suite, row, source or cycle")
        lbc = job.get("source_selection", {}).get("lbc", [])
        if lbc and receipt.get("lbc_source") not in lbc:
            raise ValueError("suite_prepared_lbc: prepared boundary source is absent from the suite's ordered ladder")
        files = receipt.get("files", [])
        if not isinstance(files, list) or not files or any(not isinstance(item, dict) or
                not isinstance(item.get("path"), str) or not re.fullmatch(r"[a-f0-9]{64}", str(item.get("sha256", ""))) for item in files):
            raise ValueError("suite_prepared_hashes: prepared receipt needs file hashes")
        for item in files:
            if _file_hash(item["path"]) != item["sha256"]:
                raise ValueError(f"suite_prepared_hash: {item['path']} changed after preparation")
        if row.get("base_config") and Path(expand([row["base_config"]], context)[0]).resolve() not in {
                Path(item["path"]).resolve() for item in files}:
            raise ValueError("suite_prepared_config: base_config is absent from the prepared authority hashes")
    if not job.get("config_edits"):
        return context
    prefix = f"artifacts.{job['row']}.{job['cycle_hour']:02d}.{job['member']}"
    folder = Path(context["work"])
    if row.get("auto_prepared"):
        # The prepare stage already derived and prepared this cycle's
        # configuration. The runner binds the published copy by name and
        # digest, so the forecast uses it as published, never a re-derivation.
        if receipt is None or not receipt.get("prepared_root") or not receipt.get("experiment_config"):
            raise ValueError("suite_prepared_identity: automatic preparation receipt names no prepared bundle")
        hashed = {Path(item["path"]).resolve() for item in receipt["files"]}
        bound = [receipt["experiment_config"], *([receipt["wps_namelist"]] if receipt.get("wps_namelist") else [])]
        if any(Path(path).resolve() not in hashed for path in bound):
            raise ValueError("suite_prepared_config: published authority is absent from the prepared receipt hashes")
        config = Path(receipt["experiment_config"])
        options = {"prepared_root": receipt["prepared_root"]}
        if receipt.get("wps_namelist"):
            options["wps_namelist"] = receipt["wps_namelist"]
        return _write_run_plan(job, row, context, folder, prefix, config, options)
    derived = derive_config(job, Path(expand([row["base_config"]], context)[0]), context)
    if row.get("kind") == "prepare":
        context["derived_config"] = str(derived)
        return context
    return _write_run_plan(job, row, context, folder, prefix, derived, None)


def derive_config(job, source, context):
    """Apply the planner's config edits to a base configuration for this cycle."""
    from .run import expand, utc
    from gpuwm.toml_document import emit_experiment_toml
    from gpuwm.experiment import load_experiment
    raw = tomllib.loads(source.read_text())
    for dotted, value in job["config_edits"].items():
        keys = dotted.split(".")
        cursor = raw
        for key in keys[:-1]:
            cursor = cursor.setdefault(key, {})
        resolved = expand([value], context)[0] if isinstance(value, str) else value
        # fetch.py:929 accepts YYYY-MM-DDTHH; :8108 requires whole fetch hours.
        # Forecast run_seconds remains exact, while forcing covers its last lead.
        if dotted == "fetch.cycle" and value == "{cycle}":
            resolved = context["cycle_hour"]
        if dotted == "fetch.hours":
            resolved = math.ceil(resolved)
        if dotted == "experiment.start_time" and value == "{start_time}":
            resolved = utc(context["cycle"]).replace(tzinfo=None)
        cursor[keys[-1]] = resolved
    folder = Path(context["work"])
    folder.mkdir(parents=True, exist_ok=True)
    config = folder / "experiment.toml"
    text = emit_experiment_toml(raw)
    if config.exists() and config.read_text() != text:
        raise ValueError("suite_artifact_changed: prepared authority changed under a launch token")
    if not config.exists():
        # The scheduler owns its mutex; replacement leaves a complete TOML.
        temporary = folder / "experiment.toml.pending"
        temporary.write_text(text)
        temporary.replace(config)
    load_experiment(config)
    return config


def _write_run_plan(job, row, context, folder, prefix, config, prepared):
    from .run import expand
    folder.mkdir(parents=True, exist_ok=True)
    context["derived_config"] = str(config)
    context[prefix + ".config"] = str(config)
    document = deepcopy(job["run_plan_document"])
    document["config"]["path"] = str(config)
    document["output_root"] = context["prepared.outdir"]
    document["run_options"].update({k: expand([v], context)[0] if isinstance(v, str) else v
                                    for k, v in row.get("run_options", {}).items()})
    document["run_options"].update(prepared or {})
    context["prepared.root"] = document["run_options"].get("prepared_root")
    context["prepared.wps_namelist"] = document["run_options"].get("wps_namelist")
    if document["run_options"].get("prepared_root"):
        # runplan.py:920 refuses cycle on an existing bundle because it is ignored.
        document["run_options"].pop("cycle", None)
    else:
        document["run_options"]["cycle"] = context["cycle_hour"]
    path = folder / "run-plan.json"
    if path.exists() and json.loads(path.read_text()) != document:
        raise ValueError("suite_artifact_changed: run plan changed under a launch token")
    write_receipt_atomically(path, document)
    from gpuwm.runplan import load_plan
    load_plan(path)
    context[prefix + ".run_plan"] = str(path)
    return context
