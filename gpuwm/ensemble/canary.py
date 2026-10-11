"""Short ordinary packed forecasts at representative prepared heads.

This is a launch check, not an ensemble product or forecast-skill result.
It uses the existing preflight, packing, initialization and forecast owners.
"""
from __future__ import annotations

import argparse
from contextvars import copy_context
from dataclasses import replace
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import threading
import time


class CanaryRefusal(ValueError):
    """A short launch found breakage before the rest of the roster prepared."""


def representatives(bindings):
    """Keep one actual prepared authority per source, in first-seen order."""
    chosen = {}
    for binding in bindings:
        if not binding.get("source"):
            raise CanaryRefusal("canary input has no source; every requested source needs its own cold-start check")
        chosen.setdefault(binding["source"], binding)
    if not chosen:
        raise CanaryRefusal("canary has no prepared inputs; packing and cold start cannot be checked")
    return tuple(chosen.values())


def _limit(inputs, steps):
    """Change only the diagnostic run length/cadence, retaining authorities."""
    exp = inputs.experiment
    if any(domain.run.use_adaptive_time_step for domain in exp.domains):
        raise CanaryRefusal("a duration-derived 50-step canary cannot certify an adaptive step count; "
                            "use a fixed-clock launch check for this configuration")
    duration = float(exp.dt_exact(exp.root.grid_id) * steps)
    if duration > exp.run_seconds:
        raise CanaryRefusal("the prepared forecast is shorter than the requested canary; "
                            "its forcing cannot support 50 model steps")
    domains = []
    for domain in exp.domains:
        if domain.start_time is not None and domain.start_time > exp.start_time:
            raise CanaryRefusal("a delayed child is not active at this canary's head; "
                                "50 root steps would leave its cold start unchecked")
        run = replace(domain.run, run_seconds=duration)
        domains.append(replace(domain, run=run, history_interval_s=duration,
                               history_begin_s=0, history_end_s=None))
    limited = replace(exp, run_seconds=duration, domains=tuple(domains))
    return replace(inputs, experiment=limited), duration


def _bind(binding):
    from gpuwm import stage_cli
    config = Path(binding["experiment_config"])
    root = Path(binding["prepared_root"])
    head = binding.get("head_sha256")
    bundle = (stage_cli.resolve_head_bundle(root, head) if head else stage_cli.resolve_bundle(root))
    if bundle["layout"] == "tree":
        from gpuwm.prepared_domain_tree_forecast import preflight_prepared_tree
        digests = stage_cli.tree_digests(bundle, config)
        values = {"prepared_root": root, "experiment_config": config,
                  "experiment_config_sha256": digests["experiment_config"], "devices": binding.get("devices")}
        values.update({"prepared_head_sha256": head} if head else
                      {"preparation_receipt_sha256": digests["preparation_receipt"]})
        return preflight_prepared_tree(**values), "tree"
    from gpuwm.experiment import load_experiment
    from gpuwm.prepared_single_domain_forecast import preflight_prepared_forecast
    exp = load_experiment(config)
    values = {"source": bundle["source"], "prepared_root": root,
              "experiment_config": config, "wps_namelist": Path(binding["wps_namelist"]),
              "run_seconds": float(exp.run_seconds), "history_interval_seconds": float(exp.root.history_interval_s),
              "devices": binding.get("devices")}
    if head:
        values.update(prepared_head_sha256=head, source_manifest_sha256=bundle.get("source_manifest_sha256"))
    else:
        digests = stage_cli.single_domain_digests(bundle)
        values.update(proof_sha256=digests["proof"], source_manifest_sha256=digests["source_manifest"],
                      prepared_content_sha256=digests["prepared_content"])
    return preflight_prepared_forecast(**values), "single"


def run_canary(bindings, *, directory, request=None, cards=None, steps=50):
    """Run real packed members on the current grant and record every result."""
    import cupy as cp
    from gpuwm.ensemble.production import PreparedEnsembleSession
    from gpuwm.ensemble.request import EnsembleRequest
    from gpuwm import prepared_single_domain_forecast, prepared_domain_tree_forecast
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    receipt = {"schema": "gpuwm-packed-head-canary-v1", "status": "running", "steps": steps,
               "prevents": "packing or cold-start failure discovered only after full-roster preparation"}
    began = time.monotonic()
    def publish():
        partial = directory / "canary.json.pending"
        partial.write_text(json.dumps(receipt, indent=2, sort_keys=True, default=str) + "\n")
        partial.replace(directory / "canary.json")
    publish()
    try:
        if isinstance(steps, bool) or not isinstance(steps, int) or steps <= 0:
            raise CanaryRefusal("canary steps must be a positive integer to bound the smoke forecast")
        visible = int(cp.cuda.runtime.getDeviceCount())
        if cards is not None and cards != visible:
            raise CanaryRefusal(f"canary expected {cards} granted cards but sees {visible}; "
                                "a different grant shape would leave the production packing failure unchecked")
        selected = representatives(bindings)
        prepared, layouts, durations = [], [], []
        for binding in selected:
            inputs, layout = _bind(binding)
            limited, duration = _limit(inputs, steps)
            prepared.append(limited); layouts.append(layout); durations.append(duration)
        if len(set(layouts)) != 1 or len(set(durations)) != 1:
            raise CanaryRefusal("canary representatives need one runner layout and a common 50-step clock; "
                                "otherwise the packed launch would not check the same member window")
        original = EnsembleRequest.from_mapping(request) if request else EnsembleRequest(len(selected))
        # Two slots per card exercise ordinary packing. Repeated inputs here
        # are smoke replicas, never reported as scientific ensemble members.
        from gpuwm.ensemble.prepared_execution import member_device_ids_for_request
        ids = member_device_ids_for_request(prepared[0], original, visible_count=visible,
                                            current_device=int(cp.cuda.runtime.getDevice()))
        count = max(len(prepared), len(ids) * 2)
        members = tuple(prepared[index % len(prepared)] for index in range(count))
        asked = replace(original, members=count, recipe=None, trajectories=(), sources=(), member_variants=(),
                        retain_member_diagnostics=False, keep_member_files=False,
                        member_device_ids=ids,
                        max_ordinary_members_per_device=original.max_ordinary_members_per_device or 2)
        receipt.update(sources=[item["source"] for item in selected], input_bindings=list(selected),
                       grant_cards=visible, smoke_members=count, run_seconds=durations[0],
                       member_device_ids=list(ids), spatial_device_count=int(getattr(prepared[0].experiment.devices, "count", 1)),
                       request=asked.receipt())
        publish()
        session = PreparedEnsembleSession(asked, output_directory=directory / "run",
            input_provider=lambda *, member_id, **unused: members[member_id])
        runner = (prepared_domain_tree_forecast.run_prepared_tree if layouts[0] == "tree" else
                  prepared_single_domain_forecast.run_prepared_forecast)
        options = {"io_mode": "history"} if layouts[0] == "tree" else {}
        result = session.run_prepared(runner, members[0], **options)
        progress = result["ensemble_progress"]["members"]
        if result["status"] != "PASS" or any(row["outer_step"] != steps for row in progress):
            raise CanaryRefusal("packed canary did not complete 50 steps for every smoke member; "
                                "a passing launch label alone cannot qualify the packing and cold start")
        digests = [{"member_id": row["member_id"],
                    "final_state_digest": row["result"].get("final_state_digest")}
                   for row in result["completed_member_results"]]
        if len(digests) != count or any(not row["final_state_digest"] for row in digests):
            raise CanaryRefusal("packed canary lacks a final-state digest for every smoke member; "
                                "its completed labels cannot prove member outputs")
        receipt.update(status="PASS", member_progress=progress, packing=result["packing"],
                       member_digests=digests)
    except (KeyboardInterrupt, SystemExit) as error:
        receipt.update(status="interrupted", error_type=type(error).__name__, error=str(error))
        raise
    except Exception as error:
        receipt.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise CanaryRefusal("packed head canary refused before remaining roster preparation: "
                            f"{type(error).__name__}: {error}; this prevents a packing or cold-start "
                            "failure after the full roster has been prepared") from error
    finally:
        receipt["seconds"] = time.monotonic() - began
        publish()
    return receipt


class HeadCanaries:
    """At most one GPU canary; each newly seen source extends the probe."""
    def __init__(self, directory, request, *, devices=None):
        self.directory = Path(directory)
        self.request = request
        self.devices = devices
        self.bindings = {}
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="packed-head-canary")
        self._futures = []

    def submit(self, binding):
        self.check()
        if self.devices is not None:
            binding = {**binding, "devices": self.devices}
        with self._lock:
            if binding["source"] in self.bindings:
                return
            self.bindings[binding["source"]] = dict(binding)
            selected = tuple(self.bindings.values())
            generation = len(self._futures) + 1
            self.directory.mkdir(parents=True, exist_ok=True)
            context = copy_context()
            future = self._executor.submit(context.run, run_canary, selected,
                directory=self.directory / f"probe-{generation:03d}", request=self.request)
            self._futures.append(future)

    def check(self):
        for future in tuple(self._futures):
            if future.done():
                future.result()

    def close(self, *, raise_error=True):
        self._executor.shutdown(wait=True, cancel_futures=not raise_error)
        if raise_error:
            self.check()

    def watch(self, directory, *, source, config, namelist):
        """Watch only this preparation's published head, including retry age."""
        from datetime import datetime, timezone
        stopped = threading.Event()
        started = time.time()
        errors = []
        def watch_head():
            try:
                while not stopped.wait(.2):
                    self.check()
                    for path in Path(directory).glob("prepare/**/boundary-stream/head.json"):
                        try:
                            head = json.loads(path.read_text())
                        except FileNotFoundError:
                            continue
                        created = datetime.fromisoformat(head["created_utc"])
                        if created.tzinfo is None:
                            created = created.replace(tzinfo=timezone.utc)
                        if created.timestamp() < started or not (head.get("decision") or {}).get("chained"):
                            continue
                        from gpuwm import stage_cli
                        if head["basis"]["proof_head"].get("schema") not in stage_cli._schema_index():
                            continue
                        self.submit({"source": source, "prepared_root": str(path.parent.parent),
                                     "experiment_config": str(config), "wps_namelist": str(namelist),
                                     "head_sha256": head["head_sha256"]})
                        return
            except BaseException as error:
                errors.append(error)
        worker = threading.Thread(target=watch_head, name="canary-head-watch", daemon=True)
        worker.start()
        def check():
            if errors:
                raise errors[0]
            self.check()
        def stop():
            stopped.set()
            worker.join()
        return check, stop


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bindings", type=Path, required=True, help="JSON members with source and prepared authority paths")
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--cards", type=int, required=True, help="exact granted visible-card count")
    args = parser.parse_args(argv)
    document = json.loads(args.bindings.read_text())
    try:
        result = run_canary(document["members"], directory=args.outdir, request=document.get("request"), cards=args.cards)
    except CanaryRefusal as error:
        print(str(error), flush=True)
        return 2
    print(json.dumps(result, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
