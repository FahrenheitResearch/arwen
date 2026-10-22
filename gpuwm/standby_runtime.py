"""Existing native preparation and forecast hosted by the standby lifecycle."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path

from gpuwm.standby import HTTPObjectStore, wait_for_object, wall_now


@dataclass(frozen=True)
class PreparedExecution:
    """Configuration-bound CPU plans and completed kernel warm receipts."""
    config_sha256: str
    boundary_interval_seconds: float
    clock: object
    schedule: object
    kernels: tuple

    def verify(self, inputs):
        from gpuwm.file_digest import sha256_file
        from gpuwm.kernel_prewarm import first_step_ready
        if sha256_file(inputs.experiment_config) != self.config_sha256:
            raise ValueError("warm execution plan belongs to another experiment configuration")
        if float(inputs.boundary_interval_seconds) != self.boundary_interval_seconds:
            raise ValueError("warm execution plan boundary cadence differs from prepared inputs")
        if not self.kernels or not all(first_step_ready(row, preparation=True, forecast=True)
                                       for row in self.kernels):
            raise ValueError("warm execution plan has incomplete kernel receipts")


class PreparedStandby:
    def __init__(self, args, initial_plan, watched):
        self.args, self.initial_plan, self.watched = args, initial_plan, watched
        self.root = Path(args.outdir)
        self.store = HTTPObjectStore(workers=args.fetch_workers, timeout=min(15, args.poll_s))
        self.handle = self.observer = self.events = None
        self.contexts = []

    def prestage(self, receipt):
        from gpuwm import capabilities, doctor, runplan, source_cli
        from gpuwm.boundary_prestage import prestage_boundaries
        from gpuwm.experiment import load_experiment
        from gpuwm.kernel_prewarm import warm_static, first_step_ready
        from gpuwm.mapped_engine_bridge import require_engine

        self.receipt = receipt
        self.experiment = load_experiment(self.args.config)
        with receipt.stage("doctor", parallelism="installation probes") as row:
            checks = doctor.collect_checks((self.initial_plan.source_id,))
            row["checks"] = [asdict(check) for check in checks]
            capabilities.require_for_command("go")
            row["mapped_engine"] = str(require_engine())

        with receipt.stage("card_admission", parallelism="device probes and host planner"):
            import argparse
            from gpuwm.core.preflight import check_main, register_cli
            parser = argparse.ArgumentParser()
            register_cli(parser.add_subparsers())
            command = ["check", str(self.args.config), "--json"]
            parsed = parser.parse_args(command)
            # Price the shape before inputs exist. The combined check also
            # checks input files, which this door deliberately has not fetched.
            code = check_main(parsed)
            if code:
                raise RuntimeError(f"forecast card/host admission failed (gpuwm check exit {code})")
            import cupy as cp
            devices = tuple(dict.fromkeys(self.experiment.devices.device_ids()))
            if cp.cuda.runtime.getDeviceCount() <= max(devices):
                raise RuntimeError(f"standby needs visible admitted cards {devices}")
            # CUDA contexts and the per-device loader caches survive in this
            # process. Actual state-dependent buffers remain the constructor's.
            for device in devices:
                with cp.cuda.Device(device):
                    cp.cuda.runtime.free(0)
                self.contexts.append(device)

        with receipt.stage("kernel_prewarm", parallelism="GPU and parallel NVRTC workers") as row:
            results = []
            for device in self.contexts:
                with cp.cuda.Device(device):
                    warmed = warm_static(self.experiment, preparation=True, forecast=True,
                                         say=lambda line: print(line, flush=True))
                    results.append({"device": device, **warmed})
                    if not first_step_ready(warmed, preparation=True, forecast=True):
                        row["results"] = results
                        raise RuntimeError(f"kernel prewarm incomplete on device {device}: {warmed['errors']}")
            row["results"] = results

        with receipt.stage("forecast_plan", parallelism="host orchestration") as row:
            import tomllib
            from gpuwm.core.clock import build_schedule, resolve_clock
            from gpuwm.file_digest import sha256_file
            from gpuwm.source_adapters import get_source_adapter
            fetch = tomllib.loads(self.args.config.read_text())["fetch"]
            interval = (float(fetch["cadence"]) * 3600 if fetch.get("cadence") is not None
                        else get_source_adapter(fetch["source"]).forcing_interval_seconds)
            clock = resolve_clock(self.experiment, lbc_interval_s=interval)
            self.execution = PreparedExecution(sha256_file(self.args.config), float(interval),
                clock, build_schedule(self.experiment, clock), tuple(results))
            row["deferred"] = ["state-dependent forecast buffers and physics initialization"]
            options = {"as_posted": False, "render_products": self.args.products}
            for name in ("geog_root", "data_dir"):
                value = getattr(self.args, name)
                if value is not None:
                    options[name] = str(value)
            raw = {"schema": runplan.PLAN_SCHEMA, "name": "standby",
                   "route": "prepared", "config": {"path": str(self.args.config)},
                   "output_root": str(self.root), "run_options": options}
            self.plan = runplan.build_plan(raw, source="gpuwm standby", base_dir=self.root,
                sha256=hashlib.sha256(json.dumps(raw, sort_keys=True).encode()).hexdigest())
            self.events = runplan.EventStream(self.root / "standby-events.jsonl", mirror=None)
            owner = self
            class Observer(runplan.RunObserver):
                def __call__(self, **event):
                    if event.get("outer_step", 0) > 0 and event.get("model_elapsed_seconds", 0) > 0:
                        owner.receipt.first_step(**event)
                    super().__call__(**event)
            self.observer = Observer(self.events)

        def prepare():
            from gpuwm.progress import event_sink
            with source_cli.in_process_adapters(), event_sink(receipt.preparation_event):
                return runplan._staged_chain(self.plan, config_path=self.args.config,
                    exp=self.experiment, observer=self.observer,
                    run_dir=self.root, prepare_only=True, defer_initial=True)

        with receipt.stage("boundary_static_prestage", parallelism="Rust workers and GPU"):
            self.handle = prestage_boundaries(prepare, devices=self.contexts)
            if not self.handle.wait_ready(timeout=self.args.wait_timeout_s):
                raise TimeoutError("boundary/static prestage did not become ready")
            receipt.update(boundary_prestage=self.handle.receipt)
        print("standby: warmed; waiting for the analysis", flush=True)

    def _analysis_inputs(self, first_path):
        """Use ordinary table composition/handoffs, with the ranged transport."""
        from gpuwm import fetch_routes
        from gpuwm.fetch import _donor_route_plan, _donor_route_primary
        from gpuwm.initial_source import initial_inputs_from_fetch
        from gpuwm.file_digest import sha256_file
        from types import SimpleNamespace
        root = self.root / "initial"
        root.mkdir()
        primary_urls = set(self.watched.urls(self.initial_plan.ladder))

        def acquire(plan, destination):
            objects = {url: obj for obj in plan.objects for url in obj.urls(plan.ladder)}
            def download(url, dest, *, magic, opener=None):
                dest = Path(dest)
                dest.parent.mkdir(parents=True, exist_ok=True)
                if url in primary_urls:
                    os.link(first_path, dest)
                    fetched_url = self.receipt.document["complete_object"]["url"]
                else:
                    obj = objects[url]
                    posted = wait_for_object(obj.urls(plan.ladder), self.store, self.receipt,
                        poll_s=self.args.poll_s, timeout_s=self.args.wait_timeout_s,
                        headline=False, stage_name=f"companion_wait:{obj.role}")
                    with self.receipt.stage(f"companion_fetch:{obj.role}",
                            parallelism=f"{self.store.workers} HTTP workers"):
                        self.store.fetch(posted, dest)
                    fetched_url = posted.url
                fetch_routes._verify_payload(dest, magic=magic, label=dest.name)
                return {"name": dest.name, "bytes": dest.stat().st_size,
                        "sha256": sha256_file(dest), "url": fetched_url}
            return fetch_routes.run_plan(plan, out=destination, downloader=download,
                file_workers=self.args.fetch_workers, probe=lambda url: True)

        donor_args = SimpleNamespace(transport=None, member=None, out=root)
        donors = []
        for donor in self.initial_plan.donors:
            plan = _donor_route_plan(donor, donor_args)
            donors.append((donor, plan, root / "donors" / donor.role))
        with self.receipt.stage("analysis_inventory", parallelism=f"{1 + len(donors)} source workers"), \
                ThreadPoolExecutor(1 + len(donors), thread_name_prefix="analysis-sources") as pool:
            jobs = [pool.submit(acquire, self.initial_plan, root)]
            jobs += [pool.submit(acquire, plan, folder) for _, plan, folder in donors]
            for job in jobs:
                job.result()
        donor_files = {donor.role: folder / _donor_route_primary(plan, donor)
                       for donor, plan, folder in donors}
        fetch_routes.write_handoff(self.initial_plan, root, donor_files=donor_files)
        return initial_inputs_from_fetch(self.initial_plan.source_id, root,
                                        root / "initial-inputs.json")

    def launch(self, path, receipt):
        from gpuwm import stage_cli
        from gpuwm.prepared_single_domain_forecast import run_prepared_forecast
        declaration = self._analysis_inputs(path)
        with receipt.stage("analysis_decode_start_state", parallelism="Rust workers and GPU"):
            prepared = self.handle.finish_initial(declaration)
            receipt.update(boundary_prestage=self.handle.receipt)
        with receipt.stage("forecast_preflight", parallelism="host verification workers"):
            inputs = stage_cli.preflight_bundle(prepared["bundle"],
                experiment_config=Path(prepared["experiment_config"]),
                wps_namelist=Path(prepared["wps_namelist"]) if prepared["wps_namelist"] else None,
                devices=self.args.devices)
        outdir = self.root / "forecast"
        outdir.mkdir()
        render_plan = {"run": outdir, "render": self.root / "png",
                       "config": self.args.config,
                       "render_products": self.args.products,
                       "publish_root": str(self.args.publish_root or self.root / "www"),
                       "cycle": self.receipt.document["cycle"]}
        self.observer.arm_first_products(render_plan)
        receipt.update(launch_wall=wall_now())
        self.observer.enter_stage("forecast", phase="forecast")
        with receipt.stage("forecast", parallelism=f"{len(self.contexts)} GPU(s)"):
            report = run_prepared_forecast(inputs, output_directory=outdir,
                observer=self.observer, prestaged_boundaries=self.handle,
                prewarmed_execution=self.execution)
        self.observer.finish_stage()
        from gpuwm.runplan import _finish_render
        _finish_render(render_plan, observer=self.observer, door="standby")
        receipt.update(forecast_report=report)
        return str(outdir / "report.json")

    def close(self):
        try:
            if self.observer is not None:
                self.observer.close()
        finally:
            try:
                if self.events is not None:
                    self.events.close()
            finally:
                if self.handle is not None:
                    self.handle.close()
