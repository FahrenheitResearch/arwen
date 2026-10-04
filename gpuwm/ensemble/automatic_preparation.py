"""Concrete acquisition and native preparation owner for automatic ensembles.

The caller retains this owner through its ordinary ensemble session and calls
``finish`` after forecasts, or ``close`` on failure. Only subprocesses launched
by this owner are stopped. Shared/preexisting source producers are never killed.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from gpuwm.ensemble.automatic_sources import _canonical, _digest
from gpuwm.ensemble.physical_store import digest_file
from gpuwm.ensemble.source_preparation import PostedSourcePreparation


def _publish(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (_canonical(value) + "\n").encode()
    if path.exists():
        if path.read_bytes() != payload:
            raise FileExistsError(f"automatic source authority already exists with different bytes: {path}")
        return
    temporary = path.with_name(path.name + ".pending")
    temporary.write_bytes(payload)
    temporary.replace(path)


def _option(arguments, flag):
    found = []
    for index, value in enumerate(arguments):
        if value == flag:
            if index + 1 == len(arguments):
                raise ValueError(f"native source template has no value for {flag}")
            found.append(arguments[index + 1])
        elif value.startswith(flag + "="):
            found.append(value.split("=", 1)[1])
    if len(found) > 1:
        raise ValueError(f"native source template repeats {flag}")
    return found[0] if found else None


def _replace_option(arguments, flag, value):
    result = list(arguments)
    for index, token in enumerate(result):
        if token == flag:
            result[index + 1] = str(value)
            return tuple(result)
        if token.startswith(flag + "="):
            result[index] = flag + "=" + str(value)
            return tuple(result)
    return (*result, flag, str(value))


def _remove_option(arguments, flag):
    result = []
    skip = False
    for token in arguments:
        if skip:
            skip = False
        elif token == flag:
            skip = True
        elif not token.startswith(flag + "="):
            result.append(token)
    return tuple(result)


def materialize_source_controls(recipe, trajectory, template, *, root):
    """Expand donor config times to native brackets through ordinary writers.

    The target domain and physics tables remain the parsed original document.
    Only acquisition time bounds and native forcing cadence change. Recentered
    members still use the original base template and its requested window.
    """
    template.verify()
    arguments = template.arguments
    first, last = recipe.acquisition_window(trajectory)
    start = trajectory.cycle + timedelta(hours=first)
    end = trajectory.cycle + timedelta(hours=last)
    config_path = _option(arguments, "--experiment-config")
    from gpuwm.source_adapters import get_source_adapter
    from gpuwm.fortran_namelist import parse_namelist
    cadence = get_source_adapter(trajectory.source).forcing_interval_seconds
    original_wps = _option(arguments, "--wps-namelist")
    actual_cadence = (None if original_wps is None else
        parse_namelist(original_wps).get("share", {}).get("interval_seconds", [None])[0])
    if start == recipe.start and end == recipe.end and cadence == actual_cadence:
        return arguments
    if config_path is None:
        raise ValueError(f"{trajectory.source} donor brackets require its existing experiment-config template")
    import tomllib
    from copy import deepcopy
    from gpuwm.experiment import load_experiment
    from gpuwm.toml_document import emit_experiment_toml
    from gpuwm.companion_domains import candidate_wps_text
    original_path = Path(config_path)
    original = load_experiment(original_path)
    if len(original.domains) != 1 and (start != recipe.start or end != recipe.end):
        raise ValueError("recentered domain trees need per-domain physical capture before native donor bracket expansion")
    raw = tomllib.loads(original_path.read_text(encoding="utf-8-sig"))
    changed = deepcopy(raw)
    if "case_data" in changed:
        from gpuwm.case_data import resolved_case_data_paths
        changed["case_data"] = resolved_case_data_paths(changed["case_data"],
            base_dir=original_path.resolve().parent, source=str(original_path))
    if "static" in changed:
        from gpuwm.static.highres_production import parse_static_table
        carrier = parse_static_table(changed["static"], source=str(original_path),
                                     base_dir=original_path.resolve().parent)
        if carrier is not None and getattr(carrier, "cache_root", None) is not None:
            changed.setdefault("static", {}).setdefault("highres", {})["cache_root"] = str(carrier.cache_root.resolve())
    changed["experiment"]["start_time"] = start.replace(tzinfo=None)
    changed["experiment"]["run_seconds"] = int((end - start).total_seconds())
    for domain in changed.get("domain", ()):
        if "start_time" in domain and (start != recipe.start or end != recipe.end):
            domain["start_time"] = start.replace(tzinfo=None)
    # Geometry/physics settings are preserved; a donor's input cadence is
    # a source fact, never the CAM output or integration cadence.
    if cadence is not None:
        # A new trajectory cannot inherit another source's crop, provider,
        # product or retention hints. Its actual fetch handoff owns those;
        # the copied config keeps only this recipe's explicit time request.
        changed["fetch"] = dict(source=trajectory.source, cadence=int(cadence) // 3600,
            cycle=trajectory.cycle.strftime("%Y-%m-%dT%H"), hours=last-first, forecast_start_hour=first)
        if trajectory.member is not None:
            changed["fetch"]["member"] = trajectory.member
        else:
            changed["fetch"].pop("member", None)
        if "case_data" in changed and "forcing_interval_s" in changed["case_data"]:
            changed["case_data"]["forcing_interval_s"] = cadence
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    target = root / "experiment.toml"
    text = emit_experiment_toml(changed)
    if target.exists() and target.read_text(encoding="utf-8") != text:
        raise FileExistsError("native donor configuration already differs from its original template")
    target.write_text(text, encoding="utf-8")
    experiment = load_experiment(target)
    if original_wps is None:
        raise ValueError("native donor bracket expansion needs the original WPS authority")
    wps = target.with_suffix(".namelist.wps")
    wps_text = candidate_wps_text(changed, original, experiment, target,
                                  original_wps=Path(original_wps))
    if wps.exists() and wps.read_text(encoding="utf-8") != wps_text:
        raise FileExistsError("native donor WPS authority changed")
    wps.write_text(wps_text, encoding="utf-8")
    _publish(root / "native-window.json", {"schema": "gpuwm-ensemble-native-window.v1",
        "trajectory_sha256": trajectory.identity, "recipe_sha256": recipe.sha256,
        "requested_start": recipe.start.isoformat(), "requested_end": recipe.end.isoformat(),
        "acquisition_start": start.isoformat(), "acquisition_end": end.isoformat(),
        "original_configuration_sha256": digest_file(original_path),
        "configuration_sha256": digest_file(target), "wps_sha256": digest_file(wps)})
    return _replace_option(_replace_option(arguments, "--experiment-config", target), "--wps-namelist", wps)


def preflight_source_head(context, specification, head):
    """Dispatch to the existing single-domain or tree CPU admission."""
    from gpuwm import stage_cli
    bundle = stage_cli.resolve_head_bundle(specification.prepared_root, head["head_sha256"])
    ordinary = dict(getattr(context.ordinary_inputs, "preflight_arguments", {}) or {})
    ordinary.update(context.preflight_options)
    arguments = specification.native_arguments
    config = _option(arguments, "--experiment-config")
    if config is None or (specification.prepared_root / "experiment.toml").is_file():
        config = specification.prepared_root / "experiment.toml"
    config = Path(config)
    if bundle["layout"] == "tree":
        from gpuwm.prepared_domain_tree_forecast import preflight_prepared_tree
        return preflight_prepared_tree(prepared_root=specification.prepared_root,
            prepared_head_sha256=head["head_sha256"], experiment_config=config,
            experiment_config_sha256=digest_file(config),
            **{key: ordinary[key] for key in ("physics_profile", "devices", "devices_options") if key in ordinary})
    from gpuwm.prepared_single_domain_forecast import preflight_prepared_forecast
    from gpuwm.experiment import load_experiment
    experiment = load_experiment(config)
    wps = _option(arguments, "--wps-namelist")
    if (specification.prepared_root / "namelist.wps").is_file():
        wps = specification.prepared_root / "namelist.wps"
    if wps is None:
        raise ValueError("native source head needs its exact WPS argument for ordinary admission")
    controls = {key: ordinary[key] for key in ("physics_profile", "expert_acknowledgements", "tiles",
                                               "devices", "devices_options") if key in ordinary}
    return preflight_prepared_forecast(source=bundle["source"], prepared_root=specification.prepared_root,
        prepared_head_sha256=head["head_sha256"], source_manifest_sha256=bundle["source_manifest_sha256"],
        experiment_config=config, wps_namelist=Path(wps), run_seconds=experiment.run_seconds,
        history_interval_seconds=ordinary.get("history_interval_seconds", 3600), **controls)


def _geometry(inputs):
    from gpuwm.native_wrf_contract import native_geometry_contract
    domains = inputs.experiment.domains
    grids = getattr(inputs, "grids", None)
    if grids is None:
        grids = (inputs.grid,)
    if len(grids) != len(domains):
        raise ValueError("ordinary source input domain/geometry count differs")
    return _digest([{ "grid_id": domain.grid_id, "parent_id": domain.parent_id,
                     "geometry": native_geometry_contract(grid, domain.run)}
                    for domain, grid in zip(domains, grids)])


@dataclass(frozen=True)
class PostedSourceBinding:
    """Native head and handoff authority while raw future inputs are pending."""
    specification: PostedSourcePreparation
    head_sha256: str

    @property
    def trajectory(self):
        return self.specification.trajectory

    def verify(self):
        from gpuwm.ingest.boundary_stream import (
            bind_head, posted_lead_marker_name, posted_lead_marker_sha256, read_replaced_json)
        from gpuwm.prep_handoff import _posted_handoff
        self.specification.verify()
        _, _, schedule, selected, _ = _posted_handoff(
            self.specification.acquisition_root, self.trajectory)
        if selected != self.trajectory:
            raise ValueError("ordinary source binding changed its original source trajectory")
        head = bind_head(self.specification.prepared_root, self.head_sha256)
        posted = head["basis"].get("as_posted")
        if not isinstance(posted, dict) or not isinstance(posted.get("input_plan"), dict):
            raise ValueError("automatic ordinary source binding needs its real posted input plan")
        first = int(schedule["leads"][0]["lead"])
        bound_markers = posted.get("start_marker_sha256")
        if not isinstance(bound_markers, dict) or str(first) not in bound_markers:
            raise ValueError("ordinary native source head omits its initial acquisition marker")
        for lead, expected in bound_markers.items():
            marker = read_replaced_json(self.specification.acquisition_root / "posting" / posted_lead_marker_name(int(lead)))
            if posted_lead_marker_sha256(marker) != expected:
                raise ValueError("ordinary native source head belongs to another acquisition's initial posted marker")
        # The ordinary native preflight validates each source's manifest,
        # member selection and target configuration. Retain those original
        # bytes rather than creating a stand-in acquisition manifest.
        return {"trajectory_sha256": self.trajectory.identity, "source": self.trajectory.source,
            "cycle": self.trajectory.cycle.isoformat(), "member": self.trajectory.member,
            "prepared_root": str(self.specification.prepared_root), "head_sha256": self.head_sha256,
            "input_plan_sha256": head["basis"]["as_posted"]["input_plan_sha256"],
            "posting_schedule_sha256": _digest(schedule),
            "source_seal": "ordinary forecast reader verifies all consumed intervals and final seal"}


class AutomaticPreparationOwner:
    """Owned native processes and the exact arguments to the existing session."""
    def __init__(self, selection, context, *, output_root, observer=None,
                 cpu_workers=1, temporary_root=None, existing_sources=None, cpu_bridge=None):
        if type(cpu_workers) is not int or cpu_workers < 1:
            raise ValueError("source preparation worker count must be a positive integer")
        if _digest(context.describe()) != selection.context_sha256:
            raise ValueError("automatic selection context changed before source preparation")
        self.selection, self.context = selection, context
        self.root = Path(output_root).resolve()
        self.observer, self.cpu_workers = observer, cpu_workers
        self.cpu_bridge = cpu_bridge
        self.temporary_root = None if temporary_root is None else Path(temporary_root).resolve()
        self.specifications = dict(existing_sources or {})
        self.processes = []
        self.source_inputs = {}
        self.shared_geography = {}
        self.session_arguments = {}
        self._closed = False
        self._finished = False

    @property
    def inputs(self):
        """Original inputs to hand the ordinary runner after source admission."""
        if self.selection.recipe is None:
            return self.context.ordinary_inputs
        recipe = self.selection.recipe
        trajectory = recipe.base if recipe.kind == "recentered" else recipe.members[0].trajectory
        return self.source_inputs[trajectory.identity]

    def _event(self, event, **details):
        if self.observer is not None:
            self.observer({"stage": "ensemble-source", "event": event, **details})

    def _launch(self, command, *, identity, role):
        log = self.root / "logs" / (identity + "." + role + ".log")
        log.parent.mkdir(parents=True, exist_ok=True)
        environment = dict(os.environ)
        environment.update(CUDA_VISIBLE_DEVICES="", GPUWM_NO_LOCAL_GPU="1",
                           OMP_NUM_THREADS=str(self.cpu_workers), OPENBLAS_NUM_THREADS=str(self.cpu_workers),
                           MKL_NUM_THREADS=str(self.cpu_workers), RAYON_NUM_THREADS=str(self.cpu_workers))
        if self.temporary_root is not None:
            temporary = self.temporary_root / identity / role
            temporary.mkdir(parents=True, exist_ok=True)
            environment["TMPDIR"] = str(temporary)
        stream = log.open("xb")
        try:
            process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=stream,
                stderr=subprocess.STDOUT, env=environment,
                start_new_session=os.name != "nt",
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        except BaseException:
            stream.close()
            raise
        from gpuwm.proc_identity import identify
        self.processes.append({"identity": identity, "role": role, "process": process,
                               "process_identity": identify(process.pid),
                               "log": log, "stream": stream, "command": list(command)})
        _publish(self.root / "processes" / (identity + "." + role + ".json"), {
            "process_identity": self.processes[-1]["process_identity"],
            "trajectory_sha256": identity, "role": role, "log": str(log), "command": list(command)})
        self._event("started", trajectory_sha256=identity, role=role, pid=process.pid, log=str(log))

    def _check_processes(self):
        for row in self.processes:
            status = row["process"].poll()
            if status not in (None, 0):
                raise RuntimeError(f"ordinary {row['role']} for {row['identity']} exited {status}; log: {row['log']}")

    def _shared_static_arguments(self, template, arguments, *, timeout):
        from gpuwm.ensemble.automatic_sources import NativeSourceTemplate
        from gpuwm.source_cli import preparation_runners
        # The options a preparation reads a prebuilt pair through, and the
        # build that makes one, are its row in the preparation runner table.
        row = preparation_runners().get(template.runner)
        geography = None if row is None else row.shared_geography
        if geography is None:
            # Breakage it prevents: members prepared through a route with
            # no declared prebuilt-geography options would each build their
            # own, and nothing would hold them to one target geography.
            raise ValueError(f"{template.runner} has no shared native geography producer; "
                             "its members could not be given one common static input. "
                             "Use a source whose preparation declares shared geography")
        if _option(arguments, geography.cache_option) is not None:
            # A caller-provided prebuilt input remains precisely that
            # authority. It is never taken from an initialized source.
            return arguments
        geog = _option(arguments, "--geog-root")
        if geog is None:
            raise ValueError("automatic native source needs shared static input or its ordinary geog-root")
        captured = NativeSourceTemplate.capture(template.runner, arguments)
        configuration = {item.role: item.sha256 for item in captured.configuration_files}
        experiment = _option(arguments, "--experiment-config")
        if experiment is not None:
            import tomllib
            tables = tomllib.loads(Path(experiment).read_text(encoding="utf-8-sig"))
            tables.pop("fetch", None)
            tables.pop("ensemble", None)
            configuration["--experiment-config"] = _digest(json.loads(json.dumps(tables, default=str)))
        identity = _digest({"runner": template.runner, "geog_root": str(Path(geog).resolve()),
            "configuration_files": configuration})
        if identity not in self.shared_geography:
            request = self.root / "shared-geography" / (identity + ".request.json")
            output = self.root / "shared-geography" / identity
            _publish(request, {"runner": template.runner, "arguments": list(arguments)})
            self._launch([sys.executable, "-m", "gpuwm.ensemble.automatic_static",
                "--request", str(request), "--output-root", str(output)], identity=identity, role="static")
            process = self.processes[-1]["process"]
            self._wait(lambda: True if process.poll() == 0 else None,
                       purpose="shared native geography", timeout=timeout)
            record = json.loads((output / "shared-geography.json").read_bytes())
            if (record.get("status") != "PASS" or record.get("runner") != template.runner or
                    digest_file(Path(record["cache"])) != record["cache_sha256"] or
                    digest_file(Path(record["receipt"])) != record["receipt_sha256"]):
                raise ValueError("shared native geography differs from its original builder receipt")
            self.shared_geography[identity] = record
        record = self.shared_geography[identity]
        for option in geography.superseded_options:
            arguments = _remove_option(arguments, option)
        arguments = _replace_option(arguments, geography.cache_option, record["cache"])
        return _replace_option(arguments, geography.receipt_option, record["receipt"])

    def _wait(self, ready, *, purpose, timeout):
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            self._check_processes()
            result = ready()
            if result is not None:
                return result
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError(f"automatic source preparation timed out waiting for {purpose}")
            self._event("waiting", purpose=purpose)
            time.sleep(0.25)

    def start(self, *, timeout=None):
        from gpuwm.ensemble.stochastic_model import StochasticModelProvider
        if self.selection.stochastic is not None:
            self.session_arguments["stochastic_provider"] = StochasticModelProvider.from_mapping(self.selection.stochastic)
        if self.selection.mode in {"ordinary", "caller-owned", "stochastic"}:
            if self.selection.mode != "ordinary":
                self.context.verify()
                _publish(self.root / "source-selection.json", self.selection.describe())
            return self
        self.context.verify()
        recipe = self.selection.recipe
        required = {trajectory.identity: trajectory for trajectory in recipe.acquisitions()}
        if set(self.specifications) - required.keys():
            raise ValueError("automatic source reuse includes a trajectory outside the frozen recipe")
        transformed = recipe.kind == "recentered"
        if transformed and len(self.context.domains) > 1:
            raise ValueError("recentered domain trees require per-domain native physical capture; "
                             "ordinary operational tree ensembles remain supported")
        _publish(self.root / "source-selection.json", self.selection.describe())
        _publish(self.root / "source-context.json", self.context.describe())
        acquisition_parent = self.root / "acquisitions"
        try:
            for identity, trajectory in required.items():
                if identity in self.specifications:
                    specification = self.specifications[identity].verify()
                    if specification.trajectory != trajectory:
                        raise ValueError("reused native source specification identifies another trajectory")
                    continue
                template = self.context.template_for(trajectory)
                arguments = materialize_source_controls(recipe, trajectory, template,
                    root=self.root / "native-controls" / identity)
                arguments = self._shared_static_arguments(template, arguments, timeout=timeout)
                fetch = recipe.fetch_argv(trajectory, acquisition_parent, p_top_pa=self.context.p_top_pa)
                self._launch([sys.executable, "-m", "gpuwm.cli", "fetch", *fetch,
                              "--fetch-workers", str(self.cpu_workers)], identity=identity, role="fetch")
                acquisition = acquisition_parent / identity
                def acquired():
                    if not acquisition.is_dir():
                        return None
                    try:
                        return PostedSourcePreparation.from_acquisition(trajectory,
                            acquisition_root=acquisition, prepared_root=self.root / "prepared" / identity,
                            physical_root=self.root / "physical" / identity, native_arguments=arguments)
                    except FileNotFoundError:
                        return None
                specification = self._wait(acquired, purpose=f"{trajectory.source} native handoff", timeout=timeout)
                self.specifications[identity] = specification
                native = [*specification.preparation_arguments, *specification.native_arguments,
                          "--output-root", str(specification.prepared_root)]
                if transformed:
                    native.extend(("--physical-output-store", str(specification.physical_root)))
                self._launch([sys.executable, "-m", "gpuwm.source_cli", *native], identity=identity, role="prepare")
            from gpuwm.ingest.boundary_stream import read_head
            heads = {}
            for identity, specification in self.specifications.items():
                def head_ready():
                    path = specification.prepared_root / "boundary-stream" / "head.json"
                    if not path.is_file():
                        return None
                    return read_head(specification.prepared_root)
                head = self._wait(head_ready, purpose=f"{specification.trajectory.source} initial native head", timeout=timeout)
                heads[identity] = head
                self.source_inputs[identity] = preflight_source_head(self.context, specification, head)
            if transformed:
                from gpuwm.ensemble.source_preparation import PostedPreparationFactory
                from gpuwm.ensemble.posted_execution import PostedRecipeExecution
                factory = PostedPreparationFactory.publish(recipe, self.specifications,
                    root=self.root / "provider", amplitude=self.selection.amplitude,
                    workers=self.cpu_workers, cpu_bridge=self.cpu_bridge)
                self.session_arguments["source_execution"] = PostedRecipeExecution(factory, self.source_inputs,
                    member_root=self.root / "members")
            else:
                self.session_arguments["member_roster"] = self._ordinary_roster(recipe, heads)
            _publish(self.root / "source-start.json", self.receipt())
            return self
        except BaseException:
            self.close()
            raise

    def _ordinary_roster(self, recipe, heads):
        from gpuwm.ensemble.member_preparation import PreparedMemberInput, PreparedMemberRoster
        geometries = {identity: _geometry(inputs) for identity, inputs in self.source_inputs.items()}
        if len(set(geometries.values())) != 1:
            raise ValueError("ordinary ensemble sources do not preserve the shared native target geometry")
        geometry = next(iter(geometries.values()))
        members = []
        for member in recipe.members:
            identity = member.trajectory.identity
            inputs = self.source_inputs[identity]
            times = tuple(recipe.start + timedelta(hours=hour) for hour in inputs.forcing_hours)
            if times and times[-1] > recipe.end:
                times = tuple(instant for instant in times if instant <= recipe.end)
            bindings = (PostedSourceBinding(self.specifications[identity], heads[identity]["head_sha256"]),)
            initial = getattr(inputs, "cache_reader", tuple(getattr(inputs, "domains", ())))
            members.append(PreparedMemberInput(member.index, member.seed, inputs, initial, initial,
                member.trajectory, bindings, times, geometry, recipe.sha256,
                preparation_receipt={"native_preparation": "unchanged ordinary source",
                    "head_sha256": heads[identity]["head_sha256"],
                    "domain_count": len(inputs.experiment.domains)}))
        return PreparedMemberRoster(recipe, members, shared_geometry_sha256=geometry)

    def receipt(self):
        return {"schema": "gpuwm-ensemble-automatic-preparation.v1",
                "selection_sha256": self.selection.sha256,
                "shared_geography": self.shared_geography,
                "status": "complete" if self._finished else "closed" if self._closed else "started",
                "sources": {identity: {"prepared_root": str(spec.prepared_root),
                    "acquisition_root": str(spec.acquisition_root), "physical_root": str(spec.physical_root)}
                    for identity, spec in self.specifications.items()},
                "owned_processes": [{"trajectory_sha256": row["identity"], "role": row["role"],
                    "pid": row["process"].pid, "returncode": row["process"].poll(),
                    "process_identity": row["process_identity"],
                    "log": str(row["log"]), "command": row["command"]} for row in self.processes]}

    def finish(self, *, timeout=None):
        self._wait(lambda: True if all(row["process"].poll() is not None for row in self.processes) else None,
                   purpose="ordinary source process completion", timeout=timeout)
        from gpuwm.ingest.boundary_stream import read_head, verify_seal
        for specification in self.specifications.values():
            verify_seal(specification.prepared_root, head=read_head(specification.prepared_root))
        self._finished = True
        if self.selection.mode != "ordinary":
            _publish(self.root / "source-finish.json", self.receipt())
        self.close()
        return self.receipt()

    def close(self):
        if self._closed:
            return
        from gpuwm.ingest.boundary_stream import request_stop
        import signal
        from gpuwm.proc_identity import signal_process
        for row in self.processes:
            if row["role"] == "prepare" and row["process"].poll() is None:
                request_stop(self.specifications[row["identity"]].prepared_root,
                             "automatic ensemble source owner stopped")
        for row in self.processes:
            process = row["process"]
            if process.poll() is None:
                signal_process(row["process_identity"], signal.SIGTERM, tree=True)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    signal_process(row["process_identity"], getattr(signal, "SIGKILL", signal.SIGTERM), tree=True)
                    process.wait()
            row["stream"].close()
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, kind, value, traceback):
        if kind is None:
            try:
                self.finish()
            finally:
                self.close()
        else:
            self.close()


def start_ensemble_sources(selection, context, *, output_root, observer=None,
                           cpu_workers=1, temporary_root=None, existing_sources=None,
                           timeout=None, cpu_bridge=None):
    owner = AutomaticPreparationOwner(selection, context, output_root=output_root,
        observer=observer, cpu_workers=cpu_workers, temporary_root=temporary_root,
        existing_sources=existing_sources, cpu_bridge=cpu_bridge)
    return owner.start(timeout=timeout)
