"""Explicit geometry fitting of an ordinary, authoritative experiment TOML.

The domain wizard owns geometry and pricing. This module supplies complete
candidate experiments; it never chooses or substitutes scientific settings.
"""
from __future__ import annotations

import copy
import hashlib
from datetime import date, datetime, time, timedelta, timezone
import json
import math
import os
from pathlib import Path
import shlex
import tomllib



def _publish_new_files(files):
    """Create companions first and the config last; unwind ordinary I/O errors."""
    created = []
    try:
        for path, content in files:
            with path.open("x", encoding="utf-8", newline="\n") as stream:
                identity = os.fstat(stream.fileno())
                created.append((path, identity))
                stream.write(content)
    except BaseException as error:
        for path, identity in reversed(created):
            try:
                # Do not remove a file another process has replaced.
                if os.path.samestat(identity, path.stat()):
                    path.unlink()
            except FileNotFoundError:
                pass
            except OSError as cleanup_error:
                error.add_note(f"Could not remove incomplete output {path}: {cleanup_error}")
        raise


def _command_path(path):
    value = str(path)
    if os.name == "nt":
        return "'" + value.replace("'", "''") + "'"
    return shlex.quote(value)

def _value(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, (datetime, date, time)):
        return value.isoformat()
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, list):
        return "[" + ", ".join(_value(v) for v in value) + "]"
    if isinstance(value, dict):
        return "{ " + ", ".join(json.dumps(k) + " = " + _value(v)
                                 for k, v in value.items()) + " }"
    raise TypeError(f"Cannot serialize TOML value {value!r}")


def render_tables(raw):
    """Preserve all parsed tables/types, including nested advanced settings."""
    lines = ["# Explicitly fitted from an editable starter; review the fit receipt.",
             "# Scientific settings and clocks remain the starter's authority."]
    for key, value in raw.items():
        if not isinstance(value, (dict, list)):
            lines.append(f"{json.dumps(key)} = {_value(value)}")
    for key, value in raw.items():
        if isinstance(value, dict):
            lines.append(f"\n[{json.dumps(key)}]")
            lines.extend(f"{json.dumps(k)} = {_value(v)}" for k, v in value.items())
        elif isinstance(value, list):
            if not all(isinstance(v, dict) for v in value):
                raise ValueError(f"Top-level {key} must be an array of tables")
            for table in value:
                lines.append(f"\n[[{json.dumps(key)}]]")
                lines.extend(f"{json.dumps(k)} = {_value(v)}" for k, v in table.items())
    return "\n".join(lines) + "\n"


def changes(before, after, prefix=""):
    """Exact parsed field diff, without hiding advanced/path changes."""
    if isinstance(before, dict) and isinstance(after, dict):
        result = []
        for key in sorted(before.keys() | after.keys()):
            path = f"{prefix}.{key}" if prefix else key
            if key not in before or key not in after:
                result.append((path, before.get(key), after.get(key)))
            else:
                result.extend(changes(before[key], after[key], path))
        return result
    if isinstance(before, list) and isinstance(after, list) and len(before) == len(after):
        return [item for i, (a, b) in enumerate(zip(before, after))
                for item in changes(a, b, f"{prefix}[{i}]")]
    return [] if before == after else [(prefix, before, after)]


class Starter:
    def __init__(self, path: Path, out: Path):
        from gpuwm.config_authority import read_config_authority
        from gpuwm.experiment import build_experiment_from_config_tables
        authority = read_config_authority(path)
        self.path = authority.source
        self.sha256 = authority.sha256
        self.out = out.resolve()
        self.original = tomllib.loads(authority.payload.decode("utf-8"))
        self.exp = build_experiment_from_config_tables(
            self.original, source=str(self.path), base_dir=authority.base_dir)
        self.raw = copy.deepcopy(self.original)
        self.wps_original = None
        self.wps_base = None
        domains = self.exp.domains
        if any(d.grid_id != i + 1 or d.parent_id != i
               for i, d in enumerate(domains)):
            raise ValueError("domain-fit currently fits linear parent chains numbered 1..N. "
                             "Other layouts still run normally; their files are unchanged.")
        if any(d.run.dx != d.run.dy for d in domains):
            raise ValueError("domain-fit currently fits square cells; the original "
                             "rectangular-cell configuration remains runnable unchanged.")
        self.ratios = tuple(d.parent_grid_ratio for d in domains[1:])
        self.dx = domains[0].run.dx
        # Resolve only paths owned by the companion schemas, never arbitrary strings.
        if "case_data" in self.raw:
            from gpuwm.case_data import resolved_case_data_paths
            self.raw["case_data"] = resolved_case_data_paths(
                self.raw["case_data"], base_dir=authority.base_dir, source=str(self.path))
            from gpuwm.namelist_import import read_namelist_role
            self.wps_base = Path(self.raw["case_data"]["wps_namelist"])
            self.wps_original = read_namelist_role(self.wps_base, "starter WPS")
            # The original WPS is a geometry declaration, not reusable fitted geometry.
            self.raw["case_data"]["wps_namelist"] = str(self.out.with_suffix(".namelist.wps"))
        if "static" in self.raw:
            from gpuwm.static.highres_production import parse_static_table
            config = parse_static_table(self.raw["static"], source=str(self.path),
                                        base_dir=authority.base_dir)
            self.raw["static"]["highres"]["cache_root"] = str(config.cache_root.resolve())

    def tables(self, dims):
        raw = copy.deepcopy(self.raw)
        for i, (nx, ny) in enumerate(dims):
            table = raw["domain"][i]
            table.update(nx=nx, ny=ny)
            if i:
                ratio = self.ratios[i - 1]
                table.update(i_parent_start=(dims[i - 1][0] - nx // ratio) // 2 + 1,
                             j_parent_start=(dims[i - 1][1] - ny // ratio) // 2 + 1)
        return raw

    def candidate(self, dims):
        from gpuwm.experiment import build_experiment_from_config_tables
        return build_experiment_from_config_tables(
            self.tables(dims), source=str(self.out), base_dir=self.out.parent)

    def wps_text(self, generated, start, hours):
        """Keep declared nongeometry WPS settings when replacing its geometry."""
        # The ordinary wizard uses display precision for dx; a template may
        # carry more digits. Preserve its exact binary64 geometry in WPS too.
        for axis in ("dx", "dy"):
            generated = generated.replace(f" {axis} = {float(self.dx):g},",
                                            f" {axis} = {float(self.dx)!r},")
        if self.wps_original is None:
            return generated, []
        from gpuwm.namelist_import import parse_namelist_text
        original = self.wps_original
        fitted = copy.deepcopy(original)
        geometry = parse_namelist_text(generated)
        # geog_data_res and I/O choices remain user-owned.
        for key in ("max_dom", "interval_seconds"):
            fitted.setdefault("share", {})[key] = geometry["share"][key]
        for key, value in geometry["geogrid"].items():
            if key != "geog_data_res":
                fitted.setdefault("geogrid", {})[key] = value
        for key, moment in (("start_date", start), ("end_date", start + timedelta(hours=hours))):
            if key in fitted["share"]:
                fitted["share"][key] = [moment.strftime("%Y-%m-%d_%H:%M:%S")] * len(self.exp.domains)
        for section, key in (("geogrid", "geog_data_path"),
                             ("geogrid", "opt_geogrid_tbl_path"),
                             ("metgrid", "opt_metgrid_tbl_path")):
            if key in fitted.get(section, {}):
                fitted[section][key] = [str((self.wps_base.parent / value).resolve())
                                        for value in fitted[section][key]]
        def value(v):
            if isinstance(v, str):
                return "'" + v.replace("'", "''") + "'"
            if isinstance(v, bool):
                return ".true." if v else ".false."
            return repr(v)
        text = "\n".join("&" + section + "\n" + "\n".join(
                f" {key} = " + ", ".join(value(v) for v in values) + ","
                for key, values in table.items()) + "\n/"
                for section, table in fitted.items()) + "\n"
        if parse_namelist_text(text) != fitted:
            raise ValueError("Starter WPS settings did not round-trip; no output written")
        return text, changes(original, fitted, "wps")


def _utc(value):
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
    return parsed


def fit_main(args):
    from gpuwm import domain_wizard as dw
    out = args.out.expanduser().resolve()
    starter = Starter(args.template, out)
    wps = out.with_suffix(".namelist.wps")
    receipt = out.with_suffix(".fit.json")
    if out == starter.path or any(p.exists() for p in (out, wps, receipt)):
        raise ValueError("Choose a new --out path: domain-fit never overwrites the "
                         "starter, an existing configuration, WPS file, or fit receipt.")
    raw = starter.raw
    source = args.source or raw.get("fetch", {}).get("source")
    if not source:
        raise ValueError("Declare --source for pricing this template's input route; "
                         "or retain its existing [fetch].source declaration.")
    source = dw.resolve_source(source)
    if raw.get("fetch", {}).get("source") and source != dw.resolve_source(raw["fetch"]["source"]):
        raise ValueError("--source conflicts with the template's [fetch].source; "
                         "edit that declaration explicitly before fitting.")
    sizing = dw.resolve_sizing_budget(args.card, args.vram_gib)
    vram, device, note = sizing.vram_gib, sizing.device_profile, sizing.note
    if not math.isfinite(vram) or vram <= 0:
        raise ValueError("VRAM must be a finite positive GiB capacity")
    free = sizing.free_bytes
    footprint = dw.load_polygon_footprint(args.polygon) if args.polygon else None
    lat, lon = ((footprint.center_lat, footprint.center_lon) if footprint else
                dw._parse_point(args.point))
    projection = raw["projection"]
    # Keep projection family, standard parallels and rotation authoritative.
    projection.update(ref_lat=lat, ref_lon=lon)
    experiment = raw["experiment"]
    if args.start_time:
        experiment["start_time"] = _utc(args.start_time)
    if args.hours is not None:
        if not math.isfinite(args.hours) or args.hours <= 0:
            raise ValueError("--hours must be finite and positive")
        experiment["run_seconds"] = args.hours * 3600
    start = experiment["start_time"]
    if isinstance(start, str):
        start = _utc(start)
    hours = experiment["run_seconds"] / 3600
    fetch = raw.get("fetch")
    interval = raw.get("case_data", {}).get("forcing_interval_s")
    if interval is None:
        interval = (float(fetch["cadence"]) * 3600 if fetch and "cadence" in fetch
                    else dw.source_forcing_interval_seconds(source))
    if fetch:
        if args.start_time:
            cycle = start - timedelta(hours=fetch.get("forecast_start_hour", 0))
            if cycle.minute or cycle.second or cycle.microsecond:
                raise ValueError("The fetch cycle must fall on an exact UTC hour; "
                                 "keep the original input window or choose an hourly start.")
            fetch["cycle"] = cycle.strftime("%Y-%m-%dT%H")
        if args.hours is not None:
            fetch["hours"] = max(1, math.ceil(hours / (interval / 3600)) * (interval / 3600))
        # Fetch paths are interpreted by the fetch command relative to its cwd,
        # not the TOML directory. Preserve that meaning when printing the new file.
        if fetch.get("out"):
            fetch["out"] = str(Path(fetch["out"]).expanduser().resolve())
    common = dict(ratios=starter.ratios, root_dx_m=starter.dx,
                  free_bytes=free, hours=hours, start_time=start,
                  projection=projection, source=source, name=experiment["name"],
                  vram_gib=vram, device_profile=device,
                  forcing_interval_seconds=interval,
                  candidate_builder=starter.candidate,
                  clearance_rows=starter.exp.spec_bdy_width + starter.exp.blend_width,
                  minimum_axis=max(dw.FIFTH_ORDER_STENCIL_AXIS,
                      dw.boundary_axis(starter.exp.spec_bdy_width, interior_points=1),
                      *(dw.boundary_axis(max(d.run.spec_zone, d.run.relax_zone),
                                         interior_points=1) for d in starter.exp.domains)))
    if footprint:
        buffers = dw._buffers_for_levels(dw.parse_level_buffers(args.buffer_km),
                                          len(starter.ratios) + 1)
        dims, exp = dw.fit_polygon_ladder(footprint=footprint, buffers_km=buffers, **common)
    else:
        if args.buffer_km is not None:
            raise ValueError("--buffer-km requires --polygon")
        dims, exp = dw.fit_ladder(**common)
    dw._pole_clearance_refusal(projection, *dims[0], starter.dx,
                               target_option="--polygon" if footprint else "--point")
    if fetch and dw.source_fetch_takes_a_crop_box(source):
        for key in ("point", "radius_km"):
            fetch.pop(key, None)
        fetch["area"] = dw.fetch_area_hint(projection, *dims[0], source=source,
                                           root_dx_m=starter.dx)
    final = starter.tables(dims)
    # Complete companion validation and final repricing of exactly published bytes.
    text = render_tables(final)
    from gpuwm.experiment import build_experiment_from_config_tables
    published = build_experiment_from_config_tables(tomllib.loads(text),
                source=str(out), base_dir=out.parent)
    phases = dw._sizing_phases(published, free_bytes=free, source=source,
                              forcing_interval_seconds=interval,
                              vram_gib=vram, profile=device)
    budget = dw.sizing_budget_bytes(published, free_bytes=free, vram_gib=vram,
                                    forcing_interval_seconds=interval, profile=device)
    if phases.peak_envelope_bytes > budget:
        raise dw.DomainFitError("Final template exceeds the canonical phase budget: "
                                + phases.verdict(budget))
    wps_text, wps_delta = starter.wps_text(dw.render_wps_namelist(
        projection, dims, starter.ratios, root_dx_m=starter.dx,
        source=source, forcing_interval_seconds=interval), start, hours)
    delta = changes(starter.original, final) + wps_delta
    print(f"Template: {starter.path}\nResolved configuration: {out}")
    if note:
        print(note)
    print("Exact changes (all other settings preserved):")
    for key, before, after in delta:
        print(f"  {key}: {before!r} -> {after!r}")
    print(phases.verdict(budget))
    if not args.write:
        print("Preview only. Add --write to create this configuration and its WPS file.")
        return 0
    proof = dict(template=str(starter.path), template_sha256=starter.sha256,
                 output=str(out), output_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
                 wps_sha256=hashlib.sha256(wps_text.encode("utf-8")).hexdigest(), source=source,
                 changes=[dict(field=k, before=a, after=b) for k, a, b in delta],
                 peak_envelope_bytes=phases.peak_envelope_bytes, budget_bytes=budget,
                 free_bytes=free, sizing_basis=("measured-available" if sizing.measured
                                               else "declared-capacity"),
                 launch_performed=False)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation: never replace a file that appeared during the fit.
    _publish_new_files(((wps, wps_text),
                       (receipt, json.dumps(proof, indent=2, default=str) + "\n"),
                       (out, text)))
    print(f"Created {out}\nCreated {wps}\nFit receipt: {receipt}")
    if fetch or "case_data" in raw:
        print(f"Next: gpuwm go {_command_path(out)} --dry-run")
        print("Then remove --dry-run to prepare fresh inputs and run. No forecast has started.")
    else:
        print(f"Next: gpuwm check {shlex.quote(str(out))}")
        print("This template declares no acquisition inputs. Connect its intended input "
              "route before launching; no source or prepared artifacts were invented.")
    return 0


def _tiles_tables(authority, mode):
    """Change only the requested tile mode and schema-owned path spellings."""
    from gpuwm.experiment import build_experiment_from_config_tables
    original = tomllib.loads(authority.payload.decode("utf-8"))
    build_experiment_from_config_tables(
        original, source=str(authority.source), base_dir=authority.base_dir)
    raw = copy.deepcopy(original)
    configured = raw.get("tiles", {})
    pins = sorted(set(configured) & {
        "tile_nx", "tile_ny", "nbuffers", "halo", "vram_budget_bytes",
        "host_budget_bytes"})
    overrides = [str(domain["grid_id"]) for domain in raw["domain"]
                 if "tiles" in domain]
    if pins or overrides or configured.get("store", "host") != "host":
        details = []
        if pins:
            details.append("[tiles] declares " + ", ".join(pins))
        if overrides:
            details.append("domain tile overrides exist on grid(s) "
                           + ", ".join(overrides))
        if configured.get("store", "host") != "host":
            details.append("[tiles] selects a device store")
        raise ValueError("domain-tiles cannot replace explicit streaming choices: "
                         + "; ".join(details)
                         + ". Review those settings explicitly; no files were written.")
    if "case_data" in raw:
        from gpuwm.case_data import resolved_case_data_paths
        raw["case_data"] = resolved_case_data_paths(
            raw["case_data"], base_dir=authority.base_dir,
            source=str(authority.source))
    if "static" in raw:
        from gpuwm.static.highres_production import parse_static_table
        highres = parse_static_table(
            raw["static"], source=str(authority.source),
            base_dir=authority.base_dir)
        if highres is not None:
            raw["static"]["highres"]["cache_root"] = str(highres.cache_root.resolve())
    rebased = copy.deepcopy(raw)
    raw["tiles"] = {**configured, "mode": mode, "store": "host"}
    unchanged = copy.deepcopy(raw)
    if "tiles" in rebased:
        unchanged["tiles"] = rebased["tiles"]
    else:
        unchanged.pop("tiles")
    if unchanged != rebased:
        raise RuntimeError("Tile selection changed unrelated configuration settings")
    return original, raw


def _tiles_wps(source, output, *, text):
    """Copy a WPS authority, rebasing only its declared directory references."""
    from gpuwm.namelist_import import parse_namelist_text
    original = parse_namelist_text(text)
    if source.parent == output.parent:
        return text
    tables = copy.deepcopy(original)
    for section, key in (("geogrid", "geog_data_path"),
                         ("geogrid", "opt_geogrid_tbl_path"),
                         ("metgrid", "opt_metgrid_tbl_path")):
        if key in tables.get(section, {}):
            tables[section][key] = [
                str((source.parent / value).resolve())
                for value in tables[section][key]]

    def value(item):
        if isinstance(item, str):
            return "'" + item.replace("'", "''") + "'"
        if isinstance(item, bool):
            return ".true." if item else ".false."
        return repr(item)

    text = "\n".join("&" + section + "\n" + "\n".join(
        f" {key} = " + ", ".join(value(item) for item in values) + ","
        for key, values in table.items()) + "\n/"
        for section, table in tables.items()) + "\n"
    if parse_namelist_text(text) != tables:
        raise ValueError("WPS settings did not round-trip; no output written")
    return text


def _tiles_memory_plan(path, experiment, *, original):
    """Use the launch planner with one device observation and available RAM."""
    from dataclasses import replace
    from gpuwm import domain_wizard as dw
    from gpuwm.core import preflight, streaming
    from gpuwm.runplan import prepared_chain_for_source, streaming_decision
    source = preflight.config_forcing_source(path, priced_only=False)
    chain = ("experiment" if "case_data" in original or source is None
             else prepared_chain_for_source(source))
    delivery = streaming_decision(experiment, chain=chain)
    if delivery is not None and delivery["refusal"]:
        raise ValueError(delivery["refusal"])
    sizing = dw.resolve_sizing_budget(None, None)
    machine = streaming.planner_machine(
        vram_bytes=sizing.free_bytes, name="domain-tiles available memory")
    available = preflight.host_available_bytes()
    if machine is None or available is None:
        raise ValueError("domain-tiles could not measure host RAM for the streamed "
                         "store; no configuration was written.")
    # A host store must fit both the canonical page-locking allowance and the
    # physical RAM available now. Never freeze either observation in the TOML.
    machine = replace(machine, host_bytes=min(machine.host_budget_bytes, available),
                      pinned_fraction=1.0)
    interval, intervals = preflight.config_forcing_schedule(path, experiment)
    phases = preflight.estimate_phases(
        experiment, source=source, machine=machine,
        forcing_interval_seconds=(interval if interval is not None else
                                  preflight.DEFAULT_FORCING_INTERVAL_SECONDS),
        forcing_intervals=intervals, ingest_forcing_interval_seconds=interval,
        vram_gib=sizing.vram_gib, profile=sizing.device_profile)
    budget = dw.sizing_budget_bytes(
        experiment, free_bytes=sizing.free_bytes, vram_gib=sizing.vram_gib,
        forcing_interval_seconds=interval, profile=sizing.device_profile)
    if len(experiment.domains) > 1:
        road = phases.tree_road
        if road is None or road.refusal is not None or not road.priced:
            raise ValueError("No fitting automatic tile plan: "
                             + (road.refusal if road is not None and road.refusal
                                else "the domain tree could not be priced"))
        rows = [{**row, **row.get("tile", {})} for row in road.rows]
    else:
        try:
            decision = streaming.decide(
                experiment.root.run, experiment.tiles, machine=machine,
                resident_estimate=phases.forecast)
        except Exception as error:
            raise ValueError(f"No fitting automatic tile plan: {error}") from error
        rows = [dict(grid_id=experiment.root.grid_id,
                     road="streamed" if decision.stream else "resident",
                     reason=decision.reason, tile_nx=decision.tile_nx,
                     tile_ny=decision.tile_ny, nbuffers=decision.nbuffers,
                     halo=decision.halo)]
        if decision.stream and phases.streamed is None:
            raise ValueError("The selected tile plan could not be priced; no output written")
    host_bytes = 0 if phases.streamed is None else phases.streamed.host_bytes
    if (phases.peak_envelope_bytes > budget
            or host_bytes > machine.host_budget_bytes):
        raise ValueError("Tile streaming does not fit the available memory: "
                         + phases.verdict(budget)
                         + f"; host store {host_bytes / dw.GIB:.2f} GiB against "
                         f"{machine.host_budget_bytes / dw.GIB:.2f} GiB available allowance")
    return dict(source=source, mode=experiment.tiles.mode, domains=rows,
                peak_envelope_bytes=phases.peak_envelope_bytes,
                budget_bytes=budget, free_bytes=sizing.free_bytes,
                host_store_bytes=host_bytes, host_budget_bytes=machine.host_budget_bytes,
                host_available_bytes=available, verdict=phases.verdict(budget),
                ingest_priced=phases.ingest_priced,
                launch_performed=False, download_performed=False)


def tiles_main(args):
    """Review or create an automatic tile copy; never resize or launch it."""
    from gpuwm.config_authority import read_config_authority
    from gpuwm.experiment import build_experiment_from_config_tables
    from gpuwm.toml_document import emit_experiment_toml
    authority = read_config_authority(args.template)
    out = args.out.expanduser().resolve()
    wps = out.with_suffix(".namelist.wps")
    receipt = out.with_suffix(".tiles.json")
    if out == authority.source or any(path.exists() for path in (out, wps, receipt)):
        raise ValueError("Choose a new --out path: domain-tiles never overwrites the "
                         "source, an existing configuration, WPS file, or tile receipt.")
    original, raw = _tiles_tables(authority, args.mode)
    text = emit_experiment_toml(raw)
    experiment = build_experiment_from_config_tables(
        tomllib.loads(text), source=str(out), base_dir=out.parent)
    source_wps = authority.source.with_suffix(".namelist.wps")
    wps_payload = source_wps.read_bytes() if source_wps.is_file() else None
    wps_text = (_tiles_wps(source_wps, wps, text=wps_payload.decode("utf-8"))
                if wps_payload is not None else None)
    if raw.get("fetch") and "case_data" not in raw:
        from gpuwm.runplan import prepared_chain_for_source
        if (prepared_chain_for_source(raw["fetch"]["source"]) == "prepared:go"
                and wps_text is None):
            raise ValueError(f"The source configuration needs its WPS companion "
                             f"{source_wps}; no files were written.")
    proof = _tiles_memory_plan(authority.source, experiment, original=original)
    if hashlib.sha256(authority.source.read_bytes()).hexdigest() != authority.sha256:
        raise ValueError("The source configuration changed during planning; review it again.")
    if wps_payload is not None and source_wps.read_bytes() != wps_payload:
        raise ValueError("The source WPS companion changed during planning; review it again.")
    delta = changes(original, raw)
    print(f"Template: {authority.source}\nResolved configuration: {out}")
    print("Area, grid spacing, physics, clocks, and other forecast settings are preserved.")
    print("Exact changes:")
    for key, before, after in delta:
        print(f"  {key}: {before!r} -> {after!r}")
    for row in proof["domains"]:
        detail = (f"; planner tile {row['tile_nx']}x{row['tile_ny']}, "
                  f"{row['nbuffers']} buffers" if row["road"] == "streamed" else "")
        print(f"  d{row['grid_id']:02d}: {row['road']}{detail}")
    print(proof["verdict"])
    print("Tile dimensions stay automatic and are checked again when the forecast starts.")
    if not proof["ingest_priced"]:
        print("Preprocessing is not priced for this input route; this estimate covers the forecast.")
    if not args.write:
        print("Preview only. Add --write to create this configuration. No forecast has started.")
        return 0
    proof.update(template=str(authority.source), template_sha256=authority.sha256,
                 output=str(out), output_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
                 changes=[dict(field=key, before=before, after=after)
                          for key, before, after in delta])
    if raw.get("fetch", {}).get("out") is not None:
        proof["fetch_out_path_basis"] = (
            "The unchanged fetch.out declaration remains relative to the fetch "
            "command's working directory; gpuwm go chooses its own cache path.")
    files = []
    if wps_text is not None:
        proof["source_wps"] = str(source_wps)
        proof["source_wps_sha256"] = hashlib.sha256(wps_payload).hexdigest()
        proof["wps_sha256"] = hashlib.sha256(wps_text.encode("utf-8")).hexdigest()
        files.append((wps, wps_text))
    files.extend(((receipt, json.dumps(proof, indent=2, default=str) + "\n"),
                  (out, text)))
    out.parent.mkdir(parents=True, exist_ok=True)
    _publish_new_files(files)
    print(f"Created {out}\nTile receipt: {receipt}")
    print(f"Next: gpuwm check {_command_path(out)}")
    print("No forecast has started.")
    return 0


def register_cli(subparsers):
    parser = subparsers.add_parser("domain-fit", help="explicitly fit an editable "
                                  "TOML to an area/device; preserve exact scientific settings")
    parser.add_argument("template", type=Path, help="ordinary complete experiment TOML")
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--point", help="center LAT,LON; fit largest centered layout")
    target.add_argument("--polygon", type=Path, help="GeoJSON area; preserve its entire footprint")
    parser.add_argument("--buffer-km", help="polygon buffer(s), outer to inner")
    device = parser.add_mutually_exclusive_group()
    device.add_argument("--card", help="existing named GPU tier")
    device.add_argument("--vram-gib", type=float, help="target total VRAM capacity in GiB; "
                       "omit device flags to detect this machine\'s GPU")
    parser.add_argument("--source", help="input source, required only without [fetch].source")
    parser.add_argument("--start-time", help="explicit new UTC start; otherwise preserve template")
    parser.add_argument("--hours", type=float, help="explicit new duration; otherwise preserve template")
    parser.add_argument("--out", type=Path, required=True, help="new ordinary TOML path")
    parser.add_argument("--write", action="store_true", help="write reviewed TOML, WPS and fit receipt")
    parser.set_defaults(func=fit_main)
    tiles = subparsers.add_parser(
        "domain-tiles", help="review an automatic tile-streaming copy; preserve forecast settings")
    tiles.add_argument("template", type=Path, help="ordinary complete experiment TOML")
    tiles.add_argument("--out", type=Path, required=True, help="new ordinary TOML path")
    tiles.add_argument("--mode", choices=("auto", "on"), default="auto",
                       help="auto streams when needed; on forces planner-selected streaming")
    tiles.add_argument("--write", action="store_true", help="create the reviewed TOML and tile receipt")
    tiles.set_defaults(func=tiles_main)
