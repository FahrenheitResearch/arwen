"""Radar latent heating in a deterministic WOOF forecast: the door hook.

``[radar_heating]`` (:class:`gpuwm.config.RadarHeatingConfig`) turns lane
6's NOAA heating rule (:mod:`gpuwm.da.radar_tten`) on for the first
``active_minutes`` of a plain forecast: the forcing attaches at the run
start, reads the window ending at ``start + n * window_minutes`` while the
clock is inside window ``n``, and passes every microphysics step through
after the forced period.  Arm B ("start at 17Z with observed Level II
windows") and arm C ("start at 18Z with nowcast windows") are the same
code; the case decides which windows it names.

What this module owns
---------------------
* the execution argument (``--radar-heating-table``) that overlays the
  table on a prepared door without editing the hash-bound experiment;
* :func:`grid_for_prepared`, the run's grid identity, computed from the
  prepared root by the same function ``tools/radar_tten_windows.py
  --grid-prepared`` uses to write the windows' grid descriptor, so the
  producer and the door agree by construction;
* the strict window read on every path: each window is held to its receipt
  and to that identity before anything is built;
* the attach on both roads the door has: one resident card (the existing
  :func:`radar_tten.build_tendency` on the state) and resident ranks on
  several cards (gather, build, scatter: the four background fields are
  gathered from the ranks' interiors onto the first card, each slot is
  built there with the same function, and each rank receives its slab,
  halos included, so the slots are the one-card slots by construction);
* the refusals, each naming its breakage; the resume rule; the receipt.

Nothing here runs a kernel of its own.  Every number comes from
:func:`radar_tten.build_tendency` and :class:`radar_tten.RadarTtenForcing`.
"""
from __future__ import annotations

import json
import re
import time
import tomllib
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import numpy as np

#: The receipt this hook writes into the run report.
SCHEMA = "gpuwm-da.forecast-heating.v1"

#: Set on the experiment state by a route that honored ``[radar_heating]``
#: (attached it, or decided a resume is past the forced period).
#: :func:`require_routed` refuses an enabled table on a state without it.
ROUTE_ATTRIBUTE = "_radar_heating_route"

#: The gravity :meth:`gpuwm.obs.target_grid.TargetGrid.from_wrfout` divides
#: geopotential by, so a grid read from a prepared root and one read from a
#: wrfout of the same state have the same columns.
STANDARD_GRAVITY = 9.81

#: Whole-domain float32 volumes the build card holds at the peak of an
#: attach, on top of the slots it keeps: the gathered inputs (or the
#: background), the window, the cone-filled copy, the float64 tendency and
#: its smoothing buffer (two volumes each) and the slot being built.
BUILD_TRANSIENT_VOLUMES = 12

REFUSAL_LABEL = "[radar_heating]"


class ForecastHeatingRefused(ValueError):
    """``[radar_heating]`` cannot be honoured as configured.  Never a warning."""


def _config_type():
    from gpuwm.config import RadarHeatingConfig
    return RadarHeatingConfig


# --------------------------------------------------------------------------
# the execution argument
# --------------------------------------------------------------------------

def execution_argument(value):
    """``None``, or the table from ``--radar-heating-table`` (JSON, a
    mapping or a config).  A relative windows root on a command line is the
    caller's working directory, made absolute here so a subprocess in
    another directory reads the same windows."""
    config_type = _config_type()
    if value is None or isinstance(value, config_type):
        return value
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError as error:
            raise ValueError(
                f"--radar-heating-table must contain a JSON object: {error}"
            ) from None
    config = config_type.from_mapping(value, source="--radar-heating-table")
    return replace(config, windows=str(Path(config.windows).resolve()))


def execution_flags(config) -> list[str]:
    """The flag that carries the table across a prepared-run subprocess."""
    config = execution_argument(config)
    if config is None or not config.enabled:
        return []
    return ["--radar-heating-table",
            json.dumps(config.to_mapping(), sort_keys=True)]


def add_execution_argument(parser) -> None:
    import argparse

    def parse(value):
        try:
            return execution_argument(value)
        except (ValueError, TypeError) as error:
            raise argparse.ArgumentTypeError(str(error)) from None

    parser.add_argument(
        "--radar-heating-table", default=None, metavar="JSON", type=parse,
        help="[radar_heating] as JSON: radar latent heating windows for the "
             "first active_minutes of the run, without editing a prepared "
             "configuration")


def apply_execution_options(experiment, config):
    """Overlay the table after the preparation identity has been checked.

    Like ``[tiles]``, the table rides beside the hash-bound experiment
    rather than inside it, so one preparation serves the unheated control
    and every heated arm.  A table given here replaces one the document
    carries: it is the later and more specific statement."""
    if config is None:
        return experiment
    return replace(experiment, radar_heating=execution_argument(config))


_TABLE_HEADER = re.compile(r"^\s*\[\s*radar_heating\s*\]\s*(#.*)?$")
_ANY_HEADER = re.compile(r"^\s*\[")


def split_experiment_text(text: str) -> tuple[str, str | None]:
    """``(base, table)``: the experiment text with its ``[radar_heating]``
    table cut out, and that table's own text (``None`` when there is none).

    The base keeps every other byte, including line endings; one blank
    line written just before the table goes with it, so a config made by
    appending the table to the prepared config splits back into exactly
    the prepared config's bytes."""
    lines = text.splitlines(keepends=True)
    head = next((n for n, line in enumerate(lines) if _TABLE_HEADER.match(line)), None)
    if head is None:
        return text, None
    end = next((n for n in range(head + 1, len(lines)) if _ANY_HEADER.match(lines[n])),
               len(lines))
    start = head - 1 if head > 0 and not lines[head - 1].strip() else head
    return "".join(lines[:start] + lines[end:]), "".join(lines[head:end])


#: The preparation documents that record the experiment config's digest
#: (the portable source manifest, the mapped evidence manifest, the proof's
#: execution inputs).  Read only to ask whether the preparation was made
#: from the config WITH its table.
_BINDING_DOCUMENTS = ("proof.json", "source-input-manifest.json",
                      "source-evidence/input-manifest.json")
_BINDING_DOCUMENT_LIMIT = 64 * 2**20


def preparation_binds(prepared_root, sha256: str) -> bool:
    """Whether a document of the preparation at ``prepared_root`` records
    ``sha256`` (a config digest).  A 64-hex digest cannot appear by chance."""
    if prepared_root is None:
        return False
    for name in _BINDING_DOCUMENTS:
        path = Path(prepared_root) / name
        try:
            if path.is_file() and path.stat().st_size <= _BINDING_DOCUMENT_LIMIT                     and sha256 in path.read_text(encoding="utf-8", errors="replace"):
                return True
        except OSError:
            continue
    return False


def detach_table_from_config(path, *, prepared_root=None):
    """``(config_path, heating)`` for the door's ``--experiment-config``.

    A config that carries ``[radar_heating]`` is normally the prepared
    config plus an execution control (one preparation serves the unheated
    control and every heated arm).  The table is cut out and the rest is
    written BESIDE the given config as ``<stem>.bound-<digest>.toml``, so
    every path the config states relative to its own directory still
    resolves to the same file, and the preparation's binding checks that
    copy byte for byte exactly as before.  The table is returned as the
    heating.  The rest must parse to the same document minus the table, or
    the split is refused: it may remove the table and nothing else.

    When the preparation at ``prepared_root`` was made from the config WITH
    the table (``gpuwm go`` on the staged route hands the door the very
    file it prepared from), the config is returned unchanged with its
    table: cutting it would break that binding.  A config without the
    table is returned as it is, with ``None``."""
    import hashlib
    import os

    path = Path(path)
    if not path.is_file():
        # The preflight refuses a missing config in its own words.
        return path, None
    # Bytes, not text mode: the binding is byte for byte, line endings too.
    raw = path.read_bytes()
    text = raw.decode("utf-8")
    base, table = split_experiment_text(text)
    if table is None:
        return path, None
    whole = tomllib.loads(text)
    heating_table = whole.pop("radar_heating", None)
    if tomllib.loads(base) != whole or heating_table is None:
        raise ForecastHeatingRefused(
            f"{path}: cutting the [radar_heating] table out changed another "
            "part of the configuration; the preparation binding would check "
            "a config that is not the one given")
    config = _config_type().from_mapping(heating_table, source=str(path),
                                         base_dir=path.parent)
    if preparation_binds(prepared_root, hashlib.sha256(raw).hexdigest()):
        return path, config
    data = base.encode("utf-8")
    digest = hashlib.sha256(data).hexdigest()
    bound = path.with_name(f"{path.stem}.bound-{digest[:16]}.toml")
    if not (bound.is_file() and bound.read_bytes() == data):
        temporary = bound.with_name(f"{bound.name}.tmp{os.getpid()}")
        try:
            temporary.write_bytes(data)
            os.replace(temporary, bound)
        except OSError as error:
            temporary.unlink(missing_ok=True)
            raise ForecastHeatingRefused(
                f"{path}: the config without its [radar_heating] table cannot "
                f"be written beside it ({error}); written anywhere else, the "
                "paths it states relative to its own directory would resolve "
                "to other files.  Pass the prepared config with "
                "--radar-heating-table instead") from None
    return bound, config


# --------------------------------------------------------------------------
# refusals
# --------------------------------------------------------------------------

def enabled(experiment) -> bool:
    heating = getattr(experiment, "radar_heating", None)
    return heating is not None and bool(getattr(heating, "enabled", False))


def refuse_tree(domain_count: int, route: str) -> None:
    """A heated nested or tree run is refused.

    The forcing heats one domain: a child's feedback would overwrite heated
    parent cells, and the child itself would run unheated under a heated
    record."""
    if int(domain_count) > 1:
        raise ForecastHeatingRefused(
            f"{REFUSAL_LABEL} on the {route}: the configuration carries "
            f"{int(domain_count)} domains and the forcing heats one.  A "
            "child's feedback would overwrite heated parent cells and the "
            "child would run unheated while the record says heated")


def require_routed(model, experiment=None) -> None:
    """Refuse an enabled table on a route that did not attach it.

    Asked by :func:`gpuwm.core.model.execute_experiment`, the one call every
    forecast route makes, so no route can drop the table silently."""
    exp = experiment
    if exp is None:
        exp = getattr(model, "_declared_experiment", None)
    if exp is None:
        context = getattr(model, "_activation_context", None)
        if isinstance(context, dict):
            exp = context.get("experiment")
    if not enabled(exp):
        return
    if getattr(model, ROUTE_ATTRIBUTE, None) is None:
        raise ForecastHeatingRefused(
            f"{REFUSAL_LABEL} is set and this route does not attach radar "
            "heating: it would integrate an unheated forecast under the "
            "name of a heated one.  The deterministic prepared door "
            "(gpuwm.prepared_single_domain_forecast, --radar-heating-table "
            "or the table in its experiment) applies it")


def resume_plan(heating, restored_seconds) -> dict:
    """Attach on a fresh start; refuse a resume inside the forced period;
    run free after it, as the uninterrupted run does.

    The slot clock starts at zero on attach, so a forcing attached to a
    state restored at minute 70 would read window 1 at minute 70."""
    if restored_seconds is None:
        return {"attach": True, "status": "FORCED"}
    restored = float(restored_seconds)
    active = float(heating.active_minutes) * 60.0
    if restored < active:
        raise ForecastHeatingRefused(
            f"{REFUSAL_LABEL}: the checkpoint is at {restored:g} s, inside "
            f"the forced period of {active:g} s.  The forcing's slot clock "
            "starts at zero on attach, so the windows would be read at the "
            "wrong times.  Restart from the run start, or from a checkpoint "
            "at or after the end of the forced period")
    return {"attach": False, "status": "RESUMED_AFTER_FORCED_PERIOD",
            "restored_seconds": restored,
            "note": "resumed after the forced period: no forcing attached, "
                    "which is what the uninterrupted run does from here"}


# --------------------------------------------------------------------------
# the grid
# --------------------------------------------------------------------------

_CACHE_CANDIDATES = ("native/prepared-cache", "prepared-cache", ".")
_GEOMETRY_CANDIDATES = ("native-geometry-receipt.json", "geometry-receipt.json")


def prepared_cache_directory(root) -> Path:
    root = Path(root)
    for candidate in _CACHE_CANDIDATES:
        directory = (root / candidate)
        if (directory / "header.json").is_file():
            return directory
    raise ForecastHeatingRefused(
        f"{root} holds no prepared cache (looked for "
        f"{[str(root / c) for c in _CACHE_CANDIDATES]}): the run's grid "
        "identity cannot be computed, and windows read without it may hold "
        "other columns")


def _geometry(root) -> dict:
    root = Path(root)
    for candidate in _GEOMETRY_CANDIDATES:
        path = root / candidate
        if path.is_file():
            document = json.loads(path.read_text(encoding="utf-8"))
            geometry = document.get("geometry")
            if isinstance(geometry, dict):
                return geometry
            raise ForecastHeatingRefused(f"{path} carries no geometry block")
    raise ForecastHeatingRefused(
        f"{root} holds no geometry receipt ({list(_GEOMETRY_CANDIDATES)})")


def grid_for_prepared(root):
    """The run's :class:`gpuwm.obs.target_grid.TargetGrid`, from a prepared
    root: the projection from its geometry receipt, ``z_w`` from the
    prepared start state's geopotential (``state/php + base/phb``, over
    :data:`STANDARD_GRAVITY`), terrain from ``base/terrain_z``.

    ``tools/radar_tten_windows.py --grid-prepared`` writes the windows'
    grid descriptor from this function and the door computes the identity
    it reads them against from this function, so the two agree by
    construction.  Every array is held to its manifest digest on read."""
    from gpuwm.ingest.prepared_cache import read_manifest_array
    from gpuwm.obs.target_grid import TargetGrid
    from gpuwm.static.projection import projection_class

    root = Path(root)
    geometry = _geometry(root)
    cache = prepared_cache_directory(root)
    header = json.loads((cache / "header.json").read_text(encoding="utf-8"))
    arrays = header.get("arrays") or {}
    for key in ("state/php", "base/phb"):
        if key not in arrays:
            raise ForecastHeatingRefused(
                f"prepared cache {cache} carries no {key}: the column heights "
                "the grid identity binds cannot be built")
    ny, nx = (int(v) for v in geometry["mass_shape"])
    projection = projection_class(str(geometry.get("map_proj", "lambert")))(
        ref_lat=float(geometry["ref_lat"]), ref_lon=float(geometry["ref_lon"]),
        truelat1=float(geometry["truelat1"]),
        truelat2=float(geometry["truelat2"]),
        stand_lon=float(geometry["stand_lon"]),
        dx=float(geometry["dx_m"]), dy=float(geometry["dy_m"]),
        e_we=nx + 1, e_sn=ny + 1,
        known_x=geometry.get("known_x"), known_y=geometry.get("known_y"),
        moad_cen_lat=geometry.get("moad_cen_lat"),
        moad_cen_lon=geometry.get("moad_cen_lon"))
    php = np.asarray(read_manifest_array(cache, "state/php", arrays["state/php"]),
                     dtype=np.float64)
    phb = np.asarray(read_manifest_array(cache, "base/phb", arrays["base/phb"]),
                     dtype=np.float64)
    if phb.ndim == 1:
        phb = phb[:, None, None]
    z_w = (php + phb) / STANDARD_GRAVITY
    terrain = None
    if "base/terrain_z" in arrays:
        terrain = np.asarray(read_manifest_array(
            cache, "base/terrain_z", arrays["base/terrain_z"]), dtype=np.float64)
    if z_w.shape[1:] != (ny, nx):
        raise ForecastHeatingRefused(
            f"prepared cache {cache} columns {z_w.shape[1:]} are not the "
            f"geometry receipt's {ny} x {nx} mass grid")
    return TargetGrid.from_projection(
        projection, z_w=z_w, terrain_m=terrain, name="prepared-d01",
        source=f"prepared:{cache.name}")


@lru_cache(maxsize=4)
def _identity_for(root: str) -> str:
    return grid_for_prepared(root).identity_sha256()


def prepared_grid_identity(root) -> str:
    """:func:`grid_for_prepared`'s identity, computed once per root and
    process (a cycle asks it once per member and leg)."""
    return _identity_for(str(Path(root).resolve()))


# --------------------------------------------------------------------------
# the windows
# --------------------------------------------------------------------------

def _utc_naive(when: datetime) -> datetime:
    if when.tzinfo is not None:
        when = when.astimezone(timezone.utc).replace(tzinfo=None)
    return when


def _parse_end(value) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    text = value.strip().replace("Z", "+00:00")
    try:
        return _utc_naive(datetime.fromisoformat(text))
    except ValueError:
        return None


def window_ends(heating, start_time) -> list[datetime]:
    start = _utc_naive(start_time)
    return [start + timedelta(minutes=m) for m in heating.window_end_minutes()]


def lead_class(receipt) -> str:
    """``forecast`` or ``oracle`` as the windows' producer stated it
    (``rw_nexrad grid-composite``); ``observed`` for a window gridded from
    observed volumes (``rw_nexrad grid-ref``), which states none."""
    source = receipt.get("source") if isinstance(receipt, dict) else None
    if isinstance(source, dict) and source.get("lead_class") in ("forecast", "oracle"):
        return str(source["lead_class"])
    return "observed"


def read_windows(heating, start_time, identity_sha256: str) -> list[dict]:
    """Every window of the forced period, held to its receipt, the run's
    grid and its own end time; the arrays are dropped, the facts kept.

    Refusals: a missing window, a window on another grid, a window whose
    receipt states no end time or another end time than its directory, a
    window whose bytes do not match its receipt."""
    from gpuwm.da import radar_tten

    paths = radar_tten.window_paths(
        heating.windows, _utc_naive(start_time), heating.active_minutes,
        heating.window_count)
    rows = []
    for end, path in zip(window_ends(heating, start_time), paths):
        host, receipt = radar_tten.read_window_host(
            path, identity_sha256=identity_sha256)
        stated = _parse_end((receipt.get("window") or {}).get("end"))
        if stated is None:
            # Both producers state it (radar_tten_windows.py passes
            # --window-end to grid-ref; grid-composite always writes it).
            raise ForecastHeatingRefused(
                f"window {path} sits under {end:%Y%m%dT%H%MZ} and its receipt "
                "states no end time: nothing shows it holds that window's "
                "storms rather than another time's")
        if stated != end:
            raise ForecastHeatingRefused(
                f"window {path} sits under {end:%Y%m%dT%H%MZ} but its receipt "
                f"says it ends {stated.isoformat()}: the slot would heat one "
                "window with another window's storms")
        source = receipt.get("source") if isinstance(receipt.get("source"), dict) else {}
        rows.append({
            "end": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "path": str(path),
            "shape": list(host.shape),
            "sha256": (receipt.get("data") or {}).get("sha256"),
            "lead_class": lead_class(receipt),
            "source_id": source.get("id"),
            "frame_valid": (receipt.get("window") or {}).get("frame_valid"),
            "counts": receipt.get("counts"),
        })
        del host
    return rows


def radar_config(heating):
    from gpuwm.da.radar_tten import RadarTtenConfig
    return RadarTtenConfig(
        latent_heat_period_min=float(heating.latent_heat_period_min),
        strict_suppression=bool(heating.strict_suppression),
        pbl_extension=bool(heating.pbl_extension))


# --------------------------------------------------------------------------
# memory
# --------------------------------------------------------------------------

def volume_bytes(nz: int, ny: int, nx: int) -> int:
    return 4 * int(nz) * int(ny) * int(nx)


def resident_bytes(heating, cfg) -> int:
    """One card: the kept slots, the theta snapshot and the build peak."""
    volume = volume_bytes(cfg.nz, cfg.ny, cfg.nx)
    return (heating.window_count + 1 + BUILD_TRANSIENT_VOLUMES) * volume


def slab_bytes_per_card(heating, cfg, specs, device_ids) -> dict[int, int]:
    """Each card's kept slab slots and theta snapshot, plus the build peak
    on the first card (the build card)."""
    devices = [int(d) for d in device_ids]
    out: dict[int, int] = {}
    for spec, dev in zip(specs, devices):
        out[dev] = out.get(dev, 0) + (heating.window_count + 1) * volume_bytes(
            cfg.nz, spec.cny, spec.cnx)
    out[devices[0]] = out.get(devices[0], 0) + BUILD_TRANSIENT_VOLUMES * volume_bytes(
        cfg.nz, cfg.ny, cfg.nx)
    return out


def ranked_reservation(heating, cfg, options, *, max_map_factor=1.0) -> dict[int, int]:
    """The bytes the forcing takes on each card of a ranked run, priced from
    the same rank plan the run will build (``streaming.ranked_specs``)."""
    from gpuwm.core import streaming

    halo = streaming.ranked_halo(cfg, max_map_factor=max_map_factor)
    specs = streaming.ranked_specs(cfg, options, halo=halo)
    return slab_bytes_per_card(heating, cfg, specs, options.device_ids())


# --------------------------------------------------------------------------
# the hook
# --------------------------------------------------------------------------

class ForecastHeating:
    """One run's forcing: validated windows, the attach, the receipt."""

    def __init__(self, heating, *, start_time, prepared_root,
                 restored_seconds=None):
        self.heating = heating
        self.start_time = _utc_naive(start_time)
        self.prepared_root = Path(prepared_root)
        self.plan = resume_plan(heating, restored_seconds)
        self.identity = None
        self.windows: list[dict] = []
        self.road = None
        self._attached: list = []
        self._receipts: list = []
        self.build: dict = {}

    # -- host-only checks, before anything touches a card -------------------
    def validate(self, cfg) -> None:
        grid = grid_for_prepared(self.prepared_root)
        if (grid.nz, grid.ny, grid.nx) != (int(cfg.nz), int(cfg.ny), int(cfg.nx)):
            raise ForecastHeatingRefused(
                f"the prepared grid is {(grid.nz, grid.ny, grid.nx)} and the "
                f"run is {(cfg.nz, cfg.ny, cfg.nx)}: windows gridded onto the "
                "preparation would be read at other columns")
        self.identity = grid.identity_sha256()
        self.windows = read_windows(self.heating, self.start_time, self.identity)

    def _provenance(self) -> dict:
        return {"built_from": f"{len(self.windows)} heating windows under "
                              f"{self.heating.windows}",
                "background": "the model state at the run start, for every "
                              "slot (NOAA's pre-forecast arrangement)",
                "grid_identity_sha256": self.identity,
                "road": self.road}

    def _ref(self, row):
        from gpuwm.da import radar_tten
        host, _ = radar_tten.read_window_host(row["path"], identity_sha256=self.identity)
        return host

    # -- attach --------------------------------------------------------------
    def attach(self, model, node, steppers=None) -> None:
        setattr(model, ROUTE_ATTRIBUTE, self)
        if not self.plan["attach"]:
            return
        if not self.windows:
            self.validate(node.cfg.run)
        from gpuwm.core import dycore
        stepper = (steppers or {}).get(int(node.cfg.grid_id))
        if stepper is None or stepper is dycore.step:
            if getattr(node.state, "_streamed_domain", None) is not None:
                raise ForecastHeatingRefused(
                    f"{REFUSAL_LABEL}: this domain is integrated out of a host "
                    "store; host-resident tiles carry no forcing, so the "
                    "heating would be skipped while the record says heated")
            self._attach_resident(node)
        elif getattr(stepper, "ranked", False):
            self._attach_ranked(node, stepper)
        else:
            raise ForecastHeatingRefused(
                f"{REFUSAL_LABEL}: this domain is tile-streamed ([tiles]); a "
                "tile buffer serves a different tile every sweep and carries "
                "no forcing, so the heating would be skipped or read at the "
                "wrong offsets.  Run it resident or on [devices] ranks")

    def _attach_resident(self, node) -> None:
        from gpuwm.da import radar_tten

        cp = radar_tten._require_device()
        self.road = "resident"
        cfg = node.cfg.run
        need = resident_bytes(self.heating, cfg)
        free = int(cp.cuda.runtime.memGetInfo()[0]) + int(
            cp.get_default_memory_pool().free_bytes())
        self.build["memory"] = {"priced_bytes": need, "free_bytes": free}
        if need > free:
            raise ForecastHeatingRefused(
                f"{REFUSAL_LABEL}: the forcing needs {need / 2**30:.2f} GiB on "
                f"this card and {free / 2**30:.2f} GiB is free; a CUDA "
                "out-of-memory would stop the run part way through the "
                "forced period")
        started = time.perf_counter()
        config = radar_config(self.heating)
        background = radar_tten.background_from_state(node.state)
        slots, receipts = [], []
        for row in self.windows:
            ref = cp.asarray(np.ascontiguousarray(self._ref(row)))
            slot, receipt = radar_tten.build_tendency(ref, config=config, **background)
            receipt["observations"] = dict(row)
            slots.append(slot)
            receipts.append(receipt)
            del ref
        del background
        forcing = radar_tten.RadarTtenForcing(
            slots, self.heating.window_end_minutes(), receipts=receipts,
            provenance=self._provenance(), mp_tend_lim=self.heating.mp_tend_lim,
            active_minutes=self.heating.active_minutes)
        radar_tten.attach(node.state, forcing, cfg)
        self._attached.append((node.state, forcing, None))
        self.build["seconds"] = round(time.perf_counter() - started, 3)

    def _attach_ranked(self, node, stepper) -> None:
        from gpuwm.da import radar_tten

        cp = radar_tten._require_device()
        run = getattr(stepper, "tiled_run", None)
        if run is None or not getattr(run, "ranked", False):
            raise ForecastHeatingRefused(
                f"{REFUSAL_LABEL}: the multi-card stepper exposes no resident "
                "ranks to attach slabs to")
        self.road = "ranks"
        cfg = node.cfg.run
        ny, nx = int(cfg.ny), int(cfg.nx)
        specs, devices, tiles = list(run.specs), list(run.devices), list(run.tiles)
        for spec in specs:
            if (spec.periodic_x or spec.periodic_y or spec.ci0 < 0 or spec.cj0 < 0
                    or spec.ci0 + spec.cnx > nx or spec.cj0 + spec.cny > ny):
                raise ForecastHeatingRefused(
                    f"{REFUSAL_LABEL}: rank {spec.index} has a wrapped compute "
                    "window; a slab of a periodic axis is not a rectangle of "
                    "the domain and would be heated at the wrong columns")
        sync = getattr(run, "sync_compute", None)
        if callable(sync):
            sync()
        build_device = devices[0]
        started = time.perf_counter()

        def gather(name):
            first = getattr(tiles[0], name, None)
            if first is None:
                return None
            if first.ndim == 1:
                with cp.cuda.Device(first.device.id):
                    host = cp.asnumpy(first)
                with cp.cuda.Device(build_device):
                    return cp.asarray(host)
            host = np.empty((first.shape[0], ny, nx), dtype=first.dtype)
            for tile, spec, dev in zip(tiles, specs, devices):
                array = getattr(tile, name)
                jl, il = spec.j0 - spec.cj0, spec.i0 - spec.ci0
                with cp.cuda.Device(dev):
                    host[:, spec.j0:spec.j1, spec.i0:spec.i1] = cp.asnumpy(
                        array[:, jl:jl + spec.j1 - spec.j0,
                              il:il + spec.i1 - spec.i0])
            with cp.cuda.Device(build_device):
                return cp.asarray(host)

        config = radar_config(self.heating)
        per_rank = [[] for _ in tiles]
        receipts = []
        with cp.cuda.Device(build_device):
            whole = SimpleNamespace(**{name: gather(name) for name in (
                "thb", "thp", "p", "phb", "php", "qv")})
            background = radar_tten.background_from_state(whole)
            del whole
            for row in self.windows:
                ref = cp.asarray(np.ascontiguousarray(self._ref(row)))
                slot, receipt = radar_tten.build_tendency(ref, config=config, **background)
                receipt["observations"] = dict(row)
                receipts.append(receipt)
                host_slot = cp.asnumpy(slot)
                del slot, ref
                for rank, (spec, dev) in enumerate(zip(specs, devices)):
                    piece = np.ascontiguousarray(
                        host_slot[:, spec.cj0:spec.cj0 + spec.cny,
                                  spec.ci0:spec.ci0 + spec.cnx])
                    with cp.cuda.Device(dev):
                        per_rank[rank].append(cp.asarray(piece))
                del host_slot
            del background
        sub_cfgs = list(getattr(run, "sub_cfgs", [cfg] * len(tiles)))
        for rank, (tile, spec, dev) in enumerate(zip(tiles, specs, devices)):
            with cp.cuda.Device(dev):
                forcing = radar_tten.RadarTtenForcing(
                    per_rank[rank], self.heating.window_end_minutes(),
                    receipts=receipts, provenance=self._provenance(),
                    mp_tend_lim=self.heating.mp_tend_lim,
                    active_minutes=self.heating.active_minutes,
                    extent=(spec.cj0, spec.ci0, ny, nx))
                setattr(tile, radar_tten.SLAB_ATTRIBUTE, (spec.cj0, spec.ci0))
                radar_tten.attach(tile, forcing, sub_cfgs[rank])
                # The uploads ran on the default stream and the rank steps
                # on its own non-blocking stream.
                cp.cuda.runtime.deviceSynchronize()
            self._attached.append((tile, forcing, rank))
        self.build["seconds"] = round(time.perf_counter() - started, 3)
        self.build["build_device"] = int(build_device)
        self.build["ranks"] = [{"rank": rank, "device": int(dev),
                                "extent": [spec.cj0, spec.ci0, spec.cny, spec.cnx]}
                               for rank, (spec, dev) in enumerate(zip(specs, devices))]

    # -- detach and record -------------------------------------------------
    def detach(self) -> None:
        from gpuwm.da import radar_tten

        for state, forcing, rank in self._attached:
            self._receipts.append((rank, forcing.receipt()))
            radar_tten.detach(state)
            if radar_tten.SLAB_ATTRIBUTE in vars(state):
                delattr(state, radar_tten.SLAB_ATTRIBUTE)
        self._attached = []

    def _forcings(self):
        return [receipt for _, receipt in self._receipts] or [
            forcing.receipt() for _, forcing, _ in self._attached]

    def applied_calls(self) -> int:
        forcings = self._forcings()
        if not forcings:
            return 0
        first = forcings[0]
        return int(sum(first.get("calls_by_slot", ()))) + int(
            first.get("calls_skipped_no_mp_heating", 0))

    def require_applied(self) -> None:
        """A forced run whose forcing was never read is refused: the record
        would say heated while no step was."""
        if self.plan["attach"] and self.applied_calls() == 0:
            raise ForecastHeatingRefused(
                f"{REFUSAL_LABEL} was attached and no microphysics step read "
                "it: the record would say heated while no step was")

    def receipt(self) -> dict:
        forcings = self._forcings()
        classes = sorted({row["lead_class"] for row in self.windows})
        rank_rows = [
            {"rank": rank,
             "calls_by_slot": receipt.get("calls_by_slot"),
             "calls_after_active": receipt.get("calls_after_active"),
             "extent": receipt.get("extent")}
            for rank, receipt in self._receipts if rank is not None]
        agree = len({json.dumps([r["calls_by_slot"], r["calls_after_active"]])
                     for r in rank_rows}) <= 1
        first = dict(forcings[0]) if forcings else None
        return {
            "schema": SCHEMA,
            "status": self.plan["status"],
            "plan": dict(self.plan),
            "config": self.heating.to_mapping(),
            "start_time": self.start_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "grid_identity_sha256": self.identity,
            "grid_identity_from": "gpuwm.da.forecast_heating.grid_for_prepared"
                                  f"({self.prepared_root})",
            "road": self.road,
            "windows": [dict(row) for row in self.windows],
            "lead_classes": classes,
            "oracle": "oracle" in classes,
            "forcing": first,
            "ranks": rank_rows,
            "ranks_agree": agree,
            "applied_calls": self.applied_calls(),
            "build": dict(self.build),
            "declared": ("every slot is built against the start state, as "
                         "NOAA's pre-forecast does; over a 2 h forced period "
                         "the warm-echo test uses the start temperatures"),
        }


def prevalidate(exp, *, prepared_root):
    """The host-only half of the attach, run by the door before any card
    is allocated: ``None`` when the table is off, else ``(identity,
    windows)`` for :func:`attach_forecast_heating`'s ``checked``.  The
    refusals are :meth:`ForecastHeating.validate`'s."""
    if not enabled(exp):
        return None
    hook = ForecastHeating(exp.radar_heating, start_time=exp.start_time,
                           prepared_root=prepared_root)
    hook.validate(exp.root.run)
    return hook.identity, list(hook.windows)


def attach_forecast_heating(model, node, exp, *, prepared_root, steppers=None,
                            restored_seconds=None, checked=None):
    """The door's one call: ``None`` when the table is off, else the
    attached :class:`ForecastHeating` (detach it in the door's ``finally``).
    ``checked`` is :func:`prevalidate`'s result for this run, so the
    windows are not validated twice; the attach still re-reads each one
    against its digest when it builds the slot."""
    if not enabled(exp):
        return None
    hook = ForecastHeating(exp.radar_heating, start_time=exp.start_time,
                           prepared_root=prepared_root,
                           restored_seconds=restored_seconds)
    if checked is not None:
        hook.identity, windows = checked
        hook.windows = [dict(row) for row in windows]
    hook.attach(model, node, steppers)
    return hook


__all__ = [
    "BUILD_TRANSIENT_VOLUMES", "ForecastHeating", "ForecastHeatingRefused",
    "ROUTE_ATTRIBUTE", "SCHEMA", "add_execution_argument",
    "apply_execution_options", "attach_forecast_heating",
    "detach_table_from_config", "enabled", "preparation_binds",
    "split_experiment_text",
    "execution_argument", "execution_flags", "grid_for_prepared",
    "lead_class", "prepared_cache_directory", "prepared_grid_identity",
    "prevalidate",
    "radar_config", "ranked_reservation", "read_windows", "refuse_tree",
    "require_routed", "resident_bytes", "resume_plan", "slab_bytes_per_card",
    "window_ends",
]
