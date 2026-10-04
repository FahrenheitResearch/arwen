"""Optional native surface member store for observation calibration."""
from __future__ import annotations

from datetime import datetime, timedelta
import hashlib
import json
from operator import index
from pathlib import Path
import threading

import numpy as np

FIELDS = {"T2": "K", "U10": "m s-1", "V10": "m s-1", "RAIN_TOTAL": "mm"}


class MemberDiagnosticArchive:
    """Write original member words without requiring full prognostic history."""
    CONTRACT = "gpuwm-ensemble-member-diagnostics.v1"

    def __init__(self, root, *, member_order, member_metadata, start_time):
        self.root = Path(root)
        self.order = tuple(index(member) for member in member_order)
        if not self.order or len(set(self.order)) != len(self.order):
            raise ValueError("calibration output needs the exact distinct global member order")
        self.metadata = {row["member_id"]: dict(row) for row in member_metadata}
        if set(self.metadata) != set(self.order):
            raise ValueError("calibration output requires a seed and provenance for every member")
        for row in self.metadata.values():
            if type(row.get("seed")) is not int or not 0 <= row["seed"] < (1 << 64):
                raise ValueError("calibration output member seeds must retain exact uint64 identity")
        self.start_time = start_time
        self.entries, self.unavailable = {}, []
        self.forecasts, self.expected = {}, {}
        self.finished = False
        self._lock = threading.RLock()
        self.directory = self.root / "member-diagnostics"
        self.directory.mkdir(parents=True, exist_ok=False)
        self.path = self.directory / "manifest.json"
        self._manifest()

    def _manifest(self):
        supplied = {(row["member_id"], row["grid_id"], row["valid_time"])
                    for row in self.entries.values()}
        missing = [row for key, row in self.expected.items() if key not in supplied]
        dynamic = [row for row in self.forecasts.values() if row["dynamic_lifecycle"]]
        record = {"schema": self.CONTRACT, "member_order": list(self.order),
            "member_metadata": [self.metadata[member] for member in self.order],
            "fields": FIELDS, "precipitation_accumulation_start": self.start_time.isoformat(),
            "forecast_volume_fields": False, "files": list(self.entries.values()),
            "unavailable": list(self.unavailable),
            "coverage": {"status": ("running" if not self.finished else
                "not_planned" if not self.forecasts else "incomplete" if missing else
                "dynamic_lifecycle" if dynamic else "complete"),
                "policy": "native initial and exact forecast-hour planes; original clocks and history selection unchanged",
                "member_forecasts": list(self.forecasts.values()),
                "expected_frames": len(self.expected), "missing_frames": missing,
                "dynamic_domains": dynamic}}
        temporary = self.path.with_suffix(".json.pending")
        temporary.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(self.path)
        return record

    def expect_member_forecast(self, member_id, experiment):
        """Record requested thin coverage without changing the output calendar.

        Dormant or rearmed children keep an explicit dynamic coverage contract;
        their realized episode windows cannot be inferred from a declaration.
        """
        member = index(member_id)
        if member not in self.metadata:
            raise ValueError("calibration coverage received a member outside the source roster")
        if experiment.start_time != self.start_time:
            raise ValueError("calibration member forecast has a different accumulation start")
        end_us = round(float(experiment.run_seconds) * 1_000_000)
        with self._lock:
            for domain in experiment.domains:
                grid = index(domain.grid_id)
                start = experiment.domain_start_time(grid)
                start_us = round((start - self.start_time).total_seconds() * 1_000_000)
                dynamic = any(getattr(domain, name, None) is not None
                              for name in ("spawn", "retire", "rearm"))
                row = {"member_id": member, "grid_id": grid,
                    "start_time": start.isoformat(),
                    "end_time": (self.start_time + timedelta(microseconds=end_us)).isoformat(),
                    "history_interval_seconds": float(domain.history_interval_s),
                    "history_begin_seconds": float(domain.history_begin_s),
                    "history_end_seconds": domain.history_end_s,
                    "dynamic_lifecycle": dynamic}
                key = (member, grid)
                if key in self.forecasts and self.forecasts[key] != row:
                    raise ValueError("calibration member coverage changed its original forecast calendar")
                self.forecasts[key] = row
                if dynamic:
                    continue
                ticks = {start_us} if 0 <= start_us <= end_us else set()
                first_hour = max(0, (start_us + 3_600_000_000 - 1) // 3_600_000_000)
                ticks.update(range(first_hour * 3_600_000_000, end_us + 1, 3_600_000_000))
                for tick in sorted(ticks):
                    valid = (self.start_time + timedelta(microseconds=tick)).isoformat()
                    self.expected[(member, grid, valid)] = {"member_id": member,
                        "grid_id": grid, "valid_time": valid,
                        "reason": "no complete native diagnostic plane at this requested time"}
            self._manifest()

    def finish(self):
        """Expose missing members, times and fields instead of inventing them."""
        with self._lock:
            self.finished = True
            return self._manifest()

    def submit(self, fields, *, member_id, valid_time, grid_id, episode, latitude, longitude,
               array_module=None):
        """Capture only initial and exact hourly diagnostic planes."""
        elapsed_us = round((valid_time - self.start_time).total_seconds() * 1_000_000)
        member = index(member_id)
        if member not in self.metadata:
            raise ValueError("calibration output received a member outside the source roster")
        target = (member, index(grid_id), valid_time.isoformat()) in self.expected
        if elapsed_us < 0 or (elapsed_us % 3_600_000_000 and not target):
            return False
        coords = tuple(np.ascontiguousarray(value, dtype=np.float32) for value in (latitude, longitude))
        if coords[0].ndim != 2 or coords[0].shape != coords[1].shape:
            raise ValueError("calibration output requires exact two-dimensional member coordinates")
        geometry = hashlib.sha256(coords[0].tobytes() + coords[1].tobytes()).hexdigest()
        key = (index(grid_id), index(episode), elapsed_us, member, geometry)
        with self._lock:
            if key in self.entries:
                raise ValueError("calibration member/grid/time output was submitted twice")
            missing = sorted(set(FIELDS) - set(fields))
            if missing:
                self.unavailable.append({"member_id": member, "grid_id": grid_id, "episode": episode,
                    "valid_time": valid_time.isoformat(), "geometry_sha256": geometry, "missing_fields": missing})
                self._manifest()
                return False
            arrays = {}
            for name in FIELDS:
                value = fields[name]
                if array_module is not None:
                    value = array_module.asnumpy(value)
                value = np.asarray(value)
                if value.dtype != np.dtype("float32") or value.shape != coords[0].shape:
                    raise ValueError("calibration fields require the original float32 diagnostic words and grid")
                arrays[name] = value
            from gpuwm.io.classic_product import ClassicProduct
            token = valid_time.strftime("%Y-%m-%d_%H-%M-%S")
            path = self.directory / f"d{grid_id:02d}-episode-{episode:03d}-{geometry[:12]}-{token}-member-{member:04d}.nc"
            if path.exists():
                raise FileExistsError("calibration output path already belongs to an existing file")
            pending = path.with_suffix(".nc.pending")
            with ClassicProduct(pending) as writer:
                writer.createDimension("member", 1)
                writer.createDimension("south_north", coords[0].shape[0])
                writer.createDimension("west_east", coords[0].shape[1])
                writer.setncattr("member_diagnostic_contract", self.CONTRACT)
                writer.setncattr("valid_time", valid_time.isoformat())
                writer.setncattr("precipitation_accumulation_start", self.start_time.isoformat())
                writer.setncattr("geometry_sha256", geometry)
                writer.setncattr("member_provenance", json.dumps(self.metadata[member], sort_keys=True))
                for name, value in (("member_id", member), ("member_seed", self.metadata[member]["seed"])):
                    writer.createVariable(name, "uint64", ("member",))[:] = np.asarray([value], np.uint64)
                for name, value in zip(("XLAT", "XLONG"), coords):
                    variable = writer.createVariable(name, "float32", ("south_north", "west_east"))
                    variable.units = "degrees_north" if name == "XLAT" else "degrees_east"
                    variable[:] = value
                for name, value in arrays.items():
                    variable = writer.createVariable(name, "float32", ("member", "south_north", "west_east"))
                    variable.units = FIELDS[name]
                    variable[:] = value[None]
            pending.replace(path)
            self.entries[key] = {"path": path.relative_to(self.root).as_posix(), "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "member_id": member,
                "seed": self.metadata[member]["seed"], "grid_id": grid_id, "episode": episode,
                "valid_time": valid_time.isoformat(), "geometry_sha256": geometry}
            self._manifest()
            return True

    def receipt(self):
        with self._lock:
            return self._manifest()
