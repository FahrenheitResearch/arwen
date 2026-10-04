"""Recenter native physical source snapshots before WRF-real initialization."""
from __future__ import annotations

from bisect import bisect_left
from dataclasses import asdict, replace
import numpy as np

from gpuwm.ensemble.native_preparation import NativeEnsemblePreparation
from gpuwm.ensemble.physical_store import NativePhysicalStore, digest_file
from gpuwm.ensemble.physical_fields import field_contract_sha256
from gpuwm.ensemble.recentered import FieldBounds, recenter_field

SCHEMA = "gpuwm-ensemble-recentered-preparation.v1"
# These are physical guard bounds. Amplitudes are experiment inputs, and no
# number here asserts calibrated forecast uncertainty.
PHYSICAL_BOUNDS = {
    "TT": FieldBounds("K", 5.0, 100.0, 400.0),
    "SPFH": FieldBounds("kg kg-1", 0.005, 0.0, 0.1),
    "UU": FieldBounds("m s-1", 15.0, -250.0, 250.0),
    "VV": FieldBounds("m s-1", 15.0, -250.0, 250.0),
    "GHT": FieldBounds("m", 200.0, -1000.0, 100000.0),
    "PSFC": FieldBounds("Pa", 2000.0, 10000.0, 120000.0),
    "T2": FieldBounds("K", 5.0, 100.0, 400.0),
    "Q2": FieldBounds("kg kg-1", 0.005, 0.0, 0.1),
    "U10": FieldBounds("m s-1", 15.0, -250.0, 250.0),
    "V10": FieldBounds("m s-1", 15.0, -250.0, 250.0),
}
_SURFACE = {"TT": "T2", "SPFH": "Q2", "UU": "U10", "VV": "V10", "GHT": "SOURCE_OROGRAPHY"}
RELATIVE_HUMIDITY_BOUNDS = {
    "RH": FieldBounds("%", 20.0, 0.0, 100.0),
    "RH2": FieldBounds("%", 20.0, 0.0, 100.0),
}


class RecenteredPhysicalPreparation:
    """One fixed donor population, native horizontal/vertical/time alignment.

    Input stores contain actual native-mapped fields, their geometry and
    verified source manifests. The base pressure levels remain coordinates;
    the normal real initializer reconstructs mass, hydrostatics and C-grid
    prognostics from the recentered temperature, humidity, wind and height.
    Soil and surface properties stay with the base trajectory.
    """
    def __init__(self, base_store, donor_stores, *, amplitude, cpu_bridge=None, workers=1,
                 bounds=None):
        self.base = NativePhysicalStore(base_store)
        self.donors = {key: NativePhysicalStore(value) for key, value in sorted(donor_stores.items())}
        if len(self.donors) < 2 or any(not isinstance(key, str) or not key for key in self.donors):
            raise ValueError("recentered preparation needs a fixed named donor population")
        if any(store.document["grid"] != self.base.document["grid"] for store in self.donors.values()):
            raise ValueError("donor and base native target geometries differ")
        for store in (self.base, *self.donors.values()):
            self._verify_field_contract(store)
        first = self.base.read(0)
        self.specific = {"SPFH", "Q2"} <= first.fields.keys()
        default_bounds = dict(PHYSICAL_BOUNDS)
        if not self.specific:
            if not {"RH", "RH2"} <= first.fields.keys():
                raise ValueError("native base needs its complete specific- or relative-humidity field pair")
            del default_bounds["SPFH"], default_bounds["Q2"]
            default_bounds.update(RELATIVE_HUMIDITY_BOUNDS)
        self.bounds = {key: replace(value, amplitude=amplitude)
                       for key, value in (default_bounds if bounds is None else bounds).items()}
        if set(self.bounds) != set(default_bounds):
            raise ValueError("recentered preparation must declare the complete physical field set")
        if any(value.units != default_bounds[key].units for key,value in self.bounds.items()):
            raise ValueError("recentered bounds units differ from the source-qualified physical fields")
        self.surface = {key:value for key,value in _SURFACE.items() if key != "SPFH"}
        self.surface["SPFH" if self.specific else "RH"] = "Q2" if self.specific else "RH2"
        self.native = NativeEnsemblePreparation(cpu_bridge)
        from gpuwm.ingest.preprocess_backend import ParallelCpuPreprocessBackend
        self.backend = ParallelCpuPreprocessBackend(bridge=cpu_bridge, workers=workers)
        self.workers = workers
        self._frames = {}
        self.identity = {
            "schema": SCHEMA, "stage": "physical fields before native real initialization",
            "grid_sha256": self.base.grid_sha256,
            "base": {"sha256": digest_file(self.base.manifest_path), "source": self.base.document["source"]},
            "donors": {key: {"sha256": digest_file(value.manifest_path), "source": value.document["source"]}
                       for key, value in self.donors.items()},
            "base_field_contract_sha256": field_contract_sha256(self.base.field_contract),
            "donor_field_contract_sha256": {key:field_contract_sha256(value.field_contract)
                                            for key,value in self.donors.items()},
            "bounds": {key: asdict(value) for key, value in self.bounds.items()},
            "donor_mean": "all donors in canonical ID order at every initial and boundary time",
            "vertical": "native WRF log-pressure interpolation onto the base physical pressure coordinate",
            "time": "native binary64 linear interpolation after vertical alignment; one final float32 rounding",
            "humidity": ("specific humidity kg kg-1" if self.specific else "relative humidity percent")
                + "; preserve the base field set and native initializer route; native Bolton conversion of donors before vertical alignment when needed",
            "unchanged": "base pressure coordinates, soil and land/water surface fields",
            "native_bridge_sha256": digest_file(self.native.path),
            "calibration": "amplitude supplied by experiment; not a skill claim"}

    @staticmethod
    def _verify_field_contract(store):
        contract = store.require_field_contract()
        expected = {key:value.units for key,value in {**PHYSICAL_BOUNDS, **RELATIVE_HUMIDITY_BOUNDS}.items()}
        expected.update(PRES="Pa", SOURCE_OROGRAPHY="m")
        inventory = store.document["frames"][0]["arrays"]
        for name,units in expected.items():
            key = "field__"+name
            if key not in contract["arrays"]:continue
            specification=contract["arrays"].get(key,{})
            basis = "grid_x" if name in ("UU","U10") else "grid_y" if name in ("VV","V10") else "scalar"
            dimensions = (["level"] if name in ("TT","SPFH","RH","UU","VV","GHT","PRES") else [])
            dimensions += ["y_stag" if basis=="grid_y" else "y", "x_stag" if basis=="grid_x" else "x"]
            if (specification.get("units") != units or specification.get("basis") != basis
                    or specification.get("dimensions") != dimensions):
                raise ValueError(f"native recentering requires source-qualified {name} units {units}, basis {basis} and dimensions {dimensions}")
        if "field__PRES" in inventory:
            if contract["vertical"]["pressure_field"] != "field__PRES":
                raise ValueError("native recentering pressure field differs from its coordinate authority")
        elif contract["vertical"]["kind"] != "pressure_levels":
            raise ValueError("hybrid native recentering requires its actual Pa pressure field")

    def _frame(self, key, index):
        token = (key, index)
        if token not in self._frames:
            # At most the two time brackets of one donor remain resident.
            self._frames = {old: value for old, value in self._frames.items() if old[0] == key}
            self._frames[token] = self.donors[key].read(index)
        return self._frames[token]

    def _pressure(self, frame):
        if "PRES" in frame.fields:
            return np.ascontiguousarray(frame.fields["PRES"], dtype=np.float32)
        return self.native.pressure_levels(frame.levels_hpa, frame.fields["PSFC"].shape)

    def _align(self, frame, target_pressure):
        fields = frame.fields
        required = (set(PHYSICAL_BOUNDS) - {"SPFH", "Q2"}) | {"SOURCE_OROGRAPHY"}
        if not required <= set(fields):
            raise ValueError(f"native donor lacks physical fields {sorted(required-set(fields))}")
        pressure = self._pressure(frame)
        surface = np.ascontiguousarray(fields["PSFC"], dtype=np.float32)
        donor_specific = {"SPFH", "Q2"} <= fields.keys()
        if not donor_specific and not {"RH", "RH2"} <= fields.keys():
            raise ValueError("native donor lacks its complete physical humidity field pair")
        if donor_specific != self.specific:
            converted = dict(fields)
            for target, original, temperature, coordinate in (
                    ("SPFH" if self.specific else "RH", "SPFH" if donor_specific else "RH", "TT", pressure),
                    ("Q2" if self.specific else "RH2", "Q2" if donor_specific else "RH2", "T2", surface)):
                converted[target] = self.native.humidity(fields[temperature], coordinate,
                                                          fields[original], to_relative=not self.specific)
            fields = converted
        # Hybrid snapshots can carry model indices in the legacy levels_hpa
        # slot. Physical PRES, not those indices, determines vertical order.
        reverse = bool(pressure[0,0,0] < pressure[-1,0,0])
        if reverse:
            pressure = np.ascontiguousarray(pressure[::-1])
        plans = {"mass": self.backend.prepare_wrf_vertical(pressure, surface, target_pressure)}
        for key, axis in (("UU", 2), ("VV", 1)):
            plans[key] = self.backend.prepare_wrf_vertical(
                self.native.pressure_stagger(pressure, axis),
                self.native.pressure_stagger(surface[None], axis)[0],
                self.native.pressure_stagger(target_pressure, axis))
        result = {key: fields[key] for key in self.bounds if key not in self.surface}
        for key, surface_name in self.surface.items():
            value = np.ascontiguousarray(fields[key][::-1] if reverse else fields[key], dtype=np.float32)
            result[key] = plans.get(key, plans["mass"]).apply(
                value, np.ascontiguousarray(fields[surface_name], dtype=np.float32),
                interp_in_logp=True, extrap="temperature" if key == "TT" else "constant")
        return result

    def _donor_at(self, key, valid_time, target_pressure):
        times = self.donors[key].times
        position = bisect_left(times, valid_time)
        if position < len(times) and times[position] == valid_time:
            return self._align(self._frame(key, position), target_pressure)
        if position == 0 or position == len(times):
            raise ValueError(f"donor {key} does not bracket initial/boundary valid time {valid_time}")
        left = self._align(self._frame(key, position-1), target_pressure)
        right = self._align(self._frame(key, position), target_pressure)
        weight = (valid_time-times[position-1]).total_seconds() / (times[position]-times[position-1]).total_seconds()
        return {field: self.native.time_blend(
            np.ascontiguousarray(left[field], dtype=np.float32),
            np.ascontiguousarray(right[field], dtype=np.float32), weight) for field in self.bounds}

    def prepare(self, outputs):
        """Write selected members with fixed donor IDs and a common mean."""
        amplitude = next(iter(self.bounds.values())).amplitude
        return self.prepare_variants({"default": {"amplitude":amplitude, "outputs":outputs}})["default"]

    def prepare_variants(self, variants):
        """Reuse native donor alignment across declared amplitude candidates."""
        if not variants:
            raise ValueError("an amplitude study needs at least one output variant")
        studies = {}
        for name, specification in variants.items():
            outputs = specification["outputs"]
            selected = tuple(outputs)
            if not selected or len(set(selected)) != len(selected) or not set(selected) <= self.donors.keys():
                raise ValueError("output members must select unique IDs from the frozen donor population")
            bounds = {key: replace(value, amplitude=specification["amplitude"]) for key,value in self.bounds.items()}
            identity = {**self.identity, "bounds": {key: asdict(value) for key,value in bounds.items()}}
            writers = {member: NativePhysicalStore(root, grid_identity=self.base.document["grid"],
                       source_identity={**identity, "selected_member": member, "frame_operations":[]},
                       field_contract=self.base.field_contract)
                       for member,root in outputs.items()}
            studies[name] = (selected, bounds, writers)
        for index, valid_time in enumerate(self.base.times):
            base = self.base.read(index)
            # Presence of physical SPFH/PRES/Q2 is the field contract. The
            # separate specific_humidity_authority flag selects the native
            # initializer's direct-q versus WRF FLAG_SH/RH route and is
            # preserved exactly, including False on native hybrid inputs.
            required = set(self.bounds) | {"SOURCE_OROGRAPHY"}
            if not required <= set(base.fields):
                raise ValueError(f"native base lacks physical fields {sorted(required-set(base.fields))}")
            target_pressure = self._pressure(base)
            donor_fields = {key: self._donor_at(key, valid_time, target_pressure) for key in self.donors}
            for selected, bounds, writers in studies.values():
                self._write_frame(base, donor_fields, selected, bounds, writers)
            print(f"prepared physical member fields at {valid_time.isoformat()} for "
                  f"{sum(len(study[0]) for study in studies.values())} member variants", flush=True)
        return {name:{member:writer.seal() for member,writer in study[2].items()} for name,study in studies.items()}

    def _write_frame(self, base, donor_fields, selected, field_bounds, writers):
        member_fields = {member: dict(base.fields) for member in selected}
        operations = {}
        for field, bounds in field_bounds.items():
            if field == "SPFH":
                floor = base.specific_humidity_undershoot_floor
                if floor is None:
                    from gpuwm.ingest.real import _specific_humidity_undershoot_bound
                    # Native snapshots can retain horizontal undershoots
                    # without an explicit envelope. Admit only
                    # this verified base's actual range. The native kernel
                    # preserves each sub-floor word and creates none.
                    floor = float(np.min(base.fields[field]))
                    native_floor, _ = _specific_humidity_undershoot_bound(True, None)
                    if floor < native_floor:
                        raise ValueError("base humidity falls below the native initializer's undershoot envelope")
                if floor is not None:
                    bounds = replace(bounds, input_lower=min(bounds.lower, floor))
            elif field in ("RH", "RH2"):
                # The native RH initializer clips these source fields.
                # Preserve already out-of-range base words, without
                # letting an in-range member cross the physical range.
                bounds = replace(bounds, input_lower=min(bounds.lower, float(np.min(base.fields[field]))),
                                 input_upper=max(bounds.upper, float(np.max(base.fields[field]))))
            values, _receipt = recenter_field(
                np.ascontiguousarray(base.fields[field], dtype=np.float32),
                np.stack([donor_fields[key][field] for key in self.donors]),
                donor_ids=tuple(self.donors), selected_ids=selected, bounds=bounds,
                mapped_grid_sha256=self.base.grid_sha256, array_module=np,
                cpu_bridge=self.native.path, workers=self.workers)
            operations[field] = {key: value for key, value in _receipt.items()
                                 if key != "selected_members"}
            for member, values_for_member in zip(selected, values):
                member_fields[member][field] = values_for_member
        for member, writer in writers.items():
            writer.write(replace(base, fields=member_fields[member]))
            writer.document["source"]["frame_operations"].append(
                {"valid_time": base.valid_time.isoformat(), "fields": operations})
