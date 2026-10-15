"""Small registry views. No forecast engine is imported for name lookup."""
from __future__ import annotations

import json
import math
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data" / "nwpsuite"
ROOT = Path(__file__).resolve().parents[2]


class SuiteRefusal(ValueError):
    """A file cannot describe the run it requests."""


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


class Registries:
    def __init__(self, directory=None):
        self.directory = Path(directory or DATA)
        self.regions = read_json(self.directory / "regions.v1.json")["regions"]
        self.shapes = read_json(self.directory / "shapes.v1.json")["shapes"]
        self.catalog = read_json(self.directory / "references.v1.json")["registries"]

    def require(self, registry, name, context):
        if registry == 'recipes':
            target = ROOT / 'configs' / 'recipes' / (name + '.toml')
            if target.parent == ROOT / 'configs' / 'recipes' and target.is_file():
                return
        if registry == "sources":
            from gpuwm.source_adapters import get_source_adapter
            try:
                get_source_adapter(name)
                return
            except ValueError:
                pass
        elif registry == "surface":
            rows = read_json(ROOT / "gpuwm" / "obs" / "data" / "surface_networks.json")
            if name in {row["id"] for row in rows["networks"]}:
                return
        elif registry == "presets":
            rows = read_json(ROOT / "gpuwm" / "data" / "model_presets" / "registry.v1.json")
            if name in {row["id"] for row in rows["presets"]}:
                return
        elif name in self.catalog.get(registry, {}):
            return
        raise SuiteRefusal(f"{context}: unknown {registry} reference {name!r}; prevents a named input or policy being silently dropped")

    def region(self, name):
        if name not in self.regions:
            raise SuiteRefusal(f"unknown region {name!r}; prevents planning an unintended geography")
        row = self.regions[name]
        self.require("polygons", row["polygon"], name)
        self.require("timezones", row["time_zone"], name)
        for role, values in row["sources"].items():
            for value in values:
                self.require("obs" if role == "obs_da" else "sources", value, name)
        for role, values in row["verify"].items():
            for value in values:
                self.require(role, value, name)
        for value in row["radars"]:
            self.require("radars", value, name)
        self.require("meshes", row["mesh"], name)
        for profile in row['profiles'].values():
            target=self.directory/'templates'/(profile+'.toml')
            if target.parent!=self.directory/'templates' or not target.is_file():
                raise SuiteRefusal(f"{name}: unknown template reference {profile!r}; prevents inheriting an unintended suite")
        return row

    def rates(self, path=None, base=None):
        if path and path != "gpuwm/data/nwpsuite/rates.v1.json":
            target = Path(path)
            if not target.is_absolute():
                target = Path(base or ".") / target
        else:
            target = self.directory / "rates.v1.json"
        rows = read_json(target)["rates"]
        required = {"id", "shape", "grid", "card_model", "cards", "build", "stepping_s_per_fh", "all_in_s_per_fh", "startup_s", "peak_mib_per_card", "kind", "source"}
        for row in rows:
            if not required <= row.keys() or row["kind"] not in ("M", "P"):
                raise SuiteRefusal(f"rate {row.get('id')}: missing provenance or measurement kind; prevents projected work being labelled measured")
            for key in ("id", "shape", "grid", "card_model", "build", "source"):
                if not isinstance(row[key], str) or not row[key].strip():
                    raise SuiteRefusal(f"rate {row.get('id')}.{key}: missing name or provenance; prevents an untraceable reservation")
            if isinstance(row['cards'], bool) or not isinstance(row['cards'], int) or row['cards'] < 0 or (row.get('role', 'forecast') == 'forecast' and row['cards'] == 0):
                raise SuiteRefusal(f"rate {row['id']}.cards: invalid card count; prevents a forecast reserving no cards")
            for key in ("stepping_s_per_fh", "all_in_s_per_fh", "startup_s", "peak_mib_per_card", "reservation_floor_s", "host_memory_mib", "capacity_bound_mib"):
                value = row.get(key)
                nullable = key in ("stepping_s_per_fh", "peak_mib_per_card") or key not in row
                if value is None and nullable:
                    continue
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                    raise SuiteRefusal(f"rate {row['id']}.{key}: invalid nonnegative number; prevents missing or invalid timing and memory being admitted as zero")
            if row.get('role', 'forecast') == 'forecast' and row['all_in_s_per_fh'] <= 0:
                raise SuiteRefusal(f"rate {row['id']}.all_in_s_per_fh: forecast rate must be positive; prevents a zero-length forecast reservation")
        ids = [row["id"] for row in rows]
        if len(ids) != len(set(ids)):
            raise SuiteRefusal("duplicate rate id; prevents ambiguous timing provenance")
        return rows
