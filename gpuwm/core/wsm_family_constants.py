"""WSM-family metadata used by preparation and runtime vertical gates."""
from functools import lru_cache
import json
from pathlib import Path

@lru_cache(maxsize=1)
def _rows():
    path=Path(__file__).resolve().parents[1]/"data/wsm_family/schemes.json"
    return json.loads(path.read_text(encoding="utf-8"))["schemes"]

def vertical_level_bounds(mp_physics):
    row=_rows()[str(int(mp_physics))]
    return int(row["min_levels"]),max(map(int,row["level_tiers"]))

def selectors():
    """Execution selectors declared by the family table."""
    return frozenset(map(int, _rows()))


def precipitation_classes(mp_physics):
    return tuple(map(str, _rows()[str(int(mp_physics))]["precip"]))


def scheme_metadata(mp_physics):
    """An independent metadata row without importing a CUDA executor."""
    from copy import deepcopy
    return deepcopy(_rows()[str(int(mp_physics))])


def source_absent_initialization(mp_physics):
    """WRF species names and allocation values declared by a family row."""
    initial = scheme_metadata(mp_physics)["source_absent_initialization"]
    return tuple(initial["wrf_fields"]), dict(initial["state_defaults"])
