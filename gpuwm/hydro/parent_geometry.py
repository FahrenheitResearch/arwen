"""Forecast-scoped projected-grid identity shared by prep and resident binding."""
from dataclasses import dataclass
import hashlib
import json


@dataclass(frozen=True)
class ParentGeometry:
    grid: object
    restart_identity: dict


def bind(grid):
    definition = grid.definition()
    digest = hashlib.sha256(json.dumps(definition, sort_keys=True,
                                      separators=(',', ':')).encode()).hexdigest()
    return ParentGeometry(grid, {
        'schema': 'arwen.nextgen.parent-geometry.v1',
        'definition': definition, 'definition_sha256': digest,
        'row_order': 'south_to_north',
        'authority': 'executed ProjectedGrid definition; forecast preparation projection',
    })
