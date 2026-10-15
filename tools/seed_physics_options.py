"""Materialize complete option declarations before consumer regeneration.

Existing implemented rows retain their current authority. Missing and
unimplemented rows can be seeded from the tracked option declaration table.
"""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEEDS = ROOT / 'gpuwm' / 'physics_option_seeds_v1.json'
REGISTRY = ROOT / 'gpuwm' / 'physics_registry_v2.json'

def load_seeds(path: Path = SEEDS) -> dict:
    if not path.exists():
        return {'schema': 'gpuwm-physics-option-seeds-v1', 'components': {}}
    data = json.loads(path.read_text(encoding='utf-8'))
    if data.get('schema') != 'gpuwm-physics-option-seeds-v1':
        raise ValueError('unsupported option seed schema would misread the component declarations')
    return data

def seed_options(registry: dict, seeds: dict) -> dict:
    result = copy.deepcopy(registry)
    for component, options in seeds['components'].items():
        target = result['components'][component]['options']
        for option_id, declaration in options.items():
            if declaration.get('implemented') is not True or not declaration.get('selectors'):
                raise ValueError(f'{component}.{option_id}: an incomplete option seed cannot supply an implemented selector')
            if option_id in target and target[option_id].get('implemented') is True:
                continue
            target[option_id] = copy.deepcopy(declaration)
    return result

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registry', type=Path, default=REGISTRY)
    parser.add_argument('--seeds', type=Path, default=SEEDS)
    parser.add_argument('--out', type=Path, default=REGISTRY)
    args = parser.parse_args()
    result = seed_options(json.loads(args.registry.read_text(encoding='utf-8')), load_seeds(args.seeds))
    args.out.write_text(json.dumps(result, sort_keys=True, separators=(',', ':'))+'\n', encoding='utf-8')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
