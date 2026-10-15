from pathlib import Path
from types import SimpleNamespace

import pytest

from gpuwm import go_cli
from gpuwm.ensemble import recipe_door


def test_preparation_admits_every_members_selected_static_fields(monkeypatch):
    selected = []

    def geography(root, *, config=None):
        assert config is not None, 'generic geography ignores selected static source'
        selected.append(config)
        return None

    monkeypatch.setattr(go_cli, 'geography_refusal', geography)
    monkeypatch.setattr(recipe_door, 'disk_refusal', lambda *a, **k: None)
    paths = [Path('member-0.toml'), Path('member-1.toml')]
    recipe_door.admit([SimpleNamespace(config=path) for path in paths],
        config=Path('base.toml'), geog_root=Path('geog'), case_root=Path('case'),
        request=None, options={'prepare_only': True}, command='ensemble')
    assert selected == paths


def test_missing_second_members_geography_refuses_before_preparation(monkeypatch):
    def geography(root, *, config=None):
        return 'missing member-1 lake depth' if config == Path('member-1.toml') else None

    monkeypatch.setattr(go_cli, 'geography_refusal', geography)
    monkeypatch.setattr(recipe_door, 'disk_refusal', lambda *a, **k: None)
    with pytest.raises(recipe_door.RecipeRefusal, match='member-1 lake depth'):
        recipe_door.admit([SimpleNamespace(config=Path(f'member-{i}.toml'))
                          for i in range(2)],
            config=Path('base.toml'), geog_root=Path('geog'), case_root=Path('case'),
            request=None, options={'prepare_only': True}, command='ensemble')
