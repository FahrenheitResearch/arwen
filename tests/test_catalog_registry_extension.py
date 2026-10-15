"""New registry rows provide catalog wording without a second option list."""
from copy import deepcopy

import pytest

from gpuwm import physics_catalog as catalog


@pytest.fixture
def registry_extension(monkeypatch):
    registry = deepcopy(catalog._registry())
    registry["components"]["microphysics"]["options"]["future-option"] = {
        "label": "Future declared microphysics", "implemented": False,
        "selectors": {"mp_physics": 707}}
    monkeypatch.setattr(catalog, "_registry", lambda: registry)
    catalog.table.cache_clear()
    yield registry
    catalog.table.cache_clear()


def test_registry_only_option_has_factual_catalog_words(registry_extension):
    words = catalog.table()["descriptions"]["microphysics"]["future-option"]
    assert "Future declared microphysics" in words


def test_missing_registry_label_is_not_a_manufactured_description(registry_extension):
    registry_extension["components"]["microphysics"]["options"]["future-option"].pop("label")
    with pytest.raises(ValueError, match="description or label"):
        catalog.table()
