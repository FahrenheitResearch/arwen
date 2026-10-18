"""Unsupported scalar mixing must name the missing requested tendency."""
from types import SimpleNamespace

import pytest

from gpuwm.config import UnsupportedConfiguration, validate_scalar_pblmix_consumer


@pytest.mark.parametrize("pbl,mp", [(0, 28), (2, 28), (11, 28), (900, 28), (1, 16), (5, 6)])
def test_scalar_mixing_refusal_names_the_unproduced_tendencies(pbl, mp):
    cfg = SimpleNamespace(scalar_pblmix=1, bl_pbl_physics=pbl, mp_physics=mp)
    with pytest.raises(UnsupportedConfiguration, match="would omit the requested vertical-mixing tendencies"):
        validate_scalar_pblmix_consumer(cfg)


@pytest.mark.parametrize("pbl,mp,mixing", [(1, 28, 1), (5, 28, 1), (0, 28, 0), (2, 6, 0)])
def test_supported_or_disabled_mixing_keeps_admission(pbl, mp, mixing):
    validate_scalar_pblmix_consumer(SimpleNamespace(
        scalar_pblmix=mixing, bl_pbl_physics=pbl, mp_physics=mp))
