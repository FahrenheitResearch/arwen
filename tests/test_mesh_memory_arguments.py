"""Mesh sizing never drops a stated memory budget without a word.

With --cells and no --card, every --vram-gib was ignored and the run
went ahead, although the option's own help says it needs --card.  With a
card, infinity left as an OverflowError traceback and NaN as "cannot
convert float NaN to integer".
"""

from __future__ import annotations

import pytest

from gpuwm.cli import build_parser
from gpuwm.mpas_mesh import MeshRequestError, _resolve_cells, load_sizing


def _args(*extra):
    return build_parser().parse_args(["mesh", "--background-km", "120", *extra])


def test_a_budget_beside_explicit_cells_still_needs_its_card():
    with pytest.raises(MeshRequestError, match="--card"):
        _resolve_cells(_args("--cells", "100", "--vram-gib", "12"), load_sizing())


def test_explicit_cells_and_a_budget_on_a_named_card_keep_working():
    sizing = load_sizing()
    assert _resolve_cells(_args("--cells", "100", "--card", "rtx-5070-ti",
                                "--vram-gib", "12"), sizing)[0] == 100
    assert _resolve_cells(_args("--cells", "100"), sizing)[0] == 100


def test_an_infinite_background_spacing_is_refused_by_name():
    from gpuwm.mpas_mesh import build_spec

    with pytest.raises(MeshRequestError, match="--background-km is inf"):
        build_spec(background_km=float("inf"))
