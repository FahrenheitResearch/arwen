"""Real parser tests for source table controls and crown bounds."""
import numpy as np
import pytest

from gpuwm.core.urban_tables import (
    UrbanTableError, load_urban_params, urban_table_path,
)
from gpuwm.core.urban_ucm import UrbanCanopyError, pack_params


def table(tmp_path, *, tree=1, scheme=0, radius=1, height=3, omit=None):
    # Custom URBPARM bytes exercise the ordinary table reader. They have an
    # explicit unpinned identity and are never called a release fixture.
    text = urban_table_path(0).read_text()
    rows = {
        'FVG': .2, 'RTREE': radius, 'HTREE': height, 'DTREE': 4,
        'LAI_TREE': 2, 'LAI_VEG': 2, 'Z0VG': .01, 'Z0HVG': .001,
        'CAPVG': 1.3e6, 'EPSVG': .93, 'EPST': .9, 'ALBVG': .2, 'ALBT': .15,
    }
    for name, value in rows.items():
        if name != omit:
            text += f'\n{name}: {value}, {value}, {value}\n'
    text += f'\nTREEOPTION: {tree}\nTTscheme: {scheme}\n'
    (tmp_path / 'URBPARM.TBL').write_text(text)
    return load_urban_params(1, tbl_dir=tmp_path, verify_sha256=False,
                             generation='wrf_480')


@pytest.mark.parametrize('tree', (0, 1))
@pytest.mark.parametrize('scheme', (0, 1))
def test_both_source_table_selectors(tmp_path, tree, scheme):
    params = table(tmp_path, tree=tree, scheme=scheme)
    assert params.scalar('TREEOPTION') == tree
    assert params.scalar('TTscheme') == scheme
    assert params.generation == 'wrf_480'
    assert np.all(params.RTREE_M_TBL == np.float32(1))
    assert not params.RTREE_TBL.flags.writeable


def test_radius_bound_matches_source(tmp_path):
    with pytest.raises(UrbanTableError, match=r'2\*rt must be <= W'):
        table(tmp_path, radius=100)


def test_height_bound_matches_source(tmp_path):
    with pytest.raises(UrbanTableError, match=r'ht \+ rt <= H'):
        table(tmp_path, height=100)


def test_inactive_tree_geometry_is_not_used(tmp_path):
    assert table(tmp_path, tree=0, height=100).scalar('TREEOPTION') == 0


def test_crown_top_equality_is_allowed(tmp_path):
    # The first source row has ZR=5. Equality ht+rt==H is admissible.
    params = table(tmp_path, height=4, radius=1)
    assert np.float32(params.HTREE_TBL[0] + params.RTREE_TBL[0]) == params.HGT_TBL[0]


@pytest.mark.parametrize('tree,scheme', ((2, 0), (0, 2), (-1, 0), (0, -1)))
def test_unknown_temperature_and_canopy_arms_are_refused(tmp_path, tree, scheme):
    with pytest.raises(UrbanTableError, match='must be 0 or 1'):
        table(tmp_path, tree=tree, scheme=scheme)


def test_missing_new_row_is_not_uninitialized_state(tmp_path):
    with pytest.raises(UrbanTableError, match='lacks.*LAI_TREE_TBL'):
        table(tmp_path, omit='LAI_TREE')


def test_table_generation_cannot_be_silently_packed_as_old_ucm(tmp_path):
    params = table(tmp_path)
    for packed in (pack_params(params), pack_params(params.values)):
        assert packed.generation == 'wrf_480'
        assert packed.table.shape[1] == 49
        assert packed.switches.shape == (14,)


def test_default_and_explicit_old_generations_have_identical_words():
    default = load_urban_params(1)
    for generation in ('wrf_461', 'wrf_470', 'wrf_471'):
        params = load_urban_params(1, generation=generation)
        for name, value in default.values.items():
            assert np.array_equal(value.view(np.uint32), params.values[name].view(np.uint32)), name
        assert pack_params(params).table.tobytes() == pack_params(default).table.tobytes()


def test_new_table_pin_does_not_accept_the_packaged_old_table():
    with pytest.raises(UrbanTableError, match='SHA-256'):
        load_urban_params(1, tbl_dir=urban_table_path(0).parent, generation='wrf_480')


def test_unknown_generation_is_not_guessed():
    with pytest.raises(UrbanTableError, match='unknown urban table generation'):
        load_urban_params(1, generation='latest')
