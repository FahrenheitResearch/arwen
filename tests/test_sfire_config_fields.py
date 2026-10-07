"""RunConfig carries the fire namelist as appended keyword-only fields.

The breakage this prevents: the fire collector declared RunConfig as a
subclass of FireRunFields, which put 169 fire fields AHEAD of the 252
existing ones in dataclasses.fields(RunConfig) (the order every config
echo, freeze and checkpoint record walks) and let the two declarations
drift apart silently.
"""
import dataclasses

from gpuwm.config import RunConfig
from gpuwm.sfire_config import FireRunFields


def test_fire_fields_follow_every_existing_field_in_order():
    names = [f.name for f in dataclasses.fields(RunConfig)]
    fire = [f.name for f in dataclasses.fields(FireRunFields)]
    assert names[-len(fire):] == fire
    # The last positional field is the chem block's last; 253 positional
    # fields since the urban line appended surface_energy_diag (was 252).
    assert len(names) - len(fire) == 253
    assert names[:-len(fire)][-1] == "aerosol_mp_coupling"


def test_fire_fields_match_their_declaration():
    ours = {f.name: f for f in dataclasses.fields(RunConfig)}
    for field in dataclasses.fields(FireRunFields):
        mine = ours[field.name]
        assert mine.kw_only, field.name
        assert str(mine.type) == getattr(field.type, "__name__", str(field.type)), field.name
        assert mine.default == field.default, field.name


def test_fire_fields_are_keyword_only_in_the_constructor():
    cfg = RunConfig(nx=10, ny=10, nz=10, dx=1000.0, dy=1000.0, ztop=10000.0,
                    dt=6.0, run_seconds=60.0, ifire=0)
    assert cfg.ifire == 0 and cfg.nx == 10
