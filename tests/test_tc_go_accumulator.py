"""Regression: go must either consume a cyclone request or refuse it before work."""
import json
from types import SimpleNamespace

import pytest

from gpuwm.go_cli import GoRefusal
from gpuwm.tc_accumulator import go_scope, load_request


@pytest.fixture
def config(tmp_path):
    p = tmp_path/'config.toml'
    p.write_text('[experiment]\nname="test"\nstart_time=2022-09-27T00:00:00\n[[domain]]\ngrid_id=1\n[[domain]]\ngrid_id=2\n')
    return p


@pytest.fixture
def request_file(tmp_path):
    p = tmp_path/'products.json'
    p.write_text(json.dumps({'common_grid':dict(south=22.,north=27.,west=-85.,east=-80.,nx=101,ny=101)}))
    return p


def test_missing_request_refuses_before_launch(config, tmp_path):
    called = []
    launch = go_scope(lambda args: called.append(True))
    with pytest.raises(GoRefusal, match='cannot read.*absent.json'):
        launch(SimpleNamespace(config=config, cyclone_products=tmp_path/'absent.json'))
    assert not called


@pytest.mark.parametrize('change,why', [
    ({'products':['nonexistent']}, 'unknown or empty products'),
    ({'products':[]}, 'unknown or empty products'),
    ({'grid_id':9}, 'absent from CONFIG'),
    ({'defer_render':True}, 'unknown request keys'),
    ({'common_grid':dict(south=27,north=22,west=-85,east=-80,nx=5,ny=5)}, 'finite ordered'),
    ({'common_grid':dict(south=22,north=27,west=-85,east=-80,nx=1,ny=5)}, 'integers >= 2'),
])
def test_bad_request_refuses(config, request_file, change, why):
    value = json.loads(request_file.read_text()); value.update(change)
    request_file.write_text(json.dumps(value))
    with pytest.raises(GoRefusal, match=why):
        load_request(request_file, config)


def test_real_dry_run_names_every_step_products(config, request_file, capsys):
    called=[]
    launch=go_scope(lambda args: called.append(True) or 0)
    assert launch(SimpleNamespace(config=config,cyclone_products=request_file,dry_run=True)) == 0
    assert called == [True]
    assert 'every committed step on a fixed Earth grid' in capsys.readouterr().out
    assert load_request(request_file,config)['products'] == ['max-wind','gust','min-pressure','track','rain']


def test_stage_only_does_not_claim_products(config,request_file):
    with pytest.raises(GoRefusal,match='needs forecast steps'):
        go_scope(lambda args:0)(SimpleNamespace(config=config,cyclone_products=request_file,prepare_only=True))


def test_rustwx_chooses_accumulator_mode(tmp_path, monkeypatch):
    from gpuwm import rustwx
    p=tmp_path/'receipt.json';p.write_text('{"schema":"gpuwm.tc-accumulator.v1"}')
    picture=tmp_path/'map.png';picture.write_bytes(b'png')
    seen=[]
    def run(command,**kwargs):
        seen.append(command)
        return SimpleNamespace(returncode=0,stdout=f'RENDERED\t{picture}\n',stderr='')
    monkeypatch.setattr(rustwx.subprocess,'run',run)
    assert rustwx.run_cyclone_product_renderer(tmp_path/'rw_wrfbatch',p)['maps']==[str(picture)]
    assert seen[0][1]=='--tc-accumulator'


def test_actual_go_rejects_nonexistent_request(config, tmp_path, capsys):
    from gpuwm.cli import main
    result = main(['go',str(config),'--cyclone-products',str(tmp_path/'no-such-product.json'),'--dry-run'])
    assert result != 0
    assert 'cannot read' in capsys.readouterr().err


@pytest.mark.parametrize('change,why', [
    ({'analysis_time':'tomorrow'}, 'ISO timestamp'),
    ({'best_track':'missing.csv'}, 'unknown request keys'),
])
def test_unused_or_invalid_metadata_refuses(config,request_file,change,why):
    value=json.loads(request_file.read_text());value.update(change)
    request_file.write_text(json.dumps(value))
    with pytest.raises(GoRefusal,match=why):
        load_request(request_file,config)


def test_incompatible_renderer_refuses_before_forecast(config,request_file,monkeypatch,tmp_path):
    from gpuwm import rustwx
    monkeypatch.setattr(rustwx,'find_renderer',lambda:tmp_path/'rw_wrfbatch')
    monkeypatch.setattr(rustwx.subprocess,'run',lambda *a,**kw:SimpleNamespace(returncode=1,stdout=''))
    called=[]
    with pytest.raises(GoRefusal,match='lacks the every-step accumulator contract'):
        go_scope(lambda args:called.append(True))(SimpleNamespace(config=config,cyclone_products=request_file))
    assert not called


@pytest.mark.parametrize('valid_time', ['2022-09-28T01:00:00Z',
    '2022-09-28T01:00:00', '2022-09-28T03:00:00+02:00'])
def test_track_epoch_follows_actual_cycle(config,request_file,valid_time):
    from gpuwm.tc_accumulator import Capture
    capture=Capture(load_request(request_file,config))
    node=SimpleNamespace(cfg=SimpleNamespace(grid_id=2),state=SimpleNamespace(elapsed_seconds=3600))
    capture.model=SimpleNamespace(walk_parent_first=lambda:[node])
    seen=[]
    capture.committed_step=lambda **kw:seen.append(kw)
    capture(grid_id=2,valid_time=valid_time)
    assert capture.request['analysis_time']=='2022-09-28T00:00:00+00:00'
    assert len(seen)==1


def test_naive_model_clock_initializes_utc_track(config,request_file):
    from datetime import datetime
    from gpuwm.tc_accumulator import Capture
    capture=Capture(load_request(request_file,config))
    capture.initialize(model=SimpleNamespace(_declared_experiment=
        SimpleNamespace(start_time=datetime(2022,9,27))))
    assert capture.request['analysis_time']=='2022-09-27T00:00:00+00:00'


def test_cyclone_setup_tiles_auto_is_accepted_and_on_refused(tmp_path, request_file):
    # cyclone-setup --vram-gib authors [tiles] mode = "auto"; --cyclone-products refused it
    # (TC phase 1: arwenrun.sh had to sed the mode to "off").  "auto" keeps the domain
    # resident when it fits, so only "on" (streams every domain) refuses up front.
    base = '[experiment]\nname="test"\nstart_time=2022-09-27T00:00:00\n[[domain]]\ngrid_id=1\n[[domain]]\ngrid_id=2\n'
    auto, on = tmp_path/'auto.toml', tmp_path/'on.toml'
    auto.write_text(base + '["tiles"]\n"mode" = "auto"\n"store" = "host"\n')
    on.write_text(base + '["tiles"]\n"mode" = "on"\n')
    assert load_request(request_file, auto)['grid_id'] == 2
    with pytest.raises(GoRefusal, match="mode = 'on' streams every domain"):
        load_request(request_file, on)


def test_auto_tiles_that_stream_the_observed_domain_refuse_at_first_sample(config, request_file):
    from gpuwm.tc_accumulator import Capture
    capture = Capture(load_request(request_file, config))
    node = SimpleNamespace(cfg=SimpleNamespace(grid_id=2), state=SimpleNamespace(_streamed_domain=object()))
    with pytest.raises(RuntimeError, match="streamed d02"):
        capture.committed_step(node=node, model=None)
