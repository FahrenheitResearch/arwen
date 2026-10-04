"""The aggregate mode binds its own native contract and durable map events."""
from pathlib import Path
from types import SimpleNamespace
import pytest
from gpuwm import rustwx


def test_resident_ensemble_sends_aggregate_file_only(tmp_path, monkeypatch):
    calls=[]
    image=tmp_path/'case'/'d01'/'ens_prob_wind10_ge_10'/'2024-05-25'/'frame.png'
    image.parent.mkdir(parents=True);image.write_bytes(b'native-event-fixture')
    def run(command, **kwargs):
        calls.append(command)
        if command[-1]=='--ensemble-products-abi':
            return SimpleNamespace(returncode=0,stdout=rustwx.RESIDENT_ENSEMBLE_ABI+'\n',stderr='')
        raise AssertionError('only the ABI probe uses subprocess.run')
    class Child:
        returncode=0
        def __init__(self,command,**kwargs):calls.append(command)
        def communicate(self):return f'RENDERED\t{image}\n',''
        def poll(self):return self.returncode
    monkeypatch.setattr(rustwx.subprocess,'run',run)
    monkeypatch.setattr(rustwx.subprocess,'Popen',Child)
    written,_=rustwx.run_ensemble_product_renderer(Path('rw_wrfbatch'),tmp_path/'aggregate.nc',
             out_dir=tmp_path/'case',fields=('wind10','temperature2'))
    assert written==[image]
    assert calls[1][1:3]==['--ensemble-products',str(tmp_path/'aggregate.nc')]
    assert '--member' not in calls[1]
    assert '--store-root' not in calls[1]


def test_old_native_renderer_cannot_claim_resident_products(monkeypatch,tmp_path):
    monkeypatch.setattr(rustwx.subprocess,'run',lambda *a,**k: SimpleNamespace(returncode=2,stdout='',stderr='unknown option'))
    with pytest.raises(RuntimeError,match='lacks the resident ensemble product contract'):
        rustwx.run_ensemble_product_renderer(Path('rw_wrfbatch'),tmp_path/'aggregate.nc',out_dir=tmp_path)


def test_interrupt_reaps_the_owned_native_child(monkeypatch,tmp_path):
    monkeypatch.setattr(rustwx.subprocess,'run',lambda *a,**k: SimpleNamespace(returncode=0,stdout=rustwx.RESIDENT_ENSEMBLE_ABI+'\n',stderr=''))
    actions=[]
    class Child:
        returncode=None
        def __init__(self,*a,**k):pass
        def communicate(self):raise SystemExit(143)
        def poll(self):return self.returncode
        def terminate(self):actions.append('terminate')
        def wait(self,timeout=None):actions.append('reap');self.returncode=-15
    monkeypatch.setattr(rustwx.subprocess,'Popen',Child)
    with pytest.raises(SystemExit):
        rustwx.run_ensemble_product_renderer(Path('rw_wrfbatch'),tmp_path/'aggregate.nc',out_dir=tmp_path)
    assert actions==['terminate','reap']
