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


def test_aggregate_resolves_ensemble_sibling_and_member_maps_keep_ordinary_binary(tmp_path, monkeypatch):
    monkeypatch.delenv("GPUWM_ENSEMBLE_RENDERER", raising=False)
    ordinary = tmp_path / rustwx.executable_name("rw_wrfbatch")
    ensemble = tmp_path / rustwx.executable_name("rw_ensbatch")
    ordinary.write_bytes(b"ordinary-native-binary-fixture")
    ensemble.write_bytes(b"ensemble-native-binary-fixture")
    monkeypatch.setattr(rustwx.bridges, "accept_resolved", lambda path: path)
    monkeypatch.setattr(rustwx, "renderer_candidates", lambda: (ordinary,))
    assert rustwx.find_ensemble_product_renderer(ordinary) == ensemble.resolve()
    assert rustwx.find_renderer() == ordinary.resolve()


def test_missing_explicit_ensemble_binary_never_falls_through(tmp_path, monkeypatch):
    selected = tmp_path / "absent-ensemble-binary"
    monkeypatch.setenv("GPUWM_ENSEMBLE_RENDERER", str(selected))
    with pytest.raises(FileNotFoundError, match="GPUWM_ENSEMBLE_RENDERER names a missing file"):
        rustwx.find_ensemble_product_renderer(tmp_path / "rw_wrfbatch")


def test_native_aggregate_probe_and_render_both_use_ensemble_binary(tmp_path, monkeypatch):
    monkeypatch.delenv("GPUWM_ENSEMBLE_RENDERER", raising=False)
    selected = tmp_path / rustwx.executable_name("rw_ensbatch")
    selected.write_bytes(b"native-ensemble-binary-fixture")
    monkeypatch.setattr(rustwx.bridges, "accept_resolved", lambda path: path)
    image = tmp_path / "native-map.png"
    image.write_bytes(b"native-event-fixture")
    calls = []
    def probe(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0, stdout=rustwx.RESIDENT_ENSEMBLE_ABI + "\n", stderr="")
    class Child:
        returncode = 0
        def __init__(self, command, **kwargs):
            calls.append(command)
        def communicate(self):
            return f"RENDERED\t{image}\n", ""
        def poll(self):
            return self.returncode
    monkeypatch.setattr(rustwx.subprocess, "run", probe)
    monkeypatch.setattr(rustwx.subprocess, "Popen", Child)
    rustwx.run_ensemble_product_renderer(tmp_path / "rw_wrfbatch", tmp_path / "aggregate.nc", out_dir=tmp_path)
    assert [command[0] for command in calls] == [str(selected.resolve())] * 2


@pytest.mark.parametrize("failure", [None, "hash", "revision"])
def test_cpu_reducer_transport_retains_words_and_checks_committed_hash(tmp_path, monkeypatch, failure):
    import hashlib
    import json
    import os
    import numpy as np
    from gpuwm.ensemble.batch_products import FieldProducts, product_memory_plan
    request = FieldProducts("wind10", "m s-1", (10,), paintball=True, postage_stamp=True)
    inventory = product_memory_plan((request,), {"wind10": (2, 2)}, members=2).inventory(2)
    expected = {}
    monkeypatch.setattr(rustwx, "find_ensemble_product_renderer", lambda renderer: renderer)
    def native(command, **kwargs):
        assert command[1] == "--ensemble-diagnostic-reduce"
        declaration = json.loads(Path(command[2]).read_text())
        assert declaration["member_order"] == [19, 3]
        assert declaration["fields"][0]["threshold_bits"] == [1092616192]
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == ""
        directory = Path(command[4])
        directory.mkdir()
        buffers = []
        for index, spec in enumerate(inventory):
            dtype = np.dtype(spec["dtype"])
            array = np.zeros(spec["shape"], dtype=dtype)
            if dtype == np.dtype("float32") and array.size > 1:
                array.view(np.uint32).flat[:2] = (0x80000000, 0x7fc01234)
            expected[spec["name"]] = array.tobytes()
            path = directory / f"words-{index}.bin"
            array.tofile(path)
            buffers.append({"name": spec["name"], "path": str(path), "dtype": dtype.str,
                "shape": list(spec["shape"]), "bytes": array.nbytes,
                "sha256": "0" * 64 if failure == "hash" else hashlib.sha256(array.tobytes()).hexdigest()})
        return SimpleNamespace(returncode=0, stderr="", stdout=json.dumps({
            "schema": "gpuwm-ensemble-diagnostic-reduce.v1", "members": 2,
            "shape": [2, 2], "buffers": buffers}))
    monkeypatch.setattr(rustwx.subprocess, "run", native)
    if failure == "revision":
        read = np.fromfile
        def changed(source, *args, **kwargs):
            values = read(source, *args, **kwargs)
            stat = Path(source.name).stat()
            os.utime(source.name, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
            return values
        monkeypatch.setattr(np, "fromfile", changed)
    def replay():
        return rustwx.reduce_ensemble_diagnostics(Path("rw_ensbatch"), packs=(),
            member_order=(19, 3), shape=(2, 2), valid_time="2026-10-04_00:00:00",
            requests=(request,), inventory=inventory, scratch_directory=tmp_path)
    if failure:
        with pytest.raises(RuntimeError, match="committed SHA-256" if failure == "hash" else "changed while"):
            replay()
    else:
        outputs, retired = replay()
        assert {name: values.tobytes() for name, values in outputs.items()} == expected
        assert len(retired) == len(inventory) + 1
    assert not list(tmp_path.glob("cpu-reduce-*"))


def test_programmatic_collector_probes_cpu_native_contract_before_forecast(tmp_path, monkeypatch):
    from datetime import datetime
    from gpuwm.ensemble.batch_product_output import HeadlineDiagnosticCollector
    checks = []
    monkeypatch.setattr(rustwx, "require_ensemble_diagnostic_reducer", lambda path: checks.append(path) or path)
    HeadlineDiagnosticCollector(tmp_path, members=2, renderer=Path("rw_wrfbatch"),
        start_time=datetime(2026, 10, 4), array_module=SimpleNamespace())
    assert checks == [Path("rw_wrfbatch")]


@pytest.mark.parametrize("abi", [rustwx.CPU_ENSEMBLE_REDUCTION_ABI, "older-native-contract"])
def test_cpu_reduction_gate_checks_the_installed_native_binary(tmp_path, monkeypatch, abi):
    selected = tmp_path / "rw_ensbatch"
    monkeypatch.setattr(rustwx, "find_ensemble_product_renderer", lambda renderer: selected)
    commands = []
    def native(command, **kwargs):
        commands.append(command)
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == ""
        return SimpleNamespace(returncode=0, stdout=abi + "\n", stderr="")
    monkeypatch.setattr(rustwx.subprocess, "run", native)
    if abi == rustwx.CPU_ENSEMBLE_REDUCTION_ABI:
        assert rustwx.require_ensemble_diagnostic_reducer() == selected
    else:
        with pytest.raises(RuntimeError, match="before starting the forecast"):
            rustwx.require_ensemble_diagnostic_reducer()
    assert commands == [[str(selected), "--ensemble-diagnostic-reduce-abi"]]
