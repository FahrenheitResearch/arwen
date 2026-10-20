"""Generated-member identity, native RNG replay and ordinary fetch handoff."""
from collections import OrderedDict
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from gpuwm import earth2_members as members
from gpuwm import earth2_runner as runner


def test_weight_package_identity_survives_cache_relocation_and_open_order():
    assets = [dict(asset="checkpoint", sha256="abc", bytes=123, path="/cache/a"),
              dict(asset="normalization", sha256="def", bytes=45, path="/cache/b")]
    original = members.weight_receipt({"revision": "pinned"}, assets)
    moved = [dict(asset, path="/another/cache/" + asset["asset"])
             for asset in reversed(assets)]
    assert members.weight_receipt({"revision": "pinned"}, moved)["checksum"] == original["checksum"]
    moved[0]["sha256"] = "changed"
    assert members.weight_receipt({"revision": "pinned"}, moved)["checksum"] != original["checksum"]
    assert original["revision"] == "pinned"


def test_population_preserves_requested_count_and_global_seed_offsets():
    assert members.ensemble_members(source="earth2-fcn3", members=3,
        base_seed=333, index_offset=17) == ("n0350", "n0351", "n0352")


@pytest.mark.parametrize("value", ["333", "n333", "n-333", "n0000333", "n4294967296"])
def test_invalid_noise_ids_cannot_alias_seed(value):
    with pytest.raises(ValueError):
        members.validate_member(value)


def test_new_model_is_only_a_table_row(monkeypatch):
    from gpuwm import earth2_run
    table = earth2_run.model_table()
    table["models"]["unseen"] = dict(table["models"]["fcn3"], aliases=["new-parent"])
    table["models"]["unseen"]["members"] = dict(
        table["models"]["fcn3"]["members"], first_seed=807)
    monkeypatch.setattr(earth2_run, "model_table", lambda: table)
    assert members.ensemble_members(source="earth2-new-parent", members=2) == ("n0807", "n0808")
    hints = members.member_fetch_hints({"source": "earth2-unseen"}, "n0808",
                                      window_seconds=7 * 3600, root=Path("data"))
    assert hints["model"] == "unseen" and hints["hours"] == 12


def test_complete_batch_roster_is_attached_before_first_fetch():
    fetches = [members.member_fetch_hints({"source": "earth2-fcn3"}, f"n{i:04d}",
              window_seconds=6 * 3600, root="data") for i in range(333, 351)]
    bound = members.batch_fetch_hints(fetches)
    assert len(bound[0]["earth2_members"].split(",")) == 18
    assert all(row["earth2_members"] == bound[0]["earth2_members"] for row in bound)


def test_native_noise_replays_identically_after_subbatch_change(monkeypatch):
    # This verifies the native state adapter contract using the same RNG API
    # as the installed model, without importing a CUDA runtime into the test.
    perturbations = ModuleType("earth2studio.perturbation")
    perturbations.Zero = lambda: lambda x, coords: (x, coords)
    monkeypatch.setitem(sys.modules, "earth2studio", ModuleType("earth2studio"))
    monkeypatch.setitem(sys.modules, "earth2studio.perturbation", perturbations)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace())

    class Native:
        def set_rng(self, *, seed, reset):
            self.rng = np.random.default_rng(seed)

        def initialize(self, count, times):
            self.states = [[self.rng.integers(0, 2**32) for _ in range(times)]
                           for _ in range(count)]

    policy = {"kind": "native_noise", "perturbation": {"class": "Zero"},
              "native_rng": {"seed_method": "set_rng", "state_initializer": "initialize",
                             "state_attribute": "states"}}
    model = Native()
    operation = runner.member_perturbation(model, policy, [333, 334, 335])
    operation(None, {"ensemble": np.array([0, 1, 2])})
    model.initialize(3, 2)
    batch = list(model.states)
    separate = []
    for i in range(3):
        operation(None, {"ensemble": np.array([i])})
        model.initialize(1, 2)
        separate.extend(model.states)
    assert batch == separate
    assert len({tuple(state) for state in batch}) == 3


def test_handoff_keeps_shared_donor_and_member_receipt(tmp_path):
    from datetime import datetime
    forecast, donor = tmp_path / "parent.nc", tmp_path / "donor.nc"
    forecast.write_bytes(b"forecast")
    donor.write_bytes(b"donor")
    result = members.publish_handoff(tmp_path / "handoff", forecast, donor,
        source="earth2-fcn3", cycle=datetime(2022, 6, 13, 18), member="n0333",
        batch_receipt=tmp_path / members.RECEIPT)
    assert result["member"] == "n0333"
    assert result["unbound_supplement_roles"] == []
    assert f"physical_analysis_surface_data={donor}" in result["argv"]
    assert (tmp_path / "handoff/inputs.txt").read_text() == str(forecast) + "\n"
    from gpuwm.prep_handoff import preparation_arguments
    assert preparation_arguments(result) == result["argv"]
    from gpuwm.fetch_routes import publishes_prep_handoff
    assert publishes_prep_handoff("earth2-fcn3")


def test_noise_handoff_verifies_actual_metadata_and_shared_donor(tmp_path):
    from datetime import datetime
    from gpuwm import earth2_run, forcing_member
    from gpuwm.member_grammar import MemberIdentityRefusal
    netcdf = pytest.importorskip("netCDF4")
    forecasts = []
    for lead in (0, 6):
        path = tmp_path / f"earth2-f{lead:04d}.nc"
        with netcdf.Dataset(path, "w") as data:
            data.setncattr("earth2_member", 0)
            data.setncattr("earth2_generation_seed", 333)
        forecasts.append({"path": str(path), "sha256": members.sha256(path)})
    donor = tmp_path / "donor.nc"
    donor.write_bytes(b"shared donor")
    receipt = {"schema": members.SCHEMA, "model": "fcn3", "weights": {"revision": "abc"},
        "identity": {"cycle": "2022-06-13T18:00:00", "members": ["n0333"]},
        "members": [{"id": "n0333", "seed": 333, "forecasts": forecasts}],
        "donor": {"path": str(donor), "sha256": members.sha256(donor)}}
    batch = tmp_path / members.RECEIPT
    batch.write_text(json.dumps(receipt))
    handoff = members.publish_handoff(tmp_path / "handoff", [f["path"] for f in forecasts],
        donor, source="earth2-fcn3", cycle=datetime(2022, 6, 13, 18),
        member="n0333", batch_receipt=batch)
    hints = {"source": "earth2-fcn3", "cycle": "2022-06-13T18", "member": "n0333",
             "earth2_batch_root": str(tmp_path)}
    verified = earth2_run.verify_handoff(hints, handoff, out=tmp_path / "verified")
    assert len(verified["files"]) == 2 and verified["seed"] == 333
    forcing_member.verify_unchanged(verified)
    donor.write_bytes(b"replaced")
    with pytest.raises(MemberIdentityRefusal, match="changed during preparation"):
        forcing_member.verify_unchanged(verified)
    with netcdf.Dataset(forecasts[1]["path"], "a") as data:
        data.setncattr("earth2_generation_seed", 334)
    with pytest.raises(MemberIdentityRefusal, match="member/seed differs"):
        earth2_run.verify_handoff(hints, handoff, out=tmp_path / "rejected")


def test_completed_batch_reuses_one_job_and_rejects_changed_identity(tmp_path, monkeypatch):
    from datetime import datetime
    from gpuwm import earth2_run, era5_arco
    calls = []
    monkeypatch.setattr(earth2_run, "earth2_python", lambda: "isolated-python")
    prefix = ["pool-helper", "--cards", "1", "e2c-test", "--", "timeout", "600"]
    monkeypatch.setenv("GPUWM_EARTH2_STAGE_PREFIX", json.dumps({"model": prefix}))

    def run(argv, *, stdout, stderr):
        assert argv[:-3] == prefix
        assert calls == ["donor"]
        calls.append("runner")
        request = json.loads(Path(argv[-1]).read_text())
        for path, seed in zip(request["member_outputs"], request["seeds"]):
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_bytes(str(seed).encode())
        stdout.write(json.dumps({"weights": {"revision": "abc"},
            "batch_size": 3, "batch_memory": {"sub_batches": 1},
            "inference_seconds": 3., "peak_gpu_gib": 27.}) + "\n")
        return SimpleNamespace(returncode=0)

    def donor(**kwargs):
        calls.append("donor")
        path = Path(kwargs["out"]) / "era5-combined.nc"
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(b"shared")
        return path

    class Pool:
        def __init__(self, *, max_workers, mp_context):
            assert max_workers == 3 and mp_context.get_start_method() == "spawn"
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def map(self, operation, jobs):
            return map(operation, jobs)

    def stage(job):
        assert calls[:2] == ["donor", "runner"]
        root, member, seed, *_ = job
        path = Path(root) / member / members.FORECAST
        return {"id": member, "forecast": str(path), "forecast_sha256": members.sha256(path)}

    monkeypatch.setattr(members.subprocess, "run", run)
    monkeypatch.setattr(era5_arco, "retrieve_era5_arco", donor)
    monkeypatch.setattr(members, "ProcessPoolExecutor", Pool)
    monkeypatch.setattr(members, "_stage_member", stage)
    kwargs = dict(model="fcn3", cycle=datetime(2022, 6, 13, 18), hours=6,
        area=[40, -100, 48, -89], out=tmp_path,
        members=["n0333", "n0334", "n0335"], workers=3)
    original = members.acquire_batch(**kwargs)
    assert members.acquire_batch(**kwargs) == original
    assert calls == ["donor", "runner"]
    admission = json.loads((tmp_path / "earth2-admission.json").read_text())
    assert admission["scheduler_prefix"] == prefix
    assert admission["returncode"] == 0
    with pytest.raises(ValueError, match="different Earth-2 batch identity"):
        members.acquire_batch(**dict(kwargs, members=["n0333", "n0334", "n0336"]))
    Path(original["donor"]["path"]).write_bytes(b"wrong")
    with pytest.raises(ValueError, match="shared donor digest changed"):
        members.acquire_batch(**kwargs)
    Path(original["donor"]["path"]).write_bytes(b"shared")
    Path(original["members"][1]["forecast"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="forecast digest changed"):
        members.acquire_batch(**kwargs)


def test_workflow_backend_routes_parallel_files_with_member_seed(tmp_path):
    submitted = []
    class Executor:
        def submit(self, operation, *args):
            submitted.append((operation, args))
            return SimpleNamespace(result=lambda: None)
    backend = runner.ParallelFrames.__new__(runner.ParallelFrames)
    backend.output = tmp_path / "earth2.nc"
    backend.members = [0, 1, 2]
    backend.seeds = [333, 334, 335]
    backend.member_outputs = [str(tmp_path / f"n{i:04d}/earth2.nc") for i in backend.seeds]
    backend.attributes = {"lat": {"units": "degrees_north"}}
    backend.geolocation = {}
    backend.executor = Executor()
    backend.pending = []
    backend.workers = 3
    coords = OrderedDict(ensemble=np.array([1, 2]),
                         lead_time=np.array([np.timedelta64(6, "h")]),
                         lat=np.arange(2), lon=np.arange(2))
    arrays = [np.array([np.full((1, 2, 2), 334), np.full((1, 2, 2), 335)])]
    backend.write(arrays, coords, ["t2m"])
    assert len(submitted) == 2
    for offset, (_, args) in enumerate(submitted):
        path, arrays, coordinates, variables, member, seed, attributes, geolocation = args
        assert path.endswith(f"n{334+offset:04d}/earth2-f0006.nc")
        assert "ensemble" not in coordinates and seed == 334 + offset
        assert np.all(arrays[0] == seed)
        assert attributes["lat"]["units"] == "degrees_north"
        assert geolocation == {}


def test_writer_contract_retains_profile_declared_lead_units():
    from gpuwm.earth2_run import producer_attributes
    from gpuwm.source_authorities import packaged_authorities
    mapping = packaged_authorities("earth2-pl13-q-netcdf-v1")["mapping"]
    attributes = producer_attributes(mapping)
    assert attributes["lead_time"]["units"] == "hours"
    assert attributes["lat"]["units"] == "degrees_north"


def test_card_memory_sizes_full_roster_and_respects_explicit_cap():
    from gpuwm.earth2_run import model_table
    policy = model_table()['models']['fcn3']['members']
    request = {'members': list(range(18)), 'member_policy': policy, 'batch_size': 'auto'}
    coords = {'batch': np.empty(0), 'time': np.empty(0), 'lead_time': np.arange(1),
              'variable': np.arange(72), 'lat': np.arange(721), 'lon': np.arange(1440)}
    large, basis = runner.select_batch_size(request, coords,
        free_bytes=90*2**30, total_bytes=96*2**30)
    small, _ = runner.select_batch_size(request, coords,
        free_bytes=29*2**30, total_bytes=32*2**30)
    assert large == 18 and basis['sub_batches'] == 1
    assert basis['capacity_is_estimate'] and small < large
    capped, _ = runner.select_batch_size(dict(request, batch_size=9), coords,
        free_bytes=90*2**30, total_bytes=96*2**30)
    assert capped == 9
    # Below the unmeasured batch contract the batch runs one member at a time
    # (the row single-forecast path), and the receipt says so.
    one, below = runner.select_batch_size(request, coords, free_bytes=20*2**30, total_bytes=32*2**30)
    assert one == 1 and below['below_contract'] and below['sub_batches'] == 18


def test_native_rng_without_member_state_remains_replayable_on_large_card():
    request = {'members': list(range(18)), 'batch_size': 'auto',
               'member_policy': {'native_rng': {'generator': 'torch'}}}
    size, basis = runner.select_batch_size(request, {}, free_bytes=90*2**30, total_bytes=96*2**30)
    assert size == 1 and basis['native_rng_serial'] and basis['sub_batches'] == 18


def test_preparation_profile_uses_noise_batch_donor_variant(tmp_path):
    from datetime import datetime
    from gpuwm import earth2_run
    from gpuwm.source_adapters import get_source_adapter
    _, row = earth2_run.resolve_model('earth2-fcn3')
    selected = earth2_run.resolve_donor_policy(row, datetime(2022, 6, 13, 18), mode='forecast')
    batch = tmp_path / members.RECEIPT
    batch.write_text(json.dumps({'source': 'earth2-fcn3', 'profile': selected['profile'],
                                 'donor': selected}))
    frame = tmp_path / 'forecast.nc'
    frame.write_bytes(b'forecast')
    donor = tmp_path / 'donor.nc'
    donor.write_bytes(b'donor')
    handoff = members.publish_handoff(tmp_path / 'n0333', frame, donor,
        source='earth2-fcn3', cycle=datetime(2022, 6, 13, 18), member='n0333', batch_receipt=batch)
    args = SimpleNamespace(input_list=tmp_path / 'n0333/inputs.txt')
    assert earth2_run.preparation_profile(args, get_source_adapter('earth2-fcn3')) == selected['profile']
    assert selected['profile'] != row['profile']
    bad = json.loads(batch.read_text()); bad['profile'] = row['profile']
    batch.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match='donor/profile pair'):
        earth2_run.preparation_profile(args, get_source_adapter('earth2-fcn3'))
