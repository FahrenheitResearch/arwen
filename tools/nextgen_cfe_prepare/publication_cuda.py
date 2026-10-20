"""Private replay consumer for sealed native hourly CFE publication.

Rust extracts and compares the native words; CUDA performs the transfer. This
is neither a 300-second routing policy nor a physical event qualification.
"""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import time


def main():
    root = Path(sys.argv[1]).resolve()
    identity = json.loads((root / 'source-identity.json').read_text())
    for relative, digest in identity['files'].items():
        assert hashlib.sha256((root / 'source' / relative).read_bytes()).hexdigest() == digest, relative
    for relative, digest in identity['artifacts'].items():
        assert hashlib.sha256((root / relative).read_bytes()).hexdigest() == digest, relative
    fixture = json.loads((root / 'fixture/fixture.json').read_text())
    assert hashlib.sha256((root / 'native-index.json').read_bytes()).hexdigest() == fixture['native_index']['index_sha256']
    mode = os.environ['GPUWM_NUMERICS']
    out = root / mode
    out.mkdir()
    import cupy as cp
    from cupy_backends.cuda.libs import nvrtc
    from gpuwm.hydro import ResidentHydroDriver, Stage
    from gpuwm.hydro.runoff import PreparedRunoffRouting
    from gpuwm.hydro.output import _snapshot, _call, restore_checkpoint
    from gpuwm.nextgen.raw_cuda import RawCuda
    from gpuwm import nvrtc_ptx_cache
    from gpuwm.certify.kernel_manifest import kernel_manifest
    cp.get_default_memory_pool().set_limit(size=64*1024*1024)
    raw = RawCuda(root / 'libnextgen_replay.so')
    stream = cp.cuda.Stream(non_blocking=True)
    specs = {k: dict(v, path=str(root / 'fixture' / v['path']))
             for k, v in fixture['inputs'].items() if k != 'expected'}
    times = {}
    with stream:
        inputs = raw.load_many(specs, stream=stream)
        steps, columns = fixture['inputs']['depth']['shape']
        transfer = PreparedRunoffRouting(depth=cp.empty(columns, cp.float64),
            area=inputs['area'], catchment_to_node=inputs['receiver'],
            node_count=len(fixture['node_ids']), period_seconds=fixture['period_seconds'],
            area_units=fixture['area_units'], identity=dict(
                native=fixture['native_index'], catchment_ids=fixture['catchment_ids'],
                node_ids=fixture['node_ids'], inputs=fixture['inputs']))
        driver = ResidentHydroDriver(device_id=0, interval_seconds=fixture['period_seconds'],
            identity={'scope': fixture['scope']},
            after=[Stage('transfer', transfer, transfer.advance)])
        pointers = {k: a.data.ptr for k, a in driver.arrays.items()}
        continuous = cp.empty((steps, len(fixture['node_ids'])), cp.float64)
        split = steps // 2
        resumed = cp.empty((steps-split, len(fixture['node_ids'])), cp.float64)
        def step(i, target):
            cp.copyto(transfer.arrays['depth'], inputs['depth'][i])
            driver.before_lsm(stream=stream)
            driver.advance(stream=stream)
            cp.copyto(target, transfer.arrays['routing_discharge_m3_s'])
        t = time.perf_counter_ns()
        for i in range(split): step(i, continuous[i])
        stream.synchronize()
        times['first_half_replay_host_ms'] = (time.perf_counter_ns()-t)/1e6
        saved_metadata = driver.checkpoint_metadata()
        t = time.perf_counter_ns()
        roster, offsets, payload = _snapshot(driver.arrays, stream)
        checkpoint, _ = _call(dict(operation='checkpoint-write', path=str(out/'restart.nc'),
                                   metadata=saved_metadata, arrays=roster), payload)
        times['rust_checkpoint_snapshot_write_ms'] = (time.perf_counter_ns()-t)/1e6
        for i in range(split, steps): step(i, continuous[i])
        stream.synchronize()
        final_metadata = driver.checkpoint_metadata()
        final_arrays = {k: a.copy() for k, a in driver.arrays.items()}
        stream.synchronize()
        t = time.perf_counter_ns()
        restored = restore_checkpoint(out/'restart.nc', driver)
        assert driver.checkpoint_metadata() == saved_metadata
        assert driver.steps == split and driver.elapsed_seconds == split*fixture['period_seconds']
        times['rust_checkpoint_read_restore_ms'] = (time.perf_counter_ns()-t)/1e6
        for i in range(driver.steps, steps): step(i, resumed[i-split])
        stream.synchronize()
        assert driver.checkpoint_metadata() == final_metadata
        assert pointers == {k: a.data.ptr for k, a in driver.arrays.items()}
        assert driver.check_health(stream=stream) == {'transfer/runoff': 0}
        exports = [(continuous, out/'native-actual.raw'),
                   (cp.ascontiguousarray(continuous[split:]), out/'continuous-tail.raw'),
                   (resumed, out/'resumed-tail.raw')]
        fields = []
        dtype = {'float32': 'f32', 'float64': 'f64', 'int32': 'i32', 'int64': 'i64'}
        for i, (name, actual) in enumerate(driver.arrays.items()):
            expected = out/f'final-{i}-continuous.raw'; dest = out/f'final-{i}-resumed.raw'
            exports.extend([(final_arrays[name], expected), (actual, dest)])
            fields.append(dict(name=name, dtype=dtype[str(actual.dtype)], shape=list(actual.shape),
                               order='c', expected={'path': str(expected)}, actual=str(dest)))
        raw.store_many(exports, stream=stream)
    for field in fields: field['expected']['sha256'] = raw.sha256(field['expected']['path'])
    fields.append(dict(name='second-half-discharge-history', dtype='f64', shape=list(resumed.shape),
        order='c', units='m3/s', expected=dict(path=str(out/'continuous-tail.raw'),
        sha256=raw.sha256(out/'continuous-tail.raw')), actual=str(out/'resumed-tail.raw')))
    evidence = dict(path=str(root/'native-index.json'), sha256=raw.sha256(root/'native-index.json'))
    provenance = dict(origin='native-bmi', sources=[dict(name='NOAA-OWP/cfe',
        revision='f9182dfb3d81c407c66c5b782114173893c30cf2', licence='Apache-2.0'),
        dict(name='NOAA-OWP/ngen', revision='be11a180565a78ff069d855b0b5dbfc23845e7d8', licence='Apache-2.0')], evidence=[evidence])
    base = dict(schema='nextgen.raw.compare/v1', unit='cfe-runoff-publication', case=fixture['scope'],
        numerics=mode, candidate_revision=identity['revision'], command=sys.argv)
    expected = specs.get('expected', fixture['inputs']['expected'])
    native_fields = [dict(name='hourly-native-nexus-discharge', dtype='f64',
        shape=expected['shape'], order='c', units='m3/s', expected=dict(
        path=str(root/'fixture'/expected['path']), sha256=expected['sha256']),
        actual=str(out/'native-actual.raw'))]
    verdicts = {}
    for name, rows, prov in [('native', native_fields, provenance),
            ('restart', fields, dict(origin='tool-self-test', sources=provenance['sources'], evidence=[evidence]))]:
        request = out/f'{name}-request.json'; verdict = out/f'{name}-verdict.json'
        request.write_text(json.dumps(dict(base, provenance=prov, fields=rows), indent=2)+'\n')
        t = time.perf_counter_ns()
        proc = subprocess.run([str(root/'nextgen-replay'), str(request), str(verdict)], capture_output=True, text=True)
        verdicts[name] = dict(returncode=proc.returncode, stdout=proc.stdout, stderr=proc.stderr,
            wall_ms=(time.perf_counter_ns()-t)/1e6, verdict=json.loads(verdict.read_text()) if verdict.exists() else None)
    record = dict(revision=identity['revision'], mode=mode, fixture=fixture,
        scope=fixture['scope'], timings=times, verdicts=verdicts, checkpoint=checkpoint,
        checkpoint_carriers=len(driver.arrays), checkpoint_payload_bytes=sum(a.nbytes for a in driver.arrays.values()),
        split_step=split, final_steps=driver.steps, final_elapsed_seconds=driver.elapsed_seconds,
        restored_step=saved_metadata['steps'], pointers_preserved=True,
        resources=transfer.kernel.attributes, cupy_used_bytes=cp.get_default_memory_pool().used_bytes(),
        cupy_reserved_bytes=cp.get_default_memory_pool().total_bytes(), allocator_cap_bytes=cp.get_default_memory_pool().get_limit(),
        cuda_graphs=False, launches_per_transfer=1, whole_step_speed_claim=False,
        compile_events=nvrtc_ptx_cache.cache_events(), kernel_manifest=kernel_manifest(),
        environment=dict(python=sys.version, cupy=cp.__version__, nvrtc=nvrtc.getVersion(),
            cuda_runtime=cp.cuda.runtime.runtimeGetVersion(), driver_api=cp.cuda.runtime.driverGetVersion(),
            gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,driver_version','--format=csv,noheader'],text=True)))
    (out/'checks.json').write_text(json.dumps(record,indent=2)+'\n')
    assert all(v['returncode']==0 for v in verdicts.values()), 'retained native/restart word verdicts'
    print(json.dumps(dict(mode=mode, carrier_count=len(driver.arrays), timings=times,
                         verdicts={k:v['returncode'] for k,v in verdicts.items()})))


if __name__ == '__main__': main()
