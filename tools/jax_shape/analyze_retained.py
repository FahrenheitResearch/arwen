"""Compare component timing weightings from compact receipts; no GPU or fields."""
from collections import defaultdict
from pathlib import Path
import hashlib
import json
import sqlite3

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'evidence/jax-shape/local'
fast_path = OUT / 'timing-fast.json'
trace_path = OUT / 'marked-launches.sqlite'
fast = json.loads(fast_path.read_text())
counts = {(r['function'], tuple(r['shape'])): r['count']
          for r in fast['counts_second_period']}
query = '''SELECT n.text AS marker, k.end-k.start AS ns
FROM NVTX_EVENTS n
JOIN CUPTI_ACTIVITY_KIND_RUNTIME r ON r.globalTid=n.globalTid
 AND r.start>=n.start AND r.end<=n.end
JOIN CUPTI_ACTIVITY_KIND_KERNEL k ON k.correlationId=r.correlationId
WHERE n.text LIKE 'shape.%' ORDER BY n.start'''
with sqlite3.connect(f'file:{trace_path}?mode=ro', uri=True) as connection:
    trace = connection.execute(query).fetchall()
assert len(trace) == len(set(marker for marker, ns in trace)) == 54
observed = {}
for marker, ns in trace:
    arm, function, shape = marker.split(':')
    observed[(arm.removeprefix('shape.'), function, tuple(map(int, shape.split(','))))] = ns / 1e6
families = defaultdict(lambda: dict(calls=0, event_before_ms=0., event_after_ms=0.,
                                    profile_before_ms=0., profile_after_ms=0.))
rows = []
for r in fast['rows']:
    function, shape = r['function'], tuple(r['shape'])
    count = counts[(function, shape)]
    row = dict(function=function, shape=shape, calls=count)
    family = families[function]
    family['calls'] += count
    for arm in ('before', 'after'):
        row['event_' + arm + '_ms'] = r['median_ms'][arm]
        row['profile_' + arm + '_ms'] = observed[(arm, function, shape)]
        for method in ('event', 'profile'):
            key = method + '_' + arm + '_ms'
            family[key] += count * row[key]
    rows.append(row)
totals = {key: sum(row[key] for row in families.values())
          for key in next(iter(families.values()))}
for method in ('event', 'profile'):
    totals[method + '_reduction_percent'] = 100 * (1 - totals[method + '_after_ms'] / totals[method + '_before_ms'])
result = dict(scope='First-call durations weighted by observed second-period counts. Neither is measured step time. Event samples used a shared card; profiler has CUDA/NVTX loss warnings and one sample per arm.',
              inputs={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (fast_path, trace_path)},
              query=query, observed_profile_launches=len(trace), totals=totals,
              families=dict(families), rows=rows)
(OUT / 'retained-timing-analysis.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(totals, indent=2))
