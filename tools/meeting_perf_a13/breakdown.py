"""Per-term admission breakdown for HRRR under [devices] grids (CPU only)."""
import json
import sys
from dataclasses import replace
from collections import defaultdict

sys.path.insert(0, "/work/meeting-perf-a13/tree")
import gpuwm
assert gpuwm.__file__.startswith("/work/meeting-perf-a13/tree"), gpuwm.__file__
from gpuwm.experiment import load_experiment
from gpuwm.core import preflight as pf
from gpuwm.core import devices_memory as dm
from gpuwm.core.mynn_pbl_scratch import mynn_pricing_rank_chunk, mynn_rank_chunk_candidates
from gpuwm.core.streaming import ranked_halo, ranked_specs
from tilestream.harness import tile_config

GIB = 2**30
exp = load_experiment(sys.argv[1])
grids = sys.argv[2].split(",")
out = {}
for grid in grids:
    gx, gy = (int(v) for v in grid.split("x"))
    n = gx * gy
    from gpuwm.core.devices import DeviceOptions
    opts = DeviceOptions(count=n, grid=grid, ids=tuple(range(n)), transport="auto")
    e = replace(exp, devices=opts)
    budget = int(30.86 * GIB)
    est = dm.estimate_devices(e, options=opts, forcing_intervals=2, source="rap-native",
                              budgets={i: budget for i in range(n)})
    cfg = e.root.run
    halo = ranked_halo(cfg)
    specs = ranked_specs(cfg, opts, halo=halo)
    # itemize the widest rank at the MINIMUM mynn chunk and at the chosen chunk
    widest = max(range(n), key=lambda r: specs[r].cnx * specs[r].cny)
    run = tile_config(cfg, specs[widest].cnx, specs[widest].cny)
    local = replace(e, devices=dm.DEVICES_OFF, domains=(replace(e.root, run=run),))
    rows = {}
    cands = mynn_rank_chunk_candidates(int(cfg.nz))
    for label, chunk in (("min_chunk", cands[0]),
                         ("chosen_chunk", est["rank_shapes"][widest].get("mynn_column_chunk"))):
        from gpuwm.boundary_fields import source_boundary_species
        with mynn_pricing_rank_chunk(chunk):
            x = pf.estimate_experiment(local, forcing_intervals=1, tile_buffer=True,
                                       boundary_species=source_boundary_species("rap-native"))
        d = x.domains[0]
        by_cat = defaultdict(int)
        top = []
        for item in d.items:
            by_cat[item.category] += item.nbytes
            top.append((item.nbytes, item.category, item.name))
        top.sort(reverse=True)
        rows[label] = dict(
            chunk=chunk,
            peak_envelope_gib=x.peak_envelope_bytes / GIB,
            alloc_estimate_gib=x.alloc_estimate_bytes / GIB,
            subtotal_gib=x.subtotal_bytes / GIB,
            resident_gib=x.resident_bytes / GIB, workspace_gib=x.workspace_bytes / GIB,
            transient_gib=x.transient_peak_bytes / GIB, non_pool_gib=x.non_pool_device_bytes / GIB,
            column_workspace_gib=x.column_workspace_bytes / GIB, k_tables_gib=x.k_tables_bytes / GIB,
            physics_tables_gib=x.physics_tables_bytes / GIB, headroom=x.headroom,
            legacy_call_peak_gib=[v / GIB for v in x.legacy_call_peak_by_domain],
            domain_transient_gib=x.domains[0].transient_bytes / GIB,
            domain_resident_gib=x.domains[0].resident_bytes / GIB,
            by_category_gib={k: v / GIB for k, v in sorted(by_cat.items(), key=lambda kv: -kv[1])},
            top30=[(round(b / GIB, 4), c, nm) for b, c, nm in top],
            )
    out[grid] = dict(
        halo=halo, rank_shape=[specs[widest].cny, specs[widest].cnx],
        interior=[specs[widest].interior_ny, specs[widest].interior_nx],
        cards=[{k: (v / GIB if k.endswith("bytes") else v) for k, v in c.items()} for c in est["cards"]],
        ranks=[{k: (v / GIB if k.endswith("bytes") else v) for k, v in r.items()} for r in est["rank_shapes"]],
        widest_rank=widest, itemized=rows)
json.dump(out, sys.stdout, indent=1, default=str)
