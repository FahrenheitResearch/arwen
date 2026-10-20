"""Complete-roster GPU products from the launcher's thin diagnostic planes."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import time


def reduce_set(root, receipt):
    import cupy as cp
    import numpy as np
    from gpuwm import rustwx
    from gpuwm.ensemble.batch_products import (FieldProducts, prepare_product_frame,
        write_product_frame, apply_member_weights, default_product_requests)
    root = Path(root)
    frames = {}
    members = sorted(receipt['completed'], key=lambda member: member['member_id'])
    tracks = []
    grids = {}
    for member in members:
        seen = set()
        for frame in member['frames']:
            key = (frame['grid_id'], frame['episode'], frame['valid_time'])
            if key in seen:
                raise ValueError('duplicate member diagnostic frame')
            seen.add(key)
            frames.setdefault(key, {})[member['member_id']] = frame['path']
            tracks.append({'member_id': member['member_id'], **{k: frame[k] for k in
                          ('grid_id', 'episode', 'valid_time', 'track')}})
    order = [member['member_id'] for member in members]
    weights = cp.asarray([member.get('weight', 1/len(members)) for member in members], dtype=cp.float64)
    weights /= weights.sum()
    from gpuwm.ensemble.family_products import family_roster, reduce_family_frame
    has_families = any(m.get('provenance', {}).get('parent_family') or m.get('provenance', {}).get('family') or m.get('parent_family') or m.get('family') for m in members)
    roster = family_roster(members) if has_families else None
    if roster:
        weights = cp.asarray([m['weight'] for m in roster['members']], dtype=cp.float64)
    if not frames or any(set(rows) != set(order) for rows in frames.values()):
        raise ValueError('whole-set products need every requested member at every common valid time')
    renderer = rustwx.find_ensemble_product_renderer()
    if renderer is None:
        raise RuntimeError('whole-set products require the public Rust ensemble renderer')
    outputs = []
    from gpuwm.ingest.cpu_backend import available_cpu_count
    reader_count = available_cpu_count()
    def load(path):
        with np.load(path, allow_pickle=False) as archive:
            return {key: archive[key] for key in archive.files}
    with ThreadPoolExecutor(max_workers=reader_count) as readers:
        for (grid, episode, valid), rows in sorted(frames.items()):
            started = time.perf_counter()
            arrays = list(readers.map(load, [rows[member] for member in order]))
            lat, lon = arrays[0]['latitude'], arrays[0]['longitude']
            grids[(grid, episode)] = (lat, lon, valid)
            if any(not np.array_equal(a['latitude'], lat) or not np.array_equal(a['longitude'], lon)
                   for a in arrays):
                raise ValueError('ensemble members have different product grids')
            # One field at a time bounds device memory independently of field
            # count. All N members remain in each probability denominator.
            requests, unavailable = default_product_requests(set.intersection(*(set(a) for a in arrays)), postage_stamp=False)
            for request in requests:
                field, units, thresholds = request.field, request.units, request.thresholds
                stack = cp.stack([cp.asarray(array[field], dtype=cp.float32) for array in arrays])
                products = prepare_product_frame({field: stack},
                    (FieldProducts(field, units, thresholds, paintball=True, postage_stamp=False, spaghetti=False),),
                    available_bytes=int(cp.cuda.runtime.memGetInfo()[0])-(256<<20))
                products()
                apply_member_weights(products, cp.asnumpy(weights))
                family_outputs = []
                if roster:
                    for family, reduced, attrs in reduce_family_frame({field: stack},
                            (FieldProducts(field, units, thresholds),), roster,
                            available_bytes=int(cp.cuda.runtime.memGetInfo()[0])-(256<<20), array_module=cp):
                        family_path = root/'products'/f'd{grid:02d}'/'families'/family/field/(valid.replace(':', '-')+'.nc')
                        family_path.parent.mkdir(parents=True, exist_ok=True)
                        write_product_frame(family_path, reduced, valid_time=valid, latitude=lat, longitude=lon,
                                            global_attrs=attrs)
                        drawn, _ = rustwx.run_ensemble_product_renderer(renderer, family_path,
                            out_dir=root/'maps'/'families'/family, fields=(field,), domain=f'd{grid:02d}',
                            products=('mean', 'prob', 'pmm'), source_label=family)
                        family_outputs.append({'family': family, 'product': str(family_path),
                                               'maps': [str(p) for p in drawn]})
                    del reduced
                stamp = valid.replace(':', '-').replace('+', '_')
                product_path = root/'products'/f'd{grid:02d}'/field/f'{stamp}.nc'
                product_path.parent.mkdir(parents=True, exist_ok=True)
                write_product_frame(product_path, products, valid_time=valid, latitude=lat, longitude=lon,
                                    global_attrs={'provenance_receipt': 'ensemble-set.json',
                                        'GPUWM_ENSEMBLE_PROVENANCE': json.dumps([m['provenance']['label'] for m in members]),
                                        'member_probability_weights': json.dumps(cp.asnumpy(weights).tolist()),
                                        'family_weights': json.dumps(roster['family_weights'] if roster else {})})
                maps, _ = rustwx.run_ensemble_product_renderer(renderer, product_path,
                    out_dir=root/'maps', fields=(field,), domain=f'd{grid:02d}',
                    products=('prob', 'paintball', 'pmm') if field in ('refl', 'rain_total') else ('prob', 'paintball'))
                row = {'valid_time': valid, 'field': field, 'product': str(product_path), 'families': family_outputs,
                       'maps': [str(path) for path in maps]}
                row['seconds_since_frame_start'] = time.perf_counter()-started
                outputs.append(row)
                del products, stack
            del arrays
    track_path = root/'tracks.json'
    track_path.write_text(json.dumps({'schema': 'gpuwm-ensemble-echo-plume.v1',
        'signal': '35 dBZ echo weighted centroid; not cyclone centre',
        'tracker': 'gpuwm.core.storm_tracking.locate_signal', 'members': order, 'fixes': tracks},
        indent=2, default=lambda value: value.item() if hasattr(value, 'item') else str(value))+'\n')
    plume_maps = []
    for (grid, episode), (lat, lon, valid) in grids.items():
        ny, nx = lat.shape
        jj, ii = cp.ogrid[:ny, :nx]
        stack = cp.zeros((len(order), ny, nx), dtype=cp.float32)
        indices = {member: index for index, member in enumerate(order)}
        for fix in tracks:
            point = fix['track']
            if point is None or (fix['grid_id'], fix['episode']) != (grid, episode):
                continue
            mask = (ii-point['ci'])**2+(jj-point['cj'])**2 <= 4.
            plane = stack[indices[fix['member_id']]]
            cp.maximum(plane, mask, out=plane)
        products = prepare_product_frame({'echo_centroid': stack},
            (FieldProducts('echo_centroid', '1', (0.5,), paintball=True),),
            available_bytes=int(cp.cuda.runtime.memGetInfo()[0])-(256<<20))
        products()
        apply_member_weights(products, cp.asnumpy(weights))
        path = root/'products'/f'd{grid:02d}'/'echo-centroid-plume'/f'{episode}.nc'
        path.parent.mkdir(parents=True, exist_ok=True)
        write_product_frame(path, products, valid_time=valid, latitude=lat, longitude=lon,
            global_attrs={'provenance_receipt': 'ensemble-set.json',
                'plume_definition': 'member echo centroid within 2 grid cells at any saved valid time; no interpolation',
                'member_probability_weights': json.dumps(cp.asnumpy(weights).tolist()),
                                        'family_weights': json.dumps(roster['family_weights'] if roster else {}),
                'signal': '35 dBZ weighted echo centroid; no storm identity association'})
        maps, _ = rustwx.run_ensemble_product_renderer(renderer, path, out_dir=root/'plume-maps',
            fields=('echo_centroid',), domain=f'd{grid:02d}', products=('prob', 'paintball'),
            source_label='Echo centroid, 2-cell radius')
        plume_maps.extend(str(path) for path in maps)
    return {'frames': outputs, 'track_plume': str(track_path), 'track_plume_maps': plume_maps,
            'parallelism': {'reduction': 'GPU', 'diagnostic_read': f'{reader_count} host threads', 'render': 'Rust Rayon'}}
