"""Validate legacy acquisition bytes without inventing producer provenance."""


def verify(spec, receipt, request, *, allow_regional=False):
    """Keep the original receipt; add checked compatibility evidence only."""
    from gpuwm.earth2_run import resolve_model, model_table
    from gpuwm.tc_tracks import inventory, cf_seconds
    _, row = resolve_model(spec['model'])
    table = model_table()
    ic = table['runtime']['initial_condition_sources'][spec['ic_source']]
    old = receipt.get('model_row', {})
    checked = {}
    unknown = []

    def available(label, source, key, expected):
        if key not in source:
            unknown.append(label)
        elif source[key] != expected:
            raise ValueError(f'legacy fetch {label} differs')
        else:
            checked[label] = source[key]

    for key in ('earth2studio_class', 'profile', 'step_hours', 'members'):
        available(key, old, key, row.get(key))
    weights = old.get('weights', {})
    available('weight_uri', weights, 'uri', row.get('weights', {}).get('uri'))
    available('model_class', request, 'model_class', row['earth2studio_class'])
    available('ic_class', request, 'ic_class', ic['earth2studio_class'])
    available('initial_condition_source', receipt, 'initial_condition_source', spec['ic_source'])
    available('earth2studio', receipt.get('runner', {}), 'earth2studio', table['runtime']['pinned_version'])
    if 'steps' in request:
        if request['steps'] * row['step_hours'] != receipt['hours']:
            raise ValueError('legacy fetch request horizon differs from receipt')
        checked['steps'] = request['steps']
    else:
        unknown.append('steps')
    metadata = inventory(receipt['forecast'])
    variables = metadata['variables']
    times = variables['time']
    initial = cf_seconds(times['values'], times['attrs']['units'], True)
    from datetime import datetime
    wanted = datetime.fromisoformat(spec['cycle'].replace('Z', '+00:00')).timestamp()
    if initial != [wanted]:
        raise ValueError('legacy fetch NetCDF initialization differs')
    lead = variables['lead_time']
    seconds = cf_seconds(lead['values'], lead['attrs']['units'], False)
    expected = list(range(0, spec['hours'] + 1, int(row['step_hours'])))
    if [value / 3600 for value in seconds if 0 <= value <= spec['hours'] * 3600] != expected:
        raise ValueError('legacy fetch NetCDF lead ladder differs')
    if 'ensemble' in variables and len(variables['ensemble']['values']) != 1:
        raise ValueError('legacy fetch has more than one member')
    if request['area'] is None or (spec['area'] is None and not allow_regional):
        lat, lon = variables['lat']['values'], variables['lon']['values']
        if min(lat) > -89.75 or max(lat) < 89.75 or min(lon) > 0 or max(lon) < 359.75:
            raise ValueError('legacy fetch native coordinates do not cover its declared global area')
    if request['area'] is not None:
        from gpuwm.earth2_fetch_cache import covers_area
        lat = variables['lat']['values']
        lon = sorted(set(float(v) % 360 for v in variables['lon']['values']))
        gaps = [(lon[(i + 1) % len(lon)] - value) % 360 for i, value in enumerate(lon)]
        gap = max(range(len(lon)), key=gaps.__getitem__)
        west = lon[(gap + 1) % len(lon)]
        east = west + 360 - gaps[gap]
        # Crop endpoints may fall between cells on the regular 0.25 degree grid.
        actual = [min(lat) - .25, west - .25, max(lat) + .25, east + .25]
        if not covers_area(actual, request['area']):
            raise ValueError('legacy fetch native coordinates do not cover its declared area')
    return dict(native_inventory_schema=metadata.get('schema'),
                checked_leads_hours=expected, member=0,
                checked_producer_fields=checked,
                unknown_producer_fields=unknown,
                producer_weights=weights.get('uri', 'unknown'),
                recorded_weight_metadata=weights,
                producer_member_policy=old.get('members', 'unknown'),
                provenance_basis='recorded fields only; installed defaults are not producer evidence',
                model_row_comparison='available fields checked; absent fields unknown',
                original_model_row=old)
