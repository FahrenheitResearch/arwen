"""Earth2Studio front door over ml-export's Rust engine. No array processing."""
from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from pathlib import Path

from gpuwm import ml_export as ml

AUTHORITY = Path(__file__).parent / 'authorities' / 'earth2-models.v1.json'


def channel_rows(channels, tables=None, levels=None):
    """Resolve lexicon templates from table rows, independent of consumers."""
    tables = tables or ml.load_tables()
    levels = levels or ml.parse_levels('wb13', tables)['hpa']
    channels = list(dict.fromkeys(c.strip() for c in channels if c.strip()))
    candidates = {}
    for source in tables.variables:
        if source['kind'] == 'level' and 'pressure' not in source.get('level_kinds', []):
            continue
        name = source['names']['earth2']
        pressures = levels
        if '{level}' in name:
            pattern = re.escape(name).replace(re.escape('{level}'), r'([1-9][0-9]*)')
            pressures = sorted(set(levels) | {int(m.group(1)) for c in channels
                                            if (m := re.fullmatch(pattern, c))})
        for pressure in pressures if '{level}' in name else [None]:
            row = copy.deepcopy(ml._request_row(source))
            channel = name.format(level=pressure)
            row['names']['earth2'] = channel
            if pressure is not None:
                row['id'] += ':' + str(pressure)
                row['pressure_hpa'] = pressure
            candidates[channel] = row
    missing = sorted(set(channels) - candidates.keys())
    if missing:
        raise ml.MlExportRefusal('no variable/operator row resolves channels: ' + ', '.join(missing))
    return [candidates[c] for c in dict.fromkeys(channels)]


def build_request(args):
    if not args.earth2:
        raise ml.MlExportRefusal('--earth2 PROFILE is required; use --list to see profiles')
    if args.out is None:
        raise ml.MlExportRefusal('--out is required')
    tables = ml.load_tables()
    if args.regrid != 'bilinear':
        raise ml.MlExportRefusal('fixed Earth2 profiles currently require --regrid bilinear')
    profile_table = ml._load('profiles')
    try:
        profile = profile_table['profiles'][args.earth2]
    except KeyError:
        raise ml.MlExportRefusal('unknown Earth2 grid profile: ' + args.earth2) from None
    consumers = json.loads(AUTHORITY.read_text())['consumers']
    channels = []
    for name in (args.consumers or '').split(','):
        if not name:
            continue
        if name not in consumers:
            raise ml.MlExportRefusal('unknown consumer: ' + name)
        channels.extend(consumers[name]['input_channels'])
    if args.channels:
        channels.extend(args.channels.split(','))
    if not channels:
        channels = profile_table['default_channels']
    rows = channel_rows(channels, tables)
    # Reuse the existing parser, provenance and Rust request contract.
    options = argparse.Namespace(**vars(args))
    options.levels = ','.join(str(p) for p in sorted({r['pressure_hpa'] for r in rows if 'pressure_hpa' in r})) or 'wb13'
    options.names = 'wb2'
    options.variables = ','.join(dict.fromkeys(r['id'].split(':')[0] for r in rows))
    options.grid = 'native'
    options.layout = 'forecast'
    request = ml.build_request(options, tables)
    request.update(variables=rows, names='earth2', layout='earth2')
    request['grid'] = dict(kind='native' if profile['kind']=='native' else 'latlon',method=args.regrid,target=profile)
    if profile['kind'] == 'healpix':
        request['chunk_yx'] = [profile['nside'], profile['nside']]
    request['provenance']['options'] += ' --earth2 ' + args.earth2 + ' --for ' + (args.consumers or '')
    return request


def register_cli(sub):
    # The same options and execution path as ml-export, with table resolution
    # inserted before its binary invocation.
    parser = sub.add_parser('export', help='export history for an Earth2Studio consumer')
    ml.add_arguments(parser, earth2=True)
    parser.add_argument('--earth2', metavar='PROFILE')
    parser.add_argument('--for', dest='consumers', metavar='CONSUMER[,CONSUMER]')
    parser.add_argument('--channels', help='additional lexicon channels, comma separated')
    parser.set_defaults(func=main)


def main(args):
    try:
        if args.list:
            document = dict(profiles=ml._load('profiles')['profiles'],
                            consumers=json.loads(AUTHORITY.read_text())['consumers'])
            print(json.dumps(document, indent=2))
            return ml.EXIT_OK
        request = build_request(args)
        exe = ml.BINARY.find()
        if exe is None:
            print(ml.BINARY.remedy(), file=sys.stderr)
            return ml.EXIT_NOT_STAGED
        ok, why = ml.BINARY.probe(exe)
        if not ok:
            raise ml.MlExportRefusal(why)
        return ml.run_binary(exe, request, as_json=args.json)
    except (ml.MlExportRefusal, FileNotFoundError) as error:
        print('export refused: ' + str(error), file=sys.stderr)
        return ml.EXIT_REFUSED
