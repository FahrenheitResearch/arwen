"""Hub-height products: metadata, acquisition and native renderer orchestration."""
from __future__ import annotations
import copy
import datetime as dt
import json
from pathlib import Path
import re
import sys
import urllib.request

_DATA = Path(__file__).with_name('data')


def catalog():
    return {name: json.loads((_DATA / filename).read_text(encoding='utf-8'))
            for name, filename in (('products', 'hubwind_products.json'),
                                   ('turbines', 'turbine_power_curves.json'),
                                   ('observations', 'hubwind_observations.json'))}


#: The native fields every hub-height product reads (rw-wrfbatch hubwind_products.rs planes()).
_HUB_FIELDS = frozenset({'PH', 'PHB', 'HGT', 'U', 'V'})


def product_names():
    return {p['name'] for p in catalog()['products']} | {'hub_all'}


def configuration(path):
    """A standalone JSON options object or [hubwind] in a forecast TOML."""
    if path is None:
        return {}
    path = Path(path)
    if path.suffix.lower() == '.json':
        data = json.loads(path.read_text(encoding='utf-8-sig'))
    else:
        import tomllib
        data = tomllib.loads(path.read_text(encoding='utf-8-sig')).get('hubwind', {})
    if not isinstance(data, dict):
        raise ValueError('hubwind configuration must be an object/table')
    return data


def _request(config, inputs, out, products, domain, size, timeidx='all'):
    config = dict(config)
    options = config.pop('options', None)
    # Ordinary [hubwind] has the calculation options directly. Request-only
    # fields (extra turbine rows and observations) live beside them.
    extras = {key: config.pop(key) for key in ('turbines', 'observations',
               'observation_max_distance_km', 'render_frames', 'compute_backend', 'verify_cuda') if key in config}
    if options is not None and config:
        raise ValueError('put calculation settings either in options or directly in hubwind')
    for obs in extras.get('observations', ()):
        obs['path'] = str(Path(obs['path']).resolve())
    return {'inputs': [str(Path(p).resolve()) for p in inputs], 'out': str(out.resolve()),
            'options': options if options is not None else config,
            'products': products, 'domain': domain, 'width': size[0], 'height': size[1],
            'record_selection': str(timeidx), **extras}


def _publish_hubwind_receipt(out, args, inputs, result):
    """Hub-height maps in the run's render summary, blank ones as undrawn.

    Breakage it prevents (2.9.4 rolling pass, 2026-10-21 17:10Z): every hub
    TI map of a YSU run was a blank "NO VALID TI" picture while the render
    summary said 0 undrawn, because hub maps never reached the summary at
    all.  Each picture with data is recorded as drawn under its product
    folder (out/<domain>/<product>/<day>/<stamp>.png); each picture with no
    valid value is recorded as skipped with the reason it states, so the
    summary counts it as undrawn and never as drawn.
    """
    from gpuwm import render_layout
    from gpuwm.render_receipts import publish_invocation
    family = lambda path: Path(path).parent.parent.name
    return publish_invocation(
        root=Path(out), engine="rust-hubwind", requested_spec=str(args.products),
        written=list(result.get('maps', ())), failures=[],
        skipped=[(family(path), reason) for path, reason in result.get('undrawn', ())],
        layout=render_layout.NESTED, inputs=list(inputs),
        families={str(Path(path).resolve()): family(path) for path in result.get('maps', ())})


def render(args):
    """Return None when this request belongs entirely to other products."""
    from gpuwm import render as rendering, rustwx
    if (getattr(args, '_hubwind_done', False) or not args.wrfout
            or getattr(args, 'list_products', False)
            or any(getattr(args, flag, None) for flag in ('pair', 'compare', 'diff'))):
        return None
    terms = rustwx.product_spec_terms(str(args.products))
    named = [term for term in terms if term.startswith('hub_')]
    all_requested = any(term.casefold() == 'all' for term in terms)
    if not named and not all_requested:
        return None
    if not named:
        # An implicit `all` takes hub-height maps only where they can be drawn:
        # on the Rust renderer, from inputs carrying the geopotential, terrain
        # and staggered winds the products read.  Otherwise the ordinary
        # catalog renders `all` as it did before hubwind existed, instead of a
        # default matplotlib render refusing or a history without U/V failing
        # with "missing native field V".  Named hub_* products still refuse.
        if args.engine == 'matplotlib':
            return None
        variables = rendering._file_variables(Path(args.wrfout[0]))
        if variables is not None and not _HUB_FIELDS <= variables:
            return None
    known = product_names()
    unknown = sorted(set(named) - known)
    if unknown:
        raise ValueError('unknown hubwind product: ' + ', '.join(unknown))
    selected = [] if all_requested or 'hub_all' in named else named
    config = configuration(getattr(args, 'hubwind_config', None))
    for p in args.wrfout:
        if not Path(p).is_file():
            raise ValueError(f'unreadable hubwind input: {p}')
    if args.engine == 'matplotlib':
        raise ValueError('hubwind products require the native Rust renderer')
    renderer = rendering.require_renderer()
    size = rendering.parse_size(args.size) or (1200, 900)
    rendering._claim_run_dir(args, args.wrfout)
    try:
        groups = {}
        for path in args.wrfout:
            match = re.search(r'wrfout_(d\d+)', Path(path).name)
            groups.setdefault(match.group(1) if match else 'd01', []).append(path)
        for tag, inputs in groups.items():
            # The domain folder is the render tree's token (d01-3km), so hub-height maps file
            # beside every other product of the same grid instead of in a bare d01/ folder.
            domain = rendering.domain_token(tag, rendering._grid_spacing_m(Path(inputs[0])))
            out = Path(args.out)
            # Domain receipts stay separate even when both domains share a run.
            request = _request(config, inputs, out / domain / 'hubwind', selected,
                               domain, size, args.timeidx)
            # Maps follow out/domain/product/day. Receipt/native statistics
            # reside beside the product folders, without another domain layer.
            request['out'] = str(out.resolve())
            request['receipt_prefix'] = domain + '-'
            manifest = out / f'{domain}-hubwind-request.json'
            manifest.parent.mkdir(parents=True, exist_ok=True)
            manifest.write_text(json.dumps(request, indent=2) + '\n', encoding='utf-8')
            result = rustwx.run_hubwind_renderer(renderer, manifest)
            _publish_hubwind_receipt(out, args, inputs, result)
            undrawn = result.get('undrawn', ())
            print(f"render: {len(result['maps'])} hubwind maps"
                  + (f", {len(undrawn)} undrawn (no valid value)" if undrawn else "")
                  + f"; {result['receipt']}")
        remaining = [term for term in terms if term not in named]
        if remaining:
            ordinary = copy.copy(args)
            ordinary.products = ','.join(remaining)
            ordinary._hubwind_done = True
            return rendering.render_main(ordinary)
        return 0
    finally:
        rendering._publish_run_dir(args)


def _fetch(args):
    rows = catalog()['observations']
    row = next((r for r in rows if r['id'] == args.network), None)
    if row is None:
        raise ValueError('unknown hubwind observation row')
    day = dt.date.fromisoformat(args.date)
    url = row['url'].format(year=day.year, doy=f'{day.timetuple().tm_yday:03}')
    args.out.mkdir(parents=True, exist_ok=True)
    target = args.out / url.rsplit('/', 1)[1]
    temporary = target.with_suffix(target.suffix + '.part')
    with urllib.request.urlopen(url, timeout=60) as response, temporary.open('wb') as stream:
        while block := response.read(65536):
            stream.write(block)
    temporary.replace(target)
    target.with_suffix('.acquisition.json').write_text(json.dumps({
        'schema': 'gpuwm.hubwind-acquisition.v1', 'network': row,
        'date': day.isoformat(), 'url': url, 'bytes': target.stat().st_size,
        'path': str(target.resolve())}, indent=2) + '\n', encoding='utf-8')
    print(target)
    return 0


def main(args):
    try:
        if args.hubwind_command == 'catalog':
            print(json.dumps(catalog(), indent=2))
            return 0
        if args.hubwind_command == 'fetch-observations':
            return _fetch(args)
        from gpuwm import rustwx
        from gpuwm.render import require_renderer
        result = rustwx.run_hubwind_renderer(require_renderer(), args.request.resolve())
        print(json.dumps(result, indent=2))
        return 0
    except (ValueError, OSError, RuntimeError) as error:
        print(f'hubwind: {error}', file=sys.stderr)
        return 2


def register_cli(sub):
    parser = sub.add_parser('hubwind', help='hub-height wind, turbine curves, run statistics and tower observations')
    commands = parser.add_subparsers(dest='hubwind_command', required=True)
    commands.add_parser('catalog', help='show turbine, hub product and observation rows').set_defaults(func=main)
    fetch = commands.add_parser('fetch-observations', help='acquire a public tall tower day with source and licence receipt')
    fetch.add_argument('--network', required=True, choices=tuple(r['id'] for r in catalog()['observations']))
    fetch.add_argument('--date', required=True, help='UTC YYYY-MM-DD')
    fetch.add_argument('--out', required=True, type=Path)
    fetch.set_defaults(func=main)
    products = commands.add_parser('products', help='run an explicit native hubwind JSON request')
    products.add_argument('--request', required=True, type=Path)
    products.set_defaults(func=main)
