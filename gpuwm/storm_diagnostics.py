"""Orchestrate Rust storm diagnostics and the production weather renderer."""
import argparse
import json
from pathlib import Path
import subprocess


def register_cli(sub):
    from gpuwm.storm_detect import register_cli as register_detector
    register_detector(sub)
    parser = sub.add_parser('storm-diagnostics', help='named-height w, Earth-relative vertical vorticity, reflectivity and time series')
    parser.add_argument('--input-list', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--heights', default='500,1000,3000', help='geometric heights AGL in metres, comma separated')
    parser.add_argument('--gravity', type=float, required=True, help='gravity used by the producing model for PH/PHB')
    parser.add_argument('--jobs', type=int, default=8)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--label', default='Storm diagnostics')
    parser.set_defaults(func=main)


def render(source, heights, label):
    from gpuwm.rustwx import find_renderer, run_renderer_series
    files = sorted(source.glob('wrfout*'))
    products = ','.join(['var:wrf_storm_refl_max'] + [
        f'var:wrf_storm_{field}_{height:.0f}m'
        for height in heights for field in ('w', 'zeta')])
    pictures, notes, skipped = [], [], []
    for domain in sorted({p.name.split('_')[1] for p in files}):
        selected = [p for p in files if p.name.split('_')[1] == domain]
        rendered, comment, omitted = run_renderer_series(
            find_renderer(), selected, store_root=source / ('render-store-' + domain),
            out_dir=source / 'png' / domain, products=products, frames='all',
            width=1400, height=1000, source_label=label)
        for picture in rendered:
            product = next(name.removeprefix('var:') for name in products.split(',') if '_var_' + name.removeprefix('var:') in picture.stem)
            target = source / 'png' / domain / product / selected[0].name[11:21] / picture.name
            target.parent.mkdir(parents=True, exist_ok=True)
            picture.replace(target)
            pictures.append(str(target))
        notes.extend(comment)
        skipped.extend(omitted)
    (source / 'render-receipt.json').write_text(json.dumps(
        dict(pictures=pictures, notes=notes, skipped=skipped), indent=2) + '\n')
    if not pictures:
        raise ValueError('storm diagnostic renderer produced no pictures: ' + '; '.join(notes))
    return pictures


def main(args):
    from gpuwm.netcdf_bridge import find_netcdf_bin, netcdf_remedy
    binary = find_netcdf_bin()
    if binary is None:
        raise ValueError(netcdf_remedy())
    capabilities = subprocess.run([str(binary), '--abi'], check=True, capture_output=True, text=True).stdout
    if 'storm_diagnostics_v1' not in capabilities:
        raise ValueError('storm-diagnostics needs this release\'s rw_netcdf; ' + netcdf_remedy())
    if args.jobs < 1:
        raise ValueError('--jobs must be positive')
    files = [str(Path(line).resolve()) for line in args.input_list.read_text().splitlines() if line.strip()]
    result = subprocess.run([str(binary), 'storm-diagnostics', str(args.out.resolve()),
                             args.heights, str(args.gravity), str(args.jobs), *files], check=False)
    if result.returncode:
        return result.returncode
    if args.render:
        print('Rendered', len(render(args.out, [float(h) for h in args.heights.split(',')], args.label)), 'pictures')
    return 0
