"""Pin an observation replay grid from recorded columns and Rust projection.

Recorded WRF horizontal rounding can differ from the engine's projection.
This makes a distinct bench target, preserving every PH, PHB and HGT word.
It does not relax the production grid admission check.
"""
import argparse
import hashlib
import json
from pathlib import Path

from gpuwm import netcdf_bridge
from gpuwm.io.classic_product import ClassicProduct
from gpuwm.obs.target_grid import TargetGrid
from gpuwm.static.projection import WPS_MAP_PROJ_NAMES, projection_class


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def build(source, destination):
    from gpuwm.static import rust_bridge
    if rust_bridge.route('observation replay target projection') is None:
        raise ValueError('native projection is unavailable; a Python projection cannot qualify this replay target')
    if destination.exists():
        raise ValueError('replay target already exists; its pinned columns cannot be overwritten')
    with netcdf_bridge.open_dataset(source) as dataset:
        keys = ('MAP_PROJ', 'TRUELAT1', 'TRUELAT2', 'STAND_LON', 'CEN_LAT', 'CEN_LON', 'DX', 'DY')
        attrs = {key: dataset.getncattr(key) for key in keys}
        values = {key: dataset.variables[key][0] for key in ('PH', 'PHB', 'HGT')}
    ny, nx = values['HGT'].shape
    projection = projection_class(WPS_MAP_PROJ_NAMES[int(attrs['MAP_PROJ'])])(
        ref_lat=float(attrs['CEN_LAT']), ref_lon=float(attrs['CEN_LON']),
        truelat1=float(attrs['TRUELAT1']), truelat2=float(attrs['TRUELAT2']),
        stand_lon=float(attrs['STAND_LON']), dx=float(attrs['DX']), dy=float(attrs['DY']),
        e_we=nx + 1, e_sn=ny + 1)
    lat, lon = projection.latlon_mass()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with ClassicProduct(destination) as product:
        for name, count in (('Time', 1), ('south_north', ny), ('west_east', nx),
                            ('bottom_top_stag', values['PH'].shape[0])):
            product.createDimension(name, count)
        product.setncatts(attrs)
        for name, value in {**values, 'XLAT': lat, 'XLONG': lon}.items():
            dims = ('Time', 'bottom_top_stag', 'south_north', 'west_east') if value.ndim == 3 else ('Time', 'south_north', 'west_east')
            product.createVariable(name, value.dtype, dims)[:] = value[None]
    grid = TargetGrid.from_wrfout(destination)
    with netcdf_bridge.open_dataset(destination) as dataset:
        for name, value in values.items():
            if dataset.variables[name][0].tobytes() != value.tobytes():
                raise ValueError(f'{name} words changed; the replay would no longer preserve recorded vertical columns')
    return {'schema': 'gpuwm-obs.replay-target.v1', 'source': str(source), 'source_sha256': sha(source),
        'target': str(destination), 'target_sha256': sha(destination), 'grid_identity_sha256': grid.identity_sha256(),
        'preserved_column_words': {name: hashlib.sha256(value.tobytes()).hexdigest() for name, value in values.items()},
        'shape': [grid.nz, grid.ny, grid.nx], 'horizontal_source': 'Rust static-fields projection from recorded attributes',
        'scope': 'distinct observation replay target; production grid admission and forecast fields unchanged'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('destination', type=Path)
    parser.add_argument('--receipt', type=Path, required=True)
    args = parser.parse_args()
    receipt = build(args.source.resolve(), args.destination.resolve())
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + '\n')
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
