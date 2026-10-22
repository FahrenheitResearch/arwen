"""Fetch explicitly selected public case inputs; no field decoding in Python.

Run from an input directory, then use case.json. ERA5 is extracted separately
by the existing native rw_zarr door using forcing-request.json.
"""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from urllib.request import urlopen

ETOPO = ('https://www.ngdc.noaa.gov/thredds/ncss/grid/global/ETOPO2022/'
         '30s/30s_surface_elev_netcdf/ETOPO_2022_v1_30s_N90W180_surface.nc'
         '?var=z&north=30.02&south=22.98&west=-87.02&east=-78.98&horizStride=1&accept=netcdf')
STATIONS = ['8725520', '8725110', '8726520']


def download(item):
    name, url = item
    with urlopen(url, timeout=180) as response:
        Path(name).write_bytes(response.read())
    return dict(path=name, url=url, bytes=Path(name).stat().st_size)


def main():
    jobs = [('etopo.nc', ETOPO)]
    for station in STATIONS:
        jobs.append((f'{station}-metadata.json',
                     f'https://api.tidesandcurrents.noaa.gov/mdapi/prod/webapi/stations/{station}.json'))
        base = ('https://api.tidesandcurrents.noaa.gov/api/prod/datagetter?'
                f'begin_date=20220927&end_date=20220930&datum=MSL&station={station}'
                '&time_zone=gmt&units=metric&format=json&application=gpuwm_surge')
        jobs += [(f'{station}-observations.json', base + '&product=hourly_height'),
                 (f'{station}-predictions.json', base + '&product=predictions&interval=h')]
    with ThreadPoolExecutor(max_workers=6) as pool:
        receipts = list(pool.map(download, jobs))
    Path('download-receipt.json').write_text(json.dumps(receipts, indent=2) + '\n')
    # Station locations are metadata, not transformed model fields.
    gauges = []
    for station in STATIONS:
        row = json.loads(Path(f'{station}-metadata.json').read_text())['stations'][0]
        gauges.append(dict(station=station, latitude=float(row['lat']), longitude=float(row['lng']),
                           observations=f'{station}-observations.json',
                           predictions=f'{station}-predictions.json'))
    case = dict(grid=dict(nx=800, ny=780, west=-87, east=-79, south=23, north=30),
                bathymetry=dict(path='etopo.nc', elevation='z', latitude='lat', longitude='lon',
                                vertical_datum='EGM2008 geoid; approximate MSL, no NAVD88 conversion',
                                source=ETOPO, license='CC0-1.0; NOAA US government public domain'),
                forcing=dict(path='era5.nc', latitude='latitude', longitude='longitude', time='time',
                             u10='U10', v10='V10', pressure='MSLP',
                             source='ERA5 / Copernicus Climate Change Service, ECMWF; Google ARCO archive',
                             license='CC-BY-4.0; explicitly selected research input'),
                duration_seconds=259200, output_interval_seconds=21600, gauges=gauges)
    Path('case.json').write_text(json.dumps(case, indent=2) + '\n')


if __name__ == '__main__':
    main()
