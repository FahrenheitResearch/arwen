"""Record SW columns with no layer above the troposphere switch, from an
already built unmodified WRF oracle.

Usage: sw_make_shallow.py BUILD_DIR OUT_NPZ
Build first with sw_build.sh WRF_SOURCE_ROOT BUILD_DIR.  The driver checks
its recorded stages against the untouched WRF RRTMG_SWRAD call before it
writes a fixture.

Why these columns exist.  setcoef_sw counts a layer as tropospheric while
log(pavel) > 4.56 (pavel above ~95.6 hPa).  The option-4 driver adds one
layer above the model top at half its pressure, so a model top below
~191 hPa leaves laytrop == nlayers: the upper-atmosphere loop of every
taumol band is empty.  Bands 16, 17, 27, 28 and 29 set sfluxzen only in
that loop, so WRF keeps the zero taumol_sw writes on entry ("to prevent
junk values when nlayers = laytrop").  The real and synthetic decks all
reach 10 hPa or higher and never take this path.

Cases (all 20 model layers, mp_physics 8, has_reqc/reqi/reqs = 1, so the
three batch together):
  c00301  clear column, psfc 970 hPa, model top 250 hPa  (laytrop = nlayers)
  c00302  liquid + ice cloud, psfc 1000 hPa, top 300 hPa (laytrop = nlayers)
  c00303  the same cloud, psfc 970 hPa, top 50 hPa: the control, whose
          upper loop runs (laytrop < nlayers)
"""
from pathlib import Path
import subprocess
import sys

import numpy as np

import sw_make_synthetic as synthetic
from sw_dumpio import read_swd

NZ = 20


def shallow_column(psfc, ptop):
    """sw_make_synthetic.base_column with the top at ``ptop`` Pa."""
    rd, g0, cp = synthetic.RD, synthetic.G0, synthetic.CP
    p8w = np.geomspace(psfc, ptop, NZ + 1)
    p3d = np.sqrt(p8w[:-1] * p8w[1:])
    t0, lapse, t_min = 288.0, 6.5e-3, 195.0
    z = -rd * 260.0 / g0 * np.log(p3d / psfc)
    t3d = np.maximum(t0 - lapse * z, t_min)
    zw = -rd * 260.0 / g0 * np.log(p8w / psfc)
    t8w = np.maximum(t0 - lapse * zw, t_min)
    rh = np.clip(0.7 - 0.5 * (z / 12000.0), 0.05, 0.95)
    es = 611.2 * np.exp(17.67 * (t3d - 273.15) / (t3d - 29.65))
    o3 = np.interp(np.log(p3d), np.log([100000, 20000, 5000, 1000]),
                   [6e-8, 1.5e-7, 2e-6, 6e-6])
    return dict(p3d=p3d, t3d=t3d, dz8w=zw[1:] - zw[:-1],
                pi3d=(p3d / 1e5) ** (rd / cp), rho3d=p3d / (rd * t3d),
                qv=np.clip(0.622 * rh * es / (p3d - rh * es), 2e-7, 0.02),
                qc=np.zeros(NZ), qr=np.zeros(NZ), qi=np.zeros(NZ),
                qs=np.zeros(NZ), qg=np.zeros(NZ), cldfra=np.zeros(NZ),
                re_cloud=np.zeros(NZ), re_ice=np.zeros(NZ),
                re_snow=np.zeros(NZ), o31d=o3, p8w=p8w, t8w=t8w,
                coszen=0.9, albedo=0.18, tsk=t0 + 1.5, xland=1.0, xice=0.0,
                snow=0.0, obscur=0.0, xlat=39.0, xlong=-98.0)


def cloudy(column):
    column['qc'][3:10] = 4.0e-4
    column['re_cloud'][3:10] = 9.0e-6
    column['qi'][12:18] = 1.0e-4
    column['re_ice'][12:18] = 4.0e-5
    column['cldfra'][3:18] = 0.8
    return column


def main(build_dir, output):
    build_dir = Path(build_dir).resolve()
    synthetic.NZ = NZ
    cases = ((301, shallow_column(97000.0, 25000.0)),
             (302, cloudy(shallow_column(100000.0, 30000.0))),
             (303, cloudy(shallow_column(97000.0, 5000.0))))
    lines = []
    for identifier, column in cases:
        synthetic.emit(lines, identifier, column, 8, (1, 1, 1), 1)
    source = build_dir / 'columns_shallow.txt'
    source.write_text(f'{len(cases)} {NZ}\n' + '\n'.join(lines) + '\n',
                      encoding='ascii')
    result = build_dir / 'fixtures_shallow.swd'
    subprocess.run([str(build_dir / 'sw_fixture_driver'), str(source),
                    str(result)], cwd=build_dir, check=True)
    records = read_swd(result)
    for identifier, _column in cases:
        case = f'c{identifier:05d}'
        nlayers = int(records[f'{case}/inatm/nlayers'])
        laytrop = int(records[f'{case}/setcoef/laytrop'])
        assert (laytrop == nlayers) == (identifier != 303), (
            case, laytrop, nlayers)
    np.savez_compressed(output, **records)
    print(f'{output}: {len(records)} independently recorded WRF values/arrays')


if __name__ == '__main__':
    main(*sys.argv[1:])
