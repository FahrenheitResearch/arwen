"""Record 80/129-layer SW controls from an already built unmodified WRF oracle.

Usage: sw_make_tall.py BUILD_DIR OUT_NPZ
Build first with sw_build.sh WRF_SOURCE_ROOT BUILD_DIR. The driver independently
checks its recorded stages against the untouched WRF RRTMG_SWRAD call.
"""
from pathlib import Path
import subprocess
import sys

import numpy as np

import sw_make_synthetic as synthetic
from sw_dumpio import read_swd


def main(build_dir, output):
    build_dir=Path(build_dir).resolve()
    records={}
    for identifier,nz in ((201,79),(202,128)):
        synthetic.NZ=nz
        column=synthetic.base_column()
        # The cloud spans the former launch boundary, so an unprocessed tail
        # changes both all-sky flux and heating; the clear-sky arm is checked too.
        cloud=slice(12,nz-5)
        column['qc'][cloud]=1.0e-4
        column['cldfra'][cloud]=0.7
        column['re_cloud'][cloud]=9.0e-6
        lines=[]
        synthetic.emit(lines,identifier,column,8,(1,1,1),1)
        source=build_dir/f'columns_tall_{nz}.txt'
        source.write_text(f'1 {nz}\n'+'\n'.join(lines)+'\n',encoding='ascii')
        result=build_dir/f'fixtures_tall_{nz}.swd'
        subprocess.run([str(build_dir/'sw_fixture_driver'),str(source),str(result)],
                       cwd=build_dir,check=True)
        new=read_swd(result)
        assert not records.keys() & new.keys()
        records.update(new)
    np.savez_compressed(output,**records)
    print(f'{output}: {len(records)} independently recorded WRF values/arrays')


if __name__=='__main__':
    main(*sys.argv[1:])
