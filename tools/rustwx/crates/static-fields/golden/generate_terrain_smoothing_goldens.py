"""Generate tiny WPS float32 smoothing goldens without external data."""
from pathlib import Path
import json
import sys
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[4]))
from gpuwm.static.terrain_smoothing import TerrainSmoothing, smooth_terrain_reference

def write(path, a):
    a = np.asarray(a, dtype='<f8')
    path.write_bytes(b'GWARR1\x00\x00' + bytes([0,a.ndim]) +
        b''.join(int(d).to_bytes(8,'little') for d in a.shape) + a.tobytes())
    return path.name

def main():
    directory = HERE/'lane2'/'terrain_smoothing'
    directory.mkdir(parents=True, exist_ok=True)
    a = np.random.default_rng(27).uniform(-100,1500,(11,13))
    a[4:7,5:8] = 0
    rows = []
    for option, passes in [('none',0),('1-2-1',1),('1-2-1',3),('1-2-1',5),
                           ('smth-desmth',1),('smth-desmth',4),
                           ('smth-desmth_special',2),('smth-desmth_special',5)]:
        setting = TerrainSmoothing(option,passes)
        rows.append({**setting.echo(),'output':write(directory/f'{option}-{passes}.bin',smooth_terrain_reference(a,setting))})
    spec = {'input':write(directory/'input.bin',a),'settings':rows}
    (directory/'goldens.json').write_text(json.dumps(spec,indent=2)+'\n',encoding='utf-8')

def convert_wps_oracle():
    fixture = HERE.parents[4]/'tests'/'fixtures'/'wps_smooth_v460'/'wps_smooth_v460.npz'
    if not fixture.is_file():
        return
    directory = HERE/'lane2'/'wps_terrain_smoothing'
    directory.mkdir(parents=True, exist_ok=True)
    planes = []
    with np.load(fixture) as oracle:
        for name in oracle['plane_names']:
            name = str(name)
            rows = []
            for index, (code, passes) in enumerate(oracle['cases']):
                option = {1:'1-2-1',2:'smth-desmth',3:'smth-desmth_special'}[int(code)]
                rows.append({'smooth_option':option if passes else 'none',
                    'smooth_passes':int(passes), 'oracle_f32':True,
                    'output':write(directory/f'{name}-{code}-{passes}.bin', oracle['out_'+name][index])})
            planes.append({'input':write(directory/f'{name}-input.bin',oracle['in_'+name]),'settings':rows})
        (directory/'goldens.json').write_text(json.dumps({'planes':planes,'provenance':str(oracle['provenance'])},indent=2)+'\n',encoding='utf-8',newline='\n')

if __name__ == '__main__':
    main()
    convert_wps_oracle()
