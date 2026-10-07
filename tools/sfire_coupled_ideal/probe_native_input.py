"""Exercise the public Rust input reader on the unchanged native ideal file."""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path


def probe(reference, output):
    from tools.sfire_coupled_ideal.run_arwen import build_configuration
    from gpuwm.fortran_namelist import parse_namelist
    from gpuwm.ingest.wrfinput import read_wrfinput
    from gpuwm.ingest.wrfinput_sfire import SFIRE_INPUT_DIMENSIONS
    reference = Path(reference)
    cfg, _, _ = build_configuration(reference, 2)
    cfg = replace(cfg, fire_smoke=True)
    expected = dict(west_east=cfg.nx, west_east_stag=cfg.nx+1,
        south_north=cfg.ny, south_north_stag=cfg.ny+1,
        bottom_top=cfg.nz, bottom_top_stag=cfg.nz+1,
        soil_layers_stag=cfg.num_soil_layers)
    restored = read_wrfinput(reference/'wrfinput_d01', cfg=cfg,
        require_complete=False, expected_dimensions=expected)
    result = {'status':'PASS', 'source_sha256':hashlib.sha256(
        (reference/'wrfinput_d01').read_bytes()).hexdigest(),
        'expected_dimensions':expected, 'present_fire_inputs':{
            name:{'shape':list(restored.raw[name].shape),
                'dtype':str(restored.raw[name].dtype),
                'sha256':hashlib.sha256(restored.raw[name].tobytes()).hexdigest()}
            for name in SFIRE_INPUT_DIMENSIONS if name in restored.raw},
        'native_smoke':{'shape':list(restored.raw['fire_smoke'].shape),
            'min':float(restored.raw['fire_smoke'].min()),
            'max':float(restored.raw['fire_smoke'].max())},
        'native_tracer_opt':parse_namelist(reference/'namelist.output')['dynamics']['tracer_opt'],
        'require_complete':False,
        'reason':'official ideal initialization lacks inactive surface carriers required by the real-case complete reader',
        'mapped_count':len(restored.mapped_variables),
        'input_dispositions':dict(restored.surface_input_dispositions),
        'source_files':{name:hashlib.sha256(Path(name).read_bytes()).hexdigest()
            for name in ('gpuwm/ingest/wrfinput.py','gpuwm/ingest/wrfinput_sfire.py',
                         'gpuwm/core/chem_sfire.py')}}
    Path(output).write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
    print(json.dumps(result))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('reference',type=Path)
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    probe(args.reference,args.output)
