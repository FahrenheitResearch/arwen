"""Stage a five-class cold-start probe from a shared warm-rain WRF input pair.

All NetCDF decoding and writing use the Rust bridges. Existing field words
are checked after writing. Added ice and snow initial/boundary fields are zero.
This is a scheme-change smoke input, not a new real.exe preparation or oracle.
"""
from pathlib import Path
import argparse
import hashlib
import json
import re
import numpy as np
from gpuwm import netcdf_bridge
from gpuwm.io.classic_tape import ClassicTape

def digest(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()

def stage(source, output, boundary):
    unchanged = {}
    added = []
    with netcdf_bridge.open_dataset(source) as src, ClassicTape(output) as dst:
        for name, dim in src.dimensions.items():
            dst.createDimension(name, None if dim.isunlimited() else len(dim))
        attrs = {name: src.getncattr(name) for name in src.ncattrs()}
        if 'MP_PHYSICS' in attrs:
            assert int(attrs['MP_PHYSICS']) == 1
            attrs['MP_PHYSICS'] = np.int32(4)
        dst.setncatts(attrs)
        values = {}
        for name, var in src.variables.items():
            dtype = np.dtype('S1') if var.is_character else var.dtype
            target = dst.createVariable(name, dtype, var.dimensions)
            target.setncatts(dict(var.attributes))
            values[name] = np.asarray(var[...], dtype=dtype)
            unchanged[name] = digest(values[name])
        templates = ([n for n in src.variables if n.startswith('QCLOUD_')]
                     if boundary else ['QCLOUD'])
        for original in templates:
            for species in ('QICE', 'QSNOW'):
                name = original.replace('QCLOUD', species, 1)
                assert name not in dst.variables
                template = src.variables[original]
                target = dst.createVariable(name, np.float32, template.dimensions)
                target.setncatts(dict(template.attributes))
                values[name] = np.zeros(values[original].shape, np.float32)
                added.append(name)
        for name, value in values.items():
            variable = dst.variables[name]
            record = bool(variable.dimensions and
                          src.dimensions[variable.dimensions[0]].isunlimited())
            if record:
                for index in range(value.shape[0]):
                    variable[index] = value[index]
            else:
                variable[0] = value
    with netcdf_bridge.open_dataset(output) as check:
        for name, expected in unchanged.items():
            var = check.variables[name]
            dtype = np.dtype('S1') if var.is_character else var.dtype
            assert digest(np.asarray(var[...], dtype=dtype)) == expected, name
        for name in added:
            assert np.count_nonzero(check.variables[name][...]) == 0, name
    return {'source': str(source), 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
            'unchanged_fields': unchanged, 'zero_added_fields': added,
            'output_sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
            'output_bytes': output.stat().st_size}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input_pair', type=Path)
    parser.add_argument('namelist', type=Path)
    parser.add_argument('out', type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    report = {name: stage(args.input_pair / name, args.out / name, name.startswith('wrfbdy'))
              for name in ('wrfinput_d01', 'wrfbdy_d01')}
    text = args.namelist.read_text()
    text, count = re.subn(r'(\bmp_physics\s*=\s*)1\b', r'\g<1>4', text)
    assert count == 1
    text, count = re.subn(r'(\bhistory_interval_s\s*=\s*)15\b', r'\g<1>300', text)
    assert count == 1
    (args.out / 'namelist.input').write_text(text)
    report['namelist_source_sha256'] = hashlib.sha256(args.namelist.read_bytes()).hexdigest()
    report['namelist_sha256'] = hashlib.sha256(text.encode()).hexdigest()
    report['scope'] = 'MP1 to MP4 cold addition of zero ice and snow; existing field words unchanged'
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    main()
