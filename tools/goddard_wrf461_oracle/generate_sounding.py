"""Emit the pinned WRF sounding DATA words without arithmetic changes."""
from pathlib import Path
import re,sys
source=(Path(sys.argv[1])/'phys/module_ra_goddard.F').read_text()
records=[]
for match in re.finditer(r'data\s*\(\(mcdat\(ilev,(\d+),ifld\).*?/(.*?)/',source,re.I|re.S):
    values=re.findall(r'[+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[eEdD][+-]?\d+)?',match[2].replace('&',''))
    if len(values)!=66:raise ValueError(len(values))
    records.extend(v.lower().replace('d','e')+('f' if '.' in v or 'e' in v.lower() else '.f') for v in values)
if len(records)!=1980:raise ValueError(len(records))
Path(sys.argv[2]).write_text('// WRF module_ra_goddard.F:366-734. climatology, level, field.\n__device__ const float gdd_sounding[10][33][6]={'+','.join(records)+'};\n')
