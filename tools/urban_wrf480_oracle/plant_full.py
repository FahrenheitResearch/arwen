"""Plant exactly one bit in the changed vegetated-ground water-flux term."""
from pathlib import Path
import argparse,json,shutil


def main():
    p=argparse.ArgumentParser();p.add_argument('listing',type=Path);p.add_argument('mutant',type=Path);a=p.parse_args()
    import re
    text=a.listing.read_text();pattern=r'(  // source line 2302\n  flxhumvg=[^\n]+\n)'
    if len(re.findall(pattern,text))!=1:raise ValueError('water-flux term anchor moved')
    changed=re.sub(pattern,r'\1  flxhumvg=__uint_as_float(__float_as_uint(flxhumvg)^1u);\n',text)
    a.mutant.write_text(changed)
    for suffix in ('.json','.bridge.cpp'):shutil.copyfile(a.listing.with_suffix(suffix),a.mutant.with_suffix(suffix))
    a.mutant.with_suffix('.mutation.json').write_text(json.dumps({'xor':1,'changed_term':'FLXHUMVG vegetated-ground water-flux feedback','source_line':2302},indent=2))


if __name__=='__main__':main()
