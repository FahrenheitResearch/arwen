"""Isolate defined entry-state corrections from the pristine 4.8 source.

The pristine scheme stays untouched. This produces a separate candidate
reference and its exact diff. A signaling-NaN check must still pass; blanket
zero initialization is not an exactness claim.
"""
from pathlib import Path
import argparse,difflib,hashlib,json

PIN='1be801c3e11aad59be8e0e23612bbe71ba4fa29cefc930520fb4c6e2e1169f6a'


def main():
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    raw=a.source.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=PIN:raise ValueError('source pin mismatch')
    text=raw.decode();changes=[]
    text=text.replace('   if(ahoption==1) then', '   if(ahoption==1.or.iri_scheme==1) then',1)
    changes.append({'cause':'irrigation uses tloc even with anthropogenic heat disabled; use the existing shared source clock formula','source_lines':[739,949]})
    anchor='   TVGP   = TVG'
    if text.count(anchor)!=1:raise ValueError('entry anchor moved')
    initial='''   SROOT = 0.0
   ETR = 0.0
   ETG = 0.0
   TGE = (1.0 - FVG) * TG + FVG * TVG
   HVG = 0.0
   ELEVG = 0.0
   HT = 0.0
   ELET = 0.0
   SLEAF = 0.0
   SVG1 = 0.0
   SVG2 = 0.0
'''
    text=text.replace(anchor,initial+anchor)
    changes.append({'cause':'SROOTP copied before SROOT is defined; TGEP copied before TGE is defined; inactive vegetation diagnostics have no assigned flux',
                    'source_lines':[1008,1009,1013,1826,1868,2303]})
    # The bottom moisture layer is a carried boundary value, not one of the
    # three solved SSTEP unknowns. Give the unassigned output an explicit
    # value in the candidate reference, without changing solved layers.
    anchor='      WPLUS = 0.0\n      RUNOFF3 = 0.'
    if text.count(anchor)!=1:raise ValueError('bottom-layer anchor moved')
    text=text.replace(anchor,'      SMC(NSOIL) = SMCP(NSOIL)\n'+anchor)
    changes.append({'cause':'SSTEP assigns only 1:NSOIL-1 but declares SMC as output; preserve the bottom boundary slot','source_lines':[4639,4673,4690]})
    a.output.write_text(text)
    a.output.with_suffix('.diff').write_text(''.join(difflib.unified_diff(raw.decode().splitlines(True),text.splitlines(True),fromfile='pristine-v4.8.0',tofile='defined-entry-state')))
    a.output.with_suffix('.json').write_text(json.dumps({'pristine_sha256':PIN,'candidate_sha256':hashlib.sha256(text.encode()).hexdigest(),'corrections':changes},indent=2))


if __name__=='__main__':main()
