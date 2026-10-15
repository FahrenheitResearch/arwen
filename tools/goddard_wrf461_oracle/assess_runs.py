"""Assess real GPU band outputs, optics receipts and planted flux mutations."""
from pathlib import Path
import argparse,gzip,json,hashlib
from check_columns import read_columns,compare_fields

def main():
 p=argparse.ArgumentParser();p.add_argument('--runs',type=Path,required=True);a=p.parse_args()
 reference_path=a.runs/'defined-reference.bin'
 raw=gzip.decompress((Path(__file__).parent/'fixtures/defined-reference.bin.gz').read_bytes())
 reference_path.write_bytes(raw)
 reference,_=read_columns(reference_path)
 report={'column_stage_pass':True,'full_port_complete':False,
  'reference_sha256':hashlib.sha256(raw).hexdigest(),
  'declared_divergence':'WRF unassigned taux replaced by visible-band cloud depth',
  'bands':{},'mutations':{},'clouds':{}}
 for mode in ('strict','default'):
  candidate,kind=read_columns(a.runs/f'gpu-{mode}.bin')
  passed,fields=compare_fields(reference,candidate,mode=='strict')
  report['bands'][mode]={'pass':passed,'solver_bits':kind,'fields':fields}
  report['column_stage_pass'] &= passed
  cloud=json.loads((a.runs/f'cloud-{mode}.json').read_text())
  report['clouds'][mode]=cloud
  report['column_stage_pass'] &= cloud['pass']
 for band,field in [('sw','sw_net'),('lw','lw_net')]:
  candidate,_=read_columns(a.runs/f'mutant-{band}.bin')
  passed,fields=compare_fields(reference,candidate,True)
  changed={name:v for name,v in fields.items() if v['different_words']}
  caught=(not passed and set(changed)=={field} and changed[field]['max_ulp']==1
          and changed[field]['different_words']==4)
  report['mutations'][band]={'caught':caught,'check_pass':passed,'changed':changed}
  report['column_stage_pass'] &= caught
 (a.runs/'ASSESSMENT.json').write_text(json.dumps(report,indent=2)+'\n')
 print('column_stage_pass',report['column_stage_pass'],'full_port_complete',False)
 for mode in ('strict','default'):
  print(mode,'bands',report['bands'][mode]['pass'],'clouds',report['clouds'][mode]['pass'])
 for band in ('sw','lw'):print(band,'mutation caught',report['mutations'][band]['caught'])
 raise SystemExit(0 if report['column_stage_pass'] else 1)
if __name__=='__main__':main()
