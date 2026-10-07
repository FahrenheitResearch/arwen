"""Byte comparisons between compiled native READs and the Rust f32 ABI."""
from pathlib import Path
import argparse,json,hashlib,subprocess
from types import SimpleNamespace
from gpuwm.static.sfire import read_fire_ideal_inputs,fire_ideal_input_requests,read_native_landuse_table

def main():
 p=argparse.ArgumentParser();p.add_argument('--build',type=Path,required=True);p.add_argument('--landuse',type=Path,required=True);a=p.parse_args()
 fixtures=Path(__file__).parent/'fixtures';out=a.build.resolve();out.mkdir(parents=True,exist_ok=True)
 rows=[]
 for name,ni,nj in (('array-special',3,2),('array-split',2,3)):
  source=(fixtures/(name+'.txt')).resolve();prefix=out/name
  subprocess.run([str(out/'run'),'array',source.name,str(prefix),str(ni),str(nj)],cwd=fixtures,check=True)
  actual=read_fire_ideal_inputs(fixtures,[dict(name='CONTROL',filename=source.name,ni=ni,nj=nj)])['fields']['CONTROL']
  expected=prefix.with_suffix('.f32').read_bytes();payload=actual.tobytes(order='C')
  assert actual.dtype.name=='float32' and actual.shape==(nj,ni) and payload==expected,name
  rows.append(dict(name=name,words=len(payload)//4,status='bit-identical',maximum_ulp=0,sha256=hashlib.sha256(payload).hexdigest()))
 prefix=out/'sounding';source=(fixtures/'input_sounding').resolve()
 subprocess.run([str(out/'run'),'sounding',str(source),str(prefix)],cwd=fixtures,check=True)
 actual=read_fire_ideal_inputs(fixtures,sounding=source)
 assert actual['_metadata']['sounding_levels']==int((out/'sounding.levels').read_text())==3
 for name,array in actual['sounding'].items():
  payload=array.tobytes(order='C');expected=(out/f'sounding.{name}.f32').read_bytes()
  assert array.dtype.name=='float32' and payload==expected,name
  rows.append(dict(name='sounding-'+name,words=len(payload)//4,status='bit-identical',maximum_ulp=0,sha256=hashlib.sha256(payload).hexdigest()))
 try:read_fire_ideal_inputs(fixtures,[dict(name='CONTROL',filename='array-special.txt',ni=2,nj=3)])
 except ValueError as error:geometry_refusal=str(error)
 else:raise AssertionError('Wrong producer dimensions were accepted')
 overflow=out/'overflow-sounding.txt';overflow.write_text('1000 300 0\n'+'0 300 0 1 2\n'*1001,newline='\n')
 try:read_fire_ideal_inputs(out,sounding=overflow)
 except ValueError as error:overflow_refusal=str(error)
 else:raise AssertionError('Native sounding overflow was accepted')
 table,metadata=read_native_landuse_table(a.landuse,'USGS')
 assert table.dtype.name=='float32' and table.shape==(metadata['luseas'],metadata['lucats'],7) and metadata['found']
 absent,absent_metadata=read_native_landuse_table(a.landuse,'ABSENT')
 assert not absent_metadata['found'] and absent.shape==(0,0,7)
 cfg=SimpleNamespace(nx=3,ny=2,sr_x=2,sr_y=3,sfc_full_init=True,fire_mountain_type=0,
  fire_read_lu=True,fire_read_tsk=True,fire_read_tmn=True,fire_read_atm_ht=True,fire_read_fire_ht=True,fire_read_fire_grad=True,
  fire_fuel_read=2,fire_fmc_read=2)
 requests=fire_ideal_input_requests(cfg);assert len(requests)==9
 cfg.sfc_full_init=False;cfg.fire_mountain_type=1;assert len(fire_ideal_input_requests(cfg))==2
 receipt=dict(schema='sfire-native-initializer-input-receipt-v1',scope='Compiled original native formatted READs versus Rust host-f32 parsing, with metadata/geometry/table controls; no new atmospheric forecast qualification',
  compiler=(out/'compiler.txt').read_text().strip(),native_source=json.loads((out/'native-source.json').read_text()),rows=rows,
  checks=len(rows),words=sum(row['words'] for row in rows),different_words=0,maximum_ulp=0,
  geometry_refusal=geometry_refusal,sounding_overflow_correction=overflow_refusal,
  landuse=dict(metadata=metadata,shape=table.shape,raw_f32_sha256=hashlib.sha256(table.tobytes(order='C')).hexdigest(),native_physical_consumers='Qualified separately by the complete original landuse_init control'),
  native_fixed_file_requests=requests)
 (out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))
if __name__=='__main__':main()
