from pathlib import Path
import re,json
r=Path('tools/uwpbl_wrf471_oracle/libm64');h=Path('gpuwm/core/kernels/glibc_flt64.cuh').read_text().splitlines()
maps={
'exp':[
 ('y = __dmul_rn(0x1p1009','overflow scaled result',['a0d0a']),
 ('kd = __fma_rn','multiply and Shift',['a0b81']),
 ('r = __fma_rn(kd','two reduction steps',['a0baa','a0bb9']),
 ('tmp = __fma_rn','polynomial and tail',['a0bc9','a0bea','a0bf3','a0bfc']),
 ('return __fma_rn(scale','normal result',['a0c0a'])],
'log':[
 ('y = __fma_rn(r3,__fma','near-one polynomial',['a12eb','a12f4','a1309','a1312','a131b','a1328','a1331','a133a','a1347']),
 ('double rhi = __fma','near-one split',['a1350','a1355']),
 ('hi = __fma','near-one high result',['a136e']),
 ('lo = __fma_rn(__dmul_rn(rhi','near-one low result',['a137b']),
 ('lo = __fma_rn(__dmul_rn(__log','near-one low correction',['a1384']),
 ('y = __fma_rn(r3,y,lo)','near-one final correction',['a1389']),
 ('r = __fma_rn(z','table reduction',['a126f']),
 ('w = __fma_rn','high logarithm scale',['a1264']),
 ('lo = __fma_rn(kd','low logarithm scale',['a128e']),
 ('y = __dadd_rn(__fma','general polynomial',['a1275','a129b','a12a4','a12ad','a12b2'])],
'pow':[
 ('r = __fma_rn(z','log table reduction',['a1921']),
 ('t1 = __fma_rn','high log scale',['a190a']),
 ('lo1 = __fma_rn','low log scale',['a1910']),
 ('lo3 = __fma_rn','log residual',['a1956']),
 ('p = __fma_rn','log polynomial',['a192f','a1938','a195f','a196c','a1975']),
 ('lo = __fma_rn(ar3','log polynomial low accumulation',['a1988']),
 ('y = __fma_rn(scale','overflow result',['a1d2a']),
 ('kd = __fma_rn','multiply and Shift',['a19e3']),
 ('r = __fma_rn(kd','two exp reduction steps',['a19f5','a1a05']),
 ('tmp = __fma_rn','exp polynomial and tail',['a1a2c','a1a43','a1a4c','a1a55']),
 ('return __fma_rn(scale','normal result',['a1a67']),
 ('elo = __fma_rn','product residual and tail',['a19ba','a19d2'])]}
text=['Each row covers every explicit FMA on that header line. Addresses are offsets in the installed libm ELF. Multiple calls on one line appear as multiple addresses.','', '| Function | Header line | Calculation | Installed instructions | Evidence |','|---|---:|---|---|---|']
for name,rows in maps.items():
 a=next(i for i,l in enumerate(h) if 'namespace g64_'+name+' {' in l);b=next((i for i in range(a+1,len(h)) if 'namespace g64_' in h[i]),len(h))
 covered=set();od=(r/(name+'.objdump.txt')).read_text().splitlines()
 for pat,description,addresses in rows:
  hits=[i for i in range(a,b) if h[i].strip().startswith(pat)];assert len(hits)==1,(name,pat,hits)
  ln=hits[0]+1;assert h[ln-1].count('__fma_rn')==len(addresses),(name,ln)
  covered.add(ln)
  refs=[]
  for addr in addresses:
   n=next(i+1 for i,l in enumerate(od) if re.match(r'\s*'+addr+':',l));refs.append(f'{name}.objdump.txt:{n}')
  text.append(f'| {name} | glibc_flt64.cuh:{ln} | {description} | '+', '.join('0x'+v for v in addresses)+' | '+', '.join(refs)+' |')
 actual={i+1 for i in range(a,b) if '__fma_rn' in h[i]};assert actual==covered,(name,actual-covered)
text+=['','CORE-MATH FMA sites use the upstream explicit fma calls. They are not matched to glibc cos or acos instructions. Every other floating-point operation in the function bodies is pinned separately.','', '| CORE-MATH function | Header lines containing explicit FMA |','|---|---|']
for name in ['cos','acos']:
 a=next(i for i,l in enumerate(h) if 'namespace g64_'+name+' {' in l);b=next((i for i in range(a+1,len(h)) if 'namespace g64_' in h[i]),len(h))
 text.append('| '+name+' | '+', '.join(str(i+1) for i in range(a,b) if '__fma_rn' in h[i])+' |')
(r/'FMA-SITES.md').write_text('\n'.join(text)+'\n')
print('FMA mapping complete')
