from pathlib import Path
import csv, itertools, json, numpy as np
import os
L=Path(os.environ['SFCLAYREV_CHECK_ROOT'])
import _sfclayrev_oracle as h
base=h.load_fixture()
cols=[]
def add(label, **kw):
    c={k:float(base.inputs[k][0]) for k in h.INPUT_FIELDS}; c.update(kw)
    if 'psfc' not in kw: c['psfc']=c['p']+c['dz8w']*0.5*c['p']/(287*c['t'])*9.81
    cols.append((label,{k:np.float32(v) for k,v in c.items()}))
rng=np.random.default_rng(7302026)
for i in range(512):
    water=i%3==0; t=rng.uniform(190,330); p=rng.uniform(35000,105000); dz=10**rng.uniform(0.5,2.7)
    add('new-mixed-sweep', t=t,tsk=t+rng.uniform(-20,25),p=p,dz8w=dz,qv=rng.uniform(0,.03),qsfc=rng.uniform(0,.03),u=rng.uniform(-90,90),v=rng.uniform(-50,50),xland=2 if water else 1,lakemask=int(water and i%2==0),mavail=rng.uniform(0,1),znt=10**rng.uniform(-7,-2) if water else 10**rng.uniform(-5,.5),ust=10**rng.uniform(-5,.8),ustm=10**rng.uniform(-5,.8),mol=rng.uniform(-3,3),hfx=rng.uniform(-500,800),qfx=rng.uniform(-.0005,.001),pblh=10**rng.uniform(0,4),zol=rng.uniform(-30,30),dx=float(rng.choice([.001,250,4999.9995,5000,5000.0005,5001,100000])))
for t,p,water in itertools.product([190.,220.,273.15,300.,330.],[35000.,55000.,101325.],[False,True]):
    es=np.float32(.6112)*np.exp(np.float32(17.67)*(np.float32(t)-np.float32(273.15))/(np.float32(t)-np.float32(29.65)))
    sat=float(np.float32(287./461.6)*es/(np.float32(p/1000)-es))
    for moisture in [0.,sat,sat*1.05]:
        for contrast in [-10.,0.,10.]:
            add('cold-warm-dry-saturated',t=t,tsk=t+contrast,p=p,qv=moisture,qsfc=0.,xland=2 if water else 1,lakemask=0,mavail=1.,znt=.0001 if water else .1,mol=0.,hfx=0.,qfx=0.)
for threshold in [0.,.001,.1,1.06,3.,10.]:
    for word in [np.nextafter(np.float32(threshold),np.float32(0)),np.float32(threshold),np.nextafter(np.float32(threshold),np.float32(np.inf))]:
        for water in [False,True]:
            add('ust-switch-boundary',ust=word,ustm=word,xland=2 if water else 1,znt=.0001 if water else .1,tsk=295.,t=290.)
for dx in [.001,1.,250.,4999.9995,5000.,5000.0005,5001.,100000.,1000000.]:
    for tsk in [280.,290.,310.]:
        add('dx-gust-threshold',dx=dx,t=290.,tsk=tsk,u=0.,v=-0.,hfx=0.,qfx=0.,mol=0.)
for dz in [1.,2.,3.,4.,20.,200.,2000.]:
    for znt in [1e-8,.0001,.1,1.,10.]:
        add('thin-thick-roughness',dz8w=dz,znt=znt,t=280.,tsk=270.,mol=.5,ust=.05,u=.1,v=0.)
out=L/'extra'; out.mkdir(exist_ok=True)
with (out/'sfclayrev-inputs.hex').open('w') as f:
    f.write('# case '+' '.join(h.INPUT_FIELDS)+'\n')
    for i,(label,c) in enumerate(cols,1): f.write(str(i)+' '+' '.join(f'{int(c[k].view(np.uint32)):08X}' for k in h.INPUT_FIELDS)+'\n')
with (out/'sfclayrev-cases.csv').open('w') as f:
    w=csv.writer(f,lineterminator='\n'); w.writerow(['case','label',*h.INPUT_FIELDS])
    for i,(label,c) in enumerate(cols,1): w.writerow([i,label,*[float(c[k]) for k in h.INPUT_FIELDS]])
arms=list(h.ARMS)+[x for x in itertools.product([0,1],range(3),range(3)) if x not in h.ARMS]
meta=dict(columns=len(cols),seed=7302026,steps=9,arms=arms,groups={label:sum(l==label for l,c in cols) for label,c in cols})
(L/'results/extra-design.json').write_text(json.dumps(meta,indent=1))
driver=(L/'src/tools/sfclayrev_wrf461_oracle/run_sfclayrev.F90').read_text()
start=driver.index('  ! arm:'); end=driver.index('  character(len=512)',start)
driver=driver[:start]+'  integer, parameter :: arms(3, narms) = reshape( &\n       (/ '+', &\n          '.join(', '.join(map(str,x)) for x in arms)+' /), &\n       (/ 3, narms /))\n\n'+driver[end:]
driver=driver.replace('nsteps = 3, narms = 6','nsteps = 9, narms = 18').replace("'(I1,3(1X,I1),1X,I1,1X,I4,34(1X,Z8.8))'","'(I2,3(1X,I1),1X,I2,1X,I4,34(1X,Z8.8))'")
(L/'extra-driver.F90').write_text(driver)
seeded=(L/'src/tools/sfclayrev_wrf461_oracle/run_sfclayrev.F90').read_text().replace('     ! LH is intent(out)', '     chs = 0.125; chs2 = 0.25; cqs2 = 0.375; flhc = 0.5; flqc = 0.001\n     ! LH is intent(out)')
(L/'seeded-driver.F90').write_text(seeded)
print(json.dumps(meta))
