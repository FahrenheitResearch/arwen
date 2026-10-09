"""New deterministic columns, outside the first lane's input draws."""
import copy,json,pathlib,sys,subprocess
import numpy as np

L=pathlib.Path('/work/pool-sfclay-noftz-mynn-mm5-eta')
sys.path.insert(0,str(L/'src/tools/myjsfc_wrf461_oracle'))
import make_cases as mc
import pack_fixture as pf

F=np.float32
rng=np.random.default_rng(10072026)
out=L/'extras'
out.mkdir(exist_ok=True)
sets=[]
meta=[]
def add(name,nz,it0,mutator,ncol=129,nsteps=8):
    cols,states=[],[]
    for i in range(ncol):
        c=mc._column(rng,mc.REGIMES[i%len(mc.REGIMES)],nz)
        s=mc._warm_state(rng,c,i)
        mutator(c,s,i)
        cols.append(c);states.append(s)
    mc._write(out/f'case_{name}.bin',cols,states,nz,nsteps,it0)
    sets.append(name)
    meta.append({'set':name,'columns':ncol,'levels':nz,'steps':nsteps,'it0':it0})
def unchanged(c,s,i): pass
add('new_seed_deep',128,900000,unchanged,nsteps=16)
add('late_two_level',2,2147483600,unchanged,nsteps=16)

def terrain(c,s,i):
    ht=[-430.,0.,8849.,9500.][i%4]
    scale=(101325*(1-2.25577e-5*ht)**5.25588)/c['pint'][0]
    c['ht']=ht;c['pint']*=scale;c['pmid']*=scale
    c['th']=c['t']*(1e5/c['pmid'])**mc.CAPA
    # Dry state keeps the prescribed pressure/temperature combination valid.
    c['qv'][:]=0.;c['qc'][:]=0.;s['qsfc']=s['qz0']=0.
    s['thz0']=c['tsk']*(1e5/c['pint'][0])**mc.CAPA
add('terrain_beyond_fixture',75,1,terrain)

def threshold(c,s,i):
    c['xland']=2.;c['mavail']=1.
    v=F([.225,.7,.1][i%3]); direction=F([-np.inf,np.inf,v][(i//3)%3])
    s['ustar']=float(np.nextafter(v,direction,dtype=F))
    c['q2'][:]=float(F(F(.2)*F(1.01)/F(2)))
    c['q2'][1]=float(np.nextafter(F(c['q2'][1]),F([-np.inf,np.inf][i%2]),dtype=F))
add('sea_and_tke_thresholds',50,2,threshold,nsteps=16)

def mask(c,s,i):
    c['xland']=float([np.nextafter(F(1.5),F(-np.inf)),F(1.5),np.nextafter(F(1.5),F(np.inf))][i%3])
    c['mavail']=[0.,1.,.5][(i//3)%3]
add('mask_boundary',25,8,mask)

def verythin(c,s,i):
    c['ht']=[0.,8849.,-430.][i%3]
    c['dz'][0]=[.02,.1,.5,4.,20.][(i//3)%5]
    c['qv'][:]=0.;c['qc'][:]=0.;s['qsfc']=s['qz0']=0.
add('thin_terrain_cancellation',2,1,verythin)

def extremes(c,s,i):
    c['t'][:]=[190.,220.,330.,340.][i%4]
    c['tsk']=c['t'][0]+[-35.,-10.,0.,35.][(i//4)%4]
    c['th']=c['t']*(1e5/c['pmid'])**mc.CAPA
    s['thz0']=c['tsk']*(1e5/c['pint'][0])**mc.CAPA
    c['u'][:]=[0.,.0001,80.,150.][(i//16)%4];c['v'][:]=0.
    c['qv'][:]=0.;c['qc'][:]=0.;s['qsfc']=s['qz0']=0.
    c['mavail']=[0.,1.][i%2]
    c['z0base']=c['znt']=[1e-6,.0001,3.,10.][i%4]
add('temperature_wind_roughness',60,1,extremes)

def neutral(c,s,i):
    c['qv'][:]=0.;c['qc'][:]=0.;s['qsfc']=s['qz0']=0.
    c['tsk']=float([np.nextafter(F(c['t'][0]),F(-np.inf)),F(c['t'][0]),np.nextafter(F(c['t'][0]),F(np.inf))][i%3])
    s['thz0']=c['tsk']*(1e5/c['pint'][0])**mc.CAPA
    c['u'][:]=[0.,-0.,1e-4][(i//3)%3];c['v'][:]=[-0.,0.][i%2]
add('neutral_signed_zero',40,1,neutral)

def subnormal(c,s,i):
    neutral(c,s,i)
    small=[np.nextafter(F(0),F(1)),F(np.finfo(F).tiny/2),np.nextafter(F(np.finfo(F).tiny),F(0))][i%3]
    c['qv'][0]=float(small);c['qc'][0]=float(small)
    c['u'][0]=float(small)*[-1,1][i%2];c['v'][0]=-float(small)
    s['qsfc']=s['qz0']=float(small)
add('subnormal_probe',40,2,subnormal)

(out/'cases.txt').write_text('\n'.join(sets)+'\n')
for name in sets:
    subprocess.run([str(L/'oracles/eta/run_myjsfc'),str(out/f'case_{name}.bin'),str(out/f'out_{name}.bin')],check=True)
    subprocess.run([str(L/'oracles/eta/run_myjsfc_O2'),str(out/f'case_{name}.bin'),str(out/f'o2_{name}.bin')],check=True)
    assert (out/f'out_{name}.bin').read_bytes()==(out/f'o2_{name}.bin').read_bytes(),name
pf.pack(out,out/'extras.npz')
(L/'receipts/extras-config.json').write_text(json.dumps(meta,indent=2)+'\n')
print(meta)
