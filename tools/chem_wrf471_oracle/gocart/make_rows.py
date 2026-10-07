"""Write source-cited component data and the exact row contract for review."""
import json
import re
from pathlib import Path

root=Path(__file__).resolve().parents[3]
dest=root/'gpuwm-data/gpuwm_data/data/chem/optics'
src=root/'wrf-src/chem/module_data_rrtmgaeropt.F'
lines=src.read_text().splitlines()
arrays={}
for n,line in enumerate(lines,1):
    if line.lstrip().startswith('!'):continue
    m=re.search(r'\bdata\s+(\w+)\s*/(.*)',line,re.I)
    if not m:continue
    name=m[1];text=m[2]
    end=n
    while '/' not in text:
        text+=' '+lines[end];end+=1
    text=text.split('/')[0].replace('&','')
    values=[]
    for term in text.split(','):
        term=term.strip()
        if '*' in term:
            repeat,val=term.split('*');repeat={'nswbands':4,'nlwbands':16}.get(repeat,repeat)
            values.extend([float(val)]*int(repeat))
        else:values.append(float(term))
    arrays[name]={'value':values,'dtype':'float32','from':f'chem/module_data_rrtmgaeropt.F:{n}-{end}'}
def number(value,line,file='chem/module_optical_averaging.F',dtype='float32'):
    return {'value':value,'dtype':dtype,'from':f'{file}:{line}'}
prep=(root/'wrf-src/chem/module_optical_averaging.F').read_text().splitlines()
def assignment_line(name):
    return next(i for i in range(3545,4187) if re.match(r'\s*'+name+r'\s*=',prep[i-1],re.I))
components={}
for key,density,kappa,dline,kline,tab in [
 ('sulfate',1.8,0.5,3816,1191,'sulf'),('ammonium',1.8,0.5,3821,1193,None),
 ('other_inorganic',2.6,0.14,3825,1196,None),('organic_carbon',1.,0.14,3827,1194,'oc'),
 ('black_carbon',1.8,0.14,3833,1194,None),('sodium',2.2,1.16,3822,1200,'seas'),
 ('chloride',2.2,1.16,3818,1199,'seas'),('dust',2.6,0.1,3826,1197,'dust'),
 ('msa',1.8,0.58,3819,581,None),('water',1.,None,3844,None,'water')]:
    dname={'sulfate':'so4','ammonium':'nh4','other_inorganic':'oin','organic_carbon':'oc',
           'black_carbon':'bc','sodium':'na','chloride':'cl','dust':'dust','msa':'msa','water':'h2o'}[key]
    c={'density_g_cm3':number(density,assignment_line('dens_'+dname))}
    if kappa is not None:c['kappa']=number(kappa,kline,'chem/module_data_mosaic_asect.F' if key=='msa' else 'chem/module_data_sorgam.F')
    if tab:
        for wave in ['sw','lw']:
            for part in ['r','i']:
                name=f'ref{part}w{wave}' if tab=='water' else f'ref{part}{wave}_{tab}'
                c[name]=arrays[name]
    components[key]=c
components['nitrate']={'density_g_cm3':number(1.8,assignment_line('dens_no3')),
 'kappa':number(.5,1192,'chem/module_data_sorgam.F'),
 'refractive_index':number([1.50,0.],assignment_line('ref_index_nh4no3')),
 'note':'mass_no3 is initialized to zero and never populated on the GOCART path'}
components['sea_salt']={k:v for k,v in components['sodium'].items() if k!='kappa'}
components['sea_salt']['kappa']=number(1.16,1198,'chem/module_data_sorgam.F')
components['sea_salt']['sodium_mass_fraction']=number([22.9898,58.4428],4012)
components['sea_salt']['chloride_mass_fraction']=number([35.4530,58.4428],4011)
components['black_carbon']['refractive_index']=number([1.85,0.71],assignment_line('ref_index_bc'))
for wave in ['sw','lw']:
 for part in ['r','i']:
  name=f'ref{part}{wave}_bc'
  components['black_carbon']['mie_grid_'+name]=arrays[name]
components['black_carbon']['note']='fixed prep index is 1.85+0.71i; the first-call fitting grid brackets 1.95+0.79i'
components['ammonium']['refractive_index']=number([1.50,0.],assignment_line('ref_index_nh4no3'))
components['other_inorganic']['refractive_index']=number([1.55,0.006],assignment_line('ref_index_oin'))
components['msa']['refractive_index']=number([1.43,0.],assignment_line('ref_index_msa'))
metadata={'components':components,'mode_constants':{
 'aitken_fraction':number(.25,3643),'minimum_diameter_um':number(.0390625,3676),
 'maximum_diameter_um':number(20.,3677),'rh_cap':number(.9,4034),
 'sigma':number([1.7,2.,2.5],'729-737','chem/module_data_sorgam.F'),
 'diameter_m':number([.01e-6,.07e-6,1e-6],'741-749','chem/module_data_sorgam.F'),
 'oc_mfac':number(1.8,20,'chem/module_data_gocartchem.F'),
 'nh4_mfac':number(1.375,18,'chem/module_data_gocartchem.F'),
 'sulfate_conversion_molecular_mass':number(96.,3924),
 'dry_air_conversion_molecular_mass':number(28.97,3924)},
 'bin_radius_um':{
 'dust':number([[.1,1.,1.8,3.,6.],[1.,1.8,3.,6.,10.]],'33-34','phys/module_data_gocart_dust.F','float64'),
 'sea_salt':number([[.1,.5,1.5,5.],[.5,1.5,5.,10.]],'2-3','chem/module_data_gocart_seas.F','float64')}}
(dest/'wrfchem_components.json').write_text(json.dumps(metadata,indent=2)+'\n')
names='so2 sulf dms msa p25 bc1 bc2 oc1 oc2 dust_1 dust_2 dust_3 dust_4 dust_5 seas_1 seas_2 seas_3 seas_4 p10'.split()
spec={'sulf':('sulfate',0,True,1.,True),'msa':('msa',7,True,1.,True),
      'p25':('other_inorganic',2,False,1.,True),'bc1':('black_carbon',5,False,1.,False),
      'bc2':('black_carbon',6,False,1.,True),'oc1':('organic_carbon',3,False,1.8,False),
      'oc2':('organic_carbon',4,False,1.8,True)}
rows=[]
for order,name in enumerate(names):
    o=None
    if name in spec:
        component,target,conversion,mult,hygro=spec[name]
        o={'wrfchem_component':component,'wrf_order':order,'size_treatment':'modal',
           'target':target,'conversion':'ppmv_sulfate' if conversion else 'ug_kg',
           'mass_multiplier':mult,'aitken_fraction':.25,'hygroscopic':hygro,
           'merra_type':component,'merra_bin':None,'secondary_target':1 if name=='sulf' else None,
           'secondary_multiplier':.375 if name=='sulf' else None}
    elif name.startswith('dust_') or name.startswith('seas_'):
        dust=name.startswith('dust_');b=int(name.split('_')[1])-1
        component='dust' if dust else 'sea_salt'
        edges=metadata['bin_radius_um'][component]['value']
        o={'wrfchem_component':component,'wrf_order':order,'size_treatment':'fixed_bin',
           'distribution_table':'dustfrc_goc9bin' if dust else 'seasfrc_goc9bin',
           'distribution_bin':b,'bin_radius_um':[edges[0][b],edges[1][b]],
           'hygroscopic':True,'merra_type':component,'merra_bin':b+1}
    if o is not None:o['from']='chem/module_optical_averaging.F:3928-4016'
    rows.append({'name':name,'optics':o})
(dest/'gocart_simple_optics_rows.json').write_text(json.dumps(rows,indent=2)+'\n')
