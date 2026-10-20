#!/usr/bin/env python3
"""Analysis charts from Rust diagnostic receipts; no weather-array processing."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--series',action='append',required=True,help='LABEL=series.json');p.add_argument('--out',type=Path,required=True);p.add_argument('--height',type=int,choices=[500,1000,3000],default=1000);a=p.parse_args()
fig,axes=plt.subplots(3,len(a.series),figsize=(6*len(a.series),10),squeeze=False,layout='constrained')
metrics=[('max_w_m_s','Maximum w (m/s)'),('max_abs_vertical_vorticity_s_1','Maximum |vertical vorticity| (s⁻¹)'),('updraft_area_w_ge_5_m_s_km2','Updraft area, w ≥ 5 m/s (km²)')]
for col,binding in enumerate(a.series):
 label,path=binding.split('=',1);receipt=json.loads(Path(path).read_text()); groups={}
 for frame in receipt['frames']:
  domain=Path(frame['source']).name.split('_')[1];time=datetime.strptime(frame['valid_time'],'%Y-%m-%d_%H:%M:%S');stat=next(s for s in frame['statistics'] if s['height_agl_m']==a.height)
  groups.setdefault(domain,[]).append((time,stat))
 start=min(t for rows in groups.values() for t,_ in rows)
 for row,(key,title) in enumerate(metrics):
  ax=axes[row,col]
  for domain,rows in sorted(groups.items()):
   rows.sort();ax.plot([(t-start).total_seconds()/3600 for t,_ in rows],[s[key] for _,s in rows],marker='.',label=domain)
  ax.set_ylabel(title);ax.set_xlabel('Hours from first diagnostic frame');ax.grid(alpha=.25);ax.legend()
  if row==0:ax.set_title(label)
fig.suptitle(f'Fallback diagnostics — {a.height} m AGL\nDomain footprints differ',fontsize=14)
a.out.parent.mkdir(parents=True,exist_ok=True);fig.savefig(a.out,dpi=150);plt.close(fig)
