"""64 columns with real Tegen climatology, seasons, edges and sun branches."""
import sys
from pathlib import Path
import sw_make_synthetic as make
days=[0.,14.999,15.,15.001,44.,74.,104.,135.,165.,196.,227.,257.,288.,318.,349.,364.999]
locations=[(-89.,-179.),(-45.,-170.),(0.,0.),(25.,15.),(35.,-30.),(65.,170.),(89.,179.),(90.,180.)]
out=[]
for cid in range(64):
    c=make.base_column(psfc=101000. if cid%2 else 85000.,t0=285. if cid%2 else 300.)
    c['xlat'],c['xlong']=locations[cid%8]
    c['coszen']=[0.95,0.02,0.5,-0.1][cid//16]
    block=[]
    make.emit(block,cid+1,c,8,(1,1,1),0)
    day=days[cid%16]
    header=block[0].split(); header[2]=str(int(day)+1); block[0]=' '.join(header)
    scalars=block[1].split(); scalars[0]=make.fmt(day); block[1]=' '.join(scalars)
    out.extend(block)
Path(sys.argv[1]).write_text(f"64 {make.NZ}\n"+'\n'.join(out)+'\n')
