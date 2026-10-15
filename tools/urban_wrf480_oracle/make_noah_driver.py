"""Extend only the recorded WRF-facing driver for the release's new state ABI."""
from pathlib import Path
import re,sys

here=Path(__file__).resolve().parent
text=(here.parent/'urban_wrf471_oracle/run_noah_hook.F90').read_text()
text=text.replace('use module_sf_urban, only: urban_param_init, urban_var_init',
                  'use module_sf_urban, only: urban_param_init, urban_var_init, TREEOPTION, TTscheme, FVG_TBL')
anchor='  real, dimension( ncase, 1 ) :: flxhumg_urb2d'
names=('tvg_urb2d','tt_urb2d','xxxvg_urb2d','cmcg_urb2d','flxhumvg_urb2d','flxhumt_urb2d','tvgl_urb3d','smg_urb3d')
decl='\n'.join('  real :: '+n+('(ncase,nsoil,1)' if n.endswith('3d') else '(ncase,1)') for n in names)
assert text.count(anchor)==1;text=text.replace(anchor,anchor+'\n'+decl)
text=text.replace('  flxhumg_urb2d=0.0','  flxhumg_urb2d=0.0\n'+'\n'.join('  '+n+'=0.0' for n in names))
keywords='\n'.join('      '+n+'='+n+', &' for n in names)
text=re.sub(r'( *flxhumg_urb2d\s*=\s*flxhumg_urb2d, *&)',r'\1\n'+keywords,text)
anchor='    call urban_param_init(dzr,dzb,dzg,nsoil,max(1,sf_urban_physics),lcz,.false.)'
text=text.replace(anchor,anchor+'\n    TREEOPTION=1;TTscheme=0;FVG_TBL=.2')
anchor="    call oracle_put('flxhumg_urb2d'//trim(suffix), flxhumg_urb2d)"
text=text.replace(anchor,anchor+'\n'+'\n'.join("    call oracle_put('"+n+"'//trim(suffix), "+n+")" for n in names))
Path(sys.argv[1]).write_text(text)
