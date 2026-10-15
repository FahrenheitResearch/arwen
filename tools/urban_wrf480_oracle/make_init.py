"""Adapt only the WRF-facing input/output driver for the new Registry ABI."""
from pathlib import Path
import re,sys

here=Path(__file__).resolve().parent
text=(here.parent/'urban_wrf471_oracle/run_init.F90').read_text()
start=text.index('  use module_sf_bep,')
end=text.index('  implicit none',start)
text=text[:start]+text[end:]
text=re.sub(r'\b(?:bep|bem)_(?:ndm|nz|ng|nwr|nf|ngb|nbui|ngr)_\b','1',text)
text=text.replace('do opt = 1, 3','do opt = 1, 1')
text=text.replace('ncol = 10','ncol = 14')
text=re.sub(r'ivgtyp\(:, 1\) = \(/ isurban, isurban, isurban,.*?17 /\)',
    'ivgtyp(:, 1) = (/ isurban,isurban,isurban,lcz(1),lcz(2),lcz(3),10,14,17,isurban,isurban,isurban,isurban,isurban /)',text,count=1,flags=re.S)
text=re.sub(r'ivgtyp\(:, 1\) = \(/ isurban, lcz\(1\),.*?17 /\)',
    'ivgtyp(:, 1) = (/ lcz,10,14,17 /)',text,count=1,flags=re.S)
text=text.replace('0.4, 0.0, 0.0 /)','0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 /)')
text=text.replace('    real, dimension(ncol, nsoil, 1) :: trl, tbl, tgl, tgrl, smr',
    '    real, dimension(ncol,1) :: tvg,tt,xxxvg,cmcg,flxhumvg,flxhumt\n'
    '    real, dimension(ncol,nsoil,1) :: tvgl,smg\n'
    '    real, dimension(ncol, nsoil, 1) :: trl, tbl, tgl, tgrl, smr')
text=text.replace('    utype = -7','    utype = -7\n    tvg=-7;tt=-7;xxxvg=-7;cmcg=-7;flxhumvg=-7;flxhumt=-7;tvgl=-7;smg=-7')
text=text.replace('      FLXHUMG_URB2D=flxhumg, &',
    '      FLXHUMG_URB2D=flxhumg, &\n'
    '      TVG_URB2D=tvg,TT_URB2D=tt,XXXVG_URB2D=xxxvg,TVGL_URB3D=tvgl,SMG_URB3D=smg, &\n'
    '      CMCG_URB2D=cmcg,FLXHUMVG_URB2D=flxhumvg,FLXHUMT_URB2D=flxhumt, &')
needle="    call oracle_put('flxhumg_urb2d', flxhumg)"
text=text.replace(needle,needle+'\n'+'\n'.join(f"    call oracle_put('{n}_urb{'3d' if n in ('tvgl','smg') else '2d'}', {n})" for n in ('tvg','tt','xxxvg','tvgl','smg','cmcg','flxhumvg','flxhumt')))
text=text.replace('    write(case,',"    if(sf_urban_physics/=1) error stop 'this driver exercises sf_urban_physics=1 only'\n    write(case,",1)
Path(sys.argv[1]).write_text(text)
