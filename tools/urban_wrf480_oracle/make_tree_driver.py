"""Extend the existing WRF-facing column driver with the new state arguments.

Only driver inputs and output serialization are changed. The release scheme
and all numerical helpers remain unchanged and are compiled separately.
"""
from pathlib import Path
import sys

here = Path(__file__).resolve().parent
text = (here.parent/'urban_wrf471_oracle/ucm_column_oracle.F90').read_text()
text = text.replace('WRF v4.7.1', 'WRF v4.8.0')
text = text.replace('  real :: lf_urb_s, z0_urb, vegfrac',
                    '  real :: tvg, tt, xxxvg, tvgl(nsoil), smg(nsoil), cmcg, flxhumvg, flxhumt\n'
                    '  real :: smg_seed(nsoil)=(/0.2,0.2,0.2,0.2/)\n'
                    '  real :: lf_urb_s, z0_urb, vegfrac')
text = text.replace('  call write_table()', '''  call get_command_argument(13, arg)
  read(arg, *) treeoption
  call get_command_argument(14, arg)
  read(arg, *) ttscheme
  call get_command_argument(15, arg)
  read(arg, *) fvg_tbl
  if(command_argument_count()>=16) then
    call get_command_argument(16,arg)
    read(arg,*) smg_seed
  end if
''')
text = text.replace('                   lf_urb_s, z0_urb, vegfrac)',
                    '                   tvg, tt, xxxvg, tvgl, smg, cmcg, flxhumvg, flxhumt, &\n'
                    '                   lf_urb_s, z0_urb, vegfrac)')
text = text.replace('    vegfrac = 0.', '''    vegfrac = 0.
    ! Explicit driver inputs. These are not computed by the oracle.
    tvg = tsurf0
    tt = tsurf0
    xxxvg = 0.
    tvgl = tgl
    smg = smg_seed
    cmcg = 0.
    flxhumvg = 0.
    flxhumt = 0.''')
start = text.index('  subroutine write_table()')
end = text.index('  end subroutine write_table', start)+len('  end subroutine write_table')
text = text[:start]+text[end:]
text = text.replace("        ',drelr_in,drelb_in,drelg_in,flxhumr_in,flxhumb_in,flxhumg_in'",
                    "        ',drelr_in,drelb_in,drelg_in,flxhumr_in,flxhumb_in,flxhumg_in' // &\n"
                    "        ',tvg_in,tt_in,xxxvg_in,tvgl1_in,tvgl2_in,tvgl3_in,tvgl4_in' // &\n"
                    "        ',smg1_in,smg2_in,smg3_in,smg4_in,cmcg_in,flxhumvg_in,flxhumt_in'")
text = text.replace("        ',drelr,drelb,drelg,flxhumr,flxhumb,flxhumg'",
                    "        ',drelr,drelb,drelg,flxhumr,flxhumb,flxhumg' // &\n"
                    "        ',tvg,tt,xxxvg,tvgl1,tvgl2,tvgl3,tvgl4' // &\n"
                    "        ',smg1,smg2,smg3,smg4,cmcg,flxhumvg,flxhumt'")
needle = "    write(u, '(6(\",\",ES16.8E3))', advance='no') drelr, drelb, drelg, flxhumr, flxhumb, flxhumg"
text = text.replace(needle, needle+"\n    write(u, '(14(\",\",ES16.8E3))', advance='no') tvg, tt, xxxvg, tvgl, smg, cmcg, flxhumvg, flxhumt")
needle = "    write(u, '(6(\",\",ES16.8E3))') drelr, drelb, drelg, flxhumr, flxhumb, flxhumg"
text = text.replace(needle, "    write(u, '(6(\",\",ES16.8E3))', advance='no') drelr, drelb, drelg, flxhumr, flxhumb, flxhumg\n"
                    "    write(u, '(14(\",\",ES16.8E3))') tvg, tt, xxxvg, tvgl, smg, cmcg, flxhumvg, flxhumt")
Path(sys.argv[1]).write_text(text)
