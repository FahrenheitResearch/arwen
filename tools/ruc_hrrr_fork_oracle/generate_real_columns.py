"""Create an exact-input CPU RUC driver from a captured fused call."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np


def generate(snapshot, directory):
    from gpuwm.core.ruc import (RUC_DRIVER_COLUMN_STATE,
        RUC_DRIVER_COLUMN_FORCING, RUC_DRIVER_PROFILE_STATE, RUC_DRIVER_ARW_FORCING)
    snapshot = Path(snapshot)
    manifest = json.loads(snapshot.with_suffix('.json').read_text())
    arrays = np.load(snapshot, allow_pickle=False)
    kwargs = manifest['driver_keywords']
    ncol = int(manifest['columns'])
    nsl = len(arrays['zs'])
    columns = RUC_DRIVER_COLUMN_STATE + RUC_DRIVER_COLUMN_FORCING + RUC_DRIVER_ARW_FORCING
    profiles = RUC_DRIVER_PROFILE_STATE
    values = {name: np.asarray(arrays['input__' + name], dtype=np.float32)
              for name in columns + profiles}
    # The fork accumulates snowfall in metres. The engine stores millimetres.
    values['snowfallac'] = (values['snowfallac'] * np.float32(.001)).astype(np.float32)
    values.update(ivgtyp=np.asarray(arrays['keyword__ivgtyp'], dtype=np.int32),
                  isltyp=np.asarray(arrays['keyword__isltyp'], dtype=np.int32),
                  zs=np.asarray(arrays['zs'], dtype=np.float32))
    mminlu = kwargs.get('mminlu', 'MODIFIED_IGBP_MODIS_NOAH')
    nlcat, nscat = 28, 19
    for name, size, category in [('landusef', nlcat, 'ivgtyp'),
                                ('soilctop', nscat, 'isltyp')]:
        if 'keyword__' + name in arrays:
            source = np.asarray(arrays['keyword__' + name], dtype=np.float32)
            size = source.shape[0]
        else:
            source = np.zeros((size, ncol), dtype=np.float32)
            source[values[category].reshape(-1) - 1, np.arange(ncol)] = 1
        values[name] = source
        if name == 'landusef':
            nlcat = size
        else:
            nscat = size
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    decl, reads, raw = [], [], []
    for name in columns + profiles + ('ivgtyp', 'isltyp', 'zs', 'landusef', 'soilctop'):
        value = values[name]
        typ = 'integer' if name in ('ivgtyp', 'isltyp') else 'real'
        if name in profiles:
            shape = '(ncol,nsl,1)'
            payload = value.T.reshape(ncol, nsl, 1)
        elif name in ('landusef', 'soilctop'):
            shape = f'(ncol,{value.shape[0]},1)'
            payload = value.T[..., None]
        elif name == 'zs':
            shape = '(nsl)'
            payload = value
        elif name in ('z3d','p8w','t3d','qv3d','qc3d','rho3d'):
            shape = '(ncol,nsl,1)'
            payload = np.broadcast_to(value.reshape(ncol,1,1), (ncol,nsl,1))
        else:
            shape = '(ncol,1)'
            payload = value.reshape(ncol,1)
        decl.append(f'  {typ} :: {name}{shape}')
        reads.append(f'  read(11) {name}')
        raw.append(np.asarray(payload, dtype='<i4' if typ == 'integer' else '<f4')
                   .tobytes(order='F'))
    data_path = directory / 'inputs.bin'
    data_path.write_bytes(b''.join(raw))
    real_names = RUC_DRIVER_COLUMN_STATE + profiles
    output_header = ['column','k','ivgtyp','isltyp'] + list(real_names)
    expressions = ['i','k','ivgtyp(i,1)','isltyp(i,1)']
    for name in real_names:
        expression = f'{name}(i,k,1)' if name in profiles else f'{name}(i,1)'
        if name == 'snowfallac':
            expression = '(' + expression + '*1000.0)'
        expressions.append(expression)
    header = ','.join(output_header)
    header_lines = (' // &\n    '.join(repr(header[start:start+78])
                                  for start in range(0,len(header),78)))
    expression_lines = (', &\n        ').join(expressions)
    def logical(name, default):
        return '.true.' if kwargs.get(name, default) else '.false.'
    water = kwargs.get('iswater') or (16 if mminlu == 'USGS' else 17)
    ice = kwargs.get('isice') or (24 if mminlu == 'USGS' else 15)
    call = '''  call lsmruc(0, pattern_spp_lsm, field_sf, dt, ktau, nsl, &
    lakemodel, lakemask, graupelncv, snowncv, rainncv, &
    zs, rainbl, snow, snowh, snowc, frzfrac, frpcpn, rhosnf, precipfr, &
    z3d, p8w, t3d, qv3d, qc3d, rho3d, glw, gsw, emiss, chklowq, chs, &
    flqc, flhc, mavail, canwat, vegfra, alb, znt, z0, snoalb, albbck, lai, &
    mminlu, landusef, nlcat, mosaic_lu, mosaic_soil, soilctop, nscat, &
    qsfc, qsg, qvg, qcg, dew, soilt1, tsnav, tbot, ivgtyp, isltyp, xland, &
    iswater, isice, xice, xice_threshold, cp0, rovcp0, g0, lv, stb, &
    soilmois, sh2o, smavail, smmax, tso, soilt, hfx, qfx, lh, sfcrunoff, &
    udrunoff, acrunoff, sfcexc, sfcevp, grdflx, snowfallac, acsnow, snom, &
    smfr3d, keepfr3dflag, myj, shdmin, shdmax, rdlai2d, &
    1,ncol+1,1,2,1,nsl+1,1,ncol,1,1,1,nsl,1,ncol,1,1,1,nsl)'''
    source = f'''program run_real_ruc
  use module_sf_ruclsm, only: lsmruc, ruclsminit
  implicit none
  integer, parameter :: ncol={ncol}, nsl={nsl}, nlcat={nlcat}, nscat={nscat}
  integer, parameter :: ktau={int(kwargs['ktau'])}, iswater={water}, isice={ice}
  integer, parameter :: mosaic_lu={int(kwargs.get('mosaic_lu',0))}, mosaic_soil={int(kwargs.get('mosaic_soil',0))}
  integer, parameter :: lakemodel={int(kwargs.get('lakemodel',0))}
  real, parameter :: dt={np.float32(kwargs['dt']):.9e}
  real, parameter :: xice_threshold={np.float32(kwargs.get('xice_threshold',.02)):.9e}
  real, parameter :: cp0=1004.5, rovcp0=287.0/1004.5, g0=9.81, lv=2.5e6, stb=5.67051e-8
  logical, parameter :: myj={logical('myj',False)}, frpcpn={logical('frpcpn',True)}
  logical, parameter :: rdlai2d={logical('rdlai2d',True)}
  character(len=32), parameter :: mminlu='{mminlu}'
  integer :: i,k,u
  character(len=1024) :: output_path
  real :: pattern_spp_lsm(ncol,nsl,1), field_sf(ncol,nsl,1)
{chr(10).join(decl)}
  open(unit=11,file='inputs.bin',access='stream',form='unformatted',status='old')
{chr(10).join(reads)}
  close(11)
  ! Initialize module parameter tables, then restore every captured input.
  call ruclsminit(sh2o,smfr3d,tso,soilmois,isltyp,ivgtyp,mminlu,xice, &
    mavail,nsl,iswater,isice,znt,.false.,.true., &
    1,ncol+1,1,2,1,nsl+1,1,ncol,1,1,1,nsl,1,ncol,1,1,1,nsl)
  open(unit=11,file='inputs.bin',access='stream',form='unformatted',status='old')
{chr(10).join(reads)}
  close(11)
  pattern_spp_lsm=0.0
  field_sf=0.0
  call scrub(10,0.0)
{call}
  call get_command_argument(1,output_path)
  if (len_trim(output_path)==0) output_path='oracle-real.csv'
  open(newunit=u,file=trim(output_path),status='replace',action='write')
  write(u,'(A)') {header_lines}
  do i=1,ncol
    do k=1,nsl
      write(u,'(*(g0,:,","))') &
        {expression_lines}
    end do
  end do
  close(u)
contains
  recursive subroutine scrub(depth, fill)
    integer, intent(in) :: depth
    real, intent(in) :: fill
    real :: pad(4096)
    pad = fill
    if (depth > 0) call scrub(depth - 1, fill)
    if (sum(pad) < -1.0e30) write(*, *) 'unreachable'
  end subroutine scrub
end program run_real_ruc
'''
    driver = directory / 'run_real.F90'
    driver.write_text(source)
    receipt = {'snapshot': snapshot.name, 'columns': ncol, 'nsl': nsl,
               'driver_keywords': kwargs, 'source_files': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                                       for p in (snapshot, snapshot.with_suffix('.json'), data_path, driver)}}
    (directory / 'inputs-receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('snapshot')
    parser.add_argument('directory')
    args = parser.parse_args()
    print(json.dumps(generate(args.snapshot, args.directory), indent=2))
