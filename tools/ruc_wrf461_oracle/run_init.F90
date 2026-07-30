program run_ruc_init_oracle
  use module_sf_ruclsm, only: ruclsminit
  implicit none

  integer, parameter :: ncol = 4, nzs = 9
  integer, parameter :: ids = 1, ide = ncol + 1
  integer, parameter :: jds = 1, jde = 2
  integer, parameter :: kds = 1, kde = nzs + 1
  integer, parameter :: ims = 1, ime = ncol
  integer, parameter :: jms = 1, jme = 1
  integer, parameter :: kms = 1, kme = nzs
  integer, parameter :: its = 1, ite = ncol
  integer, parameter :: jts = 1, jte = 1
  integer, parameter :: kts = 1, kte = nzs
  character(len=16), parameter :: names(ncol) = [character(len=16) :: &
      'warm_land', 'frozen_land', 'water', 'sea_ice']
  character(len=1024) :: output_path
  character(len=32) :: mminlu
  integer :: i, k, unit
  integer :: isltyp(ims:ime, jms:jme), ivgtyp(ims:ime, jms:jme)
  real :: tslb(ims:ime, 1:nzs, jms:jme)
  real :: smois(ims:ime, 1:nzs, jms:jme)
  real :: sh2o(ims:ime, 1:nzs, jms:jme)
  real :: smfr3d(ims:ime, 1:nzs, jms:jme)
  real :: xice(ims:ime, jms:jme), mavail(ims:ime, jms:jme)
  real :: znt(ims:ime, jms:jme)

  call get_command_argument(1, output_path)
  if (len_trim(output_path) == 0) then
    write(*, '(A)') 'usage: run_init OUTPUT.csv'
    error stop 2
  end if

  tslb = 280.0
  smois = 0.20
  tslb(2,:,1) = [272.8, 271.0, 268.0, 265.0, 270.0, 274.0, 276.0, 278.0, 280.0]
  smois(2,:,1) = [0.30, 0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37, 0.38]
  isltyp(:,1) = [1, 4, 14, 16]
  ivgtyp(:,1) = [1, 10, 17, 15]
  xice(:,1) = [0.0, 0.0, 0.0, 0.4]
  sh2o = -999.0
  smfr3d = -999.0
  mavail = -999.0
  znt = -999.0
  mminlu = 'MODIFIED_IGBP_MODIS_NOAH'

  call ruclsminit(sh2o, smfr3d, tslb, smois, isltyp, ivgtyp, &
      mminlu, xice, mavail, nzs, 17, 15, znt, .false., .true., &
      ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, &
      its, ite, jts, jte, kts, kte)

  open(newunit=unit, file=trim(output_path), status='replace', action='write')
  write(unit, '(A)') 'case,k,tslb,smois,isltyp,ivgtyp,xice,sh2o,smfr3d,mavail,znt'
  do i = 1, ncol
    do k = 1, nzs
      write(unit, '(*(g0,:,","))') trim(names(i)), k, tslb(i,k,1), &
          smois(i,k,1), isltyp(i,1), ivgtyp(i,1), xice(i,1), &
          sh2o(i,k,1), smfr3d(i,k,1), mavail(i,1), znt(i,1)
    end do
  end do
  close(unit)
end program run_ruc_init_oracle
