program run_nt_oracle
  ! THE WRF SIDE of the New Tiedtke column oracle: WRF v4.6.1's own
  ! cu_ntiedtke_driver, byte-unmodified, over a whole case in one call.
  !
  !   run_nt_oracle CASE OUT DX STEPCU ITIMESTEP BL_LANES
  !
  ! CASE holds raw little-endian float32 arrays in WRF memory order
  ! a(i,k,j) (see make_columns.py) and meta.txt (nx, ny, nz, dt).  OUT
  ! receives the six tendencies, RAINCV, PRATEC, CU_ACT_FLAG and the two
  ! folded forcing fields, in the same layout.
  !
  ! THE PATCH IS THE REAL ONE.  The driver is called once with
  ! its..ite = 1..nx and jts..jte = 1..ny, exactly what a serial WRF run
  ! with one tile hands it, so cu_ntiedtke_driver's own j loop makes one
  ! cu_ntiedtke_run call per row of nx columns.  That matters for one
  ! thing only: cuascn's llo3 is an OR over the columns of the call
  ! (cu_ntiedtke.F90:1975-2003), so the row is WRF's reduction scope.
  !
  ! THE FOLD is module_cumulus_driver.F:879-881, transcribed:
  !
  !     RTHFTEN(i,k,j)=(RTHFTEN(i,k,j)+RTHRATEN(i,k,j)+RTHBLTEN(i,k,j))*pi(i,k,j)
  !     RQVFTEN(i,k,j)=RQVFTEN(i,k,j)+RQVBLTEN(i,k,j)
  !
  ! with RTHRATEN as the radiation driver forms it: the longwave scheme
  ! writes it (RTHRATENLW=RTHRATEN in the scheme call) and the shortwave
  ! increment is added after (module_radiation_driver.F:2528 and its
  ! siblings), so RTHRATEN = LW + SW.  BL_LANES = 0 is a run without a PBL
  ! scheme: RTHBLTEN and RQVBLTEN stay at the zero the driver allocates.
  !
  ! The module_cumulus_driver.F file itself cannot be compiled outside a
  ! configured WRF (it needs the registry and the framework), so these
  ! three lines are the one transcription in this harness; the scheme
  ! files are compiled as they are.
  use iso_fortran_env, only: int32, real32
  use ccpp_kind_types, only: kind_phys
  use module_cu_ntiedtke, only: cu_ntiedtke_driver
  implicit none

  ! The WRF constants module_model_constants hands the driver
  ! (g, xlf, xls, xlv, r_d, r_v, cp = 7*r_d/2).
  real(kind_phys), parameter :: c_g = 9.81, c_rd = 287.0, c_rv = 461.6
  real(kind_phys), parameter :: c_cp = 7.0*c_rd/2.0
  real(kind_phys), parameter :: c_xlv = 2.5e6, c_xlf = 3.50e5
  real(kind_phys), parameter :: c_xls = c_xlv + c_xlf

  logical, parameter :: p_f_qv = .true., p_f_qc = .true., p_f_qr = .true.
  logical, parameter :: p_f_qi = .true., p_f_qs = .true.

  character(len=1024) :: casedir, outdir, arg, line
  character(len=64) :: key
  character(len=256) :: errmsg
  integer :: nx, ny, nz, nz1, stepcu, itimestep, bl_lanes, errflg, u, ios
  integer :: i, j, k
  real(kind_phys) :: dt, dxval

  real(kind_phys), allocatable, dimension(:,:,:) :: &
       u3d, v3d, w, t3d, qv3d, qc3d, qi3d, pi3d, rho3d, dz8w, pcps, p8w, &
       rthdynten, rthratenlw, rthratensw, rthblten, rqvdynten, rqvblten, &
       rthraten, thften, qvften, &
       rthcuten, rqvcuten, rqccuten, rqicuten, rucuten, rvcuten
  real(kind_phys), allocatable, dimension(:,:) :: &
       raincv, pratec, qfx, hfx, xland, dx
  logical, allocatable, dimension(:,:) :: cu_act_flag
  integer(int32), allocatable, dimension(:,:) :: flag_out

  if (command_argument_count() /= 6) then
     write(*,'(a)') 'usage: run_nt_oracle CASE OUT DX STEPCU ITIMESTEP BL_LANES'
     stop 2
  end if
  call get_command_argument(1, casedir)
  call get_command_argument(2, outdir)
  call get_command_argument(3, arg); read(arg,*) dxval
  call get_command_argument(4, arg); read(arg,*) stepcu
  call get_command_argument(5, arg); read(arg,*) itimestep
  call get_command_argument(6, arg); read(arg,*) bl_lanes

  ! ccpp_kind_types.F takes its DOUBLE branch silently when RWORDSIZE is
  ! not defined; refuse rather than write a double-precision answer.
  if (kind(1.0_kind_phys) /= 4) then
     write(*,'(a)') 'FATAL: kind_phys is not single precision'
     stop 3
  end if

  nx = -1; ny = -1; nz = -1; dt = -1.0
  open(newunit=u, file=trim(casedir)//'/meta.txt', status='old', action='read')
  do
     read(u,'(a)',iostat=ios) line
     if (ios /= 0) exit
     read(line,*,iostat=ios) key
     if (ios /= 0) cycle
     select case (trim(key))
     case ('nx'); read(line,*) key, nx
     case ('ny'); read(line,*) key, ny
     case ('nz'); read(line,*) key, nz
     case ('dt'); read(line,*) key, dt
     end select
  end do
  close(u)
  if (nx <= 0 .or. ny <= 0 .or. nz <= 0 .or. dt <= 0.0) then
     write(*,'(a)') 'FATAL: meta.txt is incomplete'
     stop 4
  end if
  nz1 = nz + 1

  allocate(u3d(nx,nz1,ny), v3d(nx,nz1,ny), w(nx,nz1,ny), t3d(nx,nz1,ny), &
       qv3d(nx,nz1,ny), qc3d(nx,nz1,ny), qi3d(nx,nz1,ny), pi3d(nx,nz1,ny), &
       rho3d(nx,nz1,ny), dz8w(nx,nz1,ny), pcps(nx,nz1,ny), p8w(nx,nz1,ny), &
       rthdynten(nx,nz1,ny), rthratenlw(nx,nz1,ny), rthratensw(nx,nz1,ny), &
       rthblten(nx,nz1,ny), rqvdynten(nx,nz1,ny), rqvblten(nx,nz1,ny), &
       rthraten(nx,nz1,ny), thften(nx,nz1,ny), qvften(nx,nz1,ny), &
       rthcuten(nx,nz1,ny), rqvcuten(nx,nz1,ny), rqccuten(nx,nz1,ny), &
       rqicuten(nx,nz1,ny), rucuten(nx,nz1,ny), rvcuten(nx,nz1,ny))
  allocate(raincv(nx,ny), pratec(nx,ny), qfx(nx,ny), hfx(nx,ny), &
       xland(nx,ny), dx(nx,ny), cu_act_flag(nx,ny), flag_out(nx,ny))

  ! Zeroed whole, including the kme slot of the mass fields, so no word
  ! the driver might read is whatever the allocator handed back.
  u3d = 0.0; v3d = 0.0; w = 0.0; t3d = 0.0; qv3d = 0.0; qc3d = 0.0
  qi3d = 0.0; pi3d = 0.0; rho3d = 0.0; dz8w = 0.0; pcps = 0.0; p8w = 0.0
  rthdynten = 0.0; rthratenlw = 0.0; rthratensw = 0.0; rthblten = 0.0
  rqvdynten = 0.0; rqvblten = 0.0; rthraten = 0.0; thften = 0.0
  qvften = 0.0
  rthcuten = 0.0; rqvcuten = 0.0; rqccuten = 0.0; rqicuten = 0.0
  rucuten = 0.0; rvcuten = 0.0
  raincv = 0.0; pratec = 0.0; cu_act_flag = .false.

  call read_lev('t3d', t3d);       call read_lev('qv3d', qv3d)
  call read_lev('qc3d', qc3d);     call read_lev('qi3d', qi3d)
  call read_lev('u3d', u3d);       call read_lev('v3d', v3d)
  call read_lev('pcps', pcps);     call read_lev('dz8w', dz8w)
  call read_lev('rho3d', rho3d);   call read_lev('exner', pi3d)
  call read_lev('rthdynten', rthdynten)
  call read_lev('rthratenlw', rthratenlw)
  call read_lev('rthratensw', rthratensw)
  call read_lev('rqvdynten', rqvdynten)
  if (bl_lanes /= 0) then
     call read_lev('rthblten', rthblten)
     call read_lev('rqvblten', rqvblten)
  end if
  call read_iface('p8w', p8w);     call read_iface('w', w)
  call read_sfc('xland', xland);   call read_sfc('hfx', hfx)
  call read_sfc('qfx', qfx)
  dx = dxval

  ! ---- the radiation driver's RTHRATEN, then the cumulus driver's fold --
  do j = 1, ny
     do k = 1, nz
        do i = 1, nx
           rthraten(i,k,j) = rthratenlw(i,k,j)
           rthraten(i,k,j) = rthraten(i,k,j) + rthratensw(i,k,j)
        end do
     end do
  end do
  thften = rthdynten
  qvften = rqvdynten
  do j = 1, ny
     do k = 1, nz
        do i = 1, nx
           thften(i,k,j) = (thften(i,k,j) + rthraten(i,k,j) &
                           + rthblten(i,k,j)) * pi3d(i,k,j)
           qvften(i,k,j) = qvften(i,k,j) + rqvblten(i,k,j)
        end do
     end do
  end do

  errflg = 0
  call cu_ntiedtke_driver( &
       dt=dt, itimestep=itimestep, stepcu=stepcu, hfx=hfx, &
       raincv=raincv, pratec=pratec, qfx=qfx, &
       u3d=u3d, v3d=v3d, w=w, t3d=t3d, pi3d=pi3d, rho3d=rho3d, &
       qv3d=qv3d, qc3d=qc3d, qi3d=qi3d, &
       dz8w=dz8w, pcps=pcps, p8w=p8w, xland=xland, dx=dx, &
       cu_act_flag=cu_act_flag, &
       ids=1, ide=nx+1, jds=1, jde=ny+1, kds=1, kde=nz1, &
       ims=1, ime=nx, jms=1, jme=ny, kms=1, kme=nz1, &
       its=1, ite=nx, jts=1, jte=ny, kts=1, kte=nz, &
       qvften=qvften, thften=thften, &
       f_qv=p_f_qv, f_qc=p_f_qc, f_qr=p_f_qr, f_qi=p_f_qi, f_qs=p_f_qs, &
       rthcuten=rthcuten, rqvcuten=rqvcuten, &
       rqccuten=rqccuten, rqicuten=rqicuten, &
       rucuten=rucuten, rvcuten=rvcuten, &
       grav=c_g, xlf=c_xlf, xls=c_xls, xlv=c_xlv, rd=c_rd, rv=c_rv, cp=c_cp, &
       errmsg=errmsg, errflg=errflg)
  if (errflg /= 0) then
     write(*,'(a,i0,a,a)') 'driver errflg=', errflg, ' : ', trim(errmsg)
     stop 7
  end if

  call write_lev('rthcuten', rthcuten); call write_lev('rqvcuten', rqvcuten)
  call write_lev('rqccuten', rqccuten); call write_lev('rqicuten', rqicuten)
  call write_lev('rucuten', rucuten);   call write_lev('rvcuten', rvcuten)
  call write_lev('thften', thften);     call write_lev('qvften', qvften)
  call write_sfc('raincv', raincv);     call write_sfc('pratec', pratec)
  flag_out = merge(1_int32, 0_int32, cu_act_flag)
  open(newunit=u, file=trim(outdir)//'/cu_act_flag.i32', access='stream', &
       form='unformatted', status='replace', action='write')
  write(u) flag_out
  close(u)

  write(*,'(a,i0,a,i0,a,i0,a,f8.1,a,i0,a,i0,a,i0)') 'run_nt_oracle: ', nx, &
       ' x ', ny, ' columns, nz ', nz, ', dx ', dxval, ', stepcu ', stepcu, &
       ', itimestep ', itimestep, ', bl_lanes ', bl_lanes

contains

  subroutine read_lev(name, a)
    character(len=*), intent(in) :: name
    real(kind_phys), intent(inout) :: a(:,:,:)
    real(real32), allocatable :: buf(:,:,:)
    integer :: uu
    allocate(buf(nx, nz, ny))
    open(newunit=uu, file=trim(casedir)//'/'//name//'.f32', &
         access='stream', form='unformatted', status='old', action='read')
    read(uu) buf
    close(uu)
    a(:, 1:nz, :) = buf
  end subroutine read_lev

  subroutine read_iface(name, a)
    character(len=*), intent(in) :: name
    real(kind_phys), intent(inout) :: a(:,:,:)
    real(real32), allocatable :: buf(:,:,:)
    integer :: uu
    allocate(buf(nx, nz1, ny))
    open(newunit=uu, file=trim(casedir)//'/'//name//'.f32', &
         access='stream', form='unformatted', status='old', action='read')
    read(uu) buf
    close(uu)
    a(:, 1:nz1, :) = buf
  end subroutine read_iface

  subroutine read_sfc(name, a)
    character(len=*), intent(in) :: name
    real(kind_phys), intent(inout) :: a(:,:)
    integer :: uu
    open(newunit=uu, file=trim(casedir)//'/'//name//'.f32', &
         access='stream', form='unformatted', status='old', action='read')
    read(uu) a
    close(uu)
  end subroutine read_sfc

  subroutine write_lev(name, a)
    character(len=*), intent(in) :: name
    real(kind_phys), intent(in) :: a(:,:,:)
    integer :: uu
    open(newunit=uu, file=trim(outdir)//'/'//name//'.f32', &
         access='stream', form='unformatted', status='replace', action='write')
    write(uu) a(:, 1:nz, :)
    close(uu)
  end subroutine write_lev

  subroutine write_sfc(name, a)
    character(len=*), intent(in) :: name
    real(kind_phys), intent(in) :: a(:,:)
    integer :: uu
    open(newunit=uu, file=trim(outdir)//'/'//name//'.f32', &
         access='stream', form='unformatted', status='replace', action='write')
    write(uu) a
    close(uu)
  end subroutine write_sfc

end program run_nt_oracle
