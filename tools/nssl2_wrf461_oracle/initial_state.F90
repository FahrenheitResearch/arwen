program nssl2_initial_state_oracle
  use module_mp_nssl_2mom, only: nssl_2mom_init, calcnfromq
  implicit none

  integer, parameter :: nx = 4, ny = 1, nz = 12, na = 40
  integer :: i, k, unit, case_id
  real :: nssl_params(20), an(nx,ny,nz,na), density(nx,nz+1)
  real :: qv(nx,nz), qc(nx,nz), qr(nx,nz), qi(nx,nz)
  real :: qs(nx,nz), qg(nx,nz), qh(nx,nz)
  real :: nc(nx,nz), nr(nx,nz), ni(nx,nz), ns(nx,nz)
  real :: ng(nx,nz), nh(nx,nz), qnn(nx,nz)
  real :: volg(nx,nz), volh(nx,nz), scale
  character(len=512) :: output_path

  call get_command_argument(1, output_path)
  if (len_trim(output_path) == 0) then
     error stop 'usage: nssl2_initial_state_oracle OUTPUT.csv'
  endif

  nssl_params = 0.0
  nssl_params(1:10) = [0.5e9, 0.0, 1.0, 4.0e5, 4.0e4, &
                       8.0e5, 3.0e6, 500.0, 900.0, 100.0]
  call nssl_2mom_init(nssl_params=nssl_params, ipctmp=5, mixphase=0, &
       nssl_density_on=.true., nssl_hail_on=.true.,                 &
       nssl_ccn_on=.true., nssl_icdx=6, nssl_icdxhl=6)

  an = 0.0
  qv = 0.010
  qc = 0.0
  qr = 0.0
  qi = 0.0
  qs = 0.0
  qg = 0.0
  qh = 0.0
  nc = 0.0
  nr = 0.0
  ni = 0.0
  ns = 0.0
  ng = 0.0
  nh = 0.0
  qnn = 0.5e9 / 1.225
  volg = 0.0
  volh = 0.0

  do k = 1, nz + 1
     do i = 1, nx
        density(i,k) = 1.225 * exp(-real((k - 1) * 500) / 8000.0)
     enddo
  enddo
  do k = 1, nz
     do i = 1, nx
        case_id = mod((k - 1) * nx + i - 1, 8)
        select case (case_id)
        case (0)
           ! Empty state.
        case (1)
           qc(i,k) = 0.5e-13
           qr(i,k) = 0.5e-12
           qi(i,k) = 0.5e-13
           qs(i,k) = 0.5e-13
           qg(i,k) = 0.5e-12
           qh(i,k) = 0.5e-12
        case (2)
           ! Above qxmin but below qxmin_init: returned to vapor.
           qc(i,k) = 5.0e-10
           qr(i,k) = 5.0e-10
           qi(i,k) = 5.0e-10
           qs(i,k) = 5.0e-10
           qg(i,k) = 5.0e-10
           qh(i,k) = 5.0e-10
        case (3,5,6)
           scale = 1.0 + 0.25 * real(case_id)
           qc(i,k) = scale * 2.0e-5
           qr(i,k) = scale * 4.0e-5
           qi(i,k) = scale * 3.0e-5
           qs(i,k) = scale * 5.0e-5
           qg(i,k) = scale * 6.0e-5
           qh(i,k) = scale * 7.0e-5
           if (case_id == 5) then
              nc(i,k) = -1.0
              nr(i,k) = -1.0
              ni(i,k) = -1.0
              ns(i,k) = -1.0
              ng(i,k) = -1.0
              nh(i,k) = -1.0
              volg(i,k) = -1.0
              volh(i,k) = -1.0
           endif
        case (4)
           qc(i,k) = 2.0e-4
           qr(i,k) = 3.0e-4
           qi(i,k) = 1.5e-4
           qs(i,k) = 2.5e-4
           qg(i,k) = 3.5e-4
           qh(i,k) = 4.5e-4
           nc(i,k) = 8.0e7
           nr(i,k) = 4.0e5
           ni(i,k) = 2.0e6
           ns(i,k) = 3.0e5
           ng(i,k) = 2.0e5
           nh(i,k) = 8.0e4
           volg(i,k) = qg(i,k) / 650.0
           volh(i,k) = qh(i,k) / 850.0
        case (7)
           ! qxmin_init itself is not sufficient: the source uses >.
           qc(i,k) = 1.0e-8
           qr(i,k) = 1.0e-8
           qi(i,k) = 1.0e-8
           qs(i,k) = 1.0e-8
           qg(i,k) = 1.0e-8
           qh(i,k) = 1.0e-8
        end select
     enddo
  enddo

  open(newunit=unit, file=trim(output_path), status='replace', action='write')
  write(unit,'(A)') 'phase,case,i,k,rho_kg_m3,qv,qc,qr,qi,qs,qg,qh,nc_per_kg,nr_per_kg,ni_per_kg,ns_per_kg,ng_per_kg,nh_per_kg,qnn_per_kg,qvolg_m3_per_kg,qvolh_m3_per_kg'
  call write_rows(unit, 'before')

  call calcnfromq(nx,ny,nz,an,na,0,0,density,                 &
       qcw=qc,qci=qi,qsw=qs,qrw=qr,qhw=qg,qhl=qh,             &
       ccw=nc,cci=ni,csw=ns,crw=nr,chw=ng,chl=nh,             &
       cccn=qnn,vhw=volg,vhl=volh,qv=qv)

  call write_rows(unit, 'after')
  close(unit)

  print '(A,1X,A)', 'NSSL2_INITIAL_STATE_ORACLE_COMPLETE', &
       trim(output_path)

contains

  subroutine write_rows(output_unit, phase)
    integer, intent(in) :: output_unit
    character(len=*), intent(in) :: phase
    integer :: ii, kk, current_case

    do kk = 1, nz
       do ii = 1, nx
          current_case = mod((kk - 1) * nx + ii - 1, 8)
          write(output_unit,'(A,",",I0,2(",",I0),17(",",ES24.16E3))') &
               trim(phase), current_case, ii, kk, density(ii,kk),      &
               qv(ii,kk), qc(ii,kk), qr(ii,kk), qi(ii,kk), qs(ii,kk), &
               qg(ii,kk), qh(ii,kk), nc(ii,kk), nr(ii,kk), ni(ii,kk), &
               ns(ii,kk), ng(ii,kk), nh(ii,kk), qnn(ii,kk),           &
               volg(ii,kk), volh(ii,kk)
       enddo
    enddo
  end subroutine write_rows
end program nssl2_initial_state_oracle
