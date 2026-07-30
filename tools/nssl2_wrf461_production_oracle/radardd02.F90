program nssl2_radardd02_oracle
  use module_mp_nssl_2mom, only: nssl_2mom_init, radardd02
  implicit none

  integer, parameter :: nx = 4, ny = 1, nz = 16, na = 40
  integer, parameter :: lr = 4, li = 5, ls = 6, lh = 7, lhl = 8
  integer, parameter :: lnr = 11, lni = 12, lns = 13
  integer, parameter :: lnh = 14, lnhl = 15, lvh = 16, lvhl = 17
  integer :: i, k, unit, case_id
  real :: nssl_params(20), an(nx,ny,nz,na), density(nx,ny,nz)
  real :: temperature(nx,ny,nz), dbz(nx,ny,nz)
  real :: qr, qi, qs, qg, qh, nr, ni, ns, ng, nh, volg, volh
  real :: rho, dg, dh
  character(len=512) :: output_path

  call get_command_argument(1, output_path)
  if (len_trim(output_path) == 0) then
     error stop 'usage: nssl2_radardd02_oracle OUTPUT.csv'
  endif

  nssl_params = 0.0
  nssl_params(1:10) = [0.5e9, 0.0, 1.0, 4.0e5, 4.0e4, &
                       8.0e5, 3.0e6, 500.0, 900.0, 100.0]
  call nssl_2mom_init(nssl_params=nssl_params, ipctmp=5, mixphase=0, &
       nssl_density_on=.true., nssl_hail_on=.true.,                 &
       nssl_ccn_on=.true., nssl_icdx=6, nssl_icdxhl=6)

  an = 0.0
  do k = 1, nz
     do i = 1, nx
        case_id = mod((k-1)*nx+i-1, 16)
        rho = 1.25 - 0.12*real(mod(k-1,4)) - 0.01*real(i-1)
        density(i,1,k) = rho
        temperature(i,1,k) = 238.0 + 11.0*real(mod(k-1,4)) + real(i-1)
        qr = 0.0; qi = 0.0; qs = 0.0; qg = 0.0; qh = 0.0
        nr = 0.0; ni = 0.0; ns = 0.0; ng = 0.0; nh = 0.0
        volg = 0.0; volh = 0.0
        dg = 3.0e-3; dh = 8.0e-3

        select case (case_id)
        case (0)
           continue
        case (1)
           qr = 8.0e-4; nr = qr/(1000.0*0.523599*(1.2e-3)**3)
        case (2)
           qi = 2.0e-4; ni = qi/(900.0*0.523599*(80.0e-6)**3)
        case (3)
           qs = 6.0e-4; ns = qs/(100.0*0.523599*(1.5e-3)**3)
        case (4)
           qg = 1.2e-3; ng = qg/(170.0*0.523599*dg**3); volg = qg/170.0
        case (5)
           qg = 1.2e-3; ng = qg/(500.0*0.523599*dg**3); volg = qg/500.0
        case (6)
           qh = 1.5e-3; nh = qh/(500.0*0.523599*dh**3); volh = qh/500.0
        case (7)
           qh = 1.5e-3; nh = qh/(900.0*0.523599*dh**3); volh = qh/900.0
        case (8)
           qr = 1.0e-3; nr = qr/(1000.0*0.523599*(1.5e-3)**3)
           qi = 1.0e-4; ni = qi/(900.0*0.523599*(100.0e-6)**3)
           qs = 4.0e-4; ns = qs/(100.0*0.523599*(2.0e-3)**3)
           qg = 8.0e-4; ng = qg/(350.0*0.523599*(4.0e-3)**3); volg = qg/350.0
           qh = 9.0e-4; nh = qh/(800.0*0.523599*(10.0e-3)**3); volh = qh/800.0
        case (9)
           qr = 2.0e-4; nr = qr/(1000.0*0.523599*(300.0e-6)**3)
           qs = 1.0e-3; ns = qs/(100.0*0.523599*(4.0e-3)**3)
        case (10)
           qi = 5.0e-4; ni = qi/(900.0*0.523599*(300.0e-6)**3)
           qg = 2.0e-3; ng = qg/(220.0*0.523599*(8.0e-3)**3); volg = qg/220.0
           qh = 2.0e-3; nh = qh/(650.0*0.523599*(15.0e-3)**3); volh = qh/650.0
        case (11)
           qr = 5.0e-4; nr = 1.0e-6
           qs = 5.0e-4; ns = 1.0e-6
           qg = 5.0e-4; ng = 1.0e-6; volg = qg/170.0
           qh = 5.0e-4; nh = 1.0e-6; volh = qh/900.0
        case (12)
           qr = 1.0e-12; nr = 1.0e5
           qi = 1.0e-13; ni = 1.0e5
           qs = 1.0e-13; ns = 1.0e5
           qg = 1.0e-12; ng = 1.0e5; volg = qg/500.0
           qh = 1.0e-12; nh = 1.0e5; volh = qh/900.0
        case (13)
           qr = 1.0e-2; nr = qr/(1000.0*0.523599*(5.0e-3)**3)
           qs = 8.0e-3; ns = qs/(100.0*0.523599*(9.0e-3)**3)
           qg = 1.0e-2; ng = qg/(600.0*0.523599*(18.0e-3)**3); volg = qg/600.0
           qh = 1.5e-2; nh = qh/(900.0*0.523599*(35.0e-3)**3); volh = qh/900.0
        case (14)
           qg = 1.0e-3; ng = qg/(500.0*0.523599*(5.0e-3)**3); volg = 0.0
           qh = 1.0e-3; nh = qh/(900.0*0.523599*(9.0e-3)**3); volh = 0.0
        case default
           qr = 7.0e-4; nr = qr/(1000.0*0.523599*(900.0e-6)**3)
           qi = 7.0e-5; ni = qi/(900.0*0.523599*(60.0e-6)**3)
           qs = 3.0e-4; ns = qs/(100.0*0.523599*(800.0e-6)**3)
           qg = 4.0e-4; ng = qg/(275.0*0.523599*(2.0e-3)**3); volg = qg/275.0
           qh = 5.0e-4; nh = qh/(725.0*0.523599*(6.0e-3)**3); volh = qh/725.0
        end select

        an(i,1,k,lr) = qr
        an(i,1,k,li) = qi
        an(i,1,k,ls) = qs
        an(i,1,k,lh) = qg
        an(i,1,k,lhl) = qh
        an(i,1,k,lnr) = nr*rho
        an(i,1,k,lni) = ni*rho
        an(i,1,k,lns) = ns*rho
        an(i,1,k,lnh) = ng*rho
        an(i,1,k,lnhl) = nh*rho
        an(i,1,k,lvh) = volg*rho
        an(i,1,k,lvhl) = volh*rho
     enddo
  enddo

  call radardd02(nx,ny,nz,0,na,an,temperature,dbz,density,nz, &
       4.0e5,500.0,5,nz,0)

  open(newunit=unit, file=trim(output_path), status='replace', action='write')
  write(unit,'(A)') 'case,i,k,rho_kg_m3,temperature_k,qr,qi,qs,qg,qh,nr_per_kg,ni_per_kg,ns_per_kg,ng_per_kg,nh_per_kg,qvolg_m3_per_kg,qvolh_m3_per_kg,refl_10cm_dbz'
  do k = 1, nz
     do i = 1, nx
        case_id = mod((k-1)*nx+i-1, 16)
        rho = density(i,1,k)
        write(unit,'(3(I0,","),14(ES24.16E3,","),ES24.16E3)') &
             case_id, i, k, rho, temperature(i,1,k), &
             an(i,1,k,lr), an(i,1,k,li), an(i,1,k,ls), &
             an(i,1,k,lh), an(i,1,k,lhl), an(i,1,k,lnr)/rho, &
             an(i,1,k,lni)/rho, an(i,1,k,lns)/rho, &
             an(i,1,k,lnh)/rho, an(i,1,k,lnhl)/rho, &
             an(i,1,k,lvh)/rho, an(i,1,k,lvhl)/rho, dbz(i,1,k)
     enddo
  enddo
  close(unit)

  print '(A,1X,A)', 'NSSL2_RADARDD02_ORACLE_COMPLETE', trim(output_path)
end program nssl2_radardd02_oracle
