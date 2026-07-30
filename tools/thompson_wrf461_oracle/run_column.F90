program run_thompson_column
  use module_mp_thompson, only: thompson_init, mp_gt_driver, RSLF, RSIF
  implicit none

  integer, parameter :: nx = 2, ny = 2, nz = 24
  integer :: i, j, k, scenario_id, ios
  real, parameter :: p0 = 100000.0, rd_over_cp = 287.0 / 1004.0
  real, parameter :: re_qc_bg = 2.49e-6
  real, parameter :: re_qi_bg = 4.99e-6
  real, parameter :: re_qs_bg = 9.99e-6
  real :: z(nz), temperature(nz), dt
  real :: hgt(nx,nz,ny)
  real :: qv(nx,nz,ny), qc(nx,nz,ny), qr(nx,nz,ny)
  real :: qi(nx,nz,ny), qs(nx,nz,ny), qg(nx,nz,ny)
  real :: ni(nx,nz,ny), nr(nx,nz,ny), th(nx,nz,ny)
  real :: pii(nx,nz,ny), p(nx,nz,ny), w(nx,nz,ny), dz(nx,nz,ny)
  real :: refl(nx,nz,ny), re_cloud(nx,nz,ny)
  real :: re_ice(nx,nz,ny), re_snow(nx,nz,ny)
  real :: rainnc(nx,ny), rainncv(nx,ny), snownc(nx,ny)
  real :: snowncv(nx,ny), graupelnc(nx,ny), graupelncv(nx,ny)
  real :: sr(nx,ny)
  real :: qv_before(nz), qc_before(nz), qr_before(nz)
  real :: qi_before(nz), qs_before(nz), qg_before(nz)
  real :: ni_before(nz), nr_before(nz), th_before(nz)
  character(len=32) :: scenario
  character(len=512) :: output_dir, column_path, surface_path

  call get_command_argument(1, scenario)
  call get_command_argument(2, output_dir)
  if (len_trim(scenario) == 0 .or. len_trim(output_dir) == 0) then
     error stop 'usage: run_column SCENARIO OUTPUT_DIRECTORY'
  endif
  select case (trim(scenario))
  case ('warm')
     scenario_id = 1
  case ('mixed')
     scenario_id = 2
  case ('ice')
     scenario_id = 3
  case ('condense')
     scenario_id = 4
  case ('rain-sed')
     scenario_id = 5
  case ('ice-sed')
     scenario_id = 6
  case ('cloud-sed')
     scenario_id = 7
  case ('snow-sed')
     scenario_id = 8
  case ('graupel-sed')
     scenario_id = 9
  case ('warm-auto')
     scenario_id = 10
  case ('rain-self')
     scenario_id = 11
  case ('warm-accrete')
     scenario_id = 12
  case ('ice-auto')
     scenario_id = 13
  case ('rain-evap')
     scenario_id = 14
  case ('snow-subl')
     scenario_id = 15
  case ('graupel-subl')
     scenario_id = 16
  case ('ice-dep')
     scenario_id = 17
  case ('ice-nuc')
     scenario_id = 18
  case ('snow-dep')
     scenario_id = 19
  case ('graupel-dep')
     scenario_id = 20
  case ('snow-melt')
     scenario_id = 21
  case ('graupel-melt')
     scenario_id = 22
  case ('rain-freeze')
     scenario_id = 23
  case ('cloud-freeze')
     scenario_id = 24
  case ('graupel-rime')
     scenario_id = 25
  case ('snow-rime')
     scenario_id = 26
  case ('snow-ice')
     scenario_id = 27
  case ('rain-ice')
     scenario_id = 28
  case ('rain-snow')
     scenario_id = 29
  case ('rain-graupel')
     scenario_id = 30
  case ('snow-rime-convert')
     scenario_id = 31
  case ('warm-overlap')
     scenario_id = 32
  case ('rain-snow-graupel-overlap')
     scenario_id = 33
  case ('rain-ice-graupel-overlap')
     scenario_id = 34
  case ('frozen-vapor-overlap')
     scenario_id = 35
  case ('cold-cloud-overlap')
     scenario_id = 36
  case ('frozen-vapor-nucleation-overlap')
     scenario_id = 37
  case ('cold-ice-rain-overlap')
     scenario_id = 38
  case ('cold-full-overlap')
     scenario_id = 39
  case ('cold-cloud-rain-overlap')
     scenario_id = 40
  case ('cloud-condense-sed')
     scenario_id = 41
  case ('cloud-condense-nofall')
     scenario_id = 42
  case ('cloud-rain-condense-sed')
     scenario_id = 43
  case ('cloud-rain-condense-nofall')
     scenario_id = 44
  case ('condense-fall-attempt')
     scenario_id = 45
  case ('warm-frozen-subsat')
     scenario_id = 46
  case default
     error stop 'unknown Thompson column scenario'
  end select

  if (scenario_id == 37 .or. scenario_id == 38 .or. &
      scenario_id == 39 .or. scenario_id == 40 .or. &
      scenario_id == 46) then
     ! Interaction-forcing source steps expose simultaneous-process
     ! conservation groups without relying on a long forecast to amplify
     ! ordering drift.
     dt = 50.0
  else
     dt = 10.0
  endif
  do k = 1, nz
     z(k) = (real(k) - 0.5) * 500.0
     if (scenario_id == 23) then
        temperature(k) = 230.0
     elseif (scenario_id == 18) then
        temperature(k) = 240.0
     elseif (scenario_id == 28 .or. scenario_id == 29 .or. &
             scenario_id == 30 .or. scenario_id == 33) then
        temperature(k) = 273.149
     elseif (scenario_id == 34) then
        temperature(k) = 253.15
     elseif (scenario_id == 21 .or. scenario_id == 22 .or. &
              scenario_id == 46) then
        temperature(k) = 274.15
     elseif (scenario_id == 24 .or. scenario_id == 37 .or. &
             scenario_id == 39 .or. scenario_id == 40) then
        temperature(k) = 240.0
     elseif (scenario_id == 25 .or. scenario_id == 26 .or. &
             scenario_id == 31 .or. scenario_id == 36 .or. &
             scenario_id == 38) then
        temperature(k) = 268.15
     elseif (scenario_id == 4 .or. scenario_id == 45 .or. &
         scenario_id == 5 .or. scenario_id == 7 .or. &
         scenario_id == 10 .or. scenario_id == 11 .or. scenario_id == 12 .or. &
         scenario_id == 14 .or. scenario_id == 32 .or. &
         scenario_id == 41 .or. scenario_id == 42 .or. &
         scenario_id == 43 .or. scenario_id == 44) then
       temperature(k) = 285.0
     elseif (scenario_id == 6 .or. scenario_id == 8 .or. scenario_id == 9 .or. &
             scenario_id == 13 .or. scenario_id == 15 .or. &
             scenario_id == 16 .or. scenario_id == 17 .or. &
             scenario_id == 27 .or. &
             scenario_id == 19 .or. scenario_id == 20 .or. &
             scenario_id == 35) then
        temperature(k) = 260.0
     else
        temperature(k) = max(205.0, 292.0 - 0.0065 * z(k))
     endif
  enddo

  qv = 0.0
  qc = 0.0
  qr = 0.0
  qi = 0.0
  qs = 0.0
  qg = 0.0
  ni = 0.0
  nr = 0.0
  th = 0.0
  pii = 0.0
  p = 0.0
  w = 0.0
  dz = 500.0
  ! Source-isolation geometry: retain sedimentation calls while making their
  ! transport negligible compared with the interacting source tendencies.
  if (scenario_id == 37 .or. scenario_id == 38 .or. &
      scenario_id == 39 .or. scenario_id == 40 .or. &
      scenario_id == 46) dz = 1.0e8
  ! An adversarially shallow layer amplifies the otherwise subtle distinction
  ! between WRF's held pre-adjustment cloud mass and current-density tendency
  ! conversion without activating another source process.
  if (scenario_id == 41 .or. scenario_id == 42 .or. &
      scenario_id == 43 .or. scenario_id == 44) dz = 1.0
  refl = -35.0
  re_cloud = re_qc_bg
  re_ice = re_qi_bg
  re_snow = re_qs_bg
  rainnc = 0.0
  rainncv = 0.0
  snownc = 0.0
  snowncv = 0.0
  graupelnc = 0.0
  graupelncv = 0.0
  sr = 0.0

  do j = 1, ny
     do i = 1, nx
        do k = 1, nz
           hgt(i,k,j) = z(k)
           p(i,k,j) = p0 * exp(-z(k) / 8000.0)
           pii(i,k,j) = (p(i,k,j) / p0) ** rd_over_cp
           th(i,k,j) = temperature(k) / pii(i,k,j)
           if (scenario_id == 4 .or. scenario_id == 45) then
              qv(i,k,j) = 0.012
           elseif (scenario_id == 5) then
              qv(i,k,j) = RSLF(p(i,k,j), temperature(k))
           elseif (scenario_id == 6 .or. scenario_id == 8 .or. &
                   scenario_id == 9 .or. scenario_id == 13 .or. &
                   scenario_id == 23 .or. scenario_id == 24 .or. &
                   scenario_id == 25 .or. scenario_id == 26 .or. &
                   scenario_id == 27 .or. scenario_id == 28 .or. &
                   scenario_id == 29 .or. scenario_id == 30 .or. &
                   scenario_id == 33 .or. scenario_id == 34 .or. &
                   scenario_id == 36) then
              qv(i,k,j) = RSIF(p(i,k,j), temperature(k))
           elseif (scenario_id == 37) then
              ! Activate Cooper deposition nucleation together with ice and
              ! snow deposition, forcing WRF's shared vapor limiter.
              if (z(k) >= 1250.0 .and. z(k) <= 4250.0) then
                 qv(i,k,j) = 1.30 * RSIF(p(i,k,j), temperature(k))
              else
                 qv(i,k,j) = RSIF(p(i,k,j), temperature(k))
              endif
           elseif (scenario_id == 39 .or. scenario_id == 40) then
              ! Drive the full cold rain/frozen source network through the
              ! common vapor and category conservation passes.
              if (z(k) >= 1250.0 .and. z(k) <= 4250.0) then
                 qv(i,k,j) = 1.30 * RSIF(p(i,k,j), temperature(k))
              else
                 qv(i,k,j) = RSIF(p(i,k,j), temperature(k))
              endif
           elseif (scenario_id == 38) then
              ! Start liquid-saturated while retaining cold ice deposition;
              ! deposition can activate ordinary post-source rain evaporation.
              qv(i,k,j) = RSLF(p(i,k,j), temperature(k))
           elseif (scenario_id == 7) then
              qv(i,k,j) = RSLF(p(i,k,j), temperature(k))
           elseif (scenario_id == 41 .or. scenario_id == 42 .or. &
                   scenario_id == 43 .or. scenario_id == 44) then
              qv(i,k,j) = 1.0008 * RSLF(p(i,k,j), temperature(k))
           elseif (scenario_id == 10) then
              qv(i,k,j) = RSLF(p(i,k,j), temperature(k))
           elseif (scenario_id == 11) then
              qv(i,k,j) = RSLF(p(i,k,j), temperature(k))
           elseif (scenario_id == 12 .or. scenario_id == 32) then
              qv(i,k,j) = RSLF(p(i,k,j), temperature(k))
           elseif (scenario_id == 14) then
              ! Keep the column below water saturation so this vector
              ! activates ordinary rain evaporation before rain fallout.
              qv(i,k,j) = 0.995 * RSLF(p(i,k,j), temperature(k))
           elseif (scenario_id == 46) then
              ! Ambient temperature is above freezing, but the large liquid-
              ! saturation deficit drives wet-bulb below freezing through
              ! part of the column.  Snow and graupel therefore exercise the
              ! coupled warm melt/sublimation branches in one source call.
              qv(i,k,j) = 0.90 * RSLF(p(i,k,j), temperature(k))
           elseif (scenario_id == 21 .or. scenario_id == 22) then
              ! Liquid saturation keeps the warm melt vectors out of the
              ! vapor-limited branch and makes WRF's wet-bulb temperature
              ! equal the supplied temperature.
              qv(i,k,j) = RSLF(p(i,k,j), temperature(k))
           elseif (scenario_id == 15 .or. scenario_id == 16) then
              ! Keep the cold column just below ice saturation so it
              ! activates frozen-hydrometeor sublimation without deposition
              ! nucleation.
              qv(i,k,j) = 0.995 * RSIF(p(i,k,j), temperature(k))
           elseif (scenario_id == 35) then
              ! A shallow ice deficit activates simultaneous ice, snow, and
              ! graupel sublimation while forcing the shared vapor limiter.
              qv(i,k,j) = 0.9999 * RSIF(p(i,k,j), temperature(k))
           elseif (scenario_id == 31) then
              ! Keep deposition positive but small enough that riming exceeds
              ! it by more than the conversion branch's factor-of-two gate.
              qv(i,k,j) = 1.0001 * RSIF(p(i,k,j), temperature(k))
           elseif (scenario_id == 17 .or. scenario_id == 19 .or. &
                   scenario_id == 20) then
              ! A small ice supersaturation activates cloud-ice deposition
              ! without crossing Thompson's 25-percent deposition-nucleation
              ! threshold or reaching water saturation.
              qv(i,k,j) = 1.01 * RSIF(p(i,k,j), temperature(k))
           elseif (scenario_id == 18) then
              ! Thirty-percent ice supersaturation at 240 K activates the
              ! classic Cooper deposition-nucleation branch while remaining
              ! below liquid-water saturation.
              qv(i,k,j) = 1.30 * RSIF(p(i,k,j), temperature(k))
           else
              qv(i,k,j) = max(1.0e-5, 0.014 * exp(-z(k) / 2500.0))
           endif
        enddo
     enddo
  enddo

  call initialize_case(scenario_id, z, qv, qc, qr, qi, qs, qg, ni, nr, w)
  qv_before = qv(1,:,1)
  qc_before = qc(1,:,1)
  qr_before = qr(1,:,1)
  qi_before = qi(1,:,1)
  qs_before = qs(1,:,1)
  qg_before = qg(1,:,1)
  ni_before = ni(1,:,1)
  nr_before = nr(1,:,1)
  th_before = th(1,:,1)

  call thompson_init(                                                 &
       hgt=hgt,                                                       &
       ids=1, ide=2, jds=1, jde=2, kds=1, kde=nz,                   &
       ims=1, ime=nx, jms=1, jme=ny, kms=1, kme=nz,                 &
       its=1, ite=nx, jts=1, jte=ny, kts=1, kte=nz)

  call mp_gt_driver(                                                  &
       qv=qv, qc=qc, qr=qr, qi=qi, qs=qs, qg=qg, ni=ni, nr=nr,      &
       aer_init_opt=0, wif_input_opt=0,                              &
       th=th, pii=pii, p=p, w=w, dz=dz,                             &
       dt_in=dt, itimestep=1,                                        &
       RAINNC=rainnc, RAINNCV=rainncv,                              &
       SNOWNC=snownc, SNOWNCV=snowncv,                              &
       GRAUPELNC=graupelnc, GRAUPELNCV=graupelncv, SR=sr,           &
       refl_10cm=refl, diagflag=.true., ke_diag=nz, do_radar_ref=1, &
       re_cloud=re_cloud, re_ice=re_ice, re_snow=re_snow,           &
       has_reqc=1, has_reqi=1, has_reqs=1,                          &
       ids=1, ide=2, jds=1, jde=2, kds=1, kde=nz,                   &
       ims=1, ime=nx, jms=1, jme=ny, kms=1, kme=nz,                 &
       its=1, ite=nx, jts=1, jte=ny, kts=1, kte=nz)

  column_path = trim(output_dir) // '/' // trim(scenario) // '-column.csv'
  surface_path = trim(output_dir) // '/' // trim(scenario) // '-surface.csv'
  open(newunit=ios, file=trim(column_path), status='replace', action='write')
  write(ios,'(A)') 'phase,k,z_m,p_pa,pii,w_m_s,dz_m,theta_k,temp_k,qv,qc,qr,qi,qs,qg,ni_per_kg,nr_per_kg,effc_m,effi_m,effs_m,refl_dbz'
  do k = 1, nz
     call write_row(ios, 'before', k, z(k), p(1,k,1), pii(1,k,1),      &
          w(1,k,1), dz(1,k,1), th_before(k), qv_before(k),            &
          qc_before(k), qr_before(k), qi_before(k), qs_before(k),     &
          qg_before(k), ni_before(k), nr_before(k), re_qc_bg,         &
          re_qi_bg, re_qs_bg, -35.0)
  enddo
  do k = 1, nz
     call write_row(ios, 'after', k, z(k), p(1,k,1), pii(1,k,1),       &
          w(1,k,1), dz(1,k,1), th(1,k,1), qv(1,k,1), qc(1,k,1),      &
          qr(1,k,1), qi(1,k,1), qs(1,k,1), qg(1,k,1), ni(1,k,1),     &
          nr(1,k,1), re_cloud(1,k,1), re_ice(1,k,1),                 &
          re_snow(1,k,1), refl(1,k,1))
  enddo
  close(ios)

  open(newunit=ios, file=trim(surface_path), status='replace', action='write')
  write(ios,'(A)') 'scenario,dt_s,rainnc_mm,rainncv_mm,snownc_mm,snowncv_mm,graupelnc_mm,graupelncv_mm,sr'
  write(ios,'(A,8(",",ES24.16E3))') trim(scenario), dt, rainnc(1,1), &
       rainncv(1,1), snownc(1,1), snowncv(1,1), graupelnc(1,1),       &
       graupelncv(1,1), sr(1,1)
  close(ios)

  print '(A,1X,A)', 'THOMPSON_COLUMN_ORACLE_COMPLETE', trim(scenario)

contains

  subroutine initialize_case(which, height, qv3, qc3, qr3, qi3, qs3, &
                             qg3, ni3, nr3, w3)
    integer, intent(in) :: which
    real, intent(in) :: height(nz)
    real, intent(inout) :: qv3(nx,nz,ny), qc3(nx,nz,ny)
    real, intent(inout) :: qr3(nx,nz,ny), qi3(nx,nz,ny)
    real, intent(inout) :: qs3(nx,nz,ny), qg3(nx,nz,ny)
    real, intent(inout) :: ni3(nx,nz,ny), nr3(nx,nz,ny)
    real, intent(inout) :: w3(nx,nz,ny)
    real :: cloud, rain, ice, snow, graupel, velocity
    integer :: ii, jj, kk

    do jj = 1, ny
       do ii = 1, nx
          do kk = 1, nz
             select case (which)
             case (1)
                cloud = 9.0e-4 * exp(-((height(kk)-1750.0)/900.0)**2)
                rain = 3.5e-4 * exp(-((height(kk)-1000.0)/850.0)**2)
                ice = 0.0
                snow = 0.0
                graupel = 0.0
                velocity = 1.8 * exp(-((height(kk)-1800.0)/1200.0)**2)
             case (2)
                cloud = 5.0e-4 * exp(-((height(kk)-3500.0)/1800.0)**2)
                rain = 2.5e-4 * exp(-((height(kk)-1800.0)/1300.0)**2)
                ice = 1.5e-4 * exp(-((height(kk)-6500.0)/1900.0)**2)
                snow = 4.0e-4 * exp(-((height(kk)-6000.0)/1800.0)**2)
                graupel = 2.0e-4 * exp(-((height(kk)-4500.0)/1400.0)**2)
                velocity = 4.0 * exp(-((height(kk)-4500.0)/2200.0)**2)
             case (3)
                cloud = 0.0
                rain = 0.0
                ice = 2.0e-4 * exp(-((height(kk)-8500.0)/1900.0)**2)
                snow = 1.2e-4 * exp(-((height(kk)-9000.0)/2200.0)**2)
                graupel = 0.0
                velocity = 0.8 * exp(-((height(kk)-8500.0)/1800.0)**2)
             case (4, 45)
                cloud = 0.0
                rain = 0.0
                ice = 0.0
                snow = 0.0
                graupel = 0.0
                ! Scenario 45 deliberately permits the velocity criterion,
                ! but held ANY(L_qc) remains false because the column entered
                ! cloud-free; its result must equal scenario 4 exactly.
                if (which == 45) then
                   velocity = 0.0
                else
                   velocity = 1.0
                endif
             case (5)
                cloud = 0.0
                if (height(kk) <= 4250.0) then
                   rain = 3.0e-4 * exp(-((height(kk)-1000.0)/1100.0)**2)
                else
                   rain = 0.0
                endif
                ice = 0.0
                snow = 0.0
                graupel = 0.0
                velocity = 1.0
             case (6)
                cloud = 0.0
                rain = 0.0
                if (height(kk) <= 4250.0) then
                   ice = 5.0e-7 * exp(-((height(kk)-1000.0)/1100.0)**2)
                else
                   ice = 0.0
                endif
                snow = 0.0
                graupel = 0.0
                velocity = 1.0
             case (7)
                if (kk == 2) then
                   cloud = 8.0e-6
                else
                   cloud = 0.0
                endif
                rain = 0.0
                ice = 0.0
                snow = 0.0
                graupel = 0.0
                velocity = 0.0
             case (8)
                cloud = 0.0
                rain = 0.0
                ice = 0.0
                if (height(kk) <= 4250.0) then
                   snow = 2.0e-4 * exp(-((height(kk)-1000.0)/1100.0)**2)
                else
                   snow = 0.0
                endif
                graupel = 0.0
                velocity = 1.0
             case (9)
                cloud = 0.0
                rain = 0.0
                ice = 0.0
                snow = 0.0
                if (height(kk) <= 4250.0) then
                   graupel = 2.0e-4 * exp(-((height(kk)-1000.0)/1100.0)**2)
                else
                   graupel = 0.0
                endif
                velocity = 1.0
             case (10)
                if (height(kk) <= 4250.0) then
                   cloud = 1.0e-3 * exp(-((height(kk)-1250.0)/1100.0)**2)
                else
                   cloud = 0.0
                endif
                rain = 0.0
                ice = 0.0
                snow = 0.0
                graupel = 0.0
                ! Disable cloud-droplet fallout while rain generated by
                ! autoconversion follows WRF's normal sedimentation path.
                velocity = 1.0
             case (11)
                cloud = 0.0
                if (height(kk) <= 4250.0) then
                   rain = 3.0e-4 * exp(-((height(kk)-1000.0)/1100.0)**2)
                else
                   rain = 0.0
                endif
                ice = 0.0
                snow = 0.0
                graupel = 0.0
                velocity = 1.0
             case (12)
                if (height(kk) <= 4250.0) then
                   cloud = 6.0e-6 * exp(-((height(kk)-1250.0)/1100.0)**2)
                   rain = 3.0e-4 * exp(-((height(kk)-1000.0)/1100.0)**2)
                else
                   cloud = 0.0
                   rain = 0.0
                endif
                ice = 0.0
                snow = 0.0
                graupel = 0.0
                velocity = 1.0
             case (13)
                cloud = 0.0
                rain = 0.0
                if (kk == 2) then
                   ice = 2.0e-5
                else
                   ice = 0.0
                endif
                snow = 0.0
                graupel = 0.0
                velocity = 1.0
             case (14)
                cloud = 0.0
                if (height(kk) <= 4250.0) then
                   rain = 3.0e-4 * exp(-((height(kk)-1000.0)/1100.0)**2)
                else
                   rain = 0.0
                endif
                ice = 0.0
                snow = 0.0
                graupel = 0.0
                velocity = 1.0
             case (15)
                cloud = 0.0
                rain = 0.0
                ice = 0.0
                if (height(kk) <= 4250.0) then
                   snow = 2.0e-4 * exp(-((height(kk)-1000.0)/1100.0)**2)
                else
                   snow = 0.0
                endif
                graupel = 0.0
                velocity = 1.0
             case (16)
                cloud = 0.0
                rain = 0.0
                ice = 0.0
                snow = 0.0
                if (height(kk) <= 4250.0) then
                   graupel = 2.0e-4 * exp(-((height(kk)-1000.0)/1100.0)**2)
                else
                   graupel = 0.0
                endif
                velocity = 1.0
             case (17)
                cloud = 0.0
                rain = 0.0
                if (height(kk) <= 4250.0) then
                   ice = 5.0e-7 * exp(-((height(kk)-1000.0)/1100.0)**2)
                else
                   ice = 0.0
                endif
                snow = 0.0
                graupel = 0.0
                velocity = 1.0
             case (18)
                cloud = 0.0
                rain = 0.0
                ice = 0.0
                snow = 0.0
                graupel = 0.0
                velocity = 1.0
             case (19)
                cloud = 0.0
                rain = 0.0
                ice = 0.0
                if (height(kk) <= 4250.0) then
                   snow = 2.0e-4 * exp(-((height(kk)-1000.0)/1100.0)**2)
                else
                   snow = 0.0
                endif
                graupel = 0.0
                velocity = 1.0
             case (20)
                cloud = 0.0
                rain = 0.0
                ice = 0.0
                snow = 0.0
                if (height(kk) <= 4250.0) then
                   graupel = 2.0e-4 * exp(-((height(kk)-1000.0)/1100.0)**2)
                else
                   graupel = 0.0
                endif
                velocity = 1.0
             case (21)
                cloud = 0.0
                rain = 0.0
                ice = 0.0
                if (height(kk) <= 4250.0) then
                   snow = 2.0e-4 * exp(-((height(kk)-1000.0)/1100.0)**2)
                else
                   snow = 0.0
                endif
                graupel = 0.0
                velocity = 1.0
             case (22)
                cloud = 0.0
                rain = 0.0
                ice = 0.0
                snow = 0.0
                if (height(kk) <= 4250.0) then
                   graupel = 2.0e-4 * exp(-((height(kk)-1000.0)/1100.0)**2)
                else
                   graupel = 0.0
                endif
                velocity = 1.0
             case (23)
                cloud = 0.0
                if (height(kk) <= 4250.0) then
                   rain = 3.0e-4 * exp(-((height(kk)-1000.0)/1100.0)**2)
                else
                   rain = 0.0
                endif
                ice = 0.0
                snow = 0.0
                graupel = 0.0
                velocity = 1.0
             case (24)
                if (height(kk) <= 4250.0) then
                   cloud = 3.0e-4 * exp(-((height(kk)-1000.0)/1100.0)**2)
                else
                   cloud = 0.0
                endif
                rain = 0.0
                ice = 0.0
                snow = 0.0
                graupel = 0.0
                ! Cloud sedimentation is gated off when w >= 0.1 m/s.  Ice
                ! saturation avoids deposition nucleation while retaining the
                ! real post-freezing cloud-evaporation branch.
                velocity = 1.0
             case (25)
                if (height(kk) <= 4250.0) then
                   cloud = 3.0e-4 * exp(-((height(kk)-1000.0)/1100.0)**2)
                   graupel = 2.0e-4 * exp(-((height(kk)-1200.0)/1200.0)**2)
                else
                   cloud = 0.0
                   graupel = 0.0
                endif
                rain = 0.0
                ice = 0.0
                snow = 0.0
                ! The -5 C column activates graupel-cloud riming and the
                ! peak Hallett-Mossop splinter multiplier.  Ice saturation
                ! suppresses unrelated vapor deposition.
                velocity = 1.0
             case (26)
                if (height(kk) <= 4250.0) then
                   cloud = 3.0e-4 * exp(-((height(kk)-1000.0)/1100.0)**2)
                   snow = 2.0e-4 * exp(-((height(kk)-1200.0)/1200.0)**2)
                else
                   cloud = 0.0
                   snow = 0.0
                endif
                rain = 0.0
                ice = 0.0
                graupel = 0.0
                ! Ice saturation suppresses snow vapor exchange, leaving
                ! snow-cloud riming, cloud evaporation, and fallout.
                velocity = 1.0
             case (27)
                cloud = 0.0
                rain = 0.0
                if (height(kk) <= 4250.0) then
                   ice = 5.0e-7 * exp(-((height(kk)-1000.0)/1100.0)**2)
                   snow = 2.0e-4 * exp(-((height(kk)-1200.0)/1200.0)**2)
                else
                   ice = 0.0
                   snow = 0.0
                endif
                graupel = 0.0
                ! Ice saturation suppresses vapor exchange.  A 29-micron
                ! ice diameter remains below autoconversion while exposing
                ! snow collection of cloud ice and both fallout paths.
                velocity = 1.0
             case (28)
                cloud = 0.0
                if (height(kk) <= 4250.0) then
                   rain = 3.0e-4 * exp(-((height(kk)-1000.0)/1100.0)**2)
                   ice = 1.0e-8 * exp(-((height(kk)-1000.0)/1100.0)**2)
                else
                   rain = 0.0
                   ice = 0.0
                endif
                snow = 0.0
                graupel = 0.0
                ! A 60-micron rain MVD exceeds four times Thompson's
                ! minimum ice diameter.  Near-freezing ice saturation
                ! minimizes unrelated rain evaporation and freezing.
                velocity = 1.0
             case (29)
                cloud = 0.0
                if (height(kk) <= 4250.0) then
                   rain = 3.0e-4 * exp(-((height(kk)-1000.0)/1100.0)**2)
                   snow = 2.0e-4 * exp(-((height(kk)-1200.0)/1200.0)**2)
                else
                   rain = 0.0
                   snow = 0.0
                endif
                ice = 0.0
                graupel = 0.0
                ! A near-freezing, ice-saturated column suppresses vapor
                ! exchange and minimizes rain freezing while retaining the
                ! cold rain/snow collision tables and both entry-time fallout
                ! paths.  Collision-created graupel is held until next call.
                velocity = 1.0
             case (30)
                cloud = 0.0
                if (height(kk) <= 4250.0) then
                   rain = 3.0e-4 * exp(-((height(kk)-1000.0)/1100.0)**2)
                   graupel = 2.0e-4 * exp(-((height(kk)-1200.0)/1200.0)**2)
                else
                   rain = 0.0
                   graupel = 0.0
                endif
                ice = 0.0
                snow = 0.0
                ! Near-freezing ice saturation suppresses vapor exchange and
                ! minimizes rain freezing while retaining the cold
                ! rain/graupel collision table and both entry-time fallout
                ! paths.  Classic mp=8 diagnoses 400 kg/m3 graupel.
                velocity = 1.0
             case (31)
                if (height(kk) <= 4250.0) then
                   cloud = 3.0e-4 * exp(-((height(kk)-1000.0)/1100.0)**2)
                   snow = 2.0e-4 * exp(-((height(kk)-1200.0)/1200.0)**2)
                else
                   cloud = 0.0
                   snow = 0.0
                endif
                rain = 0.0
                ice = 0.0
                graupel = 0.0
                ! One-percent ice supersaturation provides a positive snow
                ! deposition denominator while the -5 C cloud/snow column
                ! retains strong riming.  This isolates the partial rimed-
                ! snow conversion, its fall-speed boost, cloud adjustment,
                ! and entry-time (non-falling) graupel product.
                velocity = 1.0
             case (32)
                if (height(kk) <= 4250.0) then
                   ! Both cloud sinks are intentionally active from the same
                   ! incoming state: autoconversion plus rain accretion.
                   cloud = 1.0e-3 * exp(-((height(kk)-1250.0)/1100.0)**2)
                   rain = 3.0e-4 * exp(-((height(kk)-1000.0)/1100.0)**2)
                else
                   cloud = 0.0
                   rain = 0.0
                endif
                ice = 0.0
                snow = 0.0
                graupel = 0.0
                velocity = 1.0
             case (33)
                cloud = 0.0
                if (height(kk) <= 4250.0) then
                   ! Activate the rain/snow and rain/graupel collision
                   ! families from one shared incoming rain state.  This
                   ! exposes WRF's grouped rain limiter and single rain
                   ! self-collection tendency in a production-order slice.
                   rain = 3.0e-4 * exp(-((height(kk)-1000.0)/1100.0)**2)
                   snow = 2.0e-4 * exp(-((height(kk)-1200.0)/1200.0)**2)
                   graupel = 2.0e-4 * exp(-((height(kk)-1200.0)/1200.0)**2)
                else
                   rain = 0.0
                   snow = 0.0
                   graupel = 0.0
                endif
                ice = 0.0
                ! Near-freezing ice saturation suppresses vapor exchange
                ! while all three entry-time fallout paths remain active.
                velocity = 1.0
             case (34)
                cloud = 0.0
                if (height(kk) <= 4250.0) then
                   ! Complete the grouped cold-rain source vector with
                   ! simultaneous table freezing, cloud-ice collection,
                   ! graupel collection, and rain self-collection.
                   rain = 3.0e-4 * exp(-((height(kk)-1000.0)/1100.0)**2)
                   ice = 1.0e-8 * exp(-((height(kk)-1000.0)/1100.0)**2)
                   graupel = 2.0e-4 * exp(-((height(kk)-1200.0)/1200.0)**2)
                else
                   rain = 0.0
                   ice = 0.0
                   graupel = 0.0
                endif
                snow = 0.0
                ! Twenty degrees of supercooling activates Bigg freezing while
                ! ice saturation suppresses unrelated frozen-vapor exchange.
                velocity = 1.0
             case (35)
                cloud = 0.0
                rain = 0.0
                if (height(kk) <= 4250.0) then
                   ! Keep each frozen species below its 1e-6 kg/m3
                   ! collection-table floor while retaining vapor exchange
                   ! and all three entry-time fallout paths.
                   ice = 5.0e-7 * exp(-((height(kk)-1000.0)/1100.0)**2)
                   snow = 5.0e-7 * exp(-((height(kk)-1200.0)/1200.0)**2)
                   graupel = 5.0e-7 * exp(-((height(kk)-1200.0)/1200.0)**2)
                else
                   ice = 0.0
                   snow = 0.0
                   graupel = 0.0
                endif
                velocity = 1.0
             case (36)
                if (height(kk) <= 4250.0) then
                   ! Simultaneously activate cloud autoconversion, table
                   ! freezing, snow riming, graupel riming, and the -5 C
                   ! Hallett-Mossop maximum from one incoming cloud state.
                   cloud = 1.0e-3 * exp(-((height(kk)-1000.0)/1100.0)**2)
                   snow = 2.0e-4 * exp(-((height(kk)-1200.0)/1200.0)**2)
                   graupel = 2.0e-4 * exp(-((height(kk)-1200.0)/1200.0)**2)
                else
                   cloud = 0.0
                   snow = 0.0
                   graupel = 0.0
                endif
                rain = 0.0
                ice = 0.0
                velocity = 1.0
             case (37)
                cloud = 0.0
                rain = 0.0
                if (height(kk) >= 1250.0 .and. height(kk) <= 4250.0) then
                   ! Existing 200-micron ice and snow make autoconversion,
                   ! deposition, and collection overlap Cooper nucleation
                   ! and both caps.
                   ice = 1.0e-8 * exp(-((height(kk)-1000.0)/1100.0)**2)
                   snow = 6.0e-3 * exp(-((height(kk)-1200.0)/1200.0)**2)
                else
                   ice = 0.0
                   snow = 0.0
                endif
                graupel = 0.0
                velocity = 1.0
             case (38)
                cloud = 0.0
                if (height(kk) >= 1250.0 .and. height(kk) <= 4250.0) then
                   ! Large rain drops collect 200-micron ice while ice
                   ! deposition and table autoconversion compete under the
                   ! shared cloud-ice cap. New graupel remains entry-inactive.
                   rain = 3.0e-4 * exp(-((height(kk)-1200.0)/1200.0)**2)
                   ice = 5.0e-7 * exp(-((height(kk)-1000.0)/1100.0)**2)
                else
                   rain = 0.0
                   ice = 0.0
                endif
                snow = 0.0
                graupel = 0.0
                velocity = 1.0
             case (39)
                cloud = 0.0
                if (height(kk) >= 1250.0 .and. height(kk) <= 4250.0) then
                   ! One incoming state activates both complete fused groups:
                   ! nucleation/vapor/autoconversion/snow-ice plus table rain
                   ! freezing and rain/ice/snow/graupel collection.
                   rain = 3.0e-4 * exp(-((height(kk)-1200.0)/1200.0)**2)
                   ice = 1.0e-8 * exp(-((height(kk)-1000.0)/1100.0)**2)
                   snow = 6.0e-3 * exp(-((height(kk)-1200.0)/1200.0)**2)
                   graupel = 2.0e-4 * exp(-((height(kk)-1200.0)/1200.0)**2)
                else
                   rain = 0.0
                   ice = 0.0
                   snow = 0.0
                   graupel = 0.0
                endif
                velocity = 1.0
             case (40)
                if (height(kk) >= 1250.0 .and. height(kk) <= 4250.0) then
                   ! Same incoming state activates cloud autoconversion,
                   ! rain accretion, Bigg cloud/rain freezing, rain/ice
                   ! collection, and Cooper nucleation.  The exact driver
                   ! ordering subtracts both pni_wfz and pni_rfz before
                   ! forming the Cooper crystal target.
                   cloud = 5.0e-4 * exp(-((height(kk)-1200.0)/1200.0)**2)
                   rain = 3.0e-4 * exp(-((height(kk)-1200.0)/1200.0)**2)
                   ice = 1.0e-8 * exp(-((height(kk)-1000.0)/1100.0)**2)
                else
                   cloud = 0.0
                   rain = 0.0
                   ice = 0.0
                endif
                snow = 0.0
                graupel = 0.0
                velocity = 1.0
             case (41, 42)
                if (kk == 2) then
                   ! Retain entry-active cloud while a small supersaturation
                   ! changes temperature, vapor, and density before fallout.
                   ! The final cloud stays below the autoconversion threshold.
                   cloud = 2.0e-6
                else
                   cloud = 0.0
                endif
                rain = 0.0
                ice = 0.0
                snow = 0.0
                graupel = 0.0
                if (which == 42) then
                   ! Matched no-fallout member exposes WRF's exact adjusted
                   ! state before the cloud sedimentation divergence.
                   velocity = 1.0
                else
                   velocity = 0.0
                endif
             case (43, 44)
                if (kk == 2) then
                   cloud = 2.0e-6
                   ! Entry-active sub-collection rain makes WRF's preceding
                   ! rain-velocity pass refresh RHOF after cloud adjustment.
                   rain = 1.0e-8
                else
                   cloud = 0.0
                   rain = 0.0
                endif
                ice = 0.0
                snow = 0.0
                graupel = 0.0
                if (which == 44) then
                   velocity = 1.0
                else
                   velocity = 0.0
                endif
             case (46)
                cloud = 0.0
                rain = 0.0
                ice = 0.0
                if (height(kk) >= 1250.0 .and. height(kk) <= 4250.0) then
                   ! Both frozen categories are deliberately large so the
                   ! warm/subsaturated vector exposes simultaneous frozen
                   ! source terms and the same-call graupel-melt evaporation
                   ! limiter before category depletion can hide either path.
                   snow = 6.0e-3 * exp(-((height(kk)-2000.0)/1200.0)**2)
                   graupel = 6.0e-3 * exp(-((height(kk)-2000.0)/1200.0)**2)
                else
                   snow = 0.0
                   graupel = 0.0
                endif
                velocity = 1.0
             case default
                error stop 'invalid Thompson scenario id'
             end select
             qc3(ii,kk,jj) = cloud
             qr3(ii,kk,jj) = rain
             qi3(ii,kk,jj) = ice
             qs3(ii,kk,jj) = snow
             qg3(ii,kk,jj) = graupel
             if (ice > 1.0e-12) then
                if (which == 28 .or. which == 34) then
                   ! Select 6-micron ice, preserving WRF's ice-number ceiling
                   ! while satisfying the rain/ice collision size criterion.
                   ni3(ii,kk,jj) = ice * (4.0/6.0e-6)**3                &
                        / (3.1415926536*890.0)
                elseif (which == 6 .or. which == 17 .or. which == 27 .or. &
                        which == 35) then
                   ! Select a 29-micron mass-weighted diameter, below the
                   ! 30-micron ice-to-snow autoconversion threshold while
                   ! remaining under WRF's 999 L-1 ice-number ceiling.
                   ni3(ii,kk,jj) = ice * (4.0/29.0e-6)**3               &
                        / (3.1415926536*890.0)
                elseif (which == 13 .or. which == 37 .or. which == 38 .or. &
                        which == 39 .or. which == 40) then
                   ! Select a 200-micron mass-weighted diameter, within the
                   ! explicit ice-to-snow autoconversion lookup-table range;
                   ! scenario 37 overlaps it with nucleation and vapor rates.
                   ni3(ii,kk,jj) = ice * (4.0/200.0e-6)**3              &
                        / (3.1415926536*890.0)
                else
                   ni3(ii,kk,jj) = 8.0e5
                endif
             else
                ni3(ii,kk,jj) = 0.0
             endif
             if (rain > 1.0e-12) then
                if (which == 28) then
                   nr3(ii,kk,jj) = rain * (3.672/60.0e-6)**3             &
                        / (3.1415926536*1000.0)
                elseif (which == 5 .or. which == 14 .or. which == 23 .or. &
                        which == 43 .or. which == 44) then
                   ! Select a 45-micron median-volume diameter: below D0r,
                   ! so WRF's rain self-collection is inactive.  Scenario 5
                   ! isolates sedimentation; 14 admits evaporation first.
                   nr3(ii,kk,jj) = rain * (3.672/45.0e-6)**3             &
                        / (3.1415926536*1000.0)
                elseif (which == 38 .or. which == 39 .or. which == 40) then
                   ! One-millimetre MVD exceeds four times the 200-micron ice
                   ! diameter and activates simultaneous self-collection.
                   nr3(ii,kk,jj) = rain * (3.672/1000.0e-6)**3          &
                        / (3.1415926536*1000.0)
                elseif (which == 11 .or. which == 12 .or. which == 29 .or. &
                        which == 30 .or. which == 32 .or. which == 33 .or. &
                        which == 34) then
                   ! Select a 500-micron median-volume diameter above D0r,
                   ! activating rain self-collection without other species.
                   nr3(ii,kk,jj) = rain * (3.672/500.0e-6)**3            &
                        / (3.1415926536*1000.0)
                else
                   nr3(ii,kk,jj) = 3.0e5
                endif
             else
                nr3(ii,kk,jj) = 0.0
             endif
             w3(ii,kk,jj) = velocity
          enddo
       enddo
    enddo
  end subroutine initialize_case

  subroutine write_row(unit, phase, level, height, pressure, exner,     &
                       velocity, layer_dz, theta, vapor, cloud, rain,  &
                       ice, snow, graupel, ice_number, rain_number,    &
                       reqc, reqi, reqs, dbz)
    integer, intent(in) :: unit, level
    character(len=*), intent(in) :: phase
    real, intent(in) :: height, pressure, exner, velocity, layer_dz
    real, intent(in) :: theta, vapor, cloud, rain, ice, snow, graupel
    real, intent(in) :: ice_number, rain_number, reqc, reqi, reqs, dbz
    real :: temp

    temp = theta * exner
    write(unit,'(A,",",I0,19(",",ES24.16E3))') trim(phase), level,    &
         height, pressure, exner, velocity, layer_dz, theta, temp,     &
         vapor, cloud, rain, ice, snow, graupel, ice_number,           &
         rain_number, reqc, reqi, reqs, dbz
  end subroutine write_row

end program run_thompson_column
