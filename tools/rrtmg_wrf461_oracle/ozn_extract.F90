! Control program for the WRF v4.6.1 o3input=2 ozone pipeline oracle.
!
! Drives the UNMODIFIED oznini + lin_interpol2 (phys/module_ra_cam_support.F)
! and ozn_time_int + ozn_p_int (phys/module_radiation_driver.F), extracted
! verbatim into ozn_oracle_mod.F90 by ozn_build.sh (line ranges taken by
! pattern; source SHA-256 recorded beside the fixtures).  Compiled with the
! same FP32 (RWORDSIZE=4) flag set as lw_build.sh, so every stage runs in
! default REAL exactly as in a WRF FP32 build.
!
! Reads the three run/ozone*.formatted files from the CWD (ozn_build.sh
! copies them from the WRF source authority), then an input text file:
!
!   nlat                      number of test latitudes
!   xlat(1..nlat)             one value per line
!   njul                      number of time-interp cases
!   julday julian             one pair per line
!   npc                       number of pressure-interp cases
!   ncol nz ijul              per case header (ijul indexes the julian
!                             cases; ozmixt columns 1..ncol are reused)
!   p(i,k)                    ncol*nz values, one per line, column i
!                             outer, k inner BOTTOM-UP (driver order)
!
! Dumps everything (parsed climatology, lat-interpolated ozmixm, per-case
! ozmixt and o3vmr) into OUT_DIR/ozn_fixture.bin via lw_binio's
! little-endian record format.  WRF's ozmixm/ozmixin month slot 1 is never
! written by oznini (INTENT(OUT), loops start at m=2), so only slots 2..13
! are dumped.
!
! usage: ozn_extract INPUT_TXT OUT_DIR
program ozn_extract
  use ozn_oracle_mod, only: oznini, ozn_time_int, ozn_p_int, &
                            plev_ozone_save, lat_ozone_save, ozmixin_save
  use lw_binio, only: bio_open, bio_close, wr_r0, wr_r1, wr_r2, wr_r3, &
                      wr_r4, wr_i0
  implicit none

  integer, parameter :: levsiz = 59, num_months = 13
  character(len=1024) :: input_txt, out_dir
  character(len=64) :: rec
  integer :: iu, nlat, njul, npc, i, j, k, jc, pc, ncol, nz, ijul
  integer, allocatable :: juldays(:)
  real, allocatable :: juls(:)
  real, allocatable :: xlat(:,:), ozmixm(:,:,:,:), pin(:)
  real, allocatable :: ozt(:,:,:), ozmixt_all(:,:,:)
  real, allocatable :: p(:,:,:), o3(:,:,:), ozt_case(:,:,:)

  if (command_argument_count() /= 2) then
     write(*,'(A)') 'usage: ozn_extract INPUT_TXT OUT_DIR'
     error stop 2
  end if
  call get_command_argument(1, input_txt)
  call get_command_argument(2, out_dir)

  open(newunit=iu, file=trim(input_txt), form='formatted', status='old', &
       action='read')
  read(iu,*) nlat
  allocate(xlat(nlat,1))
  do i = 1, nlat
     read(iu,*) xlat(i,1)
  end do

  allocate(ozmixm(nlat, levsiz, 1, num_months), pin(levsiz))

  ! Grid bounds chosen so oznini's itf=min0(ite,ide-1) covers all nlat
  ! test points and jtf covers the single row.
  call oznini(ozmixm, pin, levsiz, num_months, xlat, &
              1, nlat+1, 1, 2, 1, 2, &
              1, nlat,   1, 1, 1, 1, &
              1, nlat,   1, 1, 1, 1)

  call bio_open(trim(out_dir)//'/ozn_fixture.bin')
  call wr_i0('meta/nlat', nlat)
  call wr_r1('static/plev', plev_ozone_save)
  call wr_r1('static/pin', pin)
  call wr_r1('static/lat_ozone', lat_ozone_save)
  call wr_r4('static/ozmixin', ozmixin_save(:,:,:,2:num_months))
  call wr_r1('static/xlat', xlat(:,1))
  call wr_r3('static/ozmixm', ozmixm(:,:,1,2:num_months))

  read(iu,*) njul
  call wr_i0('meta/njul', njul)
  allocate(juldays(njul), juls(njul))
  allocate(ozt(nlat, levsiz, 1), ozmixt_all(nlat, levsiz, njul))
  do jc = 1, njul
     read(iu,*) juldays(jc), juls(jc)
  end do
  do jc = 1, njul
     call ozn_time_int(juldays(jc), juls(jc), ozmixm, ozt, levsiz, &
                       num_months, &
                       1, nlat+1, 1, 2, 1, 2, &
                       1, nlat,   1, 1, 1, 1, &
                       1, nlat,   1, 1, 1, 1)
     ozmixt_all(:,:,jc) = ozt(:,:,1)
     write(rec,'(A,I0,A)') 'jul_', jc, '/julday'
     call wr_i0(trim(rec), juldays(jc))
     write(rec,'(A,I0,A)') 'jul_', jc, '/julian'
     call wr_r0(trim(rec), juls(jc))
     write(rec,'(A,I0,A)') 'jul_', jc, '/ozmixt'
     call wr_r2(trim(rec), ozt(:,:,1))
  end do

  read(iu,*) npc
  call wr_i0('meta/npc', npc)
  do pc = 1, npc
     read(iu,*) ncol, nz, ijul
     if (ncol > nlat .or. ijul < 1 .or. ijul > njul) then
        write(*,'(A,I0)') 'ozn_extract: bad pressure case ', pc
        error stop 3
     end if
     allocate(p(ncol, nz, 1), o3(ncol, nz, 1), ozt_case(ncol, levsiz, 1))
     do i = 1, ncol
        do k = 1, nz
           read(iu,*) p(i,k,1)
        end do
     end do
     ozt_case(:,:,1) = ozmixt_all(1:ncol,:,ijul)
     o3 = 0.0
     call ozn_p_int(p, pin, levsiz, ozt_case, o3, &
                    1, ncol+1, 1, 2, 1, nz+1, &
                    1, ncol,   1, 1, 1, nz, &
                    1, ncol,   1, 1, 1, nz)
     write(rec,'(A,I0,A)') 'pc_', pc, '/ijul'
     call wr_i0(trim(rec), ijul)
     write(rec,'(A,I0,A)') 'pc_', pc, '/p'
     call wr_r2(trim(rec), p(:,:,1))
     write(rec,'(A,I0,A)') 'pc_', pc, '/o3vmr'
     call wr_r2(trim(rec), o3(:,:,1))
     deallocate(p, o3, ozt_case)
  end do
  close(iu)
  call bio_close()
  write(*,'(A)') 'ozn_extract: OK'
end program ozn_extract
