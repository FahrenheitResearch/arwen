! TEMPO reference driver for the WOOF port (lane tempo-mp).
! Built against unmodified TEMPO 9adf9ef (tempo_v3.0.3) with -fdefault-real-8.
!  1. tempo_init with the RRFS v2 configuration (aerosol-aware, hail-aware)
!  2. dumps every module_mp_tempo_params scalar, array and lookup table as
!     named float64 records (tempo_blob.bin), read by tempo_ref_host.cpp
!  3. runs column scenarios through tempo_run (the MPAS/CCPP entry) and
!     writes the initial state and the state after every step
!     (tempo_ref_columns.bin)
program tempo_ref_driver
  use module_mp_tempo_cfgs, only : ty_tempo_cfgs
  use module_mp_tempo_driver, only : tempo_init, tempo_run, ty_tempo_driver_diags
  use module_mp_tempo_params
  implicit none

  type(ty_tempo_cfgs) :: cfgs
  integer :: u

  call tempo_init(tempo_cfgs=cfgs)

  open(newunit=u, file='tempo_blob.bin', form='unformatted', access='stream', status='replace')
  call wr(u, 'av_r', [av_r]); call wr(u, 'bv_r', [bv_r]); call wr(u, 'av_s', [av_s])
  call wr(u, 'bv_s', [bv_s]); call wr(u, 'fv_s', [fv_s]); call wr(u, 'bv_c', [bv_c])
  call wr(u, 'bv_i', [bv_i]); call wr(u, 'av_g_old', [av_g_old]); call wr(u, 'bv_g_old', [bv_g_old])
  call wr(u, 'av_g_new', [av_g_new]); call wr(u, 'bv_g_new', [bv_g_new]); call wr(u, 'fv_r', [fv_r])
  call wr(u, 'av_c', [av_c]); call wr(u, 'a_coeff', [a_coeff]); call wr(u, 'b_coeff', [b_coeff])
  call wr(u, 'av_i', [av_i]); call wr(u, 'am_s', [am_s]); call wr(u, 'bm_s', [bm_s])
  call wr(u, 'bm_g', [bm_g]); call wr(u, 'bm_i', [bm_i]); call wr(u, 'bm_r', [bm_r])
  call wr(u, 'rho_i', [rho_i]); call wr(u, 'xm0i', [xm0i]); call wr(u, 'd0c', [d0c])
  call wr(u, 'd0r', [d0r]); call wr(u, 'd0s', [d0s]); call wr(u, 'd0g', [d0g])
  call wr(u, 'd0r_max', [d0r_max]); call wr(u, 'c_cube', [c_cube]); call wr(u, 'c_sqrd', [c_sqrd])
  call wr(u, 'mu_r', [mu_r]); call wr(u, 'mu_s', [mu_s]); call wr(u, 'mu_g', [mu_g])
  call wr(u, 'mu_i', [mu_i]); call wr(u, 'nu_c_scale', [nu_c_scale])
  call wr(u, 'nu_c_max', [real(nu_c_max, 8)]); call wr(u, 'nu_c_min', [real(nu_c_min, 8)])
  call wr(u, 'naccn0', [naccn0]); call wr(u, 'naccn1', [naccn1]); call wr(u, 'nain0', [nain0])
  call wr(u, 'nain1', [nain1]); call wr(u, 'nwfa_default', [nwfa_default])
  call wr(u, 'nifa_default', [nifa_default]); call wr(u, 'aero_max', [aero_max])
  call wr(u, 'hgfrz', [hgfrz]); call wr(u, 'nt_c_o', [nt_c_o]); call wr(u, 'nt_c_l', [nt_c_l])
  call wr(u, 'nt_c_max', [nt_c_max]); call wr(u, 'nt_c_min', [nt_c_min]); call wr(u, 'tno', [tno])
  call wr(u, 'ato', [ato]); call wr(u, 'rho_s', [rho_s]); call wr(u, 'demott_nuc_ssati', [demott_nuc_ssati])
  call wr(u, 'max_ni', [real(max_ni, 8)]); call wr(u, 'icenuc_max', [icenuc_max])
  call wr(u, 'rime_threshold', [rime_threshold]); call wr(u, 'rime_conversion', [rime_conversion])
  call wr(u, 'ef_si', [ef_si]); call wr(u, 'ef_rs', [ef_rs]); call wr(u, 'ef_rg', [ef_rg])
  call wr(u, 'ef_ri', [ef_ri]); call wr(u, 'autocon_nr_factor', [autocon_nr_factor])
  call wr(u, 'timestep_conversion_rime_to_rain', [timestep_conversion_rime_to_rain])
  call wr(u, 'nrhg', [real(nrhg, 8)]); call wr(u, 'nrhg1', [real(nrhg1, 8)])
  call wr(u, 'min_qv', [min_qv]); call wr(u, 'r1', [r1]); call wr(u, 'r2', [r2])
  call wr(u, 'low_limit_mass_for_precip', [low_limit_mass_for_precip]); call wr(u, 'eps', [eps])
  call wr(u, 'meters3_to_liters', [meters3_to_liters]); call wr(u, 'gonv_min', [real(gonv_min, 8)])
  call wr(u, 'gonv_max', [real(gonv_max, 8)]); call wr(u, 't0', [t0]); call wr(u, 'rho_w', [rho_w])
  call wr(u, 'sc', [sc]); call wr(u, 'pi', [pi]); call wr(u, 'lsub', [lsub]); call wr(u, 'lvap0', [lvap0])
  call wr(u, 'rv', [rv]); call wr(u, 'rdry', [rdry]); call wr(u, 'roverrv', [roverrv]); call wr(u, 'r', [r])
  call wr(u, 'rho_not', [rho_not]); call wr(u, 'rho_not0', [rho_not0]); call wr(u, 'cp', [cp])
  call wr(u, 'r_uni', [r_uni]); call wr(u, 'kap0', [kap0]); call wr(u, 'kap1', [kap1])
  call wr(u, 'lam0', [lam0]); call wr(u, 'lam1', [lam1])
  call wr(u, 'nbins', [real(nbins, 8)]); call wr(u, 'nbc', [real(nbc, 8)]); call wr(u, 'nbr', [real(nbr, 8)])
  call wr(u, 'nbs', [real(nbs, 8)]); call wr(u, 'nbi', [real(nbi, 8)]); call wr(u, 'nbg', [real(nbg, 8)])
  call wr(u, 'ntb_i', [real(ntb_i, 8)]); call wr(u, 'ntb_i1', [real(ntb_i1, 8)]); call wr(u, 'ntb_c', [real(ntb_c, 8)])
  call wr(u, 'ntb_t', [real(ntb_t, 8)]); call wr(u, 'ntb_g1', [real(ntb_g1, 8)]); call wr(u, 'ntb_s', [real(ntb_s, 8)])
  call wr(u, 'ntb_g', [real(ntb_g, 8)]); call wr(u, 'ntb_r', [real(ntb_r, 8)]); call wr(u, 'ntb_r1', [real(ntb_r1, 8)])
  call wr(u, 'ntb_t1', [real(ntb_t1, 8)]); call wr(u, 'ntb_in', [real(ntb_in, 8)]); call wr(u, 'ntb_arc', [real(ntb_arc, 8)])
  call wr(u, 'ntb_arw', [real(ntb_arw, 8)]); call wr(u, 'ntb_art', [real(ntb_art, 8)]); call wr(u, 'ntb_arr', [real(ntb_arr, 8)])
  call wr(u, 'ntb_ark', [real(ntb_ark, 8)]); call wr(u, 'idx_bg1', [real(idx_bg1, 8)])
  call wr(u, 'nhbins', [real(nhbins, 8)]); call wr(u, 'radar_bins', [real(radar_bins, 8)])
  call wr(u, 'rho_g', rho_g); call wr(u, 'r_c', r_c); call wr(u, 'r_i', r_i); call wr(u, 'r_r', r_r)
  call wr(u, 'r_s', r_s); call wr(u, 'r_g', r_g); call wr(u, 'n0r_exp', n0r_exp); call wr(u, 'n0g_exp', n0g_exp)
  call wr(u, 'nt_i', nt_i); call wr(u, 'nt_in', nt_in); call wr(u, 'ta_na', ta_na); call wr(u, 'ta_ww', ta_ww)
  call wr(u, 'ta_tk', ta_tk); call wr(u, 'ta_ra', ta_ra); call wr(u, 'ta_ka', ta_ka); call wr(u, 'sa', sa)
  call wr(u, 'sb', sb); call wr(u, 'tc', tc)
  call wr(u, 'dim_nrhg', [real(dim_nrhg, 8)]); call wr(u, 'av_g', av_g); call wr(u, 'bv_g', bv_g)
  call wr(u, 'am_i', [am_i]); call wr(u, 'am_r', [am_r]); call wr(u, 'am_g', am_g); call wr(u, 'lfus', [lfus])
  call wr(u, 'olfus', [olfus]); call wr(u, 'orv', [orv]); call wr(u, 'ar_volume', [ar_volume])
  call wr(u, 'sc3', [sc3]); call wr(u, 'd0i', [d0i]); call wr(u, 'xm0s', [xm0s]); call wr(u, 'xm0g', [xm0g])
  call wr(u, 'obmi', [obmi]); call wr(u, 'obmr', [obmr]); call wr(u, 'oams', [oams]); call wr(u, 'obms', [obms])
  call wr(u, 'ocms', [ocms]); call wr(u, 'oamg', oamg); call wr(u, 'ocmg', ocmg); call wr(u, 'obmg', [obmg])
  call wr(u, 'cce', reshape(cce, [size(cce)])); call wr(u, 'ccg', reshape(ccg, [size(ccg)]))
  call wr(u, 'ocg1', ocg1); call wr(u, 'ocg2', ocg2); call wr(u, 'cie', cie); call wr(u, 'cig', cig)
  call wr(u, 'oig1', [oig1]); call wr(u, 'oig2', [oig2]); call wr(u, 'cre', cre); call wr(u, 'crg', crg)
  call wr(u, 'ore1', [ore1]); call wr(u, 'org1', [org1]); call wr(u, 'org2', [org2]); call wr(u, 'org3', [org3])
  call wr(u, 'cse', cse); call wr(u, 'csg', csg)
  call wr(u, 'cge', reshape(cge, [size(cge)])); call wr(u, 'cgg', reshape(cgg, [size(cgg)]))
  call wr(u, 'oge1', [oge1]); call wr(u, 'ogg1', [ogg1]); call wr(u, 'ogg2', [ogg2]); call wr(u, 'ogg3', [ogg3])
  call wr(u, 't1_qr_qc', [t1_qr_qc]); call wr(u, 't1_qr_qi', [t1_qr_qi]); call wr(u, 't2_qr_qi', [t2_qr_qi])
  call wr(u, 't1_qs_qc', [t1_qs_qc]); call wr(u, 't1_qs_qi', [t1_qs_qi]); call wr(u, 't1_qr_ev', [t1_qr_ev])
  call wr(u, 't2_qr_ev', [t2_qr_ev]); call wr(u, 't1_qs_sd', [t1_qs_sd]); call wr(u, 't2_qs_sd', [t2_qs_sd])
  call wr(u, 't1_qs_me', [t1_qs_me]); call wr(u, 't2_qs_me', [t2_qs_me]); call wr(u, 't1_qg_sd', [t1_qg_sd])
  call wr(u, 't1_qg_me', [t1_qg_me])
  call wr(u, 'nic2', [real(nic2, 8)]); call wr(u, 'nii2', [real(nii2, 8)]); call wr(u, 'nii3', [real(nii3, 8)])
  call wr(u, 'nir2', [real(nir2, 8)]); call wr(u, 'nir3', [real(nir3, 8)]); call wr(u, 'nis2', [real(nis2, 8)])
  call wr(u, 'nig2', [real(nig2, 8)]); call wr(u, 'nig3', [real(nig3, 8)]); call wr(u, 'niin2', [real(niin2, 8)])
  call wr(u, 'nic1', [nic1])
  call wr(u, 'dc', dc); call wr(u, 'dtc', dtc); call wr(u, 'di', di); call wr(u, 'dti', dti)
  call wr(u, 'dr', dr); call wr(u, 'dtr', dtr); call wr(u, 'ds', ds); call wr(u, 'dts', dts)
  call wr(u, 'dg', dg); call wr(u, 'dtg', dtg); call wr(u, 't_nc', t_nc)
  call wra(u, 'hbins', hbins, nhbins); call wra(u, 'dhbins', dhbins, nhbins)
  call wra(u, 'sbins_radar', sbins_radar, radar_bins); call wra(u, 'dsbins_radar', dsbins_radar, radar_bins)
  call wra(u, 'gbins_radar', gbins_radar, radar_bins); call wra(u, 'dgbins_radar', dgbins_radar, radar_bins)
  call wr(u, 't_efrw', reshape(t_efrw, [size(t_efrw)])); call wr(u, 't_efsw', reshape(t_efsw, [size(t_efsw)]))
  call wr(u, 'tpc_wev', reshape(tpc_wev, [size(tpc_wev)])); call wr(u, 'tnc_wev', reshape(tnc_wev, [size(tnc_wev)]))
  call wr(u, 'tnccn_act', reshape(real(tnccn_act, 8), [size(tnccn_act)]))
  call wr(u, 'tps_iaus', reshape(tps_iaus, [size(tps_iaus)])); call wr(u, 'tni_iaus', reshape(tni_iaus, [size(tni_iaus)]))
  call wr(u, 'tpi_ide', reshape(tpi_ide, [size(tpi_ide)]))
  call wr(u, 'tpi_qcfz', reshape(tpi_qcfz, [size(tpi_qcfz)])); call wr(u, 'tni_qcfz', reshape(tni_qcfz, [size(tni_qcfz)]))
  call wr(u, 'tpi_qrfz', reshape(tpi_qrfz, [size(tpi_qrfz)])); call wr(u, 'tpg_qrfz', reshape(tpg_qrfz, [size(tpg_qrfz)]))
  call wr(u, 'tni_qrfz', reshape(tni_qrfz, [size(tni_qrfz)])); call wr(u, 'tnr_qrfz', reshape(tnr_qrfz, [size(tnr_qrfz)]))
  call wr(u, 'tcs_racs1', reshape(tcs_racs1, [size(tcs_racs1)])); call wr(u, 'tmr_racs1', reshape(tmr_racs1, [size(tmr_racs1)]))
  call wr(u, 'tcs_racs2', reshape(tcs_racs2, [size(tcs_racs2)])); call wr(u, 'tmr_racs2', reshape(tmr_racs2, [size(tmr_racs2)]))
  call wr(u, 'tcr_sacr1', reshape(tcr_sacr1, [size(tcr_sacr1)])); call wr(u, 'tms_sacr1', reshape(tms_sacr1, [size(tms_sacr1)]))
  call wr(u, 'tcr_sacr2', reshape(tcr_sacr2, [size(tcr_sacr2)])); call wr(u, 'tms_sacr2', reshape(tms_sacr2, [size(tms_sacr2)]))
  call wr(u, 'tnr_racs1', reshape(tnr_racs1, [size(tnr_racs1)])); call wr(u, 'tnr_racs2', reshape(tnr_racs2, [size(tnr_racs2)]))
  call wr(u, 'tnr_sacr1', reshape(tnr_sacr1, [size(tnr_sacr1)])); call wr(u, 'tnr_sacr2', reshape(tnr_sacr2, [size(tnr_sacr2)]))
  call wr(u, 'tcg_racg', reshape(tcg_racg, [size(tcg_racg)])); call wr(u, 'tmr_racg', reshape(tmr_racg, [size(tmr_racg)]))
  call wr(u, 'tcr_gacr', reshape(tcr_gacr, [size(tcr_gacr)])); call wr(u, 'tnr_racg', reshape(tnr_racg, [size(tnr_racg)]))
  call wr(u, 'tnr_gacr', reshape(tnr_gacr, [size(tnr_gacr)]))
  close(u)
  write(*,'(A)') 'tempo_ref_driver: wrote tempo_blob.bin'

  open(newunit=u, file='tempo_ref_columns.bin', form='unformatted', access='stream', status='replace')
  ! scenario id, dt, nsteps, semi_sedi, qv scale over levels 8..30, qc seed
  call run_scenario(u, cfgs, 1, 20._8, 90, .false., 1.0_8, 0._8, 0._8)
  call run_scenario(u, cfgs, 2, 20._8, 90, .false., 1.15_8, 2.e-4_8, 0._8)
  call run_scenario(u, cfgs, 3, 20._8, 90, .true., 1.15_8, 2.e-4_8, 0._8)
  call run_scenario(u, cfgs, 4, 60._8, 30, .false., 1.15_8, 2.e-4_8, 0._8)
  ! 5: whole column 8 K warmer (melting graupel and snow, warm rain)
  call run_scenario(u, cfgs, 5, 20._8, 90, .false., 1.15_8, 2.e-4_8, 8._8)
  ! 6: whole column 18 K colder (freezing drizzle and rain aloft, ice nucleation, snow)
  call run_scenario(u, cfgs, 6, 20._8, 90, .false., 1.15_8, 2.e-4_8, -18._8)
  close(u)
  write(*,'(A)') 'tempo_ref_driver: wrote tempo_ref_columns.bin'

contains

  subroutine wr(u, name, a)
    integer, intent(in) :: u
    character(len=*), intent(in) :: name
    real(8), intent(in) :: a(:)
    character(len=48) :: nm
    nm = name
    write(u) nm, int(size(a), 8), a
  end subroutine wr

  subroutine wra(u, name, a, n)
    integer, intent(in) :: u, n
    character(len=*), intent(in) :: name
    real(8), allocatable, intent(in) :: a(:)
    real(8) :: z(n)
    if (allocated(a)) then
      call wr(u, name, a)
    else
      z = 0._8
      call wr(u, name, z)
    endif
  end subroutine wra

  subroutine run_scenario(u, cfgs_in, sid, dt, nsteps, semi_sedi, qvscale, qcseed, toffset)
    integer, intent(in) :: u, sid, nsteps
    type(ty_tempo_cfgs), intent(in) :: cfgs_in
    real(8), intent(in) :: dt, qvscale, qcseed, toffset
    logical, intent(in) :: semi_sedi
    integer, parameter :: nz = 59
    type(ty_tempo_cfgs) :: c
    type(ty_tempo_driver_diags) :: d
    real(8), dimension(1, nz, 1) :: t, p, w, dz, qv, qc, qr, qi, qs, qg, ni, nr, nc, nwfa, nifa, ng, qb, th, pii
    real(8), dimension(nz) :: klev, qv_in, qc_in, qr_in, qi_in, qs_in, qg_in, ni_in, nr_in, nc_in, &
      nwfa_in, nifa_in, theta_in, ng_in, volg_in, p_in, w_in, dz_in
    integer :: io, k, it

    open(newunit=io, file='../test/data/mpas_59lev_test.txt', status='old')
    read(io, *)
    do k = 1, nz
      read(io, *) klev(k), qv_in(k), qc_in(k), qr_in(k), qi_in(k), qs_in(k), qg_in(k), ni_in(k), &
        nr_in(k), nc_in(k), nwfa_in(k), nifa_in(k), theta_in(k), ng_in(k), volg_in(k), p_in(k), &
        w_in(k), dz_in(k)
    enddo
    close(io)

    c = cfgs_in
    c%semi_sedi_flag = semi_sedi

    t(1,:,1) = theta_in * (p_in/100000._8)**0.286_8 + toffset
    p(1,:,1) = p_in
    w(1,:,1) = w_in
    dz(1,:,1) = dz_in
    qv(1,:,1) = qv_in
    do k = 8, 30
      qv(1,k,1) = qv_in(k) * qvscale
    enddo
    qc(1,:,1) = qc_in
    if (qcseed > 0._8) then
      do k = 5, 20
        qc(1,k,1) = qc(1,k,1) + qcseed
      enddo
    endif
    qr(1,:,1) = qr_in
    qi(1,:,1) = qi_in
    qs(1,:,1) = qs_in
    qg(1,:,1) = qg_in
    ni(1,:,1) = ni_in
    nr(1,:,1) = nr_in
    nc(1,:,1) = nc_in
    nwfa(1,:,1) = nwfa_in
    nifa(1,:,1) = nifa_in
    ng(1,:,1) = ng_in
    qb(1,:,1) = volg_in * 1000._8

    write(u) int(sid, 8), int(nz, 8), int(nsteps, 8), dt, merge(1._8, 0._8, semi_sedi)
    write(u) t(1,:,1), p(1,:,1), w(1,:,1), dz(1,:,1), qv(1,:,1), qc(1,:,1), qr(1,:,1), qi(1,:,1), &
      qs(1,:,1), qg(1,:,1), ni(1,:,1), nr(1,:,1), nc(1,:,1), nwfa(1,:,1), nifa(1,:,1), ng(1,:,1), qb(1,:,1)
    do it = 1, nsteps
      call tempo_run(tempo_cfgs=c, dt=dt, itimestep=it, t=t, p=p, w=w, dz=dz, &
        qv=qv, qc=qc, qr=qr, qi=qi, qs=qs, qg=qg, ni=ni, nr=nr, &
        nc=nc, nwfa=nwfa, nifa=nifa, ng=ng, qb=qb, &
        ids=1, ide=1, jds=1, jde=1, kds=1, kde=nz, ims=1, ime=1, jms=1, jme=1, kms=1, kme=nz, &
        its=1, ite=1, jts=1, jte=1, kts=1, kte=nz, tempo_diags=d)
      write(u) t(1,:,1), p(1,:,1), w(1,:,1), dz(1,:,1), qv(1,:,1), qc(1,:,1), qr(1,:,1), qi(1,:,1), &
        qs(1,:,1), qg(1,:,1), ni(1,:,1), nr(1,:,1), nc(1,:,1), nwfa(1,:,1), nifa(1,:,1), ng(1,:,1), qb(1,:,1)
      write(u) d%rain_precip(1,1), d%ice_liquid_equiv_precip(1,1), d%snow_liquid_equiv_precip(1,1), &
        d%graupel_liquid_equiv_precip(1,1), d%frz_rain_precip(1,1), &
        d%max_hail_diameter_sfc(1,1), d%max_hail_diameter_column(1,1)
      write(u) d%refl10cm(1,:,1), d%re_cloud(1,:,1), d%re_ice(1,:,1), d%re_snow(1,:,1)
    enddo
  end subroutine run_scenario

end program tempo_ref_driver
