! Table dump runner for the fast SBM oracle (links the ACCESS build).
!
! Calls FAST_HUCMINIT(dt) exactly as WRF does at initialisation, then writes
!   <out>/tables  record "read":    every table array as READ from SBM_input_33
!                 record "derived": ima, chucm, both mg mass grids, ECOALMASSM,
!                                   BRKWEIGHT, Prob, Gain_Var_New, NND,
!                                   ikr_spon_break, the aerosol spectra and
!                                   grids, DROPRADII
!   <out>/kernels record "kernels" x 10: cwll and every cw* family after
!                                   Kernals_KS(dt, NKR, p) at 1000, 950, 850,
!                                   750, 700, 600, 500, 400, 300 and 250 hPa
! in the build's own precision (f4 native, f8 promoted; arrays WRF declares
! REAL(KIND=8) are f8 in both).
!
! The two mg mass grids: xl_mg of module_mp_fast_sbm is declared with an
! explicit PRIVATE attribute (not reachable through the one-line access
! transform), and the grid inside courant_bott_KS is a local. Both are written
! here by evaluating the WRF expression with the WRF kinds:
!   xl_mg_6078 = xl(i)*1.e3      (:6078, default-real product, stored in r8)
!   xl_mg_1656 = xl(i)*1.0D3     (:1656, default real times a double constant)
! The KC-1 probe family records the module's own xl_mg from inside
! COAL_BOTT_NEW, and pack_fixtures.py checks the two agree byte for byte.
!
! Usage: run_dump_tables <outdir> [dt]   (run where ./SBM_input_33 exists)
program run_dump_tables
  use module_mp_fast_sbm
  use fsbm_io
  implicit none
  type(fsbm_stream) :: s
  character(len=512) :: outdir, arg
  real :: dtk, pz
  real :: plist(10)
  real(8) :: xlmg6078(33), xlmg1656(33)
  integer :: n, ii

  plist = (/ 1000., 950., 850., 750., 700., 600., 500., 400., 300., 250. /)
  call get_command_argument(1, outdir)
  if (len_trim(outdir) == 0) call fsbm_fail('usage: run_dump_tables <outdir> [dt]')
  dtk = 12.0
  if (command_argument_count() >= 2) then
    call get_command_argument(2, arg)
    read(arg, *) dtk
  end if

  call fast_hucminit(dtk)

  do ii = 1, 33
    xlmg6078(ii) = xl(ii)*1.e3
    xlmg1656(ii) = xl(ii)*1.0D3
  end do

  call fsbm_open(s, trim(outdir)//'/tables')
  call fsbm_begin(s, 'read')
  call fsbm_put(s, 'dt', dtk)
  call fsbm_put(s, 'bin_mass', bin_mass)
  call fsbm_put(s, 'tab_colum', tab_colum)
  call fsbm_put(s, 'tab_dendr', tab_dendr)
  call fsbm_put(s, 'tab_snow', tab_snow)
  call fsbm_put(s, 'bin_log', bin_log)
  call fsbm_put(s, 'rlec', rlec)
  call fsbm_put(s, 'riec', riec)
  call fsbm_put(s, 'rsec', rsec)
  call fsbm_put(s, 'rgec', rgec)
  call fsbm_put(s, 'rhec', rhec)
  call fsbm_put(s, 'xl', xl)
  call fsbm_put(s, 'xi', xi)
  call fsbm_put(s, 'xs', xs)
  call fsbm_put(s, 'xg', xg)
  call fsbm_put(s, 'xh', xh)
  call fsbm_put(s, 'vr1', vr1)
  call fsbm_put(s, 'vr2', vr2)
  call fsbm_put(s, 'vr3', vr3)
  call fsbm_put(s, 'vr4', vr4)
  call fsbm_put(s, 'vr5', vr5)
  call fsbm_put(s, 'slic', slic)
  call fsbm_put(s, 'tlic', tlic)
  call fsbm_put(s, 'coefin', coefin)
  call fsbm_put(s, 'ywll_1000mb', ywll_1000mb)
  call fsbm_put(s, 'ywll_750mb', ywll_750mb)
  call fsbm_put(s, 'ywll_500mb', ywll_500mb)
  call fsbm_put(s, 'ywli_300mb', ywli_300mb)
  call fsbm_put(s, 'ywli_500mb', ywli_500mb)
  call fsbm_put(s, 'ywli_750mb', ywli_750mb)
  call fsbm_put(s, 'ywlg_300mb', ywlg_300mb)
  call fsbm_put(s, 'ywlg_500mb', ywlg_500mb)
  call fsbm_put(s, 'ywlg_750mb', ywlg_750mb)
  call fsbm_put(s, 'ywlh_300mb', ywlh_300mb)
  call fsbm_put(s, 'ywlh_500mb', ywlh_500mb)
  call fsbm_put(s, 'ywlh_750mb', ywlh_750mb)
  call fsbm_put(s, 'ywls_300mb', ywls_300mb)
  call fsbm_put(s, 'ywls_500mb', ywls_500mb)
  call fsbm_put(s, 'ywls_750mb', ywls_750mb)
  call fsbm_put(s, 'ywii_300mb', ywii_300mb)
  call fsbm_put(s, 'ywii_500mb', ywii_500mb)
  call fsbm_put(s, 'ywii_750mb', ywii_750mb)
  call fsbm_put(s, 'ywis_300mb', ywis_300mb)
  call fsbm_put(s, 'ywis_500mb', ywis_500mb)
  call fsbm_put(s, 'ywis_750mb', ywis_750mb)
  call fsbm_put(s, 'ywsg_300mb', ywsg_300mb)
  call fsbm_put(s, 'ywsg_500mb', ywsg_500mb)
  call fsbm_put(s, 'ywsg_750mb', ywsg_750mb)
  call fsbm_put(s, 'ywss_300mb', ywss_300mb)
  call fsbm_put(s, 'ywss_500mb', ywss_500mb)
  call fsbm_put(s, 'ywss_750mb', ywss_750mb)
  call fsbm_put(s, 'ro1bl', ro1bl)
  call fsbm_put(s, 'ro2bl', ro2bl)
  call fsbm_put(s, 'ro3bl', ro3bl)
  call fsbm_put(s, 'ro4bl', ro4bl)
  call fsbm_put(s, 'ro5bl', ro5bl)
  call fsbm_put(s, 'radxxo', radxxo)
  call fsbm_put(s, 'pkij', pkij)
  call fsbm_put(s, 'qkj', qkj)
  call fsbm_end(s)

  call fsbm_begin(s, 'derived')
  call fsbm_put(s, 'dt', dtk)
  call fsbm_put(s, 'ima', ima)
  call fsbm_put(s, 'chucm', chucm)
  call fsbm_put(s, 'xl_mg_6078', xlmg6078)
  call fsbm_put(s, 'xl_mg_1656', xlmg1656)
  call fsbm_put(s, 'ecoalmassm', ecoalmassm)
  call fsbm_put(s, 'brkweight', brkweight)
  call fsbm_put(s, 'prob', prob)
  call fsbm_put(s, 'gain_var_new', gain_var_new)
  call fsbm_put(s, 'nnd', nnd)
  call fsbm_put(s, 'ikr_spon_break', int(ikr_spon_break, 4))
  call fsbm_put(s, 'fccnr_con', fccnr_con)
  call fsbm_put(s, 'fccnr_mar', fccnr_mar)
  call fsbm_put(s, 'xccn', xccn)
  call fsbm_put(s, 'rccn', rccn)
  call fsbm_put(s, 'scale_ccn_factor', scale_ccn_factor)
  call fsbm_put(s, 'dropradii', dropradii)
  call fsbm_end(s)
  call fsbm_close(s)

  call fsbm_open(s, trim(outdir)//'/kernels')
  do n = 1, 10
    pz = plist(n)*1000.0
    call kernals_ks(dtk, 33, pz)
    call fsbm_begin(s, 'kernels')
    call fsbm_put(s, 'p_hpa', plist(n))
    call fsbm_put(s, 'p_z', pz)
    call fsbm_put(s, 'dtime_coal', dtk)
    call fsbm_put(s, 'cwll', cwll)
    call fsbm_put(s, 'cwli_1', cwli_1)
    call fsbm_put(s, 'cwli_2', cwli_2)
    call fsbm_put(s, 'cwli_3', cwli_3)
    call fsbm_put(s, 'cwil_1', cwil_1)
    call fsbm_put(s, 'cwil_2', cwil_2)
    call fsbm_put(s, 'cwil_3', cwil_3)
    call fsbm_put(s, 'cwlg', cwlg)
    call fsbm_put(s, 'cwlh', cwlh)
    call fsbm_put(s, 'cwls', cwls)
    call fsbm_put(s, 'cwgl', cwgl)
    call fsbm_put(s, 'cwhl', cwhl)
    call fsbm_put(s, 'cwsl', cwsl)
    call fsbm_put(s, 'cwii_1_1', cwii_1_1)
    call fsbm_put(s, 'cwii_1_2', cwii_1_2)
    call fsbm_put(s, 'cwii_1_3', cwii_1_3)
    call fsbm_put(s, 'cwii_2_1', cwii_2_1)
    call fsbm_put(s, 'cwii_2_2', cwii_2_2)
    call fsbm_put(s, 'cwii_2_3', cwii_2_3)
    call fsbm_put(s, 'cwii_3_1', cwii_3_1)
    call fsbm_put(s, 'cwii_3_2', cwii_3_2)
    call fsbm_put(s, 'cwii_3_3', cwii_3_3)
    call fsbm_put(s, 'cwis_1', cwis_1)
    call fsbm_put(s, 'cwis_2', cwis_2)
    call fsbm_put(s, 'cwis_3', cwis_3)
    call fsbm_put(s, 'cwsi_1', cwsi_1)
    call fsbm_put(s, 'cwsi_2', cwsi_2)
    call fsbm_put(s, 'cwsi_3', cwsi_3)
    call fsbm_put(s, 'cwig_1', cwig_1)
    call fsbm_put(s, 'cwig_2', cwig_2)
    call fsbm_put(s, 'cwig_3', cwig_3)
    call fsbm_put(s, 'cwih_1', cwih_1)
    call fsbm_put(s, 'cwih_2', cwih_2)
    call fsbm_put(s, 'cwih_3', cwih_3)
    call fsbm_put(s, 'cwsg', cwsg)
    call fsbm_put(s, 'cwss', cwss)
    call fsbm_end(s)
  end do
  call fsbm_close(s)

end program run_dump_tables
