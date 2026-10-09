! WRF v4.6.1's thompson_init and radar_init module state, bit for bit, for
! tests/data/thompson_wrf461_init_constants.txt: the REAL(4) gamma moments
! (crg, csg, cgg at idx_bg1) and exponents, the derived reciprocals and the
! radar module's PI5, lamda4, K_w, m_w_0, m_i_0 and size-bin tables.  Linked
! against the PRIVATE-dropped copy refl_check.py builds; run in a directory
! holding the four Thompson tables with GFORTRAN_CONVERT_UNIT='big_endian:20'
!   python refl_check.py constants BUILD_DIR TABLE_RUN_DIR > constants.txt
program constdump
  use module_mp_thompson
  use module_mp_radar
  implicit none
  integer, parameter :: nz = 3, nxp = 2
  real :: hgt(nxp,nz,2), nwfa(nxp,nz,2), nifa(nxp,nz,2), nbca(nxp,nz,2)
  real :: nwfa2d(nxp,2), nbca2d(nxp,2)
  integer :: n
  hgt = 0.; nwfa = 1.e9; nifa = 1.e6; nbca = 0.; nwfa2d = 0.; nbca2d = 0.
  do n = 1, nz
     hgt(:,n,:) = 100.*n
  enddo
  call thompson_init(hgt=hgt, nwfa2d=nwfa2d, nbca2d=nbca2d, nwfa=nwfa, nifa=nifa, nbca=nbca, &
       wif_input_opt=0, ids=1, ide=nxp, jds=1, jde=2, kds=1, kde=nz, &
       ims=1, ime=nxp, jms=1, jme=2, kms=1, kme=nz, its=1, ite=nxp, jts=1, jte=2, kts=1, kte=nz)
  do n = 1, 13
     call f('crg', n, crg(n)); call f('cre', n, cre(n))
  enddo
  do n = 1, 17
     call f('csg', n, csg(n)); call f('cse', n, cse(n))
  enddo
  do n = 1, 12
     call f('cgg', n, cgg(n,idx_bg1)); call f('cge', n, cge(n,idx_bg1))
  enddo
  call f('am_g', idx_bg1, am_g(idx_bg1)); call f('am_r', 0, am_r); call f('am_s', 0, am_s)
  call f('oams', 0, oams); call f('obms', 0, obms); call f('ocms', 0, ocms)
  call f('obmr', 0, obmr); call f('org2', 0, org2); call f('org3', 0, org3)
  call f('ogg1', 0, ogg1); call f('ogg2', 0, ogg2); call f('ogg3', 0, ogg3)
  call f('obmg', 0, obmg); call f('oge1', 0, oge1); call f('RHO_NOT', 0, RHO_NOT)
  call f('ocmg', idx_bg1, ocmg(idx_bg1))
  call f('av_g_old', 0, av_g_old); call f('bv_g_old', 0, bv_g_old)
  call f('mu_s', 0, mu_s); call f('Kap0', 0, Kap0); call f('Kap1', 0, Kap1)
  call f('Lam0', 0, Lam0); call f('Lam1', 0, Lam1); call f('av_s', 0, av_s); call f('fv_s', 0, fv_s)
  call f('av_r', 0, av_r); call f('fv_r', 0, fv_r); call f('PI', 0, PI)
  do n = 1, 10
     call f('sa', n, sa(n)); call f('sb', n, sb(n))
  enddo
  call d('PI5', 0, PI5); call d('lamda4', 0, lamda4); call d('K_w', 0, K_w)
  call d('m_w_0re', 0, dble(m_w_0)); call d('m_w_0im', 0, aimag(m_w_0))
  call d('m_i_0re', 0, dble(m_i_0)); call d('m_i_0im', 0, aimag(m_i_0))
  do n = 1, nrbins
     call d('xxDs', n, xxDs(n)); call d('xdts', n, xdts(n)); call d('simpson', n, simpson(n))
  enddo
  call d('simpson', nrbins+1, simpson(nrbins+1))
contains
  subroutine f(name, i, v)
    character(*), intent(in) :: name
    integer, intent(in) :: i
    real, intent(in) :: v
    print '(A,1X,I0,1X,Z8.8,1X,ES24.16)', name, i, transfer(v, 0), v
  end subroutine f
  subroutine d(name, i, v)
    character(*), intent(in) :: name
    integer, intent(in) :: i
    double precision, intent(in) :: v
    print '(A,1X,I0,1X,Z16.16,1X,ES26.18)', name, i, transfer(v, 0_8), v
  end subroutine d
end program constdump
