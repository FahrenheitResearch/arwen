program run_dump
  ! MORR_TWO_MOMENT_INIT's own words for morr_rimed_ice = 0 (graupel) and
  ! 1 (hail), then module_mp_radar's radar_init words (the init calls
  ! radar_init after setting the Morrison size-distribution parameters).
  use module_mp_morr_two_moment, only: morr_two_moment_init,            &
       arwen_dump_morr_constants
  use module_mp_radar, only: xxDs, xdts, xxDg, xdtg, simpson, K_w, PI5,  &
       lamda4, m_w_0, m_i_0, nrbins, xcre, xcse, xcge, xcrg, xcsg, xcgg, &
       xam_r, xbm_r, xmu_r, xobmr, xam_s, xbm_s, xmu_s, xoams, xobms,    &
       xocms, xam_g, xbm_g, xmu_g, xoamg, xobmg, xocmg, xorg2, xosg2, xogg2
  implicit none
  integer :: mode, n
  do mode = 0, 1
     call morr_two_moment_init(mode)
     call arwen_dump_morr_constants(mode)
     do n = 1, 4
        call wr('xcre', n, xcre(n)); call wr('xcse', n, xcse(n))
        call wr('xcge', n, xcge(n)); call wr('xcrg', n, xcrg(n))
        call wr('xcsg', n, xcsg(n)); call wr('xcgg', n, xcgg(n))
     enddo
     call wr('xam_r', 1, xam_r); call wr('xbm_r', 1, xbm_r)
     call wr('xmu_r', 1, xmu_r); call wr('xobmr', 1, xobmr)
     call wr('xam_s', 1, xam_s); call wr('xbm_s', 1, xbm_s)
     call wr('xmu_s', 1, xmu_s); call wr('xoams', 1, xoams)
     call wr('xobms', 1, xobms); call wr('xocms', 1, xocms)
     call wr('xam_g', 1, xam_g); call wr('xbm_g', 1, xbm_g)
     call wr('xmu_g', 1, xmu_g); call wr('xoamg', 1, xoamg)
     call wr('xobmg', 1, xobmg); call wr('xocmg', 1, xocmg)
     call wr('xorg2', 1, xorg2); call wr('xosg2', 1, xosg2)
     call wr('xogg2', 1, xogg2)
     do n = 1, nrbins
        call wd('xxDs', n, xxDs(n)); call wd('xdts', n, xdts(n))
        call wd('xxDg', n, xxDg(n)); call wd('xdtg', n, xdtg(n))
     enddo
     do n = 1, nrbins + 1
        call wd('simpson', n, simpson(n))
     enddo
     call wd('K_w', 1, K_w); call wd('PI5', 1, PI5)
     call wd('lamda4', 1, lamda4)
     call wd('m_w_0', 1, dble(m_w_0)); call wd('m_w_0', 2, dimag(m_w_0))
     call wd('m_i_0', 1, dble(m_i_0)); call wd('m_i_0', 2, dimag(m_i_0))
  enddo
contains
  subroutine wr(name, i, v)
    character(len=*), intent(in) :: name
    integer, intent(in) :: i
    real, intent(in) :: v
    write(*,'(A,1X,A,1X,I0,1X,I0,1X,Z8.8)') 'R', name, mode, i, transfer(v, 0)
  end subroutine wr
  subroutine wd(name, i, v)
    character(len=*), intent(in) :: name
    integer, intent(in) :: i
    double precision, intent(in) :: v
    write(*,'(A,1X,A,1X,I0,1X,I0,1X,Z16.16)') 'D', name, mode, i,        &
         transfer(v, 0_8)
  end subroutine wd
end program run_dump
