program run_dump
  use module_mp_thompson, only: thompson_init, arwen_dump_classic_constants
  use module_mp_radar, only: xxDs, xdts, simpson, K_w, PI5, lamda4,     &
       m_w_0, m_i_0, nrbins
  implicit none
  real :: hgt(2,2,2)
  integer :: n
  hgt = 0.0
  hgt(:,1,:) = 100.0
  hgt(:,2,:) = 200.0
  call thompson_init(hgt=hgt, ids=1, ide=3, jds=1, jde=3, kds=1, kde=2, &
       ims=1, ime=2, jms=1, jme=2, kms=1, kme=2,                        &
       its=1, ite=2, jts=1, jte=2, kts=1, kte=2)
  call arwen_dump_classic_constants()
  ! module_mp_radar's radar_init words (thompson_init calls radar_init).
  do n = 1, nrbins
     write(*,'(A,1X,I0,1X,Z16.16)') 'D xxDs', n, transfer(xxDs(n), 0_8)
     write(*,'(A,1X,I0,1X,Z16.16)') 'D xdts', n, transfer(xdts(n), 0_8)
  enddo
  do n = 1, nrbins + 1
     write(*,'(A,1X,I0,1X,Z16.16)') 'D simpson', n, transfer(simpson(n), 0_8)
  enddo
  write(*,'(A,1X,I0,1X,Z16.16)') 'D K_w', 1, transfer(K_w, 0_8)
  write(*,'(A,1X,I0,1X,Z16.16)') 'D PI5', 1, transfer(PI5, 0_8)
  write(*,'(A,1X,I0,1X,Z16.16)') 'D lamda4', 1, transfer(lamda4, 0_8)
  write(*,'(A,1X,I0,1X,Z16.16)') 'D m_w_0', 1, transfer(dble(m_w_0), 0_8)
  write(*,'(A,1X,I0,1X,Z16.16)') 'D m_w_0', 2, transfer(dimag(m_w_0), 0_8)
  write(*,'(A,1X,I0,1X,Z16.16)') 'D m_i_0', 1, transfer(dble(m_i_0), 0_8)
  write(*,'(A,1X,I0,1X,Z16.16)') 'D m_i_0', 2, transfer(dimag(m_i_0), 0_8)
end program run_dump
