! The unmodified fork's thompson_init on already present aerosols.
! Run in the oracle build directory, with its own generated tables.
program fork_start_emission
  use module_mp_thompson, only: thompson_init
  implicit none
  real :: hgt(6,3,2), nwfa(6,3,2), nifa(6,3,2), emission(6,2)
  real :: spacing(5), numbers(6), dx, dy
  integer :: i, j, k, arm
  spacing = [1000., 3000., 12000., 20000., 25000.]
  numbers = [0., 1.e6, 1.e7, 1.e8, 1.e9, 9.999e9]
  hgt(:,1,:) = 100.
  hgt(:,2,:) = 200.
  hgt(:,3,:) = 300.
  nifa = 5.e5
  do arm = 1, 5
    dx = spacing(arm)
    dy = dx * 2.
    do j = 1, 2
      do k = 1, 3
        do i = 1, 6
          nwfa(i,k,j) = numbers(i)
        enddo
      enddo
    enddo
    emission = 123.
    call thompson_init(hgt=hgt, nwfa=nwfa, nifa=nifa, nwfa2d=emission, &
         dx=dx, dy=dy, is_start=.true.,                            &
         ids=1, ide=7, jds=1, jde=3, kds=1, kde=3,                &
         ims=1, ime=6, jms=1, jme=2, kms=1, kme=3,                &
         its=1, ite=6, jts=1, jte=2, kts=1, kte=3)
    write(*,'(8(ES16.8E3,1X))') dx, dy, emission(:,1)
  enddo
end program fork_start_emission
