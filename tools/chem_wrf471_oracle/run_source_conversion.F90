program run_source_conversion
  use oracle_io
  implicit none
  real, parameter :: RD=287.04
  real :: PMID(1,1,8), T(1,1,8), SMOKE(1,1,8,1), GRID1(1,1)
  real :: mass(8), density(8), inverse(8), inputs(8,3)
  integer :: I,J,LL
  character(len=1024) :: root
  call oracle_root(root)
  I=1
  J=1
  PMID(1,1,:)=[100000.,90000.,50000.,10000.,98000.,40000.,80000.,20000.]
  T(1,1,:)=[300.,290.,260.,220.,280.,250.,310.,230.]
  SMOKE(1,1,:,1)=[0.,1.,7.,30.,0.001,1000.,0.5,15.]
  do LL=1,8
#ifdef POISON
    GRID1=huge(0.0)
    density(LL)=-huge(0.0)
    inverse(LL)=-huge(0.0)
#endif
! UPP MDLFLD.f:2442 at 1296eebb295251d0fe1cc697f47058c62d887977.
                 GRID1(I,J) = (1./RD)*(PMID(I,J,LL)/T(I,J,LL))*SMOKE(I,J,LL,1)/(1E9)
    mass(LL)=GRID1(I,J)
    density(LL)=(1./RD)*(PMID(I,J,LL)/T(I,J,LL))
    inverse(LL)=mass(LL)/density(LL)*1E9
  end do
  inputs(:,1)=mass
  inputs(:,2)=PMID(1,1,:)
  inputs(:,3)=T(1,1,:)
  call oracle_open('source_cells')
  call oracle_put('inputs',inputs)
  call oracle_put('density',density)
  call oracle_put('inverse',inverse)
  call oracle_put('source_mixing_ratio',SMOKE(1,1,:,1))
  call oracle_close()
end program
