module rrtmg_module
contains
subroutine convert_bands(tauaer300,tauaer400,tauaer600,tauaer999,waer300,waer400,waer600,waer999, &
 gaer300,gaer400,gaer600,gaer999,tauaer,ssaaer,asmaer,nz)
 integer :: nz,i,j,k,nb,ncol,kts,kte
 integer,parameter :: nbndsw=14
 real :: tauaer300(1,nz,1),tauaer400(1,nz,1),tauaer600(1,nz,1),tauaer999(1,nz,1)
 real :: waer300(1,nz,1),waer400(1,nz,1),waer600(1,nz,1),waer999(1,nz,1)
 real :: gaer300(1,nz,1),gaer400(1,nz,1),gaer600(1,nz,1),gaer999(1,nz,1)
 real :: tauaer(1,nz,14),ssaaer(1,nz,14),asmaer(1,nz,14)
 character(256) :: msg
 include 'rrtmg_parameters.inc'
 i=1;j=1;ncol=1;kts=1;kte=nz
 ! WRF initializes the arrays before entering the feedback branch.
 tauaer=0.;ssaaer=1.;asmaer=0.
 include 'rrtmg_conversion.inc'
end subroutine
end module
subroutine wrf_message(msg)
 character(*) :: msg
 print *,trim(msg)
end subroutine
subroutine wrf_error_fatal(msg)
 character(*) :: msg
 print *,trim(msg)
 error stop 'RRTMG fatal'
end subroutine
