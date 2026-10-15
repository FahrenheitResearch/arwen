program snow_age_column
  use module_sf_noahmplsm, only: noahmp_parameters, noahmp_options, ALBEDO
  use module_sf_noahmp_glacier, only: RADIATION_GLACIER, SNOW_AGE_GLACIER
  use noahmp_glacier_globals, only: opt_alb_g => opt_alb
  implicit none
  type(noahmp_parameters) :: p
  character(len=1024) :: input, output
  integer :: ios, mode
  real :: x(52), o(41), fage, albold, tauss, age_tauss, age_fage
  real :: albgrd(2), albgri(2), albd(2), albi(2), fabd(2), fabi(2)
  real :: ftdd(2), ftid(2), ftii(2), frevd(2), frevi(2), fregd(2), fregi(2)
  real :: albsnd(2), albsni(2), fsun, bgap, wgap, sag, fsr, fsa
  call get_command_argument(1, input)
  call get_command_argument(2, output)
  call noahmp_options(4,1,1,3,1,1,1,3,2,1,2,1,1,1,1,0,0,0,1,0)
  opt_alb_g = 2
  if (storage_size(x(1)) /= 32) stop 'oracle requires binary32 REAL'
  open(10,file=trim(input),access='stream',form='unformatted',status='old')
  open(11,file=trim(output),access='stream',form='unformatted',status='replace')
  do
    read(10,iostat=ios) mode, x
    if (ios < 0) exit
    if (ios /= 0) stop 'truncated oracle input'
    albold=x(42); tauss=x(43); fage=x(44)
    if (mode == 0) then
      p%tau0=x(1); p%grain_growth=x(2); p%extra_growth=x(3)
      p%dirt_soot=x(4); p%swemx=x(5)
      p%albsat=x(6:7); p%albdry=x(8:9); p%alblak=x(10:11)
      p%rhol=x(12:13); p%rhos=x(14:15); p%taul=x(16:17); p%taus=x(18:19)
      p%xl=x(20); p%omegas=x(21:22); p%betads=x(23); p%betais=x(24)
      ! Seed caller storage. These four arrays are undefined at night in
      ! upstream ALBEDO and are excluded from numerical identity grading.
      frevd=x(45:46); frevi=x(47:48); fregd=x(49:50); fregi=x(51:52)
      call ALBEDO(p,11,1,0,4,x(25),x(26),fage,x(27),x(28), &
                  x(29),x(30),x(31),x(32),x(33),x(38:41), &
                  x(34),x(35),x(36),x(37),1,1,albold,tauss, &
                  albgrd,albgri,albd,albi,fabd,fabi,ftdd,ftid,ftii, &
                  fsun,frevi,frevd,fregd,fregi,bgap,wgap,albsnd,albsni)
      o(1:36)=[fage,albold,tauss,fsun,bgap,wgap,albgrd,albgri, &
               albd,albi,fabd,fabi,ftdd,ftid,ftii,frevd,frevi, &
               fregd,fregi,albsnd,albsni]
      o(37:41)=0.0
    else
      age_tauss=x(43)
      call SNOW_AGE_GLACIER(x(25),x(29),x(34),x(35),age_tauss,age_fage)
      call RADIATION_GLACIER(x(25),x(29),x(34),x(35),x(26), &
                            x(36),x(45:46),x(47:48),albold,tauss,sag,fsr,fsa)
      o=0.0
      o(1:5)=[albold,tauss,sag,fsr,fsa]
      o(6)=age_fage
    end if
    write(11) o
  end do
  close(10); close(11)
end program
