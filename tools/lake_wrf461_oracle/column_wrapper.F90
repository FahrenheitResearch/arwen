subroutine lake_init_columns(n,seed,columns,statics,use_depth,depth_flag,default_depth,errors) bind(c)
use iso_c_binding
use module_sf_lake, only: wrf_lake=>lake,wrf_lakeini=>lakeini
implicit none
integer(c_int),value :: n
integer(c_int) :: errors(n)
real(c_float) :: columns(n,131), statics(n,71)
integer(c_int),value :: use_depth,depth_flag
real(c_float),value :: default_depth
real(c_float) :: seed(n,5)
integer :: col
integer :: iswater
real :: xice_threshold
real :: xice(1:1,1:1)
real :: tsk(1:1,1:1)
real :: xland(1:1,1:1)
real :: lakemask(1:1,1:1)
integer :: lakeflag
integer :: lake_depth_flag
integer :: use_lakedepth
logical :: restart
integer :: ims
integer :: ime
integer :: jms
integer :: jme
integer :: its
integer :: ite
integer :: jts
integer :: jte
integer :: ivgtyp(1:1,1:1)
integer :: isltyp(1:1,1:1)
real :: ht(1:1,1:1)
real :: snow(1:1,1:1)
real :: lakedepth_default
real :: lake_min_elev
real :: lakedepth2d(1:1,1:1)
real :: savedtke12d(1:1,1:1)
real :: snowdp2d(1:1,1:1)
real :: h2osno2d(1:1,1:1)
real :: snl2d(1:1,1:1)
real :: t_grnd2d(1:1,1:1)
real :: t_lake3d(1:1,1:10,1:1)
real :: lake_icefrac3d(1:1,1:10,1:1)
real :: z_lake3d(1:1,1:10,1:1)
real :: dz_lake3d(1:1,1:10,1:1)
real :: t_soisno3d(1:1,-4:10,1:1)
real :: h2osoi_ice3d(1:1,-4:10,1:1)
real :: h2osoi_liq3d(1:1,-4:10,1:1)
real :: h2osoi_vol3d(1:1,-4:10,1:1)
real :: z3d(1:1,-4:10,1:1)
real :: dz3d(1:1,-4:10,1:1)
real :: watsat3d(1:1,1:10,1:1)
real :: csol3d(1:1,1:10,1:1)
real :: tkmg3d(1:1,1:10,1:1)
real :: tkdry3d(1:1,1:10,1:1)
real :: tksatu3d(1:1,1:10,1:1)
real :: zi3d(1:1,-5:10,1:1)
logical :: lake(1:1,1:1)
real :: lake_depth(1:1,1:1)
do col=1,n
iswater=0
xice_threshold=0
xice=0
tsk=0
xland=0
lakemask=0
lakeflag=0
lake_depth_flag=0
use_lakedepth=0
restart=.false.
ims=0
ime=0
jms=0
jme=0
its=0
ite=0
jts=0
jte=0
ivgtyp=0
isltyp=0
ht=0
snow=0
lakedepth_default=0
lake_min_elev=0
lakedepth2d=0
savedtke12d=0
snowdp2d=0
h2osno2d=0
snl2d=0
t_grnd2d=0
t_lake3d=0
lake_icefrac3d=0
z_lake3d=0
dz_lake3d=0
t_soisno3d=0
h2osoi_ice3d=0
h2osoi_liq3d=0
h2osoi_vol3d=0
z3d=0
dz3d=0
watsat3d=0
csol3d=0
tkmg3d=0
tkdry3d=0
tksatu3d=0
zi3d=0
lake=.false.
lake_depth=0
ims=1
ime=1
jms=1
jme=1
its=1
ite=1
jts=1
jte=1
lakemask=1
ivgtyp=17
iswater=17
xland=2
ht=100
lake_min_elev=5
xice_threshold=0.5
lakeflag=1
use_lakedepth=use_depth
lake_depth_flag=depth_flag
lakedepth_default=default_depth
isltyp(1,1)=seed(col,1)
lake_depth(1,1)=seed(col,2)
tsk(1,1)=seed(col,3)
snow(1,1)=seed(col,4)
xice(1,1)=seed(col,5)
call wrf_lakeini(ivgtyp, &
isltyp, &
ht, &
snow, &
lake_min_elev, &
restart, &
lakedepth_default, &
lake_depth, &
lakedepth2d, &
savedtke12d, &
snowdp2d, &
h2osno2d, &
snl2d, &
t_grnd2d, &
t_lake3d, &
lake_icefrac3d, &
z_lake3d, &
dz_lake3d, &
t_soisno3d, &
h2osoi_ice3d, &
h2osoi_liq3d, &
h2osoi_vol3d, &
z3d, &
dz3d, &
zi3d, &
watsat3d, &
csol3d, &
tkmg3d, &
iswater, &
xice, &
xice_threshold, &
xland, &
tsk, &
lakemask, &
lakeflag, &
lake_depth_flag, &
use_lakedepth, &
tkdry3d, &
tksatu3d, &
lake, &
its, &
ite, &
jts, &
jte, &
ims, &
ime, &
jms, &
jme)
columns(col,1)=savedtke12d(1,1)
columns(col,2)=snowdp2d(1,1)
columns(col,3)=h2osno2d(1,1)
columns(col,4)=snl2d(1,1)
columns(col,5)=t_grnd2d(1,1)
columns(col,6)=t_lake3d(1,1,1)
columns(col,7)=t_lake3d(1,2,1)
columns(col,8)=t_lake3d(1,3,1)
columns(col,9)=t_lake3d(1,4,1)
columns(col,10)=t_lake3d(1,5,1)
columns(col,11)=t_lake3d(1,6,1)
columns(col,12)=t_lake3d(1,7,1)
columns(col,13)=t_lake3d(1,8,1)
columns(col,14)=t_lake3d(1,9,1)
columns(col,15)=t_lake3d(1,10,1)
columns(col,16)=lake_icefrac3d(1,1,1)
columns(col,17)=lake_icefrac3d(1,2,1)
columns(col,18)=lake_icefrac3d(1,3,1)
columns(col,19)=lake_icefrac3d(1,4,1)
columns(col,20)=lake_icefrac3d(1,5,1)
columns(col,21)=lake_icefrac3d(1,6,1)
columns(col,22)=lake_icefrac3d(1,7,1)
columns(col,23)=lake_icefrac3d(1,8,1)
columns(col,24)=lake_icefrac3d(1,9,1)
columns(col,25)=lake_icefrac3d(1,10,1)
columns(col,26)=t_soisno3d(1,-4,1)
columns(col,27)=t_soisno3d(1,-3,1)
columns(col,28)=t_soisno3d(1,-2,1)
columns(col,29)=t_soisno3d(1,-1,1)
columns(col,30)=t_soisno3d(1,0,1)
columns(col,31)=t_soisno3d(1,1,1)
columns(col,32)=t_soisno3d(1,2,1)
columns(col,33)=t_soisno3d(1,3,1)
columns(col,34)=t_soisno3d(1,4,1)
columns(col,35)=t_soisno3d(1,5,1)
columns(col,36)=t_soisno3d(1,6,1)
columns(col,37)=t_soisno3d(1,7,1)
columns(col,38)=t_soisno3d(1,8,1)
columns(col,39)=t_soisno3d(1,9,1)
columns(col,40)=t_soisno3d(1,10,1)
columns(col,41)=h2osoi_ice3d(1,-4,1)
columns(col,42)=h2osoi_ice3d(1,-3,1)
columns(col,43)=h2osoi_ice3d(1,-2,1)
columns(col,44)=h2osoi_ice3d(1,-1,1)
columns(col,45)=h2osoi_ice3d(1,0,1)
columns(col,46)=h2osoi_ice3d(1,1,1)
columns(col,47)=h2osoi_ice3d(1,2,1)
columns(col,48)=h2osoi_ice3d(1,3,1)
columns(col,49)=h2osoi_ice3d(1,4,1)
columns(col,50)=h2osoi_ice3d(1,5,1)
columns(col,51)=h2osoi_ice3d(1,6,1)
columns(col,52)=h2osoi_ice3d(1,7,1)
columns(col,53)=h2osoi_ice3d(1,8,1)
columns(col,54)=h2osoi_ice3d(1,9,1)
columns(col,55)=h2osoi_ice3d(1,10,1)
columns(col,56)=h2osoi_liq3d(1,-4,1)
columns(col,57)=h2osoi_liq3d(1,-3,1)
columns(col,58)=h2osoi_liq3d(1,-2,1)
columns(col,59)=h2osoi_liq3d(1,-1,1)
columns(col,60)=h2osoi_liq3d(1,0,1)
columns(col,61)=h2osoi_liq3d(1,1,1)
columns(col,62)=h2osoi_liq3d(1,2,1)
columns(col,63)=h2osoi_liq3d(1,3,1)
columns(col,64)=h2osoi_liq3d(1,4,1)
columns(col,65)=h2osoi_liq3d(1,5,1)
columns(col,66)=h2osoi_liq3d(1,6,1)
columns(col,67)=h2osoi_liq3d(1,7,1)
columns(col,68)=h2osoi_liq3d(1,8,1)
columns(col,69)=h2osoi_liq3d(1,9,1)
columns(col,70)=h2osoi_liq3d(1,10,1)
columns(col,71)=h2osoi_vol3d(1,-4,1)
columns(col,72)=h2osoi_vol3d(1,-3,1)
columns(col,73)=h2osoi_vol3d(1,-2,1)
columns(col,74)=h2osoi_vol3d(1,-1,1)
columns(col,75)=h2osoi_vol3d(1,0,1)
columns(col,76)=h2osoi_vol3d(1,1,1)
columns(col,77)=h2osoi_vol3d(1,2,1)
columns(col,78)=h2osoi_vol3d(1,3,1)
columns(col,79)=h2osoi_vol3d(1,4,1)
columns(col,80)=h2osoi_vol3d(1,5,1)
columns(col,81)=h2osoi_vol3d(1,6,1)
columns(col,82)=h2osoi_vol3d(1,7,1)
columns(col,83)=h2osoi_vol3d(1,8,1)
columns(col,84)=h2osoi_vol3d(1,9,1)
columns(col,85)=h2osoi_vol3d(1,10,1)
columns(col,86)=z3d(1,-4,1)
columns(col,87)=z3d(1,-3,1)
columns(col,88)=z3d(1,-2,1)
columns(col,89)=z3d(1,-1,1)
columns(col,90)=z3d(1,0,1)
columns(col,91)=z3d(1,1,1)
columns(col,92)=z3d(1,2,1)
columns(col,93)=z3d(1,3,1)
columns(col,94)=z3d(1,4,1)
columns(col,95)=z3d(1,5,1)
columns(col,96)=z3d(1,6,1)
columns(col,97)=z3d(1,7,1)
columns(col,98)=z3d(1,8,1)
columns(col,99)=z3d(1,9,1)
columns(col,100)=z3d(1,10,1)
columns(col,101)=dz3d(1,-4,1)
columns(col,102)=dz3d(1,-3,1)
columns(col,103)=dz3d(1,-2,1)
columns(col,104)=dz3d(1,-1,1)
columns(col,105)=dz3d(1,0,1)
columns(col,106)=dz3d(1,1,1)
columns(col,107)=dz3d(1,2,1)
columns(col,108)=dz3d(1,3,1)
columns(col,109)=dz3d(1,4,1)
columns(col,110)=dz3d(1,5,1)
columns(col,111)=dz3d(1,6,1)
columns(col,112)=dz3d(1,7,1)
columns(col,113)=dz3d(1,8,1)
columns(col,114)=dz3d(1,9,1)
columns(col,115)=dz3d(1,10,1)
columns(col,116)=zi3d(1,-5,1)
columns(col,117)=zi3d(1,-4,1)
columns(col,118)=zi3d(1,-3,1)
columns(col,119)=zi3d(1,-2,1)
columns(col,120)=zi3d(1,-1,1)
columns(col,121)=zi3d(1,0,1)
columns(col,122)=zi3d(1,1,1)
columns(col,123)=zi3d(1,2,1)
columns(col,124)=zi3d(1,3,1)
columns(col,125)=zi3d(1,4,1)
columns(col,126)=zi3d(1,5,1)
columns(col,127)=zi3d(1,6,1)
columns(col,128)=zi3d(1,7,1)
columns(col,129)=zi3d(1,8,1)
columns(col,130)=zi3d(1,9,1)
columns(col,131)=zi3d(1,10,1)
statics(col,1)=lakedepth2d(1,1)
statics(col,2)=z_lake3d(1,1,1)
statics(col,3)=z_lake3d(1,2,1)
statics(col,4)=z_lake3d(1,3,1)
statics(col,5)=z_lake3d(1,4,1)
statics(col,6)=z_lake3d(1,5,1)
statics(col,7)=z_lake3d(1,6,1)
statics(col,8)=z_lake3d(1,7,1)
statics(col,9)=z_lake3d(1,8,1)
statics(col,10)=z_lake3d(1,9,1)
statics(col,11)=z_lake3d(1,10,1)
statics(col,12)=dz_lake3d(1,1,1)
statics(col,13)=dz_lake3d(1,2,1)
statics(col,14)=dz_lake3d(1,3,1)
statics(col,15)=dz_lake3d(1,4,1)
statics(col,16)=dz_lake3d(1,5,1)
statics(col,17)=dz_lake3d(1,6,1)
statics(col,18)=dz_lake3d(1,7,1)
statics(col,19)=dz_lake3d(1,8,1)
statics(col,20)=dz_lake3d(1,9,1)
statics(col,21)=dz_lake3d(1,10,1)
statics(col,22)=watsat3d(1,1,1)
statics(col,23)=watsat3d(1,2,1)
statics(col,24)=watsat3d(1,3,1)
statics(col,25)=watsat3d(1,4,1)
statics(col,26)=watsat3d(1,5,1)
statics(col,27)=watsat3d(1,6,1)
statics(col,28)=watsat3d(1,7,1)
statics(col,29)=watsat3d(1,8,1)
statics(col,30)=watsat3d(1,9,1)
statics(col,31)=watsat3d(1,10,1)
statics(col,32)=csol3d(1,1,1)
statics(col,33)=csol3d(1,2,1)
statics(col,34)=csol3d(1,3,1)
statics(col,35)=csol3d(1,4,1)
statics(col,36)=csol3d(1,5,1)
statics(col,37)=csol3d(1,6,1)
statics(col,38)=csol3d(1,7,1)
statics(col,39)=csol3d(1,8,1)
statics(col,40)=csol3d(1,9,1)
statics(col,41)=csol3d(1,10,1)
statics(col,42)=tkmg3d(1,1,1)
statics(col,43)=tkmg3d(1,2,1)
statics(col,44)=tkmg3d(1,3,1)
statics(col,45)=tkmg3d(1,4,1)
statics(col,46)=tkmg3d(1,5,1)
statics(col,47)=tkmg3d(1,6,1)
statics(col,48)=tkmg3d(1,7,1)
statics(col,49)=tkmg3d(1,8,1)
statics(col,50)=tkmg3d(1,9,1)
statics(col,51)=tkmg3d(1,10,1)
statics(col,52)=tkdry3d(1,1,1)
statics(col,53)=tkdry3d(1,2,1)
statics(col,54)=tkdry3d(1,3,1)
statics(col,55)=tkdry3d(1,4,1)
statics(col,56)=tkdry3d(1,5,1)
statics(col,57)=tkdry3d(1,6,1)
statics(col,58)=tkdry3d(1,7,1)
statics(col,59)=tkdry3d(1,8,1)
statics(col,60)=tkdry3d(1,9,1)
statics(col,61)=tkdry3d(1,10,1)
statics(col,62)=tksatu3d(1,1,1)
statics(col,63)=tksatu3d(1,2,1)
statics(col,64)=tksatu3d(1,3,1)
statics(col,65)=tksatu3d(1,4,1)
statics(col,66)=tksatu3d(1,5,1)
statics(col,67)=tksatu3d(1,6,1)
statics(col,68)=tksatu3d(1,7,1)
statics(col,69)=tksatu3d(1,8,1)
statics(col,70)=tksatu3d(1,9,1)
statics(col,71)=tksatu3d(1,10,1)
errors(col)=0
enddo
end subroutine
subroutine lake_step_columns(n,forcing,columns,statics,output,dt,errors) bind(c)
use iso_c_binding
use module_sf_lake, only: wrf_lake=>lake,wrf_lakeini=>lakeini
implicit none
integer(c_int),value :: n
integer(c_int) :: errors(n)
real(c_float) :: columns(n,131), statics(n,71)
real(c_float),value :: dt
real(c_float) :: forcing(n,13),output(n,9)
integer :: col
integer :: ids
integer :: ide
integer :: jds
integer :: jde
integer :: kds
integer :: kde
integer :: ims
integer :: ime
integer :: jms
integer :: jme
integer :: kms
integer :: kme
integer :: its
integer :: ite
integer :: jts
integer :: jte
integer :: kts
integer :: kte
integer :: iswater
real :: xice_threshold
real :: xice(1:1,1:1)
real :: lakemask(1:1,1:1)
real :: t_phy(1:1,1:2,1:1)
real :: p8w(1:1,1:2,1:1)
real :: dz8w(1:1,1:2,1:1)
real :: qvcurr(1:1,1:2,1:1)
real :: u_phy(1:1,1:2,1:1)
real :: v_phy(1:1,1:2,1:1)
real :: glw(1:1,1:1)
real :: emiss(1:1,1:1)
real :: rainbl(1:1,1:1)
real :: swdown(1:1,1:1)
real :: albedo(1:1,1:1)
real :: xland(1:1,1:1)
real :: xlat_urb2d(1:1,1:1)
integer :: ivgtyp(1:1,1:1)
real :: dtbl
real :: z_lake3d(1:1,1:10,1:1)
real :: dz_lake3d(1:1,1:10,1:1)
real :: watsat3d(1:1,1:10,1:1)
real :: csol3d(1:1,1:10,1:1)
real :: tkmg3d(1:1,1:10,1:1)
real :: tkdry3d(1:1,1:10,1:1)
real :: tksatu3d(1:1,1:10,1:1)
real :: lakedepth2d(1:1,1:1)
real :: ht(1:1,1:1)
real :: lake_min_elev
real :: hfx(1:1,1:1)
real :: lh(1:1,1:1)
real :: grdflx(1:1,1:1)
real :: tsk(1:1,1:1)
real :: qfx(1:1,1:1)
real :: t2(1:1,1:1)
real :: th2(1:1,1:1)
real :: q2(1:1,1:1)
real :: savedtke12d(1:1,1:1)
real :: snowdp2d(1:1,1:1)
real :: h2osno2d(1:1,1:1)
real :: snl2d(1:1,1:1)
real :: t_grnd2d(1:1,1:1)
real :: t_lake3d(1:1,1:10,1:1)
real :: lake_icefrac3d(1:1,1:10,1:1)
real :: t_soisno3d(1:1,-4:10,1:1)
real :: h2osoi_ice3d(1:1,-4:10,1:1)
real :: h2osoi_liq3d(1:1,-4:10,1:1)
real :: h2osoi_vol3d(1:1,-4:10,1:1)
real :: z3d(1:1,-4:10,1:1)
real :: dz3d(1:1,-4:10,1:1)
real :: zi3d(1:1,-5:10,1:1)
do col=1,n
ids=0
ide=0
jds=0
jde=0
kds=0
kde=0
ims=0
ime=0
jms=0
jme=0
kms=0
kme=0
its=0
ite=0
jts=0
jte=0
kts=0
kte=0
iswater=0
xice_threshold=0
xice=0
lakemask=0
t_phy=0
p8w=0
dz8w=0
qvcurr=0
u_phy=0
v_phy=0
glw=0
emiss=0
rainbl=0
swdown=0
albedo=0
xland=0
xlat_urb2d=0
ivgtyp=0
dtbl=0
z_lake3d=0
dz_lake3d=0
watsat3d=0
csol3d=0
tkmg3d=0
tkdry3d=0
tksatu3d=0
lakedepth2d=0
ht=0
lake_min_elev=0
hfx=0
lh=0
grdflx=0
tsk=0
qfx=0
t2=0
th2=0
q2=0
savedtke12d=0
snowdp2d=0
h2osno2d=0
snl2d=0
t_grnd2d=0
t_lake3d=0
lake_icefrac3d=0
t_soisno3d=0
h2osoi_ice3d=0
h2osoi_liq3d=0
h2osoi_vol3d=0
z3d=0
dz3d=0
zi3d=0
ids=1
ide=1
jds=1
jde=1
kds=1
kde=2
ims=1
ime=1
jms=1
jme=1
kms=1
kme=2
its=1
ite=1
jts=1
jte=1
kts=1
kte=1
lakemask=1
ivgtyp=17
iswater=17
xland=2
ht=100
lake_min_elev=5
xice_threshold=0.5
dtbl=dt
savedtke12d(1,1)=columns(col,1)
snowdp2d(1,1)=columns(col,2)
h2osno2d(1,1)=columns(col,3)
snl2d(1,1)=columns(col,4)
t_grnd2d(1,1)=columns(col,5)
t_lake3d(1,1,1)=columns(col,6)
t_lake3d(1,2,1)=columns(col,7)
t_lake3d(1,3,1)=columns(col,8)
t_lake3d(1,4,1)=columns(col,9)
t_lake3d(1,5,1)=columns(col,10)
t_lake3d(1,6,1)=columns(col,11)
t_lake3d(1,7,1)=columns(col,12)
t_lake3d(1,8,1)=columns(col,13)
t_lake3d(1,9,1)=columns(col,14)
t_lake3d(1,10,1)=columns(col,15)
lake_icefrac3d(1,1,1)=columns(col,16)
lake_icefrac3d(1,2,1)=columns(col,17)
lake_icefrac3d(1,3,1)=columns(col,18)
lake_icefrac3d(1,4,1)=columns(col,19)
lake_icefrac3d(1,5,1)=columns(col,20)
lake_icefrac3d(1,6,1)=columns(col,21)
lake_icefrac3d(1,7,1)=columns(col,22)
lake_icefrac3d(1,8,1)=columns(col,23)
lake_icefrac3d(1,9,1)=columns(col,24)
lake_icefrac3d(1,10,1)=columns(col,25)
t_soisno3d(1,-4,1)=columns(col,26)
t_soisno3d(1,-3,1)=columns(col,27)
t_soisno3d(1,-2,1)=columns(col,28)
t_soisno3d(1,-1,1)=columns(col,29)
t_soisno3d(1,0,1)=columns(col,30)
t_soisno3d(1,1,1)=columns(col,31)
t_soisno3d(1,2,1)=columns(col,32)
t_soisno3d(1,3,1)=columns(col,33)
t_soisno3d(1,4,1)=columns(col,34)
t_soisno3d(1,5,1)=columns(col,35)
t_soisno3d(1,6,1)=columns(col,36)
t_soisno3d(1,7,1)=columns(col,37)
t_soisno3d(1,8,1)=columns(col,38)
t_soisno3d(1,9,1)=columns(col,39)
t_soisno3d(1,10,1)=columns(col,40)
h2osoi_ice3d(1,-4,1)=columns(col,41)
h2osoi_ice3d(1,-3,1)=columns(col,42)
h2osoi_ice3d(1,-2,1)=columns(col,43)
h2osoi_ice3d(1,-1,1)=columns(col,44)
h2osoi_ice3d(1,0,1)=columns(col,45)
h2osoi_ice3d(1,1,1)=columns(col,46)
h2osoi_ice3d(1,2,1)=columns(col,47)
h2osoi_ice3d(1,3,1)=columns(col,48)
h2osoi_ice3d(1,4,1)=columns(col,49)
h2osoi_ice3d(1,5,1)=columns(col,50)
h2osoi_ice3d(1,6,1)=columns(col,51)
h2osoi_ice3d(1,7,1)=columns(col,52)
h2osoi_ice3d(1,8,1)=columns(col,53)
h2osoi_ice3d(1,9,1)=columns(col,54)
h2osoi_ice3d(1,10,1)=columns(col,55)
h2osoi_liq3d(1,-4,1)=columns(col,56)
h2osoi_liq3d(1,-3,1)=columns(col,57)
h2osoi_liq3d(1,-2,1)=columns(col,58)
h2osoi_liq3d(1,-1,1)=columns(col,59)
h2osoi_liq3d(1,0,1)=columns(col,60)
h2osoi_liq3d(1,1,1)=columns(col,61)
h2osoi_liq3d(1,2,1)=columns(col,62)
h2osoi_liq3d(1,3,1)=columns(col,63)
h2osoi_liq3d(1,4,1)=columns(col,64)
h2osoi_liq3d(1,5,1)=columns(col,65)
h2osoi_liq3d(1,6,1)=columns(col,66)
h2osoi_liq3d(1,7,1)=columns(col,67)
h2osoi_liq3d(1,8,1)=columns(col,68)
h2osoi_liq3d(1,9,1)=columns(col,69)
h2osoi_liq3d(1,10,1)=columns(col,70)
h2osoi_vol3d(1,-4,1)=columns(col,71)
h2osoi_vol3d(1,-3,1)=columns(col,72)
h2osoi_vol3d(1,-2,1)=columns(col,73)
h2osoi_vol3d(1,-1,1)=columns(col,74)
h2osoi_vol3d(1,0,1)=columns(col,75)
h2osoi_vol3d(1,1,1)=columns(col,76)
h2osoi_vol3d(1,2,1)=columns(col,77)
h2osoi_vol3d(1,3,1)=columns(col,78)
h2osoi_vol3d(1,4,1)=columns(col,79)
h2osoi_vol3d(1,5,1)=columns(col,80)
h2osoi_vol3d(1,6,1)=columns(col,81)
h2osoi_vol3d(1,7,1)=columns(col,82)
h2osoi_vol3d(1,8,1)=columns(col,83)
h2osoi_vol3d(1,9,1)=columns(col,84)
h2osoi_vol3d(1,10,1)=columns(col,85)
z3d(1,-4,1)=columns(col,86)
z3d(1,-3,1)=columns(col,87)
z3d(1,-2,1)=columns(col,88)
z3d(1,-1,1)=columns(col,89)
z3d(1,0,1)=columns(col,90)
z3d(1,1,1)=columns(col,91)
z3d(1,2,1)=columns(col,92)
z3d(1,3,1)=columns(col,93)
z3d(1,4,1)=columns(col,94)
z3d(1,5,1)=columns(col,95)
z3d(1,6,1)=columns(col,96)
z3d(1,7,1)=columns(col,97)
z3d(1,8,1)=columns(col,98)
z3d(1,9,1)=columns(col,99)
z3d(1,10,1)=columns(col,100)
dz3d(1,-4,1)=columns(col,101)
dz3d(1,-3,1)=columns(col,102)
dz3d(1,-2,1)=columns(col,103)
dz3d(1,-1,1)=columns(col,104)
dz3d(1,0,1)=columns(col,105)
dz3d(1,1,1)=columns(col,106)
dz3d(1,2,1)=columns(col,107)
dz3d(1,3,1)=columns(col,108)
dz3d(1,4,1)=columns(col,109)
dz3d(1,5,1)=columns(col,110)
dz3d(1,6,1)=columns(col,111)
dz3d(1,7,1)=columns(col,112)
dz3d(1,8,1)=columns(col,113)
dz3d(1,9,1)=columns(col,114)
dz3d(1,10,1)=columns(col,115)
zi3d(1,-5,1)=columns(col,116)
zi3d(1,-4,1)=columns(col,117)
zi3d(1,-3,1)=columns(col,118)
zi3d(1,-2,1)=columns(col,119)
zi3d(1,-1,1)=columns(col,120)
zi3d(1,0,1)=columns(col,121)
zi3d(1,1,1)=columns(col,122)
zi3d(1,2,1)=columns(col,123)
zi3d(1,3,1)=columns(col,124)
zi3d(1,4,1)=columns(col,125)
zi3d(1,5,1)=columns(col,126)
zi3d(1,6,1)=columns(col,127)
zi3d(1,7,1)=columns(col,128)
zi3d(1,8,1)=columns(col,129)
zi3d(1,9,1)=columns(col,130)
zi3d(1,10,1)=columns(col,131)
lakedepth2d(1,1)=statics(col,1)
z_lake3d(1,1,1)=statics(col,2)
z_lake3d(1,2,1)=statics(col,3)
z_lake3d(1,3,1)=statics(col,4)
z_lake3d(1,4,1)=statics(col,5)
z_lake3d(1,5,1)=statics(col,6)
z_lake3d(1,6,1)=statics(col,7)
z_lake3d(1,7,1)=statics(col,8)
z_lake3d(1,8,1)=statics(col,9)
z_lake3d(1,9,1)=statics(col,10)
z_lake3d(1,10,1)=statics(col,11)
dz_lake3d(1,1,1)=statics(col,12)
dz_lake3d(1,2,1)=statics(col,13)
dz_lake3d(1,3,1)=statics(col,14)
dz_lake3d(1,4,1)=statics(col,15)
dz_lake3d(1,5,1)=statics(col,16)
dz_lake3d(1,6,1)=statics(col,17)
dz_lake3d(1,7,1)=statics(col,18)
dz_lake3d(1,8,1)=statics(col,19)
dz_lake3d(1,9,1)=statics(col,20)
dz_lake3d(1,10,1)=statics(col,21)
watsat3d(1,1,1)=statics(col,22)
watsat3d(1,2,1)=statics(col,23)
watsat3d(1,3,1)=statics(col,24)
watsat3d(1,4,1)=statics(col,25)
watsat3d(1,5,1)=statics(col,26)
watsat3d(1,6,1)=statics(col,27)
watsat3d(1,7,1)=statics(col,28)
watsat3d(1,8,1)=statics(col,29)
watsat3d(1,9,1)=statics(col,30)
watsat3d(1,10,1)=statics(col,31)
csol3d(1,1,1)=statics(col,32)
csol3d(1,2,1)=statics(col,33)
csol3d(1,3,1)=statics(col,34)
csol3d(1,4,1)=statics(col,35)
csol3d(1,5,1)=statics(col,36)
csol3d(1,6,1)=statics(col,37)
csol3d(1,7,1)=statics(col,38)
csol3d(1,8,1)=statics(col,39)
csol3d(1,9,1)=statics(col,40)
csol3d(1,10,1)=statics(col,41)
tkmg3d(1,1,1)=statics(col,42)
tkmg3d(1,2,1)=statics(col,43)
tkmg3d(1,3,1)=statics(col,44)
tkmg3d(1,4,1)=statics(col,45)
tkmg3d(1,5,1)=statics(col,46)
tkmg3d(1,6,1)=statics(col,47)
tkmg3d(1,7,1)=statics(col,48)
tkmg3d(1,8,1)=statics(col,49)
tkmg3d(1,9,1)=statics(col,50)
tkmg3d(1,10,1)=statics(col,51)
tkdry3d(1,1,1)=statics(col,52)
tkdry3d(1,2,1)=statics(col,53)
tkdry3d(1,3,1)=statics(col,54)
tkdry3d(1,4,1)=statics(col,55)
tkdry3d(1,5,1)=statics(col,56)
tkdry3d(1,6,1)=statics(col,57)
tkdry3d(1,7,1)=statics(col,58)
tkdry3d(1,8,1)=statics(col,59)
tkdry3d(1,9,1)=statics(col,60)
tkdry3d(1,10,1)=statics(col,61)
tksatu3d(1,1,1)=statics(col,62)
tksatu3d(1,2,1)=statics(col,63)
tksatu3d(1,3,1)=statics(col,64)
tksatu3d(1,4,1)=statics(col,65)
tksatu3d(1,5,1)=statics(col,66)
tksatu3d(1,6,1)=statics(col,67)
tksatu3d(1,7,1)=statics(col,68)
tksatu3d(1,8,1)=statics(col,69)
tksatu3d(1,9,1)=statics(col,70)
tksatu3d(1,10,1)=statics(col,71)
t_phy(1,1,1)=forcing(col,1)
p8w(1,1,1)=forcing(col,2)
p8w(1,2,1)=forcing(col,3)
dz8w(1,1,1)=forcing(col,4)
qvcurr(1,1,1)=forcing(col,5)
u_phy(1,1,1)=forcing(col,6)
v_phy(1,1,1)=forcing(col,7)
glw(1,1)=forcing(col,8)
emiss(1,1)=forcing(col,9)
rainbl(1,1)=forcing(col,10)
swdown(1,1)=forcing(col,11)
albedo(1,1)=forcing(col,12)
xlat_urb2d(1,1)=forcing(col,13)
call wrf_lake(t_phy, &
p8w, &
dz8w, &
qvcurr, &
u_phy, &
v_phy, &
glw, &
emiss, &
rainbl, &
dtbl, &
swdown, &
albedo, &
xlat_urb2d, &
z_lake3d, &
dz_lake3d, &
lakedepth2d, &
watsat3d, &
csol3d, &
tkmg3d, &
tkdry3d, &
tksatu3d, &
ivgtyp, &
ht, &
xland, &
iswater, &
xice, &
xice_threshold, &
lake_min_elev, &
ids, &
ide, &
jds, &
jde, &
kds, &
kde, &
ims, &
ime, &
jms, &
jme, &
kms, &
kme, &
its, &
ite, &
jts, &
jte, &
kts, &
kte, &
h2osno2d, &
snowdp2d, &
snl2d, &
z3d, &
dz3d, &
zi3d, &
h2osoi_vol3d, &
h2osoi_liq3d, &
h2osoi_ice3d, &
t_grnd2d, &
t_soisno3d, &
t_lake3d, &
savedtke12d, &
lake_icefrac3d, &
lakemask, &
hfx, &
lh, &
grdflx, &
tsk, &
qfx, &
t2, &
th2, &
q2)
columns(col,1)=savedtke12d(1,1)
columns(col,2)=snowdp2d(1,1)
columns(col,3)=h2osno2d(1,1)
columns(col,4)=snl2d(1,1)
columns(col,5)=t_grnd2d(1,1)
columns(col,6)=t_lake3d(1,1,1)
columns(col,7)=t_lake3d(1,2,1)
columns(col,8)=t_lake3d(1,3,1)
columns(col,9)=t_lake3d(1,4,1)
columns(col,10)=t_lake3d(1,5,1)
columns(col,11)=t_lake3d(1,6,1)
columns(col,12)=t_lake3d(1,7,1)
columns(col,13)=t_lake3d(1,8,1)
columns(col,14)=t_lake3d(1,9,1)
columns(col,15)=t_lake3d(1,10,1)
columns(col,16)=lake_icefrac3d(1,1,1)
columns(col,17)=lake_icefrac3d(1,2,1)
columns(col,18)=lake_icefrac3d(1,3,1)
columns(col,19)=lake_icefrac3d(1,4,1)
columns(col,20)=lake_icefrac3d(1,5,1)
columns(col,21)=lake_icefrac3d(1,6,1)
columns(col,22)=lake_icefrac3d(1,7,1)
columns(col,23)=lake_icefrac3d(1,8,1)
columns(col,24)=lake_icefrac3d(1,9,1)
columns(col,25)=lake_icefrac3d(1,10,1)
columns(col,26)=t_soisno3d(1,-4,1)
columns(col,27)=t_soisno3d(1,-3,1)
columns(col,28)=t_soisno3d(1,-2,1)
columns(col,29)=t_soisno3d(1,-1,1)
columns(col,30)=t_soisno3d(1,0,1)
columns(col,31)=t_soisno3d(1,1,1)
columns(col,32)=t_soisno3d(1,2,1)
columns(col,33)=t_soisno3d(1,3,1)
columns(col,34)=t_soisno3d(1,4,1)
columns(col,35)=t_soisno3d(1,5,1)
columns(col,36)=t_soisno3d(1,6,1)
columns(col,37)=t_soisno3d(1,7,1)
columns(col,38)=t_soisno3d(1,8,1)
columns(col,39)=t_soisno3d(1,9,1)
columns(col,40)=t_soisno3d(1,10,1)
columns(col,41)=h2osoi_ice3d(1,-4,1)
columns(col,42)=h2osoi_ice3d(1,-3,1)
columns(col,43)=h2osoi_ice3d(1,-2,1)
columns(col,44)=h2osoi_ice3d(1,-1,1)
columns(col,45)=h2osoi_ice3d(1,0,1)
columns(col,46)=h2osoi_ice3d(1,1,1)
columns(col,47)=h2osoi_ice3d(1,2,1)
columns(col,48)=h2osoi_ice3d(1,3,1)
columns(col,49)=h2osoi_ice3d(1,4,1)
columns(col,50)=h2osoi_ice3d(1,5,1)
columns(col,51)=h2osoi_ice3d(1,6,1)
columns(col,52)=h2osoi_ice3d(1,7,1)
columns(col,53)=h2osoi_ice3d(1,8,1)
columns(col,54)=h2osoi_ice3d(1,9,1)
columns(col,55)=h2osoi_ice3d(1,10,1)
columns(col,56)=h2osoi_liq3d(1,-4,1)
columns(col,57)=h2osoi_liq3d(1,-3,1)
columns(col,58)=h2osoi_liq3d(1,-2,1)
columns(col,59)=h2osoi_liq3d(1,-1,1)
columns(col,60)=h2osoi_liq3d(1,0,1)
columns(col,61)=h2osoi_liq3d(1,1,1)
columns(col,62)=h2osoi_liq3d(1,2,1)
columns(col,63)=h2osoi_liq3d(1,3,1)
columns(col,64)=h2osoi_liq3d(1,4,1)
columns(col,65)=h2osoi_liq3d(1,5,1)
columns(col,66)=h2osoi_liq3d(1,6,1)
columns(col,67)=h2osoi_liq3d(1,7,1)
columns(col,68)=h2osoi_liq3d(1,8,1)
columns(col,69)=h2osoi_liq3d(1,9,1)
columns(col,70)=h2osoi_liq3d(1,10,1)
columns(col,71)=h2osoi_vol3d(1,-4,1)
columns(col,72)=h2osoi_vol3d(1,-3,1)
columns(col,73)=h2osoi_vol3d(1,-2,1)
columns(col,74)=h2osoi_vol3d(1,-1,1)
columns(col,75)=h2osoi_vol3d(1,0,1)
columns(col,76)=h2osoi_vol3d(1,1,1)
columns(col,77)=h2osoi_vol3d(1,2,1)
columns(col,78)=h2osoi_vol3d(1,3,1)
columns(col,79)=h2osoi_vol3d(1,4,1)
columns(col,80)=h2osoi_vol3d(1,5,1)
columns(col,81)=h2osoi_vol3d(1,6,1)
columns(col,82)=h2osoi_vol3d(1,7,1)
columns(col,83)=h2osoi_vol3d(1,8,1)
columns(col,84)=h2osoi_vol3d(1,9,1)
columns(col,85)=h2osoi_vol3d(1,10,1)
columns(col,86)=z3d(1,-4,1)
columns(col,87)=z3d(1,-3,1)
columns(col,88)=z3d(1,-2,1)
columns(col,89)=z3d(1,-1,1)
columns(col,90)=z3d(1,0,1)
columns(col,91)=z3d(1,1,1)
columns(col,92)=z3d(1,2,1)
columns(col,93)=z3d(1,3,1)
columns(col,94)=z3d(1,4,1)
columns(col,95)=z3d(1,5,1)
columns(col,96)=z3d(1,6,1)
columns(col,97)=z3d(1,7,1)
columns(col,98)=z3d(1,8,1)
columns(col,99)=z3d(1,9,1)
columns(col,100)=z3d(1,10,1)
columns(col,101)=dz3d(1,-4,1)
columns(col,102)=dz3d(1,-3,1)
columns(col,103)=dz3d(1,-2,1)
columns(col,104)=dz3d(1,-1,1)
columns(col,105)=dz3d(1,0,1)
columns(col,106)=dz3d(1,1,1)
columns(col,107)=dz3d(1,2,1)
columns(col,108)=dz3d(1,3,1)
columns(col,109)=dz3d(1,4,1)
columns(col,110)=dz3d(1,5,1)
columns(col,111)=dz3d(1,6,1)
columns(col,112)=dz3d(1,7,1)
columns(col,113)=dz3d(1,8,1)
columns(col,114)=dz3d(1,9,1)
columns(col,115)=dz3d(1,10,1)
columns(col,116)=zi3d(1,-5,1)
columns(col,117)=zi3d(1,-4,1)
columns(col,118)=zi3d(1,-3,1)
columns(col,119)=zi3d(1,-2,1)
columns(col,120)=zi3d(1,-1,1)
columns(col,121)=zi3d(1,0,1)
columns(col,122)=zi3d(1,1,1)
columns(col,123)=zi3d(1,2,1)
columns(col,124)=zi3d(1,3,1)
columns(col,125)=zi3d(1,4,1)
columns(col,126)=zi3d(1,5,1)
columns(col,127)=zi3d(1,6,1)
columns(col,128)=zi3d(1,7,1)
columns(col,129)=zi3d(1,8,1)
columns(col,130)=zi3d(1,9,1)
columns(col,131)=zi3d(1,10,1)
output(col,1)=hfx(1,1)
output(col,2)=lh(1,1)
output(col,3)=grdflx(1,1)
output(col,4)=tsk(1,1)
output(col,5)=qfx(1,1)
output(col,6)=t2(1,1)
output(col,7)=th2(1,1)
output(col,8)=q2(1,1)
output(col,9)=albedo(1,1)
errors(col)=0
enddo
end subroutine
