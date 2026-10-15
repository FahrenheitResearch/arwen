! Column driver for WRF v4.6.1 Ferrier-Aligo microphysics (mp_physics=5,
! phys/module_mp_fer_hires.F, byte-unmodified).
!
! For every column in columns.in it does what WRF does for a run at that
! column's time step: fer_hires_init (GSMDT=0, so DTPH=DT; reads
! ETAMPNEW_DATA.expanded_rain, builds the saturation tables), then FER_HIRES
! once per step for NSTEPS steps, carrying the state (theta, qv, qc, qr, qi,
! F_ICE_PHY, F_RAIN_PHY, F_RIMEF_PHY, RAINNC) from step to step exactly as
! WRF's state arrays would.  QT is recomputed by FER_HIRES on entry.
!
! The patch is one point wide (its=ite=1, jts=jte=1), kts=1..kte=NZ with
! kde=NZ+1, as WRF's microphysics driver passes it.
!
! Outputs (floats as binary32 bit patterns, hex):
!   levels.csv  : col,step,k,th,qv,qt,qc,qr,qi,f_ice,f_rain,f_rimef
!   surface.csv : col,step,rainnc,rainncv,sr
! for step 1 and step NSTEPS.
!
! The main program must be compiled with -fconvert=big-endian: WRF reads
! the unformatted table with that runtime conversion.
program run_ferrier
  use module_mp_fer_hires, only: fer_hires, fer_hires_init
  implicit none

  integer :: ncol, nz, ic, k, step, nsteps, gid, iu, ul, us
  integer :: ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme
  integer :: its, ite, jts, jte, kts, kte
  integer :: hdt, hdx, hrain
  integer :: w(12)
  real :: dt, dx
  real, allocatable, dimension(:,:,:) :: p, pi, th, qv, qt, qc, qr, qi, &
       dz8w, rho, f_ice, f_rain, f_rimef
  real, dimension(1,1) :: rainnc, rainncv, sr
  integer, dimension(1,1) :: lowlyr

  if (kind(1.0) /= 4) stop 'default REAL is not binary32'

  open(newunit=iu, file='columns.in', status='old', action='read', &
       form='formatted')
  read(iu, *) ncol, nz

  ids = 1; ide = 2; jds = 1; jde = 2; kds = 1; kde = nz + 1
  ims = 1; ime = 1; jms = 1; jme = 1; kms = 1; kme = nz + 1
  its = 1; ite = 1; jts = 1; jte = 1; kts = 1; kte = nz

  allocate(p(1,kms:kme,1), pi(1,kms:kme,1), th(1,kms:kme,1), &
       qv(1,kms:kme,1), qt(1,kms:kme,1), qc(1,kms:kme,1), &
       qr(1,kms:kme,1), qi(1,kms:kme,1), dz8w(1,kms:kme,1), &
       rho(1,kms:kme,1), f_ice(1,kms:kme,1), f_rain(1,kms:kme,1), &
       f_rimef(1,kms:kme,1))

  open(newunit=ul, file='levels.csv', status='replace', action='write')
  open(newunit=us, file='surface.csv', status='replace', action='write')
  write(ul, '(A)') 'col,step,k,th,qv,qt,qc,qr,qi,f_ice,f_rain,f_rimef'
  write(us, '(A)') 'col,step,rainnc,rainncv,sr'

  do ic = 0, ncol - 1
     read(iu, '(I6,1X,I6,3(1X,Z8))') gid, nsteps, hdt, hdx, hrain
     dt = transfer(hdt, dt)
     dx = transfer(hdx, dx)
     rainnc(1,1) = transfer(hrain, dx)
     rainncv = 0.0
     sr = 0.0
     p = 0.0; pi = 1.0; th = 0.0; qv = 0.0; qt = 0.0; qc = 0.0; qr = 0.0
     qi = 0.0; dz8w = 0.0; rho = 0.0; f_ice = 0.0; f_rain = 0.0
     f_rimef = 1.0
     do k = 1, nz
        read(iu, '(12(Z8,1X))') w
        p(1,k,1)       = transfer(w(1), dt)
        pi(1,k,1)      = transfer(w(2), dt)
        th(1,k,1)      = transfer(w(3), dt)
        qv(1,k,1)      = transfer(w(4), dt)
        qc(1,k,1)      = transfer(w(5), dt)
        qr(1,k,1)      = transfer(w(6), dt)
        qi(1,k,1)      = transfer(w(7), dt)
        dz8w(1,k,1)    = transfer(w(8), dt)
        rho(1,k,1)     = transfer(w(9), dt)
        f_ice(1,k,1)   = transfer(w(10), dt)
        f_rain(1,k,1)  = transfer(w(11), dt)
        f_rimef(1,k,1) = transfer(w(12), dt)
     end do

     ! WRF calls the init once per run with the model time step; a run at
     ! this column's dt is what is reproduced.  The optional F_* arrays are
     ! not passed, so the column's own F_RIMEF input survives (a restart-like
     ! state, which is what a column taken mid-run carries).
     call fer_hires_init(0.0, dt, dx, dx, lowlyr, .false., .true., &
          ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, &
          its, ite, jts, jte, kts, kte)

     do step = 1, nsteps
        call fer_hires(step, dt, dx, dx, gid, rainnc, rainncv, &
             dz8w, rho, p, pi, th, qv, qt, lowlyr, sr, &
             f_ice, f_rain, f_rimef, qc, qr, qi, &
             ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, &
             its, ite, jts, jte, kts, kte)
        if (step == 1 .or. step == nsteps) then
           do k = 1, nz
              write(ul, '(I0,",",I0,",",I0,9(",",Z8.8))') ic, step, k, &
                   th(1,k,1), qv(1,k,1), qt(1,k,1), qc(1,k,1), qr(1,k,1), &
                   qi(1,k,1), f_ice(1,k,1), f_rain(1,k,1), f_rimef(1,k,1)
           end do
           write(us, '(I0,",",I0,3(",",Z8.8))') ic, step, rainnc(1,1), &
                rainncv(1,1), sr(1,1)
        end if
     end do
  end do
  close(iu); close(ul); close(us)
end program run_ferrier
