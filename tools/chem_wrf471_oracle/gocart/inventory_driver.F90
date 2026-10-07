! Drive WRF's emiss_opt=6 GOCART anthropogenic add on fixed columns.
! One case per (kemit, dtstep) variant; each case is a 64 x 1 patch of
! 12-level columns with a spread of densities, layer depths and emission
! magnitudes (zero, trace, urban, point-source strength, cycling with the
! column index and perturbed by a fixed irrational step so no two columns
! repeat), all six emitted species, and records chem before and after.
program inventory_driver
  use module_configure
  use module_state_description
  use inventory_harness
  use oracle_io
  implicit none
  integer, parameter :: nx = 64, nz = 12
  type(grid_config_rec_type) :: cfg
  real :: chem(nx, nz, 1, num_chem), chem0(nx, nz, 1, num_chem)
  real :: rho(nx, nz, 1), alt(nx, nz, 1), dz8w(nx, nz, 1)
  real, allocatable :: emis(:, :, :, :)
  character(len=1024) :: root
  character(len=64) :: case_name
  integer :: v, i, k, n, kemit
  real :: dt
  integer, parameter :: kemits(4) = (/ 1, 3, 9, 12 /)
  real, parameter :: dts(4) = (/ 36.0, 7.5, 60.0, 18.0 /)

  call oracle_root(root)
  do i = 1, nx
    do k = 1, nz
      rho(i, k, 1) = 1.2 * exp(-real(k - 1) * 0.11) * (1.0 + 0.0037 * i)
      alt(i, k, 1) = 1.0 / (rho(i, k, 1) * (0.99 - 0.0002 * i))
      dz8w(i, k, 1) = 20.0 + 35.0 * real(k - 1) + 0.61803 * i
      do n = 1, num_chem
        chem0(i, k, 1, n) = 1.0e-3 * real(n) * (1.0 + 0.1 * k) / (1.0 + 0.137 * i)
      end do
    end do
  end do
  do v = 1, 4
    kemit = kemits(v)
    dt = dts(v)
    cfg%emiss_opt = 6
    cfg%kemit = kemit
    allocate(emis(nx, kemit, 1, num_emis_ant))
    emis = 0.0
    do i = 1, nx
      do k = 1, kemit
        do n = param_first_scalar, num_emis_ant
          select case (mod(i - 1, 4))
          case (0); emis(i, k, 1, n) = 0.0
          case (1); emis(i, k, 1, n) = 1.3e-3 * real(n) / real(k) * (1.0 + 0.0271 * i)
          case (2); emis(i, k, 1, n) = 0.77 * real(n) / real(k * k) * (1.0 + 0.0313 * i)
          case default; emis(i, k, 1, n) = 245.0 * real(n) / real(k) * (1.0 + 0.0173 * i)
          end select
        end do
      end do
    end do
    chem = chem0
    ! ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte
    call gocart_emiss_opt6(cfg, chem, emis, rho, alt, dz8w, dt,           &
                           1, nx, 1, 1, 1, nz, 1, nx, 1, 1, 1, nz)
    write(case_name, '(A,I0,A,I0)') 'kemit', kemit, '_v', v
    call oracle_open(trim(case_name))
    call oracle_put('kemit', kemit)
    call oracle_put('dtstep', dt)
    call oracle_put('rho_phy', rho)
    call oracle_put('alt', alt)
    call oracle_put('dz8w', dz8w)
    call oracle_put('emis_ant', emis)
    call oracle_put('chem_in', chem0)
    call oracle_put('chem_out', chem)
    call oracle_close()
    deallocate(emis)
  end do
end program inventory_driver
