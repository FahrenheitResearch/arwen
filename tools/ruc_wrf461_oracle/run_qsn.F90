program run_ruc_qsn_oracle
  ! Drives the unmodified WRF v4.6.1 module_sf_ruclsm::qsn, the saturation
  ! lookup over the 5001-entry tbq table.  The sweep covers both ends of the
  ! table (the i<1 and i>5000 clamps and the exact first/last nodes) and the
  ! interpolation interior, including sub-interval probes inside a single
  ! 0.05 K bin and the ice/water switch at 273.15 K.
  !
  ! tbq is rebuilt here with exactly the arithmetic lsmruc uses; it is the
  ! same table pinned in gpuwm/data/ruc/oracle/tbq.csv, which is what the
  ! validator feeds to the gpuwm port, so each row is reproducible from the
  ! pinned pair alone.  r_raw is the unclamped index expression qsn forms
  ! internally, written out so the CSV records which branch each row took
  ! without the reader having to re-derive it.
  use module_sf_ruclsm, only: qsn
  implicit none

  integer, parameter :: nsample = 81
  real, parameter :: tn_list(nsample) = [ &
      ! --- below the table: qsn clamps i to 1 and r to 1.0 ---
      -50.0, 0.0, 100.0, 150.0, 170.0, 173.0, 173.10, 173.1400, 173.1499, &
      ! --- exactly the first node ---
      173.15, &
      ! --- interior: first bin, then a coarse sweep with fine probes ---
      173.16, 173.175, 173.19, 173.1999, &
      180.0, 190.0, 200.0, 200.0125, 200.025, 200.0375, 200.049, &
      210.0, 220.0, 230.0, 233.15, 240.0, &
      250.0, 250.001, 250.01, 250.02, 250.03, 250.04, 250.0499, &
      255.0, 260.0, 265.0, 270.0, 271.4, 272.0, &
      273.0, 273.10, 273.1499, &
      273.15, &
      273.16, 273.20, 274.0, 275.0, 280.0, 285.0, 290.0, 293.15, 295.0, &
      300.0, 305.0, 310.0, 315.0, 320.0, 330.0, 340.0, 350.0, 360.0, &
      370.0, 380.0, 390.0, 400.0, 410.0, 415.0, 420.0, 422.0, &
      423.0, 423.05, 423.10, 423.1499, &
      ! --- exactly the last node: r reaches 5001.0 and the i>5000 clamp fires ---
      423.15, &
      ! --- above the table: qsn clamps i to 5000 and r to 5001.0 ---
      423.16, 423.20, 424.0, 430.0, 450.0, 500.0, 1000.0]
  character(len=1024) :: output_path
  character(len=12) :: label
  integer :: n, k, unit
  real :: cq, evs, eis, r61, tn, value, r_raw
  real :: tbq(5001)

  call get_command_argument(1, output_path)
  if (len_trim(output_path) == 0) then
    write(*, '(A)') 'usage: run_qsn OUTPUT.csv'
    error stop 2
  end if

  cq = 173.15 - 0.05
  r61 = 6.1153 * 0.62198
  do k = 1, 5001
    cq = cq + 0.05
    evs = exp(17.67 * (cq - 273.15) / (cq - 29.65))
    eis = exp(22.514 - 6.15e3 / cq)
    if (cq >= 273.15) then
      tbq(k) = r61 * evs
    else
      tbq(k) = r61 * eis
    end if
  end do

  open(newunit=unit, file=trim(output_path), status='replace', action='write')
  write(unit, '(A)') 'case,n,tn,r_raw,qsn'

  do n = 1, nsample
    tn = tn_list(n)
    r_raw = (tn - 173.15) / .05 + 1.
    if (tn < 173.15) then
      label = 'clamp_low'
    else if (tn == 173.15) then
      label = 'node_low'
    else if (tn == 423.15) then
      label = 'node_high'
    else if (tn > 423.15) then
      label = 'clamp_high'
    else
      label = 'interior'
    end if
    value = qsn(tn, tbq)
    write(unit, '(*(g0,:,","))') trim(label), n, tn, r_raw, value
  end do
  close(unit)
end program run_ruc_qsn_oracle
