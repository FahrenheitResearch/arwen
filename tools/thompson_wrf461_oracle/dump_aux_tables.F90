program dump_thompson_aux_tables
  use module_mp_thompson, only: thompson_init, tps_iaus, tni_iaus,     &
       tpi_ide, t_Efrw, t_Efsw, tpc_wev, tnc_wev
  implicit none
  real :: hgt(2,2,2)

  hgt = 0.0
  hgt(:,1,:) = 100.0
  hgt(:,2,:) = 200.0
  call thompson_init(                                                &
       hgt=hgt,                                                      &
       ids=1, ide=3, jds=1, jde=3, kds=1, kde=2,                    &
       ims=1, ime=2, jms=1, jme=2, kms=1, kme=2,                    &
       its=1, ite=2, jts=1, jte=2, kts=1, kte=2)

  open(71, file='thompson_aux_tables.dat', form='unformatted',         &
       status='replace', action='write')
  write(71) tps_iaus
  write(71) tni_iaus
  write(71) tpi_ide
  write(71) t_Efrw
  write(71) t_Efsw
  write(71) tpc_wev
  write(71) tnc_wev
  close(71)
  print '(A)', 'THOMPSON_AUX_TABLE_DUMP_COMPLETE'
end program dump_thompson_aux_tables
