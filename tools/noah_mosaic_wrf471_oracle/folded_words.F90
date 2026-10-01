program folded_words
use module_model_constants,only: R_D,CP,XLV,RHOWATER,XLF,STBOLT
implicit none
real,parameter :: capa=R_D/CP
write(*,'(A,1X,Z8.8)') 'CAPA',transfer(capa,0)
write(*,'(A,1X,Z8.8)') 'CP',transfer(CP,0)
write(*,'(A,1X,Z8.8)') 'ROWLIW',transfer(RHOWATER*XLF,0)
write(*,'(A,1X,Z8.8)') 'XLV_RHOWATER',transfer(XLV*RHOWATER,0)
write(*,'(A,1X,Z8.8)') 'STBOLT',transfer(STBOLT,0)
write(*,'(A,1X,Z8.8)') 'fold_000',transfer(((273.15)-(1.0e-3)),0)
write(*,'(A,1X,Z8.8)') 'fold_001',transfer(((2.0)*(8.0)),0)
write(*,'(A,1X,Z8.8)') 'fold_002',transfer(((2.0)*(0.11631)),0)
write(*,'(A,1X,Z8.8)') 'fold_003',transfer(((0.55)*(2.0)),0)
write(*,'(A,1X,Z8.8)') 'fold_004',transfer(((2.501000e6)/(1004.5)),0)
write(*,'(A,1X,Z8.8)') 'fold_005',transfer(((0.0001)*(1000.0)),0)
write(*,'(A,1X,Z8.8)') 'fold_006',transfer(((1.0000e3)*(3.3350e5)),0)
write(*,'(A,1X,Z8.8)') 'fold_007',transfer(((-(1.0000e3))*(3.3350e5)),0)
write(*,'(A,1X,Z8.8)') 'fold_008',transfer(((0.1309)*(0.5)),0)
write(*,'(A,1X,Z8.8)') 'fold_009',transfer(((17.67)*(((273.15)-(29.65)))),0)
write(*,'(A,1X,Z8.8)') 'fold_010',transfer(((273.15)-(29.65)),0)
write(*,'(A,1X,Z8.8)') 'fold_011',transfer(((1.0)/(273.15)),0)
write(*,'(A,1X,Z8.8)') 'fold_012',transfer(((273.15)-(5.0)),0)
write(*,'(A,1X,Z8.8)') 'fold_013',transfer(((((2.4888e+3)*(2.83e+6)))/(2.501e+6)),0)
write(*,'(A,1X,Z8.8)') 'fold_014',transfer(((2.4888e+3)*(2.83e+6)),0)
write(*,'(A,1X,Z8.8)') 'fold_015',transfer(((0.980)-(1.0)),0)
write(*,'(A,1X,Z8.8)') 'fold_016',transfer(((2.5e6)*(1000.0)),0)
write(*,'(A,1X,Z8.8)') 'fold_017',transfer(((0.412)/(0.468)),0)
end program
