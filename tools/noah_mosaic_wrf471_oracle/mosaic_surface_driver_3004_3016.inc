           IF(IVGTYP(I,J) == ISURBAN   .or. IVGTYP(I,J) == LCZ_1 .or. IVGTYP(I,J) == LCZ_2 .or. &
             IVGTYP(I,J) == LCZ_3      .or. IVGTYP(I,J) == LCZ_4 .or. IVGTYP(I,J) == LCZ_5 .or. &
             IVGTYP(I,J) == LCZ_6      .or. IVGTYP(I,J) == LCZ_7 .or. IVGTYP(I,J) == LCZ_8 .or. &
             IVGTYP(I,J) == LCZ_9      .or. IVGTYP(I,J) == LCZ_10 .or. IVGTYP(I,J) == LCZ_11 )THEN
             U10(I,J)  = U10_URB2D(I,J)                       !urban
             V10(I,J)  = V10_URB2D(I,J)                       !urban
             PSIM(I,J) = PSIM_URB2D(I,J)                      !urban
             PSIH(I,J) = PSIH_URB2D(I,J)                      !urban
             GZ1OZ0(I,J) = GZ1OZ0_URB2D(I,J)                  !urban
!m             AKHS(I,J) = AKHS_URB2D(I,J)                    !urban
             AKHS(I,J) = CHS(I,J)                             !urban
             AKMS(I,J) = AKMS_URB2D(I,J)                      !urban
           END IF                                             !urban
