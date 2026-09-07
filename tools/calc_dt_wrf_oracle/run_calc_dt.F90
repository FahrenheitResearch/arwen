! Oracle for WRF's calc_dt (dyn_em/adapt_timestep_em.F, WRF 4.8.0).
!
! calc_dt is lifted VERBATIM except that it USEs module_utility rather
! than module_domain: the subroutine touches nothing from module_domain
! but the WRFU_TimeInterval type and its operators, and module_utility is
! their actual provider.  The arithmetic, the branch structure, the 0.1
! floor and every INT(x*precision + 0.5) quantisation are upstream's.
!
! CSV: max_cfl,target_cfl,mif,precision,last_S,last_Sn,last_Sd,
!      out_S,out_Sn,out_Sd,out_real

      SUBROUTINE calc_dt(dtInterval, max_cfl, max_increase_factor, precision, &
           last_dtInterval, target_cfl)

        USE module_utility

        TYPE(WRFU_TimeInterval) ,INTENT(OUT)      :: dtInterval
        REAL                    ,INTENT(IN)       :: max_cfl
        REAL                    ,INTENT(IN)       :: max_increase_factor
        INTEGER                 ,INTENT(IN)       :: precision
        REAL                    ,INTENT(IN)       :: target_cfl
        TYPE(WRFU_TimeInterval) ,INTENT(IN)       :: last_dtInterval
        REAL                                      :: factor
        INTEGER                                   :: num, den

        if (max_cfl < 0.001) then
           num = INT(max_increase_factor * precision + 0.5)
           den = precision
           dtInterval = last_dtInterval * num / den
        else
           if (max_cfl .gt. target_cfl) then
              factor = ( target_cfl - 0.5 * (max_cfl - target_cfl) ) / max_cfl
              factor = MAX(0.1,factor)
              num = INT(factor * precision + 0.5)
              den = precision
              dtInterval = last_dtInterval * num / den
           else
              factor = target_cfl / max_cfl
              num = INT(factor * precision + 0.5)
              den = precision
              dtInterval = last_dtInterval * num / den
           endif
        endif

      END SUBROUTINE calc_dt

      PROGRAM run_calc_dt
        USE module_utility
        IMPLICIT NONE
        EXTERNAL :: calc_dt
        TYPE(WRFU_TimeInterval) :: last, out
        REAL    :: max_cfl, target_cfl, mif, outreal
        INTEGER :: precision, rc
        INTEGER :: oS, oSn, oSd
        INTEGER :: i, j, k, m
        REAL    :: cfls(24), targs(4), mifs(3)
        INTEGER :: lastS(6), lastSn(6), lastSd(6)

        CALL WRFU_Initialize(defaultCalKind=WRFU_CAL_GREGORIAN, rc=rc)

        cfls = (/ 0.0, 0.0005, 0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5,      &
                  0.6, 0.7, 0.8, 0.84, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5,      &
                  1.8, 2.0, 3.0, 6.0, 50.0 /)
        targs = (/ 1.2, 0.84, 2.0, 0.5 /)
        mifs  = (/ 1.05, 1.51, 2.0 /)
        lastS  = (/ 30,  60,  6, 12,  1,  0 /)
        lastSn = (/  0,   0,  0,  0, 50, 33 /)
        lastSd = (/  1,   1,  1,  1,100,100 /)

        precision = 100
        write(*,'(A)') 'max_cfl,target_cfl,mif,precision,last_S,last_Sn,last_Sd,out_S,out_Sn,out_Sd,out_real'
        do m = 1, 6
          CALL WRFU_TimeIntervalSet(last, S=lastS(m), Sn=lastSn(m), Sd=lastSd(m), rc=rc)
          do i = 1, 24
            max_cfl = cfls(i)
            do j = 1, 4
              target_cfl = targs(j)
              do k = 1, 3
                mif = mifs(k)
                CALL calc_dt(out, max_cfl, mif, precision, last, target_cfl)
                CALL WRFU_TimeIntervalGet(out, S=oS, Sn=oSn, Sd=oSd, rc=rc)
                if (ABS(oSd) < 1) then
                  outreal = REAL(oS)
                else
                  outreal = REAL(oS) + REAL(oSn) / REAL(oSd)
                endif
                write(*,'(3(E16.9,","),I0,",",6(I0,","),E16.9)')            &
                     max_cfl, target_cfl, mif, precision,                   &
                     lastS(m), lastSn(m), lastSd(m), oS, oSn, oSd, outreal
              end do
            end do
          end do
        end do
      END PROGRAM run_calc_dt
