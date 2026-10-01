    DO J=JTS,JTE
    DO I=ITS,ITE
      HFX_RURAL(I,J)                = HFX(I,J)
      QFX_RURAL(I,J)                = QFX(I,J)
      GRDFLX_RURAL(I,J)             = GRDFLX(I,J)
      EMISS_RURAL(I,J)              = EMISS(I,J)
      TSK_RURAL(I,J)                = TSK(I,J)
      ALB_RURAL(I,J)                = ALBEDO(I,J)
    END DO
    END DO
IF((SF_URBAN_PHYSICS == 2).OR.(SF_URBAN_PHYSICS == 3))THEN 

  sigma_sb=5.67e-08
  do j = jts, jte
  do i = its, ite
    UMOM_URB(I,J)     = 0.
    VMOM_URB(I,J)     = 0.
    HFX_URB(I,J)      = 0.
    QFX_URB(I,J)      = 0.

    do k=kts,kte
      a_u_bep(i,k,j) = a_u_bep(i,k,j)*frc_urb2d(i,j)
      a_v_bep(i,k,j) = a_v_bep(i,k,j)*frc_urb2d(i,j)
      a_t_bep(i,k,j) = a_t_bep(i,k,j)*frc_urb2d(i,j)
      a_q_bep(i,k,j) = 0.
      a_e_bep(i,k,j) = 0.
      b_u_bep(i,k,j) = b_u_bep(i,k,j)*frc_urb2d(i,j)
      b_v_bep(i,k,j) = b_v_bep(i,k,j)*frc_urb2d(i,j)
      b_t_bep(i,k,j) = b_t_bep(i,k,j)*frc_urb2d(i,j)
      b_q_bep(i,k,j) = b_q_bep(i,k,j)*frc_urb2d(i,j)
      b_e_bep(i,k,j) = b_e_bep(i,k,j)*frc_urb2d(i,j)
      HFX_URB(I,J)   = HFX_URB(I,J) + B_T_BEP(I,K,J)*RHO(I,K,J)*CP*DZ8W(I,K,J)*VL_BEP(I,K,J)
      QFX_URB(I,J)   = QFX_URB(I,J) + B_Q_BEP(I,K,J)*DZ8W(I,K,J)*VL_BEP(I,K,J)
      UMOM_URB(I,J)  = UMOM_URB(I,J)+ (A_U_BEP(I,K,J)*U_PHY(I,K,J)+B_U_BEP(I,K,J))*DZ8W(I,K,J)*VL_BEP(I,K,J)
      VMOM_URB(I,J)  = VMOM_URB(I,J)+ (A_V_BEP(I,K,J)*V_PHY(I,K,J)+B_V_BEP(I,K,J))*DZ8W(I,K,J)*VL_BEP(I,K,J)
      vl_bep(i,k,j)  = (1.-frc_urb2d(i,j)) + vl_bep(i,k,j)*frc_urb2d(i,j)
      sf_bep(i,k,j)  = (1.-frc_urb2d(i,j)) + sf_bep(i,k,j)*frc_urb2d(i,j)
    end do

    a_u_bep(i,1,j)   = (1.-frc_urb2d(i,j))*(-ust(I,J)*ust(I,J))/dz8w(i,1,j)/   &
                          ((u_phy(i,1,j)**2+v_phy(i,1,j)**2.)**.5)+a_u_bep(i,1,j)

    a_v_bep(i,1,j)   = (1.-frc_urb2d(i,j))*(-ust(I,J)*ust(I,J))/dz8w(i,1,j)/   &
                          ((u_phy(i,1,j)**2+v_phy(i,1,j)**2.)**.5)+a_v_bep(i,1,j)

    b_t_bep(i,1,j)   = (1.-frc_urb2d(i,j))*hfx_rural(i,j)/dz8w(i,1,j)/rho(i,1,j)/CP+ & 
                           b_t_bep(i,1,j)

    b_q_bep(i,1,j)   = (1.-frc_urb2d(i,j))*qfx_rural(i,j)/dz8w(i,1,j)/rho(i,1,j)+b_q_bep(i,1,j)

    umom             = (1.-frc_urb2d(i,j))*ust(i,j)*ust(i,j)*u_phy(i,1,j)/               &
                         ((u_phy(i,1,j)**2+v_phy(i,1,j)**2.)**.5)+umom_urb(i,j)

    vmom             = (1.-frc_urb2d(i,j))*ust(i,j)*ust(i,j)*v_phy(i,1,j)/               &
                         ((u_phy(i,1,j)**2+v_phy(i,1,j)**2.)**.5)+vmom_urb(i,j)
    sf_bep(i,1,j)    = 1.

! using the emissivity and the total longwave upward radiation estimate the averaged skin temperature

  IF (FRC_URB2D(I,J).GT.0.) THEN
    rl_up_rural   = -emiss_rural(i,j)*sigma_sb*(tsk_rural(i,j)**4.)-(1.-emiss_rural(i,j))*glw(i,j)
    rl_up_tot     = (1.-frc_urb2d(i,j))*rl_up_rural     + frc_urb2d(i,j)*rl_up_urb(i,j)
    emiss(i,j)    = (1.-frc_urb2d(i,j))*emiss_rural(i,j)+ frc_urb2d(i,j)*emiss_urb(i,j)
    ts_urb2d(i,j) = (max(0.,(-rl_up_urb(i,j)-(1.-emiss_urb(i,j))*glw(i,j))/emiss_urb(i,j)/sigma_sb))**0.25
    tsk(i,j)      = (max(0., (-1.*rl_up_tot-(1.-emiss(i,j))*glw(i,j) )/emiss(i,j)/sigma_sb))**.25
    rs_abs_tot    = (1.-frc_urb2d(i,j))*swdown(i,j)*(1.-albedo(i,j))+frc_urb2d(i,j)*rs_abs_urb(i,j)

    if(swdown(i,j) > 0.)then
      albedo(i,j) = 1.-rs_abs_tot/swdown(i,j)
    else
      albedo(i,j) = alb_rural(i,j)
    endif

! rename *_urb to sh_urb2d,lh_urb2d,g_urb2d,rn_urb2d

    grdflx(i,j)   = (1.-frc_urb2d(i,j))*grdflx_rural(i,j)+ frc_urb2d(i,j)*grdflx_urb(i,j)*(-1.0) ! positive: downward direction
    qfx(i,j)      = (1.-frc_urb2d(i,j))*qfx_rural(i,j)   + qfx_urb(i,j)
    lh(i,j)       = qfx(i,j)*xlv
    hfx(i,j)      = hfx_urb(i,j)                         + (1-frc_urb2d(i,j))*hfx_rural(i,j)      ![W/m/m]
    sh_urb2d(i,j) = hfx_urb(i,j)/frc_urb2d(i,j)
    lh_urb2d(i,j) = qfx_urb(i,j)*xlv/frc_urb2d(i,j)
    g_urb2d(i,j)  = grdflx_urb(i,j)
    rn_urb2d(i,j) = rs_abs_urb(i,j)+emiss_urb(i,j)*glw(i,j)-rl_up_urb(i,j)
    ust(i,j)      = (umom**2.+vmom**2.)**.25

  ELSE

    sh_urb2d(i,j)    = 0.
    lh_urb2d(i,j)    = 0.
    g_urb2d(i,j)     = 0.
    rn_urb2d(i,j)    = 0.

  ENDIF

  enddo ! jloop
  enddo ! iloop

ENDIF ! SF_URBAN_PHYSICS == 2 or 3
