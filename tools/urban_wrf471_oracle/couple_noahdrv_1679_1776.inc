    if((sf_urban_physics.eq.2).OR.(sf_urban_physics.eq.3))then         !Bep begin
         do j=jts,jte
         do i=its,ite
            UMOM_URB(I,J)=0.
            VMOM_URB(I,J)=0.
            HFX_URB(I,J)=0.
            QFX_URB(I,J)=0.
         do k=kts,kte
            a_u_bep(i,k,j)=a_u_bep(i,k,j)*frc_urb2d(i,j)
            a_v_bep(i,k,j)=a_v_bep(i,k,j)*frc_urb2d(i,j)
            a_t_bep(i,k,j)=a_t_bep(i,k,j)*frc_urb2d(i,j)
            a_q_bep(i,k,j)=0.        
            a_e_bep(i,k,j)=0.
            b_u_bep(i,k,j)=b_u_bep(i,k,j)*frc_urb2d(i,j)
            b_v_bep(i,k,j)=b_v_bep(i,k,j)*frc_urb2d(i,j)
            b_t_bep(i,k,j)=b_t_bep(i,k,j)*frc_urb2d(i,j)
            b_q_bep(i,k,j)=b_q_bep(i,k,j)*frc_urb2d(i,j)
            b_e_bep(i,k,j)=b_e_bep(i,k,j)*frc_urb2d(i,j)
            HFX_URB(I,J)=HFX_URB(I,J)+B_T_BEP(I,K,J)*RHO(I,K,J)*CP*                &
                          DZ8W(I,K,J)*VL_BEP(I,K,J)
            QFX_URB(I,J)=QFX_URB(I,J)+B_Q_BEP(I,K,J)*               &
                          DZ8W(I,K,J)*VL_BEP(I,K,J)
            UMOM_URB(I,J)=UMOM_URB(I,J)+ (A_U_BEP(I,K,J)*U_PHY(I,K,J)+             &
                          B_U_BEP(I,K,J))*DZ8W(I,K,J)*VL_BEP(I,K,J)
            VMOM_URB(I,J)=VMOM_URB(I,J)+ (A_V_BEP(I,K,J)*V_PHY(I,K,J)+             &
                          B_V_BEP(I,K,J))*DZ8W(I,K,J)*VL_BEP(I,K,J)
            vl_bep(i,k,j)=(1.-frc_urb2d(i,j))+vl_bep(i,k,j)*frc_urb2d(i,j)
            sf_bep(i,k,j)=(1.-frc_urb2d(i,j))+sf_bep(i,k,j)*frc_urb2d(i,j)
         end do
            a_u_bep(i,1,j)=(1.-frc_urb2d(i,j))*(-ust(I,J)*ust(I,J))/dz8w(i,1,j)/   &
                           ((u_phy(i,1,j)**2+v_phy(i,1,j)**2.)**.5)+a_u_bep(i,1,j)
            a_v_bep(i,1,j)=(1.-frc_urb2d(i,j))*(-ust(I,J)*ust(I,J))/dz8w(i,1,j)/   &
                           ((u_phy(i,1,j)**2+v_phy(i,1,j)**2.)**.5)+a_v_bep(i,1,j)
            b_t_bep(i,1,j)=(1.-frc_urb2d(i,j))*hfx_rural(i,j)/dz8w(i,1,j)/rho(i,1,j)/CP+ &
                            b_t_bep(i,1,j)
            b_q_bep(i,1,j)=(1.-frc_urb2d(i,j))*qfx_rural(i,j)/dz8w(i,1,j)/rho(i,1,j)+b_q_bep(i,1,j)
            umom=(1.-frc_urb2d(i,j))*ust(i,j)*ust(i,j)*u_phy(i,1,j)/               &
                         ((u_phy(i,1,j)**2+v_phy(i,1,j)**2.)**.5)+umom_urb(i,j)
            vmom=(1.-frc_urb2d(i,j))*ust(i,j)*ust(i,j)*v_phy(i,1,j)/               &
                         ((u_phy(i,1,j)**2+v_phy(i,1,j)**2.)**.5)+vmom_urb(i,j)
            sf_bep(i,1,j)=1.
           
! compute upward longwave radiation from the rural part and total
!           rl_up_rural=-emiss_rural(i,j)*sigma_sb*(tsk_rural(i,j)**4.)-(1.-emiss_rural(i,j))*glw(i,j)
!           rl_up_tot=(1.-frc_urb2d(i,j))*rl_up_rural+frc_urb2d(i,j)*rl_up_urb(i,j)   
!           emiss(i,j)=(1.-frc_urb2d(i,j))*emiss_rural(i,j)+frc_urb2d(i,j)*emiss_urb(i,j)
! using the emissivity and the total longwave upward radiation estimate the averaged skin temperature
           IF (FRC_URB2D(I,J).GT.0.) THEN
              rl_up_rural=-emiss_rural(i,j)*sigma_sb*(tsk_rural(i,j)**4.)-(1.-emiss_rural(i,j))*glw(i,j)
              rl_up_tot=(1.-frc_urb2d(i,j))*rl_up_rural+frc_urb2d(i,j)*rl_up_urb(i,j)   
              emiss(i,j)=(1.-frc_urb2d(i,j))*emiss_rural(i,j)+frc_urb2d(i,j)*emiss_urb(i,j)
              ts_urb2d(i,j)=(max(0.,(-rl_up_urb(i,j)-(1.-emiss_urb(i,j))*glw(i,j))/emiss_urb(i,j)/sigma_sb))**0.25
              tsk(i,j)=(max(0., (-1.*rl_up_tot-(1.-emiss(i,j))*glw(i,j) )/emiss(i,j)/sigma_sb))**.25
           rs_abs_tot=(1.-frc_urb2d(i,j))*swdown(i,j)*(1.-albedo(i,j))+frc_urb2d(i,j)*rs_abs_urb(i,j)  
          if(swdown(i,j).gt.0.)then
           albedo(i,j)=1.-rs_abs_tot/swdown(i,j)
          else
           albedo(i,j)=alb_rural(i,j)
          endif
! rename *_urb to sh_urb2d,lh_urb2d,g_urb2d,rn_urb2d
         grdflx(i,j)= (1.-frc_urb2d(i,j))*grdflx_rural(i,j)+frc_urb2d(i,j)*grdflx_urb(i,j) 
         qfx(i,j)=(1.-frc_urb2d(i,j))*qfx_rural(i,j)+qfx_urb(i,j)
!        lh(i,j)=(1.-frc_urb2d(i,j))*qfx_rural(i,j)*xlv
         lh(i,j)=qfx(i,j)*xlv
         HFX(I,J) = HFX_URB(I,J)+(1-FRC_URB2D(I,J))*HFX_RURAL(I,J)         ![W/m/m]
            SH_URB2D(I,J)    = HFX_URB(I,J)/FRC_URB2D(I,J)
            LH_URB2D(I,J)    = qfx_urb(i,j)*xlv
            G_URB2D(I,J)     = grdflx_urb(i,j)
            RN_URB2D(I,J)    = rs_abs_urb(i,j)+emiss_urb(i,j)*glw(i,j)-rl_up_urb(i,j)
            ust(i,j)=(umom**2.+vmom**2.)**.25
!          if(tsk(i,j).gt.350)write(*,*)'tsk too big!',i,j,tsk(i,j)
!          if(tsk(i,j).lt.260)write(*,*)'tsk too small!',i,j,tsk(i,j),rl_up_tot,rl_up_urb(i,j),rl_up_rural
!                    print*,'ivgtyp,i,j,sigma_sb',ivgtyp(i,j),i,j,sigma_sb
!                    print*,'hfx,lh,qfx,grdflx,ts_urb2d',hfx(i,j),lh(i,j),qfx(i,j),grdflx(i,j),ts_urb2d(i,j)
!                    print*,'tsk,albedo,emiss',tsk(i,j),albedo(i,j),emiss(i,j)
!                if(i.eq.56.and.j.eq.29)then
!                    print*,'ivgtyp, qfx, hfx',ivgtyp(i,j),hfx_rural(i,j),qfx_rural(i,j)
!                    print*,'emiss_rural,emiss_urb',emiss_rural(i,j),emiss_urb(i,j)
!                    print*,'rl_up_rural,rl_up_urb(i,j)',rl_up_rural,rl_up_urb(i,j)
!                    print*,'tsk_rural,ts_urb2d(i,j),tsk',tsk_rural(i,j),ts_urb2d(i,j),tsk(i,j)
!                    print*,'reconstruction fei',((emiss(i,j)*tsk(i,j)**4.-frc_urb2d(i,j)*emiss_urb(i,j)*ts_urb2d(i,j)**4.)/(emiss_rural(i,j)*(1.-frc_urb2d(i,j))))**.25
!                    print*,'ivgtyp,hfx,hfx_urb,hfx_rural',hfx(i,j),hfx_urb(i,j),hfx_rural(i,j)
!                    print*,'lh,lh_rural',lh(i,j),lh_rural(i,j)
!                    print*,'qfx',qfx(i,j)
!                    print*,'ts_urb2d',ts_urb2d(i,j)
!                    print*,'ust',ust(i,j)
!                    print*,'swdown,glw',swdown(i,j),glw(i,j)
!                endif
            else
              SH_URB2D(I,J)    = 0.
              LH_URB2D(I,J)    = 0.
              G_URB2D(I,J)     = 0.
              RN_URB2D(I,J)    = 0.
            endif
        enddo
        enddo

       endif                                                           !Bep end
