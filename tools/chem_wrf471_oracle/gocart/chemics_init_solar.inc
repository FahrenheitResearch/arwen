!
! next is done to scale background oh and no3 in dependence on average zenith angle and day/night for no3
! this is done since background values are only available as average/month. It will not be necessary if other
! chemistry packages are used that provide oh,no3,h2o2
!
         endif
           ndystep=86400/ifix(dt)
           do j=jts,jte
                 do i=its,ite
                   tcosz(i,j)=0.
                   ttday(i,j)=0.
                   rlat=xlat(i,j)*3.1415926535590/180.
                   xlonn=xlong(i,j)
                   do n=1,ndystep
                      xtime=n*dt/60.
                      ixhour=ifix(gmt+.01)+ifix(xtime/60.)
                      xhour=float(ixhour)
                      xmin=60.*gmt+(xtime-xhour*60.)
                      gmtp=mod(xhour,24.)
                      gmtp=gmtp+xmin/60.
                      CALL szangle(1, 1, julday, gmtp, sza, cosszax,xlonn,rlat)
                      TCOSZ(i,j)=TCOSZ(I,J)+cosszax(1,1)
                      if(cosszax(1,1).gt.0.)ttday(i,j)=ttday(i,j)+dt
                    enddo
!                if(i.eq.19.and.j.eq.19)write(0,*)'in cheminit'
!                if(i.eq.19.and.j.eq.19)write(0,*)TCOSZ(i,j),ttday(i,j),julday, gmtp, sza, cosszax,xlonn,rlat
           enddo
