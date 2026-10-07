program run_sum_pm
 use oracle_io
 use pm_oracle
 implicit none
 real :: chem(3,4,2,20), alt(3,4,2), pm25(3,4,2), ec(3,4,2), pm10(3,4,2)
 integer :: c,i,j,k,n
 character(len=1024) :: root
 character(len=40) :: name
 call oracle_root(root)
 do c=0,24
 chem=0.
 do j=1,2
 do k=1,4
 do i=1,3
 alt(i,k,j)=0.7+real(k)*0.2+real(i+j)*0.03
 if(c==22) alt(i,k,j)=0.03+real(k)*0.01
 if(c==23) alt(i,k,j)=10.+real(k)
 if(c==24) alt(i,k,j)=1.
 do n=2,20
 if(c==1 .or. c>=21) chem(i,k,j,n)=real(n*n+i+3*j)*0.013/real(k)
 if(c>=2 .and. c<=20 .and. n==c) chem(i,k,j,n)=real(i+k+j)*0.31
 if(c==21) chem(i,k,j,n)=chem(i,k,j,n)*1.e20
 if(c==22) chem(i,k,j,n)=chem(i,k,j,n)*1.e-30
 enddo
 enddo
 enddo
 enddo
 write(name,'(A,I2.2)') 'case_',c
 call sum_pm_gocart(alt,chem,pm25,ec,pm10,1,4,1,3,1,4,1,3,1,2,1,4,1,3,1,2,1,4)
 call oracle_open(trim(name))
 call oracle_put('chem',chem)
 call oracle_put('alt',alt)
 call oracle_put('PM2_5_DRY',pm25)
 call oracle_put('PM10',pm10)
 call oracle_put('PM2_5_DRY_EC',ec)
 call oracle_close()
 enddo
end program
