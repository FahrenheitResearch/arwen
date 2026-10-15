end module
program categories
 use native_counts
 implicit none
 integer::ci,pi,iCell,lu_index(1),initial(4),cases(4,8),maximum
 integer(kind=I8KIND)::pixel(1)
 maximum=huge(1)
 cases(:,1)=[0,0,0,0]
 cases(:,2)=[1,1,1,1]
 cases(:,3)=[-2,-1,-1,-3]
 cases(:,4)=[maximum,2,3,4]
 cases(:,5)=[maximum-1,maximum,maximum,0]
 cases(:,6)=[-maximum-1,-maximum-1,-maximum-1,-maximum-1]
 cases(:,7)=[-maximum-1,0,-1,-2]
 cases(:,8)=[2,3,4,maximum]
 allocate(ncat(4,1))
 do ci=1,size(cases,2)
 do pi=0,5
  initial=cases(:,ci);ncat(:,1)=initial;pixel=pi
  call categorical_interp_accumulation(1,pixel)
  iCell=1
! CATEGORY_MAXLOC
  write(*,'(10(z8.8,1x))')initial,pi,ncat(:,1),lu_index(1)
 enddo
 enddo
 deallocate(ncat)
 category_min=5;category_max=3
 allocate(ncat(category_min:category_max,1))
 iCell=1
! CATEGORY_MAXLOC
 write(*,'(a,3(i0,1x))')'EMPTY ',category_min,category_max,lu_index(1)
end program
