import sys
snip = open(sys.argv[1]).read(); src = open(sys.argv[2]).read()
body = """subroutine n2m_probe()
implicit none
""" + snip + """
end subroutine n2m_probe
subroutine probe_init()
real :: np(20)
integer :: cc
np = 0.0
np(1)=0.5e9; np(2)=0.0; np(3)=1.0; np(4)=4.0e5; np(5)=4.0e4; np(6)=8.0e5; np(7)=3.0e6
np(8)=500.0; np(9)=900.0; np(10)=100.0; np(11)=0.0; np(12)=12.0; cc=0
call nssl_2mom_init(nssl_params=np, ipctmp=5, mixphase=0, nssl_density_on=.true., &
  nssl_hail_on=.true., nssl_ccn_on=.true., nssl_icdx=6, nssl_icdxhl=6, ccn_is_ccna=cc)
end subroutine probe_init
subroutine pr(n, x)
character(*) :: n; real :: x
write(*,"(a24,1x,z8.8,1x,es17.9)") n, transfer(x,0), x
end subroutine pr
subroutine pri(n, x)
character(*) :: n; integer :: x
write(*,"(a24,1x,i12)") n, x
end subroutine pri
subroutine prl(n, x)
character(*) :: n; logical :: x
write(*,"(a24,1x,l2)") n, x
end subroutine prl
subroutine prd(n, x)
character(*) :: n; double precision :: x
write(*,"(a24,1x,z16.16,1x,es25.17)") n, transfer(x,0_8), x
end subroutine prd
"""
i = src.rindex("END MODULE module_mp_nssl_2mom")
src = src[:i] + body + src[i:]
j = src.index("\n CONTAINS\n")
src = src[:j] + "\n PUBLIC :: n2m_probe\n" + src[j:]
open(sys.argv[3], "w").write(src)
