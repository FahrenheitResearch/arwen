subroutine probe(v,o) bind(C)
use iso_c_binding
use module_model_constants
real(c_float),intent(in)::v(7)
real(c_float),intent(out)::o(10)
real::tmpdz,dthrdn,tmp,mlen_s,mlen_v,deltas,kmv,khv
tmpdz=1.0/v(2)+1.0/v(3)
dthrdn=(v(4)-v(5))/tmpdz
tmp=sqrt(max(v(6),1.e-6))
mlen_s=0.76*tmp/(abs(g/v(1)*dthrdn))**0.5
deltas=1.0/v(7)
mlen_v=min(deltas,mlen_s)
kmv=0.15*tmp*mlen_v
khv=kmv*(1.0+2.0*mlen_v/deltas)
o=[tmpdz,dthrdn,tmp,g/v(1),g/v(1)*dthrdn,sqrt(abs(g/v(1)*dthrdn)),mlen_s,mlen_v,kmv,khv]
end subroutine
