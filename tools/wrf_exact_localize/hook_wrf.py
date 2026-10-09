"""Insert step-1 debug dumps into a scratch copy of WRF 4.6.1 dyn_em/solve_em.F (writes only; no arithmetic touched)."""
import sys
path = sys.argv[1]
src = open(path).read()
F3 = lambda n, a: f"     CALL locdump3('{n}', {a}, ims,ime,kms,kme,jms,jme, grid%itimestep, rk_step, LOCIT)\n"
F2 = lambda n, a: f"     CALL locdump3('{n}', {a}, ims,ime,1,1,jms,jme, grid%itimestep, rk_step, LOCIT)\n"
STATE = [("u_2", "grid%u_2"), ("v_2", "grid%v_2"), ("w_2", "grid%w_2"), ("t_2", "grid%t_2"), ("ph_2", "grid%ph_2")]
def block(tag, three, two=(), it="0"):
    s = f"     LOCIT = {it}\n     CALL locstage('{tag}')\n"
    s += "".join(F3(n, a) for n, a in three) + "".join(F2(n, a) for n, a in two)
    return s
hooks = [
 ("after", "BENCH_END(step_prep_tim)\n", "     CALL locdims(ids,ide,jds,jde,kds,kde,ims,ime,jms,jme,kms,kme, grid%itimestep)\n" +
   block("b_prep", STATE + [("ru","grid%ru"),("rv","grid%rv"),("ww","grid%ww"),("php","grid%php"),("alt","grid%alt"),("al","grid%al"),("p","grid%p")],
         [("mu_2","grid%mu_2"),("mut","grid%mut"),("muu","grid%muu"),("muv","grid%muv")])),
 ("after", "     END IF rk_step_is_one\n", block("c_part2", [("ru_tendf","ru_tendf"),("rv_tendf","rv_tendf"),("rw_tendf","rw_tendf"),("ph_tendf","ph_tendf"),("t_tendf","t_tendf")], [("mu_tendf","mu_tendf")])),
 ("after", "BENCH_END(rk_tend_tim)\n", block("d_rktend", [("ru_tend","grid%ru_tend"),("rv_tend","grid%rv_tend"),("rw_tend","rw_tend"),("ph_tend","ph_tend"),("t_tend","t_tend")], [("mu_tend","mu_tend")])),
 ("after", "BENCH_END(relax_bdy_dry_tim)\n", block("e_addtend", [("ru_tend","grid%ru_tend"),("rv_tend","grid%rv_tend"),("rw_tend","rw_tend"),("ph_tend","ph_tend"),("t_tend","t_tend")], [("mu_tend","mu_tend")])),
 ("before", "     small_steps : DO iteration = 1 , number_of_small_timesteps\n", block("f_ssprep", STATE + [("ww","grid%ww"),("p","grid%p"),("al","grid%al"),("t_1","grid%t_1"),("u_1","grid%u_1"),("t_save","grid%t_save")], [("mu_2","grid%mu_2"),("muts","grid%muts"),("mu_1","grid%mu_1")])),
 ("before", "     END DO small_steps\n", block("g_ss", STATE + [("ww","grid%ww"),("p","grid%p"),("al","grid%al")], [("mu_2","grid%mu_2"),("muts","grid%muts")], it="iteration")),
 ("after", "BENCH_END(small_step_finish_tim)\n", block("h_finish", STATE + [("ww","grid%ww")], [("mu_2","grid%mu_2")])),
 ("after", "BENCH_END(calc_p_rho_tim)\n", block("i_pphi", STATE + [("p","grid%p"),("al","grid%al"),("alt","grid%alt"),("php","grid%php")], [("mu_2","grid%mu_2")])),
 ("before", "   END DO Runge_Kutta_loop\n", block("j_rkend", STATE + [("p","grid%p"),("al","grid%al")], [("mu_2","grid%mu_2")])),
]
for where, anchor, text in hooks:
    if src.count(anchor) != 1:
        raise SystemExit(f"anchor {anchor!r} found {src.count(anchor)} times")
    src = src.replace(anchor, anchor + text if where == "after" else text + anchor)
# declare LOCIT after IMPLICIT NONE (first one)
src = src.replace("   IMPLICIT NONE\n", "   IMPLICIT NONE\n   INTEGER :: LOCIT\n", 1)
# final-state dump: before the last END SUBROUTINE solve_em
end = "END SUBROUTINE solve_em"
i = src.rindex(end)
fin = ("     LOCIT = 0\n     CALL locstage('z_end')\n" + "".join(F3(n, a).replace("rk_step", "9") for n, a in STATE + [("p","grid%p"),("al","grid%al")])
       + F2("mu_2", "grid%mu_2").replace("rk_step", "9"))
src = src[:i] + fin + src[i:]
src += r'''

SUBROUTINE locstage(tag)
  CHARACTER*(*) tag
  CHARACTER(len=64) :: cur
  COMMON /locdumpc/ cur
  cur = tag
END SUBROUTINE locstage

SUBROUTINE locdims(ids,ide,jds,jde,kds,kde,ims,ime,jms,jme,kms,kme,itimestep)
  INTEGER ids,ide,jds,jde,kds,kde,ims,ime,jms,jme,kms,kme,itimestep, l, u
  CHARACTER(len=512) :: dir
  CALL get_environment_variable('WRF_LOCDUMP', dir, l)
  IF (l == 0 .OR. itimestep /= 1) RETURN
  OPEN(newunit=u, file=dir(1:l)//'/dims.txt', status='replace', form='formatted')
  WRITE(u,*) ids,ide,jds,jde,kds,kde,ims,ime,jms,jme,kms,kme
  CLOSE(u)
END SUBROUTINE locdims

SUBROUTINE locdump3(name, a, ims,ime,kms,kme,jms,jme, itimestep, rk_step, it)
  CHARACTER*(*) name
  INTEGER ims,ime,kms,kme,jms,jme, itimestep, rk_step, it, l, u, nmax
  REAL a(ims:ime, kms:kme, jms:jme)
  CHARACTER(len=512) :: dir, fname, sval
  CHARACTER(len=64) :: cur
  COMMON /locdumpc/ cur
  CALL get_environment_variable('WRF_LOCDUMP', dir, l)
  IF (l == 0) RETURN
  nmax = 1
  CALL get_environment_variable('WRF_LOCDUMP_STEPS', sval)
  IF (LEN_TRIM(sval) > 0) READ(sval,*) nmax
  IF (itimestep > nmax) RETURN
  WRITE(fname,'(A,"/s",I3.3,"_rk",I1,"_it",I2.2,"_",A,"__",A,".bin")') dir(1:l), itimestep, rk_step, it, TRIM(cur), TRIM(name)
  OPEN(newunit=u, file=TRIM(fname), access='stream', form='unformatted', status='replace', convert='little_endian')
  WRITE(u) ims,ime,kms,kme,jms,jme
  WRITE(u) a
  CLOSE(u)
END SUBROUTINE locdump3
'''
open(path, "w").write(src)
print("hooked")
