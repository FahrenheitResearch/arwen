"""Second hook pass: per-term dumps inside rk_tendency (module_em.F) at step 1; step number via COMMON set in solve_em."""
import sys, re
d = sys.argv[1]
se = open(f"{d}/solve_em.F").read()
a = "   grid%itimestep = grid%itimestep + 1\n"
assert se.count(a) == 1
se = se.replace(a, a + "   CALL locsetstep(grid%itimestep)\n")
se += r'''
SUBROUTINE locsetstep(n)
  INTEGER n, istep
  COMMON /locstepc/ istep
  istep = n
END SUBROUTINE locsetstep

SUBROUTINE locdumpk(name, a, ims,ime,kms,kme,jms,jme, rk_step)
  CHARACTER*(*) name
  INTEGER ims,ime,kms,kme,jms,jme, rk_step, istep
  REAL a(*)
  COMMON /locstepc/ istep
  CALL locdump3(name, a, ims,ime,kms,kme,jms,jme, istep, rk_step, 0)
END SUBROUTINE locdumpk
'''
open(f"{d}/solve_em.F", "w").write(se)
lines = open(f"{d}/module_em.F").read().split("\n")
start = next(i for i, l in enumerate(lines) if re.match(r"\s*SUBROUTINE rk_tendency\b", l))
end = next(i for i, l in enumerate(lines) if re.match(r"\s*END SUBROUTINE rk_tendency\b", l))
T3 = ["ru_tend", "rv_tend", "rw_tend", "t_tend", "ph_tend"]
targets = [("CALL advect_u (", "k1_advu"), ("CALL advect_v (", "k2_advv"), ("CALL advect_w (", "k3_advw"),
           ("CALL advect_scalar (", "k4_advt"), ("CALL rhs_ph(", "k5_rhsph"), ("CALL horizontal_pressure_gradient(", "k6_hpg"),
           ("CALL pg_buoy_w(", "k7_buoy"), ("CALL w_damp ", "k8_wdamp"), ("CALL coriolis (", "k9_cor"), ("CALL curvature (", "kA_curv")]
inserts = []
for key, tag in targets:
    idx = [i for i in range(start, end) if key in lines[i]]
    assert len(idx) == 1, (key, idx)
    i = idx[0]
    while lines[i].split("!")[0].rstrip().endswith("&"):
        i += 1
    block = [f"     CALL locstage('{tag}')"] + [f"     CALL locdumpk('{n}', {n}, ims,ime,kms,kme,jms,jme, rk_step)" for n in T3]
    inserts.append((i + 1, block))
for pos, block in sorted(inserts, reverse=True):
    lines[pos:pos] = block
open(f"{d}/module_em.F", "w").write("\n".join(lines))
print("hooked", len(inserts))
