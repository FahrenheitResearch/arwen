"""Fourth WRF hook pass (writes only, no arithmetic), after hook_wrf.py, hook_wrf2.py
and hook_wrf7.py: water vapour at the stage points, the moist scalar path
(rk_scalar_tend's advect_tend and moist_tend, the acoustic mean fluxes, moist_tend
after relax_bdy_scalar/spec_bdy_scalar), and the end-of-solve dump moved before
solve_em's final RETURN (it sat after it and never ran).

usage: python3 hook_wrf8.py <WRF-4.6.1/dyn_em>

Checked on W1 2026-10-07: the build with all four passes reproduces the stock scalar
recording of iowa-convective A088 in 208 of 208 fields, and its p0/p1 history hashes
equal the builds before this pass.
"""
import sys

p = sys.argv[1] + "/solve_em.F"
s = open(p).read()
D = lambda n, a, rk="rk_step": (f"     CALL locdump3('{n}', {a}, ims,ime,kms,kme,jms,jme, "
                                f"grid%itimestep, {rk}, LOCIT)\n")
for tag in ("b_prep", "i_pphi", "j_rkend", "h_finish"):
    anchor = f"     CALL locstage('{tag}')\n"
    assert s.count(anchor) == 1, tag
    s = s.replace(anchor, anchor + D("qv", "moist(ims,kms,jms,P_QV)"))
anchor = "BENCH_END(rk_scalar_tend_tim)\n"
assert s.count(anchor) == 1
s = s.replace(anchor, anchor + "     IF (im == P_QV) THEN\n     LOCIT = 0\n     CALL locstage('q1_sctend')\n"
              + D("advect_tend", "advect_tend") + D("moist_tend", "moist_tend(ims,kms,jms,im)")
              + D("ru_m", "grid%ru_m") + D("rv_m", "grid%rv_m") + D("ww_m", "grid%ww_m")
              + "     ENDIF\n")
anchor = "BENCH_END(rlx_bdy_scalar_tim)\n"
assert s.count(anchor) == 1
s = s.replace(anchor, anchor + "     IF (im == P_QV) THEN\n     CALL locstage('q2_relax')\n"
              + D("moist_tend", "moist_tend(ims,kms,jms,im)") + "     ENDIF\n")
start = s.index("     LOCIT = 0\n     CALL locstage('z_end')\n")
end = s.index("\nEND SUBROUTINE solve_em", start)
block = s[start:end + 1]
s = s[:start] + s[end + 1:]
ret = s.rindex("   RETURN\n", 0, s.index("END SUBROUTINE solve_em"))
s = s[:ret] + block + D("qv", "moist(ims,kms,jms,P_QV)", "9") + s[ret:]
open(p, "w").write(s)
print("hooked qv, moist path, end of solve")
