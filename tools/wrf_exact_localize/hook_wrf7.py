"""Third hook pass (writes only, no arithmetic): sub-steps of first_rk_step_part2 and the
damp_opt=3 rows of advance_w.  Uses locdumpk/locstage from the earlier passes."""
import re
import sys

d = sys.argv[1]


def after_call(lines, start, end, key):
    idx = [i for i in range(start, end) if key in lines[i] and not lines[i].lstrip().startswith("!")]
    assert len(idx) == 1, (key, idx)
    i = idx[0]
    while lines[i].split("!")[0].rstrip().endswith("&"):
        i += 1
    return i + 1


# --- first_rk_step_part2: physics, vertical diffusion, horizontal diffusion -----------------
p = f"{d}/module_first_rk_step_part2.F"
lines = open(p).read().split("\n")
T = ["ru_tendf", "rv_tendf", "rw_tendf", "t_tendf", "ph_tendf"]


def blk(tag, extra=(), extra2=()):
    out = [f"     CALL locstage('{tag}')"]
    out += [f"     CALL locdumpk('{n}', {n}, ims,ime,kms,kme,jms,jme, 1)" for n in T]
    out += [f"     CALL locdumpk('{n}', {a}, ims,ime,kms,kme,jms,jme, 1)" for n, a in extra]
    out += [f"     CALL locdumpk('{n}', {a}, ims,ime,1,1,jms,jme, 1)" for n, a in extra2]
    return out


ins = []
n = len(lines)
ins.append((after_call(lines, 0, n, "CALL update_phy_ten("), blk("c0_phy")))
ins.append((after_call(lines, 0, n, "CALL vertical_diffusion_2("),
            blk("c1_vdiff", [("defor13", "grid%defor13"), ("defor23", "grid%defor23"),
                             ("xkmv", "grid%xkmv"), ("xkhv", "grid%xkhv")],
                [("ust", "grid%ust")])))
ins.append((after_call(lines, 0, n, "CALL horizontal_diffusion_2("), blk("c2_hdiff")))
for pos, b in sorted(ins, reverse=True):
    lines[pos:pos] = b
open(p, "w").write("\n".join(lines))

# --- advance_w: w before and after the damp_opt=3 rows, and the tridiagonal coefficients ----
p = f"{d}/module_small_step_em.F"
lines = open(p).read().split("\n")
s = next(i for i, l in enumerate(lines) if re.match(r"\s*SUBROUTINE advance_w\b", l))
e = next(i for i, l in enumerate(lines) if re.match(r"\s*END SUBROUTINE advance_w\b", l))
body = lines[s:e]
# declare the copy right after IMPLICIT NONE of advance_w
k_imp = next(i for i in range(s, e) if "REAL :: muthk, muthkm1" in lines[i])
k_damp = next(i for i in range(s, e) if "IF (config_flags%damp_opt .eq. 3) THEN" in lines[i])
k_jend = next(i for i in range(s, e) if "ENDDO j_loop_w" in lines[i])
k_first = next(i for i in range(k_imp + 1, e) if "j_loop_w:" in lines[i])
# first executable: locate the line declaring dampwt etc. is irrelevant; declarations end before
# the first statement that is not a declaration.  Insert the declaration right after IMPLICIT NONE.
new = list(lines)
new.insert(k_jend + 1,
           "     CALL locstage('w9_adv')\n"
           "     CALL locdumpk('w_predamp', locwpre, ims,ime,kms,kme,jms,jme, 1)\n"
           "     CALL locdumpk('w_postdamp', w, ims,ime,kms,kme,jms,jme, 1)\n"
           "     CALL locdumpk('alpha', alpha, ims,ime,kms,kme,jms,jme, 1)\n"
           "     CALL locdumpk('gamma', gamma, ims,ime,kms,kme,jms,jme, 1)\n"
           "     CALL locdumpk('a', a, ims,ime,kms,kme,jms,jme, 1)\n"
           "     CALL locdumpk('rhs_ph', ph, ims,ime,kms,kme,jms,jme, 1)")
new.insert(k_damp, "    locwpre(:,:,j) = w(:,:,j)")
new.insert(k_imp + 1, "   REAL :: locwpre(ims:ime,kms:kme,jms:jme)")
open(p, "w").write("\n".join(new))
print("hooked part2 3 points, advance_w damp rows")
