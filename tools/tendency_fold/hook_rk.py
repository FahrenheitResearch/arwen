"""Dump the stock RK theta-add arguments in an owned instrumented WRF tree."""
from pathlib import Path
import re


def hook(root):
    path=Path(root)/"dyn_em/module_em.F"
    source=path.read_text()
    start=re.search(r"(?im)^\s*subroutine rk_addtend_dry\s*\(",source).start()
    end=source.lower().index("end subroutine rk_addtend_dry",start)
    body=source[start:end]
    if "rk_add_inputs" in body:
        raise ValueError("RK hooks are already installed")
    block="   CALL locstage('rk_add_inputs')\n"
    for name in ("t_tend","t_tendf","t_save","h_diabatic"):
        block+=f"   CALL locdumpk('{name}',{name},ims,ime,kms,kme,jms,jme,rk_step)\n"
    for name in ("msfty","mut"):
        block+=f"   CALL locdumpk('{name}',{name},ims,ime,1,1,jms,jme,rk_step)\n"
    anchor="   DO j = jts,MIN(jte,jde-1)"
    if anchor not in body:
        raise ValueError("stock RK loop is required")
    body=body.replace(anchor,block+anchor,1)
    body+="   CALL locstage('rk_add_outputs')\n"
    for name in ("t_tend","t_tendf"):
        body+=f"   CALL locdumpk('{name}',{name},ims,ime,kms,kme,jms,jme,rk_step)\n"
    path.write_text(source[:start]+body+source[end:])
    path=Path(root)/"dyn_em/module_first_rk_step_part2.F"
    source=path.read_text()
    anchor="     CALL locstage('c0_phy')"
    if source.count(anchor)!=1:
        raise ValueError("the parent physics trace is required")
    source=source.replace(anchor,"     CALL locstage('c0_reference')\n"
                          "     CALL locdumpk('t_init',grid%t_init,ims,ime,kms,kme,jms,jme,1)\n"+anchor)
    path.write_text(source)


if __name__=="__main__":
    import sys
    hook(sys.argv[1])
