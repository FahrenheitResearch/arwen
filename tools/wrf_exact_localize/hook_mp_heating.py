"""Write retained microphysics heating at the already installed step end."""
from pathlib import Path


def hook(directory):
    path = Path(directory) / "dyn_em/solve_em.F"
    source = path.read_text()
    anchor = "     LOCIT = 0\n     CALL locstage('z_end')\n"
    if source.count(anchor) != 1 or "CALL locdump3('h_diabatic', grid%h_diabatic" in source:
        raise ValueError("one fresh installed end-of-step hook is required")
    block = ("     CALL locdump3('h_diabatic', grid%h_diabatic, "
             "ims,ime,kms,kme,jms,jme, grid%itimestep, 9, 0)\n")
    path.write_text(source.replace(anchor, anchor + block))


if __name__ == "__main__":
    import sys
    hook(sys.argv[1])
