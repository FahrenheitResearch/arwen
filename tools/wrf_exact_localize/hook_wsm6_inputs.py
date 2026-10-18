"""Write stock WSM6 arguments immediately before and after its real call."""
from pathlib import Path


def hook(directory):
    path = Path(directory) / "phys/module_microphysics_driver.F"
    source = path.read_text()
    lines = source.splitlines(keepends=True)
    calls = [i for i, line in enumerate(lines) if line.lstrip().startswith("CALL wsm6(")]
    if len(calls) != 1 or "CALL locstage('mp_wsm6_in')" in source:
        raise ValueError("one fresh stock WSM6 call is required")
    first = last = calls[0]
    # Preprocessor directives and blank/comment lines can interrupt the
    # physical lines of this continued call. Its closing line ends in ')'.
    while (lines[last].lstrip().startswith("#") or
           not lines[last].split("!")[0].rstrip().endswith(")")):
        last += 1
        if last == len(lines):
            raise ValueError("stock WSM6 call has no closing line")
    names = ("th", "rho", "pi_phy", "p", "dz8w", "qv_curr", "qc_curr",
             "qr_curr", "qi_curr", "qs_curr", "qg_curr")
    def block(tag):
        return [f"             CALL locstage('{tag}')\n", *(
            f"             CALL locdumpk('{n}', {n}, ims,ime,kms,kme,jms,jme, 1)\n"
            for n in names)]
    lines[last + 1:last + 1] = block("mp_wsm6_out")
    lines[first:first] = block("mp_wsm6_in")
    path.write_text("".join(lines))


if __name__ == "__main__":
    import sys
    hook(sys.argv[1])
