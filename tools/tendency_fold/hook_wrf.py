"""Write-only fold and raw cumulus dumps in an already instrumented WRF tree."""
from pathlib import Path
import re


def dumps(tag, fields):
    return f"      CALL locstage('{tag}')\n" + "".join(
        f"      CALL locdumpk('{n}', {n}, ims,ime,kms,kme,jms,jme, 1)\n"
        for n in fields)


def hook(root):
    root = Path(root)
    path = root / 'phys/module_physics_addtendc.F'
    source = path.read_text()
    if 'fold_after_radiation' in source:
        raise ValueError('fold hooks already installed')
    stop = source.lower().index('end subroutine update_phy_ten')
    head, tail = source[:stop], source[stop:]
    for call, tag in [('phy_ra_ten', 'fold_after_radiation'),
                      ('phy_bl_ten', 'fold_after_pbl'),
                      ('phy_cu_ten', 'fold_after_cumulus')]:
        lines = head.splitlines(keepends=True)
        starts = [i for i, line in enumerate(lines)
                  if re.search(r'\bCALL\s+' + call + r'\s*\(', line, re.I)]
        if len(starts) != 1:
            raise ValueError(f'{call}: expected one call, found {len(starts)}')
        i = starts[0]
        while lines[i].split('!')[0].rstrip().endswith('&'):
            i += 1
        lines.insert(i + 1, dumps(tag, ('rt_tendf', 'ru_tendf', 'rv_tendf')))
        head = ''.join(lines)
    path.write_text(head + tail)
    path = root / 'dyn_em/module_em.F'
    source = path.read_text()
    start = source.lower().index('subroutine calculate_phy_tend (')
    stop = source.lower().index('end subroutine calculate_phy_tend', start)
    body = source[start:stop]
    anchor = '   IF (config_flags%cu_physics .gt. 0) THEN\n'
    if body.count(anchor) != 1:
        raise ValueError('expected one cumulus mass coupling block')
    body = body.replace(anchor, anchor + dumps('fold_raw_cumulus',
                        ('RTHCUTEN', 'RUCUTEN', 'RVCUTEN')))
    path.write_text(source[:start] + body + source[stop:])


if __name__ == '__main__':
    import sys
    hook(sys.argv[1])
