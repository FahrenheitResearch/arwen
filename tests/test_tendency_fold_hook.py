import importlib.util
from pathlib import Path

import pytest


def module():
    path = Path(__file__).parents[1] / 'tools/tendency_fold/hook_wrf.py'
    spec = importlib.util.spec_from_file_location('fold_hook', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_hooks_follow_complete_calls_and_preserve_arithmetic(tmp_path):
    (tmp_path / 'phys').mkdir()
    (tmp_path / 'dyn_em').mkdir()
    calls = ''.join(f'   IF (active) &\n      CALL {name}(rt_tendf, &\n         bounds)\n'
                    for name in ('phy_ra_ten', 'phy_bl_ten', 'phy_cu_ten'))
    path = tmp_path / 'phys/module_physics_addtendc.F'
    path.write_text(calls + 'END SUBROUTINE update_phy_ten\n')
    em = tmp_path / 'dyn_em/module_em.F'
    em.write_text('SUBROUTINE calculate_phy_tend (x)\n'
                  '   IF (config_flags%cu_physics .gt. 0) THEN\n'
                  '      RTHCUTEN = mass*RTHCUTEN\n   ENDIF\n'
                  'END SUBROUTINE calculate_phy_tend\n')
    module().hook(tmp_path)
    cleaned = '\n'.join(s for s in path.read_text().splitlines()
                        if 'CALL loc' not in s) + '\n'
    assert cleaned == calls + 'END SUBROUTINE update_phy_ten\n'
    assert "bounds)\n      CALL locstage('fold_after_radiation')" in path.read_text()
    assert "'fold_raw_cumulus'" in em.read_text()
    with pytest.raises(ValueError, match='already installed'):
        module().hook(tmp_path)


def test_rk_hook_preserves_statements_and_captures_the_reference(tmp_path):
    root=tmp_path/'dyn_em';root.mkdir()
    text=('SUBROUTINE rk_addtend_dry (t)\n   IMPLICIT NONE\n'
          '   DO j = jts,MIN(jte,jde-1)\n'
          '      t_tend=t_tend+t_tendf/msfty\n   ENDDO\n'
          'END SUBROUTINE rk_addtend_dry\n')
    path=root/'module_em.F';path.write_text(text)
    part=root/'module_first_rk_step_part2.F'
    part.write_text("     CALL locstage('c0_phy')\n     t_tendf=t_tendf+physics\n")
    source=Path(__file__).parents[1]/'tools/tendency_fold/hook_rk.py'
    spec=importlib.util.spec_from_file_location('rk_hook',source)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    mod.hook(tmp_path)
    cleaned='\n'.join(line for line in path.read_text().splitlines()
                      if 'CALL loc' not in line)+'\n'
    assert cleaned==text
    assert "locdumpk('h_diabatic'" in path.read_text()
    assert "locdumpk('t_init',grid%t_init" in part.read_text()
    with pytest.raises(ValueError,match='already installed'):
        mod.hook(tmp_path)
