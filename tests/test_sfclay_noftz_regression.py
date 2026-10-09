"""Default arithmetic must preserve the surface and QSQ subnormal words.

Run tools/sfclay_noftz_check/prepare.sh first. These are fresh Fortran
column references, not host mirrors or tolerances. Every unit has its own
assertion, including the CK-corrected classic referee without exclusions.
"""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
from conftest import requires_gpu

pytestmark = pytest.mark.gpu
ROOT = Path(__file__).resolve().parents[1]
CHECK = ROOT / 'tools' / 'sfclay_noftz_check'
LANE = Path(os.environ.get('SFCLAY_CHECK_ROOT', '/work/pool-sfclay-noftz-mynn-mm5-eta'))

@requires_gpu
@pytest.mark.parametrize('unit', ['mynn_surface', 'mynn_pbl', 'classic', 'revised', 'eta'])
def test_default_subnormal_words(unit):
    if not (LANE / 'subnormal/oracle.bin').exists():
        pytest.skip('fresh Fortran stream is absent; portable unit fixtures grade the regression')
    result = subprocess.run([sys.executable, str(CHECK / 'check.py'), unit],
                            text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr
