"""WRF v4.7.1 sum_pm_gocart word parity on published Fortran fixtures."""
import pytest
import numpy as np
from gpuwm.core.chem_outputs import compile_diagnostic
from gpuwm.verify.chem_outputs_reference import evaluate_cpu
from gpuwm.verify.chem_oracle import ORACLE_ROOT, load, ulp_table
from chem_pm_support import staged_table, WRF_ORDER

CASES = load(ORACLE_ROOT / 'core/sum_pm')


@pytest.mark.parametrize('case', sorted(CASES))
@pytest.mark.parametrize('output', ['PM2_5_DRY', 'PM10'])
def test_sum_pm_oracle(tmp_path, case, output):
    table = staged_table(tmp_path)
    ref = CASES[case]
    fields = {table.row(n).state_attr: np.ascontiguousarray(ref['chem'][:,:,:,i+1].transpose(1,2,0))
              for i,n in enumerate(WRF_ORDER)}
    alt = np.ascontiguousarray(ref['alt'].transpose(1,2,0))
    # ArWen's smoke term (last in both sums) is absent from WRF's
    # sum_pm_gocart; a zero smoke field leaves every WRF word unchanged.
    fields[table.row('smoke').state_attr] = np.zeros_like(alt)
    diag = next(d for d in table.diagnostics if d.output_name == output)
    result = np.empty_like(alt)
    evaluate_cpu(compile_diagnostic(diag, table), fields, alt, None, result)
    expected = np.ascontiguousarray(ref[output].transpose(1,2,0))
    measured = ulp_table(result, expected)
    print(case, output, measured)
    assert measured == dict(max_ulp=0, n_nonzero=0, n=24)
    assert ulp_table(ref['PM2_5_DRY_EC'], np.zeros_like(ref['PM2_5_DRY_EC'])) == dict(max_ulp=0,n_nonzero=0,n=24)


@pytest.mark.parametrize('output', ['PM2_5_DRY', 'PM10'])
def test_smoke_joins_the_pm_sums_last(tmp_path, output):
    """The smoke row is the sums' last term: the result is WRF's GOCART sum
    plus smoke, both float32 words over alt, added in that order."""
    table = staged_table(tmp_path)
    ref = CASES[sorted(CASES)[0]]
    fields = {table.row(n).state_attr: np.ascontiguousarray(ref['chem'][:,:,:,i+1].transpose(1,2,0))
              for i,n in enumerate(WRF_ORDER)}
    alt = np.ascontiguousarray(ref['alt'].transpose(1,2,0))
    diag = next(d for d in table.diagnostics if d.output_name == output)
    assert list(diag.terms[-1]['species']) == ['smoke']
    zero = dict(fields)
    zero[table.row('smoke').state_attr] = np.zeros_like(alt)
    smoke = np.linspace(0.5, 40.0, alt.size, dtype=np.float32).reshape(alt.shape)
    fields[table.row('smoke').state_attr] = smoke
    program = compile_diagnostic(diag, table)
    base = np.empty_like(alt)
    evaluate_cpu(program, zero, alt, None, base)
    with_smoke = np.empty_like(alt)
    evaluate_cpu(program, fields, alt, None, with_smoke)
    assert np.all(with_smoke > base)
    np.testing.assert_allclose(with_smoke - base, smoke / alt, rtol=2e-6)
