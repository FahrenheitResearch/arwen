"""Exercise the real mapped-head metadata expression at its consumer."""
import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import test_prepared_single_domain_forecast as fixtures

# The metadata expressions call the module's one receipt helper (2.8.8 D-02),
# so the evaluated scope carries the real helper rather than a copy of it.
from gpuwm.mapped_direct import _soil_floor_receipts


def metadata_for(soil):
    tree = ast.parse((Path(__file__).parents[1] / 'gpuwm/mapped_direct.py').read_text())
    expressions = [item.value for call in ast.walk(tree)
                   if isinstance(call, ast.Call)
                   and isinstance(call.func, ast.Attribute)
                   and call.func.attr == 'write_head'
                   for item in call.keywords
                   if item.arg == 'metadata' and isinstance(item.value, ast.Dict)]
    assert len(expressions) == 1
    case = fixtures._cache_metadata_case()
    start = case['exp'].start_time
    from datetime import timedelta
    values = dict(times=[start, start + timedelta(hours=1)], forcing_key='forcing_hours',
                  forcing_axis=[0, 1], boundary_interval_seconds=3600,
                  receipt_identity_sha256='a' * 64, soil=soil,
                  _soil_floor_receipts=_soil_floor_receipts)
    return eval(compile(ast.Expression(expressions[0]), 'mapped-head-metadata', 'eval'), values)


@pytest.mark.parametrize('receipt', [{}, {'policy': 'wrf-ruc-layer-source-smois-max-0.005',
                                       'floored_values': 2, 'floored_columns': 1}])
def test_mapped_head_receipt_binds_without_changing_the_field_contract(receipt):
    user = metadata_for(SimpleNamespace(soil_moisture_floor=receipt))
    case = fixtures._cache_metadata_case()
    case.update(source='mapped', layout='mapped-direct-d01-v1',
                proof={'source_composition': {'receipt_content_sha256': 'a' * 64},
                       **({'soil_moisture_floor': receipt} if receipt else {})})
    case['reader'].header['metadata']['user'] = user
    fixtures.runner._validate_cache_metadata(**case)
    if receipt:
        case['proof']['soil_moisture_floor'] = dict(receipt, floored_values=3)
        with pytest.raises(ValueError, match='user metadata differs'):
            fixtures.runner._validate_cache_metadata(**case)


@pytest.mark.parametrize('receipt', [{}, {'policy': 'wrf-ruc-layer-source-smois-max-0.005',
                                       'floored_values': 2, 'floored_columns': 1}])
def test_chained_tree_root_metadata_reaches_the_same_strict_reader(receipt):
    tree = ast.parse((Path(__file__).parents[1] / 'gpuwm/mapped_direct.py').read_text())
    function = next(item for item in tree.body if isinstance(item, ast.FunctionDef)
                    and item.name == '_prepare_chained_mapped_tree')
    expressions = [item.value for item in ast.walk(function) if isinstance(item, ast.Assign)
                   and any(isinstance(target, ast.Name) and target.id == 'root_metadata'
                           for target in item.targets) and isinstance(item.value, ast.Dict)]
    assert len(expressions) == 2
    scope = dict(c=SimpleNamespace(soil=SimpleNamespace(soil_moisture_floor=receipt),
                                  target_contract={}), receipt_sha256='a' * 64,
                 tree_identity={'composition_receipt_sha256': 'a' * 64},
                 _soil_floor_receipts=_soil_floor_receipts)
    for expression in expressions:
        user = eval(compile(ast.Expression(expression), 'chained-root-metadata', 'eval'), scope)
        case = fixtures._cache_metadata_case()
        start = case['exp'].start_time
        from datetime import timedelta
        user.update(initial_valid_time=start.isoformat(),
                    last_valid_time=(start + timedelta(hours=1)).isoformat(), forcing_hours=[0, 1])
        case.update(source='mapped', layout='mapped-hierarchy-d01-v1',
                    proof={'source_composition': {'receipt_content_sha256': 'a' * 64},
                           'target_contract': {}, **({'soil_moisture_floor': receipt} if receipt else {})})
        case['reader'].header['metadata']['user'] = user
        fixtures.runner._validate_cache_metadata(**case)
        if receipt:
            case['proof']['soil_moisture_floor'] = dict(receipt, floored_values=3)
            with pytest.raises(ValueError, match='user metadata differs'):
                fixtures.runner._validate_cache_metadata(**case)
