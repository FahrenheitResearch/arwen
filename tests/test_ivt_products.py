"""The standard regional preset exposes every native AR diagnostic."""
from gpuwm import tui_products

def test_native_ar_products_are_default():
    rows=tui_products.presets()
    default=next(row for row in rows['presets'] if row['id']==rows['default'])
    assert {'ivt','iwv','freezing_level','ar_outline'} <= set(default['products'])
