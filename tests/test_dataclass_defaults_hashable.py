"""No dataclass field defaults to a value Python 3.11 refuses.

THE BREAKAGE THIS PREVENTS
--------------------------
The package declares ``requires-python = ">=3.11"``.  Python 3.11's
``dataclasses`` refuses an unhashable default when the class is defined
("mutable default ... is not allowed: use default_factory"), so one such
field makes ``import gpuwm`` fail on 3.11 for every user.  ``types.
MappingProxyType`` became hashable only in Python 3.12, so
``LegacyPosting.providers = MappingProxyType({})`` in gpuwm/fetch_routes.py
passed every gate the 2.8.1 cut ran on Python 3.13 and broke the package on
3.11, the interpreter publish.yml's test job uses.  The 2.8.1 pre-cut gate
found it by importing on 3.11.  This scan reads the source, so it holds on
any interpreter the tests run under.
"""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "gpuwm"

#: Calls whose result is unhashable on Python 3.11.
UNHASHABLE_CALLS = {"MappingProxyType", "dict", "list", "set", "bytearray",
                    "defaultdict", "OrderedDict", "Counter", "deque"}


def _is_dataclass(node: ast.ClassDef) -> bool:
    for deco in node.decorator_list:
        target = deco.func if isinstance(deco, ast.Call) else deco
        name = target.attr if isinstance(target, ast.Attribute) else getattr(target, "id", None)
        if name == "dataclass":
            return True
    return False


def _unhashable(value: ast.expr) -> bool:
    if isinstance(value, (ast.Dict, ast.List, ast.Set, ast.DictComp, ast.ListComp, ast.SetComp)):
        return True
    if isinstance(value, ast.Call):
        func = value.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
        if name in UNHASHABLE_CALLS:
            return True
        if name == "field":
            for keyword in value.keywords:
                if keyword.arg == "default" and _unhashable(keyword.value):
                    return True
    return False


def _offenders() -> list[str]:
    found = []
    for path in sorted(ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not (isinstance(node, ast.ClassDef) and _is_dataclass(node)):
                continue
            for item in node.body:
                if isinstance(item, ast.AnnAssign) and item.value is not None and _unhashable(item.value):
                    found.append(f"{path.relative_to(ROOT.parent).as_posix()}:{item.lineno} "
                                 f"{node.name}.{ast.unparse(item.target)}")
    return found


def test_no_dataclass_field_defaults_to_a_value_python_311_refuses():
    offenders = _offenders()
    assert not offenders, (
        "these dataclass fields default to an unhashable value, which Python 3.11 refuses "
        "when the class is defined (import gpuwm then fails on 3.11); use "
        "field(default_factory=...):\n  " + "\n  ".join(offenders))


def test_the_scan_sees_the_shape_that_broke_311():
    tree = ast.parse("from types import MappingProxyType\n"
                     "@dataclass(frozen=True)\nclass Row:\n    providers: dict = MappingProxyType({})\n")
    cls = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef))
    assert _is_dataclass(cls)
    assert _unhashable(cls.body[0].value)
