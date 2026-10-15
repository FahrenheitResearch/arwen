"""Prove that a loader change only registers new literal header rows.

The original measured receipt stays unchanged. Its pinned loader preimage,
all established header rows and every executable AST node remain authorities.
Callers still validate each measured CUDA unit and compiler option separately.
"""
from __future__ import annotations

import ast
import hashlib


def _parts(source: bytes):
    tree = ast.parse(source.decode('utf-8'))
    rows = None
    for node in tree.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == '_EXTRA_HEADERS':
            if rows is not None:
                raise ValueError('duplicate header declaration')
            if not isinstance(node.value, ast.Dict):
                raise ValueError('headers must remain a literal dictionary')
            pairs = [(ast.literal_eval(k), ast.literal_eval(v)) for k, v in zip(node.value.keys, node.value.values)]
            rows = dict(pairs)
            if len(rows) != len(pairs):
                raise ValueError('duplicate header key')
            if any(not isinstance(k, str) or not isinstance(v, tuple) or any(not isinstance(h, str) for h in v) for k, v in pairs):
                raise ValueError('invalid literal header row')
            node.value = ast.Dict(keys=[], values=[])
    if rows is None:
        raise ValueError('missing header declaration')
    return ast.dump(tree, include_attributes=False), rows


def require_additive_loader(current: bytes, baseline: bytes, expected_sha: str, expected_bytes: int):
    if len(baseline) != expected_bytes or hashlib.sha256(baseline).hexdigest() != expected_sha:
        raise ValueError('loader preimage does not match the measured receipt')
    old_code, old_rows = _parts(baseline)
    new_code, new_rows = _parts(current)
    if old_code != new_code:
        raise ValueError('executable loader code changed; measured recapture required')
    if any(new_rows.get(name) != headers for name, headers in old_rows.items()):
        raise ValueError('established header binding changed; measured recapture required')
    return sorted(set(new_rows) - set(old_rows))
