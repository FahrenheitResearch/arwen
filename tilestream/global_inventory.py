"""Closed per-domain arrays that one owner updates between tile sweeps."""
from __future__ import annotations


def global_keys(keys):
    from tilestream.sfire_spotting import spotting_global_keys
    from tilestream.sfire_smoke import GLOBAL_KEYS
    keys = tuple(keys)
    found = set(spotting_global_keys(keys))
    # Native single-entry history time is source-inert and belongs to the
    # domain. It must survive restart/history without becoming a tile axis.
    found.update(key for key in keys if key == 'fire/lfn_time')
    ledger = {key for key in keys if key.startswith('state/chemdiag_ledger_')}
    if ledger - set(GLOBAL_KEYS):
        raise ValueError('unknown chemistry ledger carrier would bypass tile transport')
    found.update(ledger)
    return tuple(sorted(found))
