import importlib.util
from pathlib import Path

import numpy as np


def test_word_hash_dictionary_distinguishes_signed_zero_and_reuses_words():
    path = Path(__file__).parents[1] / 'tools/tendency_fold/receipt.py'
    spec = importlib.util.spec_from_file_location('fold_receipt', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    dictionary, identities = [], {}
    first = mod.words(np.array([0., -0., 1., 1.], np.float32), dictionary, identities, True)
    second = mod.words(np.array([1., -0.], np.float32), dictionary, identities, True)
    assert first['word_ids'] == [0, 1, 2, 2]
    assert second['word_ids'] == [2, 1]
    assert len(dictionary) == 3
    assert len(set(dictionary)) == 3
    assert mod.words(np.array([2.], np.float32), dictionary, identities, False)['word_ids'] == []
    assert len(dictionary) == 3
