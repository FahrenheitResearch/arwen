"""Serial float64 verification reference for extensive cell-sum arithmetic.

This is verification only. Production masking and accumulation are Rust.
Index -1 denotes an unreachable partition; other indices are destination
flat indices, in ascending source then row-major partition order.
"""
import numpy as np


def ulp_table(actual, expected):
    """Measured word distance for finite nonnegative float64 mass totals."""
    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    assert actual.shape == expected.shape
    assert np.isfinite(actual).all() and np.isfinite(expected).all()
    assert (actual >= 0).all() and (expected >= 0).all()
    distances = [abs(int(a)-int(b)) for a,b in
                 zip(actual.view(np.uint64).flat, expected.view(np.uint64).flat)]
    return dict(max_ulp=max(distances,default=0), nonzero=sum(v != 0 for v in distances),
                count=len(distances))


def apply(index, values, valid, destination_shape, n=1):
    out = np.zeros(destination_shape, dtype=np.float64)
    mask = np.ones(destination_shape, dtype=bool)
    receipt = np.zeros(4, dtype=np.float64)
    for source, value in enumerate(np.asarray(values, dtype=np.float64).flat):
        finite = np.isfinite(value)
        usable = np.asarray(valid).flat[source] and finite
        if finite:
            receipt[0] += value
        if not usable and finite:
            receipt[3] += value
        piece = value / np.float64(n*n)
        for target in np.asarray(index).reshape(-1, n*n)[source]:
            if target == -1:
                if usable:
                    receipt[2] += piece
            elif usable:
                out.flat[target] += piece
            else:
                mask.flat[target] = False
    for value in out.flat:
        receipt[1] += value
    return out, mask, receipt
