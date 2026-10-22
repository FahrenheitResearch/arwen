"""Validate and describe source-roster probability masses without field math."""
import math


def validate_weights(weights, members):
    if weights is None:
        return None
    weights = tuple(weights)
    if len(weights) != members or any(isinstance(w, bool) or not isinstance(w, (int, float))
            or not math.isfinite(w) or w <= 0 for w in weights):
        raise ValueError('member weights need one finite positive mass per member')
    return tuple(float(w) for w in weights)


def unequal_weights(weights):
    return weights is not None and any(w != weights[0] for w in weights)


def roster_weights(order, metadata):
    rows = {row.get('member_id'): row for row in metadata}
    if not any('weight' in row for row in metadata):
        return None
    return validate_weights([rows.get(i, {}).get('weight') for i in order], len(order))


def weight_receipt(order, weights, present=None):
    weights = validate_weights(weights, len(order))
    present = set(order if present is None else present)
    rows = [(i, (weights[j] if weights is not None else 1.))
            for j, i in enumerate(order) if i in present]
    scale = max((w for _, w in rows), default=1.)
    total = math.fsum(w/scale for _, w in rows)
    return dict(member_weights=[dict(member_id=i, weight=w,
                     normalized_weight=(w/scale)/total) for i, w in rows],
        weighting='declared roster masses' if weights is not None else 'equal members',
        weight_normalization='sum of member masses present in this frame; nonfinite cells remain masked',
        spread_estimator='reliability-weighted sample variance: sum(w*(x-mean)^2)/(W-sum(w^2)/W)',
        equal_weight_arithmetic='original serial f64 count arithmetic')
