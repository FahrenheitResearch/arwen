"""Refresh a committed sweep from frozen records, without external templates.

Only new generated rows are rendered. Every retained historical ID keeps its
entire record. Corrected records receive new IDs, and lost pair coverage is
filled through the same constraint solver as gen_combos.py.
"""
import argparse
from collections import Counter, OrderedDict
import copy
import hashlib
import itertools
import json
from pathlib import Path
import random
import gen_combos as g


def coverage(rows, targets):
    words = [set(row.items()) for row in rows]
    return sum(any(target <= row for row in words) for target in targets), len(targets)

def spaces():
    a = g.Space(OrderedDict(list(g.CORE.items()) + list(g.DYN.items()) + list(g.SUB.items())))
    cb = OrderedDict(g.CORE, pbl=g.CORE['pbl'] + [9])
    b = g.Space(OrderedDict(list(cb.items()) + list(g.DYN.items()) + list(g.SUB.items()) +
                          list(g.B_EXTRA.items()) + list(g.B_SUB.items())))
    cd = OrderedDict(g.CORE)
    cd.update(rad=g.CORE['rad'] + ['rte_rrtmgp', 'analytic', 'dudhia'],
              pbl=g.CORE['pbl'] + [900], turb=g.CORE['turb'] + ['k0'])
    d = g.Space(OrderedDict(list(cd.items()) + list(g.DYN.items()) + list(g.SUB.items())))
    c = g.Space.__new__(g.Space)
    c.factors = g.C_FACTORS
    c.order = list(c.factors)
    c.domains = {n: list(s[0]) + [g.NA] if isinstance(s, tuple) else list(s) for n, s in c.factors.items()}
    c.cons = g.c_constraints(c.factors)
    c.by_var = {n: [] for n in c.order}
    for deps, fn in c.cons:
        for n in set(deps):
            c.by_var[n].append((deps, fn))
    bn = set(cb) | set(g.DYN) | set(g.B_EXTRA) | set(g.B_SUB) | {
        'ysu_topdown_pblmix', 'isftcflx', 'iz0tlnd', 'monthly_veg', 'rdmaxalb', 'opt_thcnd'}
    pa, ta = g.required_pairs(a), g.required_triples(a, ['mp', 'pbl', 'sfclay', 'lsm'])
    pb = g.required_pairs(b, lambda t: bool(set(t) & set(g.B_NEW_ITEMS)) and all(n in bn for n, _ in t))
    dn = set(g.CORE) | set(g.DYN)
    pd = g.required_pairs(d, lambda t: bool(set(t) & set(g.D_NEW_ITEMS)) and all(n in dn for n, _ in t)
                          and {n for n, _ in t} != {'rad'})
    pd = {t for t in pd if not (('rad', 'analytic') in t and any(n in g.DYN for n, _ in t))}
    pc = g.required_pairs(c)
    return dict(A=a, B=b, C=c, D=d), dict(A_pairs=pa, A_core_triples=ta, B_pairs=pb, C_pairs=pc, D_pairs=pd)


def refresh(current, historical, seed=20261007):
    sp, required = spaces()
    old = {r['id']: r for r in historical['combos']}
    original = {r['id']: r for r in current['combos']}
    maximum = {s: max((int(cid[1:]) for cid in set(old) | set(original)
                      if cid.startswith(s) and cid[1:].isdigit()), default=0) for s in sp}
    reasons = {}
    rows = []

    def assignment(rec, s):
        return {n: rec['factors'].get(n, g.NA) for n in sp[s].order}

    def valid(rec):
        s = rec['stratum']
        if s in sp:
            row = assignment(rec, s)
            if any(row[n] not in sp[s].domains[n] for n in row) or any(not fn(row) for _, fn in sp[s].cons):
                return False
        ph, dom = rec['namelist']['physics'], rec['namelist']['domains']
        return (not (ph.get('bl_pbl_physics') == 5 and ph.get('ishallow') == 1) and
                (ph['mp_physics'] != 28 or (dom.get('wif_input_opt'), dom.get('num_wif_levels'),
                                          ph.get('use_aero_icbc')) == (1, 30, True)))

    def new_id(rec):
        s = rec['stratum']
        if s in sp:
            maximum[s] += 1
            return f'{s}{maximum[s]:03d}'
        return rec['id'] + '-r290'

    for record in current['combos']:
        rec = copy.deepcopy(record)
        if not valid(rec):
            reasons[rec['id']] = 'current constraints reject this recorded assignment'
            continue
        cid = rec['id']
        if cid in old and rec != old[cid]:
            before, after = (dict(r) for r in (old[cid], rec))
            before.pop('cases'); after.pop('cases')
            if before == after and valid(old[cid]):
                rec = copy.deepcopy(old[cid])
            else:
                rec['id'] = new_id(rec)
                reasons[cid] = 'corrected frozen record has different bytes; replacement is ' + rec['id']
        rows.append(rec)

    rng = random.Random(seed)
    hrrr = next(r for r in rows if r['id'] == 'S-hrrr-v4-gsd41')

    def append(s, factors, cases=None):
        template = next(r for r in current['combos'] if r['stratum'] == s)
        rec = copy.deepcopy(template)
        rec['id'] = new_id(rec)
        rec['cases'] = cases or ['iowa-convective']
        rec['origin'] = 'constraint-aware coverage refresh from committed records'
        if s == 'C':
            nl, ws = g.apply_fork(hrrr['factors'], hrrr['namelist'], hrrr['woof_settings'], factors)
            factors = dict(hrrr['factors'], **{n: v for n, v in factors.items() if v != g.NA})
        else:
            nl, ws = g.render(factors)
        rec.update(factors={n: v for n, v in factors.items() if v != g.NA}, namelist=nl, woof_settings=ws,
                   declared_divergences=g.divergences(nl, ws))
        for name in ('also', 'note', 'case_overrides'):
            rec.pop(name, None)
        if s == 'C':
            longest = max(float(nl['physics'].get('radt', 0)), float(nl['physics'].get('cudt', 0)))
            need = int(round(longest * 60 / 15)) + 4
            if need > 20:
                rec['case_overrides'] = dict(steps=need)
        assert valid(rec), rec['id']
        rows.append(rec)

    for s in sp:
        req = set().union(*(targets for name, targets in required.items() if name.startswith(s + '_')))
        existing = [assignment(r, s) for r in rows if r['stratum'] == s]
        for factors in g.build_cover(sp[s], req, rng, candidates=40, existing=existing):
            append(s, factors)

    variants = [('gulf-coast-convective', 'coast_variant_pairs',
                 ['mp', 'pbl', 'sfclay', 'lsm', 'isftcflx', 'iz0tlnd', 'sf_lake_physics', 'rad', 'cu']),
                ('iowa-snow-ice', 'snow_variant_pairs',
                 ['mp', 'pbl', 'sfclay', 'lsm', 'ruc_soil_layers', 'mosaic_lu', 'mosaic_soil',
                  'fractional_seaice', 'monthly_veg', 'rdmaxalb', 'opt_thcnd', 'rad'])]
    variant_receipts = {}
    for case, name, factors in variants:
        targets = set()
        for rec in rows:
            if rec['stratum'] == 'A':
                f = rec['factors']
                targets.update(frozenset([(a, f[a]), (b, f[b])]) for a, b in itertools.combinations(factors, 2)
                               if a in f and b in f and f[a] != g.NA and f[b] != g.NA)
        existing = [assignment(r, 'A') for r in rows if r['stratum'] == 'A' and case in r['cases']]
        for f in g.build_cover(sp['A'], targets, rng, candidates=40, existing=existing):
            append('A', f, ['iowa-convective', case])
        covered, total = coverage([assignment(r, 'A') for r in rows if r['stratum'] == 'A' and case in r['cases']], targets)
        assert covered == total
        variant_receipts[name] = dict(covered=covered, required=total,
                                     runs=sum(r['stratum'] == 'A' and case in r['cases'] for r in rows))

    final = {r['id']: r for r in rows}
    assert len(final) == len(rows)
    for cid in set(old) & set(final):
        assert old[cid] == final[cid], cid
    for cid in ('A071', 'A195'):
        assert final[cid] == original[cid], 'recorded exact-chain case changed: ' + cid
    doc = copy.deepcopy(current)
    doc['generator'] = 'refresh_committed.py --seed ' + str(seed)
    doc['combos'] = rows
    doc['regeneration'] = dict(keep_rows_from=current['regeneration']['keep_rows_from'],
                              kept=dict(Counter(r['stratum'] for r in rows if r['id'] in old)),
                              dropped=[dict(id=cid, reason=reasons.get(cid, 'excluded by prior recorded rules'))
                                       for cid in sorted(set(old) - set(final))],
                              added=sorted(set(final) - set(old)))
    doc['counts'] = dict(combos=len(rows), by_stratum=dict(Counter(r['stratum'] for r in rows)),
                         runs_including_variant_cases=sum(len(r['cases']) for r in rows),
                         extra_repeat_runs=sum('repeat_run_0ulp' in r['checks'] for r in rows))
    doc['coverage'] = {}
    for name, targets in required.items():
        s = name[0]
        covered, total = coverage([assignment(r, s) for r in rows if r['stratum'] == s], targets)
        assert covered == total, (name, covered, total)
        doc['coverage'][name] = dict(covered=covered, required=total)
    doc['coverage'].update(variant_receipts)
    doc['wrf_rules'] = list(dict.fromkeys(current['wrf_rules'] + [
        'PBL off with km_opt=4 and epssm=0.1 is excluded because recorded WRF runs become non-finite']))
    return doc


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parent
    ap.add_argument('--current', type=Path, default=here / 'combos-v2.json')
    ap.add_argument('--historical', type=Path, default=here / 'combos-v1.json')
    ap.add_argument('--out', type=Path, default=here / 'combos.json')
    ap.add_argument('--receipt', type=Path, required=True)
    args = ap.parse_args()
    raw, historic = args.current.read_bytes(), args.historical.read_bytes()
    current, historical = json.loads(raw), json.loads(historic)
    assert current['regeneration']['keep_rows_from']['sha256'] == hashlib.sha256(historic).hexdigest()
    result = refresh(current, historical)
    result['refresh_inputs'] = dict(current_sha256=hashlib.sha256(raw).hexdigest(),
                                  historical_sha256=hashlib.sha256(historic).hexdigest(),
                                  external_templates_read=False)
    args.out.write_text(json.dumps(result, indent=1) + '\n')
    receipt = dict(counts=result['counts'], coverage=result['coverage'], regeneration=result['regeneration'],
                   inputs=result['refresh_inputs'], output_sha256=hashlib.sha256(args.out.read_bytes()).hexdigest())
    args.receipt.write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(result['counts']), flush=True)


if __name__ == '__main__':
    main()
