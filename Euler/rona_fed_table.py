'''
Aggregate the cross-silo results (mean and population std, as in the other tables) (rona_fed_or.py -> Exps/rona/<tag>/fed_or.json) over seeds.

For each detector, operating point and budget, the attacker's result is the worst case (for the
defender) over the three attack views: plans optimized on the global graph, on silo 3's view
and on silo 2's view, each replayed into every silo and OR-combined.

    python Euler/rona_fed_table.py --seeds 0 1 2
'''
import argparse
import json
import os

import numpy as np

ROOT = './Exps/rona'
VIEWS = ['', '_c3', '_c2']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    ap.add_argument('--budgets', nargs='+', type=int, default=[1, 10])
    ap.add_argument('--prefix', default='F_sub', help='F_sub (K=4) or F_k6 (K=6)')
    ap.add_argument('--views', nargs='+', default=['', '_c3', '_c2'], help="plan suffixes ('' = global view)")
    ap.add_argument('--out', default='fed_table.json')
    c = ap.parse_args()
    VIEWS = [v if v != 'global' else '' for v in c.views]
    names = ['global' if v == '' else 'silo' + v[2:] for v in VIEWS]
    out = {}
    for det, sf in (('gcn', ''), ('gmedn', '_ptfrozen')):
        for op in ('cutoff', 'fpr0.01', 'fpr0.001'):
            rows = {k: [] for k in ['clean', 'fpr', 'clean_cross'] +
                    ['b%d%s' % (b, m) for b in c.budgets for m in ('', '_st', '_flag', '_cross', '_view')]}
            seeds = []
            for s in c.seeds:
                fn = os.path.join(ROOT, '%s_%s_s%d' % (c.prefix, det, s), 'fed_or.json')
                if not os.path.exists(fn):
                    continue
                d = json.load(open(fn))
                if op not in d['clean']:
                    continue
                seeds.append(s)
                rows['clean'].append(100 * d['clean'][op]['SR_or'])
                rows['clean_cross'].append(100 * d['clean'][op]['SR_or_cross'])
                rows['fpr'].append(100 * d['clean'][op]['FPR_or'])
                for b in c.budgets:
                    cand = [d['plan_b%d%s%s' % (b, sf, v)][op] for v in VIEWS]
                    i = int(np.argmax([x['SR_or'] for x in cand]))
                    w = cand[i]
                    rows['b%d' % b].append(100 * w['SR_or'])
                    rows['b%d_st' % b].append(100 * max(x['SR_or_stealthy'] for x in cand))
                    rows['b%d_flag' % b].append(100 * min(x['inj_flag_or'] for x in cand))
                    rows['b%d_cross' % b].append(100 * max(x['SR_or_cross'] for x in cand))
                    rows['b%d_view' % b].append(names[i])
            if not seeds:
                continue
            key = '%s %s' % (det, op)
            out[key] = {'seeds': seeds}
            print('== %s (seeds %s)' % (key, seeds))
            for k, v in rows.items():
                if k.endswith('_view'):
                    out[key][k] = v
                    print('   %-12s %s' % (k, v))
                else:
                    a = np.array(v, float)
                    out[key][k] = {'mean': float(a.mean()), 'std': float(a.std()),
                                   'min': float(a.min()), 'max': float(a.max()), 'per_seed': v}
                    print('   %-12s %6.1f +- %4.1f  [%5.1f, %5.1f]  %s' % (k, a.mean(), out[key][k]['std'], a.min(), a.max(),
                                                                    ' '.join('%.1f' % x for x in v)))
    json.dump(out, open(os.path.join(ROOT, c.out), 'w'), indent=1)


if __name__ == '__main__':
    main()
