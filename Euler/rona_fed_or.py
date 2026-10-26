'''
Cross-silo detection under per-silo inference (subnet partition).

Every silo scores the edges it sees (its subgraph plus one-hop cross-silo edges) with the
shared federated model, its own validation cutoff and its own trust history. An edge seen
by several silos (a cross-silo edge) is flagged if ANY of them flags it (OR rule); a
malicious edge is missed only if every silo that sees it misses it.

Inputs are the replay files written by rona_replay.py --client k --prefix replay_c<k>:
    Exps/rona/<tag>/replay_c<k>_<plan>.npz
For each plan this prints the OR-combined miss rate (SR), the stealthy miss rate, the
OR-combined FPR and injected-flag rate, and the per-silo numbers for comparison.

    python Euler/rona_fed_or.py --tag F_sub_gcn_s0 --silos 0 1 2 3 --plans clean plan_b1_c3 plan_b1
'''
import argparse
import json
import os

import numpy as np

ROOT = './Exps/rona'


def keys(r):
    return (r['t'].astype(np.int64) << 40) | (r['src'].astype(np.int64) << 20) | r['dst'].astype(np.int64)


def combine(tag, silos, plan, prefix='replay_c', fpr=None):
    '''fpr=None: every silo uses its own validation cutoff; fpr=f: every silo flags the
    fraction f of its benign test edges (threshold on its own scores, as for TPR at fixed FPR)'''
    per = {}
    for k in silos:
        fn = os.path.join(ROOT, tag, '%s%s_%s.npz' % (prefix, k, plan))
        if not os.path.exists(fn):
            raise FileNotFoundError(fn)
        r = dict(np.load(fn))
        if fpr is None:
            r['flag'] = r['p'] <= r['cutoff']
        else:
            ben = (r['y'] == 0) & (r['inj'] == 0)
            r['flag'] = r['p'] <= np.quantile(r['p'][ben], fpr)
        r['key'] = keys(r)
        per[k] = r
    # union of all edges seen by any silo, with OR of flags
    allk = np.concatenate([per[k]['key'] for k in silos])
    ally = np.concatenate([per[k]['y'] for k in silos]).astype(bool)
    alli = np.concatenate([per[k]['inj'] for k in silos]).astype(bool)
    allf = np.concatenate([per[k]['flag'] for k in silos])
    nsee = np.ones_like(allk)
    uk, inv = np.unique(allk, return_inverse=True)
    flag = np.zeros(len(uk), bool)
    np.logical_or.at(flag, inv, allf)
    y = np.zeros(len(uk), bool)
    np.logical_or.at(y, inv, ally)
    inj = np.zeros(len(uk), bool)
    np.logical_or.at(inj, inv, alli)
    seen_by = np.zeros(len(uk), int)
    np.add.at(seen_by, inv, nsee)
    mal = y & ~inj
    ben = ~y & ~inj
    out = {'plan': plan, 'operating_point': 'cutoff' if fpr is None else 'FPR=%g per silo' % fpr,
           'n_mal': int(mal.sum()), 'n_mal_cross': int((mal & (seen_by > 1)).sum()),
           'SR_or': float(1 - flag[mal].mean()),
           'SR_or_cross': float(1 - flag[mal & (seen_by > 1)].mean()) if (mal & (seen_by > 1)).any() else None,
           'FPR_or': float(flag[ben].mean()),
           'n_inj': int(inj.sum()),
           'inj_flag_or': float(flag[inj].mean()) if inj.any() else 0.0}
    # stealthy miss (as in rona_run.evaluate): missed, and no injected edge of the same
    # snapshot is flagged by any silo
    t = uk >> 40
    alarmed = set(t[inj & flag].tolist())
    quiet = np.array([tt not in alarmed for tt in t[mal].tolist()], bool)
    out['SR_or_stealthy'] = float((~flag[mal] & quiet).mean())
    for k in silos:
        r = per[k]
        m = (r['y'] > 0) & (r['inj'] == 0)
        b = (r['y'] == 0) & (r['inj'] == 0)
        out['silo%s' % k] = {'n_mal': int(m.sum()),
                             'SR': float(1 - r['flag'][m].mean()) if m.any() else None,
                             'FPR': float(r['flag'][b].mean()),
                             'n_inj': int(r['inj'].sum()),
                             'inj_flag': float(r['flag'][r['inj'] > 0].mean()) if r['inj'].any() else 0.0}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', required=True)
    ap.add_argument('--silos', nargs='+', default=['0', '1', '2', '3'])
    ap.add_argument('--plans', nargs='+', required=True)
    ap.add_argument('--prefix', default='replay_c')
    ap.add_argument('--fpr', nargs='*', type=float, default=[0.01, 0.001],
                    help='also report every silo at these fixed per-silo FPRs')
    ap.add_argument('--out', default=None)
    c = ap.parse_args()
    res = {}
    for p in c.plans:
        res[p] = {'cutoff': combine(c.tag, c.silos, p, c.prefix)}
        for f in c.fpr:
            res[p]['fpr%g' % f] = combine(c.tag, c.silos, p, c.prefix, fpr=f)
        for k, v in res[p].items():
            short = {m: (round(x, 4) if isinstance(x, float) else x) for m, x in v.items() if not m.startswith('silo')}
            print('FEDOR', c.tag, k, json.dumps(short), flush=True)
    if c.out:
        json.dump(res, open(os.path.join(ROOT, c.tag, c.out), 'w'), indent=1)


if __name__ == '__main__':
    main()
