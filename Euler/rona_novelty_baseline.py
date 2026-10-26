"""
Novelty ("never-seen edge") baselines on the OpTC test set, CPU only, no model.
  N1: score = -count(edge)   (count = #train+val snapshots containing the directed edge)
  N2: lexicographic (edge count, dst-node count, src-node count), all ascending = more anomalous
      (node count = #train+val snapshots in which the node appears as src or dst)
Ties are broken uniformly at random: AUC is exact (average ranks); AP is averaged over 50 random
tie-breaks; ROC is exact at tie-group boundaries and linear in between (randomized threshold).
Usage (from the repository root):  python Euler/rona_novelty_baseline.py
"""
import glob
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, 'Euler')
from loaders import rona_hooks as RH
import loaders.load_optc as optc

NB = 1 << 20
M = 1 << 12
FPRS = [0.0023, 0.0064, 0.0095, 0.0107, 0.024]
TPRS = [0.855, 0.90, 1.0]


def kws():
    delta = int(0.1 * 3600)
    tr_end = optc.DATE_OF_EVIL_LANL
    val = max(tr_end // 20, delta * 2)
    base = dict(delta=delta, use_flows=False, client='All')
    return {
        'train': dict(start=0, end=tr_end - val, is_test=False, **base),
        'val': dict(start=tr_end - val, end=tr_end, is_test=False, **base),
        'test': dict(start=tr_end, end=optc.TIMES['all'], is_test=True, **base)}


def load(kw):
    p = RH._cache_path(kw)
    print('cache:', p)
    return torch.load(p, map_location='cpu'), os.path.basename(p)


def hist_counts(datas):
    ec, nc = {}, {}
    for d in datas:
        for t in range(d.T):
            ei = d.eis[t].numpy().astype(np.int64)
            for k in np.unique(ei[0] * NB + ei[1]).tolist():
                ec[k] = ec.get(k, 0) + 1
            for v in np.unique(ei.reshape(-1)).tolist():
                nc[v] = nc.get(v, 0) + 1
    return ec, nc


def codes(src, dst, ec, nc, mode):
    ce = np.array([ec.get(int(s) * NB + int(d), 0) for s, d in zip(src, dst)], dtype=np.int64)
    if mode == 'N1':
        return ce
    cd = np.array([nc.get(int(d), 0) for d in dst], dtype=np.int64)
    cs = np.array([nc.get(int(s), 0) for s in src], dtype=np.int64)
    return (ce * M + cd) * M + cs  # ascending code = more anomalous


def metrics(code, y, rng, nshuf=50):
    u, inv = np.unique(code, return_inverse=True)
    G = len(u)
    pos = np.bincount(inv, weights=y, minlength=G)
    tot = np.bincount(inv, minlength=G).astype(float)
    neg = tot - pos
    P, N = pos.sum(), neg.sum()
    cp = np.concatenate([[0], np.cumsum(pos)]) / P
    cn = np.concatenate([[0], np.cumsum(neg)]) / N
    auc = float(np.trapz(cp, cn))
    aps = []
    for _ in range(nshuf):
        perm = rng.permutation(len(y))
        order = perm[np.argsort(inv[perm], kind='stable')]
        yy = y[order]
        prec = np.cumsum(yy) / np.arange(1, len(yy) + 1)
        aps.append(float((prec * yy).sum() / P))
    res = {'AP': float(np.mean(aps)), 'AP_std': float(np.std(aps)), 'AUC': auc}
    res['TPR_at_FPR'] = {str(f): float(np.interp(f, cn, cp)) for f in FPRS}
    res['TPR_at_FPR_det'] = {str(f): float(cp[np.searchsorted(cn, f, side='right') - 1]) for f in FPRS}
    fpr_at = {}
    for r in TPRS:
        k = int(np.argmax(cp >= r - 1e-12))  # first boundary reaching r
        fpr_at[str(r)] = float(np.interp(r, cp[:k + 1], cn[:k + 1])) if r < 1 else float(cn[k])
    res['FPR_at_TPR'] = fpr_at
    res['roc'] = [[float(a), float(b)] for a, b in zip(cn, cp)]
    res['u'], res['neg'], res['pos'] = u, neg, pos
    return res


def inj_flag_rate(res, inj_codes, f):
    """Expected fraction of injected edges flagged by the clean threshold at FPR budget f."""
    neg, u = res['neg'], res['u']
    cn = np.cumsum(neg) / neg.sum()
    g = int(np.searchsorted(cn, f, side='left'))
    prev = cn[g - 1] if g > 0 else 0.0
    q = min(max((f - prev) / (neg[g] / neg.sum()), 0.0), 1.0) if neg[g] > 0 else 0.0
    gi = np.searchsorted(u, inj_codes, side='left')
    full = gi < g
    tie = (gi == g) & (u[np.minimum(gi, len(u) - 1)] == inj_codes)
    return float((full + tie * q).mean()) if len(inj_codes) else float('nan')


def main():
    K = kws()
    tr, trn = load(K['train'])
    va, van = load(K['val'])
    te, ten = load(K['test'])
    print('test T=%d edges=%d mal=%d' % (te.T, sum(e.size(1) for e in te.eis),
                                         sum(int((y > 0).sum()) for y in te.ys)))
    ec, nc = hist_counts([tr, va])
    src = np.concatenate([e[0].numpy() for e in te.eis]).astype(np.int64)
    dst = np.concatenate([e[1].numpy() for e in te.eis]).astype(np.int64)
    y = np.concatenate([(l > 0).numpy() for l in te.ys]).astype(float)
    c1 = codes(src, dst, ec, nc, 'N1')
    print('benign unseen frac %.4f, malicious unseen frac %.4f' % ((c1[y == 0] == 0).mean(), (c1[y == 1] == 0).mean()))
    rng = np.random.RandomState(0)
    out = {'cache': [trn, van, ten], 'n_mal': int(y.sum()), 'n_edges': len(y),
           'benign_unseen': float((c1[y == 0] == 0).mean()), 'mal_unseen': float((c1[y == 1] == 0).mean())}
    R = {}
    for m in ('N1', 'N2'):
        r = metrics(codes(src, dst, ec, nc, m), y, rng)
        R[m] = r
        out[m] = {k: v for k, v in r.items() if k not in ('u', 'neg', 'pos')}
        print(m, r['AP'], r['AUC'], r['TPR_at_FPR'], r['FPR_at_TPR'])
    out['models'] = []
    for p in sorted(glob.glob('Exps/rona/F_*/clean.json')):
        j = json.load(open(p))
        out['models'].append({'tag': os.path.basename(os.path.dirname(p)), 'AP': j['AP'],
                              'AUC': j['AUC'], 'TPR': j['TPR'], 'FPR': j['FPR']})
    ex = [set((e[0].numpy().astype(np.int64) * NB + e[1].numpy()).tolist()) for e in te.eis]
    att = {}
    for tagdir in sorted(glob.glob('Exps/rona/F_*/')):
        tag = os.path.basename(os.path.normpath(tagdir))
        for b in (1, 2, 5, 10):
            pp = os.path.join(tagdir, 'plan_b%d.pt' % b)
            if not os.path.exists(pp):
                continue
            plan = torch.load(pp, map_location='cpu')
            S, D = [], []
            for t, e in plan.items():
                e = np.asarray(e)
                if e.size == 0:
                    continue
                seen = set()
                for s, d in zip(e[0].tolist(), e[1].tolist()):
                    k = s * NB + d
                    if s != d and k not in ex[t] and k not in seen:
                        seen.add(k)
                        S.append(s)
                        D.append(d)
            S, D = np.array(S, dtype=np.int64), np.array(D, dtype=np.int64)
            rec = {'n_inj': int(len(S)), 'unseen_frac': float((codes(S, D, ec, nc, 'N1') == 0).mean())}
            for m in ('N1', 'N2'):
                ic = codes(S, D, ec, nc, m)
                for f in (0.0064, 0.024):
                    rec['%s_flag@%s' % (m, f)] = inj_flag_rate(R[m], ic, f)
            att['%s/b%d' % (tag, b)] = rec
    out['attack'] = att
    json.dump(out, open('Exps/novelty_baseline.json', 'w'), indent=1)


if __name__ == '__main__':
    torch.set_num_threads(4)
    main()
