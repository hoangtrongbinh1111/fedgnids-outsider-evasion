'''
Semi-synthetic lateral movement on LANL (effect of target degree).

The real red-team edges are removed from the LANL test period, and synthetic lateral movement is
inserted from the real compromised source hosts (sources of red-team edges) to targets of a chosen
in-degree regime. Each of `--pairs` (source, target) pairs is active in `--length` consecutive
windows starting at a random test window in which the target is active. Pairs are never-seen in
the train+val history (as 97.8% of OpTC malicious edges are).
  low  : targets whose median in-degree over the windows in which they receive edges is <= 3
  high : targets whose median in-degree is >= 10
Writes Exps/plans/lanl_synth_<regime>.pt = {'edges': {t: 2 x m}, 'ew': 0.5, 'remove_real': True,
'meta': {...}}.

    python Euler/rona_make_synth_lanl.py --regime low
'''
import argparse
import glob
import os
import sys
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--regime', choices=['low', 'high'], required=True)
    ap.add_argument('--pairs', type=int, default=25)
    ap.add_argument('--length', type=int, default=8)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--cache', default='./data/rona_cache_lanl')
    c = ap.parse_args()
    rng = np.random.RandomState(c.seed)
    te = torch.load(glob.glob(os.path.join(c.cache, '*', 'test_*.pt'))[0], map_location='cpu')
    hist = set()
    for f in glob.glob(os.path.join(c.cache, '*', 'train_*.pt')) + glob.glob(os.path.join(c.cache, '*', 'val_*.pt')):
        d = torch.load(f, map_location='cpu')
        for ei in d.eis:
            hist |= set((ei[0].long() * 100000 + ei[1].long()).tolist())
    n = te.num_nodes
    sources = set()
    indeg = defaultdict(list)        # node -> in-degree in each window where it receives edges
    active = defaultdict(list)       # node -> windows where it receives edges
    for t in range(te.T):
        ei = te.eis[t]
        y = te.ys[t] > 0
        sources |= set(ei[0][y].tolist())
        u, cnt = np.unique(ei[1].numpy(), return_counts=True)
        for v, k in zip(u.tolist(), cnt.tolist()):
            indeg[v].append(k)
            active[v].append(t)
    sources = sorted(sources)
    med = {v: float(np.median(k)) for v, k in indeg.items() if len(k) >= c.length}
    if c.regime == 'low':
        pool = [v for v, m in med.items() if m <= 3]
    else:
        pool = [v for v, m in med.items() if m >= 10]
    pool = [v for v in pool if v not in sources]
    rng.shuffle(pool)
    edges = defaultdict(list)
    meta = []
    for v in pool:
        if len(meta) >= c.pairs:
            break
        s = sources[len(meta) % len(sources)]
        if s * 100000 + v in hist:
            continue
        acts = np.array(active[v])
        acts = acts[acts <= te.T - c.length]
        if len(acts) == 0:
            continue
        t0 = int(rng.choice(acts))
        for t in range(t0, t0 + c.length):
            edges[t].append((s, v))
        meta.append({'src': s, 'dst': v, 't0': t0, 'median_indeg': med[v]})
    out = {'edges': {t: np.array(e, dtype=np.int64).T for t, e in edges.items()}, 'ew': 0.5,
           'remove_real': True, 'meta': meta}
    os.makedirs('./Exps/plans', exist_ok=True)
    fn = './Exps/plans/lanl_synth_%s.pt' % c.regime
    torch.save(out, fn)
    print(fn, 'pairs', len(meta), 'occurrences', sum(len(e) for e in edges.values()),
          'sources', sources, 'median target in-degree', float(np.median([m['median_indeg'] for m in meta])),
          'pool', len(pool))


if __name__ == '__main__':
    main()
