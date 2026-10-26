'''
Semi-synthetic lateral-movement campaigns on OpTC (does the attack/defense
evidence depend on the single recorded red-team campaign?).

The real red-team edges are removed from the OpTC test period (rona_hooks._synthesize), and a
synthetic campaign is inserted: `--sources` random internal hosts that are active as sources in
the test period act as compromised hosts, and `--pairs` (source, target) pairs are drawn with
targets stratified by in-degree (equal numbers of targets whose median benign in-degree over the windows in which they are
active is zero, at most the median of the positive values, or above it). Each
pair is never seen in the train+val history and is active in `--length` consecutive test windows
starting at a random window in which the target is active. Edge weight 0.5 (a single
connection attempt, like 93.8% of OpTC test edges).
Writes Exps/plans/optc_synth_c<seed>.pt = {'edges': {t: 2 x m}, 'ew': 0.5, 'remove_real': True,
'meta': [{'src', 'dst', 't0', 'median_indeg', 'stratum'}]}.

    python Euler/rona_make_synth_optc.py --seed 0
'''
import argparse
import os
import sys
from collections import Counter, defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rona_run as R  # noqa: E402
from loaders import rona_hooks as RH  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--pairs', type=int, default=60)
    ap.add_argument('--sources', type=int, default=5)
    ap.add_argument('--length', type=int, default=8)
    c = ap.parse_args()
    rng = np.random.RandomState(1000 + c.seed)
    a = R.get_args(['--tag', 'X_synth_gen', '--fed', 'Momentum', '--agg', 'gcn'])
    s, times, loader, _ = R.build_env(a)
    RH.CFG['synth'] = None
    for ph in ('train', 'val'):
        loader(**R.data_kwargs(times, ph, 'All', s.mc))
    hist = RH.HISTORY['All']
    seen = set(hist['train']) | set(hist['val'])
    te = loader(**R.data_kwargs(times, 'test', 'All', s.mc))
    n = te.num_nodes
    sizes = Counter(s.mc.values())
    internal = max(sizes, key=sizes.get)
    is_int = lambda v: s.mc.get(int(v)) == internal  # noqa: E731
    real_mal = set()
    indeg, alive, out_active = defaultdict(list), defaultdict(list), set()
    for t in range(te.T):
        ei = te.eis[t]
        y = te.ys[t] > 0
        real_mal |= set(ei[0][y].tolist()) | set(ei[1][y].tolist())
        ben = ei[:, ~y]
        out_active |= set(ben[0].tolist())
        cnt = Counter(ben[1].tolist())
        for v in set(ben[0].tolist()) | set(ben[1].tolist()):
            alive[v].append(t)
            indeg[v].append(cnt.get(v, 0))
    # target in-degree = median benign in-degree over the windows in which the host is active
    # (as source or destination); most internal hosts are workstations that rarely receive
    med = {v: float(np.median(k)) for v, k in indeg.items() if len(k) >= c.length and is_int(v)}
    pos = np.array(sorted(m for m in med.values() if m > 0))
    q1 = float(np.median(pos)) if len(pos) else 0.0
    q2 = q1
    strata = {'zero': [v for v, m in med.items() if m == 0],
              'low': [v for v, m in med.items() if 0 < m <= q1],
              'high': [v for v, m in med.items() if m > q1]}
    active = alive
    src_pool = sorted(v for v in out_active if is_int(v) and v not in real_mal)
    # redraw the compromised sources (at most 20 times) until every stratum can be filled
    for attempt in range(20):
        sources = [int(x) for x in rng.choice(src_pool, c.sources, replace=False)]
        edges = defaultdict(list)
        meta = []
        skipped = Counter()
        per = c.pairs // 3
        for name, pool in strata.items():
            pool = [v for v in pool if v not in sources]
            rng.shuffle(pool)
            got = 0
            for v in pool:
                if got >= per:
                    break
                # rotate over the compromised sources, using the first one never seen talking to v
                order = [sources[(len(meta) + i) % len(sources)] for i in range(len(sources))]
                fresh = [x for x in order if x * n + v not in seen and x != v]
                if not fresh:
                    skipped[name] += 1
                    continue
                src = fresh[0]
                acts = np.array(active[v])
                acts = acts[acts <= te.T - c.length]
                if len(acts) == 0:
                    continue
                t0 = int(rng.choice(acts))
                for t in range(t0, t0 + c.length):
                    edges[t].append((src, v))
                meta.append({'src': src, 'dst': int(v), 't0': t0, 'median_indeg': med[v], 'stratum': name})
                got += 1
        if len(meta) >= 0.9 * per * 3:
            break
    out = {'edges': {t: np.array(e, dtype=np.int64).T for t, e in edges.items()}, 'ew': 0.5,
           'remove_real': True, 'meta': meta, 'sources': sources, 'tertiles': [float(q1), float(q2)]}
    os.makedirs('./Exps/plans', exist_ok=True)
    fn = './Exps/plans/optc_synth_c%d.pt' % c.seed
    torch.save(out, fn)
    st = Counter(m['stratum'] for m in meta)
    print(fn, 'pairs', len(meta), dict(st), 'occurrences', sum(len(e) for e in edges.values()),
          'sources', sources, 'tertiles', (q1, q2), 'skipped (all sources seen)', dict(skipped),
          'pool sizes', {k: len(v) for k, v in strata.items()},
          'median indeg per stratum', {k: float(np.median([m['median_indeg'] for m in meta if m['stratum'] == k]))
                                       for k in st})


if __name__ == '__main__':
    main()
