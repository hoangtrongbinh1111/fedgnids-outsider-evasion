'''
Descriptive statistics of the OpTC graph used by Euler, to ground the threat model:
how many malicious edges / snapshots, who the attack sources are, edge reciprocity,
in-degree of attacked nodes, and how often test edges are "new" (never seen before).
'''
import glob
import json
import os
from collections import Counter, defaultdict

import numpy as np

D = './data/optc_euler_features/'
TR_END = 573290
DELTA = 360
TE_END = 745983

files = sorted([f for f in glob.glob(D + '*.txt') if not os.path.basename(f).startswith('back')],
               key=lambda f: int(os.path.basename(f)[:-4]))
snap_edges = defaultdict(set)
mal = defaultdict(set)
for f in files:
    for line in open(f):
        p = line.split(',')
        ts, s, d, lab = int(p[0]), int(p[1]), int(p[2]), int(p[8])
        if s == d or ts >= TE_END:
            continue
        t = ts // DELTA
        snap_edges[t].add((s, d))
        if lab:
            mal[t].add((s, d))

tr_snaps = [t for t in snap_edges if t * DELTA < TR_END]
te_snaps = sorted(t for t in snap_edges if t * DELTA >= TR_END)
train_edges = Counter()
for t in tr_snaps:
    train_edges.update(snap_edges[t])

all_mal = set().union(*mal.values()) if mal else set()
out = {}
out['n_train_snaps'] = len(tr_snaps)
out['n_test_snaps'] = len(te_snaps)
out['edges_per_snap_mean'] = float(np.mean([len(snap_edges[t]) for t in snap_edges]))
out['malicious_snapshot_edges'] = sum(len(v) for v in mal.values())
out['malicious_unique_edges'] = len(all_mal)
out['snaps_with_malicious'] = len(mal)
out['mal_in_train_period'] = sum(len(mal[t]) for t in mal if t * DELTA < TR_END)
srcs = Counter(s for s, _ in all_mal)
dsts = Counter(d for _, d in all_mal)
out['mal_srcs'] = len(srcs)
out['mal_dsts'] = len(dsts)
out['top_mal_srcs'] = srcs.most_common(10)
# reciprocity: fraction of edges whose reverse appears in the same snapshot
rec = [np.mean([(d, s) in snap_edges[t] for s, d in snap_edges[t]]) for t in te_snaps[:200]]
out['reciprocity_test'] = float(np.mean(rec))
# novelty: fraction of test edges never seen in training
te_benign_new, te_mal_new, nb, nm = 0, 0, 0, 0
for t in te_snaps:
    for e in snap_edges[t]:
        if e in mal[t]:
            nm += 1
            te_mal_new += e not in train_edges
        else:
            nb += 1
            te_benign_new += e not in train_edges
out['test_benign_new_frac'] = te_benign_new / max(nb, 1)
out['test_mal_new_frac'] = te_mal_new / max(nm, 1)
# in-degree of malicious dst / out-degree of malicious src in their snapshot
indeg, outdeg = [], []
for t, es in mal.items():
    ind = Counter(d for _, d in snap_edges[t])
    outd = Counter(s for s, _ in snap_edges[t])
    for s, d in es:
        indeg.append(ind[d])
        outdeg.append(outd[s])
out['mal_dst_indeg_median'] = float(np.median(indeg))
out['mal_src_outdeg_median'] = float(np.median(outdeg))
deg_all = [c for t in te_snaps[:200] for c in Counter(d for _, d in snap_edges[t]).values()]
out['indeg_median_all'] = float(np.median(deg_all))
out['mal_snapshots'] = sorted(mal.keys())[:5] + ['...'] + sorted(mal.keys())[-3:]
print(json.dumps(out, indent=1, default=str))
json.dump(out, open('./Exps/optc_stats.json', 'w'), indent=1, default=str)
