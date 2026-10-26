'''
Degrees of the endpoints of malicious edges in the test windows (LANL or OpTC cache), with the
definition used in the paper: for each malicious edge occurrence, the in-degree (number of
distinct in-neighbors, which is what the GCN aggregates) of its destination and of its source in
that window, plus their weighted in-degree (sum of edge weights), against the same statistics over
all (host, window) pairs with at least one incoming edge.

    python Euler/rona_lanl_indeg.py ./data/rona_cache_lanl
    python Euler/rona_lanl_indeg.py ./data/rona_cache
'''
import glob
import os
import sys

import numpy as np
import torch


def main(cache):
    if cache.endswith('.pt'):
        fn = cache
    else:
        fn = (glob.glob(os.path.join(cache, '*', 'test_*.pt')) + glob.glob(os.path.join(cache, 'test_*.pt')))[0]
    te = torch.load(fn, map_location='cpu')
    print(fn, 'nodes', te.num_nodes, 'windows', te.T)
    has_w = hasattr(te, 'ews') and te.ews is not None
    rows = {'dst': [], 'src': [], 'dst_w': [], 'src_w': []}
    allh, allw = [], []
    for t in range(te.T):
        ei = te.eis[t].numpy()
        y = (te.ys[t] > 0).numpy()
        w = te.ews[t].numpy().ravel() if has_w else np.ones(ei.shape[1])
        n = int(ei.max()) + 1
        deg = np.bincount(ei[1], minlength=n)
        wdeg = np.bincount(ei[1], weights=w, minlength=n)
        recv = deg > 0
        allh.append(deg[recv])
        allw.append(wdeg[recv])
        if y.any():
            s, d = ei[0][y], ei[1][y]
            rows['dst'].extend(deg[d].tolist())
            rows['src'].extend(deg[s].tolist())
            rows['dst_w'].extend(wdeg[d].tolist())
            rows['src_w'].extend(wdeg[s].tolist())
    allh, allw = np.concatenate(allh), np.concatenate(allw)
    print(cache, 'malicious occurrences', len(rows['dst']), 'weights' if has_w else 'no weights')
    for k, v in rows.items():
        v = np.array(v)
        print('  %-6s median %.2f  quartiles %s  mean %.2f' % (k, np.median(v), np.percentile(v, [25, 75]).round(2).tolist(), v.mean()))
    print('  all receiving (host, window) pairs: in-degree median %.1f mean %.2f; weighted median %.2f'
          % (np.median(allh), allh.mean(), np.median(allw)))


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else './data/rona_cache_lanl')
