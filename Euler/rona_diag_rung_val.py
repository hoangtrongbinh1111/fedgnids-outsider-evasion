'''
RUNG diagnostic per phase (validation / last training windows / test), so that the MCP
threshold gamma can be chosen without looking at test data.

Same statistics as rona_diag_rung.py (whose published numbers are the median, or for
raw_max_max the maximum, over the four encoder calls that R.evaluate makes: 2 layers on the
validation windows inside get_cutoff and 2 layers on the test windows), but every call is
labelled with its phase:
  val         the validation windows (the get_cutoff pass inside R.evaluate)
  test        the clean test windows (the scoring pass inside R.evaluate)
  train_last  the last K training windows, K = number of validation windows (extra pass)
'legacy' pools val + test exactly as rona_diag_rung.py does (reproduction check).

Statistics are taken on the first IRLS iteration (f = h), over non-self edges, as in the
original. For each gamma on a grid we also report the fraction of neighbours the MCP weight
would prune (y >= gamma) on the same embeddings (post hoc), and the fraction of nodes with
>= 1 neighbour all of whose neighbours would be pruned.

    python Euler/rona_diag_rung_val.py F_entente_rung_s0
'''
import json
import os
import sys

import numpy as np
import torch
from torch_scatter import scatter_max, scatter_min

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rona_run_b  # noqa: E402,F401  (registers the aggregators)
import models.robust_conv_ext as X  # noqa: E402
import rona_attack as A  # noqa: E402
import rona_run as R  # noqa: E402
from loaders.tdata import TData  # noqa: E402

G_GRID = [3.0, 1.0, 0.5, 0.3, 0.2, 0.1]
tag = sys.argv[1]
no_train = '--no_train' in sys.argv
torch.set_num_threads(4)
a, s, times, loader, model, h0 = A.load_model(tag, torch.device('cpu'))
enc = model.gcns.module
S = []
CUR = {'label': None}
orig = X.RobustConvExt._rung


def mcp_w(y, gam):
    return (1.0 / (2 * y) - 1.0 / (2 * gam)).clamp(min=0, max=X.CFG['rung_max_w'])


def hooked(self, h, src, dst, norm, mass, dinv, n):
    with torch.no_grad():
        fs = h * dinv.unsqueeze(1)
        raw = (fs[src] - fs[dst]).pow(2).sum(1).sqrt()
        y = (raw.pow(2) + X.CFG['rung_eps']).sqrt()
        if self.agg == 'rungl1':
            w = 1.0 / (2 * y)
        else:
            w = mcp_w(y, X.CFG['rung_gamma'])
        ns = src != dst
        k = torch.bincount(dst[ns], minlength=n)
        mx = scatter_max(w[ns], dst[ns], dim=0, dim_size=n)[0]
        mn = scatter_min(w[ns], dst[ns], dim=0, dim_size=n)[0]
        sel = k >= 3
        has = k >= 1
        data = enc.data
        rec = {'phase': 'test' if data.is_test else CUR['label'], 'T_windows': int(data.T),
               'n_edges': int(ns.sum()),
               'raw_median': float(raw[ns].median()), 'raw_p90': float(raw[ns].quantile(.9)),
               'raw_max': float(raw[ns].max()), 'y_median': float(y[ns].median()),
               'pruned_frac': float((w[ns] == 0).float().mean()),
               'w_ratio_median': float((mx[sel] / mn[sel].clamp(min=1e-30)).median())}
        for g in G_GRID:
            wg = mcp_w(y[ns], g)
            rec['pruned_frac_g%s' % g] = float((wg == 0).float().mean())
            kept = torch.bincount(dst[ns][wg > 0], minlength=n)
            rec['allpruned_node_frac_g%s' % g] = float(((kept == 0) & has).float().sum() / has.float().sum())
        S.append(rec)
    return orig(self, h, src, dst, norm, mass, dinv, n)


X.RobustConvExt._rung = hooked
CUR['label'] = 'val'
res = R.evaluate(model, h0, times, s, loader, torch.device('cpu'))
n_val = [r['T_windows'] for r in S if r['phase'] == 'val'][0]
if not no_train:
    CUR['label'] = 'train_last'
    data = loader(**R.data_kwargs(times, 'train', 'All', s.mc))
    K = n_val
    for name in ('eis', 'ews', 'masks', 'trust', 'ys', 'cnt', 'eas', 'hist_counts', 'injected',
                 'ei_sizes', 'slices'):
        f = getattr(data, name, None)
        if isinstance(f, list) and len(f) == data.T:
            setattr(data, name, f[-K:])
    data.T = K
    enc.data = data
    enc._batch_cache = {}
    enc.forward(TData.ALL, True)


def summ(rows):
    keys = [k for k in rows[0] if k not in ('phase', 'T_windows', 'n_edges')]
    out = {k: float(np.median([x[k] for x in rows])) for k in keys}
    out['pruned_frac_max'] = float(max(x['pruned_frac'] for x in rows))
    out['raw_max_max'] = float(max(x['raw_max'] for x in rows))
    out['per_layer_raw_median'] = [x['raw_median'] for x in rows]
    out['per_layer_raw_max'] = [x['raw_max'] for x in rows]
    out['n_calls'] = len(rows)
    out['windows'] = rows[0]['T_windows']
    return out


summary = {ph: summ([r for r in S if r['phase'] == ph])
           for ph in ('val', 'train_last', 'test') if any(r['phase'] == ph for r in S)}
summary['legacy_val+test'] = summ([r for r in S if r['phase'] in ('val', 'test')])
summary.update(tag=tag, eps=X.CFG['rung_eps'], gamma=X.CFG['rung_gamma'], AP=res['AP'])
for r in S:
    print('CALL', json.dumps(r))
print('DIAGV', json.dumps(summary))
