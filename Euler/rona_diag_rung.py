'''
Diagnostic for the RUNG baseline: distances between degree-normalized neighbor features in each
layer on the clean test windows, the fraction of neighbors whose MCP weight is zero (pruned),
and the spread of the weights within a neighborhood. Answers whether gamma prunes anything at
the distance scale of the trained embeddings.

    python Euler/rona_diag_rung.py F_entente_rung_s0
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

tag = sys.argv[1]
torch.set_num_threads(4)
a, s, times, loader, model, h0 = A.load_model(tag, torch.device('cpu'))
S = []
orig = X.RobustConvExt._rung


def hooked(self, h, src, dst, norm, mass, dinv, n):
    with torch.no_grad():
        fs = h * dinv.unsqueeze(1)
        raw = (fs[src] - fs[dst]).pow(2).sum(1).sqrt()
        y = (raw.pow(2) + X.CFG['rung_eps']).sqrt()
        if self.agg == 'rungl1':
            w = 1.0 / (2 * y)
        else:
            w = (1.0 / (2 * y) - 1.0 / (2 * X.CFG['rung_gamma'])).clamp(min=0, max=X.CFG['rung_max_w'])
        ns = src != dst
        k = torch.bincount(dst[ns], minlength=n)
        mx = scatter_max(w[ns], dst[ns], dim=0, dim_size=n)[0]
        mn = scatter_min(w[ns], dst[ns], dim=0, dim_size=n)[0]
        sel = k >= 3
        S.append({'raw_median': float(raw[ns].median()), 'raw_p90': float(raw[ns].quantile(.9)),
                  'raw_max': float(raw[ns].max()), 'pruned_frac': float((w[ns] == 0).float().mean()),
                  'w_ratio_median': float((mx[sel] / mn[sel].clamp(min=1e-30)).median())})
    return orig(self, h, src, dst, norm, mass, dinv, n)


X.RobustConvExt._rung = hooked
res = R.evaluate(model, h0, times, s, loader, torch.device('cpu'))
summary = {k: float(np.median([x[k] for x in S])) for k in S[0]}
summary['pruned_frac_max'] = float(max(x['pruned_frac'] for x in S))
summary['raw_max_max'] = float(max(x['raw_max'] for x in S))
print('DIAG', tag, json.dumps(summary), 'eps', X.CFG['rung_eps'], 'gamma', X.CFG['rung_gamma'], 'AP', res['AP'])
