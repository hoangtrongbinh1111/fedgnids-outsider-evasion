'''
Soft Median diagnostic per phase (validation / last training windows / test), so that the
temperature T can be chosen without looking at test data.

Same statistics as rona_diag_softmed.py (whose published numbers are the median over the four
encoder calls that R.evaluate makes: 2 layers on the validation windows, inside get_cutoff,
and 2 layers on the test windows), but every call is labelled with its phase:
  val         the validation windows (the get_cutoff pass inside R.evaluate)
  test        the clean test windows (the scoring pass inside R.evaluate)
  train_last  the last K training windows, K = number of validation windows (extra pass)
'legacy' pools val + test exactly as rona_diag_softmed.py does (reproduction check).

Per call: median / p90 of the distance between each message and the neighbourhood median,
the median over neighbourhoods (deg >= 3, self loop included) of the max/min softmax weight
ratio at the model's own T (original formula), and the median over the same neighbourhoods of
the distance range (max - min).  For any other T the max/min ratio of a neighbourhood is
exp(range / T), so the median ratio at T is exp(range_median / T) (post hoc, same embeddings).

    python Euler/rona_diag_softmed_val.py --tag F_entente_softmed_s0
'''
import argparse
import json
import math
import os
import sys

import numpy as np
import torch
from torch_scatter import scatter_add, scatter_max, scatter_min

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rona_run_b  # noqa: F401,E402  (registers the aggregators)
import models.robust_conv_ext as X  # noqa: E402
import rona_attack as A  # noqa: E402
import rona_run as R  # noqa: E402
from loaders.tdata import TData  # noqa: E402

T_GRID = [1.0, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001]
STATS = []
CUR = {'label': None, 'enc': None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', required=True)
    ap.add_argument('--no_train', action='store_true', help='skip the last-training-windows pass')
    c = ap.parse_args()
    torch.set_num_threads(4)
    a, s, times, loader, model, h0 = A.load_model(c.tag, torch.device('cpu'))
    enc = model.gcns.module
    orig = X.RobustConvExt._softmed

    def hooked(self, h, src, dst, w, mass, n):
        d = h.size(1)
        with torch.no_grad():
            xmed = self._chunked(self._median_select, h, src, dst, w, n)
            dist = (xmed[dst] - h[src]).norm(dim=1) / d ** 0.5
            z = -dist / X.CFG['softmed_T']
            e = torch.exp(z - z.max())
            sw = e / (scatter_add(e, dst, dim=0, dim_size=n)[dst] + 1e-12)
            k = torch.bincount(dst, minlength=n).float()
            mx = scatter_max(sw, dst, dim=0, dim_size=n)[0]
            mn = scatter_min(sw, dst, dim=0, dim_size=n)[0]
            sel = k >= 3
            dmx = scatter_max(dist, dst, dim=0, dim_size=n)[0]
            dmn = scatter_min(dist, dst, dim=0, dim_size=n)[0]
            rng = (dmx - dmn)[sel]
            data = enc.data
            phase = 'test' if data.is_test else CUR['label']
            STATS.append({'phase': phase, 'T_windows': int(data.T), 'n_edges': int(dist.numel()),
                          'dist_median': float(dist.median()), 'dist_p90': float(dist.quantile(0.9)),
                          'ratio_max_min_median': float((mx[sel] / mn[sel].clamp(min=1e-30)).median()),
                          'range_median': float(rng.median())})
        return orig(self, h, src, dst, w, mass, n)

    X.RobustConvExt._softmed = hooked
    CUR['label'] = 'val'
    res = R.evaluate(model, h0, times, s, loader, torch.device('cpu'))
    n_val = [st['T_windows'] for st in STATS if st['phase'] == 'val'][0]
    if not c.no_train:
        # last n_val training windows (same length as the validation period)
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
        out = {k: float(np.median([r[k] for r in rows]))
               for k in ('dist_median', 'dist_p90', 'ratio_max_min_median', 'range_median')}
        out['n_calls'] = len(rows)
        out['windows'] = rows[0]['T_windows']
        out['per_layer_dist_median'] = [r['dist_median'] for r in rows]
        out['per_layer_ratio'] = [r['ratio_max_min_median'] for r in rows]
        # post hoc median max/min weight ratio at other temperatures: exp(range_median / T),
        # taken per call and then the median over calls (as for the measured ratio)
        out['ratio_at_T'] = {str(t): float(np.median([math.exp(min(r['range_median'] / t, 700.0))
                                                      for r in rows])) for t in T_GRID}
        return out

    summary = {ph: summ([r for r in STATS if r['phase'] == ph])
               for ph in ('val', 'train_last', 'test') if any(r['phase'] == ph for r in STATS)}
    summary['legacy_val+test'] = summ([r for r in STATS if r['phase'] in ('val', 'test')])
    summary.update(tag=c.tag, T=X.CFG['softmed_T'], AP=res['AP'])
    for r in STATS:
        print('CALL', json.dumps(r))
    print('DIAGV', json.dumps(summary))


if __name__ == '__main__':
    main()
