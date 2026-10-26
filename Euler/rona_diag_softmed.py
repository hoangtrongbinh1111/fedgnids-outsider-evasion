'''
Diagnostic for the Soft Median baseline: distribution of dist/T in each layer on the clean
test windows, and how far the softmax weights are from uniform (effective neighbourhood
share). Answers whether T=1.0 makes Soft Median behave like the weighted mean.

    python Euler/rona_diag_softmed.py --tag F_entente_softmed_s0
'''
import argparse
import json
import os
import sys

import numpy as np
import torch
from torch_scatter import scatter_add

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rona_run_b  # noqa: F401
import models.robust_conv_ext as X
import rona_attack as A
import rona_run as R

STATS = []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', required=True)
    ap.add_argument('--windows', type=int, default=40)
    c = ap.parse_args()
    torch.set_num_threads(4)
    a, s, times, loader, model, h0 = A.load_model(c.tag, torch.device('cpu'))
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
            # ratio of the largest to the smallest softmax weight in each neighbourhood (deg >= 3)
            from torch_scatter import scatter_max, scatter_min
            mx = scatter_max(sw, dst, dim=0, dim_size=n)[0]
            mn = scatter_min(sw, dst, dim=0, dim_size=n)[0]
            sel = k >= 3
            STATS.append({'dist_median': float(dist.median()), 'dist_p90': float(dist.quantile(0.9)),
                          'ratio_max_min_median': float((mx[sel] / mn[sel].clamp(min=1e-30)).median())})
        return orig(self, h, src, dst, w, mass, n)

    X.RobustConvExt._softmed = hooked
    res = R.evaluate(model, h0, times, s, loader, torch.device('cpu'))
    out = {k: float(np.median([st[k] for st in STATS])) for k in STATS[0]}
    out.update(T=X.CFG['softmed_T'], AP=res['AP'], n_calls=len(STATS))
    print('DIAG', json.dumps(out))


if __name__ == '__main__':
    main()
