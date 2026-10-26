'''
Theory-vs-implementation check. A reference Weiszfeld solver started at the node's own message
m_i would stay there (the self-loop has distance 0 and an enormous weight). Here the exact a-weighted geometric median mu* is computed
from the a-weighted mean with 500 Weiszfeld steps (distances floored at 1e-6 rho to step over data
points). Per gmedn call (layer 1, layer 2 alternate) and per node with >= 2 neighbours besides itself:
    gap  = ||mu^(K) - mu*|| / rho*     deployed K-step layer vs exact median
    self = ||mu^(K) - m_i|| / rho*     deployed layer vs the node's own message (anchoring)
    med  = ||mu*    - m_i|| / rho*     exact median vs own message
rho* = a-weighted mean distance of the messages to mu*. Prints quantiles per replay and layer.

    python Euler/rona_irls_gap2.py --tag F_entente_gmedn_s0 --plugin_trust frozen \
        --plans clean Exps/rona/F_entente_gmedn_s0/plan_b10_ptfrozen.pt
'''
import builtins
import os
import runpy
import sys

import numpy as np
import torch
from torch_scatter import scatter_add

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import robust_conv as RC  # noqa: E402

STATS = {0: [], 1: []}
CALL = [0]
_orig = RC.RobustConv._irls


def _irls(self, h, src, dst, a, n):
    out = _orig(self, h, src, dst, a, n)
    with torch.no_grad():
        mu = self._wmean(h, src, dst, a, n)
        for _ in range(500):
            d = (h[src] - mu[dst]).norm(dim=1)
            rho = scatter_add(a * d, dst, dim=0, dim_size=n) / (scatter_add(a, dst, dim=0, dim_size=n) + 1e-12)
            w = a / torch.maximum(d, 1e-6 * rho[dst] + 1e-12)
            mu = self._wmean(h, src, dst, w, n)
        d = (h[src] - mu[dst]).norm(dim=1)
        rho = scatter_add(a * d, dst, dim=0, dim_size=n) / (scatter_add(a, dst, dim=0, dim_size=n) + 1e-12)
        cnt = scatter_add(torch.ones_like(a), dst, dim=0, dim_size=n)
        ok = (cnt >= 3) & (rho > 1e-6)
        r = rho.clamp(min=1e-12)
        STATS[CALL[0] % 2].append(np.stack([((out - mu).norm(dim=1) / r)[ok].cpu().numpy(),
                                            ((out - h).norm(dim=1) / r)[ok].cpu().numpy(),
                                            ((mu - h).norm(dim=1) / r)[ok].cpu().numpy()], 1))
        CALL[0] += 1
    return out


RC.RobustConv._irls = _irls
_print = builtins.print


def _p(*a, **k):
    if a and isinstance(a[0], str) and a[0].startswith('REPLAY'):
        name = a[1] if len(a) > 1 else ''
        for layer in (0, 1):
            if not STATS[layer]:
                continue
            x = np.concatenate(STATS[layer])
            q = lambda v: 'med %.3f p90 %.3f p99 %.3f' % (np.median(v), np.quantile(v, .9), np.quantile(v, .99))
            _print('IRLSGAP2', name, 'layer%d' % (layer + 1), 'n=%d' % len(x),
                   '| gap', q(x[:, 0]), '| self', q(x[:, 1]), '| med', q(x[:, 2]), flush=True)
            STATS[layer].clear()
        CALL[0] = 0
    _print(*a, **k)


builtins.print = _p
sys.argv = ['rona_replay.py'] + sys.argv[1:] + ['--prefix', 'irlsgap2']
runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'rona_replay.py'), run_name='__main__')
