'''
Numerical check of the influence bound of the paper on the RobustConv aggregators.

One target node with |C| clean in-neighbors (+ self loop, weight 1) and |A| injected
neighbors whose messages are placed adversarially far away (scale s). We report the
output shift of the target node for mean / gmed (IRLS) with and without trust, versus
the bound 2*rho/(1 - W_A/W_C) evaluated on the exact geometric median.
'''
import json
import sys

import torch

sys.path.insert(0, './Euler')
from models.robust_conv import RobustConv


def exact_gmed(h, w, iters=500):
    y = (w[:, None] * h).sum(0) / w.sum()
    for _ in range(iters):
        d = (h - y).norm(dim=1).clamp(min=1e-9)
        y = ((w / d)[:, None] * h).sum(0) / (w / d).sum()
    return y


def run(n_clean, n_adv, scale, agg, trust, tau0=0.1, ew=0.73, dim=16, seed=0, iters=3):
    torch.manual_seed(seed)
    n = n_clean + n_adv + 1          # node 0 is the target
    conv = RobustConv(n, dim, agg=agg, iters=iters)
    with torch.no_grad():
        # one-hot inputs -> message of node j is column j of the weight matrix
        W = torch.randn(dim, n)
        W[:, n_clean + 1:] = scale * torch.randn(dim, n_adv).sign()
        conv.lin.weight.copy_(W)
        conv.bias.zero_()
    x = torch.eye(n)
    src_c = torch.arange(1, n_clean + 1)
    src_a = torch.arange(n_clean + 1, n)
    ei_c = torch.stack([src_c, torch.zeros_like(src_c)])
    ei_a = torch.stack([src_a, torch.zeros_like(src_a)])
    ew_c = torch.full((n_clean,), ew)
    tr_c = torch.ones(n_clean)
    tr_a = torch.full((n_adv,), tau0 if trust else 1.0)
    with torch.no_grad():
        clean = conv(x, ei_c, ew_c, tr_c if trust else None)[0]
        adv = conv(x, torch.cat([ei_c, ei_a], 1), torch.cat([ew_c, torch.full((n_adv,), ew)]),
                   torch.cat([tr_c, tr_a]) if trust else None)[0]
        # bound on the exact weighted geometric median (self loop weight 1)
        hC = torch.cat([W[:, :1].t(), W[:, 1:n_clean + 1].t()])
        aC = torch.cat([torch.ones(1), ew_c])
        mu0 = exact_gmed(hC, aC)
        rho = ((hC - mu0).norm(dim=1) * aC).sum() / aC.sum()
        WC, WA = aC.sum().item(), (ew * tr_a).sum().item()
        bound = 2 * rho.item() / (1 - WA / WC) if WA < WC else float('inf')
    return {'agg': agg, 'trust': trust, 'n_clean': n_clean, 'n_adv': n_adv, 'scale': scale,
            'shift': (adv - clean).norm().item(), 'bound': bound, 'WA/WC': WA / WC}


if __name__ == '__main__':
    rows = []
    for n_clean in [0, 5]:
        for n_adv in [1, 2, 5, 10, 20]:
            for scale in [10.0, 1000.0]:
                for agg in ['mean', 'gmed']:
                    for trust in [False, True]:
                        rows.append(run(n_clean, n_adv, scale, agg, trust))
    for r in rows:
        print('C=%(n_clean)2d A=%(n_adv)2d s=%(scale)6.0f %(agg)4s trust=%(trust)d  '
              'shift=%(shift)10.3f  bound=%(bound)10.3f  WA/WC=%(WA/WC).2f' % r)
    json.dump(rows, open('./Exps/bound_check.json', 'w'), indent=1)
