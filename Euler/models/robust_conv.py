'''
Robust neighborhood aggregation layers for the Euler GCN encoder.

All layers share one interface:  forward(x, ei, ew=None, trust=None)
  x     : N x F node features (or, via aggregate(), precomputed messages h = lin(x))
  ei    : 2 x E edge index, messages flow ei[0] -> ei[1] (same as GCNConv)
  ew    : E edge weights (Euler's standardized event counts), or None
  trust : E temporal trust weights in (0, 1], or None (= all ones)

Aggregators ("agg"):
  gcn    symmetric-normalized sum with self loops; numerically identical to
         torch_geometric GCNConv(add_self_loops=True) with edge_weight = ew * trust
  mean   weighted mean over N(i) + self loop
  guard  GNNGuard-style: reweight neighbors by cosine similarity, prune < 0.1
  gmed   weighted geometric median of the messages of N(i) + self loop, solved by
         Weiszfeld/IRLS iterations started at the node's own (self-loop) message, which
         the attacker cannot move; unrolled, so it is differentiable
  gmedn  GCN-scaled geometric median. GCN output factorizes as
             sum_j n_ij h_j = S_i * wmean_j(h_j; n_ij),   n_ij = a_ij / sqrt(d_i d_j),  S_i = sum_j n_ij
         gmedn keeps the GCN mass S_i (degree information the inner-product decoder relies
         on) and replaces only the location wmean by the n_ij-weighted geometric median.
'''
import torch
from torch import nn
from torch_scatter import scatter_add
from torch_geometric.utils import add_remaining_self_loops


def temporal_trust(counts, tau0, cap):
    '''
    Maps the number of earlier snapshots in which an edge was seen to a trust
    weight in [tau0, 1]: tau0 for a never-seen edge, 1 once seen >= cap times.
    '''
    if cap <= 0:
        return torch.ones_like(counts, dtype=torch.float)
    return tau0 + (1 - tau0) * torch.clamp(counts.float() / cap, max=1.0)


class RobustConv(nn.Module):
    def __init__(self, in_dim, out_dim, agg='gmed', iters=3, eps=1e-6, guard_thresh=0.1):
        super().__init__()
        self.agg = agg
        self.iters = iters
        self.eps = eps
        self.guard_thresh = guard_thresh
        self.lin = nn.Linear(in_dim, out_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_dim))
        nn.init.xavier_uniform_(self.lin.weight)  # same init as GCNConv

    def forward(self, x, ei, ew=None, trust=None):
        return self.aggregate(self.lin(x), ei, ew, trust)

    def aggregate(self, h, ei, ew=None, trust=None):
        n = h.size(0)
        if ew is None:
            ew = torch.ones(ei.size(1), device=h.device)
        a = ew if trust is None else ew * trust
        ei, a = add_remaining_self_loops(ei, a, fill_value=1.0, num_nodes=n)
        src, dst = ei

        if self.agg in ('gcn', 'gmedn'):
            deg = scatter_add(a, dst, dim=0, dim_size=n)
            dinv = deg.pow(-0.5)
            dinv[torch.isinf(dinv)] = 0
            norm = dinv[src] * a * dinv[dst]
            if self.agg == 'gcn':
                return scatter_add(norm.unsqueeze(1) * h[src], dst, dim=0, dim_size=n) + self.bias
            mass = scatter_add(norm, dst, dim=0, dim_size=n).unsqueeze(1)
            return mass * self._irls(h, src, dst, norm, n) + self.bias

        if self.agg == 'guard':
            hn = h / (h.norm(dim=1, keepdim=True) + self.eps)
            sim = (hn[src] * hn[dst]).sum(1).clamp(min=0)
            sim = torch.where(src == dst, torch.ones_like(sim), sim)
            sim = torch.where(sim < self.guard_thresh, torch.zeros_like(sim), sim)
            a = a * sim

        if self.agg == 'gmed':
            return self._irls(h, src, dst, a, n) + self.bias

        return self._wmean(h, src, dst, a, n) + self.bias

    def _irls(self, h, src, dst, a, n):
        # Weiszfeld / IRLS for the a-weighted geometric median, started at the self message:
        #   c_i <- sum_j (a_ij / d_ij) h_j / sum_j (a_ij / d_ij),  d_ij = ||h_j - c_i||
        # d is normalized by its per-node weighted mean so the smoothing is scale-free.
        out = h
        for _ in range(self.iters):
            d = (h[src] - out[dst]).norm(dim=1)
            dbar = scatter_add(a * d, dst, dim=0, dim_size=n) / \
                (scatter_add(a, dst, dim=0, dim_size=n) + self.eps)
            w = a / (d / (dbar[dst] + self.eps) + 1e-2)
            out = self._wmean(h, src, dst, w, n)
        return out

    def _wmean(self, h, src, dst, w, n):
        num = scatter_add(w.unsqueeze(1) * h[src], dst, dim=0, dim_size=n)
        den = scatter_add(w, dst, dim=0, dim_size=n).unsqueeze(1)
        return num / (den + self.eps)
