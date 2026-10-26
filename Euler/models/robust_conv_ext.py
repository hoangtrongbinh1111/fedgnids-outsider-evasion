'''
Baseline robust aggregators for the RoNA comparison (NEW file; models/robust_conv.py is untouched).

Importing this module monkeypatches  models.robust_conv.RobustConv  with RobustConvExt, a
subclass that handles the extra aggregator names below and delegates every other name
(gcn, mean, guard, gmed, gmedn) to the original implementation (bit-identical results).
embedders.RGCN imports RobustConv lazily at construction time, so it picks up the subclass.

All new aggregators use the same edge weights a = ew * trust (+ remaining self loops of
weight 1), the GCN normalization n_ij = a_ij / sqrt(d_i d_j), and are scaled by the GCN
mass S_i = sum_j n_ij:   out_i = S_i * c_i + bias,  where c_i is a location estimate of the
messages {h_j} with weights n_ij. Hence every aggregator equals GCN when all messages of a
neighbourhood are identical (c_i = h).  How each differs from the published method:

  softmed  Soft Median (Geisler et al., NeurIPS'21), a port of rgnn_at_scale/aggregation.py
           `soft_median`: n_ij-weighted dimension-wise median m_i (no grad), distances
           ||m_i - h_j||_2 / sqrt(d), softmax(-dist/T) over the neighbourhood, multiplied by
           n_ij and renormalized, then multiplied by sum_j n_ij.  NOTE: the official code
           ALREADY rescales by the sum of the (normalized) edge weights (`weight_sums`), so
           the S_i scaling is not a difference here.  T=1.0 is the code default.
           Deviation: official code needs a custom CUDA kernel for the median; we use a
           pure-torch segment sort (same semantics: first element with cum. weight >= half).
  rung     RUNG (Hou et al., NeurIPS'24, github chris-hzc/RUNG model/rung.py): MCP-penalized
           estimator, quasi-Newton IRLS. Original: decoupled model (MLP, then K=10 dense
           propagation layers on a node-classification graph). Adaptation: one RUNG block
           (K=`iters` QN-IRLS steps, F0 = the layer's transformed messages h) replaces the
           neighbourhood sum of each GCN layer. Per step
               y_ij = || F_i/sqrt(d_i) - F_j/sqrt(d_j) ||_2   (d_i = weighted degree incl. self loop)
               W_ij = clamp(1/(2y_ij) - 1/(2*gamma), 0, 1e3) * [i != j]          (MCP)
               F_i <- ( sum_j W_ij n_ij F_j + lam S_i F0_i ) / ( sum_j W_ij n_ij + lam S_i )
           lam = 1/lam_hat - 1 (lam_hat=0.9 as in the repo). Deviations: (a) the denominator
           uses n_ij-mass (original: sum_j W_ij A_ij / d_i + lam), so that equal messages
           give F = F0 and the output S_i*F reduces to GCN; (b) y uses sqrt(y^2+0.01) like
           the repo's att_func; (c) the repo detaches W (Z.detach()); we keep it
           differentiable so the white-box PGD attacker sees the full gradient (adaptive);
           (d) gamma=3 default (repo script default is 36 for Cora features; not tuned here).
  rungl1   same with the l1 penalty: W_ij = 1/(2 y_ij) (no pruning).
  median   n_ij-weighted coordinate-wise median (first element whose cumulative weight
           reaches half the total, as in the official weighted_dimwise_median) times S_i.
  trmean   coordinate-wise trimmed mean: per node, drop the m lowest and m highest
           messages per coordinate (by count, among edges with a_ij > 0), m = max(1, floor(0.2 k))
           if k >= 3 else 0; n_ij-weighted mean of the rest, times S_i.
  gat      single-head GAT attention (Velickovic et al., ICLR'18) with edge weights a_ij entering
           the softmax multiplicatively; outputs sum to one over the neighbourhood (no S_i).
  gatn     the same attention as location, times the GCN mass S_i (keeps the degree scale).
  guardn   GNNGuard-style: cosine similarity of the messages (clamped at 0, self = 1), prune
           sim < 0.1, a' = a * sim, then plain GCN normalization/aggregation on a'.
           Differs from GNNGuard, which row-normalizes the similarities and uses a learned
           memory/self-loop weight; here the original `guard` similarity step is reused.

Attack surrogates (module flag ATTACK_SURROGATE): median and trmean are piecewise constant
in the edge weights (zero PGD gradient). When ATTACK_SURROGATE is True they use a smooth
relaxation in the normalized cumulative weight mass c in [0,1] of the value-sorted
neighbours: median -> sigmoid((c_hi-.5)/tau) - sigmoid((c_lo-.5)/tau) soft selection;
trmean -> exact overlap of each neighbour's mass interval with [alpha, 1-alpha]
(a weighted quantile-trimmed mean). Both are differentiable in the edge weights, and
agree with the exact aggregators when weights are equal (up to rank/mass discretization).
Final evaluation must run with ATTACK_SURROGATE=False.
'''
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.utils import softmax as seg_softmax
from torch_scatter import scatter_add, scatter_max
from torch_geometric.utils import add_remaining_self_loops

from models import robust_conv as _rc

ATTACK_SURROGATE = False

CFG = dict(softmed_T=1.0, trim=0.2, rung_gamma=3.0, rung_lam_hat=0.9, rung_eps=0.01,
           rung_max_w=1e3, surr_tau=0.1, col_chunk=4)

NEW_AGGS = ('softmed', 'rung', 'rungl1', 'median', 'trmean', 'guardn', 'gat', 'gatn', 'gmednm', 'gcnanc')
# mechanism controls for gmedn: its K=3 IRLS starts at the node's own message, whose
# self-loop then gets ~100x weight in the first step (distance 0, smoothing 1e-2).
#   gmednm  same geometric-median IRLS, started at the n-weighted mean instead (median, no anchor)
#   gcnanc  n-weighted mean with the self-loop weight x100 (the anchor, no distance-based reweighting)
ANCHOR = 100.0
_Original = _rc.RobustConv if not hasattr(_rc, '_OriginalRobustConv') else _rc._OriginalRobustConv


def _seg_sort(v, dst, n):
    '''Per column of v (E x d): order of the rows sorted by (dst, value). Returns
    order (E x d, indices into rows), seg (E, node id of sorted row r, identical for all
    columns) and ptr (n+1, segment starts).'''
    _, i1 = torch.sort(v, dim=0)
    _, i2 = torch.sort(dst[i1], dim=0, stable=True)
    order = i1.gather(0, i2)
    counts = torch.bincount(dst, minlength=n)
    ptr = torch.cat([counts.new_zeros(1), counts.cumsum(0)])
    seg = torch.repeat_interleave(torch.arange(n, device=dst.device), counts)
    return order, seg, ptr


def _seg_cumsum(ws, seg, ptr):
    '''Segment-wise cumulative sum (float64) of ws (E x d, rows sorted by segment).'''
    cum = ws.double().cumsum(0)
    zero = cum.new_zeros(1, cum.size(1))
    base = torch.cat([zero, cum])[ptr[:-1]]            # n x d
    cin = cum - base[seg]
    tot = cin[(ptr[1:] - 1).clamp(min=0)]              # n x d (same in all columns)
    return cin, tot


class RobustConvExt(_Original):
    def __init__(self, in_dim, out_dim, agg='gmed', iters=3, eps=1e-6, guard_thresh=0.1):
        super().__init__(in_dim, out_dim, agg=agg, iters=iters, eps=eps, guard_thresh=guard_thresh)
        if agg in ('gat', 'gatn'):
            # single-head GAT attention vectors (Velickovic et al., ICLR'18), Glorot init as in PyG
            self.att_src = nn.Parameter(torch.empty(1, out_dim))
            self.att_dst = nn.Parameter(torch.empty(1, out_dim))
            nn.init.xavier_uniform_(self.att_src)
            nn.init.xavier_uniform_(self.att_dst)

    def aggregate(self, h, ei, ew=None, trust=None):
        if self.agg not in NEW_AGGS:
            return super().aggregate(h, ei, ew, trust)
        n = h.size(0)
        if ew is None:
            ew = torch.ones(ei.size(1), device=h.device)
        a = ew if trust is None else ew * trust
        ei, a = add_remaining_self_loops(ei, a, fill_value=1.0, num_nodes=n)
        src, dst = ei

        if self.agg == 'guardn':
            hn = h / (h.norm(dim=1, keepdim=True) + self.eps)
            sim = (hn[src] * hn[dst]).sum(1).clamp(min=0)
            sim = torch.where(src == dst, torch.ones_like(sim), sim)
            sim = torch.where(sim < self.guard_thresh, torch.zeros_like(sim), sim)
            a = a * sim

        deg = scatter_add(a, dst, dim=0, dim_size=n)
        dinv = deg.pow(-0.5)
        dinv[torch.isinf(dinv)] = 0
        norm = dinv[src] * a * dinv[dst]
        if self.agg == 'guardn':
            return scatter_add(norm.unsqueeze(1) * h[src], dst, dim=0, dim_size=n) + self.bias
        mass = scatter_add(norm, dst, dim=0, dim_size=n).unsqueeze(1)

        if self.agg in ('gat', 'gatn'):
            # attention over N(i) + self loop; edge weight (and trust) enter multiplicatively:
            # alpha_ij = a_ij exp(e_ij) / sum_k a_ik exp(e_ik)
            e = F.leaky_relu((h[src] * self.att_src).sum(1) + (h[dst] * self.att_dst).sum(1), 0.2)
            alpha = seg_softmax(e + torch.log(a.clamp(min=1e-12)), dst, num_nodes=n)
            loc = scatter_add(alpha.unsqueeze(1) * h[src], dst, dim=0, dim_size=n)
            # gat: plain GAT (attention weights sum to one); gatn: scaled by the GCN mass S_i
            # like the other aggregators, so it keeps the degree scale the decoder relies on
            return (loc if self.agg == 'gat' else mass * loc) + self.bias

        if self.agg == 'softmed':
            c = self._softmed(h, src, dst, norm, mass, n)
            return c + self.bias                                  # already scaled by mass
        if self.agg == 'gmednm':
            out = self._wmean(h, src, dst, norm, n)
            for _ in range(self.iters):
                d = (h[src] - out[dst]).norm(dim=1)
                dbar = scatter_add(norm * d, dst, dim=0, dim_size=n) / \
                    (scatter_add(norm, dst, dim=0, dim_size=n) + self.eps)
                out = self._wmean(h, src, dst, norm / (d / (dbar[dst] + self.eps) + 1e-2), n)
            return mass * out + self.bias
        if self.agg == 'gcnanc':
            w = torch.where(src == dst, norm * ANCHOR, norm)
            return mass * self._wmean(h, src, dst, w, n) + self.bias
        if self.agg in ('rung', 'rungl1'):
            loc = self._rung(h, src, dst, norm, mass, dinv, n)
        elif self.agg == 'median':
            loc = self._median(h, src, dst, norm, n)
        else:
            loc = self._trmean(h, src, dst, norm, n)
        return mass * loc + self.bias

    # ---------------------------------------------------------------- Soft Median
    def _softmed(self, h, src, dst, w, mass, n):
        d = h.size(1)
        with torch.no_grad():
            xmed = self._chunked(self._median_select, h, src, dst, w, n)
        dist = (xmed[dst] - h[src]).norm(dim=1) / d ** 0.5       # p=2, /d^(1/p)
        z = -dist / CFG['softmed_T']
        zmax = scatter_max(z.detach(), dst, dim=0, dim_size=n)[0]
        e = torch.exp(z - zmax[dst])
        sw = e / (scatter_add(e, dst, dim=0, dim_size=n)[dst] + 1e-12)
        wv = sw * w
        fin = wv / (scatter_add(wv, dst, dim=0, dim_size=n)[dst] + 1e-12) * mass[dst, 0]
        return scatter_add(fin.unsqueeze(1) * h[src], dst, dim=0, dim_size=n)

    # ----------------------------------------------------------------- RUNG
    def _rung(self, h, src, dst, norm, mass, dinv, n):
        lam = 1.0 / CFG['rung_lam_hat'] - 1.0
        gam, eps = CFG['rung_gamma'], CFG['rung_eps']
        notself = (src != dst).float()
        base = lam * mass
        f = h
        for _ in range(self.iters):
            fs = f * dinv.unsqueeze(1)
            y = ((fs[src] - fs[dst]).pow(2).sum(1) + eps).sqrt()
            if self.agg == 'rungl1':
                wgt = 1.0 / (2 * y)
            else:
                wgt = (1.0 / (2 * y) - 1.0 / (2 * gam)).clamp(min=0, max=CFG['rung_max_w'])
            wn = (wgt * notself * norm)
            num = scatter_add(wn.unsqueeze(1) * f[src], dst, dim=0, dim_size=n) + base * h
            den = scatter_add(wn, dst, dim=0, dim_size=n).unsqueeze(1) + base
            f = num / den.clamp(min=1e-12)
        return f

    # --------------------------------------------------------------- medians
    @staticmethod
    def _chunked(fn, h, src, dst, w, n):
        '''Applies fn column-chunk-wise: the sort-based estimators need several E x d_chunk
        index/cumsum tensors, chunking bounds the transient peak on the 1M+ edge OpTC graph.
        (No inner checkpointing: reentrant checkpoint breaks torch.autograd.grad, which the
        attack uses.)'''
        k = CFG['col_chunk']
        return torch.cat([fn(h[:, c:c + k], src, dst, w, n) for c in range(0, h.size(1), k)], 1)

    def _median_select(self, h, src, dst, w, n):
        '''Exact weighted coordinate-wise median (values of actual neighbours), n x d.
        The selection is non-differentiable; the returned values are gathered from h.'''
        with torch.no_grad():
            v = h[src]
            order, seg, ptr = _seg_sort(v, dst, n)
            cin, tot = _seg_cumsum(w.detach()[order], seg, ptr)
            below = (cin < (tot / 2)[seg]).long()
            cnt = scatter_add(below, seg, dim=0, dim_size=n)
            pos = torch.min(ptr[:-1].unsqueeze(1) + cnt, (ptr[1:] - 1).unsqueeze(1))
            sel = src[order.gather(0, pos)]                    # n x d source node ids
        return h.gather(0, sel)

    def _median(self, h, src, dst, w, n):
        return self._chunked(self._median_impl, h, src, dst, w, n)

    def _trmean(self, h, src, dst, w, n):
        return self._chunked(self._trmean_impl, h, src, dst, w, n)

    def _median_impl(self, h, src, dst, w, n):
        if not ATTACK_SURROGATE:
            return self._median_select(h, src, dst, w, n)
        v = h[src]
        with torch.no_grad():
            order, seg, ptr = _seg_sort(v, dst, n)
        ws = w[order]
        cin, tot = _seg_cumsum(ws, seg, ptr)
        t = tot[seg].clamp(min=1e-12)
        c_hi = cin / t
        c_lo = c_hi - ws.double() / t
        tau = CFG['surr_tau']
        s = (torch.sigmoid((c_hi - 0.5) / tau) - torch.sigmoid((c_lo - 0.5) / tau)).to(h.dtype)
        return self._soft_combine(v.gather(0, order), s, seg, n)

    def _trmean_impl(self, h, src, dst, w, n):
        k = scatter_add((w > 0).long(), dst, dim=0, dim_size=n)       # valid neighbours
        m = torch.where(k >= 3, (CFG['trim'] * k.float()).floor().clamp(min=1).long(), torch.zeros_like(k))
        v = h[src]
        with torch.no_grad():
            order, seg, ptr = _seg_sort(v, dst, n)
        ws = w[order]
        if not ATTACK_SURROGATE:
            valid = (ws > 0).long()
            cv = valid.cumsum(0)
            basev = torch.cat([cv.new_zeros(1, cv.size(1)), cv])[ptr[:-1]]
            rank = cv - basev[seg] - 1
            keep = (valid > 0) & (rank >= m[seg].unsqueeze(1)) & (rank < (k - m)[seg].unsqueeze(1))
            return self._soft_combine(v.gather(0, order), ws * keep.to(ws.dtype), seg, n)
        cin, tot = _seg_cumsum(ws, seg, ptr)
        t = tot[seg].clamp(min=1e-12)
        c_hi, c_lo = cin / t, (cin - ws.double()) / t
        alpha = (m.double() / k.clamp(min=1).double())[seg].unsqueeze(1)
        ov = (torch.min(c_hi, 1 - alpha) - torch.max(c_lo, alpha)).clamp(min=0)
        return self._soft_combine(v.gather(0, order), (ov * t).to(h.dtype), seg, n)

    @staticmethod
    def _soft_combine(vs, e, seg, n):
        num = scatter_add(e * vs, seg, dim=0, dim_size=n)
        den = scatter_add(e, seg, dim=0, dim_size=n)
        return num / den.clamp(min=1e-12)


def install():
    if getattr(_rc.RobustConv, '__name__', '') != 'RobustConvExt':
        _rc._OriginalRobustConv = _rc.RobustConv
        _rc.RobustConv = RobustConvExt


install()
