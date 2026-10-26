'''Layer-1 source attention scores of the GAT(n) checkpoints: GAT attention is "static"
(Brody et al.): for a fixed destination the neighbour ranking is set by a_src . W x_j alone.
Where do the hosts the attacker injects from rank?  Also the attention share that injected
edges receive in layer 1 relative to the GCN share (edge weights read from the replay npz are
not stored, so unit weights are assumed for this ratio).'''
import os, sys
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rona_run_b  # noqa
import rona_run as R
from rona_attack import load_model

for s in (0, 1, 2):
    tag = 'F_entente_gatn_s%d' % s
    a, st, times, loader, model, h0 = load_model(tag, torch.device('cpu'))
    enc = model.gcns.module
    x = enc.data.xs
    with torch.no_grad():
        h = enc.c1.lin(x)
        ssrc = (h * enc.c1.att_src).sum(1).numpy()
        sdst = (h * enc.c1.att_dst).sum(1).numpy()
    z = np.load(os.path.join(R.OUT_ROOT, tag, 'replay_plan_b1.npz'))
    inj = z['inj'] == 1
    U = np.unique(z['src'][inj])
    pct = lambda v: (ssrc < v).mean() * 100
    active = np.unique(np.concatenate([z['src'], z['dst']]))
    pa = lambda v: (ssrc[active] < v).mean() * 100
    print(tag, 'controlled hosts', len(U), 'source-score percentile among active hosts:',
          [round(pa(ssrc[u]), 1) for u in U])
    # share of attention vs uniform share at attacked destinations (layer 1, unit weights)
    t, src, dst = z['t'], z['src'], z['dst']
    ratios = []
    for (tt, v) in set(zip(t[inj].tolist(), dst[inj].tolist())):
        m = (t == tt) & (dst == v)
        nb = np.concatenate([src[m], [v]])
        isinj = np.concatenate([inj[m], [False]])
        e = sdst[v] + ssrc[nb]
        e = np.where(e > 0, e, 0.2 * e)
        w = np.exp(e - e.max()); w /= w.sum()
        ratios.append(w[isinj].sum() / (isinj.sum() / len(nb)))
    print('   attention share of injected edges / unweighted share: median %.3f, mean %.3f, n=%d'
          % (np.median(ratios), np.mean(ratios), len(ratios)))
