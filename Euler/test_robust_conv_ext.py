'''Unit tests for models/robust_conv_ext.py.  Run on CPU:
    python Euler/test_robust_conv_ext.py
'''
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import robust_conv as rc
import models.robust_conv_ext as X

NEW = list(X.NEW_AGGS)
OLD = ['gcn', 'mean', 'guard', 'gmed', 'gmedn']
Orig = rc._OriginalRobustConv
Ext = X.RobustConvExt
fails = []


def check(name, ok, info=''):
    print(('PASS ' if ok else 'FAIL ') + name + ' ' + info)
    if not ok:
        fails.append(name)


def rand_graph(n=30, e=150, seed=0):
    g = torch.Generator().manual_seed(seed)
    ei = torch.randint(0, n, (2, e), generator=g)
    ei = ei[:, ei[0] != ei[1]]
    ew = torch.rand(ei.size(1), generator=g) + 0.2
    tr = torch.rand(ei.size(1), generator=g) * 0.9 + 0.1
    return ei, ew, tr


def make(cls, agg, i=8, o=6, seed=1):
    torch.manual_seed(seed)
    m = cls(i, o, agg=agg, iters=3)
    with torch.no_grad():
        m.bias.normal_()
    return m


# (ii) existing aggregators unchanged (bit-identical)
n = 30
ei, ew, tr = rand_graph(n)
x = torch.randn(n, 8)
check('patched', rc.RobustConv is Ext)
for agg in OLD:
    a, b = make(Orig, agg)(x, ei, ew, tr), make(Ext, agg)(x, ei, ew, tr)
    check('unchanged_' + agg, torch.equal(a, b))

# (i) equals GCN when all messages in each neighbourhood are identical
xc = torch.ones(n, 8) * torch.randn(1, 8)
gcn = make(Orig, 'gcn')(xc, ei, ew, tr)
for agg in NEW:
    out = make(Ext, agg)(xc, ei, ew, tr)
    err = (out - gcn).abs().max().item()
    check('gcn_equiv_' + agg, err < 1e-4, 'maxerr=%.2e' % err)

# non-degenerate: exact and surrogate agree on equal weights, different values
for agg in ('median', 'trmean'):
    m = make(Ext, agg)
    o1 = m(x, ei, torch.ones_like(ew))
    X.ATTACK_SURROGATE = True
    o2 = m(x, ei, torch.ones_like(ew))
    X.ATTACK_SURROGATE = False
    print('  info: %s exact-vs-surrogate mean abs diff %.3f (output scale %.3f)' %
          (agg, (o1 - o2).abs().mean(), o1.abs().mean()))

# (iii) gradient wrt ew finite & non-zero
for agg in NEW:
    for sur in ([False, True] if agg in ('median', 'trmean') else [False]):
        X.ATTACK_SURROGATE = sur
        m = make(Ext, agg)
        w = ew.clone().requires_grad_(True)
        # also an edge subset at weight exactly 0 (the PGD starting point m=0)
        w0 = w.detach().clone()
        w0[:40] = 0
        w0.requires_grad_(True)
        for nm, ww in (('w', w), ('w0', w0)):
            out = m(x, ei, ww, tr)
            loss = (out * torch.randn(out.shape, generator=torch.Generator().manual_seed(3))).sum()
            gr, = torch.autograd.grad(loss, ww)
            ok = bool(torch.isfinite(gr).all()) and gr.abs().sum().item() > 0
            nz0 = int((gr[:40] != 0).sum()) if nm == 'w0' else -1
            check('grad_%s%s_%s' % (agg, '_surr' if sur else '', nm), ok,
                  '|g|_1=%.3g%s' % (gr.abs().sum(), (' nonzero_at_w=0:%d/40' % nz0) if nz0 >= 0 else ''))
        X.ATTACK_SURROGATE = False
for agg in ('median', 'trmean'):  # exact version: expected (near) zero gradient
    m = make(Ext, agg)
    w = ew.clone().requires_grad_(True)
    out = m(x, ei, w, tr)
    gr, = torch.autograd.grad(out.sum(), w)
    print('  info: exact %s |grad ew|_1=%.3g (only via the GCN mass S_i; no information from the selection)' % (agg, gr.abs().sum()))

# (iv) outlier robustness: k huge-norm neighbours added to node 0
n2 = 40
g = torch.Generator().manual_seed(5)
ei2 = torch.randint(0, n2, (2, 200), generator=g)
ei2 = ei2[:, (ei2[0] != ei2[1]) & (ei2[1] != 0)]
# node 0 gets 8 benign in-neighbours
ben = torch.arange(1, 9)
ei_b = torch.cat([ei2, torch.stack([ben, torch.zeros_like(ben)])], 1)
x2 = torch.randn(n2, 8)
out_idx = torch.arange(n2 - 3, n2)  # three attacker nodes
for K in (2,):
    out_nodes = torch.arange(n2 - K, n2)
    x_att = x2.clone()
    x_att[out_nodes] = 1000.0 * torch.randn(K, 8)
    ei_a = torch.cat([ei_b, torch.stack([out_nodes, torch.zeros_like(out_nodes)])], 1)
    res = {}
    for agg in ['gcn'] + NEW + ['gmedn']:
        m = make(Ext, agg)
        with torch.no_grad():
            o0 = m(x2, ei_b)[0]
            o1 = m(x_att, ei_a)[0]
        res[agg] = ((o1 - o0).norm() / (o0.norm() + 1e-9)).item()
    print('  info: relative shift of node 0 with %d outlier neighbours (|x|~1000): %s' %
          (K, {k: round(v, 2) for k, v in res.items()}))
    for agg in NEW:
        if agg == 'guardn':
            continue
        check('outlier_%s' % agg, res[agg] < 0.05 * res['gcn'], 'shift=%.2f vs gcn=%.2f' % (res[agg], res['gcn']))
    # guardn filters by cosine similarity only: it removes outliers pointing away from the
    # node's own message (cos < 0.1) but, by design, not large-norm outliers pointing the same way.
    m = make(Ext, 'guardn')
    lin0 = m.lin(x2)[0]
    for _ in range(200):
        cand = torch.randn(K, 8)
        if ((m.lin(cand) * lin0).sum(1) < 0).all():
            break
    x_opp = x2.clone()
    x_opp[out_nodes] = 1000.0 * cand
    with torch.no_grad():
        o0 = m(x2, ei_b)[0]
        o1 = m(x_opp, ei_a)[0]
        g0 = make(Ext, 'gcn')(x2, ei_b)[0]
        g1 = make(Ext, 'gcn')(x_opp, ei_a)[0]
    sh, shg = ((o1 - o0).norm() / o0.norm()).item(), ((g1 - g0).norm() / g0.norm()).item()
    check('outlier_guardn_antialigned', sh < 0.05 * shg, 'shift=%.2f vs gcn=%.2f' % (sh, shg))
    print('  info: guardn (same-direction large-norm outlier, shift %.1f) is NOT robust: cosine ignores magnitude' % res['guardn'])

# (v) Stepper from rona_attack on a synthetic graph: gradient w.r.t. the injected-edge mask
try:
    import types
    import rona_attack as A
    from models.recurrent import GRU
    n3, F_in, hdim = 25, 25, 8
    ei3, ew3, tr3 = rand_graph(n3, 120, seed=7)
    cand = torch.stack([torch.randint(0, 5, (60,)), torch.randint(5, n3, (60,))])
    ei_all = torch.cat([ei3, cand], 1)
    tr_all = torch.cat([tr3, torch.rand(60) * 0.9 + 0.1])
    for agg in NEW:
        torch.manual_seed(0)
        enc = types.SimpleNamespace(c1=Ext(F_in, hdim, agg=agg), c2=Ext(hdim, hdim, agg=agg),
                                    relu=torch.nn.ReLU(), data=types.SimpleNamespace(xs=torch.eye(n3)))
        gru = GRU(hdim, hdim, 4)
        st = A.Stepper(types.SimpleNamespace(gcns=types.SimpleNamespace(module=enc), rnn=gru), 'cpu')
        X.ATTACK_SURROGATE = True
        m = torch.zeros(60, requires_grad=True)
        z, _ = st(ei_all, torch.cat([ew3, 0.5 * m]), tr_all, None)
        loss = torch.nn.functional.softplus(-(z[cand[0]] * z[cand[1]]).sum(1)).mean()
        gr, = torch.autograd.grad(loss, m)
        X.ATTACK_SURROGATE = False
        check('stepper_grad_' + agg, bool(torch.isfinite(gr).all()) and gr.abs().sum().item() > 0,
              '|g|_1=%.3g' % gr.abs().sum())
except Exception as e:  # report, do not hide
    import traceback
    traceback.print_exc()
    check('stepper', False, repr(e))

print('\nFAILED: %s' % fails if fails else '\nALL TESTS PASSED')
sys.exit(1 if fails else 0)
