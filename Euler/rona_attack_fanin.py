'''
Fan-in attack on the degree scale S_i of gmedn.

gmedn outputs S_i * gmed(messages), with S_i = sum_j n_ij the GCN mass of node i. The influence
bound covers the location gmed; the scale grows with the injected weight that reaches i. This
attacker spends its whole budget on that scale: in every window with malicious edges it injects
up to b*|M_t| edges from controlled hosts directly INTO the endpoints of the malicious edges
(targets first, in order of increasing clean in-weight, round-robin across endpoints), choosing
for each endpoint the controlled hosts with the most interaction history first (highest trust).
No gradients are used. Same threat model, candidate rule (no self loops, no existing edges),
injected weight and trust as rona_attack.py; plans are scored by the unmodified evaluation.

    python Euler/rona_attack_fanin.py --tag F_entente_gmedn_s0 --plugin_trust frozen --budgets 1 10
'''
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rona_attack as A
import rona_run as R
from loaders import rona_hooks as RH
from models.euler_detector import DetectorEncoder


def fanin_plan(model, times, s, loader, budget, base_counts):
    RH.CFG['inject']['test'] = {}
    DetectorEncoder.load_new_data(model.gcns, loader, R.data_kwargs(times, 'test', 'All', s.mc))
    data = model.gcns.module.data
    n = data.num_nodes
    controlled, plan, log = set(), {}, []
    for t in range(data.T):
        ei, ew = data.eis[t], data.ews[t].float()
        mal = ei[:, data.ys[t] > 0]
        if mal.size(1) == 0:
            controlled |= set(mal.flatten().tolist())
            continue
        controlled |= set(mal[0].tolist())
        existing = set((ei[0] * n + ei[1]).tolist())
        inw = torch.zeros(n).index_add_(0, ei[1], ew)
        tgts = sorted(set(mal[1].tolist()), key=lambda v: (float(inw[v]), v))
        srcs = sorted(set(mal[0].tolist()) - set(tgts), key=lambda v: (float(inw[v]), v))
        queues = []
        for v in tgts + srcs:
            hs = [a for a in controlled if a != v and a * n + v not in existing]
            hs.sort(key=lambda a: (-base_counts.get(a * n + v, 0), a))
            queues.append([(a, v) for a in hs])
        B = int(budget * mal.size(1))
        chosen = []
        while len(chosen) < B and any(queues):
            for q in queues:
                if q and len(chosen) < B:
                    chosen.append(q.pop(0))
        plan[t] = np.array([[e[0] for e in chosen], [e[1] for e in chosen]], dtype=np.int64).reshape(2, -1)
        log.append({'t': t, 'n_mal': int(mal.size(1)), 'n_inj': len(chosen), 'n_ctrl': len(controlled)})
    return plan, log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', required=True)
    ap.add_argument('--budgets', type=int, nargs='+', default=[1, 10])
    ap.add_argument('--plugin_trust', default=None, choices=['frozen', 'causal'])
    ap.add_argument('--inject_ew', type=float, default=None)
    ap.add_argument('--cpu', type=int, default=1)
    ap.add_argument('--threads', type=int, default=4)
    c = ap.parse_args()
    R.set_seed(0)
    device = torch.device('cpu') if c.cpu else torch.device('cuda')
    if c.cpu:
        torch.set_num_threads(c.threads)
    a, s, times, loader, model, h0 = A.load_model(c.tag, device)
    if c.plugin_trust:
        RH.CFG.update(trust=True, trust_mode=c.plugin_trust)
    if c.inject_ew is not None:
        RH.CFG['inject_ew'] = c.inject_ew
    hist = RH.HISTORY['All']
    base_counts = hist['train'] + hist['val']
    out_dir = os.path.join(R.OUT_ROOT, c.tag)
    suf = ('_pt' + c.plugin_trust if c.plugin_trust else '') + \
          ('_ew%g' % c.inject_ew if c.inject_ew is not None else '') + '_fanin'
    results = {'clean': R.evaluate(model, h0, times, s, loader, device)}
    print('CLEAN', json.dumps(results['clean']), flush=True)
    for b in c.budgets:
        t0 = time.time()
        plan, log = fanin_plan(model, times, s, loader, b, base_counts)
        torch.save(plan, os.path.join(out_dir, 'plan_b%d%s.pt' % (b, suf)))
        res = R.evaluate(model, h0, times, s, loader, device, test_inject=plan)
        res.update(budget=b, attack_time=time.time() - t0,
                   n_inj_total=int(sum(l['n_inj'] for l in log)))
        results['b%d' % b] = res
        print('ATTACK', json.dumps(res), flush=True)
    json.dump(results, open(os.path.join(out_dir, 'attack%s.json' % suf), 'w'), indent=1)


if __name__ == '__main__':
    main()
