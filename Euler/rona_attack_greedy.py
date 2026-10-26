'''
Gradient-free covering-edge attack (check against gradient masking).

Same threat model, budget, candidate rule, edge weight and trust as rona_attack.py, but the
plan is chosen without gradients, by evaluating the exact deployed model:
  * greedy: the pool contains every edge from a controlled host into an endpoint of a
    malicious edge of window t or into an in-neighbour of such an endpoint (the edges that
    can reach the endpoint embeddings through the two message-passing layers). Edges are
    added one at a time, each time the one that lowers the attack loss most; after the
    first full pass only the 32 most promising candidates of the previous pass are
    re-evaluated (lazy greedy). Stops early when no candidate lowers the loss.
  * random search: --random subsets of size B drawn uniformly from the full candidate set
    (controlled host -> any host), plus random subsets of the pool.
The plan with the lowest loss in each window is kept; the GRU state is propagated through
the chosen plan, and the final plan is scored by the unmodified evaluation (rona_run.evaluate).

    python Euler/rona_attack_greedy.py --tag F_entente_gmedn_s0 --plugin_trust frozen --budgets 1 2
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
import spinup
from loaders import rona_hooks as RH
from models.euler_detector import DetectorEncoder
from models.robust_conv import temporal_trust

LAZY = 32


def greedy_attack(model, h0, times, s, loader, device, budget, base_counts, n_random=200, seed=0):
    g = torch.Generator().manual_seed(seed)
    RH.CFG['inject']['test'] = {}
    DetectorEncoder.load_new_data(model.gcns, loader, R.data_kwargs(times, 'test', 'All', s.mc))
    data = model.gcns.module.data
    n = data.num_nodes
    step = A.Stepper(model, device)
    tau0, cap, use_trust, inj_ew = RH.CFG['tau0'], RH.CFG['cap'], RH.CFG['trust'], RH.CFG['inject_ew']
    controlled, plan, log = set(), {}, []
    h = h0.to(device)

    def trust_of(src, dst):
        if not use_trust:
            return torch.ones(len(src), device=device)
        cnt = torch.tensor([base_counts.get(int(a) * n + int(b), 0) for a, b in zip(src, dst)])
        return temporal_trust(cnt, tau0, cap).float().to(device)

    for t in range(data.T):
        ei = data.eis[t].to(device)
        ew = data.ews[t].float().to(device)
        tr = data.trust[t].float().to(device)
        ys = data.ys[t] > 0
        mal = data.eis[t][:, ys]
        if mal.size(1) == 0:
            with torch.no_grad():
                _, h = step(ei, ew, tr, h)
            controlled |= set(mal.flatten().tolist())
            continue
        controlled |= set(mal[0].tolist())
        ctrl = sorted(controlled)
        existing = set((data.eis[t][0] * n + data.eis[t][1]).tolist())
        ends = set(mal.flatten().tolist())
        nb = set(data.eis[t][0][torch.isin(data.eis[t][1], torch.tensor(sorted(ends)))].tolist())
        targets = sorted(ends | nb)
        pool = [(a, b) for a in ctrl for b in targets if a != b and a * n + b not in existing]
        full = [(a, b) for a in ctrl for b in range(n) if a != b and a * n + b not in existing]
        B = int(budget * mal.size(1))
        mal_d = mal.to(device)

        def loss_of(edges):
            with torch.no_grad():
                if edges:
                    src = torch.tensor([e[0] for e in edges], device=device)
                    dst = torch.tensor([e[1] for e in edges], device=device)
                    ei2 = torch.cat([ei, torch.stack([src, dst])], 1)
                    ew2 = torch.cat([ew, torch.full((len(edges),), inj_ew, device=device)])
                    tr2 = torch.cat([tr, trust_of(src.tolist(), dst.tolist())])
                else:
                    ei2, ew2, tr2 = ei, ew, tr
                z, h2 = step(ei2, ew2, tr2, h)
                sc = (z[mal_d[0]] * z[mal_d[1]]).sum(1)
                return float(torch.nn.functional.softplus(-sc).mean()), h2, torch.sigmoid(sc)

        base_loss, _, p_clean = loss_of([])
        # greedy over the pool
        chosen, cur = [], base_loss
        rank = list(range(len(pool)))
        evals = 0
        for k in range(min(B, len(pool))):
            cands = [i for i in rank if pool[i] not in chosen]
            if k > 0:
                cands = cands[:LAZY]
            scores = []
            for i in cands:
                l, _, _ = loss_of(chosen + [pool[i]])
                scores.append((l, i))
                evals += 1
            if not scores:
                break
            scores.sort()
            if scores[0][0] >= cur - 1e-9:
                break
            cur = scores[0][0]
            chosen.append(pool[scores[0][1]])
            rank = [i for _, i in scores[1:]] + [i for i in rank if i not in {j for _, j in scores}]
        best, best_loss, best_src = chosen, cur, 'greedy'
        # random search over the full candidate set and over the pool
        for r in range(n_random):
            src_set = full if r % 2 == 0 else pool
            if not src_set:
                continue
            m = min(B, len(src_set))
            idx = torch.randperm(len(src_set), generator=g)[:m].tolist()
            edges = [src_set[i] for i in idx]
            l, _, _ = loss_of(edges)
            evals += 1
            if l < best_loss:
                best, best_loss, best_src = edges, l, 'random_full' if r % 2 == 0 else 'random_pool'
        _, h, p_adv = loss_of(best)
        plan[t] = np.array([[e[0] for e in best], [e[1] for e in best]], dtype=np.int64).reshape(2, -1)
        log.append({'t': t, 'n_mal': int(mal.size(1)), 'n_inj': len(best), 'pool': len(pool), 'evals': evals,
                    'winner': best_src, 'loss_clean': base_loss, 'loss_adv': best_loss,
                    'p_clean': float(p_clean.mean()), 'p_adv': float(p_adv.mean())})
    return plan, log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', required=True)
    ap.add_argument('--budgets', type=int, nargs='+', default=[1, 2])
    ap.add_argument('--plugin_trust', default=None, choices=['frozen', 'causal'])
    ap.add_argument('--inject_ew', type=float, default=None)
    ap.add_argument('--random', type=int, default=200)
    ap.add_argument('--cpu', type=int, default=1)
    ap.add_argument('--threads', type=int, default=8)
    ap.add_argument('--suffix', default='', help='extra suffix for the result json (keeps earlier budgets)')
    c = ap.parse_args()
    R.set_seed(0)
    torch.backends.cudnn.enabled = False
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
          ('_ew%g' % c.inject_ew if c.inject_ew is not None else '') + '_greedy'
    jsuf = suf + c.suffix
    results = {'clean': R.evaluate(model, h0, times, s, loader, device)}
    print('CLEAN', json.dumps(results['clean']), flush=True)
    for b in c.budgets:
        t0 = time.time()
        hval, _ = spinup.get_cutoff(model, h0, times, {'val_nratio': 1}, s.fpweight, False, device, 'All', s.mc)
        plan, log = greedy_attack(model, hval, times, s, loader, device, b, base_counts, n_random=c.random)
        torch.save(plan, os.path.join(out_dir, 'plan_b%d%s.pt' % (b, suf)))
        json.dump(log, open(os.path.join(out_dir, 'greedy_log_b%d%s.json' % (b, suf)), 'w'))
        res = R.evaluate(model, h0, times, s, loader, device, test_inject=plan)
        res.update(budget=b, attack_time=time.time() - t0,
                   pred_p_adv=float(np.mean([l['p_adv'] for l in log])),
                   pred_p_clean=float(np.mean([l['p_clean'] for l in log])),
                   winners={w: sum(l['winner'] == w for l in log) for w in ('greedy', 'random_full', 'random_pool')})
        results['b%d' % b] = res
        print('ATTACK', json.dumps(res), flush=True)
        json.dump(results, open(os.path.join(out_dir, 'attack%s.json' % jsuf), 'w'), indent=1)


if __name__ == '__main__':
    main()
