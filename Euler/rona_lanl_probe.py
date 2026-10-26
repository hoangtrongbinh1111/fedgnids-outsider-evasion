'''
LANL covering-edge diagnosis + credible-relaxation probe (CPU, memory-frugal).

New file; changes no existing default behaviour. It loads a trained LANL model (default
L_entente_gcn_s0) exactly as rona_attack_lanl does, then runs the white-box covering-edge
PGD ONLY on the first `--max_mal_windows` windows that contain a malicious edge (the GRU
hidden state is still propagated, without gradients, through every preceding window, so the
score at each attacked window is exact). For each window it reports the clean malicious-edge
score and the best adversarial score the attacker can reach, under several threat models:

  --control src   : covering-edge SOURCES = sources of malicious edges seen so far
                    (== the paper's current outsider threat model)
  --control both  : covering-edge SOURCES = BOTH endpoints of malicious edges seen so far
                    (lateral-movement relaxation: once the attacker has authenticated INTO a
                     server it controls that host and can originate covering flows from it;
                     this matches Xu et al. USENIX'23 / Zhao & Hassan CCS'26 "a host the
                     adversary already controls")
  --control any   : covering-edge SOURCES = any host  (Xu et al. "strong capability" upper bound)

  --dst all       : covering edges may go to any node (the real attack's candidate set)
  --dst endpoints : covering edges may only TERMINATE at an endpoint of a current malicious
                    edge (concentrates the budget as INCOMING edges on the scored nodes, which
                    is the only 1-hop way to move z[u] or z[v] of the scored pair u->v)

Reads only cached TData (no GPU). Prints per-window and aggregate predicted scores and the
fraction of malicious edges pushed above the model cutoff.

    python Euler/rona_lanl_probe.py --tag L_entente_gcn_s0 --control both --dst endpoints \
        --budgets 5 10 --max_mal_windows 6 --cpu 1 --threads 6
'''
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rona_lanl
rona_lanl.patch()
import rona_run as R
import spinup
from loaders import rona_hooks as RH
from models.robust_conv import temporal_trust
from rona_attack import load_model, Stepper, project_budget
from models.euler_detector import DetectorEncoder


def controlled_set(mal_src, mal_dst, mode):
    if mode == 'src':
        return set(mal_src.tolist())
    if mode == 'both':
        return set(mal_src.tolist()) | set(mal_dst.tolist())
    return None  # 'any'


def run(model, hval, times, s, loader, device, budget, mode, dst_mode, max_mw,
        steps, samples, seed, cutoff):
    g = torch.Generator(device='cpu').manual_seed(seed)
    RH.CFG['inject']['test'] = {}
    DetectorEncoder.load_new_data(model.gcns, loader, R.data_kwargs(times, 'test', 'All', s.mc))
    data = model.gcns.module.data
    n = data.num_nodes
    step = Stepper(model, device)
    inj_ew = RH.CFG['inject_ew']
    controlled = set()
    h = hval.to(device)
    out = []
    done = 0
    with torch.no_grad():
        pass
    for t in range(data.T):
        ei = data.eis[t].to(device)
        ew = data.ews[t].float().to(device)
        tr = data.trust[t].float().to(device)
        ys = data.ys[t] > 0
        mal = data.eis[t][:, ys]
        if mal.size(1) == 0 or done >= max_mw:
            with torch.no_grad():
                _, h = step(ei, ew, tr, h)
            if mal.size(1):
                controlled |= set(mal.flatten().tolist())
            continue
        # controlled set BEFORE this window acts (attacker already pivoted here)
        ctrl_now = set(controlled)
        ctrl_now |= set(mal[0].tolist()) if mode == 'src' else set(mal.flatten().tolist())
        endpoints = sorted(set(mal.flatten().tolist()))
        # candidate covering edges
        if mode == 'any':
            src_pool = list(range(n))
        else:
            src_pool = sorted(ctrl_now)
        if dst_mode == 'endpoints':
            dst_pool = endpoints
        else:
            dst_pool = list(range(n))
        src_pool_t = torch.tensor(src_pool)
        dst_pool_t = torch.tensor(dst_pool)
        src = src_pool_t.repeat_interleave(len(dst_pool_t))
        dst = dst_pool_t.repeat(len(src_pool_t))
        existing = set((data.eis[t][0] * n + data.eis[t][1]).tolist())
        key = (src.long() * n + dst.long()).tolist()
        ok = torch.tensor([(int(a) != int(b)) and (k not in existing)
                           for a, b, k in zip(src.tolist(), dst.tolist(), key)])
        cand = torch.stack([src[ok], dst[ok]]).to(device)
        if cand.size(1) == 0:
            with torch.no_grad():
                _, h = step(ei, ew, tr, h)
            controlled |= set(mal.flatten().tolist())
            continue
        ctr = torch.ones(cand.size(1), device=device)  # GCN model: trust == 1
        B = float(budget * mal.size(1))
        mal_d = mal.to(device)
        ei_all = torch.cat([ei, cand], 1)
        tr_all = torch.cat([tr, ctr])

        def loss_of(mask):
            z, _ = step(ei_all, torch.cat([ew, inj_ew * mask]), tr_all, h)
            sc = (z[mal_d[0]] * z[mal_d[1]]).sum(1)
            return torch.nn.functional.softplus(-sc).mean(), torch.sigmoid(sc)

        m = torch.zeros(cand.size(1), device=device)
        for _ in range(steps):
            m.requires_grad_(True)
            loss, _ = loss_of(m)
            gr, = torch.autograd.grad(loss, m)
            with torch.no_grad():
                m = project_budget(m - 0.2 * gr / (gr.abs().max() + 1e-12), B)
        with torch.no_grad():
            mc = m.cpu()
            trials = [torch.zeros_like(mc).scatter_(0, torch.topk(mc, min(int(B), mc.numel())).indices, 1.0)]
            for _ in range(samples):
                bern = torch.bernoulli(mc, generator=g)
                if bern.sum() > B:
                    keep = torch.topk(mc * bern, int(B)).indices
                    bern = torch.zeros_like(mc).scatter_(0, keep, 1.0)
                trials.append(bern)
            best_m, best = None, float('inf')
            for bm in trials:
                l, _ = loss_of(bm.to(device))
                if l.item() < best:
                    best, best_m = l.item(), bm
            _, p_clean = loss_of(torch.zeros_like(m))
            _, p_adv = loss_of(best_m.to(device))
            _, h = step(ei_all, torch.cat([ew, inj_ew * best_m.to(device)]), tr_all, h)
        out.append({'t': t, 'n_mal': int(mal.size(1)), 'n_cand': int(cand.size(1)),
                    'n_inj': int(best_m.sum()), 'budget_edges': int(B),
                    'p_clean_mean': float(p_clean.mean()), 'p_adv_mean': float(p_adv.mean()),
                    'p_clean_max': float(p_clean.max()), 'p_adv_max': float(p_adv.max()),
                    'evaded_clean': int((p_clean > cutoff).sum()), 'evaded_adv': int((p_adv > cutoff).sum())})
        controlled |= set(mal.flatten().tolist())
        done += 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', default='L_entente_gcn_s0')
    ap.add_argument('--control', choices=['src', 'both', 'any'], default='both')
    ap.add_argument('--dst', choices=['all', 'endpoints'], default='endpoints')
    ap.add_argument('--budgets', type=int, nargs='+', default=[5, 10])
    ap.add_argument('--max_mal_windows', type=int, default=6)
    ap.add_argument('--steps', type=int, default=60)
    ap.add_argument('--samples', type=int, default=20)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--cpu', type=int, default=1)
    ap.add_argument('--threads', type=int, default=6)
    ap.add_argument('--suffix', default='')
    c = ap.parse_args()
    R.set_seed(c.seed)
    torch.backends.cudnn.enabled = False
    device = torch.device('cpu') if c.cpu else torch.device('cuda')
    if c.cpu:
        torch.set_num_threads(c.threads)
    tag = c.tag if c.tag.startswith('L_') else 'L_' + c.tag
    a, s, times, loader, model, h0 = load_model(tag, device)
    cutoff = float(model.cutoff)
    hval, _ = spinup.get_cutoff(model, h0, times, {'val_nratio': 1}, s.fpweight, False, device, 'All', s.mc)
    print('CUTOFF', cutoff, flush=True)
    results = {'tag': tag, 'control': c.control, 'dst': c.dst, 'cutoff': cutoff, 'budgets': {}}
    for b in c.budgets:
        t0 = time.time()
        log = run(model, hval, times, s, loader, device, b, c.control, c.dst,
                  c.max_mal_windows, c.steps, c.samples, c.seed, cutoff)
        agg = {
            'windows': len(log),
            'mal_edges': int(sum(l['n_mal'] for l in log)),
            'p_clean_mean': float(np.mean([l['p_clean_mean'] for l in log])) if log else None,
            'p_adv_mean': float(np.mean([l['p_adv_mean'] for l in log])) if log else None,
            'evaded_clean': int(sum(l['evaded_clean'] for l in log)),
            'evaded_adv': int(sum(l['evaded_adv'] for l in log)),
            'mean_cand': float(np.mean([l['n_cand'] for l in log])) if log else None,
            'time_s': time.time() - t0,
            'per_window': log,
        }
        results['budgets']['b%d' % b] = agg
        print('B%d %s' % (b, json.dumps({k: v for k, v in agg.items() if k != 'per_window'})), flush=True)
    out = os.path.join('Exps', 'lanl_probe_%s_%s_%s%s.json' % (tag, c.control, c.dst, c.suffix))
    json.dump(results, open(out, 'w'), indent=1)
    print('WROTE', out, flush=True)


if __name__ == '__main__':
    main()
