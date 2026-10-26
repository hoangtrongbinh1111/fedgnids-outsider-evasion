'''
Outsider edge-injection evasion attack against a trained (federated) Euler GNIDS.

Threat model: the attacker controls compromised hosts (sources of malicious edges and
the attacker infrastructure they talk to) but no FL client and no log pipeline. At test
snapshot t it may create up to  b * |M_t|  extra connections *originating* from hosts it
controls (C_t = endpoints of malicious edges seen up to t), to raise the decoder score
of its malicious edges M_t above the detection cutoff.

White-box, adaptive: the attacker knows the weights, the aggregator and the temporal
trust of every candidate edge. Optimization = PGD on a continuous edge mask with a
budget projection (Xu et al., IJCAI'19), followed by random-sampling discretization.
The GRU state is propagated sequentially through the perturbed snapshots, and the final
plan is re-scored by the unmodified evaluation pipeline (rona_run.evaluate).

    python Euler/rona_attack.py --tag E0_entente_gcn_s0 --budgets 1 2 5 10
'''
import argparse
import json
import os
import sys
import time
from collections import Counter

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rona_run as R
import spinup
from loaders import rona_hooks as RH
from models.embedders import detector_rgcn_rref
from models.euler_detector import DetectorEncoder, DetectorRecurrent
from models.recurrent import GRU
from models.robust_conv import temporal_trust


def load_model(tag, device):
    ck = torch.load(os.path.join(R.OUT_ROOT, tag, 'ckpt.pt'), map_location='cpu')
    a = R.get_args(['--tag', tag] + sum([['--' + k, str(v)] for k, v in ck['args'].items()
                                         if k not in ('tag', 'train_inject') and v is not None], []))
    s, times, loader, _ = R.build_env(a)
    # history counters for trust are built when train/val data are loaded; a model trained
    # on poisoned data (--train_inject, e.g. trust farming) must see the same history
    if ck['args'].get('train_inject'):
        RH.CFG['inject']['train'] = torch.load(ck['args']['train_inject'])
    loader(**R.data_kwargs(times, 'train', 'All', s.mc))
    RH.CFG['inject']['train'] = {}
    rrefs = detector_rgcn_rref(loader, R.data_kwargs(times, 'val', 'All', s.mc, device), a.hidden, a.hidden)
    model = DetectorRecurrent(GRU(a.hidden, a.hidden, a.zdim), rrefs, device).to(device)
    model.load_state_dict(ck['state'])
    model.eval()
    return a, s, times, loader, model, ck['h0'].to(device)


class Stepper:
    '''One-snapshot forward of the encoder + one GRU step, differentiable in edge weights.'''

    def __init__(self, model, device):
        self.enc = model.gcns.module
        self.rnn = model.rnn
        self.x = self.enc.data.xs.to(device)

    def __call__(self, ei, ew, trust, h):
        g = self.enc.relu(self.enc.c1(self.x, ei, ew, trust))
        g = self.enc.relu(self.enc.c2(g, ei, ew, trust))
        out, h = self.rnn.rnn(g.unsqueeze(0), h)
        return self.rnn.lin(out)[0], h


def project_budget(m, B):
    '''Euclidean projection onto {0 <= m <= 1, sum(m) <= B} by bisection on the shift.'''
    c = m.clamp(0, 1)
    if c.sum() <= B:
        return c
    lo, hi = (m.min() - 1).item(), m.max().item()
    for _ in range(40):
        mu = (lo + hi) / 2
        if (m - mu).clamp(0, 1).sum() > B:
            lo = mu
        else:
            hi = mu
    return (m - hi).clamp(0, 1)


def attack(model, h0, times, s, loader, device, budget, steps=60, samples=20, lr=0.2, seed=0,
           base_counts=None, stealth=0.0, cutoff=0.5, margin=1.0, client='All', loss_type='bce', kappa=0.5,
           control='src', cand_mode='all'):
    '''stealth > 0 adds  stealth * sum_c m_c * softplus(logit(cutoff) + 1 - score_c) / B:
    the attacker also wants every injected edge c to score above the detection cutoff,
    so that the covering edges themselves raise no alert.'''
    logit_cut = float(np.log(cutoff / (1 - cutoff)))
    g = torch.Generator(device='cpu').manual_seed(seed)
    RH.CFG['inject']['test'] = {}
    DetectorEncoder.load_new_data(model.gcns, loader, R.data_kwargs(times, 'test', client, s.mc))
    data = model.gcns.module.data
    # per-silo attack: only covering edges that this silo logs (one endpoint in the silo);
    # nodes outside the machine map never occur in the logs and belong to no silo
    silo_of = None if client in ('All', 'Fed') else (lambda v: str(RH.CFG['mc'].get(int(v))))
    n = data.num_nodes
    step = Stepper(model, device)
    tau0, cap, use_trust = RH.CFG['tau0'], RH.CFG['cap'], RH.CFG['trust']
    inj_ew = RH.CFG['inject_ew']
    controlled = set()
    prior = set()       # endpoints of the malicious edges of earlier windows (hosts already reached)
    # causal trust keeps counting during the test phase, including the attacker's own earlier
    # covering edges (rona_hooks._attach_trust); the attacker plans with exactly those counts
    causal = use_trust and RH.CFG.get('trust_mode') == 'causal'
    seen = Counter()
    plan, log = {}, []
    h = h0.to(device)
    for t in range(data.T):
        ei = data.eis[t].to(device)
        ew = data.ews[t].float().to(device)
        tr = data.trust[t].float().to(device)
        keys_t = (data.eis[t][0] * n + data.eis[t][1]).tolist()
        if causal:
            tr = temporal_trust(torch.tensor([base_counts.get(k, 0) + seen[k] for k in keys_t]),
                                tau0, cap).float().to(device)
        ys = data.ys[t] > 0
        mal = data.eis[t][:, ys]
        if mal.size(1) == 0 or budget == 0:
            with torch.no_grad():
                _, h = step(ei, ew, tr, h)
            controlled |= set(mal.flatten().tolist())
            if causal:
                seen.update(set(keys_t))
            prior |= set(mal.flatten().tolist())
            continue
        # sources are compromised when they act; with control='both' the destinations of
        # malicious edges (hosts the intruder has reached) may also originate covering edges;
        # with control='reached' only hosts reached in earlier windows (plus current sources) may
        controlled |= set((mal.flatten() if control == 'both' else mal[0]).tolist())
        if control == 'reached':
            controlled |= prior
        ctrl = torch.tensor(sorted(controlled))
        if cand_mode == 'all':
            tgt = torch.arange(n)
        else:
            # focused candidate set: covering edges INTO the endpoints of the malicious edges of
            # window t ('endpoints'), or also into their in-neighbours ('pool', = greedy pool)
            ends = torch.unique(mal.flatten())
            if cand_mode == 'pool':
                ends = torch.unique(torch.cat([ends, data.eis[t][0][torch.isin(data.eis[t][1], ends)]]))
            tgt = ends
        src = ctrl.repeat_interleave(len(tgt))
        dst = tgt.repeat(len(ctrl))
        existing = set((data.eis[t][0] * n + data.eis[t][1]).tolist())
        ok = torch.tensor([(int(a) != int(b)) and (int(a) * n + int(b) not in existing)
                           and (silo_of is None or silo_of(a) == client or silo_of(b) == client)
                           for a, b in zip(src, dst)])
        cand = torch.stack([src[ok], dst[ok]]).to(device)
        if cand.size(1) == 0:                      # nothing left to inject in this window
            with torch.no_grad():
                _, h = step(ei, ew, tr, h)
            if causal:
                seen.update(set(keys_t))
            prior |= set(mal.flatten().tolist())
            plan[t] = np.zeros((2, 0), dtype=np.int64)
            log.append({'t': t, 'n_mal': int(mal.size(1)), 'n_inj': 0, 'p_clean': float('nan'), 'p_adv': float('nan')})
            continue
        if use_trust:
            cnt = torch.tensor([base_counts.get(int(a) * n + int(b), 0) + seen[int(a) * n + int(b)]
                                for a, b in cand.t().tolist()])
            if RH.CFG.get('inject_familiar'):
                cnt = cnt.clamp(min=cap)
            ctr = temporal_trust(cnt, tau0, cap).float().to(device)
        else:
            ctr = torch.ones(cand.size(1), device=device)
        B = float(budget * mal.size(1))
        mal_d = mal.to(device)
        ei_all = torch.cat([ei, cand], 1)
        tr_all = torch.cat([tr, ctr])

        def loss_of(mask):
            z, _ = step(ei_all, torch.cat([ew, inj_ew * mask]), tr_all, h)
            sc = (z[mal_d[0]] * z[mal_d[1]]).sum(1)
            if loss_type == 'cw':
                # margin (Carlini-Wagner style) loss: only malicious edges still below
                # logit(cutoff) + kappa contribute, so effort goes to edges that can still flip
                loss = torch.relu(logit_cut + kappa - sc).mean()
            else:
                loss = torch.nn.functional.softplus(-sc).mean()
            if stealth > 0:
                sc_c = (z[cand[0]] * z[cand[1]]).sum(1)
                loss = loss + stealth * (mask * torch.nn.functional.softplus(logit_cut + margin - sc_c)).sum() / B
            return loss, torch.sigmoid(sc)

        m = torch.zeros(cand.size(1), device=device)
        for k in range(steps):
            m.requires_grad_(True)
            loss, _ = loss_of(m)
            gr, = torch.autograd.grad(loss, m)
            with torch.no_grad():
                m = project_budget(m - lr * gr / (gr.abs().max() + 1e-12), B)
        with torch.no_grad():
            best_m, best_loss = None, float('inf')
            mc = m.cpu()
            trials = [torch.zeros_like(mc).scatter_(0, torch.topk(mc, min(int(B), mc.numel())).indices, 1.0)]
            for _ in range(samples):
                bern = torch.bernoulli(mc, generator=g)
                if bern.sum() > B:
                    keep = torch.topk(mc * bern, int(B)).indices
                    bern = torch.zeros_like(mc).scatter_(0, keep, 1.0)
                trials.append(bern)
            for bm in trials:
                l, _ = loss_of(bm.to(device))
                if l.item() < best_loss:
                    best_loss, best_m = l.item(), bm
            sel = best_m.bool()
            chosen = cand[:, sel.to(device)].cpu()
            _, p_clean = loss_of(torch.zeros_like(m))
            _, p_adv = loss_of(best_m.to(device))
            _, h = step(ei_all, torch.cat([ew, inj_ew * best_m.to(device)]), tr_all, h)
        plan[t] = chosen.numpy()
        if causal:
            seen.update(set(keys_t) | set((chosen[0] * n + chosen[1]).tolist()))
        prior |= set(mal.flatten().tolist())
        log.append({'t': t, 'n_mal': int(mal.size(1)), 'n_inj': int(sel.sum()),
                    'p_clean': float(p_clean.mean()), 'p_adv': float(p_adv.mean())})
    return plan, log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', required=True)
    ap.add_argument('--budgets', type=int, nargs='+', default=[1, 2, 5, 10])
    ap.add_argument('--steps', type=int, default=60)
    ap.add_argument('--transfer_from', default=None, help='reuse plans crafted on another model')
    ap.add_argument('--stealth', type=float, default=0.0, help='weight of the "injected edges stay unflagged" term')
    ap.add_argument('--margin', type=float, default=1.0, help='logit margin above the cutoff required of injected edges')
    ap.add_argument('--inject_ew', type=float, default=None,
                    help='standardized weight of injected edges (default: rona_hooks 0.5; 1.0 = max observed)')
    ap.add_argument('--plugin_trust', default=None, choices=['frozen', 'causal'],
                    help='evaluate (and attack) with test-time temporal trust on a model trained without it')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--cpu', type=int, default=0)
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--samples', type=int, default=20, help='Bernoulli discretization samples')
    ap.add_argument('--familiar', type=int, default=0,
                    help='1: every malicious test pair counts as seen >= cap times in train+val '
                         '(idealized familiar-path attacker; model weights unchanged)')
    ap.add_argument('--familiar_cover', type=int, default=0,
                    help='1: every covering edge counts as seen >= cap times (full trust); idealized '
                         'attacker with footholds that routinely talk to the targets')
    ap.add_argument('--suffix', default='', help='extra suffix for plan/json file names')
    ap.add_argument('--loss', default='bce', choices=['bce', 'cw'],
                    help="attacker objective: 'bce' = mean -log p (default), 'cw' = margin loss")
    ap.add_argument('--kappa', type=float, default=0.5, help='logit margin above the cutoff for --loss cw')
    ap.add_argument('--cand', default='all', choices=['all', 'endpoints', 'pool'],
                    help='candidate destinations of covering edges: any host (default), the endpoints of the '
                         'malicious edges of the window, or those endpoints and their in-neighbours')
    ap.add_argument('--control', default='src', choices=['src', 'both', 'reached'],
                    help="hosts that may originate covering edges: sources of malicious edges so far (default), "
                         "both endpoints (including the current window's destinations), or 'reached': current "
                         "sources plus both endpoints of the malicious edges of earlier windows")
    ap.add_argument('--synth', default=None,
                    help='semi-synthetic test set: torch file {"edges": {t: 2 x m}, "ew": w, "remove_real": bool}')
    ap.add_argument('--client', default='All',
                    help="'All' = global-graph inference; a silo id = attack and evaluate that silo's view "
                         "(its subgraph plus one-hop cross-silo edges, its own cutoff and trust history)")
    c = ap.parse_args()
    R.set_seed(c.seed)
    # cuDNN refuses GRU backward in eval mode; the native kernel computes the same function
    torch.backends.cudnn.enabled = False
    device = torch.device('cpu') if c.cpu else torch.device('cuda')
    if c.cpu:
        torch.set_num_threads(c.threads)
    a, s, times, loader, model, h0 = load_model(c.tag, device)
    if c.plugin_trust:
        RH.CFG.update(trust=True, trust_mode=c.plugin_trust)
    if c.inject_ew is not None:
        RH.CFG['inject_ew'] = c.inject_ew
    if c.synth:
        RH.CFG['synth'] = torch.load(c.synth)
        c.suffix = '_synth' + os.path.splitext(os.path.basename(c.synth))[0].replace('lanl_synth_', '') + c.suffix
    if c.familiar:
        from rona_replay import familiarize
        print('familiarized %d malicious pairs' % len(familiarize(times, s, loader, max(RH.CFG['cap'], 1))),
              flush=True)
    if c.client != 'All':
        RH.CFG['inject']['train'] = {}
        for ph in ('train', 'val'):                      # builds this silo's trust history
            loader(**R.data_kwargs(times, ph, c.client, s.mc))
    hist = RH.HISTORY[c.client]
    base_counts = hist['train'] + hist['val'] + RH.EXTRA_TEST_COUNTS
    out_dir = os.path.join(R.OUT_ROOT, c.tag)
    results = {}
    clean = R.evaluate(model, h0, times, s, loader, device, client=c.client)
    results['clean'] = clean
    print('CLEAN', json.dumps(clean), flush=True)
    suf = '_s%g' % c.stealth if c.stealth > 0 else ''
    if c.plugin_trust:
        suf = '_pt' + c.plugin_trust + suf
    if c.inject_ew is not None:
        suf += '_ew%g' % c.inject_ew
    if c.stealth > 0 and c.margin != 1.0:
        suf += '_m%g' % c.margin
    if c.familiar:
        suf += '_fam'
    if c.familiar_cover:
        RH.CFG['inject_familiar'] = True
        suf += '_famcov'
    if c.client != 'All':
        suf += '_c' + c.client
    if c.loss == 'cw':
        suf += '_cw%g' % c.kappa
    if c.control != 'src':
        suf += '_ctl' + c.control
    if c.cand != 'all':
        suf += '_cand' + c.cand
    suf += c.suffix
    for b in c.budgets:
        t0 = time.time()
        if c.transfer_from:
            plan = torch.load(os.path.join(R.OUT_ROOT, c.transfer_from, 'plan_b%d%s.pt' % (b, suf)))
            log = None
        else:
            # cutoff + val hidden state, exactly as in evaluation
            hval, _ = spinup.get_cutoff(model, h0, times, {'val_nratio': 1}, s.fpweight, False, device,
                                        c.client, s.mc)
            plan, log = attack(model, hval, times, s, loader, device, b, steps=c.steps, seed=c.seed,
                               samples=c.samples, base_counts=base_counts, stealth=c.stealth,
                               cutoff=float(model.cutoff), margin=c.margin, client=c.client,
                               loss_type=c.loss, kappa=c.kappa, control=c.control,
                               cand_mode=c.cand)
            torch.save(plan, os.path.join(out_dir, 'plan_b%d%s.pt' % (b, suf)))
        res = R.evaluate(model, h0, times, s, loader, device, test_inject=plan, client=c.client)
        res['budget'] = b
        res['stealth'] = c.stealth
        res['margin'] = c.margin
        res['attack_time'] = time.time() - t0
        if log:
            res['pred_p_adv'] = float(np.nanmean([l['p_adv'] for l in log]))
            res['pred_p_clean'] = float(np.nanmean([l['p_clean'] for l in log]))
        results['b%d' % b] = res
        print('ATTACK', json.dumps(res), flush=True)
    name = ('attack_transfer_%s%s.json' % (c.transfer_from, suf) if c.transfer_from
            else 'attack%s.json' % suf)
    json.dump(results, open(os.path.join(out_dir, name), 'w'), indent=1)


if __name__ == '__main__':
    main()
