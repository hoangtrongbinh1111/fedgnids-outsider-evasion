'''
Replay saved injection plans on a trained checkpoint and dump per-edge results, for the
additional analyses (per-pair statistics, degree strata, TPR at fixed FPR under
attack, previously-seen malicious edges, transfer attacks).

For every evaluated plan it writes Exps/rona/<tag>/replay_<name>.npz with, per test edge
(injected edges included and marked):
  t, src, dst  snapshot index and endpoints
  p            decoder score (low = anomalous)
  y            1 for malicious edges
  inj          1 for injected (attacker) edges
  hist         number of train+val snapshots containing the directed edge
  indeg        number of distinct in-neighbours of dst in snapshot t (clean edges only)
  cutoff       validation cutoff (scalar)
The scores and cutoff are produced by exactly the same code path as rona_run.evaluate.

    python Euler/rona_replay.py --tag F_entente_gmedn_s0 --plugin_trust frozen \
        --plans clean Exps/rona/F_entente_gmedn_s0/plan_b1_ptfrozen.pt
    # make the malicious host pairs "familiar" (idealized familiar-path attacker)
    python Euler/rona_replay.py --tag F_entente_gmedn_s0 --plugin_trust frozen --familiar 1 --plans clean
'''
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rona_run as R
import spinup
from loaders import rona_hooks as RH
from loaders.tdata import TData
from models.euler_detector import DetectorEncoder


def familiarize(times, s, loader, cap):
    '''Give every malicious test pair a history count of `cap` that is used by the test phase
    only (RH.EXTRA_TEST_COUNTS), so that frozen trust gives it full weight. Train/val
    histories, the validation cutoff and the model weights are unchanged.'''
    RH.CFG['inject']['test'] = {}
    data = loader(**R.data_kwargs(times, 'test', 'All', s.mc))
    n = data.num_nodes
    pairs = set()
    for t in range(data.T):
        ys = data.ys[t] > 0
        mal = data.eis[t][:, ys]
        pairs |= set((mal[0] * n + mal[1]).tolist())
    RH.EXTRA_TEST_COUNTS.clear()
    for k in pairs:
        RH.EXTRA_TEST_COUNTS[k] = cap
    return pairs


def replay(model, h0_train, times, s, loader, device, plan, client='All'):
    model.eval()
    RH.CFG['inject']['val'] = {}
    h0, _ = spinup.get_cutoff(model, h0_train, times, {'val_nratio': 1}, s.fpweight, False, device, client, s.mc)
    RH.CFG['inject']['test'] = plan or {}
    DetectorEncoder.load_new_data(model.gcns, loader, R.data_kwargs(times, 'test', client, s.mc))
    RH.CFG['inject']['test'] = {}
    out = {k: [] for k in ('t', 'src', 'dst', 'p', 'y', 'inj', 'hist', 'indeg')}
    with torch.no_grad():
        zs = model.forward(TData.TEST, h0=h0.to(device), no_grad=True)
        data = model.gcns.module.data
        for t in range(data.T):
            ei = data.eis[t]
            inj = data.injected[t].numpy()
            out['p'].append(model.gcns.decode(ei.to(device), zs[t], True).cpu().numpy())
            out['y'].append((data.ys[t] > 0).numpy())
            out['inj'].append(inj)
            out['t'].append(np.full(ei.size(1), t))
            out['src'].append(ei[0].numpy())
            out['dst'].append(ei[1].numpy())
            out['hist'].append(data.hist_counts[t].numpy())
            clean = ei[:, ~torch.from_numpy(inj)]
            pairs = np.unique(clean[0].numpy().astype(np.int64) * data.num_nodes + clean[1].numpy())
            deg = np.bincount(pairs % data.num_nodes, minlength=data.num_nodes)
            out['indeg'].append(deg[ei[1].numpy()])
    res = {k: np.concatenate(v) for k, v in out.items()}
    res['cutoff'] = float(model.cutoff)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', required=True)
    ap.add_argument('--plans', nargs='+', default=['clean'], help="'clean' or paths to plan_b*.pt")
    ap.add_argument('--plugin_trust', default=None, choices=['frozen', 'causal'])
    ap.add_argument('--tau0', type=float, default=None)
    ap.add_argument('--cap', type=int, default=None)
    ap.add_argument('--inject_ew', type=float, default=None)
    ap.add_argument('--familiar', type=int, default=0)
    ap.add_argument('--synth', default=None, help='semi-synthetic test set (see rona_hooks._synthesize)')
    ap.add_argument('--prefix', default='replay')
    ap.add_argument('--client', default='All',
                    help="'All' = global-graph inference (as in ENTENTE's evaluation); '0'/'1'/'2' = "
                         'per-silo inference on the silo subgraph with its one-hop cross-silo edges, '
                         'validation cutoff and trust history of that silo')
    ap.add_argument('--cpu', type=int, default=1)
    ap.add_argument('--threads', type=int, default=8)
    c = ap.parse_args()
    R.set_seed(0)
    torch.backends.cudnn.enabled = False
    device = torch.device('cpu') if c.cpu else torch.device('cuda')
    if c.cpu:
        torch.set_num_threads(c.threads)
    from rona_attack import load_model
    a, s, times, loader, model, h0 = load_model(c.tag, device)
    if c.plugin_trust:
        RH.CFG.update(trust=True, trust_mode=c.plugin_trust)
    if c.tau0 is not None:
        RH.CFG['tau0'] = c.tau0
    if c.cap is not None:
        RH.CFG['cap'] = c.cap
    if c.inject_ew is not None:
        RH.CFG['inject_ew'] = c.inject_ew
    if c.synth:
        RH.CFG['synth'] = torch.load(c.synth)
    if c.familiar:
        pairs = familiarize(times, s, loader, max(RH.CFG['cap'], 1))
        print('familiarized %d malicious pairs' % len(pairs), flush=True)
    if c.client != 'All':
        RH.CFG['inject']['train'] = {}
        loader(**R.data_kwargs(times, 'train', c.client, s.mc))  # builds this silo's train history
    out_dir = os.path.join(R.OUT_ROOT, c.tag)
    for pl in c.plans:
        plan = None if pl == 'clean' else torch.load(pl, map_location='cpu')
        res = replay(model, h0, times, s, loader, device, plan, client=c.client)
        name = 'clean' if pl == 'clean' else os.path.splitext(os.path.basename(pl))[0]
        if pl != 'clean' and os.path.dirname(os.path.abspath(pl)) != os.path.abspath(out_dir):
            name += '_from_' + os.path.basename(os.path.dirname(os.path.abspath(pl)))
        fn = os.path.join(out_dir, '%s_%s.npz' % (c.prefix, name))
        np.savez_compressed(fn, **res)
        keep = res['inj'] == 0
        y, p = res['y'][keep], res['p'][keep]
        flag = p <= res['cutoff']
        summary = {'plan': name, 'SR': float(1 - flag[y].mean()), 'FPR': float(flag[~y].mean()),
                   'n_inj': int(res['inj'].sum()),
                   'inj_flag': float((res['p'][res['inj'] == 1] <= res['cutoff']).mean()) if res['inj'].any() else 0.0}
        print('REPLAY', fn, json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
