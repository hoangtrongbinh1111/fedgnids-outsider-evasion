'''
RoNA experiment driver: train a (federated) Euler GNIDS on OpTC with a chosen
neighborhood aggregator, evaluate it clean, and save a checkpoint that the attack
script (rona_attack.py) reuses.

Example (ENTENTE baseline, K=3):
    python Euler/rona_run.py --fed Momentum --agg gcn --tag entente_gcn_s0
'''
import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import loaders.load_optc as optc
import spinup
from fl_utils import build_random_graph
from loaders import rona_hooks as RH
from loaders.tdata import TData
from models import embedders
from models.embedders import detector_rgcn_rref
from models.euler_detector import DetectorEncoder, DetectorRecurrent
from models.recurrent import GRU
from run import load_machine_clusters

AUX = './Euler/'
OUT_ROOT = './Exps/rona/'


def _release_cuda_cache(*_):
    # spinup calls this no-op hook after every client step; returning cached blocks lets a
    # second GPU process (e.g. an attack job) share the card without WDDM paging.
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


spinup.calculate_model_memory_cost = _release_cuda_cache


def get_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--fed', default='Momentum', choices=['Momentum', 'FedAvg', 'WeightedFedAvg', 'FedProx', 'None'])
    ap.add_argument('--K', type=int, default=3)
    ap.add_argument('--cluster_fname', default=None)
    ap.add_argument('--agg', default='gcn', choices=['gcn', 'mean', 'guard', 'gmed', 'gmedn'])
    ap.add_argument('--fl_fix', type=int, default=1,
                    help='1: clients start from the global init, norm bounding clips the '
                         'update delta (paper Eq.10) and the ENTENTE weighted average is '
                         'normalized (see momentum_agg); 0: released code')
    ap.add_argument('--iters', type=int, default=3)
    ap.add_argument('--trust', type=int, default=0)
    ap.add_argument('--tau0', type=float, default=0.1)
    ap.add_argument('--cap', type=int, default=3)
    ap.add_argument('--trust_mode', default='frozen', choices=['frozen', 'causal'])
    ap.add_argument('--fedT', type=int, default=10)
    ap.add_argument('--lr', type=float, default=0.005)
    ap.add_argument('--hidden', type=int, default=32)
    ap.add_argument('--zdim', type=int, default=16)
    ap.add_argument('--fpweight', type=float, default=0.6)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--cpu', type=int, default=0)
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--train_inject', default=None, help='pickle of {snapshot: 2xm edges} injected in training')
    ap.add_argument('--tag', required=True)
    a = ap.parse_args(argv)
    if a.cluster_fname is None:
        a.cluster_fname = {3: 'optc_2_2.json', 4: 'optc_3_3.json'}.get(a.K, 'optc_2_2.json')
    return a


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def pick_device(a):
    if a.cpu or not torch.cuda.is_available():
        torch.set_num_threads(a.threads)
        return torch.device('cpu')
    return torch.device('cuda')


def build_env(a):
    '''Configures hooks/encoder globals and returns (spinup-style args, times, loader).'''
    mc = load_machine_clusters('./data/fl_lanl/ip_dict.json', AUX + a.cluster_fname, 'OPTC')
    RH.CFG.update(trust=bool(a.trust), tau0=a.tau0, cap=a.cap, trust_mode=a.trust_mode, mc=mc)
    # non-default silo partitions get their own per-silo cache keys (see rona_hooks._cache_path)
    RH.CFG['cluster_tag'] = None if a.cluster_fname in ('optc_2_2.json', 'optc_3_3.json') else a.cluster_fname
    embedders.RGCN.AGG = a.agg
    embedders.RGCN.ITERS = a.iters
    loader = RH.wrap_loader(optc.load_optc_dist)
    spinup.LOAD_FN = loader

    s = argparse.Namespace(
        fed=a.fed, rnn=GRU, impl='DETECT', dataset='OPTC_RONA_' + a.tag, encoder_name='GCN',
        client_list=[str(x) for x in range(a.K)], client_number=a.K, client='Fed', mc=mc,
        fedopt_epoch=5, poison=False, poison_defense=False, nb_defense=False, dp_defense=False,
        cdp_defense=False, personalization=False, equal_weights=False, flows=False,
        similarity=[1.0 / a.K] * a.K, fpweight=a.fpweight)
    delta = int(0.1 * 3600)
    tr_start, tr_end = 0, optc.DATE_OF_EVIL_LANL
    val = max((tr_end - tr_start) // 20, delta * 2)
    times = {'tr_start': tr_start, 'tr_end': tr_end - val, 'val_start': tr_end - val,
             'val_end': tr_end, 'te_times': [(tr_end, optc.TIMES['all'])], 'delta': delta}
    RH.CFG['tr_start'] = tr_start
    tr_args = {'epochs': a.fedT, 'min': 1, 'nratio': 1, 'val_nratio': 1, 'fedT': a.fedT,
               'fedMu': 0.05, 'lr': a.lr, 'patience': 100}
    return s, times, loader, tr_args


def data_kwargs(times, phase, client, mc, device=None):
    if phase == 'train':
        kw = {'start': times['tr_start'], 'end': times['tr_end'], 'is_test': False}
    elif phase == 'val':
        kw = {'start': times['val_start'], 'end': times['val_end'], 'is_test': False}
    else:
        kw = {'start': times['te_times'][0][0], 'end': times['te_times'][0][1], 'is_test': True}
    kw.update(delta=times['delta'], use_flows=False, client=client, mc=mc)
    if device is not None:
        kw['device'] = device
    return kw


def clip_delta(w, M, fed_model):
    '''Norm bounding as in ENTENTE Eq.10: w <- w_fed + delta * min(1, M / ||delta||), with
    ||delta|| the sum of per-tensor norms as in the released clip_update. A non-finite
    update is dropped (the client keeps the global weights).'''
    delta = {k: w[k] - fed_model[k] for k in w}
    norm = sum(torch.norm(v.float()) for v in delta.values())
    if not torch.isfinite(norm):
        print('NONFINITE_UPDATE dropped')
        return {k: fed_model[k].clone() for k in w}
    sf = 1.0 if norm <= M else M / norm
    print('scaling_factor: ', sf)
    return {k: fed_model[k] + sf * delta[k] for k in w}


def momentum_agg(w, w_fed, client_list, similarity, defense=False):
    '''ENTENTE aggregation (non-defense branch of fl_utils.Momentum) with a normalized
    weighted average. The released code starts w_avg at a_0 * w_0 and then adds a_0 * w_0
    again in the loop while dividing by sum(a) = 1, so every round multiplies the global
    weights by ~(1 + a_0) and biases them toward client 0; training diverges (NaN by
    round ~15). ACS weights (0.8 * S^Jac + 0.2 * softmax(mass * velocity)) and norm
    bounding are unchanged.'''
    import fl_utils
    idx = [i for i, wi in enumerate(w) if wi is not None]
    client_weights = torch.tensor(fl_utils.calculate_client_weights2(similarity, w))
    moms = []
    for i in idx:
        mass, velocity = fl_utils.calculate_contribution_metrics(w[i], w_fed)
        moms.append(mass * velocity)
    weights = 0.8 * client_weights + 0.2 * torch.softmax(torch.tensor(moms), dim=0)
    print('Weights: ', weights)
    clipped = [clip_delta(w[i], 5, w_fed) for i in idx]
    tot = weights.sum()
    return {k: sum(weights[j] * clipped[j][k] for j in range(len(idx))) / tot for k in clipped[0]}


def apply_fl_fix():
    import fl_utils
    fl_utils.clip_update = clip_delta
    spinup.clip_update = clip_delta
    spinup.Momentum = momentum_agg


def sync_client_init(rrefs_dict):
    '''Standard FL: every client starts from the global model's encoder weights.'''
    fed = rrefs_dict['fed'].module
    for k, r in rrefs_dict.items():
        if k != 'fed':
            r.module.c1.load_state_dict(fed.c1.state_dict())
            r.module.c2.load_state_dict(fed.c2.state_dict())


def train(a, s, times, loader, tr_args, device):
    worker_args = [a.hidden, a.hidden]
    rnn_args = [a.hidden, a.hidden, a.zdim]
    if a.train_inject:
        RH.CFG['inject']['train'] = torch.load(a.train_inject)
    t0 = time.time()
    if a.fed == 'None':
        s.client = 'All'
        rrefs = detector_rgcn_rref(loader, data_kwargs(times, 'train', 'All', s.mc, device), *worker_args)
        model, h0, _ = spinup.train(rrefs, tr_args, s, rnn_args, device)
    else:
        if a.fed == 'Momentum':
            global_G = build_random_graph(s.mc)
            s.similarity = []
        rrefs_dict = {}
        for c in s.client_list:
            rrefs = detector_rgcn_rref(loader, data_kwargs(times, 'train', c, s.mc, device), *worker_args)
            rrefs_dict['rrefs-client-' + c] = rrefs
            if a.fed == 'Momentum':
                s.similarity.append(rrefs.module.data.compute_graph_weights(c, global_G)[2])
        rrefs_dict['fed'] = detector_rgcn_rref(loader, data_kwargs(times, 'train', 'All', s.mc, device), *worker_args)
        if a.fl_fix:
            apply_fl_fix()
            sync_client_init(rrefs_dict)
        model, h0, _, _, _ = spinup.train_fed(rrefs_dict, tr_args, s, rnn_args, device, device, times, OUT_ROOT)
    RH.CFG['inject']['train'] = {}
    return model, h0, time.time() - t0


def evaluate(model, h0_train, times, s, loader, device, test_inject=None, return_raw=False, client='All'):
    '''Clean or attacked evaluation on the OpTC test period. Injected edges are excluded
    from the metrics; SR = fraction of malicious edges scored above the cutoff (missed).
    client != 'All': the silo scores its own view (its subgraph plus one-hop cross-silo edges)
    with its own validation cutoff.'''
    model.eval()
    RH.CFG['inject']['val'] = {}
    h0, _ = spinup.get_cutoff(model, h0_train, times, {'val_nratio': 1}, s.fpweight, False, device, client, s.mc)
    RH.CFG['inject']['test'] = test_inject or {}
    DetectorEncoder.load_new_data(model.gcns, loader, data_kwargs(times, 'test', client, s.mc))
    RH.CFG['inject']['test'] = {}
    with torch.no_grad():
        zs = model.forward(TData.TEST, h0=h0.to(device), no_grad=True)
        data = model.gcns.module.data
        P, Y, INJ, TT = [], [], [], []
        for t in range(data.T):
            ei = data.eis[t].to(device)
            P.append(model.gcns.decode(ei, zs[t], True).cpu().numpy())
            Y.append((data.ys[t] > 0).numpy())
            INJ.append(data.injected[t].numpy())
            TT.append(np.full(ei.size(1), t))
    p, y, inj, tt = np.concatenate(P), np.concatenate(Y), np.concatenate(INJ), np.concatenate(TT)
    cutoff = float(model.cutoff)
    # an injected edge that is itself flagged still raises an alert for the defender
    inj_flag = p[inj] <= cutoff
    alarmed_t = set(tt[inj][inj_flag].tolist())
    keep = ~inj
    p, y, tt = p[keep], y[keep], tt[keep]
    flag = p <= cutoff
    quiet = np.array([t not in alarmed_t for t in tt[y]])
    res = {
        'AP': float(average_precision_score(y, 1 - p)),
        'AUC': float(roc_auc_score(y, 1 - p)),
        'cutoff': cutoff,
        'TPR': float(flag[y].mean()),
        'FPR': float(flag[~y].mean()),
        'SR': float(1 - flag[y].mean()),
        'n_mal': int(y.sum()),
        'n_edges': int(len(y)),
        'n_injected': int(inj.sum()),
        'inj_flag_rate': float(inj_flag.mean()) if inj.any() else 0.0,
        # missed malicious edges in snapshots where no injected edge raised an alert
        'SR_stealthy': float((~flag[y] & quiet).mean()),
        'mal_score_mean': float(p[y].mean()),
    }
    if return_raw:
        return res, {'p': p, 'y': y}
    return res


def main():
    a = get_args()
    set_seed(a.seed)
    device = pick_device(a)
    out = os.path.join(OUT_ROOT, a.tag)
    os.makedirs(out, exist_ok=True)
    os.makedirs('./Exps/FL/', exist_ok=True)
    os.makedirs('./Exps/result/', exist_ok=True)
    s, times, loader, tr_args = build_env(a)
    model, h0, tr_time = train(a, s, times, loader, tr_args, device)
    torch.save({'state': model.state_dict(), 'h0': h0.detach().cpu(), 'args': vars(a)},
               os.path.join(out, 'ckpt.pt'))
    res = evaluate(model, h0, times, s, loader, device)
    res.update(train_time=tr_time, **{k: v for k, v in vars(a).items()})
    print('RESULT', json.dumps(res))
    json.dump(res, open(os.path.join(out, 'clean.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
