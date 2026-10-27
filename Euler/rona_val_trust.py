'''
Label-free selection of the plug-in trust parameters on the validation windows.

For every (tau0, cap) on a grid, the trained checkpoint scores the validation windows with
plug-in frozen trust, exactly as when its cutoff is computed (spinup.get_cutoff): trust counts
come from the training windows only, and the score is the link-prediction AP/AUC of the real
validation edges against the same random non-edges for every setting. No malicious labels,
attack results or test windows are used.

    python Euler/rona_val_trust.py --tag F_entente_gmedn_s0 [--cpu 1]

Writes Exps/rona/<tag>/val_trust.json.
'''
import argparse
import json
import os
import sys

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

GRID = [(0.0, 3), (0.05, 3), (0.1, 3), (0.3, 3), (0.5, 3), (1.0, 3), (0.1, 1), (0.1, 10), (0.1, 30)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', required=True)
    ap.add_argument('--cpu', type=int, default=1)
    ap.add_argument('--threads', type=int, default=8)
    ap.add_argument('--reps', type=int, default=5, help='negative-sample draws per setting')
    c = ap.parse_args()
    import rona_attack as A
    import rona_run as R
    import spinup
    from loaders import rona_hooks as RH
    from models.euler_detector import DetectorEncoder as Encoder
    from loaders.tdata import TData
    torch.backends.cudnn.enabled = False
    device = torch.device('cpu') if c.cpu else torch.device('cuda')
    if c.cpu:
        torch.set_num_threads(c.threads)
    a, s, times, loader, model, h0 = A.load_model(c.tag, device)
    out = {'tag': c.tag, 'grid': []}
    for tau0, cap in GRID:
        RH.CFG.update(trust=tau0 < 1, trust_mode='frozen', tau0=tau0, cap=cap)
        args = {'start': times['val_start'], 'end': times['val_end'], 'delta': times['delta'],
                'is_test': False, 'use_flows': False, 'client': 'All', 'mc': s.mc}
        Encoder.load_new_data(model.gcns, spinup.LOAD_FN, args)
        model.eval()
        with torch.no_grad():
            zs = Encoder.forward(model.gcns.module, TData.ALL, True).to(device)
            zs, _ = model.rnn(zs, h0.to(device), include_h=True)
        aps, aucs = [], []
        for r in range(c.reps):
            R.set_seed(r)
            with torch.no_grad():
                p, n = Encoder.score_edges(model.gcns, zs, TData.ALL, 1)
            p, n = p.cpu().numpy().ravel(), n.cpu().numpy().ravel()
            y = np.r_[np.ones(len(p)), np.zeros(len(n))]
            sc = np.r_[p, n]
            aps.append(average_precision_score(y, sc))
            aucs.append(roc_auc_score(y, sc))
        row = {'tau0': tau0, 'cap': cap, 'val_AP': float(np.mean(aps)), 'val_AP_sd': float(np.std(aps)),
               'val_AUC': float(np.mean(aucs))}
        out['grid'].append(row)
        print('VAL', json.dumps(row), flush=True)
    best = max(out['grid'], key=lambda r: r['val_AP'])
    out['best'] = best
    json.dump(out, open(os.path.join(R.OUT_ROOT, c.tag, 'val_trust.json'), 'w'), indent=1)
    print('BEST', json.dumps(best))


if __name__ == '__main__':
    main()
