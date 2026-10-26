'''
Re-evaluate a trained checkpoint with a different temporal-trust setting at test time only
("plug-in trust"): e.g. a model trained without trust, evaluated with causal trust.

    python Euler/rona_eval_trust.py --tag L_entente_gmedn_s0 --lanl 1 --trust 1 --trust_mode causal
    python Euler/rona_eval_trust.py --tag F_entente_gmedn_s0 --trust 1 --trust_mode frozen

Writes Exps/rona/<tag>/eval_trust<trust>_<mode>_t<tau0>_c<cap>.json. The validation cutoff is
computed with the training-time setting (val data is loaded before the override), so compare
AUC / TPR at fixed FPR rather than the cutoff-based SR/FPR.
'''
import argparse
import json
import os
import sys

import numpy as np
import torch
from sklearn.metrics import roc_curve

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', required=True)
    ap.add_argument('--lanl', type=int, default=0)
    ap.add_argument('--trust', type=int, default=1)
    ap.add_argument('--trust_mode', default='causal', choices=['frozen', 'causal'])
    ap.add_argument('--tau0', type=float, default=0.1)
    ap.add_argument('--cap', type=int, default=3)
    ap.add_argument('--cpu', type=int, default=0)
    c = ap.parse_args()
    if c.lanl:
        import rona_lanl
        rona_lanl.patch()
    import rona_attack as A
    import rona_run as R
    from loaders import rona_hooks as RH
    torch.backends.cudnn.enabled = False
    device = torch.device('cpu') if c.cpu else torch.device('cuda')
    a, s, times, loader, model, h0 = A.load_model(c.tag, device)
    RH.CFG.update(trust=bool(c.trust), trust_mode=c.trust_mode, tau0=c.tau0, cap=c.cap)
    res, raw = R.evaluate(model, h0, times, s, loader, device, return_raw=True)
    fpr, tpr, _ = roc_curve(raw['y'], 1 - raw['p'])
    for x in (1e-4, 1e-3, 1e-2):
        res['TPR@FPR%g' % x] = float(np.interp(x, fpr, tpr))
    res.update(eval_trust=c.trust, eval_trust_mode=c.trust_mode, eval_tau0=c.tau0, eval_cap=c.cap,
               train_args=vars(a))
    name = 'eval_trust%d_%s_t%g_c%d.json' % (c.trust, c.trust_mode, c.tau0, c.cap)
    json.dump(res, open(os.path.join(R.OUT_ROOT, c.tag, name), 'w'), indent=1, default=str)
    print('EVAL', c.tag, name, json.dumps({k: res[k] for k in ('AUC', 'AP', 'TPR@FPR0.0001',
                                                               'TPR@FPR0.001', 'TPR@FPR0.01')}))


if __name__ == '__main__':
    main()
