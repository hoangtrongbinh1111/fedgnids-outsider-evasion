"""
Dump raw decoder scores (p; low = anomalous) and labels (y) of a trained RoNA model on the
OpTC test set, clean and optionally under an injection plan, to Exps/rona/<tag>/scores_*.npz.

    python Euler/rona_dump_scores.py --tag F_entente_gmedn_trust_s0 --cpu 1 \
        --plan Exps/rona/F_entente_gmedn_trust_s0/plan_b1.pt
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rona_run as R
from rona_attack import load_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', required=True)
    ap.add_argument('--plan', default=None, help='injection plan (.pt, snapshot -> 2 x m array)')
    ap.add_argument('--cpu', type=int, default=0)
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--seed', type=int, default=0)
    c = ap.parse_args()
    R.set_seed(c.seed)
    device = torch.device('cpu') if c.cpu else torch.device('cuda')
    if c.cpu:
        torch.set_num_threads(c.threads)
    a, s, times, loader, model, h0 = load_model(c.tag, device)
    out = os.path.join(R.OUT_ROOT, c.tag)
    res, raw = R.evaluate(model, h0, times, s, loader, device, return_raw=True)
    np.savez_compressed(os.path.join(out, 'scores_clean.npz'), p=raw['p'], y=raw['y'],
                        cutoff=res['cutoff'])
    print('clean', res['AP'], res['AUC'], res['TPR'], res['FPR'], flush=True)
    if c.plan:
        plan = torch.load(c.plan, map_location='cpu')
        res, raw = R.evaluate(model, h0, times, s, loader, device, test_inject=plan, return_raw=True)
        name = 'scores_' + os.path.splitext(os.path.basename(c.plan))[0] + '.npz'
        np.savez_compressed(os.path.join(out, name), p=raw['p'], y=raw['y'], cutoff=res['cutoff'])
        print('attacked', name, res['AP'], res['AUC'], res['TPR'], res['FPR'], flush=True)


if __name__ == '__main__':
    main()
