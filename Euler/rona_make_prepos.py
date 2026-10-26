'''
Trust-farming / pre-positioning plan. An attacker that is already present during the
IDS training window creates the very edges it will later use for the attack (every unique
malicious (src, dst) pair of the test period) in k training snapshots drawn from the last
`window` snapshots of the training period, so that at test time these edges have a
history (high temporal trust) and were seen as normal by the model.

The plan is a {snapshot: 2 x m array} dict for rona_run.py --train_inject; rona_hooks
applies the 1-hop visibility rule, so every client holding src or dst receives the edges.

    python Euler/rona_make_prepos.py --k 3 --out Exps/plans/prepos_k3.pt
'''
import argparse
import glob
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from loaders import rona_hooks as RH  # noqa: F401  (TData unpickling needs the loaders package)


def largest(pattern):
    return torch.load(sorted(glob.glob(pattern), key=os.path.getsize)[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--k', type=int, required=True, help='training snapshots containing the edges')
    ap.add_argument('--window', type=int, default=300)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    test = largest(os.path.join(RH.CACHE_DIR, 'test_*.pt'))   # client 'All' is the largest
    train = largest(os.path.join(RH.CACHE_DIR, 'train_*.pt'))
    pairs = sorted({(s, d) for t in range(test.T)
                    for s, d in zip(*test.eis[t][:, test.ys[t] > 0].tolist())})
    edges = np.array(pairs).T
    rng = np.random.default_rng(a.seed)
    lo = max(0, train.T - a.window)
    snaps = sorted(rng.choice(np.arange(lo, train.T), size=min(a.k, train.T - lo), replace=False).tolist())
    plan = {int(t): edges.copy() for t in snaps}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    torch.save(plan, a.out)
    print('train T=%d, %d unique malicious pairs, injected in snapshots %s -> %s'
          % (train.T, len(pairs), snaps, a.out))


if __name__ == '__main__':
    main()
