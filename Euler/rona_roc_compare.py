'''
Matched-FPR comparison of the trained models (Exps/rona/<tag>/scores_clean.npz, written by
rona_dump_scores.py) against the novelty baselines N1/N2 (Exps/novelty_baseline.json):
TPR at fixed FPRs, plus partial AUC up to FPR 1% (normalized), seed replicates as mean +- std.

    python Euler/rona_roc_compare.py > Exps/roc_compare.md
'''
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np
from sklearn.metrics import roc_curve

FPRS = [0.001, 0.0025, 0.005, 0.01, 0.024]
PAUC_MAX = 0.01


def tpr_at(fpr, tpr, x):
    return float(np.interp(x, fpr, tpr))


def pauc(fpr, tpr, xmax=PAUC_MAX):
    grid = np.linspace(0, xmax, 1001)
    return float(np.trapz(np.interp(grid, fpr, tpr), grid) / xmax)


def main():
    rows = defaultdict(list)
    for f in sorted(glob.glob('./Exps/rona/F_*/scores_clean.npz')):
        tag = os.path.basename(os.path.dirname(f))
        m = re.match(r'(.*)_s(\d+)$', tag)
        z = np.load(f)
        fpr, tpr, _ = roc_curve(z['y'], 1 - z['p'])
        rows[m.group(1) if m else tag].append([tpr_at(fpr, tpr, x) for x in FPRS] + [pauc(fpr, tpr)])
    nb = json.load(open('./Exps/novelty_baseline.json'))
    for name in ('N1', 'N2'):
        roc = np.array(nb[name]['roc'])  # list of (fpr, tpr) points, ties interpolated linearly
        fpr, tpr = roc[:, 0], roc[:, 1]
        rows['novelty ' + name].append([tpr_at(fpr, tpr, x) for x in FPRS] + [pauc(fpr, tpr)])
    head = ['TPR@FPR=%g%%' % (100 * x) for x in FPRS] + ['pAUC@1%']
    print('| model | ' + ' | '.join(head) + ' |')
    print('|---' * (len(head) + 1) + '|')
    for k, v in sorted(rows.items(), key=lambda kv: -np.mean(kv[1], 0)[-1]):
        v = np.array(v)
        cells = ['%.3f' % c if len(v) == 1 else '%.3f±%.3f' % (c, s)
                 for c, s in zip(v.mean(0), v.std(0))]
        print('| %s%s | %s |' % (k, '' if len(v) == 1 else ' (n=%d)' % len(v), ' | '.join(cells)))


if __name__ == '__main__':
    main()
