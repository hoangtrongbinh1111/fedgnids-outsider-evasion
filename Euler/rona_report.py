'''
Collects Exps/rona/<tag>/clean.json and attack*.json into one markdown table, grouping
seed replicates (<name>_s<k>) as mean +- std.

    python Euler/rona_report.py [--prefix F_] > Exps/rona_report.md
'''
import argparse
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np

ROOT = './Exps/rona/'


def fmt(xs, pct=False):
    xs = np.array(xs, dtype=float) * (100 if pct else 1)
    if len(xs) == 1:
        return ('%.1f' if pct else '%.3f') % xs[0]
    return (('%.1f±%.1f' if pct else '%.3f±%.3f') % (xs.mean(), xs.std())) + ' (n=%d)' % len(xs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prefix', default='F_')
    a = ap.parse_args()
    # (model, attack mode, budget) -> metric -> [values over seeds]
    rows = defaultdict(lambda: defaultdict(list))
    for d in sorted(glob.glob(os.path.join(ROOT, a.prefix + '*'))):
        tag = os.path.basename(d)
        m = re.match(r'(.*)_s(\d+)$', tag)
        model = m.group(1) if m else tag
        cj = os.path.join(d, 'clean.json')
        if os.path.exists(cj):
            r = json.load(open(cj))
            for k in ('AP', 'AUC', 'SR', 'FPR'):
                rows[(model, 'clean', 0)][k].append(r[k])
        for f in glob.glob(os.path.join(d, 'attack*.json')):
            mode = os.path.basename(f)[len('attack'):-len('.json')].strip('_') or 'adaptive'
            res = json.load(open(f))
            mdl = model
            # plug-in trust (attack_pt<mode>[_...].json) is a different deployed model
            pt = re.match(r'(pt(?:frozen|causal))_?(.*)$', mode)
            if pt:
                mdl, mode = model + '+' + pt.group(1), pt.group(2) or 'adaptive'
                if mode == 'adaptive':
                    for k in ('AP', 'AUC', 'SR', 'FPR'):
                        rows[(mdl, 'clean', 0)][k].append(res['clean'][k])
            for key, r in res.items():
                if key == 'clean':
                    continue
                for k in ('AP', 'AUC', 'SR', 'SR_stealthy', 'inj_flag_rate'):
                    rows[(mdl, mode, r['budget'])][k].append(r.get(k, float('nan')))
    print('| model | attack | b | AP | AUC | SR % | SR stealthy % | inj. flagged % |')
    print('|---|---|---|---|---|---|---|---|')
    for (model, mode, b), r in sorted(rows.items(), key=lambda x: (x[0][0], x[0][1] != 'clean', x[0][1], x[0][2])):
        print('| %s | %s | %s | %s | %s | %s | %s | %s |' % (
            model, mode, b or '-', fmt(r['AP']), fmt(r['AUC']), fmt(r['SR'], True),
            fmt(r['SR_stealthy'], True) if r.get('SR_stealthy') else '-',
            fmt(r['inj_flag_rate'], True) if r.get('inj_flag_rate') else '-'))


if __name__ == '__main__':
    main()
