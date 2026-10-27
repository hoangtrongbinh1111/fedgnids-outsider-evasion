'''
Summaries over training seeds (mean and population std, rates in %):
  * main OpTC table over seeds 0-9 (ENTENTE-GCN attack.json, GMed-T attack_ptfrozen.json)
  * trust sensitivity under the adaptive attack (attack_ptfrozen_t<tau0>_c<cap>.json, b = 1, 10)
  * secondary seed-5-9 checks (gmedn without trust, GCN + trust, epsilon = 1, hard filter)

    python Euler/rona_seed_summary.py [--seeds 0 1 ... 9]   ->  Exps/seed_summary.json
'''
import argparse
import json
import os

import numpy as np

R = './Exps/rona'
KEYS = ('AP', 'AUC', 'SR', 'SR_stealthy', 'inj_flag_rate', 'FPR')


def load(tag, fname):
    p = os.path.join(R, tag, fname)
    return json.load(open(p)) if os.path.exists(p) else None


def collect(tag_fmt, fname, seeds, conds):
    per = {}
    for s in seeds:
        r = load(tag_fmt % s, fname)
        if r is None or not all(c in r for c in conds):
            continue
        per[s] = {c: {k: r[c].get(k) for k in KEYS} for c in conds}
    return per


def stats(per, conds):
    out = {'seeds': sorted(per)}
    for c in conds:
        out[c] = {}
        for k in KEYS:
            v = [per[s][c][k] for s in per if per[s][c][k] is not None]
            if not v:
                continue
            a = np.array(v, float) * (1 if k in ('AP', 'AUC') else 100)
            out[c][k] = {'mean': float(a.mean()), 'std': float(a.std()), 'min': float(a.min()),
                         'max': float(a.max()), 'per_seed': [round(float(x), 4) for x in a]}
    return out


def show(name, st, conds):
    print('== %s  seeds %s' % (name, st['seeds']))
    for c in conds:
        d = st[c]
        f = lambda k, p: ('%.3f±%.3f' if p else '%.1f±%.1f') % (d[k]['mean'], d[k]['std']) if k in d else '--'
        print('  %-6s AP %s AUC %s SR %s SRst %s flag %s FPR %s   SR per seed %s' % (
            c, f('AP', 1), f('AUC', 1), f('SR', 0), f('SR_stealthy', 0), f('inj_flag_rate', 0), f('FPR', 0),
            d.get('SR', {}).get('per_seed')))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', nargs='+', type=int, default=list(range(10)))
    c = ap.parse_args()
    out = {}
    full = ['clean', 'b1', 'b2', 'b5', 'b10']
    for name, fmt, fn in (('ENTENTE-GCN', 'F_entente_gcn_s%d', 'attack.json'),
                          ('GMed-T', 'F_entente_gmedn_s%d', 'attack_ptfrozen.json')):
        st = stats(collect(fmt, fn, c.seeds, full), full)
        out[name] = st
        show(name, st, full)
    two = ['clean', 'b1', 'b10']
    sens = {}
    for t0, cap in ((0.05, 3), (0.1, 3), (0.3, 3), (0.1, 1), (0.1, 10), (0, 1), (1, 3)):
        if (t0, cap) == (0.1, 3):
            fn = 'attack_ptfrozen.json'
        elif (t0, cap) == (1, 3):
            fn = 'attack.json'          # gmedn without trust
        else:
            fn = 'attack_ptfrozen_t%g_c%d.json' % (t0, cap)
        st = stats(collect('F_entente_gmedn_s%d', fn, c.seeds, two), two)
        sens['t%g_c%d' % (t0, cap)] = st
        show('sensitivity tau0=%g cap=%d' % (t0, cap), st, two)
    out['sensitivity'] = sens
    sec = {}
    for name, fmt, fn in (('GCN+trust', 'F_entente_gcn_s%d', 'attack_ptfrozen.json'),
                          ('GMed-T eps=1', 'F_entente_gmedn_s%d', 'attack_ptfrozen_ew1.json')):
        st = stats(collect(fmt, fn, c.seeds, two), two)
        sec[name] = st
        show(name, st, two)
    out['secondary'] = sec
    json.dump(out, open('./Exps/seed_summary.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
