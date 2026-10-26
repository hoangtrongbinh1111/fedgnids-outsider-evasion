'''
Crossed (seed x host-pair) bootstrap and pair-level reporting for the OpTC evasion results
(uncertainty that respects the data structure: 5 training seeds x 25 malicious
host pairs, and pair-level detection counts).

Inputs: replay files Exps/rona/<tag>/<prefix>_<plan>.npz written by Euler/rona_replay.py, with
per test edge t, src, dst, p (score, low = anomalous), y (malicious), inj (injected edge),
cutoff (validation cutoff). Injected edges are excluded from every metric. A malicious edge is
detected ("flagged") when p <= cutoff; the miss (evasion) rate SR is the fraction of malicious,
non-injected edge occurrences that are not flagged. The 1218 malicious occurrences belong to 25
unique directed (src, dst) host pairs, shared by all seeds and models.

Before use, every npz is checked against the SR/FPR stored in the attack JSON of the same run
(attack.json for models without trust, attack_ptfrozen.json for plug-in frozen trust); the
differences (in malicious occurrences) are written to the output.

Statistics (B replicates, fixed RNG seed, the SAME replicate draws for every model/condition so
that differences are paired):
  crossed   each replicate resamples the 5 seeds with replacement AND the 25 host pairs with
            replacement; one pair resample is applied to every seed (pairs are shared across
            seeds). SR_b = mean over resampled seeds s of
                sum_p w_bp * missed_sp / sum_p w_bp * occ_p
            with w_bp the multiplicity of pair p in replicate b (occurrence-weighted).
  seed-only only seeds are resampled (all pairs, w = 1).
  pair-only only pairs are resampled (all five seeds, each once).
  95% percentile CIs. Paired differences (GMed-T - GCN, GMed-T - Soft Median+trust, attack effect
  b=1 - clean and b=10 - clean per model) use the same replicates; seed index k of two models is
  treated as paired (same training seed). For model contrasts an unpaired-seed variant (each model
  resamples its own seeds; pair resample still shared) is also given. frac_le0 = fraction of
  replicates with difference <= 0.
  TPR at 1% FPR: per model x condition x seed, threshold = 1% quantile of the benign non-injected
  scores (low score = anomalous), detected if p <= threshold; bootstrapped as above (the
  threshold is held fixed within a replicate).
  Pair level: per model x condition x seed, number of the 25 pairs with >= 1 detected occurrence
  and with >= 50% of occurrences detected (at the model's cutoff); mean +- population SD over
  seeds. Per pair: detection fraction averaged over seeds, per model x condition.

    python Euler/rona_bootstrap.py [--B 20000] [--rng 20261025]
writes Exps/rona/bootstrap.json and Exps/rona/bootstrap.md
'''
import argparse
import json
import os

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Exps', 'rona')
SEEDS = [0, 1, 2, 3, 4]
CONDS = ['clean', 'b1', 'b10']

# model -> (tag pattern, {cond: npz basename}, attack JSON with the reference SRs, required)
MODELS = {
    'GCN': ('F_entente_gcn_s%d',
            {'clean': 'replay_clean.npz', 'b1': 'replay_plan_b1.npz', 'b10': 'replay_plan_b10.npz'},
            'attack.json', True),
    'GCN+trust': ('F_entente_gcn_s%d',
                  {'clean': 'replaypt_clean.npz', 'b1': 'replaypt_plan_b1_ptfrozen.npz',
                   'b10': 'replaypt_plan_b10_ptfrozen.npz'},
                  'attack_ptfrozen.json', False),
    'gmedn': ('F_entente_gmedn_s%d',
              {'clean': 'replaynt_clean.npz', 'b1': 'replaynt_plan_b1.npz', 'b10': 'replaynt_plan_b10.npz'},
              'attack.json', False),
    'GMed-T': ('F_entente_gmedn_s%d',
               {'clean': 'replay_clean.npz', 'b1': 'replay_plan_b1_ptfrozen.npz',
                'b10': 'replay_plan_b10_ptfrozen.npz'},
               'attack_ptfrozen.json', True),
    'SoftMed+trust': ('F_entente_softmedT001_s%d',
                      {'clean': 'replaypt_clean.npz', 'b1': 'replaypt_plan_b1_ptfrozen.npz',
                       'b10': 'replaypt_plan_b10_ptfrozen.npz'},
                      'attack_ptfrozen.json', True),
}
CONTRASTS = [('GMed-T', 'GCN'), ('GMed-T', 'SoftMed+trust'), ('GMed-T', 'gmedn'), ('GMed-T', 'GCN+trust')]
TPR_MODELS = ['GCN', 'GMed-T', 'SoftMed+trust', 'GCN+trust', 'gmedn']


def load_run(path):
    d = np.load(path)
    keep = d['inj'] == 0
    y = d['y'][keep].astype(bool)
    p = d['p'][keep].astype(np.float64)
    cut = float(d['cutoff'])
    src, dst = d['src'][keep][y].astype(np.int64), d['dst'][keep][y].astype(np.int64)
    ben = p[~y]
    thr = float(np.quantile(ben, 0.01))
    return dict(src=src, dst=dst, pm=p[y], cut=cut, thr=thr,
                SR=float((p[y] > cut).mean()), FPR=float((ben <= cut).mean()),
                FPR_thr=float((ben <= thr).mean()), n_inj=int(d['inj'].sum()))


def pair_counts(run, pairs):
    idx = {pq: i for i, pq in enumerate(pairs)}
    k = np.array([idx[(a, b)] for a, b in zip(run['src'].tolist(), run['dst'].tolist())])
    occ = np.bincount(k, minlength=len(pairs))
    miss = np.bincount(k, weights=(run['pm'] > run['cut']), minlength=len(pairs))
    miss1 = np.bincount(k, weights=(run['pm'] > run['thr']), minlength=len(pairs))
    return occ, miss, miss1


def ci(x):
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]


def stat(M, occ, Wp, Cs):
    '''M: (5, P) missed counts; occ: (P,); Wp: (B, P) pair multiplicities; Cs: (B, 5) seed
    multiplicities. Returns (B,) bootstrap SR (mean over resampled seeds of occurrence-weighted SR).'''
    per_seed = (Wp @ M.T) / (Wp @ occ)[:, None]  # (B, 5)
    return (Cs * per_seed).sum(1) / Cs.sum(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--B', type=int, default=20000)
    ap.add_argument('--rng', type=int, default=20261025)
    c = ap.parse_args()

    # ---- load and verify
    runs, checks, models = {}, [], []
    for m, (pat, files, jname, required) in MODELS.items():
        paths = {(s, k): os.path.join(ROOT, pat % s, files[k]) for s in SEEDS for k in CONDS}
        missing = [p for p in paths.values() if not os.path.exists(p)]
        if missing:
            if required:
                raise FileNotFoundError(missing)
            print('skip %s (missing %d files)' % (m, len(missing)))
            continue
        models.append(m)
        for (s, k), p in paths.items():
            r = load_run(p)
            runs[m, k, s] = r
            ref = json.load(open(os.path.join(ROOT, pat % s, jname)))[k]
            checks.append(dict(model=m, cond=k, seed=s, npz=os.path.relpath(p, ROOT).replace('\\', '/'),
                               json=(pat % s) + '/' + jname, SR_npz=r['SR'], SR_json=ref['SR'],
                               FPR_npz=r['FPR'], FPR_json=ref['FPR'], n_inj_npz=r['n_inj'],
                               n_inj_json=ref['n_injected'],
                               diff_occ=int(round((r['SR'] - ref['SR']) * len(r['pm'])))))

    first = runs[models[0], 'clean', 0]
    pairs = sorted(set(zip(first['src'].tolist(), first['dst'].tolist())))
    P = len(pairs)
    occ = None
    MISS, MISS1 = {}, {}
    for (m, k, s), r in runs.items():
        o, mi, mi1 = pair_counts(r, pairs)
        if occ is None:
            occ = o
        assert np.array_equal(o, occ), (m, k, s)
        MISS[m, k, s], MISS1[m, k, s] = mi, mi1
    n_occ = int(occ.sum())

    # ---- replicate draws (shared by everything)
    rng = np.random.default_rng(c.rng)
    B = c.B
    Cs = np.stack([np.bincount(rng.integers(0, 5, 5), minlength=5) for _ in range(B)]).astype(float)
    Wp = np.stack([np.bincount(rng.integers(0, P, P), minlength=P) for _ in range(B)]).astype(float)
    Cs_indep = {m: np.stack([np.bincount(rng.integers(0, 5, 5), minlength=5) for _ in range(B)]).astype(float)
                for m in models}
    ones_s, ones_p = np.ones((B, 5)), np.ones((B, P))

    def Mmat(m, k, tpr=False):
        D = MISS1 if tpr else MISS
        return np.stack([D[m, k, s] for s in SEEDS])

    def boot(m, k, tpr=False, Cseed=None):
        M = Mmat(m, k, tpr)
        cs = Cs if Cseed is None else Cseed
        return {'crossed': stat(M, occ, Wp, cs), 'seed': stat(M, occ, ones_p, cs), 'pair': stat(M, occ, Wp, ones_s)}

    out = dict(B=B, rng=c.rng, n_pairs=P, n_occurrences=n_occ, seeds=SEEDS, models=models,
               definitions=__doc__, checks=checks, sr={}, tpr1={}, diffs={}, pairs={}, per_pair=[])
    # ---- SR
    BS = {}
    for m in models:
        for k in CONDS:
            per = np.array([runs[m, k, s]['SR'] for s in SEEDS])
            b = boot(m, k)
            BS[m, k] = b
            out['sr']['%s|%s' % (m, k)] = dict(
                mean=float(per.mean()), sd_pop=float(per.std()), per_seed=per.tolist(),
                ci_crossed=ci(b['crossed']), ci_seed=ci(b['seed']), ci_pair=ci(b['pair']),
                se_crossed=float(b['crossed'].std()), se_seed=float(b['seed'].std()), se_pair=float(b['pair'].std()))
    # ---- TPR at 1% FPR
    BT = {}
    for m in [x for x in TPR_MODELS if x in models]:
        for k in CONDS:
            per = np.array([(runs[m, k, s]['pm'] <= runs[m, k, s]['thr']).mean() for s in SEEDS])
            b = {kk: 1 - v for kk, v in boot(m, k, tpr=True).items()}
            BT[m, k] = b
            out['tpr1']['%s|%s' % (m, k)] = dict(
                mean=float(per.mean()), sd_pop=float(per.std()), per_seed=per.tolist(),
                fpr_realized=[runs[m, k, s]['FPR_thr'] for s in SEEDS],
                ci_crossed=ci(b['crossed']), ci_seed=ci(b['seed']), ci_pair=ci(b['pair']))

    # ---- paired differences
    def dsum(d, point):
        return dict(point=float(point), ci_crossed=ci(d), frac_le0=float((d <= 0).mean()))

    for a, bm in CONTRASTS:
        if a not in models or bm not in models:
            continue
        for k in CONDS:
            pt = out['sr']['%s|%s' % (a, k)]['mean'] - out['sr']['%s|%s' % (bm, k)]['mean']
            d = BS[a, k]['crossed'] - BS[bm, k]['crossed']
            e = dsum(d, pt)
            e['ci_seed'] = ci(BS[a, k]['seed'] - BS[bm, k]['seed'])
            e['ci_pair'] = ci(BS[a, k]['pair'] - BS[bm, k]['pair'])
            # unpaired seeds: each model resamples its own seeds, pair resample shared
            du = stat(Mmat(a, k), occ, Wp, Cs_indep[a]) - stat(Mmat(bm, k), occ, Wp, Cs_indep[bm])
            e['ci_crossed_unpaired_seeds'] = ci(du)
            e['frac_le0_unpaired_seeds'] = float((du <= 0).mean())
            out['diffs']['SR %s - %s | %s' % (a, bm, k)] = e
            if (a, k) in BT and (bm, k) in BT:
                ptt = out['tpr1']['%s|%s' % (a, k)]['mean'] - out['tpr1']['%s|%s' % (bm, k)]['mean']
                out['diffs']['TPR1 %s - %s | %s' % (a, bm, k)] = dsum(BT[a, k]['crossed'] - BT[bm, k]['crossed'], ptt)
    for m in models:
        for k in ('b1', 'b10'):
            pt = out['sr']['%s|%s' % (m, k)]['mean'] - out['sr']['%s|clean' % m]['mean']
            e = dsum(BS[m, k]['crossed'] - BS[m, 'clean']['crossed'], pt)
            e['ci_seed'] = ci(BS[m, k]['seed'] - BS[m, 'clean']['seed'])
            e['ci_pair'] = ci(BS[m, k]['pair'] - BS[m, 'clean']['pair'])
            out['diffs']['SR attack %s: %s - clean' % (m, k)] = e

    # ---- pair level
    for m in models:
        for k in CONDS:
            det = np.stack([occ - MISS[m, k, s] for s in SEEDS])  # (5, P) detected occurrences
            any1 = (det >= 1).sum(1)
            half = (det >= 0.5 * occ).sum(1)
            det1 = np.stack([occ - MISS1[m, k, s] for s in SEEDS])
            out['pairs']['%s|%s' % (m, k)] = dict(
                any_per_seed=any1.tolist(), any_mean=float(any1.mean()), any_sd=float(any1.std()),
                half_per_seed=half.tolist(), half_mean=float(half.mean()), half_sd=float(half.std()),
                any_tpr1_per_seed=(det1 >= 1).sum(1).tolist(),
                half_tpr1_per_seed=(det1 >= 0.5 * occ).sum(1).tolist(),
                missed_occ_per_seed=Mmat(m, k).sum(1).astype(int).tolist())
    for i, (a, b) in enumerate(pairs):
        row = dict(src=int(a), dst=int(b), occ=int(occ[i]), det_frac={})
        for m in models:
            for k in CONDS:
                fr = [(occ[i] - MISS[m, k, s][i]) / occ[i] for s in SEEDS]
                row['det_frac']['%s|%s' % (m, k)] = dict(mean=float(np.mean(fr)), per_seed=[float(x) for x in fr])
        out['per_pair'].append(row)
    # concentration of misses: share of all missed occurrences (summed over seeds) in the worst pairs
    conc = {}
    for m in models:
        for k in CONDS:
            tot = Mmat(m, k).sum(0)
            srt = np.sort(tot)[::-1]
            T = tot.sum()
            conc['%s|%s' % (m, k)] = dict(total_missed=int(T), n_pairs_with_miss=int((tot > 0).sum()),
                                          top1=float(srt[:1].sum() / T) if T else 0.0,
                                          top3=float(srt[:3].sum() / T) if T else 0.0,
                                          top5=float(srt[:5].sum() / T) if T else 0.0)
    out['miss_concentration'] = conc

    with open(os.path.join(ROOT, 'bootstrap.json'), 'w') as f:
        json.dump(out, f, indent=1)
    write_md(out, os.path.join(ROOT, 'bootstrap.md'))
    print('wrote', os.path.join(ROOT, 'bootstrap.json'), os.path.join(ROOT, 'bootstrap.md'))


def pc(x):
    return '%.1f' % (100 * x)


def pci(c):
    return '[%s, %s]' % (pc(c[0]), pc(c[1]))


def write_md(o, fn):
    L = ['# Crossed seed x host-pair bootstrap (OpTC)', '',
         'Generated by Euler/rona_bootstrap.py; B = %d replicates, RNG seed %d; %d malicious occurrences '
         'in %d host pairs; seeds %s. All rates in %%. Injected edges excluded. CIs are 95%% percentile '
         'intervals. crossed = seeds and pairs resampled jointly (same pair draw for every seed); '
         'seed / pair = only that source resampled.' % (o['B'], o['rng'], o['n_occurrences'], o['n_pairs'], o['seeds']),
         '', '## Check of replay files against attack JSONs', '',
         '| model | cond | seed | npz | SR npz | SR json | diff (occ.) | FPR npz | FPR json |', '|---|---|---|---|---|---|---|---|---|']
    for r in o['checks']:
        L.append('| %s | %s | %d | %s | %.2f | %.2f | %+d | %.3f | %.3f |' % (
            r['model'], r['cond'], r['seed'], r['npz'], 100 * r['SR_npz'], 100 * r['SR_json'], r['diff_occ'],
            100 * r['FPR_npz'], 100 * r['FPR_json']))
    L += ['', '## Miss rate SR (%)', '',
          '| model | cond | mean | pop. SD | per seed | CI crossed | CI seed-only | CI pair-only |',
          '|---|---|---|---|---|---|---|---|']
    for key, v in o['sr'].items():
        m, k = key.split('|')
        L.append('| %s | %s | %s | %s | %s | %s | %s | %s |' % (
            m, k, pc(v['mean']), pc(v['sd_pop']), ' / '.join(pc(x) for x in v['per_seed']),
            pci(v['ci_crossed']), pci(v['ci_seed']), pci(v['ci_pair'])))
    L += ['', '## TPR at 1% FPR (%)', '',
          '| model | cond | mean | pop. SD | per seed | CI crossed | CI seed-only | CI pair-only |',
          '|---|---|---|---|---|---|---|---|']
    for key, v in o['tpr1'].items():
        m, k = key.split('|')
        L.append('| %s | %s | %s | %s | %s | %s | %s | %s |' % (
            m, k, pc(v['mean']), pc(v['sd_pop']), ' / '.join(pc(x) for x in v['per_seed']),
            pci(v['ci_crossed']), pci(v['ci_seed']), pci(v['ci_pair'])))
    L += ['', '## Paired differences (percentage points)', '',
          'Model contrasts pair seed k of both models; "unpaired seeds" resamples seeds independently per model '
          '(pair draw still shared). frac<=0 = fraction of replicates with difference <= 0. For SR differences a '
          'negative value favours the first model (fewer misses) or means the attack did not raise misses; for '
          'TPR1 differences a positive value favours the first model.', '',
          '| difference | point | CI crossed | frac<=0 | CI seed-only | CI pair-only | CI unpaired seeds | frac<=0 unpaired |',
          '|---|---|---|---|---|---|---|---|']
    for key, v in o['diffs'].items():
        L.append('| %s | %s | %s | %.4f | %s | %s | %s | %s |' % (
            key, pc(v['point']), pci(v['ci_crossed']), v['frac_le0'],
            pci(v['ci_seed']) if 'ci_seed' in v else '', pci(v['ci_pair']) if 'ci_pair' in v else '',
            pci(v['ci_crossed_unpaired_seeds']) if 'ci_crossed_unpaired_seeds' in v else '',
            '%.4f' % v['frac_le0_unpaired_seeds'] if 'frac_le0_unpaired_seeds' in v else ''))
    L += ['', '## Pairs detected (of %d), at the model cutoff' % o['n_pairs'], '',
          '| model | cond | >=1 occ. detected: mean +- SD | per seed | >=50% detected: mean +- SD | per seed | '
          '>=1 / >=50% at 1% FPR per seed | missed occ. per seed |',
          '|---|---|---|---|---|---|---|---|']
    for key, v in o['pairs'].items():
        m, k = key.split('|')
        L.append('| %s | %s | %.1f +- %.1f | %s | %.1f +- %.1f | %s | %s / %s | %s |' % (
            m, k, v['any_mean'], v['any_sd'], v['any_per_seed'], v['half_mean'], v['half_sd'], v['half_per_seed'],
            v['any_tpr1_per_seed'], v['half_tpr1_per_seed'], v['missed_occ_per_seed']))
    L += ['', '## Concentration of misses (summed over seeds)', '',
          '| model | cond | missed occ. | pairs with >=1 miss | top-1 share | top-3 share | top-5 share |',
          '|---|---|---|---|---|---|---|']
    for key, v in o['miss_concentration'].items():
        m, k = key.split('|')
        L.append('| %s | %s | %d | %d | %s | %s | %s |' % (m, k, v['total_missed'], v['n_pairs_with_miss'],
                                                          pc(v['top1']), pc(v['top3']), pc(v['top5'])))
    cols = ['%s|%s' % (m, k) for m in o['models'] for k in CONDS]
    L += ['', '## Per pair: detected fraction (%) averaged over seeds, at the model cutoff', '',
          '| src | dst | occ. | ' + ' | '.join(c.replace('|', ' ') for c in cols) + ' |',
          '|---|---|---|' + '---|' * len(cols)]
    for r in sorted(o['per_pair'], key=lambda r: -r['occ']):
        L.append('| %d | %d | %d | ' % (r['src'], r['dst'], r['occ']) +
                 ' | '.join('%.0f' % (100 * r['det_frac'][c]['mean']) for c in cols) + ' |')
    with open(fn, 'w') as f:
        f.write('\n'.join(L) + '\n')


if __name__ == '__main__':
    main()
