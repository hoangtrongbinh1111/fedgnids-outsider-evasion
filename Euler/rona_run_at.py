'''
Training entry point for the adversarial-training baseline (see rona_at.py). Same as
rona_run_b.py plus the AT options below; they are kept out of the checkpoint args (written to
<tag>/at_config.json instead), so rona_attack.py / rona_attack_b.py load the model unchanged.

    python Euler/rona_run_at.py --tag F_entente_gcnat_s0 --fed Momentum --agg gcn --fedT 30 \
        --seed 0 --at_k 4 --at_ncand 12 --at_lam 1
'''
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rona_run_b  # noqa: E402,F401  (extra aggregators + patched get_args)
import rona_at  # noqa: E402
import rona_run as R  # noqa: E402

pre = argparse.ArgumentParser(add_help=False)
pre.add_argument('--at_k', type=int, default=4, help='pseudo-anomalous pairs per training snapshot')
pre.add_argument('--at_ncand', type=int, default=12, help='candidate covering edges per pair')
pre.add_argument('--at_budget', type=int, default=1, help='covering edges inserted per pair')
pre.add_argument('--at_lam', type=float, default=1.0, help='weight of the pseudo-anomaly loss')
pre.add_argument('--at_ew', type=float, default=0.5, help='weight of inserted edges')
pre.add_argument('--at_steps', type=int, default=1, help='PGD steps of the inner maximization (1 = FGSM-style)')
pre.add_argument('--at_lr', type=float, default=0.3, help='PGD step size (fraction of the mask range)')
AT, rest = pre.parse_known_args(sys.argv[1:])
sys.argv = [sys.argv[0]] + rest

if __name__ == '__main__':
    a = R.get_args()
    rona_at.configure(on=True, k=AT.at_k, ncand=AT.at_ncand, budget=AT.at_budget, lam=AT.at_lam,
                      ew=AT.at_ew, seed=a.seed, steps=AT.at_steps, lr=AT.at_lr)
    out = os.path.join(R.OUT_ROOT, a.tag)
    os.makedirs(out, exist_ok=True)
    json.dump(dict(rona_at.AT), open(os.path.join(out, 'at_config.json'), 'w'), indent=1)
    R.main()
