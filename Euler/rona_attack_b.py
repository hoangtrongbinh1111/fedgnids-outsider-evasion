'''
Entry point for attacks on baseline-aggregator checkpoints (and hard-drop filter baselines).
Same as rona_attack.py plus
  * the new --agg names (via rona_run_b),
  * X.ATTACK_SURROGATE=True during PGD optimisation only (adaptive attack on median/trmean,
    which are piecewise constant in the edge weights); the final R.evaluate with the discrete
    plan uses the exact aggregators,
  * --tau0 / --cap overrides of the test-time temporal trust (require --plugin_trust), e.g.
    `--plugin_trust frozen --tau0 0 --cap 1` = hard-drop every edge unseen in train+val
    (T-SHIELD-like filter). Output files get the suffix _t<tau0>_c<cap> so they never collide
    with attack*.json / plan_b*.pt of the plain runs.
'''
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rona_run_b  # noqa: E402,F401  (patches rona_run.get_args, imports robust_conv_ext)
import models.robust_conv_ext as X  # noqa: E402
import rona_attack as A  # noqa: E402
from loaders import rona_hooks as RH  # noqa: E402

pre = argparse.ArgumentParser(add_help=False)
pre.add_argument('--tau0', type=float, default=None)
pre.add_argument('--cap', type=int, default=None)
OVR, rest = pre.parse_known_args(sys.argv[1:])
sys.argv = [sys.argv[0]] + rest
if (OVR.tau0 is not None or OVR.cap is not None) and '--plugin_trust' not in rest:
    sys.exit('--tau0/--cap overrides require --plugin_trust frozen|causal')

_load_model, _attack = A.load_model, A.attack


def load_model(*a, **k):
    out = _load_model(*a, **k)
    if OVR.tau0 is not None:
        RH.CFG['tau0'] = OVR.tau0
    if OVR.cap is not None:
        RH.CFG['cap'] = OVR.cap
    return out


def attack(*a, **k):
    X.ATTACK_SURROGATE = True
    try:
        return _attack(*a, **k)
    finally:
        X.ATTACK_SURROGATE = False


A.load_model, A.attack = load_model, attack

if OVR.tau0 is not None or OVR.cap is not None:
    class _Path:
        def __getattr__(self, k):
            return getattr(os.path, k)

        def join(self, *p):
            r = os.path.join(*p)
            base = os.path.basename(r)
            ext = None
            if base.startswith('attack') and base.endswith('.json'):
                ext = '.json'
            elif base.startswith('plan_b') and base.endswith('.pt'):
                ext = '.pt'
            if ext:
                tag = '_t%g_c%g' % (RH.CFG['tau0'], RH.CFG['cap'])
                r = r[:-len(ext)] + tag + ext
            return r

    class _OS:
        path = _Path()

        def __getattr__(self, k):
            return getattr(os, k)

    A.os = _OS()

if __name__ == '__main__':
    A.main()
