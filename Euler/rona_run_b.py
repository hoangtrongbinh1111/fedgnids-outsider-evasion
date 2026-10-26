'''
Entry point for the baseline-aggregator runs: same as rona_run.py but --agg also accepts
softmed | rung | rungl1 | median | trmean | guardn (see models/robust_conv_ext.py).
    python Euler/rona_run_b.py --tag F_entente_softmed_s0 --fed Momentum --agg softmed ...
'''
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import models.robust_conv_ext as X  # noqa: E402  (registers the aggregators)
import rona_run as R  # noqa: E402

_orig_get_args = R.get_args


def get_args(argv=None):
    '''Parses with the original parser, hiding new --agg values behind a legal placeholder.'''
    argv = list(sys.argv[1:] if argv is None else argv)
    # --softmed_T: Soft Median temperature (default 1.0 = code default of Geisler et al.); it is
    # stored in the checkpoint args so that attacks rebuild the same aggregator
    softmed_T = None
    if '--softmed_T' in argv:
        i = argv.index('--softmed_T')
        softmed_T = float(argv[i + 1])
        del argv[i:i + 2]
        X.CFG['softmed_T'] = softmed_T
    # --rung_gamma: MCP threshold of RUNG (default 3.0); also stored in the checkpoint args
    rung_gamma = None
    if '--rung_gamma' in argv:
        i = argv.index('--rung_gamma')
        rung_gamma = float(argv[i + 1])
        del argv[i:i + 2]
        X.CFG['rung_gamma'] = rung_gamma
    new_agg = None
    for i, tok in enumerate(argv):
        if tok == '--agg' and i + 1 < len(argv) and argv[i + 1] in X.NEW_AGGS:
            new_agg, argv[i + 1] = argv[i + 1], 'gcn'
        elif tok.startswith('--agg=') and tok[6:] in X.NEW_AGGS:
            new_agg, argv[i] = tok[6:], '--agg=gcn'
    a = _orig_get_args(argv)
    if new_agg is not None:
        a.agg = new_agg
    if softmed_T is not None:
        a.softmed_T = softmed_T
    if rung_gamma is not None:
        a.rung_gamma = rung_gamma
    return a


R.get_args = get_args

if __name__ == '__main__':
    R.main()
