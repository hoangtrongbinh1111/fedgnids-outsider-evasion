'''Run another rona_* script with the baseline aggregators of rona_run_b registered
(--agg gatn, gat, softmed, ...):  python Euler/rona_b_wrap.py Euler/rona_replay.py --tag ...'''
import os
import runpy
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rona_run_b  # noqa: E402,F401  (patches rona_run.get_args, imports robust_conv_ext)

script = sys.argv[1]
sys.argv = sys.argv[1:]
runpy.run_path(script, run_name='__main__')
