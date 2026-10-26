'''
LANL entry point of the RoNA training/evaluation driver (same CLI as rona_run.py, plus the
--lanl_* flags documented in rona_lanl.py). Outputs go to ./Exps/rona/L_<tag>/.

    python Euler/rona_run_lanl.py --fed Momentum --agg gmedn --trust 1 --fedT 30 --tag L_entente_gmedn_trust_s0
'''
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rona_lanl

R = rona_lanl.patch()

if __name__ == '__main__':
    R.main()
