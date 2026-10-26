'''
CIC-IDS-2017 entry point of the RoNA training/evaluation driver (same CLI as rona_run.py, plus the
--lanl_* flags documented in rona_cic.py). Outputs go to ./Exps/rona/C_<tag>/.

    python Euler/rona_run_cic.py --fed Momentum --agg gmedn --trust 1 --fedT 30 --tag C_entente_gmedn_trust_s0
'''
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rona_cic

R = rona_cic.patch()

if __name__ == '__main__':
    R.main()
