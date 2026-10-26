'''
CIC-IDS-2017 entry point of the white-box edge-injection attack (same CLI as rona_attack.py).
Model from ./Exps/rona/C_<tag>/ckpt.pt written by rona_run_cic.py.

    python Euler/rona_attack_cic.py --tag C_entente_gcn_s0 --budgets 1 5 20
'''
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rona_cic

rona_cic.patch()
import rona_attack as A

if __name__ == '__main__':
    sys.argv = [sys.argv[0]] + rona_cic.prefix_tag(sys.argv[1:])
    A.main()
