'''
LANL entry point of the white-box edge-injection attack (same CLI as rona_attack.py). The model is
loaded from ./Exps/rona/<tag>/ckpt.pt written by rona_run_lanl.py; the LANL windows/cluster file are
restored from the checkpoint (--lanl_cfg).

    python Euler/rona_attack_lanl.py --tag L_entente_gcn_s0 --budgets 1 2 5 10
'''
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rona_lanl

rona_lanl.patch()
import rona_attack as A

if __name__ == '__main__':
    sys.argv = [sys.argv[0]] + rona_lanl.prefix_tag(sys.argv[1:])
    A.main()
