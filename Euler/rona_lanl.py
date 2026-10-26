'''
LANL (dataset 2) counterparts of the OpTC RoNA pipeline, plugged into rona_run / rona_attack by
monkeypatching (no existing file is modified):

    import rona_lanl; rona_lanl.patch(); import rona_run as R; R.main()

What patch() swaps
  R.get_args      LANL defaults (cluster file lanl_<K>_<K>.json, tag prefix L_) + LANL-only flags
  R.build_env     LANL loader, node-id cluster map (mc: int node id -> client), LANL time windows
  R.data_kwargs   same signature as the OpTC one
  R.evaluate      chunked-over-time evaluation (GRU state carried between chunks), same metrics dict
  embedders.RGCN  patched in place: c1.lin never multiplies by a dense eye(N) (one-hot input == weight rows)
  RH.CACHE_DIR    ./data/rona_cache_lanl/<cluster stem>/ ; cached TData never stores an N x N matrix

LANL-only flags (stripped before the OpTC parser sees them; stored in the checkpoint as --lanl_cfg
so rona_attack_lanl reproduces exactly the same windows):
  --lanl_data DIR    folder holding fl_lanl/ and lanl_K_K.json   (default env LANL_DATA or
                     ./data/lanl)
  --lanl_delta S     snapshot length in seconds (1800)
  --lanl_tr_end S    end of the training period (150885 = first redteam event)
  --lanl_val S       length of the validation tail (default max(tr_end/20, 2*delta) = 7544)
  --lanl_te_start/--lanl_te_end S   test window (150885 .. 5011199)
  --lanl_chunk N     snapshots per evaluation chunk (50)
  --lanl_cache DIR   cache dir override
'''
import argparse
import json
import os
import pickle
import sys
import time

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rona_run as R
import spinup
from loaders import load_lanl as L
from loaders import rona_hooks as RH
from loaders.tdata import TData
from models import embedders
from models.euler_detector import DetectorEncoder

DEFAULT_DATA = os.environ.get('LANL_DATA', './data/lanl')

LC = {'data': DEFAULT_DATA, 'delta': 1800, 'tr_end': L.DATE_OF_EVIL_LANL, 'val': None,
      'te_start': L.DATE_OF_EVIL_LANL, 'te_end': L.TIMES['all'], 'chunk': 50, 'cache': None}

_ORIG = {}


# ----------------------------------------------------------------------------- one-hot features
class OneHotX:
    '''Stand-in for xs = eye(N): zero storage, answers size()/to() and is understood by OneHotLin.'''

    def __init__(self, n):
        self.n = n

    def size(self, d=None):
        return (self.n, self.n) if d is None else self.n

    @property
    def shape(self):
        return (self.n, self.n)

    def to(self, *a, **k):
        return self

    def __len__(self):
        return self.n


class OneHotLin(nn.Linear):
    '''lin(eye(N)) == weight.T; same parameters/state_dict keys as nn.Linear.'''

    def forward(self, x):
        if isinstance(x, OneHotX):
            return self.weight.t()
        return super().forward(x)


def _encode_range(self, mask_enum, a, b):
    '''Embeddings of snapshots a..b-1 (eval path of RGCN.inner_forward on a sub-batch).'''
    dev, n = self.device, self.data.num_nodes
    eis, ews, trs = [], [], []
    for k, t in enumerate(range(a, b)):
        ei = self.data.ei_masked(mask_enum, t)
        eis.append(ei + k * n)
        ews.append(self.data.ew_masked(mask_enum, t).float())
        tr = self.data.trust_masked(mask_enum, t)
        trs.append(tr.float() if tr is not None else torch.ones(ei.size(1)))
    ei, ew, tr = torch.cat(eis, 1).to(dev), torch.cat(ews).to(dev), torch.cat(trs).to(dev)
    h = self.c1.lin(self.data.xs.to(dev)).repeat(b - a, 1)
    x = self.c1.aggregate(h, ei, ew, tr)
    x = self.drop(self.relu(x))
    x = self.c2(x, ei, ew, tr)
    return self.relu(x).view(b - a, n, -1)


# ----------------------------------------------------------------------------- config / args
def _lanl_parser():
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('--lanl_cfg', default=None)
    p.add_argument('--lanl_data', default=None)
    p.add_argument('--lanl_delta', type=int, default=None)
    p.add_argument('--lanl_tr_end', type=int, default=None)
    p.add_argument('--lanl_val', type=int, default=None)
    p.add_argument('--lanl_te_start', type=int, default=None)
    p.add_argument('--lanl_te_end', type=int, default=None)
    p.add_argument('--lanl_chunk', type=int, default=None)
    p.add_argument('--lanl_cache', default=None)
    return p


def prefix_tag(argv, flags=('--tag', '--transfer_from')):
    argv = list(argv)
    for f in flags:
        if f in argv:
            i = argv.index(f) + 1
            if not argv[i].startswith('L_'):
                argv[i] = 'L_' + argv[i]
    return argv


def _prep(argv):
    ns, rest = _lanl_parser().parse_known_args(argv)
    if ns.lanl_cfg:
        LC.update(json.loads(ns.lanl_cfg))
    for k in ('data', 'delta', 'tr_end', 'val', 'te_start', 'te_end', 'chunk', 'cache'):
        v = getattr(ns, 'lanl_' + k)
        if v is not None:
            LC[k] = v
    rest = prefix_tag(rest)
    if '--cluster_fname' not in rest:
        K = int(rest[rest.index('--K') + 1]) if '--K' in rest else 3
        rest += ['--cluster_fname', 'lanl_%d_%d.json' % (K, K)]
    return rest


def get_args(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    a = _ORIG['get_args'](_prep(argv))
    a.lanl_cfg = json.dumps(LC)      # stored in ckpt -> attack reproduces the same windows
    return a


def _resolve(fname):
    for c in (fname, os.path.join(LC['data'], fname), os.path.join('./data', fname), os.path.join(R.AUX, fname)):
        if os.path.exists(c):
            return c
    raise FileNotFoundError('LANL cluster file %s not found (looked under %s)' % (fname, LC['data']))


def load_clusters(fname, n_nodes):
    mc = L.load_lanl_clusters(_resolve(fname))
    k = max(mc.values()) + 1
    for v in range(n_nodes):                 # ids absent from the json (e.g. the spare id N-1)
        mc.setdefault(v, k)                  # -> outlier cluster, belongs to no client
    return mc


# ----------------------------------------------------------------------------- loader
def _n_nodes():
    with open(os.path.join(LC['data'], 'fl_lanl', 'nmap.pkl'), 'rb') as f:
        return len(pickle.load(f)) + 1


def wrap_loader(loader):
    '''Same behaviour as RH.wrap_loader minus the deepcopy (a freshly loaded/unpickled TData is
    not shared with anybody, and the LANL test set is ~1 GB per client).'''
    def hooked(**kw):
        os.makedirs(RH.CACHE_DIR, exist_ok=True)
        path = RH._cache_path(kw)
        if os.path.exists(path):
            data = torch.load(path, map_location='cpu')
        else:
            data = loader(**kw)
            torch.save(data, path)
        phase, client = RH._phase(kw), kw.get('client', 'All')
        RH._inject(data, phase, client)
        RH._attach_trust(data, phase, client)
        return data
    return hooked


def build_env(a):
    import argparse as ap
    n = _n_nodes()
    L.LANL_FOLDER = os.path.join(LC['data'], 'fl_lanl') + '/'
    L._X_CACHE[n] = OneHotX(n)               # load_lanl.make_data_obj would allocate eye(n) (~1 GB)
    mc = load_clusters(a.cluster_fname, n)
    RH.CFG.update(trust=bool(a.trust), tau0=a.tau0, cap=a.cap, trust_mode=a.trust_mode, mc=mc)
    stem = os.path.splitext(os.path.basename(a.cluster_fname))[0]
    RH.CACHE_DIR = LC['cache'] or './data/rona_cache_lanl/%s/' % stem
    embedders.RGCN.AGG = a.agg
    embedders.RGCN.ITERS = a.iters
    loader = wrap_loader(L.load_lanl_dist)
    spinup.LOAD_FN = loader

    s = ap.Namespace(
        fed=a.fed, rnn=R.GRU, impl='DETECT', dataset='LANL_RONA_' + a.tag, encoder_name='GCN',
        client_list=[str(x) for x in range(a.K)], client_number=a.K, client='Fed', mc=mc,
        fedopt_epoch=5, poison=False, poison_defense=False, nb_defense=False, dp_defense=False,
        cdp_defense=False, personalization=False, equal_weights=False, flows=False,
        similarity=[1.0 / a.K] * a.K, fpweight=a.fpweight)
    delta, tr_end = LC['delta'], LC['tr_end']
    val = LC['val'] or max(tr_end // 20, delta * 2)
    times = {'tr_start': 0, 'tr_end': tr_end - val, 'val_start': tr_end - val, 'val_end': tr_end,
             'te_times': [(LC['te_start'], LC['te_end'])], 'delta': delta}
    RH.CFG['tr_start'] = 0
    tr_args = {'epochs': a.fedT, 'min': 1, 'nratio': 1, 'val_nratio': 1, 'fedT': a.fedT,
               'fedMu': 0.05, 'lr': a.lr, 'patience': 100}
    return s, times, loader, tr_args


def data_kwargs(times, phase, client, mc, device=None):
    return _ORIG['data_kwargs'](times, phase, client, mc, device)


# ----------------------------------------------------------------------------- evaluation
def _tpr_at_fpr(p, y, fpr):
    thr = np.quantile(p[~y], fpr)            # flagged <=> p <= thr
    return float((p[y] <= thr).mean())


def _metrics(p, y, inj, tt, cutoff):
    inj_flag = p[inj] <= cutoff
    alarmed_t = set(tt[inj][inj_flag].tolist())
    keep = ~inj
    p, y, tt = p[keep], y[keep], tt[keep]
    flag = p <= cutoff
    quiet = np.array([t not in alarmed_t for t in tt[y]], dtype=bool)
    has = bool(y.any())
    nan = float('nan')
    res = {
        'AP': float(average_precision_score(y, 1 - p)) if has else nan,
        'AUC': float(roc_auc_score(y, 1 - p)) if has else nan,
        'cutoff': float(cutoff),
        'TPR': float(flag[y].mean()) if has else nan,
        'FPR': float(flag[~y].mean()),
        'SR': float(1 - flag[y].mean()) if has else nan,
        'n_mal': int(y.sum()),
        'n_edges': int(len(y)),
        'n_injected': int(inj.sum()),
        'inj_flag_rate': float(inj_flag.mean()) if inj.any() else 0.0,
        'SR_stealthy': float((~flag[y] & quiet).mean()) if has else nan,
        'mal_score_mean': float(p[y].mean()) if has else nan,
        # LANL extras: AP is ~meaningless at 3.6e-5 prevalence; fixed-FPR operating points
        'TPR@FPR1e-3': _tpr_at_fpr(p, y, 1e-3) if has else nan,
        'TPR@FPR1e-4': _tpr_at_fpr(p, y, 1e-4) if has else nan,
        'n_mal_snapshots': int(len(set(tt[y].tolist()))),
    }
    return res, p, y


def evaluate(model, h0_train, times, s, loader, device, test_inject=None, return_raw=False, chunk=None,
             client='All'):
    assert client == 'All', 'per-silo evaluation is not implemented for LANL'
    '''Chunked version of rona_run.evaluate: encoder + GRU are run over `chunk` snapshots at a
    time and the GRU hidden state is carried across chunks (exactly the same function as the
    all-at-once forward, verified in rona_lanl_smoke.py).'''
    chunk = chunk or LC['chunk']
    model.eval()
    RH.CFG['inject']['val'] = {}
    h0, _ = spinup.get_cutoff(model, h0_train, times, {'val_nratio': 1}, s.fpweight, False, device, 'All', s.mc)
    RH.CFG['inject']['test'] = test_inject or {}
    DetectorEncoder.load_new_data(model.gcns, loader, data_kwargs(times, 'test', 'All', s.mc))
    RH.CFG['inject']['test'] = {}
    enc = model.gcns.module
    enc.eval()
    data = enc.data
    model.rnn = model.rnn.to(device)
    P, Y, INJ, TT = [], [], [], []
    h = h0.to(device)
    with torch.no_grad():
        for a in range(0, data.T, chunk):
            b = min(data.T, a + chunk)
            g = enc.encode_range(TData.TEST, a, b)
            zs, h = model.rnn(g, h, include_h=True)
            for k, t in enumerate(range(a, b)):
                ei = data.eis[t].to(device)
                P.append(model.gcns.decode(ei, zs[k], True).cpu().numpy())
                Y.append((data.ys[t] > 0).numpy())
                INJ.append(data.injected[t].numpy())
                TT.append(np.full(ei.size(1), t, dtype=np.int32))
            del g, zs
    p, y, inj, tt = np.concatenate(P), np.concatenate(Y), np.concatenate(INJ), np.concatenate(TT)
    res, p, y = _metrics(p, y, inj, tt, float(model.cutoff))
    if return_raw:
        return res, {'p': p, 'y': y}
    return res


# ----------------------------------------------------------------------------- patch
def patch():
    if 'get_args' not in _ORIG:
        _ORIG.update(get_args=R.get_args, data_kwargs=R.data_kwargs, evaluate=R.evaluate,
                     build_env=R.build_env)
    R.get_args = get_args
    R.build_env = build_env
    R.data_kwargs = data_kwargs
    R.evaluate = evaluate
    R.OUT_ROOT = './Exps/rona/'
    if not getattr(embedders.RGCN, '_lanl_patched', False):
        # RGCN.__init__ uses super(RGCN, self): patch the class in place, do not subclass
        orig_init = embedders.RGCN.__init__

        def init(self, *a, **k):
            orig_init(self, *a, **k)
            self.c1.lin.__class__ = OneHotLin

        embedders.RGCN.__init__ = init
        embedders.RGCN.encode_range = _encode_range
        embedders.RGCN._lanl_patched = True
    return R
