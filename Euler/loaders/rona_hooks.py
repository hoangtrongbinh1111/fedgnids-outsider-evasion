'''
Hooks around the Euler/OpTC loader used by the RoNA experiments.

wrap_loader(loader) returns a loader with the same signature that
  1. caches the raw TData on disk (loading OpTC text files is slow),
  2. injects attacker edges (covering / evasion edges) into the requested phase,
     respecting the 1-hop augmentation semantics of ENTENTE: an injected edge is
     visible to client k iff its src or dst belongs to k (or client is All/Fed),
  3. attaches causal temporal-trust weights: for an edge in snapshot t, the
     number of *earlier* snapshots (train history + earlier snapshots of the same
     phase) in which the same directed edge was observed.

Phases are inferred from the loader kwargs:
  train : not is_test and start == CFG['tr_start']
  val   : not is_test and start != CFG['tr_start']   (tail of the training period)
  test  : is_test
'''
import hashlib
import json
import os
from collections import Counter
from copy import deepcopy

import numpy as np
import torch

from models.robust_conv import temporal_trust

CACHE_DIR = './data/rona_cache/'

CFG = {
    'tr_start': 0,
    'trust': False,     # attach temporal trust
    'tau0': 0.1,
    'cap': 3,
    # 'causal': test-time counts keep accumulating within the test phase
    # 'frozen': test-time trust only uses the (attack-free) train+val profile
    'trust_mode': 'frozen',
    'mc': None,         # machine -> cluster map, for 1-hop visibility of injected edges
    # phase -> {snapshot index: np.ndarray of shape (2, m)} edges to inject
    'inject': {'train': {}, 'val': {}, 'test': {}},
    'inject_ew': 0.5,   # standardized weight of an injected (low-count) edge
    # idealized attacker whose covering edges are all familiar (each counts
    # as seen >= cap times, i.e. full trust), e.g. footholds that routinely talk to the targets
    'inject_familiar': False,
}

# client -> {'train': Counter, 'val': Counter} of edge -> #snapshots containing it
HISTORY = {}
# extra history counts used only for the test phase (idealized
# familiar-path attacker whose malicious host pairs count as previously seen)
EXTRA_TEST_COUNTS = Counter()


def _phase(kw):
    if kw.get('is_test', False):
        return 'test'
    return 'train' if kw['start'] == CFG['tr_start'] else 'val'


def _cache_path(kw):
    key = {k: kw[k] for k in ('start', 'end', 'delta', 'is_test', 'use_flows', 'client')}
    # a per-silo view depends on the silo partition; the default partition keeps its old keys
    if kw.get('client', 'All') not in ('All', 'Fed') and CFG.get('cluster_tag'):
        key['cluster'] = CFG['cluster_tag']
    h = hashlib.md5(json.dumps(key, sort_keys=True).encode()).hexdigest()[:16]
    return os.path.join(CACHE_DIR, '%s_%s.pt' % (_phase(kw), h))


def _visible(src, dst, client):
    if client in ('All', 'Fed'):
        return np.ones(len(src), dtype=bool)
    mc = CFG['mc']
    # nodes outside the machine map never occur in the logs and belong to no silo
    return np.array([str(mc.get(int(s))) == client or str(mc.get(int(d))) == client
                     for s, d in zip(src, dst)], dtype=bool)


def _edge_filter(data, t, keep):
    '''drop the edges of snapshot t where keep is False, in every per-edge field'''
    m = keep.numel()
    data.eis[t] = data.eis[t][:, keep]
    for name in ('ews', 'ys', 'cnt', 'eas', 'masks'):
        f = getattr(data, name, None)
        if not (isinstance(f, list) and len(f) > t and torch.is_tensor(f[t]) and f[t].dim() >= 1):
            continue
        if f[t].size(0) == m:
            f[t] = f[t][keep]
        elif f[t].dim() == 2 and f[t].size(1) == m:
            f[t] = f[t][:, keep]


def _synthesize(data, client):
    '''Semi-synthetic test set: remove the real red-team edges and add
    synthetic malicious edges CFG['synth']['edges'] = {t: 2 x m}, labelled malicious, with
    weight CFG['synth']['ew'] and count 1. Visibility follows the 1-hop rule of _visible.'''
    syn = CFG.get('synth')
    n = data.num_nodes
    for t in range(data.T):
        if syn.get('remove_real', True):
            _edge_filter(data, t, ~(data.ys[t] > 0))
        edges = syn['edges'].get(t)
        if edges is None or len(edges[0]) == 0:
            continue
        src, dst = np.asarray(edges[0]), np.asarray(edges[1])
        keep = _visible(src, dst, client)
        ei = data.eis[t]
        existing = set((ei[0] * n + ei[1]).tolist())
        new = [(s, d) for s, d in zip(src[keep].tolist(), dst[keep].tolist())
               if s != d and s * n + d not in existing]
        new = list(dict.fromkeys(new))
        if not new:
            continue
        add = torch.tensor(new, dtype=ei.dtype).t()
        m = add.size(1)
        data.eis[t] = torch.cat([ei, add], dim=1)
        data.ews[t] = torch.cat([data.ews[t], torch.full((m,), float(syn.get('ew', 0.5)), dtype=data.ews[t].dtype)])
        data.ys[t] = torch.cat([data.ys[t], torch.ones(m, dtype=data.ys[t].dtype)])
        if getattr(data, 'cnt', None) is not None:
            data.cnt[t] = torch.cat([data.cnt[t], torch.ones(m, dtype=data.cnt[t].dtype)])


def _inject(data, phase, client):
    if phase == 'test' and CFG.get('synth'):
        _synthesize(data, client)
    data.injected = [torch.zeros(ei.size(1), dtype=torch.bool) for ei in data.eis]
    plan = CFG['inject'].get(phase, {})
    n = data.num_nodes
    for t, edges in plan.items():
        if t >= data.T or edges is None or len(edges[0]) == 0:
            continue
        src, dst = np.asarray(edges[0]), np.asarray(edges[1])
        keep = _visible(src, dst, client)
        src, dst = src[keep], dst[keep]
        if len(src) == 0:
            continue
        ei = data.eis[t]
        existing = set((ei[0] * n + ei[1]).tolist())
        new = [(s, d) for s, d in zip(src.tolist(), dst.tolist())
               if s != d and s * n + d not in existing]
        new = list(dict.fromkeys(new))
        if not new:
            continue
        add = torch.tensor(new, dtype=ei.dtype).t()
        m = add.size(1)
        data.eis[t] = torch.cat([ei, add], dim=1)
        data.ews[t] = torch.cat([data.ews[t], torch.full((m,), CFG['inject_ew'])])
        if data.cnt is not None:
            data.cnt[t] = torch.cat([data.cnt[t], torch.ones(m, dtype=data.cnt[t].dtype)])
        if data.is_test:
            data.ys[t] = torch.cat([data.ys[t], torch.zeros(m, dtype=data.ys[t].dtype)])
        else:
            # injected edges are part of the training signal, never held out
            data.masks[t] = torch.cat([data.masks[t], torch.ones(m, dtype=torch.bool)])
        data.injected[t] = torch.cat([data.injected[t], torch.ones(m, dtype=torch.bool)])
    data.ei_sizes = [(data.ei_masked(data.TRAIN, t).size(1), data.ei_masked(data.VAL, t).size(1),
                      data.eis[t].size(1)) for t in range(data.T)]


def _attach_trust(data, phase, client):
    n = data.num_nodes
    hist = HISTORY.setdefault(client, {'train': Counter(), 'val': Counter()})
    base = Counter()
    if phase == 'val':
        base = Counter(hist['train'])
    elif phase == 'test':
        base = hist['train'] + hist['val'] + EXTRA_TEST_COUNTS
    seen = Counter()
    trust, counts = [], []
    for t in range(data.T):
        keys = (data.eis[t][0] * n + data.eis[t][1]).tolist()
        c = torch.tensor([base[k] + seen[k] for k in keys])
        if phase == 'test' and CFG.get('inject_familiar') and getattr(data, 'injected', None) is not None:
            c = torch.where(data.injected[t], c.clamp(min=CFG['cap']), c)
        counts.append(c)
        trust.append(temporal_trust(c, CFG['tau0'], CFG['cap']) if CFG['trust']
                     else torch.ones(len(keys)))
        if phase != 'test' or CFG['trust_mode'] == 'causal':
            for k in set(keys):
                seen[k] += 1
    if phase in ('train', 'val'):
        hist[phase] = seen
    data.trust = trust
    data.hist_counts = counts


def wrap_loader(loader):
    def hooked(**kw):
        os.makedirs(CACHE_DIR, exist_ok=True)
        path = _cache_path(kw)
        if os.path.exists(path):
            data = torch.load(path, map_location='cpu')
        else:
            data = loader(**kw)
            torch.save(data, path)
        data = deepcopy(data)
        phase, client = _phase(kw), kw.get('client', 'All')
        _inject(data, phase, client)
        _attach_trust(data, phase, client)
        return data
    return hooked
