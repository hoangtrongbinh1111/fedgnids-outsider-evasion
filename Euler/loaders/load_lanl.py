"""
LANL loader for ENTENTE's Euler pipeline (reconstructed; the release omitted it).

Derived from (a) iHeartGraph/Euler lanl_experiments/loaders/load_lanl.py (4-column
files ts,src,dst,label) and (b) ENTENTE's own load_optc.py (client filtering with 1-hop
augmentation, TData with slices/eas/use_flows, signature load_*_dist(**kw)).

Input: ENTENTE artifact data/fl_lanl/<10000*k>.txt, produced by split_lanl.split_auth():
    ts,src,dst,src_dom,dst_dom,auth_type,logon_type,auth_orient,success,label,src_user,dst_user
 (12 columns, NTLM auth events only, node ids from nmap.pkl, label = redteam edge)
and data/fl_lanl/nmap.pkl (list of 15610 machine names; list index = node id).
Client map: data/lanl_<K>_<K>.json, 'bottom' level, keys are *node ids as strings*
(NOT machine names, so run.py:load_machine_clusters(..., 'LANL') cannot be used; use
load_lanl_clusters below).

Edge features (cols 3-8, 10-11) are not used (use_flows=False in all our runs).
"""
from copy import deepcopy
import json
import os
import pickle

import torch

from .tdata import TData
from .load_utils import edge_tv_split, std_edge_w, standardized, std_edge_a

DATE_OF_EVIL_LANL = 150885          # first redteam event (original Euler constant)
FILE_DELTA = 10000
LANL_FOLDER = './data/fl_lanl/'
LABEL_COL = 9

TIMES = {
    '20':   228642,    # first 20 anoms
    '100':  740104,    # first 100 anoms
    '500':  1089597,   # first 500 anoms
    'all':  5011199,   # full
    'test': 10000,
}

torch.set_num_threads(1)

_X_CACHE = {}


def load_lanl_clusters(cluster_fname, level='bottom'):
    """{node id (int): cluster index} from ENTENTE's lanl_K_K.json (ids already numeric)."""
    clusters = json.load(open(cluster_fname))[level]
    mc = {}
    for ci, machines in enumerate(clusters):
        for m in machines.keys():
            mc[int(m)] = ci
    return mc


def empty_lanl(use_flows=False):
    return make_data_obj(None, [], None, None, None, use_flows=use_flows)


def load_lanl_dist(start=0, end=635015, delta=1800, is_test=False, use_flows=False,
                   ew_fn=std_edge_w, ea_fn=std_edge_a, client='All', mc=None, **_):
    if start is None or end is None:
        return empty_lanl(use_flows)
    return load_partial_lanl(start, end, delta, is_test, use_flows, ew_fn, ea_fn, client, mc)


def make_data_obj(cur_slice, eis, ys, ew_fn, ea_fn, ews=None, eas=None, use_flows=False, **kwargs):
    if 'node_map' in kwargs:
        nm = kwargs['node_map']
    else:
        nm = pickle.load(open(LANL_FOLDER + 'nmap.pkl', 'rb'))
    n = len(nm) + 1
    # one shared identity matrix (15611^2 floats ~ 0.98 GB) instead of one per TData
    if n not in _X_CACHE:
        _X_CACHE[n] = torch.eye(n)
    x = _X_CACHE[n]

    eis_t, masks = [], []
    for i in range(len(eis)):
        ei = torch.tensor(eis[i])
        eis_t.append(ei)
        if isinstance(ys, None.__class__):   # training data: random train/val edge mask
            masks.append(edge_tv_split(ei)[0])

    if not isinstance(ews, None.__class__):
        cnt = deepcopy(ews)
        ews = ew_fn(ews)
    else:
        cnt = None
    return TData(cur_slice, eis_t, x, ys, masks, ews=ews, eas=eas, use_flows=use_flows,
                 cnt=cnt, node_map=nm)


def load_partial_lanl(start=0, end=150885, delta=1800, is_test=False, use_flows=False,
                      ew_fn=standardized, ea_fn=std_edge_a, client='All', mc=None):
    if use_flows:
        raise NotImplementedError('edge features of LANL are not wired in this port')
    print('client ' + str(client) + ' start:' + str(start) + ', end:' + str(end))
    cur_slice = int(start - (start % FILE_DELTA))
    in_f = open(LANL_FOLDER + str(cur_slice) + '.txt', 'r')

    edges, ews, ys, slices, eas = [], [], [], [], []
    edges_t = {}
    node_map = pickle.load(open(LANL_FOLDER + 'nmap.pkl', 'rb'))
    filt = client not in ('All', 'Fed')

    def add_edge(et, is_anom=0):
        if et in edges_t:
            val = edges_t[et]
            edges_t[et] = (max(is_anom, val[0]), val[1] + 1)
        else:
            edges_t[et] = (is_anom, 1)

    keep_reading = True
    next_split = start + delta
    line = in_f.readline()
    while keep_reading:
        while line:
            l = line.split(',')
            ts = int(l[0])
            if ts < start:
                line = in_f.readline()
                continue
            src, dst, label = int(l[1]), int(l[2]), int(l[LABEL_COL])

            # federated client: keep an edge if src OR dst belongs to it (1-hop augmentation)
            if filt and str(mc[src]) != str(client) and str(mc[dst]) != str(client):
                line = in_f.readline()
                continue

            # close the current window
            if ts >= next_split:
                if len(edges_t):
                    edges.append(list(zip(*edges_t.keys())))
                    y, ew = list(zip(*edges_t.values()))
                    ews.append(torch.tensor(ew))
                    if is_test:
                        ys.append(torch.tensor(y))
                    slices.append(str(ts))
                    edges_t = {}
                next_split += delta
                if ts >= end:
                    keep_reading = False
                    break

            if src == dst:          # skip self-loops
                line = in_f.readline()
                continue
            add_edge((src, dst), is_anom=label)
            line = in_f.readline()

        in_f.close()
        if not keep_reading:
            break
        cur_slice += FILE_DELTA
        nxt = LANL_FOLDER + str(cur_slice) + '.txt'
        if os.path.exists(nxt):
            in_f = open(nxt, 'r')
            line = in_f.readline()
        else:
            break

    ys = ys if is_test else None
    sizes = [len(e[1]) for e in edges]
    if sizes:
        print('snapshots:', len(edges), 'mean edges:', sum(sizes) / len(sizes),
              'max:', max(sizes), 'min:', min(sizes))
    return make_data_obj(slices, edges, ys, ew_fn, ea_fn, ews=ews, eas=eas,
                         use_flows=use_flows, node_map=node_map)
