'''
Adversarial training (AT) against covering edges, adapted to the unsupervised link-prediction
detector (baseline in the spirit of Xu et al., IJCAI'19 and Galli et al.).

The detector has no malicious labels at training time, so sampled pseudo-anomalies stand in
for malicious edges. In every training forward pass (each client, each local step):
  1. per training snapshot t, sample k pseudo-anomalous pairs (u, v): u a source and v a
     destination active in t, (u, v) not an edge of t. Like a malicious edge at test time, the
     pseudo-anomalous edge u->v is present in the message-passing graph (weight inject_ew);
  2. candidate covering edges originate from u (the "compromised" source) and end at v's
     in-neighbours in t or at random active destinations (at most `ncand` per pair, existing
     edges excluded), mirroring the attacker of rona_attack.py;
  3. inner maximization: one gradient step (FGSM-style, as in fast adversarial training) of the
     attacker loss -log p(u, v) w.r.t. a continuous mask over the candidates, through the
     encoder AND the GRU; per pair the `budget` candidates with the most negative gradient are
     inserted (weight inject_ew). With steps > 1, `steps` projected gradient
     steps of size lr are taken instead and the `budget` largest mask entries per pair are kept;
  4. outer minimization: the usual BCE loss plus  lam * mean_t -log(1 - p_t(u, v))  on the
     pseudo-anomalies of the perturbed graph.
Evaluation and attacks are unchanged (AT is active only for TRAIN forward passes with grad).
'''
import numpy as np
import torch

from loaders.tdata import TData
from models import embedders as E
from models import euler_interface as EI
from models.euler_detector import DetectorEncoder

AT = dict(on=False, k=4, ncand=12, budget=1, lam=1.0, ew=0.5, seed=0, steps=1, lr=0.3)
_rng = np.random.RandomState(0)


def configure(**kw):
    global _rng
    AT.update(kw)
    _rng = np.random.RandomState(AT['seed'])


def _static(data):
    '''per training snapshot: edge keys and in-neighbour lists (cached on the data object)'''
    st = getattr(data, '_at_static', None)
    if st is not None:
        return st
    n = data.num_nodes
    st = []
    for t in range(data.T):
        ei = data.ei_masked(TData.TRAIN, t).cpu().numpy()
        keys = set((ei[0].astype(np.int64) * n + ei[1]).tolist())
        order = np.argsort(ei[1], kind='stable')
        dsts, starts = np.unique(ei[1][order], return_index=True)
        ends = np.append(starts[1:], len(order))
        inn = {int(d): ei[0][order][s:e] for d, s, e in zip(dsts, starts, ends)}
        st.append((ei, keys, inn, np.unique(ei[0]), dsts))
    data._at_static = st
    return st


def _sample(data):
    '''pseudo-anomalous pairs per snapshot (local ids) and candidate edges (batched ids)'''
    n = data.num_nodes
    pairs, cand, owner = [], [], []
    npair = 0
    for t, (ei, keys, inn, srcs, dsts) in enumerate(_static(data)):
        if len(srcs) == 0 or len(dsts) < 2:
            pairs.append(np.zeros((2, 0), np.int64))
            continue
        us = srcs[_rng.randint(len(srcs), size=AT['k'])]
        vs = dsts[_rng.randint(len(dsts), size=AT['k'])]
        ok = [(int(u), int(v)) for u, v in zip(us, vs) if u != v and u * n + v not in keys]
        ok = list(dict.fromkeys(ok))
        pairs.append(np.array(ok, np.int64).T.reshape(2, -1))
        for u, v in ok:
            ws = np.concatenate([inn.get(v, np.zeros(0, np.int64)), dsts[_rng.randint(len(dsts), size=4)]])
            ws = np.unique(ws)
            ws = ws[(ws != u) & (ws != v)]
            ws = np.array([w for w in ws if u * n + w not in keys], np.int64)
            if len(ws) > AT['ncand']:
                ws = _rng.choice(ws, AT['ncand'], replace=False)
            for w in ws:
                cand.append((t * n + u, t * n + w))
                owner.append(npair)
            npair += 1
    cand = np.array(cand, np.int64).T.reshape(2, -1)
    return pairs, cand, np.array(owner, np.int64), npair


def _pair_edges(pairs, n):
    e = [p + t * n for t, p in enumerate(pairs) if p.shape[1]]
    return np.concatenate(e, 1) if e else np.zeros((2, 0), np.int64)


# ---- encoder: append the extra (pseudo-anomalous + covering) edges to the batched graph ----
_orig_inner = E.RGCN.inner_forward


def _inner_forward(self, mask_enum):
    extra = getattr(self, '_at_extra', None)
    if extra is None or mask_enum != TData.TRAIN:
        return _orig_inner(self, mask_enum)
    T, n = self.data.T, self.data.num_nodes
    ei, ew, tr = self._batched(mask_enum)
    xe, xw = extra
    ei = torch.cat([ei, xe.to(self.device)], 1)
    ew = torch.cat([ew, xw.to(self.device)])
    tr = torch.cat([tr, torch.ones(xe.size(1), device=self.device)])
    h = self.c1.lin(self.data.xs.to(self.device)).repeat(T, 1)
    if torch.is_grad_enabled() and self.AGG != 'gcn':
        from torch.utils.checkpoint import checkpoint
        x = checkpoint(self.c1.aggregate, h, ei, ew, tr)
        x = self.drop(self.relu(x))
        x = checkpoint(self.c2.aggregate, self.c2.lin(x), ei, ew, tr)
    else:
        x = self.c1.aggregate(h, ei, ew, tr)
        x = self.drop(self.relu(x))
        x = self.c2(x, ei, ew, tr)
    return self.relu(x).view(T, n, -1)


E.RGCN.inner_forward = _inner_forward


# ---- recurrent model: inner maximization before the training forward pass ----
_orig_forward = EI.Euler_Recurrent.forward


def _pair_probs(enc, zs, pairs):
    ps = []
    for t, p in enumerate(pairs):
        if p.shape[1]:
            pt = torch.from_numpy(p).to(zs.device)
            ps.append(enc.decode(pt[0], pt[1], zs[t]))
    return torch.cat(ps) if ps else None


def _forward(self, mask_enum, include_h=False, h0=None, no_grad=False):
    enc = self.gcns.module
    enc._at_extra, enc._at_pairs = None, None
    if not (AT['on'] and mask_enum == TData.TRAIN and not no_grad and torch.is_grad_enabled()):
        return _orig_forward(self, mask_enum, include_h, h0, no_grad)
    data = enc.data
    n = data.num_nodes
    pairs, cand, owner, npair = _sample(data)
    pe = torch.from_numpy(_pair_edges(pairs, n))
    ce = torch.from_numpy(cand)
    pw = torch.full((pe.size(1),), AT['ew'])
    if npair and ce.size(1):
        dev = next(self.rnn.parameters()).device
        m = torch.zeros(ce.size(1), device=dev, requires_grad=True)
        if AT['steps'] <= 1:
            enc._at_extra = (torch.cat([pe, ce], 1), torch.cat([pw.to(dev), AT['ew'] * m]))
            zs = _orig_forward(self, mask_enum, False, h0, False)
            p = _pair_probs(enc, zs, pairs)
            g, = torch.autograd.grad(-torch.log(p + 1e-6).mean(), m)
            rank = g.detach().cpu().numpy()                  # most negative gradient first
        else:
            # multi-step inner maximization: PGD on the mask, each pair's mask projected onto
            # {0 <= m <= 1, sum <= budget}; the budget largest entries per pair are inserted
            own = torch.from_numpy(owner).to(dev)
            for _ in range(AT['steps']):
                m.requires_grad_(True)
                enc._at_extra = (torch.cat([pe, ce], 1), torch.cat([pw.to(dev), AT['ew'] * m]))
                zs = _orig_forward(self, mask_enum, False, h0, False)
                p = _pair_probs(enc, zs, pairs)
                g, = torch.autograd.grad(-torch.log(p + 1e-6).mean(), m)
                with torch.no_grad():
                    m = (m - AT['lr'] * g / (g.abs().max() + 1e-12)).clamp(0, 1)
                    tot = torch.zeros(npair, device=dev).index_add_(0, own, m)
                    m = m * (AT['budget'] / tot.clamp(min=AT['budget']))[own]
                del zs, p
            rank = -m.detach().cpu().numpy()                 # largest mask first
        chosen = []
        for i in range(npair):
            idx = np.where(owner == i)[0]
            if len(idx):
                chosen.extend(idx[np.argsort(rank[idx], kind='stable')[:AT['budget']]].tolist())
        ce = ce[:, torch.tensor(sorted(chosen), dtype=torch.long)]
    else:
        ce = torch.zeros((2, 0), dtype=torch.long)
    enc._at_extra = (torch.cat([pe, ce], 1), torch.full((pe.size(1) + ce.size(1),), AT['ew']))
    enc._at_pairs = pairs
    return _orig_forward(self, mask_enum, include_h, h0, no_grad)


EI.Euler_Recurrent.forward = _forward


# ---- loss: pseudo-anomalies of the perturbed graph must score low ----
_orig_calc_loss = DetectorEncoder.calc_loss


def _calc_loss(self, z, partition, nratio, device):
    loss = _orig_calc_loss(self, z, partition, nratio, device)
    pairs = getattr(self.module, '_at_pairs', None)
    if AT['on'] and partition == TData.TRAIN and pairs is not None:
        p = _pair_probs(self.module, z, pairs)
        if p is not None:
            loss = loss + AT['lam'] * (-torch.log(1 - p + 1e-6).mean())
        self.module._at_pairs = None
        self.module._at_extra = None
    return loss


DetectorEncoder.calc_loss = _calc_loss
