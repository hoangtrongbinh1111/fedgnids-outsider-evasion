'''Equivalence tests for the RoNA encoder changes (run on CPU).'''
import sys

import torch
from torch_geometric.nn import GCNConv

sys.path.insert(0, './Euler')
from models.robust_conv import RobustConv

torch.manual_seed(0)
N, F, H, T = 50, 50, 8, 6
x = torch.eye(N)
snaps = []
for t in range(T):
    E = 120
    ei = torch.randint(0, N, (2, E))
    ei = ei[:, ei[0] != ei[1]]
    snaps.append((ei, torch.rand(ei.size(1)) * 0.5 + 0.5, torch.rand(ei.size(1)) * 0.9 + 0.1))

# 1) agg='gcn' == GCNConv(edge_weight = ew*trust)
ref = GCNConv(F, H, add_self_loops=True)
mine = RobustConv(F, H, agg='gcn')
with torch.no_grad():
    mine.lin.weight.copy_(ref.lin.weight)
    ref.bias.normal_()
    mine.bias.copy_(ref.bias)
err = max((ref(x, ei, edge_weight=ew * tr) - mine(x, ei, ew, tr)).abs().max().item() for ei, ew, tr in snaps)
print('gcn vs GCNConv max abs err: %.2e' % err)

# 2) block-diagonal batch == per-snapshot loop, for every aggregator
for agg in ['gcn', 'mean', 'guard', 'gmed', 'gmedn']:
    conv = RobustConv(F, H, agg=agg)
    per = torch.cat([conv(x, ei, ew, tr) for ei, ew, tr in snaps])
    big_ei = torch.cat([ei + t * N for t, (ei, _, _) in enumerate(snaps)], 1)
    big = conv.aggregate(conv.lin(x).repeat(T, 1), big_ei, torch.cat([s[1] for s in snaps]),
                         torch.cat([s[2] for s in snaps]))
    print('%5s batch vs loop max abs err: %.2e' % (agg, (per - big).abs().max().item()))
