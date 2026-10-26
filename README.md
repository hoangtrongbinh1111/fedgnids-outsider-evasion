# fedgnids-outsider-evasion

Code for the paper *Robust aggregation with temporal trust against outsider evasion of federated graph-based intrusion detection* (under review).

The repository studies an outsider who controls a few compromised hosts, but no federated client and no log pipeline, and who injects *covering connections* at inference time so that its malicious edges are scored as benign by a federated graph-based network intrusion detection system (GNIDS). It contains

- the attack (white-box PGD on a relaxed edge mask, a gradient-free greedy search, margin and fan-in variants);
- the defense evaluated in the paper, **GMed-T**: degree-scaled geometric-median aggregation (`gmedn`) with a plug-in temporal trust weight computed by each silo from its own history;
- the baselines (Soft Median, RUNG, coordinate-wise median, trimmed mean, GNNGuard, GAT, adversarial training) and the analysis scripts used for the tables.

## Relation to ENTENTE

The code builds on the ENTENTE implementation (Xu et al., <https://github.com/uci-dsp-lab/ENTENTE>), which builds on Euler. Its license (BSD 3-Clause) is kept in `LICENSE`. Only the `Euler/` part of ENTENTE is used.

The released ENTENTE federated loop contains three defects, which are corrected here (`--fl_fix 1`, the default; `--fl_fix 0` reproduces the released behaviour):

1. clients start from independent random initializations instead of the global model;
2. norm bounding clips the full weight vector instead of the client update;
3. the momentum aggregation counts the first client twice, so the parameters grow by about 1.17x per round and training diverges; the corrected code normalizes the aggregation weights.

These defects will be reported to the ENTENTE authors.

## Setup

Install the environment of ENTENTE (PyTorch 1.10, PyTorch Geometric 2.2 and the matching `torch_scatter`/`torch_sparse` wheels), then `pip install -r requirements.txt`. Download the pre-processed OpTC and LANL data released with ENTENTE (<https://zenodo.org/records/16014433>) and place the `data` folder at the repository root. All commands are run from the repository root; outputs go to `Exps/rona/<tag>/`.

## Main commands

Training (corrected federated training, 30 rounds; `--agg gcn` is ENTENTE-GCN, `--agg gmedn` the aggregation of GMed-T):

```bash
python Euler/rona_run.py --fed Momentum --agg gcn   --fedT 30 --seed 0 --tag F_entente_gcn_s0
python Euler/rona_run.py --fed Momentum --agg gmedn --fedT 30 --seed 0 --tag F_entente_gmedn_s0
python Euler/rona_run.py --fed FedAvg   --agg gmedn --fedT 30 --fl_fix 0 --seed 0 --tag E0_fedavg_gmedn_s0   # released configuration
```

Attacks (`--plugin_trust frozen` adds the temporal trust of GMed-T at test time; budgets are covering edges per malicious edge of the window):

```bash
python Euler/rona_attack.py        --tag F_entente_gcn_s0   --budgets 1 10
python Euler/rona_attack.py        --tag F_entente_gmedn_s0 --plugin_trust frozen --budgets 1 10
python Euler/rona_attack.py        --tag F_entente_gmedn_s0 --plugin_trust frozen --budgets 1 10 --loss cw --kappa 0.5
python Euler/rona_attack_greedy.py --tag F_entente_gmedn_s0 --plugin_trust frozen --budgets 1     # gradient-free
python Euler/rona_attack_fanin.py  --tag F_entente_gmedn_s0 --plugin_trust frozen                  # degree-scale attack
```

`--control src|both|reached` sets which hosts may originate covering edges (sources of malicious edges so far; both endpoints; or current sources plus hosts reached in earlier windows) and `--cand all|endpoints|pool` the candidate destinations.

LANL (`rona_lanl.py` documents the data flags):

```bash
python Euler/rona_run_lanl.py   --fed Momentum --agg gmedn --fedT 30 --seed 0 --tag entente_gmedn_s0
python Euler/rona_attack_lanl.py --tag L_entente_gmedn_s0 --plugin_trust frozen --control reached --cand endpoints --budgets 1 5 10
```

Baselines and controls use `rona_run_b.py` / `rona_attack_b.py` with `--agg softmed|rung|rungl1|median|trmean|guardn|gat|gatn|gmednm|gcnanc` (`gmednm`: geometric median started at the weighted mean; `gcnanc`: weighted mean with an up-weighted self loop), and `rona_b_wrap.py Euler/rona_attack_greedy.py ...` for their gradient-free attack. Adversarial training: `rona_run_at.py --agg gcn --at_lam 0.1 --at_steps 5 --at_budget 2`.

## Analysis scripts

| Script | Purpose |
|---|---|
| `rona_replay.py` | re-scores saved attack plans (per-pair statistics, TPR at fixed FPR) |
| `rona_bootstrap.py` | crossed seed x host-pair bootstrap intervals |
| `rona_fed_or.py`, `rona_fed_table.py` | cross-silo detection with per-silo cutoffs and OR fusion |
| `rona_irls_gap2.py` | distance between the deployed K-step IRLS layer and the exact geometric median |
| `rona_bound_check.py` | numerical check of the influence bound on single target nodes |
| `rona_eval_trust.py`, `rona_novelty_baseline.py` | trust variants and novelty-rule baselines |
| `rona_make_prepos.py`, `rona_make_synth_*.py` | training-time pre-positioned edges; semi-synthetic campaigns |
| `rona_diag_*.py` | validation diagnostics used to set the Soft Median temperature and the RUNG threshold |
| `rona_qrun2.sh`, `rona_qrun_gpu.sh` | simple job queues (`train`/`job` lines) used to run the experiments |

`rona_selftest.py` checks that the modified encoder reproduces the original GCN layer, and `test_robust_conv_ext.py` tests the baseline aggregators. Files added by this study: the `rona_*` scripts, `models/robust_conv.py`, `models/robust_conv_ext.py`, `loaders/rona_hooks.py`, `loaders/load_lanl.py`, `optc_subnet.json` and `test_robust_conv_ext.py`. `loaders/tdata.py` and `models/embedders.py` are ENTENTE files extended to pass trust weights to the encoder; all other files under `Euler/` are unchanged from ENTENTE.

## License

BSD 3-Clause; see `LICENSE`.
