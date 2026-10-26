# run from the repository root; activate your Python environment first
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export CUDA_MODULE_LOADING=LAZY
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
R=Exps/rona
# training finished = checkpoint written and RESULT line logged
done_train() { [ -f $R/$1/ckpt.pt ] && grep -q RESULT $R/$1/log.txt 2>/dev/null; }
wait_train() { until done_train $1; do sleep 60; done; }
summ() { grep -E "CLEAN|ATTACK" $1 | python -c "import sys,json;[print('  ',l.split()[0], {k:round(v,4) for k,v in json.loads(l.split(' ',1)[1]).items() if k in ('budget','SR','SR_stealthy','inj_flag_rate','FPR','AP')}) for l in sys.stdin]"; }
SUB_TAGS="F_sub_gcn_s0 F_sub_gmedn_s0 F_sub_gcn_s1 F_sub_gmedn_s1 F_sub_gcn_s2 F_sub_gmedn_s2"
pt_of() { case $1 in *gmedn*) echo "--plugin_trust frozen";; esac; }
sf_of() { case $1 in *gmedn*) echo "_ptfrozen";; esac; }
