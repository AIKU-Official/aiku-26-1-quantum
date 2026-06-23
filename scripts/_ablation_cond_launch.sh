#!/bin/bash
# STEP 4 ablation 병렬 런처 (정본 cond pooling). cond diag 처방 사용, 결과 cache/cond/ablation/.
#   CRY _ablation_launch.sh 와 동일 그리드(None/Prescribed/Wrong/Multi × seeds{0,1,2}).
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
       NUMEXPR_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2

METHOD=cond
DATASETS="match2 pair3"
CONDS="None Prescribed Wrong Multi"
SEEDS="0 1 2"
NB=4; CSCALE=3.0; NTRAIN=1000; NTEST=600; NSTEPS=200; LR=0.1

LOGDIR=$ROOT/cache/cond/ablation/logs
mkdir -p "$LOGDIR"

pids=""
for d in $DATASETS; do
  for c in $CONDS; do
    for s in $SEEDS; do
      python3 scripts/_ablation_worker.py --dataset "$d" --condition "$c" --seed "$s" \
        --method $METHOD --n_blocks $NB --c_scale $CSCALE --n_train $NTRAIN \
        --n_test $NTEST --n_steps $NSTEPS --lr $LR \
        > "$LOGDIR/${d}_${c}_s${s}.log" 2>&1 &
      pids="$pids $!"
    done
  done
done

echo "launched $(echo $pids | wc -w) jobs"
fail=0
for p in $pids; do wait "$p" || fail=$((fail+1)); done
echo "ALL_DONE fail=$fail"
