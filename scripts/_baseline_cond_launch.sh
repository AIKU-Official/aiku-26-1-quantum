#!/bin/bash
# cond f0 baseline 런처: nb4 pooling reupload Bell-0, seeds {0,1,2}, 병렬. best 선택까지.
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
       NUMEXPR_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2

METHOD=cond
LOGDIR=$ROOT/cache/cond/baseline_seeds/logs
mkdir -p "$LOGDIR"
pids=""
for d in match2 pair3; do
  for s in 0 1 2; do
    python3 scripts/_baseline_worker.py --dataset "$d" --seed "$s" --method $METHOD \
      --n_blocks 4 --n_train 1000 --n_test 600 --n_steps 200 --lr 0.1 \
      > "$LOGDIR/${d}_s${s}.log" 2>&1 &
    pids="$pids $!"
  done
done
echo "launched $(echo $pids | wc -w) baseline jobs"
fail=0
for p in $pids; do wait "$p" || fail=$((fail+1)); done
echo "TRAIN_DONE fail=$fail"
python3 scripts/_baseline_select.py --method $METHOD
echo "ALL_DONE fail=$fail"
