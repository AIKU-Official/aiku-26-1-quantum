#!/bin/bash
# 실험 ③ higher-order 런처 (정본 cond). R{1,2} × seeds{0,1}. 결과 cache/cond/ho/.
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
       NUMEXPR_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2
METHOD=cond
LOGDIR=$ROOT/cache/cond/ho/logs
mkdir -p "$LOGDIR"
pids=""
for R in 1 2; do
  for s in 0 1; do
    python3 scripts/_ho_worker.py --R "$R" --seed "$s" --method $METHOD \
      --n_blocks 4 --n_train 1000 --n_test 600 --n_steps 200 --lr 0.1 \
      > "$LOGDIR/R${R}_s${s}.log" 2>&1 &
    pids="$pids $!"
  done
done
echo "launched $(echo $pids | wc -w) ho jobs"
fail=0
for p in $pids; do wait "$p" || fail=$((fail+1)); done
echo "ALL_DONE fail=$fail"
