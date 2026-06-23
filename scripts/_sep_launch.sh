#!/bin/bash
# separable cond f0 런처: seeds {0,1,2}, nb4 pooling reupload Bell-0. cache/cond/separable/.
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
       NUMEXPR_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2
LOG=$ROOT/cache/cond/separable/logs
mkdir -p "$LOG"
pids=""
for s in 0 1 2; do
  python3 scripts/_sep_worker.py --seed "$s" --method cond \
    --n_blocks 4 --n_train 1000 --n_test 600 --n_steps 200 --lr 0.1 \
    > "$LOG/s${s}.log" 2>&1 &
  pids="$pids $!"
done
echo "launched $(echo $pids | wc -w) sep f0 jobs"
fail=0
for p in $pids; do wait "$p" || fail=$((fail+1)); done
echo "ALL_DONE fail=$fail"
