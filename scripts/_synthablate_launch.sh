#!/bin/bash
# [작업2] synthetic ablation 런처 (cond). separable/offrank1/marginal × None/Prescribed/Wrong × seed{0,1,2}.
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
       NUMEXPR_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2
LOG=$ROOT/cache/cond/synthablate/logs
mkdir -p "$LOG"
pids=""
for d in separable offrank1 marginal; do
  for c in None Prescribed Wrong; do
    for s in 0 1 2; do
      python3 scripts/_synthablate_worker.py --dataset "$d" --condition "$c" --seed "$s" \
        --method cond --K 2 --n_blocks 4 --c_scale 3.0 --n_train 1000 --n_test 600 \
        --n_steps 200 --lr 0.1 > "$LOG/${d}_${c}_s${s}.log" 2>&1 &
      pids="$pids $!"
    done
  done
done
echo "launched $(echo $pids | wc -w) synthablate jobs"
fail=0
for p in $pids; do wait "$p" || fail=$((fail+1)); done
echo "ALL_DONE fail=$fail"
