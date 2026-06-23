#!/bin/bash
# 실험 ② 런처: 깊은 cascade. (A) 5조건×2데이터×2시드 + (B) depth-sweep×match2×2시드.
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
       NUMEXPR_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2
LOG=$ROOT/cache/deep/logs
mkdir -p "$LOG"
pids=""

# (A) 5조건
for d in match2 pair3; do
  for c in None Discarded Wrong Prescribed Multi; do
    for s in 0 1; do
      python3 scripts/_deep_worker.py --dataset "$d" --spec "$c" --seed "$s" \
        --n_steps 120 --n_train 400 --n_test 400 --lr 0.1 \
        > "$LOG/${d}_${c}_s${s}.log" 2>&1 &
      pids="$pids $!"
    done
  done
done

# (B) depth-sweep (match2)
for c in sweep_d0 sweep_d1 sweep_d3 sweep_d4; do
  for s in 0 1; do
    python3 scripts/_deep_worker.py --dataset match2 --spec "$c" --seed "$s" \
      --n_steps 120 --n_train 400 --n_test 400 --lr 0.1 \
      > "$LOG/match2_${c}_s${s}.log" 2>&1 &
    pids="$pids $!"
  done
done

echo "launched $(echo $pids | wc -w) jobs"
fail=0
for p in $pids; do wait "$p" || fail=$((fail+1)); done
echo "ALL_DONE fail=$fail"
