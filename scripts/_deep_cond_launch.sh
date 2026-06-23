#!/bin/bash
# 실험 ② 런처 (정본 cond). 깊은 cascade (deep_circuit_cond) + 얕은 cond Discarded 보충.
#   - deep: (A) 5조건×2데이터×2시드 + (B) depth-sweep×match2×2시드 → cache/cond/deep/
#   - shallow Discarded: 08 의 Disc−None 갭(얕음) 비교용. cond 얕은 ablation 엔 Discarded
#     가 없으므로 여기서 보충(match2,pair3 × seed0,1,2) → cache/cond/ablation/
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
       NUMEXPR_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2

METHOD=cond
LOG=$ROOT/cache/cond/deep/logs
LOGA=$ROOT/cache/cond/ablation/logs
mkdir -p "$LOG" "$LOGA"
pids=""

# (A) deep 5조건
for d in match2 pair3; do
  for c in None Discarded Wrong Prescribed Multi; do
    for s in 0 1; do
      python3 scripts/_deep_worker.py --dataset "$d" --spec "$c" --seed "$s" \
        --method $METHOD --n_steps 120 --n_train 400 --n_test 400 --lr 0.1 \
        > "$LOG/${d}_${c}_s${s}.log" 2>&1 &
      pids="$pids $!"
    done
  done
done

# (B) deep depth-sweep (match2)
for c in sweep_d0 sweep_d1 sweep_d3 sweep_d4; do
  for s in 0 1; do
    python3 scripts/_deep_worker.py --dataset match2 --spec "$c" --seed "$s" \
      --method $METHOD --n_steps 120 --n_train 400 --n_test 400 --lr 0.1 \
      > "$LOG/match2_${c}_s${s}.log" 2>&1 &
    pids="$pids $!"
  done
done

# (C) 얕은 cond Discarded 보충 (8큐빗, _ablation_worker, 동일 설정 n_train=1000 n_steps=200)
for d in match2 pair3; do
  for s in 0 1 2; do
    python3 scripts/_ablation_worker.py --dataset "$d" --condition Discarded --seed "$s" \
      --method $METHOD --n_blocks 4 --c_scale 3.0 --n_train 1000 --n_test 600 \
      --n_steps 200 --lr 0.1 > "$LOGA/${d}_Discarded_s${s}.log" 2>&1 &
    pids="$pids $!"
  done
done

echo "launched $(echo $pids | wc -w) jobs"
fail=0
for p in $pids; do wait "$p" || fail=$((fail+1)); done
echo "ALL_DONE fail=$fail"
