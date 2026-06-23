#!/bin/bash
# 처방 준수 스펙트럼: match2 cond, Under_prescribed(energy3) / Prescribed(gap4) / Wrong(3 off). seeds{0,1,2}.
#   (None / Multi_extra / Discarded 는 캐시 재사용 — routing 동일.)
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
       NUMEXPR_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2
LOG=$ROOT/cache/cond/ablation/logs
mkdir -p "$LOG"
pids=""
for c in Under_prescribed Prescribed Wrong; do
  for s in 0 1 2; do
    python3 scripts/_ablation_worker.py --dataset match2 --condition "$c" --seed "$s" \
      --method cond --n_blocks 4 --c_scale 3.0 --n_train 1000 --n_test 600 \
      --n_steps 200 --lr 0.1 > "$LOG/match2_${c}_s${s}.log" 2>&1 &
    pids="$pids $!"
  done
done
echo "launched $(echo $pids | wc -w) jobs"
fail=0
for p in $pids; do wait "$p" || fail=$((fail+1)); done
echo "ALL_DONE fail=$fail"
