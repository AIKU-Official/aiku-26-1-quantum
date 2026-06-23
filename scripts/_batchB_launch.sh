#!/bin/bash
# STEP 4 batch B: Discarded 대조군 + K-sweep(대칭 생존).
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
       NUMEXPR_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2

LOGD=$ROOT/cache/ablation/logs
LOGK=$ROOT/cache/ksweep/logs
mkdir -p "$LOGD" "$LOGK"
pids=""

# --- Discarded (batch A 와 동일 설정: n_train=1000 n_steps=200, pooling) ---
for d in match2 pair3; do
  for s in 0 1 2; do
    python3 scripts/_ablation_worker.py --dataset "$d" --condition Discarded --seed "$s" \
      --n_blocks 4 --c_scale 3.0 --n_train 1000 --n_test 600 --n_steps 200 --lr 0.1 \
      > "$LOGD/${d}_Discarded_s${s}.log" 2>&1 &
    pids="$pids $!"
  done
done

# --- K-sweep (no-pool, full readout; K=0..4) ---
for d in match2 pair3; do
  for k in 0 1 2 3 4; do
    for s in 0 1; do
      python3 scripts/_ksweep_worker.py --dataset "$d" --k "$k" --seed "$s" \
        --n_blocks 4 --c_scale 3.0 --n_train 800 --n_test 600 --n_steps 160 --lr 0.1 \
        > "$LOGK/${d}_k${k}_s${s}.log" 2>&1 &
      pids="$pids $!"
    done
  done
done

echo "launched $(echo $pids | wc -w) jobs"
fail=0
for p in $pids; do wait "$p" || fail=$((fail+1)); done
echo "ALL_DONE fail=$fail"
