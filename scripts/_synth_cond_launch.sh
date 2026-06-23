#!/bin/bash
# 실험 ④ synth 런처 (정본 cond). R{1,2,3} × seeds{0,1}, Bell-0 baseline. 결과 cache/cond/synth/.
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
       NUMEXPR_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2

METHOD=cond
RS="1 2 3"
SEEDS="0 1"
NB=4; NTRAIN=1000; NTEST=600; NSTEPS=200; LR=0.1

LOGDIR=$ROOT/cache/cond/synth/logs
mkdir -p "$LOGDIR"
pids=""
for R in $RS; do
  for s in $SEEDS; do
    python3 scripts/_synth_worker.py --R "$R" --seed "$s" --method $METHOD \
      --n_blocks $NB --n_train $NTRAIN --n_test $NTEST --n_steps $NSTEPS --lr $LR \
      > "$LOGDIR/R${R}_s${s}.log" 2>&1 &
    pids="$pids $!"
  done
done
echo "launched $(echo $pids | wc -w) synth jobs"
fail=0
for p in $pids; do wait "$p" || fail=$((fail+1)); done
echo "ALL_DONE fail=$fail"
