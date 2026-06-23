#!/bin/bash
# 실험 ① crossfit 런처 (정본 cond). fold{A,B} × seeds{0,1}, match2. 결과 cache/cond/crossfit/.
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
       NUMEXPR_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2

METHOD=cond
NB=4; NSTEPS=200; LR=0.1
LOGDIR=$ROOT/cache/cond/crossfit/logs
mkdir -p "$LOGDIR"
pids=""
for f in A B; do
  for s in 0 1; do
    python3 scripts/_crossfit_worker.py --fold "$f" --seed "$s" --method $METHOD \
      --n_blocks $NB --n_steps $NSTEPS --lr $LR \
      > "$LOGDIR/fold${f}_s${s}.log" 2>&1 &
    pids="$pids $!"
  done
done
echo "launched $(echo $pids | wc -w) crossfit jobs"
fail=0
for p in $pids; do wait "$p" || fail=$((fail+1)); done
echo "ALL_DONE fail=$fail"
