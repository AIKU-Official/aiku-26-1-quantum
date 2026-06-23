#!/bin/bash
# capacity scan 병렬 런처 (정본 cond pooling). 결과는 cache/cond/scan/ 로 분리.
#   CRY 기존 런처(_scan_launch.sh)와 동일 그리드. 기존 cache/scan/ 은 안 건드림.
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
       NUMEXPR_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2

METHOD=cond
DATASETS="match2 pair3"
NBLOCKS="1 2 3 4"
SEEDS="0 1"
NTRAIN=1000; NTEST=600; NSTEPS=200; LR=0.1

LOGDIR=$ROOT/cache/cond/scan/logs
mkdir -p "$LOGDIR"

pids=""
for d in $DATASETS; do
  for nb in $NBLOCKS; do
    for s in $SEEDS; do
      python3 scripts/_scan_worker.py --dataset "$d" --n_blocks "$nb" --seed "$s" \
        --method $METHOD --pooling 1 --reupload 1 \
        --n_train $NTRAIN --n_test $NTEST --n_steps $NSTEPS --lr $LR \
        > "$LOGDIR/${d}_nb${nb}_s${s}.log" 2>&1 &
      pids="$pids $!"
    done
  done
done

echo "launched $(echo $pids | wc -w) jobs: $pids"
fail=0
for p in $pids; do wait "$p" || fail=$((fail+1)); done
echo "ALL_DONE fail=$fail"
