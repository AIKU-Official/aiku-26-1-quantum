#!/bin/bash
# capacity scan 병렬 런처: 모든 (dataset, n_blocks, seed) 조합을 스레드 제한 후 병렬 실행.
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
       NUMEXPR_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2

DATASETS="match2 pair3"
NBLOCKS="1 2 3 4"
SEEDS="0 1"
NTRAIN=1000; NTEST=600; NSTEPS=200; LR=0.1

LOGDIR=$ROOT/cache/scan/logs
mkdir -p "$LOGDIR"

pids=""
for d in $DATASETS; do
  for nb in $NBLOCKS; do
    for s in $SEEDS; do
      python3 scripts/_scan_worker.py --dataset "$d" --n_blocks "$nb" --seed "$s" \
        --pooling 1 --reupload 1 --n_train $NTRAIN --n_test $NTEST \
        --n_steps $NSTEPS --lr $LR \
        > "$LOGDIR/${d}_nb${nb}_s${s}.log" 2>&1 &
      pids="$pids $!"
    done
  done
done

echo "launched $(echo $pids | wc -w) jobs: $pids"
fail=0
for p in $pids; do wait "$p" || fail=$((fail+1)); done
echo "ALL_DONE fail=$fail"
