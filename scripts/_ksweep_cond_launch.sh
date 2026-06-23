#!/bin/bash
# cond K-sweep 런처: 대칭 생존(no-pool, full readout) K=0..4, cond diag 처방 사용.
#   목적: cond에서도 test 정점이 K=gap-K*(=4)와 일치 → Multi>Prescribed 가 "처방이 나쁜 게
#   아니라 K*가 4에 가까웠을 뿐"임을 보여 처방 정당성 완성. 결과 cache/cond/ksweep/.
#   (CRY _batchB_launch.sh 의 K-sweep 그리드와 동일: k0..4 × {match2,pair3} × seed{0,1})
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
       NUMEXPR_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2

METHOD=cond
LOGK=$ROOT/cache/cond/ksweep/logs
mkdir -p "$LOGK"
pids=""
for d in match2 pair3; do
  for k in 0 1 2 3 4; do
    for s in 0 1; do
      python3 scripts/_ksweep_worker.py --dataset "$d" --k "$k" --seed "$s" \
        --method $METHOD --n_blocks 4 --c_scale 3.0 --n_train 800 --n_test 600 \
        --n_steps 160 --lr 0.1 > "$LOGK/${d}_k${k}_s${s}.log" 2>&1 &
      pids="$pids $!"
    done
  done
done
echo "launched $(echo $pids | wc -w) ksweep jobs"
fail=0
for p in $pids; do wait "$p" || fail=$((fail+1)); done
echo "ALL_DONE fail=$fail"
