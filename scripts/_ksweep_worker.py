# Q1(b) K-sweep 워커: "대각 4쌍 모두 대칭 생존" 라우팅에서 K=0..4 sweep.
#   - pooling 없음 + readout="all"(8큐빗 전체 확률) → 깊이 위계 제거, 모든 대각 쌍 대칭.
#   - K개 상위 대각 Bell pair(세기 c·√σ) 주입 → 정점이 K*=3 인지 K=4까지 오르는지.
import os, sys, json, time, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config, data, circuits, model as M

p = argparse.ArgumentParser()
p.add_argument("--dataset", required=True)
p.add_argument("--k", type=int, required=True)          # Bell pair 수 0..4
p.add_argument("--seed", type=int, required=True)
p.add_argument("--method", default="cry", choices=circuits.METHODS)  # 처방 출처(cond diag) 선택
p.add_argument("--n_blocks", type=int, default=4)
p.add_argument("--c_scale", type=float, default=3.0)
p.add_argument("--n_train", type=int, default=1000)
p.add_argument("--n_test", type=int, default=600)
p.add_argument("--n_steps", type=int, default=200)
p.add_argument("--lr", type=float, default=0.1)
a = p.parse_args()

C = circuits.get(a.method)   # K-sweep 은 pooling=False(대칭 생존)라 회로는 method 무관,
#   하지만 처방(diag)·캐시는 method별로 분리. = cond 처방의 정당성 검증.
diag_base = config.CACHE_DIR if a.method == "cry" else os.path.join(config.CACHE_DIR, "cond")
diag = np.load(os.path.join(diag_base, f"diag_{a.dataset}.npz"), allow_pickle=True)
ranked = [tuple(map(int, q)) for q in np.asarray(diag["ranked_pairs"])]
sing = np.asarray(diag["sing"], dtype=float)
# 상위 K 대각 쌍 (match2: ranked 상위가 대각). physical (i, 4+j), 세기 c·√σ
top = ranked[:a.k]
th = np.clip(a.c_scale * np.sqrt(sing[:a.k]), 0.0, np.pi / 2)
bell_pairs = [(int(i), int(4 + j), float(t)) for (i, j), t in zip(top, th)]

ds = data.load_dataset(a.dataset)
tr, te = M.split_train_test(ds, a.n_train, a.n_test, config.SEED)
# 대칭 생존: pooling=False, readout="all"
qn = C.make_qnode(pooling=False, n_blocks=a.n_blocks, reupload=True,
                  bell_pairs=bell_pairs if bell_pairs else None, readout="all")
nq = C.n_quantum_params(False, a.n_blocks)
nout = C.n_readout("all")

t0 = time.time()
params, hist = M.train_model(qn, nq, tr["X"], tr["y"], a.n_steps, a.lr,
                             seed=a.seed, n_out=nout)
tr_acc = M.accuracy(params, tr["X"], tr["y"], qn, nq, nout)
te_acc = M.accuracy(params, te["X"], te["y"], qn, nq, nout)
dt = time.time() - t0

kdir = circuits.cache_subdir("ksweep", a.method)  # cry=cache/ksweep, cond=cache/cond/ksweep
np.savez(os.path.join(kdir, f"{a.dataset}_k{a.k}_seed{a.seed}.npz"),
         dataset=a.dataset, k=a.k, seed=a.seed, method=a.method,
         train_acc=tr_acc, test_acc=te_acc,
         bell_pairs=np.array(bell_pairs, dtype=float) if bell_pairs else np.zeros((0, 3)))
print(json.dumps({"dataset": a.dataset, "k": a.k, "seed": a.seed, "method": a.method,
                  "train": round(tr_acc, 4), "test": round(te_acc, 4),
                  "sec": round(dt, 1)}), flush=True)
