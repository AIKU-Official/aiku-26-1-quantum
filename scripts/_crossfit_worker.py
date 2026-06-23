# 실험 ① 워커: match2 2-fold cross-fitting 용 Bell-0 baseline.
#   fold A/B 각각에 f0 학습. T_ij 추정(집계)은 in-sample/held-out 둘 다 계산.
import os, sys, json, time, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config, data, circuits, model as M

p = argparse.ArgumentParser()
p.add_argument("--fold", required=True, choices=["A", "B"])
p.add_argument("--seed", type=int, required=True)
p.add_argument("--method", default="cry", choices=circuits.METHODS)  # cry(기존)/cond(정본)
p.add_argument("--n_blocks", type=int, default=4)
p.add_argument("--n_steps", type=int, default=200)
p.add_argument("--lr", type=float, default=0.1)
a = p.parse_args()

C = circuits.get(a.method)
ds = data.load_dataset("match2")
X, y = ds["X"], ds["y"]
n = X.shape[0]
idx = np.random.default_rng(config.SEED).permutation(n)
A_idx, B_idx = idx[:n // 2], idx[n // 2:]
tr_idx = A_idx if a.fold == "A" else B_idx
ho_idx = B_idx if a.fold == "A" else A_idx

qn = C.make_qnode(pooling=True, n_blocks=a.n_blocks, reupload=True)   # 기본 routing(readout 0,4)
nq = C.n_quantum_params(True, a.n_blocks)
t0 = time.time()
params, hist = M.train_model(qn, nq, X[tr_idx], y[tr_idx], a.n_steps, a.lr, seed=a.seed)
tr_acc = M.accuracy(params, X[tr_idx], y[tr_idx], qn, nq)
ho_acc = M.accuracy(params, X[ho_idx], y[ho_idx], qn, nq)
dt = time.time() - t0

cdir = circuits.cache_subdir("crossfit", a.method)  # cry=cache/crossfit, cond=cache/cond/crossfit
np.savez(os.path.join(cdir, f"fold{a.fold}_seed{a.seed}.npz"),
         params=params, n_q=nq, fold=a.fold, seed=a.seed, n_blocks=a.n_blocks,
         method=a.method, A_idx=A_idx, B_idx=B_idx, tr_acc=tr_acc, ho_acc=ho_acc)
print(json.dumps({"fold": a.fold, "seed": a.seed, "method": a.method, "train": round(tr_acc, 4),
                  "heldout": round(ho_acc, 4), "sec": round(dt, 1)}), flush=True)
