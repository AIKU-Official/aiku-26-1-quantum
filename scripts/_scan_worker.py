# capacity scan 워커: 한 (dataset, n_blocks, seed) 조합을 학습하고 결과 저장.
#   병렬 실행을 위해 BLAS 스레드는 launcher에서 제한한다.
import os, sys, json, time, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config, data, circuits, model as M

p = argparse.ArgumentParser()
p.add_argument("--dataset", required=True)
p.add_argument("--n_blocks", type=int, required=True)
p.add_argument("--seed", type=int, required=True)
p.add_argument("--pooling", type=int, default=1)        # 1=pool, 0=nopool
p.add_argument("--reupload", type=int, default=1)
p.add_argument("--method", default="cry", choices=circuits.METHODS)  # cry(기존)/cond(정본)
p.add_argument("--n_train", type=int, default=1000)
p.add_argument("--n_test", type=int, default=600)
p.add_argument("--n_steps", type=int, default=200)
p.add_argument("--lr", type=float, default=0.1)
a = p.parse_args()

C = circuits.get(a.method)
pooling = bool(a.pooling); reupload = bool(a.reupload)
ds = data.load_dataset(a.dataset)
tr, te = M.split_train_test(ds, a.n_train, a.n_test, config.SEED)
qn = C.make_qnode(pooling=pooling, n_blocks=a.n_blocks, reupload=reupload)
nq = C.n_quantum_params(pooling, a.n_blocks)

t0 = time.time()
params, hist = M.train_model(qn, nq, tr["X"], tr["y"], a.n_steps, a.lr, seed=a.seed)
tr_acc = M.accuracy(params, tr["X"], tr["y"], qn, nq)
te_acc = M.accuracy(params, te["X"], te["y"], qn, nq)
dt = time.time() - t0

tag = "pool" if pooling else "nopool"
sdir = circuits.cache_subdir("scan", a.method)   # cry=cache/scan, cond=cache/cond/scan
fn = os.path.join(sdir, f"{a.dataset}_{tag}_nb{a.n_blocks}_seed{a.seed}.npz")
np.savez(fn, params=params, n_q=nq, pooling=pooling, reupload=reupload,
         n_blocks=a.n_blocks, dataset=a.dataset, seed=a.seed, method=a.method,
         train_acc=tr_acc, test_acc=te_acc, n_train=a.n_train,
         n_test=a.n_test, n_steps=a.n_steps, lr=a.lr,
         train_idx=tr["idx"], test_idx=te["idx"], final_cost=hist[-1])
print(json.dumps({"dataset": a.dataset, "nb": a.n_blocks, "seed": a.seed,
                  "method": a.method,
                  "tag": tag, "train": round(tr_acc, 4), "test": round(te_acc, 4),
                  "cost": round(hist[-1], 4), "sec": round(dt, 1), "file": fn}),
      flush=True)
