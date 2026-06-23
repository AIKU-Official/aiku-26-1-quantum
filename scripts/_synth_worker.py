# 실험 ④ 워커: synthR{R} 에 Bell-0 baseline 학습 (얽힘 없음). residual-C 는 집계에서.
import os, sys, json, time, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config, circuits, model as M, synthetic as S

p = argparse.ArgumentParser()
p.add_argument("--R", type=int, required=True)
p.add_argument("--seed", type=int, required=True)
p.add_argument("--method", default="cry", choices=circuits.METHODS)  # cry(기존)/cond(정본)
p.add_argument("--n_blocks", type=int, default=4)
p.add_argument("--n_train", type=int, default=1000)
p.add_argument("--n_test", type=int, default=600)
p.add_argument("--n_steps", type=int, default=200)
p.add_argument("--lr", type=float, default=0.1)
a = p.parse_args()

C = circuits.get(a.method)
ds = S.make_synthetic(a.R)
Xang = S.angle_encode(ds["X"])
y = ds["y"]
tr, te = M.split_train_test({"X": Xang, "y": y, "aux": y}, a.n_train, a.n_test, config.SEED)
qn = C.make_qnode(pooling=True, n_blocks=a.n_blocks, reupload=True, bell_pairs=None)
nq = C.n_quantum_params(True, a.n_blocks)

t0 = time.time()
params, hist = M.train_model(qn, nq, tr["X"], tr["y"], a.n_steps, a.lr, seed=a.seed)
tr_acc = M.accuracy(params, tr["X"], tr["y"], qn, nq)
te_acc = M.accuracy(params, te["X"], te["y"], qn, nq)
dt = time.time() - t0

sdir = circuits.cache_subdir("synth", a.method)  # cry=cache/synth, cond=cache/cond/synth
np.savez(os.path.join(sdir, f"R{a.R}_seed{a.seed}.npz"),
         params=params, n_q=nq, R=a.R, seed=a.seed, n_blocks=a.n_blocks, method=a.method,
         train_acc=tr_acc, test_acc=te_acc,
         train_idx=tr["idx"], test_idx=te["idx"])
print(json.dumps({"R": a.R, "seed": a.seed, "method": a.method, "train": round(tr_acc, 4),
                  "test": round(te_acc, 4), "sec": round(dt, 1)}), flush=True)
