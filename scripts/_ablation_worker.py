# STEP 4 워커: 한 (dataset, condition, seed) 를 학습. 라우팅/Bell pair 는 diag 처방에서.
import os, sys, json, time, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config, data, circuits, model as M, ablation as AB

p = argparse.ArgumentParser()
p.add_argument("--dataset", required=True)
p.add_argument("--condition", required=True, choices=AB.CONDITIONS)
p.add_argument("--seed", type=int, required=True)
p.add_argument("--method", default="cry", choices=circuits.METHODS)  # cry(기존)/cond(정본)
p.add_argument("--n_blocks", type=int, default=4)
p.add_argument("--c_scale", type=float, default=3.0)
p.add_argument("--n_train", type=int, default=1000)
p.add_argument("--n_test", type=int, default=600)
p.add_argument("--n_steps", type=int, default=200)
p.add_argument("--lr", type=float, default=0.1)
a = p.parse_args()

C = circuits.get(a.method)
# 처방 로드 → 조건별 라우팅/Bell. method별 diag 경로 분리(cond=cache/cond/diag_*).
diag_base = config.CACHE_DIR if a.method == "cry" else os.path.join(config.CACHE_DIR, "cond")
diag = np.load(os.path.join(diag_base, f"diag_{a.dataset}.npz"), allow_pickle=True)
conds = AB.build_conditions(diag, c_scale=a.c_scale)
spec = conds[a.condition]
routing, bell_pairs = spec["routing"], spec["bell"]

ds = data.load_dataset(a.dataset)
tr, te = M.split_train_test(ds, a.n_train, a.n_test, config.SEED)
qn = C.make_qnode(pooling=True, n_blocks=a.n_blocks, reupload=True,
                  bell_pairs=bell_pairs if bell_pairs else None, routing=routing)
nq = C.n_quantum_params(True, a.n_blocks)

t0 = time.time()
params, hist = M.train_model(qn, nq, tr["X"], tr["y"], a.n_steps, a.lr, seed=a.seed)
tr_acc = M.accuracy(params, tr["X"], tr["y"], qn, nq)
te_acc = M.accuracy(params, te["X"], te["y"], qn, nq)
dt = time.time() - t0

adir = circuits.cache_subdir("ablation", a.method)  # cry=cache/ablation, cond=cache/cond/ablation
fn = os.path.join(adir, f"{a.dataset}_{a.condition}_seed{a.seed}.npz")
np.savez(fn, params=params, dataset=a.dataset, condition=a.condition, seed=a.seed,
         method=a.method,
         train_acc=tr_acc, test_acc=te_acc, n_blocks=a.n_blocks, c_scale=a.c_scale,
         bell_pairs=np.array(bell_pairs, dtype=float) if bell_pairs else np.zeros((0, 3)),
         routing_A=np.array(routing["A"]), routing_B=np.array(routing["B"]),
         final_cost=hist[-1])
print(json.dumps({"dataset": a.dataset, "cond": a.condition, "seed": a.seed,
                  "method": a.method,
                  "train": round(tr_acc, 4), "test": round(te_acc, 4),
                  "n_bell": len(bell_pairs), "sec": round(dt, 1)}), flush=True)
