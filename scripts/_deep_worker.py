# 실험 ② 워커: 깊은 cascade 회로에서 한 (dataset, spec, seed) 학습.
#   spec: (A) 5조건(None/Discarded/Wrong/Prescribed/Multi) 또는 (B) sweep_dK.
#   Bell 강도는 match2 진단 σ 에서 (실험 초점이 match2 cross 구조이므로 일관 사용).
import os, sys, json, time, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config, data, model as M
from src import deep_circuit as _DC_cry, deep_circuit_cond as _DC_cond

p = argparse.ArgumentParser()
p.add_argument("--dataset", required=True)
p.add_argument("--spec", required=True)
p.add_argument("--seed", type=int, required=True)
p.add_argument("--method", default="cry", choices=["cry", "cond"])  # cry(기존)/cond(정본)
p.add_argument("--n_blocks", type=int, default=4)
p.add_argument("--c_scale", type=float, default=3.0)
p.add_argument("--n_train", type=int, default=400)
p.add_argument("--n_test", type=int, default=400)
p.add_argument("--n_steps", type=int, default=120)
p.add_argument("--lr", type=float, default=0.1)
a = p.parse_args()

DC = _DC_cond if a.method == "cond" else _DC_cry
# match2 진단 σ: method별 분리(cond=cache/cond/diag_match2.npz)
diag_base = config.CACHE_DIR if a.method == "cry" else os.path.join(config.CACHE_DIR, "cond")
sing = np.load(os.path.join(diag_base, "diag_match2.npz"))["sing"]
specs = DC.build_specs(sing, a.c_scale)
bell = specs[a.spec]

ds = data.load_dataset(a.dataset)
tr, te = M.split_train_test(ds, a.n_train, a.n_test, config.SEED)
qn = DC.make_deep_qnode(bell_pairs=bell or None, n_blocks=a.n_blocks, reupload=True)
nq = DC.n_quantum_params(a.n_blocks)

t0 = time.time()
params, hist = M.train_model(qn, nq, tr["X"], tr["y"], a.n_steps, a.lr, seed=a.seed)
tr_acc = M.accuracy(params, tr["X"], tr["y"], qn, nq)
te_acc = M.accuracy(params, te["X"], te["y"], qn, nq)
dt = time.time() - t0

ddir = os.path.join(config.CACHE_DIR, "deep") if a.method == "cry" \
    else os.path.join(config.CACHE_DIR, "cond", "deep")
os.makedirs(ddir, exist_ok=True)
np.savez(os.path.join(ddir, f"{a.dataset}_{a.spec}_seed{a.seed}.npz"),
         dataset=a.dataset, spec=a.spec, seed=a.seed, method=a.method,
         train_acc=tr_acc, test_acc=te_acc, n_bell=len(bell), final_cost=hist[-1])
print(json.dumps({"dataset": a.dataset, "spec": a.spec, "seed": a.seed, "method": a.method,
                  "train": round(tr_acc, 4), "test": round(te_acc, 4),
                  "n_bell": len(bell), "sec": round(dt, 1)}), flush=True)
