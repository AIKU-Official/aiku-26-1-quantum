# f0 reference baseline 워커: 한 (dataset, seed) 의 nb=4 pooling reupload Bell-0 모델 학습.
#   STEP 3 residual 진단의 f0 로 쓰임. --method 로 cry(circuit)/cond(circuit_cond) 선택.
#   per-seed 파일을 저장하고, _baseline_select.py 가 best-test 를 골라 canonical
#   baseline_{ds}_pool.npz 로 만든다. (원본 프로토콜: seeds {0,1,2} 중 best test)
import os, sys, json, time, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config, data, circuits, model as M

p = argparse.ArgumentParser()
p.add_argument("--dataset", required=True)
p.add_argument("--seed", type=int, required=True)
p.add_argument("--method", default="cry", choices=circuits.METHODS)
p.add_argument("--n_blocks", type=int, default=4)
p.add_argument("--n_train", type=int, default=1000)
p.add_argument("--n_test", type=int, default=600)
p.add_argument("--n_steps", type=int, default=200)
p.add_argument("--lr", type=float, default=0.1)
a = p.parse_args()

C = circuits.get(a.method)
ds = data.load_dataset(a.dataset)
tr, te = M.split_train_test(ds, a.n_train, a.n_test, config.SEED)
qn = C.make_qnode(pooling=True, n_blocks=a.n_blocks, reupload=True, bell_pairs=None)
nq = C.n_quantum_params(True, a.n_blocks)

t0 = time.time()
params, hist = M.train_model(qn, nq, tr["X"], tr["y"], a.n_steps, a.lr, seed=a.seed)
tr_acc = M.accuracy(params, tr["X"], tr["y"], qn, nq)
te_acc = M.accuracy(params, te["X"], te["y"], qn, nq)
dt = time.time() - t0

bdir = circuits.cache_subdir("baseline_seeds", a.method)
fn = os.path.join(bdir, f"{a.dataset}_pool_s{a.seed}.npz")
np.savez(fn, params=params, n_q=nq, pooling=True, reupload=True,
         n_blocks=a.n_blocks, dataset=a.dataset, seed=a.seed, method=a.method,
         split_seed=config.SEED, train_idx=tr["idx"], test_idx=te["idx"],
         train_acc=tr_acc, test_acc=te_acc, n_train=a.n_train, n_test=a.n_test,
         n_steps=a.n_steps, lr=a.lr, final_cost=hist[-1])
print(json.dumps({"dataset": a.dataset, "seed": a.seed, "method": a.method,
                  "train": round(tr_acc, 4), "test": round(te_acc, 4),
                  "sec": round(dt, 1), "file": fn}), flush=True)
