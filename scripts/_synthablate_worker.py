# [작업 2] synthetic ablation 워커 (CQ2/T1a). cond 회로.
#   데이터: separable(rank-1 cross) / offrank1(=synthetic R2, rank≥2) / marginal(=pair3).
#   처방: f0 residual 의 total cross demand T_tot 상위 K=2 쌍에 Bell pair(= "cross 있는 곳").
#     None       : 얽힘 없음(=f0).
#     Prescribed : top-2 cross 쌍에 Bell(세기 c·√T_tot), 그 쌍이 생존(routing).
#     Wrong      : 같은 A 생존 큐빗·같은 세기지만 B 파트너를 어긋나게(off cross).
#   기대: separable None≈Prescribed(rank-1 → 얽힘 무용), offrank1 Prescribed>None(얽힘 필요).
import os, sys, json, time, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config, data, circuits, model as M, synthetic as S, diagnostics as D, ablation as AB

p = argparse.ArgumentParser()
p.add_argument("--dataset", required=True, choices=["separable", "offrank1", "marginal"])
p.add_argument("--condition", required=True, choices=["None", "Prescribed", "Wrong"])
p.add_argument("--seed", type=int, required=True)
p.add_argument("--method", default="cond", choices=circuits.METHODS)
p.add_argument("--K", type=int, default=2)
p.add_argument("--n_blocks", type=int, default=4)
p.add_argument("--c_scale", type=float, default=3.0)
p.add_argument("--n_train", type=int, default=1000)
p.add_argument("--n_test", type=int, default=600)
p.add_argument("--n_steps", type=int, default=200)
p.add_argument("--lr", type=float, default=0.1)
a = p.parse_args()

C = circuits.get(a.method)
cond_base = os.path.join(config.CACHE_DIR, "cond")


def load_data_and_f0(name):
    import glob
    if name == "marginal":
        ds = data.load_dataset("pair3"); Xang = ds["X"]; y = ds["y"]
        z = np.load(os.path.join(cond_base, "baseline_pair3_pool.npz"), allow_pickle=True)
        nb = int(z["n_blocks"])
    elif name == "separable":
        ds = S.make_separable(); Xang = S.angle_encode(ds["X"]); y = ds["y"]
        zs = [np.load(f, allow_pickle=True) for f in
              glob.glob(os.path.join(cond_base, "separable", "separable_pool_s*.npz"))]
        z = max(zs, key=lambda q: float(q["test_acc"])); nb = int(z["n_blocks"])
    else:  # offrank1 = synth R2
        ds = S.make_synthetic(2); Xang = S.angle_encode(ds["X"]); y = ds["y"]
        zs = [np.load(f, allow_pickle=True) for f in
              glob.glob(os.path.join(cond_base, "synth", "R2_seed*.npz"))]
        z = max(zs, key=lambda q: float(q["test_acc"])); nb = int(z["n_blocks"])
    qn = C.make_qnode(pooling=True, n_blocks=nb, reupload=True)
    return Xang, y, np.array(z["params"]), int(z["n_q"]), nb


Xang, y, f0params, f0nq, nb = load_data_and_f0(a.dataset)
# 처방: total cross demand T_tot 상위 K 쌍
f0qn = C.make_qnode(pooling=True, n_blocks=nb, reupload=True)
r = D.residual(f0params, Xang, y, f0qn, f0nq)
T_off, T_tot = D.demand_matrix(r, Xang, return_total=True)
flat = np.argsort(T_tot.ravel())[::-1]
ranked = [tuple(map(int, np.unravel_index(k, T_tot.shape))) for k in flat]
top = ranked[:a.K]
theta = lambda i, j: float(np.clip(a.c_scale * np.sqrt(max(T_tot[i, j], 0.0)), 0.0, np.pi / 2))
# routing: top 쌍이 생존
routing = AB.build_routing(top)
# Bell pairs
if a.condition == "None":
    bell = []
elif a.condition == "Prescribed":
    bell = [(int(i), int(4 + j), theta(i, j)) for (i, j) in top]
else:  # Wrong: 같은 A 생존, B 파트너 cyclic-shift(어긋난 cross), 같은 세기
    A_ = [i for (i, j) in top]; B_ = [j for (i, j) in top]
    Bs = B_[1:] + B_[:1] if len(B_) > 1 else [(B_[0] + 1) % 4]
    bell = [(int(i), int(4 + jb), theta(i, j)) for i, j, jb in zip(A_, B_, Bs)]

ds_full = {"X": Xang, "y": y, "aux": y}
tr, te = M.split_train_test(ds_full, a.n_train, a.n_test, config.SEED)
qn = C.make_qnode(pooling=True, n_blocks=a.n_blocks, reupload=True,
                  bell_pairs=bell or None, routing=routing)
nq = C.n_quantum_params(True, a.n_blocks)

t0 = time.time()
params, hist = M.train_model(qn, nq, tr["X"], tr["y"], a.n_steps, a.lr, seed=a.seed)
tr_acc = M.accuracy(params, tr["X"], tr["y"], qn, nq)
te_acc = M.accuracy(params, te["X"], te["y"], qn, nq)
dt = time.time() - t0

adir = os.path.join(cond_base, "synthablate"); os.makedirs(adir, exist_ok=True)
np.savez(os.path.join(adir, f"{a.dataset}_{a.condition}_seed{a.seed}.npz"),
         dataset=a.dataset, condition=a.condition, seed=a.seed, method=a.method,
         train_acc=tr_acc, test_acc=te_acc, n_bell=len(bell), K=a.K,
         top_pairs=np.array(top), bell=np.array(bell, dtype=float) if bell else np.zeros((0, 3)))
print(json.dumps({"dataset": a.dataset, "cond": a.condition, "seed": a.seed,
                  "train": round(tr_acc, 4), "test": round(te_acc, 4),
                  "n_bell": len(bell), "top": top, "sec": round(dt, 1)}), flush=True)
