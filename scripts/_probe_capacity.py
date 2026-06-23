# 빠른 probe: n_blocks=3 강한 로컬 모델이 pair3를 푸는가? match2는 뒤처지는가?
import os, sys, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config, data, circuit as C, model as M
from pennylane import numpy as pnp
import pennylane as qml

np.random.seed(config.SEED)
N_TRAIN, N_TEST, N_STEPS, LR, NB = 300, 400, 250, 0.1, 3

for name in ["pair3", "match2"]:
    ds = data.load_dataset(name)
    tr, te = M.split_train_test(ds, N_TRAIN, N_TEST, config.SEED)
    qn = C.make_qnode(pooling=True, n_blocks=NB, reupload=True)
    nq = C.n_quantum_params(True, NB)
    Xtr = pnp.array(tr["X"], requires_grad=False); Ytr = pnp.array(tr["y"], requires_grad=False)
    init = 0.1 * np.random.default_rng(0).standard_normal(nq + C.N_COMBINER)
    params = pnp.array(init, requires_grad=True)
    opt = qml.AdamOptimizer(LR)
    t0 = time.time()
    for t in range(N_STEPS):
        params, c = opt.step_and_cost(lambda p: M.cost(p, Xtr, Ytr, qn, nq), params)
        if t % 25 == 0 or t == N_STEPS - 1:
            tra = M.accuracy(params, tr["X"], tr["y"], qn, nq)
            tea = M.accuracy(params, te["X"], te["y"], qn, nq)
            print(f"{name} nb={NB} step {t:3d} cost={float(c):.3f} train={tra:.3f} test={tea:.3f} ({time.time()-t0:.0f}s)", flush=True)
    print(f"== {name} DONE ==", flush=True)
