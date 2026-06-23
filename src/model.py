# CPFP-LOCC ablation: 고전 결합기 + 손실 + 학습 루프 (재사용 모듈)
#   params = [양자(N_Q) | w(4) | b(1)] 를 하나의 autograd 배열로 학습.
#   f(x) = probs(x)·w + b,  예측 = sign(f).  타깃은 ±1, 손실은 MSE.

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

from . import circuit as C


# ────────────────────────────────────────
# 0. train/test 분할
# ────────────────────────────────────────
def split_train_test(ds, n_train, n_test, seed):
    """데이터셋 dict에서 셔플 후 disjoint train/test 인덱스로 분할."""
    rng = np.random.default_rng(seed)
    n = ds["X"].shape[0]
    idx = rng.permutation(n)
    tr, te = idx[:n_train], idx[n_train:n_train + n_test]
    pack = lambda I: {"X": ds["X"][I], "y": ds["y"][I], "idx": I}
    return pack(tr), pack(te)


# ────────────────────────────────────────
# 1. 예측 / 손실 / 정확도
# ────────────────────────────────────────
def predict_f(params, X, qnode, n_q, n_out=4):
    """연속 결정함수 f = probs·w + b 를 반환 (B,).  n_out=확률 벡터 차원."""
    P = qnode(X, params[:n_q])           # (B, n_out)
    w = params[n_q:n_q + n_out]
    b = params[n_q + n_out]
    return P @ w + b


def cost(params, X, Y, qnode, n_q, n_out=4):
    f = predict_f(params, X, qnode, n_q, n_out)
    return pnp.mean((f - Y) ** 2)


def accuracy(params, X, Y, qnode, n_q, n_out=4):
    f = predict_f(params, X, qnode, n_q, n_out)
    pred = np.where(np.array(f) >= 0.0, 1.0, -1.0)
    return float(np.mean(pred == np.array(Y)))


# ────────────────────────────────────────
# 2. 학습 루프
# ────────────────────────────────────────
def train_model(qnode, n_q, Xtr, Ytr, n_steps, lr, seed, n_out=4, verbose=False):
    """Adam으로 학습. 반환: (params(np.ndarray), history(list of cost))."""
    Xtr = pnp.array(Xtr, requires_grad=False)
    Ytr = pnp.array(Ytr, requires_grad=False)
    init = 0.1 * np.random.default_rng(seed).standard_normal(n_q + n_out + 1)
    params = pnp.array(init, requires_grad=True)
    opt = qml.AdamOptimizer(lr)
    hist = []
    for t in range(n_steps):
        params, c = opt.step_and_cost(
            lambda p: cost(p, Xtr, Ytr, qnode, n_q, n_out), params)
        hist.append(float(c))
        if verbose and (t % 20 == 0 or t == n_steps - 1):
            print(f"    step {t:3d}  cost={float(c):.4f}")
    return np.array(params), hist
