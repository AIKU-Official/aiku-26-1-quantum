# CPFP-LOCC ablation: residual 기반 cross-demand 진단 T_ij
#   가설: 필요한 얽힘 = "Bell-0 baseline f0 가 설명 못한 residual r=y-f0" 의
#         cross Fourier 에너지. 각 cross 쌍 (i∈A, j∈B) 에 대해 off-rank-1 에너지를
#         T_ij 로 측정 → 어느 쌍을 얽힐지 처방.
#
#   off-rank-1(cross) 에너지 계산은 cpfp_strength_multipartite.py 의 cross_capacity
#   로직을 재사용: 2D Fourier 블록의 교차부(둘 다 비영 주파수) SVD 후
#       off-rank-1 = Σσ² − σ₀²   (rank-1=곱/분리로 설명되는 부분을 뺀 나머지)
#
#   데이터가 4점 격자({0,π/2,π,3π/2}=π/2·color) 위에 있으므로 연속 FFT 대신
#   Z_4 경험적 DFT 를 쓴다. 다른 feature 는 무작위 표본 평균으로 marginalize 된다
#   (k=0 성분 고정과 동일).

import numpy as np

from . import model as M
from . import config


# ────────────────────────────────────────
# 0. 격자 / DFT 유틸
# ────────────────────────────────────────
def color_index(X):
    """각도(0,π/2,π,3π/2) → 색 인덱스 0..3."""
    return (np.rint(X / (np.pi / 2)).astype(int)) % 4


def dft4():
    """Z_4 DFT 행렬 W[k,c] = exp(-iπ k c / 2)  (k,c ∈ {0,1,2,3})."""
    k = np.arange(4)[:, None]
    c = np.arange(4)[None, :]
    return np.exp(-1j * np.pi * k * c / 2)


# ────────────────────────────────────────
# 1. residual
# ────────────────────────────────────────
def residual(params, X, y, qnode, n_q):
    """r = y − f0(x).  f0 = baseline 연속 결정함수 (w·probs + b)."""
    f0 = np.array(M.predict_f(params, X, qnode, n_q))
    return np.asarray(y, dtype=float) - f0


# ────────────────────────────────────────
# 2. 쌍별 cross 에너지 (off-rank-1) → T_ij
# ────────────────────────────────────────
def _pair_cross_block(r, ci, cj, W):
    """다른 feature marginalize 후 (feature i, feature j) 의 2D 경험적 Fourier 계수.

    C[a,b] = (1/N) Σ_n r_n W[a, ci_n] W[b, cj_n]   (a,b ∈ {0..3})
    반환: 교차 블록 C[1:4, 1:4] (둘 다 비영 주파수, 3×3 복소)
    """
    N = len(r)
    Wi = W[:, ci]                     # (4, N)
    Wj = W[:, cj]                     # (4, N)
    C = (Wi * r[None, :]) @ Wj.T / N  # (4, 4)
    return C[1:4, 1:4]


def demand_matrix(r, X, return_total=False):
    """residual r 로부터 4×4 cross-demand 행렬 T_ij (i∈A feature, j∈B feature).

    T_ij = off-rank-1 에너지 (Σσ² − σ₀²) of cross 블록.
    return_total=True 면 cross 총에너지(Σσ²)도 함께 반환.
    """
    W = dft4()
    cidx = color_index(X)                       # (n,8)
    cA = [cidx[:, w] for w in config.PARTY_A]   # A feature 색 (wire 0..3)
    cB = [cidx[:, w] for w in config.PARTY_B]   # B feature 색 (wire 4..7)
    T_off = np.zeros((4, 4))
    T_tot = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            blk = _pair_cross_block(r, cA[i], cB[j], W)
            s = np.linalg.svd(blk, compute_uv=False)
            T_tot[i, j] = float((s ** 2).sum())
            T_off[i, j] = float((s ** 2).sum() - s[0] ** 2)
    return (T_off, T_tot) if return_total else T_off


# ────────────────────────────────────────
# 3. T_ij → 처방 (K*, 세기 √σ, 상위 쌍)
# ────────────────────────────────────────
def signed_cross_matrix(r, XA, XB):
    """signed cross-covariance C[i,j] = mean(r · x_{A,i} · x_{B,j})  (binary ±1 용).

    bilinear cross 항 γ_r(u_r·x_A)(v_r·x_B) 는 C 에 rank-1 기여 γ_r u_r v_rᵀ →
    R개 직교항이면 C 의 rank=R, singular value≈|γ_r|. SVD 로 nonlocal rank 복원.
    """
    n = len(r)
    return (XA * r[:, None]).T @ XB / n        # (4,4) 실수 signed


A_PAIRS = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]   # A degree-2 단항식


def degree2_cross_matrix(r, XA, XB):
    """degree-2-in-A × degree-1-in-B signed cross C2[p,k] = mean(r·(x_{p0}x_{p1})^A·x_k^B).

    반환: (6,4) 실수. 3-way(고차) cross 를 잡는다(degree-1 C 가 못 보는 것을 복원).
    """
    n = len(r)
    ZA = np.stack([XA[:, i] * XA[:, j] for (i, j) in A_PAIRS], axis=1)   # (n,6)
    return (ZA * r[:, None]).T @ XB / n                                  # (6,4)


def svd_rank(M, tau=0.9):
    """행렬 M 의 SVD → (singular values, 누적에너지, K*(누적 σ² ≥ tau))."""
    s = np.linalg.svd(M, compute_uv=False)
    e = s ** 2
    cum = np.cumsum(e) / (e.sum() + 1e-18)
    Kstar = int(np.searchsorted(cum, tau) + 1)
    Kstar = max(1, min(Kstar, len(s)))
    return s, cum, Kstar


def rank_gap(s, rel=0.25, floor=0.0):
    """상대 특이값 gap 기준 rank: σ_k ≥ max(rel·σ_1, floor) 인 성분 수.

    누적에너지(τ) 기준은 약한 성분(작은 γ_r)을 90% 꼬리 밑으로 잘라 과소추정할 수
    있다. 이 기준은 'σ_1 의 rel 배 이상' 인 성분을 세어, 노이즈 바닥보다 충분히 큰
    약한 성분도 rank 로 인정한다(스펙트럼 gap 탐지).

    floor : 절대 노이즈 바닥(예: residual 셔플 null σ_1). σ_1 자체가 floor 밑이면 0
            (= cross 구조 없음). pair3 처럼 cross 가 없을 때 0 을 주기 위함.
    """
    s = np.asarray(s, dtype=float)
    if s[0] <= 0:
        return 0
    thr = max(rel * s[0], floor)
    return int(np.sum(s >= thr))


def null_sigma1(r, X, n_shuffle=40, seed=2024, pct=95):
    """residual 을 셔플해 r–X 대응을 깬 뒤 demand_matrix 의 top 특이값 분포 → 노이즈 바닥.

    반환: 셔플 σ_1 의 pct 백분위수 (이보다 작은 실제 σ_1 은 cross 없음으로 간주).
    """
    rng = np.random.default_rng(seed)
    tops = []
    for _ in range(n_shuffle):
        T = demand_matrix(rng.permutation(r), X)
        tops.append(float(np.linalg.svd(T, compute_uv=False)[0]))
    return float(np.percentile(tops, pct))


def prescribe(T, energy_thresh=0.9, c=1.0, rel=0.25, floor=0.0):
    """T_ij 를 SVD → 두 기준의 K* + 세기 λ_r = c·√σ_r.

    Kstar      : energy 기준 (누적 singular energy ≥ energy_thresh)
    Kstar_gap  : 상대 gap 기준 (σ_k ≥ max(rel·σ_1, floor)).  약한 성분도 인정.
                 floor>0 이면 cross 없는 경우 0 가능.
    *모든 실험은 두 기준을 함께 보고한다.*

    반환 dict: sing, cum_energy, Kstar, Kstar_gap, lambdas(길이 Kstar), ranked_pairs
    """
    s = np.linalg.svd(T, compute_uv=False)
    e = s ** 2
    cum = np.cumsum(e) / (e.sum() + 1e-18)
    Kstar = int(np.searchsorted(cum, energy_thresh) + 1)
    Kstar = max(1, min(Kstar, len(s)))
    Kstar_gap = rank_gap(s, rel, floor)
    lambdas = c * np.sqrt(s[:Kstar])
    flat = np.argsort(T.ravel())[::-1]
    ranked = [tuple(map(int, np.unravel_index(k, T.shape))) for k in flat]
    return {"sing": s, "cum_energy": cum, "Kstar": Kstar, "Kstar_gap": Kstar_gap,
            "lambdas": lambdas, "ranked_pairs": ranked}
