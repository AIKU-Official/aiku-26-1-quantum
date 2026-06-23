# CPFP-LOCC ablation: rank-controlled 합성 데이터 (실험 ④)
#   y = sign( Σ_{r=1}^R γ_r (u_r·x_A)(v_r·x_B) + ℓ_A·x_A + ℓ_B·x_B + ε )
#   - x_A, x_B ∈ {-1,+1}^4
#   - u_r, v_r : 직교 단위벡터 (랜덤 직교행렬의 앞 R열)
#   - γ_r      : 적당히 감소(서로 다른 크기)하는 cross 세기 → singular spectrum 에서 R 식별
#   - ℓ_A, ℓ_B : 단일파티 marginal 교란항 (cross C 에는 안 섞임: E[x_B]=0)
#   - ε        : 가우시안 노이즈
#   진짜 nonlocal rank = R. 진단이 이를 복원하는지 검증한다.

import os
import numpy as np

from . import config

GAMMA = np.array([1.0, 0.75, 0.55])   # γ_1..γ_3 (R 만큼 사용)
MARGINAL_SCALE = 0.6
NOISE_STD = 0.4
N_SAMPLES = 2048


def _orthonormal_cols(rng, d, k):
    """d차원에서 직교단위벡터 k개 (랜덤 직교행렬의 앞 k열)."""
    A = rng.standard_normal((d, d))
    Q, _ = np.linalg.qr(A)
    return Q[:, :k]                    # (d, k)


def make_synthetic(R, n=N_SAMPLES, seed=1234):
    """true nonlocal rank R 데이터 생성. 반환 dict(X(±1, n×8), y(±1), meta)."""
    rng = np.random.default_rng(seed + R)
    XA = rng.choice([-1.0, 1.0], size=(n, 4))
    XB = rng.choice([-1.0, 1.0], size=(n, 4))
    U = _orthonormal_cols(rng, 4, R)   # (4,R) u_r
    V = _orthonormal_cols(rng, 4, R)   # (4,R) v_r
    g = GAMMA[:R]
    cross = ((XA @ U) * (XB @ V)) @ g                      # Σ_r γ_r (u_r·xA)(v_r·xB)
    lA = XA @ (MARGINAL_SCALE * rng.standard_normal(4))    # marginal A
    lB = XB @ (MARGINAL_SCALE * rng.standard_normal(4))    # marginal B
    eps = NOISE_STD * rng.standard_normal(n)
    score = cross + lA + lB + eps
    y = np.where(score >= 0.0, 1.0, -1.0)
    X = np.concatenate([XA, XB], axis=1)                   # (n,8) ±1
    return {"name": f"synthR{R}", "R": R, "X": X, "y": y, "XA": XA, "XB": XB,
            "U": U, "V": V, "gamma": g,
            "cross_frac": float(np.var(cross) / np.var(score))}


# A 파티 degree-2 단항식(같은 파티 두 위치 곱) 인덱스
A_PAIRS = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]   # 6개


def degree2_features(XP):
    """파티 feature(n,4 ±1) → degree-2 단항식 x_i x_j (n,6)."""
    return np.stack([XP[:, i] * XP[:, j] for (i, j) in A_PAIRS], axis=1)


def make_synthetic_ho(R, n=N_SAMPLES, seed=4321):
    """3-way(고차) cross 데이터. cross 가 degree-2-in-A × degree-1-in-B.

    y = sign( Σ_{r≤R} γ_r (a_r·Z_A)(w_r·x_B) + ℓ_A·x_A + ℓ_B·x_B + ε )
      Z_A = A 의 degree-2 단항식(6차원), a_r∈R^6 직교, w_r∈R^4 직교.
    → degree-1 cross(x_i^A x_j^B)는 0(blind), degree-2 basis 라야 복원됨.
    """
    rng = np.random.default_rng(seed + R)
    XA = rng.choice([-1.0, 1.0], size=(n, 4))
    XB = rng.choice([-1.0, 1.0], size=(n, 4))
    ZA = degree2_features(XA)                       # (n,6)
    Aco = _orthonormal_cols(rng, 6, R)              # (6,R) a_r
    Wco = _orthonormal_cols(rng, 4, R)              # (4,R) w_r
    g = GAMMA[:R]
    cross = ((ZA @ Aco) * (XB @ Wco)) @ g           # Σ_r γ_r (a_r·ZA)(w_r·xB)
    lA = XA @ (MARGINAL_SCALE * rng.standard_normal(4))
    lB = XB @ (MARGINAL_SCALE * rng.standard_normal(4))
    eps = NOISE_STD * rng.standard_normal(n)
    score = cross + lA + lB + eps
    y = np.where(score >= 0.0, 1.0, -1.0)
    X = np.concatenate([XA, XB], axis=1)
    return {"name": f"synthHO_R{R}", "R": R, "X": X, "y": y, "XA": XA, "XB": XB,
            "Aco": Aco, "Wco": Wco, "gamma": g,
            "cross_frac": float(np.var(cross) / np.var(score))}


def make_separable(n=N_SAMPLES, seed=2025):
    """separable(rank-1) cross 데이터 (연구목적 CQ2 / Task T1a).
       y = sign((u·x_A)(v·x_B)),  u,v 직교(랜덤) 단위벡터, x_A,x_B ∈ {-1,+1}^4.

    핵심: cross correlation 은 분명히 존재(σ₁ 큼)하지만 **rank-1**(=곱/분리 구조)이라
    얽힘이 불필요하다. sign(ab)=sign(a)sign(b) 로 factorize → cross C 가 정확히 rank-1
    (off-rank-1 에너지 ≈ 0). "cross 있어도 rank-1 이면 얽힘 불필요"의 학습 실험 표본.
    """
    rng = np.random.default_rng(seed)
    XA = rng.choice([-1.0, 1.0], size=(n, 4))
    XB = rng.choice([-1.0, 1.0], size=(n, 4))
    u = rng.standard_normal(4); u /= np.linalg.norm(u)     # A 단위벡터
    v = rng.standard_normal(4); v /= np.linalg.norm(v)     # B 단위벡터
    a = XA @ u
    b = XB @ v
    y = np.where(a * b >= 0.0, 1.0, -1.0)                  # = sign(a)·sign(b)
    X = np.concatenate([XA, XB], axis=1)                   # (n,8) ±1
    return {"name": "separable", "X": X, "y": y, "XA": XA, "XB": XB,
            "u": u, "v": v, "rank": 1}


def angle_encode(X):
    """±1 → 각도 {-1→0, +1→π} (RY 임베딩용 기저 인코딩)."""
    return (X + 1.0) / 2.0 * np.pi


def save_csv(ds):
    """합성 데이터를 data/ 에 CSV 로 저장 (feature 0..7, label)."""
    path = os.path.join(config.DATA_DIR, f"{ds['name']}.csv")
    arr = np.concatenate([ds["X"], ds["y"][:, None]], axis=1)
    header = ",".join([str(i) for i in range(8)] + ["y"])
    np.savetxt(path, arr, delimiter=",", header=header, comments="", fmt="%.0f")
    return path
