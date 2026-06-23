# [작업 1] match2 cross 구조 SVD 분석 (로컬 발견 서버 재현).
#   - 큐빗쌍별 cross demand T_ij (4×4): 대각 4쌍 (0,4)(1,5)(2,6)(3,7) 집중 확인.
#   - 색 4종(multi-level) → full cross SVD rank ~12 (쌍당 ~3) vs degree-1(단일변수)
#     4×4 → rank 4 (대각 깨끗) 대비.
#   데이터 본질(라벨 y) 기준. PNG dpi=300.

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config, data, diagnostics as D

config.apply_style()
PARTY_A, PARTY_B = config.PARTY_A, config.PARTY_B


def full_cross_matrix(r, X, freqs=(1, 2, 3)):
    """multi-level full Fourier cross: 행=(A feat i, freq a), 열=(B feat j, freq b).
       M[(i,a),(j,b)] = (1/N) Σ_n r_n W[a, cA_i] W[b, cB_j]*  → (4·|freqs|, 4·|freqs|) 복소."""
    W = D.dft4()
    c = D.color_index(X)
    FA = np.array([W[a, c[:, i]] for i in PARTY_A for a in freqs])      # (4F, N)
    FB = np.array([W[b, c[:, j]] for j in PARTY_B for b in freqs])      # (4F, N)
    M = (FA * r[None, :]) @ FB.conj().T / len(r)
    return M


def degree1_cross_matrix(r, X, freq=1):
    """degree-1(큐빗=단일변수) 4×4 cross: 단일 주파수만."""
    return full_cross_matrix(r, X, freqs=(freq,))


def main():
    ds = data.load_dataset("match2")
    X, y = ds["X"], ds["y"]
    r = y.astype(float)                      # 데이터 본질 cross (f0 무관)

    # ── (1) 큐빗쌍 demand T_ij (total cross 에너지) ──
    T_off, T_tot = D.demand_matrix(r, X, return_total=True)
    diag = np.array([T_tot[i, i] for i in range(4)])
    offd = T_tot[~np.eye(4, dtype=bool)]
    print("=" * 70)
    print("[작업1] match2 cross demand T_ij (total cross 에너지, 라벨 기준)")
    print("=" * 70)
    print("  행=A큐빗 i(0..3), 열=B큐빗 j(0..3)  [물리큐빗 (i, 4+j)]")
    for i in range(4):
        print("   " + "  ".join(f"{T_tot[i, j]:.4f}" for j in range(4)))
    print(f"  대각 평균 = {diag.mean():.4f}  비대각 평균 = {offd.mean():.4f}  "
          f"비율 = {diag.mean()/(offd.mean()+1e-12):.1f}x")
    print(f"  (off-rank-1 누수: 대각 T_off 평균 = "
          f"{np.mean([T_off[i,i] for i in range(4)]):.4f} — ≥2 임계 비선형성)")

    # ── (2) degree-1 (4×4) vs full (12×12) SVD ──
    M_deg1 = degree1_cross_matrix(r, X, freq=1)
    M_full = full_cross_matrix(r, X, freqs=(1, 2, 3))
    s_deg1 = np.linalg.svd(M_deg1, compute_uv=False)
    s_full = np.linalg.svd(M_full, compute_uv=False)
    rel = 0.10
    rank_deg1 = int(np.sum(s_deg1 >= rel * s_deg1[0]))
    rank_full = int(np.sum(s_full >= rel * s_full[0]))
    print("\n  degree-1 (4×4, 단일변수) σ =", np.round(s_deg1, 3))
    print(f"     → 유효 rank(σ≥{rel}σ1) = {rank_deg1}  (기대 4 = 대각 4쌍)")
    print("  full (12×12, multi-level) σ =", np.round(s_full, 3))
    print(f"     → 유효 rank(σ≥{rel}σ1) = {rank_full}  (기대 ~12 = 쌍당 ~3)")

    # ── 그림: 히트맵 + degree-1/full 스펙트럼 ──
    fig, axes = plt.subplots(1, 3, figsize=(config.W_DOUBLE, 56 * config.MM),
                             gridspec_kw={"width_ratios": [1, 1.05, 1.15], "wspace": 0.5})
    # (a) demand 히트맵
    ax = axes[0]
    vmax = T_tot.max()
    im = ax.imshow(T_tot, cmap="cividis", vmin=0, vmax=vmax, aspect="equal")
    ax.set_xticks(range(4)); ax.set_xticklabels([f"q{4+j}" for j in range(4)])
    ax.set_yticks(range(4)); ax.set_yticklabels([f"q{i}" for i in range(4)])
    ax.set_xlabel("Party B qubit"); ax.set_ylabel("Party A qubit")
    ax.set_title(f"Qubit-pair cross demand\ndiag/off = "
                 f"{diag.mean()/(offd.mean()+1e-12):.0f}x", fontsize=6.0)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{T_tot[i,j]:.2f}", ha="center", va="center",
                    fontsize=4.6, color="w" if T_tot[i, j] < vmax * 0.55 else "k")
        ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False,
                                   edgecolor=config.OKABE_ITO[4], lw=1.1))
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    # (b) degree-1 spectrum
    ax = axes[1]
    ax.bar(np.arange(1, len(s_deg1) + 1), s_deg1, 0.6, color=config.OKABE_ITO[0])
    ax.axhline(rel * s_deg1[0], color="0.6", lw=0.6, ls="--")
    ax.set_xlabel("Singular index"); ax.set_ylabel("Singular value")
    ax.set_xticks(range(1, len(s_deg1) + 1))
    ax.set_title(f"degree-1 (single var)\nrank = {rank_deg1} (clean diagonal)", fontsize=6.0)
    # (c) full spectrum
    ax = axes[2]
    ax.bar(np.arange(1, len(s_full) + 1), s_full, 0.7, color=config.OKABE_ITO[1])
    ax.axhline(rel * s_full[0], color="0.6", lw=0.6, ls="--")
    ax.set_xlabel("Singular index"); ax.set_ylabel("Singular value")
    ax.set_xticks(range(1, len(s_full) + 1, 2))
    ax.set_title(f"full multi-level (4 colors)\nrank ~ {rank_full} (~3 per pair)", fontsize=6.0)

    fig.suptitle("Task 1. match2 cross: diagonal 4 qubit-pairs (degree-1 rank 4); "
                 "multi-level color inflates full rank to ~12", fontsize=6.6, y=1.04)
    out = os.path.join(config.OUTPUT_DIR, "12_match2_cross_svd.png")
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
