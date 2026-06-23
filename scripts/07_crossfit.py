# 실험 ①. match2 K* 편향 cross-fitting 수정.
#   원인 가설: f0 readout(0,4) 이 in-sample 에서 (0,4)쌍 residual 을 흡수 → T[0,0] 낮음
#             → K* 과소추정.
#   방법: 2-fold. f0_A(fold A 학습), f0_B(fold B 학습).
#     - in-sample T  = avg( T(f0_A; A), T(f0_B; B) )           ← 흡수/과적합 편향
#     - cross-fit T  = avg( T(f0_A; held-out B), T(f0_B; A) )  ← 편향 제거
#   검증: (0,4) 쌍 T 가 회복되는가? K* 가 4로 오르는가?

import os
import sys
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config, data, circuits, diagnostics as D

config.apply_style()
TAU, REL = 0.9, 0.25


def best_fold(fold, method):
    cdir = circuits.cache_subdir("crossfit", method)
    fs = glob.glob(os.path.join(cdir, f"fold{fold}_seed*.npz"))
    zs = [np.load(f, allow_pickle=True) for f in fs]
    return max(zs, key=lambda z: float(z["ho_acc"]))    # held-out 최고


def T_from(params, n_q, X, y, method):
    C = circuits.get(method)
    qn = C.make_qnode(pooling=True, n_blocks=4, reupload=True)
    r = D.residual(np.array(params), X, y, qn, n_q)
    return D.demand_matrix(r, X)


def summarize(name, T):
    sv, cum, Ke = D.svd_rank(T, TAU)
    Kg = D.rank_gap(sv, REL)
    diag = np.array([T[i, i] for i in range(4)])
    print(f"\n── {name} ──")
    for i in range(4):
        print("   " + "  ".join(f"{T[i, j]:.4f}" for j in range(4)))
    print(f"  대각 = {np.round(diag,4)}  (pair (0,4)=T[0,0]={T[0,0]:.4f})")
    print(f"  σ(T) = {np.round(sv,4)}  cum={np.round(cum,3)}")
    print(f"  K*(energy τ={TAU}) = {Ke}   K_gap(rel={REL}) = {Kg}")
    return dict(T=T, sv=sv, Ke=Ke, Kg=Kg, diag=diag)


def run_method(method):
    pool_name = {"cry": "CRY", "cond": "cond (Cong)"}[method]
    ds = data.load_dataset("match2")
    X, y = ds["X"], ds["y"]
    zA, zB = best_fold("A", method), best_fold("B", method)
    A_idx, B_idx = zA["A_idx"], zA["B_idx"]
    pA, pB = np.array(zA["params"]), np.array(zB["params"])
    nqA, nqB = int(zA["n_q"]), int(zB["n_q"])
    print("=" * 64)
    print(f"실험 ① [{pool_name}]. match2 cross-fitting (T_ij K* 편향 수정)")
    print("=" * 64)
    print(f"  f0_A: train={float(zA['tr_acc']):.3f} heldout={float(zA['ho_acc']):.3f}")
    print(f"  f0_B: train={float(zB['tr_acc']):.3f} heldout={float(zB['ho_acc']):.3f}")

    # in-sample: 학습한 fold 에서 T (흡수/과적합 편향)
    T_in = 0.5 * (T_from(pA, nqA, X[A_idx], y[A_idx], method) +
                  T_from(pB, nqB, X[B_idx], y[B_idx], method))
    # cross-fit: held-out fold 에서 T (편향 제거)
    T_cf = 0.5 * (T_from(pA, nqA, X[B_idx], y[B_idx], method) +
                  T_from(pB, nqB, X[A_idx], y[A_idx], method))

    r_in = summarize("in-sample T (overfit/absorption bias)", T_in)
    r_cf = summarize("cross-fit T (bias removed)", T_cf)

    print("\n[(0,4) 쌍 회복]  in-sample T[0,0] = "
          f"{T_in[0,0]:.4f}  →  cross-fit T[0,0] = {T_cf[0,0]:.4f}")
    print(f"[K* energy]  in-sample = {r_in['Ke']}  →  cross-fit = {r_cf['Ke']}")
    print(f"[K_gap]      in-sample = {r_in['Kg']}  →  cross-fit = {r_cf['Kg']}")

    # ── 그림: 히트맵 2개(공유 스케일) + 대각 막대 ──
    vmax = max(T_in.max(), T_cf.max())
    fig, axes = plt.subplots(1, 3, figsize=(config.W_DOUBLE, 56 * config.MM),
                             gridspec_kw={"width_ratios": [1, 1, 1.1], "wspace": 0.5})
    for ax, T, title, rr in [(axes[0], T_in, "in-sample\n(absorption bias)", r_in),
                             (axes[1], T_cf, "cross-fit\n(bias removed)", r_cf)]:
        im = ax.imshow(T, cmap="cividis", vmin=0, vmax=vmax, aspect="equal")
        ax.set_xticks(range(4)); ax.set_xticklabels([f"b{j}" for j in range(4)])
        ax.set_yticks(range(4)); ax.set_yticklabels([f"a{i}" for i in range(4)])
        ax.set_xlabel("Party B feature"); ax.set_ylabel("Party A feature")
        ax.set_title(f"{title}\nK*={rr['Ke']} (gap {rr['Kg']})", fontsize=5.8)
        for i in range(4):
            for j in range(4):
                ax.text(j, i, f"{T[i,j]:.2f}", ha="center", va="center",
                        fontsize=4.6, color="w" if T[i, j] < vmax * 0.55 else "k")
        ax.add_patch(plt.Rectangle((-0.5, -0.5), 1, 1, fill=False,
                                   edgecolor=config.OKABE_ITO[4], lw=1.1))  # (0,4)강조
    # (c) 대각 비교 막대
    ax = axes[2]
    xx = np.arange(4); w = 0.38
    ax.bar(xx - w / 2, r_in["diag"], w, color=config.OKABE_ITO[5], label="in-sample")
    ax.bar(xx + w / 2, r_cf["diag"], w, color=config.OKABE_ITO[0], label="cross-fit")
    ax.set_xticks(xx); ax.set_xticklabels([f"({i},{4+i})" for i in range(4)],
                                          fontsize=5.0)
    ax.set_xlabel("Diagonal cross pair (A,B)")
    ax.set_ylabel("Residual T (off-rank-1)")
    ax.set_title("(0,4) pair recovers", fontsize=6.0)
    ax.legend(loc="upper left", fontsize=5.4, handlelength=1.1)

    fig.suptitle(f"Experiment 1 [{pool_name}]. Cross-fitting lifts residual T "
                 "(partial (0,4) recovery); (0,4) weakest = structural, rank-4 by gap",
                 fontsize=6.2, y=1.04)
    suffix = "" if method == "cry" else f"_{method}"
    out = os.path.join(config.OUTPUT_DIR, f"07_crossfit{suffix}.png")
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"\nsaved: {out}")
    return dict(T_in=T_in, T_cf=T_cf, r_in=r_in, r_cf=r_cf)


def compare(res_by_method):
    methods = [m for m in ("cry", "cond") if m in res_by_method]
    if len(methods) < 2:
        return
    print("\n" + "=" * 64)
    print("실험 ① CRY vs cond — (0,4)쌍 회복 & K_gap 비교")
    print("=" * 64)
    for m in methods:
        rr = res_by_method[m]
        print(f"  {m:>4}: T[0,0] in={rr['T_in'][0,0]:.4f}→cf={rr['T_cf'][0,0]:.4f}  "
              f"| K*(e) in={rr['r_in']['Ke']}→cf={rr['r_cf']['Ke']}  "
              f"| K_gap in={rr['r_in']['Kg']}→cf={rr['r_cf']['Kg']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="cry",
                    choices=list(circuits.METHODS) + ["both"])
    args = ap.parse_args()
    methods = list(circuits.METHODS) if args.method == "both" else [args.method]
    res_by_method = {}
    for m in methods:
        if not glob.glob(os.path.join(circuits.cache_subdir("crossfit", m), "fold*_seed*.npz")):
            print(f"[{m}] crossfit 결과 없음 (스킵)"); continue
        res_by_method[m] = run_method(m)
    if len(res_by_method) == 2:
        compare(res_by_method)


if __name__ == "__main__":
    main()
