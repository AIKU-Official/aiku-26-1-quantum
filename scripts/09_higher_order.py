# 실험 ③. Higher-order(3-way) cross: pairwise(degree-1) T 의 한계 vs degree-2 복구.
#   데이터: cross 가 degree-2-in-A × degree-1-in-B (true rank R).
#   - degree-1 C(4×4): blind (노이즈 바닥) → cross 못 잡음.
#   - degree-2 C(6×4): rank R 복원.

import os
import sys
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config, circuits, model as M, synthetic as S, diagnostics as D

config.apply_style()
RS = [1, 2]
REL = 0.25


def best(R, method):
    hdir = circuits.cache_subdir("ho", method)
    zs = [np.load(f, allow_pickle=True) for f in glob.glob(os.path.join(hdir, f"R{R}_seed*.npz"))]
    return max(zs, key=lambda z: float(z["test_acc"]))


def null_floor(r, XA, XB, fn, n_shuffle=40, seed=2024, pct=95):
    rng = np.random.default_rng(seed)
    tops = [float(np.linalg.svd(fn(rng.permutation(r), XA, XB), compute_uv=False)[0])
            for _ in range(n_shuffle)]
    return float(np.percentile(tops, pct))


def run_method(method):
    C = circuits.get(method)
    pool_name = {"cry": "CRY", "cond": "cond (Cong)"}[method]
    print("=" * 70)
    print(f"실험 ③ [{pool_name}] Higher-order cross: degree-1 (blind) vs degree-2 (recovers)")
    print("=" * 70)
    res = {}
    for R in RS:
        ds = S.make_synthetic_ho(R)
        Xang = S.angle_encode(ds["X"]); y = ds["y"]; XA, XB = ds["XA"], ds["XB"]
        z = best(R, method)
        qn = C.make_qnode(pooling=True, n_blocks=int(z["n_blocks"]), reupload=True)
        r = D.residual(np.array(z["params"]), Xang, y, qn, int(z["n_q"]))
        C1 = D.signed_cross_matrix(r, XA, XB)         # degree-1 (4×4)
        C2 = D.degree2_cross_matrix(r, XA, XB)        # degree-2 (6×4)
        s1 = np.linalg.svd(C1, compute_uv=False); s2 = np.linalg.svd(C2, compute_uv=False)
        f1 = null_floor(r, XA, XB, D.signed_cross_matrix)
        f2 = null_floor(r, XA, XB, D.degree2_cross_matrix)
        K1 = D.rank_gap(s1, REL, f1); K2 = D.rank_gap(s2, REL, f2)
        res[R] = dict(C1=C1, C2=C2, s1=s1, s2=s2, K1=K1, K2=K2, f1=f1, f2=f2,
                      tr=float(z["train_acc"]), te=float(z["test_acc"]))
        print(f"\n── R={R} (baseline tr={res[R]['tr']:.3f} te={res[R]['te']:.3f}) ──")
        print(f"  degree-1 C σ={np.round(s1,3)}  null={f1:.3f}  gap-K*(gated)={K1}  (기대 0)")
        print(f"  degree-2 C σ={np.round(s2,3)}  null={f2:.3f}  gap-K*(gated)={K2}  (기대 {R})")

    print("\n" + "=" * 48)
    print(f"{'R':>3} {'deg1 gap-K*':>12} {'deg2 gap-K*':>12} {'true':>5}")
    print("-" * 48)
    for R in RS:
        print(f"{R:>3} {res[R]['K1']:>12} {res[R]['K2']:>12} {R:>5}")

    # ── 그림 (R=2 heatmaps + 두 R 의 스펙트럼) ──
    A_PAIRS = ["x0x1", "x0x2", "x0x3", "x1x2", "x1x3", "x2x3"]
    fig, axes = plt.subplots(1, 3, figsize=(config.W_DOUBLE, 58 * config.MM),
                             gridspec_kw={"width_ratios": [1, 1.2, 1.1], "wspace": 0.55})
    Rshow = 2
    C1, C2 = res[Rshow]["C1"], res[Rshow]["C2"]
    vlim = max(np.abs(C2).max(), np.abs(C1).max())
    # (a) degree-1
    ax = axes[0]
    im = ax.imshow(C1, cmap="RdBu_r", vmin=-vlim, vmax=vlim, aspect="equal")
    ax.set_xticks(range(4)); ax.set_xticklabels([f"b{j}" for j in range(4)])
    ax.set_yticks(range(4)); ax.set_yticklabels([f"a{i}" for i in range(4)])
    ax.set_title(f"degree-1 C (blind)\ngap-K*={res[Rshow]['K1']}", fontsize=6.0)
    ax.set_xlabel("B position"); ax.set_ylabel("A position")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    # (b) degree-2
    ax = axes[1]
    im = ax.imshow(C2, cmap="RdBu_r", vmin=-vlim, vmax=vlim, aspect="auto")
    ax.set_xticks(range(4)); ax.set_xticklabels([f"b{j}" for j in range(4)])
    ax.set_yticks(range(6)); ax.set_yticklabels(A_PAIRS, fontsize=5)
    ax.set_title(f"degree-2 C (recovers R={Rshow})\ngap-K*={res[Rshow]['K2']}", fontsize=6.0)
    ax.set_xlabel("B position"); ax.set_ylabel("A pair (degree-2)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    # (c) 스펙트럼
    ax = axes[2]
    for R in RS:
        ax.plot(range(1, len(res[R]["s2"]) + 1), res[R]["s2"], "-o", ms=3, lw=1.0,
                color=config.OKABE_ITO[2] if R == 1 else config.OKABE_ITO[0],
                label=f"deg-2, R={R}")
    ax.plot(range(1, len(res[Rshow]["s1"]) + 1), res[Rshow]["s1"], "-s", ms=2.6, lw=0.9,
            color=config.OKABE_ITO[4], label=f"deg-1, R={Rshow} (blind)")
    ax.set_xlabel("Singular value index"); ax.set_ylabel("Singular value")
    ax.set_title("deg-2 recovers rank;\ndeg-1 stays at noise", fontsize=6.0)
    ax.legend(loc="upper right", fontsize=5.0, handlelength=1.3)

    fig.suptitle(f"Experiment 3 [{pool_name}]. Pairwise (degree-1) blind to 3-way "
                 "cross; degree-2 basis recovers it", fontsize=6.4, y=1.05)
    suffix = "" if method == "cry" else f"_{method}"
    out = os.path.join(config.OUTPUT_DIR, f"09_higher_order{suffix}.png")
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"\nsaved: {out}")
    return res


def compare(res_by_method):
    methods = [m for m in ("cry", "cond") if m in res_by_method]
    if len(methods) < 2:
        return
    print("\n" + "=" * 56)
    print("실험 ③ CRY vs cond — degree-1(blind) vs degree-2(recovers R) gap-K*")
    print("=" * 56)
    print(f"{'R':>3} | " + " | ".join(f"{m}: d1/d2" for m in methods) + " | true")
    for R in RS:
        cells = [f"{res_by_method[m][R]['K1']}/{res_by_method[m][R]['K2']}" for m in methods]
        print(f"{R:>3} | " + " | ".join(f"{c:>9}" for c in cells) + f" | {R}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="cry",
                    choices=list(circuits.METHODS) + ["both"])
    args = ap.parse_args()
    methods = list(circuits.METHODS) if args.method == "both" else [args.method]
    res_by_method = {}
    for m in methods:
        if not glob.glob(os.path.join(circuits.cache_subdir("ho", m), "R*_seed*.npz")):
            print(f"[{m}] ho 결과 없음 (스킵)"); continue
        res_by_method[m] = run_method(m)
    if len(res_by_method) == 2:
        compare(res_by_method)


if __name__ == "__main__":
    main()
