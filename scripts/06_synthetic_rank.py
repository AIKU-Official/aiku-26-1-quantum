# 실험 ④. Rank-controlled synthetic: residual C 의 singular spectrum 이 true rank R 복원?
#   각 R(=1,2,3): Bell-0 baseline f0 → residual r=y−f0 → signed C → SVD.
#   검증: K*(τ=0.9) == R ?  C(y)(레퍼런스) 와 비교.

import os
import sys
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config, circuits, model as M, synthetic as S, diagnostics as D

config.apply_style()
TAU = 0.9
REL = 0.25          # gap 기준: σ_k ≥ REL·σ_1
RS = [1, 2, 3]


def best_baseline(R, method):
    """R 에 대해 test acc 최고 baseline npz 반환."""
    sdir = circuits.cache_subdir("synth", method)
    fs = glob.glob(os.path.join(sdir, f"R{R}_seed*.npz"))
    zs = [np.load(f, allow_pickle=True) for f in fs]
    return max(zs, key=lambda z: float(z["test_acc"]))


def run_method(method):
    C = circuits.get(method)
    pool_name = {"cry": "CRY", "cond": "cond (Cong)"}[method]
    print("=" * 70)
    print(f"실험 ④ [{pool_name}]. Rank-controlled synthetic: residual C 가 rank R 복원?")
    print("=" * 70)

    res = {}
    for R in RS:
        ds = S.make_synthetic(R)
        Xang = S.angle_encode(ds["X"])
        z = best_baseline(R, method)
        qn = C.make_qnode(pooling=True, n_blocks=int(z["n_blocks"]), reupload=True)
        nq = int(z["n_q"])
        r = D.residual(np.array(z["params"]), Xang, ds["y"], qn, nq)
        Cres = D.signed_cross_matrix(r, ds["XA"], ds["XB"])
        s_res, cum_res, K_res = D.svd_rank(Cres, TAU)
        Kgap = D.rank_gap(s_res, REL)
        Cy = D.signed_cross_matrix(ds["y"], ds["XA"], ds["XB"])     # 레퍼런스
        s_y, cum_y, K_y = D.svd_rank(Cy, TAU)
        res[R] = dict(s_res=s_res, cum_res=cum_res, K_res=K_res, Kgap=Kgap,
                      s_y=s_y, K_y=K_y, tr=float(z["train_acc"]),
                      te=float(z["test_acc"]), gamma=ds["gamma"])

        print(f"\n── R={R} (baseline train={res[R]['tr']:.3f} test={res[R]['te']:.3f}) ──")
        print(f"  γ_r                 = {np.round(ds['gamma'],3)}")
        print(f"  C(residual) σ       = {np.round(s_res,4)}  cum={np.round(cum_res,3)}")
        print(f"  σ/σ1                = {np.round(s_res/s_res[0],3)}")
        print(f"     → K*(에너지 τ={TAU}) = {K_res}   "
              f"{'✓' if K_res==R else '✗'}   "
              f"K_gap(σ≥{REL}σ1) = {Kgap}   {'✓' if Kgap==R else '✗'}   (true R={R})")
        print(f"  C(label)    σ       = {np.round(s_y,4)}  → K*(τ)={K_y}")

    # ── 표 ──
    print("\n" + "=" * 56)
    print(f"{'R':>3} {'K*(energy τ=.9)':>16} {'K_gap(rel=.25)':>15} {'true':>5}")
    print("-" * 56)
    for R in RS:
        e_ok = "✓" if res[R]['K_res'] == R else "✗"
        g_ok = "✓" if res[R]['Kgap'] == R else "✗"
        print(f"{R:>3} {str(res[R]['K_res'])+' '+e_ok:>16} "
              f"{str(res[R]['Kgap'])+' '+g_ok:>15} {R:>5}")

    # ── 그림: (a) singular spectra, (b) K* vs R ──
    colors = [config.OKABE_ITO[0], config.OKABE_ITO[1], config.OKABE_ITO[2]]
    fig, axes = plt.subplots(1, 2, figsize=(config.W_DOUBLE * 0.8, 62 * config.MM),
                             gridspec_kw={"wspace": 0.38})
    ax = axes[0]
    idx = np.arange(1, 5); w = 0.25
    for k, R in enumerate(RS):
        ax.bar(idx + (k - 1) * w, res[R]["s_res"], w, color=colors[k],
               label=f"R={R} (K*={res[R]['K_res']})")
    ax.set_xlabel("Singular value index")
    ax.set_ylabel("Singular value of residual C")
    ax.set_xticks(idx)
    ax.set_title("Residual cross spectrum recovers rank", fontsize=6.4)
    ax.legend(loc="upper right", fontsize=5.4, handlelength=1.1)

    ax = axes[1]
    Kres = [res[R]["K_res"] for R in RS]
    Kgap = [res[R]["Kgap"] for R in RS]
    ax.plot(RS, RS, "--", color="0.6", lw=0.8, label="K = R (ideal)")
    ax.plot(RS, Kgap, "-o", ms=4, lw=1.3, color=config.OKABE_ITO[2],
            label="K_gap (σ≥0.25σ₁)")
    ax.plot(RS, Kres, "-s", ms=3.4, lw=1.0, color=config.OKABE_ITO[4],
            label="K* (energy τ=0.9)")
    ax.set_xlabel("True nonlocal rank R")
    ax.set_ylabel("Recovered rank")
    ax.set_xticks(RS); ax.set_yticks(RS)
    ax.set_title("Gap criterion recovers R;\nenergy-K* undercounts weak γ", fontsize=6.0)
    ax.legend(loc="upper left", fontsize=5.2, handlelength=1.4)

    fig.suptitle(f"Experiment 4 [{pool_name}]. Rank-controlled synthetic: K* "
                 "recovers true nonlocal rank", fontsize=7.0, y=1.04)
    suffix = "" if method == "cry" else f"_{method}"
    out = os.path.join(config.OUTPUT_DIR, f"06_synthetic_rank{suffix}.png")
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"\nsaved: {out}")
    return res


def compare(res_by_method):
    methods = [m for m in ("cry", "cond") if m in res_by_method]
    if len(methods) < 2:
        return
    print("\n" + "=" * 64)
    print("실험 ④ CRY vs cond — rank 복원 비교 (K*energy / K_gap, true R)")
    print("=" * 64)
    print(f"{'R':>3} | " + " | ".join(f"{m}:K*(e)/Kgap" for m in methods) + " | true")
    for R in RS:
        cells = []
        for m in methods:
            rr = res_by_method[m][R]
            cells.append(f"{rr['K_res']}/{rr['Kgap']}")
        print(f"{R:>3} | " + " | ".join(f"{c:>12}" for c in cells) + f" | {R}")

    # 그림: K_gap vs R, 두 방식 overlay
    fig, ax = plt.subplots(figsize=(config.W_SINGLE, 62 * config.MM))
    ax.plot(RS, RS, "--", color="0.6", lw=0.8, label="K = R (ideal)")
    sty = {"cry": dict(ls="--", marker="s", color=config.OKABE_ITO[1]),
           "cond": dict(ls="-", marker="o", color=config.OKABE_ITO[2])}
    for m in methods:
        Kgap = [res_by_method[m][R]["Kgap"] for R in RS]
        ax.plot(RS, Kgap, ms=4, lw=1.3, label=f"K_gap — {m}", **sty[m])
    ax.set_xlabel("True nonlocal rank R"); ax.set_ylabel("Recovered rank (K_gap)")
    ax.set_xticks(RS); ax.set_yticks(RS)
    ax.legend(loc="upper left", fontsize=5.4, handlelength=1.6)
    ax.set_title("Experiment 4. K_gap recovers R — CRY vs cond", fontsize=6.4)
    out = os.path.join(config.OUTPUT_DIR, "06_synthetic_rank_compare.png")
    fig.savefig(out, dpi=300); plt.close(fig)
    print(f"\nsaved comparison: {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="cry",
                    choices=list(circuits.METHODS) + ["both"])
    args = ap.parse_args()
    methods = list(circuits.METHODS) if args.method == "both" else [args.method]
    res_by_method = {}
    for m in methods:
        if not glob.glob(os.path.join(circuits.cache_subdir("synth", m), "R*_seed*.npz")):
            print(f"[{m}] synth 결과 없음 (스킵)"); continue
        res_by_method[m] = run_method(m)
    if len(res_by_method) == 2:
        compare(res_by_method)


if __name__ == "__main__":
    main()
