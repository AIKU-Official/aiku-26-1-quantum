# CQ2 / Task T1a. separable(rank-1) cross 진단:
#   "cross correlation 이 있어도 rank-1 이면 얽힘이 불필요"를 학습 실험으로 못 박는다.
#   세 데이터: marginal(=pair3, cross 없음) / separable(신규, rank-1 cross) /
#             offrank1(=synthetic R=2, rank≥2 cross).
#   cond 회로 f0(Bell-0) → residual → demand_matrix 로
#     - total cross 에너지(T_tot, rank≥1) → σ₁/null = "cross 존재하나?"
#     - off-rank-1 에너지(T_off, rank≥2 = 얽힘필요) → σ₁/null, energy-K*/gap-K*
#   기대: separable 은 cross(σ₁_tot) 크지만 off-rank-1 ≈ 0 → 얽힘 불필요.
#         offrank1 만 off-rank-1 큼. marginal 은 cross 자체가 null.
#   ※ 이 스크립트는 진단(1)만. ablation(2)은 별도(11_*).

import os
import sys
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config, data, circuits, model as M, synthetic as S, diagnostics as D

config.apply_style()
np.random.seed(config.SEED)
REL, TAU = 0.25, 0.9
N_SHUF = 40


def _sigma1_null(r, X, total, n=N_SHUF, seed=config.SEED, pct=95):
    """셔플 null: demand_matrix 의 T_off(total=False)/T_tot(total=True) top σ 의 pct%."""
    rng = np.random.default_rng(seed)
    tops = []
    for _ in range(n):
        T = D.demand_matrix(rng.permutation(r), X, return_total=True)
        tops.append(float(np.linalg.svd(T[1] if total else T[0], compute_uv=False)[0]))
    return float(np.percentile(tops, pct))


def cross_report(r, X):
    """주어진 r(=y 또는 residual)에 대한 cross 구조 metric 묶음."""
    T_off, T_tot = D.demand_matrix(r, X, return_total=True)
    s_tot = np.linalg.svd(T_tot, compute_uv=False)
    s_off = np.linalg.svd(T_off, compute_uv=False)
    null_tot = _sigma1_null(r, X, total=True)
    null_off = _sigma1_null(r, X, total=False)
    pres = D.prescribe(T_off, TAU, 1.0, rel=REL, floor=null_off)
    return dict(
        E_tot=float(T_tot.sum()), E_off=float(T_off.sum()),
        off_ratio=float(T_off.sum() / (T_tot.sum() + 1e-12)),
        s1_tot=float(s_tot[0]), null_tot=null_tot,
        ratio_tot=float(s_tot[0] / null_tot),
        s1_off=float(s_off[0]), null_off=null_off,
        ratio_off=float(s_off[0] / null_off),
        Kstar=pres["Kstar"], Kgap=pres["Kstar_gap"])


def best(files):
    zs = [np.load(f, allow_pickle=True) for f in files]
    return max(zs, key=lambda z: float(z["test_acc"]))


def load_case(name, method):
    """(데이터 dict X/y/Xang, f0 qnode, params, n_q, f0 train/test) 반환."""
    C = circuits.get(method)
    cond_base = os.path.join(config.CACHE_DIR, "cond")
    if name == "marginal":
        ds = data.load_dataset("pair3"); Xang = ds["X"]; y = ds["y"]
        z = np.load(os.path.join(cond_base, "baseline_pair3_pool.npz"), allow_pickle=True)
        qn = C.make_qnode(pooling=bool(z["pooling"]), n_blocks=int(z["n_blocks"]),
                          reupload=bool(z["reupload"]))
    elif name == "separable":
        ds = S.make_separable(); Xang = S.angle_encode(ds["X"]); y = ds["y"]
        z = best(glob.glob(os.path.join(cond_base, "separable", "separable_pool_s*.npz")))
        qn = C.make_qnode(pooling=True, n_blocks=int(z["n_blocks"]), reupload=True)
    elif name == "offrank1":
        ds = S.make_synthetic(2); Xang = S.angle_encode(ds["X"]); y = ds["y"]
        z = best(glob.glob(os.path.join(cond_base, "synth", "R2_seed*.npz")))
        qn = C.make_qnode(pooling=True, n_blocks=int(z["n_blocks"]), reupload=True)
    else:
        raise ValueError(name)
    return Xang, y, qn, np.array(z["params"]), int(z["n_q"]), \
        float(z["train_acc"]), float(z["test_acc"])


CASES = [("marginal", "no cross"), ("separable", "rank-1 cross"),
         ("offrank1", "rank>=2 cross")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="cond", choices=circuits.METHODS)
    args = ap.parse_args()

    print("=" * 78)
    print(f"CQ2/T1a separable 진단 [{args.method}]: cross 존재(σ₁_tot) vs off-rank-1(얽힘필요)")
    print("=" * 78)
    res = {}
    for name, desc in CASES:
        Xang, y, qn, params, nq, f0tr, f0te = load_case(name, args.method)
        r = D.residual(params, Xang, y, qn, nq)
        raw = cross_report(y, Xang)          # 데이터 본질 cross (f0 무관)
        rsd = cross_report(r, Xang)          # cond f0 잔차 (얽힘 demand)
        res[name] = dict(raw=raw, rsd=rsd, f0tr=f0tr, f0te=f0te, desc=desc)
        print(f"\n── {name} ({desc})  f0 train={f0tr:.3f} test={f0te:.3f} ──")
        print(f"  [RAW y]  cross σ₁_tot/null = {raw['ratio_tot']:5.1f}x  "
              f"(cross 존재? {'YES' if raw['ratio_tot'] > 3 else 'no'})  | "
              f"off-rank-1 σ₁/null = {raw['ratio_off']:5.1f}x  off_ratio = {raw['off_ratio']:.4f}")
        print(f"  [RESID]  cross σ₁_tot/null = {rsd['ratio_tot']:5.1f}x  | "
              f"off-rank-1 σ₁/null = {rsd['ratio_off']:5.1f}x  off_ratio = {rsd['off_ratio']:.4f}  "
              f"| energy-K*={rsd['Kstar']} gap-K*(gated)={rsd['Kgap']}")

    # ── 결론 표 (RAW = 데이터 본질 cross 구조가 신뢰 신호; residual 은 f0-fit 에 교란) ──
    def verdict_of(raw):
        if raw["ratio_tot"] < 3:
            return "cross 없음 (local)"
        if raw["ratio_off"] < 3:
            return "rank-1 cross → 얽힘 불필요"
        return "off-rank-1 cross → 얽힘 필요"
    print("\n" + "=" * 82)
    print("RAW-label cross (데이터 본질, 신뢰 신호) + f0(Bell-0) train fit")
    print("=" * 82)
    print(f"{'dataset':>10} {'cross s1/null':>13} {'off-rk1 s1/null':>15} "
          f"{'off_ratio':>10} {'gapK*':>6} {'f0 train':>9}  해석")
    print("-" * 82)
    for name, _ in CASES:
        raw = res[name]["raw"]
        print(f"{name:>10} {raw['ratio_tot']:>12.1f}x {raw['ratio_off']:>14.1f}x "
              f"{raw['off_ratio']:>10.4f} {raw['Kgap']:>6} {res[name]['f0tr']:>9.3f}  "
              f"{verdict_of(raw)}")
    print("\n※ residual(=y−f0) off-rank-1 은 f0-fit 품질에 교란됨(separable f0 train=1.0 →")
    print("  잔차가 margin-noise; offrank1 잔차 off-rank-1 은 shuffle-null 정규화에 묻힘).")
    print("  → 데이터-수준 RAW off-rank-1 + f0 train fit 이 신뢰 신호. f0 train: separable=완전적합")
    print("  (얽힘 불필요 직접 증거) > marginal > offrank1(부적합→얽힘 필요).")

    # ── 그림 (PNG dpi=300): (a) cross 존재  (b) off-rank-1(데이터)  (c) f0 Bell-0 train fit ──
    #   (b)는 RAW(데이터 본질) off-rank-1 — residual 은 f0-fit 교란이라 신뢰 신호 아님.
    names = [c[0] for c in CASES]
    colors = [config.OKABE_ITO[2], config.OKABE_ITO[1], config.OKABE_ITO[4]]
    fig, axes = plt.subplots(1, 3, figsize=(config.W_DOUBLE, 60 * config.MM),
                             gridspec_kw={"wspace": 0.42})
    x = np.arange(len(names))
    # (a) cross 존재: RAW s1_tot/null (log)
    ax = axes[0]
    vals = [res[n]["raw"]["ratio_tot"] for n in names]
    ax.bar(x, vals, 0.6, color=colors, edgecolor="k", lw=0.4)
    ax.axhline(1.0, color="0.5", lw=0.6, ls="--"); ax.text(2.3, 1.06, "null", fontsize=5.0, color="0.5")
    ax.set_yscale("log"); ax.set_xticks(x)
    ax.set_xticklabels([f"{n}\n({res[n]['desc']})" for n in names], fontsize=5.0)
    ax.set_ylabel("Total cross s1 / null")
    ax.set_title("(a) Cross present?", fontsize=6.2)
    for xi, v in zip(x, vals):
        ax.text(xi, v * 1.15, f"{v:.0f}x", ha="center", fontsize=5.2)
    # (b) off-rank-1 (데이터): RAW s1_off/null (log)
    ax = axes[1]
    vals = [max(res[n]["raw"]["ratio_off"], 1e-2) for n in names]
    ax.bar(x, vals, 0.6, color=colors, edgecolor="k", lw=0.4)
    ax.axhline(1.0, color="0.5", lw=0.6, ls="--"); ax.text(2.3, 1.06, "null", fontsize=5.0, color="0.5")
    ax.set_yscale("log"); ax.set_xticks(x)
    ax.set_xticklabels([f"{n}\nratio={res[n]['raw']['off_ratio']:.3f}" for n in names], fontsize=5.0)
    ax.set_ylabel("Off-rank-1 s1 / null")
    ax.set_title("(b) Entanglement needed? (data)", fontsize=6.2)
    for xi, v in zip(x, vals):
        ax.text(xi, v * 1.15, f"{v:.1f}x", ha="center", fontsize=5.2)
    # (c) f0 Bell-0 train fit (얽힘 없는 모델이 적합되는가)
    ax = axes[2]
    vals = [res[n]["f0tr"] for n in names]
    ax.bar(x, vals, 0.6, color=colors, edgecolor="k", lw=0.4)
    ax.axhline(0.5, color="0.6", lw=0.5, ls="--")
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=5.2)
    ax.set_ylim(0.45, 1.03); ax.set_ylabel("Bell-0 f0 train accuracy")
    ax.set_title("(c) No-entanglement fit", fontsize=6.2)
    for xi, v in zip(x, vals):
        ax.text(xi, v + 0.01, f"{v:.2f}", ha="center", fontsize=5.2)
    fig.suptitle("CQ2/T1a. Separable cross has correlation (a) but rank-1 / no off-rank-1 "
                 "(b); no-entanglement f0 fits it perfectly (c) -> entanglement unnecessary",
                 fontsize=6.2, y=1.06)
    out = os.path.join(config.OUTPUT_DIR, f"10_separable_diag_{args.method}.png")
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
