# STEP 4. Ablation 비교 (None / Prescribed / Wrong / Multi)
#   - 두 데이터셋 × 4조건 × 다시드 학습 결과(cache/ablation/*.npz) 집계
#   - test accuracy 막대그래프(평균±SEM) + train accuracy 패널
#     (match2 Prescribed 에서 train 도 오르는지 = 표현력이 실제로 열렸는지)
#
#   검증:
#     match2: Prescribed > Wrong ≈ None  (대각 얽힘이 cross 를 풀어 정확도↑)
#     pair3 : 네 조건 ≈ 동일       (marginal 이라 얽힘 무용)

import os
import sys
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config, circuits, ablation as AB

config.apply_style()

COND_COLOR = {
    "None": config.OKABE_ITO[7],         # black
    "Discarded": config.OKABE_ITO[5],    # light blue (washed-out 대조)
    "Wrong": config.OKABE_ITO[4],        # vermillion
    "Prescribed": config.OKABE_ITO[0],   # blue
    "Multi_extra": config.OKABE_ITO[2],  # green (대각 처방 + 비대각 과잉)
}


def load(method):
    """cache/(cond/)ablation/*.npz → (dataset,cond) → {train:[..], test:[..]}."""
    adir = circuits.cache_subdir("ablation", method)
    recs = {}
    for f in sorted(glob.glob(os.path.join(adir, "*.npz"))):
        z = np.load(f, allow_pickle=True)
        key = (str(z["dataset"]), str(z["condition"]))
        d = recs.setdefault(key, {"train": [], "test": []})
        d["train"].append(float(z["train_acc"]))
        d["test"].append(float(z["test_acc"]))
    return recs


def sem(x):
    x = np.asarray(x)
    return x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0


def aggregate(method):
    """한 method의 (dataset,cond)→통계 + 콘솔 표/검증요약. 반환: agg dict."""
    recs = load(method)
    if not recs:
        return None
    datasets = ["match2", "pair3"]
    conds = [c for c in AB.CONDITIONS if any((d, c) in recs for d in datasets)]
    pool_name = {"cry": "CRY", "cond": "cond (Cong)"}[method]

    # ── 표 ────────────────────────────────────────────────
    print("=" * 72)
    print(f"STEP 4 [{pool_name}] ablation: test / train accuracy (mean±SEM over seeds)")
    print("=" * 72)
    print(f"{'dataset':>8} {'condition':>11} | {'test':>14} | {'train':>14} | seeds")
    print("-" * 72)
    agg = {}
    for d in datasets:
        for c in conds:
            if (d, c) not in recs:
                continue
            te = np.array(recs[(d, c)]["test"]); tr = np.array(recs[(d, c)]["train"])
            agg[(d, c)] = dict(te_m=te.mean(), te_s=sem(te), tr_m=tr.mean(), tr_s=sem(tr))
            print(f"{d:>8} {c:>11} | {te.mean():.3f} ± {sem(te):.3f} "
                  f"| {tr.mean():.3f} ± {sem(tr):.3f} | {np.round(te,3)}")
        print("-" * 72)

    # ── 검증 요약 ──────────────────────────────────────────
    core = ["None", "Wrong", "Prescribed", "Multi_extra"]
    print("\n검증 요약")
    for d in datasets:
        if all((d, c) in agg for c in core):
            P = agg[(d, "Prescribed")]["te_m"]; W = agg[(d, "Wrong")]["te_m"]
            N = agg[(d, "None")]["te_m"]; Mu = agg[(d, "Multi_extra")]["te_m"]
            print(f"  {d}: Prescribed={P:.3f}  Wrong={W:.3f}  None={N:.3f}  "
                  f"Multi_extra={Mu:.3f}   | ΔP-N={P-N:+.3f}  ΔP-W={P-W:+.3f}")
            if d == "match2":
                tP = agg[(d, "Prescribed")]["tr_m"]; tN = agg[(d, "None")]["tr_m"]
                print(f"     train(match2): None={tN:.3f} → Prescribed={tP:.3f} "
                      f"(Δtrain={tP-tN:+.3f}; >0 이면 표현력이 실제로 열림)")
    agg["_conds"] = conds
    return agg


def _figure(method, agg):
    datasets = ["match2", "pair3"]
    conds = agg["_conds"]
    fig, axes = plt.subplots(1, 2, figsize=(config.W_DOUBLE, 64 * config.MM),
                             gridspec_kw={"wspace": 0.32})
    xpos = np.arange(len(datasets)); nC = len(conds); width = 0.16
    for which, ax, ylab, title in [
        ("test", axes[0], "Test accuracy", "Test accuracy"),
        ("train", axes[1], "Train accuracy", "Train accuracy (expressivity)")]:
        for k, c in enumerate(conds):
            m = [agg[(d, c)][f"{which[:2]}_m"] for d in datasets]
            e = [agg[(d, c)][f"{which[:2]}_s"] for d in datasets]
            ax.bar(xpos + (k - (nC - 1) / 2) * width, m, width, yerr=e, capsize=2,
                   error_kw={"elinewidth": 0.6}, color=COND_COLOR[c], label=c)
        ax.axhline(0.5, color="0.6", lw=0.5, ls="--")
        ax.set_xticks(xpos)
        ax.set_xticklabels([f"{d}\n({'cross' if d=='match2' else 'marginal'})"
                            for d in datasets])
        ax.set_ylabel(ylab)
        ax.set_ylim(0.45, 1.0)
        ax.set_title(title, fontsize=6.6)
    axes[0].legend(loc="upper right", fontsize=5.4, handlelength=1.1, ncol=2,
                   columnspacing=0.8)
    pool_name = {"cry": "CRY", "cond": "cond (Cong)"}[method]
    fig.suptitle(f"STEP 4 [{pool_name}]. Prescribed entanglement helps cross "
                 "(match2), useless for marginal (pair3)", fontsize=7.0, y=1.04)
    suffix = "" if method == "cry" else f"_{method}"
    out = os.path.join(config.OUTPUT_DIR, f"04_ablation{suffix}.png")
    fig.savefig(out, dpi=300); fig.savefig(out.replace(".png", ".pdf"))
    plt.close(fig)
    print(f"\nsaved: {out}")


def _compare_figure(aggs):
    """CRY vs cond: match2 train cap 돌파 + 위계를 한 그림에. (핵심 결과)"""
    methods = [m for m in ("cry", "cond") if m in aggs]
    conds = ["None", "Wrong", "Prescribed", "Multi_extra"]
    sty = {"cry": dict(hatch="//", alpha=0.55), "cond": dict(alpha=0.95)}
    fig, axes = plt.subplots(1, 2, figsize=(config.W_DOUBLE, 66 * config.MM),
                             gridspec_kw={"wspace": 0.30})
    for which, ax, ylab in [("te", axes[0], "match2 Test acc"),
                            ("tr", axes[1], "match2 Train acc")]:
        x = np.arange(len(conds)); width = 0.38
        for mi, m in enumerate(methods):
            agg = aggs[m]
            vals = [agg[("match2", c)][f"{which}_m"] for c in conds]
            errs = [agg[("match2", c)][f"{which}_s"] for c in conds]
            ax.bar(x + (mi - 0.5) * width, vals, width, yerr=errs, capsize=2,
                   error_kw={"elinewidth": 0.6},
                   color=[COND_COLOR[c] for c in conds],
                   edgecolor="k", lw=0.4, label=m, **sty[m])
        ax.axhline(0.5, color="0.6", lw=0.5, ls="--")
        ax.set_xticks(x); ax.set_xticklabels(conds, rotation=20, fontsize=5.4)
        ax.set_ylabel(ylab); ax.set_ylim(0.45, 1.0)
    # train cap 라인(None train) 표시
    for m in methods:
        cap = aggs[m][("match2", "None")]["tr_m"]
        axes[1].axhline(cap, color=("0.3" if m == "cond" else "0.6"),
                        lw=0.6, ls=":")
        axes[1].text(0, cap + 0.005, f"{m} cap={cap:.2f}", fontsize=4.8,
                     color=("0.3" if m == "cond" else "0.6"))
    axes[0].legend(loc="upper left", fontsize=5.4, handlelength=1.0)
    fig.suptitle("STEP 4. match2 — entanglement breaks the train cap "
                 "(CRY hatched vs cond solid)", fontsize=7.0, y=1.04)
    out = os.path.join(config.OUTPUT_DIR, "04_ablation_compare.png")
    fig.savefig(out, dpi=300); fig.savefig(out.replace(".png", ".pdf"))
    plt.close(fig)
    print(f"\nsaved comparison: {out}")

    print("\n" + "=" * 72)
    print("CRY vs cond — match2 train cap 돌파 비교")
    print("=" * 72)
    for m in methods:
        a = aggs[m]
        N = a[("match2", "None")]; P = a[("match2", "Prescribed")]
        print(f"  {m:>4}: train None={N['tr_m']:.3f} → Prescribed={P['tr_m']:.3f} "
              f"(Δ={P['tr_m']-N['tr_m']:+.3f})  | test None={N['te_m']:.3f} → "
              f"Prescribed={P['te_m']:.3f} (Δ={P['te_m']-N['te_m']:+.3f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="cry",
                    choices=list(circuits.METHODS) + ["both"])
    args = ap.parse_args()
    methods = list(circuits.METHODS) if args.method == "both" else [args.method]
    aggs = {}
    for m in methods:
        agg = aggregate(m)
        if agg is None:
            print(f"[{m}] ablation 결과 없음 (스킵)"); continue
        aggs[m] = agg
        _figure(m, agg)
    if len(aggs) == 2:
        _compare_figure(aggs)


if __name__ == "__main__":
    main()
