# [작업2] synthetic ablation 집계 (CQ2/T1a). 세 데이터 None/Prescribed/Wrong 나란히.
#   검증: separable None≈Prescribed (rank-1 cross → 얽힘 무용),
#         offrank1 Prescribed>None (rank≥2 → 얽힘 필요), marginal 평탄.

import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config

config.apply_style()
ADIR = os.path.join(config.CACHE_DIR, "cond", "synthablate")
DATASETS = ["marginal", "separable", "offrank1"]
DESC = {"marginal": "no cross", "separable": "rank-1 cross", "offrank1": "rank>=2 cross"}
CONDS = ["None", "Prescribed", "Wrong"]
CCOLOR = {"None": config.OKABE_ITO[7], "Prescribed": config.OKABE_ITO[0],
          "Wrong": config.OKABE_ITO[4]}


def sem(x):
    x = np.asarray(x)
    return x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0


def main():
    recs = {}
    for f in sorted(glob.glob(os.path.join(ADIR, "*.npz"))):
        z = np.load(f, allow_pickle=True)
        k = (str(z["dataset"]), str(z["condition"]))
        d = recs.setdefault(k, {"tr": [], "te": []})
        d["tr"].append(float(z["train_acc"])); d["te"].append(float(z["test_acc"]))
    if not recs:
        print("결과 없음"); return

    print("=" * 74)
    print("[작업2] synthetic ablation (cond): None / Prescribed / Wrong  test/train")
    print("=" * 74)
    agg = {}
    for d in DATASETS:
        print(f"\n── {d} ({DESC[d]}) ──")
        for c in CONDS:
            if (d, c) not in recs:
                continue
            te = np.array(recs[(d, c)]["te"]); tr = np.array(recs[(d, c)]["tr"])
            agg[(d, c)] = dict(te_m=te.mean(), te_s=sem(te), tr_m=tr.mean(), tr_s=sem(tr))
            print(f"  {c:11s} test={te.mean():.3f}±{sem(te):.3f}  train={tr.mean():.3f}±{sem(tr):.3f}")
        if all((d, c) in agg for c in CONDS):
            P, N = agg[(d, "Prescribed")], agg[(d, "None")]
            dlt = P["te_m"] - N["te_m"]
            verdict = ("얽힘 불필요 (None≈Prescribed)" if abs(dlt) < 0.03
                       else ("얽힘 도움 (Prescribed>None)" if dlt > 0 else "얽힘 해로움"))
            print(f"  → ΔP-N(test) = {dlt:+.3f}  | train None={N['tr_m']:.3f} "
                  f"Prescribed={P['tr_m']:.3f}  ⇒ {verdict}")

    # ── 그림 (PNG dpi=300): test / train 두 패널, 데이터별 3조건 ──
    fig, axes = plt.subplots(1, 2, figsize=(config.W_DOUBLE, 64 * config.MM),
                             gridspec_kw={"wspace": 0.30})
    x = np.arange(len(DATASETS)); w = 0.26
    for which, ax, ylab in [("te", axes[0], "Test accuracy"),
                            ("tr", axes[1], "Train accuracy")]:
        for k, c in enumerate(CONDS):
            m = [agg[(d, c)][f"{which}_m"] for d in DATASETS]
            e = [agg[(d, c)][f"{which}_s"] for d in DATASETS]
            ax.bar(x + (k - 1) * w, m, w, yerr=e, capsize=2, error_kw={"elinewidth": 0.6},
                   color=CCOLOR[c], label=c)
        ax.axhline(0.5, color="0.6", lw=0.5, ls="--")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{d}\n({DESC[d]})" for d in DATASETS], fontsize=5.4)
        ax.set_ylabel(ylab); ax.set_ylim(0.45, 1.03)
    axes[0].legend(loc="upper left", fontsize=5.6, handlelength=1.1)
    fig.suptitle("Task 2 (CQ2/T1a). Entanglement helps only rank>=2 cross (offrank1); "
                 "useless for rank-1 separable & marginal", fontsize=6.6, y=1.04)
    out = os.path.join(config.OUTPUT_DIR, "11_synthablate_cond.png")
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
