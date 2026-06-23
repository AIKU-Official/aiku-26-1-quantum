# [처방 준수 스펙트럼] match2 cond: 처방을 얼마나 따랐나에 따른 정확도.
#   None < Wrong < Under-prescribed < Prescribed > Over-prescribed.
#   "처방을 정확히 따르는 게 최적, 덜 해도(under) 더 해도(over) 손해" 를 한눈에.
#
#   조건(내부명 → 표시라벨):
#     None             → None            (0 pairs)         얽힘 0
#     Wrong            → Wrong           (3 off-diag)      틀린 위치
#     Under_prescribed → Under-prescribed(energy-K*, 3 diag) (0,4) 누락 = 하다 만 처방
#     Prescribed       → Prescribed      (gap-K*, 4 diag)  처방대로 다 함
#     Multi_extra      → Over-prescribed (4 diag + 2 off)  처방 초과
import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config

config.apply_style()
ADIR = os.path.join(config.CACHE_DIR, "cond", "ablation")
# 처방 준수 스펙트럼 순서
ORDER = ["None", "Wrong", "Under_prescribed", "Prescribed", "Multi_extra"]
LABEL = {"None": "None", "Wrong": "Wrong",
         "Under_prescribed": "Under-prescribed", "Prescribed": "Prescribed",
         "Multi_extra": "Over-prescribed"}
SUB = {"None": "(0 pairs)", "Wrong": "(3 off-diag)",
       "Under_prescribed": "(energy-K*, 3 diag)", "Prescribed": "(gap-K*, 4 diag)",
       "Multi_extra": "(4 diag + 2 off)"}
# 회색→파랑(정확)→초록(과잉): 처방 거리감
COL = {"None": "0.55", "Wrong": config.OKABE_ITO[4],
       "Under_prescribed": config.OKABE_ITO[5], "Prescribed": config.OKABE_ITO[0],
       "Multi_extra": config.OKABE_ITO[2]}


def sem(x):
    x = np.asarray(x)
    return x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0


def main():
    recs = {}
    for f in sorted(glob.glob(os.path.join(ADIR, "match2_*.npz"))):
        z = np.load(f, allow_pickle=True)
        c = str(z["condition"])
        d = recs.setdefault(c, {"tr": [], "te": [], "nb": int(z["bell_pairs"].shape[0])})
        d["tr"].append(float(z["train_acc"])); d["te"].append(float(z["test_acc"]))

    print("=" * 80)
    print("[처방 준수 스펙트럼] match2 cond — None<Wrong<Under<Prescribed>Over")
    print("=" * 80)
    print(f"{'condition':>18} {'n_bell':>6} {'test':>14} {'train':>14}")
    print("-" * 80)
    agg = {}
    for c in ORDER:
        if c not in recs:
            print(f"{c:>18}   (결과 없음)")
            continue
        te = np.array(recs[c]["te"]); tr = np.array(recs[c]["tr"])
        agg[c] = dict(te_m=te.mean(), te_s=sem(te), tr_m=tr.mean(), tr_s=sem(tr),
                      nb=recs[c]["nb"])
        print(f"{LABEL[c]:>18} {recs[c]['nb']:>6} {te.mean():.3f}±{sem(te):.3f}  "
              f"{tr.mean():.3f}±{sem(tr):.3f}")

    print("\n── 스토리 (test) ──")
    if all(c in agg for c in ORDER):
        seq = " < ".join(f"{LABEL[c]} {agg[c]['te_m']:.3f}" for c in ORDER[:4])
        print(f"  {seq}")
        print(f"  Prescribed {agg['Prescribed']['te_m']:.3f} > Over-prescribed "
              f"{agg['Multi_extra']['te_m']:.3f}")
        dU = agg["Prescribed"]["te_m"] - agg["Under_prescribed"]["te_m"]
        dO = agg["Multi_extra"]["te_m"] - agg["Prescribed"]["te_m"]
        print(f"  → 덜 하면(under→exact) Δ={dU:+.3f}(+이면 (0,4) 채워야 함);  "
              f"더 하면(exact→over) Δ={dO:+.3f}({'손해' if dO < 0 else '이득'})")
        print(f"  ⇒ 정확히 처방(gap-K* 4쌍)이 최적: 덜 해도 더 해도 손해.")

    # ── 그림: 5막대 스펙트럼 (test / train) ──
    present = [c for c in ORDER if c in agg]
    fig, axes = plt.subplots(1, 2, figsize=(config.W_DOUBLE * 1.06, 72 * config.MM),
                             gridspec_kw={"wspace": 0.22})
    x = np.arange(len(present))
    for which, ax, ylab in [("te", axes[0], "Test accuracy"),
                            ("tr", axes[1], "Train accuracy")]:
        m = [agg[c][f"{which}_m"] for c in present]
        e = [agg[c][f"{which}_s"] for c in present]
        bars = ax.bar(x, m, 0.66, yerr=e, capsize=2, error_kw={"elinewidth": 0.6},
                      color=[COL[c] for c in present], edgecolor="k", lw=0.4)
        # Prescribed 강조 테두리
        for c, b in zip(present, bars):
            if c == "Prescribed":
                b.set_edgecolor(config.OKABE_ITO[0]); b.set_linewidth(1.4)
        ax.axhline(0.5, color="0.6", lw=0.5, ls="--")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{LABEL[c]}\n{SUB[c]}" for c in present], fontsize=4.2)
        ax.tick_params(axis="x", pad=2)
        ax.set_ylabel(ylab); ax.set_ylim(0.45, 1.08)
        # 값 라벨: error bar(±SEM) 위에 얹어 막대·오차막대에 가리지 않게
        for c, xi, v, ee in zip(present, x, m, e):
            ax.text(xi, v + ee + 0.014, f"{v:.2f}", ha="center", fontsize=5.2,
                    fontweight=("bold" if c == "Prescribed" else "normal"),
                    color=(config.OKABE_ITO[0] if c == "Prescribed" else "k"))
        # 'optimal' 표시(test 패널): 값 라벨보다 더 위에서 짧은 화살표 → 값 라벨과 안 겹침
        if which == "te" and "Prescribed" in present:
            pi = present.index("Prescribed")
            vP = agg["Prescribed"]["te_m"]; eP = agg["Prescribed"]["te_s"]
            ax.annotate("optimal", xy=(pi, vP + eP + 0.032),
                        xytext=(pi, vP + eP + 0.094),
                        ha="center", fontsize=5.6, color=config.OKABE_ITO[0],
                        fontweight="bold",
                        arrowprops=dict(arrowstyle="->", color=config.OKABE_ITO[0],
                                        lw=0.9, shrinkA=0, shrinkB=0))
    fig.suptitle("Prescription adherence: following gap-K* exactly is optimal "
                 "(None < Wrong < Under < Prescribed > Over)", fontsize=6.2, y=1.03)
    out = os.path.join(config.OUTPUT_DIR, "13_prescribe_fix_cond.png")
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
