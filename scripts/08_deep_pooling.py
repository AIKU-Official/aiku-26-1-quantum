# 실험 ②. 깊은 cascade pooling: Discarded ≈ None 분리 + 깊이별 gain 게이팅.
#   (A) 5조건 ablation (deep): None ≈ Discarded < Wrong < Prescribed ≈ Multi 인가?
#   (B) depth-sweep: 단일 대각 Bell pair 의 test gain 이 pool-hop 깊이↑ 에서 0 으로?
#   비교: 얕은(8큐빗) 아키텍처의 Discarded−None 갭 vs 깊은(12큐빗) 갭.

import os
import sys
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config

config.apply_style()


def _deep_dir(method):
    return os.path.join(config.CACHE_DIR, "deep") if method == "cry" \
        else os.path.join(config.CACHE_DIR, "cond", "deep")


def _shallow_dir(method):
    return os.path.join(config.CACHE_DIR, "ablation") if method == "cry" \
        else os.path.join(config.CACHE_DIR, "cond", "ablation")


CONDS = ["None", "Discarded", "Wrong", "Prescribed", "Multi"]
COND_COLOR = {"None": config.OKABE_ITO[7], "Discarded": config.OKABE_ITO[5],
              "Wrong": config.OKABE_ITO[4], "Prescribed": config.OKABE_ITO[0],
              "Multi": config.OKABE_ITO[2]}
SWEEP = [("sweep_d0", 0), ("sweep_d1", 1), ("sweep_d3", 3), ("sweep_d4", 4)]


def sem(x):
    x = np.asarray(x)
    return x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0


def load(d):
    recs = {}
    for f in sorted(glob.glob(os.path.join(d, "*.npz"))):
        z = np.load(f, allow_pickle=True)
        k = (str(z["dataset"]), str(z["spec"] if "spec" in z else z["condition"]))
        r = recs.setdefault(k, {"tr": [], "te": []})
        r["tr"].append(float(z["train_acc"])); r["te"].append(float(z["test_acc"]))
    return recs


def run_method(method):
    pool_name = {"cry": "CRY", "cond": "cond (Cong)"}[method]
    deep = load(_deep_dir(method))
    shal = load(_shallow_dir(method))

    # ── 표 (deep 5조건) ──
    print("=" * 70)
    print(f"실험 ② [{pool_name}] 깊은 cascade: 5조건 test/train (mean±SEM)")
    print("=" * 70)
    for d in ["match2", "pair3"]:
        print(f"\n── {d} ──")
        for c in CONDS:
            if (d, c) in deep:
                te = deep[(d, c)]["te"]; tr = deep[(d, c)]["tr"]
                print(f"  {c:11s} test={np.mean(te):.3f}±{sem(te):.3f} "
                      f"train={np.mean(tr):.3f}±{sem(tr):.3f}")

    # Discarded−None 갭: 얕음 vs 깊음
    def gap(recs, d):
        if (d, "None") in recs and (d, "Discarded") in recs:
            return (np.mean(recs[(d, "Discarded")]["te"]) -
                    np.mean(recs[(d, "None")]["te"]))
        return None
    g_sh = gap(shal, "match2"); g_dp = gap(deep, "match2")
    print("\n[Discarded − None gap, match2]")
    print(f"  얕은 pooling(8큐빗, 2-hop):  "
          + (f"{g_sh:+.3f}" if g_sh is not None else "N/A (shallow Discarded 없음)"))
    print(f"  깊은 pooling(12큐빗, 3–4hop): "
          + (f"{g_dp:+.3f}" if g_dp is not None else "N/A"))

    # ── depth sweep gain ──
    if ("match2", "None") not in deep:
        print("\n[경고] deep match2 None 없음 → depth-sweep 스킵"); return None
    none_te = np.mean(deep[("match2", "None")]["te"])
    print("\n[depth-sweep] 단일 대각 Bell pair gain vs 깊이 (match2)")
    sweep_pts = []
    for spec, dep in SWEEP:
        if ("match2", spec) in deep:
            te = deep[("match2", spec)]["te"]
            gain = np.mean(te) - none_te
            sweep_pts.append((dep, np.mean(te), sem(te), gain))
            print(f"  depth {dep}: test={np.mean(te):.3f}  gain={gain:+.3f}")

    # ── 그림 ──
    fig, axes = plt.subplots(1, 2, figsize=(config.W_DOUBLE, 64 * config.MM),
                             gridspec_kw={"wspace": 0.34})
    # (a) deep 5조건 test
    ax = axes[0]
    xpos = np.arange(2); nC = len(CONDS); w = 0.16
    for k, c in enumerate(CONDS):
        m = [np.mean(deep[(d, c)]["te"]) for d in ["match2", "pair3"]]
        e = [sem(deep[(d, c)]["te"]) for d in ["match2", "pair3"]]
        ax.bar(xpos + (k - (nC - 1) / 2) * w, m, w, yerr=e, capsize=2,
               error_kw={"elinewidth": 0.6}, color=COND_COLOR[c], label=c)
    ax.axhline(0.5, color="0.6", lw=0.5, ls="--")
    ax.set_xticks(xpos); ax.set_xticklabels(["match2\n(cross)", "pair3\n(marginal)"])
    ax.set_ylabel("Test accuracy"); ax.set_ylim(0.45, 1.0)
    ax.set_title("Deep cascade: Discarded falls to None", fontsize=6.2)
    ax.legend(loc="upper right", fontsize=5.0, handlelength=1.0, ncol=2,
              columnspacing=0.7)

    # (b) gain vs depth + 얕은/깊은 갭
    ax = axes[1]
    deps = [p[0] for p in sweep_pts]; gains = [p[3] for p in sweep_pts]
    errs = [p[2] for p in sweep_pts]
    ax.errorbar(deps, gains, yerr=errs, marker="o", ms=4, lw=1.3, capsize=2,
                elinewidth=0.6, color=config.OKABE_ITO[0],
                label="single diagonal pair (deep arch)")
    ax.axhline(0.0, color="0.6", lw=0.5, ls="--")
    # 참조: 얕은 아키텍처 Discarded−None 갭
    if g_sh is not None:
        ax.axhline(g_sh, color=config.OKABE_ITO[4], lw=0.8, ls=":",
                   label=f"shallow arch Disc−None ({g_sh:+.2f})")
    ax.set_xlabel("Bell pair pool-hop depth from readout")
    ax.set_ylabel("Test accuracy gain vs None")
    ax.set_xticks([0, 1, 2, 3, 4])
    ax.set_title("Entanglement benefit is gated by pooling depth", fontsize=6.2)
    ax.legend(loc="upper right", fontsize=5.0, handlelength=1.4)

    fig.suptitle(f"Experiment 2 [{pool_name}]. Deeper pooling washes out far-placed "
                 "entanglement (Discarded → None)", fontsize=6.6, y=1.04)
    suffix = "" if method == "cry" else f"_{method}"
    out = os.path.join(config.OUTPUT_DIR, f"08_deep_pooling{suffix}.png")
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"\nsaved: {out}")
    return dict(g_sh=g_sh, g_dp=g_dp, sweep=sweep_pts, none_te=none_te)


def compare(res_by_method):
    methods = [m for m in ("cry", "cond") if m in res_by_method and res_by_method[m]]
    if len(methods) < 2:
        return
    print("\n" + "=" * 64)
    print("실험 ② CRY vs cond — Discarded−None 갭(얕음/깊음) & depth-sweep gain")
    print("=" * 64)
    for m in methods:
        rr = res_by_method[m]
        sh = f"{rr['g_sh']:+.3f}" if rr['g_sh'] is not None else "N/A"
        dp = f"{rr['g_dp']:+.3f}" if rr['g_dp'] is not None else "N/A"
        sweep = "  ".join(f"d{d}:{g:+.3f}" for (d, _, _, g) in rr["sweep"])
        print(f"  {m:>4}: Disc−None shallow={sh} deep={dp}  | gain[{sweep}]")
    # depth-sweep gain overlay
    fig, ax = plt.subplots(figsize=(config.W_SINGLE, 62 * config.MM))
    sty = {"cry": dict(ls="--", marker="s", color=config.OKABE_ITO[1]),
           "cond": dict(ls="-", marker="o", color=config.OKABE_ITO[2])}
    for m in methods:
        rr = res_by_method[m]
        deps = [p[0] for p in rr["sweep"]]; gains = [p[3] for p in rr["sweep"]]
        errs = [p[2] for p in rr["sweep"]]
        ax.errorbar(deps, gains, yerr=errs, ms=4, lw=1.3, capsize=2, elinewidth=0.6,
                    label=f"single diag pair — {m}", **sty[m])
    ax.axhline(0.0, color="0.6", lw=0.5, ls="--")
    ax.set_xlabel("Bell pair pool-hop depth from readout")
    ax.set_ylabel("Test gain vs None"); ax.set_xticks([0, 1, 2, 3, 4])
    ax.legend(loc="upper right", fontsize=5.2, handlelength=1.6)
    ax.set_title("Exp 2. Depth-gated entanglement — CRY vs cond", fontsize=6.2)
    out = os.path.join(config.OUTPUT_DIR, "08_deep_pooling_compare.png")
    fig.savefig(out, dpi=300); plt.close(fig)
    print(f"\nsaved comparison: {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="cry", choices=["cry", "cond", "both"])
    args = ap.parse_args()
    methods = ["cry", "cond"] if args.method == "both" else [args.method]
    res_by_method = {}
    for m in methods:
        if not glob.glob(os.path.join(_deep_dir(m), "*.npz")):
            print(f"[{m}] deep 결과 없음 (스킵)"); continue
        res_by_method[m] = run_method(m)
    if len([m for m in res_by_method if res_by_method.get(m)]) == 2:
        compare(res_by_method)


if __name__ == "__main__":
    main()
