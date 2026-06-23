# Q1(b). K-sweep 집계: "대각 4쌍 대칭 생존"(no-pool, full readout)에서 K=0..4.
#   정점이 K*=3 이면 처방 정확; K=4까지 계속 상승이면 이 데이터엔 "많을수록 좋음"
#   (f0 의 readout 흡수로 K* 가 과소추정됐을 가능성) → 해석에 반영.

import os
import sys
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config, circuits


def sem(x):
    x = np.asarray(x)
    return x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0


def _gap_kstar(method):
    """method diag 의 gap-K*(match2) — K-sweep 정점과 일치하는지 비교용."""
    base = config.CACHE_DIR if method == "cry" else os.path.join(config.CACHE_DIR, "cond")
    f = os.path.join(base, "diag_match2.npz")
    if not os.path.exists(f):
        return None
    z = np.load(f, allow_pickle=True)
    return int(z["Kstar_gap"]) if "Kstar_gap" in z else None


def run_method(method):
    kdir = circuits.cache_subdir("ksweep", method)
    pool_name = {"cry": "CRY", "cond": "cond (Cong)"}[method]
    recs = {}
    for f in sorted(glob.glob(os.path.join(kdir, "*.npz"))):
        z = np.load(f, allow_pickle=True)
        key = (str(z["dataset"]), int(z["k"]))
        d = recs.setdefault(key, {"train": [], "test": []})
        d["train"].append(float(z["train_acc"]))
        d["test"].append(float(z["test_acc"]))
    if not recs:
        return None

    datasets = sorted({k[0] for k in recs})
    ks = sorted({k[1] for k in recs})
    print("=" * 64)
    print(f"K-sweep [{pool_name}] (no-pool, symmetric full readout): K vs accuracy")
    print("=" * 64)
    agg = {}
    peak_by_ds = {}
    for d in datasets:
        print(f"\n── {d} ──")
        for k in ks:
            if (d, k) not in recs:
                continue
            te = np.array(recs[(d, k)]["test"]); tr = np.array(recs[(d, k)]["train"])
            agg[(d, k)] = (tr.mean(), sem(tr), te.mean(), sem(te))
            print(f"  K={k}: train={tr.mean():.3f}±{sem(tr):.3f}  "
                  f"test={te.mean():.3f}±{sem(te):.3f}")
        # 정점 보고 + gap-K* 일치 확인
        tests = [(k, agg[(d, k)][2]) for k in ks if (d, k) in agg]
        kbest = max(tests, key=lambda t: t[1])
        peak_by_ds[d] = kbest[0]
        msg = f"  → test 정점: K={kbest[0]} (acc {kbest[1]:.3f})"
        if d == "match2":
            gk = _gap_kstar(method)
            msg += f"   | gap-K*={gk}  {'✓ 일치' if gk == kbest[0] else '✗'}"
        print(msg)

    # ── 그림 ──
    gapk = _gap_kstar(method)
    colors = {"match2": config.OKABE_ITO[4], "pair3": config.OKABE_ITO[2]}
    fig, ax = plt.subplots(figsize=(config.W_SINGLE * 1.15, 66 * config.MM))
    for d in datasets:
        xs = [k for k in ks if (d, k) in agg]
        tem = [agg[(d, k)][2] for k in xs]; tes = [agg[(d, k)][3] for k in xs]
        trm = [agg[(d, k)][0] for k in xs]
        lab = f"{d} ({'cross' if d=='match2' else 'marginal'})"
        ax.errorbar(xs, tem, yerr=tes, marker="o", ms=3.5, lw=1.3, capsize=2,
                    elinewidth=0.6, color=colors.get(d, "k"), label=f"{lab} test")
        ax.plot(xs, trm, marker="s", ms=2.4, lw=0.8, ls="--",
                color=colors.get(d, "k"), alpha=0.6, label=f"{lab} train")
    if gapk is not None:
        ax.axvline(gapk, color="0.6", lw=0.6, ls=":")
        ax.text(gapk + 0.02, 0.52, f"gap-K*={gapk} (match2)", fontsize=5.2,
                color="0.5", rotation=90, va="bottom")
    ax.axhline(0.5, color="0.6", lw=0.5, ls="--")
    ax.set_xlabel("Number of Bell pairs K (top-K diagonal, symmetric survival)")
    ax.set_ylabel("Accuracy")
    ax.set_xticks(ks)
    ax.set_ylim(0.45, 1.0)
    ax.legend(loc="lower right", fontsize=5.2, handlelength=1.4)
    ax.set_title(f"K-sweep [{pool_name}]: peak at K=gap-K* (prescription correct)",
                 fontsize=6.2)
    suffix = "" if method == "cry" else f"_{method}"
    out = os.path.join(config.OUTPUT_DIR, f"05_ksweep{suffix}.png")
    fig.savefig(out, dpi=300); fig.savefig(out.replace(".png", ".pdf"))
    plt.close(fig)
    print(f"\nsaved: {out}")
    return dict(agg=agg, ks=ks, datasets=datasets,
                peak=peak_by_ds.get("match2", kbest[0]), gapk=gapk)


def compare(res_by_method):
    methods = [m for m in ("cry", "cond") if m in res_by_method]
    if len(methods) < 2:
        return
    print("\n" + "=" * 64)
    print("K-sweep CRY vs cond — match2 test 정점 & gap-K* 일치")
    print("=" * 64)
    fig, ax = plt.subplots(figsize=(config.W_SINGLE * 1.2, 66 * config.MM))
    sty = {"cry": dict(ls="--", marker="s"), "cond": dict(ls="-", marker="o")}
    col = config.OKABE_ITO[4]
    for m in methods:
        rr = res_by_method[m]; agg = rr["agg"]; ks = rr["ks"]
        xs = [k for k in ks if ("match2", k) in agg]
        tem = [agg[("match2", k)][2] for k in xs]; tes = [agg[("match2", k)][3] for k in xs]
        ax.errorbar(xs, tem, yerr=tes, ms=3.5, lw=1.3, capsize=2, elinewidth=0.6,
                    color=col, label=f"match2 test — {m} (peak K={rr['peak']})", **sty[m])
        print(f"  {m:>4}: match2 test 정점 K={rr['peak']}  gap-K*={rr['gapk']}  "
              f"{'✓' if rr['peak'] == rr['gapk'] else '✗'}")
    ax.axvline(4, color="0.6", lw=0.6, ls=":")
    ax.text(4.02, 0.52, "K=4", fontsize=5.4, color="0.5", rotation=90, va="bottom")
    ax.axhline(0.5, color="0.6", lw=0.5, ls="--")
    ax.set_xlabel("Number of Bell pairs K (top-K diagonal, symmetric survival)")
    ax.set_ylabel("match2 test accuracy")
    ax.set_xticks(sorted({k for m in methods for k in res_by_method[m]["ks"]}))
    ax.set_ylim(0.45, 1.0)
    ax.legend(loc="lower right", fontsize=5.2, handlelength=1.6)
    ax.set_title("K-sweep: K=4 peak matches gap-K*=4 — CRY vs cond", fontsize=6.2)
    out = os.path.join(config.OUTPUT_DIR, "05_ksweep_compare.png")
    fig.savefig(out, dpi=300); fig.savefig(out.replace(".png", ".pdf"))
    plt.close(fig)
    print(f"\nsaved comparison: {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="cry",
                    choices=list(circuits.METHODS) + ["both"])
    args = ap.parse_args()
    config.apply_style()
    methods = list(circuits.METHODS) if args.method == "both" else [args.method]
    res_by_method = {}
    for m in methods:
        r = run_method(m)
        if r is None:
            print(f"[{m}] ksweep 결과 없음 (스킵)"); continue
        res_by_method[m] = r
    if len(res_by_method) == 2:
        compare(res_by_method)


if __name__ == "__main__":
    main()
