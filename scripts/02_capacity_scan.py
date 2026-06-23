# STEP 2 (확장). 로컬 모델 capacity 스캔
#   핵심 질문: "충분히 강한 *로컬* 모델이면 pair3(순수 marginal)는 풀리는가?
#              그리고 match2(cross)는 같은 capacity에서도 안 풀리는가?"
#   - 파티 간 게이트는 절대 없음(강한 로컬 모델일 뿐).
#   - n_blocks(깊이)를 키우며 두 데이터셋 train/test acc 기록 → capacity-vs-acc 곡선.
#   - 해석:
#       pair3 ↑(0.85+) & match2 가 뚜렷이 낮게 남음 → 가설 정합(cross는 로컬로 불가).
#       pair3 가 안 오름 → readout 구조적 병목 → STOP & 재설계.
#   결과 워커(_scan_worker.py)가 cache/scan/*.npz 에 저장한 것을 집계/플롯한다.

import os
import sys
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config, circuits


def scan_dir(method):
    return circuits.cache_subdir("scan", method)


def load_scan(method):
    """cache/(cond/)scan/*.npz 를 (dataset, nb) → {seed: (train,test,file)} 로 모은다."""
    recs = {}
    for f in sorted(glob.glob(os.path.join(scan_dir(method), "*.npz"))):
        z = np.load(f, allow_pickle=True)
        key = (str(z["dataset"]), int(z["n_blocks"]))
        recs.setdefault(key, {})[int(z["seed"])] = {
            "train": float(z["train_acc"]), "test": float(z["test_acc"]),
            "n_q": int(z["n_q"]), "file": f,
        }
    return recs


def aggregate(method):
    """한 method의 (dataset,nb)→통계 dict 반환 + 콘솔 표 출력."""
    recs = load_scan(method)
    if not recs:
        return None, [], []
    datasets = sorted({k[0] for k in recs})
    nbs = sorted({k[1] for k in recs})
    print(f"\n[{method}] dataset nb n_q | train | test")
    agg = {}
    for d in datasets:
        for nb in nbs:
            if (d, nb) not in recs:
                continue
            seeds = recs[(d, nb)]
            tr = np.array([v["train"] for v in seeds.values()])
            te = np.array([v["test"] for v in seeds.values()])
            n_q = list(seeds.values())[0]["n_q"]
            best = max(seeds.values(), key=lambda v: v["test"])
            sem = lambda x: x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0
            agg[(d, nb)] = dict(train_m=tr.mean(), train_s=sem(tr),
                                test_m=te.mean(), test_s=sem(te), n_q=n_q,
                                best_file=best["file"], best_test=best["test"])
            print(f"  {d:>8} {nb:>3} {n_q:>4} | {tr.mean():.3f}±{sem(tr):.3f} "
                  f"| {te.mean():.3f}±{sem(te):.3f}")
    return agg, datasets, nbs


def _print_diag(method, agg, nbs):
    print(f"\n[{method}] 진단: pair3(marginal) 풀리고 match2(cross) 막히나?")
    for nb in nbs:
        if ("pair3", nb) in agg and ("match2", nb) in agg:
            p = agg[("pair3", nb)]["test_m"]; m = agg[("match2", nb)]["test_m"]
            pt = agg[("pair3", nb)]["train_m"]; mt = agg[("match2", nb)]["train_m"]
            print(f"  nb={nb}: pair3 test={p:.3f}(tr {pt:.3f})  "
                  f"match2 test={m:.3f}(tr {mt:.3f})  gap={p-m:+.3f}")


def main():
    config.apply_style()
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="cry", choices=list(circuits.METHODS) + ["both"])
    args = ap.parse_args()
    methods = list(circuits.METHODS) if args.method == "both" else [args.method]

    aggs = {}
    for mth in methods:
        agg, datasets, nbs = aggregate(mth)
        if agg is None:
            print(f"[{mth}] 결과 없음 (스킵)")
            continue
        aggs[mth] = (agg, datasets, nbs)
        _single_figure(mth, agg, datasets, nbs)
        _print_diag(mth, agg, nbs)

    if len(aggs) == 2:
        _compare_figure(aggs)
    print()


def _single_figure(method, agg, datasets, nbs):
    """capacity(n_blocks) vs accuracy 곡선 (method별 별도 파일)."""
    colors = {"match2": config.OKABE_ITO[4], "pair3": config.OKABE_ITO[2]}
    fig, ax = plt.subplots(figsize=(config.W_SINGLE * 1.15, 68 * config.MM))
    for d in datasets:
        xs = [nb for nb in nbs if (d, nb) in agg]
        tem = [agg[(d, nb)]["test_m"] for nb in xs]
        tes = [agg[(d, nb)]["test_s"] for nb in xs]
        trm = [agg[(d, nb)]["train_m"] for nb in xs]
        lab = f"{d} ({'cross' if d=='match2' else 'marginal'})"
        ax.errorbar(xs, tem, yerr=tes, marker="o", ms=3, lw=1.2, capsize=2,
                    elinewidth=0.6, color=colors.get(d, "k"),
                    label=f"{lab} — test")
        ax.plot(xs, trm, marker="s", ms=2.4, lw=0.8, ls="--",
                color=colors.get(d, "k"), alpha=0.6, label=f"{lab} — train")
    ax.axhline(0.5, color="0.6", lw=0.5, ls=":")
    ax.text(nbs[0], 0.515, "chance", fontsize=5.2, color="0.5")
    ax.axhline(0.85, color="0.8", lw=0.5, ls=":")
    ax.text(nbs[0], 0.86, "target 0.85", fontsize=5.2, color="0.6")
    ax.set_xlabel("Local model capacity (n_blocks, no cross-party gates)")
    ax.set_ylabel("Accuracy")
    ax.set_xticks(nbs)
    ax.set_ylim(0.45, 1.0)
    ax.legend(loc="lower right", fontsize=5.0, handlelength=1.4, ncol=1)
    pool_name = {"cry": "CRY", "cond": "cond (Cong)"}[method]
    ax.set_title(f"STEP 2 [{pool_name} pooling]. marginal solvable, cross is not",
                 fontsize=6.4)
    suffix = "" if method == "cry" else f"_{method}"
    out = os.path.join(config.OUTPUT_DIR, f"02_capacity_scan{suffix}.png")
    fig.savefig(out, dpi=300); fig.savefig(out.replace(".png", ".pdf"))
    plt.close(fig)
    print(f"  saved: {out}")


def _compare_figure(aggs):
    """CRY vs cond 나란히 비교(test acc). 두 방식 모두 같은 질적 결론을 보이는지."""
    colors = {"match2": config.OKABE_ITO[4], "pair3": config.OKABE_ITO[2]}
    styles = {"cry": dict(ls="--", marker="s", alpha=0.85),
              "cond": dict(ls="-", marker="o", alpha=1.0)}
    fig, ax = plt.subplots(figsize=(config.W_SINGLE * 1.25, 70 * config.MM))
    for mth, (agg, datasets, nbs) in aggs.items():
        for d in datasets:
            xs = [nb for nb in nbs if (d, nb) in agg]
            tem = [agg[(d, nb)]["test_m"] for nb in xs]
            tes = [agg[(d, nb)]["test_s"] for nb in xs]
            ax.errorbar(xs, tem, yerr=tes, ms=3, lw=1.1, capsize=2, elinewidth=0.5,
                        color=colors.get(d, "k"), **styles[mth],
                        label=f"{d} — {mth}")
    ax.axhline(0.5, color="0.6", lw=0.5, ls=":")
    ax.axhline(0.85, color="0.8", lw=0.5, ls=":")
    ax.set_xlabel("Local model capacity (n_blocks)")
    ax.set_ylabel("Test accuracy")
    nbs_all = sorted({nb for (_, _, nbs) in aggs.values() for nb in nbs})
    ax.set_xticks(nbs_all)
    ax.set_ylim(0.45, 1.0)
    ax.legend(loc="lower right", fontsize=5.0, handlelength=1.8, ncol=2)
    ax.set_title("STEP 2. CRY vs cond pooling — same qualitative split",
                 fontsize=6.4)
    out = os.path.join(config.OUTPUT_DIR, "02_capacity_scan_compare.png")
    fig.savefig(out, dpi=300); fig.savefig(out.replace(".png", ".pdf"))
    plt.close(fig)
    print(f"\nsaved comparison: {out}")


if __name__ == "__main__":
    main()
