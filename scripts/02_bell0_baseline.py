# STEP 2. 4+4 회로 + Bell-0 (K=0) baseline 학습
#   - Bell pair 없이(얽힘 X) 두 데이터셋(match2/pair3) × {pooling 有/無} baseline 학습
#   - test accuracy 표 출력 + 막대그래프 저장
#   - best param 을 cache/ 에 저장 → STEP 3 residual 진단에서 재사용
#
#   가설상 기대: match2(cross 구조)는 얽힘 없는 baseline이 cross를 못 잡아 낮고,
#                pair3(marginal 구조)는 baseline만으로도 비교적 잘 맞아야 한다.

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config, data, circuit as C, model as M

config.apply_style()
np.random.seed(config.SEED)

# ────────────────────────────────────────
# 0. 전역 상수
# ────────────────────────────────────────
SPLIT_SEED = config.SEED          # 모든 config가 동일한 train/test 분할 사용
N_TRAIN, N_TEST = 400, 400
N_STEPS = 150
LR = 0.05
BASELINE_SEEDS = [0, 1, 2]        # param 초기화 시드
POOLING_MODES = [True, False]     # QCNN(pool) vs 순수 유니터리


def train_baseline(name, pooling):
    """한 (데이터셋, pooling) 조합의 K=0 baseline 을 여러 시드로 학습.

    반환: dict(accs, best_params, best_seed, n_q, train/test idx)
    """
    ds = data.load_dataset(name)
    tr, te = M.split_train_test(ds, N_TRAIN, N_TEST, SPLIT_SEED)
    qnode = C.make_qnode(pooling=pooling, bell_pairs=None)   # K=0
    n_q = C.n_quantum_params(pooling)

    accs, best = [], None
    for s in BASELINE_SEEDS:
        params, _ = M.train_model(qnode, n_q, tr["X"], tr["y"],
                                  N_STEPS, LR, seed=s)
        acc = M.accuracy(params, te["X"], te["y"], qnode, n_q)
        tr_acc = M.accuracy(params, tr["X"], tr["y"], qnode, n_q)
        accs.append(acc)
        if best is None or acc > best["test_acc"]:
            best = {"params": params, "seed": s, "test_acc": acc, "train_acc": tr_acc}
        print(f"    seed {s}: train={tr_acc:.3f}  test={acc:.3f}")
    return {
        "accs": np.array(accs), "best_params": best["params"],
        "best_seed": best["seed"], "best_test": best["test_acc"],
        "best_train": best["train_acc"], "n_q": n_q,
        "train_idx": tr["idx"], "test_idx": te["idx"],
    }


def main():
    print("=" * 64)
    print("STEP 2. Bell-0 (K=0) baseline  —  얽힘 없는 로컬 모델")
    print("=" * 64)

    results = {}   # (name, pooling) -> dict
    for name in config.DATASETS:
        for pooling in POOLING_MODES:
            tag = "pool" if pooling else "nopool"
            print(f"\n── {name} | {tag} (n_q={C.n_quantum_params(pooling)}) ──")
            r = train_baseline(name, pooling)
            results[(name, pooling)] = r
            # cache 저장 (STEP 3 재사용)
            cpath = os.path.join(config.CACHE_DIR, f"baseline_{name}_{tag}.npz")
            np.savez(cpath, params=r["best_params"], n_q=r["n_q"],
                     pooling=pooling, dataset=name, split_seed=SPLIT_SEED,
                     train_idx=r["train_idx"], test_idx=r["test_idx"],
                     accs=r["accs"], best_seed=r["best_seed"])
            print(f"    saved cache: {cpath}")

    # ── 정확도 표 ────────────────────────────────────────────
    print("\n" + "=" * 64)
    print("STEP 2 결과: Bell-0 baseline test accuracy (mean±SEM over seeds)")
    print("=" * 64)
    print(f"{'dataset':>10} | {'pooling':>8} | {'mean±SEM':>14} | {'best':>6} | seeds")
    print("-" * 64)
    for name in config.DATASETS:
        for pooling in POOLING_MODES:
            r = results[(name, pooling)]
            a = r["accs"]; m = a.mean(); sem = a.std(ddof=1) / np.sqrt(len(a))
            tag = "pool" if pooling else "nopool"
            print(f"{name:>10} | {tag:>8} | {m:.3f} ± {sem:.3f}  | "
                  f"{r['best_test']:.3f} | {np.round(a, 3)}")

    # ── 막대그래프 ────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(config.W_SINGLE, 62 * config.MM))
    names = list(config.DATASETS)
    xpos = np.arange(len(names)); width = 0.36
    for k, pooling in enumerate(POOLING_MODES):
        m = [results[(n, pooling)]["accs"].mean() for n in names]
        e = [results[(n, pooling)]["accs"].std(ddof=1) / np.sqrt(len(BASELINE_SEEDS))
             for n in names]
        ax.bar(xpos + (k - 0.5) * width, m, width, yerr=e, capsize=2,
               color=config.OKABE_ITO[0 if pooling else 1],
               label=("QCNN (pool)" if pooling else "Unitary (no pool)"),
               error_kw={"elinewidth": 0.6})
    ax.axhline(0.5, color="0.6", lw=0.5, ls="--")
    ax.text(len(names) - 1.1, 0.515, "chance", fontsize=5.2, color="0.5")
    ax.set_xticks(xpos)
    ax.set_xticklabels([f"{n}\n({'cross' if n=='match2' else 'marginal'})" for n in names])
    ax.set_ylabel("Test accuracy")
    ax.set_ylim(0.45, 1.0)
    ax.legend(loc="upper right", handlelength=1.2)
    ax.set_title("STEP 2. Bell-0 baseline (no entanglement)", fontsize=6.6)
    out = os.path.join(config.OUTPUT_DIR, "02_bell0_baseline.png")
    fig.savefig(out, dpi=300); fig.savefig(out.replace(".png", ".pdf"))
    plt.close(fig)
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
