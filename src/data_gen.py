# CPFP-v2 STEP 0: 합성 데이터 재설계 (marginal leakage 제거)
# ─────────────────────────────────────────────────────────────────────────────
# 목적: 한쪽 party 만으로는 라벨 예측 불가(Acc≈0.5), 양쪽 합쳐야만 예측 가능한
#       cross-only 합성 데이터 두 종류 생성 + leakage 검정 + head 호환 확인.
#
# 각도 규약: 모든 feature 각도는 [0, π] (encoding_head PartyAngleHead 와 일관).
# ground-truth edge set S = {(0,4),(1,5)} (A-side i, B-side j).
#
# 수학 메모(한국어):
#   cos(x_i−x_j) 의 x_j~U[0,π] marginal = (2/π)sin(x_i) ≠ 0 → marginal leakage 발생.
#   cos(2(x_i−x_j)) 의 marginal = 0 (2π 주기를 완전히 덮음) → leakage 없음.
#   b(x)=sign(x−π/2) 의 bilinear product b(x_i)b(x_j) 도 marginal=0 (b 대칭).
# 따라서 Type1(cosine) 은 frequency-2 를 채택(freq-1 은 leak 시연용으로 함께 검정),
#        Type2(non-cosine) 는 bilinear product 사용.

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import mutual_info_classif

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from encoding_head import EncodingHead

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
RESULTS_DATA = os.path.join(ROOT, "results", "data")
FIG_DIR = os.path.join(ROOT, "figures")
for d in (DATA_DIR, RESULTS_DATA, FIG_DIR):
    os.makedirs(d, exist_ok=True)

SEED = 2024
N = 2048
PARTY_A = [0, 1, 2, 3]
PARTY_B = [4, 5, 6, 7]
S_TRUE = [(0, 4), (1, 5)]
GAMMA = {(0, 4): 1.0, (1, 5): 0.6}      # 비대칭 가중 → 정확 상쇄 회피
OI = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]


# ─────────────────────────────────────────────────────────────────────────────
# 1. 데이터 생성
# ─────────────────────────────────────────────────────────────────────────────
def _features(rng):
    """8 feature ~ Uniform[0, π]."""
    return rng.uniform(0.0, np.pi, size=(N, 8))


def gen_cosine(freq=2, sigma=0.40, seed=SEED):
    """Type 1: y = sign( Σ_{(i,j)∈S} γ_ij cos(freq·(x_i − x_j)) + ε ). local term 없음."""
    rng = np.random.RandomState(seed)
    X = _features(rng)
    score = np.zeros(N)
    for (i, j), g in GAMMA.items():
        score += g * np.cos(freq * (X[:, i] - X[:, j]))
    score += rng.normal(0, sigma, size=N)
    y = np.where(score >= 0, 1.0, -1.0)
    return X, y


def gen_noncosine(sigma=0.30, seed=SEED + 1):
    """Type 2: y = sign( Σ γ_ij b(x_i)·b(x_j) + ε ), b(x)=sign(x−π/2). bilinear product."""
    rng = np.random.RandomState(seed)
    X = _features(rng)
    b = np.sign(X - np.pi / 2.0)
    score = np.zeros(N)
    for (i, j), g in GAMMA.items():
        score += g * b[:, i] * b[:, j]
    score += rng.normal(0, sigma, size=N)
    y = np.where(score >= 0, 1.0, -1.0)
    return X, y


# ─────────────────────────────────────────────────────────────────────────────
# 2. Marginal leakage 검정
# ─────────────────────────────────────────────────────────────────────────────
def _clf_acc(Xin, y, kind, n_rep=3):
    """train/test split 반복 평균 test accuracy. kind: 'logistic' | 'mlp'."""
    accs = []
    for s in range(n_rep):
        rng = np.random.RandomState(100 + s)
        idx = rng.permutation(len(y))
        ntr = int(0.7 * len(y))
        tr, te = idx[:ntr], idx[ntr:]
        if kind == "logistic":
            clf = LogisticRegression(max_iter=2000)
        else:
            clf = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=800,
                                early_stopping=True, random_state=s)
        clf.fit(Xin[tr], y[tr])
        accs.append(clf.score(Xin[te], y[te]))
    return float(np.mean(accs))


def leakage_test(X, y):
    """Acc(A→y), Acc(B→y), Acc(A,B→y) 를 logistic·mlp 로 측정 + per-feature MI."""
    XA, XB, XAB = X[:, PARTY_A], X[:, PARTY_B], X
    res = {}
    for name, Xin in [("A", XA), ("B", XB), ("AB", XAB)]:
        res[name] = {"logistic": _clf_acc(Xin, y, "logistic"),
                     "mlp": _clf_acc(Xin, y, "mlp")}
        res[name]["worst"] = max(res[name]["logistic"], res[name]["mlp"])
    yb = (y > 0).astype(int)
    mi = mutual_info_classif(X, yb, discrete_features=False, random_state=SEED)
    res["mi_per_feature"] = mi.tolist()
    # 판정: max(Acc_A, Acc_B) < 0.55 이고 Acc_AB 유의하게 높음(>0.62)
    a, b, ab = res["A"]["worst"], res["B"]["worst"], res["AB"]["worst"]
    res["pass"] = bool(max(a, b) < 0.55 and ab > 0.62)
    res["margin_fail"] = bool(max(a, b) > 0.60)
    res["class_balance_pos"] = float((y > 0).mean())
    return res


# ─────────────────────────────────────────────────────────────────────────────
# 3. head 호환 sanity
# ─────────────────────────────────────────────────────────────────────────────
def head_sanity(X):
    head = EncodingHead(A_cols=PARTY_A, B_cols=PARTY_B, mode="binary")
    rng = np.random.RandomState(SEED)
    idx = rng.permutation(len(X)); ntr = int(0.7 * len(X))
    head.fit(X[idx[:ntr]])
    out = head.transform(X)
    ang = out["angles"]; pm1 = out["pm1_features"]
    return {
        "ang_min": float(ang.min()), "ang_max": float(ang.max()),
        "ang_in_0_pi": bool(ang.min() >= -1e-9 and ang.max() <= np.pi + 1e-6),
        "pm1_values": sorted(np.unique(pm1).tolist()),
        "pm1_shape": list(pm1.shape),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. 저장 / 그림 / 메인
# ─────────────────────────────────────────────────────────────────────────────
def save_dataset(tag, X, y, leak, sanity):
    meta = {"S_true": [list(e) for e in S_TRUE], "gamma": {f"{i}-{j}": g for (i, j), g in GAMMA.items()},
            "angle_range": [0, float(np.pi)], "N": int(len(y)),
            "leakage": leak, "head_sanity": sanity}
    np.savez(os.path.join(DATA_DIR, f"synth_{tag}.npz"), X=X, y=y,
             S_true=np.array(S_TRUE), meta=json.dumps(meta))
    json.dump(meta, open(os.path.join(DATA_DIR, f"synth_{tag}_meta.json"), "w"), indent=2)
    # CSV (col0-7 feature, col8 label) — 검수용
    import csv
    with open(os.path.join(DATA_DIR, f"synth_{tag}.csv"), "w", newline="") as fp:
        w = csv.writer(fp); w.writerow([str(k) for k in range(8)] + ["label"])
        for n in range(len(y)):
            w.writerow(list(np.round(X[n], 6)) + [int(y[n])])


def set_style():
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
        "font.size": 7, "axes.titlesize": 8, "axes.labelsize": 7,
        "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
        "axes.linewidth": 0.7, "savefig.dpi": 350, "figure.dpi": 150,
    })


def bar_panel(ax, leak, title):
    groups = ["A-only", "B-only", "A+B"]
    log = [leak["A"]["logistic"], leak["B"]["logistic"], leak["AB"]["logistic"]]
    mlp = [leak["A"]["mlp"], leak["B"]["mlp"], leak["AB"]["mlp"]]
    x = np.arange(3); w = 0.38
    ax.bar(x - w/2, log, w, color=OI[0], label="logistic")
    ax.bar(x + w/2, mlp, w, color=OI[1], label="MLP")
    ax.axhline(0.5, color="0.4", ls="--", lw=0.8, label="chance 0.5")
    ax.axhline(0.55, color=OI[2], ls=":", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(groups)
    ax.set_ylim(0.4, 1.0); ax.set_ylabel("Test accuracy")
    verdict = "PASS" if leak["pass"] else "LEAK"
    ax.set_title(f"{title}  [{verdict}]")
    ax.legend(loc="upper left")


def main():
    print("=" * 64); print("STEP 0: 합성 데이터 재설계 + leakage 검정"); print("=" * 64)
    set_style()

    # Type1 freq-1 (leak 시연), freq-2 (채택 후보), Type2
    datasets = {}
    print("\n[생성 & 검정]")
    for tag, (X, y) in {
        "cosine_freq1": gen_cosine(freq=1),
        "cosine": gen_cosine(freq=2),
        "noncosine": gen_noncosine(),
    }.items():
        leak = leakage_test(X, y)
        sanity = head_sanity(X)
        datasets[tag] = (X, y, leak, sanity)
        print(f"\n── {tag}  (pos_frac={leak['class_balance_pos']:.3f})")
        print(f"   Acc  A-only={leak['A']['worst']:.3f}  B-only={leak['B']['worst']:.3f}  "
              f"A+B={leak['AB']['worst']:.3f}   → {'PASS' if leak['pass'] else 'LEAK/FAIL'}")
        print(f"   head: ang∈[{sanity['ang_min']:.3f},{sanity['ang_max']:.3f}] "
              f"in[0,π]={sanity['ang_in_0_pi']}  pm1={sanity['pm1_values']}")

    # 채택: cosine(freq2), noncosine. freq1 은 저장 안 함(leak 시연용).
    accepted = []
    for tag in ["cosine", "noncosine"]:
        X, y, leak, sanity = datasets[tag]
        if leak["pass"]:
            save_dataset(tag, X, y, leak, sanity)
            accepted.append(tag)
            print(f"\n[저장] synth_{tag}.npz/.csv/_meta.json  (PASS)")
        else:
            print(f"\n[경고] {tag} leakage 검정 미통과 — 저장 보류 (재설계 필요)")

    # 그림: 3 패널 (freq1 reject / cosine accept / noncosine accept)
    fig, ax = plt.subplots(1, 3, figsize=(9.5, 3.3))
    bar_panel(ax[0], datasets["cosine_freq1"][2], "(a) cosine freq=1 (rejected)")
    bar_panel(ax[1], datasets["cosine"][2], "(b) cosine freq=2 (Type 1)")
    bar_panel(ax[2], datasets["noncosine"][2], "(c) non-cosine bilinear (Type 2)")
    fig.suptitle("Marginal leakage test: single-party accuracy must ≈ 0.5", y=1.02)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "step0_leakage_test.png")
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    print(f"\n[그림] {out}")

    # 검정 결과 저장
    summary = {tag: {"leakage": d[2], "head_sanity": d[3]} for tag, d in datasets.items()}
    summary["accepted"] = accepted
    summary["S_true"] = [list(e) for e in S_TRUE]
    json.dump(summary, open(os.path.join(RESULTS_DATA, "step0_summary.json"), "w"), indent=2)
    print("\n[요약 저장] results/data/step0_summary.json")
    print("accepted datasets:", accepted)


if __name__ == "__main__":
    main()
