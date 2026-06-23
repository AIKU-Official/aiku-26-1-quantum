# CPFP-v2 STEP 1: loss 통일(BCE) + loss-consistent residual + Metric B 미분 가능성 사전 확인
# ─────────────────────────────────────────────────────────────────────────────
# 목적:
#   (1-1) 전 실험 BCE(logistic) 통일. label ±1 → t=(y+1)/2 ∈ {0,1}.
#   (1-2) Bell-0(E=∅) 회로를 BCE 로 재학습(Θ + W). residual g_n=t_n−p0(x_n),
#         curvature h_n=p0(1−p0), H=diag(h). 기존 r=y−f0 대신 g_n 사용.
#   (1-3) ★Metric B probe 미분 가능성: 학습된 Bell-0 기준, edge probe 회로
#         (pre-shared Bell resource RY(θ)+CNOT → local embedding → local QCNN)
#         의 u_e=∂z/∂θ|_{θ=0} 가 defer_measurements 회로에서 유한·nonzero 인지.
#
# 회로 입력은 head 경유([0,π] 각도), Fourier 입력은 pm1(±1). seed 2024 기본, 절대경로.
#
# ★사전 발견(중요): C.init_params 기본 W=[.25,.25,.25,.25](균등)이면
#   z=W·probs=0.25·Σprobs=0.25 로 상수 → tangent 항등 0. 따라서 probe 는
#   반드시 '학습된(또는 non-degenerate) W' 기준이어야 한다(1-3 사양과 일치).

import os, json, time
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import circuit as C
from encoding_head import EncodingHead

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
RESULTS_DATA = os.path.join(ROOT, "results", "data")
FIG_DIR = os.path.join(ROOT, "figures")
for d in (RESULTS_DATA, FIG_DIR):
    os.makedirs(d, exist_ok=True)

PARTY_A, PARTY_B = [0, 1, 2, 3], [4, 5, 6, 7]
DIAG_CROSS = [(0, 4), (1, 5), (2, 6), (3, 7)]
DATASETS = ["cosine", "noncosine"]
SEEDS = [2024, 2025, 2026]          # STEP 1: 3 seed (seed별 학습곡선용). STEP 3 에서 6 seed.
N_TRAIN, N_TEST = 512, 512
EPOCHS, LR = 60, 0.05
OI = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]


# ─────────────────────────────────────────────────────────────────────────────
# 데이터: synth npz → head 경유
# ─────────────────────────────────────────────────────────────────────────────
def load_synth(tag, split_seed=2024):
    """synth_{tag}.npz → train/test split → head(train fit) → angles + pm1."""
    d = np.load(os.path.join(DATA_DIR, f"synth_{tag}.npz"), allow_pickle=True)
    Xraw, y = d["X"].astype(float), d["y"].astype(float)
    N = len(y)
    rng = np.random.RandomState(split_seed)
    perm = rng.permutation(N)
    idx_tr, idx_te = perm[:N_TRAIN], perm[N_TRAIN:N_TRAIN + N_TEST]

    head = EncodingHead(A_cols=PARTY_A, B_cols=PARTY_B, mode="binary")
    head.fit(Xraw[idx_tr])                                  # train 만 fit (누수 방지)
    ang_tr = head.transform(Xraw[idx_tr])["angles"]
    ang_te = head.transform(Xraw[idx_te])["angles"]
    full = head.transform(Xraw)
    return {
        "tag": tag, "Xraw": Xraw, "y": y,
        "Xtr_ang": ang_tr, "Xte_ang": ang_te, "Xfull_ang": full["angles"],
        "pm1_full": full["pm1_features"],
        "ytr": y[idx_tr], "yte": y[idx_te], "yfull": y,
        "idx_tr": idx_tr, "idx_te": idx_te,
        "S_true": d["S_true"].tolist(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# BCE 학습 (Bell-0, E=∅). z=probs·W = logit, p0=σ(z), t=(y+1)/2.
# ─────────────────────────────────────────────────────────────────────────────
def _sigmoid(z):
    return 1.0 / (1.0 + pnp.exp(-z))


def logits(qn, ang, p, theta, E_list, chunk=512):
    """z(x)=probs·W (logit). ang:(N,8)→(N,)."""
    out = []
    for s in range(0, ang.shape[0], chunk):
        xb = pnp.array(ang[s:s + chunk].T, requires_grad=False)
        pr = qn(xb, p, theta, E_list)
        out.append(np.asarray(pr) @ np.asarray(p["W"]))
    return np.concatenate(out)


def bce_np(z, t):
    p0 = 1.0 / (1.0 + np.exp(-z)); eps = 1e-7
    p0 = np.clip(p0, eps, 1 - eps)
    return float(-np.mean(t * np.log(p0) + (1 - t) * np.log(1 - p0)))


def acc_from_logit(z, y):
    return float((np.where(z >= 0.0, 1.0, -1.0) == y).mean())


def train_bell0_bce(ds, seed, epochs=EPOCHS, lr=LR):
    """E=∅ Bell-0 을 BCE 로 학습. residual g/h + 학습곡선 반환."""
    qn = C.make_qnode()
    p0 = C.init_seed_params(seed)                 # ★non-degenerate W (random)
    theta0, E_list = C.init_entangler(set(), seed=seed)   # E=∅
    n_ent = 0

    xT = pnp.array(ds["Xtr_ang"].T, requires_grad=False)
    t_tr = pnp.array((ds["ytr"] + 1.0) / 2.0, requires_grad=False)   # {0,1}
    W_before = np.asarray(p0["W"]).copy()
    flat = C.pack(p0, theta0)

    def cost(flat):
        d, th = C.unpack(flat, n_ent)
        pr = qn(xT, d, th, E_list)
        z = pr @ d["W"]
        p = _sigmoid(z); eps = 1e-7; p = pnp.clip(p, eps, 1 - eps)
        return -pnp.mean(t_tr * pnp.log(p) + (1 - t_tr) * pnp.log(1 - p))

    opt = qml.AdamOptimizer(stepsize=lr)
    hist = {"train_bce": [], "test_bce": [], "train_acc": [], "test_acc": [], "grad_norm": []}
    for ep in range(epochs):
        g = qml.grad(cost)(flat)
        gnorm = float(np.linalg.norm(np.asarray(g)))
        flat, c = opt.step_and_cost(cost, flat)
        d, th = C.unpack(flat, n_ent)
        z_tr = logits(qn, ds["Xtr_ang"], d, th, E_list)
        z_te = logits(qn, ds["Xte_ang"], d, th, E_list)
        hist["train_bce"].append(float(c))
        hist["test_bce"].append(bce_np(z_te, (ds["yte"] + 1) / 2))
        hist["train_acc"].append(acc_from_logit(z_tr, ds["ytr"]))
        hist["test_acc"].append(acc_from_logit(z_te, ds["yte"]))
        hist["grad_norm"].append(gnorm)

    d, th = C.unpack(flat, n_ent)
    W_after = np.asarray(d["W"]).copy()
    # full set residual (M 진단/metric 입력용)
    z_full = logits(qn, ds["Xfull_ang"], d, th, E_list)
    p0_full = 1.0 / (1.0 + np.exp(-z_full))
    t_full = (ds["yfull"] + 1) / 2
    g_full = t_full - p0_full                  # 1차 residual (BCE gradient)
    h_full = p0_full * (1 - p0_full)           # curvature
    z_te = logits(qn, ds["Xte_ang"], d, th, E_list)
    return {
        "seed": seed, "tag": ds["tag"],
        "hist": {k: np.array(v) for k, v in hist.items()},
        "W_before": W_before, "W_after": W_after,
        "grad_first": hist["grad_norm"][0], "grad_last": hist["grad_norm"][-1],
        "W_delta_norm": float(np.linalg.norm(W_after - W_before)),
        "test_acc": acc_from_logit(z_te, ds["yte"]),
        "train_acc": hist["train_acc"][-1],
        "p_final": {k: np.asarray(v) for k, v in d.items()},
        "theta_final": np.asarray(th),
        "z_full": z_full, "p0_full": p0_full, "g_full": g_full, "h_full": h_full,
        "idx_tr": ds["idx_tr"], "idx_te": ds["idx_te"],
    }


# ─────────────────────────────────────────────────────────────────────────────
# 1-3. Metric B probe 미분 가능성 (pre-shared Bell resource)
# ─────────────────────────────────────────────────────────────────────────────
def make_probe_qnode():
    """probe: RY(θ)_i + CNOT(i→j) [pre-shared Bell] → local embedding → local QCNN."""
    dev = qml.device("default.qubit", wires=C.N_QUBITS)

    @qml.qnode(dev, interface="autograd", diff_method="backprop")
    @qml.defer_measurements
    def probe(x, theta, i, j, p):
        # 1) pre-shared Bell resource (RY(θ)|0>=cos|0>+sin|1>, CNOT → cos|00>+sin|11>)
        qml.RY(theta, wires=i)
        qml.CNOT(wires=[i, j])
        # 2) local embedding (이후 전부 local — cross-party gate 없음)
        for k in range(C.N_QUBITS):
            qml.RY(x[k], wires=k)
        # 3) local QCNN per party
        C._party_block(p, 0, C.PARTY_A)
        C._party_block(p, 1, C.PARTY_B)
        return qml.probs(wires=C.READOUT)

    return probe


def tangent_ue(probe, ang, p, i, j):
    """u_e(x_n)=∂z(x_n;θ)/∂θ|_{θ=0}, z=probs·W. ang:(B,8)→(B,)."""
    xT = pnp.array(ang.T, requires_grad=False)
    W = pnp.array(p["W"], requires_grad=False)

    def zvec(theta):
        pr = probe(xT, theta, i, j, p)
        return pr @ W
    return np.asarray(qml.jacobian(zvec)(pnp.array(0.0, requires_grad=True)))


def probe_check(ds, fit, edges, n_probe=32):
    """학습된 Bell-0 모델로 probe 미분 가능성 + 시간 측정 + FD cross-check."""
    probe = make_probe_qnode()
    p = fit["p_final"]
    ang = ds["Xtr_ang"][:n_probe]
    res = {}
    for (i, j) in edges:
        t0 = time.time()
        u = tangent_ue(probe, ang, p, i, j)
        dt = time.time() - t0
        # FD cross-check on sample 0
        xT = pnp.array(ang.T, requires_grad=False); W = np.asarray(p["W"]); eps = 1e-4
        zp = float(np.asarray(probe(xT, pnp.array(eps), i, j, p))[0] @ W)
        zm = float(np.asarray(probe(xT, pnp.array(-eps), i, j, p))[0] @ W)
        fd = (zp - zm) / (2 * eps)
        res[f"{i}-{j}"] = {
            "finite": bool(np.all(np.isfinite(u))),
            "nonzero": int(np.sum(np.abs(u) > 1e-12)), "n": int(u.size),
            "u_abs_mean": float(np.abs(u).mean()), "u_abs_max": float(np.abs(u).max()),
            "u0_autograd": float(u[0]), "u0_fd": float(fd),
            "fd_match": bool(abs(u[0] - fd) < 1e-6 + 1e-3 * abs(u[0])),
            "time_sec": dt,
        }
    return res


# ─────────────────────────────────────────────────────────────────────────────
# 그림
# ─────────────────────────────────────────────────────────────────────────────
def set_style():
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
        "font.size": 7, "axes.titlesize": 8, "axes.labelsize": 7,
        "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6,
        "axes.linewidth": 0.7, "savefig.dpi": 350, "figure.dpi": 150,
    })


def make_figure(fits):
    """multi-panel: 학습곡선(BCE/acc, seed별) + g/h 분포 (두 데이터셋)."""
    set_style()
    fig, ax = plt.subplots(2, 4, figsize=(13, 6))
    for r, tag in enumerate(DATASETS):
        seedfits = [f for f in fits if f["tag"] == tag]
        # (col0) train/test BCE per seed
        for k, f in enumerate(seedfits):
            ep = np.arange(1, len(f["hist"]["train_bce"]) + 1)
            ax[r, 0].plot(ep, f["hist"]["train_bce"], color=OI[k], lw=1.0, label=f"seed{f['seed']} tr")
            ax[r, 0].plot(ep, f["hist"]["test_bce"], color=OI[k], lw=1.0, ls="--")
        ax[r, 0].set_title(f"{tag}: BCE loss (—train, --test)")
        ax[r, 0].set_xlabel("epoch"); ax[r, 0].set_ylabel("BCE"); ax[r, 0].legend(ncol=1)
        # (col1) train/test acc per seed
        for k, f in enumerate(seedfits):
            ep = np.arange(1, len(f["hist"]["train_acc"]) + 1)
            ax[r, 1].plot(ep, f["hist"]["train_acc"], color=OI[k], lw=1.0)
            ax[r, 1].plot(ep, f["hist"]["test_acc"], color=OI[k], lw=1.0, ls="--")
        ax[r, 1].axhline(0.5, color="0.4", ls=":", lw=0.8)
        ax[r, 1].set_title(f"{tag}: accuracy (—train, --test)")
        ax[r, 1].set_xlabel("epoch"); ax[r, 1].set_ylabel("acc"); ax[r, 1].set_ylim(0.4, 1.0)
        # (col2) g distribution (seed 2024)
        f0 = seedfits[0]
        ax[r, 2].hist(f0["g_full"], bins=40, color=OI[0], alpha=0.8)
        ax[r, 2].set_title(f"{tag}: residual g=t-p0 (seed{f0['seed']})")
        ax[r, 2].set_xlabel("g_n"); ax[r, 2].set_ylabel("count")
        # (col3) h distribution (seed 2024)
        ax[r, 3].hist(f0["h_full"], bins=40, color=OI[1], alpha=0.8)
        ax[r, 3].axvline(0.25, color="0.4", ls=":", lw=0.8)
        ax[r, 3].set_title(f"{tag}: curvature h=p0(1-p0)")
        ax[r, 3].set_xlabel("h_n"); ax[r, 3].set_ylabel("count")
    fig.suptitle("STEP 1: Bell-0 BCE training + loss-consistent residual (g, h)", y=1.01)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "step1_bell0_training.png")
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70); print("STEP 1: BCE Bell-0 + residual + Metric B probe check"); print("=" * 70)
    t_start = time.time()
    fits = []
    summary = {"datasets": {}, "probe": {}, "config": {
        "N_TRAIN": N_TRAIN, "N_TEST": N_TEST, "EPOCHS": EPOCHS, "LR": LR, "SEEDS": SEEDS}}

    for tag in DATASETS:
        ds = load_synth(tag)
        summary["datasets"][tag] = {"S_true": ds["S_true"], "seeds": {}}
        for seed in SEEDS:
            t0 = time.time()
            fit = train_bell0_bce(ds, seed)
            dt = time.time() - t0
            fits.append({**fit, "tag": tag})
            np.savez(os.path.join(RESULTS_DATA, f"step1_bell0_{tag}_seed{seed}.npz"),
                     **{k: v for k, v in fit.items() if k not in ("hist", "p_final")},
                     **{f"hist_{k}": v for k, v in fit["hist"].items()},
                     **{f"p_{k}": v for k, v in fit["p_final"].items()})
            summary["datasets"][tag]["seeds"][str(seed)] = {
                "test_acc": fit["test_acc"], "train_acc": fit["train_acc"],
                "grad_first": fit["grad_first"], "grad_last": fit["grad_last"],
                "W_delta_norm": fit["W_delta_norm"],
                "W_before": fit["W_before"].tolist(), "W_after": fit["W_after"].tolist(),
                "final_train_bce": float(fit["hist"]["train_bce"][-1]),
                "final_test_bce": float(fit["hist"]["test_bce"][-1]),
                "g_mean": float(fit["g_full"].mean()), "g_std": float(fit["g_full"].std()),
                "h_mean": float(fit["h_full"].mean()), "h_max": float(fit["h_full"].max()),
                "train_time_sec": dt,
            }
            print(f"[{tag} seed{seed}] test_acc={fit['test_acc']:.3f} "
                  f"train_acc={fit['train_acc']:.3f}  BCE tr={fit['hist']['train_bce'][-1]:.3f}/"
                  f"te={fit['hist']['test_bce'][-1]:.3f}  |ΔW|={fit['W_delta_norm']:.3f} "
                  f"grad {fit['grad_first']:.2e}→{fit['grad_last']:.2e}  ({dt:.0f}s)")

    # ── 1-3 probe 미분 가능성 (seed 2024 학습 모델, 16 edge) ──
    print("\n[1-3] Metric B probe differentiability (trained Bell-0, seed 2024)")
    ALL_CROSS = [(i, j) for i in PARTY_A for j in PARTY_B]
    for tag in DATASETS:
        ds = load_synth(tag)
        fit2024 = next(f for f in fits if f["tag"] == tag and f["seed"] == 2024)
        pr = probe_check(ds, fit2024, ALL_CROSS, n_probe=32)
        summary["probe"][tag] = pr
        ok = all(v["finite"] and v["nonzero"] == v["n"] for v in pr.values())
        per_edge_time = np.mean([v["time_sec"] for v in pr.values()])
        fd_ok = all(v["fd_match"] for v in pr.values())
        print(f"  {tag}: all finite&nonzero={ok}  FD-match={fd_ok}  "
              f"mean jac time/edge(B=32)={per_edge_time:.2f}s")
        # 회로-인지 라우팅: diag edge 별 |u|mean
        for (i, j) in DIAG_CROSS:
            v = pr[f"{i}-{j}"]
            print(f"     edge{(i, j)}: |u|mean={v['u_abs_mean']:.3e} max={v['u_abs_max']:.3e} "
                  f"nonzero={v['nonzero']}/{v['n']}")

    # 비용 환산 (B=32 기준 → STEP 3 는 full N. 선형 환산)
    t_edge32 = np.mean([v["time_sec"] for tag in DATASETS for v in summary["probe"][tag].values()])
    cost_full = t_edge32 * (N_TRAIN / 32.0)        # full-train batch 환산(대략)
    est = cost_full * 16 * 6 * 2
    summary["probe_cost"] = {
        "time_per_edge_B32_sec": float(t_edge32),
        "est_time_per_edge_fulltrain_sec": float(cost_full),
        "est_total_16edge_6seed_2ds_sec": float(est),
        "est_total_min": float(est / 60.0),
    }
    print(f"\n[비용 환산] edge당 jac B=32 ≈{t_edge32:.2f}s → full-train(B={N_TRAIN}) ≈{cost_full:.1f}s/edge")
    print(f"   16 edge × 6 seed × 2 ds ≈ {est/60:.1f} min (선형 환산, 상한 추정)")

    fig_path = make_figure(fits)
    print(f"\n[그림] {fig_path}")
    json.dump(summary, open(os.path.join(RESULTS_DATA, "step1_summary.json"), "w"), indent=2)
    print(f"[요약] {os.path.join(RESULTS_DATA, 'step1_summary.json')}")
    print(f"[총 시간] {(time.time() - t_start) / 60:.1f} min")


if __name__ == "__main__":
    main()
