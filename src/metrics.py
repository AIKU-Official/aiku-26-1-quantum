# CPFP-v2 STEP 2: 네 metric 구현(A / A' / B / B⊥) + 단위테스트 (우열 판단은 STEP 3)
# ─────────────────────────────────────────────────────────────────────────────
# 공통 인터페이스: 입력 (공식 Bell-0 모델 p, head 거친 데이터, residual g, curvature h),
#                 출력 16 cross edge (i,j) score. seed 2024, 절대경로, BCE, [0,π] head.
#
# ★비용 주의(STEP 2 측정): tangent u_e / M_0 jacobian 은 batch 에 대해 O(B^2)
#   (broadcasted output 의 reverse-mode jacobian). 따라서 작은 chunk(=32/64)로 쪼개
#   concat 하면 O(B) 로 선형화된다. 본 모듈은 전부 chunked 로 계산한다.
#
# ★전제(STEP 1 확인): probe 의 z=W·probs 에서 W 가 균등(=0.25)이면 z 상수→tangent 0.
#   반드시 '학습된(공식) Bell-0 의 W' 사용.

import os, json, time, itertools
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import circuit as C
import train_bell0 as S1

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DATA = os.path.join(ROOT, "results", "data")
FIG_DIR = os.path.join(ROOT, "figures")
os.makedirs(RESULTS_DATA, exist_ok=True)

PARTY_A, PARTY_B = [0, 1, 2, 3], [4, 5, 6, 7]
N_PARTY = 4
ALL_CROSS = [(i, j) for i in PARTY_A for j in PARTY_B]     # 16 edge, ALL_CROSS 순서 고정
DIAG_CROSS = [(0, 4), (1, 5), (2, 6), (3, 7)]
LAMBDA = 1e-3                          # B/B⊥ 분모 정칙화 기본값
EPS_WHITEN = 1e-6                      # A' whitening G 의 εI
CHUNK_TAN, CHUNK_M0 = 32, 64           # O(B^2) 회피용 chunk


# ─────────────────────────────────────────────────────────────────────────────
# Walsh-Fourier cross-power (Metric A 의 기반). rank-1 차감 없음.
# ─────────────────────────────────────────────────────────────────────────────
def build_walsh(pm1):
    """pm1 (N,8) → 캐릭터 행렬 Chi (N,256) 와 pair별 기여 mask.

    S=bitmask 0..255, bit k = feature k 포함. χ_S(n)=∏_{k∈S} pm1[n,k].
    pair (i∈A, j∈B): bit i & bit j 가 모두 켜진 S 가 기여.
    """
    N = pm1.shape[0]
    Chi = np.ones((N, 256))
    for S in range(256):
        cols = [k for k in range(8) if (S >> k) & 1]
        if cols:
            Chi[:, S] = np.prod(pm1[:, cols], axis=1)
    masks = {}
    for (i, j) in ALL_CROSS:
        masks[(i, j)] = np.array(
            [bool((S >> i) & 1 and (S >> j) & 1) for S in range(256)])
    return Chi, masks


def cpfp_M(residual, Chi, masks):
    """residual r (N,) → cross term 계수 → M_ij=‖B_ij‖_F (4×4). rank-1 차감 없음."""
    N = residual.shape[0]
    coeffs = (Chi.T @ residual) / N                     # (256,) Walsh 계수
    M = np.zeros((4, 4))
    for a, i in enumerate(PARTY_A):
        for b, j in enumerate(PARTY_B):
            sel = masks[(i, j)]
            M[a, b] = np.sqrt(np.sum(coeffs[sel] ** 2))
    return M


# ─────────────────────────────────────────────────────────────────────────────
# 2-0. Bell-0 restart 규약 → 공식 Bell-0
# ─────────────────────────────────────────────────────────────────────────────
def train_bell0_official(ds, seed, R=3, epochs=S1.EPOCHS, lr=S1.LR, fail_acc=0.55):
    """seed 마다 R restart(init만 다름) → train BCE 최소 restart 선택.
    모든 restart 가 train_acc<fail_acc 면 optimization failure(excluded=True).
    ★선택은 train BCE 로만(test/metric 미사용)."""
    restarts = []
    for r in range(R):
        init_seed = seed * 10 + r       # init 만 다르게(재현 가능). 데이터 split 은 불변.
        fit = S1.train_bell0_bce(ds, init_seed, epochs=epochs, lr=lr)
        fit["restart_idx"] = r
        fit["init_seed"] = init_seed
        fit["final_train_bce"] = float(fit["hist"]["train_bce"][-1])
        restarts.append(fit)
    best = min(restarts, key=lambda f: f["final_train_bce"])   # train BCE 최소
    excluded = all(f["train_acc"] < fail_acc for f in restarts)
    info = {
        "seed": seed, "tag": ds["tag"], "R": R, "excluded": bool(excluded),
        "selected_restart": best["restart_idx"], "selected_init_seed": best["init_seed"],
        "restarts": [{"restart_idx": f["restart_idx"], "init_seed": f["init_seed"],
                      "train_bce": f["final_train_bce"], "train_acc": f["train_acc"],
                      "test_acc": f["test_acc"]} for f in restarts],
    }
    return best, info


# ─────────────────────────────────────────────────────────────────────────────
# Bell resource primitive (STEP 4 와 공유) + probe
# ─────────────────────────────────────────────────────────────────────────────
class BellEdge:
    """pre-shared Bell resource edge: |ψ_θ⟩=cos(θ/2)|00⟩+sin(θ/2)|11⟩ (RY(θ)_i + CNOT(i→j))."""
    def __init__(self, i, j, theta):
        self.i, self.j, self.theta = i, j, theta


def apply_bell_resource(edges):
    """edges: [BellEdge,...]. RY(θ)_i 후 CNOT(i→j). θ=0 이면 무얽힘. (i,j 절대 wire)"""
    for e in edges:
        qml.RY(e.theta, wires=e.i)
        qml.CNOT(wires=[e.i, e.j])


def make_probe_qnode():
    """probe: apply_bell_resource → local embedding → local QCNN. (STEP1 검증 회로와 동일)"""
    dev = qml.device("default.qubit", wires=C.N_QUBITS)

    @qml.qnode(dev, interface="autograd", diff_method="backprop")
    @qml.defer_measurements
    def probe(x, theta, i, j, p):
        apply_bell_resource([BellEdge(i, j, theta)])      # pre-shared Bell on edge
        for k in range(C.N_QUBITS):                        # local embedding (이후 전부 local)
            qml.RY(x[k], wires=k)
        C._party_block(p, 0, PARTY_A)
        C._party_block(p, 1, PARTY_B)
        return qml.probs(wires=C.READOUT)

    return probe


# ─────────────────────────────────────────────────────────────────────────────
# tangent u_e (chunked, O(B)) / Bell-0 param jacobian M_0 (chunked)
# ─────────────────────────────────────────────────────────────────────────────
def tangent_ue(probe, ang, p, i, j, chunk=CHUNK_TAN):
    """u_e(x_n)=∂z(x_n;θ)/∂θ|_{θ=0}, z=probs·W. chunk 로 O(B) 선형화."""
    W = pnp.array(p["W"], requires_grad=False)
    out = []
    for s in range(0, ang.shape[0], chunk):
        xT = pnp.array(ang[s:s + chunk].T, requires_grad=False)

        def zvec(theta):
            return probe(xT, theta, i, j, p) @ W
        out.append(np.asarray(qml.jacobian(zvec)(pnp.array(0.0, requires_grad=True))))
    return np.concatenate(out)


def bell0_param_jacobian(ang, p, chunk=CHUNK_M0):
    """M_0[n,k]=∂z_0(x_n;η)/∂η_k. η=flat Bell-0 파라미터(E=∅). chunk 로 선형화."""
    qn = C.make_qnode()
    flat0 = C.pack(p, np.zeros(0))

    def z0(flat, xT):
        d, th = C.unpack(flat, 0)
        return qn(xT, d, th, []) @ d["W"]
    rows = []
    for s in range(0, ang.shape[0], chunk):
        xT = pnp.array(ang[s:s + chunk].T, requires_grad=False)
        rows.append(np.asarray(qml.jacobian(lambda f: z0(f, xT))(flat0)))
    return np.concatenate(rows, axis=0)            # (N, 142)


# ─────────────────────────────────────────────────────────────────────────────
# 2-1. 네 metric
# ─────────────────────────────────────────────────────────────────────────────
def _grid_to_scores(M):
    """4×4 grid(M[a,b]) → {(i,j): score} (ALL_CROSS 순서)."""
    return {(PARTY_A[a], PARTY_B[b]): float(M[a, b]) for a in range(4) for b in range(4)}


def metric_A(g, pm1):
    """Metric A: Fourier cross-power M_ij=‖B_ij‖_F. residual g 사용, rank-1 차감 없음."""
    Chi, masks = build_walsh(pm1)
    M = cpfp_M(g, Chi, masks)                    # 4×4
    return _grid_to_scores(M), {"M_grid": M}


def _party_chars(pm1, cols):
    """nonempty subset character 행렬 (N,15) + 각 character 가 포함하는 feature 집합."""
    feats, subsets = [], []
    for rsz in range(1, len(cols) + 1):
        for comb in itertools.combinations(cols, rsz):
            feats.append(np.prod(pm1[:, list(comb)], axis=1))
            subsets.append(set(comb))
    return np.stack(feats, axis=1), subsets        # (N,15), list of sets


def _inv_sqrt(Gmat):
    w, V = np.linalg.eigh(Gmat)
    w = np.clip(w, 1e-12, None)
    return (V / np.sqrt(w)) @ V.T


def metric_Aprime(g, h, pm1, eps=EPS_WHITEN):
    """Metric A': centered+weighted-whitened Fourier cross-power."""
    N = pm1.shape[0]
    PhiA, subA = _party_chars(pm1, PARTY_A)        # (N,15)
    PhiB, subB = _party_chars(pm1, PARTY_B)
    sh = h.sum()
    PhiA = PhiA - (h @ PhiA) / sh                  # weighted mean 제거 (h)
    PhiB = PhiB - (h @ PhiB) / sh
    GA = (PhiA.T @ (h[:, None] * PhiA)) / N + eps * np.eye(PhiA.shape[1])
    GB = (PhiB.T @ (h[:, None] * PhiB)) / N + eps * np.eye(PhiB.shape[1])
    PsiA = PhiA @ _inv_sqrt(GA)
    PsiB = PhiB @ _inv_sqrt(GB)
    Cw = (PsiA.T @ (g[:, None] * PsiB)) / N        # (15,15)
    M = np.zeros((4, 4))
    for a, i in enumerate(PARTY_A):
        rowsel = [k for k, S in enumerate(subA) if i in S]
        for b, j in enumerate(PARTY_B):
            colsel = [k for k, S in enumerate(subB) if j in S]
            M[a, b] = np.linalg.norm(Cw[np.ix_(rowsel, colsel)])
    sv = np.linalg.svd(Cw, compute_uv=False)
    r_eff = float((sv ** 2).sum() ** 2 / (sv ** 4).sum())     # effective rank (진단용)
    return _grid_to_scores(M), {"C_white": Cw, "eff_rank": r_eff, "singular_values": sv}


def metric_B(g, h, U, lam=LAMBDA):
    """Metric B: Bell-tangent utility. U[(i,j)] = u_e (N,). Score=(g·u)²/(u·H·u+λ)."""
    sc = {}
    for e in ALL_CROSS:
        u = U[e]
        sc[e] = float((g @ u) ** 2 / (h @ (u ** 2) + lam))
    return sc


def metric_Bperp(g, h, U, M0, lam=LAMBDA):
    """Metric B⊥: Bell-0 projection 제거 tangent. u⊥=(I−P_0)u, P_0=M_0(M_0^T H M_0+λI)^{-1}M_0^T H."""
    A0 = (M0.T @ (h[:, None] * M0)) + lam * np.eye(M0.shape[1])      # (142,142)
    A0inv = np.linalg.inv(A0)
    sc, uperp = {}, {}
    for e in ALL_CROSS:
        u = U[e]
        coef = A0inv @ (M0.T @ (h * u))            # (142,)
        up = u - M0 @ coef                         # (I-P_0)u
        uperp[e] = up
        sc[e] = float((g @ up) ** 2 / (h @ (up ** 2) + lam))
    return sc, uperp


def compute_tangents(probe, ang, p, edges=ALL_CROSS, chunk=CHUNK_TAN):
    """16 edge u_e dict + per-edge 시간."""
    U, times = {}, {}
    for e in edges:
        t0 = time.time()
        U[e] = tangent_ue(probe, ang, p, e[0], e[1], chunk=chunk)
        times[e] = time.time() - t0
    return U, times


# ─────────────────────────────────────────────────────────────────────────────
# 단위테스트 helper
# ─────────────────────────────────────────────────────────────────────────────
def _rank_order(score_dict):
    return [e for e, _ in sorted(score_dict.items(), key=lambda kv: -kv[1])]


def set_style():
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
                         "font.size": 7, "axes.titlesize": 8, "savefig.dpi": 300})


def main():
    print("=" * 72); print("STEP 2: four metrics (A/A'/B/B⊥) implementation + unit tests"); print("=" * 72)
    summary = {"config": {"N_metric_cosine": 512, "N_metric_noncosine_sanity": 256,
                          "LAMBDA": LAMBDA, "CHUNK_TAN": CHUNK_TAN, "CHUNK_M0": CHUNK_M0,
                          "R_restart": 3}, "datasets": {}}
    probe = make_probe_qnode()

    # ── 2-0: 공식 Bell-0 (seed 2024, cosine) — restart 규약 end-to-end 검증 ──
    print("\n[2-0] official Bell-0 via R=3 restart protocol (seed 2024, cosine)")
    t0 = time.time()
    ds = S1.load_synth("cosine")
    best, info = train_bell0_official(ds, 2024, R=3)
    print(f"   restarts train_bce={[round(r['train_bce'],3) for r in info['restarts']]} "
          f"train_acc={[round(r['train_acc'],3) for r in info['restarts']]}")
    print(f"   selected restart={info['selected_restart']} (min train BCE), excluded={info['excluded']}  "
          f"({time.time()-t0:.0f}s)")
    json.dump(info, open(os.path.join(RESULTS_DATA, "step2_official_bell0_cosine_seed2024.json"), "w"), indent=2)
    p = best["p_final"]

    # metric 입력: 공식 Bell-0 의 train-set residual/curvature + train 각도
    N_m = 512
    ang = ds["Xtr_ang"][:N_m]
    # train-set 위 residual/curvature (공식 Bell-0 forward)
    z_tr = S1.logits(C.make_qnode(), ang, p, np.zeros(0), [])
    p0_tr = 1.0 / (1.0 + np.exp(-z_tr))
    g = (ds["ytr"][:N_m] + 1) / 2 - p0_tr
    h = p0_tr * (1 - p0_tr)
    head = S1  # pm1 from full transform; recompute on train slice via head used in load_synth
    # pm1 for train slice: reuse load_synth's head by re-transforming (train fit already inside)
    # load_synth 는 pm1_full(전체) 제공 → train idx 로 슬라이스
    pm1 = ds["pm1_full"][ds["idx_tr"]][:N_m]

    # ── tangents + M_0 (chunked, 시간측정) ──
    print("\n[cost] computing 16-edge tangents (chunked) + M_0 ...")
    tU = time.time(); U, etimes = compute_tangents(probe, ang, p, ALL_CROSS); tU = time.time() - tU
    tM = time.time(); M0 = bell0_param_jacobian(ang, p); tM = time.time() - tM
    print(f"   tangents 16 edge: {tU:.1f}s ({tU/16:.1f}s/edge, B={N_m} chunk={CHUNK_TAN})   M_0 {M0.shape}: {tM:.1f}s")

    # ── 네 metric 계산 ──
    scA, dA = metric_A(g, pm1)
    scAp, dAp = metric_Aprime(g, h, pm1)
    scB = metric_B(g, h, U)
    scBp, uperp = metric_Bperp(g, h, U, M0)

    # ── 단위테스트: shape ──
    print("\n[unit] shape check (each metric returns 16 edge scores)")
    for nm, sc in [("A", scA), ("A'", scAp), ("B", scB), ("B⊥", scBp)]:
        assert len(sc) == 16 and set(sc.keys()) == set(ALL_CROSS), nm
        print(f"   {nm}: 16 scores OK   top3={[f'{e}:{sc[e]:.3e}' for e in _rank_order(sc)[:3]]}")

    # ── 단위테스트: B finite/nonzero 재확인 ──
    allfin = all(np.all(np.isfinite(U[e])) for e in ALL_CROSS)
    allnz = all(np.any(np.abs(U[e]) > 1e-12) for e in ALL_CROSS)
    print(f"\n[unit] Metric B tangents finite={allfin} nonzero={allnz} (trained W)")

    # chunk vs monolithic 일치(작은 batch) — 정확성 보증
    u_mono = np.asarray(S1.tangent_ue(probe, ang[:32], p, 0, 4))
    u_chunk = tangent_ue(probe, ang[:32], p, 0, 4, chunk=16)
    print(f"   chunk-consistency edge(0,4) B=32: max|Δ|={np.max(np.abs(u_mono-u_chunk)):.2e}")

    # ── 단위테스트: λ 민감도(순위 안정성) ──
    print("\n[unit] lambda sensitivity (rank stability of B / B⊥)")
    lam_ranks = {}
    for lam in [1e-2, 1e-3, 1e-4]:
        rB = _rank_order(metric_B(g, h, U, lam=lam))
        rBp = _rank_order(metric_Bperp(g, h, U, M0, lam=lam)[0])
        lam_ranks[str(lam)] = {"B_top3": [list(e) for e in rB[:3]], "Bperp_top3": [list(e) for e in rBp[:3]]}
        print(f"   λ={lam:.0e}  B top3={rB[:3]}   B⊥ top3={rBp[:3]}")

    # ── effective rank (C_white) — SVD 강등(진단용만) ──
    print(f"\n[diag] A' C_white effective rank = {dAp['eff_rank']:.2f} (진단용; Bell pair 수 결정에 미사용)")

    # ── 결과 저장 + 그림 ──
    def grid(sc):
        return np.array([[sc[(PARTY_A[a], PARTY_B[b])] for b in range(4)] for a in range(4)])
    np.savez(os.path.join(RESULTS_DATA, "step2_metrics_cosine_seed2024.npz"),
             A=grid(scA), Aprime=grid(scAp), B=grid(scB), Bperp=grid(scBp),
             g=g, h=h, M0=M0, C_white=dAp["C_white"], eff_rank=dAp["eff_rank"],
             S_true=np.array(ds["S_true"]))

    set_style()
    fig, ax = plt.subplots(1, 4, figsize=(13, 3.2))
    S = [tuple(e) for e in ds["S_true"]]
    for k, (nm, sc) in enumerate([("A (Fourier)", scA), ("A' (whitened)", scAp),
                                  ("B (tangent)", scB), ("B-perp (proj)", scBp)]):
        Gd = grid(sc)
        im = ax[k].imshow(Gd, cmap="viridis", aspect="equal")
        ax[k].set_xticks(range(4)); ax[k].set_xticklabels([f"B{j}" for j in PARTY_B])
        ax[k].set_yticks(range(4)); ax[k].set_yticklabels([f"A{i}" for i in PARTY_A])
        ax[k].set_title(nm); fig.colorbar(im, ax=ax[k], fraction=0.046)
        for (i, j) in S:                       # ground-truth 셀 테두리
            a, b = PARTY_A.index(i), PARTY_B.index(j)
            ax[k].add_patch(plt.Rectangle((b - .5, a - .5), 1, 1, fill=False, ec="red", lw=2))
    fig.suptitle("STEP 2: four metric score grids (cosine, seed2024). Red = ground-truth S (NOT a verdict — STEP 3 judges).", y=1.03)
    fig.tight_layout()
    figp = os.path.join(FIG_DIR, "step2_metric_grids.png")
    fig.savefig(figp, bbox_inches="tight"); plt.close(fig)
    print(f"\n[fig] {figp}")

    # ── noncosine shape sanity (STEP1 모델 재사용, N=256, 재학습 없음) ──
    print("\n[sanity] noncosine shape check (reuse STEP1 seed2024 model, N=256)")
    dn = np.load(os.path.join(RESULTS_DATA, "step1_bell0_noncosine_seed2024.npz"), allow_pickle=True)
    pn = {k[2:]: dn[k] for k in dn.files if k.startswith("p_")}
    dsn = S1.load_synth("noncosine"); Nn = 256
    angn = dsn["Xtr_ang"][:Nn]
    zn = S1.logits(C.make_qnode(), angn, pn, np.zeros(0), [])
    p0n = 1/(1+np.exp(-zn)); gn = (dsn["ytr"][:Nn]+1)/2 - p0n; hn = p0n*(1-p0n)
    pm1n = dsn["pm1_full"][dsn["idx_tr"]][:Nn]
    Un, _ = compute_tangents(probe, angn, pn, ALL_CROSS)
    M0n = bell0_param_jacobian(angn, pn)
    nc_ok = (len(metric_A(gn, pm1n)[0]) == 16 and len(metric_Aprime(gn, hn, pm1n)[0]) == 16
             and len(metric_B(gn, hn, Un)) == 16 and len(metric_Bperp(gn, hn, Un, M0n)[0]) == 16)
    print(f"   noncosine all four metrics return 16 scores: {nc_ok}")

    # ── STEP 3 비용 환산 ──
    per_edge = tU / 16
    tangent_seed = per_edge * 16
    cost = {
        "per_edge_tangent_sec_B512": float(per_edge),
        "M0_jacobian_sec_B512": float(tM),
        "metricB_Bperp_per_seed_sec": float(tangent_seed + tM),
        "bell0_restart_R3_per_seed_sec": float(3 * 220),
        "prescription_8cond_per_seed_sec_est": float(8 * 220),
    }
    # STEP3 총: (R3 학습 + tangent+M0 + 8조건 처방학습) × 6 seed × 2 ds
    per_seed = cost["bell0_restart_R3_per_seed_sec"] + cost["metricB_Bperp_per_seed_sec"] + cost["prescription_8cond_per_seed_sec_est"]
    cost["est_per_seed_sec"] = float(per_seed)
    cost["est_step3_total_min"] = float(per_seed * 6 * 2 / 60)
    summary["cost"] = cost
    print("\n[STEP 3 cost estimate]")
    print(f"   per-edge tangent(B=512,chunked)={per_edge:.1f}s  M_0={tM:.0f}s  B+B⊥/seed={cost['metricB_Bperp_per_seed_sec']:.0f}s")
    print(f"   per seed ≈ {per_seed/60:.1f} min (R3 학습 11min + tangent/M0 + 8조건 처방 29min)")
    print(f"   ★ STEP 3 총 ≈ {cost['est_step3_total_min']/60:.1f} hr (6 seed × 2 ds). 크면 N_metric subsample(256) 권장 → tangent/M0 4×↓")

    summary["datasets"]["cosine"] = {
        "S_true": ds["S_true"], "excluded": info["excluded"],
        "scores": {nm: grid(sc).tolist() for nm, sc in
                   [("A", scA), ("Aprime", scAp), ("B", scB), ("Bperp", scBp)]},
        "rank_top3": {nm: [list(e) for e in _rank_order(sc)[:3]] for nm, sc in
                      [("A", scA), ("Aprime", scAp), ("B", scB), ("Bperp", scBp)]},
        "lambda_ranks": lam_ranks, "eff_rank_Cwhite": dAp["eff_rank"],
        "tangent_finite": allfin, "tangent_nonzero": allnz,
    }
    summary["datasets"]["noncosine_sanity"] = {"all_four_16": bool(nc_ok)}
    json.dump(summary, open(os.path.join(RESULTS_DATA, "step2_summary.json"), "w"), indent=2)
    print(f"\n[summary] {os.path.join(RESULTS_DATA, 'step2_summary.json')}")


if __name__ == "__main__":
    main()
