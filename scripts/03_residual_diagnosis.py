# STEP 3. Residual 진단 T_ij
#   - reference baseline f0(cache/baseline_{ds}_pool.npz) 로 residual r=y-f0 계산
#   - 쌍별 cross off-rank-1 에너지 T_ij(4×4) 계산 (i∈A feature, j∈B feature)
#   - 두 데이터셋 히트맵 나란히 (공유 스케일) → 대비
#   - 각각 SVD → K*(누적에너지 0.9) 와 세기 √σ_r 출력/저장
#
#   검증:
#     match2: 대각 쌍 (0,0),(1,1),(2,2),(3,3) [= 물리큐빗 (0,4),(1,5),(2,6),(3,7)] 지목?
#     pair3 : 전체적으로 낮은가?

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config, data, circuits, model as M, diagnostics as D

config.apply_style()
np.random.seed(config.SEED)

ENERGY_THRESH = 0.9   # K* 누적 singular energy 임계
REL = 0.25            # gap-K* 상대 임계 (σ_k ≥ REL·σ_1)
C_SCALE = 1.0         # 세기 λ_r = C_SCALE · √σ_r
# match2 ground truth: 대각 cross 쌍 (A feature i, B feature i)
GT_PAIRS_MATCH2 = [(i, i) for i in range(4)]


def _baseline_path(name, method):
    base = config.CACHE_DIR if method == "cry" else os.path.join(config.CACHE_DIR, "cond")
    return os.path.join(base, f"baseline_{name}_pool.npz")


def _diag_path(name, method):
    base = config.CACHE_DIR if method == "cry" else os.path.join(config.CACHE_DIR, "cond")
    return os.path.join(base, f"diag_{name}.npz")


def load_f0(name, method):
    """reference baseline f0 (nb=4 pooling reupload) 로드 → (params, qnode, n_q, z)."""
    C = circuits.get(method)
    z = np.load(_baseline_path(name, method), allow_pickle=True)
    qnode = C.make_qnode(pooling=bool(z["pooling"]), n_blocks=int(z["n_blocks"]),
                         reupload=bool(z["reupload"]))
    return np.array(z["params"]), qnode, int(z["n_q"]), z


def run_method(method):
    """한 pooling method 의 residual 진단 전체. 반환: results dict (비교용)."""
    pool_name = {"cry": "CRY", "cond": "cond (Cong)"}[method]
    print("=" * 70)
    print(f"STEP 3 [{pool_name}]. Residual 진단 T_ij  (off-rank-1 cross 에너지)")
    print("=" * 70)

    results = {}
    for name in config.DATASETS:
        ds = data.load_dataset(name)
        params, qnode, n_q, z = load_f0(name, method)
        # residual 은 전체 데이터에서 (Fourier 추정 표본 최대화)
        X, y = ds["X"], ds["y"]
        r = D.residual(params, X, y, qnode, n_q)
        T_off, T_tot = D.demand_matrix(r, X, return_total=True)
        # 노이즈 바닥: residual 셔플 null σ_1 (95%) → cross 없으면 gap-K* 0 으로 게이트
        floor = D.null_sigma1(r, X, n_shuffle=40, seed=config.SEED)
        pres = D.prescribe(T_off, ENERGY_THRESH, C_SCALE, rel=REL, floor=floor)
        gap_rel = D.rank_gap(pres["sing"], REL, 0.0)          # 상대 gap (게이트 없음)
        results[name] = dict(T_off=T_off, T_tot=T_tot, pres=pres, floor=floor,
                             gap_rel=gap_rel,
                             f0_train=float(z["train_acc"]), f0_test=float(z["test_acc"]),
                             resid_var=float(np.var(r)))

        print(f"\n── {name} (f0 train={float(z['train_acc']):.3f} "
              f"test={float(z['test_acc']):.3f}, Var[r]={np.var(r):.3f}) ──")
        print("  T_ij (off-rank-1 cross energy)  행=A feature i, 열=B feature j:")
        for i in range(4):
            print("   ", "  ".join(f"{T_off[i, j]:.4f}" for j in range(4)))
        # 대각/비대각 평균 (cross 구조 진단)
        diag = np.array([T_off[i, i] for i in range(4)])
        offd = T_off[~np.eye(4, dtype=bool)]
        print(f"  대각 평균={diag.mean():.4f}  비대각 평균={offd.mean():.4f}  "
              f"비율(대각/비대각)={diag.mean()/(offd.mean()+1e-9):.2f}")
        # top 쌍
        top = pres["ranked_pairs"][:4]
        print(f"  상위 4 쌍 (i,j)[물리큐빗(i,4+j)]: "
              + ", ".join(f"({i},{j})[{i},{4+j}]={T_off[i,j]:.3f}" for i, j in top))
        # SVD 처방 (두 기준 모두 보고)
        print(f"  SVD σ = {np.round(pres['sing'], 4)}  (노이즈바닥 σ1_null={floor:.4f})")
        print(f"  누적에너지 = {np.round(pres['cum_energy'], 3)}")
        print(f"  K*(energy τ={ENERGY_THRESH}) = {pres['Kstar']}   "
              f"| gap-K*(rel={REL}) = {gap_rel}   "
              f"| gap-K*(noise-gated) = {pres['Kstar_gap']}")
        print(f"  세기 λ_r=√σ = {np.round(pres['lambdas'], 4)}")
        # match2 ground truth 일치 검사
        if name == "match2":
            topK = set(pres["ranked_pairs"][:4])
            hit = len(topK & set(GT_PAIRS_MATCH2))
            print(f"  [GT 검증] 상위4 쌍 중 대각(ground truth) 일치: {hit}/4")

        # cache 저장 (STEP 4 처방 재사용) — 두 기준 모두 저장. method별 경로 분리.
        np.savez(_diag_path(name, method),
                 T_off=T_off, T_tot=T_tot, sing=pres["sing"],
                 cum_energy=pres["cum_energy"], Kstar=pres["Kstar"],
                 Kstar_gap=pres["Kstar_gap"], gap_rel=gap_rel, null_floor=floor,
                 lambdas=pres["lambdas"],
                 ranked_pairs=np.array(pres["ranked_pairs"]))

    # ── 두 기준 비교 표 ──────────────────────────────────────
    print("\n" + "=" * 64)
    print("K* 기준 비교 (energy τ=0.9  vs  gap σ≥0.25σ₁)")
    print("=" * 64)
    print(f"{'dataset':>8} {'K*(energy)':>11} {'gap-K*(rel)':>12} "
          f"{'gap-K*(gated)':>14} {'σ1':>8} {'σ1_null':>9}")
    print("-" * 64)
    for name in config.DATASETS:
        rr = results[name]
        print(f"{name:>8} {rr['pres']['Kstar']:>11} {rr['gap_rel']:>12} "
              f"{rr['pres']['Kstar_gap']:>14} {rr['pres']['sing'][0]:>8.4f} "
              f"{rr['floor']:>9.4f}")

    # ── 그림: 히트맵(공유 스케일) + SVD 누적에너지 ──────────────
    names = list(config.DATASETS)
    vmax = max(results[n]["T_off"].max() for n in names)
    fig, axes = plt.subplots(1, 3, figsize=(config.W_DOUBLE, 56 * config.MM),
                             gridspec_kw={"width_ratios": [1, 1, 1.05], "wspace": 0.45})
    for col, name in enumerate(names):
        ax = axes[col]
        T = results[name]["T_off"]
        im = ax.imshow(T, cmap="cividis", vmin=0, vmax=vmax, aspect="equal")
        ax.set_xticks(range(4)); ax.set_xticklabels([f"b{j}" for j in range(4)])
        ax.set_yticks(range(4)); ax.set_yticklabels([f"a{i}" for i in range(4)])
        ax.set_xlabel("Party B feature"); ax.set_ylabel("Party A feature")
        ax.set_title(f"{name} ({'cross' if name=='match2' else 'marginal'})\n"
                     "residual T_ij (off-rank-1)", fontsize=6.0)
        for i in range(4):
            for j in range(4):
                ax.text(j, i, f"{T[i, j]:.2f}", ha="center", va="center",
                        fontsize=4.6, color="w" if T[i, j] < vmax * 0.55 else "k")
        # match2 대각(GT) 강조
        if name == "match2":
            for i in range(4):
                ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False,
                                           edgecolor=config.OKABE_ITO[4], lw=1.0))
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # (c) SVD 누적 singular energy
    axc = axes[2]
    for name in names:
        cum = results[name]["pres"]["cum_energy"]
        col = config.OKABE_ITO[4] if name == "match2" else config.OKABE_ITO[2]
        ke = results[name]["pres"]["Kstar"]
        kg = results[name]["pres"]["Kstar_gap"]
        axc.plot(range(1, len(cum) + 1), cum, "-o", ms=3, lw=1.0, color=col,
                 label=f"{name} (K*={ke}, gap={kg})")
    axc.axhline(ENERGY_THRESH, color="0.6", lw=0.5, ls="--")
    axc.text(1, ENERGY_THRESH + 0.01, f"{ENERGY_THRESH:g}", fontsize=5.2, color="0.5")
    axc.set_xlabel("Rank (SVD of T_ij)")
    axc.set_ylabel("Cumulative singular energy")
    axc.set_xticks(range(1, 5))
    axc.set_ylim(0, 1.05)
    axc.set_title("Prescription: K* & strengths", fontsize=6.2)
    axc.legend(loc="lower right", fontsize=5.2, handlelength=1.3)

    fig.suptitle(f"STEP 3 [{pool_name}]. Residual cross-demand: match2 picks the "
                 "diagonal, pair3 stays flat", fontsize=7.0, y=1.04)
    suffix = "" if method == "cry" else f"_{method}"
    out = os.path.join(config.OUTPUT_DIR, f"03_residual_diagnosis{suffix}.png")
    fig.savefig(out, dpi=300); fig.savefig(out.replace(".png", ".pdf"))
    plt.close(fig)
    print(f"\nsaved: {out}")
    return results


def _diag_offdiag_ratio(T):
    diag = np.array([T[i, i] for i in range(4)])
    offd = T[~np.eye(4, dtype=bool)]
    return diag.mean(), offd.mean(), diag.mean() / (offd.mean() + 1e-9)


def compare_figure(res_by_method):
    """CRY vs cond 비교: match2 T_ij 히트맵 나란히 + diag/offdiag 비율 + K* 표."""
    methods = [m for m in ("cry", "cond") if m in res_by_method]
    if len(methods) < 2:
        return
    # (a,b) match2 히트맵 CRY vs cond, (c) match2 SVD 누적에너지 overlay
    Tmats = {m: res_by_method[m]["match2"]["T_off"] for m in methods}
    vmax = max(T.max() for T in Tmats.values())
    fig, axes = plt.subplots(1, 3, figsize=(config.W_DOUBLE, 56 * config.MM),
                             gridspec_kw={"width_ratios": [1, 1, 1.05], "wspace": 0.45})
    for col, m in enumerate(methods):
        ax = axes[col]; T = Tmats[m]
        im = ax.imshow(T, cmap="cividis", vmin=0, vmax=vmax, aspect="equal")
        ax.set_xticks(range(4)); ax.set_xticklabels([f"b{j}" for j in range(4)])
        ax.set_yticks(range(4)); ax.set_yticklabels([f"a{i}" for i in range(4)])
        ax.set_xlabel("Party B feature"); ax.set_ylabel("Party A feature")
        dm, om, ratio = _diag_offdiag_ratio(T)
        pname = {"cry": "CRY", "cond": "cond"}[m]
        ax.set_title(f"match2 T_ij — {pname}\ndiag/off = {ratio:.0f}×", fontsize=6.0)
        for i in range(4):
            for j in range(4):
                ax.text(j, i, f"{T[i, j]:.2f}", ha="center", va="center",
                        fontsize=4.6, color="w" if T[i, j] < vmax * 0.55 else "k")
        for i in range(4):
            ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False,
                                       edgecolor=config.OKABE_ITO[4], lw=1.0))
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    axc = axes[2]
    sty = {"cry": dict(ls="--", marker="s"), "cond": dict(ls="-", marker="o")}
    for m in methods:
        for name in config.DATASETS:
            cum = res_by_method[m][name]["pres"]["cum_energy"]
            col = config.OKABE_ITO[4] if name == "match2" else config.OKABE_ITO[2]
            axc.plot(range(1, len(cum) + 1), cum, ms=3, lw=1.0, color=col,
                     **sty[m], label=f"{name}-{m}")
    axc.axhline(ENERGY_THRESH, color="0.6", lw=0.5, ls=":")
    axc.set_xlabel("Rank (SVD of T_ij)"); axc.set_ylabel("Cumulative singular energy")
    axc.set_xticks(range(1, 5)); axc.set_ylim(0, 1.05)
    axc.legend(loc="lower right", fontsize=4.8, handlelength=1.5, ncol=2)
    axc.set_title("K* — CRY vs cond", fontsize=6.2)
    fig.suptitle("STEP 3. CRY vs cond: diagonal-cross structure reproduced",
                 fontsize=7.0, y=1.04)
    out = os.path.join(config.OUTPUT_DIR, "03_residual_diagnosis_compare.png")
    fig.savefig(out, dpi=300); fig.savefig(out.replace(".png", ".pdf"))
    plt.close(fig)
    print(f"\nsaved comparison: {out}")

    # 콘솔 비교 표
    print("\n" + "=" * 70)
    print("CRY vs cond 비교 (match2 diag/off 비율, GT hit, K*)")
    print("=" * 70)
    for name in config.DATASETS:
        print(f"\n[{name}]")
        for m in methods:
            rr = res_by_method[m][name]
            dm, om, ratio = _diag_offdiag_ratio(rr["T_off"])
            print(f"  {m:>4}: diag={dm:.4f} off={om:.4f} ratio={ratio:6.1f}×  "
                  f"K*(e)={rr['pres']['Kstar']} gap(rel)={rr['gap_rel']} "
                  f"gap(gated)={rr['pres']['Kstar_gap']}  σ1={rr['pres']['sing'][0]:.4f} "
                  f"σ1_null={rr['floor']:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="cry",
                    choices=list(circuits.METHODS) + ["both"])
    args = ap.parse_args()
    methods = list(circuits.METHODS) if args.method == "both" else [args.method]
    res_by_method = {}
    for m in methods:
        if not os.path.exists(_baseline_path("match2", m)):
            print(f"[{m}] baseline 없음 → 스킵 ({_baseline_path('match2', m)})")
            continue
        res_by_method[m] = run_method(m)
    if len(res_by_method) == 2:
        compare_figure(res_by_method)


if __name__ == "__main__":
    main()
