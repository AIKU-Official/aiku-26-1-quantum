# CPFP-LOCC ablation: 조건 구성 (라우팅 + Bell pair 처방)
#   진단(diag cache)으로부터 조건을 만든다. 라우팅은 고정(처방 상위쌍이 생존 큐빗이
#   되도록)하고 조건별로 Bell pair 위치/개수만 바꾼다. **처방의 기본은 gap-K***
#   (energy-K* 는 readout 흡수로 약한 (0,4) 모드를 놓쳐 과소처방 → 폐기. energy-K* 는
#   diagnostics.prescribe 가 진단용으로 여전히 반환하지만 ablation 처방엔 안 씀).
#
#   - None       : Bell pair 없음 (얽힘 없는 baseline).
#   - Wrong      : Prescribed 와 같은 개수·세기지만 B-와이어 cyclic-shift → off-diagonal.
#   - Prescribed : 진단 gap-K* 대각쌍 전부에 Bell pair(세기 c·√σ_r). [정식 처방]
#   - Multi_extra: Prescribed 위에 비대각 Bell pair 추가 = "필요 이상" (CQ4 최적성 검정).
#   - Discarded  : 올바른 대각쌍을 조기폐기 큐빗에 배치(깊이 대조; 실험 ②에서만 사용).

import numpy as np

CONDITIONS = ["None", "Discarded", "Wrong", "Under_prescribed",
              "Prescribed", "Multi_extra"]
DEFAULT_ROUTING = {"A": [0, 1, 2, 3], "B": [4, 5, 6, 7]}
# default routing 의 조기폐기(early) feature = order[1],order[3] = {1,3} (양 파티 공통)
EARLY_FEATS = [1, 3]
# Multi_extra 가 대각 4쌍 위에 더 거는 비대각 feature 쌍 (물리큐빗 (0,5),(1,4))
EXTRA_OFFDIAG = [(0, 1), (1, 0)]


def build_routing(pres_feat):
    """prescribed feature 쌍에서 라우팅(생존 깊이) 구성.

    상위 1·2번째 쌍의 A/B feature 가 각각 final/mid 생존 큐빗이 되도록 order 구성.
    order 포맷: [keep(final), discard_early, mid_keep(mid), mid_discard]
    반환: {"A":[wires 0-3 perm], "B":[wires 4-7 perm]}
    """
    Af, Bf = [], []
    for (i, j) in pres_feat:
        if i not in Af:
            Af.append(i)
        if j not in Bf:
            Bf.append(j)
    Af += [a for a in range(4) if a not in Af]
    Bf += [b for b in range(4) if b not in Bf]
    # final=Af[0], mid=Af[1], early=Af[2],Af[3]
    a_order = [Af[0], Af[2], Af[1], Af[3]]
    b_order = [4 + Bf[0], 4 + Bf[2], 4 + Bf[1], 4 + Bf[3]]
    return {"A": a_order, "B": b_order}


def build_conditions(diag, c_scale=3.0, multi_k=4):
    """diag(npz/dict)에서 조건별 (routing, bell_pairs) 를 만든다.

    반환: conditions[name] = {"routing": {...}, "bell": [(wi,wj,theta),...]}
      theta = clip(c_scale·√σ_r, 0, π/2),  Bell concurrence = sin(theta)

    조건(검증 위계  None ≈ Discarded < Wrong < Prescribed ≈ Multi):
      None       : 얽힘 없음.
      Discarded  : '올바른' 대각 쌍이지만 조기폐기 큐빗에 배치(default routing).
                   → readout 까지 못 가 washed-out (깊이 축 대조).
      Wrong      : 생존 큐빗·같은 세기지만 off-diagonal 재배선(특징쌍 축 대조).
      Prescribed : 상위 K* 대각 쌍, 상위 2쌍이 deep 생존이 되도록 라우팅.
      Multi      : 상위 multi_k 대각 쌍.
    """
    ranked = [tuple(map(int, p)) for p in np.asarray(diag["ranked_pairs"])]
    sing = np.asarray(diag["sing"], dtype=float)
    Kstar = int(diag["Kstar"])          # energy-K* (약한 (0,4) 놓침 → under-prescribed)
    Kgap = int(diag["Kstar_gap"])       # gap-K*    (대각 4쌍 전부 → 정식 처방)

    def thetas_for(k):
        return np.clip(c_scale * np.sqrt(sing[:k]), 0.0, np.pi / 2)

    def pack(phys, th):
        return [(int(a), int(b), float(t)) for (a, b), t in zip(phys, th)]

    # ── 처방 준수 스펙트럼 (None < Wrong < Under < Prescribed > Over) ──
    #   라우팅은 gap 처방 쌍이 생존하도록 고정(모든 조건 동일 아키텍처), Bell 만 바꾼다.
    K = Kgap if Kgap > 0 else max(Kstar, 1)
    routing = build_routing(ranked[:K]) if K > 0 else DEFAULT_ROUTING

    # Under_prescribed: energy-K* 대각쌍 (match2: 3쌍, (0,4) 누락 = 하다 만 처방)
    under_feat = ranked[:Kstar] if Kstar > 0 else []
    under_phys = [(i, 4 + j) for (i, j) in under_feat]
    under_th = thetas_for(Kstar) if Kstar > 0 else np.zeros(0)

    # Prescribed: gap-K* 대각쌍 (match2: 4쌍 = (0,4) 포함, 정식 처방)
    pres_feat = ranked[:Kgap] if Kgap > 0 else []
    pres_phys = [(i, 4 + j) for (i, j) in pres_feat]
    pres_th = thetas_for(Kgap) if Kgap > 0 else np.zeros(0)

    # Wrong: Under(틀린 위치 대조)와 같은 개수·세기지만 B cyclic-shift → off-diagonal.
    Aw = [i for (i, j) in under_feat]
    Bw = [4 + j for (i, j) in under_feat]
    if len(under_feat) >= 2:
        wrong_phys = list(zip(Aw, Bw[1:] + Bw[:1]))
    elif len(under_feat) == 1:
        wrong_phys = [(Aw[0], 4 + ranked[-1][1])]
    else:
        wrong_phys = []

    # Multi_extra (Over-prescribed): gap 대각쌍 + 비대각 추가 = 처방 초과.
    me_phys = pres_phys + [(i, 4 + j) for (i, j) in EXTRA_OFFDIAG]
    me_th = list(pres_th) + [float(thetas_for(1)[0])] * len(EXTRA_OFFDIAG)

    # Discarded: 올바른 대각쌍이지만 조기폐기 큐빗에 배치(깊이 대조용; 실험 ②에서 사용).
    disc = [(i, j) for (i, j) in ranked if i == j and i in EARLY_FEATS]
    if not disc:
        disc = [(EARLY_FEATS[0], EARLY_FEATS[0]), (EARLY_FEATS[1], EARLY_FEATS[1])]
    disc = disc[:2]
    disc_th = thetas_for(len(disc))
    disc_phys = [(i, 4 + j) for (i, j) in disc]

    return {
        "None": {"routing": routing, "bell": []},
        "Discarded": {"routing": DEFAULT_ROUTING, "bell": pack(disc_phys, disc_th)},
        "Wrong": {"routing": routing, "bell": pack(wrong_phys, under_th)},
        "Under_prescribed": {"routing": routing, "bell": pack(under_phys, under_th)},
        "Prescribed": {"routing": routing, "bell": pack(pres_phys, pres_th)},
        "Multi_extra": {"routing": routing, "bell": pack(me_phys, me_th)},
    }
