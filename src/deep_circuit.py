# 실험 ②: 깊은 cascade pooling (12큐빗, 파티당 6큐빗 = 4 feature + 2 ancilla)
#   목적: 조기폐기 큐빗을 readout 에서 멀리(여러 pool-hop) 떨어뜨려, Bell pair 가
#         완전히 washed-out 되는지 확인 → "pooling 깊이가 얽힘 효과의 게이팅 변수".
#
#   레이아웃:
#     파티 A 큐빗 0,1,2,3 = feature(X col 0..3),  ancilla 4,5
#     파티 B 큐빗 6,7,8,9 = feature(X col 4..7),  ancilla 10,11
#   cascade chain(=readout 로부터의 깊이 순서):
#     A: [3, 1, 4, 2, 0, 5]  → readout=q3.  depth: feat3=0, feat1=1, anc4=2,
#                                              feat2=3, feat0=4, anc5=5
#     B: [9, 7,10, 8, 6,11]  → readout=q9.  depth: feat3=0, feat1=1, anc10=2,
#                                              feat2=3, feat0=4, anc11=5
#   pool: chain[k] → chain[k-1] (cascade) → chain[k] 는 readout 에서 k hop.
#   readout = probs(q3, q9).
#
#   ⇒ 대각 feature 쌍의 깊이: feat3=(3,9) d0, feat1=(1,7) d1, feat2=(2,8) d3,
#     feat0=(0,6) d4.  shallow {feat3,feat1}, deep {feat2,feat0}.

import numpy as np
import pennylane as qml

N_QUBITS = 12
CHAIN_A = [3, 1, 4, 2, 0, 5]
CHAIN_B = [9, 7, 10, 8, 6, 11]
ANCILLAS = {4, 5, 10, 11}
N_COMBINER = 5

# feature 큐빗 → 깊이 (Bell pair 배치/검증용)
FEAT_DEPTH = {0: 4, 1: 1, 2: 3, 3: 0}     # A feature p (qubit p) 의 깊이
READOUT = (3, 9)

dev = qml.device("default.qubit", wires=N_QUBITS)


def _col_of(q):
    """feature 큐빗 → X 컬럼. ancilla 면 None."""
    if 0 <= q <= 3:
        return q
    if 6 <= q <= 9:
        return q - 2
    return None


def n_party_params(n_blocks):
    # block 당: 6 RY + 5 CRY = 11 ; pool cascade: 5
    return 11 * n_blocks + 5


def n_quantum_params(n_blocks):
    return 2 * n_party_params(n_blocks)


def _cascade_party(qp, chain, X, n_blocks, reupload):
    i = 0
    for _ in range(n_blocks):
        if reupload:
            for q in chain:
                c = _col_of(q)
                if c is not None:
                    qml.RY(X[:, c], wires=q)
        for q in chain:
            qml.RY(qp[i], wires=q); i += 1
        for a, b in zip(chain[:-1], chain[1:]):
            qml.CRY(qp[i], wires=[a, b]); i += 1
    # cascade pool: 깊은 쪽(chain[-1])부터 readout(chain[0])으로
    for k in range(len(chain) - 1, 0, -1):
        qml.CRY(qp[i], wires=[chain[k], chain[k - 1]]); i += 1
    return chain[0]


def make_deep_qnode(bell_pairs=None, n_blocks=4, reupload=True):
    per = n_party_params(n_blocks)

    @qml.qnode(dev, interface="autograd", diff_method="backprop")
    def circuit(X, qp):
        # 1. 사전공유 Bell pair (cross-party, 임베딩 전)
        if bell_pairs:
            for (i, j, theta) in bell_pairs:
                qml.RY(theta, wires=i)
                qml.CNOT(wires=[i, j])
        # 2. feature 임베딩 (ancilla 제외)
        for p in range(4):
            qml.RY(X[:, p], wires=p)         # A feat p
            qml.RY(X[:, 4 + p], wires=6 + p)  # B feat p
        # 3. 파티별 cascade (파티 간 게이트 없음)
        ra = _cascade_party(qp[:per], CHAIN_A, X, n_blocks, reupload)
        rb = _cascade_party(qp[per:2 * per], CHAIN_B, X, n_blocks, reupload)
        return qml.probs(wires=[ra, rb])

    return circuit


def diag_pair(p, theta):
    """대각 feature 쌍 p 의 물리 Bell pair (A qubit p, B qubit 6+p, theta)."""
    return (p, 6 + p, float(theta))


def build_specs(sing, c_scale=3.0):
    """match2 진단 σ 로부터 (A) 5조건 + (B) depth-sweep Bell pair 스펙.

    σ 순서대로 강도 θ_r = clip(c·√σ_r, 0, π/2).
    """
    th = np.clip(c_scale * np.sqrt(np.asarray(sing, float)), 0, np.pi / 2)
    # 대각 feature 를 깊이별로: shallow={3(d0),1(d1)}, deep={2(d3),0(d4)}
    specs = {
        # (A) 5조건
        "None": [],
        "Prescribed": [diag_pair(3, th[0]), diag_pair(1, th[1])],     # 대각·shallow
        "Discarded": [diag_pair(2, th[0]), diag_pair(0, th[1])],      # 대각·deep
        "Wrong": [(3, 7, float(th[0])), (1, 9, float(th[1]))],        # off-diag·shallow
        "Multi": [diag_pair(3, th[0]), diag_pair(1, th[1]),
                  diag_pair(2, th[2]), diag_pair(0, th[3])],          # 대각 4쌍
        # (B) depth-sweep: 단일 대각쌍, 동일 강도 θ[0], 깊이만 변화
        "sweep_d0": [diag_pair(3, th[0])],
        "sweep_d1": [diag_pair(1, th[0])],
        "sweep_d3": [diag_pair(2, th[0])],
        "sweep_d4": [diag_pair(0, th[0])],
    }
    return specs


CONDITIONS_A = ["None", "Discarded", "Wrong", "Prescribed", "Multi"]
SWEEP = [("sweep_d0", 0), ("sweep_d1", 1), ("sweep_d3", 3), ("sweep_d4", 4)]
