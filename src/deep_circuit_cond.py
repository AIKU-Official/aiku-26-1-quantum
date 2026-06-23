# 실험 ② cond 버전: 깊은 cascade pooling (12큐빗) — 정본 mid-circuit measurement pooling.
#   src/deep_circuit.py 의 cond 대응. cascade pool 의 CRY(chain[k]→chain[k-1]) 를
#   정본 방식(측정+조건부 RX)으로 교체. conv/임베딩/Bell/readout/chain 레이아웃 동일.
#
#   ※ cond 는 discard 큐빗을 **측정으로 완전히 제거**한다 → 깊은 곳의 조기폐기 큐빗에
#     놓인 Bell pair 가 CRY(coherent funnel)보다 **더 깨끗하게 washed-out** 될 것으로
#     예상(아키텍처 축 가설을 더 강하게 입증). 검증은 scripts/08.
#
#   파라미터 회계: block 당 6 RY + 5 CRY = 11. cascade pool 5개 → cond 는 pool당 2
#                (m==0/m==1) = 10. n_party_params = 11·n_blocks + 10 (CRY: +5).
#
#   ⚠️ backprop+MCM 비용: cascade 5 pool/party = 10 MCM. default.qubit 가 MCM 을
#     어떻게 전개하는지에 따라 속도가 크게 달라질 수 있어, 사용 전 grad 벤치마크로
#     실측 확인할 것(8큐빗 cond 는 CRY 와 동속이었음).

import numpy as np
import pennylane as qml

N_QUBITS = 12
CHAIN_A = [3, 1, 4, 2, 0, 5]
CHAIN_B = [9, 7, 10, 8, 6, 11]
ANCILLAS = {4, 5, 10, 11}
N_COMBINER = 5

FEAT_DEPTH = {0: 4, 1: 1, 2: 3, 3: 0}
READOUT = (3, 9)

dev = qml.device("default.qubit", wires=N_QUBITS)


def _col_of(q):
    if 0 <= q <= 3:
        return q
    if 6 <= q <= 9:
        return q - 2
    return None


def n_party_params(n_blocks):
    # block 당: 6 RY + 5 CRY = 11 ; cond pool cascade: 5 pool × 2 = 10
    return 11 * n_blocks + 10


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
    # cond cascade pool: 깊은 쪽(chain[-1])부터 readout(chain[0])으로
    #   chain[k](discard) 측정 → chain[k-1](keep) 에 조건부 RX
    for k in range(len(chain) - 1, 0, -1):
        m = qml.measure(chain[k])
        qml.cond(m == 0, qml.RX)(qp[i], wires=chain[k - 1]);     i += 1
        qml.cond(m == 1, qml.RX)(qp[i], wires=chain[k - 1]);     i += 1
    return chain[0]


def make_deep_qnode(bell_pairs=None, n_blocks=4, reupload=True):
    per = n_party_params(n_blocks)

    @qml.qnode(dev, interface="autograd", diff_method="backprop")
    def circuit(X, qp):
        if bell_pairs:
            for (i, j, theta) in bell_pairs:
                qml.RY(theta, wires=i)
                qml.CNOT(wires=[i, j])
        for p in range(4):
            qml.RY(X[:, p], wires=p)
            qml.RY(X[:, 4 + p], wires=6 + p)
        ra = _cascade_party(qp[:per], CHAIN_A, X, n_blocks, reupload)
        rb = _cascade_party(qp[per:2 * per], CHAIN_B, X, n_blocks, reupload)
        return qml.probs(wires=[ra, rb])

    return circuit


def diag_pair(p, theta):
    return (p, 6 + p, float(theta))


def build_specs(sing, c_scale=3.0):
    """match2 진단 σ → (A) 5조건 + (B) depth-sweep. deep_circuit.build_specs 와 동일."""
    th = np.clip(c_scale * np.sqrt(np.asarray(sing, float)), 0, np.pi / 2)
    specs = {
        "None": [],
        "Prescribed": [diag_pair(3, th[0]), diag_pair(1, th[1])],
        "Discarded": [diag_pair(2, th[0]), diag_pair(0, th[1])],
        "Wrong": [(3, 7, float(th[0])), (1, 9, float(th[1]))],
        "Multi": [diag_pair(3, th[0]), diag_pair(1, th[1]),
                  diag_pair(2, th[2]), diag_pair(0, th[3])],
        "sweep_d0": [diag_pair(3, th[0])],
        "sweep_d1": [diag_pair(1, th[0])],
        "sweep_d3": [diag_pair(2, th[0])],
        "sweep_d4": [diag_pair(0, th[0])],
    }
    return specs


CONDITIONS_A = ["None", "Discarded", "Wrong", "Prescribed", "Multi"]
SWEEP = [("sweep_d0", 0), ("sweep_d1", 1), ("sweep_d3", 3), ("sweep_d4", 4)]
