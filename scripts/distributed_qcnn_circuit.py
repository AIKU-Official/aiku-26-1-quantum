"""
분산형 QCNN 회로 재현 (image 8 형식과 정확히 일치)
- 8 큐빗 (4+4 두 파티), Cong QCNN cond pooling
- conv: 이웃쌍 CNOT + RY  /  pool: 측정 후 조건부 RX 2개
- 생존경로 A: {0,1,2,3}->{1,3}->{3},  B: {4,5,6,7}->{5,7}->{7}
"""
import numpy as np
import pennylane as qml
import matplotlib.pyplot as plt

N = 8
dev = qml.device("default.qubit", wires=N)


class PsiPrep(qml.operation.Operation):
    """|Ψ⟩ 상태 준비 박스 (사전공유 Bell pair + 각 파티 angle encoding)"""
    grad_method = None

    def label(self, decimals=None, base_label=None, cache=None):
        return r"$|\Psi\rangle$"

    @staticmethod
    def compute_decomposition(wires):
        return []


def conv(wires, p, idx):
    """Cong QCNN convolution: 이웃쌍 CNOT + 각 큐빗 RY"""
    for i in range(0, len(wires) - 1, 2):
        qml.CNOT(wires=[wires[i], wires[i + 1]])
    for w in wires:
        qml.RY(p[idx[0]], wires=w); idx[0] += 1


def pool(wires, p, idx):
    """Cong QCNN pooling (cond): 짝수 측정 후 홀수에 조건부 RX 2개"""
    survivors = []
    for i in range(0, len(wires) - 1, 2):
        m = qml.measure(wires[i])
        qml.cond(m == 0, qml.RX)(p[idx[0]], wires=wires[i + 1]); idx[0] += 1
        qml.cond(m == 1, qml.RX)(p[idx[0]], wires=wires[i + 1]); idx[0] += 1
        survivors.append(wires[i + 1])
    return survivors


@qml.qnode(dev)
def distributed_qcnn(p):
    idx = [0]
    PsiPrep(wires=range(N))                       # |Ψ⟩

    for w in range(N):                            # 초기 RY
        qml.RY(p[idx[0]], wires=w); idx[0] += 1

    conv(list(range(N)), p, idx)                  # Block1 conv (8->)
    s1 = pool(list(range(N)), p, idx)             # Block1 pool -> [1,3,5,7]
    qml.Barrier(wires=range(N), only_visual=True)

    conv(s1, p, idx)                              # Block2 conv
    s2 = pool(s1, p, idx)                         # Block2 pool -> [3,7]
    qml.Barrier(wires=range(N), only_visual=True)

    for w in s2:                                  # 최종 RY (readout A=3, B=7)
        qml.RY(p[idx[0]], wires=w); idx[0] += 1

    return [qml.expval(qml.PauliZ(w)) for w in s2]


p = np.zeros(200)
fig, ax = qml.draw_mpl(distributed_qcnn, decimals=None, style="black_white")(p)
fig.savefig("/home/claude/circuit_repro.png", dpi=150, bbox_inches="tight")
print("saved")
