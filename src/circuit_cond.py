# CPFP-LOCC ablation — STEP A 실험용 복사본 (정본 Cong et al. QCNN pooling).
#
#   ⚠️ 이 파일은 src/circuit.py 의 **실험용 복사본**이다. 기존 circuit.py(CRY
#      coherent pooling)는 그대로 보존하고, 여기서는 **pooling만** 정본 방식인
#      mid-circuit measurement + conditional rotation 으로 교체한다.
#
#         m = qml.measure(discard_wire)
#         qml.cond(m == 0, qml.RX)(params[i],   wires=keep_wire)
#         qml.cond(m == 1, qml.RX)(params[i+1], wires=keep_wire)
#
#   conv/embedding/Bell/readout/파티분할은 circuit.py 와 동일. 4+4 유지.
#     파티 A: 0~3 --pool1--> 0,2 --pool2--> 0
#     파티 B: 4~7 --pool1--> 4,6 --pool2--> 4
#
#   ── 파라미터 회계 차이 (CRY → cond) ──────────────────────────────
#     CRY pool : pool 1개당 RY param 1개.
#     cond pool: pool 1개당 RX param 2개 (m==0 분기, m==1 분기). → 파티당 pool
#                param 이 (CRY: 3개) → (cond: 6개) 로 늘어난다. 그래서 이 파일의
#                n_party_params 는 circuit.py 와 다르며, 두 방식의 파라미터 수는
#                직접 비교하지 말 것(표현력 비교는 별도 분석에서).
#
#   ── backprop/호환성 주의 ───────────────────────────────────────
#     diff_method="backprop"(analytic) 에서 mid-circuit measurement 는
#     deferred-measurement 원리로 전개되어 **MCM 1개당 보조 wire 1개**가 붙는다.
#     8큐빗 + 6 MCM = 유효 14 wire(2^14 상태벡터). 파라미터 broadcasting(배치)과
#     autograd grad 모두 호환됨은 확인했으나(STEP A smoke test), 속도/메모리는
#     CRY 버전보다 무겁다. 벤치마크 결과는 STEP A 보고 참조.

import numpy as np
import pennylane as qml

# ────────────────────────────────────────
# 0. 전역 상수 (circuit.py 와 동일)
# ────────────────────────────────────────
N_QUBITS = 8
PARTY_A_WIRES = [0, 1, 2, 3]
PARTY_B_WIRES = [4, 5, 6, 7]
N_COMBINER = 5

dev = qml.device("default.qubit", wires=N_QUBITS)


# ── 파라미터 수 ────────────────────────────
def n_party_params(n_blocks, pooling):
    """파티당 양자 파라미터 수 (cond pooling 회계).
       4q conv layer = 8, 2q conv layer = 4.
       cond pool = RX 2개(m==0, m==1 분기)."""
    if pooling:
        # 4q stage(n_blocks×8) + pool1(2 pool × 2 = 4) + 2q stage(n_blocks×4)
        #  + pool2(1 pool × 2 = 2)
        return 8 * n_blocks + 4 + 4 * n_blocks + 2
    else:
        # pooling 없는 분기는 circuit.py 와 동일(측정 없음)
        return 16 * n_blocks


def n_quantum_params(pooling, n_blocks):
    return 2 * n_party_params(n_blocks, pooling)


# ────────────────────────────────────────
# 1. 빌딩 블록 (circuit.py 와 동일 — conv 는 coherent CRY 유지)
# ────────────────────────────────────────
def _conv4(p, q):
    """4큐빗 conv layer: trainable RY 4개 + ring CRY 4개 (파라미터 8개)."""
    for i, w in enumerate(q):
        qml.RY(p[i], wires=w)
    ring = [(q[0], q[1]), (q[2], q[3]), (q[1], q[2]), (q[3], q[0])]
    for i, (a, b) in enumerate(ring):
        qml.CRY(p[4 + i], wires=[a, b])


def _conv2(p, a, b):
    """2큐빗 conv layer: RY 2개 + 양방향 CRY 2개 (파라미터 4개)."""
    qml.RY(p[0], wires=a)
    qml.RY(p[1], wires=b)
    qml.CRY(p[2], wires=[a, b])
    qml.CRY(p[3], wires=[b, a])


def _cond_pool(p2, discard, keep):
    """정본 pooling 1회: discard 큐빗을 mid-circuit 측정 후, 측정 결과에 따라
       keep 큐빗에 조건부 RX. p2 = (param_if_0, param_if_1)."""
    m = qml.measure(discard)
    qml.cond(m == 0, qml.RX)(p2[0], wires=keep)
    qml.cond(m == 1, qml.RX)(p2[1], wires=keep)


# ────────────────────────────────────────
# 2. 로컬 QCNN (cond pooling)
# ────────────────────────────────────────
def local_qcnn(qp, X, order, n_blocks, pooling, reupload):
    """한 파티의 로컬 QCNN. 파티 간 게이트 없음. pooling=정본 cond 방식.

    order : [keep, discard_early, mid_keep, mid_discard]
    반환  : 생존(readout) 큐빗 = order[0]
    """
    q = list(order)
    idx = 0

    def take(n):
        nonlocal idx
        s = qp[idx:idx + n]; idx += n
        return s

    # ── 4q feature stage ──
    for _ in range(n_blocks):
        if reupload:
            for w in q:
                qml.RY(X[:, w], wires=w)
        _conv4(take(8), q)

    if pooling:
        # pool1 (4→2): keep q[0]<-discard q[1] ; keep q[2]<-discard q[3]
        _cond_pool(take(2), discard=q[1], keep=q[0])
        _cond_pool(take(2), discard=q[3], keep=q[2])
        # 2q stage on survivors (q[0], q[2])
        for _ in range(n_blocks):
            if reupload:
                qml.RY(X[:, q[0]], wires=q[0])
                qml.RY(X[:, q[2]], wires=q[2])
            _conv2(take(4), q[0], q[2])
        # pool2 (2→1): keep q[0]<-discard q[2]
        _cond_pool(take(2), discard=q[2], keep=q[0])
        return q[0]
    else:
        for _ in range(n_blocks):
            if reupload:
                for w in q:
                    qml.RY(X[:, w], wires=w)
            _conv4(take(8), q)
        return q[0]


# ────────────────────────────────────────
# 3. 회로 빌더 (circuit.py 와 동일 인터페이스)
# ────────────────────────────────────────
def default_routing():
    """기본 pooling 라우팅 → 생존 큐빗 (0, 4)."""
    return {"A": [0, 1, 2, 3], "B": [4, 5, 6, 7]}


def n_readout(readout):
    return 2 ** N_QUBITS if readout == "all" else 4


def make_qnode(pooling=True, n_blocks=3, reupload=True,
               bell_pairs=None, routing=None, readout="pair"):
    """4+4 회로 qnode 생성 (cond pooling). 인터페이스는 circuit.py 와 동일.

    backprop(analytic)에서 MCM 은 deferred-measurement 로 전개됨.
    """
    if routing is None:
        routing = default_routing()
    per = n_party_params(n_blocks, pooling)

    @qml.qnode(dev, interface="autograd", diff_method="backprop")
    def circuit(X, qp):
        # 1. 사전공유 Bell pair (LOCC: 데이터 임베딩 '전'에만)
        if bell_pairs:
            for (i, j, theta) in bell_pairs:
                qml.RY(theta, wires=i)
                qml.CNOT(wires=[i, j])
        # 2. 로컬 각도 임베딩
        for k in range(N_QUBITS):
            qml.RY(X[:, k], wires=k)
        # 3. 로컬 QCNN (파티 간 게이트 없음)
        ra = local_qcnn(qp[:per], X, routing["A"], n_blocks, pooling, reupload)
        rb = local_qcnn(qp[per:2 * per], X, routing["B"], n_blocks, pooling, reupload)
        # 4. readout
        if readout == "all":
            return qml.probs(wires=list(range(N_QUBITS)))
        return qml.probs(wires=[ra, rb])

    return circuit
