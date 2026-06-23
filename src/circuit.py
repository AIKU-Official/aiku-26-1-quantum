# CPFP-LOCC ablation: 8큐빗 4+4 회로 (capacity 조절 가능한 강한 로컬 모델)
#   - 큐빗 0~3 = 파티 A, 4~7 = 파티 B. 각 파티 4 feature angle encoding.
#   - 각 파티 안에서만 로컬 QCNN(conv→pool). **파티 간 게이트 절대 없음.**
#   - 얽힘은 embedding 전 사전공유 Bell pair(LOCC)로만 주입(STEP 4).
#   - readout: 각 파티 pooling 후 살아남는 큐빗 1개씩 → qml.probs(2큐빗)=4확률.
#
#   capacity 노브:
#     n_blocks : conv 레이어 반복 수(깊이). ↑ → 표현력 ↑ (파티 간 게이트는 안 늘림)
#     reupload : True면 각 block 마다 자기 파티 데이터 RY 재업로드(degree↑, marginal
#                threshold 함수 표현에 필수적)
#
# ── pooling 생존 경로 (반드시 추적) ─────────────────────────────
#   conv=2큐빗 합성곱(trainable RY + CRY), pool=CRY(discard→keep): 정보를 keep으로
#   넘기고 discard는 더 이상 측정/사용 안 함. "일찍 버려지는 큐빗"에 얽힘을 놓으면
#   readout까지 못 살아남으므로 Bell pair 후보 큐빗은 아래 생존 큐빗과 일치해야 한다.
#
#   파티 A (order=[0,1,2,3]): {0,1,2,3} --pool1--> {0,2} --pool2--> {0}
#       pool1: CRY(1→0) keep0 ; CRY(3→2) keep2     pool2: CRY(2→0) keep0
#       ⇒ 생존(readout) = order[0]=0   (생존 깊이: 0 > 2 > {1,3})
#   파티 B (order=[4,5,6,7]): {4,5,6,7} --pool1--> {4,6} --pool2--> {4}
#       ⇒ 생존(readout) = order[0]=4   (생존 깊이: 4 > 6 > {5,7})
#   ※ routing 의 order 리스트를 바꾸면 어떤 큐빗을 생존시킬지 제어 가능(STEP 4).
#     생존 쌍 순위: (order_A[0],order_B[0]) 최강 > (order_A[2],order_B[2]) > 나머지

import numpy as np
import pennylane as qml

# ────────────────────────────────────────
# 0. 전역 상수
# ────────────────────────────────────────
N_QUBITS = 8
PARTY_A_WIRES = [0, 1, 2, 3]
PARTY_B_WIRES = [4, 5, 6, 7]
N_COMBINER = 5        # 선형결합 w(4) + b(1)

dev = qml.device("default.qubit", wires=N_QUBITS)


# ── 파라미터 수 ────────────────────────────
def n_party_params(n_blocks, pooling):
    """파티당 양자 파라미터 수.
       4q conv layer = 8 (RY4 + CRY4), 2q conv layer = 4, pool = CRY 1개."""
    if pooling:
        # 4q stage(n_blocks×8) + pool1(2) + 2q stage(n_blocks×4) + pool2(1)
        return 8 * n_blocks + 2 + 4 * n_blocks + 1
    else:
        # 4q stage 두 번(pool 대신) = 2×n_blocks×8
        return 16 * n_blocks


def n_quantum_params(pooling, n_blocks):
    return 2 * n_party_params(n_blocks, pooling)


# ────────────────────────────────────────
# 1. 빌딩 블록
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


# ────────────────────────────────────────
# 2. 로컬 QCNN (capacity 조절)
# ────────────────────────────────────────
def local_qcnn(qp, X, order, n_blocks, pooling, reupload):
    """한 파티의 로컬 QCNN. 파티 간 게이트 없음.

    qp       : 이 파티의 양자 파라미터 슬라이스
    X        : (B,8) 배치 입력 (재업로드용; 이 파티 wire의 feature만 사용)
    order    : 이 파티의 큐빗 순서 [keep, discard_early, mid_keep, mid_discard]
    반환     : 생존(readout) 큐빗 = order[0]
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
                qml.RY(X[:, w], wires=w)     # 자기 파티 데이터 재업로드
        _conv4(take(8), q)

    if pooling:
        # pool1 (4→2): keep q[0], q[2]
        qml.CRY(take(1)[0], wires=[q[1], q[0]])
        qml.CRY(take(1)[0], wires=[q[3], q[2]])
        # 2q stage on survivors (q[0], q[2])
        for _ in range(n_blocks):
            if reupload:
                qml.RY(X[:, q[0]], wires=q[0])
                qml.RY(X[:, q[2]], wires=q[2])
            _conv2(take(4), q[0], q[2])
        # pool2 (2→1): keep q[0]
        qml.CRY(take(1)[0], wires=[q[2], q[0]])
        return q[0]
    else:
        # pool 대신 4q conv stage 한 번 더 (모든 큐빗 생존, readout만 q[0])
        for _ in range(n_blocks):
            if reupload:
                for w in q:
                    qml.RY(X[:, w], wires=w)
            _conv4(take(8), q)
        return q[0]


# ────────────────────────────────────────
# 3. 회로 빌더
# ────────────────────────────────────────
def default_routing():
    """기본 pooling 라우팅 → 생존 큐빗 (0, 4)."""
    return {"A": [0, 1, 2, 3], "B": [4, 5, 6, 7]}


def n_readout(readout):
    """readout 모드별 확률 벡터 차원 (= 고전 결합기 w 길이)."""
    return 2 ** N_QUBITS if readout == "all" else 4


def make_qnode(pooling=True, n_blocks=3, reupload=True,
               bell_pairs=None, routing=None, readout="pair"):
    """4+4 회로 qnode 생성.

    pooling    : True=QCNN(conv+pool), False=순수 유니터리(pool 없음)
    n_blocks   : capacity 깊이
    reupload   : 데이터 재업로드 여부
    bell_pairs : [(i,j,theta), ...] embedding 전 사전공유 Bell pair (i∈A, j∈B).
                 RY(theta) 후 CNOT(i→j) → concurrence=sin(theta). (baseline=None)
    routing    : {"A":[...], "B":[...]} 큐빗 순서(생존 제어). None이면 default.
    readout    : "pair"=두 파티 생존 큐빗 결합확률(4); "all"=8큐빗 전체 확률(256).
                 "all" 은 pooling 없이 "대각 4쌍 모두 대칭 생존" K-sweep 용.

    반환: circuit(X, qp) → qml.probs
    """
    if routing is None:
        routing = default_routing()
    per = n_party_params(n_blocks, pooling)

    @qml.qnode(dev, interface="autograd", diff_method="backprop")
    def circuit(X, qp):
        # 1. 사전공유 Bell pair (LOCC: 데이터 임베딩 '전'에만 주입)
        if bell_pairs:
            for (i, j, theta) in bell_pairs:
                qml.RY(theta, wires=i)
                qml.CNOT(wires=[i, j])
        # 2. 로컬 각도 임베딩 (feature k → wire k, 각 파티 자기 데이터만)
        for k in range(N_QUBITS):
            qml.RY(X[:, k], wires=k)
        # 3. 로컬 QCNN (파티 간 게이트 없음)
        ra = local_qcnn(qp[:per], X, routing["A"], n_blocks, pooling, reupload)
        rb = local_qcnn(qp[per:2 * per], X, routing["B"], n_blocks, pooling, reupload)
        # 4. readout
        if readout == "all":
            return qml.probs(wires=list(range(N_QUBITS)))   # 대칭(모두 생존)
        return qml.probs(wires=[ra, rb])                    # 두 생존 큐빗 결합확률

    return circuit
