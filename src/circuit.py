# CPFP final: 분산 QCNN(brickwall) 회로 구현
# ─────────────────────────────────────────────────────────────────────────────
# 참고 이미지: DQCNN_ansatz_brickwall.png (배선은 이미지를 우선)
#
# 구조 요약
#   - 8 qubit.  Party A = [0,1,2,3],  Party B = [4,5,6,7]
#   - 인코딩      : qubit 마다 Ry(x_k) 하나씩 (8 feature)
#   - 얽힘 자리   : 인코딩 직후, cross pair 집합 E = {(i,j) | i∈A, j∈B} 에만
#                   trainable 2-qubit entangler 1종(IsingZZ(θ_ij))을 건다.
#                   Bell-0 baseline 은 E=∅(빈 집합).
#   - 각 party brickwall QCNN 2층:
#       conv1(U) brickwall → pool1(측정+conditional V) → conv2(U₂) → pool2 → readout
#   - 출력 : qml.probs(wires=[3,7]) → [P00,P01,P10,P11], f(x)=W·probs
#
# mid-circuit 측정 + conditional V 구조는 그대로 유지하되, autograd/backprop 와의
# 호환을 위해 qml.defer_measurements 트랜스폼을 QNode 에 적용한다(측정→제어연산은
# deferred 형태로 컴파일되지만 pooling 의 "측정 결과에 따른 V 적용" 의미는 보존).

import os

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 그림 출력 경로 (repo 루트의 figures/. 디렉토리는 generate_all_figures 에서 보장)
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(_REPO_ROOT, "figures")

# ─────────────────────────────────────────────────────────────────────────────
# 0. 전역 배선 정의
# ─────────────────────────────────────────────────────────────────────────────
N_QUBITS = 8
PARTY_A = [0, 1, 2, 3]
PARTY_B = [4, 5, 6, 7]
READOUT = [3, 7]                       # 각 party 의 최종 생존 qubit

# 모든 cross pair (i∈A, j∈B) — E 는 이 집합의 부분집합으로 호출 시 지정
ALL_CROSS_PAIRS = [(i, j) for i in PARTY_A for j in PARTY_B]
# 대각 cross pair: (0,4),(1,5),(2,6),(3,7) — match2 의 자연스러운 처방 예시
DIAG_CROSS_PAIRS = [(PARTY_A[k], PARTY_B[k]) for k in range(4)]


# ─────────────────────────────────────────────────────────────────────────────
# 1. 파라미터 개수 규약
# ─────────────────────────────────────────────────────────────────────────────
# conv U / U₂ (2-qubit trainable unitary) 분해 — 이미지 하단 분해를 따른다:
#   [양 wire 단일큐빗 회전(각 3각)] → CNOT(ctrl=아래, tgt=위)
#   → Rz(위)·Ry(아래) → CNOT(ctrl=위, tgt=아래) → Ry(아래)
#   → CNOT(ctrl=아래, tgt=위) → [양 wire 단일큐빗 회전(각 3각)]
# 단일큐빗 회전 1개 = Rot(φ,θ,ω) = 3 파라미터.
#   앞단 2 wire × 3 = 6,  가운데 Rz+Ry+Ry = 3,  뒷단 2 wire × 3 = 6  → 합 15
N_U_PARAMS = 15

# pool V (conditional 1-qubit unitary) — Rot = 3 파라미터
N_V_PARAMS = 3

# entangler(IsingZZ) — pair 당 1 파라미터
N_ENT_PARAMS = 1


def param_shapes():
    """학습 파라미터 텐서들의 shape 규약을 dict 로 반환.

    party 마다 동일 구조 → axis 0 = party(2개) 로 묶는다.
      U1a, U1b : conv1 의 두 brickwall U (예: A 의 U(0,1), U(2,3))   shape (2,2,15)
      U1c      : conv1 의 가운데 U (예: A 의 U(1,2))                  shape (2,15)
      U2       : conv2 의 U₂ (예: A 의 U(1,3))                        shape (2,15)
      V1       : pool1 의 conditional V (qubit 0,2 측정 → 1,3 에 V)  shape (2,2,3)
      V2       : pool2 의 conditional V (qubit 1 측정 → 3 에 V)      shape (2,3)
      W        : combiner classical weight                          shape (4,)
    """
    return {
        "U1_pairs": (2, 2, N_U_PARAMS),   # [party, {pair0,pair1}, 15]
        "U1_mid":   (2, N_U_PARAMS),      # [party, 15]
        "U2":       (2, N_U_PARAMS),      # [party, 15]
        "V1":       (2, 2, N_V_PARAMS),   # [party, {cond0,cond1}, 3]
        "V2":       (2, N_V_PARAMS),      # [party, 3]
        "W":        (4,),
    }


def init_params(seed=2024, scale=0.1):
    """규약에 맞는 초기 파라미터 dict 생성 (작은 무작위값). W 는 균등 초기화."""
    rng = np.random.RandomState(seed)
    shapes = param_shapes()
    p = {}
    for k, shp in shapes.items():
        if k == "W":
            p[k] = np.ones(shp) * 0.25
        else:
            p[k] = rng.uniform(-scale, scale, size=shp)
    return p


def init_entangler(E, seed=2024, scale=0.1):
    """얽힘 자리 파라미터 θ_ij 를 E 의 pair 순서대로 1차원 배열로 생성.

    반환: (theta: shape (len(E),),  E_list: 정렬된 pair 리스트)
    E=∅ 이면 길이 0 배열.
    """
    E_list = list(E)
    rng = np.random.RandomState(seed + 777)
    theta = rng.uniform(-scale, scale, size=len(E_list))
    return np.asarray(theta, dtype=float), E_list


def init_seed_params(seed):
    """seed 기반 학습 시작 파라미터. W 는 대칭 깨기 위해 작은 무작위 초기화."""
    p = init_params(seed=seed)
    rng = np.random.RandomState(seed + 13)
    p["W"] = rng.uniform(-0.5, 0.5, size=4)
    return p


def pack(p, theta):
    """파라미터 dict p + entangler theta → 단일 1차원 trainable 벡터(autograd)."""
    parts = [np.asarray(p[k]).ravel() for k in param_shapes()] + [np.asarray(theta).ravel()]
    return pnp.array(np.concatenate(parts), requires_grad=True)


def unpack(flat, n_ent):
    """flat 벡터 → (파라미터 dict, theta). param_shapes() 순서를 그대로 따른다."""
    d, i = {}, 0
    for k, sh in param_shapes().items():
        n = int(np.prod(sh))
        d[k] = pnp.reshape(flat[i:i + n], sh)
        i += n
    theta = flat[i:i + n_ent]
    return d, theta


# ─────────────────────────────────────────────────────────────────────────────
# 2. 게이트 블록
# ─────────────────────────────────────────────────────────────────────────────
def _entangler(theta, wire_i, wire_j):
    """통일된 trainable 2-qubit entangler. 모든 실험 조건에서 동일 게이트(IsingZZ)."""
    qml.IsingZZ(theta, wires=[wire_i, wire_j])


def conv_u_block(params, wires):
    """이미지 하단 분해를 따른 trainable 2-qubit unitary U 를 구성하는 단일 core 함수.

    게이트를 **큐에 넣지 않고** 리스트로 반환한다(학습 회로는 qml.apply 로 inline 적용,
    ConvU.compute_decomposition 은 이 리스트를 그대로 분해로 사용 → 단일 정의 소스).

    params: (15,)  배치 순서
      0:3   앞단 Rot(위)        3:6   앞단 Rot(아래)
      6     중앙 Rz(위)         7     중앙 Ry(아래)   8  Ry(아래, 2번째)
      9:12  뒷단 Rot(위)        12:15 뒷단 Rot(아래)
    wires = [w_top, w_bot]
    배선(이미지): CNOT(ctrl=아래,tgt=위) → Rz(위)/Ry(아래)
                  → CNOT(ctrl=위,tgt=아래) → Ry(아래)
                  → CNOT(ctrl=아래,tgt=위)
    """
    w_top, w_bot = wires
    with qml.QueuingManager.stop_recording():
        ops = [
            # 앞단 단일큐빗 회전
            qml.Rot(params[0], params[1], params[2], wires=w_top),
            qml.Rot(params[3], params[4], params[5], wires=w_bot),
            # CNOT(control=아래, target=위)
            qml.CNOT(wires=[w_bot, w_top]),
            # Rz(위), Ry(아래)
            qml.RZ(params[6], wires=w_top),
            qml.RY(params[7], wires=w_bot),
            # CNOT(control=위, target=아래)
            qml.CNOT(wires=[w_top, w_bot]),
            # Ry(아래)
            qml.RY(params[8], wires=w_bot),
            # CNOT(control=아래, target=위)
            qml.CNOT(wires=[w_bot, w_top]),
            # 뒷단 단일큐빗 회전
            qml.Rot(params[9], params[10], params[11], wires=w_top),
            qml.Rot(params[12], params[13], params[14], wires=w_bot),
        ]
    return ops


class ConvU(qml.operation.Operation):
    """conv U 를 단일 박스로 표현하기 위한 custom Operation (그림 전용).

    num_wires=2, 단일 길이-15 파라미터 배열. 분해는 conv_u_block 을 그대로 호출.
    """

    num_wires = 2
    num_params = 1
    ndim_params = (1,)
    grad_method = None

    @staticmethod
    def compute_decomposition(params, wires):
        return conv_u_block(params, wires)

    def label(self, decimals=None, base_label=None, cache=None):
        return base_label or "U"


def _conv_U(params, w_top, w_bot):
    """학습 회로용: conv_u_block 의 게이트를 inline(qml.apply)으로 큐에 적용.

    게이트 내용·순서는 Phase 1 과 동일 → 수치 결과 불변.
    """
    for op in conv_u_block(params, [w_top, w_bot]):
        qml.apply(op)


def _pool(v_params, measured_wire, target_wire):
    """conditional pooling: measured_wire 를 mid-circuit 측정 →
    그 결과(=1)일 때 conditional V(Rot) 를 target_wire 에 적용. 살아남는 = target_wire."""
    m = qml.measure(measured_wire)
    qml.cond(m, qml.Rot)(v_params[0], v_params[1], v_params[2], wires=target_wire)


def _party_block(p, party_idx, wires):
    """한 party([q0,q1,q2,q3] 형태 wires)에 대한 brickwall QCNN 2층.

    반환: 이 party 의 readout qubit (wires[3]).
    wires 국소 인덱스: a0,a1,a2,a3 = wires[0..3]
      conv1 brickwall: U(a0,a1), U(a2,a3), U(a1,a2)
      pool1          : (a0,a1)->a1, (a2,a3)->a3   생존 {a1,a3}
      conv2          : U(a1,a3)
      pool2          : (a1,a3)->a3                 생존 {a3} = readout
    """
    a0, a1, a2, a3 = wires
    # conv1 brickwall
    _conv_U(p["U1_pairs"][party_idx][0], a0, a1)   # U(a0,a1)
    _conv_U(p["U1_pairs"][party_idx][1], a2, a3)   # U(a2,a3)
    _conv_U(p["U1_mid"][party_idx],      a1, a2)   # U(a1,a2)  (brick 엇물림)
    # pool1: a0 측정→a1 에 V, a2 측정→a3 에 V
    _pool(p["V1"][party_idx][0], a0, a1)
    _pool(p["V1"][party_idx][1], a2, a3)
    # conv2: U(a1,a3)
    _conv_U(p["U2"][party_idx], a1, a3)
    # pool2: a1 측정→a3 에 V
    _pool(p["V2"][party_idx], a1, a3)
    return a3


# ─────────────────────────────────────────────────────────────────────────────
# 3. 회로 본체 (probs 반환)
# ─────────────────────────────────────────────────────────────────────────────
def circuit_probs(x, p, theta, E_list):
    """8-qubit 분산 QCNN. probs(wires=[3,7]) 반환.

    x       : (8,) 각도 feature
    p       : init_params() dict
    theta   : (len(E_list),) entangler 파라미터
    E_list  : cross pair 리스트 (E=∅ 이면 빈 리스트) — 얽힘 자리에만 게이트
    """
    # 1) 인코딩: qubit 마다 Ry(x_k)
    for k in range(N_QUBITS):
        qml.RY(x[k], wires=k)
    # 2) 얽힘 자리: E 의 cross pair 에만 entangler
    for idx, (i, j) in enumerate(E_list):
        _entangler(theta[idx], i, j)
    # 3) 각 party brickwall QCNN
    _party_block(p, 0, PARTY_A)
    _party_block(p, 1, PARTY_B)
    # 4) 출력
    return qml.probs(wires=READOUT)


def make_qnode(dev=None):
    """defer_measurements 적용 + backprop(autograd) QNode 생성.

    반환된 qnode 시그니처: qnode(x, p, theta, E_list)
    """
    if dev is None:
        dev = qml.device("default.qubit", wires=N_QUBITS)

    @qml.qnode(dev, interface="autograd", diff_method="backprop")
    @qml.defer_measurements
    def qnode(x, p, theta, E_list):
        return circuit_probs(x, p, theta, E_list)

    return qnode


def predict(qnode, x, p, theta, E_list):
    """f(x) = W·probs = W1*P00 + W2*P01 + W3*P10 + W4*P11."""
    probs = qnode(x, p, theta, E_list)
    return qml.math.dot(p["W"], probs)


# ─────────────────────────────────────────────────────────────────────────────
# 4. 그림 출력 (검증용) — 학습과 동일한 conv_u_block / _pool 만 호출 (단일 소스)
# ─────────────────────────────────────────────────────────────────────────────
# 그림 전용 고정 파라미터 (seed 2024). 전역 np.random 상태를 건드리지 않도록
# 별도 RandomState 사용. 값 자체는 abstract(decimals=None)에선 표시되지 않음.
_DRAW_P = init_params(seed=2024)
_DRAW_X = np.random.RandomState(2024).uniform(0, 2 * np.pi, size=N_QUBITS)
_DRAW_THETA = 0.5    # entangler 각 (raw 에서만 숫자로 표시)


def _draw_qfunc(E, use_barrier, expand_u):
    """그림 전용 구조 회로 (defer 미적용 → measure+cond 직접 사용).

    expand_u=False : conv U 를 ConvU 단일 박스로 (abstract)
    expand_u=True  : 동일한 conv_u_block 게이트를 inline 으로 펼침 (raw)
    use_barrier    : stage 구분 Barrier(only_visual=True) 삽입 여부만 토글
    → barrier 외 게이트는 두 버전 동일 (U 접기 차이만 존재).
    """
    p, x = _DRAW_P, _DRAW_X

    def U(prm, w):
        if expand_u:
            for op in conv_u_block(np.asarray(prm), w):
                qml.apply(op)
        else:
            ConvU(np.asarray(prm), wires=w)

    def barrier():
        if use_barrier:
            qml.Barrier(wires=range(N_QUBITS), only_visual=True)

    # (1) Ry 인코딩 8개
    for k in range(N_QUBITS):
        qml.RY(x[k], wires=k)
    barrier()
    # (2) entangle: E 의 cross pair 에만 IsingZZ
    for (i, j) in sorted(E):
        qml.IsingZZ(_DRAW_THETA, wires=[i, j])
    barrier()
    # (3) conv1 brickwall: A=(0,1),(2,3),(1,2) / B=(4,5),(6,7),(5,6)
    U(p["U1_pairs"][0][0], [0, 1]); U(p["U1_pairs"][0][1], [2, 3]); U(p["U1_mid"][0], [1, 2])
    U(p["U1_pairs"][1][0], [4, 5]); U(p["U1_pairs"][1][1], [6, 7]); U(p["U1_mid"][1], [5, 6])
    barrier()
    # (4) pool1: qubit 0,2,4,6 measure → V 를 1,3,5,7 에 cond (학습과 동일한 _pool)
    _pool(p["V1"][0][0], 0, 1); _pool(p["V1"][0][1], 2, 3)
    _pool(p["V1"][1][0], 4, 5); _pool(p["V1"][1][1], 6, 7)
    barrier()
    # (5) conv2: ConvU on (1,3),(5,7)
    U(p["U2"][0], [1, 3]); U(p["U2"][1], [5, 7])
    barrier()
    # (6) pool2: qubit 1,5 measure → V 를 3,7 에 cond
    _pool(p["V2"][0], 1, 3); _pool(p["V2"][1], 5, 7)
    barrier()
    # (7) readout
    return qml.probs(wires=READOUT)


def _make_draw_qnode(use_barrier, expand_u):
    dev = qml.device("default.qubit", wires=N_QUBITS)

    @qml.qnode(dev)
    def qn(E):
        return _draw_qfunc(E, use_barrier, expand_u)

    return qn


def draw_circuit(E, style="black_white", level="top", use_barrier=True,
                 decimals=None, fname=None, title=None):
    """그림 전용 통합 함수. abstract / raw 를 옵션만 바꿔 생성.

    level="top"     → abstract: U 를 단일 박스(ConvU)로 접음.
    level="device"  → raw: U 를 실제 게이트로 펼침. (PL 0.42 의 device-level draw 는
                      mid-circuit 측정을 defer 시켜 pool 그림을 바꾸므로, spec 의
                      "expand 후 그리기" fallback 으로 conv_u_block 을 inline 펼침 →
                      측정+cond 는 네이티브로 보존.)
    use_barrier     → Barrier(only_visual=True) 삽입 여부 토글.
    decimals        → None 이면 게이트 박스에 숫자 표시 안 함(abstract), 2 면 표시(raw).
    """
    expand_u = (level != "top")
    qml.drawer.use_style(style)
    qn = _make_draw_qnode(use_barrier, expand_u)
    fig, ax = qml.draw_mpl(qn, decimals=decimals)(E)
    if title:
        ax.set_title(title, fontsize=12)
    if fname:
        os.makedirs(FIG_DIR, exist_ok=True)
        out = os.path.join(FIG_DIR, fname)
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"saved: {out}   (E={sorted(E)}, level={level}, barrier={use_barrier}, decimals={decimals})")
        return out
    return fig, ax


def draw_conv_u_definition(fname="conv_u_definition.png", decimals=2, style="black_white"):
    """2큐빗 미니 QNode 로 conv_u_block(wires=[0,1]) 만 그린 U 분해도.

    학습 회로와 동일한 conv_u_block 을 호출 → 정의가 일치함을 시각 확인.
    """
    qml.drawer.use_style(style)
    prm = np.asarray(init_params(seed=2024)["U1_pairs"][0][0])   # 길이-15 예시 파라미터
    dev2 = qml.device("default.qubit", wires=2)

    @qml.qnode(dev2)
    def u_def(params):
        for op in conv_u_block(params, [0, 1]):
            qml.apply(op)
        return qml.probs(wires=[0, 1])

    fig, ax = qml.draw_mpl(u_def, decimals=decimals)(prm)
    ax.set_title("conv_u_block decomposition (U on wires [0,1])", fontsize=12)
    os.makedirs(FIG_DIR, exist_ok=True)
    out = os.path.join(FIG_DIR, fname)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out}   (conv_u_block decomposition)")
    return out


def generate_all_figures():
    """spec 의 5장 figure 를 circuit.py 단일 실행으로 생성."""
    os.makedirs(FIG_DIR, exist_ok=True)
    # Bell-0 (E=∅)
    draw_circuit(set(), level="top", use_barrier=True, decimals=None,
                 fname="circuit_bell0.png",
                 title="Distributed QCNN (brickwall) - abstract, Bell-0 (E = empty)")
    draw_circuit(set(), level="device", use_barrier=False, decimals=2,
                 fname="circuit_bell0_raw.png",
                 title="Distributed QCNN (brickwall) - raw (U expanded), Bell-0 (E = empty)")
    # Prescribed (E={(3,7)})
    draw_circuit({(3, 7)}, level="top", use_barrier=True, decimals=None,
                 fname="circuit_prescribed.png",
                 title="Distributed QCNN (brickwall) - abstract, prescribed E = {(3,7)}")
    draw_circuit({(3, 7)}, level="device", use_barrier=False, decimals=2,
                 fname="circuit_prescribed_raw.png",
                 title="Distributed QCNN (brickwall) - raw (U expanded), prescribed E = {(3,7)}")
    # conv_u_block 정의도
    draw_conv_u_definition()


# ─────────────────────────────────────────────────────────────────────────────
# 5. 자체 점검 + 그림 생성
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    np.random.seed(2024)
    p = init_params(seed=2024)
    qn = make_qnode()

    x = np.random.uniform(0, 2 * np.pi, size=8)

    # Bell-0 baseline: E=∅  (수치 결과 통합 전과 동일한지 확인)
    theta0, E0 = init_entangler(set())
    pr0 = qn(x, p, theta0, E0)
    print("Bell-0  probs:", np.round(np.asarray(pr0), 4), " sum=", float(np.sum(pr0)))
    print("Bell-0  f(x) =", float(predict(qn, x, p, theta0, E0)))

    # 처방 예시: 대각 cross pair 2개
    E_ex = [(0, 4), (1, 5)]
    theta_ex, E_ex = init_entangler(E_ex)
    pr1 = qn(x, p, theta_ex, E_ex)
    print("Prescribed probs:", np.round(np.asarray(pr1), 4), " sum=", float(np.sum(pr1)))
    print("param shapes:", {k: np.asarray(v).shape for k, v in p.items()})

    # ── 그림 5장 생성 ──────────────────────────────────────
    print("\n[figures]")
    generate_all_figures()
    print("done.")
