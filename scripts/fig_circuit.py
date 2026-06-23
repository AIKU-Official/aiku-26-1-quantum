# 논문용 회로 figure: **실제 학습에 쓴** 정본(cond) 4+4 회로를 qml.draw_mpl 로 출력.
#   ⚠️ 예시 재현본(outputs/distributed_qcnn_circuit.py)을 다시 그리는 게 아니라,
#      우리가 학습에 사용한 src/circuit_cond.py 의 make_qnode/local_qcnn/_conv4/
#      _cond_pool 을 그대로 import 해서 그린다 (--method cond 정본).
#
#   (a) Bell-0 baseline  : 사전공유 Bell pair 없음 (angle encoding 만)
#   (b) Prescribed       : gap-K*=4, 대각 4쌍 (0-4,1-5,2-6,3-7) Bell pair 삽입
#
#   가독성: n_blocks=1, reupload=False (단일 angle encoding), 더미 입력 1개.
#   ※ 실제 학습은 n_blocks=4 + re-uploading 사용 — 여기선 구조 가독성 위해 단순화.
#
#   생존경로(실제 코드대로): A {0,1,2,3}->{0,2}->{0},  B {4,5,6,7}->{4,6}->{4}
#       readout = (q0, q4).  (첨부 이미지의 {1,3}->{3}/{5,7}->{7} 와는 다름 —
#       실제 코드가 정본이므로 코드대로 그린다. 자세한 차이는 콘솔 보고 참조.)

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pennylane as qml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config
from src.circuits import get

# 실제 학습 회로(정본 cond pooling) 모듈을 그대로 사용
C = get("cond")          # == src.circuit_cond

FIG_DIR = os.path.join(config.BASE_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

N_BLOCKS = 1
# 게이트 종류를 '값'으로 구분(라벨로 읽기 쉽게):
#   encoding RY(x)=0.79 (모든 wire 동일) / Bell RY(θ)=1.20 / trainable conv/pool=0.50
ENC_VAL = round(np.pi / 4, 2)          # 0.79  angle encoding (uniform dummy)
X_DUMMY = np.full((1, 8), np.pi / 4)   # RY(0.79) on every wire = encoding column
THETA = 1.20                            # Bell rotation (distinct value)
CONV_VAL = 0.50                         # trainable conv / cond-pool RX param
# Prescribed: gap-K*=4 대각 4쌍 (파티 A 큐빗 i ↔ 파티 B 큐빗 4+i)
BELL_PAIRS = [(0, 4, THETA), (1, 5, THETA), (2, 6, THETA), (3, 7, THETA)]

KEY = ("Gate key:   RY(0.79) = angle encoding RY(x)      "
       "RY(1.20) = pre-shared Bell rotation RY(theta)      "
       "RY(0.50) = trainable conv RY      CRY = conv entangler      "
       "RX(0.50)|m = measurement-conditioned pool rotation")

A_BG = "#E8F1FA"     # 파티 A 배경 (옅은 파랑)
B_BG = "#FCEFE3"     # 파티 B 배경 (옅은 주황)


def shade_parties(ax):
    """wire 0~3 = 파티 A, 4~7 = 파티 B 배경색 + 라벨.
       cond pooling 은 MCM deferred 전개로 보조 wire(8~)가 붙을 수 있으므로
       현재 축 y범위를 읽어서 음영을 안전하게 채운다."""
    y0, y1 = ax.get_ylim()                 # draw_mpl: y 위=작은값
    top = min(y0, y1) - 0.0
    bot = max(y0, y1)
    ax.axhspan(-0.5, 3.5, color=A_BG, zorder=-10)
    ax.axhspan(3.5, 7.5, color=B_BG, zorder=-10)
    ax.axhline(3.5, color="0.5", lw=1.0, ls="--", zorder=-5)
    x0 = ax.get_xlim()[0]
    ax.text(x0 - 0.25, 1.5, "Party A  (QPU 1)\nwires 0-3", rotation=90,
            va="center", ha="center", fontsize=12, fontweight="bold", color="#0072B2")
    ax.text(x0 - 0.25, 5.5, "Party B  (QPU 2)\nwires 4-7", rotation=90,
            va="center", ha="center", fontsize=12, fontweight="bold", color="#D55E00")


def draw_case(bell_pairs, title, fname):
    # 실제 학습 회로 그대로 (정본 cond pooling, 파티 간 게이트 없음)
    qnode = C.make_qnode(pooling=True, n_blocks=N_BLOCKS, reupload=False,
                         bell_pairs=bell_pairs)
    n_q = C.n_quantum_params(True, N_BLOCKS)
    qp = np.full(n_q, CONV_VAL)         # 일정한 더미 파라미터 (라벨 깔끔)

    fig, ax = qml.draw_mpl(
        qnode, style="black_white", decimals=2, fontsize=14,
        wire_order=list(range(8)),
    )(X_DUMMY, qp)

    fig.set_size_inches(17, 7.8)
    shade_parties(ax)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=14)
    fig.text(0.5, 0.012, KEY, ha="center", va="bottom", fontsize=9.5,
             family="monospace",
             bbox=dict(boxstyle="round,pad=0.4", fc="#f5f5f5", ec="0.6", lw=0.8))
    out = os.path.join(FIG_DIR, fname)
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved: {out}")


def main():
    # (a) Bell-0 baseline
    draw_case(
        None,
        "(a) Bell-0 baseline  -  no entanglement   "
        "(angle encoding RY(x) -> local QCNN: conv (RY+CRY) / cond pool (measure + RX) -> readout q0,q4)",
        "fig_circuit_bell0.png")
    # (b) Prescribed (gap-K*=4, 대각 4쌍)
    draw_case(
        BELL_PAIRS,
        "(b) Prescribed  -  pre-shared Bell pairs RY(theta)+CNOT on diagonal "
        "(0,4),(1,5),(2,6),(3,7) before encoding   [gap-K* = 4]",
        "fig_circuit_prescribed.png")


if __name__ == "__main__":
    main()
