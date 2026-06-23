# 논문용 회로 figure: src/circuit.py 의 4+4 회로를 qml.draw_mpl 로 출력.
#   (a) Bell-0 baseline (얽힘 없음)
#   (b) Prescribed (대각 사전공유 Bell pair 포함)
#   가독성: n_blocks=1, reupload=False(단일 angle encoding), 더미 입력 1개.
#   파티 A(0~3)/B(4~7) 배경색 + 라벨로 구분. 영어 라벨, PNG dpi=300.
#   ※ 학습 모델은 n_blocks=4 + re-uploading 사용 — 여기선 구조 가독성 위해 단순화.

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pennylane as qml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config, circuit as C

os.makedirs(config.OUTPUT_DIR, exist_ok=True)

N_BLOCKS = 1
# 게이트 종류를 '값'으로 구분(주석 달기 쉽게):
#   encoding RY(x)=0.79 (모든 wire 동일) / Bell RY(θ)=1.20 / trainable conv RY=0.50
ENC_VAL = round(np.pi / 4, 2)          # 0.79  angle encoding (uniform dummy)
X_DUMMY = np.full((1, 8), np.pi / 4)   # RY(0.79) on every wire = encoding column
THETA = 1.20                            # Bell rotation (distinct value)
CONV_VAL = 0.50                         # trainable conv/pool param
# Prescribed 대각 Bell pair (파티 A 큐빗 i ↔ 파티 B 큐빗 4+i). 가독성 위해 2쌍.
BELL_PAIRS = [(0, 4, THETA), (2, 6, THETA)]

KEY = ("Gate key:   RY(0.79) = angle encoding RY(x)      "
       "RY(1.20) = pre-shared Bell rotation RY(θ)      "
       "RY(0.50) = trainable conv      CRY = conv/pool entangler")

A_BG = "#E8F1FA"     # 파티 A 배경 (옅은 파랑)
B_BG = "#FCEFE3"     # 파티 B 배경 (옅은 주황)


def shade_parties(ax, n_wires=8):
    """wire 0~3 = 파티 A, 4~7 = 파티 B 배경색 + 라벨."""
    # draw_mpl: wire k 는 y=k (위=0, 아래로 증가)
    ax.axhspan(-0.5, 3.5, color=A_BG, zorder=-10)
    ax.axhspan(3.5, 7.5, color=B_BG, zorder=-10)
    ax.axhline(3.5, color="0.5", lw=1.0, ls="--", zorder=-5)
    x0 = ax.get_xlim()[0]
    ax.text(x0 - 0.2, 1.5, "Party A  (QPU 1)\nwires 0–3", rotation=90,
            va="center", ha="center", fontsize=12, fontweight="bold", color="#0072B2")
    ax.text(x0 - 0.2, 5.5, "Party B  (QPU 2)\nwires 4–7", rotation=90,
            va="center", ha="center", fontsize=12, fontweight="bold", color="#D55E00")


def draw_case(bell_pairs, title, fname):
    qnode = C.make_qnode(pooling=True, n_blocks=N_BLOCKS, reupload=False,
                         bell_pairs=bell_pairs)
    n_q = C.n_quantum_params(True, N_BLOCKS)
    qp = np.full(n_q, CONV_VAL)         # 일정한 더미 파라미터 (라벨 깔끔)

    fig, ax = qml.draw_mpl(
        qnode, style="black_white", decimals=2, fontsize=15,
        wire_order=list(range(8)),
    )(X_DUMMY, qp)

    fig.set_size_inches(16, 7.4)
    shade_parties(ax)
    ax.set_title(title, fontsize=15, fontweight="bold", pad=14)
    fig.text(0.5, 0.015, KEY, ha="center", va="bottom", fontsize=10.5,
             family="monospace",
             bbox=dict(boxstyle="round,pad=0.4", fc="#f5f5f5", ec="0.6", lw=0.8))
    out = os.path.join(config.OUTPUT_DIR, fname)
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved: {out}")


def main():
    # (a) Bell-0 baseline
    draw_case(None,
              "(a) Bell-0 baseline  —  no entanglement  "
              "(angle encoding RY(x) → local QCNN conv/pool → readout)",
              "fig_circuit_bell0.png")
    # (b) Prescribed
    draw_case(BELL_PAIRS,
              "(b) Prescribed  —  pre-shared Bell pairs RY(θ)+CNOT on diagonal "
              "(0,4),(2,6) before encoding",
              "fig_circuit_prescribed.png")


if __name__ == "__main__":
    main()
