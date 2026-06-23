# CPFP-LOCC ablation: 전역 설정 (경로 / 시드 / 그림 스타일)
#   모든 스크립트가 이 모듈을 import 해서 동일한 경로·스타일·시드를 공유한다.

import os
import numpy as np
import matplotlib as mpl

# ────────────────────────────────────────
# 0. 전역 상수
# ────────────────────────────────────────
SEED = 2024
np.random.seed(SEED)

# 경로 (repo 루트 기준 상대경로 — 이 파일은 <repo>/src/config.py 이므로 두 단계 상위가 루트)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
SRC_DIR = os.path.join(BASE_DIR, "src")
OUTPUT_DIR = os.path.join(BASE_DIR, "figures")   # 그림 출력 (= README 의 figures/)
CACHE_DIR = os.path.join(BASE_DIR, "cache")
for _d in (OUTPUT_DIR, CACHE_DIR):
    os.makedirs(_d, exist_ok=True)

# 데이터셋 파일명 (논리 이름 → CSV)
DATASETS = {
    "match2": "color4match2_angle_sample_dataset.csv",  # 대각 cross 구조 (얽힘 도움 예상)
    "pair3": "color4pair3_angle_sample_dataset.csv",    # 파티 내부 marginal 구조 (얽힘 무용 예상)
}

# 색(0,1,2,3) → 각도 매핑: x ∈ [0,2π] 주기성 + Fourier 직교성 보장
ANGLE_MAP = {0: 0.0, 1: np.pi / 2, 2: np.pi, 3: 3 * np.pi / 2}
VALID_ANGLES = np.array(sorted(ANGLE_MAP.values()))

# 파티 분할: 8D = 4(A) + 4(B)
N_FEATURES = 8
PARTY_A = [0, 1, 2, 3]   # 파티 A 큐빗/feature 인덱스
PARTY_B = [4, 5, 6, 7]   # 파티 B 큐빗/feature 인덱스

# ────────────────────────────────────────
# 1. 그림 스타일 (Nature 풍, 그림 내 텍스트는 영어만)
# ────────────────────────────────────────
MM = 1 / 25.4
W_SINGLE = 89 * MM
W_DOUBLE = 183 * MM

# Okabe-Ito 색맹 안전 팔레트
OKABE_ITO = ["#0072B2", "#E69F00", "#009E73", "#CC79A7",
             "#D55E00", "#56B4E9", "#F0E442", "#000000"]


def apply_style():
    """matplotlib 전역 스타일을 Nature 풍으로 설정한다."""
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7,
        "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 6,
        "axes.linewidth": 0.5, "lines.linewidth": 1.0,
        "xtick.major.width": 0.5, "ytick.major.width": 0.5,
        "xtick.direction": "out", "ytick.direction": "out",
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "figure.dpi": 150, "savefig.dpi": 300,
        "savefig.bbox": "tight", "pdf.fonttype": 42, "ps.fonttype": 42,
    })
