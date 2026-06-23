# CPFP-LOCC ablation: 데이터 로딩 / 색→각도 검증 / 파티 분할
#   CSV 컬럼 규약: 0~7 = 8개 각도 feature, 8 = 보조 통계(라벨 결정 카운트),
#                  9 = 라벨(±1).
#   - match2: col8 = 대각 cross 매치 수(a_i==b_i), 라벨 = +1 iff col8>=2  → cross 구조
#   - pair3 : col8 = 파티 내부(marginal) 통계, 라벨 = +1 iff col8>=2     → cross 없음

import os
import numpy as np
import pandas as pd

from . import config


# ────────────────────────────────────────
# 0. 각도 검증
# ────────────────────────────────────────
def verify_angles(X, atol=1e-6):
    """모든 feature가 [0,2π) 안의 4개 허용 각도 {0,π/2,π,3π/2}인지 검증.

    반환: (in_range: bool, all_on_grid: bool)
    """
    in_range = bool(np.all(X >= -atol) and np.all(X < 2 * np.pi + atol))
    # 각 값이 허용 각도 중 하나에 가까운가
    dist = np.min(np.abs(X[..., None] - config.VALID_ANGLES[None, None, :]), axis=-1)
    all_on_grid = bool(np.all(dist <= atol))
    return in_range, all_on_grid


# ────────────────────────────────────────
# 1. 데이터셋 로딩
# ────────────────────────────────────────
def load_dataset(name):
    """논리 이름('match2'/'pair3')으로 CSV를 로드한다.

    반환 dict:
      X    : (n,8) float  각도 feature
      y    : (n,)  float  라벨 ±1
      aux  : (n,)  int    col8 보조 통계
      XA   : (n,4) 파티 A feature,  XB : (n,4) 파티 B feature
    """
    if name not in config.DATASETS:
        raise KeyError(f"알 수 없는 데이터셋: {name} (가능: {list(config.DATASETS)})")
    path = os.path.join(config.DATA_DIR, config.DATASETS[name])
    df = pd.read_csv(path)
    arr = df.values.astype(float)
    X = arr[:, :config.N_FEATURES]
    aux = arr[:, 8].astype(int)
    y = arr[:, 9].astype(float)
    return {
        "name": name,
        "path": path,
        "X": X,
        "y": y,
        "aux": aux,
        "XA": X[:, config.PARTY_A],
        "XB": X[:, config.PARTY_B],
    }


# ────────────────────────────────────────
# 2. 요약 통계
# ────────────────────────────────────────
def summarize(ds):
    """라벨/색/각도 분포 요약을 dict로 반환한다."""
    X, y, aux = ds["X"], ds["y"], ds["aux"]
    in_range, on_grid = verify_angles(X)
    # 색 인덱스(0~3) 복원: 각도 → 가장 가까운 허용 각도의 순서
    color_idx = np.argmin(
        np.abs(X[..., None] - config.VALID_ANGLES[None, None, :]), axis=-1)
    colors, color_counts = np.unique(color_idx, return_counts=True)
    labels, label_counts = np.unique(y, return_counts=True)
    return {
        "n": X.shape[0],
        "n_features": X.shape[1],
        "angles_in_range": in_range,
        "angles_on_grid": on_grid,
        "label_dist": dict(zip(labels.tolist(), label_counts.tolist())),
        "pos_frac": float(np.mean(y > 0)),
        "color_dist": dict(zip(colors.tolist(), color_counts.tolist())),
        "aux_dist": {int(k): int(v)
                     for k, v in zip(*np.unique(aux, return_counts=True))},
    }
