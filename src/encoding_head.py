# CPFP final: 인코딩 head — raw feature → Ry 인코딩 각도(angle) 변환
# ─────────────────────────────────────────────────────────────────────────────
# 단일 변환 방식(분기 없음): 컬럼별 min-max 선형 매핑 → [0, π].
#
#     angle = (x - col_min) / (col_max - col_min) * π
#
# 왜 [0, 2π) 가 아니라 [0, π] 인가:
#   Ry(0) = Ry(2π) = I 이므로 [0, 2π) 를 쓰면 컬럼의 최소값과 최대값이
#   wrap-around 로 같은 게이트(항등)에 매핑돼 회로상 구분되지 않는다.
#   [0, π] 로 매핑하면 col_min → Ry(0), col_max → Ry(π) 로 끝점이 확실히 구분된다.
#
# party-preserving:
#   A_cols 는 A 끼리, B_cols 는 B 끼리 따로 min/max(및 median) fit.
#   A·B 전체를 한 scaler 로 fit 하지 않는다(party 경계 혼입 방지). 변환은 컬럼별
#   독립이므로 한 party 컬럼 값을 바꿔도 다른 party 각도는 불변.
#
# 누수 방지:
#   min/max(및 binary median)는 train 에서만 fit, test 엔 fit 금지.
#   test 가 train 범위를 벗어나면 각도가 [0, π] 밖으로 나갈 수 있으나 clip 하지 않고
#   그대로 둔다(범위 이탈 비율은 self-test 에서 리포트만).

import numpy as np

DEFAULT_A_COLS = [0, 1, 2, 3]
DEFAULT_B_COLS = [4, 5, 6, 7]


class EncodingHead:
    """raw feature 행렬 (N, n_cols) → Ry 인코딩 각도로 변환하는 head.

    mode:
      "binary" (기본, 검증됨):
          연속 각도(angles) + 컬럼별 median threshold 로 ±1 이진화한
          pm1_features 를 함께 반환. pm1_features 는 Walsh-Fourier M_ij 입력 전제
          (M_ij 는 binary feature 를 가정). median 도 train 에서만 fit.
      "continuous" (미검증 확장):
          연속 각도만 반환(pm1_features=None). M_ij 는 binary 전제이므로
          continuous 입력에는 estimator 교체가 필요한 미검증 경로다.
    """

    def __init__(self, A_cols=DEFAULT_A_COLS, B_cols=DEFAULT_B_COLS, mode="binary"):
        if mode not in ("binary", "continuous"):
            raise ValueError(f"mode 는 'binary' 또는 'continuous' (받은 값: {mode})")
        self.A_cols = list(A_cols)
        self.B_cols = list(B_cols)
        self.mode = mode
        # fit 산출물
        self.col_min_ = None
        self.col_max_ = None
        self.median_ = None
        self.n_cols_ = None

    # ────────────────────────────────────────
    # fit: train 에서만 통계 추정 (party 별로 따로)
    # ────────────────────────────────────────
    def fit(self, X_train):
        X = np.asarray(X_train, dtype=float)
        n_cols = X.shape[1]
        col_min = np.full(n_cols, np.nan)
        col_max = np.full(n_cols, np.nan)
        median = np.full(n_cols, np.nan)

        # party 경계 혼입 방지: A_cols 는 A 컬럼만, B_cols 는 B 컬럼만 사용해 fit.
        # (컬럼별 독립 min-max 라 결과적으로 컬럼별 통계지만, 한 scaler 로 전체를
        #  묶지 않음을 명시적으로 보장한다.)
        for cols in (self.A_cols, self.B_cols):
            if len(cols) == 0:
                continue
            sub = X[:, cols]
            col_min[cols] = sub.min(axis=0)
            col_max[cols] = sub.max(axis=0)
            median[cols] = np.median(sub, axis=0)

        self.col_min_ = col_min
        self.col_max_ = col_max
        self.median_ = median
        self.n_cols_ = n_cols
        return self

    # ────────────────────────────────────────
    # transform: 단일 방식 min-max → [0, π]
    # ────────────────────────────────────────
    def transform(self, X):
        if self.col_min_ is None:
            raise RuntimeError("fit 먼저 호출해야 함 (train 통계 미설정)")
        X = np.asarray(X, dtype=float)

        denom = self.col_max_ - self.col_min_
        # 상수 컬럼(denom==0) 또는 미fit 컬럼(denom==nan) → 0 으로 매핑(0 division 방지)
        bad = ~(denom > 0)
        with np.errstate(invalid="ignore", divide="ignore"):
            angles = (X - self.col_min_) / denom * np.pi
        angles[:, bad] = 0.0

        if self.mode == "binary":
            # 컬럼별 median threshold 로 ±1 이진화 (x >= median → +1, else -1)
            pm1 = np.where(X >= self.median_, 1.0, -1.0)
            return {"angles": angles, "pm1_features": pm1}
        # continuous: 연속 각도만 (미검증 경로)
        return {"angles": angles, "pm1_features": None}

    def fit_transform(self, X_train):
        return self.fit(X_train).transform(X_train)

    # ────────────────────────────────────────
    # 진단: test 의 [0, π] 범위 이탈 비율 (clip 안 함, 리포트용)
    # ────────────────────────────────────────
    def out_of_range_fraction(self, X):
        ang = self.transform(X)["angles"]
        below = ang < 0.0
        above = ang > np.pi
        return {
            "frac_below_0": float(below.mean()),
            "frac_above_pi": float(above.mean()),
            "frac_outside": float((below | above).mean()),
        }


# ─────────────────────────────────────────────────────────────────────────────
# self-test
# ─────────────────────────────────────────────────────────────────────────────
def _run_self_test():
    np.random.seed(2024)
    lines = []
    PI = np.pi

    def ok(msg):
        lines.append(f"- ✅ {msg}")

    # ── (1) color-match 값(0,1,2,3) → [0,π] 매핑 ──────────────
    # 각 컬럼이 색 0,1,2,3 을 모두 포함하도록 구성
    base_col = np.array([0, 1, 2, 3], dtype=float)
    Xcolor = np.tile(base_col[:, None], (1, 8))   # (4,8), 각 컬럼 = [0,1,2,3]
    head = EncodingHead(mode="binary").fit(Xcolor)
    ang = head.transform(Xcolor)["angles"]
    expected = np.array([0.0, PI / 3, 2 * PI / 3, PI])     # 0, π/3, 2π/3, π
    assert np.allclose(ang[:, 0], expected), ang[:, 0]
    # 4개 값이 서로 구분되는지
    uniq = np.unique(np.round(ang[:, 0], 8))
    assert uniq.size == 4, uniq
    ok(f"color(0,1,2,3) → [0, π/3, 2π/3, π] 매핑 + 4값 구분 (unique={uniq.size})")

    # ── (2) 임의 연속값(스케일 제각각) → (N,8), train 범위 내 [0,π] ──
    scales = np.array([1, 10, 100, 0.1, 5, -3, 1000, 0.01])
    offs = np.array([0, -50, 5, 2, -1, 100, 0, -7])
    Xc = np.random.randn(200, 8) * scales + offs
    head2 = EncodingHead(mode="continuous").fit(Xc)
    out = head2.transform(Xc)
    angc = out["angles"]
    assert angc.shape == (200, 8), angc.shape
    assert out["pm1_features"] is None  # continuous → pm1 없음
    assert angc.min() >= -1e-9 and angc.max() <= PI + 1e-9, (angc.min(), angc.max())
    ok(f"연속 8컬럼 → 출력 {angc.shape}, train 범위 내 [0,π] (min={angc.min():.3f}, max={angc.max():.3f})")

    # ── (3) party-preservation ───────────────────────────────
    X1 = np.random.randn(100, 8) * scales + offs
    a1 = EncodingHead().fit(X1).transform(X1)["angles"]
    # A_cols(0:4)만 흔들기 → B쪽(4:8) 각도 불변
    X_pa = X1.copy(); X_pa[:, 0:4] = np.random.randn(100, 4) * 7 + 3
    a_pa = EncodingHead().fit(X_pa).transform(X_pa)["angles"]
    assert np.allclose(a_pa[:, 4:8], a1[:, 4:8]), "B 각도가 A 변화에 영향받음"
    # B_cols(4:8)만 흔들기 → A쪽(0:4) 각도 불변
    X_pb = X1.copy(); X_pb[:, 4:8] = np.random.randn(100, 4) * 9 - 2
    a_pb = EncodingHead().fit(X_pb).transform(X_pb)["angles"]
    assert np.allclose(a_pb[:, 0:4], a1[:, 0:4]), "A 각도가 B 변화에 영향받음"
    ok("party-preservation: A 변화 → B 각도 불변, B 변화 → A 각도 불변")

    # ── (4) 상수 컬럼 → 0 ────────────────────────────────────
    Xconst = np.random.randn(50, 8) * scales + offs
    Xconst[:, 2] = 4.2   # col 2 상수
    a_const = EncodingHead().fit(Xconst).transform(Xconst)["angles"]
    assert np.allclose(a_const[:, 2], 0.0), a_const[:, 2]
    ok("상수 컬럼(col_max==col_min) → 0 매핑 (0 division 방지)")

    # ── (5) test 범위 이탈(누수 방지) 리포트 — clip 안 함 ──────
    Xtr = np.random.randn(300, 8) * scales + offs
    head5 = EncodingHead(mode="binary").fit(Xtr)
    Xte = np.random.randn(300, 8) * scales * 1.6 + offs   # train 보다 넓은 분포
    frac = head5.out_of_range_fraction(Xte)
    ang_te = head5.transform(Xte)["angles"]
    ok(f"test 범위 이탈(누수 방지, clip 없음): outside [0,π] = {frac['frac_outside']*100:.2f}% "
       f"(below0={frac['frac_below_0']*100:.2f}%, abovePi={frac['frac_above_pi']*100:.2f}%); "
       f"test 각도 range=[{ang_te.min():.3f}, {ang_te.max():.3f}]")

    # ── (6) binary path: pm1_features ±1 만, median 분리 확인 ──
    pm1 = head5.transform(Xtr)["pm1_features"]
    assert set(np.unique(pm1)).issubset({-1.0, 1.0}), np.unique(pm1)
    assert pm1.shape == Xtr.shape
    ok(f"binary path: pm1_features ∈ {{-1,+1}}, shape {pm1.shape} (median train fit)")

    # ── 보고 markdown ────────────────────────────────────────
    report = [
        "## encoding_head self-test 결과",
        "",
        "**변환 방식(단일):** 컬럼별 min-max 선형 매핑 → `angle = (x-min)/(max-min)·π`, 범위 [0, π].",
        "party 별 따로 fit, train 에서만 fit, 상수 컬럼은 0, 범위 이탈 clip 없음.",
        "",
        "### 검증 항목",
        *lines,
        "",
        "### path 구분",
        "- **binary (검증됨):** 연속 각도 + median 이진화 pm1_features(±1) 반환. "
        "Walsh-Fourier M_ij 입력 전제를 만족 → Phase 2 기본 경로.",
        "- **continuous (미검증 확장):** 연속 각도만 반환(pm1_features=None). "
        "M_ij 는 binary feature 가정이라 continuous 입력에는 estimator 교체가 필요한 미검증 경로.",
    ]
    print("\n".join(report))


if __name__ == "__main__":
    _run_self_test()
