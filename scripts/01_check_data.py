# STEP 1. 데이터 확인
#   - 두 CSV 로드, shape/라벨 분포/색 분포/보조통계 확인
#   - 색→각도 매핑(범위·격자) 검증 + 밸런스 출력
#   - 색 분포 / 라벨 분포 막대그래프 저장(그림 내 텍스트 영어)

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# scripts/ 에서 src/ 를 패키지로 import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config, data

config.apply_style()
np.random.seed(config.SEED)


def main():
    summaries = {}
    print("=" * 64)
    print("STEP 1. 데이터 확인 (color→angle 매핑 검증, 밸런스)")
    print("=" * 64)

    for name in config.DATASETS:
        ds = data.load_dataset(name)
        s = data.summarize(ds)
        summaries[name] = (ds, s)

        print(f"\n── {name}  ({config.DATASETS[name]}) " + "─" * 18)
        print(f"  shape           : ({s['n']}, {s['n_features']})  feature + aux + label")
        print(f"  angles in [0,2π): {s['angles_in_range']}   on 4-grid: {s['angles_on_grid']}")
        print(f"  허용 각도       : {np.round(config.VALID_ANGLES, 4)}  (= 색 0,1,2,3)")
        print(f"  라벨 분포       : {s['label_dist']}   (+1 비율 {s['pos_frac']:.3f})")
        print(f"  색 분포(0..3)   : {s['color_dist']}")
        print(f"  보조통계 col8   : {s['aux_dist']}")
        # 밸런스 경고
        bal = "OK (≈50/50)" if abs(s['pos_frac'] - 0.5) < 0.1 else "주의: 불균형"
        print(f"  밸런스          : {bal}")

    # ── 그림: 색 분포 + 라벨 분포 ────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(config.W_DOUBLE * 0.75, 90 * config.MM))
    for col, name in enumerate(config.DATASETS):
        ds, s = summaries[name]
        # (상) 색 분포
        ax = axes[0, col]
        cidx = sorted(s["color_dist"])
        ax.bar(cidx, [s["color_dist"][c] for c in cidx],
               color=config.OKABE_ITO[0], width=0.7)
        ax.set_title(f"{name}: color distribution", fontsize=6.4)
        ax.set_xlabel("Color index (0,1,2,3)")
        ax.set_ylabel("Count (over all features)")
        ax.set_xticks(cidx)
        # (하) 라벨 분포
        ax = axes[1, col]
        labs = sorted(s["label_dist"])
        ax.bar([str(int(l)) for l in labs], [s["label_dist"][l] for l in labs],
               color=config.OKABE_ITO[1], width=0.6)
        ax.axhline(s["n"] / 2, color="0.6", lw=0.5, ls="--")
        ax.set_title(f"{name}: label balance", fontsize=6.4)
        ax.set_xlabel("Label")
        ax.set_ylabel("Count")

    fig.suptitle("STEP 1. Dataset check: color & label balance", fontsize=7.0)
    fig.tight_layout()
    out_png = os.path.join(config.OUTPUT_DIR, "01_data_check.png")
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_png.replace(".png", ".pdf"))
    plt.close(fig)
    print(f"\nsaved: {out_png}")


if __name__ == "__main__":
    main()
