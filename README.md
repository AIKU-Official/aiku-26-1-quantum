# CPFP: Cross-Party Fourier Power for Distributed Quantum Machine Learning

📢 2026년 1학기 [AIKU](https://github.com/AIKU-Official) 활동으로 진행한 프로젝트입니다.

분산 양자컴퓨터에서 고전 데이터를 학습할 때, **어떤 큐비트 쌍을 얽어야 하는지**를 데이터로부터 진단·처방하는 방법 **CPFP(Cross-Party Fourier Power)** 를 제안하고 검증합니다.

## 소개

고전 데이터를 두 대의 QPU(A, B)에 4+4로 나누어 학습하면, 각 파티는 자기 데이터만 보기 때문에 두 파티에 걸친 상관(cross-correlation)을 잃습니다. 이 끊긴 결합을 메우는 자원이 사전 공유 얽힘(pre-shared Bell pair)입니다. 단, 얽힘은 비싸므로 "아무 데나" 얽으면 비효율적입니다. 본 연구는 **얽힘 없는 모델의 잔차(residual)를 분석하면 어느 큐비트 쌍을 얽어야 하는지 알아낼 수 있다**는 가설을 세우고, 이를 CPFP 파이프라인으로 구현·검증했습니다.

## 방법론

파티 간 양자 게이트 없이(LOCC 설정), 얽힘은 임베딩 전 사전 공유 Bell pair로만 주입합니다.

- **분산 QCNN (8큐비트, 4+4):** 각 파티가 독립 QCNN 학습. conv(RY + ring CRY) → conditional pooling(mid-circuit measurement 후 조건부 회전) → 생존 큐비트 readout → 고전 결합. 파티 간 표현력은 늘리지 않아 cross 구조는 얽힘 없이 풀 수 없음.
- **CPFP 진단·처방:** ① 얽힘 없는 baseline 학습 → ② residual `r = y − f0` 추출 → ③ 파티 간 cross-correlation 행렬 `C` 구성 → ④ rank-1(분리 가능 성분) 제거 후 SVD로 얽힘 수요가 큰 쌍을 처방. 분리 가능한 cross는 고전 결합기가 이미 표현하므로 제거해야 진짜 비분리(off-rank-1) cross 수요만 남음.

## 환경 설정

```bash
pip install -r requirements.txt
```

PennyLane (default.qubit, autograd backend), numpy, pandas, matplotlib.

## 사용 방법

`scripts/` 의 워커를 순서대로 실행하면 데이터 점검 → baseline 학습 → residual 진단 → ablation 비교가 재현됩니다. 결과 그림은 `figures/` 에 저장됩니다.

## 예시 결과

**1. 얽힘은 분리 불가능한 cross에서만 필요.** 데이터를 marginal / separable(rank-1) / offrank1(rank≥2)로 나누면, separable cross는 얽힘 없이도 정확도 1.00으로 풀리고(고전적으로 분해 가능), offrank1에서만 None 0.73 → Prescribed 0.89로 얽힘이 도움. 진단 단계에서 rank-1 제거가 정당함을 실증.

**2. 진단이 cross 구조를 정확히 짚음.** match2(cross)에서는 residual cross-demand 행렬의 대각 성분만 솟고(diag/off ≈ 55배), pair3(marginal)에서는 전면 0. cross가 없으면 거짓 양성 없이 0을 줌.

**3. "올바른 위치"가 성능을 결정.** 개수·강도·라우팅을 통제한 채 위치만 바꾼 Under vs Wrong에서 올바른 배치가 +0.123(p=0.03) 우월. 더 많이 얽은 Over는 오히려 하락. 또한 cross 데이터(match2)에서만 처방이 효과 있고 marginal(pair3)에서는 무의미.

> 핵심 수치 (3 seed, paired t-test): None 0.580 < Wrong 0.736 < Under 0.859 < **Prescribed 0.914** > Over 0.841. Prescribed > Wrong은 +0.178 (p=0.0071).

그림 (`figures/`):

- `02_capacity_scan_cond.png` — 로컬 모델은 cross(match2)를 못 풀고 marginal(pair3)만 풂
- `03_residual_diagnosis_cond.png` — residual cross-demand 행렬: match2는 대각, pair3는 0
- `04_ablation_cond.png` — None < Wrong < Under < Prescribed > Over
- `13_prescribe_fix_cond.png` — 처방 얽힘은 cross(match2)에만 효과, marginal(pair3)엔 무의미
- `10_separable_diag_cond.png`, `11_synthablate_cond.png` — 얽힘은 off-rank-1 cross에서만 필요
- `06_synthetic_rank_cond.png`, `12_match2_cross_svd.png` — residual 스펙트럼이 nonlocal rank 복원
- `distributed_qcnn_circuit.png` — 분산 QCNN 회로도

**한계.** 위 성공은 4색 데이터 × 4-level 각도 인코딩 × Z₄ 진단 basis가 정확히 정렬된 toy 설정에 의존하며(임의 데이터로의 일반화는 검증 중), Prescribed 우위에는 "올바른 feature pairing"과 "pooling 깊이 생존"이 함께 기여해 완전 분리는 미완입니다. 통계는 3 seed 기준입니다. **후속**으로 임의 구조(고차 상관·3파티)로의 일반화, 쌍별 얽힘 세기 처방, 실제 이미지 데이터 적용을 계획합니다.

> 참고: Y. Kim, K. Hwang, H. Kwon, and Y. Kim, "The power of entanglement in distributed quantum machine learning," arXiv:2605.03864, 2026.

## 팀원

- [박주현]() (팀장): 프로젝트 총괄, 가설 제안 및 이론 정립, 실험 설계 및 검증
- [장서현](): 이론 정식화, 수학적 프레임워크 구성, 파이프라인 도식화
- [박서연](): 데이터셋 설계 및 제작 (검증용 toy dataset 설계)
