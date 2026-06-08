# Automotive Ethernet IDS

차량용 이더넷 네트워크에서 발생하는 사이버 공격을 탐지·분류하는 딥러닝 기반 침입 탐지 시스템(IDS)입니다.  
PCAP에서 생성한 웨이블릿 이미지를 입력으로 받아 **2단계 게이트 파이프라인**으로 공격 유무와 유형을 분류합니다.

---

## 탐지 대상 클래스

| 레이블 | 이름 | 설명 |
|--------|------|------|
| 0 | Normal | 정상 트래픽 |
| 1 | F\_I | Flooding/Injection |
| 2 | P\_I | Packet Injection |
| 3 | M\_F | MAC Flooding |
| 4 | C\_D | Content Disruption |
| 5 | C\_R | Content Replay |
| 6 | Unknown | 낮은 confidence 미지 공격 후보 (추론 전용) |

---

## 파이프라인 구조

```
입력 이미지
    │
    ▼
[Stage 1: CAE 이상 탐지]
    MSE ≤ τ  ──────────────→  Normal (종료)
    MSE > τ
    │
    ▼
[Stage 2: DCNN 6-class 분류]
    max_prob ≥ thr  ────────→  F_I / P_I / M_F / C_D / C_R
    max_prob < thr  ────────→  Unknown (미탐 제로데이)
```

| 단계 | 모델 | 역할 |
|------|------|------|
| Phase 1 | DCNN (2-class) | 이진 분류 베이스라인 (Normal vs Attack) |
| Phase 2 | DCNN (6-class) | 공격 유형 분류, 클래스 불균형 보정 |
| Phase 3 | CAE + TwoStagePipeline | 비지도 이상 탐지 게이트 + 엔드투엔드 파이프라인 |
| Phase 4 | LOAO 평가 | Leave-One-Attack-Out 제로데이 평가 (stub) |

---

## 프로젝트 구조

```
.
├── configs/
│   ├── cae.yaml          # CAE 하이퍼파라미터
│   ├── experiment.yaml   # 데이터 경로, use_cae 스위치 등
│   ├── model.yaml        # DCNN 구조 설정
│   └── train.yaml        # 학습률, 배치 크기, seeds 등
├── data/
│   ├── raw/              # 원본 PCAP (git 제외 — 별도 공유)
│   └── processed/        # dataset_train.npz, dataset_test.npz, split_manifest.json (git 제외)
├── experiments/
│   └── leave_one_out.py  # Phase 4 LOAO 평가 (stub)
├── results/
│   ├── checkpoints/      # cae_best.pth, s2_seed_*_best.pth (git 제외)
│   ├── figures/          # 혼동 행렬, MSE 히스토그램, ROC 곡선
│   └── tables/           # CSV 결과 테이블, tau_values.json
├── scripts/
│   └── make_stub_dataset.py  # 실제 데이터 없이 스모크 테스트용 stub 생성
├── src/
│   ├── models/
│   │   ├── dcnn.py       # TOW-IDS DCNN (SepConv 기반)
│   │   └── cae.py        # Convolutional Autoencoder
│   ├── pipeline/
│   │   └── two_stage.py  # TwoStagePipeline (CAE + S2 연결)
│   ├── train/
│   │   ├── train_s1.py   # Phase 1: 이진 분류 학습
│   │   ├── train_s2.py   # Phase 2: 6-class 분류 학습
│   │   └── train_cae.py  # Phase 3: CAE 학습 + tau 계산
│   └── utils/
│       ├── io.py          # 데이터 로드/저장
│       ├── metrics.py     # 평가 지표
│       ├── seed.py        # 재현성 seed 고정
│       └── split.py       # train/val/test 분할 manifest 생성
├── tests/
│   └── test_two_stage.py  # confidence threshold 보정 회귀 테스트
├── requirements.txt
└── spec_phase0_to_3.docx  # 전체 설계 스펙 문서
```

---

## 환경 세팅

Python **3.10 이상**을 사용합니다.

```bash
# 1. 가상환경 생성 및 활성화
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. 의존성 설치
pip install -r requirements.txt
```

---

## 데이터 준비

PCAP 및 전처리 데이터는 용량 문제로 git에 포함되지 않습니다.  
팀원에게 별도로 받은 뒤 아래 경로에 배치합니다.

```
data/processed/
├── dataset_train.npz   # 학습/val용 (논문 원본 split)
└── dataset_test.npz    # 평가 전용 frozen test — 절대 학습에 사용 금지
```

NPZ 스키마:

- `X`: `float32`, `(N, 3, H, W)`, 값 범위 `[0, 1]`
- `y`: `int64`, `(N,)`, 레이블 `{0, 1, 2, 3, 4, 5}`
- `meta_json`: JSON bytes (메타데이터, 선택)
- `pcap_id`: `int64`, `(N,)` (선택, 시간 누수 방지용)

---

## 실행 방법

> 모든 명령어는 **프로젝트 루트**(`MS/`)에서 실행합니다.

### 0. 데이터 분할 (최초 1회)

```bash
python -m src.utils.split \
  --train-npz data/processed/dataset_train.npz \
  --test-npz  data/processed/dataset_test.npz
```

- `dataset_train.npz` → train 90% / val 10% (stratified, seed=42)
- `dataset_test.npz` → frozen test 전체 (분할 없음)

생성 결과: `data/processed/split_manifest.json`, `data/processed/normal_only_idx.npy`

### 1. Phase 1 — 이진 분류 학습 (Normal vs Attack)

```bash
python -m src.train.train_s1
```

결과: `results/tables/s1_baseline.csv`

### 2. Phase 2 — 6-class 공격 분류 학습

```bash
python -m src.train.train_s2
```

결과:

- `results/tables/s2_summary.csv`, `results/tables/s2_per_class.csv`
- `results/figures/cm_s2_norm.png`, `results/figures/cm_s2_raw.png`
- `results/checkpoints/s2_seed_<seed>_best.pth`

### 3. Phase 3 — CAE 학습 및 파이프라인 구성

```bash
python -m src.train.train_cae
```

결과:

- `results/checkpoints/cae_best.pth`
- `results/tables/tau_values.json`, `results/tables/tau_sensitivity.csv`
- `results/figures/mse_histogram.png`, `results/figures/roc_cae.png`

### 4. 학습된 체크포인트로 파이프라인 추론

```python
import torch
from src.pipeline.two_stage import TwoStagePipeline

device = torch.device('mps' if torch.backends.mps.is_available() else
                       'cuda' if torch.cuda.is_available() else 'cpu')

pipeline = TwoStagePipeline.from_checkpoints(
    cae_ckpt_path='results/checkpoints/cae_best.pth',
    s2_model='results/checkpoints/s2_seed_0_best.pth',
    tau_json_path='results/tables/tau_values.json',
    conf_thr=0.5,
    use_cae=True,
    device=device,
)
```

---

## 스모크 테스트 (실제 데이터 없을 때)

```bash
# 1. stub 데이터 생성 (N=200, 랜덤 이미지)
python scripts/make_stub_dataset.py

# 2. split manifest 생성
python -m src.utils.split \
  --train-npz data/processed/dataset_train.npz \
  --test-npz  data/processed/dataset_test.npz

# 3. 각 학습 스크립트 1-epoch 테스트
python -m src.train.train_s1 --smoke
python -m src.train.train_s2 --smoke
python -m src.train.train_cae --smoke

# 4. 파이프라인 회귀 테스트
python -m unittest discover -s tests -v
```

---

## 실험 결과 (논문 원본 split 기준)

데이터: `dataset_train.npz` 18,808개 / `dataset_test.npz` 12,368개 (frozen)

### Phase 1 — S1 Binary DCNN (5 seeds)

| 지표 | mean ± std |
|------|-----------|
| Accuracy | 0.9120 ± 0.0404 |
| F1 (binary) | 0.8923 ± 0.0515 |
| FPR | 0.0219 ± 0.0127 |
| ROC-AUC | 0.9334 ± 0.0505 |

### Phase 2 — S2 6-class DCNN (5 seeds)

| 지표 | mean ± std |
|------|-----------|
| Accuracy | 0.8886 ± 0.0284 |
| Macro-F1 | 0.8384 ± 0.0563 |
| C\_R Recall | 0.9727 ± 0.0165 |

클래스별 Recall (낮은 순): C_D 0.353 → F_I 0.898 → Normal 0.944 → C_R 0.973 → M_F 0.987 → P_I 0.998

### Phase 3 — CAE (seed=42, val set)

| 지표 | 값 |
|------|---|
| ROC-AUC | 0.9587 |
| tau\_2σ | 0.003183 |
| Normal FPR @ tau\_2σ | 3.96% |
| Attack TPR @ tau\_2σ | 74.4% |

---

## 주요 설정 파일

### `configs/experiment.yaml`

```yaml
experiment:
  train_npz_path: data/processed/dataset_train.npz   # 학습/val 분할 원본
  test_npz_path:  data/processed/dataset_test.npz    # frozen test (절대 학습에 사용 금지)
  manifest_path:  data/processed/split_manifest.json
  use_cae: false   # true → Stage 1 CAE 게이트 활성화 / false → S2 단독 모드
  conf_thr: 0.5
  output_dir: results/
```

### `configs/train.yaml`

```yaml
train:
  lr: 1.0e-3
  batch_size: 32
  epochs: 100
  patience: 5
  seeds: [0, 1, 2, 3, 4]
```

### `configs/cae.yaml`

```yaml
cae:
  latent_dim: 128
  lr: 1.0e-3
  batch_size: 64
  epochs: 150
  patience: 12
  noise_std: 0.05   # denoising CAE
```

---

## 재현성

- S1 / S2: `configs/train.yaml`의 `seeds` 리스트 전체 순회 후 mean±std 리포트
- 모든 단계: 동일한 `split_manifest.json`의 frozen test set 사용
- CAE: `seed=42` 고정 (한 번만 학습)

---

## 주의사항

- `data/processed/split_manifest.json`은 **절대 수정하지 마세요.**  
  S1·S2·S3가 동일한 `test_idx`를 공유해야 공정한 비교가 됩니다.
- `dataset_test.npz`는 최종 평가 전까지 학습에 사용하지 않습니다.
- `configs/experiment.yaml`의 `use_cae: false`이면 S2-only 모드로 동작합니다.

---

## References

1. M. L. Han et al., ["TOW-IDS"](https://doi.org/10.1109/TIFS.2022.3221893), *IEEE TIFS*, 2023.
2. L. F. Marques da Luz et al., ["Multi-stage Deep Learning-based IDS for Automotive Ethernet"](https://doi.org/10.1016/j.adhoc.2024.103548), *Ad Hoc Networks*, 2024.
3. S. Jeong et al., ["AERO"](https://doi.org/10.1109/TII.2023.3324949), *IEEE TII*, 2024.
4. M. S. G. A. Leandro et al., ["SeqWatch"](https://doi.org/10.5753/sbrc.2025.5949), *SBRC*, 2025.
5. F. Chollet, ["Xception"](https://openaccess.thecvf.com/content_cvpr_2017/html/Chollet_Xception_Deep_Learning_CVPR_2017_paper.html), *CVPR*, 2017.
6. P. Vincent et al., ["Denoising Autoencoders"](https://doi.org/10.1145/1390156.1390294), *ICML*, 2008.
7. D. Hendrycks and K. Gimpel, ["OOD Baseline"](https://openreview.net/forum?id=Hkg4TI9xl), *ICLR*, 2017.

## License

현재 이 저장소에는 별도의 오픈소스 라이선스가 적용되어 있지 않습니다.
