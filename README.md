# Automotive Ethernet IDS

차량용 이더넷 네트워크에서 발생하는 사이버 공격을 탐지·분류하는 딥러닝 기반 침입 탐지 시스템(IDS)입니다.  
PCAP에서 생성한 웨이블릿 이미지를 입력으로 받아 **2단계 게이트 파이프라인**으로 공격 유무와 유형을 분류합니다.

> **현재 상태:** Phase 0~3의 모델, 학습, 데이터 분할 및 파이프라인 코드는 구현되어 있습니다. 현재 저장소의 `dataset_v0.npz`는 동작 확인용 stub이며, PCAP 전처리 코드와 실제 웨이블릿 NPZ는 팀원 전달 후 연결할 예정입니다. Phase 4 LOAO 평가는 아직 stub입니다.

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
| 6 | Unknown | 낮은 confidence로 분류된 미지 공격 후보 (추론 전용) |

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
│   ├── raw/              # 원본 PCAP + 레이블 CSV (git 제외 — 별도 공유)
│   └── processed/        # dataset_v0.npz, split_manifest.json (git 제외 — 별도 공유)
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
│       └── split.py       # 시간적 누수 방지 train/val/test 분할
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

PCAP 및 전처리 데이터는 용량 문제로 git에 포함되지 않습니다. 팀원에게 별도로 받은 뒤 아래 경로에 배치합니다.

```
data/
├── raw/
│   ├── Automotive_Ethernet_with_Attack_original_10_17_19_50_training.pcap
│   ├── Automotive_Ethernet_with_Attack_original_10_17_20_04_test.pcap
│   ├── y_train.csv
│   └── y_test.csv
└── processed/
    ├── dataset_v0.npz        # 현재는 N=200 랜덤 이미지 stub
    └── split_manifest.json   # 고정된 train/val/test 인덱스
```

학습 코드가 기대하는 NPZ 스키마는 다음과 같습니다.

- `X`: `float32`, `(N, 3, H, W)`, 값 범위 `[0, 1]`
- `y`: `int64`, `(N,)`, 레이블 `{0, 1, 2, 3, 4, 5}`
- `meta`: JSON 직렬화 가능한 메타데이터. 가능하면 시간 누수 방지를 위한 `pcap_id` 포함

실제 NPZ를 받으면 `configs/experiment.yaml`의 `npz_path`를 변경하고 split manifest를 새 데이터 기준으로 최초 1회 생성해야 합니다.

---

## 실행 방법

> 모든 명령어는 **프로젝트 루트**(`MS/`)에서 실행합니다.

### 0. 데이터 분할 (최초 1회)

학습 전에 split manifest를 먼저 생성합니다. `pcap_id`가 있으면 PCAP 단위로, 없으면 인덱스 순서와 guard gap을 사용해 분할합니다.

```bash
python -m src.utils.split
```

생성 결과: `data/processed/split_manifest.json`, `data/processed/normal_only_idx.npy`

### 1. Phase 1 — 이진 분류 학습 (Normal vs Attack)

```bash
python -m src.train.train_s1
```

S1도 S2/S3와 동일한 frozen manifest의 train/val/test 인덱스를 사용합니다.

결과: `results/tables/s1_baseline.csv`

### 2. Phase 2 — 6-class 공격 분류 학습

```bash
python -m src.train.train_s2
```

결과:

- `results/tables/s2_summary.csv`
- `results/tables/s2_per_class.csv`
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

### 4. 학습된 체크포인트로 파이프라인 구성

`TwoStagePipeline.from_checkpoints()`는 S2 모델 객체 또는 `train_s2.py`가 저장한 checkpoint 경로를 받을 수 있습니다.

```python
import torch

from src.pipeline.two_stage import TwoStagePipeline

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
pipeline = TwoStagePipeline.from_checkpoints(
    cae_ckpt_path='results/checkpoints/cae_best.pth',
    s2_model='results/checkpoints/s2_seed_0_best.pth',
    tau_json_path='results/tables/tau_values.json',
    conf_thr=0.5,
    use_cae=True,
    device=device,
)
```

`calibrate_conf_thr()`는 validation normal FPR 제한을 만족하는 후보 중 가장 높은 threshold를 선택합니다. 제한을 만족하는 후보가 없으면 normal FPR이 가장 낮은 값으로 fallback합니다.

---

## 스모크 테스트 (실제 데이터 없을 때)

실제 데이터 없이 코드가 정상 동작하는지 빠르게 확인합니다.

```bash
# 1. stub 데이터 생성 (N=200, 랜덤 이미지)
python scripts/make_stub_dataset.py

# 2. split manifest 생성
python -m src.utils.split

# 3. 각 학습 스크립트 1-epoch 테스트
python -m src.train.train_s1 --smoke
python -m src.train.train_s2 --smoke
python -m src.train.train_cae --smoke

# 4. 파이프라인 회귀 테스트
python -m unittest discover -s tests -v
```

---

## 주요 설정 파일

### `configs/experiment.yaml`

```yaml
experiment:
  npz_path: data/processed/dataset_v0.npz
  manifest_path: data/processed/split_manifest.json
  use_cae: false   # true → Stage 1 CAE 게이트 활성화 / false → S2 단독 모드
  conf_thr: 0.5    # Stage 2 Unknown 판단 임계값
  output_dir: results/
```

### `configs/train.yaml`

```yaml
train:
  lr: 1.0e-3
  batch_size: 32
  epochs: 100
  patience: 5
  seeds: [0, 1, 2, 3, 4]   # 5개 seed 평균으로 결과 리포트
```

### `configs/cae.yaml`

```yaml
cae:
  latent_dim: 128
  lr: 1.0e-3
  batch_size: 64
  epochs: 150
  patience: 12
  noise_std: 0.05   # 노이즈 제거 학습 (denoising CAE)
```

---

## 재현성

모든 학습은 seed를 고정해 재현 가능합니다.

- S1 / S2: `configs/train.yaml`의 `seeds` 리스트 전체를 순회하고 평균±표준편차로 리포트
- S1 / S2 / S3: 동일한 `split_manifest.json`의 frozen test set 사용
- S2: seed별 validation macro-F1 best checkpoint 저장
- CAE: `seed=42` 고정 (한 번만 학습 후 모든 fold에서 재사용)

---

## 주의사항

- `data/processed/split_manifest.json`은 **절대 수정하지 마세요.**  
  S1·S2·S3가 동일한 `test_idx`를 공유해야 공정한 비교가 됩니다.
- 실제 NPZ로 교체한 뒤에는 기존 stub manifest를 재사용하지 말고 실제 데이터 기준으로 새 manifest를 생성하세요.
- 데이터 분할은 시간적 누수 방지를 위해 `pcap_id` 기반 PCAP 블록 분할을 우선 사용하고, 메타데이터가 없을 때만 인덱스 순서 기반 fallback을 사용합니다.
- `configs/experiment.yaml`의 `use_cae`는 CAE 검증 결과에 따라 수동으로 결정합니다. `false`이면 S2-only 모드입니다.

## References

이 프로젝트의 구조와 평가 방법은 다음 연구를 참고했습니다.

1. M. L. Han, B. I. Kwak, and H. K. Kim, ["TOW-IDS: Intrusion Detection System Based on Three Overlapped Wavelets for Automotive Ethernet"](https://doi.org/10.1109/TIFS.2022.3221893), *IEEE Transactions on Information Forensics and Security*, 2023.
2. L. F. Marques da Luz, P. F. de Araujo-Filho, and D. R. Campelo, ["Multi-stage Deep Learning-based Intrusion Detection System for Automotive Ethernet Networks"](https://doi.org/10.1016/j.adhoc.2024.103548), *Ad Hoc Networks*, 2024.
3. S. Jeong, H. K. Kim, M. L. Han, and B. I. Kwak, ["AERO: Automotive Ethernet Real-Time Observer for Anomaly Detection in In-Vehicle Networks"](https://doi.org/10.1109/TII.2023.3324949), *IEEE Transactions on Industrial Informatics*, 2024.
4. M. S. G. A. Leandro et al., ["SeqWatch: Unsupervised Sequence-based Intrusion Detection System for Automotive Ethernet"](https://doi.org/10.5753/sbrc.2025.5949), *SBRC*, 2025.
5. F. Chollet, ["Xception: Deep Learning with Depthwise Separable Convolutions"](https://openaccess.thecvf.com/content_cvpr_2017/html/Chollet_Xception_Deep_Learning_CVPR_2017_paper.html), *CVPR*, 2017.
6. K. He et al., ["Deep Residual Learning for Image Recognition"](https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html), *CVPR*, 2016.
7. P. Vincent et al., ["Extracting and Composing Robust Features with Denoising Autoencoders"](https://doi.org/10.1145/1390156.1390294), *ICML*, 2008.
8. D. Hendrycks and K. Gimpel, ["A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks"](https://openreview.net/forum?id=Hkg4TI9xl), *ICLR*, 2017.

## Contact

버그 제보, 실행 문의 및 개선 제안은 [GitHub Issues](https://github.com/kwonhoyeong/automotive-ethernet-ids/issues)에 등록해 주세요.

## Acknowledgements

Automotive Ethernet IDS 연구 기반을 제공한 TOW-IDS 저자들과 데이터 전처리 및 평가를 함께 진행하는 프로젝트 팀원들에게 감사드립니다.

## License

현재 이 저장소에는 별도의 오픈소스 라이선스가 적용되어 있지 않습니다. 사용 또는 재배포가 필요한 경우 저장소 소유자에게 문의해 주세요.
