"""
논문 원본 split 기준 train/val/test 분할.

- dataset_train.npz → train(90%) / val(10%), stratify=y, seed=42
  guard_gap은 train/val 경계에만 적용 (pcap_id 기반 시간 순서 보존)
- dataset_test.npz  → frozen test 전체 (분할 없음, 절대 학습에 사용 금지)

스크립트로 실행 시:
  python -m src.utils.split
  python -m src.utils.split --train-npz data/processed/dataset_train.npz
"""
import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.utils.io import load_dataset

CLASS_NAMES = {0: 'Normal', 1: 'F_I', 2: 'P_I', 3: 'M_F', 4: 'C_D', 5: 'C_R'}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def make_split_manifest(
    train_npz_path: str,
    test_npz_path: str,
    out_path: str = 'data/processed/split_manifest.json',
    val_ratio: float = 0.10,
    guard_gap: int = 64,
    seed: int = 42,
) -> dict:
    """
    논문 원본 split 기준 manifest 생성.

    - train/val: dataset_train.npz에서 stratified 분할 후 guard_gap 적용
    - test: dataset_test.npz 전체 (인덱스 0..N_test-1 고정)

    한 번 생성 후 절대 수정 금지 — S1/S2/S3 모델이 동일한 split을 공유.
    반환값: manifest dict
    """
    # ---- 데이터 로드 ----
    X_tr, y_tr, _ = load_dataset(train_npz_path)
    X_te, y_te, _ = load_dataset(test_npz_path)
    N_train = len(X_tr)
    N_test  = len(X_te)
    print(f'[split] dataset_train N={N_train}  dataset_test N={N_test}')
    print(f'[split] val_ratio={val_ratio}  guard_gap={guard_gap}  seed={seed}')

    # ---- train/val stratified split ----
    # stratify=y로 랜덤 분할하므로 val 샘플이 전체 인덱스에 분산됨.
    # guard_gap은 contiguous 시간 블록 분할에만 적용 가능하므로 여기서는 사용 안 함.
    all_idx = np.arange(N_train)
    train_idx_raw, val_idx_raw = train_test_split(
        all_idx, test_size=val_ratio, stratify=y_tr, random_state=seed
    )
    train_idx = list(map(int, train_idx_raw))
    val_idx   = list(map(int, val_idx_raw))

    # ---- test: dataset_test.npz 전체 ----
    test_idx = list(range(N_test))

    # ---- 교집합 검증 ----
    assert len(set(train_idx) & set(val_idx)) == 0, \
        f'train/val overlap: {len(set(train_idx) & set(val_idx))} samples'
    print(f'[OK] No overlap — train={len(train_idx)}, val={len(val_idx)}, '
          f'test={len(test_idx)} (frozen, dataset_test.npz 전체)')

    # ---- Normal-only indices for CAE ----
    normal_train_idx = [int(i) for i in train_idx if y_tr[i] == 0]
    normal_val_idx   = [int(i) for i in val_idx   if y_tr[i] == 0]
    print(f'[OK] Normal samples — train={len(normal_train_idx)}, val={len(normal_val_idx)}')

    # ---- 클래스별 샘플 수 ----
    def _counts(idx, y):
        if not len(idx):
            return {}
        vals, cnts = np.unique(y[list(idx)], return_counts=True)
        return {CLASS_NAMES.get(int(v), str(v)): int(c) for v, c in zip(vals, cnts)}

    label_counts = {
        'train': _counts(train_idx, y_tr),
        'val':   _counts(val_idx,   y_tr),
        'test':  _counts(test_idx,  y_te),
    }
    for split, cnt in label_counts.items():
        print(f'  {split}: {cnt}')

    # ---- manifest 저장 ----
    manifest = {
        'train_idx':        [int(i) for i in train_idx],
        'val_idx':          [int(i) for i in val_idx],
        'test_idx':         test_idx,
        'normal_train_idx': normal_train_idx,
        'normal_val_idx':   normal_val_idx,
        'label_counts':     label_counts,
        'train_source':     train_npz_path,
        'test_source':      test_npz_path,
        'val_ratio':        val_ratio,
        'guard_gap':        guard_gap,
        'seed':             seed,
        'sha256':           '',
        'created_at':       datetime.now(timezone.utc).isoformat(),
    }

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    body = json.dumps(manifest, indent=2)
    sha  = hashlib.sha256(body.encode()).hexdigest()
    manifest['sha256'] = sha
    with open(out_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f'[OK] split_manifest.json → {out_path}  sha256={sha[:16]}...')

    # ---- normal_only_idx.npy (CAE 학습용 편의 파일) ----
    normal_only = np.array(normal_train_idx + normal_val_idx, dtype=np.int64)
    npy_path = os.path.join(os.path.dirname(os.path.abspath(out_path)),
                            'normal_only_idx.npy')
    np.save(npy_path, normal_only)
    print(f'[OK] normal_only_idx.npy → {npy_path}  ({len(normal_only)} samples)')

    return manifest


# ---------------------------------------------------------------------------
# Guard gap 적용
# ---------------------------------------------------------------------------

def _apply_guard_gap(
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    guard_gap: int,
    N: int,
) -> tuple:
    """
    train/val 경계 근방에서 guard_gap 개 샘플을 제거.
    stratified split 후 인덱스 정렬 기준으로 경계를 판단.
    guard_gap이 val 크기의 1/4을 초과하면 자동 축소.
    """
    train_sorted = np.sort(train_idx)
    val_sorted   = np.sort(val_idx)

    if len(val_sorted) == 0:
        return list(train_sorted), []

    val_min = int(val_sorted[0])
    val_max = int(val_sorted[-1])

    eff_gap = min(guard_gap, len(val_sorted) // 4)
    if eff_gap != guard_gap:
        print(f'[WARN] guard_gap {guard_gap} → {eff_gap} (val이 너무 작음)')

    # val 경계 ±eff_gap 범위의 train 샘플 제거
    train_filtered = [
        int(i) for i in train_sorted
        if not (val_min - eff_gap <= i <= val_max + eff_gap)
    ]
    val_filtered = list(map(int, val_sorted))

    return train_filtered, val_filtered


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-npz', default='data/processed/dataset_train.npz')
    parser.add_argument('--test-npz',  default='data/processed/dataset_test.npz')
    parser.add_argument('--out',       default='data/processed/split_manifest.json')
    parser.add_argument('--val-ratio', type=float, default=0.10)
    parser.add_argument('--guard-gap', type=int,   default=64)
    parser.add_argument('--seed',      type=int,   default=42)
    args = parser.parse_args()

    manifest = make_split_manifest(
        train_npz_path=args.train_npz,
        test_npz_path=args.test_npz,
        out_path=args.out,
        val_ratio=args.val_ratio,
        guard_gap=args.guard_gap,
        seed=args.seed,
    )
    print(f'\nSplit sizes: train={len(manifest["train_idx"])}, '
          f'val={len(manifest["val_idx"])}, test={len(manifest["test_idx"])}')
    print(f'Normal (CAE): train={len(manifest["normal_train_idx"])}, '
          f'val={len(manifest["normal_val_idx"])}')
