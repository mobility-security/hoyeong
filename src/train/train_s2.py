"""
Phase 2: 6-class 공격 유형 분류 DCNN (클래스 가중치 균형 손실 적용).
조기 종료 기준: val_macro_f1 최대화.

사용법:
  python -m src.train.train_s2           # 전체 실행: 100 epoch × 5 seed
  python -m src.train.train_s2 --smoke   # 스모크 테스트: 1 epoch, seed 0만
"""
import argparse
import copy
import os
import sys

import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score,
)
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.models.dcnn import DCNN
from src.utils.io import load_dataset
from src.utils.seed import set_seed

CLASS_NAMES = ['Normal', 'F_I', 'P_I', 'M_F', 'C_D', 'C_R']
NUM_CLASSES  = 6


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_loader(X: np.ndarray, y: np.ndarray, batch_size: int,
                shuffle: bool = False) -> DataLoader:
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


class EarlyStopping:
    """min(손실) 및 max(F1) 모드를 모두 지원하는 조기 종료."""

    def __init__(self, patience: int = 5, mode: str = 'max'):
        assert mode in ('min', 'max')
        self.patience = patience
        self.mode = mode
        self.best = float('-inf') if mode == 'max' else float('inf')
        self.counter = 0
        self.best_weights = None

    def _improved(self, score: float) -> bool:
        return score > self.best if self.mode == 'max' else score < self.best

    def step(self, score: float, model: nn.Module) -> bool:
        """학습을 중단해야 할 때 True 반환."""
        if self._improved(score):
            self.best = score
            self.counter = 0
            self.best_weights = copy.deepcopy(model.state_dict())
        else:
            self.counter += 1
        return self.counter >= self.patience

    def restore_best(self, model: nn.Module) -> None:
        if self.best_weights is not None:
            model.load_state_dict(self.best_weights)


def _infer(model: nn.Module, loader: DataLoader,
           device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(y_true, y_pred, y_prob_전체클래스) 반환."""
    model.eval()
    trues, preds, probs = [], [], []
    with torch.no_grad():
        for xb, yb in loader:
            logits = model(xb.to(device))
            p = F.softmax(logits, dim=1).cpu().numpy()
            probs.extend(p)
            preds.extend(logits.argmax(dim=1).cpu().numpy())
            trues.extend(yb.numpy())
    return (np.array(trues, dtype=np.int64),
            np.array(preds, dtype=np.int64),
            np.array(probs, dtype=np.float32))


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def _save_confusion_matrices(y_true: np.ndarray, y_pred: np.ndarray,
                              seed: int, out_dir: str) -> None:
    labels = list(range(NUM_CLASSES))

    cm_raw  = confusion_matrix(y_true, y_pred, labels=labels)
    # normalize='true': each row = recall per class (avoid division by zero)
    row_sums = cm_raw.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_norm = cm_raw / row_sums

    os.makedirs(out_dir, exist_ok=True)

    for cm, tag, fmt in [(cm_norm, 'norm', '.3f'), (cm_raw, 'raw', 'd')]:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            cm, annot=True, fmt=fmt, cmap='Blues', ax=ax,
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
        )
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title(f'S2 Confusion Matrix ({tag}) — seed={seed}')
        plt.tight_layout()
        path = os.path.join(out_dir, f'cm_s2_{tag}.png')
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f'  Saved: {path}')


# ---------------------------------------------------------------------------
# Per-seed training
# ---------------------------------------------------------------------------

def train_one_seed(
    seed: int,
    X: np.ndarray,
    y: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    manifest: dict,
    cfg_train,
    device: torch.device,
    out_dir: str,
    max_epochs: int | None = None,
    dropout: float = 0.5,
) -> dict:
    set_seed(seed)

    epochs   = max_epochs if max_epochs is not None else int(cfg_train.epochs)
    lr       = float(cfg_train.lr)
    bs       = int(cfg_train.batch_size)
    patience = int(cfg_train.patience)

    train_idx = np.array(manifest['train_idx'])
    val_idx   = np.array(manifest['val_idx'])

    # 클래스 가중치는 train split만 사용 (val/test 정보 유출 방지)
    y_train = y[train_idx]
    classes = np.arange(NUM_CLASSES)
    weights = compute_class_weight(class_weight='balanced',
                                   classes=classes, y=y_train)
    class_weights = torch.FloatTensor(weights).to(device)

    train_loader = make_loader(X[train_idx], y_train,    bs, shuffle=True)
    val_loader   = make_loader(X[val_idx],   y[val_idx], bs)
    test_loader  = make_loader(X_test,       y_test,     bs)

    model     = DCNN(num_classes=NUM_CLASSES, dropout=dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn   = nn.CrossEntropyLoss(weight=class_weights)
    stopper   = EarlyStopping(patience=patience, mode='max')

    for epoch in range(1, epochs + 1):
        # --- train ---
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss_fn(model(xb), yb).backward()
            optimizer.step()

        # --- val macro-F1 ---
        y_vt, y_vp, _ = _infer(model, val_loader, device)
        val_macro_f1 = f1_score(y_vt, y_vp, average='macro', zero_division=0)
        print(f'  [seed={seed}] epoch={epoch}/{epochs}  '
              f'val_macro_f1={val_macro_f1:.4f}'
              f'  (best={stopper.best:.4f}, patience={stopper.counter}/{patience})')

        if stopper.step(val_macro_f1, model):
            print(f'  EarlyStopping at epoch {epoch}, restoring best weights.')
            break

    stopper.restore_best(model)

    # 결합 파이프라인에서 동일한 best weights를 재사용할 수 있도록 seed별 저장.
    ckpt_dir = os.path.join(out_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f's2_seed_{seed}_best.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'num_classes': NUM_CLASSES,
        'dropout': float(dropout),
        'seed': seed,
        'val_macro_f1': float(stopper.best),
        'input_shape': tuple(X.shape[1:]),
    }, ckpt_path)
    print(f'  Saved checkpoint: {ckpt_path}')

    # --- test evaluation ---
    y_true, y_pred, _ = _infer(model, test_loader, device)

    report = classification_report(
        y_true, y_pred,
        target_names=CLASS_NAMES,
        labels=list(range(NUM_CLASSES)),
        digits=4,
        output_dict=True,
        zero_division=0,
    )
    macro_f1    = report['macro avg']['f1-score']
    weighted_f1 = report['weighted avg']['f1-score']
    accuracy    = report['accuracy']

    print(f'  [seed={seed}] test → acc={accuracy:.4f}  '
          f'macro_f1={macro_f1:.4f}  weighted_f1={weighted_f1:.4f}')

    # 혼동 행렬 PNG 저장 (마지막 seed가 덮어쓰는 것은 의도된 동작)
    _save_confusion_matrices(y_true, y_pred, seed,
                             os.path.join(out_dir, 'figures'))

    # Per-class rows
    per_class_rows = []
    for name in CLASS_NAMES:
        if name in report:
            per_class_rows.append({
                'seed':      seed,
                'class':     name,
                'precision': report[name]['precision'],
                'recall':    report[name]['recall'],
                'f1':        report[name]['f1-score'],
                'support':   report[name]['support'],
            })

    return {
        'seed':          seed,
        'accuracy':      accuracy,
        'macro_f1':      macro_f1,
        'weighted_f1':   weighted_f1,
        'per_class':     per_class_rows,
        'val_macro_f1_best': stopper.best,
        'checkpoint_path': ckpt_path,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true',
                        help='1 epoch, seed 0 only')
    args = parser.parse_args()

    cfg_model = OmegaConf.load('configs/model.yaml').model
    cfg_train = OmegaConf.load('configs/train.yaml').train
    cfg_exp   = OmegaConf.load('configs/experiment.yaml').experiment

    assert int(cfg_model.num_classes) == NUM_CLASSES, \
        f'configs/model.yaml num_classes must be {NUM_CLASSES}, got {cfg_model.num_classes}'

    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f'Using device: {device}')

    X,      y,      _ = load_dataset(cfg_exp.train_npz_path)
    X_test, y_test, _ = load_dataset(cfg_exp.test_npz_path)
    print(f'Train: X={X.shape}  labels={sorted(set(y.tolist()))}')
    print(f'Test:  X={X_test.shape}')

    with open(cfg_exp.manifest_path) as f:
        manifest = json.load(f)
    print(f'Manifest: train={len(manifest["train_idx"])}, '
          f'val={len(manifest["val_idx"])}, test(frozen)={len(manifest["test_idx"])}')

    seeds      = [0] if args.smoke else list(cfg_train.seeds)
    max_epochs = 1   if args.smoke else None
    out_dir    = cfg_exp.output_dir.rstrip('/')

    summary_rows = []
    all_per_class = []
    for seed in seeds:
        print(f'\n=== seed={seed} ===')
        m = train_one_seed(seed, X, y, X_test, y_test, manifest, cfg_train,
                           device, out_dir, max_epochs,
                           dropout=float(cfg_model.dropout))
        summary_rows.append({
            'seed':             m['seed'],
            'accuracy':         m['accuracy'],
            'macro_f1':         m['macro_f1'],
            'weighted_f1':      m['weighted_f1'],
            'val_macro_f1_best': m['val_macro_f1_best'],
            'checkpoint_path':   m['checkpoint_path'],
        })
        all_per_class.extend(m['per_class'])

    # ---- summary CSV ----
    df_sum = pd.DataFrame(summary_rows)
    num_cols = ['accuracy', 'macro_f1', 'weighted_f1', 'val_macro_f1_best']
    mean_row = {c: df_sum[c].mean() for c in num_cols}
    std_row  = {c: df_sum[c].std()  for c in num_cols}
    mean_row['seed'] = 'mean'
    std_row['seed']  = 'std'
    df_sum = pd.concat([df_sum, pd.DataFrame([mean_row, std_row])], ignore_index=True)

    # ---- per-class CSV ----
    df_pc = pd.DataFrame(all_per_class)
    if not df_pc.empty:
        pc_num = ['precision', 'recall', 'f1', 'support']
        pc_mean = df_pc.groupby('class')[pc_num].mean().reset_index()
        pc_std  = df_pc.groupby('class')[pc_num].std().reset_index()
        pc_mean['seed'] = 'mean'
        pc_std['seed']  = 'std'
        df_pc = pd.concat([df_pc, pc_mean, pc_std], ignore_index=True)

    os.makedirs(os.path.join(out_dir, 'tables'), exist_ok=True)
    sum_path = os.path.join(out_dir, 'tables', 's2_summary.csv')
    pc_path  = os.path.join(out_dir, 'tables', 's2_per_class.csv')
    df_sum.to_csv(sum_path, index=False)
    df_pc.to_csv(pc_path,   index=False)

    print(f'\ns2_summary.csv  → {sum_path}')
    print(f's2_per_class.csv → {pc_path}')
    print(df_sum.to_string(index=False))


if __name__ == '__main__':
    main()
