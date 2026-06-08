"""
Phase 1 베이스라인: 이진 분류 DCNN (Normal vs Attack).

사용법:
  python -m src.train.train_s1           # 전체 실행: 100 epoch × 5 seed
  python -m src.train.train_s1 --smoke   # 스모크 테스트: 1 epoch, seed 0만
"""
import argparse
import copy
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, roc_auc_score,
)
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.models.dcnn import DCNN
from src.utils.io import load_dataset
from src.utils.seed import set_seed


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def binarize(y: np.ndarray) -> np.ndarray:
    """다중 클래스 레이블을 이진으로 변환: 0=Normal, 1=Attack."""
    return (y != 0).astype(np.int64)


class EarlyStopping:
    def __init__(self, patience: int = 5):
        self.patience = patience
        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None

    def step(self, val_loss: float, model: nn.Module) -> bool:
        """학습을 중단해야 할 때 True 반환."""
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
            self.best_weights = copy.deepcopy(model.state_dict())
        else:
            self.counter += 1
        return self.counter >= self.patience

    def restore_best(self, model: nn.Module) -> None:
        if self.best_weights is not None:
            model.load_state_dict(self.best_weights)


def make_loader(X: np.ndarray, y: np.ndarray, batch_size: int,
                shuffle: bool = False) -> DataLoader:
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def eval_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                 y_prob: np.ndarray) -> dict:
    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, zero_division=0)
    cm  = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = float('nan')
    return {'accuracy': acc, 'f1_binary': f1, 'fpr': fpr, 'fnr': fnr,
            'auc_roc': auc}


# ---------------------------------------------------------------------------
# Per-seed training
# ---------------------------------------------------------------------------

def train_one_seed(
    seed: int,
    X: np.ndarray,
    y_bin: np.ndarray,
    X_test: np.ndarray,
    y_test_bin: np.ndarray,
    manifest: dict,
    cfg_train,
    device: torch.device,
    max_epochs: int | None = None,
) -> dict:
    set_seed(seed)

    epochs   = max_epochs if max_epochs is not None else int(cfg_train.epochs)
    lr       = float(cfg_train.lr)
    bs       = int(cfg_train.batch_size)
    patience = int(cfg_train.patience)

    # train/val: dataset_train.npz 인덱스 / test: dataset_test.npz 전체
    idx_train = np.asarray(manifest['train_idx'], dtype=np.int64)
    idx_val   = np.asarray(manifest['val_idx'],   dtype=np.int64)

    train_loader = make_loader(X[idx_train],   y_bin[idx_train], bs, shuffle=True)
    val_loader   = make_loader(X[idx_val],     y_bin[idx_val],   bs)
    test_loader  = make_loader(X_test, y_test_bin, bs)

    model     = DCNN(num_classes=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn   = nn.CrossEntropyLoss()
    stopper   = EarlyStopping(patience=patience)

    val_loss_history = []

    for epoch in range(1, epochs + 1):
        # train
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss_fn(model(xb), yb).backward()
            optimizer.step()

        # validate
        model.eval()
        running_loss, n = 0.0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                running_loss += loss_fn(model(xb), yb).item() * len(xb)
                n += len(xb)
        val_loss = running_loss / n
        val_loss_history.append(val_loss)
        print(f'  [seed={seed}] epoch={epoch}/{epochs}  val_loss={val_loss:.4f}'
              f'  (best={stopper.best_loss:.4f}, patience={stopper.counter}/{patience})')

        if stopper.step(val_loss, model):
            print(f'  EarlyStopping: stopped at epoch {epoch}, restoring best weights.')
            break

    stopper.restore_best(model)

    # test
    model.eval()
    preds, probs, trues = [], [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            logits = model(xb.to(device))
            p = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            probs.extend(p)
            preds.extend(logits.argmax(dim=1).cpu().numpy())
            trues.extend(yb.numpy())

    metrics = eval_metrics(np.array(trues), np.array(preds), np.array(probs))
    metrics['seed'] = seed
    metrics['val_loss_final'] = val_loss_history[-1] if val_loss_history else float('nan')
    metrics['epochs_run'] = len(val_loss_history)
    print(f'  [seed={seed}] test → acc={metrics["accuracy"]:.4f}  '
          f'f1={metrics["f1_binary"]:.4f}  fpr={metrics["fpr"]:.4f}  '
          f'fnr={metrics["fnr"]:.4f}  auc={metrics["auc_roc"]:.4f}')
    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true',
                        help='Smoke test: 1 epoch, seed 0 only')
    args = parser.parse_args()

    cfg_train = OmegaConf.load('configs/train.yaml').train
    cfg_exp   = OmegaConf.load('configs/experiment.yaml').experiment

    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f'Using device: {device}')

    X,      y,      _ = load_dataset(cfg_exp.train_npz_path)
    X_test, y_test, _ = load_dataset(cfg_exp.test_npz_path)
    y_bin      = binarize(y)
    y_test_bin = binarize(y_test)
    unique, counts = np.unique(y_bin, return_counts=True)
    print(f'Train: X={X.shape}  binary={dict(zip(unique.tolist(), counts.tolist()))}')
    print(f'Test:  X={X_test.shape}')

    with open(cfg_exp.manifest_path) as f:
        manifest = json.load(f)
    print(f'Manifest: train={len(manifest["train_idx"])}, '
          f'val={len(manifest["val_idx"])}, test(frozen)={len(manifest["test_idx"])}')

    seeds      = [0] if args.smoke else list(cfg_train.seeds)
    max_epochs = 1   if args.smoke else None

    rows = []
    for seed in seeds:
        print(f'\n=== seed={seed} ===')
        m = train_one_seed(seed, X, y_bin, X_test, y_test_bin,
                           manifest, cfg_train, device, max_epochs)
        rows.append({
            'seed':          m['seed'],
            'accuracy':      m['accuracy'],
            'f1_binary':     m['f1_binary'],
            'fpr':           m['fpr'],
            'fnr':           m['fnr'],
            'auc_roc':       m['auc_roc'],
            'val_loss_final': m['val_loss_final'],
            'epochs_run':    m['epochs_run'],
        })

    df = pd.DataFrame(rows)
    numeric_cols = ['accuracy', 'f1_binary', 'fpr', 'fnr', 'auc_roc',
                    'val_loss_final', 'epochs_run']

    mean_row = {c: df[c].mean() for c in numeric_cols}
    std_row  = {c: df[c].std()  for c in numeric_cols}
    mean_row['seed'] = 'mean'
    std_row['seed']  = 'std'

    df = pd.concat(
        [df, pd.DataFrame([mean_row, std_row])],
        ignore_index=True,
    )

    out_dir = os.path.join(cfg_exp.output_dir.rstrip('/'), 'tables')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 's1_baseline.csv')
    df.to_csv(out_path, index=False)
    print(f'\ns1_baseline.csv → {out_path}')
    print(df.to_string(index=False))


if __name__ == '__main__':
    main()
