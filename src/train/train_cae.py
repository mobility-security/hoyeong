"""
Phase 3: CAE 학습 — 정상(Normal) 샘플만 사용, seed=42 고정.
CAE는 한 번만 학습하며 이후 모든 LOAO fold에서 재사용.

사용법:
  python -m src.train.train_cae           # 전체 실행 (150 epoch)
  python -m src.train.train_cae --smoke   # 스모크 테스트: 1 epoch
"""
import argparse
import copy
import json
import os
import sys
from datetime import datetime, timezone

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.models.cae import CAE
from src.utils.io import load_dataset
from src.utils.seed import set_seed

CLASS_NAMES = {0: 'Normal', 1: 'F_I', 2: 'P_I', 3: 'M_F', 4: 'C_D', 5: 'C_R'}
ATTACK_COLORS = ['#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _make_loader(X: np.ndarray, batch_size: int, shuffle: bool = False) -> DataLoader:
    ds = TensorDataset(torch.from_numpy(X))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def compute_mse_batch(model: CAE, X_arr: np.ndarray,
                      device: torch.device, batch_size: int = 64) -> np.ndarray:
    """이미지별 복원 오차(MSE) 계산: shape (N,) 반환."""
    model.eval()
    results = []
    for i in range(0, len(X_arr), batch_size):
        xb = torch.from_numpy(X_arr[i : i + batch_size]).to(device)
        with torch.no_grad():
            xhat = model(xb)
            mse  = ((xhat - xb) ** 2).mean(dim=(1, 2, 3))
        results.extend(mse.cpu().numpy().tolist())
    return np.array(results, dtype=np.float32)


# ---------------------------------------------------------------------------
# Tau computation
# ---------------------------------------------------------------------------

def compute_tau(mse_normal_val: np.ndarray) -> dict:
    """val-normal MSE로부터 tau 후보값 계산. val 전용 — train set 절대 미사용."""
    n = len(mse_normal_val)
    mu    = float(mse_normal_val.mean())
    sigma = float(mse_normal_val.std(ddof=1)) if n > 1 else 0.0
    log_m = np.log(mse_normal_val + 1e-8)
    return {
        'mu':          mu,
        'sigma':       sigma,
        'tau_2sigma':  mu + 2 * sigma,
        'tau_3sigma':  mu + 3 * sigma,
        'p95':         float(np.percentile(mse_normal_val, 95)),
        'p99':         float(np.percentile(mse_normal_val, 99)),
        'tau_log':     float(np.exp(log_m.mean() + 2 * log_m.std(ddof=1))) if n > 1
                       else float(mu + 2 * sigma),
        'headline_tau': 'tau_2sigma',
        'created_at':  datetime.now(timezone.utc).isoformat(),
    }


def compute_tau_sensitivity(mse_all: np.ndarray, y_all: np.ndarray,
                             taus: dict) -> pd.DataFrame:
    """tau 후보별 FPR / TPR / Youden's J 민감도 테이블 생성 (oracle 포함)."""
    y_binary    = (y_all != 0).astype(int)
    normal_mask = y_all == 0
    attack_mask = ~normal_mask

    def _row(name, tau_val, note):
        pred_anom = mse_all > tau_val
        n_fpr = float(pred_anom[normal_mask].mean()) if normal_mask.any() else 0.0
        a_tpr = float(pred_anom[attack_mask].mean()) if attack_mask.any() else 0.0
        return dict(tau_type=name, tau_val=float(tau_val),
                    normal_fpr=n_fpr, attack_tpr=a_tpr,
                    youden_j=a_tpr - n_fpr, note=note)

    rows = [
        _row('tau_2sigma', taus['tau_2sigma'], 'headline'),
        _row('tau_3sigma', taus['tau_3sigma'], '—'),
        _row('p95',        taus['p95'],        '—'),
        _row('p99',        taus['p99'],        '—'),
        _row('tau_log',    taus['tau_log'],    '—'),
    ]

    try:
        fp_arr, tp_arr, thresh_arr = roc_curve(y_binary, mse_all)
        best = int(np.argmax(tp_arr - fp_arr))
        rows.append(dict(tau_type='youden_j',
                         tau_val=float(thresh_arr[best]),
                         normal_fpr=float(fp_arr[best]),
                         attack_tpr=float(tp_arr[best]),
                         youden_j=float(tp_arr[best] - fp_arr[best]),
                         note='oracle 상한만'))
    except ValueError:
        pass

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_mse_histogram(mse_all: np.ndarray, y_all: np.ndarray,
                       taus: dict, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))

    mse_normal = mse_all[y_all == 0]
    if len(mse_normal):
        bins = max(5, min(50, len(mse_normal) // 2 + 1))
        ax.hist(mse_normal, bins=bins, alpha=0.6, label='Normal', color='steelblue')

    for idx, (cls_id, cls_name) in enumerate(
            {1: 'F_I', 2: 'P_I', 3: 'M_F', 4: 'C_D', 5: 'C_R'}.items()):
        mse_cls = mse_all[y_all == cls_id]
        if len(mse_cls):
            bins = max(5, min(50, len(mse_cls) // 2 + 1))
            ax.hist(mse_cls, bins=bins, alpha=0.4,
                    label=cls_name, color=ATTACK_COLORS[idx % len(ATTACK_COLORS)])

    ax.axvline(taus['tau_2sigma'], color='red',    ls='--', lw=1.5, label='tau_2σ')
    ax.axvline(taus['tau_3sigma'], color='orange', ls=':',  lw=1.5, label='tau_3σ')
    ax.set_xlabel('Reconstruction MSE')
    ax.set_ylabel('Count')
    ax.set_title('CAE Reconstruction Error — Normal vs Attacks (val set)')
    ax.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f'[OK] mse_histogram.png → {out_path}')


def plot_roc(mse_all: np.ndarray, y_all: np.ndarray, out_path: str) -> float:
    y_binary = (y_all != 0).astype(int)
    fig, ax  = plt.subplots(figsize=(6, 5))
    auc = float('nan')
    try:
        fpr_arr, tpr_arr, _ = roc_curve(y_binary, mse_all)
        auc = roc_auc_score(y_binary, mse_all)
        ax.plot(fpr_arr, tpr_arr, lw=2, label=f'CAE  AUC={auc:.3f}')
    except ValueError as e:
        ax.text(0.5, 0.5, f'ROC N/A:\n{e}', ha='center', va='center',
                transform=ax.transAxes, fontsize=9)
    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('CAE ROC Curve (val set, binary: Normal vs Attack)')
    ax.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f'[OK] roc_cae.png → {out_path}  AUC={auc:.4f}')
    return auc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true', help='1 epoch smoke test')
    args = parser.parse_args()

    cfg_cae = OmegaConf.load('configs/cae.yaml').cae
    cfg_exp = OmegaConf.load('configs/experiment.yaml').experiment
    set_seed(42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f'Using device: {device}')

    # ---- Load data + manifest ----
    # CAE는 train 데이터만 사용 (test는 절대 사용 금지)
    X, y, _ = load_dataset(cfg_exp.train_npz_path)
    with open(cfg_exp.manifest_path) as f:
        manifest = json.load(f)

    normal_train_idx = np.array(manifest['normal_train_idx'])
    normal_val_idx   = np.array(manifest['normal_val_idx'])
    val_idx          = np.array(manifest['val_idx'])

    print(f'Normal — train={len(normal_train_idx)}, val={len(normal_val_idx)}')
    assert len(normal_train_idx) > 0, 'No normal training samples!'

    X_normal_train = X[normal_train_idx]
    X_normal_val   = X[normal_val_idx] if len(normal_val_idx) else X_normal_train[:2]

    bs      = int(cfg_cae.batch_size)
    lr      = float(cfg_cae.lr)
    epochs  = 1 if args.smoke else int(cfg_cae.epochs)
    patience = int(cfg_cae.patience)

    train_loader = _make_loader(X_normal_train, bs, shuffle=True)
    val_loader   = _make_loader(X_normal_val,   bs, shuffle=False)

    # ---- Build model ----
    input_shape = tuple(X.shape[1:])
    model = CAE(input_shape=input_shape,
                latent_dim=int(cfg_cae.latent_dim),
                noise_std=float(cfg_cae.noise_std)).to(device)
    n_params = model.n_params()
    print(f'CAE params: {n_params:,}  '
          f'({"OK < 5 M" if n_params < 5_000_000 else "WARN > 5 M"})')
    assert n_params < 5_000_000, f'Model too large: {n_params}'

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min',
        patience=int(cfg_cae.lr_patience),
        factor=float(cfg_cae.lr_factor),
    )

    best_loss, best_weights, patience_ctr = float('inf'), None, 0

    # ---- Training loop ----
    for epoch in range(1, epochs + 1):
        model.train()
        tr_loss_acc, tr_n = 0.0, 0
        for (xb,) in train_loader:
            xb = xb.to(device)
            optimizer.zero_grad()
            xhat = model(xb, training=True)
            loss = ((xhat - xb) ** 2).mean(dim=(1, 2, 3)).mean()
            loss.backward()
            optimizer.step()
            tr_loss_acc += loss.item() * len(xb)
            tr_n        += len(xb)
        tr_loss = tr_loss_acc / tr_n

        model.eval()
        vl_loss_acc, vl_n = 0.0, 0
        with torch.no_grad():
            for (xb,) in val_loader:
                xb = xb.to(device)
                xhat = model(xb)
                vl_loss_acc += ((xhat - xb) ** 2).mean(dim=(1, 2, 3)).mean().item() * len(xb)
                vl_n        += len(xb)
        vl_loss = vl_loss_acc / max(vl_n, 1)
        scheduler.step(vl_loss)

        print(f'  epoch={epoch}/{epochs}  train_loss={tr_loss:.6f}'
              f'  val_loss={vl_loss:.6f}'
              f'  (best={best_loss:.6f}, patience={patience_ctr}/{patience})')

        if vl_loss < best_loss:
            best_loss    = vl_loss
            patience_ctr = 0
            best_weights = copy.deepcopy(model.state_dict())
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print(f'  EarlyStopping at epoch {epoch}.')
                break

    if best_weights is not None:
        model.load_state_dict(best_weights)

    # ---- Save model ----
    os.makedirs('results/checkpoints', exist_ok=True)
    ckpt_path = 'results/checkpoints/cae_best.pth'
    torch.save({'model_state_dict': model.state_dict(),
                'input_shape': input_shape,
                'latent_dim':  int(cfg_cae.latent_dim),
                'val_loss':    best_loss}, ckpt_path)
    print(f'[OK] cae_best.pth → {ckpt_path}')

    # ---- Tau 계산 (val-normal 전용 — 훈련 세트 사용 금지) ----
    mse_normal_val = compute_mse_batch(model, X_normal_val, device, bs)
    taus = compute_tau(mse_normal_val)

    tbl_dir = os.path.join(cfg_exp.output_dir.rstrip('/'), 'tables')
    fig_dir = os.path.join(cfg_exp.output_dir.rstrip('/'), 'figures')
    os.makedirs(tbl_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    tau_path = os.path.join(tbl_dir, 'tau_values.json')
    with open(tau_path, 'w') as f:
        json.dump(taus, f, indent=2)
    print(f'[OK] tau_values.json → {tau_path}')
    print(f'     mu={taus["mu"]:.6f}  sigma={taus["sigma"]:.6f}'
          f'  tau_2σ={taus["tau_2sigma"]:.6f}')

    # ---- 시각화 (val 전체: Normal + 공격 클래스 포함) ----
    mse_val = compute_mse_batch(model, X[val_idx], device, bs)
    y_val   = y[val_idx]

    plot_mse_histogram(mse_val, y_val, taus,
                       os.path.join(fig_dir, 'mse_histogram.png'))
    auc = plot_roc(mse_val, y_val,
                   os.path.join(fig_dir, 'roc_cae.png'))

    df_sens = compute_tau_sensitivity(mse_val, y_val, taus)
    sens_path = os.path.join(tbl_dir, 'tau_sensitivity.csv')
    df_sens.to_csv(sens_path, index=False)
    print(f'[OK] tau_sensitivity.csv → {sens_path}')
    print(df_sens.to_string(index=False))

    print(f'\n=== CAE Summary ===')
    print(f'  params={n_params:,}  best_val_loss={best_loss:.6f}  ROC-AUC={auc:.4f}')
    print(f'  tau_2σ={taus["tau_2sigma"]:.6f}  '
          f'tau_3σ={taus["tau_3sigma"]:.6f}  '
          f'p95={taus["p95"]:.6f}')


if __name__ == '__main__':
    main()
