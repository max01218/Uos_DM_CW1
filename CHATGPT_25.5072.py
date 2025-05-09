# -*- coding: utf-8 -*-
"""
Train a VARnet-style regressor on Dengue feature data and report MAE.

Usage (example):
    python dengue_varnet_mae.py \
        --train_csv dengue_features_train.csv \
        --val_split_year 2006 \
        --epochs 100 \
        --batch_size 32 \
        --seq_len 52

The script will:
  1. Load the training CSV.
  2. Fill missing values per–city with forward/backward fill.
  3. Robust‑scale (+ arcsinh) all feature columns.
  4. Train a VARnet‑Regressor (Conv → DWT → Conv → FEFT → Conv → FC).
  5. Report MAE on the hold‑out set (year ≥ val_split_year).
  6. Optionally (‑‑test_csv) produce a Kaggle‑style submission file.

Requires: pytorch, numpy, pandas, scikit‑learn, pywt, tqdm.
"""

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import RobustScaler
from torch.utils.data import DataLoader, Dataset
from tqdm import trange
import pywt
import matplotlib.pyplot as plt

# ---------- Dataset ---------- #
class DengueSeqDataset(Dataset):
    """Sliding-window weekly sequences → label (total_cases of last week)."""

    def __init__(self, df: pd.DataFrame, feature_cols: List[str], label_col: str = "total_cases", seq_len: int = 52,
                 scaler: RobustScaler = None, train: bool = True,
                 imputation_source_values: pd.Series = None):
        self.seq_len = seq_len
        self.train = train
        self.feature_cols = feature_cols
        self.label_col = label_col
        
        df_processed = df.copy()

        # fill NA inside each city block (ffill‑>bfill)
        for col_name in self.feature_cols:
            df_processed[col_name] = df_processed.groupby("city")[col_name].transform(lambda s: s.ffill().bfill())

        # Scaler and Imputation logic
        self.scaler = scaler or RobustScaler(quantile_range=(25, 75))
        # Check if the scaler instance passed is already fitted
        is_scaler_fitted = hasattr(self.scaler, "center_") and self.scaler.center_ is not None

        if self.train and not is_scaler_fitted: # This is the primary training data instance (e.g., train_ds)
            # Learn imputation values from this data and fit the scaler
            self.imputation_values = df_processed[self.feature_cols].median()
            df_processed[self.feature_cols] = df_processed[self.feature_cols].fillna(self.imputation_values)
            
            if df_processed[self.feature_cols].isnull().any().any():
                # This should not happen if median() worked and columns are numeric
                raise ValueError("NaNs found in training data after median imputation. Check feature columns for all-NaN groups or non-numeric data unable to compute median.")
            
            data_arcsinh = np.arcsinh(df_processed[self.feature_cols].to_numpy())
            self.scaler.fit(data_arcsinh)
            feats_scaled = self.scaler.transform(data_arcsinh).astype(np.float32)
        
        else: # Validation or test phase (scaler should be fitted and imputation_source_values provided)
            if not is_scaler_fitted:
                 raise ValueError("Scaler must be fitted for validation/test data if 'train' is False or scaler was pre-fitted.")
            if imputation_source_values is None:
                raise ValueError("Validation/Test data needs imputation_source_values from training.")
            
            self.imputation_values = imputation_source_values # Store for reference if needed
            df_processed[self.feature_cols] = df_processed[self.feature_cols].fillna(self.imputation_values)

            # Final check for NaNs, can happen if a whole column in train_ds.imputation_values was NaN (e.g. all-NaN feature in train)
            if df_processed[self.feature_cols].isnull().any().any():
                print("Warning: NaNs in validation/test data after imputation with training medians. Filling remaining with 0.")
                # If imputation_values itself had NaNs (e.g., a feature was all NaN in training), fill those specific columns with 0.
                for col in self.feature_cols:
                    if df_processed[col].isnull().any():
                        df_processed[col] = df_processed[col].fillna(0)
            
            data_arcsinh = np.arcsinh(df_processed[self.feature_cols].to_numpy())
            feats_scaled = self.scaler.transform(data_arcsinh).astype(np.float32)
        

        if self.train: # Original logic for loading labels, self.train indicates if labels should be loaded
            labels_data = df_processed[self.label_col].values
            # Ensure labels are numeric before casting, handle potential NaNs in labels if necessary
            if pd.api.types.is_numeric_dtype(labels_data):
                 self.labels = labels_data.astype(np.float32)
            else:
                 # Attempt conversion, raise error if it fails for non-NaNs
                 try:
                     self.labels = pd.to_numeric(labels_data, errors='raise').astype(np.float32)
                 except ValueError as e:
                     raise ValueError(f"Label column '{self.label_col}' contains non-numeric data that cannot be converted: {e}")


        # build (start_idx, end_idx) pairs for sliding windows
        self.windows = []
        for city in df.city.unique():
            city_idx = df[df.city == city].index.tolist()
            for i, end in enumerate(city_idx):
                start = max(city_idx[0], end - seq_len + 1)
                self.windows.append((start, end))

        self.feats_scaled = feats_scaled

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        start, end = self.windows[idx]
        seq = self.feats_scaled[start:end + 1]  # inclusive
        pad = self.seq_len - seq.shape[0]
        if pad > 0:  # pad at beginning (older weeks)
            seq = np.pad(seq, ((pad, 0), (0, 0)), mode="constant")
        if self.train:
            label = self.labels[end]
            return torch.from_numpy(seq), torch.tensor(label)
        return torch.from_numpy(seq)


# ---------- Model components ---------- #
class DWT1D(nn.Module):
    def __init__(self, wave: str = "bior2.2"):
        super().__init__()
        self.wave = pywt.Wavelet(wave)

    def forward(self, x):  # x input shape: [B, C_in, L_in]
        # pywt operates per-sample -> loop over batch
        out_batch = []
        # In the loop, sample_in_batch has shape [C_in, L_in]
        for sample_in_batch in x:
            # Perform DWT on each channel (C_in) of sample_in_batch
            # .cpu().detach().numpy() is essential for pywt
            coeffs_list_numpy = [pywt.dwt(sample_in_batch[c].cpu().detach().numpy(), self.wave) for c in range(sample_in_batch.shape[0])]
            
            # Separate approximation (a) and detail (d) coefficients.
            # Each c[0] (for approx) and c[1] (for detail) is a numpy array.
            # Convert them to tensors on the correct device with float32 dtype.
            a_tensors_per_channel = [torch.tensor(coeffs_np[0], device=x.device, dtype=torch.float32) for coeffs_np in coeffs_list_numpy]
            d_tensors_per_channel = [torch.tensor(coeffs_np[1], device=x.device, dtype=torch.float32) for coeffs_np in coeffs_list_numpy]

            # Stack lists of tensors to get shape [C_in, L_dwt]
            a_coeffs_stacked = torch.stack(a_tensors_per_channel) # Shape: [C_in, L_dwt]
            d_coeffs_stacked = torch.stack(d_tensors_per_channel) # Shape: [C_in, L_dwt]

            # Get the length after DWT (L_dwt)
            L_dwt = a_coeffs_stacked.shape[1]

            # Downsample the original sample_in_batch to match DWT output length L_dwt
            # sample_in_batch has shape [C_in, L_in]
            # For F.interpolate, input needs shape [Batch_for_interp, Channels_for_interp, Length_for_interp]
            # Here, C_in (number of features/channels) acts as the "batch" for interpolation,
            # and we add a dummy channel dimension (1).
            sample_for_interpolate = sample_in_batch.unsqueeze(1)  # Shape: [C_in, 1, L_in]
            sample_downsampled_interpolated = F.interpolate(sample_for_interpolate, size=L_dwt, mode='linear', align_corners=False)
            sample_downsampled = sample_downsampled_interpolated.squeeze(1)  # Shape: [C_in, L_dwt]

            # Concatenate the downsampled original sample, and the a and d coefficients along the channel dimension (dim=0 for features).
            # Each input tensor to cat (sample_downsampled, a_coeffs_stacked, d_coeffs_stacked) has shape [C_in, L_dwt].
            # After cat(dim=0), the shape becomes [3*C_in, L_dwt].
            concatenated_features = torch.cat([sample_downsampled, a_coeffs_stacked, d_coeffs_stacked], dim=0)
            out_batch.append(concatenated_features)
        
        # Stack the results from the batch dimension.
        # Each item in out_batch has shape [3*C_in, L_dwt].
        # After stack, final output shape: [B, 3*C_in, L_dwt].
        return torch.stack(out_batch)


class FEFT(nn.Module):
    def __init__(self, in_len: int, proj_len: int = 700):
        super().__init__()
        self.U = nn.Parameter(torch.randn(proj_len, in_len // 2 + 1, dtype=torch.float32) * 0.02)

    def forward(self, x):  # x shape [B, C_x, L_x] (e.g., B, 32, 28)
        f_fft = torch.fft.rfft(x, dim=-1, norm="ortho")  # f_fft shape [B, C_x, L_x//2+1] (complex)
        f_real = torch.view_as_real(f_fft)  # f_real shape [B, C_x, L_x//2+1, 2]
        
        # Reshape f_real to [B, C_x*2, L_x//2+1] for einsum
        # The 'k' dimension for einsum should be L_x//2+1, matching self.U's last dimension
        # The 'l' dimension for einsum will be C_x*2
        current_B, current_Cx, current_Lx_half_plus_1, two = f_real.shape
        
        f_permuted = f_real.permute(0, 1, 3, 2) # -> [B, C_x, 2, L_x//2+1]
        f_reshaped = f_permuted.reshape(current_B, current_Cx * two, current_Lx_half_plus_1) # -> [B, C_x*2, L_x//2+1]
        
        # self.U has shape [proj_len, L_x//2+1]
        # einsum("blk, mk -> bml", f_reshaped, self.U)
        # f_reshaped: b=B, l=C_x*2, k=L_x//2+1
        # self.U:     m=proj_len, k=L_x//2+1
        # k dimensions match.
        # proj output: b=B, m=proj_len, l=C_x*2
        proj = torch.einsum("blk, mk -> bml", f_reshaped, self.U)  # proj shape [B, proj_len, C_x*2]
        
        return proj.permute(0, 2, 1)  # permuted proj shape [B, C_x*2, proj_len]


class VARnetReg(nn.Module):
    def __init__(self, n_feats: int, seq_len: int):
        super().__init__()

        # Calculate the output length of DWT dynamically
        # This assumes DWT1D uses the default wave="bior2.2" and mode="symmetric"
        dwt_wave_name = "bior2.2" # Should match DWT1D's default or be a param
        wave_obj = pywt.Wavelet(dwt_wave_name)
        filter_len = wave_obj.dec_len
        # pywt.dwt default mode is 'symmetric'
        L_dwt = pywt.dwt_coeff_len(seq_len, filter_len, mode='symmetric')

        self.conv_pre = nn.Sequential(
            nn.Conv1d(n_feats, 64, 9, padding=4), nn.GELU(),
            nn.Conv1d(64, 64, 3, padding=1), nn.GELU(),
            nn.Conv1d(64, 64, 3, padding=1), nn.GELU(),
        )
        self.dwt = DWT1D() # Assumes wave="bior2.2"
        self.conv_mid = nn.Sequential(
            nn.Conv1d(64 * 3, 96, 3, padding=1), nn.GELU(),
            nn.Conv1d(96, 48, 3, padding=1), nn.GELU(),
            nn.Conv1d(48, 32, 3, padding=1), nn.GELU(),
        )
        self.feft = FEFT(in_len=L_dwt, proj_len=256) # Use dynamically calculated L_dwt
        self.conv_post = nn.Sequential(
            nn.Conv1d(32 * 2, 128, 3, padding=1), nn.GELU(),
            nn.Conv1d(128, 32, 3, padding=1), nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(32, 1))

    def forward(self, x):  # x [B, L, C]
        x = x.permute(0, 2, 1)  # [B, C, L]
        x = self.conv_pre(x)
        x = self.dwt(x)
        x = self.conv_mid(x)
        x = self.feft(x)
        x = self.conv_post(x)
        return self.head(x).squeeze(-1)


# ---------- Train helpers ---------- #

def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    crit = nn.L1Loss()
    for seqs, labels in loader:
        seqs, labels = seqs.to(device), labels.to(device)
        optimizer.zero_grad()
        preds = model(seqs)
        loss = crit(preds, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * seqs.size(0)
    return total_loss / len(loader.dataset)


def evaluate(model, loader, device):
    model.eval()
    preds_all, labels_all = [], []
    with torch.no_grad():
        for seqs, labels in loader:
            seqs = seqs.to(device)
            preds = model(seqs).cpu().numpy()
            preds_all.append(preds)
            labels_all.append(labels.numpy())
    preds_all = np.concatenate(preds_all)
    labels_all = np.concatenate(labels_all)
    mae = mean_absolute_error(labels_all, preds_all)
    return mae, preds_all, labels_all


# ---------- Main ---------- #

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using", device)

    train_df = pd.read_csv(args.train_csv)

    # 修正 feature_cols 定义，只包含数值列
    numeric_cols = train_df.select_dtypes(include=np.number).columns.tolist()
    feature_cols = [
        c for c in numeric_cols 
        if c not in ("total_cases", "year", "weekofyear") # 排除目标和原始时间列
    ]
    
    if args.val_split_year:
        train_part = train_df[train_df.year < args.val_split_year].reset_index(drop=True)
        val_part = train_df[train_df.year >= args.val_split_year].reset_index(drop=True)
    else:
        raise ValueError("Please specify --val_split_year to create hold-out set.")

    # Scaler instance to be used. train_ds will fit it.
    # val_ds and test_ds will use the instance fitted by train_ds.
    shared_scaler = RobustScaler(quantile_range=(25, 75))
    
    # For train_ds, train=True. It will learn imputation_values and fit shared_scaler.
    train_ds = DengueSeqDataset(train_part, feature_cols, label_col="total_cases", 
                                seq_len=args.seq_len, scaler=shared_scaler, train=True)
    
    # For val_ds, train=True (to load labels). It uses the fitted shared_scaler from train_ds
    # and the imputation_values learned by train_ds.
    val_ds = DengueSeqDataset(val_part, feature_cols, label_col="total_cases", 
                              seq_len=args.seq_len, scaler=train_ds.scaler, 
                              train=True, # train=True to load labels
                              imputation_source_values=train_ds.imputation_values)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = VARnetReg(n_feats=len(feature_cols), seq_len=args.seq_len).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)

    best_mae, best_state = float("inf"), None
    for epoch in trange(args.epochs, desc="Epoch"):
        tr_loss = train_epoch(model, train_loader, optimizer, device)
        mae, _, _ = evaluate(model, val_loader, device)
        if mae < best_mae:
            best_mae, best_state = mae, model.state_dict()
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:03d}: train_L1={tr_loss:.3f} | val_MAE={mae:.3f}")

    print(f"\nBest validation MAE: {best_mae:.3f}")
    model.load_state_dict(best_state)

    # Optionally predict test set
    if args.test_csv:
        test_df = pd.read_csv(args.test_csv)
        # For test_ds, train=False. It uses fitted scaler and imputation_values from train_ds.
        test_ds = DengueSeqDataset(test_df, feature_cols, label_col="total_cases", # label_col passed but not used if train=False
                                   seq_len=args.seq_len, scaler=train_ds.scaler, 
                                   train=False, # train=False to not load labels / not attempt to fit scaler
                                   imputation_source_values=train_ds.imputation_values)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
        model.eval()
        preds = []
        with torch.no_grad():
            for seqs in test_loader:
                seqs = seqs.to(device)
                preds.extend(model(seqs).cpu().numpy())
        submit = pd.read_csv(args.submission_fmt)
        submit["total_cases"] = np.round(np.clip(preds, 0, None)).astype(int)
        out_path = Path(args.out_csv)
        submit.to_csv(out_path, index=False)
        print("Saved Kaggle‑style predictions →", out_path)

        # --- Plotting Submission Predictions ---
        try:
            print(f"\nGenerating plot for submission file: {out_path}")
            submission_df = pd.read_csv(out_path)
            
            # Create a proper date column for plotting
            # Ensure weekofyear is zero-padded if necessary for consistent string formatting
            submission_df['weekofyear_str'] = submission_df['weekofyear'].astype(str).str.zfill(2)
            submission_df['date'] = pd.to_datetime(submission_df['year'].astype(str) + submission_df['weekofyear_str'] + '1', format='%Y%U%w')
            # %U for week number (Sunday as first day), %w for weekday (1 for Monday)
            # If your week starts on Monday and week number is ISO, use %G%V%u
            submission_df = submission_df.sort_values(by=['city', 'date']).reset_index(drop=True)

            cities = submission_df['city'].unique()
            n_cities = len(cities)
            
            fig, axes = plt.subplots(n_cities, 1, figsize=(15, 5 * n_cities), sharex=False)
            if n_cities == 1:
                axes = [axes] # Make it iterable if only one city

            for i, city in enumerate(cities):
                city_df = submission_df[submission_df['city'] == city]
                ax = axes[i]
                ax.plot(city_df['date'], city_df['total_cases'], label=f'Predicted Cases - {city.upper()}', marker='.')
                ax.set_xlabel('Time')
                ax.set_ylabel('Predicted Total Cases')
                ax.set_title(f'Predicted Dengue Cases for {city.upper()}')
                ax.legend()
                ax.grid(True)
                ax.tick_params(axis='x', rotation=45)
            
            fig.suptitle('Submission Predictions by City', fontsize=16)
            plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to make space for suptitle
            submission_plot_filename = "prediction_plot.png"
            plt.savefig(submission_plot_filename)
            print(f"Submission predictions plot saved to {submission_plot_filename}")
            # plt.show() # Uncomment to display
        except Exception as e:
            print(f"Error generating submission plot: {e}")
        # --- End Submission Plotting ---


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train_csv", type=str, default="DATA/dengue_features_train.csv")
    p.add_argument("--test_csv", type=str, default="DATA/dengue_features_test.csv")
    p.add_argument("--submission_fmt", type=str, default="DATA/submission_format.csv")
    p.add_argument("--out_csv", type=str, default="my_submission.csv")
    p.add_argument("--val_split_year", type=int, default=2006, help="Hold-out year boundary for validation")
    p.add_argument("--seq_len", type=int, default=52)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-3)
    args = p.parse_args()

    main(args)
