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
from typing import List, Dict

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

    def __init__(self, df: pd.DataFrame, feature_cols: List[str], city_name: str, # Added city_name for potential city-specific logic
                 label_col: str = "total_cases", seq_len: int = 52,
                 scaler: RobustScaler = None, train: bool = True,
                 imputation_source_values: pd.Series = None):
        self.seq_len = seq_len
        self.train = train
        self.feature_cols = feature_cols
        self.label_col = label_col
        self.city_name = city_name
        
        # Filter dataframe for the specific city BEFORE any processing
        # This assumes df passed is the global df, and we filter here.
        # If df is already city-specific, this line can be removed/adjusted.
        df_city_specific = df[df['city'] == city_name].copy()
        
        if df_city_specific.empty:
            # Handle cases where a city might not be in the provided df slice (e.g. empty test_df for a city)
            print(f"Warning: No data for city '{city_name}' in the provided dataframe slice. Dataset will be empty.")
            self.feats_scaled = np.array([], dtype=np.float32).reshape(0, len(feature_cols) if feature_cols else 0)
            self.labels = np.array([], dtype=np.float32)
            self.windows = []
            return
            
        df_processed = df_city_specific.copy() 

        # fill NA inside the current city block (ffill->bfill)
        # No need for groupby city as df_processed is already city-specific
        for col_name in self.feature_cols:
            if col_name in df_processed.columns:
                 df_processed[col_name] = df_processed[col_name].ffill().bfill()
            else:
                print(f"Warning: Feature column '{col_name}' not found in data for city '{self.city_name}'.")

        # Scaler and Imputation logic
        self.scaler = scaler # Use the passed scaler (could be new or pre-fitted)
        is_scaler_fitted = hasattr(self.scaler, "center_") and self.scaler.center_ is not None

        if self.train and not is_scaler_fitted: # Primary training for this city
            self.imputation_values = df_processed[self.feature_cols].median()
            df_processed[self.feature_cols] = df_processed[self.feature_cols].fillna(self.imputation_values)
            
            if df_processed[self.feature_cols].isnull().any().any():
                nan_cols = df_processed[self.feature_cols].isnull().sum()
                nan_cols_info = nan_cols[nan_cols > 0]
                print(f"Warning: NaNs found in training data for city '{self.city_name}' after median imputation: {nan_cols_info}. Filling with 0.")
                df_processed[self.feature_cols] = df_processed[self.feature_cols].fillna(0)
            
            # Convert to numpy before arcsinh, ensure all data is numeric
            numpy_data = df_processed[self.feature_cols].apply(pd.to_numeric, errors='coerce').fillna(0).to_numpy()
            data_arcsinh = np.arcsinh(numpy_data)
            self.scaler.fit(data_arcsinh)
            feats_scaled = self.scaler.transform(data_arcsinh).astype(np.float32)
        
        else: # Validation or test phase for this city
            if not is_scaler_fitted:
                 raise ValueError(f"Scaler must be fitted for validation/test data for city '{self.city_name}'.")
            if imputation_source_values is None and self.train: # train=True for val, needs imputation values
                raise ValueError(f"Validation data for city '{self.city_name}' needs imputation_source_values from training.")
            
            if imputation_source_values is not None:
                self.imputation_values = imputation_source_values
                df_processed[self.feature_cols] = df_processed[self.feature_cols].fillna(self.imputation_values)

            if df_processed[self.feature_cols].isnull().any().any():
                nan_cols = df_processed[self.feature_cols].isnull().sum()
                nan_cols_info = nan_cols[nan_cols > 0]
                print(f"Warning: NaNs in validation/test data for city '{self.city_name}' after imputation. Columns: {nan_cols_info}. Filling with 0.")
                df_processed[self.feature_cols] = df_processed[self.feature_cols].fillna(0)
            
            numpy_data = df_processed[self.feature_cols].apply(pd.to_numeric, errors='coerce').fillna(0).to_numpy()
            data_arcsinh = np.arcsinh(numpy_data)
            feats_scaled = self.scaler.transform(data_arcsinh).astype(np.float32)
        
        if self.train: 
            labels_data = df_processed[self.label_col].values
            if pd.api.types.is_numeric_dtype(labels_data):
                 self.labels = labels_data.astype(np.float32)
            else:
                 try:
                     self.labels = pd.to_numeric(labels_data, errors='raise').astype(np.float32)
                 except ValueError as e:
                     raise ValueError(f"Label column '{self.label_col}' in city '{self.city_name}' contains non-numeric data: {e}")

        self.windows = []
        # df_processed is already city-specific. No need to filter by city again for indices.
        city_indices = df_processed.index.tolist() # Use indices from df_processed
        if city_indices: # Only proceed if there are indices
            min_city_idx = city_indices[0]
            for i, end_original_idx in enumerate(city_indices):
                # Sliding window should operate on the length of the current city_processed_df
                # map end_original_idx to its position in df_processed for label indexing if needed
                # However, self.labels and self.feats_scaled are now derived from df_processed, so use its length and relative indices
                current_pos_in_df_processed = i 
                
                # start_idx and end_idx for windowing should be relative to df_processed, not the global df
                # The self.feats_scaled and self.labels are indexed from 0 to len(df_processed)-1
                # end_for_window is the current position in df_processed
                end_for_window = current_pos_in_df_processed 
                start_for_window = max(0, end_for_window - seq_len + 1)
                self.windows.append((start_for_window, end_for_window)) 
        
        self.feats_scaled = feats_scaled

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        if not self.windows: # Handle empty dataset
            # This case should ideally be caught earlier, but as a safeguard:
            dummy_seq_shape = (self.seq_len, self.feats_scaled.shape[1] if self.feats_scaled.ndim == 2 and self.feats_scaled.shape[1] > 0 else 1)
            dummy_seq = torch.zeros(dummy_seq_shape, dtype=torch.float32)
            if self.train:
                return dummy_seq, torch.tensor(0.0, dtype=torch.float32)
            return dummy_seq

        start, end = self.windows[idx]
        seq = self.feats_scaled[start:end + 1]
        pad = self.seq_len - seq.shape[0]
        if pad > 0:  
            seq = np.pad(seq, ((pad, 0), (0, 0)), mode="constant", constant_values=0) # Pad with 0
        elif pad < 0: # Should not happen if window logic is correct, but truncate if it does
            seq = seq[-self.seq_len:, :]
            
        if self.train:
            label = self.labels[end] # end is an index into self.labels (derived from df_processed)
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

def train_model_for_city(city_name: str, train_city_df: pd.DataFrame, val_city_df: pd.DataFrame, 
                         feature_cols: List[str], args: argparse.Namespace, device: torch.device):
    """Trains and evaluates a model for a single city."""
    print(f"\n--- Training for city: {city_name.upper()} ---")

    # Scaler and imputation values are learned from this city's training data ONLY
    city_scaler = RobustScaler(quantile_range=(25, 75))
    
    train_ds_city = DengueSeqDataset(train_city_df, feature_cols, city_name=city_name,
                                     seq_len=args.seq_len, scaler=city_scaler, train=True)
    
    if not train_ds_city.windows: # Skip if no training windows could be formed
        print(f"Skipping city {city_name.upper()} due to insufficient training data after processing.")
        return None, float('inf')

    val_ds_city = DengueSeqDataset(val_city_df, feature_cols, city_name=city_name,
                                 seq_len=args.seq_len, scaler=train_ds_city.scaler, 
                                 train=True, # To load labels
                                 imputation_source_values=train_ds_city.imputation_values)
    
    if not val_ds_city.windows: # Skip if no validation windows
        print(f"Skipping city {city_name.upper()} due to insufficient validation data after processing.")
        # Still return the trained model based on training data if any, but MAE is inf
        # Or decide to return None if val is critical
        return None, float('inf') 


    train_loader_city = DataLoader(train_ds_city, batch_size=args.batch_size, shuffle=True)
    val_loader_city = DataLoader(val_ds_city, batch_size=args.batch_size, shuffle=False)

    model_city = VARnetReg(n_feats=len(feature_cols), seq_len=args.seq_len).to(device)
    optimizer_city = torch.optim.AdamW(model_city.parameters(), lr=args.lr, weight_decay=1e-5)

    best_mae_city, best_state_city = float("inf"), None
    
    # Use a unique description for trange for each city
    epoch_iterator = trange(args.epochs, desc=f"Epoch ({city_name.upper()})")
    for epoch in epoch_iterator:
        tr_loss = train_epoch(model_city, train_loader_city, optimizer_city, device)
        mae, _, _ = evaluate(model_city, val_loader_city, device)
        if mae < best_mae_city:
            best_mae_city = mae
            best_state_city = model_city.state_dict()
        
        # Update trange description with current metrics
        epoch_iterator.set_postfix({"train_L1": f"{tr_loss:.3f}", "val_MAE": f"{mae:.3f}"})
        # if (epoch + 1) % 10 == 0 or epoch == 0: # Printing can be too verbose with trange postfix
            # print(f"Epoch {epoch+1:03d} ({city_name.upper()}): train_L1={tr_loss:.3f} | val_MAE={mae:.3f}")

    print(f"Best validation MAE for {city_name.upper()}: {best_mae_city:.3f}")
    if best_state_city:
        model_city.load_state_dict(best_state_city)
        return model_city, best_mae_city
    return None, best_mae_city # Return None if no best state was found (e.g. all MAEs were inf)


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
            labels_all.append(labels.numpy()) # labels are already numpy arrays from DataLoader
    
    if not preds_all or not labels_all:
        return float('inf'), np.array([]), np.array([]) # Handle empty evaluation
        
    preds_all = np.concatenate(preds_all)
    labels_all = np.concatenate(labels_all)
    mae = mean_absolute_error(labels_all, preds_all)
    return mae, preds_all, labels_all


# ---------- Main ---------- #

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using", device)

    full_train_df = pd.read_csv(args.train_csv)
    full_train_df['week_start_date'] = pd.to_datetime(full_train_df['week_start_date'])

    # Define feature columns (globally, as they are common)
    numeric_cols = full_train_df.select_dtypes(include=np.number).columns.tolist()
    feature_cols = [
        c for c in numeric_cols 
        if c not in ("total_cases", "year", "weekofyear", "city") # city is not a feature for VARnetReg
    ]
    
    cities = full_train_df['city'].unique()
    city_models: Dict[str, VARnetReg] = {}
    city_maes: Dict[str, float] = {}

    for city_name in cities:
        # Split data for the current city
        city_df = full_train_df[full_train_df['city'] == city_name]
        
        if args.val_split_year:
            train_part_city = city_df[city_df.year < args.val_split_year].reset_index(drop=True)
            val_part_city = city_df[city_df.year >= args.val_split_year].reset_index(drop=True)
        else:
            # Fallback or error if no split year, though argparse has default
            # For simplicity, assume val_split_year is always provided as per arg default
            print("Warning: --val_split_year not specified effectively. Ensure data splitting is correct.")
            # Example: use last 20% for validation if no year split, requires more logic
            split_idx = int(len(city_df) * 0.8)
            train_part_city = city_df.iloc[:split_idx].reset_index(drop=True)
            val_part_city = city_df.iloc[split_idx:].reset_index(drop=True)

        if train_part_city.empty or val_part_city.empty:
            print(f"Skipping city {city_name.upper()} due to empty train or validation set after split.")
            city_maes[city_name] = float('inf')
            continue

        model_city, mae_city = train_model_for_city(city_name, train_part_city, val_part_city, 
                                                    feature_cols, args, device)
        if model_city:
            city_models[city_name] = model_city
        city_maes[city_name] = mae_city

    print("\n--- Overall City MAEs ---")
    total_mae = 0
    num_valid_cities = 0
    for city_name, mae in city_maes.items():
        print(f"City {city_name.upper()}: Best Validation MAE = {mae:.3f}")
        if mae != float('inf'):
            total_mae += mae
            num_valid_cities +=1
    
    if num_valid_cities > 0:
        average_mae = total_mae / num_valid_cities
        print(f"Average Best Validation MAE across {num_valid_cities} cities: {average_mae:.3f}")
    else:
        print("No models were successfully trained for any city.")


    # Optionally predict test set using per-city models
    if args.test_csv:
        print("\n--- Predicting on Test Set ---")
        test_df_full = pd.read_csv(args.test_csv)
        test_df_full['week_start_date'] = pd.to_datetime(test_df_full['week_start_date'])
        
        all_city_preds_list = []

        for city_name in cities: # Iterate through known cities from training
            if city_name not in city_models:
                print(f"No trained model for city {city_name}. Predictions for this city will be missing or default.")
                # Create empty df for this city to maintain submission format if needed
                # Or fill with a default value like 0 if required by submission
                city_test_df_original_format = test_df_full[test_df_full['city'] == city_name][['city', 'year', 'weekofyear']]
                if not city_test_df_original_format.empty:
                    city_test_df_original_format['total_cases'] = 0 # Default prediction
                    all_city_preds_list.append(city_test_df_original_format)
                continue

            model_city = city_models[city_name]
            model_city.eval() # Ensure model is in eval mode

            # Get the scaler and imputation values from the training phase of this city model
            # This requires train_ds_city to be accessible or its relevant properties stored.
            # For now, we re-create a temporary train_ds to get the scaler for the test data.
            # This is not ideal, better to store scalers/imputation_values per city.
            # Simplified: Assume train_ds_city.scaler and train_ds_city.imputation_values are stored with the model or accessible.
            # Let's retrieve it from a dummy train_ds for that city, this implies re-fitting scaler which is incorrect for test. 
            # A better way: store scalers from training.
            # For now, we'll assume the scaler and imputation values are part of the 'model_city' somehow (which they are not directly)
            # This part needs careful re-architecture to correctly pass/retrieve city-specific scalers/imputation values.
            
            # Correct approach: Retrieve the scaler used for this city during its training.
            # We need to have stored these. For this refactor, we assume they are part of `train_ds` 
            # instances, which are not directly kept. We'll reconstruct the scaler for the *training part* 
            # of this city to get its properties for the test set. This is an approximation.
            
            temp_train_df_for_scaler = full_train_df[full_train_df['city'] == city_name]
            if args.val_split_year: # Use the same split logic as training
                 temp_train_part_city_for_scaler = temp_train_df_for_scaler[temp_train_df_for_scaler.year < args.val_split_year]
            else: # Fallback split, ensure consistency
                 split_idx_temp = int(len(temp_train_df_for_scaler) * 0.8)
                 temp_train_part_city_for_scaler = temp_train_df_for_scaler.iloc[:split_idx_temp]

            if temp_train_part_city_for_scaler.empty:
                print(f"Cannot prepare scaler for city {city_name} for test set due to no training data.")
                city_test_df_original_format = test_df_full[test_df_full['city'] == city_name][['city', 'year', 'weekofyear']]
                if not city_test_df_original_format.empty:
                    city_test_df_original_format['total_cases'] = 0
                    all_city_preds_list.append(city_test_df_original_format)
                continue

            city_specific_scaler_for_test = RobustScaler(quantile_range=(25,75))
            # Create a temporary dataset just to fit scaler and get imputation values for THIS city
            # This is a bit inefficient but ensures city-specific preprocessing parameters
            temp_ds_for_params = DengueSeqDataset(temp_train_part_city_for_scaler, feature_cols, city_name=city_name,
                                                  seq_len=args.seq_len, scaler=city_specific_scaler_for_test, train=True)
            
            city_scaler_fitted = temp_ds_for_params.scaler
            city_imputation_values = temp_ds_for_params.imputation_values

            city_test_df = test_df_full[test_df_full['city'] == city_name]
            if city_test_df.empty:
                continue

            test_ds_city = DengueSeqDataset(city_test_df, feature_cols, city_name=city_name,
                                          seq_len=args.seq_len, scaler=city_scaler_fitted,
                                          train=False, imputation_source_values=city_imputation_values)
            
            if not test_ds_city.windows:
                print(f"No test windows for city {city_name}. Defaulting predictions if necessary.")
                city_test_df_original_format = city_test_df[['city', 'year', 'weekofyear']]
                city_test_df_original_format['total_cases'] = 0
                all_city_preds_list.append(city_test_df_original_format)
                continue

            test_loader_city = DataLoader(test_ds_city, batch_size=args.batch_size, shuffle=False)
            
            city_preds_on_test = []
            with torch.no_grad():
                for seqs_test in test_loader_city:
                    seqs_test = seqs_test.to(device)
                    city_preds_on_test.extend(model_city(seqs_test).cpu().numpy())
            
            # Align predictions with the original test_df format for this city
            # The test_ds_city.windows correspond to the rows in city_test_df that had valid sequences
            # We need to map these predictions back. The submission format needs all test rows.
            
            # Create a dataframe with the original identifiers for the city
            city_test_df_original_format = test_df_full[test_df_full['city'] == city_name][['city', 'year', 'weekofyear']].copy()
            city_test_df_original_format['total_cases'] = 0 # Default for rows not predicted
            
            # Get the indices from city_test_df that correspond to test_ds_city.windows
            # This is tricky because test_ds_city.windows are relative to the data *after* city filtering
            # and the test_ds_city.feats_scaled is built from city_test_df
            # A simpler approach: if test_ds_city makes predictions, they are in order of city_test_df rows *that formed sequences*
            
            # The number of predictions should match the number of windows in test_ds_city
            if len(city_preds_on_test) == len(test_ds_city.windows):
                 # The predictions in city_preds_on_test correspond to the *last week* of each window.
                 # We need to identify which rows in city_test_df these correspond to.
                 # The 'end' index in test_ds_city.windows refers to an index in the city_test_df (after filtering for city)
                predicted_indices_in_city_test_df = [window[1] for window in test_ds_city.windows]
                
                # Get the original global indices from city_test_df for these predicted rows
                original_indices = city_test_df.iloc[predicted_indices_in_city_test_df].index

                # Create a temporary Series with predictions aligned to original test_df_full indices
                preds_series = pd.Series(np.round(np.clip(city_preds_on_test, 0, None)).astype(int), index=original_indices)
                
                # Update the city_test_df_original_format using these original indices
                # This ensures that predictions are placed in the correct rows of the submission format slice for this city
                city_test_df_original_format.loc[original_indices, 'total_cases'] = preds_series
                
            elif not city_preds_on_test and not test_ds_city.windows : # No windows, no preds
                 pass # Already defaulted to 0
            else:
                print(f"Warning: Mismatch in number of predictions ({len(city_preds_on_test)}) and test windows ({len(test_ds_city.windows)}) for city {city_name}. Defaulting city predictions to 0.")
                # city_test_df_original_format is already defaulted to 0

            all_city_preds_list.append(city_test_df_original_format)

        if all_city_preds_list:
            final_submission_df = pd.concat(all_city_preds_list).reset_index(drop=True)
            
            # Ensure the order matches the submission format exactly
            submission_fmt_df = pd.read_csv(args.submission_fmt)[['city', 'year', 'weekofyear']]
            final_submission_df = pd.merge(submission_fmt_df, final_submission_df, on=['city', 'year', 'weekofyear'], how='left')
            final_submission_df['total_cases'] = final_submission_df['total_cases'].fillna(0).astype(int) # Fill any potentially unmerged rows

            out_path = Path(args.out_csv)
            final_submission_df.to_csv(out_path, index=False)
            print("Saved Kaggle‑style predictions →", out_path)

            # --- Plotting Submission Predictions ---
            try:
                print(f"\nGenerating plot for submission file: {out_path}")
                submission_df_plot = pd.read_csv(out_path)
                submission_df_plot['weekofyear_str'] = submission_df_plot['weekofyear'].astype(str).str.zfill(2)
                submission_df_plot['date'] = pd.to_datetime(submission_df_plot['year'].astype(str) + submission_df_plot['weekofyear_str'] + '1', format='%Y%U%w')
                submission_df_plot = submission_df_plot.sort_values(by=['city', 'date']).reset_index(drop=True)

                plot_cities = submission_df_plot['city'].unique()
                n_plot_cities = len(plot_cities)
                
                fig, axes = plt.subplots(n_plot_cities, 1, figsize=(15, 5 * n_plot_cities), squeeze=False)

                for i, city_plot_name in enumerate(plot_cities):
                    city_df_plot = submission_df_plot[submission_df_plot['city'] == city_plot_name]
                    ax = axes[i, 0]
                    ax.plot(city_df_plot['date'], city_df_plot['total_cases'], label=f'Predicted Cases - {city_plot_name.upper()}', marker='.')
                    ax.set_xlabel('Time')
                    ax.set_ylabel('Predicted Total Cases')
                    ax.set_title(f'Predicted Dengue Cases for {city_plot_name.upper()}')
                    ax.legend()
                    ax.grid(True)
                    ax.tick_params(axis='x', rotation=45)
                
                fig.suptitle('Submission Predictions by City', fontsize=16)
                plt.tight_layout(rect=[0, 0, 1, 0.96]) 
                submission_plot_filename = "prediction_plot.png"
                plt.savefig(submission_plot_filename)
                print(f"Submission predictions plot saved to {submission_plot_filename}")
            except Exception as e:
                print(f"Error generating submission plot: {e}")
            # --- End Submission Plotting ---
        else:
            print("No predictions generated for the test set.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train_csv", type=str, default="DATA/dengue_features_train.csv")
    p.add_argument("--test_csv", type=str, default="DATA/dengue_features_test.csv")
    p.add_argument("--submission_fmt", type=str, default="DATA/submission_format.csv")
    p.add_argument("--out_csv", type=str, default="my_submission.csv") # Changed default name
    p.add_argument("--val_split_year", type=int, default=2006, help="Hold-out year boundary for validation")
    p.add_argument("--seq_len", type=int, default=52)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-3)
    args = p.parse_args()

    main(args)
