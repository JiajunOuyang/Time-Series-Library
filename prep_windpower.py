"""
WindPower data preparation for Time-Series-Library
1. Fix column name (time -> date)
2. Create baseline dataset (5 aggregate features)
3. Correlation analysis -> Create main dataset (5 agg + Top-20 turbine features)
"""
import pandas as pd
import numpy as np
import os

# ============================================================
# Load raw data
# ============================================================
raw_path = 'dataset/WindPower/WindPower.csv'
df = pd.read_csv(raw_path)
print(f'Raw data: {df.shape}')

# Rename time -> date for Dataset_Custom compatibility
df.rename(columns={'time': 'date'}, inplace=True)

# ============================================================
# Feature groups
# ============================================================
AGG_COLS = ['YG', 'WG', 'FS', 'FXJ', 'ZT']  # aggregate features
TARGET_COL = 'YG'
NON_TARGET_AGG = ['WG', 'FS', 'FXJ', 'ZT']

# All columns: date, YG, WG, FS, FXJ, ZT, 1_YG, 1_WG, ..., 57_ZT
all_cols = list(df.columns)
turbine_cols = [c for c in all_cols if c not in ['date'] + AGG_COLS]
print(f'Aggregate features: {len(AGG_COLS)}')
print(f'Turbine-level features: {len(turbine_cols)}')

# ============================================================
# Correlation analysis
# ============================================================
# Compute absolute Pearson correlation of each turbine feature with target YG
corrs = df[turbine_cols].corrwith(df[TARGET_COL]).abs().sort_values(ascending=False)

print(f'\n=== Top 30 most correlated turbine features with {TARGET_COL} ===')
for i, (col, corr) in enumerate(corrs.head(30).items()):
    print(f'  {i+1}. {col}: {corr:.4f}')

# ============================================================
# Create datasets
# ============================================================
TOP_K = 20
selected_turbine = corrs.head(TOP_K).index.tolist()
print(f'\nSelected Top-{TOP_K}: {selected_turbine}')

# Baseline: date + 5 aggregate features
baseline_cols = ['date'] + AGG_COLS
df_baseline = df[baseline_cols].copy()

# Main: date + 5 aggregate features + Top-20 turbine features
main_cols = ['date'] + AGG_COLS + selected_turbine
df_main = df[main_cols].copy()

# ============================================================
# Save datasets
# ============================================================
baseline_dir = 'dataset/WindPower_baseline'
main_dir = 'dataset/WindPower_main'
os.makedirs(baseline_dir, exist_ok=True)
os.makedirs(main_dir, exist_ok=True)

df_baseline.to_csv(f'{baseline_dir}/WindPower_baseline.csv', index=False)
df_main.to_csv(f'{main_dir}/WindPower_main.csv', index=False)

print(f'\n=== Dataset Summary ===')
print(f'Baseline: {df_baseline.shape[1]} cols ({df_baseline.shape[1]-1} features) -> {baseline_dir}/WindPower_baseline.csv')
print(f'  Columns: {list(df_baseline.columns)}')
print(f'Main:     {df_main.shape[1]} cols ({df_main.shape[1]-1} features) -> {main_dir}/WindPower_main.csv')
print(f'  Columns: {list(df_main.columns)}')

# Also save the original WindPower with 'date' column
df.to_csv('dataset/WindPower/WindPower_date.csv', index=False)
print(f'\nFull data with date col: dataset/WindPower/WindPower_date.csv')

# Save selected features list for reference
with open('dataset/WindPower/selected_features.txt', 'w') as f:
    f.write(f'Target: {TARGET_COL}\n')
    f.write(f'Aggregate features (always included): {AGG_COLS}\n')
    f.write(f'Top-{TOP_K} turbine features by correlation:\n')
    for i, col in enumerate(selected_turbine):
        f.write(f'  {i+1}. {col} (r={corrs[col]:.4f})\n')

print('Done!')
