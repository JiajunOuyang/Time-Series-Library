"""Rerun failed/timeout experiments"""
import os, sys, subprocess, time
from datetime import datetime

os.environ['PYTHONIOENCODING'] = 'utf-8'

# ===== MuST: full re-train with fixed scales =====
MUST_RUNS = [
    {'group': 'baseline', 'data': 'WindPower_baseline', 'data_path': 'WindPower_baseline.csv', 'root_path': './dataset/WindPower_baseline/', 'enc_in': 5, 'dec_in': 5},
    {'group': 'main', 'data': 'WindPower_main', 'data_path': 'WindPower_main.csv', 'root_path': './dataset/WindPower_main/', 'enc_in': 25, 'dec_in': 25},
]

# ===== Timed-out: test-only (checkpoint already exists) =====
TEST_ONLY = [
    # NS_Transformer
    ('Nonstationary_Transformer', 'WP_B_NT_336', 'baseline', 'WindPower_baseline', 5, 336),
    ('Nonstationary_Transformer', 'WP_B_NT_720', 'baseline', 'WindPower_baseline', 5, 720),
    ('Nonstationary_Transformer', 'WP_M_NT_336', 'main', 'WindPower_main', 25, 336),
    ('Nonstationary_Transformer', 'WP_M_NT_720', 'main', 'WindPower_main', 25, 720),
    # Informer
    ('Informer', 'WP_B_IF_720', 'baseline', 'WindPower_baseline', 5, 720),
    ('Informer', 'WP_M_IF_720', 'main', 'WindPower_main', 25, 720),
    # PatchTST
    ('PatchTST', 'WP_M_PT_336', 'main', 'WindPower_main', 25, 336),
    ('PatchTST', 'WP_M_PT_720', 'main', 'WindPower_main', 25, 720),
]

BASE_CMD = [
    sys.executable, '-u', 'run.py',
    '--task_name', 'long_term_forecast',
    '--seq_len', '96', '--label_len', '48',
    '--e_layers', '2', '--d_layers', '1', '--factor', '3',
    '--d_model', '256', '--d_ff', '512', '--dropout', '0.1',
    '--des', 'E', '--checkpoints', './c/', '--itr', '1',
    '--train_epochs', '10', '--batch_size', '32', '--patience', '3',
    '--learning_rate', '0.0001', '--lradj', 'type1', '--num_workers', '0',
    '--gpu', '0', '--use_gpu', '--features', 'MS', '--target', 'YG', '--freq', '15min',
]

total = len(MUST_RUNS) * 4 + len(TEST_ONLY)
current = 0

print(f"Total tasks: {total}")
print(f"  MuST re-train: {len(MUST_RUNS) * 4}")
print(f"  Test-only: {len(TEST_ONLY)}")

# ---- Phase 1: MuST ----
for cfg in MUST_RUNS:
    for pred_len in [96, 192, 336, 720]:
        current += 1
        model_id = f"WP_{cfg['group'][0].upper()}_MS_{pred_len}"
        root = cfg['root_path']
        data_path = cfg['data_path']

        cmd = BASE_CMD + [
            '--is_training', '1',
            '--model', 'MuST',
            '--model_id', model_id,
            '--data', cfg['data'],
            '--data_path', data_path,
            '--root_path', root,
            '--enc_in', str(cfg['enc_in']),
            '--dec_in', str(cfg['dec_in']),
            '--c_out', '1',
            '--pred_len', str(pred_len),
            '--scales', '96', '48', '24', '12', '6',
        ]
        print(f"\n[{current}/{total}] MuST {cfg['group']} pred={pred_len} | {datetime.now():%H:%M:%S}")
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace',
                             timeout=900, env={**os.environ, 'PYTHONIOENCODING': 'utf-8'})
            if r.returncode == 0:
                mse_line = [l for l in r.stdout.split('\n') if 'mse:' in l]
                mse = mse_line[-1].strip() if mse_line else 'N/A'
                print(f"  [OK] {mse}")
            else:
                print(f"  [FAIL] exit={r.returncode}")
        except subprocess.TimeoutExpired:
            print(f"  [TIMEOUT]")

# ---- Phase 2: Test-only ----
for model, model_id, group, data, enc_in, pred_len in TEST_ONLY:
    current += 1
    root = f'./dataset/WindPower_{group}/'
    data_path = f'WindPower_{group}.csv'

    cmd = BASE_CMD + [
        '--is_training', '0',
        '--model', model,
        '--model_id', model_id,
        '--data', f'WindPower_{group}',
        '--data_path', data_path,
        '--root_path', root,
        '--enc_in', str(enc_in),
        '--dec_in', str(enc_in),
        '--c_out', '1',
        '--pred_len', str(pred_len),
    ]
    if model == 'TimeMixer':
        cmd += ['--down_sampling_layers', '1', '--down_sampling_window', '2',
                '--down_sampling_method', 'avg', '--channel_independence', '0']

    print(f"\n[{current}/{total}] {model} {group} pred={pred_len} [test-only] | {datetime.now():%H:%M:%S}")
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace',
                         timeout=600, env={**os.environ, 'PYTHONIOENCODING': 'utf-8'})
        if r.returncode == 0:
            mse_line = [l for l in r.stdout.split('\n') if 'mse:' in l]
            mse = mse_line[-1].strip() if mse_line else 'N/A'
            print(f"  [OK] {mse}")
        else:
            print(f"  [FAIL] exit={r.returncode}")
            lines = r.stdout.strip().split('\n')[-3:]
            for l in lines:
                print(f"    {l}")
    except subprocess.TimeoutExpired:
        print(f"  [TIMEOUT]")

print(f"\n{'='*50}")
print("Rerun complete!")
