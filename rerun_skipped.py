"""Rerun previously skipped models: TimesNet, FEDformer, Autoformer (24 runs)"""
import os, sys, subprocess, time
from datetime import datetime

os.environ['PYTHONIOENCODING'] = 'utf-8'

MODELS = ['FEDformer', 'Autoformer']  # TimesNet: FFT too slow on RTX 3050 even with d_model=128
PRED_LENS = [96, 192, 336, 720]

BASELINE = {'name': 'baseline', 'data': 'WindPower_baseline', 'data_path': 'WindPower_baseline.csv',
            'root_path': './dataset/WindPower_baseline/', 'enc_in': 5, 'dec_in': 5, 'group': 'B'}
MAIN = {'name': 'main', 'data': 'WindPower_main', 'data_path': 'WindPower_main.csv',
        'root_path': './dataset/WindPower_main/', 'enc_in': 25, 'dec_in': 25, 'group': 'M'}

MODEL_SHORT = {'TimesNet': 'TN', 'FEDformer': 'FD', 'Autoformer': 'AF'}

TIMEOUT = 3600  # 1 hour max per run for these slow models

BASE_CMD = [
    sys.executable, '-u', 'run.py',
    '--task_name', 'long_term_forecast',
    '--seq_len', '96', '--label_len', '48',
    '--e_layers', '2', '--d_layers', '1', '--factor', '3',
    '--d_model', '128', '--d_ff', '256', '--dropout', '0.1',
    '--des', 'E', '--checkpoints', './c/', '--itr', '1',
    '--train_epochs', '5', '--batch_size', '16', '--patience', '3',
    '--learning_rate', '0.0001', '--lradj', 'type1', '--num_workers', '0',
    '--gpu', '0', '--use_gpu', '--features', 'MS', '--target', 'YG', '--freq', '15min',
    '--top_k', '3',
]

total = len(MODELS) * len(PRED_LENS) * 2
current = 0

# Skip if already have valid results
def already_ok(model, pred_len, cfg):
    sn = MODEL_SHORT[model]
    model_id = f"WP_{cfg['group']}_{sn}_{pred_len}"
    setting = f"long_term_forecast_{model_id}_{model}_WindPower_{cfg['name']}_ftMS_sl96_ll48_pl{pred_len}_dm256_nh8_el2_dl1_df512_expand2_dc4_fc3_ebtimeF_dtTrue_E_0"
    metrics_path = f'./results/{setting}/metrics.npy'
    if os.path.exists(metrics_path):
        import numpy as np
        m = np.load(metrics_path)
        if not np.isnan(m).any():
            return True
    return False

for cfg in [BASELINE, MAIN]:
    for model in MODELS:
        for pred_len in PRED_LENS:
            current += 1

            if already_ok(model, pred_len, cfg):
                print(f"\n[{current}/{total}] {model} {cfg['name']} pred={pred_len} - SKIP (done)")
                continue

            sn = MODEL_SHORT[model]
            model_id = f"WP_{cfg['group']}_{sn}_{pred_len}"

            cmd = BASE_CMD + [
                '--is_training', '1',
                '--model', model,
                '--model_id', model_id,
                '--data', cfg['data'],
                '--data_path', cfg['data_path'],
                '--root_path', cfg['root_path'],
                '--enc_in', str(cfg['enc_in']),
                '--dec_in', str(cfg['dec_in']),
                '--c_out', '1',
                '--pred_len', str(pred_len),
            ]

            print(f"\n[{current}/{total}] {model} {cfg['name']} pred={pred_len} | {datetime.now():%H:%M:%S}")
            print(f"  Timeout: {TIMEOUT}s")

            start = time.time()
            try:
                r = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8',
                                  errors='replace', timeout=TIMEOUT,
                                  env={**os.environ, 'PYTHONIOENCODING': 'utf-8'})
                elapsed = time.time() - start
                if r.returncode == 0:
                    mse_lines = [l for l in r.stdout.split('\n') if 'mse:' in l]
                    mse = mse_lines[-1].strip() if mse_lines else 'N/A'
                    print(f"  [OK] {elapsed:.0f}s | {mse}")
                else:
                    print(f"  [FAIL] exit={r.returncode} ({elapsed:.0f}s)")
                    lines = r.stdout.strip().split('\n')[-5:]
                    for l in lines:
                        print(f"    {l}")
            except subprocess.TimeoutExpired:
                elapsed = time.time() - start
                print(f"  [TIMEOUT] after {elapsed:.0f}s")

print(f"\n{'='*50}")
print("All skipped model runs complete!")
