"""
Batch runner for WindPower experiments
12 models x 4 pred_lens x 2 experiment groups = 96 runs
"""
import os
import sys
import subprocess
import time
import csv
import io
from datetime import datetime
from collections import defaultdict

# Force UTF-8 encoding for Windows compatibility
os.environ['PYTHONIOENCODING'] = 'utf-8'

# ============================================================
# Configuration
# ============================================================
MODELS = [
    'TimeXer', 'iTransformer', 'TimeMixer', 'PatchTST', 'DLinear',
    'TimesNet', 'FEDformer', 'Nonstationary_Transformer', 'Autoformer',
    'Informer', 'MuST', 'ProSTA'
]

# Short aliases for models (avoid Windows 260-char path limit)
MODEL_SHORT = {
    'TimeXer': 'TX', 'iTransformer': 'iT', 'TimeMixer': 'TM',
    'PatchTST': 'PT', 'DLinear': 'DL', 'TimesNet': 'TN',
    'FEDformer': 'FD', 'Nonstationary_Transformer': 'NT',
    'Autoformer': 'AF', 'Informer': 'IF', 'MuST': 'MS', 'ProSTA': 'PS'
}
GROUP_SHORT = {'baseline': 'B', 'main': 'M'}

PRED_LENS = [96, 192, 336, 720]

BASELINE_CONFIG = {
    'name': 'baseline',
    'data': 'WindPower_baseline',
    'data_path': 'WindPower_baseline.csv',
    'root_path': './dataset/WindPower_baseline/',
    'enc_in': 5,
    'dec_in': 5,
    'c_out': 1,
    'features': 'MS',
    'target': 'YG',
    'freq': '15min',
}

MAIN_CONFIG = {
    'name': 'main',
    'data': 'WindPower_main',
    'data_path': 'WindPower_main.csv',
    'root_path': './dataset/WindPower_main/',
    'enc_in': 25,
    'dec_in': 25,
    'c_out': 1,
    'features': 'MS',
    'target': 'YG',
    'freq': '15min',
}

COMMON_ARGS = {
    'task_name': 'long_term_forecast',
    'is_training': 1,
    'seq_len': 96,
    'label_len': 48,
    'e_layers': 2,
    'd_layers': 1,
    'factor': 3,
    'd_model': 256,
    'd_ff': 512,
    'dropout': 0.1,
    'des': 'E',
    'checkpoints': './c/',
    'itr': 1,
    'train_epochs': 10,
    'batch_size': 32,
    'patience': 3,
    'learning_rate': 0.0001,
    'lradj': 'type1',
    'num_workers': 0,
    'gpu': 0,
}
COMMON_FLAGS = ['use_gpu']

RESULTS_BASE = 'results_WindPower'
TIMEOUT_SECONDS = 900  # 15 min max per experiment
SKIP_MODELS = ['TimesNet', 'FEDformer', 'Autoformer']  # FFT-based models too slow on RTX 3050
os.makedirs(RESULTS_BASE, exist_ok=True)

# ============================================================
# Build command
# ============================================================
def build_command(model, pred_len, config):
    short_name = MODEL_SHORT.get(model, model)
    gs = GROUP_SHORT.get(config['name'], config['name'])
    model_id = f"WP_{gs}_{short_name}_{pred_len}"
    cmd_parts = [sys.executable, '-u', 'run.py']

    for key, val in COMMON_ARGS.items():
        cmd_parts.append(f'--{key}')
        cmd_parts.append(str(val))

    for flag in COMMON_FLAGS:
        cmd_parts.append(f'--{flag}')

    for key in ['data', 'data_path', 'root_path', 'features', 'target', 'freq', 'enc_in', 'dec_in', 'c_out']:
        cmd_parts.append(f'--{key}')
        cmd_parts.append(str(config[key]))

    cmd_parts += ['--model', model]
    cmd_parts += ['--model_id', model_id]
    cmd_parts += ['--pred_len', str(pred_len)]

    # Model-specific top_k
    if model in ('TimesNet', 'TimeMixer', 'TimeXer'):
        cmd_parts += ['--top_k', '5']

    # TimeMixer requires multi-scale downsampling + no channel independence
    if model == 'TimeMixer':
        cmd_parts += ['--down_sampling_layers', '1']
        cmd_parts += ['--down_sampling_window', '2']
        cmd_parts += ['--down_sampling_method', 'avg']
        cmd_parts += ['--channel_independence', '0']

    # MuST: scales must be <= seq_len (96)
    if model == 'MuST':
        cmd_parts += ['--scales', '96', '48', '24', '12', '6']

    return cmd_parts

# ============================================================
# Skip check
# ============================================================
def check_already_done(model, pred_len, config):
    """Check if experiment already has checkpoint saved (completed successfully)"""
    short_name = MODEL_SHORT.get(model, model)
    gs = GROUP_SHORT.get(config['name'], config['name'])
    model_id = f"WP_{gs}_{short_name}_{pred_len}"
    setting = f"long_term_forecast_{model_id}_{model}_WindPower_{config['name']}_ftMS_sl96_ll48_pl{pred_len}_dm256_nh8_el2_dl1_df512_expand2_dc4_fc3_ebtimeF_dtTrue_E_0"
    ckpt_path = f'{CHECKPOINTS_DIR}/{setting}/checkpoint.pth'
    return os.path.exists(ckpt_path)

CHECKPOINTS_DIR = 'c'

# ============================================================
# Run experiments
# ============================================================
def run_experiments():
    total = len(MODELS) * len(PRED_LENS) * 2
    current = 0
    results_log = []
    log_file = open(f'{RESULTS_BASE}/full_log.txt', 'a', encoding='utf-8')

    for config in [BASELINE_CONFIG, MAIN_CONFIG]:
        group_name = config['name']
        header = f"\n{'='*60}\nGroup: {group_name.upper()} ({config['enc_in']} features)\n{'='*60}"
        print(header)
        log_file.write(header + '\n')

        for model in MODELS:
            # Skip blacklisted models
            if model in SKIP_MODELS:
                for pred_len in PRED_LENS:
                    current += 1
                    msg_skip = f"\n[{current}/{total}] {model} | pred_len={pred_len} | {config['name']} - SKIP (blacklist)"
                    print(msg_skip)
                    log_file.write(msg_skip + '\n')
                    results_log.append({
                        'group': config['name'],
                        'model': model,
                        'pred_len': pred_len,
                        'status': 'SKIP_SLOW',
                        'time': 0
                    })
                continue

            for pred_len in PRED_LENS:
                current += 1
                model_id = f"WindPower_{group_name}_{model}_{pred_len}"

                cmd = build_command(model, pred_len, config)

                # Skip if already done
                if check_already_done(model, pred_len, config):
                    msg2 = f"\n[{current}/{total}] {model} | pred_len={pred_len} | {config['name']} - SKIP (done)"
                    print(msg2)
                    log_file.write(msg2 + '\n')
                    results_log.append({
                        'group': config['name'],
                        'model': model,
                        'pred_len': pred_len,
                        'status': 'SKIP',
                        'time': 0
                    })
                    continue

                cmd_str = ' '.join(cmd)

                msg = f"\n[{current}/{total}] {model} | pred_len={pred_len} | {group_name}"
                print(msg)
                print(f"  Time: {datetime.now().strftime('%H:%M:%S')}")
                log_file.write(msg + '\n')

                start_time = time.time()
                status = 'UNKNOWN'
                try:
                    result = subprocess.run(
                        cmd,
                        capture_output=True,
                        text=True,
                        encoding='utf-8',
                        errors='replace',
                        timeout=TIMEOUT_SECONDS,
                        env={**os.environ, 'PYTHONIOENCODING': 'utf-8'}
                    )
                    elapsed = time.time() - start_time

                    # Write output to log
                    log_file.write(result.stdout)
                    if result.stderr:
                        log_file.write(result.stderr)
                    log_file.flush()

                    if result.returncode == 0:
                        status = 'OK'
                        print(f"  [OK] Completed in {elapsed:.1f}s")
                    else:
                        status = f'FAIL(exit={result.returncode})'
                        print(f"  [FAIL] Exit code: {result.returncode}")
                        # Print last few lines for quick debugging
                        lines = result.stdout.strip().split('\n')[-5:]
                        for l in lines:
                            print(f"    {l}")

                except Exception as e:
                    elapsed = time.time() - start_time
                    status = f'ERROR: {e}'
                    print(f"  [ERROR] {e}")

                results_log.append({
                    'group': group_name,
                    'model': model,
                    'pred_len': pred_len,
                    'status': status,
                    'time': elapsed
                })

            # Save after each model completes all pred_lens
            save_results(results_log)

    log_file.close()
    print(f"\n{'='*60}")
    print("ALL EXPERIMENTS COMPLETED")
    print(f"{'='*60}")
    print_summary(results_log)
    return results_log

def save_results(log):
    with open(f'{RESULTS_BASE}/run_status.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['group', 'model', 'pred_len', 'status', 'time'])
        writer.writeheader()
        writer.writerows(log)

def print_summary(log):
    ok = sum(1 for r in log if r['status'] == 'OK')
    fail = len(log) - ok
    print(f"\nTotal: {len(log)} | OK: {ok} | Failed: {fail}")

    model_stats = defaultdict(lambda: {'OK': 0, 'FAIL': 0})
    for r in log:
        if r['status'] == 'OK':
            model_stats[r['model']]['OK'] += 1
        else:
            model_stats[r['model']]['FAIL'] += 1

    print("\nModel Summary:")
    for model in MODELS:
        s = model_stats[model]
        print(f"  {model}: {s['OK']}/{s['OK']+s['FAIL']} completed")

if __name__ == '__main__':
    if '--dry-run' in sys.argv:
        print("DRY RUN - Commands to execute:")
        for config in [BASELINE_CONFIG, MAIN_CONFIG]:
            for model in MODELS:
                for pred_len in PRED_LENS:
                    cmd = ' '.join(build_command(model, pred_len, config))
                    print(f"\n# {config['name']} | {model} | pred_len={pred_len}")
                    print(cmd)
        print(f"\nTotal: {len(MODELS) * len(PRED_LENS) * 2} runs")
    else:
        run_experiments()
