"""Rerun all MuST experiments with interpolation fix"""
import os, sys, subprocess, time

os.environ['PYTHONIOENCODING'] = 'utf-8'

configs = [
    ('baseline', 'WindPower_baseline', './dataset/WindPower_baseline/', 'WindPower_baseline.csv', 5, 'B'),
    ('main', 'WindPower_main', './dataset/WindPower_main/', 'WindPower_main.csv', 25, 'M'),
]

for group, data, root, data_path, enc_in, gs in configs:
    for pred_len in [96, 192, 336, 720]:
        model_id = f'WP_{gs}_MS_{pred_len}'

        cmd = [
            sys.executable, '-u', 'run.py',
            '--task_name', 'long_term_forecast', '--is_training', '1',
            '--model', 'MuST', '--model_id', model_id,
            '--data', data, '--root_path', root, '--data_path', data_path,
            '--features', 'MS', '--target', 'YG', '--freq', '15min',
            '--seq_len', '96', '--label_len', '48', '--pred_len', str(pred_len),
            '--enc_in', str(enc_in), '--dec_in', str(enc_in), '--c_out', '1',
            '--d_model', '256', '--d_ff', '512', '--dropout', '0.1',
            '--des', 'E', '--checkpoints', './c/', '--itr', '1',
            '--e_layers', '2', '--d_layers', '1', '--factor', '3',
            '--train_epochs', '10', '--batch_size', '32', '--patience', '3',
            '--learning_rate', '0.0001', '--lradj', 'type1', '--num_workers', '0',
            '--gpu', '0', '--use_gpu',
            '--scales', '96', '48', '24', '12', '6',
        ]

        print(f'\nMuST {group} pred={pred_len}')
        start = time.time()
        r = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8',
                          errors='replace', timeout=900,
                          env={**os.environ, 'PYTHONIOENCODING': 'utf-8'})
        elapsed = time.time() - start
        if r.returncode == 0:
            mse_lines = [l for l in r.stdout.split('\n') if 'mse:' in l]
            print(f'  [OK] {elapsed:.0f}s | {mse_lines[-1].strip() if mse_lines else "N/A"}')
        else:
            print(f'  [FAIL] exit={r.returncode} ({elapsed:.0f}s)')
            lines = r.stdout.strip().split('\n')[-3:]
            for l in lines: print(f'    {l}')

print('\nAll MuST done!')
