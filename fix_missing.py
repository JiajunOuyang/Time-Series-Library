"""Fix missing results: TimeMixer baseline test-only + re-test NaN experiments"""
import os, sys, subprocess

os.environ['PYTHONIOENCODING'] = 'utf-8'

# TimeMixer baseline missing test results (checkpoint exists, just need test)
MISSING_TESTS = [
    ('TimeMixer', 'WP_B_TM_96', 'baseline', 5, 96),
    ('TimeMixer', 'WP_B_TM_192', 'baseline', 5, 192),
    ('TimeMixer', 'WP_B_TM_336', 'baseline', 5, 336),
]

for model, model_id, group, enc_in, pred_len in MISSING_TESTS:
    root = f'./dataset/WindPower_{group}/'
    data_path = f'WindPower_{group}.csv'

    cmd = [
        sys.executable, '-u', 'run.py',
        '--task_name', 'long_term_forecast', '--is_training', '0',
        '--model', model, '--model_id', model_id,
        '--data', f'WindPower_{group}',
        '--data_path', data_path, '--root_path', root,
        '--features', 'MS', '--target', 'YG', '--freq', '15min',
        '--seq_len', '96', '--label_len', '48', '--pred_len', str(pred_len),
        '--enc_in', str(enc_in), '--dec_in', str(enc_in), '--c_out', '1',
        '--d_model', '256', '--d_ff', '512', '--dropout', '0.1',
        '--des', 'E', '--checkpoints', './c/', '--itr', '1',
        '--e_layers', '2', '--d_layers', '1', '--factor', '3',
        '--num_workers', '0', '--gpu', '0', '--use_gpu',
        '--down_sampling_layers', '1', '--down_sampling_window', '2',
        '--down_sampling_method', 'avg', '--channel_independence', '0',
    ]

    print(f"TimeMixer {group} pred={pred_len} [test-only]")
    r = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8',
                      errors='replace', timeout=900,
                      env={**os.environ, 'PYTHONIOENCODING': 'utf-8'})
    if r.returncode == 0:
        mse_lines = [l for l in r.stdout.split('\n') if 'mse:' in l]
        print(f"  [OK] {mse_lines[-1].strip() if mse_lines else 'N/A'}")
    else:
        print(f"  [FAIL] exit={r.returncode}")
        lines = r.stdout.strip().split('\n')[-5:]
        for l in lines: print(f"    {l}")

print("Done!")
