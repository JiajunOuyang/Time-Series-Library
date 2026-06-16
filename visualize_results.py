"""
Visualization script for WindPower experiment results
Reads checkpoints/test results and generates:
1. Prediction vs Actual plots per model/pred_len/group
2. Summary metric tables (MSE, MAE)
3. Comparison report between baseline and main experiments
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from collections import defaultdict

# ============================================================
# Configuration
# ============================================================
MODELS = [
    'TimeXer', 'iTransformer', 'TimeMixer', 'PatchTST', 'DLinear',
    'TimesNet', 'FEDformer', 'Nonstationary_Transformer', 'Autoformer',
    'Informer', 'MuST', 'ProSTA'
]
PRED_LENS = [96, 192, 336, 720]
GROUPS = ['baseline', 'main']

CHECKPOINTS_DIR = 'c'
RESULTS_BASE = 'results_WindPower'
RES_DIR = 'test_results'

# Short name mapping (must match batch_runner.py)
MODEL_SHORT = {
    'TimeXer': 'TX', 'iTransformer': 'iT', 'TimeMixer': 'TM',
    'PatchTST': 'PT', 'DLinear': 'DL', 'TimesNet': 'TN',
    'FEDformer': 'FD', 'Nonstationary_Transformer': 'NT',
    'Autoformer': 'AF', 'Informer': 'IF', 'MuST': 'MS', 'ProSTA': 'PS'
}
GROUP_SHORT = {'baseline': 'B', 'main': 'M'}

# ============================================================
# Data loading
# ============================================================
def load_experiment_results(model, pred_len, group):
    """Load metrics for a specific experiment run"""
    short_name = MODEL_SHORT.get(model, model)
    gs = GROUP_SHORT.get(group, group)
    model_id = f"WP_{gs}_{short_name}_{pred_len}"
    setting = f"long_term_forecast_{model_id}_{model}_WindPower_{group}_ftMS_sl96_ll48_pl{pred_len}_dm256_nh8_el2_dl1_df512_expand2_dc4_fc3_ebtimeF_dtTrue_E_0"

    cp_path = f'{CHECKPOINTS_DIR}/{setting}'
    res_path = f'./results/{setting}'
    metrics_path = f'{res_path}/metrics.npy'
    pred_path = f'{res_path}/pred.npy'
    true_path = f'{res_path}/true.npy'

    result = {
        'model': model,
        'pred_len': pred_len,
        'group': group,
        'mse': None,
        'mae': None,
    }

    if os.path.exists(metrics_path):
        metrics = np.load(metrics_path)
        result['mse'] = float(metrics[0])
        result['mae'] = float(metrics[1])
        result['pred'] = pred_path if os.path.exists(pred_path) else None
        result['true'] = true_path if os.path.exists(true_path) else None
        result['status'] = 'OK'
    else:
        result['status'] = 'NO_RESULTS'

    return result

def collect_all_results():
    """Collect results from all experiments"""
    all_results = []
    for group in GROUPS:
        for model in MODELS:
            for pred_len in PRED_LENS:
                r = load_experiment_results(model, pred_len, group)
                all_results.append(r)
                if r['status'] == 'OK':
                    print(f"  [OK] {model} pred={pred_len} {group}: MSE={r['mse']:.4f}, MAE={r['mae']:.4f}")
                else:
                    print(f"  [MISS] {model} pred={pred_len} {group}")

    return pd.DataFrame(all_results)

# ============================================================
# Visualization
# ============================================================
def plot_prediction_curves(df_results):
    """Generate prediction vs actual plots for each model"""
    os.makedirs(f'{RESULTS_BASE}/plots/baseline', exist_ok=True)
    os.makedirs(f'{RESULTS_BASE}/plots/main', exist_ok=True)

    for group in GROUPS:
        subset = df_results[df_results['group'] == group]
        for model in MODELS:
            fig, axes = plt.subplots(1, 4, figsize=(20, 5))
            fig.suptitle(f'{model} - {group.upper()} (5 features)' if group == 'baseline'
                         else f'{model} - {group.upper()} (25 features)',
                         fontsize=14, fontweight='bold')

            for i, pred_len in enumerate(PRED_LENS):
                row = subset[(subset['model'] == model) & (subset['pred_len'] == pred_len)]
                ax = axes[i]

                if len(row) > 0 and row.iloc[0]['status'] == 'OK':
                    r = row.iloc[0]
                    try:
                        pred = np.load(r['pred'])
                        true = np.load(r['true'])
                        # Take first sample for visualization
                        if len(pred.shape) == 3:
                            pred_sample = pred[0, -pred_len:, -1]
                            true_sample = true[0, -pred_len:, -1]
                        else:
                            pred_sample = pred[0, -pred_len:]
                            true_sample = true[0, -pred_len:]

                        ax.plot(true_sample, label='Actual', color='blue', alpha=0.7)
                        ax.plot(pred_sample, label='Predicted', color='red', alpha=0.7, linestyle='--')
                        ax.set_title(f'pred_len={pred_len}\nMSE={r["mse"]:.4f}, MAE={r["mae"]:.4f}')
                        ax.legend(loc='upper right', fontsize=7)
                    except Exception as e:
                        ax.set_title(f'pred_len={pred_len}\nError: {e}')
                        ax.text(0.5, 0.5, 'Data load error', ha='center', va='center')
                else:
                    ax.set_title(f'pred_len={pred_len}\nNo data')
                    ax.text(0.5, 0.5, 'N/A', ha='center', va='center')

                ax.grid(True, alpha=0.3)

            plt.tight_layout()
            save_path = f'{RESULTS_BASE}/plots/{group}/{model}_predictions.png'
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  Saved: {save_path}")

def plot_metrics_comparison(df_results):
    """Create comparative bar charts of MSE and MAE across models"""
    os.makedirs(f'{RESULTS_BASE}/plots', exist_ok=True)

    # Filter only OK results
    ok_df = df_results[df_results['status'] == 'OK'].copy()

    if len(ok_df) == 0:
        print("No results to plot!")
        return

    # MSE comparison by pred_len
    fig, axes = plt.subplots(2, 2, figsize=(20, 14))

    for i, pred_len in enumerate(PRED_LENS):
        ax = axes[i // 2, i % 2]
        data = ok_df[ok_df['pred_len'] == pred_len]

        x = np.arange(len(MODELS))
        width = 0.35

        baseline_data = data[data['group'] == 'baseline']
        main_data = data[data['group'] == 'main']

        baseline_mse = []
        main_mse = []
        for model in MODELS:
            b = baseline_data[baseline_data['model'] == model]
            m = main_data[main_data['model'] == model]
            baseline_mse.append(b['mse'].values[0] if len(b) > 0 else 0)
            main_mse.append(m['mse'].values[0] if len(m) > 0 else 0)

        ax.bar(x - width/2, baseline_mse, width, label='Baseline (5 feat)', alpha=0.8)
        ax.bar(x + width/2, main_mse, width, label='Main (25 feat)', alpha=0.8)

        ax.set_title(f'MSE Comparison - pred_len={pred_len}')
        ax.set_xticks(x)
        ax.set_xticklabels(MODELS, rotation=45, ha='right', fontsize=8)
        ax.legend()

    plt.tight_layout()
    save_path = f'{RESULTS_BASE}/plots/mse_comparison.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")

def create_summary_table(df_results):
    """Create CSV summary tables"""
    ok_df = df_results[df_results['status'] == 'OK'].copy()
    if len(ok_df) == 0:
        print("No results for summary!")
        return

    # Pivot table: models x pred_len for each group
    for group in GROUPS:
        group_data = ok_df[ok_df['group'] == group]
        pivot_mse = group_data.pivot_table(values='mse', index='model', columns='pred_len', aggfunc='first')
        pivot_mae = group_data.pivot_table(values='mae', index='model', columns='pred_len', aggfunc='first')

        pivot_mse.to_csv(f'{RESULTS_BASE}/{group}_mse_summary.csv')
        pivot_mae.to_csv(f'{RESULTS_BASE}/{group}_mae_summary.csv')
        print(f"  Saved: {RESULTS_BASE}/{group}_mse_summary.csv")
        print(f"  Saved: {RESULTS_BASE}/{group}_mae_summary.csv")

    # Combined comparison report
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("WindPower Experiment Report: Baseline vs Main")
    report_lines.append("=" * 80)
    report_lines.append(f"Baseline: 5 features (YG, WG, FS, FXJ, ZT)")
    report_lines.append(f"Main: 25 features (5 aggregate + Top-20 turbine YG by correlation)")
    report_lines.append("=" * 80)

    for pred_len in PRED_LENS:
        report_lines.append(f"\n--- Prediction Length: {pred_len} ---")
        report_lines.append(f"{'Model':<30} {'Base MSE':>10} {'Main MSE':>10} {'Diff':>10} {'Better':>10}")
        report_lines.append("-" * 72)
        for model in MODELS:
            b = ok_df[(ok_df['model'] == model) & (ok_df['pred_len'] == pred_len) & (ok_df['group'] == 'baseline')]
            m = ok_df[(ok_df['model'] == model) & (ok_df['pred_len'] == pred_len) & (ok_df['group'] == 'main')]
            if len(b) > 0 and len(m) > 0:
                b_mse = b['mse'].values[0]
                m_mse = m['mse'].values[0]
                diff = b_mse - m_mse
                better = 'MAIN' if diff > 0 else 'BASE'
                report_lines.append(f"{model:<30} {b_mse:>10.4f} {m_mse:>10.4f} {diff:>+10.4f} {better:>10}")

    with open(f'{RESULTS_BASE}/comparison_report.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    print(f"  Saved: {RESULTS_BASE}/comparison_report.txt")

    # Print report
    for line in report_lines:
        print(line)

# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    print("Collecting experiment results...")
    print("=" * 60)
    df = collect_all_results()

    print(f"\n{'='*60}")
    print(f"Collected {len(df)} results ({len(df[df['status']=='OK'])} OK, {len(df[df['status']!='OK'])} missing)")

    if len(df[df['status'] == 'OK']) > 0:
        print("\nGenerating prediction curves...")
        plot_prediction_curves(df)

        print("\nGenerating metrics comparison...")
        plot_metrics_comparison(df)

        print("\nCreating summary tables...")
        create_summary_table(df)
    else:
        print("\nNo successful results to visualize. Check if experiments have completed.")
