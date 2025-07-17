import json
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
plt.rcParams['font.family'] = 'Times New Roman'

def load_results_with_std(results_dir):
    """Load all JSON result files and calculate both mean and std for each configuration."""
    results = defaultdict(lambda: defaultdict(list))

    for filename in os.listdir(results_dir):
        if filename.endswith('_results.json'):
            filepath = os.path.join(results_dir, filename)

            with open(filepath, 'r') as f:
                data = json.load(f)

            # Extract method and configuration from filename
            if 'adapter+' in filename:
                method = 'adapter+'
                if '_0_2_' in filename:
                    config = 'remove_0_2'
                elif '_0_5_' in filename:
                    config = 'remove_0_5'
                elif '_0_8_' in filename:
                    config = 'remove_0_8'
                else:
                    config = 'baseline'
            elif 'houlsby' in filename:
                method = 'houlsby'
                if '_0_2_' in filename:
                    config = 'remove_0_2'
                elif '_0_5_' in filename:
                    config = 'remove_0_5'
                elif '_0_8_' in filename:
                    config = 'remove_0_8'
                else:
                    config = 'baseline'
            else:
                continue

            results[method][config].append(data['validation_accuracy'])

    return results


def calculate_stats(results):
    """Calculate mean, std, and relative decrease for each configuration."""
    stats = {}

    for method in results:
        stats[method] = {}

        # Calculate means and stds
        means = {}
        stds = {}
        for config in results[method]:
            means[config] = np.mean(results[method][config])
            stds[config] = np.std(results[method][config])

        # Calculate relative decreases
        if 'baseline' in means:
            baseline_acc = means['baseline']
            baseline_std = stds['baseline']

            for config in means:
                if config == 'baseline':
                    rel_decrease = 0.0
                    rel_decrease_std = 0.0
                else:
                    current_acc = means[config]
                    current_std = stds[config]
                    rel_decrease = ((baseline_acc - current_acc) / baseline_acc) * 100
                    # Error propagation for relative decrease
                    rel_decrease_std = (100 / baseline_acc) * np.sqrt(
                        current_std ** 2 + (current_acc * baseline_std / baseline_acc) ** 2)

                stats[method][config] = {
                    'accuracy_mean': means[config],
                    'accuracy_std': stds[config],
                    'rel_decrease_mean': rel_decrease,
                    'rel_decrease_std': rel_decrease_std
                }

    return stats


def create_comprehensive_plot(stats):
    """Create a comprehensive plot with both methods, error bars, and dual y-axes."""

    # Configuration labels and data
    configs = ['No removal', 'Remove first\n3 layers', 'Remove first\n6 layers', 'Remove first\n9 layers']
    config_keys = ['baseline', 'remove_0_2', 'remove_0_5', 'remove_0_8']

    # Extract data for both methods
    adapter_rel_means = [stats['adapter+'][key]['rel_decrease_mean'] for key in config_keys]
    adapter_rel_stds = [stats['adapter+'][key]['rel_decrease_std'] for key in config_keys]
    adapter_acc_means = [stats['adapter+'][key]['accuracy_mean'] for key in config_keys]
    adapter_acc_stds = [stats['adapter+'][key]['accuracy_std'] for key in config_keys]

    houlsby_rel_means = [stats['houlsby'][key]['rel_decrease_mean'] for key in config_keys]
    houlsby_rel_stds = [stats['houlsby'][key]['rel_decrease_std'] for key in config_keys]
    houlsby_acc_means = [stats['houlsby'][key]['accuracy_mean'] for key in config_keys]
    houlsby_acc_stds = [stats['houlsby'][key]['accuracy_std'] for key in config_keys]

    # Create figure and primary axis
    fig, ax1 = plt.subplots(figsize=(14, 8))

    # Set up x-axis
    x = np.arange(len(configs))
    width = 0.35

    # Plot relative decrease bars with error bars
    bars1 = ax1.bar(x - width / 2, adapter_rel_means, width, yerr=adapter_rel_stds,
                    label='Adapter+ (Rel. Decrease)', color='#2196F3', alpha=0.8,
                    edgecolor='black', capsize=5)
    bars2 = ax1.bar(x + width / 2, houlsby_rel_means, width, yerr=houlsby_rel_stds,
                    label='Houlsby (Rel. Decrease)', color='#FF9800', alpha=0.8,
                    edgecolor='black', capsize=5)

    # Primary y-axis (relative decrease)
    ax1.set_xlabel('Configuration', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Relative Decrease in Validation Accuracy (%)', fontsize=12, fontweight='bold', color='black')
    ax1.set_xticks(x)
    ax1.set_xticklabels(configs)
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim(0, max(max(adapter_rel_means), max(houlsby_rel_means)) * 1.15)

    # Add value labels on bars (relative decrease)
    for bar, mean, std in zip(bars1, adapter_rel_means, adapter_rel_stds):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., height + std + 0.1,
                 f'{mean:.1f}%', ha='center', va='bottom', fontweight='bold',
                 color='#2196F3', fontsize=10)

    for bar, mean, std in zip(bars2, houlsby_rel_means, houlsby_rel_stds):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., height + std + 0.1,
                 f'{mean:.1f}%', ha='center', va='bottom', fontweight='bold',
                 color='#FF9800', fontsize=10)

    # Create secondary y-axis for absolute accuracy
    ax2 = ax1.twinx()

    # Plot absolute accuracy lines with error bars
    line1 = ax2.errorbar(x - width / 4, adapter_acc_means, yerr=adapter_acc_stds,
                         fmt='o-', linewidth=2, markersize=8, capsize=5,
                         label='Adapter+ (Accuracy)', color='#1976D2', alpha=0.9)
    line2 = ax2.errorbar(x + width / 4, houlsby_acc_means, yerr=houlsby_acc_stds,
                         fmt='s-', linewidth=2, markersize=8, capsize=5,
                         label='Houlsby (Accuracy)', color='#F57C00', alpha=0.9)

    # Secondary y-axis settings
    ax2.set_ylabel('Absolute Validation Accuracy', fontsize=12, fontweight='bold', color='black')
    ax2.set_ylim(min(min(adapter_acc_means), min(houlsby_acc_means)) * 0.95,
                 max(max(adapter_acc_means), max(houlsby_acc_means)) * 1.02)

    # Add value labels for accuracy
    for i, (mean, std) in enumerate(zip(adapter_acc_means, adapter_acc_stds)):
        ax2.text(x[i] - width / 4, mean + std + 0.005, f'{mean:.3f}',
                 ha='center', va='bottom', fontweight='bold', color='#1976D2', fontsize=9)

    for i, (mean, std) in enumerate(zip(houlsby_acc_means, houlsby_acc_stds)):
        ax2.text(x[i] + width / 4, mean + std + 0.005, f'{mean:.3f}',
                 ha='center', va='bottom', fontweight='bold', color='#F57C00', fontsize=9)

    # Title and legend
    ax1.set_title('Adapter+ vs Houlsby: Robustness to First Layers Removal in CoLA',
                  fontsize=16, fontweight='bold', pad=20)

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', bbox_to_anchor=(0, 0.55))

    plt.tight_layout()
    return fig


def create_summary_table(stats):
    """Create a summary table with all statistics."""
    print("\n" + "=" * 80)
    print("COMPREHENSIVE RESULTS SUMMARY")
    print("=" * 80)

    configs = ['No removal', 'Remove first\n3 layers', 'Remove first\n6 layers', 'Remove first\n9 layers']
    config_names = ['Baseline', 'Remove 0-2', 'Remove 0-5', 'Remove 0-8']

    for method in ['adapter+', 'houlsby']:
        print(f"\n{method.upper()} METHOD:")
        print("-" * 40)
        print(f"{'Config':<12} {'Accuracy':<12} {'Std':<8} {'Rel.Dec.%':<10} {'Std':<8}")
        print("-" * 40)

        for config, name in zip(configs, config_names):
            if config in stats[method]:
                data = stats[method][config]
                print(f"{name:<12} {data['accuracy_mean']:.4f}      {data['accuracy_std']:.4f}   "
                      f"{data['rel_decrease_mean']:.2f}%      {data['rel_decrease_std']:.2f}")


def main():
    # Set the path to your results directory
    results_dir = 'figure6_results'  # Change this to your actual results directory path

    # Load and process results
    print("Loading results with standard deviations...")
    results = load_results_with_std(results_dir)

    print("Calculating statistics...")
    stats = calculate_stats(results)

    # Create comprehensive plot
    print("Creating comprehensive plot...")
    fig = create_comprehensive_plot(stats)
    plt.show()

    # Print summary table
    create_summary_table(stats)

    # Optionally save the plot
    fig.savefig('figure6.png', dpi=300, bbox_inches='tight')

    return stats


if __name__ == "__main__":
    stats = main()