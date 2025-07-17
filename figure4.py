import json
import os
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
plt.rcParams['font.family'] = 'Times New Roman'


def load_results_data(results_dir):
    """Load all JSON files from results directory"""
    data = []

    for filename in os.listdir(results_dir):
        if filename.endswith('.json'):
            filepath = os.path.join(results_dir, filename)
            try:
                with open(filepath, 'r') as f:
                    result = json.load(f)
                    data.append({
                        'method': result['method'],
                        'trainable_parameters': result['trainable_parameters'],
                        'validation_accuracy': result['validation_accuracy'] * 100  # Convert to percentage
                    })
            except Exception as e:
                print(f"Error loading {filename}: {e}")

    return data


def group_by_method(data):
    """Group data by method and trainable parameters, collecting multiple accuracy values"""
    grouped = defaultdict(lambda: defaultdict(list))
    for item in data:
        method = item['method']
        params = item['trainable_parameters']
        accuracy = item['validation_accuracy']
        grouped[method][params].append(accuracy)
    return grouped


def analyze_parameter_ranges(grouped_data):
    """Analyze and print parameter ranges for each method"""
    print("\n" + "=" * 50)
    print("PARAMETER RANGE ANALYSIS")
    print("=" * 50)

    all_methods = list(grouped_data.keys())
    print(f"Methods found: {all_methods}")

    for method, param_dict in grouped_data.items():
        if param_dict:
            all_params = list(param_dict.keys())
            all_accuracies = []

            for params, accuracies in param_dict.items():
                all_accuracies.extend(accuracies)

            print(f"\n{method.upper()}:")
            print(f"  Number of configurations: {len(param_dict)}")
            print(f"  Parameter range: {min(all_params):,} to {max(all_params):,}")
            print(f"  Parameter range (scientific): {min(all_params):.2e} to {max(all_params):.2e}")
            print(f"  Accuracy range: {min(all_accuracies):.2f}% to {max(all_accuracies):.2f}%")

            # Print individual configurations
            print(f"  Individual configurations:")
            for i, (params, accuracies) in enumerate(sorted(param_dict.items())):
                mean_acc = np.mean(accuracies)
                std_err = np.std(accuracies, ddof=1) / np.sqrt(len(accuracies))
                print(f"    {i + 1}: {params:,} params -> {mean_acc:.2f}% ± {std_err:.2f}% (n={len(accuracies)})")


def calculate_statistics(grouped_data):
    """Calculate mean and standard error for each method-parameter combination"""
    stats_data = defaultdict(list)

    for method, param_dict in grouped_data.items():
        for params, accuracies in param_dict.items():
            accuracies_array = np.array(accuracies)
            mean_acc = np.mean(accuracies_array)
            std_err = np.std(accuracies_array, ddof=1) / np.sqrt(len(accuracies_array))

            stats_data[method].append({
                'trainable_parameters': params,
                'mean_accuracy': mean_acc,
                'std_error': std_err,
                'num_runs': len(accuracies)
            })

    return stats_data


def create_plot(grouped_data, output_path='figure4.png'):
    """Create the parameter efficiency plot with error bars"""

    # Calculate statistics
    stats_data = calculate_statistics(grouped_data)

    # Set up the plot
    plt.figure(figsize=(10, 6))

    # Define colors and markers (same as before)
    colors = {
        'adapter+': 'blue',
        'houlsby': 'red',
        'lora': 'green'
    }

    markers = {
        'adapter+': 'o',
        'houlsby': 's',
        'lora': '^'
    }

    print("\nPLOTTING INFORMATION:")
    print("-" * 30)

    # Plot each method
    for method, data_points in stats_data.items():
        print(f"\nProcessing method: {method}")

        if method in colors:
            # Sort by trainable parameters
            data_points = sorted(data_points, key=lambda x: x['trainable_parameters'])

            params = [d['trainable_parameters'] for d in data_points]
            mean_accuracies = [d['mean_accuracy'] for d in data_points]
            std_errors = [d['std_error'] for d in data_points]

            print(f"  Will plot {len(params)} points for {method}")
            print(f"  Color: {colors[method]}, Marker: {markers[method]}")

            plt.errorbar(params, mean_accuracies, yerr=std_errors,
                         color=colors[method],
                         marker=markers[method],
                         markersize=8,
                         linewidth=2,
                         capsize=5,
                         capthick=1,
                         label=method.capitalize())
        else:
            print(f"  Skipping {method} (not in predefined colors)")

    # Rest of the plotting code remains the same...
    # Customize the plot
    plt.xscale('log')
    plt.xlabel('Num trainable parameters / task', fontsize=12)
    plt.ylabel('Validation accuracy (%)', fontsize=12)
    plt.title('CoLA Validation Accuracy vs Number of Trainable Parameters per Fine-tuning Method', fontsize=14)

    # Set axis limits similar to your reference plot
    plt.xlim(1.5e4, 1e7)
    plt.ylim(74, 86)

    # Add grid
    plt.grid(True, alpha=0.3)

    # Add legend
    plt.legend(fontsize=11)

    # Adjust layout
    plt.tight_layout()

    # Save the plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved as {output_path}")

    # Show the plot
    plt.show()


def main():
    """Main function to generate the plot"""
    results_dir = 'figure4_results'  # Change this to your results directory path

    # Check if directory exists
    if not os.path.exists(results_dir):
        print(f"Error: Directory '{results_dir}' not found!")
        return

    # Load data
    print("Loading data from JSON files...")
    data = load_results_data(results_dir)

    if not data:
        print("No data loaded. Check your JSON files!")
        return

    print(f"Loaded {len(data)} data points")

    # Group by method and parameters
    grouped_data = group_by_method(data)

    # Print summary
    print("\nBASIC SUMMARY:")
    for method, param_dict in grouped_data.items():
        total_runs = sum(len(accuracies) for accuracies in param_dict.values())
        print(f"{method}: {len(param_dict)} configurations, {total_runs} total runs")

    # Create plot
    create_plot(grouped_data)


if __name__ == "__main__":
    main()