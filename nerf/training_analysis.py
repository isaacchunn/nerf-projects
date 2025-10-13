import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime
from pathlib import Path
import seaborn as sns

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Define what gets imported with "from training_analysis import *"
__all__ = [
    'get_experiment_list',
    'load_training_data', 
    'load_metrics_data',
    'plot_training_curves',
    'plot_test_metrics', 
    'get_experiment_summary',
    'print_experiment_summary',
    'create_summary_comparison',
    'analyze_all_experiments',
    'quick_summary'
]

def get_experiment_list(logs_dir="logs"):
    """Get list of all experiment directories."""
    experiments = []
    logs_path = Path(logs_dir)
    if logs_path.exists():
        experiments = [d.name for d in logs_path.iterdir() if d.is_dir()]
    return sorted(experiments)

def load_training_data(experiment_name, logs_dir="logs"):
    """Load training data from both JSONL and CSV files if available."""
    exp_path = Path(logs_dir) / experiment_name
    
    data = {}
    
    # Try to load JSONL file
    jsonl_path = exp_path / "training_log.jsonl"
    if jsonl_path.exists():
        jsonl_data = []
        with open(jsonl_path, 'r') as f:
            for line in f:
                try:
                    jsonl_data.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    continue
        if jsonl_data:
            data['jsonl'] = pd.DataFrame(jsonl_data)
            # Convert timestamp to datetime
            if 'timestamp' in data['jsonl'].columns:
                data['jsonl']['timestamp'] = pd.to_datetime(data['jsonl']['timestamp'])
    
    # Try to load CSV file
    csv_path = exp_path / "training_log.csv"
    if csv_path.exists():
        try:
            data['csv'] = pd.read_csv(csv_path)
            if 'timestamp' in data['csv'].columns:
                data['csv']['timestamp'] = pd.to_datetime(data['csv']['timestamp'])
        except Exception as e:
            print(f"Error loading CSV for {experiment_name}: {e}")
    
    return data

def load_metrics_data(experiment_name, logs_dir="logs"):
    """Load test metrics data from metrics files."""
    exp_path = Path(logs_dir) / experiment_name
    metrics_path = exp_path / "metrics"
    
    if not metrics_path.exists():
        return None
    
    # Load the comprehensive training_metrics.json if it exists
    training_metrics_path = metrics_path / "training_metrics.json"
    if training_metrics_path.exists():
        with open(training_metrics_path, 'r') as f:
            return json.load(f)
    
    # Otherwise, load individual metrics files
    metrics_files = sorted([f for f in metrics_path.glob("metrics_*.json")])
    if not metrics_files:
        return None
    
    metrics_history = []
    for metrics_file in metrics_files:
        try:
            with open(metrics_file, 'r') as f:
                metric_data = json.load(f)
                # Extract iteration from filename
                iteration = int(metrics_file.stem.split('_')[1])
                metric_data['iteration'] = iteration
                metrics_history.append(metric_data)
        except Exception as e:
            print(f"Error loading {metrics_file}: {e}")
    
    return {'metrics_history': metrics_history} if metrics_history else None

def plot_training_curves(experiment_name, logs_dir="logs", save_dir="plots"):
    """Plot training curves for a single experiment."""
    data = load_training_data(experiment_name, logs_dir)
    
    if not data:
        print(f"No training data found for {experiment_name}")
        return
    
    # Use JSONL data if available, otherwise CSV
    df = data.get('jsonl', data.get('csv'))
    if df is None or df.empty:
        print(f"No valid training data for {experiment_name}")
        return
    
    # Create save directory
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Training Curves - {experiment_name}', fontsize=16, fontweight='bold')
    
    # Plot Loss
    if 'loss' in df.columns:
        axes[0, 0].plot(df['iteration'], df['loss'], label='Fine Loss', linewidth=2)
        if 'loss_coarse' in df.columns:
            axes[0, 0].plot(df['iteration'], df['loss_coarse'], label='Coarse Loss', linewidth=2)
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_yscale('log')
    
    # Plot PSNR
    if 'psnr' in df.columns:
        axes[0, 1].plot(df['iteration'], df['psnr'], label='Fine PSNR', linewidth=2)
        if 'psnr_coarse' in df.columns:
            axes[0, 1].plot(df['iteration'], df['psnr_coarse'], label='Coarse PSNR', linewidth=2)
        axes[0, 1].set_xlabel('Iteration')
        axes[0, 1].set_ylabel('PSNR (dB)')
        axes[0, 1].set_title('Training PSNR')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    
    # Plot Learning Rate
    if 'learning_rate' in df.columns:
        axes[1, 0].plot(df['iteration'], df['learning_rate'], linewidth=2, color='orange')
        axes[1, 0].set_xlabel('Iteration')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_title('Learning Rate Schedule')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_yscale('log')
    
    # Plot training time progression
    if 'timestamp' in df.columns:
        # Calculate time elapsed from start
        start_time = df['timestamp'].iloc[0]
        time_elapsed = (df['timestamp'] - start_time).dt.total_seconds() / 3600  # hours
        axes[1, 1].plot(df['iteration'], time_elapsed, linewidth=2, color='green')
        axes[1, 1].set_xlabel('Iteration')
        axes[1, 1].set_ylabel('Time Elapsed (hours)')
        axes[1, 1].set_title('Training Time Progress')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_filename = save_path / f"{experiment_name}_training_curves.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"Saved training curves plot: {plot_filename}")
    plt.show()

def plot_test_metrics(experiment_name, logs_dir="logs", save_dir="plots"):
    """Plot test metrics evolution for a single experiment."""
    metrics_data = load_metrics_data(experiment_name, logs_dir)
    
    if not metrics_data or 'metrics_history' not in metrics_data:
        print(f"No test metrics data found for {experiment_name}")
        return
    
    # Extract metrics history
    history = metrics_data['metrics_history']
    if not history:
        return
    
    # Convert to DataFrame
    metrics_list = []
    for entry in history:
        if 'metrics' in entry:
            row = {'iteration': entry['iteration']}
            row.update(entry['metrics'])
            if 'timestamp' in entry:
                row['timestamp'] = entry['timestamp']
            metrics_list.append(row)
    
    if not metrics_list:
        return
    
    df = pd.DataFrame(metrics_list)
    
    # Create save directory
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)
    
    # Create subplots for test metrics
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Test Metrics Evolution - {experiment_name}', fontsize=16, fontweight='bold')
    
    # Plot PSNR
    if 'avg_psnr' in df.columns:
        axes[0, 0].plot(df['iteration'], df['avg_psnr'], linewidth=2, marker='o')
        if 'std_psnr' in df.columns:
            axes[0, 0].fill_between(df['iteration'], 
                                   df['avg_psnr'] - df['std_psnr'], 
                                   df['avg_psnr'] + df['std_psnr'], 
                                   alpha=0.3)
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].set_ylabel('PSNR (dB)')
        axes[0, 0].set_title('Test PSNR')
        axes[0, 0].grid(True, alpha=0.3)
    
    # Plot SSIM
    if 'avg_ssim' in df.columns:
        axes[0, 1].plot(df['iteration'], df['avg_ssim'], linewidth=2, marker='o', color='orange')
        if 'std_ssim' in df.columns:
            axes[0, 1].fill_between(df['iteration'], 
                                   df['avg_ssim'] - df['std_ssim'], 
                                   df['avg_ssim'] + df['std_ssim'], 
                                   alpha=0.3, color='orange')
        axes[0, 1].set_xlabel('Iteration')
        axes[0, 1].set_ylabel('SSIM')
        axes[0, 1].set_title('Test SSIM')
        axes[0, 1].grid(True, alpha=0.3)
    
    # Plot LPIPS
    if 'avg_lpips' in df.columns:
        axes[1, 0].plot(df['iteration'], df['avg_lpips'], linewidth=2, marker='o', color='red')
        if 'std_lpips' in df.columns:
            axes[1, 0].fill_between(df['iteration'], 
                                   df['avg_lpips'] - df['std_lpips'], 
                                   df['avg_lpips'] + df['std_lpips'], 
                                   alpha=0.3, color='red')
        axes[1, 0].set_xlabel('Iteration')
        axes[1, 0].set_ylabel('LPIPS')
        axes[1, 0].set_title('Test LPIPS (lower is better)')
        axes[1, 0].grid(True, alpha=0.3)
    
    # Plot MSE
    if 'avg_mse' in df.columns:
        axes[1, 1].plot(df['iteration'], df['avg_mse'], linewidth=2, marker='o', color='green')
        if 'std_mse' in df.columns:
            axes[1, 1].fill_between(df['iteration'], 
                                   df['avg_mse'] - df['std_mse'], 
                                   df['avg_mse'] + df['std_mse'], 
                                   alpha=0.3, color='green')
        axes[1, 1].set_xlabel('Iteration')
        axes[1, 1].set_ylabel('MSE')
        axes[1, 1].set_title('Test MSE')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_yscale('log')
    
    plt.tight_layout()
    
    # Save plot
    plot_filename = save_path / f"{experiment_name}_test_metrics.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"Saved test metrics plot: {plot_filename}")
    plt.show()

def get_experiment_summary(experiment_name, logs_dir="logs"):
    """Get final metrics and training time for an experiment."""
    summary = {
        'experiment': experiment_name,
        'final_psnr': None,
        'final_ssim': None,
        'final_lpips': None,
        'total_training_time': None,
        'total_iterations': None,
        'final_train_psnr': None
    }
    
    # Get training data for training time and final training PSNR
    training_data = load_training_data(experiment_name, logs_dir)
    if training_data:
        df = training_data.get('jsonl', training_data.get('csv'))
        if df is not None and not df.empty:
            summary['total_iterations'] = df['iteration'].max()
            if 'psnr' in df.columns:
                summary['final_train_psnr'] = df['psnr'].iloc[-1]
            
            # Calculate training time
            if 'timestamp' in df.columns:
                start_time = df['timestamp'].iloc[0]
                end_time = df['timestamp'].iloc[-1]
                duration = end_time - start_time
                summary['total_training_time'] = duration.total_seconds() / 3600  # hours
    
    # Get test metrics
    metrics_data = load_metrics_data(experiment_name, logs_dir)
    if metrics_data and 'metrics_history' in metrics_data:
        # Get the final metrics (last entry)
        final_metrics = metrics_data['metrics_history'][-1]
        if 'metrics' in final_metrics:
            metrics = final_metrics['metrics']
            summary['final_psnr'] = metrics.get('avg_psnr')
            summary['final_ssim'] = metrics.get('avg_ssim')
            summary['final_lpips'] = metrics.get('avg_lpips')
    
    return summary

def print_experiment_summary(experiment_name, logs_dir="logs"):
    """Print formatted summary for a single experiment."""
    summary = get_experiment_summary(experiment_name, logs_dir)
    
    print(f"\n{'='*60}")
    print(f"EXPERIMENT SUMMARY: {experiment_name}")
    print(f"{'='*60}")
    
    if summary['total_iterations']:
        print(f"Total Iterations: {summary['total_iterations']:,}")
    
    if summary['total_training_time']:
        hours = int(summary['total_training_time'])
        minutes = int((summary['total_training_time'] - hours) * 60)
        print(f"Training Time: {hours}h {minutes}m ({summary['total_training_time']:.2f} hours)")
    
    if summary['final_train_psnr']:
        print(f"Final Training PSNR: {summary['final_train_psnr']:.2f} dB")
    
    print("\nFINAL TEST METRICS:")
    print("-" * 30)
    if summary['final_psnr']:
        print(f"PSNR: {summary['final_psnr']:.2f} dB")
    
    if summary['final_ssim']:
        print(f"SSIM: {summary['final_ssim']:.4f}")
    
    if summary['final_lpips']:
        print(f"LPIPS: {summary['final_lpips']:.4f}")
    
    # Check for missing data
    missing_data = []
    if summary['final_psnr'] is None:
        missing_data.append("PSNR")
    if summary['final_ssim'] is None:
        missing_data.append("SSIM")
    if summary['final_lpips'] is None:
        missing_data.append("LPIPS")
    
    if missing_data:
        print(f"\nNote: Missing test metrics - {', '.join(missing_data)}")

def create_summary_comparison(logs_dir="logs", save_dir="plots"):
    """Create comparison plots for all experiments."""
    experiments = get_experiment_list(logs_dir)
    summaries = []
    
    print("Loading experiment summaries...")
    for exp in experiments:
        summary = get_experiment_summary(exp, logs_dir)
        summaries.append(summary)
    
    # Convert to DataFrame
    df = pd.DataFrame(summaries)
    
    # Filter out experiments with no test metrics
    df_metrics = df.dropna(subset=['final_psnr', 'final_ssim', 'final_lpips'])
    
    if df_metrics.empty:
        print("No experiments with complete test metrics found.")
        return
    
    # Create save directory
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(20, 14))
    fig.suptitle('Experiment Comparison - Final Test Metrics', fontsize=16, fontweight='bold')
    
    # PSNR comparison
    axes[0, 0].bar(range(len(df_metrics)), df_metrics['final_psnr'], color='skyblue')
    axes[0, 0].set_xlabel('Experiments')
    axes[0, 0].set_ylabel('PSNR (dB)')
    axes[0, 0].set_title('Final Test PSNR')
    axes[0, 0].set_xticks(range(len(df_metrics)))
    axes[0, 0].set_xticklabels(df_metrics['experiment'], rotation=45, ha='center')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, v in enumerate(df_metrics['final_psnr']):
        axes[0, 0].text(i, v + 0.2, f'{v:.1f}', ha='center', va='bottom')
    
    # SSIM comparison
    axes[0, 1].bar(range(len(df_metrics)), df_metrics['final_ssim'], color='lightcoral')
    axes[0, 1].set_xlabel('Experiments')
    axes[0, 1].set_ylabel('SSIM')
    axes[0, 1].set_title('Final Test SSIM')
    axes[0, 1].set_xticks(range(len(df_metrics)))
    axes[0, 1].set_xticklabels(df_metrics['experiment'], rotation=45, ha='center')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, v in enumerate(df_metrics['final_ssim']):
        axes[0, 1].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
    
    # LPIPS comparison (lower is better)
    axes[1, 0].bar(range(len(df_metrics)), df_metrics['final_lpips'], color='lightgreen')
    axes[1, 0].set_xlabel('Experiments')
    axes[1, 0].set_ylabel('LPIPS')
    axes[1, 0].set_title('Final Test LPIPS (lower is better)')
    axes[1, 0].set_xticks(range(len(df_metrics)))
    axes[1, 0].set_xticklabels(df_metrics['experiment'], rotation=45, ha='center')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, v in enumerate(df_metrics['final_lpips']):
        axes[1, 0].text(i, v + 0.005, f'{v:.3f}', ha='center', va='bottom')
    
    # Training time comparison
    df_time = df.dropna(subset=['total_training_time'])
    if not df_time.empty:
        axes[1, 1].bar(range(len(df_time)), df_time['total_training_time'], color='gold')
        axes[1, 1].set_xlabel('Experiments')
        axes[1, 1].set_ylabel('Training Time (hours)')
        axes[1, 1].set_title('Total Training Time')
        axes[1, 1].set_xticks(range(len(df_time)))
        axes[1, 1].set_xticklabels(df_time['experiment'], rotation=45, ha='center')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, v in enumerate(df_time['total_training_time']):
            axes[1, 1].text(i, v + 0.2, f'{v:.1f}h', ha='center', va='bottom')
    
    plt.tight_layout(pad=3.0)
    
    # Save plot
    plot_filename = save_path / "experiment_comparison.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"Saved comparison plot: {plot_filename}")
    plt.show()

def analyze_all_experiments(logs_dir="logs", save_dir="plots"):
    """Analyze all experiments: create individual plots and summary."""
    experiments = get_experiment_list(logs_dir)
    
    if not experiments:
        print(f"No experiments found in {logs_dir}")
        return
    
    print(f"Found {len(experiments)} experiments:")
    for exp in experiments:
        print(f"  - {exp}")
    
    print("\nGenerating individual training curves...")
    for exp in experiments:
        print(f"\nProcessing {exp}...")
        plot_training_curves(exp, logs_dir, save_dir)
        plot_test_metrics(exp, logs_dir, save_dir)
        print_experiment_summary(exp, logs_dir)
    
    print("\nGenerating comparison plots...")
    create_summary_comparison(logs_dir, save_dir)
    
    print(f"\nAnalysis complete! All plots saved to '{save_dir}' directory.")

# Example usage functions
def quick_summary(logs_dir="logs"):
    """Print a quick summary table of all experiments."""
    experiments = get_experiment_list(logs_dir)
    summaries = [get_experiment_summary(exp, logs_dir) for exp in experiments]
    
    print(f"\n{'='*100}")
    print("QUICK SUMMARY - ALL EXPERIMENTS")
    print(f"{'='*100}")
    print(f"{'Experiment':<25} {'Iterations':<12} {'Time (h)':<10} {'Train PSNR':<12} {'Test PSNR':<11} {'SSIM':<8} {'LPIPS':<8}")
    print("-" * 100)
    
    for summary in summaries:
        exp_name = summary['experiment'][:24]  # Truncate long names
        iterations = f"{summary['total_iterations']:,}" if summary['total_iterations'] else "N/A"
        time_str = f"{summary['total_training_time']:.1f}" if summary['total_training_time'] else "N/A"
        train_psnr = f"{summary['final_train_psnr']:.1f}" if summary['final_train_psnr'] else "N/A"
        test_psnr = f"{summary['final_psnr']:.1f}" if summary['final_psnr'] else "N/A"
        ssim = f"{summary['final_ssim']:.3f}" if summary['final_ssim'] else "N/A"
        lpips = f"{summary['final_lpips']:.3f}" if summary['final_lpips'] else "N/A"
        
        print(f"{exp_name:<25} {iterations:<12} {time_str:<10} {train_psnr:<12} {test_psnr:<11} {ssim:<8} {lpips:<8}")

if __name__ == "__main__":
    # Example usage
    print("NeRF Training Analysis Tool")
    print("=" * 50)
    
    # Quick summary
    quick_summary()
    
    # Full analysis (uncomment to run)
    # analyze_all_experiments()
