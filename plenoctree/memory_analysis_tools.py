"""
Memory analysis tools for PlenOctree research.
Provides utilities to analyze memory efficiency metrics from JSON logs.
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional, Tuple
import os
from pathlib import Path


class MemoryAnalyzer:
    """Analyzes memory efficiency metrics from JSON log files."""
    
    def __init__(self, log_files: List[str]):
        """
        Initialize analyzer with log files.
        
        Args:
            log_files: List of paths to JSON log files
        """
        self.log_files = log_files
        self.data = []
        self.load_data()
    
    def load_data(self):
        """Load data from all JSON log files."""
        for log_file in self.log_files:
            try:
                with open(log_file, 'r') as f:
                    file_data = json.load(f)
                    if isinstance(file_data, list):
                        self.data.extend(file_data)
                    else:
                        self.data.append(file_data)
                print(f"Loaded {len(file_data) if isinstance(file_data, list) else 1} entries from {log_file}")
            except Exception as e:
                print(f"Error loading {log_file}: {e}")
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert data to pandas DataFrame for analysis."""
        if not self.data:
            return pd.DataFrame()
        
        # Flatten the data structure
        flattened_data = []
        for entry in self.data:
            row = {
                'timestamp': entry.get('timestamp'),
                'step': entry.get('step'),
                'phase': entry.get('phase')
            }
            
            # Add metrics
            metrics = entry.get('metrics', {})
            for key, value in metrics.items():
                row[f'metric_{key}'] = value
            
            # Add additional info
            additional_info = entry.get('additional_info', {})
            for key, value in additional_info.items():
                row[f'info_{key}'] = value
            
            flattened_data.append(row)
        
        df = pd.DataFrame(flattened_data)
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        return df
    
    def analyze_memory_efficiency(self, phase: Optional[str] = None) -> Dict[str, float]:
        """
        Analyze memory efficiency metrics.
        
        Args:
            phase: Filter by specific phase (training, evaluation, octree_evaluation)
        
        Returns:
            Dictionary of analysis results
        """
        df = self.to_dataframe()
        if df.empty:
            return {}
        
        if phase:
            df = df[df['phase'] == phase]
        
        results = {}
        
        # Memory Efficiency Index (MEI) analysis
        if 'info_memory_efficiency_index' in df.columns:
            mei_col = 'info_memory_efficiency_index'
            results['avg_memory_efficiency_index'] = df[mei_col].mean()
            results['max_memory_efficiency_index'] = df[mei_col].max()
            results['min_memory_efficiency_index'] = df[mei_col].min()
            results['std_memory_efficiency_index'] = df[mei_col].std()
        
        # Peak Memory Efficiency
        if 'info_peak_memory_efficiency_index' in df.columns:
            pmei_col = 'info_peak_memory_efficiency_index'
            results['avg_peak_memory_efficiency_index'] = df[pmei_col].mean()
            results['max_peak_memory_efficiency_index'] = df[pmei_col].max()
        
        # Quality-Memory Trade-off analysis
        if 'info_quality_memory_tradeoff' in df.columns:
            qmt_col = 'info_quality_memory_tradeoff'
            results['avg_quality_memory_tradeoff'] = df[qmt_col].mean()
            results['max_quality_memory_tradeoff'] = df[qmt_col].max()
        
        # LPIPS-Memory Efficiency
        if 'info_lpips_memory_efficiency' in df.columns:
            lme_col = 'info_lpips_memory_efficiency'
            results['avg_lpips_memory_efficiency'] = df[lme_col].mean()
            results['max_lpips_memory_efficiency'] = df[lme_col].max()
        
        # Combined Quality-Memory Index
        if 'info_combined_quality_memory_index' in df.columns:
            cqmi_col = 'info_combined_quality_memory_index'
            results['avg_combined_quality_memory_index'] = df[cqmi_col].mean()
            results['max_combined_quality_memory_index'] = df[cqmi_col].max()
        
        # Memory usage statistics
        gpu_cols = [col for col in df.columns if 'gpu_allocated_gb' in col]
        if gpu_cols:
            gpu_col = gpu_cols[0]
            results['avg_gpu_memory_gb'] = df[gpu_col].mean()
            results['max_gpu_memory_gb'] = df[gpu_col].max()
            results['min_gpu_memory_gb'] = df[gpu_col].min()
        
        # System memory statistics
        sys_cols = [col for col in df.columns if 'system_used_gb' in col]
        if sys_cols:
            sys_col = sys_cols[0]
            results['avg_system_memory_gb'] = df[sys_col].mean()
            results['max_system_memory_gb'] = df[sys_col].max()
        
        return results
    
    def compare_phases(self) -> pd.DataFrame:
        """Compare memory efficiency across different phases."""
        df = self.to_dataframe()
        if df.empty:
            return pd.DataFrame()
        
        # Group by phase and calculate statistics
        efficiency_metrics = [
            'info_memory_efficiency_index',
            'info_peak_memory_efficiency_index', 
            'info_quality_memory_tradeoff',
            'info_lpips_memory_efficiency',
            'info_combined_quality_memory_index'
        ]
        
        memory_metrics = [
            'metric_gpu_allocated_gb',
            'metric_system_used_gb',
            'metric_peak_gpu_allocated_gb'
        ]
        
        quality_metrics = [
            'metric_psnr',
            'metric_ssim',
            'metric_lpips'
        ]
        
        all_metrics = efficiency_metrics + memory_metrics + quality_metrics
        available_metrics = [col for col in all_metrics if col in df.columns]
        
        if 'phase' in df.columns and available_metrics:
            comparison = df.groupby('phase')[available_metrics].agg(['mean', 'std', 'max', 'min'])
            return comparison
        
        return pd.DataFrame()
    
    def plot_memory_efficiency_trends(self, save_path: Optional[str] = None):
        """Plot memory efficiency trends over time/steps."""
        df = self.to_dataframe()
        if df.empty:
            print("No data to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Memory Efficiency Analysis', fontsize=16)
        
        # Plot 1: Memory Efficiency Index over steps
        if 'info_memory_efficiency_index' in df.columns and 'step' in df.columns:
            training_data = df[df['phase'] == 'training']
            if not training_data.empty:
                axes[0, 0].plot(training_data['step'], training_data['info_memory_efficiency_index'], 
                               label='MEI', alpha=0.7)
                axes[0, 0].set_xlabel('Step')
                axes[0, 0].set_ylabel('Memory Efficiency Index (PSNR/GB)')
                axes[0, 0].set_title('Memory Efficiency Index Over Training')
                axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Memory Usage over steps
        if 'metric_gpu_allocated_gb' in df.columns:
            training_data = df[df['phase'] == 'training']
            if not training_data.empty:
                axes[0, 1].plot(training_data['step'], training_data['metric_gpu_allocated_gb'], 
                               label='GPU Memory', color='red', alpha=0.7)
                axes[0, 1].set_xlabel('Step')
                axes[0, 1].set_ylabel('Memory Usage (GB)')
                axes[0, 1].set_title('GPU Memory Usage Over Training')
                axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Quality vs Memory Trade-off
        quality_cols = [col for col in df.columns if 'metric_psnr' in col]
        memory_cols = [col for col in df.columns if 'metric_gpu_allocated_gb' in col]
        
        if quality_cols and memory_cols:
            quality_col = quality_cols[0]
            memory_col = memory_cols[0]
            
            # Color by phase
            phases = df['phase'].unique()
            colors = plt.cm.Set1(np.linspace(0, 1, len(phases)))
            
            for phase, color in zip(phases, colors):
                phase_data = df[df['phase'] == phase]
                if not phase_data.empty and quality_col in phase_data.columns and memory_col in phase_data.columns:
                    axes[1, 0].scatter(phase_data[memory_col], phase_data[quality_col], 
                                     label=phase, alpha=0.6, color=color)
            
            axes[1, 0].set_xlabel('Memory Usage (GB)')
            axes[1, 0].set_ylabel('PSNR')
            axes[1, 0].set_title('Quality vs Memory Trade-off')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Efficiency Indices Comparison
        efficiency_metrics = [
            'info_memory_efficiency_index',
            'info_quality_memory_tradeoff',
            'info_lpips_memory_efficiency'
        ]
        
        available_efficiency = [col for col in efficiency_metrics if col in df.columns]
        if available_efficiency:
            efficiency_data = []
            labels = []
            
            for metric in available_efficiency:
                data = df[metric].dropna()
                if not data.empty:
                    efficiency_data.append(data)
                    labels.append(metric.replace('info_', '').replace('_', ' ').title())
            
            if efficiency_data:
                axes[1, 1].boxplot(efficiency_data, labels=labels)
                axes[1, 1].set_ylabel('Efficiency Value')
                axes[1, 1].set_title('Distribution of Efficiency Indices')
                axes[1, 1].tick_params(axis='x', rotation=45)
                axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        else:
            plt.show()
    
    def generate_report(self, output_path: str):
        """Generate a comprehensive memory efficiency report."""
        report = []
        report.append("# Memory Efficiency Analysis Report\n")
        report.append(f"Generated from {len(self.log_files)} log files with {len(self.data)} total entries.\n")
        
        # Overall analysis
        overall_analysis = self.analyze_memory_efficiency()
        if overall_analysis:
            report.append("## Overall Memory Efficiency Metrics\n")
            for key, value in overall_analysis.items():
                if isinstance(value, float):
                    report.append(f"- **{key.replace('_', ' ').title()}**: {value:.4f}\n")
                else:
                    report.append(f"- **{key.replace('_', ' ').title()}**: {value}\n")
            report.append("\n")
        
        # Phase comparison
        phase_comparison = self.compare_phases()
        if not phase_comparison.empty:
            report.append("## Phase Comparison\n")
            report.append("```\n")
            report.append(phase_comparison.to_string())
            report.append("\n```\n\n")
        
        # Phase-specific analysis
        df = self.to_dataframe()
        if not df.empty and 'phase' in df.columns:
            phases = df['phase'].unique()
            for phase in phases:
                phase_analysis = self.analyze_memory_efficiency(phase)
                if phase_analysis:
                    report.append(f"## {phase.title()} Phase Analysis\n")
                    for key, value in phase_analysis.items():
                        if isinstance(value, float):
                            report.append(f"- **{key.replace('_', ' ').title()}**: {value:.4f}\n")
                        else:
                            report.append(f"- **{key.replace('_', ' ').title()}**: {value}\n")
                    report.append("\n")
        
        # Memory recommendations
        report.append("## Memory Efficiency Recommendations\n")
        if overall_analysis:
            mei = overall_analysis.get('avg_memory_efficiency_index', 0)
            max_gpu = overall_analysis.get('max_gpu_memory_gb', 0)
            
            if mei > 0:
                if mei < 5.0:
                    report.append("- **Low Memory Efficiency**: Consider model compression or architecture optimization.\n")
                elif mei < 10.0:
                    report.append("- **Moderate Memory Efficiency**: Good balance, but room for improvement.\n")
                else:
                    report.append("- **High Memory Efficiency**: Excellent memory utilization.\n")
            
            if max_gpu > 0:
                if max_gpu > 20.0:
                    report.append("- **High Memory Usage**: Consider batch size reduction or gradient checkpointing.\n")
                elif max_gpu > 10.0:
                    report.append("- **Moderate Memory Usage**: Monitor for memory leaks during long training.\n")
                else:
                    report.append("- **Low Memory Usage**: Efficient memory utilization.\n")
        
        # Save report
        with open(output_path, 'w') as f:
            f.writelines(report)
        
        print(f"Memory efficiency report saved to {output_path}")


def analyze_directory(log_dir: str, output_dir: str = None):
    """
    Analyze all JSON log files in a directory.
    
    Args:
        log_dir: Directory containing JSON log files
        output_dir: Directory to save analysis results (defaults to log_dir)
    """
    if output_dir is None:
        output_dir = log_dir
    
    # Find all JSON log files
    log_files = []
    for file_path in Path(log_dir).rglob("*.json"):
        if any(keyword in file_path.name for keyword in ['metrics', 'evaluation', 'memory']):
            log_files.append(str(file_path))
    
    if not log_files:
        print(f"No JSON log files found in {log_dir}")
        return
    
    print(f"Found {len(log_files)} log files:")
    for log_file in log_files:
        print(f"  - {log_file}")
    
    # Create analyzer
    analyzer = MemoryAnalyzer(log_files)
    
    # Generate report
    report_path = os.path.join(output_dir, "memory_efficiency_report.md")
    analyzer.generate_report(report_path)
    
    # Generate plots
    plot_path = os.path.join(output_dir, "memory_efficiency_plots.png")
    analyzer.plot_memory_efficiency_trends(plot_path)
    
    # Save comparison data
    comparison = analyzer.compare_phases()
    if not comparison.empty:
        csv_path = os.path.join(output_dir, "phase_comparison.csv")
        comparison.to_csv(csv_path)
        print(f"Phase comparison data saved to {csv_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze memory efficiency from JSON logs")
    parser.add_argument("log_dir", help="Directory containing JSON log files")
    parser.add_argument("--output_dir", help="Output directory for analysis results")
    
    args = parser.parse_args()
    analyze_directory(args.log_dir, args.output_dir)
