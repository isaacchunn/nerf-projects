#!/usr/bin/env python3
"""
Enhanced Scene Analyzer - Comprehensive Visualization Suite

Creates detailed, publication-ready visualizations for ALL collected metrics.
Generates additional plots beyond the standard analyzer to ensure every metric is visualized.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
from typing import Dict, List, Optional
import warnings

warnings.filterwarnings('ignore')

# Import unified theme
from visualization_theme import (Colors, Typography, PlotElements, 
                                 PlotTemplates, get_scene_color, get_metric_color)


class EnhancedSceneAnalyzer:
    """Creates comprehensive visualizations for ALL collected metrics."""
    
    def __init__(self, base_path: str = "/mnt/d/GitHub/nerf-projects/plenoctree/data/Plenoctree/checkpoints"):
        self.base_path = Path(base_path)
        
    def load_efficiency_csv(self, scene: str) -> Optional[pd.DataFrame]:
        """Load the efficiency metrics CSV for a scene."""
        csv_path = self.base_path / "syn_sh16" / scene / "analysis" / f"{scene}_efficiency_metrics.csv"
        
        if not csv_path.exists():
            print(f"❌ No efficiency CSV found for {scene}")
            return None
        
        try:
            df = pd.read_csv(csv_path)
            print(f"✅ Loaded {len(df)} data points for {scene}")
            return df
        except Exception as e:
            print(f"❌ Error loading CSV: {e}")
            return None
    
    def create_memory_comparison_plot(self, df: pd.DataFrame, scene: str, output_dir: Path):
        """Create detailed memory comparison: current vs peak."""
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        fig.suptitle(f'Detailed Memory Analysis - {scene.upper()}', 
                     fontsize=22, fontweight='bold', y=0.98)
        
        # 1. Current vs Peak Memory
        ax1 = axes[0, 0]
        ax1.plot(df['step'], df['memory_gb'], 'b-', linewidth=2.5, label='Current Memory', marker='o', markersize=4)
        ax1.plot(df['step'], df['peak_memory_gb'], 'r--', linewidth=2.5, label='Peak Memory', marker='s', markersize=4)
        ax1.fill_between(df['step'], df['memory_gb'], df['peak_memory_gb'], alpha=0.2, color='orange')
        ax1.set_title('Current vs Peak Memory Usage\nShaded area shows memory headroom', 
                     fontweight='bold', fontsize=15)
        ax1.set_xlabel('Training Step', fontweight='bold')
        ax1.set_ylabel('Memory (GB)', fontweight='bold')
        ax1.legend(loc='best', fontsize=11)
        ax1.grid(True, alpha=0.3)
        
        # 2. Memory Delta
        ax2 = axes[0, 1]
        memory_delta = df['peak_memory_gb'] - df['memory_gb']
        ax2.plot(df['step'], memory_delta, 'purple', linewidth=2.5, marker='o', markersize=4)
        ax2.fill_between(df['step'], 0, memory_delta, alpha=0.3, color='purple')
        ax2.set_title('Memory Headroom\nPeak - Current Memory', 
                     fontweight='bold', fontsize=15)
        ax2.set_xlabel('Training Step', fontweight='bold')
        ax2.set_ylabel('Memory Delta (GB)', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Add statistics
        avg_delta = memory_delta.mean()
        max_delta = memory_delta.max()
        ax2.axhline(y=avg_delta, color='red', linestyle='--', alpha=0.7, label=f'Avg: {avg_delta:.2f} GB')
        ax2.legend(loc='best')
        
        # 3. Memory Utilization Percentage
        ax3 = axes[1, 0]
        if df['peak_memory_gb'].max() > 0:
            utilization = (df['memory_gb'] / df['peak_memory_gb']) * 100
            ax3.plot(df['step'], utilization, 'g-', linewidth=2.5, marker='o', markersize=4)
            ax3.fill_between(df['step'], 0, utilization, alpha=0.3, color='green')
            ax3.set_title('Memory Utilization Efficiency\nCurrent / Peak Memory (%)', 
                         fontweight='bold', fontsize=15)
            ax3.set_xlabel('Training Step', fontweight='bold')
            ax3.set_ylabel('Utilization (%)', fontweight='bold')
            ax3.set_ylim([0, 105])
            ax3.axhline(y=100, color='red', linestyle='--', alpha=0.5, label='100% (Optimal)')
            ax3.legend(loc='best')
            ax3.grid(True, alpha=0.3)
        
        # 4. Memory Statistics Summary
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        stats_text = f"""
MEMORY STATISTICS SUMMARY

Current Memory:
  • Mean:    {df['memory_gb'].mean():.2f} GB
  • Min:     {df['memory_gb'].min():.2f} GB
  • Max:     {df['memory_gb'].max():.2f} GB
  • Std Dev: {df['memory_gb'].std():.2f} GB

Peak Memory:
  • Mean:    {df['peak_memory_gb'].mean():.2f} GB
  • Min:     {df['peak_memory_gb'].min():.2f} GB
  • Max:     {df['peak_memory_gb'].max():.2f} GB
  • Std Dev: {df['peak_memory_gb'].std():.2f} GB

Memory Headroom:
  • Average: {avg_delta:.2f} GB
  • Maximum: {max_delta:.2f} GB
  • Minimum: {memory_delta.min():.2f} GB

Average Utilization: {utilization.mean():.1f}%
        """
        
        ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, 
                fontsize=12, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=1", facecolor='#F0F0F0', 
                         edgecolor='#333333', linewidth=2, alpha=0.9))
        
        plt.tight_layout()
        
        output_path = output_dir / f"{scene}_detailed_memory_analysis.png"
        fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        
        print(f"  ✅ {output_path.name}")
    
    def create_efficiency_comparison_plot(self, df: pd.DataFrame, scene: str, output_dir: Path):
        """Compare all efficiency metrics: MEI, QMT, MEPV."""
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle(f'Comprehensive Efficiency Metrics Comparison - {scene.upper()}', 
                     fontsize=22, fontweight='bold', y=0.98)
        
        # 1. MEI: Current vs Peak
        ax1 = axes[0, 0]
        ax1.plot(df['step'], df['mei'], 'b-', linewidth=3, label='Current MEI', marker='o', markersize=5)
        ax1.plot(df['step'], df['peak_mei'], 'b--', linewidth=2.5, label='Peak MEI', marker='s', markersize=4, alpha=0.7)
        ax1.set_title('Memory Efficiency Index (MEI)\nPSNR per GB - Higher is Better', 
                     fontweight='bold', fontsize=14)
        ax1.set_xlabel('Training Step', fontweight='bold')
        ax1.set_ylabel('MEI (PSNR/GB)', fontweight='bold')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        
        # 2. QMT: Current vs Peak
        ax2 = axes[0, 1]
        if df['qmt'].notna().any():
            ax2.plot(df['step'], df['qmt'], 'g-', linewidth=3, label='Current QMT', marker='o', markersize=5)
            ax2.plot(df['step'], df['peak_qmt'], 'g--', linewidth=2.5, label='Peak QMT', marker='s', markersize=4, alpha=0.7)
            ax2.set_title('Quality-Memory Tradeoff (QMT)\n(PSNR x SSIM) per GB - Higher is Better', 
                         fontweight='bold', fontsize=14)
            ax2.set_xlabel('Training Step', fontweight='bold')
            ax2.set_ylabel('QMT', fontweight='bold')
            ax2.legend(loc='best')
            ax2.grid(True, alpha=0.3)
        
        # 3. MEPV: Standard vs Quality-weighted
        ax3 = axes[0, 2]
        if df['mepv'].notna().any():
            ax3.plot(df['step'], df['mepv'], 'purple', linewidth=3, label='MEPV', marker='o', markersize=5)
            if df['mepv_quality'].notna().any():
                ax3_twin = ax3.twinx()
                ax3_twin.plot(df['step'], df['mepv_quality'], 'orange', linewidth=2.5, 
                            label='Quality-weighted MEPV', marker='s', markersize=4, linestyle='--')
                ax3_twin.set_ylabel('Quality MEPV (PSNR x MVoxels/GB)', fontweight='bold', color='orange')
                ax3_twin.tick_params(axis='y', labelcolor='orange')
                ax3_twin.legend(loc='upper right')
            
            ax3.set_title('Memory Efficiency Per Voxel (MEPV)\nMillion Voxels per GB - Higher is Better', 
                         fontweight='bold', fontsize=14)
            ax3.set_xlabel('Training Step', fontweight='bold')
            ax3.set_ylabel('MEPV (MVoxels/GB)', fontweight='bold', color='purple')
            ax3.tick_params(axis='y', labelcolor='purple')
            ax3.legend(loc='upper left')
            ax3.grid(True, alpha=0.3)
        
        # 4. All Efficiency Metrics Normalized (0-1 scale)
        ax4 = axes[1, 0]
        
        # Normalize each metric to 0-1 for comparison
        metrics_to_compare = {}
        if df['mei'].notna().any():
            mei_norm = (df['mei'] - df['mei'].min()) / (df['mei'].max() - df['mei'].min())
            ax4.plot(df['step'], mei_norm, 'b-', linewidth=2.5, label='MEI (norm)', marker='o', markersize=4)
            metrics_to_compare['MEI'] = df['mei'].iloc[-1]
        
        if df['qmt'].notna().any():
            qmt_norm = (df['qmt'] - df['qmt'].min()) / (df['qmt'].max() - df['qmt'].min())
            ax4.plot(df['step'], qmt_norm, 'g-', linewidth=2.5, label='QMT (norm)', marker='s', markersize=4)
            metrics_to_compare['QMT'] = df['qmt'].iloc[-1]
        
        if df['mepv'].notna().any():
            mepv_norm = (df['mepv'] - df['mepv'].min()) / (df['mepv'].max() - df['mepv'].min())
            ax4.plot(df['step'], mepv_norm, 'purple', linewidth=2.5, label='MEPV (norm)', marker='^', markersize=4)
            metrics_to_compare['MEPV'] = df['mepv'].iloc[-1]
        
        if df['combined_efficiency'].notna().any():
            comb_norm = (df['combined_efficiency'] - df['combined_efficiency'].min()) / (df['combined_efficiency'].max() - df['combined_efficiency'].min())
            ax4.plot(df['step'], comb_norm, 'orange', linewidth=2.5, label='Combined (norm)', marker='d', markersize=4)
            metrics_to_compare['Combined'] = df['combined_efficiency'].iloc[-1]
        
        ax4.set_title('All Efficiency Metrics Normalized\nComparing Relative Performance', 
                     fontweight='bold', fontsize=14)
        ax4.set_xlabel('Training Step', fontweight='bold')
        ax4.set_ylabel('Normalized Value (0-1)', fontweight='bold')
        ax4.legend(loc='best', fontsize=10)
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim([-0.05, 1.05])
        
        # 5. Efficiency Metrics Bar Comparison (Final Values)
        ax5 = axes[1, 1]
        if metrics_to_compare:
            colors_bar = ['#4A90E2', '#50C878', '#9B59B6', '#F39C12']
            bars = ax5.bar(range(len(metrics_to_compare)), list(metrics_to_compare.values()), 
                          color=colors_bar[:len(metrics_to_compare)], alpha=0.8, edgecolor='white', linewidth=2)
            ax5.set_xticks(range(len(metrics_to_compare)))
            ax5.set_xticklabels(list(metrics_to_compare.keys()), fontweight='bold')
            ax5.set_title('Final Efficiency Metrics\nHigher is Better', 
                         fontweight='bold', fontsize=14)
            ax5.set_ylabel('Metric Value', fontweight='bold')
            ax5.grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for bar, val in zip(bars, metrics_to_compare.values()):
                height = bar.get_height()
                ax5.text(bar.get_x() + bar.get_width()/2., height + height*0.02,
                        f'{val:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
        
        # 6. Peak vs Current Efficiency Comparison
        ax6 = axes[1, 2]
        comparison_data = []
        if df['mei'].notna().any():
            comparison_data.append(['MEI', df['mei'].iloc[-1], df['peak_mei'].iloc[-1]])
        if df['qmt'].notna().any():
            comparison_data.append(['QMT', df['qmt'].iloc[-1], df['peak_qmt'].iloc[-1]])
        if df['combined_efficiency'].notna().any():
            comparison_data.append(['Combined', df['combined_efficiency'].iloc[-1], 
                                   df['peak_combined_efficiency'].iloc[-1]])
        
        if comparison_data:
            metrics_names = [d[0] for d in comparison_data]
            current_vals = [d[1] for d in comparison_data]
            peak_vals = [d[2] for d in comparison_data]
            
            x = np.arange(len(metrics_names))
            width = 0.35
            
            bars1 = ax6.bar(x - width/2, current_vals, width, label='Current', 
                           color='#4A90E2', alpha=0.8, edgecolor='white', linewidth=2)
            bars2 = ax6.bar(x + width/2, peak_vals, width, label='Peak Memory', 
                           color='#E74C3C', alpha=0.8, edgecolor='white', linewidth=2)
            
            ax6.set_title('Current vs Peak Memory Efficiency\nComparing Best vs Average', 
                         fontweight='bold', fontsize=14)
            ax6.set_ylabel('Efficiency Value', fontweight='bold')
            ax6.set_xticks(x)
            ax6.set_xticklabels(metrics_names, fontweight='bold')
            ax6.legend(loc='best')
            ax6.grid(True, alpha=0.3, axis='y')
            
            # Add value labels
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    ax6.text(bar.get_x() + bar.get_width()/2., height + height*0.02,
                            f'{height:.2f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        output_path = output_dir / f"{scene}_efficiency_comparison_detailed.png"
        fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        
        print(f"  ✅ {output_path.name}")
    
    def create_quality_metrics_detailed(self, df: pd.DataFrame, scene: str, output_dir: Path):
        """Create detailed quality metrics visualization."""
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        fig.suptitle(f'Detailed Quality Metrics Analysis - {scene.upper()}', 
                     fontsize=22, fontweight='bold', y=0.98)
        
        # 1. All Quality Metrics Together
        ax1 = axes[0, 0]
        ax1_psnr = ax1
        ax1_psnr.plot(df['step'], df['psnr'], 'r-', linewidth=3, label='PSNR', marker='o', markersize=5)
        ax1_psnr.set_xlabel('Training Step', fontweight='bold')
        ax1_psnr.set_ylabel('PSNR (dB)', fontweight='bold', color='r')
        ax1_psnr.tick_params(axis='y', labelcolor='r')
        
        if df['ssim'].notna().any():
            ax1_ssim = ax1_psnr.twinx()
            ax1_ssim.plot(df['step'], df['ssim'], 'b-', linewidth=2.5, label='SSIM', marker='s', markersize=4)
            ax1_ssim.set_ylabel('SSIM (0-1)', fontweight='bold', color='b')
            ax1_ssim.tick_params(axis='y', labelcolor='b')
        
        ax1_psnr.set_title('Quality Metrics: PSNR and SSIM\nDual-axis comparison', 
                          fontweight='bold', fontsize=15)
        ax1_psnr.grid(True, alpha=0.3)
        ax1_psnr.legend(loc='upper left')
        if df['ssim'].notna().any():
            ax1_ssim.legend(loc='upper right')
        
        # 2. Combined Quality Score
        ax2 = axes[0, 1]
        if df['combined_quality'].notna().any():
            ax2.plot(df['step'], df['combined_quality'], 'purple', linewidth=3, marker='o', markersize=5)
            ax2.fill_between(df['step'], 0, df['combined_quality'], alpha=0.3, color='purple')
            ax2.set_title('Combined Quality Score\nPSNR x SSIM x (1-LPIPS) - Higher is Better', 
                         fontweight='bold', fontsize=15)
            ax2.set_xlabel('Training Step', fontweight='bold')
            ax2.set_ylabel('Combined Quality', fontweight='bold')
            ax2.grid(True, alpha=0.3)
            
            # Add final value annotation
            final_val = df['combined_quality'].iloc[-1]
            ax2.axhline(y=final_val, color='red', linestyle='--', alpha=0.5, 
                       label=f'Final: {final_val:.2f}')
            ax2.legend(loc='best')
        
        # 3. LPIPS Detailed (Lower is Better)
        ax3 = axes[1, 0]
        if df['lpips'].notna().any() and df['lpips'].notna().sum() > 0:
            lpips_data = df['lpips'].dropna()
            steps_data = df['step'][df['lpips'].notna()]
            ax3.plot(steps_data, lpips_data, 'orange', linewidth=3, marker='o', markersize=5)
            ax3.fill_between(steps_data, 0, lpips_data, alpha=0.3, color='orange')
            ax3.set_title('LPIPS Perceptual Metric\nLower is Better', 
                         fontweight='bold', fontsize=15)
            ax3.set_xlabel('Training Step', fontweight='bold')
            ax3.set_ylabel('LPIPS (0-1)', fontweight='bold')
            ax3.grid(True, alpha=0.3)
            
            # Add threshold line (LPIPS < 0.1 is excellent)
            ax3.axhline(y=0.1, color='green', linestyle='--', alpha=0.5, 
                       label='Excellent (< 0.1)')
            ax3.legend(loc='best')
        else:
            ax3.text(0.5, 0.5, 'LPIPS data not available', 
                    ha='center', va='center', transform=ax3.transAxes,
                    fontsize=14, fontweight='bold', color='gray')
            ax3.set_title('LPIPS Perceptual Metric', fontweight='bold', fontsize=15)
        
        # 4. Quality Statistics Summary
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        lpips_text = "N/A"
        if df['lpips'].notna().any() and df['lpips'].notna().sum() > 0:
            lpips_text = f"{df['lpips'].mean():.4f} (mean)"
        
        ssim_text = "N/A"
        if df['ssim'].notna().any():
            ssim_text = f"{df['ssim'].mean():.4f} (mean)"
        
        combined_text = "N/A"
        if df['combined_quality'].notna().any():
            combined_text = f"{df['combined_quality'].mean():.2f} (mean)"
        
        stats_text = f"""
QUALITY METRICS SUMMARY

PSNR (Peak Signal-to-Noise Ratio):
  • Final:   {df['psnr'].iloc[-1]:.2f} dB
  • Mean:    {df['psnr'].mean():.2f} dB
  • Max:     {df['psnr'].max():.2f} dB
  • Min:     {df['psnr'].min():.2f} dB
  • Range:   {df['psnr'].max() - df['psnr'].min():.2f} dB

SSIM (Structural Similarity):
  • Final:   {df['ssim'].iloc[-1] if df['ssim'].notna().any() else "N/A"}
  • Mean:    {ssim_text}

LPIPS (Perceptual Similarity):
  • Final:   {df['lpips'].iloc[-1] if df['lpips'].notna().any() and df['lpips'].notna().sum() > 0 else "N/A"}
  • Mean:    {lpips_text}

Combined Quality Score:
  • Final:   {df['combined_quality'].iloc[-1]:.2f} if combined_text != "N/A" else "N/A"
  • Mean:    {combined_text}

Quality Rating: {"Excellent" if df['psnr'].iloc[-1] > 30 else "Good" if df['psnr'].iloc[-1] > 25 else "Fair"}
        """
        
        ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, 
                fontsize=11, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=1", facecolor='#F0F0F0', 
                         edgecolor='#333333', linewidth=2, alpha=0.9))
        
        plt.tight_layout()
        
        output_path = output_dir / f"{scene}_quality_metrics_detailed.png"
        fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        
        print(f"  ✅ {output_path.name}")
    
    def create_training_progression_overview(self, df: pd.DataFrame, scene: str, output_dir: Path):
        """Create comprehensive training progression showing all key metrics over time."""
        fig = plt.figure(figsize=(24, 14))
        fig.suptitle(f'Complete Training Progression Overview - {scene.upper()}', 
                     fontsize=24, fontweight='bold', y=0.98)
        
        gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)
        
        # 1. PSNR Evolution
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(df['step'], df['psnr'], 'r-', linewidth=3, marker='o', markersize=4)
        ax1.fill_between(df['step'], df['psnr'].min(), df['psnr'], alpha=0.3, color='red')
        ax1.set_title('PSNR Evolution\nHigher is Better', fontweight='bold', fontsize=14)
        ax1.set_xlabel('Training Step', fontweight='bold')
        ax1.set_ylabel('PSNR (dB)', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # 2. Memory Evolution
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(df['step'], df['memory_gb'], 'b-', linewidth=2.5, label='Current', marker='o', markersize=4)
        ax2.plot(df['step'], df['peak_memory_gb'], 'b--', linewidth=2, label='Peak', marker='s', markersize=3, alpha=0.7)
        ax2.set_title('Memory Evolution\nLower is Better', fontweight='bold', fontsize=14)
        ax2.set_xlabel('Training Step', fontweight='bold')
        ax2.set_ylabel('Memory (GB)', fontweight='bold')
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)
        
        # 3. MEI Evolution
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.plot(df['step'], df['mei'], 'g-', linewidth=3, marker='o', markersize=4)
        ax3.fill_between(df['step'], 0, df['mei'], alpha=0.3, color='green')
        ax3.set_title('MEI Evolution\nHigher is Better', fontweight='bold', fontsize=14)
        ax3.set_xlabel('Training Step', fontweight='bold')
        ax3.set_ylabel('MEI (PSNR/GB)', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # 4. SSIM Evolution
        ax4 = fig.add_subplot(gs[1, 0])
        if df['ssim'].notna().any():
            ax4.plot(df['step'], df['ssim'], 'purple', linewidth=3, marker='o', markersize=4)
            ax4.fill_between(df['step'], df['ssim'].min(), df['ssim'], alpha=0.3, color='purple')
            ax4.set_title('SSIM Evolution\nHigher is Better', fontweight='bold', fontsize=14)
            ax4.set_xlabel('Training Step', fontweight='bold')
            ax4.set_ylabel('SSIM (0-1)', fontweight='bold')
            ax4.grid(True, alpha=0.3)
        
        # 5. QMT Evolution
        ax5 = fig.add_subplot(gs[1, 1])
        if df['qmt'].notna().any():
            ax5.plot(df['step'], df['qmt'], 'orange', linewidth=3, marker='o', markersize=4)
            ax5.fill_between(df['step'], 0, df['qmt'], alpha=0.3, color='orange')
            ax5.set_title('QMT Evolution\nHigher is Better', fontweight='bold', fontsize=14)
            ax5.set_xlabel('Training Step', fontweight='bold')
            ax5.set_ylabel('QMT', fontweight='bold')
            ax5.grid(True, alpha=0.3)
        
        # 6. MEPV Evolution
        ax6 = fig.add_subplot(gs[1, 2])
        if df['mepv'].notna().any():
            ax6.plot(df['step'], df['mepv'], 'teal', linewidth=3, marker='o', markersize=4)
            ax6.fill_between(df['step'], 0, df['mepv'], alpha=0.3, color='teal')
            ax6.set_title('MEPV Evolution\nHigher is Better', fontweight='bold', fontsize=14)
            ax6.set_xlabel('Training Step', fontweight='bold')
            ax6.set_ylabel('MEPV (MVoxels/GB)', fontweight='bold')
            ax6.grid(True, alpha=0.3)
        
        # 7. Combined Quality
        ax7 = fig.add_subplot(gs[2, 0])
        if df['combined_quality'].notna().any():
            ax7.plot(df['step'], df['combined_quality'], 'magenta', linewidth=3, marker='o', markersize=4)
            ax7.fill_between(df['step'], 0, df['combined_quality'], alpha=0.3, color='magenta')
            ax7.set_title('Combined Quality\nPSNR x SSIM x (1-LPIPS)', fontweight='bold', fontsize=14)
            ax7.set_xlabel('Training Step', fontweight='bold')
            ax7.set_ylabel('Combined Quality', fontweight='bold')
            ax7.grid(True, alpha=0.3)
        
        # 8. Combined Efficiency
        ax8 = fig.add_subplot(gs[2, 1])
        if df['combined_efficiency'].notna().any():
            ax8.plot(df['step'], df['combined_efficiency'], 'brown', linewidth=3, marker='o', markersize=4)
            ax8.fill_between(df['step'], 0, df['combined_efficiency'], alpha=0.3, color='brown')
            ax8.set_title('Combined Efficiency\nHigher is Better', fontweight='bold', fontsize=14)
            ax8.set_xlabel('Training Step', fontweight='bold')
            ax8.set_ylabel('Combined Efficiency', fontweight='bold')
            ax8.grid(True, alpha=0.3)
        
        # 9. Comprehensive Stats
        ax9 = fig.add_subplot(gs[2, 2])
        ax9.axis('off')
        
        # Format values first to avoid f-string issues
        ssim_val = f"{df['ssim'].iloc[-1]:.4f}" if df['ssim'].notna().any() else "N/A"
        lpips_val = f"{df['lpips'].iloc[-1]:.4f}" if df['lpips'].notna().any() and df['lpips'].notna().sum() > 0 else "N/A"
        qmt_val = f"{df['qmt'].iloc[-1]:.2f}" if df['qmt'].notna().any() else "N/A"
        mepv_val = f"{df['mepv'].iloc[-1]:.3f}" if df['mepv'].notna().any() else "N/A"
        performance = "EXCELLENT" if df['psnr'].iloc[-1] > 30 else "GOOD"
        
        final_stats = f"""
FINAL TRAINING RESULTS

Quality Metrics:
  PSNR:  {df['psnr'].iloc[-1]:.2f} dB
  SSIM:  {ssim_val}
  LPIPS: {lpips_val}

Memory Metrics:
  Current: {df['memory_gb'].iloc[-1]:.2f} GB
  Peak:    {df['peak_memory_gb'].iloc[-1]:.2f} GB

Efficiency Metrics:
  MEI:     {df['mei'].iloc[-1]:.2f}
  QMT:     {qmt_val}
  MEPV:    {mepv_val}

Training Statistics:
  Total Steps: {len(df):,}
  PSNR Gain:   {df['psnr'].iloc[-1] - df['psnr'].iloc[0]:.2f} dB
  Memory Δ:    {df['memory_gb'].iloc[-1] - df['memory_gb'].iloc[0]:.2f} GB

{scene.upper()} Performance: {performance}
        """
        
        ax9.text(0.1, 0.9, final_stats, transform=ax9.transAxes, 
                fontsize=12, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=1", facecolor='#E8F4FD', 
                         edgecolor='#2E86AB', linewidth=2, alpha=0.9))
        
        plt.tight_layout()
        
        output_path = output_dir / f"{scene}_complete_training_progression.png"
        fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        
        print(f"  ✅ {output_path.name}")
    
    def analyze_scene(self, scene: str):
        """Generate all enhanced visualizations for a scene."""
        print(f"\n🎨 Generating ENHANCED visualizations for {scene}...")
        
        # Load data
        df = self.load_efficiency_csv(scene)
        if df is None or len(df) == 0:
            print(f"❌ No data available for {scene}")
            return
        
        # Create output directory
        output_dir = self.base_path / "syn_sh16" / scene / "analysis"
        output_dir.mkdir(exist_ok=True, parents=True)
        
        print(f"\n📊 Creating detailed visualizations:")
        
        # Create all enhanced plots
        self.create_memory_comparison_plot(df, scene, output_dir)
        self.create_efficiency_comparison_plot(df, scene, output_dir)
        self.create_quality_metrics_detailed(df, scene, output_dir)
        self.create_training_progression_overview(df, scene, output_dir)
        
        print(f"\n✅ All enhanced visualizations created for {scene}!")
    
    def analyze_all_scenes(self):
        """Generate enhanced visualizations for all scenes."""
        scenes = []
        syn_sh16_path = self.base_path / "syn_sh16"
        
        if syn_sh16_path.exists():
            for item in syn_sh16_path.iterdir():
                if item.is_dir() and (item / "analysis" / f"{item.name}_efficiency_metrics.csv").exists():
                    scenes.append(item.name)
        
        if not scenes:
            print("❌ No scenes found!")
            return
        
        print(f"🚀 Generating ENHANCED visualizations for {len(scenes)} scenes...")
        print(f"Scenes: {scenes}\n")
        
        for scene in sorted(scenes):
            try:
                self.analyze_scene(scene)
            except Exception as e:
                print(f"❌ Error analyzing {scene}: {e}")
                import traceback
                traceback.print_exc()
        
        print(f"\n🎉 ENHANCED visualization suite complete!")
        print(f"\n📁 New visualizations per scene:")
        print(f"  • {'{scene}'}_detailed_memory_analysis.png")
        print(f"  • {'{scene}'}_efficiency_comparison_detailed.png")
        print(f"  • {'{scene}'}_quality_metrics_detailed.png")
        print(f"  • {'{scene}'}_complete_training_progression.png")
        print(f"\n💡 Total: 4 additional charts × {len(scenes)} scenes = {4 * len(scenes)} new visualizations!")


def main():
    """Run the enhanced scene analyzer."""
    analyzer = EnhancedSceneAnalyzer()
    analyzer.analyze_all_scenes()


if __name__ == "__main__":
    main()

