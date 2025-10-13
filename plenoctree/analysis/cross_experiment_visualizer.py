#!/usr/bin/env python3
"""
Cross-Experiment Visualizer

Creates beautiful overlay visualizations comparing all experiments/scenes.
Produces publication-quality charts in the root analysis folder.
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


class CrossExperimentVisualizer:
    """Creates beautiful cross-experiment comparison visualizations."""
    
    def __init__(self, base_path: str = "/mnt/d/GitHub/nerf-projects/plenoctree/data/Plenoctree/checkpoints"):
        self.base_path = Path(base_path)
        self.output_dir = self.base_path / "analysis"
        self.output_dir.mkdir(exist_ok=True)
        
        # Use unified theme colors
        self.scene_colors = Colors.SCENES
        
    def get_comparison_memory(self, metrics: Dict) -> float:
        """Get the most reliable memory metric for cross-experiment comparison."""
        # Priority order for accuracy and consistency
        if 'nvidia_smi_used_gb' in metrics and metrics['nvidia_smi_used_gb'] > 0:
            return metrics['nvidia_smi_used_gb']
        elif 'gpu_reserved_gb' in metrics and metrics['gpu_reserved_gb'] > 0:
            return metrics['gpu_reserved_gb']
        elif 'gpu_allocated_gb' in metrics and metrics['gpu_allocated_gb'] > 0:
            return metrics['gpu_allocated_gb']
        else:
            return metrics.get('process_rss_gb', 0.001)
    
    def discover_scenes(self, config_name="syn_sh16") -> List[str]:
        """Discover all available scenes."""
        config_path = self.base_path / config_name
        scenes = []
        if config_path.exists():
            for item in config_path.iterdir():
                if item.is_dir() and (item / "full_pipeline_metrics.json").exists():
                    scenes.append(item.name)
        print(f"🔍 Found {len(scenes)} scenes: {scenes}")
        return sorted(scenes)
    
    def load_final_metrics(self, scene: str, config_name="syn_sh16") -> Optional[Dict]:
        """Load final evaluation metrics for a scene."""
        scene_path = self.base_path / config_name / scene
        
        # Try to load octree evaluation files (prioritize regular evaluation for complete data)
        eval_files = [
            scene_path / "octrees" / "octree_evaluation_metrics.json",  # Has max_depth
            scene_path / "octrees" / "octree_compression_evaluation_metrics.json",
            scene_path / "octree_compression_evaluation_metrics.json",
            scene_path / "nerf_evaluation_final.json",
        ]
        
        for eval_file in eval_files:
            if eval_file.exists():
                try:
                    with open(eval_file) as f:
                        data = json.load(f)
                    
                    # Handle list or dict format
                    if isinstance(data, list) and len(data) > 0:
                        entry = data[0]
                    else:
                        entry = data
                    
                    if 'metrics' in entry:
                        additional_info = entry.get('additional_info', {})
                        return {
                            'scene': scene,
                            'psnr': entry['metrics']['psnr'],
                            'ssim': entry['metrics']['ssim'],
                            'lpips': entry['metrics']['lpips'],
                            'memory_gb': self.get_comparison_memory(entry['metrics']),
                            'octree_capacity': additional_info.get('octree_capacity', 0),
                            'octree_file_size_mb': additional_info.get('octree_file_size_mb', 0),
                            'init_grid_depth': additional_info.get('max_depth', additional_info.get('init_grid_depth', 0)),
                            'source_file': eval_file.name
                        }
                except Exception as e:
                    print(f"⚠️ Error loading {eval_file}: {e}")
                    continue
        
        print(f"❌ No evaluation data found for {scene}")
        return None
    
    def collect_all_scene_data(self) -> pd.DataFrame:
        """Collect final metrics from all scenes."""
        scenes = self.discover_scenes()
        all_data = []
        
        for scene in scenes:
            metrics = self.load_final_metrics(scene)
            if metrics:
                # Calculate efficiency metrics
                mei = metrics['psnr'] / metrics['memory_gb']
                qmt = (metrics['psnr'] * metrics['ssim']) / metrics['memory_gb']
                
                # Calculate MEPV if voxel count available (HIGHER IS BETTER)
                mepv = None
                mepv_quality = None
                if metrics['octree_capacity'] and metrics['octree_capacity'] > 0:
                    mepv = (metrics['octree_capacity'] / 1e6) / metrics['memory_gb']
                    mepv_quality = metrics['psnr'] * (metrics['octree_capacity'] / 1e6) / metrics['memory_gb']
                
                # Calculate new storage-aware metrics
                storage_aware_mei = None
                voxel_density_efficiency = None
                if (metrics.get('octree_capacity') and metrics.get('octree_file_size_mb') and 
                    metrics.get('init_grid_depth')):
                    
                    # Calculate compression ratio and occupancy ratio
                    init_grid_depth = metrics['init_grid_depth']
                    reso = 2 ** (init_grid_depth + 1)
                    total_possible_voxels = reso ** 3
                    compression_ratio = total_possible_voxels / metrics['octree_capacity']
                    occupancy_ratio = metrics['octree_capacity'] / total_possible_voxels
                    storage_size_gb = metrics['octree_file_size_mb'] / 1024
                    
                    if storage_size_gb > 0:
                        # Storage-Aware Memory Efficiency Index (MEI) - CORRECT FORMULA
                        import math
                        log_compression = math.log10(max(compression_ratio, 1.0))
                        storage_aware_mei = (metrics['psnr'] * log_compression) / storage_size_gb
                        
                        # Voxel Density Efficiency (VDE) - unchanged
                        voxel_density_efficiency = (metrics['psnr'] * occupancy_ratio) / storage_size_gb
                
                metrics.update({
                    'mei': mei,
                    'qmt': qmt,
                    'mepv': mepv,
                    'mepv_quality': mepv_quality,
                    'storage_aware_mei': storage_aware_mei,
                    'voxel_density_efficiency': voxel_density_efficiency,
                })
                all_data.append(metrics)
        
        df = pd.DataFrame(all_data)
        
        # Save to CSV
        csv_path = self.output_dir / "cross_experiment_comparison.csv"
        df.to_csv(csv_path, index=False)
        print(f"💾 Cross-experiment data saved: {csv_path}")
        
        return df
    
    def create_quality_metrics_overlay(self, df: pd.DataFrame):
        """Create beautiful overlay plot of PSNR, SSIM, and LPIPS across all scenes."""
        fig, axes = plt.subplots(1, 3, figsize=(22, 7))
        fig.suptitle('Quality Metrics Comparison Across All Scenes', 
                     fontsize=22, fontweight='bold', y=1.02)
        
        scenes = df['scene'].tolist()
        x_pos = np.arange(len(scenes))
        
        # 1. PSNR Comparison
        ax1 = axes[0]
        bars1 = ax1.bar(x_pos, df['psnr'], color=[self.scene_colors.get(s, '#95A5A6') for s in scenes],
                       alpha=0.85, edgecolor='white', linewidth=2)
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars1, df['psnr'])):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{val:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        ax1.set_title('PSNR Quality Metric', fontweight='bold', pad=15, fontsize=16)
        ax1.set_ylabel('PSNR (dB) - Higher is Better', fontweight='bold', fontsize=13)
        ax1.set_xlabel('Scene', fontweight='bold')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(scenes, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # 2. SSIM Comparison
        ax2 = axes[1]
        bars2 = ax2.bar(x_pos, df['ssim'], color=[self.scene_colors.get(s, '#95A5A6') for s in scenes],
                       alpha=0.85, edgecolor='white', linewidth=2)
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars2, df['ssim'])):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{val:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        ax2.set_title('SSIM Quality Metric', fontweight='bold', pad=15, fontsize=16)
        ax2.set_ylabel('SSIM - Higher is Better', fontweight='bold', fontsize=13)
        ax2.set_xlabel('Scene', fontweight='bold')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(scenes, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 3. LPIPS Comparison
        ax3 = axes[2]
        bars3 = ax3.bar(x_pos, df['lpips'], color=[self.scene_colors.get(s, '#95A5A6') for s in scenes],
                       alpha=0.85, edgecolor='white', linewidth=2)
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars3, df['lpips'])):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                    f'{val:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        ax3.set_title('LPIPS Perceptual Metric', fontweight='bold', pad=15, fontsize=16)
        ax3.set_ylabel('LPIPS - Lower is Better', fontweight='bold', fontsize=13)
        ax3.set_xlabel('Scene', fontweight='bold')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(scenes, rotation=45, ha='right')
        ax3.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        output_path = self.output_dir / "quality_metrics_comparison.png"
        fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"✅ Quality metrics comparison saved: {output_path}")
        
    def create_memory_efficiency_overlay(self, df: pd.DataFrame):
        """Create overlay plot showing the two new storage-focused efficiency metrics with LPIPS color scheme."""
        fig, axes = plt.subplots(1, 2, figsize=(18, 9))
        fig.suptitle('Storage-Aware Efficiency Metrics Comparison Across All Scenes', 
                     fontsize=18, fontweight='bold', y=0.95)
        
        scenes = df['scene'].tolist()
        x_pos = np.arange(len(scenes))
        
        # Import Colors for LPIPS color scheme
        from visualization_theme import Colors
        storage_mei_color = Colors.LPIPS  # '#F77F00' - Orange (same as LPIPS)
        vde_color = '#FF9500'  # Slightly different orange for contrast
        
        # 1. Storage-Aware Memory Efficiency Index (MEI)
        ax1 = axes[0]
        if 'storage_aware_mei' in df.columns and df['storage_aware_mei'].notna().any():
            bars1 = ax1.bar(x_pos, df['storage_aware_mei'], 
                           color=storage_mei_color, alpha=0.8, edgecolor='white', linewidth=2)
            
            # Add gradient effect by varying alpha
            for i, bar in enumerate(bars1):
                bar.set_alpha(0.7 + 0.3 * (i / len(bars1)))
            
            for bar, val in zip(bars1, df['storage_aware_mei']):
                if pd.notna(val):
                    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + bar.get_height()*0.02,
                            f'{val:.0f}', ha='center', va='bottom', fontweight='bold', fontsize=11, color='white',
                            bbox=dict(boxstyle='round,pad=0.2', facecolor=storage_mei_color, alpha=0.8))
            
            ax1.set_title('Storage-Aware Memory Efficiency Index\n(PSNR × log₁₀(Compression Ratio)) / Storage GB ↑', 
                         fontweight='bold', pad=10, fontsize=12)
            ax1.set_ylabel('Storage-Aware MEI', fontweight='bold', fontsize=13)
        else:
            ax1.text(0.5, 0.5, 'Storage-Aware MEI\n\nRequires octree evaluation\nwith file size data', 
                    ha='center', va='center', transform=ax1.transAxes,
                    fontsize=14, fontweight='bold', color='gray',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor='#F8F9FA', edgecolor='gray'))
            ax1.set_title('Storage-Aware MEI (Not Available)', fontweight='bold', pad=20, fontsize=15)
        
        ax1.set_xlabel('Scene', fontweight='bold', fontsize=13)
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(scenes, rotation=45, ha='right', fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y', linestyle='--')
        ax1.set_facecolor('#FAFBFC')
        
        # 2. Voxel Density Efficiency (VDE)
        ax2 = axes[1]
        if 'voxel_density_efficiency' in df.columns and df['voxel_density_efficiency'].notna().any():
            bars2 = ax2.bar(x_pos, df['voxel_density_efficiency'], 
                           color=vde_color, alpha=0.8, edgecolor='white', linewidth=2)
            
            # Add gradient effect by varying alpha
            for i, bar in enumerate(bars2):
                bar.set_alpha(0.7 + 0.3 * (i / len(bars2)))
            
            for bar, val in zip(bars2, df['voxel_density_efficiency']):
                if pd.notna(val):
                    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + bar.get_height()*0.02,
                            f'{val:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=11, color='white',
                            bbox=dict(boxstyle='round,pad=0.2', facecolor=vde_color, alpha=0.8))
            
            ax2.set_title('Voxel Density Efficiency\n(PSNR × Occupancy Ratio) / Storage GB ↑', 
                         fontweight='bold', pad=10, fontsize=12)
            ax2.set_ylabel('Voxel Density Efficiency', fontweight='bold', fontsize=13)
        else:
            ax2.text(0.5, 0.5, 'Voxel Density Efficiency\n\nRequires octree evaluation\nwith voxel density data', 
                    ha='center', va='center', transform=ax2.transAxes,
                    fontsize=14, fontweight='bold', color='gray',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor='#F8F9FA', edgecolor='gray'))
            ax2.set_title('VDE (Not Available)', fontweight='bold', pad=20, fontsize=15)
        
        ax2.set_xlabel('Scene', fontweight='bold', fontsize=13)
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(scenes, rotation=45, ha='right', fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y', linestyle='--')
        ax2.set_facecolor('#FAFBFC')
        
        # Add explanatory text at the bottom
        explanation = """Storage-Aware MEI measures quality×compression efficiency per GB. VDE measures quality×voxel density per GB. Higher values indicate better storage efficiency."""
        fig.text(0.5, 0.02, explanation, ha='center', va='bottom', fontsize=11, 
                style='italic', color='#666666')
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.85, bottom=0.15, left=0.08, right=0.95, wspace=0.25)
        
        output_path = self.output_dir / "storage_efficiency_comparison.png"
        fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"✅ Storage-aware efficiency comparison saved: {output_path}")
    
    def create_scatter_overlay(self, df: pd.DataFrame):
        """Create scatter plot showing quality vs memory tradeoff."""
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        fig.suptitle('Quality vs Memory Tradeoff Analysis', 
                     fontsize=22, fontweight='bold', y=1.02)
        
        # 1. PSNR vs Memory
        ax1 = axes[0]
        for _, row in df.iterrows():
            color = self.scene_colors.get(row['scene'], '#95A5A6')
            ax1.scatter(row['memory_gb'], row['psnr'], s=300, color=color, 
                       alpha=0.7, edgecolors='white', linewidth=2, label=row['scene'])
            ax1.text(row['memory_gb'], row['psnr'] + 0.3, row['scene'], 
                    ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        ax1.set_title('PSNR vs Memory Usage\nTop-right quadrant = Best (High Quality, Low Memory)', 
                     fontweight='bold', pad=15, fontsize=16)
        ax1.set_xlabel('Memory Usage (GB)', fontweight='bold')
        ax1.set_ylabel('PSNR (dB)', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # 2. Combined Quality vs Memory
        ax2 = axes[1]
        # Combined quality = PSNR * SSIM * (1 - LPIPS)
        df['combined_quality'] = df['psnr'] * df['ssim'] * (1 - df['lpips'])
        
        for _, row in df.iterrows():
            color = self.scene_colors.get(row['scene'], '#95A5A6')
            ax2.scatter(row['memory_gb'], row['combined_quality'], s=300, color=color, 
                       alpha=0.7, edgecolors='white', linewidth=2)
            ax2.text(row['memory_gb'], row['combined_quality'] + 0.5, row['scene'], 
                    ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        ax2.set_title('Combined Quality vs Memory\n(PSNR x SSIM x (1-LPIPS)) - Top-left is Best', 
                     fontweight='bold', pad=15, fontsize=16)
        ax2.set_xlabel('Memory Usage (GB)', fontweight='bold')
        ax2.set_ylabel('Combined Quality Score', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        output_path = self.output_dir / "quality_memory_tradeoff.png"
        fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"✅ Quality-memory tradeoff saved: {output_path}")
    
    def create_radar_chart(self, df: pd.DataFrame):
        """Create radar chart comparing normalized metrics across scenes."""
        # Normalize metrics to 0-1 scale
        metrics_to_plot = ['psnr', 'ssim', 'mei', 'qmt']
        
        # Normalize each metric
        df_norm = df.copy()
        for metric in metrics_to_plot:
            if metric in df.columns and df[metric].notna().any():
                min_val = df[metric].min()
                max_val = df[metric].max()
                if max_val > min_val:
                    df_norm[metric] = (df[metric] - min_val) / (max_val - min_val)
                else:
                    df_norm[metric] = 0.5
        
        # For LPIPS, invert (lower is better)
        if 'lpips' in df.columns:
            min_val = df['lpips'].min()
            max_val = df['lpips'].max()
            if max_val > min_val:
                df_norm['lpips_inv'] = 1 - (df['lpips'] - min_val) / (max_val - min_val)
            else:
                df_norm['lpips_inv'] = 0.5
        
        # Create radar chart for each scene
        num_scenes = len(df)
        cols = 4
        rows = (num_scenes + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(20, 5*rows), 
                                subplot_kw=dict(projection='polar'))
        fig.suptitle('Multi-Metric Performance Comparison (Normalized)', 
                     fontsize=22, fontweight='bold', y=0.995)
        
        if num_scenes == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        categories = ['PSNR', 'SSIM', 'LPIPS\n(inverted)', 'MEI', 'QMT']
        num_vars = len(categories)
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        for idx, (_, row) in enumerate(df_norm.iterrows()):
            ax = axes[idx]
            scene = row['scene']
            color = self.scene_colors.get(scene, '#95A5A6')
            
            values = [
                row.get('psnr', 0),
                row.get('ssim', 0),
                row.get('lpips_inv', 0),
                row.get('mei', 0),
                row.get('qmt', 0),
            ]
            values += values[:1]  # Complete the circle
            
            ax.plot(angles, values, 'o-', linewidth=3, color=color, label=scene)
            ax.fill(angles, values, alpha=0.25, color=color)
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories, fontsize=10)
            ax.set_ylim(0, 1)
            ax.set_title(scene.upper(), fontweight='bold', pad=20, fontsize=14)
            ax.grid(True, alpha=0.3)
        
        # Hide extra subplots
        for idx in range(num_scenes, len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        
        output_path = self.output_dir / "radar_comparison.png"
        fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"✅ Radar comparison saved: {output_path}")
    
    def create_summary_table(self, df: pd.DataFrame):
        """Create a beautiful summary table visualization."""
        fig, ax = plt.subplots(figsize=(20, 12))
        fig.suptitle('Cross-Experiment Summary Table', 
                     fontsize=22, fontweight='bold', y=0.98)
        
        ax.axis('off')
        
        # Prepare table data
        table_data = []
        for _, row in df.iterrows():
            mepv_str = f"{row['mepv']:.2f}" if pd.notna(row['mepv']) else "N/A"
            table_data.append([
                row['scene'],
                f"{row['psnr']:.2f}",
                f"{row['ssim']:.4f}",
                f"{row['lpips']:.4f}",
                f"{row['memory_gb']:.2f}",
                f"{row['mei']:.1f}",
                f"{row['qmt']:.1f}",
                mepv_str,
            ])
        
        # Create table
        table = ax.table(cellText=table_data,
                        colLabels=['Scene', 'PSNR (dB)\nHigher Better', 'SSIM\nHigher Better', 'LPIPS\nLower Better', 
                                  'Memory (GB)\nLower Better', 'MEI\nHigher Better', 'QMT\nHigher Better', 'MEPV\nHigher Better'],
                        cellLoc='center',
                        loc='center',
                        colWidths=[0.15, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.13])
        
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1, 3)
        
        # Style the table
        for i in range(len(table_data) + 1):
            for j in range(8):
                cell = table[(i, j)]
                if i == 0:  # Header row
                    cell.set_facecolor('#2E86AB')
                    cell.set_text_props(weight='bold', color='white', fontsize=13)
                else:  # Data rows
                    scene = table_data[i-1][0]
                    color = self.scene_colors.get(scene, '#FFFFFF')
                    if j == 0:  # Scene name column
                        cell.set_facecolor(color)
                        cell.set_text_props(weight='bold', fontsize=12)
                    else:
                        # Alternate row colors
                        if i % 2 == 1:
                            cell.set_facecolor('#F8F9FA')
                        else:
                            cell.set_facecolor('#FFFFFF')
                        cell.set_text_props(fontsize=11)
                
                cell.set_edgecolor('#DDDDDD')
                cell.set_linewidth(1.5)
        
        # Add legend for arrows
        legend_text = """
        Legend:
        
        MEI  = Memory Efficiency Index (PSNR / Memory GB)
        QMT  = Quality-Memory Tradeoff (PSNR x SSIM / Memory GB)
        MEPV = Memory Efficiency Per Voxel (Million Voxels / GB)
        
        All efficiency metrics: Higher values indicate better performance
        LPIPS: Lower values indicate better perceptual quality
        """
        
        ax.text(0.5, 0.05, legend_text, 
               horizontalalignment='center', verticalalignment='center',
               transform=ax.transAxes, fontsize=11, family='monospace',
               bbox=dict(boxstyle="round,pad=0.8", facecolor='#F8F9FA', 
                        edgecolor='#2E86AB', linewidth=2))
        
        plt.tight_layout()
        
        output_path = self.output_dir / "summary_table.png"
        fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"✅ Summary table saved: {output_path}")
    
    def print_methodology_note(self):
        """Print methodology documentation."""
        print("")
        print("="*80)
        print("📋 CROSS-EXPERIMENT ANALYSIS METHODOLOGY")
        print("="*80)
        print("")
        print("Memory Source Priority (for cross-experiment comparison):")
        print("  1. nvidia-smi GPU memory (system-level, most accurate)")
        print("  2. GPU reserved memory (PyTorch memory pool)")
        print("  3. GPU allocated memory (active tensors only)")
        print("  4. Process RSS (CPU fallback)")
        print("")
        print("Why this matters:")
        print("  • nvidia-smi shows actual GPU RAM usage (closest to 'true' usage)")
        print("  • GPU allocated may underestimate (doesn't include PyTorch overhead)")
        print("  • Consistent across different GPU models (12GB vs 16GB vs 24GB)")
        print("")
        print("Key Metrics (all measured at Final Compressed Octree Evaluation):")
        print("  • PSNR: Peak Signal-to-Noise Ratio (dB) - Higher is better")
        print("  • SSIM: Structural Similarity Index (0-1) - Higher is better")
        print("  • LPIPS: Learned Perceptual Image Patch Similarity - Lower is better")
        print("  • MEI: Memory Efficiency Index (PSNR / Memory GB) - Higher is better")
        print("  • QMT: Quality-Memory Tradeoff (PSNR × SSIM / Memory GB) - Higher is better")
        print("  • MEPV: Memory Efficiency Per Voxel (MVoxels / GB) ⬆ HIGHER IS BETTER")
        print("  • Quality MEPV: PSNR × MVoxels / GB ⬆ HIGHER IS BETTER")
        print("")
        print("All comparisons use:")
        print("  ✓ Same measurement methodology across all scenes")
        print("  ✓ Final evaluation stage for fair comparison")
        print("  ✓ Hardware-independent metrics (ratios, not absolute values)")
        print("")
        print("="*80)
        print("")
    
    def generate_all_visualizations(self):
        """Generate all cross-experiment visualizations."""
        print("\n🎨 Cross-Experiment Visualization Suite\n")
        
        # Print methodology
        self.print_methodology_note()
        
        # Collect data from all scenes
        print("📊 Collecting data from all scenes...")
        df = self.collect_all_scene_data()
        
        if len(df) == 0:
            print("❌ No data found! Cannot create visualizations.")
            return
        
        print(f"\n✅ Collected data from {len(df)} scenes\n")
        
        # Generate visualizations
        print("🎨 Generating visualizations...\n")
        
        self.create_quality_metrics_overlay(df)
        self.create_memory_efficiency_overlay(df)
        self.create_scatter_overlay(df)
        self.create_radar_chart(df)
        self.create_summary_table(df)
        
        print(f"\n✅ All visualizations saved to: {self.output_dir}")
        print("\nGenerated files:")
        print("  • quality_metrics_comparison.png")
        print("  • storage_efficiency_comparison.png")
        print("  • quality_memory_tradeoff.png")
        print("  • radar_comparison.png")
        print("  • summary_table.png")
        print("  • cross_experiment_comparison.csv")
        print("\n🎉 Cross-experiment analysis complete!")


def main():
    """Run the cross-experiment visualizer."""
    visualizer = CrossExperimentVisualizer()
    visualizer.generate_all_visualizations()


if __name__ == "__main__":
    main()

