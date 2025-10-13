#!/usr/bin/env python3
"""
Enhanced Memory Efficiency Metrics Analyzer

Analyzes memory efficiency metrics including proper MSF calculation when voxel counts are available.
Can work with existing training data and enhance it with octree evaluation results.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
from typing import Dict, List, Optional, Tuple
import warnings

warnings.filterwarnings('ignore')

# Import unified theme
from visualization_theme import (Colors, Typography, PlotElements, 
                                 PlotTemplates, get_scene_color, get_metric_color)

class EfficiencyMetricsAnalyzer:
    """Enhanced analyzer for memory efficiency metrics including proper MSF calculation."""
    
    def __init__(self, base_path: str = "/mnt/d/GitHub/nerf-projects/plenoctree/data/Plenoctree/checkpoints"):
        self.base_path = Path(base_path)
    
    def get_comparison_memory(self, stage_data: Dict) -> float:
        """Get the most reliable memory metric for cross-experiment comparison.
        
        Priority order for accuracy and consistency:
        1. nvidia-smi GPU memory (system-level, most accurate)
        2. GPU reserved memory (PyTorch memory pool)
        3. GPU allocated memory (active tensors only)
        4. Process RSS (CPU fallback)
        """
        # Priority 1: nvidia-smi (most accurate)
        if 'nvidia_smi_used_gb' in stage_data and stage_data['nvidia_smi_used_gb'] > 0:
            return stage_data['nvidia_smi_used_gb']
        # Priority 2: GPU reserved memory
        elif 'gpu_reserved_gb' in stage_data and stage_data['gpu_reserved_gb'] > 0:
            return stage_data['gpu_reserved_gb']
        # Priority 3: GPU allocated memory
        elif 'gpu_allocated_gb' in stage_data and stage_data['gpu_allocated_gb'] > 0:
            return stage_data['gpu_allocated_gb']
        # Priority 4: Process memory (fallback)
        else:
            return stage_data.get('process_rss_gb', 0.001)
        
    def load_training_metrics(self, scene_path: Path) -> List[Dict]:
        """Load NeRF training metrics from JSON file."""
        metrics_file = scene_path / "nerf_training_metrics.json"
        if not metrics_file.exists():
            print(f"⚠️ No training metrics found at {metrics_file}")
            return []
            
        try:
            with open(metrics_file, 'r') as f:
                data = json.load(f)
            print(f"✅ Loaded {len(data)} training steps from {metrics_file.name}")
            return data
        except Exception as e:
            print(f"❌ Error loading {metrics_file}: {e}")
            return []
    
    def load_octree_evaluation(self, scene_path: Path) -> Optional[Dict]:
        """Load octree evaluation metrics that contain voxel counts and storage metrics."""
        # Check multiple possible locations for octree evaluation data
        possible_files = [
            scene_path / "octrees" / "octree_evaluation_metrics.json",  # Most likely location
            scene_path / "octrees" / "octree_compression_evaluation_metrics.json",  # Compressed evaluation
            scene_path / "octree_evaluation_final.json",
            scene_path / "octrees" / "octree_evaluation_final.json", 
            scene_path / "nerf_evaluation_final.json",
            scene_path / "octree_optimization_metrics.json",
            scene_path / "octrees" / "octree_optimization_metrics.json"
        ]
        
        for eval_file in possible_files:
            if eval_file.exists():
                try:
                    with open(eval_file, 'r') as f:
                        data = json.load(f)
                    
                    # Check if this file contains octree capacity information
                    if isinstance(data, list) and len(data) > 0:
                        # Check the last entry for capacity info
                        last_entry = data[-1]
                        if 'additional_info' in last_entry and 'octree_capacity' in last_entry['additional_info']:
                            capacity = last_entry['additional_info']['octree_capacity']
                            print(f"✅ Found octree capacity: {capacity:,} nodes in {eval_file.name}")
                            # Check for storage metrics
                            if 'storage_aware_mei' in last_entry['additional_info']:
                                print(f"✅ Found storage metrics in {eval_file.name}")
                            return last_entry
                    elif isinstance(data, dict):
                        if 'octree_capacity' in data or ('additional_info' in data and 'octree_capacity' in data['additional_info']):
                            capacity = data.get('octree_capacity', data.get('additional_info', {}).get('octree_capacity', 0))
                            print(f"✅ Found octree capacity: {capacity:,} nodes in {eval_file.name}")
                            # Check for storage metrics
                            if data.get('additional_info', {}).get('storage_aware_mei'):
                                print(f"✅ Found storage metrics in {eval_file.name}")
                            return data
                            
                except Exception as e:
                    print(f"⚠️ Error reading {eval_file}: {e}")
                    continue
        
        print(f"⚠️ No octree evaluation data with capacity found for {scene_path.name}")
        return None
    
    def calculate_enhanced_efficiency_metrics(self, training_data: List[Dict], 
                                            octree_capacity: Optional[int] = None,
                                            octree_eval_data: Optional[Dict] = None) -> pd.DataFrame:
        """Calculate enhanced efficiency metrics including proper MSF when voxel count is available."""
        
        metrics_list = []
        
        for entry in training_data:
            step = entry.get('step', 0)
            metrics = entry.get('metrics', {})
            additional_info = entry.get('additional_info', {})
            
            # Extract quality metrics
            psnr = metrics.get('psnr', 0)
            ssim = None
            lpips = None
            
            # Try to find SSIM/LPIPS in various places
            if 'ssim' in metrics:
                ssim = metrics['ssim']
            elif 'ssim' in additional_info:
                ssim = additional_info['ssim']
                
            if 'lpips' in metrics:
                lpips = metrics['lpips']
            elif 'lpips' in additional_info:
                lpips = additional_info['lpips']
            
            # Extract memory usage - prioritize nvidia-smi for accuracy
            # Use nvidia-smi > gpu_reserved > gpu_allocated > process memory
            memory_gb = None
            peak_memory_gb = None
            
            # Check for nvidia-smi memory (most accurate)
            if metrics.get('nvidia_smi_used_gb') and metrics.get('nvidia_smi_used_gb') > 0:
                memory_gb = metrics.get('nvidia_smi_used_gb')
                peak_memory_gb = max(memory_gb, metrics.get('peak_gpu_allocated_gb', memory_gb))
            # Fallback to GPU reserved memory
            elif metrics.get('gpu_reserved_gb') and metrics.get('gpu_reserved_gb') > 0:
                memory_gb = metrics.get('gpu_reserved_gb')
                peak_memory_gb = metrics.get('peak_gpu_reserved_gb', memory_gb)
            # Fallback to GPU allocated memory
            elif metrics.get('gpu_allocated_gb') and metrics.get('gpu_allocated_gb') > 0:
                memory_gb = metrics.get('gpu_allocated_gb')
                peak_memory_gb = metrics.get('peak_gpu_allocated_gb', memory_gb)
            # Last resort: process memory
            else:
                memory_gb = (metrics.get('process_rss_gb') or 
                           metrics.get('system_used_gb', 0))
                peak_memory_gb = (metrics.get('peak_process_rss_gb') or 
                                metrics.get('peak_system_used_gb', memory_gb))
            
            if memory_gb <= 0:
                memory_gb = 0.001  # Avoid division by zero
            if peak_memory_gb <= 0:
                peak_memory_gb = memory_gb
            
            # Calculate existing efficiency metrics
            mei = psnr / memory_gb
            peak_mei = psnr / peak_memory_gb
            
            # Original (incorrect) MSF - just memory usage (lower is better)
            msf_original = memory_gb
            
            # Proper MEPV (Memory Efficiency Per Voxel) - HIGHER IS BETTER
            # Inverted from MSF to make "higher = more efficient"
            mepv = None
            mepv_quality = None
            if octree_capacity and octree_capacity > 0:
                # Million voxels per GB - higher is better
                mepv = (octree_capacity / 1e6) / memory_gb
                # Quality-weighted efficiency - higher is better
                mepv_quality = psnr * (octree_capacity / 1e6) / memory_gb
            
            # QMT calculation
            qmt = None
            peak_qmt = None
            if ssim is not None:
                qmt = (psnr * ssim) / memory_gb
                peak_qmt = (psnr * ssim) / peak_memory_gb
            
            # Combined quality index
            combined_quality = psnr
            if ssim is not None:
                combined_quality *= ssim
            if lpips is not None:
                combined_quality *= (1.0 - lpips)
            
            combined_efficiency = combined_quality / memory_gb
            peak_combined_efficiency = combined_quality / peak_memory_gb
            
            # Add storage-aware metrics if available from octree evaluation
            storage_aware_mei = None
            voxel_density_efficiency = None
            storage_efficiency = None
            
            if octree_eval_data and 'additional_info' in octree_eval_data:
                additional_info = octree_eval_data['additional_info']
                storage_aware_mei = additional_info.get('storage_aware_mei')
                voxel_density_efficiency = additional_info.get('voxel_density_efficiency')
                storage_efficiency = additional_info.get('storage_efficiency')
            
            row = {
                'step': step,
                'psnr': psnr,
                'ssim': ssim,
                'lpips': lpips,
                'memory_gb': memory_gb,
                'peak_memory_gb': peak_memory_gb,
                'mei': mei,
                'peak_mei': peak_mei,
                'msf_original': msf_original,
                'mepv': mepv,  # NEW: Higher is better
                'mepv_quality': mepv_quality,  # NEW: Quality-weighted efficiency
                'qmt': qmt,
                'peak_qmt': peak_qmt,
                'combined_quality': combined_quality,
                'combined_efficiency': combined_efficiency,
                'peak_combined_efficiency': peak_combined_efficiency,
                'octree_capacity': octree_capacity,
                # NEW: Storage-aware metrics
                'storage_aware_mei': storage_aware_mei,
                'voxel_density_efficiency': voxel_density_efficiency,
                'storage_efficiency': storage_efficiency
            }
            
            metrics_list.append(row)
        
        return pd.DataFrame(metrics_list)
    
    def create_efficiency_plots(self, df: pd.DataFrame, scene: str, output_dir: Path):
        """Create comprehensive efficiency metrics plots."""
        
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create figure with subplots for efficiency metrics
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Memory Efficiency Analysis - {scene.upper()}', fontsize=18, fontweight='bold')
        
        # 1. Memory Efficiency Index (MEI)
        ax1 = axes[0, 0]
        ax1.plot(df['step'], df['mei'], 'b-', linewidth=2, label='Current MEI')
        ax1.plot(df['step'], df['peak_mei'], 'b--', linewidth=2, alpha=0.7, label='Peak MEI')
        ax1.set_title('Memory Efficiency Index (MEI)\nPSNR per GB - Higher is Better', fontweight='bold', fontsize=13)
        ax1.set_xlabel('Training Step', fontweight='bold', fontsize=11)
        ax1.set_ylabel('MEI (PSNR/GB)', fontweight='bold', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. MEPV (Memory Efficiency Per Voxel) - HIGHER IS BETTER
        ax2 = axes[0, 1]
        if df['mepv'].notna().any():
            # Show both memory and MEPV on dual axes
            ax2.plot(df['step'], df['msf_original'], 'r-', linewidth=2.5, label='Memory Usage', alpha=0.7)
            ax2.set_ylabel('Memory Usage (GB)', color='#D32F2F', fontweight='bold', fontsize=12)
            ax2.tick_params(axis='y', labelcolor='#D32F2F')
            
            ax2_twin = ax2.twinx()
            ax2_twin.plot(df['step'], df['mepv'], 'g-', linewidth=3, label='MEPV', marker='o', markersize=6)
            ax2_twin.set_ylabel('MEPV (MVoxels/GB)\nHigher is Better', fontweight='bold', color='#2E7D32', fontsize=12)
            ax2_twin.tick_params(axis='y', labelcolor='#2E7D32')
            ax2.set_title('Memory Efficiency Per Voxel (MEPV)\nMillion Voxels per GB - Higher is Better', fontweight='bold', fontsize=13)
            
            # Combine legends
            lines1, labels1 = ax2.get_legend_handles_labels()
            lines2, labels2 = ax2_twin.get_legend_handles_labels()
            ax2_twin.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10)
        else:
            ax2.plot(df['step'], df['msf_original'], 'r-', linewidth=2, label='Memory Usage')
            ax2.set_ylabel('Memory Usage (GB)', fontweight='bold', fontsize=12)
            ax2.set_title('Memory Usage\n(MEPV requires voxel count)', fontweight='bold', fontsize=13)
            ax2.legend(loc='upper right')
        ax2.set_xlabel('Training Step', fontweight='bold', fontsize=11)
        ax2.grid(True, alpha=0.3)
        
        # 3. Quality-Memory Trade-off (QMT)
        ax3 = axes[0, 2]
        if df['qmt'].notna().any():
            ax3.plot(df['step'], df['qmt'], 'purple', linewidth=2, label='Current QMT')
            ax3.plot(df['step'], df['peak_qmt'], 'purple', linewidth=2, alpha=0.7, linestyle='--', label='Peak QMT')
            ax3.set_ylabel('QMT (PSNR×SSIM/GB)')
        else:
            # Fallback to MEI if no SSIM available
            ax3.plot(df['step'], df['mei'], 'purple', linewidth=2, label='MEI (no SSIM)')
            ax3.set_ylabel('MEI (PSNR/GB)')
        ax3.set_title('Quality-Memory Tradeoff (QMT)\n(PSNR x SSIM) per GB - Higher is Better', fontweight='bold', fontsize=13)
        ax3.set_xlabel('Training Step', fontweight='bold', fontsize=11)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Combined Efficiency Evolution
        ax4 = axes[1, 0]
        ax4.plot(df['step'], df['combined_efficiency'], 'orange', linewidth=2, label='Combined Efficiency')
        ax4.plot(df['step'], df['peak_combined_efficiency'], 'orange', linewidth=2, alpha=0.7, linestyle='--', label='Peak Combined')
        ax4.set_title('Combined Quality Efficiency\n(PSNR x SSIM x (1-LPIPS)) per GB', fontweight='bold', fontsize=13)
        ax4.set_xlabel('Training Step', fontweight='bold', fontsize=11)
        ax4.set_ylabel('Combined Efficiency', fontweight='bold', fontsize=12)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Storage-Aware Memory Efficiency Index
        ax5 = axes[1, 1]
        # Check if we have storage-aware metrics from octree evaluation
        if hasattr(df, 'storage_aware_mei') or 'storage_aware_mei' in df.columns:
            if df['storage_aware_mei'].notna().any():
                ax5.plot(df['step'], df['storage_aware_mei'], 'teal', linewidth=3, label='Storage-Aware MEI', marker='s', markersize=6)
                ax5.set_ylabel('Storage-Aware MEI\n(PSNR × log(Compression)) / Storage GB', fontweight='bold', fontsize=12)
                ax5.set_title('Storage-Aware Memory Efficiency Index\nHigher is Better', fontweight='bold', fontsize=13)
                ax5.legend()
            else:
                ax5.text(0.5, 0.5, 'Storage-Aware MEI\nRequires octree evaluation\nwith file size data', 
                        ha='center', va='center', transform=ax5.transAxes, fontsize=12, fontweight='bold', color='gray')
                ax5.set_title('Storage-Aware MEI (Not Available)', fontweight='bold', fontsize=13)
        else:
            # Fallback to memory vs quality scatter
            scatter = ax5.scatter(df['memory_gb'], df['psnr'], c=df['step'], cmap='viridis', alpha=0.7)
            ax5.set_xlabel('Memory Usage (GB)')
            ax5.set_ylabel('PSNR')
            ax5.set_title('Memory vs Quality Tradeoff\n(Color = Training Step)', fontweight='bold', fontsize=13)
            plt.colorbar(scatter, ax=ax5, label='Training Step')
        ax5.set_xlabel('Training Step', fontweight='bold', fontsize=11)
        ax5.grid(True, alpha=0.3)
        
        # 6. Voxel Density Efficiency (VDE) - NEW!
        ax6 = axes[1, 2]
        # Check if we have VDE metrics from octree evaluation
        if hasattr(df, 'voxel_density_efficiency') or 'voxel_density_efficiency' in df.columns:
            if df['voxel_density_efficiency'].notna().any():
                ax6.plot(df['step'], df['voxel_density_efficiency'], 'darkgreen', linewidth=3, label='VDE', marker='d', markersize=6)
                ax6.set_ylabel('VDE\n(PSNR × Occupancy Ratio) / Storage GB', fontweight='bold', fontsize=12)
                ax6.set_title('Voxel Density Efficiency (VDE)\nHigher is Better', fontweight='bold', fontsize=13)
                ax6.legend()
                ax6.set_xlabel('Training Step', fontweight='bold', fontsize=11)
                ax6.grid(True, alpha=0.3)
            else:
                ax6.text(0.5, 0.5, 'Voxel Density Efficiency\nRequires octree evaluation\nwith voxel density data', 
                        ha='center', va='center', transform=ax6.transAxes, fontsize=12, fontweight='bold', color='gray')
                ax6.set_title('VDE (Not Available)', fontweight='bold', fontsize=13)
        else:
            # Show efficiency summary statistics as fallback
            ax6.axis('off')
            
            # Calculate summary statistics
            final_mei = df['mei'].iloc[-1] if len(df) > 0 else 0
            max_mei = df['mei'].max() if len(df) > 0 else 0
            final_memory = df['memory_gb'].iloc[-1] if len(df) > 0 else 0
            final_psnr = df['psnr'].iloc[-1] if len(df) > 0 else 0
            
            summary_text = f"""
EFFICIENCY SUMMARY

Final Metrics:
• PSNR: {final_psnr:.2f} dB
• Memory: {final_memory:.2f} GB
• MEI: {final_mei:.2f}

Peak Performance:
• Max MEI: {max_mei:.2f}
• Steps: {len(df):,}

Storage Metrics:
"""
            
            if df['mepv'].notna().any():
                final_mepv = df['mepv'].iloc[-1]
                summary_text += f"• MEPV: {final_mepv:.3f} MVoxels/GB\n"
                if df['mepv_quality'].notna().any():
                    final_mepv_quality = df['mepv_quality'].iloc[-1]
                    summary_text += f"• Quality MEPV: {final_mepv_quality:.1f}\n"
                if df['octree_capacity'].iloc[0]:
                    summary_text += f"• Voxel Count: {df['octree_capacity'].iloc[0]:,}\n"
            else:
                summary_text += "• Need octree evaluation\n  for storage metrics"
            
            ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes, fontsize=11,
                    verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5))
        
        plt.tight_layout()
        
        # Save the plot
        plot_path = output_dir / f"{scene}_efficiency_metrics.png"
        fig.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close(fig)
        
        print(f"📊 Efficiency metrics plot saved: {plot_path.name}")
        
        # Create separate storage metrics plot if data is available
        storage_plot_path = self.create_storage_metrics_plot(df, scene, output_dir)
        
        return plot_path
    
    def create_storage_metrics_plot(self, df: pd.DataFrame, scene: str, output_dir: Path):
        """Create dedicated storage metrics plot with Storage-Aware MEI and VDE side by side."""
        
        # Check if we have storage metrics data
        has_storage_mei = 'storage_aware_mei' in df.columns and df['storage_aware_mei'].notna().any()
        has_vde = 'voxel_density_efficiency' in df.columns and df['voxel_density_efficiency'].notna().any()
        
        if not (has_storage_mei or has_vde):
            print(f"⚠️  No storage metrics data available for {scene} - skipping storage plot")
            return None
        
        # Import visualization theme for consistent colors
        from visualization_theme import Colors
        
        # Set up the plotting style to match LPIPS visualization
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create figure with 1x2 subplots for side-by-side comparison
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        fig.suptitle(f'Storage-Aware Efficiency Metrics - {scene.upper()}', 
                     fontsize=20, fontweight='bold', y=0.95)
        
        # Color scheme matching LPIPS (orange theme with variations)
        storage_mei_color = Colors.LPIPS  # '#F77F00' - Orange (same as LPIPS)
        vde_color = '#FF9500'  # Slightly different orange for contrast
        
        # 1. Storage-Aware Memory Efficiency Index (Left)
        ax1 = axes[0]
        if has_storage_mei:
            ax1.plot(df['step'], df['storage_aware_mei'], color=storage_mei_color, 
                    linewidth=3, label='Storage-Aware MEI', marker='s', markersize=6, 
                    markerfacecolor='white', markeredgecolor=storage_mei_color, markeredgewidth=2)
            
            # Add gradient fill under the curve
            ax1.fill_between(df['step'], df['storage_aware_mei'], alpha=0.2, color=storage_mei_color)
            
            ax1.set_ylabel('Storage-Aware MEI\n(PSNR × log₁₀(Compression)) / Storage GB', 
                          fontweight='bold', fontsize=13)
            ax1.set_title('Storage-Aware Memory Efficiency Index\nHigher is Better ↑', 
                         fontweight='bold', fontsize=15, pad=20)
            
            # Add value annotation for final point
            final_val = df['storage_aware_mei'].iloc[-1]
            final_step = df['step'].iloc[-1]
            ax1.annotate(f'{final_val:.1f}', 
                        xy=(final_step, final_val), 
                        xytext=(10, 10), textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor=storage_mei_color, alpha=0.8),
                        fontsize=12, fontweight='bold', color='white')
        else:
            ax1.text(0.5, 0.5, 'Storage-Aware MEI\n\nRequires octree evaluation\nwith file size data', 
                    ha='center', va='center', transform=ax1.transAxes, 
                    fontsize=14, fontweight='bold', color='gray',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor='#F8F9FA', edgecolor='gray'))
            ax1.set_title('Storage-Aware MEI (Not Available)', fontweight='bold', fontsize=15)
        
        ax1.set_xlabel('Training Step', fontweight='bold', fontsize=13)
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.set_facecolor('#FAFBFC')
        
        # 2. Voxel Density Efficiency (Right)
        ax2 = axes[1]
        if has_vde:
            ax2.plot(df['step'], df['voxel_density_efficiency'], color=vde_color, 
                    linewidth=3, label='VDE', marker='d', markersize=6,
                    markerfacecolor='white', markeredgecolor=vde_color, markeredgewidth=2)
            
            # Add gradient fill under the curve
            ax2.fill_between(df['step'], df['voxel_density_efficiency'], alpha=0.2, color=vde_color)
            
            ax2.set_ylabel('VDE\n(PSNR × Occupancy Ratio) / Storage GB', 
                          fontweight='bold', fontsize=13)
            ax2.set_title('Voxel Density Efficiency\nHigher is Better ↑', 
                         fontweight='bold', fontsize=15, pad=20)
            
            # Add value annotation for final point
            final_val = df['voxel_density_efficiency'].iloc[-1]
            final_step = df['step'].iloc[-1]
            ax2.annotate(f'{final_val:.2f}', 
                        xy=(final_step, final_val), 
                        xytext=(10, 10), textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor=vde_color, alpha=0.8),
                        fontsize=12, fontweight='bold', color='white')
        else:
            ax2.text(0.5, 0.5, 'Voxel Density Efficiency\n\nRequires octree evaluation\nwith voxel density data', 
                    ha='center', va='center', transform=ax2.transAxes, 
                    fontsize=14, fontweight='bold', color='gray',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor='#F8F9FA', edgecolor='gray'))
            ax2.set_title('VDE (Not Available)', fontweight='bold', fontsize=15)
        
        ax2.set_xlabel('Training Step', fontweight='bold', fontsize=13)
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.set_facecolor('#FAFBFC')
        
        # Add explanatory text at the bottom
        explanation = """Storage-Aware MEI measures quality×compression efficiency per GB of storage. VDE measures quality×voxel density per GB of storage."""
        fig.text(0.5, 0.02, explanation, ha='center', va='bottom', fontsize=11, 
                style='italic', color='#666666')
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.85, bottom=0.15)
        
        # Save the plot
        storage_plot_path = output_dir / f"{scene}_storage_efficiency_metrics.png"
        fig.savefig(storage_plot_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close(fig)
        
        if has_storage_mei or has_vde:
            print(f"📊 Storage efficiency metrics plot saved: {storage_plot_path.name}")
        
        return storage_plot_path
    
    def analyze_scene_efficiency(self, scene: str) -> Optional[pd.DataFrame]:
        """Analyze efficiency metrics for a single scene."""
        
        scene_path = self.base_path / "syn_sh16" / scene
        if not scene_path.exists():
            print(f"❌ Scene directory not found: {scene_path}")
            return None
        
        print(f"\n🔍 Analyzing efficiency metrics for {scene}...")
        
        # Load training data
        training_data = self.load_training_metrics(scene_path)
        if not training_data:
            return None
        
        # Try to load octree evaluation data for voxel counts and storage metrics
        octree_eval = self.load_octree_evaluation(scene_path)
        octree_capacity = None
        if octree_eval:
            octree_capacity = (octree_eval.get('octree_capacity') or 
                             octree_eval.get('additional_info', {}).get('octree_capacity'))
        
        # Calculate enhanced metrics with storage-aware data
        df = self.calculate_enhanced_efficiency_metrics(training_data, octree_capacity, octree_eval)
        
        # Create output directory
        output_dir = scene_path / "analysis"
        output_dir.mkdir(exist_ok=True)
        
        # Create efficiency plots
        plot_path = self.create_efficiency_plots(df, scene, output_dir)
        
        # Save enhanced metrics to CSV for further analysis
        csv_path = output_dir / f"{scene}_efficiency_metrics.csv"
        df.to_csv(csv_path, index=False)
        print(f"💾 Enhanced metrics saved: {csv_path.name}")
        
        # Print summary
        self.print_efficiency_summary(df, scene, octree_capacity)
        
        return df
    
    def print_efficiency_summary(self, df: pd.DataFrame, scene: str, octree_capacity: Optional[int]):
        """Print a summary of efficiency metrics."""
        
        print(f"\n📈 EFFICIENCY SUMMARY - {scene.upper()}")
        print("=" * 50)
        
        if len(df) == 0:
            print("❌ No data to analyze")
            return
        
        final_row = df.iloc[-1]
        
        print(f"Training Steps: {len(df):,}")
        print(f"Final PSNR: {final_row['psnr']:.2f} dB")
        print(f"Final Memory: {final_row['memory_gb']:.2f} GB")
        print(f"Peak Memory: {final_row['peak_memory_gb']:.2f} GB")
        
        print(f"\n📊 EFFICIENCY INDICES:")
        print(f"• MEI (Current): {final_row['mei']:.2f} PSNR/GB")
        print(f"• MEI (Peak): {final_row['peak_mei']:.2f} PSNR/GB")
        print(f"• Max MEI: {df['mei'].max():.2f} PSNR/GB")
        
        if octree_capacity:
            print(f"\n🌳 OCTREE INFO:")
            print(f"• Voxel Count: {octree_capacity:,} nodes")
            print(f"• MEPV (Higher is better): {final_row['mepv']:.3f} MVoxels/GB ⬆")
            if final_row['mepv_quality'] is not None:
                print(f"• Quality-weighted MEPV: {final_row['mepv_quality']:.1f} ⬆")
        else:
            print(f"\n⚠️ Memory Only: {final_row['msf_original']:.2f} GB")
            print("   Run octree evaluation to get proper MEPV calculation")
        
        if final_row['qmt'] is not None:
            print(f"\n🎯 QMT: {final_row['qmt']:.2f} (PSNR×SSIM)/GB")
        
        print("=" * 50)
    
    def discover_scenes(self) -> List[str]:
        """Discover available scenes in the checkpoint directory."""
        
        syn_sh16_path = self.base_path / "syn_sh16"
        if not syn_sh16_path.exists():
            print(f"❌ Base path not found: {syn_sh16_path}")
            return []
        
        scenes = []
        for item in syn_sh16_path.iterdir():
            if item.is_dir() and (item / "nerf_training_metrics.json").exists():
                scenes.append(item.name)
        
        print(f"🔍 Found {len(scenes)} scenes with training data: {scenes}")
        return sorted(scenes)
    
    def create_cross_scene_comparison(self, all_scenes_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Compare memory efficiency across all scenes using standardized metrics."""
        comparison_data = []
        
        for scene, df in all_scenes_data.items():
            if len(df) == 0:
                continue
                
            # Use final evaluation stage for fair comparison
            final_stage = df.iloc[-1]
            
            # Build comparison row with standardized memory metric
            row = {
                'scene': scene,
                'psnr': final_stage['psnr'],
                'ssim': final_stage['ssim'],
                'lpips': final_stage['lpips'],
                'memory_gb': final_stage['memory_gb'],
                'peak_memory_gb': final_stage['peak_memory_gb'],
                'octree_capacity': final_stage.get('octree_capacity', 0),
                'mei': final_stage['mei'],
                'peak_mei': final_stage['peak_mei'],
                'qmt': final_stage['qmt'],
            }
            
            # Add proper MEPV if voxel count available (HIGHER IS BETTER)
            if row['octree_capacity'] and row['octree_capacity'] > 0:
                row['mepv'] = final_stage['mepv']
                row['mepv_quality'] = final_stage['mepv_quality']
            else:
                row['mepv'] = None
                row['mepv_quality'] = None
            
            comparison_data.append(row)
        
        return pd.DataFrame(comparison_data)
    
    def print_methodology_note(self):
        """Print important notes about memory measurement methodology."""
        print("")
        print("="*70)
        print("📋 MEMORY MEASUREMENT METHODOLOGY")
        print("="*70)
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
        print("  • MEI (Memory Efficiency Index): PSNR / Memory GB")
        print("  • MEPV (Memory Efficiency Per Voxel): MVoxels / GB ⬆ HIGHER IS BETTER")
        print("  • QMT (Quality-Memory Tradeoff): (PSNR × SSIM) / Memory GB")
        print("  • Quality MEPV: PSNR × MVoxels / GB ⬆ HIGHER IS BETTER")
        print("")
        print("="*70)
        print("")
    
    def analyze_all_scenes(self) -> Dict[str, pd.DataFrame]:
        """Analyze efficiency metrics for all discovered scenes."""
        
        scenes = self.discover_scenes()
        if not scenes:
            print("❌ No scenes found!")
            return {}
        
        # Print methodology
        self.print_methodology_note()
        
        print(f"\n🚀 Analyzing efficiency metrics for {len(scenes)} scenes...")
        
        results = {}
        for scene in scenes:
            try:
                df = self.analyze_scene_efficiency(scene)
                if df is not None:
                    results[scene] = df
            except Exception as e:
                print(f"❌ Error analyzing {scene}: {e}")
                import traceback
                traceback.print_exc()
        
        print(f"\n✅ Efficiency analysis complete for {len(results)} scenes!")
        
        # Create cross-scene comparison
        if results:
            print("\n📊 Creating cross-scene comparison...")
            comparison_df = self.create_cross_scene_comparison(results)
            
            # Save comparison table
            comparison_path = self.base_path / "cross_scene_efficiency_comparison.csv"
            comparison_df.to_csv(comparison_path, index=False)
            print(f"💾 Cross-scene comparison saved: {comparison_path}")
            
            # Print comparison summary
            print("\n" + "="*70)
            print("📊 CROSS-SCENE EFFICIENCY COMPARISON")
            print("="*70)
            print(comparison_df.to_string(index=False))
            print("="*70)
        
        return results


def main():
    """Run the efficiency metrics analyzer on all scenes."""
    analyzer = EfficiencyMetricsAnalyzer()
    results = analyzer.analyze_all_scenes()
    
    if results:
        print(f"\n🎉 Successfully analyzed {len(results)} scenes:")
        for scene in results:
            print(f"  • {scene}")
    else:
        print("❌ No scenes analyzed successfully")


if __name__ == "__main__":
    main()
