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

class EfficiencyMetricsAnalyzer:
    """Enhanced analyzer for memory efficiency metrics including proper MSF calculation."""
    
    def __init__(self, base_path: str = "/mnt/d/GitHub/nerf-projects/plenoctree/data/Plenoctree/checkpoints"):
        self.base_path = Path(base_path)
        
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
        """Load octree evaluation metrics that contain voxel counts."""
        # Check multiple possible locations for octree evaluation data
        possible_files = [
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
                            return last_entry
                    elif isinstance(data, dict):
                        if 'octree_capacity' in data or ('additional_info' in data and 'octree_capacity' in data['additional_info']):
                            capacity = data.get('octree_capacity', data.get('additional_info', {}).get('octree_capacity', 0))
                            print(f"✅ Found octree capacity: {capacity:,} nodes in {eval_file.name}")
                            return data
                            
                except Exception as e:
                    print(f"⚠️ Error reading {eval_file}: {e}")
                    continue
        
        print(f"⚠️ No octree evaluation data with capacity found for {scene_path.name}")
        return None
    
    def calculate_enhanced_efficiency_metrics(self, training_data: List[Dict], 
                                            octree_capacity: Optional[int] = None) -> pd.DataFrame:
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
            
            # Original (incorrect) MSF - just memory usage
            msf_original = memory_gb
            
            # Proper MSF calculation if voxel count is available
            msf_proper = None
            if octree_capacity and octree_capacity > 0:
                msf_proper = memory_gb / (octree_capacity / 1e6)  # Memory per million voxels
            
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
                'msf_proper': msf_proper,
                'qmt': qmt,
                'peak_qmt': peak_qmt,
                'combined_quality': combined_quality,
                'combined_efficiency': combined_efficiency,
                'peak_combined_efficiency': peak_combined_efficiency,
                'octree_capacity': octree_capacity
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
        fig.suptitle(f'Memory Efficiency Analysis - {scene.upper()}', fontsize=16, fontweight='bold')
        
        # 1. Memory Efficiency Index (MEI)
        ax1 = axes[0, 0]
        ax1.plot(df['step'], df['mei'], 'b-', linewidth=2, label='Current MEI')
        ax1.plot(df['step'], df['peak_mei'], 'b--', linewidth=2, alpha=0.7, label='Peak MEI')
        ax1.set_title('Memory Efficiency Index\n(PSNR / Memory GB)', fontweight='bold')
        ax1.set_xlabel('Training Step')
        ax1.set_ylabel('MEI (PSNR/GB)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Memory Scalability Factor (MSF)
        ax2 = axes[0, 1]
        ax2.plot(df['step'], df['msf_original'], 'r-', linewidth=2, label='Original MSF (Memory Only)')
        if df['msf_proper'].notna().any():
            ax2.plot(df['step'], df['msf_proper'], 'g-', linewidth=2, label='Proper MSF (Memory/MVoxels)')
            ax2.set_ylabel('MSF (GB/MVoxels)')
            ax2.set_title('Memory Scalability Factor\n(Memory / Million Voxels)', fontweight='bold')
        else:
            ax2.set_ylabel('Memory Usage (GB)')
            ax2.set_title('Memory Usage\n(Proper MSF requires voxel count)', fontweight='bold')
        ax2.set_xlabel('Training Step')
        ax2.legend()
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
        ax3.set_title('Quality-Memory Trade-off\n(PSNR × SSIM / Memory GB)', fontweight='bold')
        ax3.set_xlabel('Training Step')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Combined Efficiency Evolution
        ax4 = axes[1, 0]
        ax4.plot(df['step'], df['combined_efficiency'], 'orange', linewidth=2, label='Combined Efficiency')
        ax4.plot(df['step'], df['peak_combined_efficiency'], 'orange', linewidth=2, alpha=0.7, linestyle='--', label='Peak Combined')
        ax4.set_title('Combined Quality Efficiency\n(PSNR×SSIM×(1-LPIPS) / Memory)', fontweight='bold')
        ax4.set_xlabel('Training Step')
        ax4.set_ylabel('Combined Efficiency')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Memory vs Quality Scatter
        ax5 = axes[1, 1]
        scatter = ax5.scatter(df['memory_gb'], df['psnr'], c=df['step'], cmap='viridis', alpha=0.7)
        ax5.set_xlabel('Memory Usage (GB)')
        ax5.set_ylabel('PSNR')
        ax5.set_title('Memory vs Quality Trade-off\n(Color = Training Step)', fontweight='bold')
        plt.colorbar(scatter, ax=ax5, label='Training Step')
        ax5.grid(True, alpha=0.3)
        
        # 6. Efficiency Summary Statistics
        ax6 = axes[1, 2]
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

MSF Status:
"""
        
        if df['msf_proper'].notna().any():
            final_msf = df['msf_proper'].iloc[-1]
            summary_text += f"• Proper MSF: {final_msf:.3f} GB/MVoxel\n"
            if df['octree_capacity'].iloc[0]:
                summary_text += f"• Voxel Count: {df['octree_capacity'].iloc[0]:,}\n"
        else:
            summary_text += "• Need octree evaluation\n  for proper MSF calculation"
        
        ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5))
        
        plt.tight_layout()
        
        # Save the plot
        plot_path = output_dir / f"{scene}_efficiency_metrics.png"
        fig.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close(fig)
        
        print(f"📊 Efficiency metrics plot saved: {plot_path.name}")
        return plot_path
    
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
        
        # Try to load octree evaluation data for voxel counts
        octree_eval = self.load_octree_evaluation(scene_path)
        octree_capacity = None
        if octree_eval:
            octree_capacity = (octree_eval.get('octree_capacity') or 
                             octree_eval.get('additional_info', {}).get('octree_capacity'))
        
        # Calculate enhanced metrics
        df = self.calculate_enhanced_efficiency_metrics(training_data, octree_capacity)
        
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
            print(f"• MSF (Proper): {final_row['msf_proper']:.3f} GB/MVoxel")
        else:
            print(f"\n⚠️ MSF: Original (memory only): {final_row['msf_original']:.2f} GB")
            print("   Run octree evaluation to get proper MSF calculation")
        
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
    
    def analyze_all_scenes(self) -> Dict[str, pd.DataFrame]:
        """Analyze efficiency metrics for all discovered scenes."""
        
        scenes = self.discover_scenes()
        if not scenes:
            print("❌ No scenes found!")
            return {}
        
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
