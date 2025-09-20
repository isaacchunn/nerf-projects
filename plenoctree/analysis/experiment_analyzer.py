#!/usr/bin/env python3
"""
Beautiful PlenOctree Pipeline Analyzer

Creates comprehensive and individual plots with proper averaging and beautiful styling.
Includes post-compression evaluation for accurate quality metrics.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

def format_3_sig_figs(value):
    """Format number to 3 significant figures consistently."""
    if value == 0:
        return "0.00"
    elif abs(value) >= 100:
        return f"{value:.1f}"
    elif abs(value) >= 10:
        return f"{value:.2f}"
    elif abs(value) >= 1:
        return f"{value:.3f}"
    elif abs(value) >= 0.1:
        return f"{value:.3f}"
    elif abs(value) >= 0.01:
        return f"{value:.3f}"
    else:
        return f"{value:.3g}"

# Set beautiful plot style
plt.style.use('default')  # Start with clean default
sns.set_palette("husl")

# Beautiful plot configuration with enhanced styling
plt.rcParams.update({
    'figure.dpi': 200,
    'savefig.dpi': 200,
    'savefig.bbox': 'tight',
    'savefig.facecolor': 'white',
    'font.size': 12,
    'axes.titlesize': 18,
    'axes.labelsize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 12,
    'figure.titlesize': 20,
    'axes.grid': True,
    'grid.alpha': 0.25,
    'grid.linewidth': 0.8,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 1.5,
    'axes.edgecolor': '#333333',
    'axes.facecolor': '#FAFAFA',
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'xtick.major.size': 6,
    'ytick.major.size': 6,
    'xtick.color': '#666666',
    'ytick.color': '#666666',
    'axes.labelcolor': '#333333',
    'grid.linewidth': 0.8,
    'lines.linewidth': 3,
    'lines.markersize': 8,
})

class SimplePlenOctreeAnalyzer:
    """Beautiful analyzer with proper averaging and post-compression evaluation"""
    
    def __init__(self, base_path="/mnt/d/GitHub/nerf-projects/plenoctree/data/Plenoctree/checkpoints"):
        self.base_path = Path(base_path)
        
    def discover_scenes(self, config_name="syn_sh16"):
        """Auto-discover all available scenes"""
        config_path = self.base_path / config_name
        scenes = []
        if config_path.exists():
            scenes = [d.name for d in config_path.iterdir() if d.is_dir()]
            print(f"📍 Found scenes: {scenes}")
        return scenes
    
    def load_scene_data(self, scene, config_name="syn_sh16"):
        """Load all metrics for a scene"""
        scene_path = self.base_path / config_name / scene
        octrees_path = scene_path / "octrees"
        
        data = {}
        
        # Load NeRF training metrics (training time data)
        nerf_training_file = scene_path / "nerf_training_metrics.json"
        if nerf_training_file.exists():
            with open(nerf_training_file) as f:
                data['nerf_training'] = json.load(f)
                print(f"  ✅ NeRF Training: {nerf_training_file.name}")
        
        # Load NeRF evaluation (post-training quality)
        nerf_eval_file = scene_path / "nerf_evaluation_final.json"
        if nerf_eval_file.exists():
            with open(nerf_eval_file) as f:
                data['nerf_eval'] = json.load(f)
                print(f"  ✅ NeRF Evaluation: {nerf_eval_file.name}")
        
        # Load octree extraction metrics (contains creation + initial evaluation)
        extraction_file = octrees_path / "octree_extraction_metrics.json"
        if extraction_file.exists():
            with open(extraction_file) as f:
                data['extraction'] = json.load(f)
                print(f"  ✅ Octree Extraction: {extraction_file.name}")
        
        # Load initial octree evaluation metrics
        initial_eval_file = octrees_path / "octree_initial_evaluation_metrics.json"
        if initial_eval_file.exists():
            with open(initial_eval_file) as f:
                data['initial_evaluation'] = json.load(f)
                print(f"  ✅ Initial Evaluation: {initial_eval_file.name}")
        
        # Load octree optimization metrics
        optimization_file = octrees_path / "octree_optimization_metrics.json"
        if optimization_file.exists():
            with open(optimization_file) as f:
                data['optimization'] = json.load(f)
                print(f"  ✅ Octree Optimization: {optimization_file.name}")
        
        # Load optimized octree evaluation (evaluation after optimization)
        evaluation_file = octrees_path / "octree_evaluation_metrics.json"
        if evaluation_file.exists():
            with open(evaluation_file) as f:
                data['evaluation'] = json.load(f)
                print(f"  ✅ Optimized Octree Evaluation: {evaluation_file.name}")
        
        # Load compressed octree evaluation (post-compression quality)
        # Check both main directory (old location) and octrees directory (new location)
        compression_eval_file = scene_path / "octree_compression_evaluation_metrics.json"
        compression_eval_file_octrees = octrees_path / "octree_compression_evaluation_metrics.json"
        
        if compression_eval_file_octrees.exists():
            with open(compression_eval_file_octrees) as f:
                data['compression_evaluation'] = json.load(f)
                print(f"  ✅ Compression Evaluation: {compression_eval_file_octrees.name} (from octrees/)")
        elif compression_eval_file.exists():
            with open(compression_eval_file) as f:
                data['compression_evaluation'] = json.load(f)
                print(f"  ✅ Compression Evaluation: {compression_eval_file.name} (from main directory)")
        
        # Load full pipeline metrics (timing data)
        pipeline_file = scene_path / "full_pipeline_metrics.json"
        if pipeline_file.exists():
            try:
                with open(pipeline_file) as f:
                    content = f.read()
                    # Fix common JSON formatting issues
                    content = content.replace('"step": O1,', '"step": "O1",')
                    content = content.replace('"step": O2,', '"step": "O2",')
                    content = content.replace('"step": O3,', '"step": "O3",')
                    content = content.replace('"step": O4,', '"step": "O4",')
                    content = content.replace('"step": O5,', '"step": "O5",')
                    content = content.replace('"step": O6,', '"step": "O6",')
                    data['pipeline_timing'] = json.loads(content)
                    print(f"  ✅ Pipeline Timing: {pipeline_file.name}")
            except json.JSONDecodeError as e:
                print(f"  ⚠️  Pipeline Timing JSON error: {e} - skipping timing data")
                data['pipeline_timing'] = None
        
        return data
    
    def calculate_averages(self, metrics_list, metric_name):
        """Calculate average of a metric from a list of entries"""
        values = []
        for entry in metrics_list:
            if entry.get('metrics', {}).get(metric_name) is not None:
                values.append(entry['metrics'][metric_name])
        return np.mean(values) if values else None
    
    def extract_timing_summary(self, scene_data):
        """Extract timing information from individual JSON files to avoid duplicates from incomplete pipeline runs"""
        timing_summary = {
            'stage_timings': {},
            'total_pipeline_time': None,
            'detailed_breakdowns': {}
        }
        
        print("  🔄 Extracting timing data from individual metrics files...")
        
        # Step 1: Extract NeRF training time from timestamps
        if 'nerf_training' in scene_data and scene_data['nerf_training']:
            training_data = scene_data['nerf_training']
            if len(training_data) >= 2:
                # Calculate duration from first and last timestamps
                first_entry = training_data[0]
                last_entry = training_data[-1]
                
                first_timestamp_str = first_entry.get('timestamp', '')
                last_timestamp_str = last_entry.get('timestamp', '')
                
                if first_timestamp_str and last_timestamp_str:
                    try:
                        from datetime import datetime
                        # Parse ISO format timestamps
                        first_dt = datetime.fromisoformat(first_timestamp_str.replace('Z', '+00:00'))
                        last_dt = datetime.fromisoformat(last_timestamp_str.replace('Z', '+00:00'))
                        
                        training_duration = (last_dt - first_dt).total_seconds()
                        if training_duration > 0:
                            timing_summary['stage_timings']['NeRF-SH Training'] = training_duration
                            print(f"  ⏱️  Found NeRF-SH Training: {training_duration:.1f}s (from {len(training_data)} timestamps)")
                    except (ValueError, TypeError) as e:
                        print(f"  ⚠️  Could not parse training timestamps: {e}")
                        # Fallback to time_interval if available
                        if 'additional_info' in last_entry:
                            info = last_entry['additional_info']
                            if 'time_interval' in info:
                                timing_summary['stage_timings']['NeRF-SH Training'] = info['time_interval']
                                print(f"  ⏱️  Found NeRF-SH Training: {info['time_interval']:.1f}s (fallback)")
        
        # Step 2: Extract NeRF evaluation time
        if 'nerf_eval' in scene_data and scene_data['nerf_eval']:
            eval_data = scene_data['nerf_eval'][-1]  # Latest evaluation
            if 'additional_info' in eval_data:
                info = eval_data['additional_info']
                if 'total_eval_time' in info:
                    timing_summary['stage_timings']['NeRF-SH Evaluation'] = info['total_eval_time']
                    print(f"  ⏱️  Found NeRF-SH Evaluation: {info['total_eval_time']:.1f}s")
        
        # Step 3: Extract octree construction and evaluation times from individual files
        timing_stages = [
            # (data_key, stage_name, primary_timing_field, fallback_timing_field)
            ('extraction', 'Initial Octree Construction', 'total_extraction_time', 'eval_time'),
            ('initial_evaluation', 'Initial Octree Evaluation', 'total_eval_time', None),
            ('optimization', 'Optimize Octree', 'total_optimization_time', None),
            ('evaluation', 'Optimized Octree Evaluation', 'total_eval_time', None),
            ('compression', 'Compress Octree', 'total_compression_time', 'average_time_per_file'),
            ('compression_evaluation', 'Compressed Octree Evaluation', 'total_eval_time', None)
        ]
        
        for data_key, stage_name, primary_timing_field, fallback_timing_field in timing_stages:
            if data_key in scene_data and scene_data[data_key]:
                data_list = scene_data[data_key]
                
                # For evaluation stages, use the last entry (most recent)
                # For process stages (extraction, optimization, compression), use total time
                if isinstance(data_list, list) and data_list:
                    entry = data_list[-1]  # Latest entry
                    
                    if isinstance(entry, dict) and 'additional_info' in entry:
                        info = entry['additional_info']
                        
                        # Try primary timing field first
                        if primary_timing_field in info and info[primary_timing_field] > 0:
                            timing_summary['stage_timings'][stage_name] = info[primary_timing_field]
                            print(f"  ⏱️  Found {stage_name}: {info[primary_timing_field]:.1f}s")
                        # Try fallback timing field
                        elif fallback_timing_field and fallback_timing_field in info and info[fallback_timing_field] > 0:
                            timing_summary['stage_timings'][stage_name] = info[fallback_timing_field]
                            print(f"  ⏱️  Found {stage_name}: {info[fallback_timing_field]:.1f}s (fallback)")
                        # Look for step timing for extraction phase
                        elif data_key == 'extraction' and 'step_timings' in info:
                            step_timings = info['step_timings']
                            if isinstance(step_timings, dict):
                                # Sum up all extraction steps
                                total_extraction = sum(step_timings.values())
                                if total_extraction > 0:
                                    timing_summary['stage_timings'][stage_name] = total_extraction
                                    print(f"  ⏱️  Found {stage_name}: {total_extraction:.1f}s (from step timings)")
        
        # Calculate total pipeline time
        if timing_summary['stage_timings']:
            total_time = sum(timing_summary['stage_timings'].values())
            timing_summary['total_pipeline_time'] = total_time
            print(f"  🏁 Total pipeline time: {total_time:.1f}s ({total_time/60:.1f}m)")
        else:
            print("  ⚠️  No timing data found in individual metrics files")
        
        # Extract detailed stage breakdowns
        stage_details = {
            'extraction': {
                'total_time_key': 'total_extraction_time',
                'sub_timings': ['step1_time', 'step2_time', 'save_time', 'eval_time']
            },
            'optimization': {
                'total_time_key': 'total_optimization_time', 
                'sub_timings': ['epoch_time', 'eval_time']
            },
            'initial_evaluation': {
                'total_time_key': 'total_eval_time',
                'sub_timings': ['avg_time_per_image', 'video_export_time', 'images_export_time']
            },
            'evaluation': {
                'total_time_key': 'total_eval_time',
                'sub_timings': ['avg_time_per_image', 'video_export_time', 'images_export_time']
            },
            'compression': {
                'total_time_key': 'total_compression_time',
                'sub_timings': ['load_time', 'quantization_time', 'save_time']
            },
            'compression_evaluation': {
                'total_time_key': 'total_eval_time',
                'sub_timings': ['avg_time_per_image']
            }
        }
        
        for stage_name, config in stage_details.items():
            if stage_name in scene_data and scene_data[stage_name]:
                stage_breakdown = {}
                
                # Find summary entries with total time
                summary_entries = [entry for entry in scene_data[stage_name] 
                                 if config['total_time_key'] in entry.get('additional_info', {})]
                
                if summary_entries:
                    total_time = summary_entries[0]['additional_info'][config['total_time_key']]
                    stage_breakdown['total_time'] = total_time
                    
                    # Extract sub-timings
                    sub_times = {}
                    for entry in scene_data[stage_name]:
                        for timing_key in config['sub_timings']:
                            if timing_key in entry.get('additional_info', {}):
                                if timing_key not in sub_times:
                                    sub_times[timing_key] = []
                                sub_times[timing_key].append(entry['additional_info'][timing_key])
                    
                    # Average sub-timings (for stages with multiple entries)
                    for key, values in sub_times.items():
                        stage_breakdown[key] = np.mean(values) if values else 0
                    
                    timing_summary['detailed_breakdowns'][stage_name] = stage_breakdown
        
        return timing_summary
    
    def extract_pipeline_stages(self, scene_data):
        """
        Extract 4 key evaluation stages (evaluation results only, no training/optimization process):
        1. NeRF Evaluation - Post-training NeRF quality
        2. Initial Octree Evaluation - Baseline octree quality after extraction  
        3. Optimized Octree Evaluation - Quality after optimization (final result)
        4. Compressed Octree Evaluation - Quality after compression
        """
        
        stages = []
        data_sources = []
        
        # Stage 1: NeRF Evaluation (Post-training quality)
        if 'nerf_eval' in scene_data:
            nerf_eval = scene_data['nerf_eval'][-1]
            nerf_info = nerf_eval.get('additional_info', {})
            stages.append({
                'name': '1. NeRF Evaluation',
                'psnr': nerf_eval['metrics']['psnr'],
                'ssim': nerf_eval['metrics']['ssim'],
                'lpips': nerf_eval['metrics']['lpips'],
                'memory': nerf_eval['metrics']['gpu_allocated_gb'],
                'gpu_reserved': nerf_eval['metrics'].get('gpu_reserved_gb', 0),
                'system_memory': nerf_eval['metrics'].get('system_used_gb', 0),
                'process_memory': nerf_eval['metrics'].get('process_rss_gb', 0),
                'memory_efficiency': nerf_info.get('memory_efficiency_index'),
                'peak_memory_efficiency': nerf_info.get('peak_memory_efficiency_index'),
                'quality_memory_tradeoff': nerf_info.get('quality_memory_tradeoff'),
                'lpips_memory_efficiency': nerf_info.get('lpips_memory_efficiency'),
                'combined_quality_memory_index': nerf_info.get('combined_quality_memory_index'),
                'timing': nerf_info.get('total_eval_time', 0)
            })
            data_sources.append("NeRF evaluation results")
        
        # Stage 2: Initial Octree Evaluation (Baseline octree quality)
        if 'initial_evaluation' in scene_data and scene_data['initial_evaluation']:
            initial_eval = scene_data['initial_evaluation'][0]
            initial_info = initial_eval.get('additional_info', {})
            stages.append({
                'name': '2. Initial Octree Evaluation',
                'psnr': initial_eval['metrics']['psnr'],
                'ssim': initial_eval['metrics']['ssim'],
                'lpips': initial_eval['metrics']['lpips'],
                'memory': initial_eval['metrics']['gpu_allocated_gb'],
                'gpu_reserved': initial_eval['metrics'].get('gpu_reserved_gb', 0),
                'system_memory': initial_eval['metrics'].get('system_used_gb', 0),
                'process_memory': initial_eval['metrics'].get('process_rss_gb', 0),
                'memory_efficiency': initial_info.get('memory_efficiency_index'),
                'peak_memory_efficiency': initial_info.get('peak_memory_efficiency_index'),
                'quality_memory_tradeoff': initial_info.get('quality_memory_tradeoff'),
                'lpips_memory_efficiency': initial_info.get('lpips_memory_efficiency'),
                'combined_quality_memory_index': initial_info.get('combined_quality_memory_index'),
                'timing': initial_info.get('total_eval_time', 0)
            })
            data_sources.append("Initial octree evaluation results")
        elif 'extraction' in scene_data:
            # Fallback to extraction evaluation if initial_evaluation not available
            extraction_eval = [m for m in scene_data['extraction'] if m.get('phase') == 'octree_evaluation']
            if extraction_eval:
                octree_initial = extraction_eval[-1]
                stages.append({
                    'name': '2. Octree Construction\n(Initial)',
                    'psnr': octree_initial['metrics']['psnr'],
                    'ssim': octree_initial['metrics']['ssim'],
                    'lpips': octree_initial['metrics']['lpips'],
                    'memory': octree_initial['metrics']['gpu_allocated_gb']
                })
                data_sources.append("Post-extraction evaluation (fallback)")
        
        # Stage 3: Optimized Octree Evaluation (Quality after optimization)
        if 'evaluation' in scene_data and scene_data['evaluation']:
            eval_data = scene_data['evaluation'][0]
            eval_info = eval_data.get('additional_info', {})
            stages.append({
                'name': '3. Optimized Octree Evaluation',
                'psnr': eval_data['metrics']['psnr'],
                'ssim': eval_data['metrics']['ssim'],
                'lpips': eval_data['metrics']['lpips'],
                'memory': eval_data['metrics']['gpu_allocated_gb'],
                'gpu_reserved': eval_data['metrics'].get('gpu_reserved_gb', 0),
                'system_memory': eval_data['metrics'].get('system_used_gb', 0),
                'process_memory': eval_data['metrics'].get('process_rss_gb', 0),
                'memory_efficiency': eval_info.get('memory_efficiency_index'),
                'peak_memory_efficiency': eval_info.get('peak_memory_efficiency_index'),
                'quality_memory_tradeoff': eval_info.get('quality_memory_tradeoff'),
                'lpips_memory_efficiency': eval_info.get('lpips_memory_efficiency'),
                'combined_quality_memory_index': eval_info.get('combined_quality_memory_index'),
                'timing': eval_info.get('total_eval_time', 0)
            })
            data_sources.append("Optimized octree evaluation results")
        
        # Stage 4: Compressed Octree Evaluation (Post-compression quality assessment)
        if 'compression_evaluation' in scene_data and scene_data['compression_evaluation']:
            # We have actual post-compression evaluation!
            comp_eval_data = scene_data['compression_evaluation'][0]
            comp_info = comp_eval_data.get('additional_info', {})
            stages.append({
                'name': '4. Compressed Octree Evaluation',
                'psnr': comp_eval_data['metrics']['psnr'],
                'ssim': comp_eval_data['metrics']['ssim'],
                'lpips': comp_eval_data['metrics']['lpips'],
                'memory': comp_eval_data['metrics']['gpu_allocated_gb'],
                'gpu_reserved': comp_eval_data['metrics'].get('gpu_reserved_gb', 0),
                'system_memory': comp_eval_data['metrics'].get('system_used_gb', 0),
                'process_memory': comp_eval_data['metrics'].get('process_rss_gb', 0),
                'memory_efficiency': comp_info.get('memory_efficiency_index'),
                'peak_memory_efficiency': comp_info.get('peak_memory_efficiency_index'),
                'quality_memory_tradeoff': comp_info.get('quality_memory_tradeoff'),
                'lpips_memory_efficiency': comp_info.get('lpips_memory_efficiency'),
                'combined_quality_memory_index': comp_info.get('combined_quality_memory_index'),
                'timing': comp_info.get('total_eval_time', 0),
                'compression_ratio': comp_eval_data['metrics'].get('compression_ratio'),
                'file_size_mb': comp_info.get('octree_file_size_mb')
            })
            data_sources.append("Compressed octree evaluation results")
        elif 'compression' in scene_data:
            # No post-compression evaluation - estimate quality drop
            if len(stages) >= 2:  # Need at least compression stage
                prev_stage = stages[-2]  # Use stage before compression
                
                # Estimate quality drop due to compression (typical values)
                psnr_drop = 0.5  # Typical PSNR drop from quantization
                ssim_drop = 0.01  # Typical SSIM drop
                lpips_increase = 0.005  # Typical LPIPS increase
                
                stages.append({
                    'name': '6. Compression Evaluation\n(Estimated)',
                    'psnr': max(0, prev_stage['psnr'] - psnr_drop) if prev_stage['psnr'] else None,
                    'ssim': max(0, prev_stage['ssim'] - ssim_drop) if prev_stage['ssim'] else None,
                    'lpips': min(1, prev_stage['lpips'] + lpips_increase) if prev_stage['lpips'] else None,
                    'memory': stages[-1]['memory'] if stages else None  # Use compression memory
                })
                data_sources.append("Estimated post-compression quality drop")
        
        return stages, data_sources
    
    def create_beautiful_plot(self, scene, stages, data_sources, plot_type='comprehensive'):
        """Create beautiful plots with modern styling"""
        
        if not stages:
            print(f"⚠️ No pipeline data for {scene}")
            return None
        
        # Extract data
        stage_names = [s['name'] for s in stages]
        psnr_vals = [s['psnr'] for s in stages if s['psnr'] is not None]
        ssim_vals = [s['ssim'] for s in stages if s['ssim'] is not None]
        lpips_vals = [s['lpips'] for s in stages if s['lpips'] is not None]
        memory_vals = [s['memory'] for s in stages if s['memory'] is not None]
        
        # Color schemes
        # Modern beautiful color scheme with better contrast
        colors = {
            'psnr': '#FF6B6B',      # Coral red - warm and attention-grabbing
            'ssim': '#4ECDC4',      # Teal - professional and calming  
            'lpips': '#FFE66D',     # Bright yellow - energetic but readable
            'memory': '#A8E6CF',    # Mint green - fresh and clean
            'improvement': '#88D8B0', # Seafoam - positive growth color
            'background': '#F8F9FA', # Light neutral background
            'shadow': '#00000015'    # Subtle shadow for depth
        }
        
        if plot_type == 'comprehensive':
            return self._create_comprehensive_plot(scene, stages, data_sources, colors)
        else:
            return self._create_individual_plot(scene, stages, plot_type, colors)
    
    def _create_comprehensive_plot(self, scene, stages, data_sources, colors):
        """Create comprehensive plot with beautiful styling"""
        
        # Extract data for plotting
        stage_names = [s['name'] for s in stages]
        psnr_vals = [s['psnr'] for s in stages]
        ssim_vals = [s['ssim'] or 0 for s in stages]
        lpips_vals = [s['lpips'] or 0 for s in stages]
        memory_vals = [s['memory'] or 0 for s in stages]
        
        # Extract system memory data
        system_memory_vals = [s.get('system_memory', 0) for s in stages]
        process_memory_vals = [s.get('process_memory', 0) for s in stages]
        
        # Create figure with bigger size for better visibility
        fig = plt.figure(figsize=(24, 14))
        fig.patch.set_facecolor('white')
        
        # Simplified main title
        fig.suptitle(f'PlenOctree Pipeline Analysis - {scene.upper()}', 
                    fontsize=22, fontweight='bold', y=0.95)
        
        # Create simplified grid layout (2x3 instead of 4x3)
        gs = fig.add_gridspec(2, 3, hspace=0.4, wspace=0.3)
        
        # 1. PSNR progression with gradient background
        ax1 = fig.add_subplot(gs[0, 0])
        x_pos = range(len(stage_names))
        # Add subtle shadow for depth
        ax1.plot(x_pos, psnr_vals, 'o-', linewidth=5, markersize=13, 
                color=colors['shadow'], markerfacecolor=colors['shadow'], 
                markeredgewidth=0, alpha=0.3, zorder=1)
        
        # Main line with beautiful styling
        line1 = ax1.plot(x_pos, psnr_vals, 'o-', linewidth=4, markersize=12, 
                        color=colors['psnr'], markerfacecolor='white', 
                        markeredgewidth=3, markeredgecolor=colors['psnr'], zorder=2)[0]
        
        # Add value labels above points
        for i, val in enumerate(psnr_vals):
            if val is not None:
                ax1.text(i, val + max(psnr_vals) * 0.01, format_3_sig_figs(val), 
                        ha='center', va='bottom', fontweight='bold', fontsize=10,
                        bbox=dict(boxstyle="round,pad=0.2", facecolor='white', 
                                 edgecolor=colors['psnr'], alpha=0.8))
        
        # Add gradient background
        ax1.fill_between(x_pos, psnr_vals, alpha=0.2, color=colors['psnr'])
        
        ax1.set_title('📊 PSNR Progression\nPeak Signal-to-Noise Ratio (Higher = Better)', 
                     fontweight='bold', pad=20)
        ax1.set_ylabel('PSNR (dB)', fontweight='bold')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels([s.split()[1] if len(s.split()) > 1 else s for s in stage_names], rotation=0, ha='center')
        
        # Add value annotations with beautiful styling
        for i, val in enumerate(psnr_vals):
            if val is not None:
                ax1.annotate(f'{val:.1f}', (i, val), 
                           textcoords="offset points", xytext=(0,15), ha='center', 
                           fontweight='bold', fontsize=11,
                           bbox=dict(boxstyle="round,pad=0.3", facecolor='white', 
                                   edgecolor=colors['psnr'], alpha=0.9))
        
        # 2. SSIM progression
        ax2 = fig.add_subplot(gs[0, 1])
        valid_ssim_idx = [i for i, val in enumerate(ssim_vals) if val > 0]
        valid_ssim_vals = [ssim_vals[i] for i in valid_ssim_idx]
        
        if valid_ssim_vals:
            # Shadow effect
            ax2.plot(valid_ssim_idx, valid_ssim_vals, 'o-', linewidth=5, markersize=13,
                    color=colors['shadow'], markerfacecolor=colors['shadow'], 
                    markeredgewidth=0, alpha=0.3, zorder=1)
            
            # Main line
            ax2.plot(valid_ssim_idx, valid_ssim_vals, 'o-', linewidth=4, markersize=12,
                    color=colors['ssim'], markerfacecolor='white', 
                    markeredgewidth=3, markeredgecolor=colors['ssim'], zorder=2)
            
            # Add value labels above points
            for i, val in zip(valid_ssim_idx, valid_ssim_vals):
                if val is not None and val > 0:
                    ax2.text(i, val + max(valid_ssim_vals) * 0.01, format_3_sig_figs(val), 
                            ha='center', va='bottom', fontweight='bold', fontsize=10,
                            bbox=dict(boxstyle="round,pad=0.2", facecolor='white', 
                                     edgecolor=colors['ssim'], alpha=0.8))
            
            ax2.fill_between(valid_ssim_idx, valid_ssim_vals, alpha=0.2, color=colors['ssim'])
        
        ax2.set_title('📈 SSIM Progression\nStructural Similarity Index (Higher = Better)', 
                     fontweight='bold', pad=20)
        ax2.set_ylabel('SSIM (0-1)', fontweight='bold')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels([s.split()[1] if len(s.split()) > 1 else s for s in stage_names], rotation=0, ha='center')
        
        # 3. LPIPS progression
        ax3 = fig.add_subplot(gs[0, 2])
        valid_lpips_idx = [i for i, val in enumerate(lpips_vals) if val > 0]
        valid_lpips_vals = [lpips_vals[i] for i in valid_lpips_idx]
        
        if valid_lpips_vals:
            # Shadow effect
            ax3.plot(valid_lpips_idx, valid_lpips_vals, 'o-', linewidth=5, markersize=13,
                    color=colors['shadow'], markerfacecolor=colors['shadow'], 
                    markeredgewidth=0, alpha=0.3, zorder=1)
            
            # Main line
            ax3.plot(valid_lpips_idx, valid_lpips_vals, 'o-', linewidth=4, markersize=12,
                    color=colors['lpips'], markerfacecolor='white', 
                    markeredgewidth=3, markeredgecolor=colors['lpips'], zorder=2)
            
            # Add value labels above points
            for i, val in zip(valid_lpips_idx, valid_lpips_vals):
                if val is not None and val > 0:
                    ax3.text(i, val + max(valid_lpips_vals) * 0.02, format_3_sig_figs(val), 
                            ha='center', va='bottom', fontweight='bold', fontsize=10,
                            bbox=dict(boxstyle="round,pad=0.2", facecolor='white', 
                                     edgecolor=colors['lpips'], alpha=0.8))
            
            ax3.fill_between(valid_lpips_idx, valid_lpips_vals, alpha=0.2, color=colors['lpips'])
        
        ax3.set_title('📉 LPIPS Progression\nLearned Perceptual Similarity (Lower = Better)', 
                     fontweight='bold', pad=20)
        ax3.set_ylabel('LPIPS (0-1)', fontweight='bold')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels([s.split()[1] if len(s.split()) > 1 else s for s in stage_names], rotation=0, ha='center')
        
        # 4. Memory usage comparison (GPU, System, Process)
        ax4 = fig.add_subplot(gs[1, 0])
        
        # Create grouped bar chart for different memory types
        bar_width = 0.25
        r1 = [x - bar_width for x in x_pos]
        r2 = x_pos  
        r3 = [x + bar_width for x in x_pos]
        
        # Extract GPU reserved values (more comprehensive than allocated)
        gpu_reserved_vals = [s.get('gpu_reserved', 0) for s in stages]
        
        bars1 = ax4.bar(r1, memory_vals, bar_width, label='GPU Allocated', color='#E74C3C', alpha=0.8)
        bars2 = ax4.bar(r2, gpu_reserved_vals, bar_width, label='GPU Reserved', color='#C0392B', alpha=0.8)
        bars3 = ax4.bar(r3, system_memory_vals, bar_width, label='System RAM', color='#3498DB', alpha=0.8)
        
        # Add value labels on bars
        for bars, vals in zip([bars1, bars2, bars3], [memory_vals, gpu_reserved_vals, system_memory_vals]):
            for bar, val in zip(bars, vals):
                if val > 0:
                    ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.05,
                            format_3_sig_figs(val), ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        ax4.set_title('💾 Memory Usage Comparison\n⚠️ GPU Allocated may underestimate true usage', 
                     fontweight='bold', pad=20)
        ax4.set_ylabel('Memory (GB)', fontweight='bold')
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels([s.split()[1] if len(s.split()) > 1 else s for s in stage_names], rotation=0, ha='center')
        ax4.legend(loc='upper left')
        ax4.grid(True, alpha=0.3)
        
        # 5. Quality improvements with beautiful bars
        if len(psnr_vals) >= 2 and psnr_vals[1] is not None:
            ax5 = fig.add_subplot(gs[1, 1])
            nerf_baseline = psnr_vals[1]  # NeRF Evaluation baseline
            improvements = []
            improvement_labels = []
            
            for i, (psnr, name) in enumerate(zip(psnr_vals, stage_names)):
                if i > 1 and psnr is not None:  # Skip NeRF training and evaluation
                    improvement = psnr - nerf_baseline
                    improvements.append(improvement)
                    improvement_labels.append(name.split('\n')[0])
            
            if improvements:
                bar_colors = [colors['improvement'] if imp > 0 else colors['lpips'] for imp in improvements]
                bars = ax5.bar(range(len(improvements)), improvements, color=bar_colors, 
                              alpha=0.8, edgecolor='white', linewidth=2)
                
                ax5.set_title('⬆️ PSNR Improvements Over NeRF\nOctree Optimization Benefits', 
                             fontweight='bold', pad=20)
                ax5.set_ylabel('PSNR Improvement (dB)', fontweight='bold')
                ax5.set_xticks(range(len(improvements)))
                ax5.set_xticklabels([label.split()[1] if len(label.split()) > 1 else label for label in improvement_labels], rotation=0, ha='center')
                ax5.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=2)
                
                # Add value labels
                for bar, imp in zip(bars, improvements):
                    height = bar.get_height()
                    ax5.text(bar.get_x() + bar.get_width()/2., height + (0.1 if height > 0 else -0.3),
                           f'{imp:+.2f}', ha='center', va='bottom' if height > 0 else 'top', 
                           fontweight='bold', fontsize=11,
                           bbox=dict(boxstyle="round,pad=0.3", facecolor='white', 
                                   edgecolor=bar_colors[bars.index(bar)], alpha=0.9))
        
        # 6. Memory efficiency indices from JSON data
        if len(stages) > 0:
            ax6 = fig.add_subplot(gs[1, 2])
            
            # Extract memory efficiency indices (exclude NeRF - stage 0 due to minimal memory usage)
            memory_eff_vals = []
            quality_memory_vals = []
            lpips_memory_vals = []
            combined_memory_vals = []
            
            for i, stage in enumerate(stages):
                if i == 0 and 'nerf' in stage['name'].lower():
                    # Skip NeRF evaluation for memory efficiency (minimal memory usage)
                    memory_eff_vals.append(None)
                    quality_memory_vals.append(None)
                    lpips_memory_vals.append(None)
                    combined_memory_vals.append(None)
                else:
                    memory_eff_vals.append(stage.get('memory_efficiency'))
                    quality_memory_vals.append(stage.get('quality_memory_tradeoff'))
                    lpips_memory_vals.append(stage.get('lpips_memory_efficiency'))
                    combined_memory_vals.append(stage.get('combined_quality_memory_index'))
            
            # Plot memory efficiency index progression
            valid_memory_eff = [v for v in memory_eff_vals if v is not None]
            if valid_memory_eff:
                x_eff = [i for i, v in enumerate(memory_eff_vals) if v is not None]
                ax6.plot(x_eff, valid_memory_eff, 'o-', color=colors['improvement'], 
                        linewidth=3, markersize=8, label='Memory Efficiency Index')
            
            # Plot combined quality-memory index if available
            valid_combined = [v for v in combined_memory_vals if v is not None]
            if valid_combined:
                x_combined = [i for i, v in enumerate(combined_memory_vals) if v is not None]
                ax6.plot(x_combined, valid_combined, 's-', color=colors['ssim'], 
                        linewidth=3, markersize=8, label='Combined Quality-Memory Index')
            
            ax6.set_title('⚡ Memory Efficiency Indices\nHigher = More Efficient', 
                         fontweight='bold', pad=20)
            ax6.set_ylabel('Efficiency Index', fontweight='bold')
            ax6.set_xticks(range(len(stages)))
            ax6.set_xticklabels([s['name'].split()[1] if len(s['name'].split()) > 1 else s['name'] for s in stages], rotation=0, ha='center')
            ax6.legend(loc='upper left')
            ax6.grid(True, alpha=0.3)
        
        
        # Adjust layout to prevent overlapping labels
        plt.tight_layout()
        
        return fig
    
    def _create_individual_plot(self, scene, stages, plot_type, colors):
        """Create individual plots with beautiful styling"""
        
        # Extract data
        stage_names = [s['name'] for s in stages]
        
        if plot_type == 'psnr':
            values = [s['psnr'] for s in stages]
            color = colors['psnr']
            title = f'📊 PSNR Progression - {scene.upper()}'
            subtitle = 'Peak Signal-to-Noise Ratio (Higher = Better)'
            ylabel = 'PSNR (dB)'
            format_str = '{:.2f} dB'
        elif plot_type == 'ssim':
            values = [s['ssim'] for s in stages]
            color = colors['ssim']
            title = f'📈 SSIM Progression - {scene.upper()}'
            subtitle = 'Structural Similarity Index (Higher = Better)'
            ylabel = 'SSIM (0-1)'
            format_str = '{:.4f}'
        elif plot_type == 'lpips':
            values = [s['lpips'] for s in stages]
            color = colors['lpips']
            title = f'📉 LPIPS Progression - {scene.upper()}'
            subtitle = 'Learned Perceptual Similarity (Lower = Better)'
            ylabel = 'LPIPS (0-1)'
            format_str = '{:.4f}'
        elif plot_type == 'memory':
            values = [s['memory'] for s in stages]
            color = colors['memory']
            title = f'💾 Memory Usage - {scene.upper()}'
            subtitle = 'GPU Memory Requirements per Stage'
            ylabel = 'Memory (GB)'
            format_str = '{:.2f} GB'
        else:
            return None
        
        # Filter out None values
        valid_data = [(i, val, name) for i, (val, name) in enumerate(zip(values, stage_names)) if val is not None]
        if not valid_data:
            return None
        
        indices, valid_values, valid_names = zip(*valid_data)
        
        # Create beautiful individual plot
        fig, ax = plt.subplots(1, 1, figsize=(14, 10))
        fig.patch.set_facecolor('white')
        
        if plot_type == 'memory':
            # Bar plot for memory
            bars = ax.bar(indices, valid_values, color=color, alpha=0.8, 
                         edgecolor='white', linewidth=3, width=0.6)
            
            # Add value labels on bars
            for bar, val in zip(bars, valid_values):
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + max(valid_values)*0.02,
                       format_str.format(val), ha='center', va='bottom', 
                       fontweight='bold', fontsize=13,
                       bbox=dict(boxstyle="round,pad=0.4", facecolor='white', 
                               edgecolor=color, alpha=0.9, linewidth=2))
        else:
            # Line plot for quality metrics
            line = ax.plot(indices, valid_values, 'o-', linewidth=5, markersize=15,
                          color=color, markerfacecolor='white', 
                          markeredgewidth=4, markeredgecolor=color)[0]
            
            # Add gradient fill
            ax.fill_between(indices, valid_values, alpha=0.2, color=color)
            
            # Add value labels
            for i, val in zip(indices, valid_values):
                ax.annotate(format_str.format(val), (i, val), 
                           textcoords="offset points", xytext=(0,20), ha='center', 
                           fontweight='bold', fontsize=13,
                           bbox=dict(boxstyle="round,pad=0.4", facecolor='white', 
                                   edgecolor=color, alpha=0.9, linewidth=2))
        
        # Beautiful title with background
        ax.set_title(f'{title}\n{subtitle}', fontsize=18, fontweight='bold', pad=30,
                    bbox=dict(boxstyle="round,pad=0.8", facecolor="#F8F9FA", 
                             edgecolor=color, alpha=0.9, linewidth=3))
        
        ax.set_ylabel(ylabel, fontsize=16, fontweight='bold')
        ax.set_xlabel('Pipeline Stage', fontsize=16, fontweight='bold')
        ax.set_xticks(indices)
        ax.set_xticklabels([name.split('\n')[0] for name in valid_names], 
                          rotation=45, ha='right', fontsize=12)
        
        # Improve grid and spines
        ax.grid(True, alpha=0.3, linewidth=1.2)
        ax.set_axisbelow(True)
        
        # Add subtle background gradient
        ax.set_facecolor('#FAFBFC')
        
        plt.tight_layout()
        return fig
    
    def create_timing_visualization(self, scene, timing_summary, output_dir):
        """Create timing table visualization with stage times and total"""
        if not timing_summary['stage_timings'] and not timing_summary['detailed_breakdowns']:
            print(f"⚠️  No timing data available for {scene}")
            return None
        
        # Create clean timing table (single plot with table)
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        fig.suptitle(f'🕒 Pipeline Timing Analysis - {scene.upper()}', 
                    fontsize=16, fontweight='bold', y=0.95)
        
        ax.axis('off')
        
        # Prepare table data
        table_data = []
        total_time = 0
        
        if timing_summary['stage_timings']:
            for stage, time_val in timing_summary['stage_timings'].items():
                # Clean stage name
                clean_stage = stage.replace('Pipeline Step: ', '').replace('Step', '').strip()
                time_str = format_3_sig_figs(time_val)
                minutes_str = format_3_sig_figs(time_val / 60)
                table_data.append([clean_stage, f"{time_str}s", f"{minutes_str}m"])
                total_time += time_val
        
        # Add total row
        if total_time > 0:
            total_time_str = format_3_sig_figs(total_time)
            total_minutes_str = format_3_sig_figs(total_time / 60)
            table_data.append(['━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━', '━━━━━━━━━━━━━━━━━', '━━━━━━━━━━━━━━━━━'])
            table_data.append([f'🏁 TOTAL PIPELINE TIME', f"{total_time_str}s", f"{total_minutes_str}m"])
        
        if table_data:
            # Create beautiful table
            table = ax.table(cellText=table_data,
                           colLabels=['Pipeline Stage', 'Duration (seconds)', 'Duration (minutes)'],
                           cellLoc='center',
                           loc='center',
                           colWidths=[0.55, 0.22, 0.23])
            
            table.auto_set_font_size(False)
            table.set_fontsize(12)
            table.scale(1, 2.5)
            
            # Beautiful table styling
            for i in range(len(table_data) + 1):
                for j in range(3):
                    cell = table[(i, j)]
                    if i == 0:  # Header row
                        cell.set_facecolor('#2E86AB')
                        cell.set_text_props(weight='bold', color='white', fontsize=14)
                    elif i == len(table_data):  # Total row
                        cell.set_facecolor('#F39F5A')
                        cell.set_text_props(weight='bold', fontsize=13)
                    elif i == len(table_data) - 1:  # Separator row
                        cell.set_facecolor('#FFFFFF')
                        cell.set_text_props(color='#CCCCCC', fontsize=10)
                    else:  # Regular data rows
                        if i % 2 == 1:
                            cell.set_facecolor('#F8F9FA')  # Light gray for even rows
                        else:
                            cell.set_facecolor('#FFFFFF')  # White for odd rows
                        cell.set_text_props(fontsize=11)
                    
                    # Set border properties
                    cell.set_edgecolor('#DDDDDD')
                    cell.set_linewidth(1)
            
            # Add subtitle with summary
            if total_time > 0:
                hours = total_time / 3600
                if hours >= 1:
                    summary_text = f"Total: {format_3_sig_figs(total_time)}s ({format_3_sig_figs(total_time/60)}m = {format_3_sig_figs(hours)}h)"
                else:
                    summary_text = f"Total: {format_3_sig_figs(total_time)}s ({format_3_sig_figs(total_time/60)} minutes)"
                
                ax.text(0.5, 0.15, summary_text, 
                       horizontalalignment='center', verticalalignment='center',
                       transform=ax.transAxes, fontsize=14, fontweight='bold',
                       bbox=dict(boxstyle="round,pad=0.5", facecolor='#E8F4FD', 
                                edgecolor='#2E86AB', linewidth=2))
        
        plt.tight_layout()
        
        # Save timing plot
        timing_plot_path = output_dir / f"{scene}_timing_analysis.png"
        plt.savefig(timing_plot_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close()
        
        print(f"⏱️  Timing analysis: {timing_plot_path.name}")
        return timing_plot_path

    def analyze_scene(self, scene):
        """Analyze a single scene and save beautiful plots"""
        print(f"\n🔍 Analyzing {scene}...")
        
        # Load data and extract pipeline stages
        scene_data = self.load_scene_data(scene)
        stages, data_sources = self.extract_pipeline_stages(scene_data)
        
        print(f"📊 Found {len(stages)} pipeline stages:")
        for i, (stage, source) in enumerate(zip(stages, data_sources)):
            print(f"  {i+1}. {stage['name'].replace(chr(10), ' ')} - {source}")
        
        output_dir = self.base_path / "syn_sh16" / scene / "analysis"
        output_dir.mkdir(exist_ok=True)
        
        saved_plots = []
        
        # Create comprehensive plot
        comprehensive_fig = self.create_beautiful_plot(scene, stages, data_sources, 'comprehensive')
        if comprehensive_fig:
            comprehensive_path = output_dir / f"{scene}_pipeline_analysis.png"
            comprehensive_fig.savefig(comprehensive_path, dpi=150, bbox_inches='tight', 
                                    facecolor='white', edgecolor='none')
            plt.close(comprehensive_fig)
            saved_plots.append(comprehensive_path)
            print(f"✅ Comprehensive analysis: {comprehensive_path}")
        
        # Create individual plots
        plot_types = ['psnr', 'ssim', 'lpips', 'memory']
        for plot_type in plot_types:
            individual_fig = self.create_beautiful_plot(scene, stages, data_sources, plot_type)
            if individual_fig:
                individual_path = output_dir / f"{scene}_{plot_type}_progression.png"
                individual_fig.savefig(individual_path, dpi=150, bbox_inches='tight',
                                     facecolor='white', edgecolor='none')
                plt.close(individual_fig)
                saved_plots.append(individual_path)
                print(f"   📊 {individual_path.name}")
        
        # Create timing visualization
        timing_summary = self.extract_timing_summary(scene_data)
        timing_path = self.create_timing_visualization(scene, timing_summary, output_dir)
        if timing_path:
            saved_plots.append(timing_path)
        
        return saved_plots
    
    def analyze_all_scenes(self):
        """Auto-discover and analyze all scenes with beautiful plots"""
        scenes = self.discover_scenes()
        
        if not scenes:
            print("❌ No scenes found!")
            return
        
        print(f"\n🚀 Creating beautiful analysis plots for {len(scenes)} scenes...")
        print("📈 Using proper averaging and post-compression evaluation")
        
        for scene in scenes:
            try:
                self.analyze_scene(scene)
            except Exception as e:
                print(f"❌ Error analyzing {scene}: {e}")
                import traceback
                traceback.print_exc()
        
        print(f"\n✅ Beautiful analysis complete! All plots saved with modern styling.")


def main():
    analyzer = SimplePlenOctreeAnalyzer()
    analyzer.analyze_all_scenes()


if __name__ == "__main__":
    main()