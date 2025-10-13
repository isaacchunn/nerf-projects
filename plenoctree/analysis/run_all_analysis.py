#!/usr/bin/env python3
"""
Master Analysis Runner

Runs all analysis scripts in the correct order:
1. Efficiency metrics analyzer (per-scene analysis with updated metrics)
2. Experiment analyzer (per-scene pipeline analysis)
3. Cross-experiment visualizer (overall comparison)
"""

import sys
from pathlib import Path

# Add analysis directory to path
sys.path.insert(0, str(Path(__file__).parent))

from efficiency_metrics_analyzer import EfficiencyMetricsAnalyzer
from experiment_analyzer import SimplePlenOctreeAnalyzer
from cross_experiment_visualizer import CrossExperimentVisualizer
from enhanced_scene_analyzer import EnhancedSceneAnalyzer


def main():
    """Run all analysis scripts in sequence."""
    
    print("="*80)
    print("🚀 MASTER ANALYSIS RUNNER")
    print("="*80)
    print("\nThis script will run:")
    print("  1. Efficiency Metrics Analyzer (updated with MEPV)")
    print("  2. Experiment Analyzer (per-scene pipeline analysis)")
    print("  3. Cross-Experiment Visualizer (overall comparison)")
    print("  4. Enhanced Scene Analyzer (comprehensive detailed plots)")
    print("\n" + "="*80 + "\n")
    
    # Step 1: Run efficiency metrics analyzer
    print("\n" + "="*80)
    print("STEP 1: EFFICIENCY METRICS ANALYZER")
    print("="*80 + "\n")
    
    try:
        efficiency_analyzer = EfficiencyMetricsAnalyzer()
        efficiency_results = efficiency_analyzer.analyze_all_scenes()
        print(f"\n✅ Efficiency metrics analysis complete for {len(efficiency_results)} scenes")
    except Exception as e:
        print(f"\n❌ Error in efficiency metrics analyzer: {e}")
        import traceback
        traceback.print_exc()
    
    # Step 2: Run experiment analyzer
    print("\n" + "="*80)
    print("STEP 2: EXPERIMENT ANALYZER (PIPELINE ANALYSIS)")
    print("="*80 + "\n")
    
    try:
        experiment_analyzer = SimplePlenOctreeAnalyzer()
        experiment_analyzer.analyze_all_scenes()
        print("\n✅ Experiment analysis complete")
    except Exception as e:
        print(f"\n❌ Error in experiment analyzer: {e}")
        import traceback
        traceback.print_exc()
    
    # Step 3: Run cross-experiment visualizer
    print("\n" + "="*80)
    print("STEP 3: CROSS-EXPERIMENT VISUALIZER (OVERALL COMPARISON)")
    print("="*80 + "\n")
    
    try:
        cross_visualizer = CrossExperimentVisualizer()
        cross_visualizer.generate_all_visualizations()
        print("\n✅ Cross-experiment visualization complete")
    except Exception as e:
        print(f"\n❌ Error in cross-experiment visualizer: {e}")
        import traceback
        traceback.print_exc()
    
    # Step 4: Run enhanced scene analyzer
    print("\n" + "="*80)
    print("STEP 4: ENHANCED SCENE ANALYZER (COMPREHENSIVE DETAILED PLOTS)")
    print("="*80 + "\n")
    
    try:
        enhanced_analyzer = EnhancedSceneAnalyzer()
        enhanced_analyzer.analyze_all_scenes()
        print("\n✅ Enhanced scene analysis complete")
    except Exception as e:
        print(f"\n❌ Error in enhanced scene analyzer: {e}")
        import traceback
        traceback.print_exc()
    
    # Final summary
    print("\n" + "="*80)
    print("🎉 ALL ANALYSIS COMPLETE!")
    print("="*80)
    print("\nGenerated outputs:")
    print("\n📁 Per-Scene Analysis (in each scene's analysis/ folder):")
    print("\n  Standard Visualizations (7):")
    print("    • {scene}_efficiency_metrics.png - Updated with MEPV (Higher is Better)")
    print("    • {scene}_pipeline_analysis.png - Comprehensive pipeline overview")
    print("    • {scene}_psnr_progression.png - PSNR evolution")
    print("    • {scene}_ssim_progression.png - SSIM evolution")
    print("    • {scene}_lpips_progression.png - LPIPS evolution")
    print("    • {scene}_memory_progression.png - Memory usage")
    print("    • {scene}_timing_analysis.png - Timing breakdown")
    print("\n  Enhanced Visualizations (4) - NEW!")
    print("    • {scene}_detailed_memory_analysis.png - Current vs Peak memory analysis")
    print("    • {scene}_efficiency_comparison_detailed.png - All efficiency metrics compared")
    print("    • {scene}_quality_metrics_detailed.png - Comprehensive quality analysis")
    print("    • {scene}_complete_training_progression.png - 9-panel training overview")
    print("\n  Data Files:")
    print("    • {scene}_efficiency_metrics.csv - All metrics data")
    
    print("\n📁 Cross-Experiment Analysis (in checkpoints/analysis/ folder):")
    print("  • quality_metrics_comparison.png - PSNR, SSIM, LPIPS across scenes")
    print("  • memory_efficiency_comparison.png - Memory & efficiency metrics")
    print("  • quality_memory_tradeoff.png - Scatter plots")
    print("  • radar_comparison.png - Multi-metric radar charts")
    print("  • summary_table.png - Comprehensive table")
    print("  • cross_experiment_comparison.csv - Raw data")
    print("  • cross_scene_efficiency_comparison.csv - Efficiency comparison")
    
    print("\n💡 Key Updates:")
    print("  ✓ Standardized memory measurements (nvidia-smi priority)")
    print("  ✓ Fixed MSF → MEPV (Higher is Better)")
    print("  ✓ Cross-scene comparison with unified methodology")
    print("  ✓ Beautiful, publication-ready visualizations")
    
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()

