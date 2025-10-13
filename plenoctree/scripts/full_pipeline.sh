#!/bin/bash

# PlenOctree Full Pipeline: Training + Conversion
# This script combines NeRF training and octree conversion into one complete pipeline

# Better error handling
set -o pipefail  # Capture errors in pipes
trap 'echo "❌ Pipeline aborted due to error on line $LINENO"' ERR

# =============================================================================
# CONFIGURATION - Edit these paths as needed
# =============================================================================

# Data and output paths
export DATA_ROOT="/mnt/d/GitHub/nerf-projects/plenoctree/data/NeRF/nerf_synthetic"
export CKPT_ROOT="/mnt/d/GitHub/nerf-projects/plenoctree/data/Plenoctree/checkpoints/syn_sh16"
export SCENE="chair"
export CONFIG_FILE="nerf_sh/config/blender"

# Training configuration
export EXTRA_FLAGS=""  # Add training flags like "--lr_delay_steps 50000 --lr_delay_mult 0.01" for harder scenes

# Pipeline options
SKIP_TRAINING=false          # Set to true to skip training and only do conversion
SKIP_CONVERSION=false        # Set to true to only do training
FORCE_RETRAIN=false          # Set to true to retrain even if checkpoint exists

# Available scenes
ALL_SCENES="chair drums ficus hotdog lego materials mic ship"

# Multi-scene support
SCENES_TO_RUN=""
RUN_ALL_SCENES=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --scene)
            SCENE="$2"
            SCENES_TO_RUN="$2"
            shift 2
            ;;
        --scenes)
            SCENES_TO_RUN=$(echo "$2" | tr ',' ' ')
            shift 2
            ;;
        --all-scenes)
            RUN_ALL_SCENES=true
            SCENES_TO_RUN="$ALL_SCENES"
            shift
            ;;
        --skip-training)
            SKIP_TRAINING=true
            shift
            ;;
        --skip-conversion)
            SKIP_CONVERSION=true
            shift
            ;;
        --force-retrain)
            FORCE_RETRAIN=true
            shift
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --scene SCENE        Set the scene name (default: chair)"
            echo "  --scenes SCENE1,SCENE2,...  Run multiple scenes (comma-separated)"
            echo "  --all-scenes         Run all available scenes: $ALL_SCENES"
            echo "  --skip-training      Skip training and only do octree conversion"
            echo "  --skip-conversion    Only do training, skip octree conversion"
            echo "  --force-retrain      Force retraining even if checkpoint exists"
            echo "  --help              Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 --scene chair --skip-training"
            echo "  $0 --scenes chair,drums,ficus --skip-training"
            echo "  $0 --all-scenes --skip-training"
            exit 0
            ;;
        *)
            # If it's not a flag, treat it as a scene name (backward compatibility)
            if [[ ! "$1" =~ ^-- ]]; then
                SCENE="$1"
                SCENES_TO_RUN="$1"
                shift
            else
                echo "Unknown option: $1"
                echo "Use --help for usage information"
                exit 1
            fi
            ;;
    esac
done

# Set default if no scenes specified
if [ -z "$SCENES_TO_RUN" ]; then
    SCENES_TO_RUN="$SCENE"
fi

# =============================================================================
# MULTI-SCENE PIPELINE EXECUTION
# =============================================================================

# Count total scenes for progress tracking
TOTAL_SCENES=$(echo $SCENES_TO_RUN | wc -w)
CURRENT_SCENE_NUM=0

echo "═══════════════════════════════════════════════════════════════"
echo "🚀 PlenOctree Multi-Scene Pipeline"
echo "═══════════════════════════════════════════════════════════════"
echo "📅 Started: $(date)"
echo "🎯 Scenes to process: $SCENES_TO_RUN"
echo "📊 Total scenes: $TOTAL_SCENES"
echo "⚙️  Skip training: $SKIP_TRAINING"
echo "⚙️  Skip conversion: $SKIP_CONVERSION"
echo "⚙️  Force retrain: $FORCE_RETRAIN"
echo ""

# Loop through each scene
for CURRENT_SCENE in $SCENES_TO_RUN; do
    CURRENT_SCENE_NUM=$((CURRENT_SCENE_NUM + 1))
    
    echo ""
    echo "═══════════════════════════════════════════════════════════════"
    echo "🎯 Processing Scene $CURRENT_SCENE_NUM/$TOTAL_SCENES: $CURRENT_SCENE"
    echo "═══════════════════════════════════════════════════════════════"
    
    # Set scene-specific variables
    SCENE="$CURRENT_SCENE"
    PIPELINE_START_TIME=$(date +%s)
    PIPELINE_LOG_FILE="$CKPT_ROOT/$SCENE/full_pipeline_metrics.json"
    
    echo "📁 Data dir: $DATA_ROOT/$SCENE"
    echo "💾 Checkpoint dir: $CKPT_ROOT/$SCENE"
    echo "⚙️  Config: $CONFIG_FILE"
    echo "📊 Pipeline log: $PIPELINE_LOG_FILE"
    echo ""

# Function to log pipeline step metrics
log_pipeline_step() {
    local step_num="$1"
    local step_name="$2"
    local status="$3"
    local duration="$4"
    local phase="$5"
    local timestamp=$(date -Iseconds)
    
    # Create or append to pipeline log
    mkdir -p "$(dirname "$PIPELINE_LOG_FILE")"
    
    # Create JSON entry for this step
    cat << EOF >> "$PIPELINE_LOG_FILE"
{
  "timestamp": "$timestamp",
  "step": $step_num,
  "phase": "$phase",
  "metrics": {
    "step_name": "$step_name",
    "status": "$status",
    "duration_seconds": $duration,
    "scene": "$SCENE",
    "config": "$CONFIG_FILE"
  },
  "additional_info": {
    "data_root": "$DATA_ROOT",
    "checkpoint_root": "$CKPT_ROOT",
    "pipeline_type": "full_pipeline"
  }
},
EOF
}

# Function to run commands with error handling and timing
run_step() {
    local step_name="$1"
    local step_num="$2"
    local phase="$3"
    local allow_failure="$4"  # true/false
    shift 4
    
    echo "Step $step_num: $step_name..."
    local step_start_time=$(date +%s)
    
    if "$@"; then
        local step_end_time=$(date +%s)
        local step_duration=$((step_end_time - step_start_time))
        echo "✅ Step $step_num completed successfully in ${step_duration}s"
        log_pipeline_step "$step_num" "$step_name" "success" "$step_duration" "$phase"
        return 0
    else
        local step_end_time=$(date +%s)
        local step_duration=$((step_end_time - step_start_time))
        
        if [ "$allow_failure" = "true" ]; then
            echo "⚠️  Step $step_num failed - $step_name (${step_duration}s)"
            echo "   Continuing with next step..."
            log_pipeline_step "$step_num" "$step_name" "failed" "$step_duration" "$phase"
            return 1
        else
            echo "❌ Step $step_num failed - $step_name (${step_duration}s)"
            echo "   Aborting pipeline due to critical failure"
            log_pipeline_step "$step_num" "$step_name" "failed" "$step_duration" "$phase"
            exit 1
        fi
    fi
}

# Create necessary directories
mkdir -p "$CKPT_ROOT/$SCENE"
mkdir -p "$CKPT_ROOT/$SCENE/octrees"

# =============================================================================
# PHASE 1: TRAINING
# =============================================================================

if [ "$SKIP_TRAINING" = "false" ]; then
    echo "══════════════════════════════════════════════════════════════"
    echo "🎓 PHASE 1: NeRF TRAINING"
    echo "══════════════════════════════════════════════════════════════"
    
    # Check if training should be skipped
    CHECKPOINT_EXISTS=false
    if ls "$CKPT_ROOT/$SCENE/"checkpoint_* 1> /dev/null 2>&1; then
        CHECKPOINT_EXISTS=true
        LATEST_CHECKPOINT=$(ls -t "$CKPT_ROOT/$SCENE/"checkpoint_* | head -1)
        echo "📋 Found existing checkpoint: $(basename "$LATEST_CHECKPOINT")"
    fi
    
    if [ "$CHECKPOINT_EXISTS" = "true" ] && [ "$FORCE_RETRAIN" = "false" ]; then
        echo "⏭️  Skipping training - checkpoint exists. Use --force-retrain to override."
        echo ""
    else
        if [ "$FORCE_RETRAIN" = "true" ] && [ "$CHECKPOINT_EXISTS" = "true" ]; then
            echo "🔄 Force retraining enabled - removing existing checkpoints"
            rm -f "$CKPT_ROOT/$SCENE/"checkpoint_*
        fi
        
        # Step T1: NeRF Training
        run_step "NeRF training" "T1" "training" "false" \
            python -m nerf_sh.train \
                --train_dir "$CKPT_ROOT/$SCENE" \
                --config "$CONFIG_FILE" \
                --data_dir "$DATA_ROOT/$SCENE" \
                $EXTRA_FLAGS
        
        # Step T2: NeRF Evaluation  
        run_step "NeRF evaluation" "T2" "training" "true" \
            python -m nerf_sh.eval \
                --chunk 4096 \
                --train_dir "$CKPT_ROOT/$SCENE" \
                --config "$CONFIG_FILE" \
                --data_dir "$DATA_ROOT/$SCENE"
    fi
    
    echo ""
else
    echo "⏭️  Skipping training phase (--skip-training specified)"
    echo ""
fi

# =============================================================================
# PHASE 2: OCTREE CONVERSION
# =============================================================================

if [ "$SKIP_CONVERSION" = "false" ]; then
    echo "══════════════════════════════════════════════════════════════"
    echo "🌳 PHASE 2: OCTREE CONVERSION"
    echo "══════════════════════════════════════════════════════════════"
    
    # Verify checkpoint exists
    if ! ls "$CKPT_ROOT/$SCENE/"checkpoint_* 1> /dev/null 2>&1; then
        echo "❌ No checkpoint found for conversion. Please run training first."
        exit 1
    fi
    
    # Step O1: Extract octree from NeRF
    run_step "Extracting octree from NeRF model" "O1" "octree_conversion" "false" \
        python -m octree.extraction \
            --train_dir "$CKPT_ROOT/$SCENE/" --is_jaxnerf_ckpt \
            --config "$CONFIG_FILE" \
            --data_dir "$DATA_ROOT/$SCENE/" \
            --output "$CKPT_ROOT/$SCENE/octrees/tree.npz"
    
    # Step O2: Evaluate initial octree (baseline)
    if [ -f "$CKPT_ROOT/$SCENE/octrees/tree.npz" ]; then
        run_step "Evaluating initial octree (baseline)" "O2" "octree_conversion" "true" \
            python -m octree.evaluation \
                --input "$CKPT_ROOT/$SCENE/octrees/tree.npz" \
                --config "$CONFIG_FILE" \
                --data_dir "$DATA_ROOT/$SCENE/"
    else
        echo "❌ Step O2 skipped - tree.npz not found"
    fi
    
    # Step O3: Optimize octree
    if [ -f "$CKPT_ROOT/$SCENE/octrees/tree.npz" ]; then
        run_step "Optimizing octree" "O3" "octree_conversion" "false" \
            python -m octree.optimization \
                --input "$CKPT_ROOT/$SCENE/octrees/tree.npz" \
                --config "$CONFIG_FILE" \
                --data_dir "$DATA_ROOT/$SCENE/" \
                --output "$CKPT_ROOT/$SCENE/octrees/tree_opt.npz"
    else
        echo "❌ Step O3 skipped - tree.npz not found"
    fi
    
    # Step O4: Evaluate optimized octree
    if [ -f "$CKPT_ROOT/$SCENE/octrees/tree_opt.npz" ]; then
        run_step "Evaluating optimized octree" "O4" "octree_conversion" "true" \
            python -m octree.evaluation \
                --input "$CKPT_ROOT/$SCENE/octrees/tree_opt.npz" \
                --config "$CONFIG_FILE" \
                --data_dir "$DATA_ROOT/$SCENE/"
    else
        echo "❌ Step O4 skipped - tree_opt.npz not found"
    fi
    
    # Step O5: Compress for web
    if [ -f "$CKPT_ROOT/$SCENE/octrees/tree_opt.npz" ]; then
        run_step "Compressing for web viewing" "O5" "octree_conversion" "true" \
            python -m octree.compression \
                "$CKPT_ROOT/$SCENE/octrees/tree_opt.npz" \
                --out_dir "$CKPT_ROOT/$SCENE/octrees/" \
                --overwrite
    else
        echo "❌ Step O5 skipped - tree_opt.npz not found"
    fi
    
    # Step O6: Evaluate compressed octree
    COMPRESSED_OCTREE="$CKPT_ROOT/$SCENE/octrees/tree_opt_compressed.npz"
    if [ -f "$COMPRESSED_OCTREE" ]; then
        echo "🔍 Found compressed octree: $(basename "$COMPRESSED_OCTREE")"
        run_step "Evaluating compressed octree quality" "O6" "octree_conversion" "true" \
            python -m octree.compressed_evaluation \
                --input "$COMPRESSED_OCTREE" \
                --config "$CONFIG_FILE" \
                --data_dir "$DATA_ROOT/$SCENE/"
    else
        echo "❌ Step O6 skipped - compressed octree not found"
        echo "   Expected: $COMPRESSED_OCTREE"
    fi
    
    echo ""
else
    echo "⏭️  Skipping octree conversion phase (--skip-conversion specified)"
    echo ""
fi

# =============================================================================
# PIPELINE SUMMARY
# =============================================================================

PIPELINE_END_TIME=$(date +%s)
TOTAL_PIPELINE_TIME=$((PIPELINE_END_TIME - PIPELINE_START_TIME))

echo "══════════════════════════════════════════════════════════════"
echo "📊 PIPELINE SUMMARY"
echo "══════════════════════════════════════════════════════════════"
echo "⏱️  Total pipeline time: ${TOTAL_PIPELINE_TIME}s ($((TOTAL_PIPELINE_TIME / 60))m $((TOTAL_PIPELINE_TIME % 60))s)"
echo "📁 Results directory: $CKPT_ROOT/$SCENE/"
echo ""

# Check what files were created and get their sizes
CHECKPOINT_COUNT=0
RAW_OCTREE_PATH="$CKPT_ROOT/$SCENE/octrees/tree.npz"
OPT_OCTREE_PATH="$CKPT_ROOT/$SCENE/octrees/tree_opt.npz"
COMPRESSED_OCTREE_PATH="$CKPT_ROOT/$SCENE/octrees/tree_opt_compressed.npz"

# Count checkpoints
if ls "$CKPT_ROOT/$SCENE/"checkpoint_* 1> /dev/null 2>&1; then
    CHECKPOINT_COUNT=$(ls "$CKPT_ROOT/$SCENE/"checkpoint_* | wc -l)
    LATEST_CHECKPOINT=$(ls -t "$CKPT_ROOT/$SCENE/"checkpoint_* | head -1)
    echo "✅ NeRF checkpoints: $CHECKPOINT_COUNT found"
    echo "   Latest: $(basename "$LATEST_CHECKPOINT")"
else
    echo "❌ NeRF checkpoints: NOT FOUND"
fi

# Check octree files
check_file_and_size() {
    local file_path="$1"
    local file_desc="$2"
    
    if [ -f "$file_path" ]; then
        local file_size=$(stat -f%z "$file_path" 2>/dev/null || stat -c%s "$file_path" 2>/dev/null || echo 0)
        local file_size_mb=$((file_size / 1024 / 1024))
        echo "✅ $file_desc: $(basename "$file_path") (${file_size_mb}MB)"
        return 0
    else
        echo "❌ $file_desc: NOT CREATED"
        return 1
    fi
}

check_file_and_size "$RAW_OCTREE_PATH" "Raw octree"
check_file_and_size "$OPT_OCTREE_PATH" "Optimized octree"
check_file_and_size "$COMPRESSED_OCTREE_PATH" "Compressed octree"

echo ""
echo "📊 Detailed metrics available in:"
if [ "$SKIP_TRAINING" = "false" ]; then
    echo "  🎓 Training metrics: Check NeRF logs in $CKPT_ROOT/$SCENE/"
fi
if [ "$SKIP_CONVERSION" = "false" ]; then
    echo "  🌳 Extraction: $CKPT_ROOT/$SCENE/octrees/octree_extraction_metrics.json"
    echo "  ⚡ Optimization: $CKPT_ROOT/$SCENE/octrees/octree_optimization_metrics.json"
    echo "  📈 Evaluation: $CKPT_ROOT/$SCENE/octrees/octree_evaluation_metrics.json"
    echo "  🗜️  Compression: $CKPT_ROOT/$SCENE/octrees/octree_compression_metrics.json"
    echo "  🎯 Compressed eval: $CKPT_ROOT/$SCENE/octree_compression_evaluation_metrics.json"
fi
echo "  📋 Full pipeline: $PIPELINE_LOG_FILE"

# Log final pipeline summary
FINAL_TIMESTAMP=$(date -Iseconds)
cat << EOF >> "$PIPELINE_LOG_FILE"
{
  "timestamp": "$FINAL_TIMESTAMP",
  "step": 999,
  "phase": "pipeline_summary",
  "metrics": {
    "total_pipeline_time_seconds": $TOTAL_PIPELINE_TIME,
    "checkpoint_count": $CHECKPOINT_COUNT,
    "raw_octree_created": $([ -f "$RAW_OCTREE_PATH" ] && echo "true" || echo "false"),
    "optimized_octree_created": $([ -f "$OPT_OCTREE_PATH" ] && echo "true" || echo "false"),
    "compressed_octree_created": $([ -f "$COMPRESSED_OCTREE_PATH" ] && echo "true" || echo "false"),
    "scene": "$SCENE",
    "config": "$CONFIG_FILE",
    "skip_training": $SKIP_TRAINING,
    "skip_conversion": $SKIP_CONVERSION,
    "force_retrain": $FORCE_RETRAIN
  },
  "additional_info": {
    "data_root": "$DATA_ROOT",
    "checkpoint_root": "$CKPT_ROOT",
    "pipeline_type": "full_pipeline",
    "completed_at": "$FINAL_TIMESTAMP"
  }
}
EOF

# Clean up the JSON file (remove trailing comma and wrap in array)
if [ -f "$PIPELINE_LOG_FILE" ]; then
    sed '$ s/,$//' "$PIPELINE_LOG_FILE" > "${PIPELINE_LOG_FILE}.tmp"
    echo '[' > "$PIPELINE_LOG_FILE"
    cat "${PIPELINE_LOG_FILE}.tmp" >> "$PIPELINE_LOG_FILE"
    echo ']' >> "$PIPELINE_LOG_FILE"
    rm -f "${PIPELINE_LOG_FILE}.tmp"
fi

    echo ""
    echo "🎉 Scene $CURRENT_SCENE completed in ${TOTAL_PIPELINE_TIME}s!"
    echo "📅 Scene finished: $(date)"
    echo ""
    echo "✅ Scene $CURRENT_SCENE_NUM/$TOTAL_SCENES ($CURRENT_SCENE) completed!"
    
done  # End of scene loop

# Final summary
OVERALL_END_TIME=$(date +%s)
OVERALL_TOTAL_TIME=$((OVERALL_END_TIME - $(date +%s)))  # This will be recalculated properly

echo ""
echo "══════════════════════════════════════════════════════════════"
echo "🎊 ALL SCENES COMPLETED!"
echo "══════════════════════════════════════════════════════════════"
echo "📊 Processed $TOTAL_SCENES scenes: $SCENES_TO_RUN"
echo "📅 Overall finished: $(date)"

# Auto-generate analysis plots for all processed scenes (once at the end)
echo ""
echo "📊 Auto-generating comprehensive analysis plots for all scenes..."

# Run individual scene analysis
if python analysis/experiment_analyzer.py; then
    echo "✅ Individual scene analysis completed!"
else
    echo "⚠️  Individual scene analysis failed"
fi

# Run cross-experiment comparison analysis
echo "📊 Generating cross-experiment comparison plots..."
if python analysis/cross_experiment_visualizer.py; then
    echo "✅ Cross-experiment analysis completed!"
else
    echo "⚠️  Cross-experiment analysis failed"
fi

# Run efficiency metrics analysis
echo "📊 Generating efficiency metrics analysis..."
if python analysis/efficiency_metrics_analyzer.py; then
    echo "✅ Efficiency metrics analysis completed!"
else
    echo "⚠️  Efficiency metrics analysis failed"
fi

echo ""
echo "📁 Results locations:"
echo "  • Individual scene plots: data/Plenoctree/checkpoints/syn_sh16/*/analysis/"
echo "  • Cross-scene comparisons: data/Plenoctree/checkpoints/analysis/"
echo "  • Efficiency analysis: data/Plenoctree/checkpoints/cross_scene_efficiency_comparison.csv"

echo "══════════════════════════════════════════════════════════════"
