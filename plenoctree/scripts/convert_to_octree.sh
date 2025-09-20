#!/bin/bash

# Better error handling - continue on errors but report them
set -o pipefail  # Capture errors in pipes
trap 'echo "❌ Script aborted due to error on line $LINENO"' ERR

# Configuration - Edit these paths as needed
export DATA_ROOT="/mnt/d/GitHub/nerf-projects/plenoctree/data/NeRF/nerf_synthetic"
export CKPT_ROOT="/mnt/d/GitHub/nerf-projects/plenoctree/data/Plenoctree/checkpoints/syn_sh16"
export SCENE="chair"
# export CHECKPOINT_NUM="checkpoint_200000"  # Specify the exact checkpoint to use
export CONFIG_FILE="nerf_sh/config/blender"

# Pipeline timing and logging
PIPELINE_START_TIME=$(date +%s)
PIPELINE_LOG_FILE="$CKPT_ROOT/$SCENE/octrees/pipeline_summary_metrics.json"

echo "=== PlenOctree Conversion Pipeline ==="
echo "Scene: $SCENE"
# echo "Checkpoint dir: $CKPT_ROOT/$SCENE/$CHECKPOINT_NUM"
echo "Checkpoint dir: $CKPT_ROOT/$SCENE/"
echo "Data dir: $DATA_ROOT/$SCENE"
echo "Pipeline log: $PIPELINE_LOG_FILE"
echo ""

# Function to log pipeline step metrics
log_pipeline_step() {
    local step_num="$1"
    local step_name="$2"
    local status="$3"
    local duration="$4"
    local timestamp=$(date -Iseconds)
    
    # Create or append to pipeline log
    mkdir -p "$(dirname "$PIPELINE_LOG_FILE")"
    
    # Create JSON entry for this step
    cat << EOF >> "$PIPELINE_LOG_FILE"
{
  "timestamp": "$timestamp",
  "step": $step_num,
  "phase": "pipeline_step",
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
    "phase": "octree_pipeline"
  }
},
EOF
}

# Function to run commands with error handling and timing
run_step() {
    local step_name="$1"
    local step_num="$2"
    shift 2
    
    echo "Step $step_num: $step_name..."
    local step_start_time=$(date +%s)
    
    if "$@"; then
        local step_end_time=$(date +%s)
        local step_duration=$((step_end_time - step_start_time))
        echo "✅ Step $step_num completed successfully in ${step_duration}s"
        log_pipeline_step "$step_num" "$step_name" "success" "$step_duration"
        return 0
    else
        local step_end_time=$(date +%s)
        local step_duration=$((step_end_time - step_start_time))
        echo "❌ Step $step_num failed - $step_name (${step_duration}s)"
        echo "⚠️  Continuing with next step..."
        log_pipeline_step "$step_num" "$step_name" "failed" "$step_duration"
        return 1
    fi
}

# Create octrees directory if it doesn't exist
mkdir -p "$CKPT_ROOT/$SCENE/octrees"

# Step 1: Extract octree
run_step "Extracting octree from NeRF model" "1" \
    python -m octree.extraction \
    --train_dir "$CKPT_ROOT/$SCENE/" --is_jaxnerf_ckpt \
    --config "$CONFIG_FILE" \
    --data_dir "$DATA_ROOT/$SCENE/" \
    --output "$CKPT_ROOT/$SCENE/octrees/tree.npz"

echo ""

# Step 2: Evaluate initial octree (baseline)
if [ -f "$CKPT_ROOT/$SCENE/octrees/tree.npz" ]; then
    run_step "Evaluating initial octree (baseline)" "2" \
        python -m octree.evaluation \
        --input "$CKPT_ROOT/$SCENE/octrees/tree.npz" \
        --config "$CONFIG_FILE" \
        --data_dir "$DATA_ROOT/$SCENE/"
else
    echo "❌ Step 2 skipped - tree.npz not found"
fi

echo ""

# Step 3: Optimize octree
if [ -f "$CKPT_ROOT/$SCENE/octrees/tree.npz" ]; then
    run_step "Optimizing octree" "3" \
        python -m octree.optimization \
        --input "$CKPT_ROOT/$SCENE/octrees/tree.npz" \
        --config "$CONFIG_FILE" \
        --data_dir "$DATA_ROOT/$SCENE/" \
        --output "$CKPT_ROOT/$SCENE/octrees/tree_opt.npz"
else
    echo "❌ Step 3 skipped - tree.npz not found"
fi

echo ""

# Step 4: Evaluate optimized octree
if [ -f "$CKPT_ROOT/$SCENE/octrees/tree_opt.npz" ]; then
    run_step "Evaluating optimized octree" "4" \
        python -m octree.evaluation \
        --input "$CKPT_ROOT/$SCENE/octrees/tree_opt.npz" \
        --config "$CONFIG_FILE" \
        --data_dir "$DATA_ROOT/$SCENE/"
else
    echo "❌ Step 4 skipped - tree_opt.npz not found"
fi

echo ""

# Step 5: Compress for web
if [ -f "$CKPT_ROOT/$SCENE/octrees/tree_opt.npz" ]; then
    run_step "Compressing for web viewing" "5" \
        python -m octree.compression \
        "$CKPT_ROOT/$SCENE/octrees/tree_opt.npz" \
        --out_dir "$CKPT_ROOT/$SCENE/octrees/" \
        --overwrite
else
    echo "❌ Step 5 skipped - tree_opt.npz not found"
fi

echo ""

# Step 6: Evaluate compressed octree  
# The compressed octree is saved as tree_opt_compressed.npz in the octrees directory
COMPRESSED_OCTREE="$CKPT_ROOT/$SCENE/octrees/tree_opt_compressed.npz"
if [ -f "$COMPRESSED_OCTREE" ]; then
    echo "Found compressed octree: $(basename "$COMPRESSED_OCTREE")"
    run_step "Evaluating compressed octree quality" "6" \
        python -m octree.compressed_evaluation \
        --input "$COMPRESSED_OCTREE" \
        --config "$CONFIG_FILE" \
        --data_dir "$DATA_ROOT/$SCENE/"
else
    echo "❌ Step 6 skipped - compressed octree not found"
    echo "   Expected: $COMPRESSED_OCTREE"
fi

# Calculate total pipeline time
PIPELINE_END_TIME=$(date +%s)
TOTAL_PIPELINE_TIME=$((PIPELINE_END_TIME - PIPELINE_START_TIME))

echo ""
echo "=== Pipeline Summary ==="
echo "Results directory: $CKPT_ROOT/$SCENE/"
echo "Total pipeline time: ${TOTAL_PIPELINE_TIME}s"

# Check what files were created and get their sizes
RAW_OCTREE_PATH="$CKPT_ROOT/$SCENE/octrees/tree.npz"
OPT_OCTREE_PATH="$CKPT_ROOT/$SCENE/octrees/tree_opt.npz"

RAW_SIZE=0
OPT_SIZE=0
COMPRESSED_SIZE=0
RAW_STATUS="NOT CREATED"
OPT_STATUS="NOT CREATED"
COMPRESSED_STATUS="NOT CREATED"

if [ -f "$RAW_OCTREE_PATH" ]; then
    RAW_SIZE=$(stat -f%z "$RAW_OCTREE_PATH" 2>/dev/null || stat -c%s "$RAW_OCTREE_PATH" 2>/dev/null || echo 0)
    RAW_SIZE_MB=$((RAW_SIZE / 1024 / 1024))
    echo "✅ Raw octree: $RAW_OCTREE_PATH (${RAW_SIZE_MB}MB)"
    RAW_STATUS="CREATED"
else
    echo "❌ Raw octree: NOT CREATED"
fi

if [ -f "$OPT_OCTREE_PATH" ]; then
    OPT_SIZE=$(stat -f%z "$OPT_OCTREE_PATH" 2>/dev/null || stat -c%s "$OPT_OCTREE_PATH" 2>/dev/null || echo 0)
    OPT_SIZE_MB=$((OPT_SIZE / 1024 / 1024))
    echo "✅ Optimized octree: $OPT_OCTREE_PATH (${OPT_SIZE_MB}MB)"
    OPT_STATUS="CREATED"
else
    echo "❌ Optimized octree: NOT CREATED"
fi

# Check for compressed files
COMPRESSED_COUNT=0
if ls "$CKPT_ROOT/$SCENE/"*.npz 1> /dev/null 2>&1; then
    COMPRESSED_COUNT=$(ls "$CKPT_ROOT/$SCENE/"*.npz | wc -l)
    # Calculate total size of compressed files
    for file in "$CKPT_ROOT/$SCENE/"*.npz; do
        if [ "$file" != "$RAW_OCTREE_PATH" ] && [ "$file" != "$OPT_OCTREE_PATH" ]; then
            FILE_SIZE=$(stat -f%z "$file" 2>/dev/null || stat -c%s "$file" 2>/dev/null || echo 0)
            COMPRESSED_SIZE=$((COMPRESSED_SIZE + FILE_SIZE))
        fi
    done
    COMPRESSED_SIZE_MB=$((COMPRESSED_SIZE / 1024 / 1024))
    echo "✅ Compressed files: ${COMPRESSED_COUNT} files in $CKPT_ROOT/$SCENE/ (${COMPRESSED_SIZE_MB}MB)"
    COMPRESSED_STATUS="CREATED"
else
    echo "❌ Compressed files: NOT CREATED"
fi

# Log final pipeline summary
FINAL_TIMESTAMP=$(date -Iseconds)
cat << EOF >> "$PIPELINE_LOG_FILE"
{
  "timestamp": "$FINAL_TIMESTAMP",
  "step": 999,
  "phase": "pipeline_summary",
  "metrics": {
    "total_pipeline_time_seconds": $TOTAL_PIPELINE_TIME,
    "raw_octree_status": "$RAW_STATUS",
    "optimized_octree_status": "$OPT_STATUS",
    "compressed_files_status": "$COMPRESSED_STATUS",
    "raw_octree_size_bytes": $RAW_SIZE,
    "optimized_octree_size_bytes": $OPT_SIZE,
    "compressed_files_size_bytes": $COMPRESSED_SIZE,
    "compressed_files_count": $COMPRESSED_COUNT,
    "scene": "$SCENE",
    "config": "$CONFIG_FILE"
  },
  "additional_info": {
    "raw_octree_path": "$RAW_OCTREE_PATH",
    "optimized_octree_path": "$OPT_OCTREE_PATH",
    "output_directory": "$CKPT_ROOT/$SCENE/",
    "data_root": "$DATA_ROOT",
    "checkpoint_root": "$CKPT_ROOT",
    "phase": "pipeline_complete"
  }
}
EOF

# Clean up the JSON file (remove trailing comma)
if [ -f "$PIPELINE_LOG_FILE" ]; then
    # Remove the last comma and wrap in array brackets
    sed '$ s/,$//' "$PIPELINE_LOG_FILE" > "${PIPELINE_LOG_FILE}.tmp"
    echo '[' > "$PIPELINE_LOG_FILE"
    cat "${PIPELINE_LOG_FILE}.tmp" >> "$PIPELINE_LOG_FILE"
    echo ']' >> "$PIPELINE_LOG_FILE"
    rm -f "${PIPELINE_LOG_FILE}.tmp"
fi

echo ""
echo "📊 Individual stage metrics available in:"
echo "  - Extraction: $CKPT_ROOT/$SCENE/octrees/octree_extraction_metrics.json"
echo "  - Optimization: $CKPT_ROOT/$SCENE/octrees/octree_optimization_metrics.json"
echo "  - Evaluation: $CKPT_ROOT/$SCENE/octrees/octree_evaluation_metrics.json"
echo "  - Compression: $CKPT_ROOT/$SCENE/octrees/octree_compression_metrics.json"
echo "  - Pipeline: $PIPELINE_LOG_FILE"

# Auto-generate analysis plots
echo ""
echo "📊 Auto-generating experiment analysis plots..."
if python analysis/experiment_analyzer.py; then
    echo "✅ Analysis plots generated successfully!"
    echo "📁 Check: $CKPT_ROOT/$SCENE/analysis/"
else
    echo "⚠️  Analysis plot generation failed (pipeline still completed successfully)"
fi

echo ""
echo "🏁 Script completed (with comprehensive logging) in ${TOTAL_PIPELINE_TIME}s"
