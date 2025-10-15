#!/bin/bash
# Complete metrics extraction pipeline
# Usage: bash get_all_metrics.sh

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate plenoxel

echo "================================================================================"
echo "COMPLETE METRICS EXTRACTION PIPELINE"
echo "================================================================================"

SUCCESSFUL=0
FAILED=0

# Loop through all checkpoints
for CKPT_DIR in ckpt/*/; do
    SCENE_NAME=$(basename "$CKPT_DIR")
    CKPT_FILE="$CKPT_DIR/ckpt.npz"
    ARGS_FILE="$CKPT_DIR/args.json"
    
    # Skip if no checkpoint file
    if [ ! -f "$CKPT_FILE" ]; then
        continue
    fi
    
    echo ""
    echo "================================================================================"
    echo "$SCENE_NAME"
    echo "================================================================================"
    
    # Read data_dir from args.json
    if [ ! -f "$ARGS_FILE" ]; then
        echo "✗ No args.json found, skipping"
        FAILED=$((FAILED + 1))
        continue
    fi
    
    DATA_DIR=$(python -c "import json; print(json.load(open('$ARGS_FILE'))['data_dir'])" 2>/dev/null)
    if [ -z "$DATA_DIR" ]; then
        echo "✗ Could not read data_dir from args.json"
        FAILED=$((FAILED + 1))
        continue
    fi
    
    # Step 1: Generate test renders if needed
    TEST_RENDERS_DIR="$CKPT_DIR/test_renders"
    if [ ! -d "$TEST_RENDERS_DIR" ] || [ -z "$(ls -A $TEST_RENDERS_DIR/*.png 2>/dev/null)" ]; then
        echo "[1/2] Generating test renders..."
        python render_imgs.py "$CKPT_FILE" "$DATA_DIR"
        if [ $? -ne 0 ]; then
            echo "✗ Failed to generate renders"
            FAILED=$((FAILED + 1))
            continue
        fi
        echo "✓ Test renders generated"
    else
        echo "[1/2] Test renders already exist"
    fi
    
    # Step 2: Compute metrics
    echo "[2/2] Computing PSNR, SSIM, LPIPS..."
    python calc_metrics.py "$TEST_RENDERS_DIR" "$DATA_DIR" --dataset_type auto
    if [ $? -ne 0 ]; then
        echo "✗ Failed to compute metrics"
        FAILED=$((FAILED + 1))
    else
        echo "✓ Metrics computed"
        SUCCESSFUL=$((SUCCESSFUL + 1))
    fi
done

# Step 3: Extract all metrics
echo ""
echo "================================================================================"
echo "EXTRACTING ALL METRICS"
echo "================================================================================"

python extract_metrics.py

if [ $? -eq 0 ]; then
    echo ""
    echo "================================================================================"
    echo "PIPELINE COMPLETE!"
    echo "================================================================================"
    echo "✓ Successfully processed: $SUCCESSFUL scenes"
    [ $FAILED -gt 0 ] && echo "✗ Failed: $FAILED scenes"
    echo ""
    echo "Results saved to: metrics_summary/all_metrics.csv"
    echo "================================================================================"
else
    echo "✗ Failed to extract final metrics"
    exit 1
fi

