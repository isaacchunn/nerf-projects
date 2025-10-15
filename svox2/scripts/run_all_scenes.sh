#!/bin/bash

# Simple wrapper to train all NeRF Synthetic scenes with advanced metrics
# Usage: ./run_all_scenes.sh [gpu_id]

GPU=${1:-0}  # Default to GPU 0 if not specified

SCENES="lego mic ship chair ficus materials drums hotdog"

echo "========================================="
echo "Training All NeRF Synthetic Scenes"
echo "GPU: $GPU"
echo "Scenes: $SCENES"
echo "========================================="
echo ""

for scene in $SCENES; do
    echo ""
    echo "========================================="
    echo "Starting: $scene"
    echo "========================================="
    
    ./scripts/train.sh single $scene $GPU syn "" \
        --log_advanced_metrics \
        --log_fdr \
        --advanced_metrics_every 1 \
        --fdr_every 1 \
        --log_floater_viz \
        --log_density_render \
        --fdr_density_threshold 0.01 \
        --fdr_main_object_threshold 0.1
    
    exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        echo "✅ $scene completed successfully"
    else
        echo "❌ $scene failed with exit code: $exit_code"
        read -p "Continue to next scene? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Stopping batch training."
            exit 1
        fi
    fi
    
    echo ""
    echo "⏳ Waiting 5 seconds before next scene..."
    sleep 5
done

echo ""
echo "========================================="
echo "✅ All scenes completed!"
echo "========================================="

