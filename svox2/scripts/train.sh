#!/bin/bash

# Simplified svox2 training script
# Supports: single scene, batch training, and config creation

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
OPT_DIR="$ROOT_DIR/opt"

# Predefined scene sets
NERF_SYNTHETIC="lego mic ship chair ficus materials drums hotdog"
NERF_LLFF="flower fortress horns leaves orchids room trex fern"
TANKS_TEMPLES="Barn Caterpillar Family Ignatius Truck"

show_help() {
    echo "Simplified svox2 Training Script"
    echo ""
    echo "USAGE:"
    echo "  $0 <mode> [options...]"
    echo ""
    echo "MODES:"
    echo "  single <scene> <gpu> [config] [data_root] [--param=value ...]"
    echo "    Train a single scene"
    echo ""
    echo "  batch <config> <gpu> <data_root> <scenes...>"
    echo "    Train multiple scenes sequentially on same GPU"
    echo "    (use 'nerf_synthetic', 'nerf_llff', or 'tanks_temples' for all)"
    echo ""
    echo "  config <base_config> <output_name> [param1=value1 ...]"
    echo "    Create custom config file"
    echo ""
    echo "EXAMPLES:"
    echo "  $0 single lego 0"
    echo "  $0 single lego 0 syn /path/to/data --n_iters=150000"
    echo "  $0 batch syn 0 /path/to/data lego chair ficus"
    echo "  $0 batch syn 0 /path/to/data nerf_synthetic"
    echo "  $0 config syn fast_config n_iters=100000"
    echo ""
    echo "CONFIGS: syn, llff, tnt, custom, custom_alt, co3d"
    echo ""
    echo "LOGGING:"
    echo "  - All logs saved to: logs/<experiment_name_timestamp>.log"
    echo "  - Console output shown live + saved to log file"
}

resolve_config() {
    local config=$1
    case $config in
        syn|synthetic) echo "configs/syn.json" ;;
        llff|forward_facing) echo "configs/llff.json" ;;
        tnt|tanks_temples) echo "configs/tnt.json" ;;
        custom) echo "configs/custom.json" ;;
        custom_alt) echo "configs/custom_alt.json" ;;
        co3d) echo "configs/co3d.json" ;;
        *.json) echo "$config" ;;
        *) echo "configs/$config.json" ;;
    esac
}

find_data_dir() {
    local scene=$1
    local data_root=$2
    
    if [ -n "$data_root" ]; then
        if [ -d "$data_root/$scene" ]; then
            echo "$data_root/$scene"
            return
        fi
        # Try in subdirectories
        for dataset_dir in "$data_root"/*; do
            if [ -d "$dataset_dir/$scene" ]; then
                echo "$dataset_dir/$scene"
                return
            fi
        done
    fi
    
    # Try common locations
    for root in "$ROOT_DIR/data" "$ROOT_DIR/../data" "/data" "$HOME/data"; do
        if [ -d "$root" ]; then
            for dataset in "nerf_synthetic" "nerf_llff_data" "tanks_temples" "co3d"; do
                if [ -d "$root/$dataset/$scene" ]; then
                    echo "$root/$dataset/$scene"
                    return
                fi
            done
        fi
    done
    
    # Return scene as-is (might be full path)
    echo "$scene"
}

launch_single() {
    local scene=$1
    local gpu=$2
    local config=${3:-syn}
    local data_root=$4
    
    # Extract extra args (everything after the 4th argument)
    local extra_args=""
    if [ $# -gt 4 ]; then
        shift 4
        extra_args="$@"
    fi
    
    local config_file=$(resolve_config "$config")
    local data_dir=$(find_data_dir "$scene" "$data_root")
    
    # Create experiment name with timestamp
    local timestamp=$(date +%m%d_%H%M%S)
    local exp_name="${scene}_${config}_${timestamp}"
    
    # Create logs directory
    local logs_dir="$ROOT_DIR/logs"
    mkdir -p "$logs_dir"
    local log_file="$logs_dir/${exp_name}.log"
    
    echo "=== Training Configuration ==="
    echo "Scene: $scene"
    echo "GPU: $gpu"
    echo "Config: $config_file"
    echo "Data Dir: $data_dir"
    echo "Experiment: $exp_name"
    echo "Log File: $log_file"
    echo "Extra Args: $extra_args"
    echo "=========================="
    
    if [ ! -d "$data_dir" ]; then
        echo "Warning: Data directory not found: $data_dir"
    fi
    
    # Create checkpoint directory
    local ckpt_dir="$OPT_DIR/ckpt/$exp_name"
    mkdir -p "$ckpt_dir"
    
    # Build command
    local cmd="python -u opt.py -t $ckpt_dir $data_dir -c $config_file $extra_args"
    
    echo "Starting training..."
    echo "Command: $cmd"
    echo "Press Ctrl+C to stop"
    echo ""
    
    cd "$OPT_DIR"
    
    # Run with tee to show output and save to log simultaneously
    CUDA_VISIBLE_DEVICES=$gpu $cmd 2>&1 | tee "$log_file"
    
    local exit_code=${PIPESTATUS[0]}
    
    if [ $exit_code -eq 0 ]; then
        echo ""
        echo "✅ Training completed successfully!"
        echo "📁 Checkpoint: $ckpt_dir"
        echo "📄 Log: $log_file"
    else
        echo ""
        echo "❌ Training failed with exit code: $exit_code"
        echo "📄 Check log: $log_file"
    fi
    
    return $exit_code
}

launch_batch() {
    local config=$1
    local gpu=$2
    local data_root=$3
    shift 3
    local scenes="$@"
    
    # Handle predefined scene sets
    case "$1" in
        nerf_synthetic|all_synthetic) scenes="$NERF_SYNTHETIC" ;;
        nerf_llff|all_llff) scenes="$NERF_LLFF" ;;
        tanks_temples|all_tnt) scenes="$TANKS_TEMPLES" ;;
    esac
    
    echo "=== Batch Training ==="
    echo "Config: $config"
    echo "GPU: $gpu"
    echo "Data Root: $data_root"
    echo "Scenes: $scenes"
    echo "====================="
    
    local scene_count=0
    local total_scenes=$(echo $scenes | wc -w)
    
    for scene in $scenes; do
        scene_count=$((scene_count + 1))
        echo ""
        echo "[$scene_count/$total_scenes] Training $scene on GPU $gpu..."
        launch_single "$scene" "$gpu" "$config" "$data_root"
        
        if [ $scene_count -lt $total_scenes ]; then
            echo ""
            echo "⏳ Waiting 5 seconds before next scene..."
            sleep 5
        fi
    done
    
    echo ""
    echo "✅ All batch experiments completed!"
}

make_config() {
    local base_config=$1
    local output_name=$2
    shift 2
    local params="$@"
    
    local base_file=$(resolve_config "$base_config")
    local output_file="$OPT_DIR/configs/${output_name}.json"
    
    if [ ! -f "$OPT_DIR/$base_file" ]; then
        echo "Base config not found: $OPT_DIR/$base_file"
        exit 1
    fi
    
    echo "Creating config: $output_file"
    echo "Base: $base_file"
    echo "Parameters: $params"
    
    cp "$OPT_DIR/$base_file" "$output_file"
    
    if command -v jq &> /dev/null; then
        for param in $params; do
            local key=$(echo $param | cut -d'=' -f1)
            local value=$(echo $param | cut -d'=' -f2-)
            
            if [[ $value =~ ^[0-9e.-]+$ ]] || [[ $value =~ ^\[.*\]$ ]]; then
                jq ".$key = $value" "$output_file" > "${output_file}.tmp" && mv "${output_file}.tmp" "$output_file"
            else
                jq ".$key = \"$value\"" "$output_file" > "${output_file}.tmp" && mv "${output_file}.tmp" "$output_file"
            fi
            echo "  $key = $value"
        done
    else
        echo "Warning: jq not found, using sed"
        for param in $params; do
            local key=$(echo $param | cut -d'=' -f1)
            local value=$(echo $param | cut -d'=' -f2-)
            sed -i "s/\"$key\":[^,}]*/\"$key\": $value/" "$output_file"
            echo "  $key = $value"
        done
    fi
    
    echo "Config created: $output_file"
    echo ""
    cat "$output_file"
}

# Main script logic
if [ $# -eq 0 ]; then
    show_help
    exit 1
fi

MODE=$1
shift

case $MODE in
    single|s)
        if [ $# -lt 2 ]; then
            echo "Usage: $0 single <scene> <gpu> [config] [data_root] [--param=value ...]"
            exit 1
        fi
        # Parse arguments properly for single mode
        scene=$1
        gpu=$2
        config=${3:-syn}
        data_root=$4
        shift 4 2>/dev/null || shift $#  # Shift safely
        launch_single "$scene" "$gpu" "$config" "$data_root" "$@"
        ;;
    batch|b)
        if [ $# -lt 4 ]; then
            echo "Usage: $0 batch <config> <gpu> <data_root> <scenes...>"
            exit 1
        fi
        launch_batch "$@"
        ;;
    config|cfg)
        if [ $# -lt 2 ]; then
            echo "Usage: $0 config <base_config> <output_name> [param1=value1 ...]"
            exit 1
        fi
        make_config "$@"
        ;;
    help|h|-h|--help)
        show_help
        ;;
    *)
        echo "Unknown mode: $MODE"
        echo "Use '$0 help' for usage information"
        exit 1
        ;;
esac