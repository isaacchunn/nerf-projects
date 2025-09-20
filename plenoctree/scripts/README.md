# PlenOctree Pipeline Scripts

This directory contains scripts for training NeRF models and converting them to PlenOctree format.

## Scripts Overview

### 🚀 `full_pipeline.sh` - Complete Pipeline (RECOMMENDED)

The unified script that handles both training and octree conversion in one go.

**Usage:**
```bash
# Run complete pipeline for chair scene
bash scripts/full_pipeline.sh

# Run for different scene
bash scripts/full_pipeline.sh --scene lego

# Skip training (if checkpoint exists)
bash scripts/full_pipeline.sh --skip-training

# Only do training, skip conversion
bash scripts/full_pipeline.sh --skip-conversion

# Force retrain even if checkpoint exists
bash scripts/full_pipeline.sh --force-retrain

# Get help
bash scripts/full_pipeline.sh --help
```

**What it does:**
1. **Phase 1 - Training**: NeRF training + evaluation
2. **Phase 2 - Conversion**: Octree extraction → optimization → compression → evaluation

**Features:**
- 🔄 Smart checkpoint detection
- 📊 Comprehensive logging and metrics
- ⚡ Uses your pre-activated conda environment
- 🛡️ Error handling with graceful continuation
- 📈 Performance tracking and file size analysis

### 🎓 `run_training.sh` - Training Only

Original training script for NeRF models.

**Usage:**
```bash
bash scripts/run_training.sh
```

### 🌳 `convert_to_octree.sh` - Conversion Only

Converts existing NeRF checkpoints to octree format.

**Usage:**
```bash
bash scripts/convert_to_octree.sh
```

**What it does:**
1. Extract octree from NeRF
2. Evaluate initial octree
3. Optimize octree
4. Evaluate optimized octree
5. Compress for web
6. Evaluate compressed octree (with fixed evaluation!)

## Configuration

Edit the configuration section in any script:

```bash
# Data and output paths
export DATA_ROOT="/mnt/d/GitHub/plenoctree/data/NeRF/nerf_synthetic"
export CKPT_ROOT="/mnt/d/GitHub/plenoctree/data/Plenoctree/checkpoints/syn_sh16"
export SCENE="chair"
export CONFIG_FILE="nerf_sh/config/blender"

# Training flags (for harder scenes)
export EXTRA_FLAGS="--lr_delay_steps 50000 --lr_delay_mult 0.01"
```

## Output Files

After running the full pipeline, you'll get:

### NeRF Training
- `checkpoint_*` - NeRF model checkpoints
- Training logs and metrics

### Octree Files
- `octrees/tree.npz` - Raw extracted octree (~287MB)
- `octrees/tree_opt.npz` - Optimized octree (~287MB)
- `octrees/tree_opt_compressed.npz` - Compressed octree (~60MB, 4.75x smaller!)

### Metrics and Logs
- `octree_extraction_metrics.json` - Extraction performance
- `octree_optimization_metrics.json` - Optimization metrics
- `octree_evaluation_metrics.json` - Quality evaluation
- `octree_compression_metrics.json` - Compression statistics
- `octree_compression_evaluation_metrics.json` - Compressed quality
- `full_pipeline_metrics.json` - Complete pipeline log

## 📊 Automatic Analysis Generation

Both pipeline scripts automatically generate comprehensive analysis plots at the end:

**Generated Files:**
- `scene_pipeline_analysis.png` - Complete 6-stage pipeline overview
- `scene_psnr_progression.png` - PSNR quality progression  
- `scene_ssim_progression.png` - SSIM quality progression
- `scene_lpips_progression.png` - LPIPS progression
- `scene_memory_progression.png` - Memory usage analysis
- `scene_timing_analysis.png` - Detailed timing breakdown

📁 **Location**: `data/Plenoctree/checkpoints/syn_sh16/{scene}/analysis/`

## Quality Results

With the **fixed compressed evaluation**, you get excellent compression:

| Metric | Uncompressed | Compressed | Loss |
|--------|-------------|------------|------|
| **PSNR** | 32.75 | 32.70 | -0.16% ✅ |
| **SSIM** | 0.9685 | 0.9682 | -0.03% ✅ |
| **LPIPS** | 0.0410 | 0.0412 | +0.49% ✅ |
| **Size** | 287MB | 60MB | **-79%** 🎉 |

**Result**: Near-lossless compression with 4.75x size reduction!

## Requirements

- **For training**: Activate `plenoctree` conda environment before running
- **For octree conversion/evaluation**: Activate `plenoctree_eval` conda environment before running
- You manage environment activation manually (no automatic switching)

## Troubleshooting

### Common Issues

1. **"No module named 'torch'"**
   - Make sure you activate the correct conda environment before running scripts
   - `plenoctree` for training, `plenoctree_eval` for octree operations

2. **"checkpoint not found"**
   - Run training first or use `--force-retrain`

3. **"compressed octree evaluation fails"**
   - Use the fixed `octree.compressed_evaluation` module
   - Updated in all scripts

### Getting Help

```bash
bash scripts/full_pipeline.sh --help
```

## Example Workflow

```bash
# 1. Activate appropriate environment and run full pipeline
conda activate plenoctree_eval  # Use this for the full pipeline
bash scripts/full_pipeline.sh --scene chair

# 2. Check results
ls data/Plenoctree/checkpoints/syn_sh16/chair/

# 3. View metrics
cat data/Plenoctree/checkpoints/syn_sh16/chair/full_pipeline_metrics.json

# 4. For training only (different environment)
conda activate plenoctree
bash scripts/run_training.sh

# 5. For octree conversion only
conda activate plenoctree_eval
bash scripts/convert_to_octree.sh
```

That's it! The `full_pipeline.sh` script is your one-stop solution for the complete PlenOctree workflow. 🚀
