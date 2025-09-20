#  Copyright 2021 The PlenOctree Authors.
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#
#  1. Redistributions of source code must retain the above copyright notice,
#  this list of conditions and the following disclaimer.
#
#  2. Redistributions in binary form must reproduce the above copyright notice,
#  this list of conditions and the following disclaimer in the documentation
#  and/or other materials provided with the distribution.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
#  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
#  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
#  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
#  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
#  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
#  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
#  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
#  POSSIBILITY OF SUCH DAMAGE.
"""Evaluate a compressed plenoctree on test set.

This script handles compressed octrees by reconstructing the full data format
and creating a temporary uncompressed version for evaluation.

Usage:

export DATA_ROOT=./data/NeRF/nerf_synthetic/
export CKPT_ROOT=./data/PlenOctree/checkpoints/syn_sh16
export SCENE=chair
export CONFIG_FILE=nerf_sh/config/blender

python -m octree.compressed_evaluation \
    --input $CKPT_ROOT/$SCENE/tree_opt.npz \
    --config $CONFIG_FILE \
    --data_dir $DATA_ROOT/$SCENE/
"""
import torch
import numpy as np
import os
import time
from absl import app
from absl import flags
from tqdm import tqdm
import imageio

from octree.nerf import models
from octree.nerf import utils
from octree.nerf import datasets

import svox
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from json_logger import create_logger
from memory_tracker import create_memory_tracker

FLAGS = flags.FLAGS

utils.define_flags()

flags.DEFINE_string(
    "input",
    "./tree_opt.npz",
    "Input compressed octree npz file",
)
flags.DEFINE_string(
    "write_vid",
    None,
    "If specified, writes rendered video to given path (*.mp4)",
)
flags.DEFINE_string(
    "write_images",
    None,
    "If specified, writes images to given path (*.png)",
)

device = "cuda" if torch.cuda.is_available() else "cpu"

def reconstruct_compressed_octree(compressed_path, device="cpu"):
    """
    Reconstruct a compressed octree into a format that svox can load.
    
    Args:
        compressed_path: Path to the compressed .npz file
        device: Device to load the octree on
        
    Returns:
        Tuple of (octree, temp_path) or (None, None) if failed
    """
    print(f"🔄 Loading compressed octree from {compressed_path}")
    z = np.load(compressed_path)
    
    print(f"📋 Available keys: {list(z.keys())}")
    
    # Check if this is actually a compressed format
    if 'quant_colors' not in z or 'quant_map' not in z or 'sigma' not in z:
        print("❌ This doesn't appear to be a compressed octree format")
        return None, None
    
    print("✅ Confirmed compressed octree format")
    
    # Extract compressed data
    quant_colors = z['quant_colors']  # Shape: (channels, num_colors, 3)
    quant_map = z['quant_map']        # Shape: (channels, nodes, 2, 2, 2)  
    sigma = z['sigma']                # Shape: (nodes, 2, 2, 2)
    
    print(f"📊 Compressed data shapes:")
    print(f"   quant_colors: {quant_colors.shape}")
    print(f"   quant_map: {quant_map.shape}")
    print(f"   sigma: {sigma.shape}")
    
    # Get dimensions
    num_channels, num_colors, color_dim = quant_colors.shape
    num_nodes = quant_map.shape[1]
    
    print(f"🔢 Reconstructing data for {num_nodes:,} nodes, {num_channels} channels")
    
    # Reconstruct RGB data from quantized format
    print("   🎨 Reconstructing RGB data...")
    
    # CORRECTED: Each quantized channel represents one basis function across R,G,B
    # Original data structure: (nodes, 2, 2, 2, 48) where 48 = 16 basis * 3 RGB
    # Compressed: 16 channels, each with RGB triplets for one basis function
    basis_dim = num_channels  # 16 basis functions
    reconstructed_rgb = np.zeros((num_nodes, 2, 2, 2, basis_dim * 3), dtype=np.float32)
    
    # Vectorized reconstruction with correct channel mapping
    for basis_idx in tqdm(range(basis_dim), desc="Basis functions"):
        # Get color indices for this basis function
        color_indices = quant_map[basis_idx]  # Shape: (nodes, 2, 2, 2)
        
        # Get RGB colors for this basis function
        basis_colors = quant_colors[basis_idx]  # Shape: (num_colors, 3)
        
        # Vectorized lookup
        valid_mask = color_indices < num_colors
        valid_indices = color_indices[valid_mask]
        
        # Correct channel mapping: basis_idx maps to [R, G, B] = [basis_idx, basis_idx+16, basis_idx+32]
        r_channel = basis_idx
        g_channel = basis_idx + basis_dim
        b_channel = basis_idx + 2 * basis_dim
        
        # Reconstruct RGB for this basis function
        if valid_indices.size > 0:
            reconstructed_colors = basis_colors[valid_indices]  # Shape: (valid_count, 3)
            reconstructed_rgb[valid_mask, r_channel] = reconstructed_colors[:, 0]  # R
            reconstructed_rgb[valid_mask, g_channel] = reconstructed_colors[:, 1]  # G  
            reconstructed_rgb[valid_mask, b_channel] = reconstructed_colors[:, 2]  # B
    
    # Combine RGB and sigma (alpha) data
    print("   🔗 Combining RGB and alpha data...")
    data = np.zeros((num_nodes, 2, 2, 2, basis_dim * 3 + 1), dtype=np.float16)
    data[..., :-1] = reconstructed_rgb.astype(np.float16)  # RGB channels (48 channels)
    data[..., -1] = sigma  # Alpha channel (1 channel)
    
    print(f"   ✅ Reconstructed data shape: {data.shape}")
    
    # Create the full octree dictionary with all required fields
    z_dict = dict(z)
    
    # Add the reconstructed data
    z_dict['data'] = data
    
    # Add missing metadata fields with reasonable defaults
    if 'parent_depth' not in z_dict:
        # parent_depth should have shape (num_nodes, 2) based on original format
        parent_depth = np.zeros((num_nodes, 2), dtype=np.int32)
        z_dict['parent_depth'] = parent_depth
        print("   ✓ Added parent_depth")
    
    if 'n_internal' not in z_dict:
        z_dict['n_internal'] = num_nodes
        print("   ✓ Added n_internal")
        
    if 'n_free' not in z_dict:
        z_dict['n_free'] = 0
        print("   ✓ Added n_free")
        
    if 'depth_limit' not in z_dict:
        z_dict['depth_limit'] = 10
        print("   ✓ Added depth_limit")
        
    if 'geom_resize_fact' not in z_dict:
        z_dict['geom_resize_fact'] = 1.0
        print("   ✓ Added geom_resize_fact")
    
    # Remove compressed format fields to avoid conflicts
    del z_dict['quant_colors']
    del z_dict['quant_map'] 
    del z_dict['sigma']
    print("   ✓ Removed compressed format fields")
    
    # Save temporarily for svox loading
    temp_path = compressed_path.replace('.npz', '_temp_reconstructed.npz')
    print(f"💾 Saving reconstructed octree to {temp_path}")
    np.savez_compressed(temp_path, **z_dict)
    
    try:
        # Load with svox
        print("🔄 Loading reconstructed octree with svox...")
        tree = svox.N3Tree.load(temp_path, map_location=device)
        print("✅ Successfully loaded reconstructed octree")
        return tree, temp_path
        
    except Exception as e:
        print(f"❌ Failed to load reconstructed octree: {e}")
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return None, None

@torch.no_grad()
def main(unused_argv):
    eval_start_time = time.time()
    
    utils.set_random_seed(20200823)
    utils.update_flags(FLAGS)

    dataset = datasets.get_dataset("test", FLAGS)

    print('🎯 Evaluating compressed octree:', FLAGS.input)
    
    # Reconstruct and load the compressed octree
    tree, temp_path = reconstruct_compressed_octree(FLAGS.input, device)
    
    if tree is None:
        print("❌ Failed to reconstruct compressed octree")
        return
    
    # Initialize JSON logger and memory tracker
    log_dir = os.path.dirname(FLAGS.input) if os.path.dirname(FLAGS.input) else "."
    
    # Always treat as compressed for this script
    octree_type = "compressed"
    log_filename = "octree_compression_evaluation_metrics.json"
    eval_type_display = "Compressed octree"
    
    metrics_logger = create_logger(log_dir, log_filename)
    memory_tracker = create_memory_tracker()
    
    print(f"📊 Evaluation type: {eval_type_display}")
    print(f"📝 Logging to: {log_filename}")
    print(f"🔍 Octree type detected: {octree_type}")
    
    # Capture baseline memory after loading octree
    baseline_snapshot = memory_tracker.capture_snapshot(0)
    memory_tracker.print_memory_summary(baseline_snapshot)

    # Get octree information for logging
    octree_file_size_mb = os.path.getsize(FLAGS.input) / (1024 * 1024) if os.path.exists(FLAGS.input) else 0.0
    octree_capacity = int(tree.n_internal + tree.n_leaves) if hasattr(tree, 'n_internal') and hasattr(tree, 'n_leaves') else 0
    
    print("🔄 Starting compressed octree evaluation...")
    print(f"   Dataset size: {dataset.size} images")
    print(f"   LPIPS enabled: True")
    print(f"   Frames output: {FLAGS.write_vid is not None or FLAGS.write_images is not None}")
    
    # Perform evaluation using the standard pipeline
    avg_psnr, avg_ssim, avg_lpips, out_frames = utils.eval_octree(tree, dataset, FLAGS,
            want_lpips=True,
            want_frames=FLAGS.write_vid is not None or FLAGS.write_images is not None)
    
    print("✅ Compressed octree evaluation completed!")
    print('Average PSNR', avg_psnr, 'SSIM', avg_ssim, 'LPIPS', avg_lpips)
    print(f'Octree capacity: {octree_capacity:,} nodes, File size: {octree_file_size_mb:.1f} MB')
    
    # Clean up temporary file
    if temp_path and os.path.exists(temp_path):
        os.remove(temp_path)
        print(f"🧹 Cleaned up temporary file: {temp_path}")
    
    # Log octree evaluation metrics to JSON with memory tracking
    eval_time = time.time() - eval_start_time
    timing_info = {
        "total_eval_time": eval_time,
        "dataset_size": dataset.size,
        "avg_time_per_image": eval_time / dataset.size,
        "images_per_second": dataset.size / eval_time
    }
    octree_info = {
        "octree_path": FLAGS.input,
        "device": device,
        "octree_capacity": octree_capacity,
        "octree_file_size_mb": octree_file_size_mb,
        "data_format": "compressed",
        "evaluation_method": "reconstructed"
    }
    
    # Capture final memory snapshot and calculate efficiency indices
    final_snapshot = memory_tracker.capture_snapshot(1)
    memory_metrics = memory_tracker.get_memory_metrics(final_snapshot)
    efficiency_indices = memory_tracker.calculate_efficiency_indices(
        psnr=avg_psnr,
        ssim=avg_ssim,
        lpips=avg_lpips,
        snapshot=final_snapshot
    )
    
    # Enhanced evaluation metrics
    evaluation_metrics = {
        "psnr": avg_psnr,
        "ssim": avg_ssim,
        "lpips": avg_lpips,
        "quality_index": avg_psnr * avg_ssim * (1.0 - avg_lpips),  # Combined quality metric
        "compression_ratio": octree_capacity / (dataset.size * dataset.h * dataset.w) if dataset.size > 0 else 0.0
    }
    evaluation_metrics.update(memory_metrics)
    
    # Combine all additional info
    additional_info = timing_info.copy()
    additional_info.update(octree_info)
    additional_info.update(efficiency_indices)
    
    print("📝 Logging evaluation metrics to JSON...")
    print(f"   Log file: {log_filename}")
    print(f"   Metrics keys: {list(evaluation_metrics.keys())}")
    
    metrics_logger.log_metrics(0, "compressed_octree_evaluation", evaluation_metrics, additional_info)
    print("✅ Metrics logged successfully!")

    # Handle video/image output with timing
    if FLAGS.write_vid is not None and len(out_frames):
        video_start = time.time()
        print('Writing to', FLAGS.write_vid)
        imageio.mimwrite(FLAGS.write_vid, out_frames)
        video_time = time.time() - video_start
        
        video_metrics = {
            "video_export_time": video_time,
            "video_frames": len(out_frames),
            "video_path": FLAGS.write_vid
        }
        metrics_logger.log_metrics(1, "video_export", video_metrics, {"phase": "video_output"})
        print(f'Video exported in {video_time:.2f}s')

    if FLAGS.write_images is not None and len(out_frames):
        images_start = time.time()
        print('Writing to', FLAGS.write_images)
        os.makedirs(FLAGS.write_images, exist_ok=True)
        for idx, frame in tqdm(enumerate(out_frames)):
            imageio.imwrite(os.path.join(FLAGS.write_images, f"{idx:03d}.png"), frame)
        images_time = time.time() - images_start
        
        images_metrics = {
            "images_export_time": images_time,
            "images_count": len(out_frames),
            "images_directory": FLAGS.write_images
        }
        metrics_logger.log_metrics(2, "images_export", images_metrics, {"phase": "images_output"})
        print(f'Images exported in {images_time:.2f}s')
    
    # Final summary
    print(f'\n=== Compressed Octree Evaluation Complete ===')
    print(f'Total evaluation time: {eval_time:.2f}s')
    print(f'Average time per image: {eval_time/dataset.size:.3f}s')
    print(f'Processing rate: {dataset.size/eval_time:.1f} images/second')
    print(f'Quality metrics - PSNR: {avg_psnr:.3f}, SSIM: {avg_ssim:.4f}, LPIPS: {avg_lpips:.4f}')
    print(f'Octree specs - Capacity: {octree_capacity:,} nodes, Size: {octree_file_size_mb:.1f} MB')
    print(f'Evaluation method: Reconstructed from compressed format')
    print(f'Metrics logged to: {os.path.join(log_dir, log_filename)}')
    
    # Print memory summary
    memory_tracker.print_memory_summary(final_snapshot)

if __name__ == "__main__":
    app.run(main)
