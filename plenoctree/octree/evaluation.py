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
"""Evluate a plenoctree on test set.

Usage:

export DATA_ROOT=./data/NeRF/nerf_synthetic/
export CKPT_ROOT=./data/PlenOctree/checkpoints/syn_sh16
export SCENE=chair
export CONFIG_FILE=nerf_sh/config/blender

python -m octree.evaluation \
    --input $CKPT_ROOT/$SCENE/octrees/tree_opt.npz \
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
    "Input octree npz from optimization.py",
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

@torch.no_grad()
def main(unused_argv):
    eval_start_time = time.time()
    
    utils.set_random_seed(20200823)
    utils.update_flags(FLAGS)

    dataset = datasets.get_dataset("test", FLAGS)

    print('N3Tree load', FLAGS.input)
    t = svox.N3Tree.load(FLAGS.input, map_location=device)

    # Initialize JSON logger and memory tracker
    log_dir = os.path.dirname(FLAGS.input) if os.path.dirname(FLAGS.input) else "."
    
    # Determine octree type and appropriate log file
    input_dir = os.path.dirname(FLAGS.input)
    input_basename = os.path.basename(FLAGS.input)
    
    # Detect octree type based on file location and name
    if input_basename == "tree.npz" and input_dir.endswith("/octrees"):
        octree_type = "initial"
        log_filename = "octree_initial_evaluation_metrics.json"
        eval_type_display = "Initial octree (baseline)"
    elif input_basename == "tree_opt.npz" and input_dir.endswith("/octrees"):
        octree_type = "optimized"
        log_filename = "octree_evaluation_metrics.json"
        eval_type_display = "Optimized octree"
    elif input_basename == "tree_opt.npz" and not input_dir.endswith("/octrees"):
        octree_type = "compressed"
        log_filename = "octree_compression_evaluation_metrics.json"
        eval_type_display = "Compressed octree"
    else:
        octree_type = "unknown"
        log_filename = "octree_evaluation_metrics.json"
        eval_type_display = "Unknown octree type"
    
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
    octree_capacity = int(t.n_internal + t.n_leaves) if hasattr(t, 'n_internal') and hasattr(t, 'n_leaves') else 0
    
    print("🔄 Starting octree evaluation...")
    print(f"   Dataset size: {dataset.size} images")
    print(f"   LPIPS enabled: True")
    print(f"   Frames output: {FLAGS.write_vid is not None or FLAGS.write_images is not None}")
    
    avg_psnr, avg_ssim, avg_lpips, out_frames = utils.eval_octree(t, dataset, FLAGS,
            want_lpips=True,
            want_frames=FLAGS.write_vid is not None or FLAGS.write_images is not None)
    
    print("✅ Octree evaluation completed!")
    print('Average PSNR', avg_psnr, 'SSIM', avg_ssim, 'LPIPS', avg_lpips)
    print(f'Octree capacity: {octree_capacity:,} nodes, File size: {octree_file_size_mb:.1f} MB')
    
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
        "data_format": str(t.data_format) if hasattr(t, 'data_format') else None,
        "max_depth": int(t.max_depth) if hasattr(t, 'max_depth') else None
    }
    
    # Capture final memory snapshot and calculate efficiency indices
    final_snapshot = memory_tracker.capture_snapshot(1)
    memory_metrics = memory_tracker.get_memory_metrics(final_snapshot)
    # Get init_grid_depth from octree max_depth (since it's not a flag in evaluation)
    init_grid_depth = int(t.max_depth) if hasattr(t, 'max_depth') else None
    
    efficiency_indices = memory_tracker.calculate_efficiency_indices(
        psnr=avg_psnr,
        ssim=avg_ssim,
        lpips=avg_lpips,
        snapshot=final_snapshot,
        octree_capacity=octree_capacity,
        octree_file_size_mb=octree_file_size_mb,
        init_grid_depth=init_grid_depth
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
    
    metrics_logger.log_metrics(0, "octree_evaluation", evaluation_metrics, additional_info)
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
    print(f'\n=== Evaluation Complete ===')
    print(f'Total evaluation time: {eval_time:.2f}s')
    print(f'Average time per image: {eval_time/dataset.size:.3f}s')
    print(f'Processing rate: {dataset.size/eval_time:.1f} images/second')
    print(f'Quality metrics - PSNR: {avg_psnr:.3f}, SSIM: {avg_ssim:.4f}, LPIPS: {avg_lpips:.4f}')
    print(f'Octree specs - Capacity: {octree_capacity:,} nodes, Size: {octree_file_size_mb:.1f} MB')
    print(f'Metrics logged to: {os.path.join(log_dir, "octree_evaluation_metrics.json")}')
    
    # Print memory summary
    memory_tracker.print_memory_summary(final_snapshot)

if __name__ == "__main__":
    app.run(main)
