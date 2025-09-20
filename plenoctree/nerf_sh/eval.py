# coding=utf-8
# Modifications Copyright 2021 The PlenOctree Authors.
# Original Copyright 2021 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Evaluation script for Nerf."""

import os
# Get rid of ugly TF logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import functools
from os import path

from absl import app
from absl import flags

# Apply JAX compatibility fix before importing flax
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    import jax_compatibility_fix
except ImportError:
    pass  # Fix not needed or not available

import flax
from flax.metrics import tensorboard
from flax.training import checkpoints
import jax
from jax import random
import numpy as np
import time
import torch

from nerf_sh.nerf import datasets
from nerf_sh.nerf import models
from nerf_sh.nerf import utils
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from json_logger import create_logger
from memory_tracker import create_memory_tracker

FLAGS = flags.FLAGS

utils.define_flags()


def main(unused_argv):
    rng = random.PRNGKey(20200823)
    rng, key = random.split(rng)

    utils.update_flags(FLAGS)
    utils.check_flags(FLAGS)

    dataset = datasets.get_dataset("test", FLAGS)
    model, state = models.get_model_state(key, FLAGS, restore=False)

    # Rendering is forced to be deterministic even if training was randomized, as
    # this eliminates "speckle" artifacts.
    render_pfn = utils.get_render_pfn(model, randomized=False)

    # Compiling to the CPU because it's faster and more accurate.
    ssim_fn = jax.jit(functools.partial(utils.compute_ssim, max_val=1.0), backend="cpu")
    
    # Initialize memory tracker and JSON logger for evaluation metrics
    memory_tracker = create_memory_tracker()
    metrics_logger = create_logger(FLAGS.train_dir, "nerf_evaluation_final.json")
    print('* Memory tracker and JSON logger initialized for evaluation')

    last_step = 0
    out_dir = path.join(
        FLAGS.train_dir, "path_renders" if FLAGS.render_path else "test_preds"
    )
    if not FLAGS.eval_once:
        summary_writer = tensorboard.SummaryWriter(path.join(FLAGS.train_dir, "eval"))
    while True:
        print('Loading model')
        state = checkpoints.restore_checkpoint(FLAGS.train_dir, state)
        step = int(state.optimizer.state.step)
        if step <= last_step:
            continue
        if FLAGS.save_output and (not utils.isdir(out_dir)):
            utils.makedirs(out_dir)
        psnrs = []
        ssims = []
        lpips_scores = []
        eval_start_time = time.time()
        
        # Initialize LPIPS for perceptual evaluation
        lpips_fn = None
        try:
            import lpips
            lpips_fn = lpips.LPIPS(net='vgg').cuda() if jax.devices()[0].platform == 'gpu' else lpips.LPIPS(net='vgg')
            print('* LPIPS initialized for perceptual evaluation on GPU')
        except (ImportError, Exception) as e:
            print(f'* LPIPS not available ({e}), skipping perceptual evaluation')
            
        # Initialize step-by-step logger for training metrics
        step_logger = create_logger(FLAGS.train_dir, "nerf_evaluation_steps.json")
        
        # Capture baseline memory before evaluation
        baseline_memory = memory_tracker.capture_snapshot(step=step)
        if not FLAGS.eval_once:
            showcase_index = np.random.randint(0, dataset.size)
        for idx in range(dataset.size):
            print(f"Evaluating {idx+1}/{dataset.size}")
            batch = next(dataset)
            if idx % FLAGS.approx_eval_skip != 0:
                continue
            pred_color, pred_disp, pred_acc = utils.render_image(
                functools.partial(render_pfn, state.optimizer.target),
                batch["rays"],
                rng,
                FLAGS.dataset == "llff",
                chunk=FLAGS.chunk,
            )
            if jax.host_id() != 0:  # Only record via host 0.
                continue
            if not FLAGS.eval_once and idx == showcase_index:
                showcase_color = pred_color
                showcase_disp = pred_disp
                showcase_acc = pred_acc
                if not FLAGS.render_path:
                    showcase_gt = batch["pixels"]
            # Calculate quality metrics for non-path rendering
            if not FLAGS.render_path:
                psnr = utils.compute_psnr(((pred_color - batch["pixels"]) ** 2).mean())
                ssim = ssim_fn(pred_color, batch["pixels"])
                print(f"PSNR = {psnr:.4f}, SSIM = {ssim:.4f}")
                psnrs.append(float(psnr))
                ssims.append(float(ssim))
                
                # Calculate LPIPS only every 5 iterations to save computation time
                lpips_score = None
                if lpips_fn is not None and idx % 5 == 0:
                    # Convert to tensor format for LPIPS (CHW format, range [0,1])
                    # Convert JAX arrays to numpy first
                    pred_np = np.array(pred_color) if hasattr(pred_color, 'device') else pred_color
                    gt_np = np.array(batch["pixels"]) if hasattr(batch["pixels"], 'device') else batch["pixels"]
                    pred_tensor = torch.from_numpy(pred_np).permute(2, 0, 1).unsqueeze(0).float()
                    gt_tensor = torch.from_numpy(gt_np).permute(2, 0, 1).unsqueeze(0).float()
                    
                    # Ensure tensors are on GPU for faster computation
                    if jax.devices()[0].platform == 'gpu':
                        pred_tensor = pred_tensor.cuda()
                        gt_tensor = gt_tensor.cuda()
                    
                    with torch.no_grad():
                        lpips_score = lpips_fn(pred_tensor, gt_tensor).item()
                        lpips_scores.append(lpips_score)
                        print(f"LPIPS = {lpips_score:.4f}")
                
                # Log each step to training steps JSON
                step_metrics = {
                    "psnr": float(psnr),
                    "ssim": float(ssim),
                    "lpips": lpips_score if lpips_score is not None else None,
                    "lpips_calculated": lpips_score is not None
                }
                step_additional_info = {
                    "image_idx": idx,
                    "model_step": step
                }
                step_logger.log_metrics(idx, "evaluation_step", step_metrics, step_additional_info)
            if FLAGS.save_output:
                utils.save_img(pred_color, path.join(out_dir, "{:03d}.png".format(idx)))
                utils.save_img(
                    pred_disp[Ellipsis, 0],
                    path.join(out_dir, "disp_{:03d}.png".format(idx)),
                )
        # Calculate final evaluation metrics and log to JSON
        if (jax.host_id() == 0) and (not FLAGS.render_path):
            eval_end_time = time.time()
            total_eval_time = eval_end_time - eval_start_time
            
            # Calculate average metrics
            avg_psnr = np.mean(np.array(psnrs)) if psnrs else 0.0
            avg_ssim = np.mean(np.array(ssims)) if ssims else 0.0
            avg_lpips = np.mean(np.array(lpips_scores)) if lpips_scores else 0.0
            
            print(f"\n=== FINAL EVALUATION RESULTS ===")
            print(f"Average PSNR: {avg_psnr:.4f}")
            print(f"Average SSIM: {avg_ssim:.4f}")
            print(f"Average LPIPS: {avg_lpips:.4f} (calculated from {len(lpips_scores)} samples)")
            print(f"Evaluation Time: {total_eval_time:.2f}s")
            print(f"Images Evaluated: {len(psnrs)}")
            print(f"LPIPS samples: {len(lpips_scores)}/{len(psnrs)} images")
            
            # Capture final memory snapshot
            final_memory = memory_tracker.capture_snapshot(step=step)
            memory_metrics = memory_tracker.get_memory_metrics(final_memory)
            
            # Calculate memory efficiency indices
            efficiency_indices = memory_tracker.calculate_efficiency_indices(
                psnr=avg_psnr,
                ssim=avg_ssim,
                lpips=avg_lpips,
                snapshot=final_memory
            )
            
            # Prepare timing and performance info
            timing_info = {
                "total_eval_time": total_eval_time,
                "images_evaluated": len(psnrs),
                "avg_time_per_image": total_eval_time / len(psnrs) if psnrs else 0.0,
                "total_rays": len(psnrs) * 640000,  # Assuming 800x800 images
                "rays_per_sec": (len(psnrs) * 640000) / total_eval_time if total_eval_time > 0 else 0.0,
                "lpips_samples": len(lpips_scores),
                "lpips_coverage": len(lpips_scores) / len(psnrs) if psnrs else 0.0
            }
            
            # Create separate final results JSON logger
            final_results_logger = create_logger(FLAGS.train_dir, "nerf_evaluation_summary.json")
            
            # Log comprehensive final results to separate JSON file
            final_results = {
                "model_step": step,
                "evaluation_timestamp": eval_end_time,
                "dataset_info": {
                    "dataset_size": dataset.size,
                    "images_evaluated": len(psnrs)
                },
                "metrics": {
                    "psnr": {
                        "average": avg_psnr,
                        "samples": len(psnrs),
                        "all_values": psnrs
                    },
                    "ssim": {
                        "average": avg_ssim,
                        "samples": len(ssims),
                        "all_values": ssims
                    },
                    "lpips": {
                        "average": avg_lpips,
                        "samples": len(lpips_scores),
                        "coverage_ratio": len(lpips_scores) / len(psnrs) if psnrs else 0.0,
                        "calculated_every_n_images": 5,
                        "all_values": lpips_scores,
                        "available": lpips_fn is not None
                    }
                },
                "timing_performance": timing_info,
                "memory_metrics": memory_metrics,
                "efficiency_indices": efficiency_indices,
                "configuration": {
                    "evaluation_type": "post_training_full_test_set",
                    "lpips_optimization": "calculated_every_5_iterations",
                    "gpu_acceleration": jax.devices()[0].platform == 'gpu'
                }
            }
            
            # Use log_metrics to log final results
            final_results_logger.log_metrics(step, "final_evaluation", final_results["metrics"], final_results)
            
            # Also log to the original evaluation metrics file for backwards compatibility
            metrics_logger.log_octree_evaluation(
                step=step,
                psnr=avg_psnr,
                ssim=avg_ssim,
                lpips=avg_lpips,
                timing_info=timing_info,
                octree_info={
                    "evaluation_type": "post_training_full_test_set",
                    "model_step": step,
                    "dataset_size": dataset.size,
                    "lpips_available": lpips_fn is not None
                },
                memory_metrics=memory_metrics,
                efficiency_indices=efficiency_indices
            )
            
            print(f"Step-by-step metrics saved to: {FLAGS.train_dir}/nerf_evaluation_steps.json")
            print(f"Final results saved to: {FLAGS.train_dir}/nerf_evaluation_summary.json")
            print(f"Final metrics saved to: {FLAGS.train_dir}/nerf_evaluation_final.json")
            
        if (not FLAGS.eval_once) and (jax.host_id() == 0):
            summary_writer.image("pred_color", showcase_color, step)
            summary_writer.image("pred_disp", showcase_disp, step)
            summary_writer.image("pred_acc", showcase_acc, step)
            if not FLAGS.render_path:
                summary_writer.scalar("psnr", avg_psnr, step)
                summary_writer.scalar("ssim", avg_ssim, step)
                if lpips_fn is not None:
                    summary_writer.scalar("lpips", avg_lpips, step)
                summary_writer.image("target", showcase_gt, step)
        if FLAGS.eval_once:
            break
        if int(step) >= FLAGS.max_steps:
            break
        last_step = step


if __name__ == "__main__":
    app.run(main)
