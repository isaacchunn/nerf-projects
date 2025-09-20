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
"""Optimize a plenoctree through finetuning on train set.

Usage:

export DATA_ROOT=./data/NeRF/nerf_synthetic/
export CKPT_ROOT=./data/PlenOctree/checkpoints/syn_sh16
export SCENE=chair
export CONFIG_FILE=nerf_sh/config/blender

python -m octree.optimization \
    --input $CKPT_ROOT/$SCENE/tree.npz \
    --config $CONFIG_FILE \
    --data_dir $DATA_ROOT/$SCENE/ \
    --output $CKPT_ROOT/$SCENE/octrees/tree_opt.npz
"""
import svox
import torch
import torch.cuda
import numpy as np
import json
import imageio
import os.path as osp
import os
import time
from argparse import ArgumentParser
from tqdm import tqdm
from torch.optim import SGD, Adam
from warnings import warn

from absl import app
from absl import flags

from octree.nerf import datasets
from octree.nerf import utils

# Import JSON logging and memory tracking
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from json_logger import create_logger
from memory_tracker import create_memory_tracker

FLAGS = flags.FLAGS

utils.define_flags()

flags.DEFINE_string(
    "input",
    "./tree.npz",
    "Input octree npz from extraction.py",
)
flags.DEFINE_string(
    "output",
    "./tree_opt.npz",
    "Output octree npz",
)
flags.DEFINE_integer(
    'render_interval',
    0,
    'render interval')
flags.DEFINE_integer(
    'val_interval',
    2,
    'validation interval')
flags.DEFINE_integer(
    'num_epochs',
    80,
    'epochs to train for')
flags.DEFINE_bool(
    'sgd',
    True,
    'use SGD optimizer instead of Adam')
flags.DEFINE_float(
    'lr',
    1e7,
    'optimizer step size')
flags.DEFINE_float(
    'sgd_momentum',
    0.0,
    'sgd momentum')
flags.DEFINE_bool(
    'sgd_nesterov',
    False,
    'sgd nesterov momentum?')
flags.DEFINE_string(
    "write_vid",
    None,
    "If specified, writes rendered video to given path (*.mp4)",
)

# Manual 'val' set
flags.DEFINE_bool(
    "split_train",
    None,
    "If specified, splits train set instead of loading val set",
)
flags.DEFINE_float(
    "split_holdout_prop",
    0.2,
    "Proportion of images to hold out if split_train is set",
)

# Do not save since it is slow
flags.DEFINE_bool(
    "nosave",
    False,
    "If set, does not save (for speed)",
)

flags.DEFINE_bool(
    "continue_on_decrease",
    False,
    "If set, continues training even if validation PSNR decreases",
)

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.autograd.set_detect_anomaly(True)


def main(unused_argv):
    optimization_start_time = time.time()
    
    utils.set_random_seed(20200823)
    utils.update_flags(FLAGS)
    
    # Initialize JSON logger and memory tracker
    log_dir = os.path.dirname(FLAGS.output) if os.path.dirname(FLAGS.output) else "."
    metrics_logger = create_logger(log_dir, "octree_optimization_metrics.json")
    memory_tracker = create_memory_tracker()
    
    # Capture baseline memory after initialization
    baseline_snapshot = memory_tracker.capture_snapshot(0)
    memory_tracker.print_memory_summary(baseline_snapshot)

    def get_data(stage):
        assert stage in ["train", "val", "test"]
        dataset = datasets.get_dataset(stage, FLAGS)
        focal = dataset.focal
        all_c2w = dataset.camtoworlds
        all_gt = dataset.images.reshape(-1, dataset.h, dataset.w, 3)
        all_c2w = torch.from_numpy(all_c2w).float().to(device)
        all_gt = torch.from_numpy(all_gt).float()
        return focal, all_c2w, all_gt

    focal, train_c2w, train_gt = get_data("train")
    if FLAGS.split_train:
        test_sz = int(train_c2w.size(0) * FLAGS.split_holdout_prop)
        print('Splitting train to train/val manually, holdout', test_sz)
        perm = torch.randperm(train_c2w.size(0))
        test_c2w = train_c2w[perm[:test_sz]]
        test_gt = train_gt[perm[:test_sz]]
        train_c2w = train_c2w[perm[test_sz:]]
        train_gt = train_gt[perm[test_sz:]]
    else:
        print('Using given val set')
        test_focal, test_c2w, test_gt = get_data("val")
        assert focal == test_focal
    H, W = train_gt[0].shape[:2]

    vis_dir = osp.splitext(FLAGS.input)[0] + '_render'
    os.makedirs(vis_dir, exist_ok=True)

    print('N3Tree load')
    t = svox.N3Tree.load(FLAGS.input, map_location=device)
    #  t.nan_to_num_()

    if 'llff' in FLAGS.config:
        ndc_config = svox.NDCConfig(width=W, height=H, focal=focal)
    else:
        ndc_config = None
    r = svox.VolumeRenderer(t, step_size=FLAGS.renderer_step_size, ndc=ndc_config)

    if FLAGS.sgd:
        print('Using SGD, lr', FLAGS.lr)
        if FLAGS.lr < 1.0:
            warn('For SGD please adjust LR to about 1e7')
        optimizer = SGD(t.parameters(), lr=FLAGS.lr, momentum=FLAGS.sgd_momentum,
                        nesterov=FLAGS.sgd_nesterov)
    else:
        adam_eps = 1e-4 if t.data.dtype is torch.float16 else 1e-8
        print('Using Adam, eps', adam_eps, 'lr', FLAGS.lr)
        optimizer = Adam(t.parameters(), lr=FLAGS.lr, eps=adam_eps)

    n_train_imgs = len(train_c2w)
    n_test_imgs = len(test_c2w)

    def run_test_step(i):
        print('Evaluating')
        eval_start_time = time.time()
        
        # Initialize LPIPS if available
        try:
            import lpips
            lpips_vgg = lpips.LPIPS(net="vgg").eval().to(device)
            use_lpips = True
        except ImportError:
            print("Warning: LPIPS not available, skipping LPIPS calculation")
            use_lpips = False
            lpips_vgg = None
        
        with torch.no_grad():
            tpsnr = 0.0
            tssim = 0.0
            tlpips = 0.0
            
            for j, (c2w, im_gt) in enumerate(zip(test_c2w, test_gt)):
                im = r.render_persp(c2w, height=H, width=W, fx=focal, fast=False)
                im = im.cpu().clamp_(0.0, 1.0)
                im_gt_cpu = im_gt.cpu()

                # Calculate PSNR
                mse = ((im - im_gt_cpu) ** 2).mean()
                psnr = -10.0 * np.log(mse) / np.log(10.0)
                tpsnr += psnr.item()
                
                # Calculate SSIM
                ssim = utils.compute_ssim(im, im_gt_cpu, max_val=1.0).mean()
                tssim += ssim.item()
                
                # Calculate LPIPS if available
                if use_lpips:
                    im_gpu = im.to(device)
                    im_gt_gpu = im_gt_cpu.to(device)
                    lpips_val = lpips_vgg(im_gt_gpu.permute([2, 0, 1]).contiguous(),
                                         im_gpu.permute([2, 0, 1]).contiguous(), normalize=True)
                    tlpips += lpips_val.item()

                if FLAGS.render_interval > 0 and j % FLAGS.render_interval == 0:
                    vis = torch.cat((im_gt_cpu, im), dim=1)
                    vis = (vis * 255).numpy().astype(np.uint8)
                    imageio.imwrite(f"{vis_dir}/{i:04}_{j:04}.png", vis)
                    
            tpsnr /= n_test_imgs
            tssim /= n_test_imgs
            if use_lpips:
                tlpips /= n_test_imgs
            
            # Capture memory snapshot and log validation metrics
            eval_snapshot = memory_tracker.capture_snapshot(i)
            memory_metrics = memory_tracker.get_memory_metrics(eval_snapshot)
            efficiency_indices = memory_tracker.calculate_efficiency_indices(
                tpsnr, tssim, tlpips if use_lpips else None, eval_snapshot
            )
            
            eval_time = time.time() - eval_start_time
            timing_info = {
                "eval_time": eval_time,
                "dataset_size": n_test_imgs,
                "phase": "validation"
            }
            
            # Log validation metrics
            validation_metrics = {
                "psnr": tpsnr,
                "ssim": tssim
            }
            if use_lpips:
                validation_metrics["lpips"] = tlpips
            
            validation_metrics.update(memory_metrics)
            
            additional_info = timing_info.copy()
            additional_info.update(efficiency_indices)
            
            metrics_logger.log_metrics(i, "validation", validation_metrics, additional_info)
            
            return tpsnr, tssim, tlpips if use_lpips else 0.0

    # Initial validation
    initial_val_psnr, initial_val_ssim, initial_val_lpips = run_test_step(0)
    best_validation_psnr = initial_val_psnr
    print('** initial val psnr ', best_validation_psnr, 'ssim', initial_val_ssim, 'lpips', initial_val_lpips)
    
    best_t = None
    for i in range(FLAGS.num_epochs):
        epoch_start_time = time.time()
        print('epoch', i)
        tpsnr = 0.0
        
        for j, (c2w, im_gt) in tqdm(enumerate(zip(train_c2w, train_gt)), total=n_train_imgs):
            im = r.render_persp(c2w, height=H, width=W, fx=focal, cuda=True)
            im_gt_ten = im_gt.to(device=device)
            im = torch.clamp(im, 0.0, 1.0)
            mse = ((im - im_gt_ten) ** 2).mean()
            im_gt_ten = None

            optimizer.zero_grad()
            t.data.grad = None  # This helps save memory weirdly enough
            mse.backward()
            #  print('mse', mse, t.data.grad.min(), t.data.grad.max())
            optimizer.step()
            #  t.data.data -= eta * t.data.grad
            psnr = -10.0 * np.log(mse.detach().cpu()) / np.log(10.0)
            tpsnr += psnr.item()
            
        tpsnr /= n_train_imgs
        epoch_time = time.time() - epoch_start_time
        print('** train_psnr', tpsnr)
        
        # Capture memory snapshot and log training metrics
        train_snapshot = memory_tracker.capture_snapshot(i + 1)
        memory_metrics = memory_tracker.get_memory_metrics(train_snapshot)
        
        # Log training metrics
        training_metrics = {
            "psnr": tpsnr,
            "mse": ((tpsnr / -10.0) * np.log(10.0)),  # Convert back to MSE for logging
            "learning_rate": FLAGS.lr
        }
        training_metrics.update(memory_metrics)
        
        timing_info = {
            "epoch_time": epoch_time,
            "dataset_size": n_train_imgs,
            "phase": "training"
        }
        
        metrics_logger.log_metrics(i + 1, "training", training_metrics, timing_info)

        # Validation step
        if i % FLAGS.val_interval == FLAGS.val_interval - 1 or i == FLAGS.num_epochs - 1:
            validation_psnr, validation_ssim, validation_lpips = run_test_step(i + 1)
            print('** val psnr ', validation_psnr, 'ssim', validation_ssim, 'lpips', validation_lpips, 'best', best_validation_psnr)
            
            if validation_psnr > best_validation_psnr:
                best_validation_psnr = validation_psnr
                best_t = t.clone(device='cpu')  # SVOX 0.2.22
                print('** New best model saved')
            elif not FLAGS.continue_on_decrease:
                print('Stop since overfitting')
                break
    # Log final optimization summary
    total_optimization_time = time.time() - optimization_start_time
    final_snapshot = memory_tracker.capture_snapshot(-1)
    final_memory_metrics = memory_tracker.get_memory_metrics(final_snapshot)
    
    summary_info = {
        "total_optimization_time": total_optimization_time,
        "total_epochs": FLAGS.num_epochs,
        "best_validation_psnr": best_validation_psnr,
        "initial_validation_psnr": initial_val_psnr,
        "improvement": best_validation_psnr - initial_val_psnr,
        "optimizer": "SGD" if FLAGS.sgd else "Adam",
        "learning_rate": FLAGS.lr,
        "output_path": FLAGS.output
    }
    summary_info.update(final_memory_metrics)
    
    metrics_logger.log_metrics(-1, "optimization_summary", {"final_best_psnr": best_validation_psnr}, summary_info)
    
    if not FLAGS.nosave:
        if best_t is not None:
            print('Saving best model to', FLAGS.output)
            best_t.save(FLAGS.output, compress=False)
            print(f'Optimization completed in {total_optimization_time:.2f}s')
            print(f'Best validation PSNR: {best_validation_psnr:.4f} (improvement: {best_validation_psnr - initial_val_psnr:.4f})')
        else:
            print('Did not improve upon initial model')
            
    print(f'Metrics logged to: {os.path.join(log_dir, "octree_optimization_metrics.json")}')

if __name__ == "__main__":
    app.run(main)
