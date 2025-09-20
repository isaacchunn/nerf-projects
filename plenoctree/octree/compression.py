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
"""Compress a plenoctree.

Including quantization using median cut algorithm.

Usage:
python compression.py x.npz [y.npz ...]
"""
import sys
import numpy as np
import os.path as osp
import torch
from svox.helpers import _get_c_extension
from tqdm import tqdm
import os
import argparse
import time

# Import JSON logging and memory tracking
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from json_logger import create_logger
from memory_tracker import create_memory_tracker

@torch.no_grad()
def main():
    compression_start_time = time.time()
    
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str, nargs='+', default=None, help='Input npz(s)')
    parser.add_argument('--noquant', action='store_true',
            help='Disable quantization')
    parser.add_argument('--bits', type=int, default=16,
            help='Quantization bits (order)')
    parser.add_argument('--out_dir', type=str, default='min_alt',
            help='Where to write compressed npz')
    parser.add_argument('--overwrite', action='store_true',
            help='Overwrite existing compressed npz')
    parser.add_argument('--weighted', action='store_true',
            help='Use weighted median cut (seems quite useless)')
    parser.add_argument('--sigma_thresh', type=float, default=2.0,
            help='Kill voxels under this sigma')
    parser.add_argument('--retain', type=int, default=0,
            help='Do not compress first x SH coeffs, needed for some scenes to keep ok quality')

    args = parser.parse_args()
    
    # Initialize JSON logger and memory tracker  
    # Check if we're compressing from octrees directory, if so put metrics there too
    if any('octrees' in fname for fname in args.input):
        # If input files are from octrees directory, put metrics there
        octrees_dir = None
        for fname in args.input:
            if 'octrees' in fname:
                octrees_dir = os.path.dirname(fname)
                break
        if octrees_dir:
            metrics_logger = create_logger(octrees_dir, "octree_compression_metrics.json")
        else:
            metrics_logger = create_logger(args.out_dir, "octree_compression_metrics.json")
    else:
        metrics_logger = create_logger(args.out_dir, "octree_compression_metrics.json")
    memory_tracker = create_memory_tracker()
    
    # Capture baseline memory
    baseline_snapshot = memory_tracker.capture_snapshot(0)
    memory_tracker.print_memory_summary(baseline_snapshot)

    _C = _get_c_extension()
    os.makedirs(args.out_dir, exist_ok=True)

    if args.noquant:
        print('Quantization disabled, only applying deflate')
    else:
        print('Quantization enabled')

    file_index = 0
    total_files = len(args.input)
    compression_results = []
    
    for fname in args.input:
        file_start_time = time.time()
        base_name = osp.basename(fname)
        
        # If output would overwrite input (same directory), use compressed suffix
        input_dir = osp.realpath(osp.dirname(fname))
        output_dir = osp.realpath(args.out_dir)
        
        if input_dir == output_dir:
            name, ext = osp.splitext(base_name)
            fname_c = osp.join(args.out_dir, f"{name}_compressed{ext}")
            print(f"   📁 Same directory detected - using compressed suffix: {base_name} → {name}_compressed{ext}")
        else:
            fname_c = osp.join(args.out_dir, base_name)
            print(f"   📁 Different directories - using original name: {base_name}")
            
        print(f'Compressing {fname} to {fname_c} ({file_index + 1}/{total_files})')
        
        # Get original file size
        original_size_mb = osp.getsize(fname) / (1024 * 1024)
        
        if not args.overwrite and osp.exists(fname_c):
            print(' > skip')
            file_index += 1
            continue

        load_start = time.time()
        z = np.load(fname)
        load_time = time.time() - load_start

        if not args.noquant:
            if 'quant_colors' in z.files:
                print(' > skip since source already compressed')
                file_index += 1
                continue
                
        # Capture memory snapshot after loading
        load_snapshot = memory_tracker.capture_snapshot(file_index * 10 + 1)
        
        # Get octree information
        original_data_shape = z['data'].shape if 'data' in z else None
        original_elements = np.prod(original_data_shape) if original_data_shape else 0
        
        z = dict(z)
        del z['parent_depth']
        del z['geom_resize_fact']
        del z['n_free']
        del z['n_internal']
        del z['depth_limit']

        if not args.noquant:
            quantization_start = time.time()
            
            data = torch.from_numpy(z['data'])
            sigma = data[..., -1].reshape(-1)
            snz = sigma > args.sigma_thresh
            retained_voxels = snz.sum().item()
            total_voxels = snz.numel()
            voxel_retention_rate = retained_voxels / total_voxels
            
            sigma[~snz] = 0.0

            data = data[..., :-1]
            N = data.size(1)
            basis_dim = data.size(-1) // 3

            data = data.reshape(-1, 3, basis_dim).float()[snz].unbind(-1)
            if args.retain:
                retained = data[:args.retain]
                data = data[args.retain:]
            else:
                retained = None

            all_quant_colors = []
            all_quant_maps = []

            if args.weighted:
                weights = 1.0 - np.exp(-0.01 * sigma.float())
            else:
                weights = torch.empty((0,))

            # Track quantization progress
            quant_detail_start = time.time()
            for i, d in tqdm(enumerate(data), total=len(data), desc="Quantizing channels"):
                colors, color_id_map = _C.quantize_median_cut(d.contiguous(),
                                                              weights,
                                                              args.bits)
                color_id_map_full = np.zeros((snz.shape[0],), dtype=np.uint16)
                color_id_map_full[snz] = color_id_map

                all_quant_colors.append(colors.numpy().astype(np.float16))
                all_quant_maps.append(color_id_map_full.reshape(-1, N, N, N).astype(np.uint16))
            
            quant_detail_time = time.time() - quant_detail_start
            
            quant_map = np.stack(all_quant_maps, axis=0)
            quant_colors = np.stack(all_quant_colors, axis=0)
            del all_quant_maps
            del all_quant_colors
            
            z['quant_colors'] = quant_colors
            z['quant_map'] = quant_map
            z['sigma'] = sigma.reshape(-1, N, N, N)
            
            if args.retain:
                all_retained = []
                for i in range(args.retain):
                    retained_wz = np.zeros((snz.shape[0], 3), dtype=np.float16)
                    retained_wz[snz] = retained[i]
                    all_retained.append(retained_wz.reshape(-1, N, N, N, 3))
                all_retained = np.stack(all_retained, axis=0)
                del retained
                z['data_retained'] = all_retained
            del z['data']
            
            quantization_time = time.time() - quantization_start
        else:
            quantization_time = 0.0
            retained_voxels = total_voxels = original_elements
            voxel_retention_rate = 1.0
            quant_detail_time = 0.0
            
        # Save compressed file
        save_start = time.time()
        np.savez_compressed(fname_c, **z)
        save_time = time.time() - save_start
        
        # Calculate compression metrics
        compressed_size_mb = osp.getsize(fname_c) / (1024 * 1024)
        compression_ratio = original_size_mb / compressed_size_mb if compressed_size_mb > 0 else 0.0
        size_reduction_percent = (1.0 - compressed_size_mb / original_size_mb) * 100 if original_size_mb > 0 else 0.0
        
        file_time = time.time() - file_start_time
        
        print(f' > Size {original_size_mb:.1f} MB -> {compressed_size_mb:.1f} MB (Ratio: {compression_ratio:.1f}x, Reduction: {size_reduction_percent:.1f}%)')
        
        # Capture memory snapshot after compression
        compress_snapshot = memory_tracker.capture_snapshot(file_index * 10 + 2)
        memory_metrics = memory_tracker.get_memory_metrics(compress_snapshot)
        
        # Log compression metrics
        compression_metrics = {
            "original_size_mb": original_size_mb,
            "compressed_size_mb": compressed_size_mb,
            "compression_ratio": compression_ratio,
            "size_reduction_percent": size_reduction_percent,
            "quantization_enabled": not args.noquant,
            "quantization_bits": args.bits,
            "sigma_threshold": args.sigma_thresh,
            "retained_coeffs": args.retain,
            "weighted_quantization": args.weighted,
            "total_voxels": total_voxels,
            "retained_voxels": retained_voxels,
            "voxel_retention_rate": voxel_retention_rate
        }
        compression_metrics.update(memory_metrics)
        
        timing_info = {
            "total_file_time": file_time,
            "load_time": load_time,
            "quantization_time": quantization_time,
            "quantization_detail_time": quant_detail_time,
            "save_time": save_time,
            "input_file": fname,
            "output_file": fname_c,
            "file_index": file_index + 1,
            "total_files": total_files
        }
        
        metrics_logger.log_metrics(file_index + 1, "file_compression", compression_metrics, timing_info)
        
        # Store results for summary
        compression_results.append({
            "file": osp.basename(fname),
            "original_size_mb": original_size_mb,
            "compressed_size_mb": compressed_size_mb,
            "compression_ratio": compression_ratio,
            "compression_time": file_time
        })
        
        file_index += 1


    # Log compression summary
    total_compression_time = time.time() - compression_start_time
    final_snapshot = memory_tracker.capture_snapshot(999)
    final_memory_metrics = memory_tracker.get_memory_metrics(final_snapshot)
    
    if compression_results:
        total_original_size = sum(r["original_size_mb"] for r in compression_results)
        total_compressed_size = sum(r["compressed_size_mb"] for r in compression_results)
        overall_compression_ratio = total_original_size / total_compressed_size if total_compressed_size > 0 else 0.0
        overall_size_reduction = (1.0 - total_compressed_size / total_original_size) * 100 if total_original_size > 0 else 0.0
        
        summary_metrics = {
            "total_files_processed": len(compression_results),
            "total_original_size_mb": total_original_size,
            "total_compressed_size_mb": total_compressed_size,
            "overall_compression_ratio": overall_compression_ratio,
            "overall_size_reduction_percent": overall_size_reduction,
            "average_compression_ratio": np.mean([r["compression_ratio"] for r in compression_results]),
            "quantization_enabled": not args.noquant,
            "quantization_bits": args.bits,
            "sigma_threshold": args.sigma_thresh,
            "retained_coeffs": args.retain,
            "weighted_quantization": args.weighted
        }
        summary_metrics.update(final_memory_metrics)
        
        summary_timing = {
            "total_compression_time": total_compression_time,
            "average_time_per_file": total_compression_time / len(compression_results),
            "output_directory": args.out_dir,
            "files_processed": [r["file"] for r in compression_results]
        }
        
        metrics_logger.log_metrics(999, "compression_summary", summary_metrics, summary_timing)
        
        # Print comprehensive summary
        print(f'\n=== Compression Complete ===')
        print(f'Total files processed: {len(compression_results)}')
        print(f'Total compression time: {total_compression_time:.2f}s')
        print(f'Overall size: {total_original_size:.1f} MB -> {total_compressed_size:.1f} MB')
        print(f'Overall compression ratio: {overall_compression_ratio:.1f}x')
        print(f'Overall size reduction: {overall_size_reduction:.1f}%')
        print(f'Average time per file: {total_compression_time/len(compression_results):.2f}s')
        print(f'Quantization: {"Enabled" if not args.noquant else "Disabled"}')
        if not args.noquant:
            print(f'  - Bits: {args.bits}')
            print(f'  - Sigma threshold: {args.sigma_thresh}')
            print(f'  - Retained coeffs: {args.retain}')
            print(f'  - Weighted: {args.weighted}')
        # Show the correct metrics file location
        if any('octrees' in fname for fname in args.input):
            octrees_dir = None
            for fname in args.input:
                if 'octrees' in fname:
                    octrees_dir = os.path.dirname(fname)
                    break
            if octrees_dir:
                metrics_path = osp.join(octrees_dir, "octree_compression_metrics.json")
            else:
                metrics_path = osp.join(args.out_dir, "octree_compression_metrics.json")
        else:
            metrics_path = osp.join(args.out_dir, "octree_compression_metrics.json")
        print(f'Metrics logged to: {metrics_path}')
        
        # Print memory summary
        memory_tracker.print_memory_summary(final_snapshot)
    else:
        print("No files were processed.")

if __name__ == '__main__':
    main()
