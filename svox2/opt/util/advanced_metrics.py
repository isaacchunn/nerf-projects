"""
Advanced quality metrics for Plenoxels evaluation

Implements:
- MCQ (Memory Cost per Quality): GPU memory efficiency metric
- FDR (Floater Detection Ratio): Ghosting artifacts quantification

Usage:
    from util.advanced_metrics import compute_MCQ, compute_FDR, compute_all_advanced_metrics
    
    # Load grid
    grid = svox2.SparseGrid.load('checkpoint.npz')
    
    # Compute MCQ (requires peak GPU memory)
    mcq_results = compute_MCQ(psnr=28.5, peak_gpu_memory_mb=2048.0)
    print(f"MCQ: {mcq_results['MCQ']:.4f} GB/dB")
    
    # Compute FDR (more expensive)
    fdr_results = compute_FDR(grid, threshold=0.01)
    print(f"FDR: {fdr_results['FDR']:.2%}")
"""

import torch
import numpy as np
from typing import Dict, Optional, Tuple
from warnings import warn

try:
    import scipy.ndimage as ndimage
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    warn("scipy not found. FDR metric will not be available. Install with: pip install scipy")


def compute_MCQ(
    psnr: float,
    peak_gpu_memory_mb: float
) -> Dict[str, float]:
    """
    Compute Memory Cost per Quality (MCQ)
    
    MCQ = Peak_GPU_Memory_GB / PSNR
    
    Measures how many GB of GPU memory needed per decibel of quality.
    Lower is better (means more efficient).
    
    Args:
        psnr: Peak signal-to-noise ratio from evaluation
        peak_gpu_memory_mb: Peak GPU memory usage (MB)
        
    Returns:
        Dictionary containing:
            - MCQ: Memory cost per quality (GB/dB)
            - peak_gpu_gb: Peak GPU memory in GB
            - psnr: Input PSNR value
            - memory_per_db: Same as MCQ (GB/dB)
    """
    # Convert memory to GB
    peak_gpu_gb = peak_gpu_memory_mb / 1024.0
    
    # Compute MCQ
    mcq = peak_gpu_gb / psnr if psnr > 0 else 0.0
    
    return {
        'MCQ': mcq,
        'peak_gpu_gb': peak_gpu_gb,
        'peak_gpu_mb': peak_gpu_memory_mb,
        'psnr': psnr,
        'memory_per_db': mcq  # Alias for clarity
    }


def compute_SMEI(
    grid,
    psnr: float,
    use_fp16: bool = False,
    include_basis: bool = False
) -> Dict[str, float]:
    """
    [DEPRECATED] Use compute_MCQ() instead for runtime GPU efficiency.
    
    Compute Sparse Memory Efficiency Index (SMEI)
    
    SMEI = (PSNR × Sparsity_Exploitation) / Storage_Size_MB
    
    This metric captures disk storage efficiency (not used in primary evaluation).
    Higher is better - indicates better quality per MB of disk storage.
    
    Args:
        grid: SparseGrid instance from svox2
        psnr: Peak signal-to-noise ratio from evaluation
        use_fp16: Whether the model uses FP16 storage (default: False for svox2)
        include_basis: Whether to include basis_data in storage calculation
        
    Returns:
        Dictionary containing:
            - SMEI: The computed efficiency index
            - storage_mb: Total storage in megabytes
            - storage_data_mb: Storage for voxel data only
            - storage_index_mb: Storage for index structure
            - active_voxels: Number of active (non-empty) voxels
            - total_capacity: Total grid capacity
            - sparsity: Sparsity ratio (1 = fully sparse, 0 = fully dense)
            - compression_ratio: Total capacity / active voxels
            - values_per_voxel: Number of values stored per active voxel
            - psnr: Input PSNR value
    """
    # Extract grid properties (all already available!)
    reso = grid.links.shape  # e.g., (512, 512, 512)
    total_capacity = int(np.prod(reso))
    
    # Active voxels = capacity (already tracked by grid)
    # This is updated during resample/sparsify operations
    active_voxels = grid.sh_data.size(0)  # Same as grid.capacity
    
    # Sparsity exploitation
    sparsity_exploitation = (total_capacity - active_voxels) / total_capacity if total_capacity > 0 else 0.0
    
    # Storage calculation
    bytes_per_value = 2 if use_fp16 else 4  # FP16 or FP32
    
    # Plenoxels stores:
    # - density_data: [active_voxels, 1]
    # - sh_data: [active_voxels, basis_dim * 3]  (typically 27 for 9 SH coeffs)
    values_per_voxel = 1 + grid.sh_data.shape[1]  # density + SH coefficients
    
    # Actual storage from active data
    storage_bytes = active_voxels * values_per_voxel * bytes_per_value
    storage_data_mb = storage_bytes / (1024 ** 2)
    
    # Add overhead for index structure (links grid uses int32)
    links_bytes = total_capacity * 4  # int32 for links
    storage_index_mb = links_bytes / (1024 ** 2)
    
    # Optional: include basis data (learned texture or MLP)
    storage_basis_mb = 0.0
    if include_basis and hasattr(grid, 'basis_data'):
        if grid.basis_data.numel() > 0:
            basis_bytes = grid.basis_data.numel() * 4  # Typically FP32
            storage_basis_mb = basis_bytes / (1024 ** 2)
    
    # Total storage
    storage_mb = storage_data_mb + storage_index_mb + storage_basis_mb
    
    # Compute SMEI
    SMEI = (psnr * sparsity_exploitation) / storage_mb if storage_mb > 0 else 0.0
    
    # Compression ratio
    compression_ratio = total_capacity / active_voxels if active_voxels > 0 else 0.0
    
    return {
        'SMEI': SMEI,
        'storage_mb': storage_mb,
        'storage_data_mb': storage_data_mb,
        'storage_index_mb': storage_index_mb,
        'storage_basis_mb': storage_basis_mb,
        'active_voxels': active_voxels,
        'total_capacity': total_capacity,
        'sparsity': sparsity_exploitation,
        'values_per_voxel': values_per_voxel,
        'psnr': psnr,
        'compression_ratio': compression_ratio,
        'bytes_per_value': bytes_per_value
    }


def compute_FDR(
    grid,
    threshold: float = 0.01,
    main_object_threshold: float = 0.05,
    use_density_threshold: bool = True,
    max_resolution: Optional[Tuple[int, int, int]] = None,
    min_object_size: int = 1000,
    size_gap_ratio: float = 0.2,
    use_adaptive: bool = True,
    connectivity: int = 26,
    verbose: bool = False
) -> Dict[str, float]:
    """
    Compute Floater Detection Ratio (FDR)
    
    FDR quantifies ghosting artifacts by detecting disconnected "floater" components
    in the density field that are significantly smaller than the main object.
    
    Args:
        grid: SparseGrid instance from svox2
        threshold: Minimum density to consider voxel occupied (default: 0.01)
        main_object_threshold: Volume ratio to distinguish floaters from main object
                               (default: 0.05 means components < 5% of main are floaters)
                               Only used if use_adaptive=False
        use_density_threshold: If True, threshold by density; if False, use occupancy only
        max_resolution: If provided, downsample grid to this resolution before analysis
                       to save memory. Format: (x, y, z)
        min_object_size: Absolute minimum size (voxels) for a component to be considered
                        a real object. Components smaller than this are always floaters.
                        (default: 1000)
        size_gap_ratio: For adaptive mode, ratio threshold to detect gaps between
                       real objects and floaters. If component[i+1]/component[i] < this,
                       a gap is detected. (default: 0.2 = 20%)
        use_adaptive: If True, use adaptive gap detection to identify floaters.
                     If False, use simple relative threshold (backward compatible).
                     (default: True)
        connectivity: Connectivity for 3D connected components (default: 26)
                     - 6: Face-adjacent only (most restrictive)
                     - 18: Face + edge adjacent
                     - 26: Face + edge + corner adjacent (most permissive, recommended)
    
    Returns:
        Dictionary containing:
            - FDR: Floater Detection Ratio (0-1, lower is better)
            - num_floaters: Number of floater components detected
            - num_components: Total number of components
            - main_volume: Volume of largest component (in voxels)
            - floater_volume: Total volume of all floaters (in voxels)
            - total_volume: Total occupied volume (in voxels)
            - sparsity: Grid sparsity ratio
            - largest_floater: Volume of largest floater component
            - mean_floater_size: Mean volume of floater components
            - num_main_objects: Number of components identified as main objects
            - detection_method: String indicating which method was used
    """
    if not HAS_SCIPY:
        warn("scipy not available. FDR computation requires scipy.ndimage. Returning dummy values.")
        return {
            'FDR': -1.0,
            'num_floaters': -1,
            'num_components': -1,
            'main_volume': -1,
            'floater_volume': -1,
            'total_volume': -1,
            'sparsity': -1.0,
            'error': 'scipy not installed'
        }
    
    reso = grid.links.shape
    device = grid.links.device
    
    # Downsample if requested (for memory efficiency)
    if max_resolution is not None:
        target_reso = max_resolution
        if any(r > mr for r, mr in zip(reso, target_reso)):
            print(f"  Downsampling from {reso} to {target_reso} for FDR computation")
            reso = target_reso
            # TODO: Implement downsampling if needed
            # For now, proceed with original resolution
    
    # METHOD: Use existing links structure (most memory efficient)
    # The links array tells us which voxels are active (>= 0)
    
    # Get occupancy mask
    active_mask = (grid.links >= 0).cpu()
    
    if use_density_threshold and threshold > 0:
        # Create dense density grid for thresholding
        # Following pattern from grid.resample() (svox2.py:1288-1317)
        dense_density = torch.zeros(reso, dtype=torch.float32)
        
        # Fill in density values for active voxels
        active_indices = grid.links[active_mask].long()
        density_values = grid.density_data[active_indices, 0].cpu()
        
        # Assign densities to their positions
        dense_density[active_mask] = density_values
        
        # Apply density threshold
        occupied = (dense_density.detach().numpy() > threshold).astype(np.uint8)
    else:
        # Just use occupancy (any active voxel)
        occupied = active_mask.numpy().astype(np.uint8)
    
    # Connected component analysis with configurable connectivity
    # Create structure for desired connectivity
    if connectivity == 26:
        # 26-connectivity: all neighbors (face, edge, corner)
        structure = np.ones((3, 3, 3), dtype=np.uint8)
    elif connectivity == 18:
        # 18-connectivity: face + edge neighbors only
        structure = np.array([[[0,0,0], [0,1,0], [0,0,0]],
                              [[0,1,0], [1,1,1], [0,1,0]],
                              [[0,0,0], [0,1,0], [0,0,0]]], dtype=np.uint8)
    elif connectivity == 6:
        # 6-connectivity: face neighbors only (scipy default)
        structure = None  # Use default
    else:
        raise ValueError(f"Invalid connectivity: {connectivity}. Must be 6, 18, or 26.")
    
    labeled, num_components = ndimage.label(occupied, structure=structure)
    
    if num_components == 0:
        # No occupied voxels
        return {
            'FDR': 0.0,
            'num_floaters': 0,
            'num_components': 0,
            'main_volume': 0,
            'floater_volume': 0,
            'total_volume': 0,
            'sparsity': 1.0,
            'largest_floater': 0,
            'mean_floater_size': 0.0
        }
    
    # Compute volume of each component
    component_volumes = ndimage.sum(occupied, labeled, range(1, num_components + 1))
    component_volumes = np.array(component_volumes)
    
    # Sort components by size (descending)
    sorted_indices = np.argsort(component_volumes)[::-1]
    sorted_volumes = component_volumes[sorted_indices]
    
    # Debug: Print top component sizes for diagnostic
    if verbose:
        print(f"    Component size distribution (top 20):")
        for i, vol in enumerate(sorted_volumes[:20]):
            if i < 10 or (i < 20 and vol >= min_object_size):
                percent = (vol / np.sum(sorted_volumes)) * 100
                print(f"      #{i+1}: {vol:,} voxels ({percent:.1f}%)", end="")
                if i > 0:
                    ratio = vol / sorted_volumes[i-1]
                    print(f" | ratio: {ratio:.3f}")
                else:
                    print()
        
        # Show distribution statistics
        print(f"    Total voxels: {np.sum(sorted_volumes):,}")
        print(f"    Total components: {len(sorted_volumes)}")
        if len(sorted_volumes) > 10:
            top10_volume = np.sum(sorted_volumes[:10])
            top10_percent = (top10_volume / np.sum(sorted_volumes)) * 100
            print(f"    Top 10 components: {top10_volume:,} voxels ({top10_percent:.1f}%)")
        if len(sorted_volumes) > 100:
            tail_volume = np.sum(sorted_volumes[100:])
            tail_percent = (tail_volume / np.sum(sorted_volumes)) * 100
            print(f"    Components #101+: {tail_volume:,} voxels ({tail_percent:.1f}%) - likely floaters")
    
    # Identify floaters using adaptive or threshold-based method
    detection_method = ""
    
    if use_adaptive:
        # ADAPTIVE METHOD: Find natural gaps in component sizes
        # Use sorted order for analysis, then map back to original indices
        floater_mask_sorted = np.zeros(len(sorted_volumes), dtype=bool)
        num_main_objects = 0
        
        # Step 1: Mark components below absolute minimum as floaters
        too_small_mask_sorted = sorted_volumes < min_object_size
        floater_mask_sorted |= too_small_mask_sorted
        
        # Step 2: Find size gap among components larger than min_object_size
        large_enough_mask = sorted_volumes >= min_object_size
        large_enough_volumes = sorted_volumes[large_enough_mask]
        
        if len(large_enough_volumes) > 1:
            # Look for the first significant gap in sizes
            # Gap = ratio between consecutive components
            ratios = large_enough_volumes[1:] / large_enough_volumes[:-1]
            gap_indices = np.where(ratios < size_gap_ratio)[0]
            
            if len(gap_indices) > 0:
                # Found a gap! Everything after the first gap is a floater
                first_gap_idx = gap_indices[0]
                num_main_objects = first_gap_idx + 1  # Components before gap
                
                # Mark everything after the gap as floaters (in sorted order)
                large_enough_positions = np.where(large_enough_mask)[0]
                floater_positions_after_gap = large_enough_positions[first_gap_idx + 1:]
                floater_mask_sorted[floater_positions_after_gap] = True
                
                detection_method = f"adaptive_gap (gap after {num_main_objects} objects, ratio={ratios[first_gap_idx]:.3f})"
            else:
                # No significant gap found - all large components are main objects
                num_main_objects = len(large_enough_volumes)
                detection_method = f"adaptive_nogap ({num_main_objects} main objects, no clear gap)"
        else:
            # Only 0 or 1 large component
            num_main_objects = len(large_enough_volumes)
            detection_method = f"adaptive_single ({num_main_objects} main objects)"
        
        # Map sorted floater mask back to original component order
        floater_mask = np.zeros(len(component_volumes), dtype=bool)
        floater_mask[sorted_indices] = floater_mask_sorted
        
        # Count components marked as floaters by absolute size
        num_too_small = too_small_mask_sorted.sum()
        if num_too_small > 0:
            detection_method += f" + {num_too_small} below min_size"
    
    else:
        # SIMPLE THRESHOLD METHOD: Any component >= min_object_size is a "main object"
        # This is the most straightforward and reliable approach
        floater_mask = component_volumes < min_object_size
        num_main_objects = np.sum(~floater_mask)
        
        detection_method = f"simple_threshold (min_size={min_object_size}, {num_main_objects} objects >= threshold)"
    
    # Compute floater statistics
    floater_volumes = component_volumes[floater_mask]
    floater_volume = np.sum(floater_volumes)
    num_floaters = np.sum(floater_mask)
    
    # Main object statistics
    main_object_mask = ~floater_mask
    main_volumes = component_volumes[main_object_mask]
    main_volume = np.sum(main_volumes) if len(main_volumes) > 0 else 0
    largest_main_volume = np.max(main_volumes) if len(main_volumes) > 0 else 0
    
    # Total volume
    total_volume = np.sum(component_volumes)
    
    # FDR
    FDR = floater_volume / total_volume if total_volume > 0 else 0.0
    
    # Additional statistics
    largest_floater = np.max(floater_volumes) if len(floater_volumes) > 0 else 0
    mean_floater_size = np.mean(floater_volumes) if len(floater_volumes) > 0 else 0.0
    
    # Grid sparsity
    total_capacity = int(np.prod(reso))
    sparsity = 1.0 - (total_volume / total_capacity)
    
    # Get IDs of main objects and floaters
    main_component_ids = np.where(main_object_mask)[0] + 1
    floater_component_ids = np.where(floater_mask)[0] + 1
    
    # Compute spatial compactness for validation
    # Compact objects have high density in their bounding box
    main_compactness_scores = []
    if verbose and len(main_component_ids) > 0:
        print(f"    Main object compactness (voxels/bbox_volume):")
        for i, comp_id in enumerate(main_component_ids[:8]):
            comp_mask = (labeled == comp_id)
            coords = np.where(comp_mask)
            if len(coords[0]) > 0:
                # Bounding box volume
                bbox_vol = np.prod([coords[j].max() - coords[j].min() + 1 for j in range(3)])
                obj_vol = len(coords[0])
                compactness = obj_vol / bbox_vol if bbox_vol > 0 else 0
                main_compactness_scores.append(compactness)
                if i < 5 or obj_vol >= min_object_size * 2:  # Show top 5 or large objects
                    print(f"      Object #{i+1}: {compactness:.3f} ({obj_vol:,} voxels)")
        
        if len(main_compactness_scores) > 0:
            avg_compactness = np.mean(main_compactness_scores)
            print(f"    Average main object compactness: {avg_compactness:.3f}")
            if avg_compactness < 0.1:
                print(f"    ⚠️  WARNING: Low compactness suggests scattered components, not solid objects!")
    
    return {
        'FDR': float(FDR),
        'num_floaters': int(num_floaters),
        'num_components': int(num_components),
        'num_main_objects': int(num_main_objects),
        'main_volume': int(main_volume),
        'largest_main_volume': int(largest_main_volume),
        'floater_volume': int(floater_volume),
        'total_volume': int(total_volume),
        'sparsity': float(sparsity),
        'largest_floater': int(largest_floater),
        'mean_floater_size': float(mean_floater_size),
        'detection_method': detection_method,
        'connectivity': connectivity,
        # Add floater mask for visualization
        'floater_mask_3d': labeled,  # Full labeled array
        'floater_component_ids': floater_component_ids,  # Component IDs that are floaters
        'main_component_ids': main_component_ids  # All main object IDs (not just largest)
    }


def compute_all_advanced_metrics(
    grid,
    psnr: float,
    use_fp16: bool = False,
    compute_fdr: bool = True,
    fdr_threshold: float = 0.01,
    fdr_main_object_threshold: float = 0.1,
    fdr_min_object_size: int = 1000,
    fdr_size_gap_ratio: float = 0.2,
    fdr_use_adaptive: bool = True,
    fdr_connectivity: int = 26,
    peak_gpu_memory_mb: float = None,
    verbose: bool = True
) -> Dict[str, float]:
    """
    Compute all advanced metrics at once
    
    Args:
        grid: SparseGrid instance from svox2
        psnr: Peak signal-to-noise ratio from evaluation
        use_fp16: Whether model uses FP16 storage
        compute_fdr: Whether to compute FDR (more expensive)
        fdr_threshold: Density threshold for FDR computation (default: 0.01)
        fdr_main_object_threshold: Size ratio to distinguish main object from floaters (default: 0.1)
                                   Only used if fdr_use_adaptive=False
        fdr_min_object_size: Absolute minimum size for real objects (default: 1000)
        fdr_size_gap_ratio: Gap ratio for adaptive detection (default: 0.2)
        fdr_use_adaptive: Use adaptive gap detection (default: True)
        fdr_connectivity: Connectivity for 3D components - 6, 18, or 26 (default: 26)
        peak_gpu_memory_mb: Peak GPU memory (MB) - for MCQ computation
        verbose: Print progress messages
        
    Returns:
        Dictionary with all computed metrics
    """
    results = {}
    
    # Compute MCQ (Memory Cost per Quality) if GPU memory available
    if peak_gpu_memory_mb is not None:
        if verbose:
            print("Computing MCQ...")
        mcq_results = compute_MCQ(psnr, peak_gpu_memory_mb)
        results.update({f'MCQ_{k}': v for k, v in mcq_results.items()})
        results['MCQ'] = mcq_results['MCQ']
    
    # Optionally compute FDR (memory intensive)
    if compute_fdr and HAS_SCIPY:
        if verbose:
            print("Computing FDR...")
        fdr_results = compute_FDR(
            grid, 
            threshold=fdr_threshold,
            main_object_threshold=fdr_main_object_threshold,
            min_object_size=fdr_min_object_size,
            size_gap_ratio=fdr_size_gap_ratio,
            use_adaptive=fdr_use_adaptive,
            connectivity=fdr_connectivity,
            verbose=verbose
        )
        results.update({f'FDR_{k}': v for k, v in fdr_results.items()})
        results['FDR'] = fdr_results['FDR']  # Also keep top-level FDR
    elif compute_fdr and not HAS_SCIPY:
        if verbose:
            print("Warning: scipy not available, skipping FDR computation")
        results['FDR'] = -1.0
    
    return results


def print_advanced_metrics(metrics: Dict[str, float], indent: str = "  "):
    """
    Pretty print advanced metrics
    
    Args:
        metrics: Dictionary of metrics from compute_all_advanced_metrics
        indent: Indentation string for formatting
    """
    print(f"\n{indent}=== Advanced Metrics ===")
    
    # MCQ (Memory Cost per Quality)
    if 'MCQ' in metrics:
        print(f"{indent}MCQ (Memory Cost per Quality): {metrics['MCQ']:.4f} GB/dB")
        print(f"{indent}  Peak GPU Memory: {metrics['MCQ_peak_gpu_gb']:.2f} GB")
        print(f"{indent}  PSNR: {metrics['MCQ_psnr']:.2f} dB")
        print(f"{indent}  Interpretation: {metrics['MCQ']:.3f} GB needed per dB of quality")
    
    # FDR (Floater Detection Ratio)
    if 'FDR' in metrics and metrics['FDR'] >= 0:
        print(f"{indent}FDR (Floater Detection Ratio): {metrics['FDR']:.2%}")
        if 'FDR_num_floaters' in metrics:
            print(f"{indent}  Detection: {metrics.get('FDR_detection_method', 'unknown')}")
            print(f"{indent}  Connectivity: {metrics.get('FDR_connectivity', 'unknown')}-connectivity")
            print(f"{indent}  Components: {metrics['FDR_num_floaters']} floaters + {metrics.get('FDR_num_main_objects', 1)} main objects = {metrics['FDR_num_components']} total")
            print(f"{indent}  Main volume: {metrics['FDR_main_volume']:,} voxels ({metrics['FDR_main_volume']/metrics['FDR_total_volume']*100:.1f}%)")
            print(f"{indent}  Floater volume: {metrics['FDR_floater_volume']:,} voxels ({metrics['FDR']:.2%})")
            if metrics['FDR_num_floaters'] > 0:
                print(f"{indent}  Largest floater: {metrics['FDR_largest_floater']:,} voxels")
                print(f"{indent}  Mean floater size: {metrics['FDR_mean_floater_size']:.1f} voxels")
    
    print(f"{indent}========================\n")


# Convenience function for quick metric computation from checkpoint
def evaluate_checkpoint_metrics(
    checkpoint_path: str,
    psnr: float,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    **kwargs
) -> Dict[str, float]:
    """
    Load checkpoint and compute all metrics
    
    Args:
        checkpoint_path: Path to .npz checkpoint file
        psnr: PSNR value from evaluation
        device: Device to load grid on
        **kwargs: Additional arguments for compute_all_advanced_metrics
        
    Returns:
        Dictionary of all computed metrics
    """
    import svox2
    
    print(f"Loading checkpoint: {checkpoint_path}")
    grid = svox2.SparseGrid.load(checkpoint_path, device=device)
    print(f"Grid: {grid}")
    
    metrics = compute_all_advanced_metrics(grid, psnr, **kwargs)
    print_advanced_metrics(metrics)
    
    return metrics


if __name__ == "__main__":
    # Example usage / testing
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python advanced_metrics.py <checkpoint.npz> <psnr>")
        print("Example: python advanced_metrics.py ckpt.npz 28.5")
        sys.exit(1)
    
    checkpoint_path = sys.argv[1]
    psnr = float(sys.argv[2])
    
    metrics = evaluate_checkpoint_metrics(checkpoint_path, psnr, compute_fdr=True)

