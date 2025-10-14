"""
Advanced quality metrics for Plenoxels evaluation

Implements:
- SMEI (Sparse Memory Efficiency Index): Quality-memory tradeoff metric
- FDR (Floater Detection Ratio): Ghosting artifacts quantification

Usage:
    from util.advanced_metrics import compute_SMEI, compute_FDR, compute_all_advanced_metrics
    
    # Load grid
    grid = svox2.SparseGrid.load('checkpoint.npz')
    
    # Compute SMEI (lightweight)
    smei_results = compute_SMEI(grid, psnr=28.5, use_fp16=False)
    print(f"SMEI: {smei_results['SMEI']:.4f}")
    
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


def compute_SMEI(
    grid,
    psnr: float,
    use_fp16: bool = False,
    include_basis: bool = False
) -> Dict[str, float]:
    """
    Compute Sparse Memory Efficiency Index (SMEI)
    
    SMEI = (PSNR × Sparsity_Exploitation) / Storage_Size_MB
    
    This metric captures the quality-memory tradeoff for sparse voxel grids.
    Higher is better - indicates better quality per MB of storage.
    
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
    main_object_threshold: float = 0.1,
    use_density_threshold: bool = True,
    max_resolution: Optional[Tuple[int, int, int]] = None
) -> Dict[str, float]:
    """
    Compute Floater Detection Ratio (FDR)
    
    FDR quantifies ghosting artifacts by detecting disconnected "floater" components
    in the density field that are significantly smaller than the main object.
    
    Args:
        grid: SparseGrid instance from svox2
        threshold: Minimum density to consider voxel occupied (default: 0.01)
        main_object_threshold: Volume ratio to distinguish floaters from main object
                               (default: 0.1 means components < 10% of main are floaters)
        use_density_threshold: If True, threshold by density; if False, use occupancy only
        max_resolution: If provided, downsample grid to this resolution before analysis
                       to save memory. Format: (x, y, z)
    
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
    
    # Connected component analysis
    labeled, num_components = ndimage.label(occupied)
    
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
    
    # Identify main object (largest component)
    main_volume = np.max(component_volumes)
    main_component_idx = np.argmax(component_volumes)
    
    # Identify floaters (components significantly smaller than main object)
    floater_mask = component_volumes < (main_volume * main_object_threshold)
    
    # Exclude the main component from floater consideration
    if not floater_mask[main_component_idx]:
        # Main component is not considered a floater (as expected)
        pass
    else:
        # This shouldn't happen since main component is largest
        floater_mask[main_component_idx] = False
    
    # Compute floater statistics
    floater_volumes = component_volumes[floater_mask]
    floater_volume = np.sum(floater_volumes)
    num_floaters = np.sum(floater_mask)
    
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
    
    return {
        'FDR': float(FDR),
        'num_floaters': int(num_floaters),
        'num_components': int(num_components),
        'main_volume': int(main_volume),
        'floater_volume': int(floater_volume),
        'total_volume': int(total_volume),
        'sparsity': float(sparsity),
        'largest_floater': int(largest_floater),
        'mean_floater_size': float(mean_floater_size)
    }


def compute_all_advanced_metrics(
    grid,
    psnr: float,
    use_fp16: bool = False,
    compute_fdr: bool = True,
    fdr_threshold: float = 0.01,
    verbose: bool = True
) -> Dict[str, float]:
    """
    Compute all advanced metrics at once
    
    Args:
        grid: SparseGrid instance from svox2
        psnr: Peak signal-to-noise ratio from evaluation
        use_fp16: Whether model uses FP16 storage
        compute_fdr: Whether to compute FDR (more expensive)
        fdr_threshold: Density threshold for FDR computation
        verbose: Print progress messages
        
    Returns:
        Dictionary with all computed metrics
    """
    results = {}
    
    # Always compute SMEI (lightweight)
    if verbose:
        print("Computing SMEI...")
    smei_results = compute_SMEI(grid, psnr, use_fp16=use_fp16)
    results.update({f'SMEI_{k}': v for k, v in smei_results.items()})
    results['SMEI'] = smei_results['SMEI']  # Also keep top-level SMEI
    
    # Optionally compute FDR (memory intensive)
    if compute_fdr and HAS_SCIPY:
        if verbose:
            print("Computing FDR...")
        fdr_results = compute_FDR(grid, threshold=fdr_threshold)
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
    
    # SMEI metrics
    if 'SMEI' in metrics:
        print(f"{indent}SMEI: {metrics['SMEI']:.4f}")
        if 'SMEI_storage_mb' in metrics:
            print(f"{indent}  Storage: {metrics['SMEI_storage_mb']:.2f} MB")
            print(f"{indent}    - Data: {metrics['SMEI_storage_data_mb']:.2f} MB")
            print(f"{indent}    - Index: {metrics['SMEI_storage_index_mb']:.2f} MB")
        if 'SMEI_active_voxels' in metrics:
            print(f"{indent}  Active voxels: {metrics['SMEI_active_voxels']:,}")
            print(f"{indent}  Total capacity: {metrics['SMEI_total_capacity']:,}")
            print(f"{indent}  Sparsity: {metrics['SMEI_sparsity']:.2%}")
            print(f"{indent}  Compression: {metrics['SMEI_compression_ratio']:.2f}x")
    
    # FDR metrics
    if 'FDR' in metrics and metrics['FDR'] >= 0:
        print(f"{indent}FDR: {metrics['FDR']:.2%}")
        if 'FDR_num_floaters' in metrics:
            print(f"{indent}  Floaters: {metrics['FDR_num_floaters']} / {metrics['FDR_num_components']} components")
            print(f"{indent}  Main volume: {metrics['FDR_main_volume']:,} voxels")
            print(f"{indent}  Floater volume: {metrics['FDR_floater_volume']:,} voxels")
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

