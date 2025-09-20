"""
Memory tracking utilities for PlenOctree training and evaluation.
Provides comprehensive memory monitoring for GPU and system memory usage.
"""

import psutil
import torch
import jax
import numpy as np
import gc
import os
import subprocess
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass


@dataclass
class MemorySnapshot:
    """Container for memory usage at a specific point in time."""
    timestamp: str
    step: int
    
    # GPU Memory (in GB)
    gpu_allocated: float
    gpu_reserved: float
    gpu_max_allocated: float
    gpu_max_reserved: float
    gpu_total: float
    gpu_free: float
    
    # System Memory (in GB)
    system_used: float
    system_available: float
    system_total: float
    system_percent: float
    
    # Process Memory (in GB)
    process_rss: float  # Resident Set Size
    process_vms: float  # Virtual Memory Size
    process_percent: float
    
    # JAX Memory (if available)
    jax_memory: Optional[float] = None
    
    # Nvidia-SMI GPU Memory (actual system-level usage)
    nvidia_smi_used_gb: Optional[float] = None
    nvidia_smi_total_gb: Optional[float] = None


class MemoryTracker:
    """Tracks memory usage throughout training and evaluation."""
    
    def __init__(self):
        # Check for both PyTorch and JAX GPU availability
        self.has_torch_gpu = torch.cuda.is_available()
        self.has_jax_gpu = self._check_jax_gpu()
        self.has_gpu = self.has_torch_gpu or self.has_jax_gpu
        self.device = "cuda" if self.has_gpu else "cpu"
        self.process = psutil.Process(os.getpid())
        
        print(f"Memory tracker initialized: PyTorch GPU: {self.has_torch_gpu}, JAX GPU: {self.has_jax_gpu}")
        
        # Peak memory tracking
        self.peak_gpu_allocated = 0.0
        self.peak_gpu_reserved = 0.0
        self.peak_system_used = 0.0
        self.peak_process_rss = 0.0
        
        # Initialize with current memory state to avoid zero peak values
        self._initialize_peaks()
        
        # Baseline memory (before training starts)
        self.baseline_snapshot: Optional[MemorySnapshot] = None
        
    def _bytes_to_gb(self, bytes_value: Union[int, float]) -> float:
        """Convert bytes to gigabytes."""
        return bytes_value / (1024 ** 3)
    
    def _check_jax_gpu(self) -> bool:
        """Check if JAX has GPU devices available."""
        try:
            import jax
            devices = jax.devices()
            return any(device.platform == 'gpu' for device in devices)
        except Exception:
            return False
    
    def _get_nvidia_smi_memory(self) -> Optional[Dict[str, float]]:
        """Get actual GPU memory usage from nvidia-smi (more accurate than PyTorch)."""
        try:
            # Run nvidia-smi to get memory info
            result = subprocess.run([
                'nvidia-smi', '--query-gpu=memory.used,memory.total,memory.free', 
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0:
                # Parse the output
                lines = result.stdout.strip().split('\n')
                if lines:
                    # Take first GPU (index 0)
                    used, total, free = map(int, lines[0].split(', '))
                    return {
                        'used_mb': used,
                        'total_mb': total,
                        'free_mb': free,
                        'used_gb': used / 1024.0,
                        'total_gb': total / 1024.0,
                        'free_gb': free / 1024.0
                    }
        except Exception as e:
            # nvidia-smi not available or failed
            pass
        return None
    
    def _initialize_peaks(self):
        """Initialize peak memory tracking with current values to avoid zero division."""
        try:
            if self.has_gpu:
                if self.has_torch_gpu:
                    self.peak_gpu_allocated = self._bytes_to_gb(torch.cuda.memory_allocated())
                    self.peak_gpu_reserved = self._bytes_to_gb(torch.cuda.memory_reserved())
                elif self.has_jax_gpu:
                    gpu_allocated, gpu_reserved, _, _, _, _ = self._get_jax_gpu_memory()
                    self.peak_gpu_allocated = gpu_allocated
                    self.peak_gpu_reserved = gpu_reserved
            
            system_memory = psutil.virtual_memory()
            self.peak_system_used = self._bytes_to_gb(system_memory.used)
            
            process_memory = self.process.memory_info()
            self.peak_process_rss = self._bytes_to_gb(process_memory.rss)
        except Exception as e:
            # If initialization fails, use small default values to avoid zero division
            print(f"Warning: Could not initialize peak memory tracking: {e}")
            self.peak_gpu_allocated = 0.001  # 1MB default
            self.peak_gpu_reserved = 0.001
            self.peak_system_used = 0.001
            self.peak_process_rss = 0.001
    
    def _get_jax_gpu_memory(self) -> tuple:
        """Get JAX GPU memory usage."""
        try:
            import jax
            
            # Try to get GPU memory info from nvidia-ml-py if available
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                
                gpu_total = self._bytes_to_gb(info.total)
                gpu_used = self._bytes_to_gb(info.used)
                gpu_free = self._bytes_to_gb(info.free)
                
                # For JAX, we'll use the GPU usage as both allocated and reserved
                gpu_allocated = gpu_used
                gpu_reserved = gpu_used
                gpu_max_allocated = gpu_used
                gpu_max_reserved = gpu_used
                
                return gpu_allocated, gpu_reserved, gpu_max_allocated, gpu_max_reserved, gpu_total, gpu_free
                
            except ImportError:
                # Fallback: use JAX device memory if available
                devices = jax.devices()
                if devices and hasattr(devices[0], 'memory_stats'):
                    stats = devices[0].memory_stats()
                    gpu_allocated = self._bytes_to_gb(stats.get('bytes_in_use', 0))
                    # For JAX without pynvml, we have limited info
                    return gpu_allocated, gpu_allocated, gpu_allocated, gpu_allocated, 0.0, 0.0
                else:
                    # Use nvidia-smi via subprocess as last resort
                    return self._get_gpu_memory_nvidia_smi()
                    
        except Exception as e:
            print(f"Warning: Could not get JAX GPU memory: {e}")
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    
    def _get_gpu_memory_nvidia_smi(self) -> tuple:
        """Get GPU memory using nvidia-smi as fallback."""
        try:
            import subprocess
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,nounits,noheader'],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                if lines:
                    used_mb, total_mb = map(int, lines[0].split(', '))
                    gpu_used = used_mb / 1024  # Convert MB to GB
                    gpu_total = total_mb / 1024
                    gpu_free = gpu_total - gpu_used
                    
                    return gpu_used, gpu_used, gpu_used, gpu_used, gpu_total, gpu_free
            
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        except Exception:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    
    def _get_jax_memory(self) -> Optional[float]:
        """Get JAX memory usage if available."""
        try:
            # JAX memory tracking is more complex and device-specific
            if hasattr(jax, 'devices'):
                devices = jax.devices()
                if devices and hasattr(devices[0], 'memory_stats'):
                    stats = devices[0].memory_stats()
                    return self._bytes_to_gb(stats.get('bytes_in_use', 0))
            return None
        except Exception:
            return None
    
    def capture_snapshot(self, step: int = 0, timestamp: Optional[str] = None) -> MemorySnapshot:
        """Capture current memory state."""
        from datetime import datetime
        if timestamp is None:
            timestamp = datetime.now().isoformat()
        
        # GPU Memory - handle both PyTorch and JAX
        if self.has_gpu:
            # Try PyTorch first, but fall back to JAX if no usage detected
            pytorch_allocated = 0.0
            if self.has_torch_gpu:
                pytorch_allocated = self._bytes_to_gb(torch.cuda.memory_allocated())
            
            # If PyTorch shows no usage but we have JAX GPU, use JAX detection
            if pytorch_allocated == 0.0 and self.has_jax_gpu:
                # JAX GPU memory (more comprehensive for JAX workloads)
                gpu_allocated, gpu_reserved, gpu_max_allocated, gpu_max_reserved, gpu_total, gpu_free = self._get_jax_gpu_memory()
            elif self.has_torch_gpu:
                # PyTorch GPU memory (for octree operations)
                gpu_allocated = pytorch_allocated
                gpu_reserved = self._bytes_to_gb(torch.cuda.memory_reserved())
                gpu_max_allocated = self._bytes_to_gb(torch.cuda.max_memory_allocated())
                gpu_max_reserved = self._bytes_to_gb(torch.cuda.max_memory_reserved())
                
                # Get total GPU memory
                gpu_props = torch.cuda.get_device_properties(0)
                gpu_total = self._bytes_to_gb(gpu_props.total_memory)
                gpu_free = gpu_total - gpu_reserved
            else:
                gpu_allocated = gpu_reserved = gpu_max_allocated = gpu_max_reserved = 0.0
                gpu_total = gpu_free = 0.0
            
            # Update peaks
            self.peak_gpu_allocated = max(self.peak_gpu_allocated, gpu_allocated)
            self.peak_gpu_reserved = max(self.peak_gpu_reserved, gpu_reserved)
        else:
            gpu_allocated = gpu_reserved = gpu_max_allocated = gpu_max_reserved = 0.0
            gpu_total = gpu_free = 0.0
        
        # System Memory
        system_memory = psutil.virtual_memory()
        system_used = self._bytes_to_gb(system_memory.used)
        system_available = self._bytes_to_gb(system_memory.available)
        system_total = self._bytes_to_gb(system_memory.total)
        system_percent = system_memory.percent
        
        self.peak_system_used = max(self.peak_system_used, system_used)
        
        # Process Memory
        process_memory = self.process.memory_info()
        process_rss = self._bytes_to_gb(process_memory.rss)
        process_vms = self._bytes_to_gb(process_memory.vms)
        process_percent = self.process.memory_percent()
        
        self.peak_process_rss = max(self.peak_process_rss, process_rss)
        
        # JAX Memory
        jax_memory = self._get_jax_memory()
        
        # Nvidia-SMI Memory (actual GPU usage)
        nvidia_smi_info = self._get_nvidia_smi_memory()
        nvidia_smi_used_gb = nvidia_smi_info['used_gb'] if nvidia_smi_info else None
        nvidia_smi_total_gb = nvidia_smi_info['total_gb'] if nvidia_smi_info else None
        
        snapshot = MemorySnapshot(
            timestamp=timestamp,
            step=step,
            gpu_allocated=gpu_allocated,
            gpu_reserved=gpu_reserved,
            gpu_max_allocated=gpu_max_allocated,
            gpu_max_reserved=gpu_max_reserved,
            gpu_total=gpu_total,
            gpu_free=gpu_free,
            system_used=system_used,
            system_available=system_available,
            system_total=system_total,
            system_percent=system_percent,
            process_rss=process_rss,
            process_vms=process_vms,
            process_percent=process_percent,
            jax_memory=jax_memory,
            nvidia_smi_used_gb=nvidia_smi_used_gb,
            nvidia_smi_total_gb=nvidia_smi_total_gb
        )
        
        if self.baseline_snapshot is None:
            self.baseline_snapshot = snapshot
        
        return snapshot
    
    def get_memory_metrics(self, snapshot: MemorySnapshot) -> Dict[str, float]:
        """Extract memory metrics for logging."""
        metrics = {
            "gpu_allocated_gb": snapshot.gpu_allocated,
            "gpu_reserved_gb": snapshot.gpu_reserved,
            "gpu_max_allocated_gb": snapshot.gpu_max_allocated,
            "gpu_max_reserved_gb": snapshot.gpu_max_reserved,
            "gpu_utilization_percent": (snapshot.gpu_allocated / snapshot.gpu_total * 100) if snapshot.gpu_total > 0 else 0.0,
            "system_used_gb": snapshot.system_used,
            "system_percent": snapshot.system_percent,
            "process_rss_gb": snapshot.process_rss,
            "process_percent": snapshot.process_percent,
            "peak_gpu_allocated_gb": self.peak_gpu_allocated,
            "peak_gpu_reserved_gb": self.peak_gpu_reserved,
            "peak_system_used_gb": self.peak_system_used,
            "peak_process_rss_gb": self.peak_process_rss
        }
        
        if snapshot.jax_memory is not None:
            metrics["jax_memory_gb"] = snapshot.jax_memory
        
        # Nvidia-SMI GPU memory (actual usage)
        if snapshot.nvidia_smi_used_gb is not None:
            metrics["nvidia_smi_used_gb"] = snapshot.nvidia_smi_used_gb
            metrics["nvidia_smi_total_gb"] = snapshot.nvidia_smi_total_gb
            metrics["nvidia_smi_utilization_percent"] = (snapshot.nvidia_smi_used_gb / snapshot.nvidia_smi_total_gb * 100) if snapshot.nvidia_smi_total_gb > 0 else 0.0
        
        # Memory delta from baseline if available
        if self.baseline_snapshot:
            metrics.update({
                "gpu_allocated_delta_gb": snapshot.gpu_allocated - self.baseline_snapshot.gpu_allocated,
                "system_used_delta_gb": snapshot.system_used - self.baseline_snapshot.system_used,
                "process_rss_delta_gb": snapshot.process_rss - self.baseline_snapshot.process_rss
            })
        
        return metrics
    
    def calculate_efficiency_indices(self, 
                                   psnr: float, 
                                   ssim: Optional[float] = None, 
                                   lpips: Optional[float] = None,
                                   snapshot: Optional[MemorySnapshot] = None) -> Dict[str, float]:
        """
        Calculate memory efficiency indices.
        
        Args:
            psnr: Peak Signal-to-Noise Ratio
            ssim: Structural Similarity Index (optional)
            lpips: Learned Perceptual Image Patch Similarity (optional)
            snapshot: Memory snapshot (if None, captures current state)
        
        Returns:
            Dictionary of efficiency indices
        """
        if snapshot is None:
            snapshot = self.capture_snapshot()
        
        indices = {}
        
        # Primary memory for calculations (use GPU if available, otherwise process memory)
        primary_memory = snapshot.gpu_allocated if self.has_gpu and snapshot.gpu_allocated > 0 else snapshot.process_rss
        peak_memory = self.peak_gpu_allocated if self.has_gpu else self.peak_process_rss
        
        # Safety check: ensure we have non-zero values for calculations
        if primary_memory <= 0:
            primary_memory = 0.001  # 1MB minimum to avoid division by zero
        if peak_memory <= 0:
            peak_memory = primary_memory  # Use current memory if no peak recorded
        
        # Memory Efficiency Index (MEI): PSNR per GB of memory
        indices["memory_efficiency_index"] = psnr / primary_memory
        indices["peak_memory_efficiency_index"] = psnr / peak_memory
        
        # Memory Scalability Factor (MSF): Memory usage normalized metric
        indices["memory_scalability_factor"] = primary_memory
        
        # Quality-Memory Trade-off (QMT)
        if ssim is not None:
            indices["quality_memory_tradeoff"] = (psnr * ssim) / primary_memory
            indices["peak_quality_memory_tradeoff"] = (psnr * ssim) / peak_memory
        
        # LPIPS-Memory Efficiency (lower LPIPS is better, so invert)
        if lpips is not None:
            # Use (1 - lpips) so higher is better, then divide by memory
            indices["lpips_memory_efficiency"] = (1.0 - lpips) / primary_memory
            indices["peak_lpips_memory_efficiency"] = (1.0 - lpips) / peak_memory
        
        # Combined Quality Index (if all metrics available)
        if ssim is not None and lpips is not None:
            # Combined quality: PSNR * SSIM * (1 - LPIPS)
            combined_quality = psnr * ssim * (1.0 - lpips)
            indices["combined_quality_memory_index"] = combined_quality / primary_memory
            indices["peak_combined_quality_memory_index"] = combined_quality / peak_memory
        
        return indices
    
    def get_model_size_estimate(self, model_state: Any = None) -> Dict[str, float]:
        """
        Estimate model size and parameter count.
        
        Args:
            model_state: Model state (JAX optimizer state or PyTorch model)
        
        Returns:
            Dictionary with model size information
        """
        info = {}
        
        try:
            if model_state is not None:
                # Try to calculate parameter count and size
                if hasattr(model_state, 'optimizer'):
                    # JAX-style optimizer state
                    params = model_state.optimizer.target
                    if hasattr(params, 'keys'):
                        total_params = 0
                        total_bytes = 0
                        for key, value in jax.tree_util.tree_flatten(params)[0]:
                            if hasattr(value, 'shape'):
                                param_count = np.prod(value.shape)
                                total_params += param_count
                                # Assume float32 (4 bytes per parameter)
                                total_bytes += param_count * 4
                        
                        info["model_parameters"] = total_params
                        info["model_size_gb"] = self._bytes_to_gb(total_bytes)
                        
                elif hasattr(model_state, 'parameters'):
                    # PyTorch-style model
                    total_params = sum(p.numel() for p in model_state.parameters())
                    total_bytes = sum(p.numel() * p.element_size() for p in model_state.parameters())
                    
                    info["model_parameters"] = total_params
                    info["model_size_gb"] = self._bytes_to_gb(total_bytes)
                    
        except Exception as e:
            print(f"Warning: Could not calculate model size: {e}")
        
        return info
    
    def cleanup_memory(self):
        """Force garbage collection and GPU memory cleanup."""
        gc.collect()
        if self.has_gpu:
            torch.cuda.empty_cache()
    
    def reset_peak_tracking(self):
        """Reset peak memory tracking."""
        self.peak_gpu_allocated = 0.0
        self.peak_gpu_reserved = 0.0
        self.peak_system_used = 0.0
        self.peak_process_rss = 0.0
        if self.has_gpu:
            torch.cuda.reset_peak_memory_stats()
    
    def print_memory_summary(self, snapshot: MemorySnapshot):
        """Print a human-readable memory summary."""
        print(f"\n{'='*60}")
        print(f"MEMORY SUMMARY - Step {snapshot.step}")
        print(f"{'='*60}")
        
        if self.has_gpu:
            print(f"GPU Memory:")
            print(f"  Allocated: {snapshot.gpu_allocated:.2f} GB")
            print(f"  Reserved:  {snapshot.gpu_reserved:.2f} GB")
            print(f"  Total:     {snapshot.gpu_total:.2f} GB")
            print(f"  Free:      {snapshot.gpu_free:.2f} GB")
            print(f"  Utilization: {(snapshot.gpu_allocated/snapshot.gpu_total*100):.1f}%")
        
        print(f"System Memory:")
        print(f"  Used:      {snapshot.system_used:.2f} GB ({snapshot.system_percent:.1f}%)")
        print(f"  Available: {snapshot.system_available:.2f} GB")
        print(f"  Total:     {snapshot.system_total:.2f} GB")
        
        print(f"Process Memory:")
        print(f"  RSS:       {snapshot.process_rss:.2f} GB ({snapshot.process_percent:.1f}%)")
        print(f"  VMS:       {snapshot.process_vms:.2f} GB")
        
        print(f"Peak Memory:")
        print(f"  GPU:       {self.peak_gpu_allocated:.2f} GB")
        print(f"  System:    {self.peak_system_used:.2f} GB")
        print(f"  Process:   {self.peak_process_rss:.2f} GB")
        
        print(f"{'='*60}\n")


# Convenience functions
def create_memory_tracker() -> MemoryTracker:
    """Create a MemoryTracker instance."""
    return MemoryTracker()


def get_current_memory_usage() -> Dict[str, float]:
    """Get current memory usage as a simple dictionary."""
    tracker = MemoryTracker()
    snapshot = tracker.capture_snapshot()
    return tracker.get_memory_metrics(snapshot)
