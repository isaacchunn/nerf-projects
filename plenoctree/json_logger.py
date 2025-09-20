"""
JSON logging utility for PlenOctree training and evaluation metrics.
Provides consistent logging across the entire pipeline.
"""

import json
import os
from datetime import datetime
from typing import Dict, Any, Optional
import numpy as np
import jax.numpy as jnp


class MetricsLogger:
    """Handles JSON logging of training and evaluation metrics."""
    
    def __init__(self, log_dir: str, log_filename: str = "metrics_log.json", clean_existing: bool = True):
        """
        Initialize the metrics logger.
        
        Args:
            log_dir: Directory to save the JSON log file
            log_filename: Name of the JSON log file
            clean_existing: Whether to remove existing log file before creating new one
        """
        self.log_dir = log_dir
        self.log_file = os.path.join(log_dir, log_filename)
        
        # Ensure directory exists
        os.makedirs(log_dir, exist_ok=True)
        
        # Clean existing file if requested (for robustness against overlapping experiments)
        if clean_existing and os.path.exists(self.log_file):
            os.remove(self.log_file)
            print(f"🧹 Cleaned existing metrics file: {log_filename}")
        
        # Initialize empty log file
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w') as f:
                f.write('[\n')
                f.write(']\n')
    
    def _convert_to_serializable(self, obj: Any) -> Any:
        """Convert numpy/jax arrays and other non-serializable objects to Python types."""
        if isinstance(obj, (np.ndarray, jnp.ndarray)):
            return obj.tolist() if obj.size > 1 else float(obj)
        elif isinstance(obj, (np.floating, jnp.floating)):
            return float(obj)
        elif isinstance(obj, (np.integer, jnp.integer)):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: self._convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._convert_to_serializable(item) for item in obj]
        else:
            return obj
    
    def log_metrics(self, 
                   step: int, 
                   phase: str, 
                   metrics: Dict[str, Any], 
                   additional_info: Optional[Dict[str, Any]] = None):
        """
        Log metrics to JSON file.
        
        Args:
            step: Current training/evaluation step
            phase: Phase of training (e.g., "training", "evaluation", "octree_eval")
            metrics: Dictionary of metrics to log
            additional_info: Additional information to include in the log
        """
        timestamp = datetime.now().isoformat()
        
        log_entry = {
            "timestamp": timestamp,
            "step": int(step),
            "phase": phase,
            "metrics": self._convert_to_serializable(metrics)
        }
        
        if additional_info:
            log_entry["additional_info"] = self._convert_to_serializable(additional_info)
        
        # Read existing log, add new entry, and write back
        try:
            with open(self.log_file, 'r') as f:
                content = f.read().strip()
            
            # Remove the closing bracket
            if content.endswith(']'):
                content = content[:-1].rstrip()
            
            # Add comma if there are existing entries
            if content != '[':
                content += ','
            
            # Add new entry
            content += '\n' + json.dumps(log_entry, indent=2)
            content += '\n]'
            
            with open(self.log_file, 'w') as f:
                f.write(content)
                
        except Exception as e:
            print(f"Warning: Failed to write to metrics log: {e}")
    
    def log_training_step(self, step: int, stats: Any, lr: float, timing_info: Dict[str, float], 
                         memory_metrics: Optional[Dict[str, float]] = None,
                         efficiency_indices: Optional[Dict[str, float]] = None):
        """
        Log training step metrics.
        
        Args:
            step: Training step
            stats: Training statistics object
            lr: Learning rate
            timing_info: Dictionary with timing information
            memory_metrics: Dictionary with memory usage metrics
            efficiency_indices: Dictionary with memory efficiency indices
        """
        metrics = {
            "loss": float(stats.loss[0]) if hasattr(stats.loss, '__getitem__') else float(stats.loss),
            "psnr": float(stats.psnr[0]) if hasattr(stats.psnr, '__getitem__') else float(stats.psnr),
            "learning_rate": float(lr),
            "weight_l2": float(stats.weight_l2[0]) if hasattr(stats.weight_l2, '__getitem__') else float(stats.weight_l2)
        }
        
        # Add coarse metrics if available
        if hasattr(stats, 'loss_c') and stats.loss_c is not None:
            loss_c_val = stats.loss_c[0] if hasattr(stats.loss_c, '__getitem__') else stats.loss_c
            if float(loss_c_val) != 0.0:
                metrics["loss_coarse"] = float(loss_c_val)
                metrics["psnr_coarse"] = float(stats.psnr_c[0]) if hasattr(stats.psnr_c, '__getitem__') else float(stats.psnr_c)
        
        # Add sparsity loss if available
        if hasattr(stats, 'loss_sp') and stats.loss_sp is not None:
            loss_sp_val = stats.loss_sp[0] if hasattr(stats.loss_sp, '__getitem__') else stats.loss_sp
            if float(loss_sp_val) != 0.0:
                metrics["sparsity_loss"] = float(loss_sp_val)
        
        # Add memory metrics if provided
        if memory_metrics:
            metrics.update(memory_metrics)
        
        # Combine timing info with efficiency indices
        additional_info = timing_info.copy() if timing_info else {}
        if efficiency_indices:
            additional_info.update(efficiency_indices)
        
        self.log_metrics(step, "training", metrics, additional_info if additional_info else None)
    
    def log_evaluation_step(self, step: int, psnr: float, ssim: float, 
                           timing_info: Dict[str, float], 
                           additional_metrics: Optional[Dict[str, float]] = None,
                           memory_metrics: Optional[Dict[str, float]] = None,
                           efficiency_indices: Optional[Dict[str, float]] = None):
        """
        Log evaluation step metrics.
        
        Args:
            step: Evaluation step
            psnr: PSNR value
            ssim: SSIM value
            timing_info: Dictionary with timing information
            additional_metrics: Any additional metrics to log
            memory_metrics: Dictionary with memory usage metrics
            efficiency_indices: Dictionary with memory efficiency indices
        """
        metrics = {
            "psnr": float(psnr),
            "ssim": float(ssim)
        }
        
        if additional_metrics:
            metrics.update({k: float(v) for k, v in additional_metrics.items()})
        
        # Add memory metrics if provided
        if memory_metrics:
            metrics.update(memory_metrics)
        
        # Combine timing info with efficiency indices
        additional_info = timing_info.copy() if timing_info else {}
        if efficiency_indices:
            additional_info.update(efficiency_indices)
        
        self.log_metrics(step, "evaluation", metrics, additional_info if additional_info else None)
    
    def log_octree_evaluation(self, step: int, psnr: float, ssim: float, lpips: float,
                             timing_info: Optional[Dict[str, float]] = None,
                             octree_info: Optional[Dict[str, Any]] = None,
                             memory_metrics: Optional[Dict[str, float]] = None,
                             efficiency_indices: Optional[Dict[str, float]] = None):
        """
        Log octree evaluation metrics.
        
        Args:
            step: Evaluation step
            psnr: PSNR value
            ssim: SSIM value
            lpips: LPIPS value
            timing_info: Dictionary with timing information
            octree_info: Additional octree-specific information
            memory_metrics: Dictionary with memory usage metrics
            efficiency_indices: Dictionary with memory efficiency indices
        """
        metrics = {
            "psnr": float(psnr),
            "ssim": float(ssim),
            "lpips": float(lpips)
        }
        
        # Add memory metrics if provided
        if memory_metrics:
            metrics.update(memory_metrics)
        
        additional_info = {}
        if timing_info:
            additional_info.update(timing_info)
        if octree_info:
            additional_info.update(octree_info)
        if efficiency_indices:
            additional_info.update(efficiency_indices)
        
        self.log_metrics(step, "octree_evaluation", metrics, additional_info if additional_info else None)


# Convenience function for easy import
def create_logger(log_dir: str, log_filename: str = "metrics_log.json", clean_existing: bool = True) -> MetricsLogger:
    """Create a MetricsLogger instance."""
    return MetricsLogger(log_dir, log_filename, clean_existing)
