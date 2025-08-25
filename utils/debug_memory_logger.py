"""
GPU Memory Debug Logger for CUDA OOM troubleshooting.
Provides concise logging of GPU memory usage at key points.
"""

import torch
import logging
import time
from typing import Optional, Dict, Any

class GPUMemoryLogger:
    """Lightweight GPU memory logger for debugging CUDA OOM errors."""
    
    def __init__(self, log_file: str = "debug_logger_virus.log"):
        """Initialize the GPU memory logger.
        
        Args:
            log_file: Path to the log file
        """
        self.log_file = log_file
        self.logger = logging.getLogger("gpu_memory_debug")
        
        # Create file handler
        file_handler = logging.FileHandler(log_file, mode='w')
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        # Remove existing handlers and add our file handler
        self.logger.handlers.clear()
        self.logger.addHandler(file_handler)
        self.logger.setLevel(logging.INFO)
        
        self.start_time = time.time()
        
        # Log initial state
        if torch.cuda.is_available():
            self.log_memory("INIT", "Logger initialized")
        else:
            self.logger.info("INIT - No CUDA available")
    
    def log_memory(self, stage: str, description: str, additional_info: Optional[Dict[str, Any]] = None):
        """Log current GPU memory usage.
        
        Args:
            stage: Stage identifier (e.g., "BATCH_START", "FORWARD", etc.)
            description: Brief description of what's happening
            additional_info: Additional information to log
        """
        if not torch.cuda.is_available():
            return
            
        try:
            # Get memory statistics
            allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            reserved = torch.cuda.memory_reserved() / 1024**3   # GB
            max_allocated = torch.cuda.max_memory_allocated() / 1024**3  # GB
            
            # Format the message
            elapsed = time.time() - self.start_time
            msg = f"[{elapsed:.1f}s] {stage} - {description} | Alloc: {allocated:.2f}GB | Reserved: {reserved:.2f}GB | Max: {max_allocated:.2f}GB"
            
            # Add additional info if provided
            if additional_info:
                info_str = " | ".join([f"{k}: {v}" for k, v in additional_info.items()])
                msg += f" | {info_str}"
            
            self.logger.info(msg)
            
        except Exception as e:
            self.logger.error(f"Error logging memory: {e}")
    
    def log_tensor_info(self, stage: str, tensor_name: str, tensor: torch.Tensor):
        """Log information about a specific tensor.
        
        Args:
            stage: Stage identifier
            tensor_name: Name of the tensor
            tensor: The tensor to analyze
        """
        if tensor is None:
            self.logger.info(f"[{time.time() - self.start_time:.1f}s] {stage} - {tensor_name}: None")
            return
        
        try:
            size_mb = tensor.numel() * tensor.element_size() / 1024**2
            info = {
                "shape": str(list(tensor.shape)),
                "dtype": str(tensor.dtype),
                "device": str(tensor.device),
                "size_mb": f"{size_mb:.2f}"
            }
            self.log_memory(stage, f"Tensor {tensor_name}", info)
        except Exception as e:
            self.logger.error(f"Error logging tensor {tensor_name}: {e}")
    
    def log_batch_info(self, stage: str, batch: Dict[str, Any]):
        """Log information about a batch.
        
        Args:
            stage: Stage identifier  
            batch: The batch dictionary
        """
        try:
            batch_info = {}
            total_size_mb = 0
            
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    size_mb = value.numel() * value.element_size() / 1024**2
                    total_size_mb += size_mb
                    batch_info[f"{key}_shape"] = str(list(value.shape))
                elif isinstance(value, dict):
                    # Handle nested dictionaries (like source_samples, target_samples)
                    for nested_key, nested_value in value.items():
                        if isinstance(nested_value, torch.Tensor):
                            size_mb = nested_value.numel() * nested_value.element_size() / 1024**2
                            total_size_mb += size_mb
                            batch_info[f"{key}_{nested_key}_shape"] = str(list(nested_value.shape))
            
            batch_info["total_batch_mb"] = f"{total_size_mb:.2f}"
            self.log_memory(stage, "Batch loaded", batch_info)
            
        except Exception as e:
            self.logger.error(f"Error logging batch info: {e}")
    
    def log_model_info(self, stage: str, model_name: str, model):
        """Log information about a model.
        
        Args:
            stage: Stage identifier
            model_name: Name of the model
            model: The model to analyze
        """
        try:
            if model is None:
                self.logger.info(f"[{time.time() - self.start_time:.1f}s] {stage} - Model {model_name}: None")
                return
                
            # Count parameters
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            # Estimate model size (rough approximation)
            param_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2
            
            info = {
                "total_params": f"{total_params:,}",
                "trainable_params": f"{trainable_params:,}",
                "param_size_mb": f"{param_size_mb:.2f}"
            }
            self.log_memory(stage, f"Model {model_name}", info)
            
        except Exception as e:
            self.logger.error(f"Error logging model info for {model_name}: {e}")
    
    def reset_peak_memory(self):
        """Reset peak memory statistics."""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            self.log_memory("RESET", "Peak memory stats reset")
    
    def log_cuda_info(self):
        """Log CUDA device information."""
        if not torch.cuda.is_available():
            self.logger.info("CUDA_INFO - CUDA not available")
            return
            
        try:
            device_count = torch.cuda.device_count()
            current_device = torch.cuda.current_device()
            device_name = torch.cuda.get_device_name(current_device)
            total_memory = torch.cuda.get_device_properties(current_device).total_memory / 1024**3
            
            info = {
                "device_count": device_count,
                "current_device": current_device,
                "device_name": device_name,
                "total_memory_gb": f"{total_memory:.2f}"
            }
            self.log_memory("CUDA_INFO", "Device information", info)
            
        except Exception as e:
            self.logger.error(f"Error logging CUDA info: {e}")

# Global instance
_debug_logger = None

def get_debug_logger(log_file: str = "/orcd/archive/abugoot/001/Projects/paolo/CoupledDistributionEmbeddings/debug_logger_virus.log") -> GPUMemoryLogger:
    """Get the global debug logger instance."""
    global _debug_logger
    if _debug_logger is None:
        _debug_logger = GPUMemoryLogger(log_file)
    return _debug_logger
