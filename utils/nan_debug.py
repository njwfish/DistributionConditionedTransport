import logging
import os
from typing import Any, Dict, Iterable, Optional, Tuple

import torch


class NaNDebugLogger:
    """Dedicated logger for NaN/Inf diagnostics with rich tensor/model stats.

    Writes to a separate file to avoid polluting standard logs.
    """

    def __init__(self, log_file_path: str):
        self.log_file_path = os.path.abspath(log_file_path)
        os.makedirs(os.path.dirname(self.log_file_path), exist_ok=True)

        self.logger = logging.getLogger("nan_debug")

        # Ensure we do not duplicate handlers if re-initialized
        if not any(isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", None) == self.log_file_path for h in self.logger.handlers):
            file_handler = logging.FileHandler(self.log_file_path, mode="w")
            formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
            self.logger.setLevel(logging.INFO)

        self.logger.info(f"NaN debug logger initialized at: {self.log_file_path}")

    def log(self, message: str, extra: Optional[Dict[str, Any]] = None, level: int = logging.INFO):
        if extra:
            kv = " | ".join(f"{k}={v}" for k, v in extra.items())
            message = f"{message} | {kv}"
        self.logger.log(level, message)

    def _safe_stats(self, tensor: torch.Tensor) -> Dict[str, Any]:
        stats: Dict[str, Any] = {}
        try:
            stats["shape"] = list(tensor.shape)
            stats["dtype"] = str(tensor.dtype)
            stats["device"] = str(tensor.device)
            if tensor.numel() == 0:
                stats["empty"] = True
                return stats
            t = tensor.detach()
            if t.is_floating_point():
                # Compute stats on finite values only to avoid reliance on torch.nan* ops
                finite_mask = torch.isfinite(t)
                num_finite = int(finite_mask.sum().item())
                stats["num_finite"] = num_finite
                if num_finite > 0:
                    t_f = t[finite_mask]
                    stats["min"] = float(t_f.min().item())
                    stats["max"] = float(t_f.max().item())
                    stats["mean"] = float(t_f.mean().item())
                    # Handle 1-element tensors for std safely
                    if t_f.numel() > 1:
                        stats["std"] = float(t_f.std(unbiased=False).item())
                    else:
                        stats["std"] = 0.0
                else:
                    stats["all_nonfinite"] = True
            else:
                stats["min"] = int(t.min().item())
                stats["max"] = int(t.max().item())
        except Exception as e:
            stats["error"] = f"{type(e).__name__}: {e}"
        try:
            stats["num_nan"] = int(torch.isnan(tensor).sum().item()) if tensor.is_floating_point() else 0
            stats["num_inf"] = int(torch.isinf(tensor).sum().item()) if tensor.is_floating_point() else 0
        except Exception:
            pass
        return stats

    def log_tensor(self, name: str, tensor: Optional[torch.Tensor]):
        if tensor is None:
            self.logger.info(f"{name}: None")
            return
        stats = self._safe_stats(tensor)
        self.log(f"Tensor '{name}'", stats)

    def log_tensors_dict(self, prefix: str, data: Dict[str, Any], max_items: int = 50):
        count = 0
        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                self.log_tensor(f"{prefix}.{key}", value)
                count += 1
                if count >= max_items:
                    self.log(f"{prefix}: truncated tensor logging after {max_items} items")
                    break
            elif isinstance(value, dict):
                inner = {f"{key}.{k}": v for k, v in value.items()}
                self.log_tensors_dict(prefix, inner, max_items=max_items)

    def log_batch(self, stage: str, batch: Dict[str, Any]):
        try:
            shapes: Dict[str, Any] = {}
            total_mb = 0.0
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    shapes[key] = list(value.shape)
                    total_mb += (value.numel() * value.element_size()) / 1024.0 / 1024.0
                elif isinstance(value, dict):
                    for k, v in value.items():
                        if isinstance(v, torch.Tensor):
                            shapes[f"{key}.{k}"] = list(v.shape)
                            total_mb += (v.numel() * v.element_size()) / 1024.0 / 1024.0
            self.log(f"Batch {stage}", {"shapes": shapes, "total_mb": f"{total_mb:.2f}"})
        except Exception as e:
            self.log(f"Error logging batch at stage {stage}", {"error": f"{type(e).__name__}: {e}"}, level=logging.ERROR)

    def log_model_summary(self, model_name: str, model: torch.nn.Module):
        try:
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            self.log(
                f"Model {model_name}",
                {
                    "total_params": f"{total_params:,}",
                    "trainable_params": f"{trainable_params:,}",
                },
            )
        except Exception as e:
            self.log(f"Error logging model summary for {model_name}", {"error": f"{type(e).__name__}: {e}"}, level=logging.ERROR)

    def _iter_named_parameters(self, model: torch.nn.Module) -> Iterable[Tuple[str, torch.nn.Parameter]]:
        for name, param in model.named_parameters(recurse=True):
            yield name, param

    def log_grad_stats(self, model_name: str, model: torch.nn.Module, top_k: int = 10):
        try:
            grad_norms: Dict[str, float] = {}
            for name, param in self._iter_named_parameters(model):
                if param.grad is None:
                    continue
                grad = param.grad.detach()
                if grad.numel() == 0:
                    continue
                norm = float(torch.norm(grad).item())
                grad_norms[name] = norm
            if not grad_norms:
                self.log(f"Gradients for {model_name}", {"info": "no grads present"})
                return
            # Top largest gradient norms
            top = sorted(grad_norms.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
            summary = {name: f"{val:.6f}" for name, val in top}
            total_norm = sum(v * v for v in grad_norms.values()) ** 0.5
            self.log(f"Grad norms for {model_name}", {"total_l2_norm": f"{total_norm:.6f}", "top": summary})
        except Exception as e:
            self.log(f"Error logging grad stats for {model_name}", {"error": f"{type(e).__name__}: {e}"}, level=logging.ERROR)

    def log_optimizer(self, optimizer: torch.optim.Optimizer):
        try:
            lr_values = [group.get("lr", None) for group in optimizer.param_groups]
            self.log("Optimizer state", {"lrs": lr_values, "num_groups": len(optimizer.param_groups)})
        except Exception as e:
            self.log("Error logging optimizer state", {"error": f"{type(e).__name__}: {e}"}, level=logging.ERROR)

    def log_scalar(self, name: str, value: Any, context: Optional[Dict[str, Any]] = None):
        ctx = {"value": value}
        if context:
            ctx.update(context)
        self.log(f"Scalar {name}", ctx)


_GLOBAL_LOGGER: Optional[NaNDebugLogger] = None


def set_nan_logger(log_file_path: str) -> NaNDebugLogger:
    global _GLOBAL_LOGGER
    _GLOBAL_LOGGER = NaNDebugLogger(log_file_path)
    return _GLOBAL_LOGGER


def get_nan_logger(log_file_path: Optional[str] = None) -> NaNDebugLogger:
    global _GLOBAL_LOGGER
    if _GLOBAL_LOGGER is None:
        if log_file_path is None:
            log_file_path = os.path.abspath("nan_debug.log")
        _GLOBAL_LOGGER = NaNDebugLogger(log_file_path)
    return _GLOBAL_LOGGER


