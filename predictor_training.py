import os
import time
from typing import Optional, Tuple, Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class PredictorTrainer:
    def __init__(
        self,
        num_epochs: int = 100,
        log_interval: int = 10,
        save_interval: int = 20,
        eval_interval: int = 5,
        early_stopping: bool = True,
        patience: int = 10,
        use_tqdm: bool = False,
    ) -> None:
        self.num_epochs = num_epochs
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.eval_interval = eval_interval
        self.early_stopping = early_stopping
        self.patience = patience
        self.use_tqdm = use_tqdm

        self.best_loss = float("inf")
        self.no_improve_count = 0

    def _compute_batch_predictor_loss(
        self,
        encoder: nn.Module,
        predictor: nn.Module,
        batch: Dict,
        device: torch.device,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        losses: Dict[str, torch.Tensor] = {}

        if isinstance(batch["source_samples"], torch.Tensor):
            source_samples = batch["source_samples"].to(device)
            target_samples = batch["target_samples"].to(device)

            source_latent = encoder(source_samples)
            target_latent = encoder(target_samples)

            if getattr(predictor, "requires_condition", False):
                condition_scalars = (
                    batch["source_idx"].to(device),
                    batch["target_idx"].to(device),
                )
                pred_loss, _ = predictor.loss(
                    source_latent, target_latent, condition_scalars
                )
            else:
                pred_loss, _ = predictor.loss(source_latent, target_latent)

        else:
            source_samples = {}
            target_samples = {}

            for key, value in batch["source_samples"].items():
                source_samples[key] = value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in batch["target_samples"].items():
                target_samples[key] = value.to(device) if isinstance(value, torch.Tensor) else value

            source_latent = encoder(source_samples)
            target_latent = encoder(target_samples)

            if getattr(predictor, "requires_condition", False):
                condition_scalars = (
                    batch["source_idx"].to(device),
                    batch["target_idx"].to(device),
                )
                pred_loss, _ = predictor.loss(
                    source_latent, target_latent, condition_scalars
                )
            else:
                pred_loss, _ = predictor.loss(source_latent, target_latent)

        losses["predictor_loss"] = pred_loss
        return pred_loss, losses

    def _evaluate(
        self,
        encoder: nn.Module,
        predictor: nn.Module,
        dataloader: DataLoader,
        device: torch.device,
    ) -> float:
        encoder.eval()
        predictor.eval()

        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in dataloader:
                pred_loss, _ = self._compute_batch_predictor_loss(
                    encoder, predictor, batch, device
                )
                total_loss += float(pred_loss.item())
                num_batches += 1

        return total_loss / max(1, num_batches)

    def train(
        self,
        encoder: nn.Module,
        predictor: nn.Module,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        device: Optional[torch.device] = None,
        output_dir: str = "./outputs",
    ) -> Tuple[str, Dict]:
        start_time = time.time()
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        os.makedirs(output_dir, exist_ok=True)

        encoder.to(device)
        predictor.to(device)
        encoder.eval()
        for param in encoder.parameters():
            param.requires_grad = False

        stats = {
            "train_losses": [],
            "eval_losses": [],
            "best_epoch": 0,
            "total_time": 0.0,
        }

        try:
            for epoch in range(self.num_epochs):
                predictor.train()

                epoch_losses = []
                if self.use_tqdm:
                    try:
                        from tqdm import tqdm  # local import to avoid hard dep
                        iterator = tqdm(dataloader, desc=f"Predictor Epoch {epoch+1}/{self.num_epochs}")
                    except Exception:
                        iterator = dataloader
                else:
                    iterator = dataloader

                for batch_index, batch in enumerate(iterator):
                    optimizer.zero_grad()

                    pred_loss, losses = self._compute_batch_predictor_loss(
                        encoder, predictor, batch, device
                    )

                    pred_loss.backward()
                    optimizer.step()

                    loss_value = float(pred_loss.item())
                    epoch_losses.append(loss_value)

                    if batch_index % self.log_interval == 0:
                        # Keep logging simple: stdout only
                        print(
                            f"[Predictor][Epoch {epoch+1}][Batch {batch_index}/{len(dataloader)}] loss={loss_value:.6f}",
                            flush=True,
                        )

                avg_epoch_loss = sum(epoch_losses) / max(1, len(epoch_losses))
                stats["train_losses"].append(avg_epoch_loss)
                print(f"[Predictor] Epoch {epoch+1} complete. Avg loss={avg_epoch_loss:.6f}", flush=True)

                # Save checkpoints periodically
                if (epoch + 1) % self.save_interval == 0:
                    ckpt_path = os.path.join(output_dir, f"predictor_checkpoint_epoch_{epoch+1}.pt")
                    torch.save(
                        {
                            "epoch": epoch + 1,
                            "predictor_state_dict": predictor.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "loss": avg_epoch_loss,
                        },
                        ckpt_path,
                    )
                    print(f"[Predictor] Saved checkpoint to {ckpt_path}", flush=True)

                # Evaluate and early stop
                if ((epoch + 1) % self.eval_interval == 0) or ((epoch + 1) == self.num_epochs):
                    eval_loss = self._evaluate(encoder, predictor, dataloader, device)
                    stats["eval_losses"].append(eval_loss)
                    print(f"[Predictor] Eval loss={eval_loss:.6f}", flush=True)

                    # Track best and save best model
                    if eval_loss < self.best_loss:
                        self.best_loss = eval_loss
                        stats["best_epoch"] = epoch + 1
                        best_model_path = os.path.join(output_dir, "predictor_best_model.pt")
                        torch.save(
                            {
                                "epoch": epoch + 1,
                                "predictor_state_dict": predictor.state_dict(),
                                "optimizer_state_dict": optimizer.state_dict(),
                                "loss": eval_loss,
                            },
                            best_model_path,
                        )
                        print(f"[Predictor] New best model saved to {best_model_path}", flush=True)
                        self.no_improve_count = 0
                    else:
                        self.no_improve_count += 1
                        print(
                            f"[Predictor] No improvement for {self.no_improve_count} evals",
                            flush=True,
                        )

                    if self.early_stopping and self.no_improve_count >= self.patience:
                        print(
                            f"[Predictor] Early stopping after epoch {epoch+1}",
                            flush=True,
                        )
                        break

        finally:
            stats["total_time"] = time.time() - start_time
            print(f"[Predictor] Training completed in {stats['total_time']:.2f}s", flush=True)

        return output_dir, stats


