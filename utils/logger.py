"""
Experiment Logger.

Handles logging of training metrics, hyperparameters, and checkpoints.
Supports CSV logging (always) and optional WandB integration.
"""

import os
import csv
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List


class ExperimentLogger:
    """
    Unified experiment logger.
    
    Logs to:
    1. Console (always)
    2. CSV file (always) — easy to load in pandas
    3. JSON config dump (always) — full experiment reproducibility
    4. WandB (optional) — for real-time monitoring
    
    Args:
        save_dir: Root directory for experiment outputs
        experiment_name: Name for this experiment run
        config: Full experiment configuration dict
        use_wandb: Whether to enable WandB logging
    """

    def __init__(
        self,
        save_dir: str = "./experiments",
        experiment_name: Optional[str] = None,
        config: Optional[Dict] = None,
        use_wandb: bool = False,
    ):
        # Create experiment directory
        if experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_name = f"exp_{timestamp}"

        self.exp_dir = Path(save_dir) / experiment_name
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.exp_dir / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.viz_dir = self.exp_dir / "visualizations"
        self.viz_dir.mkdir(exist_ok=True)

        # Save config
        self.config = config or {}
        with open(self.exp_dir / "config.json", "w") as f:
            json.dump(self.config, f, indent=2, default=str)

        # CSV logger
        self.csv_path = self.exp_dir / "metrics.csv"
        self._csv_header_written = False
        self._all_metrics: List[Dict] = []

        # Timing
        self._start_time = time.time()
        self._epoch_start = None

        # WandB
        self.use_wandb = use_wandb
        self._wandb_run = None
        if use_wandb:
            self._init_wandb(experiment_name)

        print(f"\n[Logger] Experiment: {experiment_name}")
        print(f"[Logger] Save dir:   {self.exp_dir}")

    def _init_wandb(self, name: str):
        """Initialize WandB run."""
        try:
            import wandb
            project = self.config.get("logging", {}).get(
                "project_name", "multiscale-vision-encoder"
            )
            self._wandb_run = wandb.init(
                project=project,
                name=name,
                config=self.config,
            )
        except ImportError:
            print("[Logger] wandb not installed, falling back to CSV only")
            self.use_wandb = False

    def log_epoch_start(self, epoch: int):
        """Mark the start of an epoch for timing."""
        self._epoch_start = time.time()
        print(f"\n{'='*50}")
        print(f"Epoch {epoch}")
        print(f"{'='*50}")

    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
        epoch: Optional[int] = None,
        prefix: str = "",
    ):
        """
        Log metrics to all backends.
        
        Args:
            metrics: Dict of metric name → value
            step: Global step number
            epoch: Current epoch
            prefix: Prefix for metric names (e.g., 'train/', 'val/')
        """
        # Add prefix
        prefixed = {f"{prefix}{k}": v for k, v in metrics.items()}

        # Add metadata
        row = {"epoch": epoch, "step": step, "timestamp": time.time() - self._start_time}
        row.update(prefixed)

        # Console
        parts = [f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}"
                 for k, v in prefixed.items()]
        prefix_str = f"[{prefix.rstrip('/')}] " if prefix else ""
        print(f"  {prefix_str}{' | '.join(parts)}")

        # CSV
        self._log_csv(row)

        # WandB
        if self.use_wandb and self._wandb_run:
            import wandb
            wandb.log(prefixed, step=step)

        self._all_metrics.append(row)

    def _log_csv(self, row: Dict):
        """Append a row to the CSV log."""
        mode = "a" if self._csv_header_written else "w"
        with open(self.csv_path, mode, newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if not self._csv_header_written:
                writer.writeheader()
                self._csv_header_written = True
            writer.writerow(row)

    def log_epoch_end(self, epoch: int):
        """Log epoch completion with timing."""
        if self._epoch_start:
            elapsed = time.time() - self._epoch_start
            print(f"  Epoch {epoch} completed in {elapsed:.1f}s")

    def save_checkpoint(
        self,
        state: Dict[str, Any],
        epoch: int,
        is_best: bool = False,
        filename: Optional[str] = None,
    ):
        """Save model checkpoint."""
        import torch

        if filename is None:
            filename = f"checkpoint_epoch{epoch:03d}.pt"

        path = self.checkpoint_dir / filename
        torch.save(state, path)
        print(f"  [Checkpoint] Saved: {path}")

        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(state, best_path)
            print(f"  [Checkpoint] New best model saved!")

    def get_metrics_history(self) -> List[Dict]:
        """Return all logged metrics."""
        return self._all_metrics

    def get_viz_dir(self) -> Path:
        """Return visualization directory path."""
        return self.viz_dir

    def finish(self):
        """Finalize logging."""
        total_time = time.time() - self._start_time
        print(f"\n{'='*50}")
        print(f"Experiment completed in {total_time:.1f}s ({total_time/3600:.2f}h)")
        print(f"Results saved to: {self.exp_dir}")
        print(f"{'='*50}")

        if self.use_wandb and self._wandb_run:
            import wandb
            wandb.finish()
