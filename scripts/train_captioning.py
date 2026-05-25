"""
Training Script for COCO Image Captioning.

Full training loop with:
- Mixed precision training (AMP)
- Gradient clipping
- Cosine LR schedule with warmup
- Periodic evaluation and checkpointing
- Scale weight logging for analysis

Usage:
    python scripts/train_captioning.py --config configs/default.yaml
    python scripts/train_captioning.py --config configs/default.yaml --baseline
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import build_encoder, CaptioningHead, load_config
from data import build_coco_dataloader
from utils import CaptioningEvaluator, ExperimentLogger
from utils.visualization import plot_scale_weights, plot_training_curves


def build_optimizer(params, config: dict):
    """Build optimizer from config."""
    train_cfg = config.get("training", {})
    return torch.optim.AdamW(
        params,
        lr=train_cfg.get("learning_rate", 1e-4),
        weight_decay=train_cfg.get("weight_decay", 0.01),
        betas=(train_cfg.get("beta1", 0.9), train_cfg.get("beta2", 0.999)),
    )


def build_scheduler(optimizer, config: dict, steps_per_epoch: int):
    """Build cosine LR scheduler with warmup."""
    train_cfg = config.get("training", {})
    epochs = train_cfg.get("epochs", 30)
    warmup_epochs = train_cfg.get("warmup_epochs", 5)
    min_lr = train_cfg.get("min_lr", 1e-6)

    warmup_steps = warmup_epochs * steps_per_epoch
    total_steps = epochs * steps_per_epoch

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return max(min_lr / train_cfg.get("learning_rate", 1e-4),
                    0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)).item()))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_one_epoch(
    encoder, caption_head, dataloader, optimizer, scheduler,
    scaler, device, config, logger, epoch, global_step,
):
    """Train for one epoch."""
    encoder.train()
    caption_head.train()

    train_cfg = config.get("training", {})
    clip_grad = train_cfg.get("gradient_clip", 1.0)
    accum_steps = train_cfg.get("accumulation_steps", 1)
    log_interval = config.get("logging", {}).get("log_interval", 50)
    use_amp = train_cfg.get("mixed_precision", True)

    total_loss = 0.0
    num_batches = 0

    for batch_idx, batch in enumerate(dataloader):
        images = batch["image"].to(device)
        caption_ids = batch["caption_ids"].to(device)
        caption_mask = batch["caption_mask"].to(device)

        with autocast(enabled=use_amp):
            # Encode
            enc_output = encoder(images)
            visual_features = enc_output["features"]

            # Caption loss
            cap_output = caption_head(visual_features, caption_ids, caption_mask)
            loss = cap_output["loss"] / accum_steps

        scaler.scale(loss).backward()

        if (batch_idx + 1) % accum_steps == 0:
            scaler.unscale_(optimizer)
            grad_norm = nn.utils.clip_grad_norm_(
                list(encoder.parameters()) + list(caption_head.parameters()),
                clip_grad,
            )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

        total_loss += loss.item() * accum_steps
        num_batches += 1
        global_step += 1

        if batch_idx % log_interval == 0:
            avg_loss = total_loss / num_batches
            lr = optimizer.param_groups[0]["lr"]
            logger.log_metrics(
                {"loss": avg_loss, "lr": lr, "batch": batch_idx},
                step=global_step, epoch=epoch, prefix="train/",
            )

    return total_loss / max(num_batches, 1), global_step


@torch.no_grad()
def evaluate(
    encoder, caption_head, dataloader, device,
    evaluator, config, max_batches: int = 50,
):
    """Evaluate on validation set."""
    encoder.eval()
    caption_head.eval()

    total_loss = 0.0
    num_batches = 0
    all_scale_weights = []

    candidates = []
    references = []

    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= max_batches:
            break

        images = batch["image"].to(device)
        caption_ids = batch["caption_ids"].to(device)
        caption_mask = batch["caption_mask"].to(device)

        # Encode
        enc_output = encoder(images, return_intermediate=True)
        visual_features = enc_output["features"]

        # Loss
        cap_output = caption_head(visual_features, caption_ids, caption_mask)
        total_loss += cap_output["loss"].item()
        num_batches += 1

        # Collect scale weights
        if enc_output.get("scale_weights") is not None:
            all_scale_weights.append(enc_output["scale_weights"].cpu())

        # Generate captions for metric evaluation (first few batches)
        if batch_idx < 10:
            beam_size = config.get("captioning", {}).get("beam_size", 5)
            generated = caption_head.generate(visual_features, beam_size=1)

            # Decode to text (simple: map IDs back)
            for i in range(generated.size(0)):
                gen_ids = generated[i].tolist()
                # Simple decode (in practice, use proper tokenizer)
                gen_text = " ".join(str(t) for t in gen_ids if t > 2)
                candidates.append(gen_text)
                references.append(batch["raw_captions"][i])

    metrics = {"val_loss": total_loss / max(num_batches, 1)}

    # Compute captioning metrics if we have enough samples
    if len(candidates) >= 10:
        cap_metrics = evaluator.evaluate(candidates, references)
        metrics.update(cap_metrics)

    # Aggregate scale weights
    if all_scale_weights:
        weights = torch.cat(all_scale_weights, dim=0)
        mean_weights = weights.mean(dim=0)
        if mean_weights.dim() == 2:
            mean_weights = mean_weights.mean(dim=0)
        for i, w in enumerate(mean_weights.tolist()):
            metrics[f"scale_weight_{i}"] = w

    return metrics, all_scale_weights


def main():
    parser = argparse.ArgumentParser(description="Train Captioning")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--baseline", action="store_true", help="Use baseline single-scale encoder")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--exp-name", type=str, default=None)
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Setup
    device = torch.device(args.device)
    exp_name = args.exp_name or ("captioning_baseline" if args.baseline else "captioning_multiscale")

    logger = ExperimentLogger(
        save_dir=config.get("logging", {}).get("save_dir", "./experiments"),
        experiment_name=exp_name,
        config=config,
        use_wandb=config.get("logging", {}).get("use_wandb", False),
    )

    # Build model
    encoder = build_encoder(config, baseline=args.baseline).to(device)
    caption_head = CaptioningHead(
        visual_dim=encoder.output_dim,
        **{k: v for k, v in config.get("captioning", {}).items()
           if k in ["vocab_size", "decoder_dim", "decoder_layers", "decoder_heads",
                     "max_length", "label_smoothing"]},
    ).to(device)

    # Build data
    train_loader = build_coco_dataloader(config, split="train")
    val_loader = build_coco_dataloader(config, split="val")

    # Build optimizer
    trainable_params = list(encoder.parameters()) + list(caption_head.parameters())
    trainable_params = [p for p in trainable_params if p.requires_grad]
    optimizer = build_optimizer(trainable_params, config)
    scheduler = build_scheduler(optimizer, config, len(train_loader))
    scaler = GradScaler(enabled=config.get("training", {}).get("mixed_precision", True))

    # Evaluator
    evaluator = CaptioningEvaluator()

    # Training loop
    train_cfg = config.get("training", {})
    epochs = train_cfg.get("epochs", 30)
    best_cider = 0.0
    global_step = 0
    history = {"train_loss": [], "val_loss": [], "CIDEr": []}

    print(f"\nTraining for {epochs} epochs on {device}")
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    for epoch in range(1, epochs + 1):
        logger.log_epoch_start(epoch)

        # Train
        train_loss, global_step = train_one_epoch(
            encoder, caption_head, train_loader, optimizer, scheduler,
            scaler, device, config, logger, epoch, global_step,
        )
        history["train_loss"].append(train_loss)

        # Evaluate
        if epoch % train_cfg.get("eval_every", 1) == 0:
            val_metrics, scale_weights = evaluate(
                encoder, caption_head, val_loader, device, evaluator, config,
            )
            logger.log_metrics(val_metrics, step=global_step, epoch=epoch, prefix="val/")
            history["val_loss"].append(val_metrics["val_loss"])
            history["CIDEr"].append(val_metrics.get("CIDEr", 0))

            # Save scale weight visualization
            if scale_weights:
                sw = torch.cat(scale_weights[:4], dim=0)[:16]
                layers = config.get("model", {}).get("extraction_layers", [3, 6, 9, 12])
                layer_names = [f"Layer {l}" for l in layers[-sw.shape[-1]:]]
                plot_scale_weights(
                    sw, layer_names,
                    save_path=str(logger.get_viz_dir() / f"scale_weights_ep{epoch:03d}.png"),
                )

            # Checkpoint
            is_best = val_metrics.get("CIDEr", 0) > best_cider
            if is_best:
                best_cider = val_metrics.get("CIDEr", 0)

            if epoch % train_cfg.get("save_every", 5) == 0 or is_best:
                logger.save_checkpoint(
                    {
                        "epoch": epoch,
                        "encoder": encoder.state_dict(),
                        "caption_head": caption_head.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "best_cider": best_cider,
                        "config": config,
                    },
                    epoch=epoch,
                    is_best=is_best,
                )

        logger.log_epoch_end(epoch)

    # Final plots
    plot_training_curves(history, save_path=str(logger.get_viz_dir() / "training_curves.png"))
    logger.finish()


if __name__ == "__main__":
    main()
