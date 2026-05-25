"""
Training Script for RefCOCO Visual Grounding.

Trains the multi-scale encoder + grounding head on RefCOCO.
Evaluates with Acc@0.5IoU — the primary metric for showing that
multi-scale features improve fine-grained grounding.

Usage:
    python scripts/train_grounding.py --config configs/default.yaml
    python scripts/train_grounding.py --config configs/default.yaml --baseline
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from models import build_encoder, GroundingHead, load_config
from data import build_refcoco_dataloader
from utils import GroundingEvaluator, ExperimentLogger, compute_iou
from utils.visualization import plot_scale_weights, plot_training_curves, plot_grounding_result


def train_one_epoch(
    encoder, grounding_head, dataloader, optimizer, scheduler,
    scaler, device, config, logger, epoch, global_step,
):
    """Train for one epoch on RefCOCO."""
    encoder.train()
    grounding_head.train()

    train_cfg = config.get("training", {})
    clip_grad = train_cfg.get("gradient_clip", 1.0)
    log_interval = config.get("logging", {}).get("log_interval", 50)
    use_amp = train_cfg.get("mixed_precision", True)

    total_loss = 0.0
    total_l1 = 0.0
    total_giou = 0.0
    num_batches = 0

    for batch_idx, batch in enumerate(dataloader):
        images = batch["image"].to(device)
        text_ids = batch["text_ids"].to(device)
        text_mask = batch["text_mask"].to(device)
        target_bbox = batch["bbox"].to(device)

        optimizer.zero_grad()

        with autocast(enabled=use_amp):
            # Encode image
            enc_output = encoder(images)
            # Use patch features (exclude CLS) for spatial grounding
            patch_features = enc_output["patch_features"]

            # Grounding
            ground_output = grounding_head(
                visual_features=patch_features,
                text_ids=text_ids,
                text_mask=text_mask,
                target_bbox=target_bbox,
            )
            loss = ground_output["loss"]

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(
            list(encoder.parameters()) + list(grounding_head.parameters()),
            clip_grad,
        )
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += loss.item()
        total_l1 += ground_output["l1_loss"].item()
        total_giou += ground_output["giou_loss"].item()
        num_batches += 1
        global_step += 1

        if batch_idx % log_interval == 0:
            logger.log_metrics(
                {
                    "loss": total_loss / num_batches,
                    "l1_loss": total_l1 / num_batches,
                    "giou_loss": total_giou / num_batches,
                    "lr": optimizer.param_groups[0]["lr"],
                },
                step=global_step, epoch=epoch, prefix="train/",
            )

    avg_loss = total_loss / max(num_batches, 1)
    return avg_loss, global_step


@torch.no_grad()
def evaluate(
    encoder, grounding_head, dataloader, device,
    config, logger=None, epoch=0, max_batches: int = 100,
    save_viz: bool = False,
):
    """Evaluate on RefCOCO validation set."""
    encoder.eval()
    grounding_head.eval()

    all_pred_bboxes = []
    all_target_bboxes = []
    total_loss = 0.0
    num_batches = 0
    all_scale_weights = []

    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= max_batches:
            break

        images = batch["image"].to(device)
        text_ids = batch["text_ids"].to(device)
        text_mask = batch["text_mask"].to(device)
        target_bbox = batch["bbox"].to(device)

        enc_output = encoder(images, return_intermediate=True)
        patch_features = enc_output["patch_features"]

        ground_output = grounding_head(
            visual_features=patch_features,
            text_ids=text_ids,
            text_mask=text_mask,
            target_bbox=target_bbox,
        )

        total_loss += ground_output["loss"].item()
        num_batches += 1

        all_pred_bboxes.append(ground_output["pred_bbox"].cpu())
        all_target_bboxes.append(target_bbox.cpu())

        if enc_output.get("scale_weights") is not None:
            all_scale_weights.append(enc_output["scale_weights"].cpu())

        # Save grounding visualizations for first batch
        if save_viz and batch_idx == 0 and logger:
            for i in range(min(4, images.size(0))):
                iou_val = compute_iou(
                    ground_output["pred_bbox"][i:i+1],
                    target_bbox[i:i+1],
                ).item()
                plot_grounding_result(
                    images[i].cpu(),
                    ground_output["pred_bbox"][i].cpu(),
                    target_bbox[i].cpu(),
                    batch["expression"][i],
                    iou_val,
                    save_path=str(logger.get_viz_dir() / f"grounding_ep{epoch:03d}_{i}.png"),
                )

    # Compute metrics
    preds = torch.cat(all_pred_bboxes, dim=0)
    targets = torch.cat(all_target_bboxes, dim=0)

    metrics = GroundingEvaluator.evaluate(preds, targets)
    metrics["val_loss"] = total_loss / max(num_batches, 1)

    # Scale weight stats
    if all_scale_weights:
        weights = torch.cat(all_scale_weights, dim=0)
        mean_w = weights.mean(dim=0)
        if mean_w.dim() == 2:
            mean_w = mean_w.mean(dim=0)
        for i, w in enumerate(mean_w.tolist()):
            metrics[f"scale_weight_{i}"] = w

    return metrics, all_scale_weights


def main():
    parser = argparse.ArgumentParser(description="Train Grounding")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--exp-name", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device(args.device)
    exp_name = args.exp_name or ("grounding_baseline" if args.baseline else "grounding_multiscale")

    logger = ExperimentLogger(
        save_dir=config.get("logging", {}).get("save_dir", "./experiments"),
        experiment_name=exp_name,
        config=config,
        use_wandb=config.get("logging", {}).get("use_wandb", False),
    )

    # Build model
    encoder = build_encoder(config, baseline=args.baseline).to(device)
    grounding_head = GroundingHead(
        visual_dim=encoder.output_dim,
        hidden_dim=config.get("grounding", {}).get("hidden_dim", 256),
        num_layers=config.get("grounding", {}).get("num_layers", 2),
        bbox_loss_weight=config.get("grounding", {}).get("bbox_loss_weight", 5.0),
        giou_loss_weight=config.get("grounding", {}).get("giou_loss_weight", 2.0),
    ).to(device)

    # Data
    train_loader = build_refcoco_dataloader(config, split="train")
    val_loader = build_refcoco_dataloader(config, split="val")

    # Optimizer
    trainable_params = [p for p in list(encoder.parameters()) + list(grounding_head.parameters())
                        if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=config.get("training", {}).get("learning_rate", 1e-4),
        weight_decay=config.get("training", {}).get("weight_decay", 0.01),
    )

    train_cfg = config.get("training", {})
    total_steps = train_cfg.get("epochs", 30) * len(train_loader)
    warmup_steps = train_cfg.get("warmup_epochs", 5) * len(train_loader)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=train_cfg.get("learning_rate", 1e-4),
        total_steps=total_steps,
        pct_start=warmup_steps / total_steps,
        anneal_strategy="cos",
    )

    scaler = GradScaler(enabled=train_cfg.get("mixed_precision", True))

    # Training loop
    epochs = train_cfg.get("epochs", 30)
    best_acc = 0.0
    global_step = 0
    history = {"train_loss": [], "val_loss": [], "Acc@0.5": [], "mean_IoU": []}

    print(f"\nTraining grounding for {epochs} epochs on {device}")

    for epoch in range(1, epochs + 1):
        logger.log_epoch_start(epoch)

        train_loss, global_step = train_one_epoch(
            encoder, grounding_head, train_loader, optimizer, scheduler,
            scaler, device, config, logger, epoch, global_step,
        )
        history["train_loss"].append(train_loss)

        # Evaluate
        if epoch % train_cfg.get("eval_every", 1) == 0:
            val_metrics, scale_weights = evaluate(
                encoder, grounding_head, val_loader, device,
                config, logger, epoch, save_viz=(epoch % 5 == 0),
            )
            logger.log_metrics(val_metrics, step=global_step, epoch=epoch, prefix="val/")
            history["val_loss"].append(val_metrics["val_loss"])
            history["Acc@0.5"].append(val_metrics.get("Acc@0.5", 0))
            history["mean_IoU"].append(val_metrics.get("mean_IoU", 0))

            # Save scale weights
            if scale_weights:
                sw = torch.cat(scale_weights[:4], dim=0)[:16]
                layers = config.get("model", {}).get("extraction_layers", [3, 6, 9, 12])
                layer_names = [f"Layer {l}" for l in layers[-sw.shape[-1]:]]
                plot_scale_weights(
                    sw, layer_names,
                    save_path=str(logger.get_viz_dir() / f"scale_weights_ep{epoch:03d}.png"),
                )

            # Checkpoint
            is_best = val_metrics.get("Acc@0.5", 0) > best_acc
            if is_best:
                best_acc = val_metrics.get("Acc@0.5", 0)

            if epoch % train_cfg.get("save_every", 5) == 0 or is_best:
                logger.save_checkpoint(
                    {
                        "epoch": epoch,
                        "encoder": encoder.state_dict(),
                        "grounding_head": grounding_head.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "best_acc": best_acc,
                        "config": config,
                    },
                    epoch=epoch,
                    is_best=is_best,
                )

        logger.log_epoch_end(epoch)

    plot_training_curves(history, save_path=str(logger.get_viz_dir() / "training_curves.png"))
    logger.finish()

    print(f"\nBest Acc@0.5: {best_acc:.4f}")


if __name__ == "__main__":
    main()
