"""
Evaluation Script.

Loads a trained checkpoint and evaluates on validation/test splits.
Supports both captioning (COCO) and grounding (RefCOCO) tasks.

Usage:
    python scripts/evaluate.py --checkpoint experiments/captioning_multiscale/checkpoints/best_model.pt --task captioning
    python scripts/evaluate.py --checkpoint experiments/grounding_multiscale/checkpoints/best_model.pt --task grounding
"""

import sys
import argparse
import torch
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from models import build_encoder, CaptioningHead, GroundingHead
from data import build_coco_dataloader, build_refcoco_dataloader
from utils import CaptioningEvaluator, GroundingEvaluator, compute_iou
from utils.visualization import (
    plot_scale_weights, plot_grounding_result, plot_ablation_comparison,
)


def evaluate_captioning(encoder, caption_head, dataloader, device, save_dir):
    """Full captioning evaluation."""
    encoder.eval()
    caption_head.eval()
    evaluator = CaptioningEvaluator()

    candidates = []
    references = []
    all_scale_weights = []

    print("Running captioning evaluation...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            images = batch["image"].to(device)

            enc_output = encoder(images, return_intermediate=True)
            visual_features = enc_output["features"]

            # Generate
            generated = caption_head.generate(visual_features, beam_size=5)

            for i in range(generated.size(0)):
                gen_ids = generated[i].tolist()
                gen_text = " ".join(str(t) for t in gen_ids if t > 2)
                candidates.append(gen_text)
                references.append(batch["raw_captions"][i])

            if enc_output.get("scale_weights") is not None:
                all_scale_weights.append(enc_output["scale_weights"].cpu())

            if (batch_idx + 1) % 20 == 0:
                print(f"  Processed {batch_idx + 1} batches...")

    # Compute metrics
    metrics = evaluator.evaluate(candidates, references)

    print("\n" + "=" * 50)
    print("Captioning Results")
    print("=" * 50)
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    # Save results
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / "captioning_results.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Scale weight analysis
    if all_scale_weights:
        weights = torch.cat(all_scale_weights, dim=0)
        mean_w = weights.mean(dim=0)
        if mean_w.dim() == 2:
            mean_w = mean_w.mean(dim=0)
        print(f"\n  Mean scale weights: {mean_w.tolist()}")

        plot_scale_weights(
            weights[:32],
            [f"Layer {l}" for l in [3, 6, 9, 12][:weights.shape[-1]]],
            save_path=str(save_dir / "eval_scale_weights.png"),
        )

    return metrics


def evaluate_grounding(encoder, grounding_head, dataloader, device, save_dir):
    """Full grounding evaluation."""
    encoder.eval()
    grounding_head.eval()

    all_preds = []
    all_targets = []
    all_scale_weights = []

    print("Running grounding evaluation...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
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
            )

            all_preds.append(ground_output["pred_bbox"].cpu())
            all_targets.append(target_bbox.cpu())

            if enc_output.get("scale_weights") is not None:
                all_scale_weights.append(enc_output["scale_weights"].cpu())

            # Save some visualizations
            if batch_idx < 3:
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
                        save_path=str(Path(save_dir) / f"eval_grounding_{batch_idx}_{i}.png"),
                    )

            if (batch_idx + 1) % 20 == 0:
                print(f"  Processed {batch_idx + 1} batches...")

    # Compute metrics
    preds = torch.cat(all_preds, dim=0)
    targets = torch.cat(all_targets, dim=0)

    metrics = GroundingEvaluator.evaluate(preds, targets)

    print("\n" + "=" * 50)
    print("Grounding Results")
    print("=" * 50)
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    # Save
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / "grounding_results.json", "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--task", type=str, choices=["captioning", "grounding"], required=True)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save-dir", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(args.device)

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)
    config = ckpt["config"]

    # Determine if baseline
    is_baseline = "baseline" in args.checkpoint.lower()

    # Build encoder
    encoder = build_encoder(config, baseline=is_baseline).to(device)
    encoder.load_state_dict(ckpt["encoder"])

    save_dir = args.save_dir or str(Path(args.checkpoint).parent.parent / "eval_results")

    if args.task == "captioning":
        caption_head = CaptioningHead(
            visual_dim=encoder.output_dim,
            **{k: v for k, v in config.get("captioning", {}).items()
               if k in ["vocab_size", "decoder_dim", "decoder_layers",
                        "decoder_heads", "max_length"]},
        ).to(device)
        caption_head.load_state_dict(ckpt["caption_head"])

        val_loader = build_coco_dataloader(config, split="val")
        evaluate_captioning(encoder, caption_head, val_loader, device, save_dir)

    elif args.task == "grounding":
        grounding_head = GroundingHead(
            visual_dim=encoder.output_dim,
            hidden_dim=config.get("grounding", {}).get("hidden_dim", 256),
            num_layers=config.get("grounding", {}).get("num_layers", 2),
        ).to(device)
        grounding_head.load_state_dict(ckpt["grounding_head"])

        val_loader = build_refcoco_dataloader(config, split="val")
        evaluate_grounding(encoder, grounding_head, val_loader, device, save_dir)


if __name__ == "__main__":
    main()
