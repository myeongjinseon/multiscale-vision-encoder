"""
Ablation Study Runner.

Systematically runs all planned ablation experiments:
A1. Number of scales (1 → 4)
A2. Fusion method (simple_avg, concat, weighted_avg, scale_attention)
A3. Residual refinement (with / without)
A4. Backbone comparison (CLIP, DINOv2, ViT)
A5. Layer selection (which intermediate layers)

Each experiment modifies the default config, trains for a reduced number
of epochs (quick ablation), and logs results for comparison.

Usage:
    python scripts/run_ablations.py --config configs/default.yaml --task grounding
    python scripts/run_ablations.py --config configs/default.yaml --task captioning --quick
"""

import sys
import copy
import json
import argparse
import torch
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from models import build_encoder, CaptioningHead, GroundingHead, load_config
from data import build_coco_dataloader, build_refcoco_dataloader
from utils import CaptioningEvaluator, GroundingEvaluator, ExperimentLogger
from utils.visualization import plot_ablation_comparison


def modify_config(base_config: dict, overrides: dict) -> dict:
    """Create a modified config with specific overrides."""
    config = copy.deepcopy(base_config)
    for key_path, value in overrides.items():
        parts = key_path.split(".")
        obj = config
        for p in parts[:-1]:
            obj = obj.setdefault(p, {})
        obj[parts[-1]] = value
    return config


def quick_train_and_eval(config, task, device, epochs=5, max_eval_batches=30):
    """
    Quick training for ablation (fewer epochs, smaller eval).
    Returns metric dict.
    """
    # Override epochs for quick ablation
    config = copy.deepcopy(config)
    config["training"]["epochs"] = epochs
    config["training"]["eval_every"] = epochs  # Only eval at the end

    # Build encoder
    is_baseline = config.get("_is_baseline", False)
    encoder = build_encoder(config, baseline=is_baseline).to(device)

    if task == "captioning":
        head = CaptioningHead(
            visual_dim=encoder.output_dim,
            **{k: v for k, v in config.get("captioning", {}).items()
               if k in ["vocab_size", "decoder_dim", "decoder_layers",
                        "decoder_heads", "max_length", "label_smoothing"]},
        ).to(device)
        train_loader = build_coco_dataloader(config, split="train")
        val_loader = build_coco_dataloader(config, split="val")
    else:
        head = GroundingHead(
            visual_dim=encoder.output_dim,
            hidden_dim=config.get("grounding", {}).get("hidden_dim", 256),
        ).to(device)
        train_loader = build_refcoco_dataloader(config, split="train")
        val_loader = build_refcoco_dataloader(config, split="val")

    # Quick training
    trainable = [p for p in list(encoder.parameters()) + list(head.parameters())
                 if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=1e-4, weight_decay=0.01)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    for epoch in range(1, epochs + 1):
        encoder.train()
        head.train()
        for batch_idx, batch in enumerate(train_loader):
            images = batch["image"].to(device)
            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                enc_out = encoder(images)

                if task == "captioning":
                    cap_ids = batch["caption_ids"].to(device)
                    cap_mask = batch["caption_mask"].to(device)
                    out = head(enc_out["features"], cap_ids, cap_mask)
                else:
                    text_ids = batch["text_ids"].to(device)
                    text_mask = batch["text_mask"].to(device)
                    bbox = batch["bbox"].to(device)
                    out = head(enc_out["patch_features"], text_ids, text_mask, bbox)

                loss = out["loss"]

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

    # Evaluate
    encoder.eval()
    head.eval()

    with torch.no_grad():
        if task == "captioning":
            evaluator = CaptioningEvaluator()
            candidates, references = [], []
            for batch_idx, batch in enumerate(val_loader):
                if batch_idx >= max_eval_batches:
                    break
                images = batch["image"].to(device)
                enc_out = encoder(images)
                gen = head.generate(enc_out["features"], beam_size=1)
                for i in range(gen.size(0)):
                    candidates.append(" ".join(str(t) for t in gen[i].tolist() if t > 2))
                    references.append(batch["raw_captions"][i])
            metrics = evaluator.evaluate(candidates, references)
        else:
            all_preds, all_targets = [], []
            for batch_idx, batch in enumerate(val_loader):
                if batch_idx >= max_eval_batches:
                    break
                images = batch["image"].to(device)
                text_ids = batch["text_ids"].to(device)
                text_mask = batch["text_mask"].to(device)
                enc_out = encoder(images)
                out = head(enc_out["patch_features"], text_ids, text_mask)
                all_preds.append(out["pred_bbox"].cpu())
                all_targets.append(batch["bbox"])

            preds = torch.cat(all_preds)
            targets = torch.cat(all_targets)
            metrics = GroundingEvaluator.evaluate(preds, targets)

    # Add scale weight info
    with torch.no_grad():
        sample = next(iter(val_loader))
        enc_out = encoder(sample["image"].to(device))
        if enc_out.get("scale_weights") is not None:
            sw = enc_out["scale_weights"]
            mean_w = sw.mean(dim=0)
            if mean_w.dim() == 2:
                mean_w = mean_w.mean(dim=0)
            metrics["scale_weights"] = mean_w.tolist()

    return metrics


# ============================================================
#  ABLATION DEFINITIONS
# ============================================================

def get_ablation_configs(base_config: dict) -> dict:
    """Define all ablation experiments."""
    ablations = {}

    # A1: Number of scales
    for n in [1, 2, 3, 4]:
        all_layers = [3, 6, 9, 12]
        active = all_layers[-n:]
        name = f"A1_scales_{n}"
        ablations[name] = modify_config(base_config, {
            "ablation.num_active_scales": n,
            "ablation.active_layers": active,
        })

    # A2: Fusion method
    for method in ["simple_avg", "concat", "weighted_avg", "scale_attention"]:
        name = f"A2_fusion_{method}"
        ablations[name] = modify_config(base_config, {
            "ablation.fusion_method": method,
        })

    # A3: Residual refinement
    for use_refine in [False, True]:
        name = f"A3_refine_{'yes' if use_refine else 'no'}"
        ablations[name] = modify_config(base_config, {
            "ablation.use_residual_refine": use_refine,
        })

    # A4: Backbone
    for backbone in ["clip-vit-base-patch16", "dinov2-base", "vit-base-patch16"]:
        short_name = backbone.split("-")[0]
        name = f"A4_backbone_{short_name}"
        ablations[name] = modify_config(base_config, {
            "model.backbone": backbone,
        })

    # A5: Layer selection
    layer_configs = {
        "shallow_3_6": [3, 6],
        "mid_6_9": [6, 9],
        "deep_9_12": [9, 12],
        "even_3_6_9_12": [3, 6, 9, 12],
        "dense_2_4_6_8_10_12": [2, 4, 6, 8, 10, 12],
    }
    for desc, layers in layer_configs.items():
        name = f"A5_layers_{desc}"
        cfg = modify_config(base_config, {
            "model.extraction_layers": layers,
            "model.num_scales": len(layers),
            "ablation.num_active_scales": len(layers),
        })
        ablations[name] = cfg

    # Baseline (single-scale, final layer only)
    baseline_cfg = copy.deepcopy(base_config)
    baseline_cfg["_is_baseline"] = True
    ablations["baseline_single_scale"] = baseline_cfg

    return ablations


def main():
    parser = argparse.ArgumentParser(description="Run Ablation Studies")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--task", type=str, choices=["captioning", "grounding"], default="grounding")
    parser.add_argument("--quick", action="store_true", help="Use fewer epochs (3 instead of 5)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--filter", type=str, default=None, help="Run only ablations matching this prefix (e.g., 'A1')")
    args = parser.parse_args()

    base_config = load_config(args.config)
    device = torch.device(args.device)
    epochs = 3 if args.quick else 5

    # Get all ablation configs
    ablations = get_ablation_configs(base_config)

    # Filter if requested
    if args.filter:
        ablations = {k: v for k, v in ablations.items() if k.startswith(args.filter)}

    print(f"\n{'='*60}")
    print(f"Ablation Study: {len(ablations)} experiments")
    print(f"Task: {args.task} | Epochs per run: {epochs} | Device: {device}")
    print(f"{'='*60}")

    results = {}

    for i, (name, config) in enumerate(ablations.items()):
        print(f"\n[{i+1}/{len(ablations)}] Running: {name}")
        print("-" * 40)

        try:
            metrics = quick_train_and_eval(
                config, args.task, device, epochs=epochs,
            )
            results[name] = metrics

            # Print key metrics
            if args.task == "captioning":
                print(f"  CIDEr: {metrics.get('CIDEr', 0):.4f} | BLEU-4: {metrics.get('BLEU-4', 0):.4f}")
            else:
                print(f"  Acc@0.5: {metrics.get('Acc@0.5', 0):.4f} | mean_IoU: {metrics.get('mean_IoU', 0):.4f}")

            if "scale_weights" in metrics:
                print(f"  Scale weights: {[f'{w:.3f}' for w in metrics['scale_weights']]}")

        except Exception as e:
            print(f"  FAILED: {e}")
            results[name] = {"error": str(e)}

    # Save all results
    save_dir = Path(base_config.get("logging", {}).get("save_dir", "./experiments"))
    save_dir = save_dir / f"ablation_{args.task}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    save_dir.mkdir(parents=True, exist_ok=True)

    with open(save_dir / "ablation_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Summary table
    print(f"\n\n{'='*70}")
    print("ABLATION SUMMARY")
    print(f"{'='*70}")
    primary_metric = "CIDEr" if args.task == "captioning" else "Acc@0.5"

    sorted_results = sorted(
        [(k, v) for k, v in results.items() if "error" not in v],
        key=lambda x: x[1].get(primary_metric, 0),
        reverse=True,
    )

    print(f"{'Experiment':<35} {primary_metric:>10}")
    print("-" * 50)
    for name, metrics in sorted_results:
        val = metrics.get(primary_metric, 0)
        print(f"{name:<35} {val:>10.4f}")

    # Plot comparison
    clean_results = {k: v for k, v in results.items() if "error" not in v}
    if clean_results:
        plot_ablation_comparison(
            clean_results, primary_metric,
            save_path=str(save_dir / "ablation_comparison.png"),
        )

    print(f"\nResults saved to: {save_dir}")


if __name__ == "__main__":
    main()
