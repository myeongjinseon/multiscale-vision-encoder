"""
Attention Visualization Tool — Qualitative Analysis.

Standalone script for generating publication-quality visualizations
of the multi-scale encoder's behavior. Produces:

1. Per-scale attention maps: What each backbone layer "sees"
2. Fused attention maps: What the final representation focuses on
3. Scale weight distributions: How fusion weights vary across images
4. Grounding overlays: Predicted vs ground-truth bounding boxes
5. Comparative views: Multi-scale vs single-scale attention patterns

These visualizations are essential for the paper's qualitative analysis
section — they show *why* multi-scale features help, not just *that* they help.

Usage:
    python visualization/visualize_attention.py \
        --checkpoint experiments/grounding_multiscale/checkpoints/best_model.pt \
        --images path/to/images/ \
        --output experiments/visualizations/
    
    # With RefCOCO expressions:
    python visualization/visualize_attention.py \
        --checkpoint experiments/grounding_multiscale/checkpoints/best_model.pt \
        --task grounding --split val --num-samples 20
"""

import os
import sys
import argparse
import torch
import numpy as np
from pathlib import Path
from typing import List, Optional, Dict, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import matplotlib.gridspec as gridspec
    from matplotlib.colors import LinearSegmentedColormap
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

from models import build_encoder, GroundingHead, CaptioningHead, load_config
from data import build_refcoco_dataloader, build_coco_dataloader
from utils import compute_iou


# ============================================================
#  IMAGE UTILITIES
# ============================================================

def denormalize(tensor: torch.Tensor) -> np.ndarray:
    """Convert normalized image tensor to displayable numpy array."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = (tensor.cpu() * std + mean).clamp(0, 1)
    return img.permute(1, 2, 0).numpy()


def reshape_patch_attention(
    attention: torch.Tensor,
    image_size: int = 224,
    patch_size: int = 16,
    include_cls: bool = True,
) -> np.ndarray:
    """Reshape flat patch attention to 2D spatial grid."""
    h = w = image_size // patch_size
    attn = attention.cpu().numpy()
    if include_cls and len(attn) == h * w + 1:
        attn = attn[1:]  # Remove CLS
    return attn.reshape(h, w)


def upsample_attention(attn_2d: np.ndarray, target_size: int = 224) -> np.ndarray:
    """Upsample 2D attention map to image resolution."""
    from PIL import Image as PILImage
    attn_uint8 = (attn_2d * 255).clip(0, 255).astype(np.uint8)
    upsampled = np.array(
        PILImage.fromarray(attn_uint8).resize(
            (target_size, target_size), PILImage.BILINEAR
        )
    ).astype(np.float32) / 255.0
    return upsampled


# ============================================================
#  VISUALIZATION FUNCTIONS
# ============================================================

def visualize_per_scale_attention(
    image: torch.Tensor,
    raw_features: List[torch.Tensor],
    layer_names: List[str],
    save_path: Optional[str] = None,
    title: str = "Per-Scale Feature Norms",
):
    """
    Visualize what each backbone layer "sees" by plotting the
    L2 norm of patch features at each scale.
    
    High-norm regions indicate where each layer has the strongest
    activations — shallow layers highlight edges/textures, deep
    layers highlight semantically important regions.
    """
    if not HAS_MPL:
        return

    img = denormalize(image)
    K = len(raw_features)

    fig, axes = plt.subplots(1, K + 1, figsize=(4 * (K + 1), 4))

    # Original image
    axes[0].imshow(img)
    axes[0].set_title("Input", fontsize=11)
    axes[0].axis("off")

    for i, (feat, name) in enumerate(zip(raw_features, layer_names)):
        # Compute per-patch L2 norm as a proxy for activation strength
        # feat: (N, D) — skip CLS token
        patch_feat = feat[1:] if feat.shape[0] > 196 else feat
        norms = torch.norm(patch_feat, dim=-1).cpu()

        # Normalize to [0, 1]
        norms = (norms - norms.min()) / (norms.max() - norms.min() + 1e-8)
        attn_2d = reshape_patch_attention(norms, include_cls=False)
        attn_up = upsample_attention(attn_2d)

        axes[i + 1].imshow(img)
        axes[i + 1].imshow(attn_up, cmap="jet", alpha=0.55)
        axes[i + 1].set_title(name, fontsize=11)
        axes[i + 1].axis("off")

    plt.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close(fig)


def visualize_fused_vs_scales(
    image: torch.Tensor,
    raw_features: List[torch.Tensor],
    fused_features: torch.Tensor,
    layer_names: List[str],
    save_path: Optional[str] = None,
):
    """
    Compare individual scale attention with the fused result.
    Shows how the adaptive fusion combines information from all scales.
    """
    if not HAS_MPL:
        return

    img = denormalize(image)
    K = len(raw_features)

    fig = plt.figure(figsize=(4 * (K + 2), 4))
    gs = gridspec.GridSpec(1, K + 2, width_ratios=[1] * (K + 1) + [1.2])

    # Original
    ax = fig.add_subplot(gs[0])
    ax.imshow(img)
    ax.set_title("Input", fontsize=10)
    ax.axis("off")

    # Per-scale
    for i, (feat, name) in enumerate(zip(raw_features, layer_names)):
        patch_feat = feat[1:] if feat.shape[0] > 196 else feat
        norms = torch.norm(patch_feat, dim=-1).cpu()
        norms = (norms - norms.min()) / (norms.max() - norms.min() + 1e-8)
        attn_2d = reshape_patch_attention(norms, include_cls=False)
        attn_up = upsample_attention(attn_2d)

        ax = fig.add_subplot(gs[i + 1])
        ax.imshow(img)
        ax.imshow(attn_up, cmap="jet", alpha=0.55)
        ax.set_title(name, fontsize=10)
        ax.axis("off")

    # Fused
    fused_patch = fused_features[1:] if fused_features.shape[0] > 196 else fused_features
    fused_norms = torch.norm(fused_patch, dim=-1).cpu()
    fused_norms = (fused_norms - fused_norms.min()) / (fused_norms.max() - fused_norms.min() + 1e-8)
    fused_2d = reshape_patch_attention(fused_norms, include_cls=False)
    fused_up = upsample_attention(fused_2d)

    ax = fig.add_subplot(gs[K + 1])
    ax.imshow(img)
    ax.imshow(fused_up, cmap="hot", alpha=0.6)
    ax.set_title("★ Fused", fontsize=11, fontweight="bold", color="darkred")
    ax.axis("off")

    plt.suptitle("Per-Scale vs Adaptive Fusion", fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close(fig)


def visualize_scale_weight_distribution(
    scale_weights: torch.Tensor,
    layer_names: List[str],
    save_path: Optional[str] = None,
    title: str = "Scale Weight Distribution",
):
    """
    Box plot showing the distribution of scale attention weights
    across all images. Reveals systematic preferences and variance.
    """
    if not HAS_MPL:
        return

    weights = scale_weights.cpu().numpy()
    if weights.ndim == 3:
        weights = weights.mean(axis=1)  # Average over heads

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Box plot
    bp = axes[0].boxplot(
        [weights[:, i] for i in range(weights.shape[1])],
        labels=layer_names,
        patch_artist=True,
    )
    colors = plt.cm.Set2(np.linspace(0, 0.8, weights.shape[1]))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    axes[0].set_ylabel("Attention Weight", fontsize=11)
    axes[0].set_xlabel("Scale (Backbone Layer)", fontsize=11)
    axes[0].set_title("Weight Distribution", fontsize=12, fontweight="bold")
    axes[0].grid(axis="y", alpha=0.3)

    # Histogram per scale
    for i, name in enumerate(layer_names):
        axes[1].hist(
            weights[:, i], bins=30, alpha=0.6, label=name,
            edgecolor="gray", linewidth=0.5,
        )
    axes[1].set_xlabel("Weight Value", fontsize=11)
    axes[1].set_ylabel("Count", fontsize=11)
    axes[1].set_title("Weight Histograms", fontsize=12, fontweight="bold")
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.3)

    plt.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close(fig)


def visualize_grounding_gallery(
    images: List[torch.Tensor],
    pred_bboxes: List[torch.Tensor],
    target_bboxes: List[torch.Tensor],
    expressions: List[str],
    ious: List[float],
    save_path: Optional[str] = None,
    title: str = "Grounding Results Gallery",
):
    """
    Create a gallery of grounding results showing successes and failures.
    """
    if not HAS_MPL:
        return

    n = len(images)
    cols = min(4, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    if rows == 1 and cols == 1:
        axes = [[axes]]
    elif rows == 1:
        axes = [axes]
    elif cols == 1:
        axes = [[ax] for ax in axes]

    for idx in range(n):
        r, c = idx // cols, idx % cols
        ax = axes[r][c]

        img = denormalize(images[idx])
        H, W = img.shape[:2]
        ax.imshow(img)

        # Ground truth (green)
        t = target_bboxes[idx].cpu().numpy()
        rect_gt = patches.Rectangle(
            ((t[0] - t[2] / 2) * W, (t[1] - t[3] / 2) * H),
            t[2] * W, t[3] * H,
            linewidth=2.5, edgecolor="lime", facecolor="none", label="GT",
        )
        ax.add_patch(rect_gt)

        # Prediction (red/cyan based on IoU)
        p = pred_bboxes[idx].cpu().numpy()
        color = "cyan" if ious[idx] >= 0.5 else "red"
        rect_pred = patches.Rectangle(
            ((p[0] - p[2] / 2) * W, (p[1] - p[3] / 2) * H),
            p[2] * W, p[3] * H,
            linewidth=2.5, edgecolor=color, facecolor="none",
            linestyle="--", label="Pred",
        )
        ax.add_patch(rect_pred)

        # Title with expression and IoU
        status = "✓" if ious[idx] >= 0.5 else "✗"
        expr_short = expressions[idx][:40] + "..." if len(expressions[idx]) > 40 else expressions[idx]
        ax.set_title(f'{status} IoU={ious[idx]:.2f}\n"{expr_short}"', fontsize=9)
        ax.axis("off")

    # Hide empty axes
    for idx in range(n, rows * cols):
        r, c = idx // cols, idx % cols
        axes[r][c].axis("off")

    plt.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close(fig)


def visualize_multiscale_vs_baseline(
    image: torch.Tensor,
    ms_features: torch.Tensor,
    bl_features: torch.Tensor,
    ms_bbox: Optional[torch.Tensor] = None,
    bl_bbox: Optional[torch.Tensor] = None,
    gt_bbox: Optional[torch.Tensor] = None,
    expression: str = "",
    save_path: Optional[str] = None,
):
    """
    Side-by-side comparison of multi-scale vs baseline encoder attention.
    This is the key visualization for the paper — shows the concrete
    benefit of multi-scale features.
    """
    if not HAS_MPL:
        return

    img = denormalize(image)
    H, W = img.shape[:2]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Input
    axes[0].imshow(img)
    if gt_bbox is not None:
        t = gt_bbox.cpu().numpy()
        rect = patches.Rectangle(
            ((t[0] - t[2] / 2) * W, (t[1] - t[3] / 2) * H),
            t[2] * W, t[3] * H,
            linewidth=2.5, edgecolor="lime", facecolor="none",
        )
        axes[0].add_patch(rect)
    axes[0].set_title(f'Input\n"{expression[:50]}"', fontsize=10)
    axes[0].axis("off")

    # Baseline attention
    bl_patch = bl_features[1:] if bl_features.shape[0] > 196 else bl_features
    bl_norms = torch.norm(bl_patch, dim=-1).cpu()
    bl_norms = (bl_norms - bl_norms.min()) / (bl_norms.max() - bl_norms.min() + 1e-8)
    bl_2d = reshape_patch_attention(bl_norms, include_cls=False)
    bl_up = upsample_attention(bl_2d)

    axes[1].imshow(img)
    axes[1].imshow(bl_up, cmap="jet", alpha=0.55)
    if bl_bbox is not None:
        p = bl_bbox.cpu().numpy()
        rect = patches.Rectangle(
            ((p[0] - p[2] / 2) * W, (p[1] - p[3] / 2) * H),
            p[2] * W, p[3] * H,
            linewidth=2.5, edgecolor="red", facecolor="none", linestyle="--",
        )
        axes[1].add_patch(rect)
    axes[1].set_title("Baseline (Single-Scale)", fontsize=11)
    axes[1].axis("off")

    # Multi-scale attention
    ms_patch = ms_features[1:] if ms_features.shape[0] > 196 else ms_features
    ms_norms = torch.norm(ms_patch, dim=-1).cpu()
    ms_norms = (ms_norms - ms_norms.min()) / (ms_norms.max() - ms_norms.min() + 1e-8)
    ms_2d = reshape_patch_attention(ms_norms, include_cls=False)
    ms_up = upsample_attention(ms_2d)

    axes[2].imshow(img)
    axes[2].imshow(ms_up, cmap="hot", alpha=0.6)
    if ms_bbox is not None:
        p = ms_bbox.cpu().numpy()
        rect = patches.Rectangle(
            ((p[0] - p[2] / 2) * W, (p[1] - p[3] / 2) * H),
            p[2] * W, p[3] * H,
            linewidth=2.5, edgecolor="cyan", facecolor="none", linestyle="--",
        )
        axes[2].add_patch(rect)
    axes[2].set_title("★ Multi-Scale (Ours)", fontsize=11, fontweight="bold", color="darkred")
    axes[2].axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close(fig)


# ============================================================
#  MAIN: BATCH VISUALIZATION PIPELINE
# ============================================================

@torch.no_grad()
def run_visualization(args):
    """Run full visualization pipeline on validation data."""
    device = torch.device(args.device)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)
    config = ckpt["config"]

    # Build multi-scale encoder
    encoder = build_encoder(config, baseline=False).to(device)
    encoder.load_state_dict(ckpt["encoder"], strict=False)
    encoder.eval()

    # Optionally build baseline for comparison
    baseline_encoder = None
    if args.compare_baseline:
        baseline_encoder = build_encoder(config, baseline=True).to(device)
        baseline_encoder.eval()

    # Load data
    if args.task == "grounding":
        dataloader = build_refcoco_dataloader(config, split=args.split)
    else:
        dataloader = build_coco_dataloader(config, split="val")

    # Optional grounding head
    grounding_head = None
    if args.task == "grounding" and "grounding_head" in ckpt:
        grounding_head = GroundingHead(
            visual_dim=encoder.output_dim,
            hidden_dim=config.get("grounding", {}).get("hidden_dim", 256),
        ).to(device)
        grounding_head.load_state_dict(ckpt["grounding_head"])
        grounding_head.eval()

    layers = config.get("model", {}).get("extraction_layers", [3, 6, 9, 12])
    layer_names = [f"Layer {l}" for l in layers]

    all_scale_weights = []
    gallery_images, gallery_preds, gallery_targets = [], [], []
    gallery_expressions, gallery_ious = [], []

    print(f"\nGenerating visualizations for {args.num_samples} samples...")

    sample_count = 0
    for batch_idx, batch in enumerate(dataloader):
        images = batch["image"].to(device)
        B = images.shape[0]

        # Forward pass with intermediate features
        enc_output = encoder(images, return_intermediate=True)

        if enc_output.get("scale_weights") is not None:
            all_scale_weights.append(enc_output["scale_weights"].cpu())

        for i in range(B):
            if sample_count >= args.num_samples:
                break

            # 1. Per-scale attention
            if "raw_features" in enc_output:
                per_scale = [f[i] for f in enc_output["raw_features"]]
                visualize_per_scale_attention(
                    images[i], per_scale, layer_names,
                    save_path=str(output_dir / f"per_scale_{sample_count:03d}.png"),
                )

                # 2. Fused vs scales comparison
                visualize_fused_vs_scales(
                    images[i], per_scale, enc_output["features"][i],
                    layer_names,
                    save_path=str(output_dir / f"fused_vs_scales_{sample_count:03d}.png"),
                )

            # 3. Multi-scale vs baseline
            if baseline_encoder is not None:
                bl_output = baseline_encoder(images[i:i+1])

                expr = batch.get("expression", [""])[i] if "expression" in batch else ""
                visualize_multiscale_vs_baseline(
                    images[i],
                    enc_output["features"][i],
                    bl_output["features"][0],
                    expression=expr,
                    save_path=str(output_dir / f"ms_vs_baseline_{sample_count:03d}.png"),
                )

            # 4. Grounding results
            if args.task == "grounding" and grounding_head is not None:
                text_ids = batch["text_ids"][i:i+1].to(device)
                text_mask = batch["text_mask"][i:i+1].to(device)
                target = batch["bbox"][i:i+1].to(device)

                ground_out = grounding_head(
                    enc_output["patch_features"][i:i+1], text_ids, text_mask,
                )
                pred = ground_out["pred_bbox"]
                iou_val = compute_iou(pred, target).item()

                gallery_images.append(images[i].cpu())
                gallery_preds.append(pred[0].cpu())
                gallery_targets.append(target[0].cpu())
                gallery_expressions.append(batch["expression"][i])
                gallery_ious.append(iou_val)

            sample_count += 1

        if sample_count >= args.num_samples:
            break

    # 5. Scale weight distribution
    if all_scale_weights:
        combined = torch.cat(all_scale_weights, dim=0)
        visualize_scale_weight_distribution(
            combined, layer_names,
            save_path=str(output_dir / "scale_weight_distribution.png"),
        )

    # 6. Grounding gallery
    if gallery_images:
        # Sort by IoU for gallery (show best and worst)
        indices = sorted(range(len(gallery_ious)), key=lambda x: gallery_ious[x])
        worst_n = indices[:8]
        best_n = indices[-8:]

        visualize_grounding_gallery(
            [gallery_images[i] for i in best_n],
            [gallery_preds[i] for i in best_n],
            [gallery_targets[i] for i in best_n],
            [gallery_expressions[i] for i in best_n],
            [gallery_ious[i] for i in best_n],
            save_path=str(output_dir / "gallery_best.png"),
            title="Best Grounding Results (Highest IoU)",
        )

        visualize_grounding_gallery(
            [gallery_images[i] for i in worst_n],
            [gallery_preds[i] for i in worst_n],
            [gallery_targets[i] for i in worst_n],
            [gallery_expressions[i] for i in worst_n],
            [gallery_ious[i] for i in worst_n],
            save_path=str(output_dir / "gallery_worst.png"),
            title="Hardest Cases (Lowest IoU)",
        )

    print(f"\nDone! {sample_count} samples visualized → {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Attention Visualization")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--task", type=str, choices=["captioning", "grounding"], default="grounding")
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--output", type=str, default="experiments/visualizations")
    parser.add_argument("--compare-baseline", action="store_true")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    run_visualization(args)


if __name__ == "__main__":
    main()
