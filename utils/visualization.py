"""
Visualization Utilities.

Tools for analyzing and visualizing the multi-scale encoder's behavior:
1. Scale attention weight heatmaps (which layers matter per image)
2. Patch-level attention maps (where the model looks)
3. Grounding visualization (predicted vs target bbox)
4. Training curves and metric comparison plots
"""

import torch
import numpy as np
from typing import List, Dict, Optional, Tuple
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("[Viz] matplotlib not available, visualization disabled")


def plot_scale_weights(
    scale_weights: torch.Tensor,
    layer_names: List[str],
    save_path: Optional[str] = None,
    title: str = "Scale Attention Weights",
) -> Optional[np.ndarray]:
    """
    Visualize per-image scale attention weights as a heatmap.
    
    Shows how the model distributes attention across backbone depths
    for each image in the batch. Useful for understanding:
    - Do images with small objects upweight shallow layers?
    - Does the model learn task-specific scale preferences?
    
    Args:
        scale_weights: (B, K) or (B, heads, K) attention weights
        layer_names: Names for each scale (e.g., ['Layer 3', 'Layer 6', ...])
        save_path: Where to save the figure
        title: Figure title
        
    Returns:
        Image as numpy array if matplotlib available
    """
    if not HAS_MPL:
        return None

    if scale_weights.dim() == 3:
        # Multi-head: average over heads for visualization
        weights = scale_weights.mean(dim=1).cpu().numpy()
    else:
        weights = scale_weights.cpu().numpy()

    B, K = weights.shape

    fig, ax = plt.subplots(figsize=(max(6, K * 1.5), max(4, B * 0.3)))

    im = ax.imshow(weights, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)

    ax.set_xticks(range(K))
    ax.set_xticklabels(layer_names, fontsize=10)
    ax.set_ylabel("Image Index", fontsize=11)
    ax.set_xlabel("Scale (Backbone Layer)", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")

    # Add value annotations
    for i in range(B):
        for j in range(K):
            ax.text(j, i, f"{weights[i, j]:.2f}",
                    ha="center", va="center", fontsize=8,
                    color="white" if weights[i, j] > 0.5 else "black")

    plt.colorbar(im, ax=ax, label="Attention Weight")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Viz] Saved scale weights to {save_path}")

    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return img


def plot_attention_map(
    image: torch.Tensor,
    attention: torch.Tensor,
    save_path: Optional[str] = None,
    title: str = "Patch Attention",
    patch_size: int = 16,
) -> Optional[np.ndarray]:
    """
    Overlay attention map on the input image.
    
    Shows which spatial regions the model focuses on,
    useful for qualitative analysis of grounding behavior.
    
    Args:
        image: (3, H, W) input image tensor (normalized)
        attention: (N,) per-patch attention weights (N = num_patches)
        save_path: Where to save
        title: Figure title
        patch_size: Patch size used by the backbone
    """
    if not HAS_MPL:
        return None

    # Denormalize image
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = (image.cpu() * std + mean).clamp(0, 1)
    img = img.permute(1, 2, 0).numpy()

    H, W = img.shape[:2]
    h_patches = H // patch_size
    w_patches = W // patch_size

    # Reshape attention to spatial grid
    attn = attention.cpu().numpy()
    if len(attn) == h_patches * w_patches + 1:
        attn = attn[1:]  # Remove CLS token
    attn_map = attn.reshape(h_patches, w_patches)

    # Upsample to image resolution
    from PIL import Image as PILImage
    attn_resized = np.array(
        PILImage.fromarray((attn_map * 255).astype(np.uint8)).resize((W, H))
    ).astype(np.float32) / 255.0

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(img)
    axes[0].set_title("Input Image", fontsize=11)
    axes[0].axis("off")

    axes[1].imshow(attn_map, cmap="hot", interpolation="nearest")
    axes[1].set_title("Attention Map (Patches)", fontsize=11)
    axes[1].axis("off")

    axes[2].imshow(img)
    axes[2].imshow(attn_resized, cmap="jet", alpha=0.5)
    axes[2].set_title("Overlay", fontsize=11)
    axes[2].axis("off")

    plt.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    fig.canvas.draw()
    result = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    result = result.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return result


def plot_grounding_result(
    image: torch.Tensor,
    pred_bbox: torch.Tensor,
    target_bbox: torch.Tensor,
    expression: str,
    iou: float,
    save_path: Optional[str] = None,
) -> Optional[np.ndarray]:
    """
    Visualize grounding prediction vs ground truth.
    
    Args:
        image: (3, H, W) input image
        pred_bbox: (4,) predicted box [cx, cy, w, h] normalized
        target_bbox: (4,) target box [cx, cy, w, h] normalized
        expression: Referring expression text
        iou: IoU between prediction and target
        save_path: Where to save
    """
    if not HAS_MPL:
        return None

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = (image.cpu() * std + mean).clamp(0, 1)
    img = img.permute(1, 2, 0).numpy()
    H, W = img.shape[:2]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(img)

    # Draw target box (green)
    t = target_bbox.cpu().numpy()
    tx = (t[0] - t[2] / 2) * W
    ty = (t[1] - t[3] / 2) * H
    tw, th = t[2] * W, t[3] * H
    rect_gt = patches.Rectangle((tx, ty), tw, th,
                                  linewidth=3, edgecolor="lime",
                                  facecolor="none", linestyle="-",
                                  label="Ground Truth")
    ax.add_patch(rect_gt)

    # Draw prediction box (red)
    p = pred_bbox.cpu().numpy()
    px = (p[0] - p[2] / 2) * W
    py = (p[1] - p[3] / 2) * H
    pw, ph = p[2] * W, p[3] * H
    rect_pred = patches.Rectangle((px, py), pw, ph,
                                    linewidth=3, edgecolor="red",
                                    facecolor="none", linestyle="--",
                                    label="Prediction")
    ax.add_patch(rect_pred)

    ax.legend(loc="upper right", fontsize=10)
    ax.set_title(f'"{expression}"\nIoU: {iou:.3f}', fontsize=12)
    ax.axis("off")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    fig.canvas.draw()
    result = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    result = result.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return result


def plot_training_curves(
    metrics: Dict[str, List[float]],
    save_path: Optional[str] = None,
    title: str = "Training Curves",
) -> Optional[np.ndarray]:
    """
    Plot training/validation metrics over epochs.
    
    Args:
        metrics: Dict mapping metric name to list of values per epoch
        save_path: Where to save
        title: Figure title
    """
    if not HAS_MPL:
        return None

    n_metrics = len(metrics)
    cols = min(3, n_metrics)
    rows = (n_metrics + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    if n_metrics == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for i, (name, values) in enumerate(metrics.items()):
        ax = axes[i]
        epochs = range(1, len(values) + 1)
        ax.plot(epochs, values, "o-", linewidth=2, markersize=4)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(name)
        ax.set_title(name, fontweight="bold")
        ax.grid(True, alpha=0.3)

    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    fig.canvas.draw()
    result = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    result = result.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return result


def plot_ablation_comparison(
    results: Dict[str, Dict[str, float]],
    metric_name: str = "CIDEr",
    save_path: Optional[str] = None,
) -> Optional[np.ndarray]:
    """
    Bar chart comparing ablation study results.
    
    Args:
        results: {experiment_name: {metric_name: value}}
        metric_name: Which metric to plot
        save_path: Where to save
    """
    if not HAS_MPL:
        return None

    names = list(results.keys())
    values = [results[n].get(metric_name, 0) for n in names]

    fig, ax = plt.subplots(figsize=(max(6, len(names) * 1.2), 5))

    colors = plt.cm.Set2(np.linspace(0, 1, len(names)))
    bars = ax.bar(range(len(names)), values, color=colors, edgecolor="gray", linewidth=0.5)

    # Add value labels
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.2f}", ha="center", va="bottom", fontsize=10)

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel(metric_name, fontsize=11)
    ax.set_title(f"Ablation Study — {metric_name}", fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    fig.canvas.draw()
    result = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    result = result.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return result
