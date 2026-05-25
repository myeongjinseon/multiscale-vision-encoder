"""
Multi-Scale Vision Encoder — Full Pipeline.

Assembles the complete encoder from components:
    Input Image → Backbone → Feature Projection → Scale-wise Attention → Residual Refinement → Output

This is the main module that researchers interact with. It handles
configuration, forward pass orchestration, and provides methods for
analysis (scale weight extraction, feature visualization, etc.).

The encoder is designed as a drop-in replacement for standard vision
encoders in multimodal architectures. Its output is a sequence of
token embeddings (B, N, D) that can be fed to any language model
or multimodal decoder, just like CLIP or DINOv2 outputs.
"""

import torch
import torch.nn as nn
import yaml
from typing import Optional, Tuple, Dict, List
from pathlib import Path

from .backbone import VisionBackbone, build_backbone
from .feature_projection import FeatureProjection
from .scale_attention import build_fusion_module, ScaleAttentionFusion
from .residual_refine import ResidualRefinement


class MultiScaleVisionEncoder(nn.Module):
    """
    Complete multi-scale vision encoder.
    
    This encoder extracts features at multiple depths from a pretrained
    backbone and fuses them adaptively. The result is a representation
    that captures both fine-grained local detail (from shallow layers)
    and global semantic meaning (from deep layers).
    
    Compared to using only the final backbone output:
    - Better at grounding small objects and attributes
    - Better spatial relationship understanding
    - Adaptively emphasizes different scales per image
    - Minimal parameter overhead (projection + attention + refinement)
    
    Args:
        config: Configuration dictionary or path to YAML file
    """

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        model_cfg = config.get("model", config)
        ablation_cfg = config.get("ablation", {})

        # Determine active scales for ablation
        num_active = ablation_cfg.get("num_active_scales", model_cfg.get("num_scales", 4))
        all_layers = model_cfg.get("extraction_layers", [3, 6, 9, 12])
        # Use the last N layers (deeper layers first in ablation)
        self.active_layers = all_layers[-num_active:]
        self.num_scales = len(self.active_layers)
        self.hidden_dim = model_cfg.get("hidden_dim", 512)

        # 1. Backbone
        backbone_cfg = {
            "backbone": model_cfg.get("backbone", "clip-vit-base-patch16"),
            "extraction_layers": self.active_layers,
            "freeze_backbone": model_cfg.get("freeze_backbone", True),
            "image_size": config.get("data", {}).get("image_size", 224),
        }
        self.backbone = build_backbone(backbone_cfg)

        # 2. Feature Projection
        self.feature_projection = FeatureProjection(
            layer_dims=self.backbone.layer_dims[:self.num_scales],
            output_dim=self.hidden_dim,
            dropout=model_cfg.get("scale_attention", {}).get("dropout", 0.1),
        )

        # 3. Scale-wise Fusion
        fusion_method = ablation_cfg.get("fusion_method", "scale_attention")
        sa_cfg = model_cfg.get("scale_attention", {})
        self.fusion = build_fusion_module(
            method=fusion_method,
            dim=self.hidden_dim,
            num_scales=self.num_scales,
            num_heads=sa_cfg.get("num_heads", 8),
            temperature=sa_cfg.get("temperature", 1.0),
            dropout=sa_cfg.get("dropout", 0.1),
        )

        # 4. Residual Refinement
        self.use_refinement = ablation_cfg.get("use_residual_refine", True)
        if self.use_refinement:
            rr_cfg = model_cfg.get("residual_refine", {})
            self.refinement = ResidualRefinement(
                dim=self.hidden_dim,
                num_blocks=rr_cfg.get("num_blocks", 2),
                ffn_ratio=rr_cfg.get("ffn_ratio", 4),
                num_heads=sa_cfg.get("num_heads", 8),
                dropout=rr_cfg.get("dropout", 0.1),
                use_cross_attention=rr_cfg.get("use_cross_attention", False),
            )

        self._print_summary()

    def _print_summary(self):
        """Print model configuration summary."""
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        print(f"\n{'='*60}")
        print(f"Multi-Scale Vision Encoder")
        print(f"{'='*60}")
        print(f"  Backbone:       {self.config.get('model', {}).get('backbone', 'N/A')}")
        print(f"  Active layers:  {self.active_layers}")
        print(f"  Hidden dim:     {self.hidden_dim}")
        print(f"  Fusion method:  {self.config.get('ablation', {}).get('fusion_method', 'scale_attention')}")
        print(f"  Refinement:     {'Yes' if self.use_refinement else 'No'}")
        print(f"  Trainable:      {trainable:,} params")
        print(f"  Frozen:         {frozen:,} params")
        print(f"  Total:          {trainable + frozen:,} params")
        print(f"{'='*60}\n")

    def forward(
        self,
        images: torch.Tensor,
        return_intermediate: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass through the multi-scale encoder.
        
        Args:
            images: (B, 3, H, W) input images
            return_intermediate: If True, return intermediate representations
                                 for visualization and analysis
                                 
        Returns:
            Dictionary containing:
            - 'features': (B, N, D) final multi-scale representation
            - 'cls_token': (B, D) final CLS token
            - 'scale_weights': (B, K) attention weights per scale (if available)
            - 'intermediate': dict of intermediate features (if requested)
        """
        output = {}

        # 1. Extract multi-scale features from backbone
        raw_features, backbone_cls = self.backbone(images)
        # raw_features: List of (B, N, D_i), one per active layer

        if return_intermediate:
            output["raw_features"] = [f.detach() for f in raw_features]

        # 2. Project all scales to unified dimension
        projected = self.feature_projection(raw_features)
        # projected: (B, K, N, D)

        if return_intermediate:
            output["projected_features"] = projected.detach()

        # 3. Fuse scales via attention
        fused, scale_weights = self.fusion(projected, return_weights=True)
        # fused: (B, N, D), scale_weights: (B, K) or (B, heads, K)

        output["scale_weights"] = scale_weights

        if return_intermediate:
            output["pre_refinement"] = fused.detach()

        # 4. Refine fused representation
        if self.use_refinement:
            # Use the deepest projected features as anchor for optional cross-attn
            deep_anchor = projected[:, -1]  # (B, N, D) — deepest scale
            refined = self.refinement(fused, deep_anchor)
        else:
            refined = fused

        # Output
        output["features"] = refined                # (B, N, D) full sequence
        output["cls_token"] = refined[:, 0]         # (B, D) CLS token
        output["patch_features"] = refined[:, 1:]   # (B, N-1, D) patch tokens

        return output

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """Simple interface: returns just the feature sequence."""
        return self.forward(images)["features"]

    def get_scale_weights(self, images: torch.Tensor) -> torch.Tensor:
        """Get per-image scale attention weights for analysis."""
        output = self.forward(images)
        return output.get("scale_weights")

    @property
    def output_dim(self) -> int:
        return self.hidden_dim

    @property
    def num_patches(self) -> int:
        return self.backbone.num_patches


class BaselineEncoder(nn.Module):
    """
    Baseline: Standard single-scale encoder using only final backbone output.
    
    This serves as the comparison point. It takes only the final layer's
    representation from the backbone and optionally projects it to the
    target dimension. No multi-scale fusion, no refinement.
    """

    def __init__(self, config: dict):
        super().__init__()
        model_cfg = config.get("model", config)
        self.hidden_dim = model_cfg.get("hidden_dim", 512)

        backbone_cfg = {
            "backbone": model_cfg.get("backbone", "clip-vit-base-patch16"),
            "extraction_layers": [model_cfg.get("extraction_layers", [3, 6, 9, 12])[-1]],
            "freeze_backbone": model_cfg.get("freeze_backbone", True),
            "image_size": config.get("data", {}).get("image_size", 224),
        }
        self.backbone = build_backbone(backbone_cfg)

        # Simple projection from backbone dim to hidden dim
        self.proj = nn.Sequential(
            nn.Linear(self.backbone.embed_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
        )

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[Baseline Encoder] Trainable: {trainable:,} params")

    def forward(
        self, images: torch.Tensor, return_intermediate: bool = False
    ) -> Dict[str, torch.Tensor]:
        features, cls_token = self.backbone(images)
        final_features = features[-1]  # Only the last layer
        projected = self.proj(final_features)

        return {
            "features": projected,
            "cls_token": projected[:, 0],
            "patch_features": projected[:, 1:],
            "scale_weights": None,
        }

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        return self.forward(images)["features"]

    @property
    def output_dim(self) -> int:
        return self.hidden_dim


def build_encoder(config: dict, baseline: bool = False) -> nn.Module:
    """Build encoder from config."""
    if baseline:
        return BaselineEncoder(config)
    return MultiScaleVisionEncoder(config)


def load_config(path: str) -> dict:
    """Load YAML configuration file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)
