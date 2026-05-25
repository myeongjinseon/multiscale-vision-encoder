"""
Pretrained Vision Backbone with Intermediate Feature Extraction.

Wraps CLIP ViT, DINOv2, or standard ViT to extract features from
specified intermediate layers. This is the foundation of the multi-scale
approach — instead of using only the final representation, we tap into
the hierarchy of representations learned at different depths.

Key insight: Shallow layers capture local patterns (edges, textures),
middle layers capture parts and object regions, and deep layers capture
global semantic meaning. All of these are valuable for multimodal tasks.
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Optional, Dict
from enum import Enum


class BackboneType(Enum):
    CLIP_VIT = "clip-vit-base-patch16"
    DINOV2 = "dinov2-base"
    VIT = "vit-base-patch16"


class IntermediateFeatureExtractor(nn.Module):
    """
    Hook-based feature extractor that captures activations at specified layers.
    
    We use forward hooks rather than modifying the backbone's forward() method,
    which means the backbone remains completely untouched — important for
    preserving pretrained representations exactly.
    """

    def __init__(self):
        super().__init__()
        self._features: Dict[int, torch.Tensor] = {}
        self._hooks = []

    def register_hook(self, layer_idx: int, module: nn.Module):
        """Register a forward hook on the specified module."""
        def hook_fn(mod, input, output):
            # For transformer blocks, output is typically the hidden states
            if isinstance(output, tuple):
                self._features[layer_idx] = output[0]
            else:
                self._features[layer_idx] = output

        handle = module.register_forward_hook(hook_fn)
        self._hooks.append(handle)

    def get_features(self) -> Dict[int, torch.Tensor]:
        return self._features

    def clear(self):
        self._features.clear()

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


class VisionBackbone(nn.Module):
    """
    Unified wrapper for pretrained vision backbones.
    
    Supports:
    - CLIP ViT-B/16: Already text-aligned, strong baseline for multimodal
    - DINOv2-Base: Self-supervised, excellent intermediate features
    - ViT-B/16: Standard supervised ViT (ImageNet pretrained)
    
    Outputs:
    - List of intermediate features: [F_1, F_2, ..., F_K]
      Each F_i has shape (B, N, D_i) where N = num_patches + 1 (CLS token)
    - Final CLS token embedding for reference
    
    Args:
        backbone_type: Which pretrained model to use
        extraction_layers: List of layer indices to extract features from (1-indexed)
        freeze: Whether to freeze backbone weights
        image_size: Input image resolution
    """

    def __init__(
        self,
        backbone_type: str = "clip-vit-base-patch16",
        extraction_layers: List[int] = [3, 6, 9, 12],
        freeze: bool = True,
        image_size: int = 224,
    ):
        super().__init__()
        self.backbone_type = backbone_type
        self.extraction_layers = sorted(extraction_layers)
        self.image_size = image_size

        # Load backbone and set up extraction
        self.backbone, self.embed_dim, self.num_patches = self._load_backbone(backbone_type)
        self.extractor = IntermediateFeatureExtractor()
        self._register_extraction_hooks()

        if freeze:
            self._freeze_backbone()

        # Store per-layer dimensions (all same for ViT, may differ for others)
        self.layer_dims = [self.embed_dim] * len(extraction_layers)

    def _load_backbone(self, backbone_type: str) -> Tuple[nn.Module, int, int]:
        """Load pretrained backbone and return (model, embed_dim, num_patches)."""

        if backbone_type == "clip-vit-base-patch16":
            return self._load_clip_vit()
        elif backbone_type == "dinov2-base":
            return self._load_dinov2()
        elif backbone_type == "vit-base-patch16":
            return self._load_vit()
        else:
            raise ValueError(f"Unknown backbone: {backbone_type}")

    def _load_clip_vit(self) -> Tuple[nn.Module, int, int]:
        """Load CLIP ViT-B/16 visual encoder."""
        try:
            import open_clip
            model, _, preprocess = open_clip.create_model_and_transforms(
                "ViT-B-16", pretrained="openai"
            )
            visual = model.visual
            embed_dim = visual.conv1.out_channels if hasattr(visual, 'conv1') else 768
            num_patches = (self.image_size // 16) ** 2
            return visual, embed_dim, num_patches
        except ImportError:
            print("open_clip not available, falling back to mock backbone")
            return self._create_mock_backbone()

    def _load_dinov2(self) -> Tuple[nn.Module, int, int]:
        """Load DINOv2-Base."""
        try:
            model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
            embed_dim = 768
            num_patches = (self.image_size // 14) ** 2
            return model, embed_dim, num_patches
        except Exception:
            print("DINOv2 not available, falling back to mock backbone")
            return self._create_mock_backbone()

    def _load_vit(self) -> Tuple[nn.Module, int, int]:
        """Load standard ViT-B/16 from timm."""
        try:
            import timm
            model = timm.create_model("vit_base_patch16_224", pretrained=True)
            embed_dim = 768
            num_patches = (self.image_size // 16) ** 2
            return model, embed_dim, num_patches
        except ImportError:
            print("timm not available, falling back to mock backbone")
            return self._create_mock_backbone()

    def _create_mock_backbone(self) -> Tuple[nn.Module, int, int]:
        """
        Create a lightweight mock backbone for testing and development.
        Mimics ViT-B/16 structure with 12 transformer blocks.
        """
        embed_dim = 768
        num_patches = (self.image_size // 16) ** 2

        class MockViT(nn.Module):
            def __init__(self, dim, n_patches, n_layers=12):
                super().__init__()
                self.patch_embed = nn.Conv2d(3, dim, kernel_size=16, stride=16)
                self.cls_token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
                self.pos_embed = nn.Parameter(
                    torch.randn(1, n_patches + 1, dim) * 0.02
                )
                self.blocks = nn.ModuleList([
                    nn.TransformerEncoderLayer(
                        d_model=dim, nhead=12, dim_feedforward=dim * 4,
                        dropout=0.0, activation="gelu", batch_first=True,
                        norm_first=True,
                    )
                    for _ in range(n_layers)
                ])
                self.norm = nn.LayerNorm(dim)

            def forward(self, x):
                B = x.shape[0]
                x = self.patch_embed(x).flatten(2).transpose(1, 2)
                cls = self.cls_token.expand(B, -1, -1)
                x = torch.cat([cls, x], dim=1) + self.pos_embed
                for blk in self.blocks:
                    x = blk(x)
                x = self.norm(x)
                return x

        return MockViT(embed_dim, num_patches), embed_dim, num_patches

    def _get_transformer_blocks(self) -> nn.ModuleList:
        """Get the list of transformer blocks from the backbone."""
        backbone = self.backbone

        # Try common attribute names for the transformer blocks
        for attr in ["blocks", "resblocks", "transformer.resblocks",
                      "encoder.layers", "layers"]:
            parts = attr.split(".")
            obj = backbone
            try:
                for p in parts:
                    obj = getattr(obj, p)
                if isinstance(obj, (nn.ModuleList, nn.Sequential)):
                    return obj
            except AttributeError:
                continue

        raise AttributeError(
            f"Cannot find transformer blocks in backbone. "
            f"Available attributes: {[a for a in dir(backbone) if not a.startswith('_')]}"
        )

    def _register_extraction_hooks(self):
        """Register forward hooks at specified layers."""
        blocks = self._get_transformer_blocks()
        for layer_idx in self.extraction_layers:
            # Convert 1-indexed to 0-indexed
            block = blocks[layer_idx - 1]
            self.extractor.register_hook(layer_idx, block)

    def _freeze_backbone(self):
        """Freeze all backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        print(f"[Backbone] Frozen {sum(p.numel() for p in self.backbone.parameters()):,} parameters")

    def forward(self, images: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Extract multi-scale features from the backbone.
        
        Args:
            images: (B, 3, H, W) input images
            
        Returns:
            features: List of (B, N, D) tensors from each extraction layer
            cls_token: (B, D) final CLS token embedding
        """
        self.extractor.clear()

        # Forward pass through backbone (hooks capture intermediate features)
        with torch.set_grad_enabled(not self._is_frozen()):
            output = self.backbone(images)

        # Collect intermediate features in layer order
        raw_features = self.extractor.get_features()
        features = [raw_features[idx] for idx in self.extraction_layers]

        # Get CLS token from final layer output
        if isinstance(output, torch.Tensor):
            if output.dim() == 3:
                cls_token = output[:, 0]  # (B, D)
            else:
                cls_token = output  # Already pooled
        else:
            cls_token = features[-1][:, 0]

        return features, cls_token

    def _is_frozen(self) -> bool:
        """Check if backbone is frozen."""
        for param in self.backbone.parameters():
            return not param.requires_grad
        return False

    def get_layer_dim(self, layer_idx: int) -> int:
        """Get the embedding dimension at a specific layer."""
        idx = self.extraction_layers.index(layer_idx)
        return self.layer_dims[idx]

    @property
    def output_dim(self) -> int:
        return self.embed_dim


def build_backbone(config: dict) -> VisionBackbone:
    """Factory function to build backbone from config dict."""
    return VisionBackbone(
        backbone_type=config.get("backbone", "clip-vit-base-patch16"),
        extraction_layers=config.get("extraction_layers", [3, 6, 9, 12]),
        freeze=config.get("freeze_backbone", True),
        image_size=config.get("image_size", 224),
    )
