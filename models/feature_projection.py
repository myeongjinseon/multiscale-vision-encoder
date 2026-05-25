"""
Feature Projection Module.

Each intermediate layer of the backbone may have different dimensions
(especially for hierarchical backbones like Swin). Even for ViT where
all layers share the same dimension, a learned projection allows the
model to transform each scale's features into a representation space
optimized for fusion.

The projection also serves as a "bottleneck" that can compress features
if the backbone dim is larger than our target fusion dim D, reducing
computational cost in downstream modules.
"""

import torch
import torch.nn as nn
from typing import List


class SingleScaleProjection(nn.Module):
    """
    Project features from one scale to the unified dimension.
    
    Architecture: Linear → LayerNorm → GELU → Linear (with residual if dims match)
    
    The two-layer design with nonlinearity is important: a single linear
    projection would limit the model to affine transformations of the
    original features. The nonlinearity allows learning more complex
    mappings that can, for example, suppress noisy dimensions while
    amplifying informative ones.
    """

    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        self.use_residual = (input_dim == output_dim)

        self.proj = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

        # Initialize close to identity when possible
        if self.use_residual:
            nn.init.zeros_(self.proj[-2].weight)  # Second linear
            nn.init.zeros_(self.proj[-2].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, D_in) features from one backbone layer
        Returns:
            (B, N, D_out) projected features
        """
        projected = self.proj(x)
        if self.use_residual:
            return x + projected
        return projected


class FeatureProjection(nn.Module):
    """
    Multi-scale feature projection module.
    
    Takes features from K different backbone layers (possibly different dims)
    and projects each to a common dimension D. This alignment is necessary
    before any fusion operation.
    
    Also handles optional spatial alignment: if features from different layers
    have different spatial resolutions (e.g., Swin Transformer), we interpolate
    to a common resolution.
    
    Args:
        layer_dims: List of input dimensions, one per extraction layer
        output_dim: Target unified dimension D
        dropout: Dropout probability in projection layers
        align_spatial: Whether to interpolate features to common spatial size
    """

    def __init__(
        self,
        layer_dims: List[int],
        output_dim: int,
        dropout: float = 0.1,
        align_spatial: bool = False,
    ):
        super().__init__()
        self.num_scales = len(layer_dims)
        self.output_dim = output_dim
        self.align_spatial = align_spatial

        # One projection per scale
        self.projections = nn.ModuleList([
            SingleScaleProjection(dim, output_dim, dropout)
            for dim in layer_dims
        ])

        # Learnable scale embeddings (additive, distinguishes which scale)
        # This helps downstream attention know which scale a feature came from
        self.scale_embeddings = nn.Parameter(
            torch.randn(self.num_scales, 1, 1, output_dim) * 0.02
        )

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Project all scales to unified dimension and stack.
        
        Args:
            features: List of K tensors, each (B, N_i, D_i)
                      For ViT: all N_i are the same (num_patches + 1)
                      For Swin: N_i may differ across scales
                      
        Returns:
            stacked: (B, K, N, D) tensor of aligned, projected features
                     K = num_scales, N = unified spatial size, D = output_dim
        """
        assert len(features) == self.num_scales, \
            f"Expected {self.num_scales} feature maps, got {len(features)}"

        B = features[0].shape[0]
        projected = []

        for i, (feat, proj) in enumerate(zip(features, self.projections)):
            # Project: (B, N_i, D_i) → (B, N_i, D)
            p = proj(feat)

            # Add scale embedding so the model knows which layer this came from
            p = p + self.scale_embeddings[i]

            projected.append(p)

        # Spatial alignment if needed (for hierarchical backbones)
        if self.align_spatial and not self._same_spatial_size(projected):
            projected = self._align_spatial_sizes(projected)

        # Stack: List of (B, N, D) → (B, K, N, D)
        stacked = torch.stack(projected, dim=1)

        return stacked

    def _same_spatial_size(self, features: List[torch.Tensor]) -> bool:
        """Check if all features have the same spatial dimension."""
        sizes = [f.shape[1] for f in features]
        return len(set(sizes)) == 1

    def _align_spatial_sizes(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Interpolate all features to the largest spatial size.
        
        For ViT, this is typically not needed (all layers have same N).
        For Swin or other hierarchical backbones, shallow layers have
        more spatial tokens, so we upsample deeper features to match.
        """
        max_n = max(f.shape[1] for f in features)
        aligned = []

        for feat in features:
            if feat.shape[1] == max_n:
                aligned.append(feat)
            else:
                # Treat token sequence as 1D and interpolate
                # (B, N, D) → (B, D, N) → interpolate → (B, D, max_n) → (B, max_n, D)
                f = feat.transpose(1, 2)
                f = nn.functional.interpolate(f, size=max_n, mode="linear", align_corners=False)
                aligned.append(f.transpose(1, 2))

        return aligned
