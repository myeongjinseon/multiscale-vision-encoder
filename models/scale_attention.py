"""
Scale-wise Attention Module — Core Contribution.

This module adaptively fuses features from different backbone depths by
learning input-dependent scale weights. Unlike fixed fusion strategies
(concat, average, or FPN), this approach allows the model to dynamically
emphasize shallow features for fine-grained tasks (small object grounding)
and deep features for semantic tasks (captioning).

Three fusion strategies are implemented for ablation comparison:
1. ScaleAttentionFusion (proposed): Learnable, input-adaptive
2. ConcatFusion: Simple concatenation + linear projection (baseline)
3. WeightedAvgFusion: Learnable but input-independent weights (baseline)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class ScaleAttentionFusion(nn.Module):
    """
    Adaptive scale-wise attention for multi-scale feature fusion.
    
    Mechanism:
    1. Compute a summary vector per scale (mean-pooling over spatial tokens)
    2. Pass through a shared attention network to get scale importance scores
    3. Apply softmax (with temperature) to get normalized weights
    4. Weighted sum over scales → fused representation
    
    The key design choice is that attention weights are computed per-image,
    allowing the model to adapt its feature emphasis based on image content.
    An image with many small objects will naturally upweight shallow features,
    while a scenic landscape will upweight deep semantic features.
    
    Optional: multi-head scale attention, where different heads can attend
    to different scale combinations — analogous to multi-head attention
    attending to different positions.
    
    Args:
        dim: Feature dimension D (after projection)
        num_scales: Number of feature scales K
        num_heads: Number of attention heads (1 for simple, >1 for multi-head)
        temperature: Softmax temperature (higher = more uniform weights)
        dropout: Dropout on attention weights
    """

    def __init__(
        self,
        dim: int,
        num_scales: int = 4,
        num_heads: int = 1,
        temperature: float = 1.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dim = dim
        self.num_scales = num_scales
        self.num_heads = num_heads
        self.temperature = temperature
        self.head_dim = dim // num_heads

        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"

        # Scale-level attention network
        # Input: scale summary (D) → attention logit (num_heads)
        self.scale_query = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, num_heads),
        )

        # Optional: per-head projection for value
        if num_heads > 1:
            self.value_proj = nn.Linear(dim, dim)
            self.output_proj = nn.Linear(dim, dim)
        else:
            self.value_proj = nn.Identity()
            self.output_proj = nn.Identity()

        self.attn_dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

        # For logging / visualization
        self._last_scale_weights: Optional[torch.Tensor] = None

    def forward(
        self,
        multi_scale_features: torch.Tensor,
        return_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Fuse multi-scale features via adaptive attention.
        
        Args:
            multi_scale_features: (B, K, N, D) stacked projected features
                B = batch, K = num_scales, N = num_patches+1, D = dim
            return_weights: Whether to return the attention weights for analysis
            
        Returns:
            fused: (B, N, D) fused multi-scale representation
            weights: (B, K) or (B, num_heads, K) attention weights (if requested)
        """
        B, K, N, D = multi_scale_features.shape
        assert K == self.num_scales, f"Expected {self.num_scales} scales, got {K}"

        # Step 1: Compute scale summaries by mean-pooling spatial tokens
        # (B, K, N, D) → (B, K, D)
        scale_summaries = multi_scale_features.mean(dim=2)

        # Step 2: Compute attention logits per scale
        # (B, K, D) → (B, K, num_heads)
        attn_logits = self.scale_query(scale_summaries)

        # Step 3: Normalize with temperature-scaled softmax
        # (B, K, num_heads) → (B, num_heads, K) for easier broadcasting
        attn_logits = attn_logits.permute(0, 2, 1)  # (B, num_heads, K)
        scale_weights = F.softmax(attn_logits / self.temperature, dim=-1)
        scale_weights = self.attn_dropout(scale_weights)

        # Save for visualization
        self._last_scale_weights = scale_weights.detach()

        # Step 4: Weighted fusion
        if self.num_heads == 1:
            # Simple single-head: (B, 1, K) × (B, K, N, D) → (B, N, D)
            weights = scale_weights.squeeze(1).unsqueeze(-1).unsqueeze(-1)  # (B, K, 1, 1)
            fused = (weights * multi_scale_features).sum(dim=1)  # (B, N, D)
        else:
            # Multi-head: each head has its own scale weights
            values = self.value_proj(multi_scale_features)  # (B, K, N, D)
            values = values.view(B, K, N, self.num_heads, self.head_dim)
            values = values.permute(0, 3, 1, 2, 4)  # (B, heads, K, N, head_dim)

            # (B, heads, K, 1, 1) × (B, heads, K, N, head_dim) → sum over K
            w = scale_weights.unsqueeze(-1).unsqueeze(-1)  # (B, heads, K, 1, 1)
            fused = (w * values).sum(dim=2)  # (B, heads, N, head_dim)
            fused = fused.permute(0, 2, 1, 3).reshape(B, N, D)  # (B, N, D)
            fused = self.output_proj(fused)

        fused = self.norm(fused)

        if return_weights:
            return fused, scale_weights
        return fused, None

    def get_last_scale_weights(self) -> Optional[torch.Tensor]:
        """Return the most recent scale attention weights for visualization."""
        return self._last_scale_weights


class ConcatFusion(nn.Module):
    """
    Baseline: Concatenate all scales and project down.
    
    Simple but effective baseline. The model must learn which dimensions
    from which scales are important through the projection layer.
    Disadvantage: not input-adaptive — same projection for all images.
    """

    def __init__(self, dim: int, num_scales: int = 4, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(dim * num_scales, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
        )

    def forward(
        self, multi_scale_features: torch.Tensor, return_weights: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, K, N, D = multi_scale_features.shape
        # (B, K, N, D) → (B, N, K*D)
        concat = multi_scale_features.permute(0, 2, 1, 3).reshape(B, N, K * D)
        fused = self.proj(concat)  # (B, N, D)
        return fused, None


class WeightedAvgFusion(nn.Module):
    """
    Baseline: Learnable but input-independent weights.
    
    A single set of weights learned during training, applied uniformly
    to all images. This captures the "average" importance of each scale
    for the training distribution, but cannot adapt per-image.
    """

    def __init__(self, num_scales: int = 4):
        super().__init__()
        self.scale_logits = nn.Parameter(torch.zeros(num_scales))
        self.norm = nn.LayerNorm(512)  # Will be overridden

    def forward(
        self, multi_scale_features: torch.Tensor, return_weights: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, K, N, D = multi_scale_features.shape

        # Fixed weights (same for all images)
        weights = F.softmax(self.scale_logits, dim=0)  # (K,)
        weights = weights.view(1, K, 1, 1)  # Broadcast

        fused = (weights * multi_scale_features).sum(dim=1)  # (B, N, D)

        if hasattr(self, '_norm_initialized') and not self._norm_initialized:
            self.norm = nn.LayerNorm(D).to(fused.device)
            self._norm_initialized = True

        if return_weights:
            return fused, weights.squeeze()
        return fused, None


class SimpleAvgFusion(nn.Module):
    """Baseline: Plain average across all scales. No learnable parameters."""

    def forward(
        self, multi_scale_features: torch.Tensor, return_weights: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        fused = multi_scale_features.mean(dim=1)  # (B, N, D)
        return fused, None


def build_fusion_module(
    method: str,
    dim: int,
    num_scales: int = 4,
    num_heads: int = 1,
    temperature: float = 1.0,
    dropout: float = 0.1,
) -> nn.Module:
    """Factory function for fusion modules."""
    if method == "scale_attention":
        return ScaleAttentionFusion(dim, num_scales, num_heads, temperature, dropout)
    elif method == "concat":
        return ConcatFusion(dim, num_scales, dropout)
    elif method == "weighted_avg":
        return WeightedAvgFusion(num_scales)
    elif method == "simple_avg":
        return SimpleAvgFusion()
    else:
        raise ValueError(f"Unknown fusion method: {method}")
