"""
Residual Refinement Module.

After fusing multi-scale features, the fused representation may contain
redundancies or conflicts between different scales. The refinement module
cleans up the fused representation through:

1. Feed-Forward Network (FFN) with residual connection — standard
   Transformer-style refinement that lets the model adjust feature values
   
2. Optional Cross-Attention with Deep Feature Anchor — the fused
   representation can "consult" the original deep features to maintain
   strong semantic grounding while preserving local detail from shallow layers

The residual connections are critical: they ensure that the refinement
can only ADD information, not destroy the carefully fused multi-scale signal.
"""

import torch
import torch.nn as nn
from typing import Optional


class FeedForwardBlock(nn.Module):
    """
    Standard Transformer FFN block with pre-norm and residual.
    
    Architecture: LayerNorm → Linear → GELU → Dropout → Linear → Dropout + Residual
    
    The expansion ratio (typically 4x) allows the network to project into
    a higher-dimensional space where nonlinear transformations can refine
    the representation before projecting back down.
    """

    def __init__(self, dim: int, ffn_ratio: int = 4, dropout: float = 0.1):
        super().__init__()
        hidden_dim = dim * ffn_ratio

        self.norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

        # Initialize last linear close to zero for stable training
        nn.init.normal_(self.ffn[-2].weight, std=0.02)
        nn.init.zeros_(self.ffn[-2].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, D)
        Returns:
            (B, N, D) refined features with residual
        """
        return x + self.ffn(self.norm(x))


class CrossAttentionBlock(nn.Module):
    """
    Cross-attention that lets fused features attend to the deep feature anchor.
    
    This is the "semantic anchor" mechanism: after fusion mixes information
    from all scales, this block allows the representation to re-align with
    the original deep semantic features. This prevents the local detail from
    shallow layers from "diluting" the semantic signal.
    
    Query: fused multi-scale features (potentially noisy mix)
    Key/Value: original deep features (clean semantic representation)
    
    The cross-attention selectively pulls in semantic information where
    the fused representation needs it, without overwriting local detail
    in regions where it's useful.
    """

    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            query: (B, N, D) fused features
            key_value: (B, N, D) deep feature anchor
        Returns:
            (B, N, D) refined features with residual
        """
        B, N, D = query.shape

        # Pre-norm
        q = self.norm_q(query)
        kv = self.norm_kv(key_value)

        # Project and reshape for multi-head
        q = self.q_proj(q).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(kv).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(kv).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        # Apply and reshape
        out = (attn @ v).transpose(1, 2).reshape(B, N, D)
        out = self.proj_dropout(self.out_proj(out))

        # Residual
        return query + out


class ResidualRefinement(nn.Module):
    """
    Full residual refinement module.
    
    Consists of N blocks, each containing:
    1. (Optional) Cross-attention with deep feature anchor
    2. Feed-forward refinement with residual
    
    The number of blocks controls the depth of refinement. In practice,
    1-2 blocks are sufficient — more blocks add parameters and computation
    without proportional benefit, since the input is already well-structured
    from the backbone.
    
    Args:
        dim: Feature dimension D
        num_blocks: Number of refinement blocks
        ffn_ratio: FFN expansion ratio
        num_heads: Attention heads for cross-attention
        dropout: Dropout probability
        use_cross_attention: Whether to include cross-attention with deep anchor
    """

    def __init__(
        self,
        dim: int,
        num_blocks: int = 2,
        ffn_ratio: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_cross_attention: bool = False,
    ):
        super().__init__()
        self.use_cross_attention = use_cross_attention

        blocks = []
        for _ in range(num_blocks):
            block_modules = {}
            if use_cross_attention:
                block_modules["cross_attn"] = CrossAttentionBlock(dim, num_heads, dropout)
            block_modules["ffn"] = FeedForwardBlock(dim, ffn_ratio, dropout)
            blocks.append(nn.ModuleDict(block_modules))

        self.blocks = nn.ModuleList(blocks)
        self.final_norm = nn.LayerNorm(dim)

    def forward(
        self,
        fused_features: torch.Tensor,
        deep_anchor: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Refine fused multi-scale features.
        
        Args:
            fused_features: (B, N, D) output from scale-wise attention fusion
            deep_anchor: (B, N, D) original deep features for cross-attention
                         Required if use_cross_attention=True
                         
        Returns:
            refined: (B, N, D) refined multi-scale representation
        """
        x = fused_features

        for block in self.blocks:
            if self.use_cross_attention and "cross_attn" in block:
                assert deep_anchor is not None, \
                    "deep_anchor required when use_cross_attention=True"
                x = block["cross_attn"](x, deep_anchor)
            x = block["ffn"](x)

        return self.final_norm(x)

    def count_parameters(self) -> int:
        """Count trainable parameters in this module."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
