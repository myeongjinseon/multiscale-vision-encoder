"""
Visual Grounding Head for RefCOCO.

Given multi-scale visual features and a referring expression (text query),
predicts the bounding box of the referred object. This directly evaluates
the encoder's fine-grained grounding ability — the core hypothesis of
this research.

Architecture:
    Visual features (B, N, D) + Text features (B, L, D) 
    → Cross-modal fusion → Spatial attention → Bbox prediction

If multi-scale features truly preserve local detail better than single-scale,
this task should show the most improvement — referring expressions often
describe small objects, spatial relationships, and fine-grained attributes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import math


class TextEncoder(nn.Module):
    """
    Lightweight text encoder for referring expressions.
    
    Uses a small Transformer to encode the text query. In practice,
    you could also use a frozen BERT or CLIP text encoder.
    """

    def __init__(
        self,
        vocab_size: int = 30522,
        embed_dim: int = 256,
        num_layers: int = 2,
        num_heads: int = 4,
        max_length: int = 40,
        dropout: float = 0.1,
        pad_token_id: int = 0,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        self.token_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_token_id)
        self.pos_embedding = nn.Embedding(max_length, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B, L = input_ids.shape
        positions = torch.arange(L, device=input_ids.device).unsqueeze(0)

        x = self.token_embedding(input_ids) + self.pos_embedding(positions)

        if attention_mask is not None:
            src_key_padding_mask = ~attention_mask.bool()
        else:
            src_key_padding_mask = None

        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        return self.norm(x)


class CrossModalFusion(nn.Module):
    """
    Fuse visual and text features via cross-attention.
    
    The text query attends to visual features to find relevant regions,
    then visual features attend to text to understand what to localize.
    This bidirectional attention is important for grounding because:
    - Text→Visual: identifies which visual regions match the description
    - Visual→Text: each visual patch understands which text words refer to it
    """

    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()

        # Visual attends to text (Visual→Text)
        self.v2t_attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.v2t_norm = nn.LayerNorm(dim)
        self.v2t_ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )

        # Text attends to visual (Text→Visual)
        self.t2v_attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.t2v_norm = nn.LayerNorm(dim)
        self.t2v_ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        visual: torch.Tensor,
        text: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            visual: (B, N_v, D) visual features (patch tokens, no CLS)
            text: (B, N_t, D) text features
            text_mask: (B, N_t) text padding mask
            
        Returns:
            fused_visual: (B, N_v, D) text-aware visual features
            fused_text: (B, N_t, D) visual-aware text features
        """
        key_padding = ~text_mask.bool() if text_mask is not None else None

        # Visual → Text cross-attention
        v_norm = self.v2t_norm(visual)
        v_attn, _ = self.v2t_attn(v_norm, text, text, key_padding_mask=key_padding)
        visual = visual + v_attn
        visual = visual + self.v2t_ffn(visual)

        # Text → Visual cross-attention
        t_norm = self.t2v_norm(text)
        t_attn, _ = self.t2v_attn(t_norm, visual, visual)
        text = text + t_attn
        text = text + self.t2v_ffn(text)

        return visual, text


class SpatialAttentionMap(nn.Module):
    """
    Generate spatial attention map from fused visual-text features.
    
    Produces a per-patch relevance score indicating how likely each
    spatial position is to contain the referred object.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.query_proj = nn.Linear(dim, dim)
        self.spatial_proj = nn.Linear(dim, dim)
        self.scale = dim ** -0.5

    def forward(
        self, visual: torch.Tensor, text_summary: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            visual: (B, N_v, D) fused visual features
            text_summary: (B, D) text query summary
            
        Returns:
            attn_map: (B, N_v) per-patch relevance scores
        """
        q = self.query_proj(text_summary).unsqueeze(1)  # (B, 1, D)
        k = self.spatial_proj(visual)                     # (B, N_v, D)

        attn = (q * k).sum(dim=-1) * self.scale  # (B, N_v)
        return attn.sigmoid()


class BboxPredictor(nn.Module):
    """
    Predict bounding box from attended visual features.
    
    Uses attention-weighted visual features + text summary to predict
    a normalized bounding box (cx, cy, w, h) in [0, 1].
    """

    def __init__(self, dim: int, hidden_dim: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4),
            nn.Sigmoid(),  # Output in [0, 1]
        )

    def forward(
        self, visual: torch.Tensor, attn_map: torch.Tensor, text_summary: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            visual: (B, N_v, D) visual features
            attn_map: (B, N_v) spatial attention weights
            text_summary: (B, D) text summary
            
        Returns:
            bbox: (B, 4) predicted box (cx, cy, w, h) normalized
        """
        # Attention-weighted visual summary
        weighted = (attn_map.unsqueeze(-1) * visual).sum(dim=1)  # (B, D)

        # Combine with text summary
        combined = torch.cat([weighted, text_summary], dim=-1)  # (B, 2D)

        return self.mlp(combined)


class GroundingHead(nn.Module):
    """
    Complete visual grounding module.
    
    Pipeline:
    1. Encode text query
    2. Project visual features to grounding dim
    3. Cross-modal fusion (bidirectional attention)
    4. Spatial attention map
    5. Bbox prediction
    
    Loss: L1 + GIoU (standard for detection/grounding)
    
    Args:
        visual_dim: Input visual feature dimension
        hidden_dim: Internal dimension for grounding
        num_layers: Number of cross-modal fusion layers
        bbox_loss_weight: Weight for L1 bbox loss
        giou_loss_weight: Weight for GIoU loss
    """

    def __init__(
        self,
        visual_dim: int = 512,
        hidden_dim: int = 256,
        num_layers: int = 2,
        bbox_loss_weight: float = 5.0,
        giou_loss_weight: float = 2.0,
    ):
        super().__init__()
        self.bbox_loss_weight = bbox_loss_weight
        self.giou_loss_weight = giou_loss_weight

        # Text encoder
        self.text_encoder = TextEncoder(embed_dim=hidden_dim)

        # Visual projection
        self.visual_proj = nn.Sequential(
            nn.Linear(visual_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        # Cross-modal fusion layers
        self.fusion_layers = nn.ModuleList([
            CrossModalFusion(hidden_dim) for _ in range(num_layers)
        ])

        # Spatial attention and bbox prediction
        self.spatial_attn = SpatialAttentionMap(hidden_dim)
        self.bbox_predictor = BboxPredictor(hidden_dim)

    def forward(
        self,
        visual_features: torch.Tensor,
        text_ids: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
        target_bbox: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            visual_features: (B, N, D) encoder output (patch tokens)
            text_ids: (B, L) referring expression token IDs
            text_mask: (B, L) text padding mask
            target_bbox: (B, 4) ground truth bbox (cx, cy, w, h) normalized
            
        Returns:
            dict with 'pred_bbox', 'spatial_attn', and optionally 'loss'
        """
        # Encode text
        text_features = self.text_encoder(text_ids, text_mask)  # (B, L, D_g)

        # Project visual features
        visual = self.visual_proj(visual_features)  # (B, N, D_g)

        # Cross-modal fusion
        for fusion in self.fusion_layers:
            visual, text_features = fusion(visual, text_features, text_mask)

        # Text summary (mean pool over non-padded tokens)
        if text_mask is not None:
            text_sum = (text_features * text_mask.unsqueeze(-1).float()).sum(1)
            text_sum = text_sum / text_mask.sum(1, keepdim=True).float().clamp(min=1)
        else:
            text_sum = text_features.mean(dim=1)

        # Spatial attention
        attn_map = self.spatial_attn(visual, text_sum)  # (B, N)

        # Predict bbox
        pred_bbox = self.bbox_predictor(visual, attn_map, text_sum)  # (B, 4)

        output = {
            "pred_bbox": pred_bbox,
            "spatial_attn": attn_map,
        }

        # Compute loss if targets provided
        if target_bbox is not None:
            l1_loss = F.l1_loss(pred_bbox, target_bbox, reduction="mean")
            giou_loss = self._giou_loss(pred_bbox, target_bbox)

            output["loss"] = (
                self.bbox_loss_weight * l1_loss
                + self.giou_loss_weight * giou_loss
            )
            output["l1_loss"] = l1_loss
            output["giou_loss"] = giou_loss

        return output

    def _giou_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Generalized IoU loss.
        
        Args:
            pred: (B, 4) predicted boxes (cx, cy, w, h)
            target: (B, 4) target boxes (cx, cy, w, h)
        """
        # Convert center format to corner format
        pred_x1y1 = pred[:, :2] - pred[:, 2:] / 2
        pred_x2y2 = pred[:, :2] + pred[:, 2:] / 2
        target_x1y1 = target[:, :2] - target[:, 2:] / 2
        target_x2y2 = target[:, :2] + target[:, 2:] / 2

        # Intersection
        inter_x1y1 = torch.max(pred_x1y1, target_x1y1)
        inter_x2y2 = torch.min(pred_x2y2, target_x2y2)
        inter_wh = (inter_x2y2 - inter_x1y1).clamp(min=0)
        inter_area = inter_wh[:, 0] * inter_wh[:, 1]

        # Union
        pred_area = pred[:, 2] * pred[:, 3]
        target_area = target[:, 2] * target[:, 3]
        union_area = pred_area + target_area - inter_area

        iou = inter_area / union_area.clamp(min=1e-6)

        # Enclosing box
        enclose_x1y1 = torch.min(pred_x1y1, target_x1y1)
        enclose_x2y2 = torch.max(pred_x2y2, target_x2y2)
        enclose_wh = (enclose_x2y2 - enclose_x1y1).clamp(min=0)
        enclose_area = enclose_wh[:, 0] * enclose_wh[:, 1]

        giou = iou - (enclose_area - union_area) / enclose_area.clamp(min=1e-6)

        return (1 - giou).mean()
