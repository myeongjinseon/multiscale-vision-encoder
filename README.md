# Multi-Scale Vision Encoder for Enhanced Multimodal Alignment

A research project exploring multi-scale feature aggregation from pretrained vision
backbones to improve fine-grained grounding and multimodal alignment quality.

## Core Hypothesis

Existing vision encoders that rely solely on final-layer representations lose critical
local information (small objects, attributes, spatial relationships). By extracting and
adaptively fusing features from multiple depths of a pretrained backbone, we can preserve
both global semantics and fine-grained local detail — improving downstream multimodal tasks.

## Architecture

```
Input Image
    │
    ▼
┌─────────────────────────┐
│  Pretrained ViT Backbone │
│  (CLIP / DINOv2 / ViT)  │
│                          │
│  Layer 3  → F₁ (local)  │──┐
│  Layer 6  → F₂ (parts)  │──┤
│  Layer 9  → F₃ (object) │──┤
│  Layer 12 → F₄ (global) │──┘
└─────────────────────────┘   │
                              ▼
                   ┌──────────────────┐
                   │ Feature Projection│
                   │ (align to dim D)  │
                   └────────┬─────────┘
                            ▼
                   ┌──────────────────┐
                   │ Scale-wise Attn   │
                   │ (adaptive fusion) │
                   └────────┬─────────┘
                            ▼
                   ┌──────────────────┐
                   │ Residual Refine   │
                   │ (FFN + skip)      │
                   └────────┬─────────┘
                            ▼
                  Multi-Scale Representation
                     → Multimodal Decoder
```

## Project Structure

```
multiscale-vision-encoder/
├── models/
│   ├── backbone.py           # Pretrained backbone with intermediate extraction
│   ├── feature_projection.py # Per-scale projection to unified dim
│   ├── scale_attention.py    # Scale-wise attention fusion
│   ├── residual_refine.py    # Residual refinement block
│   ├── multiscale_encoder.py # Full encoder pipeline
│   ├── captioning_head.py    # COCO Captioning decoder
│   └── grounding_head.py     # RefCOCO grounding head
├── data/
│   ├── coco_captions.py      # COCO Captions dataloader
│   └── refcoco.py            # RefCOCO dataloader
├── utils/
│   ├── metrics.py            # CIDEr, BLEU, METEOR, IoU
│   ├── visualization.py      # Attention maps, scale weights
│   └── logger.py             # Experiment logging
├── configs/
│   └── default.yaml          # Experiment configurations
├── scripts/
│   ├── train_captioning.py   # Train on COCO Captions
│   ├── train_grounding.py    # Train on RefCOCO
│   ├── evaluate.py           # Evaluation script
│   └── run_ablations.py      # Ablation study runner
├── visualization/
│   └── visualize_attention.py # Qualitative analysis tools
└── experiments/              # Saved checkpoints & logs
```

## Evaluation

- **COCO Captions**: CIDEr, BLEU-4, METEOR, SPICE (global semantic quality)
- **RefCOCO**: Accuracy@0.5IoU (fine-grained grounding ability)

## Ablation Studies

1. Number of scales (1 → 2 → 3 → 4)
2. Fusion method (concat vs avg vs scale-attention)
3. Residual refinement (with / without)
4. Backbone comparison (CLIP vs DINOv2 vs ViT)
5. Layer selection (which intermediate layers)
