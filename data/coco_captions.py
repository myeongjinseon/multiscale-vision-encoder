"""
COCO Captions Dataset.

Loads COCO 2017 images with their caption annotations.
Each image has 5 reference captions. During training, one caption
is randomly selected per image. During evaluation, all 5 are used
for metric computation.

Expected directory structure:
    data/coco/
    ├── train2017/          # Training images
    ├── val2017/            # Validation images
    └── annotations/
        ├── captions_train2017.json
        └── captions_val2017.json
"""

import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from typing import Dict, List, Optional, Tuple
from collections import defaultdict


class COCOCaptionsDataset(Dataset):
    """
    COCO Captions dataset for image captioning evaluation.
    
    Args:
        root: Root directory of COCO dataset
        ann_file: Path to annotation JSON file (relative to root)
        image_dir: Path to image directory (relative to root)
        transform: Image transforms
        tokenizer: Text tokenizer (if None, returns raw text)
        max_length: Maximum caption token length
        split: 'train' or 'val'
    """

    def __init__(
        self,
        root: str,
        ann_file: str = "annotations/captions_train2017.json",
        image_dir: str = "train2017",
        transform: Optional[transforms.Compose] = None,
        tokenizer=None,
        max_length: int = 50,
        split: str = "train",
    ):
        self.root = root
        self.image_dir = os.path.join(root, image_dir)
        self.max_length = max_length
        self.split = split
        self.tokenizer = tokenizer

        # Default transforms
        if transform is None:
            if split == "train":
                self.transform = transforms.Compose([
                    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(0.1, 0.1, 0.1),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                    ),
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                    ),
                ])
        else:
            self.transform = transform

        # Load annotations
        self._load_annotations(os.path.join(root, ann_file))

    def _load_annotations(self, ann_path: str):
        """Load and index COCO caption annotations."""
        if os.path.exists(ann_path):
            with open(ann_path, "r") as f:
                data = json.load(f)

            # Build image ID → filename mapping
            self.id_to_filename = {
                img["id"]: img["file_name"] for img in data["images"]
            }

            # Build image ID → captions mapping
            self.id_to_captions = defaultdict(list)
            for ann in data["annotations"]:
                self.id_to_captions[ann["image_id"]].append(ann["caption"])

            # List of unique image IDs
            self.image_ids = list(self.id_to_captions.keys())
        else:
            # Create mock data for testing
            print(f"[COCO] Annotation file not found: {ann_path}")
            print("[COCO] Using mock data for development")
            self.image_ids = list(range(100))
            self.id_to_filename = {i: f"mock_{i}.jpg" for i in self.image_ids}
            self.id_to_captions = {
                i: [f"A mock caption for image {i}."] * 5
                for i in self.image_ids
            }
            self._use_mock = True

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        image_id = self.image_ids[idx]
        captions = self.id_to_captions[image_id]

        # Load image
        if hasattr(self, "_use_mock"):
            image = Image.new("RGB", (224, 224))
        else:
            img_path = os.path.join(self.image_dir, self.id_to_filename[image_id])
            image = Image.open(img_path).convert("RGB")

        image = self.transform(image)

        # Select caption
        if self.split == "train":
            # Random caption during training
            caption = captions[torch.randint(len(captions), (1,)).item()]
        else:
            caption = captions[0]  # First caption for deterministic eval

        # Tokenize
        if self.tokenizer is not None:
            encoded = self.tokenizer(
                caption,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            caption_ids = encoded["input_ids"].squeeze(0)
            caption_mask = encoded["attention_mask"].squeeze(0)
        else:
            # Simple character-level tokenization for testing
            caption_ids = torch.zeros(self.max_length, dtype=torch.long)
            caption_mask = torch.zeros(self.max_length, dtype=torch.long)
            tokens = [ord(c) % 30000 + 1 for c in caption[:self.max_length - 2]]
            caption_ids[0] = 101  # BOS
            for i, t in enumerate(tokens):
                caption_ids[i + 1] = t
            caption_ids[len(tokens) + 1] = 102  # EOS
            caption_mask[:len(tokens) + 2] = 1

        return {
            "image": image,
            "caption_ids": caption_ids,
            "caption_mask": caption_mask,
            "image_id": image_id,
            "raw_captions": captions,  # All 5 captions for eval
        }


def build_coco_dataloader(
    config: dict,
    split: str = "train",
    tokenizer=None,
) -> DataLoader:
    """Build COCO Captions dataloader from config."""
    data_cfg = config.get("data", {})
    coco_cfg = data_cfg.get("coco", {})

    if split == "train":
        ann_file = coco_cfg.get("train_ann", "annotations/captions_train2017.json")
        image_dir = coco_cfg.get("train_images", "train2017")
    else:
        ann_file = coco_cfg.get("val_ann", "annotations/captions_val2017.json")
        image_dir = coco_cfg.get("val_images", "val2017")

    dataset = COCOCaptionsDataset(
        root=coco_cfg.get("root", "./data/coco"),
        ann_file=ann_file,
        image_dir=image_dir,
        tokenizer=tokenizer,
        max_length=config.get("captioning", {}).get("max_length", 50),
        split=split,
    )

    train_cfg = config.get("training", {})
    return DataLoader(
        dataset,
        batch_size=train_cfg.get("batch_size", 32),
        shuffle=(split == "train"),
        num_workers=data_cfg.get("num_workers", 4),
        pin_memory=data_cfg.get("pin_memory", True),
        drop_last=(split == "train"),
        collate_fn=coco_collate_fn,
    )


def coco_collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Custom collate function that handles variable-length captions."""
    images = torch.stack([b["image"] for b in batch])
    caption_ids = torch.stack([b["caption_ids"] for b in batch])
    caption_mask = torch.stack([b["caption_mask"] for b in batch])
    image_ids = [b["image_id"] for b in batch]
    raw_captions = [b["raw_captions"] for b in batch]

    return {
        "image": images,
        "caption_ids": caption_ids,
        "caption_mask": caption_mask,
        "image_id": image_ids,
        "raw_captions": raw_captions,
    }
