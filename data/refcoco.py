"""
RefCOCO Dataset for Visual Grounding.

RefCOCO contains referring expressions that describe specific objects
in images. Each sample consists of an image, a text query, and a
ground-truth bounding box of the referred object.

This directly tests fine-grained grounding: the model must localize
the correct object among potentially many candidates based on attributes,
spatial relations, and contextual descriptions.

Datasets:
- RefCOCO (UNC split): 19,994 images, 142,210 expressions
- RefCOCO+: No location words (forces attribute understanding)
- RefCOCOg: Longer, more complex expressions

Expected directory structure:
    data/refcoco/
    ├── instances.json          # COCO instances for bbox
    ├── refs(unc).p             # RefCOCO referring expressions
    └── images/                 # COCO images (symlink to coco/train2017)
"""

import os
import json
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from typing import Dict, List, Optional, Tuple
from collections import defaultdict


class RefCOCODataset(Dataset):
    """
    RefCOCO dataset for referring expression comprehension.
    
    Each sample: (image, referring_expression, target_bbox)
    Task: Given image + expression → predict bbox of referred object
    
    Args:
        root: Root directory of RefCOCO data
        dataset: Which dataset variant ('refcoco', 'refcoco+', 'refcocog')
        split_by: Split scheme ('unc' for RefCOCO/RefCOCO+, 'umd' for RefCOCOg)
        split: 'train', 'val', 'testA', 'testB'
        image_dir: Path to COCO images
        transform: Image transforms
        tokenizer: Text tokenizer
        max_length: Maximum text token length
    """

    def __init__(
        self,
        root: str,
        dataset: str = "refcoco",
        split_by: str = "unc",
        split: str = "train",
        image_dir: Optional[str] = None,
        transform: Optional[transforms.Compose] = None,
        tokenizer=None,
        max_length: int = 40,
    ):
        self.root = root
        self.split = split
        self.max_length = max_length
        self.tokenizer = tokenizer

        # Image directory (usually COCO train2017)
        self.image_dir = image_dir or os.path.join(root, "images")

        # Transforms
        if transform is None:
            if split == "train":
                self.transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(0.1, 0.1, 0.1),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ])
        else:
            self.transform = transform

        # Load data
        self.samples = self._load_data(dataset, split_by, split)
        self._flip_enabled = (split == "train")

    def _load_data(
        self, dataset: str, split_by: str, split: str
    ) -> List[Dict]:
        """Load RefCOCO annotations and build sample list."""
        ref_file = os.path.join(self.root, f"refs({split_by}).p")
        inst_file = os.path.join(self.root, "instances.json")

        if os.path.exists(ref_file) and os.path.exists(inst_file):
            return self._load_real_data(ref_file, inst_file, split)
        else:
            print(f"[RefCOCO] Data not found at {self.root}")
            print("[RefCOCO] Using mock data for development")
            return self._create_mock_data()

    def _load_real_data(
        self, ref_file: str, inst_file: str, split: str
    ) -> List[Dict]:
        """Load actual RefCOCO data."""
        # Load referring expressions
        with open(ref_file, "rb") as f:
            refs = pickle.load(f)

        # Load COCO instances for bbox info
        with open(inst_file, "r") as f:
            instances = json.load(f)

        # Build annotation ID → bbox mapping
        ann_id_to_bbox = {}
        ann_id_to_image_id = {}
        for ann in instances["annotations"]:
            ann_id_to_bbox[ann["id"]] = ann["bbox"]  # [x, y, w, h]
            ann_id_to_image_id[ann["id"]] = ann["image_id"]

        # Build image ID → filename mapping
        image_id_to_file = {
            img["id"]: img["file_name"] for img in instances["images"]
        }
        image_id_to_size = {
            img["id"]: (img["width"], img["height"])
            for img in instances["images"]
        }

        # Build samples
        samples = []
        for ref in refs:
            if ref["split"] != split:
                continue

            ann_id = ref["ann_id"]
            if ann_id not in ann_id_to_bbox:
                continue

            image_id = ann_id_to_image_id[ann_id]
            bbox = ann_id_to_bbox[ann_id]  # [x, y, w, h] absolute
            img_w, img_h = image_id_to_size[image_id]

            # Convert to normalized center format [cx, cy, w, h]
            cx = (bbox[0] + bbox[2] / 2) / img_w
            cy = (bbox[1] + bbox[3] / 2) / img_h
            bw = bbox[2] / img_w
            bh = bbox[3] / img_h
            norm_bbox = [cx, cy, bw, bh]

            for sent in ref["sentences"]:
                samples.append({
                    "image_id": image_id,
                    "image_file": image_id_to_file[image_id],
                    "expression": sent["sent"],
                    "bbox": norm_bbox,
                    "ann_id": ann_id,
                    "ref_id": ref["ref_id"],
                })

        print(f"[RefCOCO] Loaded {len(samples)} samples for split '{split}'")
        return samples

    def _create_mock_data(self) -> List[Dict]:
        """Create mock data for development/testing."""
        import random
        expressions = [
            "the red car on the left",
            "a person wearing blue",
            "the large dog",
            "small cup on the table",
            "woman standing near the window",
            "the green book",
            "a cat sitting on the couch",
            "the man with glasses",
            "white plate in the center",
            "the tall building",
        ]
        samples = []
        for i in range(200):
            cx = random.uniform(0.2, 0.8)
            cy = random.uniform(0.2, 0.8)
            w = random.uniform(0.1, 0.4)
            h = random.uniform(0.1, 0.4)
            samples.append({
                "image_id": i,
                "image_file": f"mock_{i}.jpg",
                "expression": expressions[i % len(expressions)],
                "bbox": [cx, cy, w, h],
                "ann_id": i,
                "ref_id": i,
            })
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        # Load image
        img_path = os.path.join(self.image_dir, sample["image_file"])
        if os.path.exists(img_path):
            image = Image.open(img_path).convert("RGB")
        else:
            image = Image.new("RGB", (224, 224))

        # Handle random flip for training (must also flip bbox)
        flipped = False
        if self._flip_enabled and torch.rand(1).item() > 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            flipped = True

        image = self.transform(image)

        # Bbox
        bbox = torch.tensor(sample["bbox"], dtype=torch.float32)
        if flipped:
            bbox[0] = 1.0 - bbox[0]  # Flip cx

        # Tokenize expression
        expression = sample["expression"]
        if self.tokenizer is not None:
            encoded = self.tokenizer(
                expression,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            text_ids = encoded["input_ids"].squeeze(0)
            text_mask = encoded["attention_mask"].squeeze(0)
        else:
            text_ids = torch.zeros(self.max_length, dtype=torch.long)
            text_mask = torch.zeros(self.max_length, dtype=torch.long)
            tokens = [ord(c) % 30000 + 1 for c in expression[:self.max_length - 2]]
            text_ids[0] = 101
            for i, t in enumerate(tokens):
                text_ids[i + 1] = t
            text_ids[len(tokens) + 1] = 102
            text_mask[:len(tokens) + 2] = 1

        return {
            "image": image,
            "text_ids": text_ids,
            "text_mask": text_mask,
            "bbox": bbox,
            "expression": expression,
            "image_id": sample["image_id"],
            "ref_id": sample["ref_id"],
        }


def refcoco_collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate function for RefCOCO batches."""
    return {
        "image": torch.stack([b["image"] for b in batch]),
        "text_ids": torch.stack([b["text_ids"] for b in batch]),
        "text_mask": torch.stack([b["text_mask"] for b in batch]),
        "bbox": torch.stack([b["bbox"] for b in batch]),
        "expression": [b["expression"] for b in batch],
        "image_id": [b["image_id"] for b in batch],
        "ref_id": [b["ref_id"] for b in batch],
    }


def build_refcoco_dataloader(
    config: dict,
    split: str = "train",
    tokenizer=None,
) -> DataLoader:
    """Build RefCOCO dataloader from config."""
    data_cfg = config.get("data", {})
    refcoco_cfg = data_cfg.get("refcoco", {})
    train_cfg = config.get("training", {})

    dataset = RefCOCODataset(
        root=refcoco_cfg.get("root", "./data/refcoco"),
        dataset=refcoco_cfg.get("dataset", "refcoco"),
        split_by=refcoco_cfg.get("split_by", "unc"),
        split=split,
        tokenizer=tokenizer,
        max_length=40,
    )

    return DataLoader(
        dataset,
        batch_size=train_cfg.get("batch_size", 32),
        shuffle=(split == "train"),
        num_workers=data_cfg.get("num_workers", 4),
        pin_memory=data_cfg.get("pin_memory", True),
        drop_last=(split == "train"),
        collate_fn=refcoco_collate_fn,
    )
